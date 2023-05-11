// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_transpose_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_resample_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_arg_min_max_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_concat_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gather_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_squeeze_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_slice_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_split_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_split_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_instance_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_mean_variance_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_reduce_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_fill_sequence_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/buildflags.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_deque.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_union_unsignedlong_unsignedlongsequence.h"


#if BUILDFLAG(BUILD_WEBNN_WITH_XNNPACK)
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"
#endif

#pragma optimize("", off) // TODO:::DELETE

namespace blink {

namespace {

MLGraphBuilder::BackendForTesting* g_backend_for_testing = nullptr;

bool IsFloatingPointType(V8MLOperandType::Enum operand_type) {
  switch (operand_type) {
    case V8MLOperandType::Enum::kFloat32:
    case V8MLOperandType::Enum::kFloat16:
      return true;
    case V8MLOperandType::Enum::kInt32:
    case V8MLOperandType::Enum::kUint32:
    case V8MLOperandType::Enum::kInt8:
    case V8MLOperandType::Enum::kUint8:
    case V8MLOperandType::Enum::kInt64:
    case V8MLOperandType::Enum::kUint64:
      return false;
  }
}

bool IsBooleanType(V8MLOperandType::Enum operand_type) {
  // Boolean types are unsigned 8-bit values.
  switch (operand_type) {
    case V8MLOperandType::Enum::kFloat32:
    case V8MLOperandType::Enum::kFloat16:
    case V8MLOperandType::Enum::kInt32:
    case V8MLOperandType::Enum::kUint32:
    case V8MLOperandType::Enum::kInt8:
    case V8MLOperandType::Enum::kInt64:
    case V8MLOperandType::Enum::kUint64:
      return false;
    case V8MLOperandType::Enum::kUint8:
      return true;
  }
}

bool IsIndexType(V8MLOperandType::Enum operand_type) {
  // Index types are integers, signed or unsigned.
  switch (operand_type) {
    case V8MLOperandType::Enum::kFloat32:
    case V8MLOperandType::Enum::kFloat16:
    case V8MLOperandType::Enum::kInt8:
    case V8MLOperandType::Enum::kUint8:
      return false;
    case V8MLOperandType::Enum::kInt32:
    case V8MLOperandType::Enum::kUint32:
    case V8MLOperandType::Enum::kInt64:
    case V8MLOperandType::Enum::kUint64:
      return true;
  }
}

bool ValidateClampOptions(const MLClampOptions* options,
                          ExceptionState& exception_state) {
  // The generated code of MLClampOptions uses blink::ToRestrictedFloat to
  // convert the min/max value to a single precision float. It will throw on
  // non-finite values.
  if (options->hasMinValue() && options->hasMaxValue()) {
    if (options->minValue() > options->maxValue()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("The min value (%f) should be less than or equal to "
                         "the max value (%f).",
                         options->minValue(), options->maxValue()));
      return false;
    }
  }
  return true;
}

bool ValidateAxis(uint32_t axis,
                  uint32_t dimension_count,
                  const char* operator_name,
                  ExceptionState& exception_state) {
  if (axis >= dimension_count)
  {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "The %s axis (%u) must be within the dimension count (%u).",
              operator_name,
              axis, dimension_count));
      return false;
  }
  return true;
}

bool ValidateAxes(base::span<const uint32_t> axes,
                  uint32_t dimension_count,
                  const char* operator_name,
                  ExceptionState& exception_state) {
  Vector<uint32_t> seen_axes(dimension_count);

  for (auto axis : axes)
  {
    if (axis >= dimension_count)
    {
      exception_state.ThrowDOMException(
           DOMExceptionCode::kDataError,
            String::Format(
                "The %s axis (%u) must be less than the dimension count (%u).",
                operator_name,
                axis, dimension_count));
       return false;
    }
    if (seen_axes[axis])
    {
      exception_state.ThrowDOMException(
           DOMExceptionCode::kDataError,
            String::Format(
                "Each %s axis (%u) must only occur once.",
                operator_name, axis));
       return false;
    }
    seen_axes[axis] = true;
  }
  return true;
}

// Generates a 32-bit mask, validating all axes fit within 32 dimensions.
bool ValidateAxesMask(base::span<const uint32_t> axes,
                      const char* operator_name,
                      ExceptionState& exception_state,
                      /*out*/ uint32_t& axes_mask) {
  uint32_t current_mask = 0x00000000;
  axes_mask = current_mask;

  const uint32_t maximum_rank = 32; // Only have 32 bits available.

  for (auto axis : axes)
  {
    if (axis >= maximum_rank)
    {
      exception_state.ThrowDOMException(
           DOMExceptionCode::kDataError,
            String::Format(
                "%s axis (%u) is beyond the maximum (%u).",
                operator_name,
                axis, maximum_rank));
      return false;
    }
    current_mask |= 1 << axis;
  }
  axes_mask = current_mask;
  return true;
}

// Computes the number of elements given the dimensions.
// Note this expects dimensions that are already known and validated
// and thus cannot overflow, rather than untrusted parameters like
// reshape's new shape before validation.
uint32_t ComputeElementCount(base::span<const uint32_t> dimensions)
{
  return std::accumulate(dimensions.begin(), dimensions.end(), 1u, std::multiplies<uint32_t>{});
}

// Increases the rank to a minimum count by padding with leading ones.
Vector<uint32_t> ExpandDimensions(
    const base::span<const uint32_t> original_dimensions,
    wtf_size_t minimum_rank) {

  wtf_size_t old_rank = static_cast<wtf_size_t>(original_dimensions.size());
  wtf_size_t new_rank = std::max(minimum_rank, static_cast<wtf_size_t>(old_rank));
  wtf_size_t leading_filler_count = new_rank - old_rank;

  Vector<uint32_t> expanded_dimensions(new_rank, 1u);
  std::copy(original_dimensions.begin(), original_dimensions.end(),
            expanded_dimensions.begin() + leading_filler_count);
  return expanded_dimensions;
}

// Broadcast the input shapes and return the output shape.
// If bidirectional is true, its behavior follows the numpy-broadcasting-rule:
// https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules.
// Otherwise, it unidirectionally broadcasts the lhs to the rhs.
// The ignorable tail count is useful for cases like MatMul, where you want
// to ignore the trailing tail of dimensions and only broadcast the leading
// ones, because the trailing part (returned as 0's) will be filled in later.
absl::optional<Vector<uint32_t>> BroadcastShapes(
    base::span<const uint32_t> dims_lhs,
    base::span<const uint32_t> dims_rhs,
    bool bidirectional = true,
    wtf_size_t ignorable_tail_count = 0
) {
  // If bidirectional is true, the rank of the output shape is the maximum
  // rank of the input shapes. Otherwise it is as the same as the rhs' rank.
  auto rank_lhs = static_cast<wtf_size_t>(dims_lhs.size());
  auto rank_rhs = static_cast<wtf_size_t>(dims_rhs.size());
  auto rank_output = bidirectional ? std::max(rank_lhs, rank_rhs) : rank_rhs;
  Vector<uint32_t> dims_output(rank_output);

  // Note the loop effectively works backwards from the end of the dimensions
  // array (the counter is forward, but accesses are relative the end).
  for (wtf_size_t i = ignorable_tail_count; i < rank_output; ++i) {

    auto dim_lhs = i < rank_lhs ? dims_lhs[rank_lhs - i - 1] : 1;
#if 0 // TODO:::DELETE - Broadcasting needs to work correctly with zero dimensions.
      // Ultimately this should handle empty tensors in the lower layer via nop.
      // A DCHECK is not appropriate here.
    DCHECK_GT(dim_lhs, uint32_t(0));
#endif
    auto dim_rhs = i < rank_rhs ? dims_rhs[rank_rhs - i - 1] : 1;
#if 0 // TODO:::DELETE - Broadcasting needs to work correctly with zero dimensions. 
    DCHECK_GT(dim_rhs, uint32_t(0));
#endif

    // If bidirectional is true, two dimensions are compatible when they are
    // equal, or one of them is 1. Otherwise, two dimensions are compatible
    // when they are equal, or the lhs dimension is 1.
    if (bidirectional) {
      if (dim_lhs != dim_rhs && dim_lhs != 1 && dim_rhs != 1) {
        return absl::nullopt;
      }
    } else if (dim_lhs != dim_rhs && dim_lhs != 1) {
      return absl::nullopt;
    }

    // If bidirectional is true, for each dimension of the output tensor, its
    // size is the maximum size along that dimension of the input shapes.
    // Otherwise, its size is the same as the rhs.
    dims_output[rank_output - i - 1] =
        bidirectional ? std::max(dim_lhs, dim_rhs) : dim_rhs;
  }
  return dims_output;
}

MLOperand* BuildUnaryOperator(MLGraphBuilder* builder,
                              MLOperator::OperatorKind kind,
                              const MLOperand* input,
                              ExceptionState& exception_state) {
  String error_message;
  auto* ml_operator = MakeGarbageCollected<MLOperator>(builder, kind);

  Vector<uint32_t> output_dimensions = input->Dimensions();
  auto* output = MLOperand::ValidateAndCreateOutput(
      builder, input->Type(), std::move(output_dimensions), ml_operator, /*out*/ error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }

  ml_operator->Connect({input}, {output});
  return output;
}

MLOperand* BuildUnaryOperator(
    MLGraphBuilder* builder,
    MLOperator::OperatorKind kind,
    const MLOperand* input,
    Vector<uint32_t> output_dimensions,
    V8MLOperandType::Enum output_data_type,
    const bindings::DictionaryBase* options,
    ExceptionState& exception_state) {
  String error_message;
  auto* ml_operator = MakeGarbageCollected<MLOperator>(builder, kind, options);

  auto* output = MLOperand::ValidateAndCreateOutput(
      builder, output_data_type, std::move(output_dimensions), ml_operator,
      /*out*/ error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }

  ml_operator->Connect({input}, {output});
  return output;
}

MLOperand* BuildElementwiseBinary(MLGraphBuilder* builder,
                                  MLOperator::OperatorKind kind,
                                  const MLOperand* a,
                                  const MLOperand* b,
                                  V8MLOperandType::Enum output_data_type,
                                  ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input types don't match.");
    return nullptr;
  }
  absl::optional<Vector<uint32_t>> dims_output =
      BroadcastShapes(a->Dimensions(), b->Dimensions());
  if (!dims_output) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input shapes are not broadcastable.");
    return nullptr;
  }

  auto* binary = MakeGarbageCollected<MLOperator>(builder, kind);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      builder, output_data_type, std::move(dims_output.value()), binary, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  binary->Connect({a, b}, {output});
  return output;
}

MLOperand* BuildArgMinMax(MLGraphBuilder* graph_builder,
                          MLOperator::OperatorKind operator_kind,
                          const char* operator_name,
                          const MLOperand* input,
                          const MLArgMinMaxOptions* options,
                          ExceptionState& exception_state) {
  // Validate axis.
  uint32_t axis = options->axis();
  auto& input_dimensions = input->Dimensions();
  if (!ValidateAxis(axis,
                    input_dimensions.size(),
                    operator_name,
                    exception_state)) {
    return nullptr;
  }

  // Determine output size, eliminating the active axis or keeping it with size 1.
  Vector<uint32_t> output_dimensions = input_dimensions;
  if (options->keepDimensions())
  {
    output_dimensions[axis] = 1;
  }
  else
  {
    output_dimensions.EraseAt(axis);
  }

  return BuildUnaryOperator(graph_builder, operator_kind, input,
                            output_dimensions, V8MLOperandType::Enum::kInt64, options,
                            exception_state);
}

MLOperand* BuildReductionOperator(MLGraphBuilder* graph_builder,
                                  MLOperator::OperatorKind operator_kind,
                                  const char* operator_name,
                                  const MLOperand* input,
                                  const MLReduceOptions* options,
                                  ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  Vector<uint32_t> axes;
  uint32_t axes_mask = 0xFFFFFFFF; // Remove all axes by default, if none passed.

  // Verify the axes are within the input rank and not duplicated.
  if (options->hasAxes())
  {
    axes = options->axes();
    if (!ValidateAxes(axes, input_rank, operator_name, exception_state)) {
      return nullptr;
    }
    if (!ValidateAxesMask(options->axes(),
                          operator_name,
                          exception_state,
                          /*out*/ axes_mask)) {
      return nullptr;
    }
  }
  else // Reduce all dimensions if permutations are missing.
  {
    axes.resize(input_rank);
    std::iota(axes.begin(), axes.end(), 0u);
    // axes_mask already 0xFFFFFFFF.
  }

  // Set dimension to 1 that are reduced.
  // or erase them entirely if MLReduceOptions::keepDimensions = false.
  Vector<uint32_t> output_dimensions = input_dimensions;
  wtf_size_t output_rank = input_rank;
  bool keep_dimensions = options->keepDimensions();
  for (wtf_size_t i = 0; i < output_rank; /*increment in loop*/)
  {
    wtf_size_t advance_count = 1;
    if (axes_mask & (1 << i)) {
      if (keep_dimensions) {
        output_dimensions[i] = 1u; // Reduce dimension.
      }
      else {
          output_dimensions.EraseAt(i); // Remove reduced dimension.
          advance_count = 0; // Stay at the current index.
          --output_rank;
      }
    }
    i += advance_count;
  }

  // Pass the normalized options onward, simplifying the lower level's job.
  MLReduceOptions* normalized_options = MLReduceOptions::Create();
  normalized_options->setAxes(axes);

  return BuildUnaryOperator(graph_builder, operator_kind, input,
                            output_dimensions, input->Type(), normalized_options,
                            exception_state);
}

struct PaddingSizes {
  uint32_t begin;
  uint32_t end;
};

// Calculate the padding given auto pad, input size, filter size, stride and
// dilation. Return the calculated padding sizes if no error.
absl::optional<PaddingSizes> CalculatePaddingForAutoPad(
    V8MLAutoPad::Enum auto_pad,
    const uint32_t input_size,
    const uint32_t filter_size,
    const uint32_t stride,
    const uint32_t dilation) {
  auto checked_output_size =
      (base::MakeCheckedNum<uint32_t>(input_size) + stride - 1) / stride;
  auto checked_dilated_filter_size =
      (base::MakeCheckedNum<uint32_t>(filter_size) - 1) * dilation + 1;
  auto checked_needed_input_size =
      (checked_output_size - 1) * stride + checked_dilated_filter_size;
  if (!checked_needed_input_size.IsValid()) {
    return absl::nullopt;
  }
  auto checked_total_padding =
      checked_needed_input_size.ValueOrDie() > input_size
          ? checked_needed_input_size - input_size
          : base::MakeCheckedNum<uint32_t>(0);
  base::CheckedNumeric<uint32_t> checked_padding_begin, checked_padding_end;
  switch (auto_pad) {
    case V8MLAutoPad::Enum::kSameUpper:
      checked_padding_begin = checked_total_padding / 2;
      checked_padding_end = (checked_total_padding + 1) / 2;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      checked_padding_begin = (checked_total_padding + 1) / 2;
      checked_padding_end = checked_total_padding / 2;
      break;
    default:
      NOTREACHED();
  }

  uint32_t padding_begin, padding_end;
  if (!checked_padding_begin.AssignIfValid(&padding_begin) ||
      !checked_padding_end.AssignIfValid(&padding_end)) {
    return absl::nullopt;
  }
  return PaddingSizes({.begin = padding_begin, .end = padding_end});
}

// Calculate the output size for conv2d based on WebNN spec:
// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
// Return the calculated output size if no error.
absl::optional<double> CalculateConv2dOutputSize(
    const uint32_t input_size,
    const uint32_t filter_size,
    const uint32_t beginning_padding,
    const uint32_t ending_padding,
    const uint32_t stride,
    const uint32_t dilation,
    const bool backward_direction,
    String& error_message) {
  // Calculate the dilated filter sizes.
  auto checked_effective_filter_size =
      (base::MakeCheckedNum<uint32_t>(filter_size) - 1) * dilation + 1;
  if (!checked_effective_filter_size.IsValid()) {
    error_message = "The effective filter size is too large.";
    return absl::nullopt;
  }

  // TODO:::
  //if (backward_direction)
  //{
  //
  //}

  // Calculate the output size in double precision floating point number that
  // ensures all dimension values of type uint32_t can be exactly represented.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Precision_limitations_on_integer_values
  // The max value of checked_output_size should be 3 * UINT_MAX + 1,
  // which is smaller than the max safe integer value for double type.
  auto checked_output_size =
      (base::MakeCheckedNum<double>(input_size) -
       checked_effective_filter_size + beginning_padding + ending_padding) /
          stride +
      1;

  if (checked_output_size.ValueOrDie() < 0) {
    error_message = "The input size is too small to fill the window.";
    return absl::nullopt;
  }

  // Check if the value is valid for rounding to uint32_t type.
  if (!checked_output_size.IsValid<uint32_t>()) {
    error_message = "The output size is too large.";
    return absl::nullopt;
  }

  return checked_output_size.ValueOrDie();
}

struct FloatSize2D {
  double height;
  double width;
};

// Validate and calculate the output spatial dimensions of conv2d given
// input sizes, filter sizes, padding, strides and dilations.
// Return the calculated output sizes in double precision floating point
// number if no errors.
absl::optional<FloatSize2D> ValidateAndCalculateConv2dOutputSizes(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t filter_height,
    const uint32_t filter_width,
    const Vector<uint32_t>& padding,
    const Vector<uint32_t>& strides,
    const Vector<uint32_t>& dilations,
    const V8MLAutoPad auto_pad,
    const bool backward_direction,
    ExceptionState& exception_state) {
  // Validate padding and get its values.
  if (padding.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of padding should be 4.");
    return absl::nullopt;
  }
  uint32_t padding_beginning_height = padding[0];
  uint32_t padding_ending_height = padding[1];
  uint32_t padding_beginning_width = padding[2];
  uint32_t padding_ending_width = padding[3];

  // Validate strides and get its values.
  if (strides.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of strides should be 2.");
    return absl::nullopt;
  }
  if (std::any_of(strides.begin(), strides.end(),
                  [](uint32_t x) { return x == 0; })) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "All strides should be greater than 0.");
    return absl::nullopt;
  }
  const uint32_t stride_height = strides[0];
  const uint32_t stride_width = strides[1];

  // Validate dilations and get its values.
  if (dilations.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of dilations should be 2.");
    return absl::nullopt;
  }
  if (std::any_of(dilations.begin(), dilations.end(),
                  [](uint32_t x) { return x == 0; })) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "All dilations should be greater than 0.");
    return absl::nullopt;
  }
  const uint32_t dilation_height = dilations[0];
  const uint32_t dilation_width = dilations[1];

  // When the autoPad is other than "explicit", the values in the
  // options.padding array are ignored and the explicit padding values need to
  // be calculated.
  if (auto_pad != V8MLAutoPad::Enum::kExplicit) {
    auto padding_sizes_height = MLGraphBuilder::CalculatePaddingForAutoPad(
        auto_pad.AsEnum(), input_height, filter_height, stride_height,
        dilation_height);
    if (!padding_sizes_height) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "Overflow occurred when calculating "
          "the padding along the height dimension.");
      return absl::nullopt;
    }
    padding_beginning_height = padding_sizes_height.value().begin;
    padding_ending_height = padding_sizes_height.value().end;
    auto padding_sizes_width = MLGraphBuilder::CalculatePaddingForAutoPad(
        auto_pad.AsEnum(), input_width, filter_width, stride_width,
        dilation_width);
    if (!padding_sizes_width) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "Overflow occurred when calculating "
          "the padding along the width dimension.");
      return absl::nullopt;
    }
    padding_beginning_width = padding_sizes_width.value().begin;
    padding_ending_width = padding_sizes_width.value().end;
  }

  String error_message;
  auto float_output_height = CalculateConv2dOutputSize(
      input_height, filter_height, padding_beginning_height,
      padding_ending_height, stride_height, dilation_height, backward_direction,
      error_message);
  if (!float_output_height) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Failed to calculate the output height: " + error_message);
    return absl::nullopt;
  }

  auto float_output_width = CalculateConv2dOutputSize(
      input_width, filter_width, padding_beginning_width, padding_ending_width,
      stride_width, dilation_width, backward_direction, error_message);
  if (!float_output_width) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Failed to calculate the output width: " + error_message);
    return absl::nullopt;
  }

  return FloatSize2D({.height = float_output_height.value(),
                      .width = float_output_width.value()});
}

MLOperand* BuildPool2d(MLGraphBuilder* builder,
                       MLOperator::OperatorKind kind,
                       const MLOperand* input,
                       const MLPool2dOptions* options,
                       ExceptionState& exception_state) {
  // Validate input operand and set its sizes.
  const auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input should be a 4-D tensor.");
    return nullptr;
  }
  // The layout option specifies the layout format of the input tensor.
  uint32_t input_batches, input_channels, input_height, input_width;
  switch (options->layout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, channels, height, width]
      input_batches = input_shape[0];
      input_channels = input_shape[1];
      input_height = input_shape[2];
      input_width = input_shape[3];
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, channels]
      input_batches = input_shape[0];
      input_height = input_shape[1];
      input_width = input_shape[2];
      input_channels = input_shape[3];
      break;
  }

  // Validate windowDimensions and get its values. If not present, the window
  // dimensions are assumed to be the height and width dimensions of the input
  // shape. The current WebNN spec defines the windowDimensions as signed
  // integer:
  // https://www.w3.org/TR/webnn/#dom-mlpool2doptions-windowdimensions
  // However, there is a proposal of using unsigned integer:
  // https://github.com/webmachinelearning/webnn/pull/294
  // Before the change merged, the signed integers are checked_cast to
  // unsigned integers for output shape calculation.
  uint32_t window_height = input_height;
  uint32_t window_width = input_width;
  if (options->hasWindowDimensions()) {
    if (options->windowDimensions().size() != 2) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The length of window dimensions should be 2.");
      return nullptr;
    }
    if (std::any_of(options->windowDimensions().begin(),
                    options->windowDimensions().end(),
                    [](uint32_t x) { return x == 0; })) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "All window dimensions should be greater than 0.");
      return nullptr;
    }
    window_height = options->windowDimensions()[0];
    window_width = options->windowDimensions()[1];
  }

  // Reuse ValidateAndCalculateConv2dOutputSizes to calculate pool2d output
  // sizes.
  const auto output_sizes = ValidateAndCalculateConv2dOutputSizes(
      input_height, input_width, window_height, window_width,
      // If padding is not present, the values are assumed to be [0,0,0,0].
      options->getPaddingOr({0, 0, 0, 0}),
      // If strides is not present, the values are assumed to be [1,1].
      options->getStridesOr({1, 1}),
      // If dilations is not present, the values are assumed to be [1, 1].
      options->getDilationsOr({1, 1}), options->autoPad(), false,
      exception_state);

  if (!output_sizes) {
    return nullptr;
  }
  const uint32_t floor_output_height =
      base::ClampFloor<uint32_t>(output_sizes.value().height);
  const uint32_t ceil_output_height =
      base::ClampCeil<uint32_t>(output_sizes.value().height);
  const uint32_t floor_output_width =
      base::ClampFloor<uint32_t>(output_sizes.value().width);
  const uint32_t ceil_output_width =
      base::ClampCeil<uint32_t>(output_sizes.value().width);

  uint32_t output_height, output_width;
  if (options->hasOutputSizes()) {
    // TODO(ningxin.hu@intel.com): report a DevTools warning message if
    // rounding type is provided but ignored.
    if (options->outputSizes().size() != 2) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The length of output sizes should be 2.");
      return nullptr;
    }
    if (std::any_of(options->outputSizes().begin(),
                    options->outputSizes().end(),
                    [](uint32_t x) { return x == 0; })) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "All output sizes should be greater than 0.");
      return nullptr;
    }
    uint32_t user_output_height = options->outputSizes()[0];
    uint32_t user_output_width = options->outputSizes()[1];

    // Check whether the user supplied output sizes is either floor or ceil
    // rounding of the calculated output sizes. The backend implementation
    // should check whether the indicated rounding type is supported.
    if ((user_output_height == floor_output_height &&
         user_output_width == floor_output_width) ||
        (user_output_height == ceil_output_height &&
         user_output_width == ceil_output_width)) {
      output_height = user_output_height;
      output_width = user_output_width;
    } else {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          (floor_output_height == ceil_output_height &&
           floor_output_width == ceil_output_width)
              ? String::Format("The output sizes should be [%u, %u].",
                               floor_output_height, floor_output_width)
              : String::Format(
                    "The output sizes should be either [%u, %u] or [%u, %u].",
                    floor_output_height, floor_output_width, ceil_output_height,
                    ceil_output_width));
      return nullptr;
    }
  } else {
    switch (options->roundingType().AsEnum()) {
      case V8MLRoundingType::Enum::kFloor:
        output_height = floor_output_height;
        output_width = floor_output_width;
        break;
      case V8MLRoundingType::Enum::kCeil:
        output_height = ceil_output_height;
        output_width = ceil_output_width;
        break;
    }
  }
  // The layout option specifies the layout format of the output tensor.
  Vector<uint32_t> output_shape;
  switch (options->layout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, channels, height, width]
      output_shape = {input_batches, input_channels, output_height,
                      output_width};
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, channels]
      output_shape = {input_batches, output_height, output_width,
                      input_channels};
      break;
  }
  // Create pool2d operator and its output operand. Connect the pool2d
  // operator to its input and output operands.
  auto* pool2d = MakeGarbageCollected<MLOperator>(builder, kind, options);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      builder, input->Type(), std::move(output_shape), pool2d, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  pool2d->Connect({input}, {output});
  return output;
}

bool CheckValueExpectedVsActual(uint32_t expected_value,
                                uint32_t actual_value,
                                const char* value_name,
                                ExceptionState& exception_state) {
  if (actual_value != expected_value) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        String::Format("The actual value (%u) for %s does not match the "
                       "expected value (%u)",
                       actual_value, value_name, expected_value));
    return false;
  }
  return true;
}

// !!!!
MLOperand* BuildConv2d(MLGraphBuilder* builder,
                       MLOperator::OperatorKind operator_kind,
                       const MLOperand* input,
                       const MLOperand* filter,
                       const MLConvOptionsInternal* options,
                       ExceptionState& exception_state) {
  // Validate input operand and set its sizes.
  const auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input should be a 4-D tensor.");
    return nullptr;
  }
  // The input layout option specifies the layout format of the input tensor.
  uint32_t input_batches, input_channels, input_height, input_width;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, input_channels, height, width]
      input_batches = input_shape[0];
      input_channels = input_shape[1];
      input_height = input_shape[2];
      input_width = input_shape[3];
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, input_channels]
      input_batches = input_shape[0];
      input_height = input_shape[1];
      input_width = input_shape[2];
      input_channels = input_shape[3];
      break;
  }

  // Validate filter operand and set its sizes.
  if (filter->Type() != input->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The filter type doesn't match the input type.");
    return nullptr;
  }

  const auto filter_shape = filter->Dimensions();
  if (!CheckValueExpectedVsActual(filter_shape.size(), 4, "filter tensor rank",
                                  exception_state)) {
    return nullptr;
  }

  // The filter layout specifies the filter layout format.
  uint32_t filter_height, filter_width, output_channels, filter_input_channels;
  switch (options->filterLayout().AsEnum()) {
    case V8MLConvFilterOperandLayoutInternal::Enum::kHwio:
      // "hwio": [height, width, input_channels/groups, output_channels]
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      filter_input_channels = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConvFilterOperandLayoutInternal::Enum::kOhwi:
      // "ohwi": [output_channels, height, width, input_channels/groups]
      output_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      filter_input_channels = filter_shape[3];
      break;
    case V8MLConvFilterOperandLayoutInternal::Enum::kIhwo:
      // "ihwo": [input_channels/groups, height, width, output_channels]
      filter_input_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConvFilterOperandLayoutInternal::Enum::kOihw:
      // "oihw": [output_channels, input_channels/groups, height, width]
      output_channels = filter_shape[0];
      filter_input_channels = filter_shape[1];
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      break;
      // TODO: Fill in other enums.
    case V8MLConvFilterOperandLayoutInternal::Enum::kHwoi:
      //--// "ihwo": [input_channels/groups, height, width, output_channels]
      //--filter_input_channels = filter_shape[0];
      //--filter_height = filter_shape[1];
      //--filter_width = filter_shape[2];
      //--output_channels = filter_shape[3];
      break;
    case V8MLConvFilterOperandLayoutInternal::Enum::kIohw:
      //--// "oihw": [output_channels, input_channels/groups, height, width]
      //--output_channels = filter_shape[0];
      //--filter_input_channels = filter_shape[1];
      //--filter_height = filter_shape[2];
      //--filter_width = filter_shape[3];
      break;
  }

  // Validate bias operand if it is present.
  if (options->hasBias()) {
    const auto bias_shape = options->bias()->Dimensions();
    if (bias_shape.size() != 1) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The bias should be a 1-D tensor.");
      return nullptr;
    }
    if (bias_shape[0] != output_channels) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("The bias shape should be [%u].", output_channels));
      return nullptr;
    }
    if (options->bias()->Type() != input->Type()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The bias type doesn't match input type.");
      return nullptr;
    }
  }
  // Validate groups.
  if (options->groups() == 0) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups should be greater than 0.");
    return nullptr;
  }
  if (input_channels % options->groups() != 0 ||
      filter_input_channels != input_channels / options->groups()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups must evenly divide the input "
                                      "channels to filter input channels.");
    return nullptr;
  }

  absl::optional<FloatSize2D> output_sizes;
  if (options->hasOutputSizes()) {

    if (!CheckValueExpectedVsActual(options->outputSizes().size(), 2,
                                    "outputSizes size", exception_state)) {
      return nullptr;
    }

    const auto& explicit_output_sizes = options->outputSizes();
    output_sizes = FloatSize2D{double(explicit_output_sizes[0]), double(explicit_output_sizes[1])};
  } else {
    if (options->hasPadding() &&
        !CheckValueExpectedVsActual(options->padding().size(), 4,
                                    "padding size", exception_state)) {
      return nullptr;
    }
    if (options->hasStrides() &&
        !CheckValueExpectedVsActual(options->strides().size(), 2,
                                    "strides size", exception_state)) {
      return nullptr;
    }
    if (options->hasDilations() &&
        !CheckValueExpectedVsActual(options->dilations().size(), 2,
                                    "dilations size", exception_state)) {
      return nullptr;
    }

    output_sizes = ValidateAndCalculateConv2dOutputSizes(
        input_height, input_width, filter_height, filter_width,
        // If padding is not present, the values are assumed to be [0,0,0,0].
        options->getPaddingOr({0, 0, 0, 0}),
        // If strides is not present, the values are assumed to be [1,1].
        options->getStridesOr({1, 1}),
        // If dilations is not present, the values are assumed to be [1, 1].
        options->getDilationsOr({1, 1}), options->autoPad(),
        operator_kind == MLOperator::OperatorKind::kConvTranspose2d,
        exception_state);
  }
  if (!output_sizes) {
    return nullptr;
  }

  const uint32_t output_height =
      base::ClampFloor<uint32_t>(output_sizes.value().height);
  const uint32_t output_width =
      base::ClampFloor<uint32_t>(output_sizes.value().width);
  // The input layout option specifies the layout format of the output tensor.
  Vector<uint32_t> output_shape;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, output_channels, height, width]
      output_shape = {input_batches, output_channels, output_height,
                      output_width};
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, output_channels]
      output_shape = {input_batches, output_height, output_width,
                      output_channels};
      break;
  }
  // Create conv2d operator and its output operand. Connect the conv2d
  // operator to its input and output operands.
  auto* conv2d = MakeGarbageCollected<MLOperator>(
      builder, MLOperator::OperatorKind::kConv2d, options);
  HeapVector<Member<const MLOperand>> inputs = {input, filter};
  if (options->hasBias()) {
    inputs.push_back(options->bias());
  }
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      builder, input->Type(), std::move(output_shape), conv2d, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  conv2d->Connect(std::move(inputs), {output});
  return output;
}

}  // namespace

// static
MLGraphBuilder* MLGraphBuilder::Create(MLContext* context) {
  return MakeGarbageCollected<MLGraphBuilder>(context);
}

MLGraphBuilder::MLGraphBuilder(MLContext* context) : ml_context_(context) {}

MLGraphBuilder::~MLGraphBuilder() = default;

void MLGraphBuilder::Trace(Visitor* visitor) const {
  visitor->Trace(ml_context_);
  ScriptWrappable::Trace(visitor);
}

MLContext* MLGraphBuilder::GetContext() const {
  return ml_context_;
}

// static
absl::optional<MLGraphBuilder::PaddingSizes>
MLGraphBuilder::CalculatePaddingForAutoPad(V8MLAutoPad::Enum auto_pad,
                                           const uint32_t input_size,
                                           const uint32_t filter_size,
                                           const uint32_t stride,
                                           const uint32_t dilation) {
  auto checked_output_size =
      (base::MakeCheckedNum<uint32_t>(input_size) + stride - 1) / stride;
  auto checked_dilated_filter_size =
      (base::MakeCheckedNum<uint32_t>(filter_size) - 1) * dilation + 1;
  auto checked_needed_input_size =
      (checked_output_size - 1) * stride + checked_dilated_filter_size;
  if (!checked_needed_input_size.IsValid()) {
    return absl::nullopt;
  }
  auto checked_total_padding =
      checked_needed_input_size.ValueOrDie() > input_size
          ? checked_needed_input_size - input_size
          : base::MakeCheckedNum<uint32_t>(0);
  base::CheckedNumeric<uint32_t> checked_padding_begin, checked_padding_end;
  switch (auto_pad) {
    case V8MLAutoPad::Enum::kSameUpper:
      checked_padding_begin = checked_total_padding / 2;
      checked_padding_end = (checked_total_padding + 1) / 2;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      checked_padding_begin = (checked_total_padding + 1) / 2;
      checked_padding_end = checked_total_padding / 2;
      break;
    default:
      NOTREACHED();
  }
  uint32_t padding_begin, padding_end;
  if (!checked_padding_begin.AssignIfValid(&padding_begin) ||
      !checked_padding_end.AssignIfValid(&padding_end)) {
    return absl::nullopt;
  }
  return PaddingSizes({.begin = padding_begin, .end = padding_end});
}

MLOperand* MLGraphBuilder::input(String name,
                                 const MLOperandDescriptor* desc,
                                 ExceptionState& exception_state) {
  String error_message;
  // If no dimensions, it represents a scalar. Set dimensions to {1}.
  Vector<uint32_t> dimensions = desc->getDimensionsOr({1});
  auto* input_operand = MLOperand::ValidateAndCreateInput(
      this, desc->type().AsEnum(), std::move(dimensions), std::move(name),
      error_message);
  if (!input_operand) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  return input_operand;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc,
                                    NotShared<DOMArrayBufferView> buffer_view,
                                    ExceptionState& exception_state) {
  String error_message;
  // If no dimensions, it represents a scalar. Set dimensions to {1}.
  Vector<uint32_t> dimensions = desc->getDimensionsOr({1});
  auto* constant_operand = MLOperand::ValidateAndCreateConstant(
      this, desc->type().AsEnum(), std::move(dimensions), buffer_view.Get(),
      error_message);
  if (!constant_operand) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  return constant_operand;
}

MLOperand* MLGraphBuilder::clamp(const MLOperand* input,
                                 const MLClampOptions* options,
                                 ExceptionState& exception_state) {
  if (!ValidateClampOptions(options, exception_state)) {
    return nullptr;
  }
  auto* clamp = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kClamp, options);
  // According to WebNN spec
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-clamp, the output tensor of
  // clamp has the same type and dimensions as its input.
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), input->Dimensions(), clamp, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  clamp->Connect({input}, {output});
  return output;
}

MLOperator* MLGraphBuilder::clamp(const MLClampOptions* options,
                                  ExceptionState& exception_state) {
  if (!ValidateClampOptions(options, exception_state)) {
    return nullptr;
  }
  // Create the clamp operator that would be used as an activation function.
  return MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kClamp, options);
}

MLOperand* MLGraphBuilder::conv2d(const MLOperand* input,
                                  const MLOperand* filter,
                                  const MLConv2dOptions* options,
                                  ExceptionState& exception_state) {
    //!!!
  // Validate input operand and set its sizes.
  const auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input should be a 4-D tensor.");
    return nullptr;
  }
  // The input layout option specifies the layout format of the input tensor.
  uint32_t input_batches, input_channels, input_height, input_width;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, input_channels, height, width]
      input_batches = input_shape[0];
      input_channels = input_shape[1];
      input_height = input_shape[2];
      input_width = input_shape[3];
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, input_channels]
      input_batches = input_shape[0];
      input_height = input_shape[1];
      input_width = input_shape[2];
      input_channels = input_shape[3];
      break;
  }

  // Validate filter operand and set its sizes.
  if (filter->Type() != input->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The filter type doesn't match the input type.");
    return nullptr;
  }
  const auto filter_shape = filter->Dimensions();
  if (filter_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The filter should be a 4-D tensor.");
    return nullptr;
  }
  // The filter layout specifies the filter layout format.
  uint32_t filter_height, filter_width, output_channels, filter_input_channels;
  switch (options->filterLayout().AsEnum()) {
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      // "hwio": [height, width, input_channels/groups, output_channels]
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      filter_input_channels = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      // "ohwi": [output_channels, height, width, input_channels/groups]
      output_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      filter_input_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      // "ihwo": [input_channels/groups, height, width, output_channels]
      filter_input_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      // "oihw": [output_channels, input_channels/groups, height, width]
      output_channels = filter_shape[0];
      filter_input_channels = filter_shape[1];
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      break;
  }
  // Validate bias operand if it is present.
  if (options->hasBias()) {
    const auto bias_shape = options->bias()->Dimensions();
    if (bias_shape.size() != 1) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The bias should be a 1-D tensor.");
      return nullptr;
    }
    if (bias_shape[0] != output_channels) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("The bias shape should be [%u].", output_channels));
      return nullptr;
    }
    if (options->bias()->Type() != input->Type()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The bias type doesn't match input type.");
      return nullptr;
    }
  }
  // Validate groups.
  if (options->groups() == 0) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups should be greater than 0.");
    return nullptr;
  }
  if (input_channels % options->groups() != 0 ||
      filter_input_channels != input_channels / options->groups()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups must evenly divide the input "
                                      "channels to filter input channels.");
    return nullptr;
  }

  const auto output_sizes = ValidateAndCalculateConv2dOutputSizes(
      input_height, input_width, filter_height, filter_width,
      // If padding is not present, the values are assumed to be [0,0,0,0].
      options->getPaddingOr({0, 0, 0, 0}),
      // If strides is not present, the values are assumed to be [1,1].
      options->getStridesOr({1, 1}),
      // If dilations is not present, the values are assumed to be [1, 1].
      options->getDilationsOr({1, 1}), options->autoPad(), false, exception_state);
  if (!output_sizes) {
    return nullptr;
  }
  const uint32_t output_height =
      base::ClampFloor<uint32_t>(output_sizes.value().height);
  const uint32_t output_width =
      base::ClampFloor<uint32_t>(output_sizes.value().width);
  // The input layout option specifies the layout format of the output tensor.
  Vector<uint32_t> output_shape;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, output_channels, height, width]
      output_shape = {input_batches, output_channels, output_height,
                      output_width};
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, output_channels]
      output_shape = {input_batches, output_height, output_width,
                      output_channels};
      break;
  }
  // Create conv2d operator and its output operand. Connect the conv2d
  // operator to its input and output operands.
  auto* conv2d = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kConv2d, options);
  HeapVector<Member<const MLOperand>> inputs = {input, filter};
  if (options->hasBias()) {
    inputs.push_back(options->bias());
  }
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(output_shape), conv2d, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  conv2d->Connect(std::move(inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::convTranspose2d(const MLOperand* input,
                                  const MLOperand* filter,
                                  const MLConvTranspose2dOptions* options,
                                  ExceptionState& exception_state) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "Not implemented.");

    // TODO:::COMPLETE stub
    return nullptr;

#if 0 // TODO:
  // Validate input operand and set its sizes.
  const auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input should be a 4-D tensor.");
    return nullptr;
  }
  // The input layout option specifies the layout format of the input tensor.
  uint32_t input_batches, input_channels, input_height, input_width;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, input_channels, height, width]
      input_batches = input_shape[0];
      input_channels = input_shape[1];
      input_height = input_shape[2];
      input_width = input_shape[3];
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, input_channels]
      input_batches = input_shape[0];
      input_height = input_shape[1];
      input_width = input_shape[2];
      input_channels = input_shape[3];
      break;
  }

  // Validate filter operand and set its sizes.
  if (filter->Type() != input->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The filter type doesn't match the input type.");
    return nullptr;
  }
  const auto filter_shape = filter->Dimensions();
  if (filter_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The filter should be a 4-D tensor.");
    return nullptr;
  }
  // The filter layout specifies the filter layout format.
  uint32_t filter_height, filter_width, output_channels, filter_input_channels;
  switch (options->filterLayout().AsEnum()) {
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      // "hwio": [height, width, input_channels/groups, output_channels]
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      filter_input_channels = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      // "ohwi": [output_channels, height, width, input_channels/groups]
      output_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      filter_input_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      // "ihwo": [input_channels/groups, height, width, output_channels]
      filter_input_channels = filter_shape[0];
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      // "oihw": [output_channels, input_channels/groups, height, width]
      output_channels = filter_shape[0];
      filter_input_channels = filter_shape[1];
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      break;
  }
  // Validate bias operand if it is present.
  if (options->hasBias()) {
    const auto bias_shape = options->bias()->Dimensions();
    if (bias_shape.size() != 1) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The bias should be a 1-D tensor.");
      return nullptr;
    }
    if (bias_shape[0] != output_channels) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("The bias shape should be [%u].", output_channels));
      return nullptr;
    }
    if (options->bias()->Type() != input->Type()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The bias type doesn't match input type.");
      return nullptr;
    }
  }
  // Validate groups.
  if (options->groups() == 0) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups should be greater than 0.");
    return nullptr;
  }
  if (input_channels % options->groups() != 0 ||
      filter_input_channels != input_channels / options->groups()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The groups must evenly divide the input "
                                      "channels to filter input channels.");
    return nullptr;
  }

  const auto output_sizes = ValidateAndCalculateConv2dOutputSizes(
      input_height, input_width, filter_height, filter_width,
      // If padding is not present, the values are assumed to be [0,0,0,0].
      options->getPaddingOr({0, 0, 0, 0}),
      // If strides is not present, the values are assumed to be [1,1].
      options->getStridesOr({1, 1}),
      // If dilations is not present, the values are assumed to be [1, 1].
      options->getDilationsOr({1, 1}), options->autoPad(), exception_state);
  if (!output_sizes) {
    return nullptr;
  }
  const uint32_t output_height =
      base::ClampFloor<uint32_t>(output_sizes.value().height);
  const uint32_t output_width =
      base::ClampFloor<uint32_t>(output_sizes.value().width);
  // The input layout option specifies the layout format of the output tensor.
  Vector<uint32_t> output_shape;
  switch (options->inputLayout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      // "nchw": [batches, output_channels, height, width]
      output_shape = {input_batches, output_channels, output_height,
                      output_width};
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      // "nhwc": [batches, height, width, output_channels]
      output_shape = {input_batches, output_height, output_width,
                      output_channels};
      break;
  }
  // Create conv2d operator and its output operand. Connect the conv2d
  // operator to its input and output operands.
  auto* conv2d = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kConv2d, options);
  HeapVector<Member<const MLOperand>> inputs = {input, filter};
  if (options->hasBias()) {
    inputs.push_back(options->bias());
  }
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(output_shape), conv2d, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  conv2d->Connect(std::move(inputs), {output});
  return output;
#endif
}

#define BUILD_ELEMENTWISE_BINARY_OP(op, op_kind)                              \
  MLOperand* MLGraphBuilder::op(const MLOperand* a, const MLOperand* b,       \
                                ExceptionState& exception_state) {            \
    return BuildElementwiseBinary(this, MLOperator::OperatorKind::op_kind, a, \
                                  b, a->Type(), exception_state);             \
  }

BUILD_ELEMENTWISE_BINARY_OP(add, kAdd)
BUILD_ELEMENTWISE_BINARY_OP(sub, kSub)
BUILD_ELEMENTWISE_BINARY_OP(mul, kMul)
BUILD_ELEMENTWISE_BINARY_OP(div, kDiv)
BUILD_ELEMENTWISE_BINARY_OP(min, kMin)
BUILD_ELEMENTWISE_BINARY_OP(max, kMax)

MLOperand* MLGraphBuilder::gemm(const MLOperand* a,
                                const MLOperand* b,
                                const MLGemmOptions* options,
                                ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The types of first two inputs don't match.");
    return nullptr;
  }
  // According to WebNN spec:
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gemm, the first input 2-D
  // tensor with shape [M, K] if aTranspose is false, or [K, M] if aTranspose
  // is true.
  auto shape_a = a->Dimensions();
  if (shape_a.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The first input must be a 2-D tensor.");
    return nullptr;
  }
  if (options->aTranspose()) {
    shape_a.Reverse();
  }
  // The second input 2-D tensor with shape [K, N] if bTranspose is false, or
  // [N, K] if bTranspose is true.
  auto shape_b = b->Dimensions();
  if (shape_b.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The second input must be a 2-D tensor.");
    return nullptr;
  }
  if (options->bTranspose()) {
    shape_b.Reverse();
  }
  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix.
  if (shape_a[1] != shape_b[0]) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        String::Format(
            "The number of columns (%u) in the %sfirst matrix isn't equal to "
            "the number of rows (%u) in the %ssecond matrix.",
            shape_a[1], options->aTranspose() ? "transposed " : "", shape_b[0],
            options->bTranspose() ? "transposed " : ""));
    return nullptr;
  }

  // The output is 2-D tensor of shape [M, N].
  Vector<uint32_t> output_shape = {shape_a[0], shape_b[1]};
  // The third input tensor c is either a scalar, or of the shape that is
  // unidirectionally broadcastable to the output shape [M, N].
  if (options->hasC()) {
    const auto* c = options->c();
    if (c->Type() != a->Type()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The third input type doesn't match other inputs' type.");
      return nullptr;
    }
    const auto shape_c = options->c()->Dimensions();
    if (shape_c.size() > 2) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The third input tensor should be "
                                        "either a scalar or a 2-D tensor.");
      return nullptr;
    }
    if (!BroadcastShapes(shape_c, output_shape, false)) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The third input tensor isn't unidirectionally broadcastable to "
          "the "
          "output tensor.");
      return nullptr;
    }
  }
  auto* gemm = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kGemm, options);
  HeapVector<Member<const MLOperand>> inputs = {a, b};
  if (options->hasC()) {
    inputs.push_back(options->c());
  }
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, a->Type(), std::move(output_shape), gemm, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  gemm->Connect(std::move(inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::hardSwish(const MLOperand* input,
                                     ExceptionState& exception_state) {
  // The input type must be one of the floating point types. Although this
  // constraint is not specified in current WebNN spec, there is a feature
  // request for that: https://github.com/webmachinelearning/webnn/issues/283
  if (!IsFloatingPointType(input->Type())) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input type must be one of the floating point types.");
    return nullptr;
  }
  auto* hard_swish = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kHardSwish);
  // According to WebNN spec
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-hard-swish, the output
  // tensor of hard-swish has the same type and dimensions as its input.
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), input->Dimensions(), hard_swish, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  hard_swish->Connect({input}, {output});
  return output;
}

MLOperator* MLGraphBuilder::hardSwish(ExceptionState& exception_state) {
  // Create the hard-swish operator that would be used as an activation
  // function.
  return MakeGarbageCollected<MLOperator>(this,
                                          MLOperator::OperatorKind::kHardSwish);
}

MLOperand* MLGraphBuilder::averagePool2d(const MLOperand* input,
                                         const MLPool2dOptions* options,
                                         ExceptionState& exception_state) {
  return BuildPool2d(this, MLOperator::OperatorKind::kAveragePool2d, input,
                     options, exception_state);
}

MLOperand* MLGraphBuilder::maxPool2d(const MLOperand* input,
                                     const MLPool2dOptions* options,
                                     ExceptionState& exception_state) {
  return BuildPool2d(this, MLOperator::OperatorKind::kMaxPool2d, input, options,
                     exception_state);
}

MLOperand* MLGraphBuilder::relu(const MLOperand* input,
                                ExceptionState& exception_state) {
  auto* relu =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OperatorKind::kRelu);
  // According to WebNN spec
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-relu, the output tensor of
  // relu has the same type and dimensions as its input.
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), input->Dimensions(), relu, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  relu->Connect({input}, {output});
  return output;
}

MLOperator* MLGraphBuilder::relu(ExceptionState& exception_state) {
  // Create the relu operator that would be used as an activation function.
  return MakeGarbageCollected<MLOperator>(this,
                                          MLOperator::OperatorKind::kRelu);
}

MLOperand* MLGraphBuilder::reshape(const MLOperand* input,
                                   const Vector<int32_t>& new_shape,
                                   ExceptionState& exception_state) {
  bool has_minus1 = false;
  wtf_size_t minus1_dim_index = 0;
  base::CheckedNumeric<size_t> checked_newshape_number_of_elements = 1;
  Vector<uint32_t> output_shape;
  if (new_shape.size() == 0) {
    // The empty new shape means reshaping to scalar, set output shape to {1}.
    output_shape = {1};
  } else {
    output_shape.resize(new_shape.size());
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-reshape, only one
    // component of new shape can be the special value of -1.
    for (wtf_size_t i = 0; i < new_shape.size(); ++i) {
      auto d = new_shape[i];
      // TODO:::DELETE this special -1 behavior per the pending issue
      // https://github.com/webmachinelearning/webnn/issues/388
      // to remove magic values like null or -1.
#if 1 // But keep it for now because Wanming may need it for the WebNN EP.
      if (d < -1 || d == 0) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The value of new shape should be positive or -1.");
        return nullptr;
      } else if (d == -1) {
        if (has_minus1) {
          exception_state.ThrowDOMException(
              DOMExceptionCode::kDataError,
              "Only one component of new shape can be -1.");
          return nullptr;
        }
        has_minus1 = true;
        minus1_dim_index = i;
#else
      if (d <= 0) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The value of new shape may not have 0 in it.");
        return nullptr;
#endif
      } else {
        checked_newshape_number_of_elements *= d;
        output_shape[i] = d;
      }
    }
  }
  size_t newshape_number_of_elements;
  if (!checked_newshape_number_of_elements.AssignIfValid(
          &newshape_number_of_elements)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The number of elements in the new shape is too large.");
    return nullptr;
  }
  DCHECK_NE(newshape_number_of_elements, size_t(0));
  if (has_minus1) {
    // The size of the dimension with the value -1 is computed so that the total
    // size remains constant.
    if (input->NumberOfElements() % newshape_number_of_elements != size_t(0)) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "The number of elements (%zu) in the input tensor can't be "
              "divided evenly by the number of elements (%zu) in the "
              "new shape.",
              input->NumberOfElements(), newshape_number_of_elements));
      return nullptr;
    }
    // Check whether the quotient of type size_t is in the range of dimension of
    // type uint32_t.
    if (!base::CheckDiv(input->NumberOfElements(), newshape_number_of_elements)
             .AssignIfValid(&output_shape[minus1_dim_index])) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The size of dimension with the value -1 is too large.");
      return nullptr;
    }
  } else {
    // The number of elements implied by new shape must be the same as the
    // number of elements in the input tensor.
    if (input->NumberOfElements() != newshape_number_of_elements) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "The number of elements (%zu) in the new shape doesn't match "
              "the number of elements (%zu) in the input tensor.",
              newshape_number_of_elements, input->NumberOfElements()));
      return nullptr;
    }
  }
  auto* reshape = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kReshape);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(output_shape), reshape, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  reshape->Connect({input}, {output});
  return output;
}

MLOperand* MLGraphBuilder::resample2d(const MLOperand* input,
                                      const MLResample2dOptions* options,
                                      ExceptionState& exception_state) {
  // According to WebNN spec:
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d, the input
  // must be a 4-D tensor.
  const auto& input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input must be a 4-D tensor.");
    return nullptr;
  }

  const auto axes = options->getAxesOr({2, 3});
  const wtf_size_t input_rank = input_shape.size();
  if (!ValidateAxes(axes, input_rank, "resample2d", exception_state)) {
    return nullptr;
  }

  if (axes.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of axes should be 2.");
    return nullptr;
  } else if (!((axes[0] == 0 && axes[1] == 1) ||
               (axes[0] == 1 && axes[1] == 2) ||
               (axes[0] == 2 && axes[1] == 3))) {
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d,
    // the valid values in the sequence are [0, 1], [1, 2] or [2, 3].
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The values of axes are invalid.");
    return nullptr;
  }

  Vector<uint32_t> output_shape(input_shape);
  Vector<float> scales(axes.size(), 1.0f);

  if (options->hasSizes()) {
    if (options->hasScales()) {
      auto* execution_context = GetContext()->GetML()->GetExecutionContext();
      if (!execution_context) {
        exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                          "Execution context is invalid.");
        return nullptr;
      }
      execution_context->AddConsoleMessage(MakeGarbageCollected<ConsoleMessage>(
          mojom::blink::ConsoleMessageSource::kJavaScript,
          mojom::blink::ConsoleMessageLevel::kWarning,
          "When sizes and scales are both specified, scales argument is "
          "ignored."));
    }
    if (options->sizes().size() != 2) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The length of sizes should be 2.");
      return nullptr;
    } else if (std::any_of(options->sizes().begin(), options->sizes().end(),
                           [](uint32_t x) { return x == 0; })) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "All sizes should be greater than 0.");
      return nullptr;
    }

    output_shape[axes[0]] = options->sizes()[0];
    output_shape[axes[1]] = options->sizes()[1];

    // Compute the scales from the new shape.
    for (wtf_size_t i = 0; i < input_rank; ++i)
    {
      scales[i] = static_cast<float>(output_shape[i]) / input_shape[i];
    }
  } else {
    scales = options->getScalesOr({1.0f, 1.0f});
    if (scales.size() != 2) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The length of scales should be 2.");
      return nullptr;
    } else if (std::any_of(scales.begin(), scales.end(),
                           [](float x) { return x <= 0.0f; })) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "All scales should be greater than 0.");
      return nullptr;
    }
    base::CheckedNumeric<uint32_t> checked_output_height =
        input_shape[axes[0]] * scales[0];
    if (!checked_output_height.AssignIfValid(&output_shape[axes[0]])) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The scale height is too large.");
      return nullptr;
    }
    base::CheckedNumeric<uint32_t> checked_output_width =
        input_shape[axes[1]] * scales[1];
    if (!checked_output_width.AssignIfValid(&output_shape[axes[1]])) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The scale width is too large.");
      return nullptr;
    }
  }

  // Pass the normalized options onward, simplifying the lower level's job.
  // Then the axes parameter and scales consistently exist.
  MLResample2dOptions* normalized_options = MLResample2dOptions::Create();
  normalized_options->setAxes(axes);
  normalized_options->setScales(scales);
  // Do not set sizes, since the output shape is already set,
  // and since it would potentially conflict with scales.

  auto* resample2d = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kResample2d, normalized_options);
  String error_message;
  // According to WebNN spec
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d, the output
  // tensor of resample2d has the same type as its input.
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(output_shape), resample2d, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  resample2d->Connect({input}, {output});
  return output;
}

MLOperand* MLGraphBuilder::softmax(const MLOperand* input,
                                   ExceptionState& exception_state) {
  // According to WebNN spec:
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax, The input must
  // be a 2-D tensor.
  if (input->Dimensions().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input must be a 2-D tensor.");
    return nullptr;
  }
  // The input type must be one of the floating point types.
  if (!IsFloatingPointType(input->Type())) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input type must be one of the floating point types.");
    return nullptr;
  }
  auto* softmax = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kSoftmax);
  // The output tensor has the same shape as the input tensor.
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), input->Dimensions(), softmax, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  softmax->Connect({input}, {output});
  return output;
}

MLOperand* MLGraphBuilder::sigmoid(const MLOperand* input,
                                   ExceptionState& exception_state) {
  auto* sigmoid = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kSigmoid);
  // According to WebNN spec
  // https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-sigmoid, the
  // output tensor of sigmoid has the same type and dimensions as its input.
  // And the input type must be one of the floating point types.
  if (!IsFloatingPointType(input->Type())) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input type must be one of the floating point types.");
    return nullptr;
  }
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), input->Dimensions(), sigmoid, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  sigmoid->Connect({input}, {output});
  return output;
}

MLOperator* MLGraphBuilder::sigmoid(ExceptionState& exception_state) {
  // Create the sigmoid operator that would be used as an activation function.
  return MakeGarbageCollected<MLOperator>(this,
                                          MLOperator::OperatorKind::kSigmoid);
}

MLOperand* MLGraphBuilder::elementwiseIf(const MLOperand* condition,
                                         const MLOperand* true_value,
                                         const MLOperand* false_value,
                                         ExceptionState& exception_state) {
  if (!IsBooleanType(condition->Type())) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input condition type must be a Boolean data type.");
    return nullptr;
  }

  if (true_value->Type() != false_value->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input types don't match.");
    return nullptr;
  }
  absl::optional<Vector<uint32_t>> value_dimensions =
      BroadcastShapes(true_value->Dimensions(), false_value->Dimensions());
  if (!value_dimensions) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input shapes are not broadcastable.");
    return nullptr;
  }
  absl::optional<Vector<uint32_t>> output_dimensions =
      BroadcastShapes(condition->Dimensions(), *value_dimensions);
  if (!output_dimensions) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The input shapes are not broadcastable.");
    return nullptr;
  }

  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kElementWiseIf);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(this, true_value->Type(),
                                                    output_dimensions.value(),
                                                    ml_operator, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  ml_operator->Connect({condition, true_value, false_value}, {output});
  return output;
}

MLOperand* MLGraphBuilder::argMax(const MLOperand* input,
                                  const MLArgMinMaxOptions* options,
                                  ExceptionState& exception_state) {
  return BuildArgMinMax(this, MLOperator::OperatorKind::kArgMax, "argMax", input, options,
                        exception_state);
}

MLOperand* MLGraphBuilder::argMin(const MLOperand* input,
                                  const MLArgMinMaxOptions* options,
                                  ExceptionState& exception_state) {
  return BuildArgMinMax(this, MLOperator::OperatorKind::kArgMin, "argMin", input, options,
                        exception_state);
}

MLOperand* MLGraphBuilder::cast(const MLOperand* input,
                                V8MLOperandType data_type,
                                ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kCast, input,
                            input->Dimensions(), data_type.AsEnum(), /*options*/ nullptr,
                            exception_state);
}

MLOperand* MLGraphBuilder::concat(const HeapVector<Member<MLOperand>>& inputs,
                                  uint32_t axis,
                                  ExceptionState& exception_state) {
  wtf_size_t input_count = inputs.size();
  if (input_count <= 0) {
    exception_state.ThrowDOMException(
         DOMExceptionCode::kDataError,
         "Concat requires at least one input.");
     return nullptr;
  }

  // Set the output dimensions initially to the first input,
  // concatenating each successive one in the loop below.
  auto& first_input = inputs.front();
  Vector<uint32_t> output_dimensions = first_input->Dimensions();

  if (!ValidateAxis(axis,
                    output_dimensions.size(),
                    "concat",
                    exception_state)) {
    return nullptr;
  }

  base::CheckedNumeric<size_t> checked_output_axis_length = 0;

  // Validate input dimensions are compatible with each other, and compute the
  // total length of the active axis dimension.
  for (wtf_size_t i = 0; i < input_count; ++i) {
    auto& input = inputs[i];
    auto& input_dimensions = input->Dimensions();

    if (input_dimensions.size() != output_dimensions.size()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("All input tensors must have the same size. Input %u "
                         "has a size of %u but input 0 has a size of %u.",
                         i, input_dimensions.size(), output_dimensions.size()));
      return nullptr;
    }

    checked_output_axis_length += input_dimensions[axis];
  }

  // Set the length of the active axis.
  uint32_t output_axis_length;
  if (!checked_output_axis_length.AssignIfValid(&output_axis_length)) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The number of elements in the new shape is too large.");
      return nullptr;
  }
  output_dimensions[axis] = output_axis_length;

  MLConcatOptionsInternal* options = MLConcatOptionsInternal::Create();
  options->setAxis(axis);

  String error_message;
  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kConcat, options);

  auto* output = MLOperand::ValidateAndCreateOutput(
      this, first_input->Type(), std::move(output_dimensions), ml_operator,
      /*out*/ error_message);
  if (!output) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        error_message);
      return nullptr;
  }

  HeapVector<Member<const MLOperand>> copied_inputs(inputs);
  ml_operator->Connect(std::move(copied_inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::expand(const MLOperand* input,
                                  const Vector<uint32_t>& new_shape,
                                  ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const auto new_shape_dimension_count = new_shape.size();
  base::CheckedNumeric<size_t> checked_new_shape_number_of_elements = 1;

  if (new_shape_dimension_count != input_dimensions.size()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("The new shape's dimension count (%u) must match the "
                         "input tensor's (%u).",
                         new_shape_dimension_count,
                         input->Dimensions().size()));
      return nullptr;
  }

  for (wtf_size_t i = 0; i < new_shape_dimension_count; ++i) {
      auto old_size = input_dimensions[i];
      auto new_size = new_shape[i];
      if (new_size < old_size || (new_size > old_size && old_size != 1)) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "The each value in the new shape (%u) must either equal the old "
              "shape (%u) "
              "or broadcast a single size dimension input to a greater size.",
              new_size, old_size));
      return nullptr;
      }
      checked_new_shape_number_of_elements *= new_size;
  }

  // Check for overflow.
  size_t new_shape_number_of_elements;
  if (!checked_new_shape_number_of_elements.AssignIfValid(
          &new_shape_number_of_elements)) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The number of elements in the new shape is too large.");
      return nullptr;
  }

  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kExpand);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(new_shape), ml_operator, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  ml_operator->Connect({input}, {output});
  return output;
}

MLOperand* MLGraphBuilder::cos(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kCos, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::erf(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kErf, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::exp(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kExp, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::floor(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kFloor, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::ceil(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kCeil, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::reciprocal(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kReciprocal, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::logicalNot(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kLogicalNot, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::flattenTo2d(const MLOperand* input,
                                       uint32_t axis,
                                       ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  if (!ValidateAxis(axis,
                    input_rank,
                    "flattenTo2d",
                    exception_state)) {
    return nullptr;
  }

  // Flatten the leading and trailing portion of the dimensions
  // (where axis is the split point) into a 2D tensor.
  Vector<uint32_t> output_dimensions(2, 0u);
  base::span<const uint32_t> input_dimensions_span = input_dimensions;
  base::span<const uint32_t> leading_dimensions = input_dimensions_span.first(axis);
  base::span<const uint32_t> trailing_dimensions = input_dimensions_span.subspan(axis);
  output_dimensions[0] = ComputeElementCount(leading_dimensions);
  output_dimensions[1] = ComputeElementCount(trailing_dimensions);

  // Resolve flattenTo2d into a reshape operator.
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kReshape, input,
                            output_dimensions, input->Type(), /*options*/nullptr,
                            exception_state);
}

MLOperand* MLGraphBuilder::gather(const MLOperand* input,
                                  const MLOperand* indices,
                                  const MLGatherOptions* options,
                                  ExceptionState& exception_state) {
  wtf_size_t input_rank = input->Dimensions().size();
  wtf_size_t indices_rank = indices->Dimensions().size(); // >= 0
  wtf_size_t output_rank = input_rank + indices_rank - 1;
  uint32_t axis = options->axis();

  if (!IsIndexType(indices->Type())) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Gather's indices element type must be int32/uint32.");
    return nullptr;
  }

  if (input_rank <= 0) {
    exception_state.ThrowDOMException(
         DOMExceptionCode::kDataError,
          String::Format(
              "Gather's input rank (%u) requires at least 1 dimension.",
              input_rank));
     return nullptr;
  }
  if (axis >= input_rank) {
     exception_state.ThrowDOMException(
         DOMExceptionCode::kDataError,
         String::Format(
             "Gather's axis (%u) must be within the input tensor rank (%u).",
             axis, input_rank));
     return nullptr;
  }
  if (input_rank + indices_rank < 1) {
     exception_state.ThrowDOMException(
         DOMExceptionCode::kDataError,
         String::Format("Gather's input rank (%u) and indices rank (%u) "
                        "combined must be at least 1.",
                        input_rank, indices_rank));
     return nullptr;
  }

  const Vector<uint32_t>& inputDimensions = input->Dimensions();
  const Vector<uint32_t>& indicesDimensions = indices->Dimensions();
  Vector<uint32_t> output_dimensions(output_rank, 1u);

  // The input dimensions following the gather axis determine the final output
  // dimensions.
  int32_t output_dimension = output_rank - 1;
  int32_t input_dimension = input_rank - 1;
  for (; input_dimension > int32_t(axis);
       --output_dimension, --input_dimension) {
     output_dimensions[output_dimension] = inputDimensions[input_dimension];
  }

  // The shape of the index tensor is reflected in the middle dimensions of the
  // output tensor.
  int32_t index_dimension = indices_rank - 1;
  for (; index_dimension >= 0; --output_dimension, --index_dimension) {
     output_dimensions[output_dimension] = indicesDimensions[index_dimension];
  }

  // The gather dimension is skipped for the purposes of sizing because the
  // index values choose slices across it. Preceding input dimensions
  // determine the shape of the output's leading dimensions.
  input_dimension = axis - 1;
  for (; output_dimension >= 0 && input_dimension >= 0;
       --output_dimension, --input_dimension) {
     output_dimensions[output_dimension] = inputDimensions[input_dimension];
  }

  String error_message;
  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kGather, options);

  auto* output = MLOperand::ValidateAndCreateOutput(
      this, input->Type(), std::move(output_dimensions), ml_operator,
      /*out*/ error_message);
  if (!output) {
     exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                       error_message);
     return nullptr;
  }

  ml_operator->Connect({input, indices}, {output});
  return output;
}

MLOperand* MLGraphBuilder::equal(const MLOperand* a,
                                 const MLOperand* b,
                                 ExceptionState& exception_state) {
  return BuildElementwiseBinary(this, MLOperator::OperatorKind::kEqual, a, b,
                                V8MLOperandType::Enum::kUint8,
                                exception_state);
}

MLOperand* MLGraphBuilder::greater(const MLOperand* a,
                                   const MLOperand* b,
                                   ExceptionState& exception_state) {
  return BuildElementwiseBinary(this, MLOperator::OperatorKind::kGreater, a, b,
                                V8MLOperandType::Enum::kUint8,
                                exception_state);
}

MLOperand* MLGraphBuilder::lesser(const MLOperand* a,
                                   const MLOperand* b,
                                   ExceptionState& exception_state) {
  return BuildElementwiseBinary(this, MLOperator::OperatorKind::kLesser, a, b,
                                V8MLOperandType::Enum::kUint8,
                                exception_state);
}

MLOperand* MLGraphBuilder::identity(const MLOperand* input,
                                    ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kIdentity, input,
                            exception_state);
}

MLOperand* MLGraphBuilder::instanceNormalization(
    const MLOperand* input,
    const MLInstanceNormalizationOptions* options,
    ExceptionState& exception_state) {
  auto& input_dimensions = input->Dimensions();

  if (input_dimensions.size() != 4)
  {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "instanceNormalization's input tensor rank (%u) should be 4D.",
              input_dimensions.size()));
      return nullptr;
  }

  // Determine which axis of the input holds the feature axis.
  static_assert(uint32_t(V8MLInputOperandLayout::kEnumSize) == 2, "Update switch for the new layout.");
  uint32_t scale_bias_dimension = 0;
  switch (options->layout().AsEnum())
  {
  case V8MLInputOperandLayout::Enum::kNchw:
      scale_bias_dimension = 1; // Channel after batch.
      break;
  case V8MLInputOperandLayout::Enum::kNhwc:
      scale_bias_dimension = 3; // Channel last.
      break;
  default:
      NOTREACHED();
  }

  const uint32_t expected_scale_bias_length = input_dimensions[scale_bias_dimension];

  // Expect a 1D array for scale and bias, equal to the feature axis.
  auto verify_scale_or_bias = [&](MLOperand& ml_operand, const char* tensor_name)
  {
    auto& dimensions = ml_operand.Dimensions();
    if (dimensions.size() != 1 || dimensions.front() != expected_scale_bias_length)
    {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format(
              "instanceNormalization's %s tensor rank (%u) should be 4D.",
              tensor_name,
              dimensions.size()));
      return false;
    }
    return true;
  };

  // Collect the inputs, with optional scale and bias.
  HeapVector<Member<const MLOperand>> inputs = {input};

  if (options->hasScale()) {
    if (!verify_scale_or_bias(*options->scale(), "scale"))
    {
        return nullptr;
    }
    inputs.push_back(options->scale());
  }
  if (options->hasBias()) {
    if (!verify_scale_or_bias(*options->bias(), "bias"))
    {
        return nullptr;
    }
    inputs.push_back(options->bias());
  }

  // Create the instance normalization operator, and connect IO.
  auto* ml_operator = MakeGarbageCollected<MLOperator>(this, MLOperator::OperatorKind::kInstanceNormalization, options);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(this, input->Type(), input_dimensions, ml_operator, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError, error_message);
    return nullptr;
  }
  ml_operator->Connect(std::move(inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::meanVarianceNormalization(
    const MLOperand* input,
    const MLMeanVarianceNormalizationOptions* options,
    ExceptionState& exception_state) {
  auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  // Verify the axes are within the input rank and not duplicated.
  Vector<uint32_t> axes;
  if (options->hasAxes())
  {
    axes = options->axes();
    if (!ValidateAxes(axes, input_rank, "meanVarianceNormalization", exception_state)) {
      return nullptr;
    }
  }
  else // Reduce all dimensions if permutations are missing, consistent with reduction operators.
  {
    axes.resize(input_rank);
    std::iota(axes.begin(), axes.end(), 0u);
  }

  // Expect a 1D array for scale and bias, equal to the feature axis.
  auto verify_scale_or_bias = [&](MLOperand& ml_operand,
                                  const char* tensor_name) {
    auto& dimensions = ml_operand.Dimensions();
    if (dimensions.size() != input_dimensions.size()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          String::Format("meanVarianceNormalization's %s tensor must "
                         "match the input tensor rank (%u), not (%u).",
                         tensor_name, input_dimensions.size(),
                         dimensions.size()));
      return false;
    }

    for (wtf_size_t i = 0, rank = input_dimensions.size(); i < rank; ++i) {
      if (dimensions[i] != input_dimensions[i] && dimensions[i] != 1) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format("meanVarianceNormalization's %s tensor dimension (%u) "
                           "must be either 1 or match the corresponding input "
                           "dimension (%u).",
                           tensor_name, dimensions[i], input_dimensions[i]));
        return false;
      }
    }
    return true;
  };

  // Collect the inputs, with optional scale and bias.
  HeapVector<Member<const MLOperand>> inputs = {input};

  if (options->hasScale()) {
    if (!verify_scale_or_bias(*options->scale(), "scale")) {
      return nullptr;
    }
    inputs.push_back(options->scale());
  }
  if (options->hasBias()) {
    if (!verify_scale_or_bias(*options->bias(), "bias"))
    {
        return nullptr;
    }
    inputs.push_back(options->bias());
  }

  // Pass the normalized options onward, simplifying the lower level's job.
  MLMeanVarianceNormalizationOptions* normalized_options = MLMeanVarianceNormalizationOptions::Create();
  normalized_options->setScale(options->scale());
  normalized_options->setBias(options->bias());
  normalized_options->setEpsilon(options->epsilon());
  normalized_options->setAxes(axes);

  // Create the mean variance normalization operator, and connect IO.
  auto* ml_operator = MakeGarbageCollected<MLOperator>(this, MLOperator::OperatorKind::kMeanVarianceNormalization, normalized_options);
  String error_message;
  auto* output = MLOperand::ValidateAndCreateOutput(this, input->Type(), input_dimensions, ml_operator, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError, error_message);
    return nullptr;
  }
  ml_operator->Connect(std::move(inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::matmul(const MLOperand* a,
                                  const MLOperand* b,
                                  ExceptionState& exception_state) {
  // Massage the two input tensor's rank accordingly:
  // - If a is 1-D, it is converted to a 2-D tensor by prepending a 1 to its dimensions.
  // - If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to its dimensions.
  // - If either a or b have rank N > 2, the higher dimensions are broadcast to each other,
  //   with the output rank being the greater of the two.
  // Then the inputs are treated as a stack of matrices. If both were 1D, it's treated as
  // as dot product (which happens naturally as a by-product of expansion).

  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The types of first two inputs don't match.");
    return nullptr;
  }

  Vector<uint32_t> a_dimensions = a->Dimensions();
  Vector<uint32_t> b_dimensions = b->Dimensions();
  // MLGraphBuilder::input should have coerced to 1 already.
  DCHECK_GT(a_dimensions.size(), uint32_t(0));
  DCHECK_GT(b_dimensions.size(), uint32_t(0));

  // Massage the sizes first, before additional broadcastability checks.
  // After this point, both arrays are at least the same size, simplifying
  // later checks in the code.
  wtf_size_t output_rank = std::max(a_dimensions.size(), b_dimensions.size());
  if (a_dimensions.size() == 1)
  {
    a_dimensions.push_front(1u);
  }
  if (a_dimensions.size() == 1)
  {
    b_dimensions.push_back(1u);
  }
  a_dimensions = ExpandDimensions(a_dimensions, output_rank);
  b_dimensions = ExpandDimensions(b_dimensions, output_rank);

  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix.
  const uint32_t a_cols = a_dimensions[output_rank - 1];
  const uint32_t a_rows = a_dimensions[output_rank - 2];
  const uint32_t b_cols = b_dimensions[output_rank - 1];
  const uint32_t b_rows = b_dimensions[output_rank - 2];
  if (a_cols != b_rows) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        String::Format(
            "The number of columns (%u) in the first matrix isn't equal to "
            "the number of rows (%u) in the second matrix.",
            a_cols, b_rows));
    return nullptr;
  }

  // Figure out the output shape by broadcasting all the dimensions except the last two.
  // The output is 2-D tensor of shape [M, N].
  absl::optional<Vector<uint32_t>> optional_output_dimensions = BroadcastShapes(a_dimensions, b_dimensions, true, 2);
  if (!optional_output_dimensions) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError, "The matmul input shapes are not broadcastable.");
    return nullptr;
  }
  auto& output_dimensions = *optional_output_dimensions;
  DCHECK(output_rank == output_dimensions.size());
  output_dimensions[output_rank - 2] = a_rows;
  output_dimensions[output_rank - 1] = b_cols;

  String error_message;

  // Create empty options for gemm, since MatMul uses the defaults.
  auto* options = MLGemmOptions::Create();

  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kGemm, options);
  HeapVector<Member<const MLOperand>> inputs = {a, b};
  auto* output = MLOperand::ValidateAndCreateOutput(
      this, a->Type(), std::move(output_dimensions), ml_operator, error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }
  ml_operator->Connect(std::move(inputs), {output});
  return output;
}

MLOperand* MLGraphBuilder::pad(
    const MLOperand* input,
    const Vector<uint32_t>& beginningPadding,
    const Vector<uint32_t>& endingPadding,
    const MLPadOptions* options,
    ExceptionState& exception_state) {
  // TODO:::COMPLETE
  return nullptr;
}

MLOperand* MLGraphBuilder::pow(const MLOperand* a,
                               const MLOperand* b,
                               ExceptionState& exception_state) {
  return BuildElementwiseBinary(this, MLOperator::OperatorKind::kPow, a, b, a->Type(),
                               exception_state);
}

MLOperand* MLGraphBuilder::fillSequence(V8MLOperandType output_data_type,
                                        const Vector<uint32_t>& output_shape,
                                        const MLFillSequenceOptions* options,
                                        ExceptionState& exception_state) {
  String error_message;
  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kFillSequence, options);

  auto* output = MLOperand::ValidateAndCreateOutput(this, output_data_type.AsEnum(),
                                                    output_shape, ml_operator,
                                                    /*out*/ error_message);
  if (!output) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      error_message);
    return nullptr;
  }

  ml_operator->Connect({}, {output});
  return output;
}

MLOperand* MLGraphBuilder::reduceL1(const MLOperand* input,
                                    const MLReduceOptions* options,
                                    ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceL1,
                                "reduceL1", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceL2(const MLOperand* input,
                                    const MLReduceOptions* options,
                                    ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceL2,
                                "reduceL2", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceLogSum(const MLOperand* input,
                                        const MLReduceOptions* options,
                                        ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceLogSum,
                                "reduceLogSum", input, options,
                                exception_state);
}

MLOperand* MLGraphBuilder::reduceLogSumExp(const MLOperand* input,
                                           const MLReduceOptions* options,
                                           ExceptionState& exception_state) {
  return BuildReductionOperator(
      this, MLOperator::OperatorKind::kReduceLogSumExp, "reduceLogSumExp",
      input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceMax(const MLOperand* input,
                                     const MLReduceOptions* options,
                                     ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceMax,
                                "reduceMax", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceMean(const MLOperand* input,
                                      const MLReduceOptions* options,
                                      ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceMean,
                                "reduceMean", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceMin(const MLOperand* input,
                                     const MLReduceOptions* options,
                                     ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceMin,
                                "reduceMin", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceProduct(const MLOperand* input,
                                         const MLReduceOptions* options,
                                         ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceProduct,
                                "reduceProduct", input, options,
                                exception_state);
}

MLOperand* MLGraphBuilder::reduceSum(const MLOperand* input,
                                     const MLReduceOptions* options,
                                     ExceptionState& exception_state) {
  return BuildReductionOperator(this, MLOperator::OperatorKind::kReduceSum,
                                "reduceSum", input, options, exception_state);
}

MLOperand* MLGraphBuilder::reduceSumSquare(const MLOperand* input,
                                           const MLReduceOptions* options,
                                           ExceptionState& exception_state) {
  return BuildReductionOperator(
      this, MLOperator::OperatorKind::kReduceSumSquare, "reduceSumSquare",
      input, options, exception_state);
}

Vector<uint32_t> MLGraphBuilder::shape(
    const MLOperand* input,
    ExceptionState& exception_state) {
  return input->Dimensions();
}

MLOperand* MLGraphBuilder::sin(const MLOperand* input,
                               ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kSin, input,
                               exception_state);
}

MLOperand* MLGraphBuilder::slice(const MLOperand* input,
                                 const Vector<uint32_t>& starts,
                                 const Vector<uint32_t>& sizes,
                                 ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  if (starts.size() != input_rank || sizes.size() != input_rank) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        String::Format("Slice's starts length (%u) and sizes length (%u) must "
                       "match the input rank (%u).",
                       starts.size(), sizes.size(), input_rank));
    return nullptr;
  }

  // Ensure starts and sizes are within valid dimensions.
  for (wtf_size_t i = 0; i < input_rank; ++i)
  {
      uint32_t dimension_length = input_dimensions[i];
      if (sizes[i] > dimension_length ||
          sizes[i] == 0 ||
          starts[i] > dimension_length - sizes[i])
      {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format("Slice's starts (%u) and sizes (%u) must fit within the dimension length (%u) and be non-empty.",
                           starts[i], sizes[i], dimension_length));
        return nullptr;
      }
  }

  MLSliceOptionsInternal* options = MLSliceOptionsInternal::Create();
  options->setStarts(starts);
  options->setSizes(sizes);

  return BuildUnaryOperator(this, MLOperator::OperatorKind::kSlice, input,
                            sizes, input->Type(), /*options*/ options,
                            exception_state);
}

HeapVector<Member<MLOperand>> MLGraphBuilder::split(
    const MLOperand* input,
    blink::V8UnionUnsignedLongOrUnsignedLongSequence* splits,
    const MLSplitOptions* options,
    ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();
  const uint32_t axis = options->hasAxis() ? options->axis() : 0;

  if (!ValidateAxis(axis, input_rank, "split", exception_state)) {
      return {};
  }

  const auto is_split_single_scalar = splits->IsUnsignedLong();
  const auto input_axis_length = input_dimensions[axis];
  Vector<uint32_t> resolved_splits;

  if (is_split_single_scalar) {
      // Convert from a single split count to a splits array.
      const wtf_size_t split_count = splits->GetAsUnsignedLong();
      if (split_count <= 0 || split_count > input_axis_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format("The split count (%u) must be greater than 0 and no "
                           "greater than the axis length (%u).",
                           split_count, input_axis_length));
        return {};
      }

      uint32_t axis_length_per_output = input_axis_length / split_count;
      if (input_axis_length % split_count != 0) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format("The split count (%u) must divide evenly into the "
                           "axis length (%u).",
                           split_count, input_axis_length));
        return {};
      }
      resolved_splits.resize(split_count);
      resolved_splits.Fill(axis_length_per_output);
  } else {
      base::CheckedNumeric<size_t> checked_output_axis_length = 0;

      resolved_splits = splits->GetAsUnsignedLongSequence();
      if (resolved_splits.empty()) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The split count must be greater than 0.");
        return {};
      }

      for (wtf_size_t i = 0, split_count = resolved_splits.size();
           i < split_count; ++i) {
        const auto split_size = resolved_splits[i];
        if (split_size <= 0) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format(
                "All split sizes must be >= 0 but index %u is zero.", i));
        return {};
        }

        checked_output_axis_length += split_size;
      }

      // Set the length of the active axis.
      uint32_t output_axis_length;
      if (!checked_output_axis_length.AssignIfValid(&output_axis_length)) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The number of split values is too large.");
        return {};
      }

      if (output_axis_length != input_axis_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            String::Format("The sum total of split sizes (%u) must equal the "
                           "input axis length (%u).", output_axis_length,
                           input_axis_length));
        return {};
      }
  }

  // Normalize the options so backends have a simpler time.
  MLSplitOptionsInternal* normalized_options = MLSplitOptionsInternal::Create();
  normalized_options->setAxis(axis);
  normalized_options->setSplits(resolved_splits);

  auto* ml_operator = MakeGarbageCollected<MLOperator>(
      this, MLOperator::OperatorKind::kSplit, normalized_options);

  // Create multiple output tensors.
  String error_message;
  HeapVector<Member<MLOperand>> outputs;

  for (wtf_size_t i = 0, split_count = resolved_splits.size(); i < split_count; ++i) {
      Vector<uint32_t> output_dimensions = input_dimensions;
      output_dimensions[axis] = resolved_splits[i];

      MLOperand* output = MLOperand::ValidateAndCreateOutput(
          this, input->Type(), std::move(output_dimensions), ml_operator,
          /*out*/ error_message);
      if (!output) {
        exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                          error_message);
        return {};
      }
      outputs.push_back(output);
  }

  HeapVector<Member<const MLOperand>> copied_outputs(outputs);
  ml_operator->Connect({input}, std::move(copied_outputs));
  return outputs;
}

MLOperand* MLGraphBuilder::sqrt(const MLOperand* input,
                                ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kSqrt, input,
                               exception_state);
}

MLOperand* MLGraphBuilder::squeeze(
    const MLOperand* input,
    const MLSqueezeOptions* options,
    ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  uint32_t axes_mask = 0xFFFFFFFF; // Remove all axes by default, if none passed.
  if (options->hasAxes()) {
    auto& axes = options->axes();
    if (!ValidateAxes(axes, input_rank, "squeeze", exception_state)) {
      return nullptr;
    }
    if (!ValidateAxesMask(options->axes(),
                          "squeeze",
                          exception_state,
                          /*out*/ axes_mask)) {
      return nullptr;
    }
  }

  // Strip any dimensions of size 1 from the output.
  Vector<uint32_t> output_dimensions = input_dimensions;
  for (wtf_size_t i = 0, output_rank = output_dimensions.size(); i < output_rank; )
  {
    if (axes_mask & (1 << i) && output_dimensions[i] == 1u) {
      output_dimensions.EraseAt(i);
      --output_rank;
    }
    else
    {
      ++i; // Preserve this dimension.
    }
  }

  // Resolve squeeze into a reshape operator.
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kReshape, input,
                            output_dimensions, input->Type(), /*options*/nullptr,
                            exception_state);
}

MLOperand* MLGraphBuilder::tan(const MLOperand* input,
                                ExceptionState& exception_state) {
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kTan, input,
                               exception_state);
}

MLOperand* MLGraphBuilder::transpose(
    const MLOperand* input,
    const MLTransposeOptions* options,
    ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  Vector<uint32_t> permutation;

  // Verify the permutations are within the input rank and not duplicated.
  if (options->hasPermutation())
  {
    permutation = options->permutation();
    if (permutation.size() != input_rank)
    {
        exception_state.ThrowDOMException(
             DOMExceptionCode::kDataError,
              String::Format(
                  "Transposes's permutation rank (%u) must must match the input rank (%u).",
                  permutation.size(), input_rank));
         return nullptr;
    }

    Vector<uint32_t> seen_axes(input_rank);

    if (!ValidateAxes(permutation, input_rank, "transpose", exception_state))
    {
        return nullptr;
    }
  }
  else // Reverse all dimensions if permutations are missing.
  {
    for (wtf_size_t i = 0; i < input_rank; ++i)
    {
      permutation.push_back(input_rank - i - 1);
    }
  }

  // Permute the dimensions.
  Vector<uint32_t> output_dimensions(input_rank);
  for (wtf_size_t i = 0; i < input_rank; ++i)
  {
    output_dimensions[i] = input_dimensions[permutation[i]];
  }

  // Pass the normalized options onward, simplifying the lower level's job.
  // Then the permutation parameter consistently exists.
  MLTransposeOptions* normalized_options = MLTransposeOptions::Create();
  normalized_options->setPermutation(permutation);

  return BuildUnaryOperator(this, MLOperator::OperatorKind::kTranspose, input,
                            output_dimensions, input->Type(), normalized_options,
                            exception_state);
}

MLOperand* MLGraphBuilder::triangularMatrix(
    const MLOperand* input,
    const MLTriangularMatrixOptions* options,
    ExceptionState& exception_state) {
  // TODO:::COMPLETE
  return nullptr;
}

MLOperand* MLGraphBuilder::unsqueeze(
    const MLOperand* input,
    const MLSqueezeOptions* options,
    ExceptionState& exception_state) {
  const auto& input_dimensions = input->Dimensions();
  const wtf_size_t input_rank = input_dimensions.size();

  // Verify all axes are within bounds and not duplicated.
  Vector<uint32_t> ordered_axes = options->getAxesOr({});
  if (!ValidateAxes(ordered_axes, input_rank + ordered_axes.size(), "unsqueeze", exception_state)) {
      return nullptr;
  }

  // Axes are allowed in any order, but insertion wants them in ascending order.
  std::sort(ordered_axes.begin(), ordered_axes.end());

  // Insert dimensions of size 1 into the output.
  Vector<uint32_t> output_dimensions = input_dimensions;
  for (uint32_t axis : ordered_axes)
  {
    output_dimensions.insert(axis, 1u);
  }

  // Resolve unsqueeze into a reshape operator.
  return BuildUnaryOperator(this, MLOperator::OperatorKind::kReshape, input,
                            output_dimensions, input->Type(), /*options*/nullptr,
                            exception_state);
}

ScriptPromise MLGraphBuilder::build(ScriptState* script_state,
                                    const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  auto promise = resolver->Promise();

  if (g_backend_for_testing) {
    g_backend_for_testing->BuildGraphAsyncImpl(ml_context_, named_outputs,
                                               resolver);
    return promise;
  }

#if BUILDFLAG(BUILD_WEBNN_WITH_XNNPACK)
  if (ml_context_->GetDevicePreference() == V8MLDevicePreference::Enum::kAuto ||
      ml_context_->GetDevicePreference() == V8MLDevicePreference::Enum::kCpu) {
    MLGraphXnnpack::ValidateAndBuildAsync(ml_context_, named_outputs, resolver);
    return promise;
  }
#endif

  // The Context is GPU device or low power preference, the graph is built by
  // MojoGraph object.
  if (GetContext()->GetDevicePreference() == V8MLDevicePreference::Enum::kGpu) {
    if (ml_context_->IsWebnnMojoContextEnabled()) {
      MojoGraph::ValidateAndBuildAsync(ml_context_, named_outputs, resolver);
    } else {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError,
          "The context for mojo must be enable with "
          "the option \"--enable-features=WebnnMojoContext\" in the command "
          "line"));
    }
    return promise;
  }
  resolver->Reject(MakeGarbageCollected<DOMException>(
      DOMExceptionCode::kNotSupportedError, "Not implemented"));
  return promise;
}

MLGraph* MLGraphBuilder::buildSync(ScriptState* script_state,
                                   const MLNamedOperands& named_outputs,
                                   ExceptionState& exception_state) {
  if (g_backend_for_testing) {
    return g_backend_for_testing->BuildGraphSyncImpl(
        script_state, ml_context_, named_outputs, exception_state);
  }

#if BUILDFLAG(BUILD_WEBNN_WITH_XNNPACK)
  if (ml_context_->GetDevicePreference() == V8MLDevicePreference::Enum::kAuto ||
      ml_context_->GetDevicePreference() == V8MLDevicePreference::Enum::kCpu) {
    return MLGraphXnnpack::ValidateAndBuildSync(script_state, ml_context_,
                                                named_outputs, exception_state);
  }
#endif

  // The Context is GPU device or low power preference, the graph is built by
  // MojoGraph object.
  if (GetContext()->GetDevicePreference() == V8MLDevicePreference::Enum::kGpu) {
    if (ml_context_->IsWebnnMojoContextEnabled()) {
      return MojoGraph::ValidateAndBuildSync(script_state, ml_context_,
                                             named_outputs, exception_state);
    } else {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kNotSupportedError,
          "The context for mojo must be enable with "
          "the option \"--enable-features=WebnnMojoContext\" in the command "
          "line");
      return nullptr;
    }
  }

  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
  return nullptr;
}

// static
void MLGraphBuilder::SetBackendForTesting(
    MLGraphBuilder::BackendForTesting* backend_for_testing) {
  g_backend_for_testing = backend_for_testing;
}

// static
void MLGraphBuilder::SortOperators(
    const MLNamedOperands& named_outputs,
    HeapVector<Member<const MLOperand>>& inputs,
    HeapVector<Member<const MLOperand>>& constants,
    HeapVector<Member<const MLOperator>>& sorted_operators) {
  HeapDeque<Member<const MLOperator>> operators_to_do;
  HeapHashSet<Member<const MLOperator>> operators_done;
  for (const auto& output : named_outputs) {
    operators_to_do.push_back(output.second->Operator());
  }
  while (operators_to_do.size() > 0) {
    const auto& op = operators_to_do.back();
    if (!operators_done.Contains(op.Get())) {
      bool can_add = true;
      for (const auto& input : op->Inputs()) {
        const auto* dependent_op = input->Operator();
        if (dependent_op && !operators_done.Contains(dependent_op)) {
          // As the dependent operator is not done, skip processing of this
          // operator and push the dependent operator into the to-do stack.
          can_add = false;
          operators_to_do.push_back(dependent_op);
        }
      }
      if (can_add) {
        // All dependent operators are done, process and add it into the
        // done set.
        for (const auto& input : op->Inputs()) {
          if (input->Kind() == MLOperand::kInput) {
            inputs.push_back(input.Get());
          } else if (input->Kind() == MLOperand::kConstant) {
            constants.push_back(input.Get());
          }
        }
        sorted_operators.push_back(op.Get());
        operators_done.insert(op.Get());
        operators_to_do.pop_back();
      }
    } else {
      operators_to_do.pop_back();
    }
  }
}

}  // namespace blink
