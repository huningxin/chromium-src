// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_model_info.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_transpose_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_concat_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gather_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_reduce_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_arg_min_max_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_slice_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_split_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_split_options_internal.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_resample_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_instance_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_mean_variance_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_fill_sequence_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#pragma optimize("", off) // TODO:::DELETE

namespace blink {

#define DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(/*MLOperator* */ ml_operator,       \
                                           /*uint32_t*/ expected_input_count,  \
                                           /*uint32_t*/ expected_output_count) \
  DCHECK_EQ(ml_operator->Inputs().size(), expected_input_count);               \
  DCHECK_EQ(ml_operator->Outputs().size(), expected_output_count);

base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment) {
  size_t remainder = value % aligment;
  if (remainder != 0) {
    value += aligment - remainder;
  }

  return value;
}

namespace {

using ml::webnn::mojom::blink::AutoPad;
using ml::webnn::mojom::blink::Conv2dFilterOperandLayout;
using ml::webnn::mojom::blink::OperatorType;
using ml::webnn::mojom::blink::InputOperandLayout;
using ml::webnn::mojom::blink::OperandType;
using ml::webnn::mojom::blink::OperationInfo;
using ml::webnn::mojom::blink::OperationInfoPtr;
using ml::webnn::mojom::blink::Pool2dType;
using ml::webnn::mojom::blink::RoundingType;

OperandType BlinkOperandTypeToMojo(V8MLOperandType::Enum type) {
  static_assert(int32_t(ml::webnn::mojom::blink::OperandType::kMaxValue) + 1 == 6);
  static_assert(int32_t(V8MLOperandType::kEnumSize) == 6);

  switch (type) {
    case V8MLOperandType::Enum::kFloat32:
      return OperandType::kFloat32;
    case V8MLOperandType::Enum::kFloat16:
      return OperandType::kFloat16;
    case V8MLOperandType::Enum::kInt32:
      return OperandType::kInt32;
    case V8MLOperandType::Enum::kUint32:
      return OperandType::kUint32;
    case V8MLOperandType::Enum::kInt8:
      return OperandType::kInt8;
    case V8MLOperandType::Enum::kUint8:
      return OperandType::kUint8;
  }
}

InputOperandLayout BlinkInputOperandLayoutToMojo(
    V8MLInputOperandLayout::Enum type) {
  switch (type) {
    case V8MLInputOperandLayout::Enum::kNchw:
      return InputOperandLayout::kNchw;
    case V8MLInputOperandLayout::Enum::kNhwc:
      return InputOperandLayout::kNhwc;
  }
}

Conv2dFilterOperandLayout BlinkConv2dFilterOperandLayoutToMojo(
    V8MLConv2dFilterOperandLayout::Enum type) {
  switch (type) {
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      return Conv2dFilterOperandLayout::kOihw;
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      return Conv2dFilterOperandLayout::kHwio;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      return Conv2dFilterOperandLayout::kOhwi;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      return Conv2dFilterOperandLayout::kIhwo;
  }
}

AutoPad BlinkAutoPadToMojo(V8MLAutoPad::Enum type) {
  switch (type) {
    case V8MLAutoPad::Enum::kExplicit:
      return AutoPad::kExplicit;
    case V8MLAutoPad::Enum::kSameUpper:
      return AutoPad::kSameUpper;
    case V8MLAutoPad::Enum::kSameLower:
      return AutoPad::kSameLower;
  }
}

RoundingType BlinkRoundingTypeToMojo(V8MLRoundingType::Enum type) {
  switch (type) {
    case V8MLRoundingType::Enum::kFloor:
      return RoundingType::kFloor;
    case V8MLRoundingType::Enum::kCeil:
      return RoundingType::kCeil;
  }
}

Pool2dType BlinkPool2dTypeToMojo(MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kAveragePool2d:
      return Pool2dType::kAveragePool2d;
    default:
      NOTREACHED();
      return Pool2dType::kUnknown;
  }
}

OperatorType BlinkOperatorKindToMojoType(
    MLOperator::OperatorKind type) {
  static_assert(int32_t(MLOperator::OperatorKind::kTotal) == 61);
  static_assert(int32_t(OperatorType::kMaxValue) + 1 == 61);

  // Keep these two enumerations in sync.
  // Favor readability over convention here.
  // clang-format off
  static_assert(uint32_t(MLOperator::OperatorKind::kClamp) == uint32_t(OperatorType::kClamp));
  static_assert(uint32_t(MLOperator::OperatorKind::kConv2d) == uint32_t(OperatorType::kConv2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kConvTranspose2d) == uint32_t(OperatorType::kConvTranspose2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kMatmul) == uint32_t(OperatorType::kMatmul));
  static_assert(uint32_t(MLOperator::OperatorKind::kGemm) == uint32_t(OperatorType::kGemm));
  static_assert(uint32_t(MLOperator::OperatorKind::kAdd) == uint32_t(OperatorType::kAdd));
  static_assert(uint32_t(MLOperator::OperatorKind::kSub) == uint32_t(OperatorType::kSub));
  static_assert(uint32_t(MLOperator::OperatorKind::kMul) == uint32_t(OperatorType::kMul));
  static_assert(uint32_t(MLOperator::OperatorKind::kDiv) == uint32_t(OperatorType::kDiv));
  static_assert(uint32_t(MLOperator::OperatorKind::kMax) == uint32_t(OperatorType::kMax));
  static_assert(uint32_t(MLOperator::OperatorKind::kMin) == uint32_t(OperatorType::kMin));
  static_assert(uint32_t(MLOperator::OperatorKind::kPow) == uint32_t(OperatorType::kPow));
  static_assert(uint32_t(MLOperator::OperatorKind::kEqual) == uint32_t(OperatorType::kEqual));
  static_assert(uint32_t(MLOperator::OperatorKind::kGreater) == uint32_t(OperatorType::kGreater));
  static_assert(uint32_t(MLOperator::OperatorKind::kLesser) == uint32_t(OperatorType::kLesser));
  static_assert(uint32_t(MLOperator::OperatorKind::kRelu) == uint32_t(OperatorType::kRelu));
  static_assert(uint32_t(MLOperator::OperatorKind::kClamp) == uint32_t(OperatorType::kClamp));
  static_assert(uint32_t(MLOperator::OperatorKind::kSigmoid) == uint32_t(OperatorType::kSigmoid));
  static_assert(uint32_t(MLOperator::OperatorKind::kHardSwish) == uint32_t(OperatorType::kHardSwish));
  static_assert(uint32_t(MLOperator::OperatorKind::kSoftmax) == uint32_t(OperatorType::kSoftmax));
  static_assert(uint32_t(MLOperator::OperatorKind::kAveragePool2d) == uint32_t(OperatorType::kAveragePool2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kMaxPool2d) == uint32_t(OperatorType::kMaxPool2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kResample2d) == uint32_t(OperatorType::kResample2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kIdentity) == uint32_t(OperatorType::kIdentity));
  static_assert(uint32_t(MLOperator::OperatorKind::kExp) == uint32_t(OperatorType::kExp));
  static_assert(uint32_t(MLOperator::OperatorKind::kSqrt) == uint32_t(OperatorType::kSqrt));
  static_assert(uint32_t(MLOperator::OperatorKind::kSin) == uint32_t(OperatorType::kSin));
  static_assert(uint32_t(MLOperator::OperatorKind::kCos) == uint32_t(OperatorType::kCos));
  static_assert(uint32_t(MLOperator::OperatorKind::kTan) == uint32_t(OperatorType::kTan));
  static_assert(uint32_t(MLOperator::OperatorKind::kErf) == uint32_t(OperatorType::kErf));
  static_assert(uint32_t(MLOperator::OperatorKind::kReciprocal) == uint32_t(OperatorType::kReciprocal));
  static_assert(uint32_t(MLOperator::OperatorKind::kLogicalNot) == uint32_t(OperatorType::kLogicalNot));
  static_assert(uint32_t(MLOperator::OperatorKind::kReshape) == uint32_t(OperatorType::kReshape));
  static_assert(uint32_t(MLOperator::OperatorKind::kSqueeze) == uint32_t(OperatorType::kSqueeze));
  static_assert(uint32_t(MLOperator::OperatorKind::kUnsqueeze) == uint32_t(OperatorType::kUnsqueeze));
  static_assert(uint32_t(MLOperator::OperatorKind::kFlattenTo2d) == uint32_t(OperatorType::kFlattenTo2d));
  static_assert(uint32_t(MLOperator::OperatorKind::kConcat) == uint32_t(OperatorType::kConcat));
  static_assert(uint32_t(MLOperator::OperatorKind::kSlice) == uint32_t(OperatorType::kSlice));
  static_assert(uint32_t(MLOperator::OperatorKind::kSplit) == uint32_t(OperatorType::kSplit));
  static_assert(uint32_t(MLOperator::OperatorKind::kTranspose) == uint32_t(OperatorType::kTranspose));
  static_assert(uint32_t(MLOperator::OperatorKind::kPad) == uint32_t(OperatorType::kPad));
  static_assert(uint32_t(MLOperator::OperatorKind::kExpand) == uint32_t(OperatorType::kExpand));
  static_assert(uint32_t(MLOperator::OperatorKind::kGather) == uint32_t(OperatorType::kGather));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceL1) == uint32_t(OperatorType::kReduceL1));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceL2) == uint32_t(OperatorType::kReduceL2));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceLogSum) == uint32_t(OperatorType::kReduceLogSum));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceLogSumExp) == uint32_t(OperatorType::kReduceLogSumExp));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceMax) == uint32_t(OperatorType::kReduceMax));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceMean) == uint32_t(OperatorType::kReduceMean));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceMin) == uint32_t(OperatorType::kReduceMin));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceProduct) == uint32_t(OperatorType::kReduceProduct));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceSum) == uint32_t(OperatorType::kReduceSum));
  static_assert(uint32_t(MLOperator::OperatorKind::kReduceSumSquare) == uint32_t(OperatorType::kReduceSumSquare));
  static_assert(uint32_t(MLOperator::OperatorKind::kArgMax) == uint32_t(OperatorType::kArgMax));
  static_assert(uint32_t(MLOperator::OperatorKind::kArgMin) == uint32_t(OperatorType::kArgMin));
  static_assert(uint32_t(MLOperator::OperatorKind::kCast) == uint32_t(OperatorType::kCast));
  static_assert(uint32_t(MLOperator::OperatorKind::kInstanceNormalization) == uint32_t(OperatorType::kInstanceNormalization));
  static_assert(uint32_t(MLOperator::OperatorKind::kMeanVarianceNormalization) == uint32_t(OperatorType::kMeanVarianceNormalization));
  static_assert(uint32_t(MLOperator::OperatorKind::kElementWiseIf) == uint32_t(OperatorType::kElementWiseIf));
  static_assert(uint32_t(MLOperator::OperatorKind::kFillSequence) == uint32_t(OperatorType::kFillSequence));
  static_assert(uint32_t(MLOperator::OperatorKind::kTriangularMatrix) == uint32_t(OperatorType::kTriangularMatrix));
  // clang-format on

  return static_cast<OperatorType>(type);
}

ml::webnn::mojom::blink::ClampOptionsPtr BlinkClampOptionsToMojo(
    const MLClampOptions* ml_options) {
  const float min = ml_options->hasMinValue()
                        ? ml_options->minValue()
                        : -std::numeric_limits<float>::infinity();
  const float max = ml_options->hasMaxValue()
                        ? ml_options->maxValue()
                        : +std::numeric_limits<float>::infinity();
  auto options = ml::webnn::mojom::blink::ClampOptions::New();
  options->minValue = min;
  options->maxValue = max;

  return options;
}

OperationInfoPtr FusionOperation(const MLOperator* activation) {
  switch (activation->Kind()) {
    case MLOperator::OperatorKind::kClamp: {
      auto clamp = ml::webnn::mojom::blink::Clamp::New();
      clamp->input_index = std::numeric_limits<uint64_t>::max();
      clamp->options = BlinkClampOptionsToMojo(
          static_cast<const MLClampOptions*>(activation->Options()));
      clamp->output_index = std::numeric_limits<uint64_t>::max();
      auto operation = OperationInfo::NewClamp(std::move(clamp));
      return operation;
    }
    case MLOperator::OperatorKind::kRelu: {
      auto relu = ml::webnn::mojom::blink::Relu::New();
      relu->input_index = std::numeric_limits<uint64_t>::max();
      relu->output_index = std::numeric_limits<uint64_t>::max();
      auto operation = OperationInfo::NewRelu(std::move(relu));
      return operation;
    }
    default: {
      NOTREACHED();
      return nullptr;
    }
  }
}

ml::webnn::mojom::blink::Conv2dOptionsPtr BlinkConv2dOptionsToMojo(
    const MLConv2dOptions* ml_options,
    const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
  auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<uint32_t>(4, 0);
  options->strides = ml_options->hasStrides()
                         ? ml_options->strides()
                         : Vector<uint32_t>(2, static_cast<uint32_t>(1));
  options->dilations = ml_options->hasDilations()
                           ? ml_options->dilations()
                           : Vector<uint32_t>(2, static_cast<uint32_t>(1));
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->groups = ml_options->groups();
  options->inputLayout =
      BlinkInputOperandLayoutToMojo(ml_options->inputLayout().AsEnum());
  options->filterLayout =
      BlinkConv2dFilterOperandLayoutToMojo(ml_options->filterLayout().AsEnum());
  options->bias_index = ml_options->hasBias()
                            ? operand_index_map.at(ml_options->bias())
                            : std::numeric_limits<uint64_t>::max();
  if (ml_options->hasActivation()) {
    options->activation = FusionOperation(ml_options->activation());
  }

  return options;
}

ml::webnn::mojom::blink::Pool2dOptionsPtr BlinkPool2dOptionsToMojo(
    const MLPool2dOptions* ml_options) {
  auto options = ml::webnn::mojom::blink::Pool2dOptions::New();
  options->window_dimensions = ml_options->hasWindowDimensions()
                                   ? ml_options->windowDimensions()
                                   : Vector<uint32_t>();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<uint32_t>(4, 0);
  options->strides = ml_options->hasStrides()
                         ? ml_options->strides()
                         : Vector<uint32_t>(2, static_cast<uint32_t>(1));
  options->dilations = ml_options->hasDilations()
                           ? ml_options->dilations()
                           : Vector<uint32_t>(2, static_cast<uint32_t>(1));
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->layout =
      BlinkInputOperandLayoutToMojo(ml_options->layout().AsEnum());
  options->rounding_type =
      BlinkRoundingTypeToMojo(ml_options->roundingType().AsEnum());
  options->output_sizes = ml_options->hasOutputSizes()
                              ? ml_options->outputSizes()
                              : Vector<uint32_t>();
  return options;
}

ml::webnn::mojom::blink::GemmOptionsPtr BlinkGemmOptionsToMojo(
    const MLGemmOptions* ml_options,
    const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
  auto options = ml::webnn::mojom::blink::GemmOptions::New();
  options->c_index = ml_options->hasC() ? operand_index_map.at(ml_options->c())
                                        : std::numeric_limits<uint64_t>::max();
  options->alpha = ml_options->hasAlpha() ? ml_options->alpha() : 1.0;
  options->beta = ml_options->hasBeta() ? ml_options->beta() : 1.0;
  options->a_transpose =
      ml_options->hasATranspose() ? ml_options->aTranspose() : false;
  options->b_transpose =
      ml_options->hasBTranspose() ? ml_options->bTranspose() : false;
  return options;
}

}  // namespace

MojoModelInfo::MojoModelInfo() {
  model_info_ = ml::webnn::mojom::blink::ModelInfo::New();
}

MojoModelInfo::~MojoModelInfo() = default;

void MojoModelInfo::Trace(Visitor* visitor) const {
  visitor->Trace(operand_index_map_);
  visitor->Trace(constant_index_map_);
}

void MojoModelInfo::AddInput(const MLOperand* input) {
  // Add input operand descriptor to the model.
  size_t index = AddOperandToModel(input);
  // Add the input to the model.
  auto named_input = ml::webnn::mojom::blink::NamedOperand::New();
  named_input->name = input->Name();
  named_input->index = index;
  model_info_->inputs.push_back(std::move(named_input));
}

void MojoModelInfo::AddConstant(const MLOperand* constant) {
  // Add const operand descriptor to the model.
  size_t index = AddOperandToModel(constant);
  // All constant data will share a big shared memory, so hold the index of
  // constant temporarily in member variable.
  constant_index_map_.insert(constant, index);
}

void MojoModelInfo::AddOutput(String name, const MLOperand* output) {
  if (operand_index_map_.find(output) == operand_index_map_.end()) {
    return;
  }
  auto named_output = ml::webnn::mojom::blink::NamedOperand::New();
  named_output->name = std::move(name);
  named_output->index = operand_index_map_.at(output);
  model_info_->outputs.push_back(std::move(named_output));
}

bool MojoModelInfo::AreOperandsInIndexMap(const Member<const MLOperand>* operands,
                                          size_t operand_count) const {
  for (size_t i = 0; i < operand_count; ++i) {
    if (!operand_index_map_.Contains(operands[i])) {
      return false;
    }
  }
  return true;
}

uint64_t MojoModelInfo::GetOperandIndex(/*nullable*/ const MLOperand* operand) const
{
    if (operand == nullptr)
    {
        // Sentinel value represents no tensor.
        return std::numeric_limits<uint64_t>::max();
    }
    return operand_index_map_.at(operand);
}

void MojoModelInfo::AddClamp(const MLOperator* ml_clamp) {
  DCHECK_EQ(ml_clamp->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_clamp->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  DCHECK_EQ(ml_clamp->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_clamp->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  // Add operand descriptor to the model.
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto clamp = ml::webnn::mojom::blink::Clamp::New();
  clamp->input_index = operand_index_map_.at(input);
  clamp->options = BlinkClampOptionsToMojo(
      static_cast<const MLClampOptions*>(ml_clamp->Options()));
  clamp->output_index = output_index;
  auto operation = OperationInfo::NewClamp(std::move(clamp));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddConv2d(const MLOperator* ml_conv2d) {
  DCHECK_GE(ml_conv2d->Inputs().size(), static_cast<uint32_t>(2));
  auto* input = ml_conv2d->Inputs()[0].Get();
  auto* filter = ml_conv2d->Inputs()[1].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end() ||
      operand_index_map_.find(filter) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_conv2d->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_conv2d->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto conv2d = ml::webnn::mojom::blink::Conv2d::New();
  conv2d->input_index = operand_index_map_.at(input);
  conv2d->filter_index = operand_index_map_.at(filter);
  const MLConv2dOptions* ml_options =
      static_cast<const MLConv2dOptions*>(ml_conv2d->Options());
  conv2d->options = BlinkConv2dOptionsToMojo(ml_options, operand_index_map_);
  conv2d->output_index = output_index;
  auto operation = OperationInfo::NewConv2d(std::move(conv2d));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddConvTranspose2d(const MLOperator* ml_conv2d) {
#if 0 // TODO:::
  DCHECK_GE(ml_conv2d->Inputs().size(), static_cast<uint32_t>(2));
  auto* input = ml_conv2d->Inputs()[0].Get();
  auto* filter = ml_conv2d->Inputs()[1].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end() ||
      operand_index_map_.find(filter) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_conv2d->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_conv2d->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto conv2d = ml::webnn::mojom::blink::Conv2d::New();
  conv2d->input_index = operand_index_map_.at(input);
  conv2d->filter_index = operand_index_map_.at(filter);
  const MLConv2dOptions* ml_options =
      static_cast<const MLConv2dOptions*>(ml_conv2d->Options());
  conv2d->options = BlinkConv2dOptionsToMojo(ml_options, operand_index_map_);
  conv2d->output_index = output_index;
  auto operation = OperationInfo::NewConv2d(std::move(conv2d));
  model_info_->operations.push_back(std::move(operation));
#endif
}

void MojoModelInfo::AddElementWiseUnary(const MLOperator* ml_operator) {
  DCHECK_EQ(ml_operator->Inputs().size(), static_cast<uint32_t>(1));
  DCHECK_EQ(ml_operator->Outputs().size(), static_cast<uint32_t>(1));

  // Verify inputs exist and output does not yet exist.
  auto* input = ml_operator->Inputs()[0].Get();
  auto* output = ml_operator->Outputs()[0].Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::ElementWiseUnary::New();
  mojom_operator->operator_type = BlinkOperatorKindToMojoType(ml_operator->Kind());
  mojom_operator->input_index = operand_index_map_.at(input);
  mojom_operator->output_index = output_index;
  auto operation = OperationInfo::NewElementWiseUnary(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddElementWiseBinary(const MLOperator* ml_binary) {
  DCHECK_EQ(ml_binary->Inputs().size(), static_cast<uint32_t>(2));
  DCHECK_EQ(ml_binary->Outputs().size(), static_cast<uint32_t>(1));

  // Verify inputs exist and output does not yet exist.
  auto* a = ml_binary->Inputs()[0].Get();
  auto* b = ml_binary->Inputs()[1].Get();
  auto* output = ml_binary->Outputs()[0].Get();
  if (!operand_index_map_.Contains(a) ||
      !operand_index_map_.Contains(b)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  // Create mojom operator from JS blink type.
  auto binary = ml::webnn::mojom::blink::ElementWiseBinary::New();
  binary->operator_type = BlinkOperatorKindToMojoType(ml_binary->Kind());
  binary->a_index = operand_index_map_.at(a);
  binary->b_index = operand_index_map_.at(b);
  binary->output_index = output_index;
  auto operation = OperationInfo::NewElementWiseBinary(std::move(binary));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddGemm(const MLOperator* ml_gemm) {
  DCHECK_GE(ml_gemm->Inputs().size(), static_cast<uint32_t>(2));
  auto* a = ml_gemm->Inputs()[0].Get();
  auto* b = ml_gemm->Inputs()[1].Get();
  if (operand_index_map_.find(a) == operand_index_map_.end() ||
      operand_index_map_.find(b) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_gemm->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_gemm->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto gemm = ml::webnn::mojom::blink::Gemm::New();
  gemm->a_index = operand_index_map_.at(a);
  gemm->b_index = operand_index_map_.at(b);
  const MLGemmOptions* ml_options =
      static_cast<const MLGemmOptions*>(ml_gemm->Options());
  gemm->options = BlinkGemmOptionsToMojo(ml_options, operand_index_map_);
  gemm->output_index = output_index;
  auto operation = OperationInfo::NewGemm(std::move(gemm));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddPool2d(const MLOperator* ml_pool2d) {
  DCHECK_EQ(ml_pool2d->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_pool2d->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_pool2d->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_pool2d->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto pool2d = ml::webnn::mojom::blink::Pool2d::New();
  pool2d->type = BlinkPool2dTypeToMojo(ml_pool2d->Kind());
  pool2d->input_index = operand_index_map_.at(input);
  const MLPool2dOptions* ml_options =
      static_cast<const MLPool2dOptions*>(ml_pool2d->Options());
  pool2d->options = BlinkPool2dOptionsToMojo(ml_options);
  pool2d->output_index = output_index;
  auto operation = OperationInfo::NewPool2d(std::move(pool2d));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddRelu(const MLOperator* ml_relu) {
  DCHECK_EQ(ml_relu->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_relu->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_relu->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_relu->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto relu = ml::webnn::mojom::blink::Relu::New();
  relu->input_index = operand_index_map_.at(input);
  relu->output_index = output_index;
  auto operation = OperationInfo::NewRelu(std::move(relu));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddReshape(const MLOperator* ml_reshape) {
  DCHECK_EQ(ml_reshape->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_reshape->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_reshape->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_reshape->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto reshape = ml::webnn::mojom::blink::Reshape::New();
  reshape->input_index = operand_index_map_.at(input);
  reshape->output_index = output_index;
  auto operation = OperationInfo::NewReshape(std::move(reshape));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddSoftmax(const MLOperator* ml_softmax) {
  DCHECK_EQ(ml_softmax->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_softmax->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_softmax->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_softmax->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto softmax = ml::webnn::mojom::blink::Softmax::New();
  softmax->input_index = operand_index_map_.at(input);
  softmax->output_index = output_index;
  auto operation = OperationInfo::NewSoftmax(std::move(softmax));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddArgMinMax(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));

  size_t output_index = AddOperandToModel(output);

  const MLArgMinMaxOptions* ml_options =
      static_cast<const MLArgMinMaxOptions*>(ml_operator->Options());

  auto mojom_operator = ml::webnn::mojom::blink::ArgMinMax::New();
  mojom_operator->operator_type = BlinkOperatorKindToMojoType(ml_operator->Kind());
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->axis = ml_options->axis();
  mojom_operator->keep_dimensions = ml_options->keepDimensions();
  mojom_operator->select_last_index = ml_options->selectLastIndex();
  mojom_operator->output_index = output_index;

  auto operation_info = OperationInfo::NewArgMinMax(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddCast(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  // Verify inputs exist and output does not yet exist.
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::Cast::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->output_index = output_index;
  mojom_operator->data_type = BlinkOperandTypeToMojo(output->Type());
  auto operation_info = OperationInfo::NewCast(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddConcat(const MLOperator* ml_operator) {
  DCHECK_GE(ml_operator->Inputs().size(), 0u);
  DCHECK_EQ(ml_operator->Outputs().size(), 1u);

  // Verify inputs exist and output does not yet exist.
  auto& inputs = ml_operator->Inputs();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!AreOperandsInIndexMap(inputs.data(), inputs.size()))
  {
      return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLConcatOptionsInternal* ml_options =
      static_cast<const MLConcatOptionsInternal*>(ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::Concat::New();
  for (const Member<const MLOperand>& input : inputs)
  {
    mojom_operator->input_indices.push_back(GetOperandIndex(input));
  }
  mojom_operator->output_index = output_index;
  mojom_operator->axis = ml_options->axis();
  auto operation_info = OperationInfo::NewConcat(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddSlice(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  // Verify inputs exist and output does not yet exist.
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLSliceOptionsInternal* ml_options =
      static_cast<const MLSliceOptionsInternal*>(ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::Slice::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->starts = ml_options->starts();
  mojom_operator->sizes = ml_options->sizes();
  mojom_operator->output_index = output_index;
  auto operation_info = OperationInfo::NewSlice(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddSplit(const MLOperator* ml_operator) {
  DCHECK_EQ(ml_operator->Inputs().size(), 1u);
  DCHECK_GE(ml_operator->Outputs().size(), 0u);

  // Verify inputs exist and output does not yet exist.
  auto& outputs = ml_operator->Outputs();
  const MLOperand* input = ml_operator->Inputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!AreOperandsInIndexMap(outputs.data(), outputs.size()));

  const MLSplitOptionsInternal* ml_options =
      static_cast<const MLSplitOptionsInternal*>(ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::Split::New();
  for (const Member<const MLOperand>& output : outputs)
  {
    size_t output_index = AddOperandToModel(output);
    mojom_operator->output_indices.push_back(output_index);
  }
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->axis = ml_options->axis();
  auto operation_info = OperationInfo::NewSplit(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddExpand(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  auto mojom_operator = ml::webnn::mojom::blink::Expand::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->output_index = output_index;

  auto operation_info = OperationInfo::NewExpand(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddGather(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 2u, 1u);

  auto& inputs = ml_operator->Inputs();
  const MLOperand* input = inputs[0].Get();
  const MLOperand* indices = inputs[1].Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!AreOperandsInIndexMap(inputs.data(), inputs.size()))
  {
      return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLGatherOptions* ml_options =
      static_cast<const MLGatherOptions*>(ml_operator->Options());

  auto mojom_operator = ml::webnn::mojom::blink::Gather::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->indices_index = GetOperandIndex(indices);
  mojom_operator->axis = ml_options->axis();
  mojom_operator->output_index = output_index;

  auto operation_info = OperationInfo::NewGather(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddInstanceNormalization(const MLOperator* ml_operator) {
  DCHECK_GE(ml_operator->Inputs().size(), 1u);
  DCHECK_LE(ml_operator->Inputs().size(), 3u);
  DCHECK_EQ(ml_operator->Outputs().size(), 1u);

  // Verify inputs exist and output does not yet exist.
  auto& inputs = ml_operator->Inputs();
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!AreOperandsInIndexMap(inputs.data(), inputs.size()))
  {
      return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLInstanceNormalizationOptions* ml_options =
      static_cast<const MLInstanceNormalizationOptions*>(
          ml_operator->Options());

  // Add another enum if extending this to ensure the two enums are castable below.
  static_assert(InputOperandLayout::kMaxValue == InputOperandLayout::kNhwc);
  static_assert(uint32_t(InputOperandLayout::kNchw) ==
                uint32_t(V8MLInputOperandLayout::Enum::kNchw));
  static_assert(uint32_t(InputOperandLayout::kNhwc) ==
                uint32_t(V8MLInputOperandLayout::Enum::kNhwc));

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::InstanceNormalization::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->scale_index = GetOperandIndex(ml_options->getScaleOr(nullptr));
  mojom_operator->bias_index = GetOperandIndex(ml_options->getBiasOr(nullptr));
  mojom_operator->epsilon = ml_options->epsilon();
  mojom_operator->layout = static_cast<InputOperandLayout>(ml_options->layout().AsEnum());
  mojom_operator->output_index = output_index;
  auto operation_info = OperationInfo::NewInstanceNormalization(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddMeanVarianceNormalization(const MLOperator* ml_operator) {
  DCHECK_GE(ml_operator->Inputs().size(), 1u);
  DCHECK_LE(ml_operator->Inputs().size(), 3u);
  DCHECK_EQ(ml_operator->Outputs().size(), 1u);

  // Verify inputs exist and output does not yet exist.
  auto& inputs = ml_operator->Inputs();
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!AreOperandsInIndexMap(inputs.data(), inputs.size()))
  {
      return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLMeanVarianceNormalizationOptions* ml_options =
      static_cast<const MLMeanVarianceNormalizationOptions*>(
          ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::MeanVarianceNormalization::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->scale_index = GetOperandIndex(ml_options->getScaleOr(nullptr));
  mojom_operator->bias_index = GetOperandIndex(ml_options->getBiasOr(nullptr));
  mojom_operator->epsilon = ml_options->epsilon();
  mojom_operator->axes = ml_options->axes();
  mojom_operator->output_index = output_index;
  auto operation_info = OperationInfo::NewMeanVarianceNormalization(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddPad(const MLOperator* ml_operator) {
    // TODO:
}

void MojoModelInfo::AddFillSequence(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 0u, 1u);

  // Verify inputs exist and output does not yet exist.
  const MLOperand* output = ml_operator->Outputs().front().Get();
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLFillSequenceOptions* ml_options =
      static_cast<const MLFillSequenceOptions*>(ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::FillSequence::New();
  mojom_operator->start = ml_options->start();
  mojom_operator->delta = ml_options->delta();
  mojom_operator->output_index = output_index;
  auto operation = OperationInfo::NewFillSequence(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddReduce(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  // Verify inputs exist and output does not yet exist.
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }

  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLReduceOptions* ml_options =
      static_cast<const MLReduceOptions*>(ml_operator->Options());

  // Create mojom operator from JS blink type.
  auto mojom_operator = ml::webnn::mojom::blink::Reduce::New();
  mojom_operator->operator_type = BlinkOperatorKindToMojoType(ml_operator->Kind());
  mojom_operator->axes = ml_options->axes();
  mojom_operator->keep_dimensions = ml_options->keepDimensions();
  mojom_operator->input_index = operand_index_map_.at(input);
  mojom_operator->output_index = output_index;
  auto operation = OperationInfo::NewReduce(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddResample2d(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  // Verify inputs exist and output does not yet exist.
  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLResample2dOptions* ml_options =
      static_cast<const MLResample2dOptions*>(ml_operator->Options());

  DCHECK(ml_options->hasScales());
  DCHECK(ml_options->hasAxes());

  // Create mojom operator from JS blink type.
  
  static_assert(
      V8MLInterpolationMode::kEnumSize == 2,
      "A new enum was added - verify these mappings are still correct.");
  static_assert(
      uint32_t(V8MLInterpolationMode::Enum::kNearestNeighbor) ==
      uint32_t(ml::webnn::mojom::InterpolationMode::kNearestNeighbor));
  static_assert(uint32_t(V8MLInterpolationMode::Enum::kLinear) ==
                uint32_t(ml::webnn::mojom::InterpolationMode::kLinear));

  auto mojom_operator = ml::webnn::mojom::blink::Resample2d::New();
  mojom_operator->input_index = GetOperandIndex(input);
  mojom_operator->scales = ml_options->scales();
  mojom_operator->axes = ml_options->axes();
  mojom_operator->interpolation_mode =
      static_cast<ml::webnn::mojom::InterpolationMode>(
          ml_options->mode().AsEnum());
  mojom_operator->output_index = output_index;

  auto operation_info = OperationInfo::NewResample2d(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddTranspose(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 1u, 1u);

  const MLOperand* input = ml_operator->Inputs().front().Get();
  const MLOperand* output = ml_operator->Outputs().front().Get();
  if (!operand_index_map_.Contains(input)) {
    return;
  }
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  const MLTransposeOptions* ml_options =
      static_cast<const MLTransposeOptions*>(ml_operator->Options());

  auto mojom_operator = ml::webnn::mojom::blink::Transpose::New();
  mojom_operator->input_index = operand_index_map_.at(input);
  mojom_operator->output_index = output_index;
  mojom_operator->permutation = ml_options->permutation();

  auto operation_info = OperationInfo::NewTranspose(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::AddTriangularMatrix(const MLOperator* ml_operator) {
    // TODO:
}

void MojoModelInfo::AddElementWiseIf(const MLOperator* ml_operator) {
  DCHECK_OPERATOR_INPUT_OUTPUT_COUNT(ml_operator, 3u, 1u);

  auto& inputs = ml_operator->Inputs();
  if (!AreOperandsInIndexMap(inputs.data(), inputs.size()))
  {
      return;
  }
  const MLOperand* condition = inputs[0].Get();
  const MLOperand* true_value = inputs[1].Get();
  const MLOperand* false_value = inputs[2].Get();
  const MLOperand* output = ml_operator->Outputs()[0].Get();
  DCHECK(!operand_index_map_.Contains(output));
  size_t output_index = AddOperandToModel(output);

  auto mojom_operator = ml::webnn::mojom::blink::ElementWiseIf::New();
  mojom_operator->condition_index = operand_index_map_.at(condition);
  mojom_operator->true_value_index = operand_index_map_.at(true_value);
  mojom_operator->false_value_index = operand_index_map_.at(false_value);
  mojom_operator->output_index = output_index;
  auto operation_info = OperationInfo::NewElementWiseIf(std::move(mojom_operator));
  model_info_->operations.push_back(std::move(operation_info));
}

void MojoModelInfo::FillConstantsWithArrayBuffer() {
  // Copy constant data to shared memory.
  base::CheckedNumeric<size_t> constants_buffer_length(0);
  for (const auto& [constant, index] : constant_index_map_) {
    wtf_size_t size = base::checked_cast<wtf_size_t>(
        constant->ArrayBufferView()->byteLength());
    constants_buffer_length += Align(size, kBufferAlignment);
  }
  base::MappedReadOnlyRegion constants_shm_region =
      base::ReadOnlySharedMemoryRegion::Create(
          constants_buffer_length.ValueOrDie());
  auto constants_info = ml::webnn::mojom::blink::ConstantsInfo::New();
  base::CheckedNumeric<size_t> aligned_constant_offset(0);
  for (const auto& [constant, index] : constant_index_map_) {
    auto memory_info = ml::webnn::mojom::blink::MemoryInfo::New();
    auto* array_buffer_view = constant->ArrayBufferView();
    memory_info->byte_offset = aligned_constant_offset.ValueOrDie();
    memory_info->byte_length = array_buffer_view->byteLength();
    aligned_constant_offset +=
        Align(memory_info->byte_length, kBufferAlignment);

    uint8_t* address = constants_shm_region.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    constants_info->memory_info.insert(index, std::move(memory_info));
  }
  constants_info->shared_memory = constants_shm_region.region.Duplicate();
  model_info_->constants =
      constant_index_map_.size() == 0 ? nullptr : std::move(constants_info);
}

ModelInfoPtr MojoModelInfo::GetModelInfo() {
  return std::move(model_info_);
}

size_t MojoModelInfo::AddOperandToModel(const MLOperand* output) {
  // Create the operand descriptor for mojo.
  auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
  desc->data_type = BlinkOperandTypeToMojo(output->Type());
  desc->dimensions = output->Dimensions();
  // Add operand descriptor into model.
  model_info_->operands.push_back(std::move(desc));
  // The index used to identify operand on the server side, each operation
  // generate a output operand that will be inserted in a hash map with the
  // MLOperand and index, the index is incremented by one.
  size_t output_index = model_info_->operands.size() - 1;
  operand_index_map_.insert(output, output_index);
  return output_index;
}

}  // namespace blink
