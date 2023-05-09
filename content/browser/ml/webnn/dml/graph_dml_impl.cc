// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

#include "base/logging.h"
#include "base/memory/ptr_util.h"
#include "base/containers/span.h"
#include "content/browser/ml/webnn/dml/execution_context.h"
#include "content/browser/ml/webnn/dml/execution_resources.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
#include "content/browser/ml/webnn/dml/upload_resource.h"
#include "mojo/public/c/system/types.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

// TODO:::DELETE
#pragma optimize("", off) // TODO:::DELETE

namespace content::webnn {

namespace {

using ml::webnn::mojom::AutoPad;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::Conv2dFilterOperandLayout;
using ml::webnn::mojom::Conv2dOptions;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::InputOperandLayout;
using ml::webnn::mojom::MemoryInfoPtr;
using ml::webnn::mojom::ModelInfoPtr;
using ml::webnn::mojom::OperandType;
using ml::webnn::mojom::OperationInfo;

enum TransposeType { NhwcToNchw, NchwToNhwc };

std::vector<UINT> transposeStrides(TransposeType transposeType,
                                   const std::vector<UINT>& input_dims) {
  UINT nStride = 0, cStride = 0, hStride = 0, wStride = 0;
  switch (transposeType) {
    case NhwcToNchw:
      nStride = input_dims[1] * input_dims[2] * input_dims[3];
      hStride = input_dims[2] * input_dims[3];
      wStride = input_dims[3];
      cStride = 1;
      return {nStride, cStride, hStride, wStride};
    case NchwToNhwc:
      nStride = input_dims[1] * input_dims[2] * input_dims[3];
      cStride = input_dims[2] * input_dims[3];
      hStride = input_dims[3];
      wStride = 1;
      return {nStride, hStride, wStride, cStride};
    default:
      DCHECK(0);
      break;
  }
}

std::array<uint32_t, 4> getLayoutToLayoutPermutation(InputOperandLayout original_layout, InputOperandLayout new_layout)
{
  static_assert(uint32_t(InputOperandLayout::kMaxValue) == 1, "Update getLayoutToLayoutPermutation for the new layout.");
  switch (original_layout) {
    case InputOperandLayout::kNchw:
      switch (new_layout) {
        case InputOperandLayout::kNchw: return {0,1,2,3};
        case InputOperandLayout::kNhwc: return {0,2,3,1};
      }
      break;
    case InputOperandLayout::kNhwc:
      switch (new_layout) {
        case InputOperandLayout::kNchw: return {0,3,1,2};
        case InputOperandLayout::kNhwc: return {0,1,2,3};
      }
      break;
  }
}

std::vector<uint32_t> transposeStrides(base::span<const uint32_t> original_strides, base::span<const uint32_t> permutation)
{
  auto dimension_count = original_strides.size();
  std::vector<uint32_t> new_strides;
  new_strides.reserve(dimension_count);
  for (auto axis : permutation)
  {
    DCHECK(axis < dimension_count); // This should have already been validated.
    new_strides.push_back(original_strides[axis]);
  }
  return new_strides;
}

std::vector<UINT> transposeStridesToNchw(
    const std::vector<UINT>& input_dims,
    const DML_TENSOR_DESC* input_tensor_desc) {
  const DML_BUFFER_TENSOR_DESC* bufferDesc =
      reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(input_tensor_desc->Desc);
  DCHECK(bufferDesc != nullptr && bufferDesc->DimensionCount == 4);
  auto* strides = bufferDesc->Strides;
  if (strides != nullptr) {
    return {strides[0], strides[3], strides[1], strides[2]};
  } else {
    return transposeStrides(NhwcToNchw, input_dims);
  }
}

DML_OPERATOR_DESC* CreateFusedOperator(
    const OperationInfo* activation,
    DML_ACTIVATION_LINEAR_OPERATOR_DESC& dmlActicationOperatorDesc,
    DML_OPERATOR_DESC& dmlFusedOperatorDesc) {
  if (activation == nullptr) {
    return nullptr;
  }

  dmlActicationOperatorDesc.InputTensor = nullptr;
  dmlActicationOperatorDesc.OutputTensor = nullptr;
  dmlActicationOperatorDesc.Alpha = 0.0;
  dmlActicationOperatorDesc.Beta = 0.0;
  switch (activation->which()) {
    case OperationInfo::Tag::kRelu:
      dmlFusedOperatorDesc.Type = DML_OPERATOR_ACTIVATION_RELU;
      break;
    case OperationInfo::Tag::kClamp:
      return nullptr;
    default:
      LOG(ERROR) << "This fusion type is not supported.";
      DCHECK(0);
  }
  dmlFusedOperatorDesc.Desc = &dmlActicationOperatorDesc;
  return &dmlFusedOperatorDesc;
}

std::vector<UINT> transposeDimensions(TransposeType transposeType,
                                      const std::vector<UINT>& input_dims) {
  DCHECK(input_dims.size() == 4);
  std::vector<UINT> newInputDims(4);
  switch (transposeType) {
    case NhwcToNchw:
      newInputDims[0] = input_dims[0];
      newInputDims[1] = input_dims[3];
      newInputDims[2] = input_dims[1];
      newInputDims[3] = input_dims[2];
      break;
    case NchwToNhwc:
      newInputDims[0] = input_dims[0];
      newInputDims[1] = input_dims[2];
      newInputDims[2] = input_dims[3];
      newInputDims[3] = input_dims[1];
      break;
    default:
      DCHECK(0);
      break;
  }
  return newInputDims;
}

std::vector<UINT> transposeFilterDimensionsAsOihw(
    Conv2dFilterOperandLayout filterLayout,
    const std::vector<UINT>& filterDims) {
  std::vector<UINT> newFilterDims(4);
  switch (filterLayout) {
    case Conv2dFilterOperandLayout::kOhwi:
      newFilterDims.resize(4);
      newFilterDims[0] = filterDims[0];
      newFilterDims[1] = filterDims[3];
      newFilterDims[2] = filterDims[1];
      newFilterDims[3] = filterDims[2];
      break;
    case Conv2dFilterOperandLayout::kHwio:
      newFilterDims[0] = filterDims[3];
      newFilterDims[1] = filterDims[2];
      newFilterDims[2] = filterDims[0];
      newFilterDims[3] = filterDims[1];
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      newFilterDims[0] = filterDims[3];
      newFilterDims[1] = filterDims[0];
      newFilterDims[2] = filterDims[1];
      newFilterDims[3] = filterDims[2];
      break;
    default:
      DCHECK(0);
      break;
  }
  return newFilterDims;
}

std::vector<UINT> transposeFilterStridesAsOihw(
    Conv2dFilterOperandLayout filterLayout,
    const std::vector<UINT>& filterDims) {
  UINT hStride = 0, wStride = 0, iStride = 0, oStride = 0;
  switch (filterLayout) {
    case Conv2dFilterOperandLayout::kHwio:
      hStride = filterDims[1] * filterDims[2] * filterDims[3];
      wStride = filterDims[2] * filterDims[3];
      iStride = filterDims[3];
      oStride = 1;
      break;
    case Conv2dFilterOperandLayout::kOhwi:
      oStride = filterDims[1] * filterDims[2] * filterDims[3];
      hStride = filterDims[2] * filterDims[3];
      wStride = filterDims[3];
      iStride = 1;
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      iStride = filterDims[1] * filterDims[2] * filterDims[3];
      hStride = filterDims[2] * filterDims[3];
      wStride = filterDims[3];
      oStride = 1;
      break;
    default:
      DCHECK(0);
      break;
  }
  return {oStride, iStride, hStride, wStride};
}

uint16_t CastFloat32ToFloat16(float float32_value) noexcept {
  static uint32_t constexpr float16_mantissa_count = 10;
  static int32_t constexpr float32to16_mantissa_count_difference = 23 - 10;
  static int32_t constexpr float32vs16_exponent_adjustment = 127 - 15;
  static uint32_t constexpr float16_sign_mask = 0b1'00000'0000000000;
  static uint32_t constexpr float16_mantissa_mask = 0b0'00000'1111111111;
  static uint32_t constexpr float16_exponent_mask = 0b0'11111'0000000000;
  static uint32_t constexpr float16_mantissa_and_exponentMask = 0b0'11111'1111111111;
  static uint32_t constexpr float32_mantissa_and_exponent_mask = 0b01111111'10000000'00000000'00000000;

  // Shift the mantissa, exponent, and sign from the 32-bit locations to 16-bit.
  // Sature the exponent if greater than float16 can represent.
  // float32 denorms are flushed to zero.

  uint32_t const float32_bit_value = reinterpret_cast<uint32_t&>(float32_value);
  uint32_t const sign = (float32_bit_value >> 16) & float16_sign_mask;
  int32_t const float32_mantissa_and_exponent =
      float32_bit_value & float32_mantissa_and_exponent_mask;
  int32_t const float16_mantissa_and_exponent =
      (float32_mantissa_and_exponent >> float32to16_mantissa_count_difference) -
      (float32vs16_exponent_adjustment << float16_mantissa_count);
  uint32_t const float16_denorm_mask =
      (float16_mantissa_and_exponent > int32_t(float16_mantissa_mask))
          ? float16_mantissa_and_exponentMask
          : 0;
  uint32_t const float16_saturation_ask =
      (float16_mantissa_and_exponent >= int32_t(float16_mantissa_and_exponentMask))
          ? float16_exponent_mask
          : 0;
  uint32_t const float16_bit_value =
      (float16_mantissa_and_exponent & float16_denorm_mask) | float16_saturation_ask |
      sign;
  return uint16_t(float16_bit_value);
}

DML_TENSOR_DATA_TYPE GetTensorDataType(OperandType type) {
  // clang-format off
  switch (type)
  {
  case OperandType::kFloat32: return DML_TENSOR_DATA_TYPE_FLOAT32;
  case OperandType::kFloat16: return DML_TENSOR_DATA_TYPE_FLOAT16;
  case OperandType::kInt32:   return DML_TENSOR_DATA_TYPE_INT32;
  case OperandType::kUint32:  return DML_TENSOR_DATA_TYPE_UINT32;
  case OperandType::kInt8:    return DML_TENSOR_DATA_TYPE_INT8;
  case OperandType::kUint8:   return DML_TENSOR_DATA_TYPE_UINT8;
  default:
    LOG(ERROR) << "This data type is not supported";
    return DML_TENSOR_DATA_TYPE_UNKNOWN;
  }
  // clang-format on
}

DML_SCALAR_UNION GetScalarUnion(DML_TENSOR_DATA_TYPE tensorDataType, float value)
{
  DML_SCALAR_UNION valueUnion = {};

  // clang-format off
  switch (tensorDataType)
  {
  case DML_TENSOR_DATA_TYPE_FLOAT32: valueUnion.Float32 = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_FLOAT16: valueUnion.UInt16  = CastFloat32ToFloat16(value); break;
  case DML_TENSOR_DATA_TYPE_UINT32:  valueUnion.UInt32  = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_UINT16:  valueUnion.UInt16  = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_UINT8:   valueUnion.UInt8   = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_INT32:   valueUnion.Int32   = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_INT16:   valueUnion.Int16   = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_INT8:    valueUnion.Int8    = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_FLOAT64: valueUnion.Float64 = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_UINT64:  valueUnion.UInt64  = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_INT64:   valueUnion.Int64   = static_cast<float>(value); break;
  case DML_TENSOR_DATA_TYPE_UNKNOWN: /* keep zeroed */ break;
  default:                           /* keep zeroed */ break;
  }
  // clang-format on

  return valueUnion;
}

DML_REDUCE_FUNCTION MapOperatorTypeToReductionFuntion(OperatorType operator_type) {
  // clang-format off
  switch (operator_type)
  {
  case OperatorType::kReduceL1:        return DML_REDUCE_FUNCTION_L1;
  case OperatorType::kReduceL2:        return DML_REDUCE_FUNCTION_L2;
  case OperatorType::kReduceLogSum:    return DML_REDUCE_FUNCTION_LOG_SUM;
  case OperatorType::kReduceLogSumExp: return DML_REDUCE_FUNCTION_LOG_SUM_EXP;
  case OperatorType::kReduceMax:       return DML_REDUCE_FUNCTION_MAX;
  case OperatorType::kReduceMean:      return DML_REDUCE_FUNCTION_AVERAGE;
  case OperatorType::kReduceMin:       return DML_REDUCE_FUNCTION_MIN;
  case OperatorType::kReduceProduct:   return DML_REDUCE_FUNCTION_MULTIPLY;
  case OperatorType::kReduceSum:       return DML_REDUCE_FUNCTION_SUM;
  case OperatorType::kReduceSumSquare: return DML_REDUCE_FUNCTION_SUM_SQUARE;
  default:
    LOG(ERROR) << "This operator type is not supported for reduction";
    return DML_REDUCE_FUNCTION_MIN;
  }
  // clang-format on
}

TensorDesc GetBroadcastedTensorDesc(NodeOutput* input_node,
                                    base::span<const UINT> broadcasted_dims,
                                    uint32_t ignorable_tail_count = 0) {
  TensorDesc broadcasted_tensor(input_node->GetTensorDesc());
  broadcasted_tensor.BroadcastTo(broadcasted_dims, TensorDesc::Alignment::kTrailing, ignorable_tail_count);
  return broadcasted_tensor;
}

}  // namespace

#define DAWN_INTERNAL_ERROR(MESSAGE)            \
  do {                                          \
    error_messages_ = MESSAGE;                  \
    DCHECK(0);                                  \
    build_result_ = BuildResult::kUnknownError; \
    return;                                     \
  } while (0)

#define CREATE_BINARY_OPERATOR(type, a_tensor_desc, b_tensor_desc, \
                               output_tensor_desc, node)           \
  DML_ELEMENT_WISE_##type##_OPERATOR_DESC operator_desc{};         \
  operator_desc.ATensor = a_tensor_desc;                           \
  operator_desc.BTensor = b_tensor_desc;                           \
  operator_desc.OutputTensor = output_tensor_desc;                 \
  node = graph_desc_builder_->CreateOperatorNode(                  \
      DML_OPERATOR_ELEMENT_WISE_##type, &operator_desc);

#define CREATE_UNARY_OPERATOR(type, input_tensor_desc,                \
                              output_tensor_desc, node)               \
  DML_##type##_OPERATOR_DESC operator_desc{};                         \
  operator_desc.InputTensor = input_tensor_desc;                      \
  operator_desc.OutputTensor = output_tensor_desc;                    \
  node = graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_##type, \
                                                 &operator_desc);

// static
void GraphDMLImpl::Create(mojo::PendingReceiver<Graph> receiver,
                          scoped_refptr<ExecutionContext> execution_context) {
  mojo::MakeSelfOwnedReceiver<Graph>(
      base::WrapUnique(new GraphDMLImpl(execution_context)),
      std::move(receiver));
}

GraphDMLImpl::~GraphDMLImpl() = default;

GraphDMLImpl::GraphDMLImpl(scoped_refptr<ExecutionContext> execution_context)
    : execution_context_(execution_context),
      input_resource_uploader_(
          std::make_unique<UploadResource>(execution_context_.get())),
      output_resource_readback_(
          std::make_unique<ReadbackResource>(execution_context_.get())),
      graph_desc_builder_(std::make_unique<GraphDescBuilder>(
          execution_context->GetDMLDevice())) {}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc,
                            UINT64 index) {
  // TODO: return directly if BuildResult has error message.
  Node input_node = graph_desc_builder_->CreateInputNode(std::move(name));
  TensorDesc tensor_desc(GetTensorDataType(desc->data_type), desc->dimensions);
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      input_node, 0, std::move(tensor_desc));
  node_output_map_[index] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc, UINT64 index) {
  // TODO: return directly if BuildResult has error message.
  if (node_output_map_.find(index) != node_output_map_.end()) {
    LOG(ERROR) << "There are issues in sorting graph";
    return;
  }
  Node constant_node = graph_desc_builder_->CreateConstantNode(index);
  TensorDesc tensor_desc(GetTensorDataType(desc->data_type),
                         DML_TENSOR_FLAG_OWNED_BY_DML, desc->dimensions);
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      constant_node, 0, std::move(tensor_desc));
  node_output_map_[index] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddElementWiseUnary(UINT64 input_index,
                                        OperatorType operator_type,
                                        OperandDescriptorPtr output_desc,
                                        UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  auto* input_node_output = node_output_map_[input_index].get();
  auto output_dims = output_desc->dimensions;

  auto& input_tensor_desc = input_node_output->GetTensorDesc();
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dims);
  Node node;

  switch (operator_type) {
    case OperatorType::kCos: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_COS, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kErf: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_ERF, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kExp: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_EXP, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kIdentity: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_IDENTITY, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kSin: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_SIN, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kTan: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_TAN, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kSqrt: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_SQRT, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kSigmoid: {
      CREATE_UNARY_OPERATOR(ACTIVATION_SIGMOID, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kRelu: {
      CREATE_UNARY_OPERATOR(ACTIVATION_RELU, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kReciprocal: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_RECIP, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    case OperatorType::kLogicalNot: {
      CREATE_UNARY_OPERATOR(ELEMENT_WISE_LOGICAL_NOT, input_tensor_desc.Get(), output_tensor_desc.Get(), node);
    } break;
    default:
      DAWN_INTERNAL_ERROR("Unary elementwise op is not implemented.");
  }

  graph_desc_builder_->Connect({input_node_output}, {node});
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddElementWiseBinary(UINT64 a_index,
                                        UINT64 b_index,
                                        OperatorType operator_type,
                                        OperandDescriptorPtr output_desc,
                                        UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.contains(a_index));
  DCHECK(node_output_map_.contains(b_index));

  auto* a_node_output = node_output_map_[a_index].get();
  auto* b_node_output = node_output_map_[b_index].get();
  auto output_dims = output_desc->dimensions;

  TensorDesc a_broadcasted_tensor =
      GetBroadcastedTensorDesc(a_node_output, output_dims);
  TensorDesc b_broadcasted_tensor =
      GetBroadcastedTensorDesc(b_node_output, output_dims);

  TensorDesc output_tensor(GetTensorDataType(output_desc->data_type), output_dims);
  Node node;
  switch (operator_type) {
    case OperatorType::kAdd: {
      CREATE_BINARY_OPERATOR(ADD, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kDiv: {
      CREATE_BINARY_OPERATOR(DIVIDE, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kMul: {
      CREATE_BINARY_OPERATOR(MULTIPLY, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kSub: {
      CREATE_BINARY_OPERATOR(SUBTRACT, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kMax: {
      CREATE_BINARY_OPERATOR(MAX, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kMin: {
      CREATE_BINARY_OPERATOR(MIN, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kEqual: {
      CREATE_BINARY_OPERATOR(LOGICAL_EQUALS, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kGreater: {
      CREATE_BINARY_OPERATOR(LOGICAL_GREATER_THAN, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kLesser: {
      CREATE_BINARY_OPERATOR(LOGICAL_LESS_THAN, a_broadcasted_tensor.Get(),
                             b_broadcasted_tensor.Get(), output_tensor.Get(),
                             node);
    } break;
    case OperatorType::kPow: {
      DML_ELEMENT_WISE_POW_OPERATOR_DESC operator_desc{};
      operator_desc.InputTensor = a_broadcasted_tensor.Get();
      operator_desc.ExponentTensor = b_broadcasted_tensor.Get();
      operator_desc.OutputTensor = output_tensor.Get();
      node = graph_desc_builder_->CreateOperatorNode(
          DML_OPERATOR_ELEMENT_WISE_POW, &operator_desc);
    } break;

    default:
      DAWN_INTERNAL_ERROR("Binary elementwise op is not implemented.");
  }
  graph_desc_builder_->Connect({a_node_output, b_node_output}, {node});
  auto node_output =
      graph_desc_builder_->CreateNodeOutput(node, 0, std::move(output_tensor));
  node_output_map_[output_index] = std::move(node_output);
  return;
}

std::unique_ptr<NodeOutput> GraphDMLImpl::Clamp(NodeOutput* input_node,
                                                const ClampOptions* options) {
  auto& input_tensor = input_node->GetTensorDesc();
  TensorDesc output_tensor(input_tensor.GetDataType(),
                           input_tensor.GetDimensions());
  DML_ELEMENT_WISE_CLIP_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor.Get();
  operator_desc.OutputTensor = output_tensor.Get();
  operator_desc.ScaleBias = nullptr;
  operator_desc.Min = options->minValue;
  operator_desc.Max = options->maxValue;
  Node operator_node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_ELEMENT_WISE_CLIP, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {operator_node});
  return graph_desc_builder_->CreateNodeOutput(operator_node, 0,
                                               std::move(output_tensor));
}

void GraphDMLImpl::AddClamp(UINT64 input_index,
                            ClampOptionsPtr options,
                            UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  auto* input_node = node_output_map_[input_index].get();
  node_output_map_[output_index] = Clamp(input_node, options.get());
  return;
}

void GraphDMLImpl::EmulateFusedOperator(const OperationInfo* activation,
                                        std::unique_ptr<NodeOutput>& input_node,
                                        const std::vector<UINT>& input_dims) {
  // HardSwish and Clamp are not supported for fusion, so we add
  // them directly to
  // emulate. Currently we implement Relu6 operator by Clamp.
  if (activation == nullptr) {
    return;
  }

  if (activation->is_clamp()) {
    auto& clamp = activation->get_clamp();
    input_node = Clamp(input_node.get(), clamp->options.get());
  }
  return;
}

void GraphDMLImpl::TransposeOutputToNhwc(
    std::unique_ptr<NodeOutput>& input_node,
    const std::vector<UINT>& nchwOutputDims) {
  auto nhwcOutputStrides = transposeStrides(NchwToNhwc, nchwOutputDims);
  auto nhwcOutputDims = transposeDimensions(NchwToNhwc, nchwOutputDims);
  auto& input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc nhwc_tensor_desc(input_tensor_desc.GetDataType(),
                              input_tensor_desc.GetFlags(), nhwcOutputDims,
                              nhwcOutputStrides);
  auto node = input_node->GetNode();
  input_node = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(nhwc_tensor_desc));
  return;
}

void GraphDMLImpl::AddConv2d(UINT64 input_index,
                             UINT64 filter_index,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_index) != node_output_map_.end());
  DCHECK(node_output_map_.find(filter_index) != node_output_map_.end());

  auto* input_node = node_output_map_[input_index].get();
  auto* filter_node = node_output_map_[filter_index].get();

  auto& input_node_desc = input_node->GetTensorDesc();
  auto input_dims = input_node_desc.GetDimensions();
  auto filterDims = filter_node->GetTensorDesc().GetDimensions();
  auto output_dims = output_desc->dimensions;
  std::vector<UINT> input_nchw_dims = input_dims, filter_nchw_dims = filterDims,
                    output_nchw_dims = output_dims;

  DML_TENSOR_DESC* input_tensor_desc = input_node_desc.Get();
  TensorDesc nhwc_tensor_desc;
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    input_nchw_dims = transposeDimensions(NhwcToNchw, input_dims);
    output_nchw_dims = transposeDimensions(NhwcToNchw, output_dims);
    auto input_nchw_Strides =
        transposeStridesToNchw(input_dims, input_tensor_desc);

    nhwc_tensor_desc =
        TensorDesc(input_node_desc.GetDataType(), input_node_desc.GetFlags(),
                   input_nchw_dims, input_nchw_Strides);
    input_tensor_desc = nhwc_tensor_desc.Get();
  }

  DML_TENSOR_DESC* filter_tensor_desc = filter_node->GetTensorDesc().Get();
  TensorDesc new_filter_tensor_desc;
  if (options->filterLayout != Conv2dFilterOperandLayout::kOihw) {
    filter_nchw_dims =
        transposeFilterDimensionsAsOihw(options->filterLayout, filterDims);
    auto filter_oihw_strides =
        transposeFilterStridesAsOihw(options->filterLayout, filterDims);

    auto& fileter_desc = filter_node->GetTensorDesc();
    new_filter_tensor_desc =
        TensorDesc(fileter_desc.GetDataType(), fileter_desc.GetFlags(),
                   filter_nchw_dims, filter_oihw_strides);
    filter_tensor_desc = new_filter_tensor_desc.Get();
  }

  std::vector<NodeOutput*> input_nodes = {input_node, filter_node};
  TensorDesc bias_tensor_desc;
  if (options->bias_index != std::numeric_limits<uint64_t>::max()) {
    DCHECK(node_output_map_.find(options->bias_index) !=
           node_output_map_.end());
    auto* bias_node = node_output_map_[options->bias_index].get();
    auto& bias_desc = bias_node->GetTensorDesc();
    auto bias_dims = bias_desc.GetDimensions();
    if (bias_dims[0] != filter_nchw_dims[0] || bias_dims.size() != 1) {
      DAWN_INTERNAL_ERROR(
          "The bias should be 1-D tensor with the shape of [output_channels].");
    }

    // Reshape bias from 1-D to 4-D for NCHW layout.
    std::vector<UINT> bias_expand_dims = {1, bias_dims[0], 1, 1};
    bias_tensor_desc = TensorDesc(bias_desc.GetDataType(), bias_desc.GetFlags(),
                                  bias_expand_dims);
    input_nodes.push_back(bias_node);
  }

  // FIXME(nhu): strides, dilations, padding should be uint32_t
  // need to fix the spec.
  std::vector<UINT> strides = options->strides;
  std::vector<UINT> dilations = options->dilations;

  base::span<UINT> input_nchw_dims_span(input_nchw_dims);
  base::span<UINT> filter_nchw_dims_span(filter_nchw_dims);
  std::vector<UINT> padding =
      options->auto_pad == AutoPad::kExplicit
          ? ExplicitPadding<Conv2dOptions>(options.get())
          : ImplicitPadding<Conv2dOptions>(
              options.get(),
              input_nchw_dims_span,
              filter_nchw_dims_span
          );
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};
  std::vector<UINT> defaultOutPadding = {0, 0};

  DML_ACTIVATION_LINEAR_OPERATOR_DESC dmlActicationOperatorDesc{};
  DML_OPERATOR_DESC dmlFusedOperatorDesc = {};
  DML_OPERATOR_DESC* fusedActivation =
      CreateFusedOperator(options->activation.get(), dmlActicationOperatorDesc,
                          dmlFusedOperatorDesc);

  TensorDesc output_tensor(input_node_desc.GetDataType(), output_nchw_dims);
  DML_CONVOLUTION_OPERATOR_DESC operator_desc{};
  operator_desc.InputTensor = input_tensor_desc;
  operator_desc.FilterTensor = filter_tensor_desc;
  operator_desc.BiasTensor = bias_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor.Get();

  operator_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
  operator_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  operator_desc.DimensionCount = input_dims.size() - 2;
  operator_desc.Strides = strides.data();
  operator_desc.Dilations = dilations.data();
  operator_desc.StartPadding = startPadding.data();
  operator_desc.EndPadding = endPadding.data();
  operator_desc.OutputPadding = defaultOutPadding.data();
  operator_desc.GroupCount = static_cast<UINT>(options->groups);
  operator_desc.FusedActivation = fusedActivation;

  Node operator_node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_CONVOLUTION, &operator_desc);
  graph_desc_builder_->Connect(std::move(input_nodes), operator_node);
  auto output_node = graph_desc_builder_->CreateNodeOutput(
      operator_node, 0, std::move(output_tensor));

  // Transpose output from nchw->nhwc.
  if (options->inputLayout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, output_nchw_dims);
  }

  EmulateFusedOperator(options->activation.get(), output_node, output_dims);
  node_output_map_[output_index] = std::move(output_node);
  return;
}

void GraphDMLImpl::AddReshape(UINT64 input_index,
                              OperandDescriptorPtr output_desc,
                              UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_index) != node_output_map_.end());

  auto output_dims = output_desc->dimensions;
  auto* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc output_tensor(input_tensor_desc.GetDataType(),
                           input_tensor_desc.GetFlags(), output_dims);
  // Reshape is not a real node in DML, just need to update node output with new
  // tensor.
  auto node = input_node->GetNode();
  node_output_map_[output_index] =
      graph_desc_builder_->CreateNodeOutput(node, 0, std::move(output_tensor));
  return;
}

void GraphDMLImpl::AddGemm(UINT64 a_index,
                           UINT64 b_index,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc,
                           UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(a_index) != node_output_map_.end());
  DCHECK(node_output_map_.find(b_index) != node_output_map_.end());

  auto* a_node_output = node_output_map_[a_index].get();
  auto* b_node_output = node_output_map_[b_index].get();
  auto& output_dims = output_desc->dimensions;

  TensorDesc a_tensor_desc =
      GetBroadcastedTensorDesc(a_node_output, output_dims, 2);
  TensorDesc b_tensor_desc =
      GetBroadcastedTensorDesc(b_node_output, output_dims, 2);
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dims);

  DCHECK(a_tensor_desc.GetDimensions().size() == b_tensor_desc.GetDimensions().size());
  DCHECK(b_tensor_desc.GetDimensions().size() == output_tensor_desc.GetDimensions().size());

  // The operand c is optional.
  TensorDesc c_tensor_desc;
  std::vector<NodeOutput*> input_nodes = {a_node_output, b_node_output};
  if (options->c_index != std::numeric_limits<uint64_t>::max()) {
    DCHECK(node_output_map_.find(options->c_index) != node_output_map_.end());
    auto* c_node_output = node_output_map_[options->c_index].get();

    // Broadcast C's shape up to the output rank for DML. It enters as either
    // a scalar or as a shape that is unidirectionally broadcastable to the
    // shape [M, N] as defined in WebNN Spec.
    c_tensor_desc = GetBroadcastedTensorDesc(c_node_output, output_dims);
    input_nodes.push_back(c_node_output);
  }

  DML_MATRIX_TRANSFORM a_transpose = options->a_transpose
                                         ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                         : DML_MATRIX_TRANSFORM_NONE;
  DML_MATRIX_TRANSFORM b_transpose = options->b_transpose
                                         ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                         : DML_MATRIX_TRANSFORM_NONE;
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = a_tensor_desc.Get();
  gemm_desc.BTensor = b_tensor_desc.Get();
  gemm_desc.CTensor = c_tensor_desc.Get();
  gemm_desc.OutputTensor = output_tensor_desc.Get();
  gemm_desc.TransA = a_transpose;
  gemm_desc.TransB = b_transpose;
  gemm_desc.Alpha = options->alpha;
  gemm_desc.Beta = options->beta;

  Node operator_node =
      graph_desc_builder_->CreateOperatorNode(DML_OPERATOR_GEMM, &gemm_desc);
  graph_desc_builder_->Connect(std::move(input_nodes), {operator_node});
  node_output_map_[output_index] = graph_desc_builder_->CreateNodeOutput(
      operator_node, 0, std::move(output_tensor_desc));
  return;
}

void GraphDMLImpl::AddPool2d(UINT64 input_index,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_index) != node_output_map_.end());

  auto* input_node = node_output_map_[input_index].get();
  auto& input_node_desc = input_node->GetTensorDesc();
  auto input_dims = input_node_desc.GetDimensions();
  auto output_dims = output_desc->dimensions;
  std::vector<UINT> input_nchw_dims = input_dims,
                    output_nchw_dims = output_dims;

  DML_TENSOR_DESC* input_tensor_desc = input_node_desc.Get();
  TensorDesc nhwc_input_tensor;
  if (options->layout == InputOperandLayout::kNhwc) {
    input_nchw_dims = transposeDimensions(NhwcToNchw, input_dims);
    output_nchw_dims = transposeDimensions(NhwcToNchw, output_dims);
    auto input_nchw_strides =
        transposeStridesToNchw(input_dims, input_tensor_desc);

    nhwc_input_tensor =
        TensorDesc(input_node_desc.GetDataType(), input_node_desc.GetFlags(),
                   input_nchw_dims, input_nchw_strides);
    input_tensor_desc = nhwc_input_tensor.Get();
  }

  std::vector<UINT> strides = options->strides;
  std::vector<UINT> dilations = options->dilations;

  std::vector<UINT> windowSizes;
  if (!options->window_dimensions.empty()) {
    windowSizes = options->window_dimensions;
  } else {
    windowSizes = {input_nchw_dims[2], input_nchw_dims[3]};
  }

  // TODO:: Support AutoPad::kSameUpper and kSameLower;
  auto padding = options->auto_pad == AutoPad::kExplicit
                     ? ExplicitPadding<Pool2dOptions>(options.get())
                     : ImplicitPadding<Pool2dOptions>(
                           options.get(), input_nchw_dims, windowSizes);
  std::vector<UINT> startPadding = {padding[0], padding[2]};
  std::vector<UINT> endPadding = {padding[1], padding[3]};

  TensorDesc output_tensor(input_node_desc.GetDataType(), output_nchw_dims);
  Node operator_node;
  if (type == Pool2dType::kAveragePool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of average pool2d are not supported.");
    }
    DML_AVERAGE_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = input_tensor_desc;
    dml_desc.OutputTensor = output_tensor.Get();
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.IncludePadding = false;
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_AVERAGE_POOLING, &dml_desc);
  } else if (type == Pool2dType::kL2Pool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      DAWN_INTERNAL_ERROR("The dilations of L2 pool2d are not supported.");
    }

    DML_LP_POOLING_OPERATOR_DESC dml_desc = {};
    dml_desc.InputTensor = input_tensor_desc;
    dml_desc.OutputTensor = output_tensor.Get();
    dml_desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    dml_desc.Strides = strides.data();
    dml_desc.WindowSize = windowSizes.data();
    dml_desc.StartPadding = startPadding.data();
    dml_desc.EndPadding = endPadding.data();
    dml_desc.P = 2;
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_LP_POOLING, &dml_desc);
  } else if (type == Pool2dType::kMaxPool2d) {
    if (dilations[0] != 1 || dilations[1] != 1) {
      for (size_t i = 0; i < windowSizes.size(); ++i) {
        uint32_t paddedInputSize =
            output_nchw_dims[2 + i] + startPadding[i] + endPadding[i];
        uint32_t dilatedWindowSize = 1 + (windowSizes[i] - 1) * dilations[i];
        output_nchw_dims[2 + i] =
            (dilatedWindowSize >= paddedInputSize)
                ? 1
                : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
      }
    }

    output_tensor = TensorDesc(input_node_desc.GetDataType(), output_nchw_dims);
    DML_MAX_POOLING2_OPERATOR_DESC desc = {};
    desc.InputTensor = input_tensor_desc;
    desc.OutputTensor = output_tensor.Get();
    desc.OutputIndicesTensor = nullptr;
    desc.DimensionCount = static_cast<UINT>(windowSizes.size());
    desc.Strides = strides.data();
    desc.WindowSize = windowSizes.data();
    desc.StartPadding = startPadding.data();
    desc.EndPadding = endPadding.data();
    desc.Dilations = dilations.data();
    operator_node = graph_desc_builder_->CreateOperatorNode(
        DML_OPERATOR_MAX_POOLING2, &desc);
  } else {
    DAWN_INTERNAL_ERROR("This pool2d type is not supported.");
  }
  graph_desc_builder_->Connect({input_node}, operator_node);
  auto output_node = graph_desc_builder_->CreateNodeOutput(
      operator_node, 0, std::move(output_tensor));

  // Transpose output from nchw->nhwc.
  if (options->layout == InputOperandLayout::kNhwc) {
    TransposeOutputToNhwc(output_node, output_nchw_dims);
  }

  node_output_map_[output_index] = std::move(output_node);

  return;
}

void GraphDMLImpl::AddRelu(UINT64 input_index,
                           OperandDescriptorPtr output_desc,
                           UINT64 output_index) {
  return AddElementWiseUnary(input_index, OperatorType::kRelu,
                             std::move(output_desc), output_index);
}

void GraphDMLImpl::AddSoftmax(UINT64 input_index,
                              OperandDescriptorPtr output_desc,
                              UINT64 output_index) {
  // TODO: return directly if BuildResult has error message.
  DCHECK(node_output_map_.find(input_index) != node_output_map_.end());

  auto* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  Node node;
  CREATE_UNARY_OPERATOR(ACTIVATION_SOFTMAX, input_tensor_desc.Get(), input_tensor_desc.Get(), node);
  graph_desc_builder_->Connect({input_node}, {node});
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type),
                                input_tensor_desc.GetDimensions());
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
  return;
}

void GraphDMLImpl::AddElementWiseIf(UINT64 condition_index,
                                    UINT64 true_value_index,
                                    UINT64 false_value_index,
                                    OperandDescriptorPtr output_desc,
                                    UINT64 output_index) {
  DCHECK(node_output_map_.contains(condition_index));
  DCHECK(node_output_map_.contains(true_value_index));
  DCHECK(node_output_map_.contains(false_value_index));

  NodeOutput* condition_node = node_output_map_[condition_index].get();
  NodeOutput* true_value_node = node_output_map_[true_value_index].get();
  NodeOutput* false_value_node = node_output_map_[false_value_index].get();

  // Broadcast each of the inputs to the output.
  auto output_dimensions = output_desc->dimensions;
  TensorDesc condition_broadcasted_tensor =
      GetBroadcastedTensorDesc(condition_node, output_dimensions);
  TensorDesc true_value_broadcasted_tensor =
      GetBroadcastedTensorDesc(true_value_node, output_dimensions);
  TensorDesc false_value_broadcasted_tensor =
      GetBroadcastedTensorDesc(false_value_node, output_dimensions);
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type),
                                output_dimensions);

  DML_ELEMENT_WISE_IF_OPERATOR_DESC operator_desc = {};
  operator_desc.ConditionTensor = condition_broadcasted_tensor.Get();
  operator_desc.ATensor = true_value_broadcasted_tensor.Get();
  operator_desc.BTensor = false_value_broadcasted_tensor.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_ELEMENT_WISE_IF, &operator_desc);

  graph_desc_builder_->Connect(
      {condition_node, true_value_node, false_value_node}, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddArgMinMax(OperatorType operator_type,
                                UINT64 input_index,
                                uint32_t axis,
                                bool select_last_index,
                                OperandDescriptorPtr output_desc,
                                UINT64 output_index) {
  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& input_dimensions = input_tensor_desc.GetDimensions();
  DCHECK(node_output_map_.contains(input_index));

  // Determine output sizes. Ignore output_desc->dimensions for the dimensions,
  // since DirectML expects the output dimensions to have the same rank as the
  // input, and output_desc->dimensions may have removed dimensions if
  // keepDimensions was false.
  std::vector<uint32_t> output_dimensions = input_dimensions;
  DCHECK(axis < output_dimensions.size());
  output_dimensions[axis] = 1u;
  auto output_data_type = GetTensorDataType(output_desc->data_type);
  TensorDesc output_tensor_desc(output_data_type, output_dimensions);

  // DML accepts multiple axes. So pass the single index along.
  std::array<uint32_t, 1> axes = {axis};

  // Note DML_ARGMIN_OPERATOR_DESC and DML_ARGMAX_OPERATOR_DESC are
  // identical in structure layout. So we can use for either.
  DML_ARGMIN_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.AxisCount = 1u;
  operator_desc.Axes = axes.data();
  operator_desc.AxisDirection =
      static_cast<DML_AXIS_DIRECTION>(select_last_index);

  DML_OPERATOR_TYPE dml_operator_type = DML_OPERATOR_INVALID;
  switch (operator_type) {
    case OperatorType::kArgMin:
      dml_operator_type = DML_OPERATOR_ARGMIN;
      break;
    case OperatorType::kArgMax:
      dml_operator_type = DML_OPERATOR_ARGMAX;
      break;
    default:
      NOTREACHED();
  }
  Node node = graph_desc_builder_->CreateOperatorNode(dml_operator_type,
                                                      &operator_desc);

  graph_desc_builder_->Connect({input_node}, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddCast(UINT64 input_index,
                           OperandType data_type,
                           OperandDescriptorPtr output_desc,
                           UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& output_dimensions = output_desc->dimensions;
  auto output_data_type = GetTensorDataType(data_type);
  TensorDesc output_tensor_desc(output_data_type, output_dimensions);

  DML_CAST_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_CAST, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddConcat(base::span<const uint64_t> input_indices,
                             uint32_t axis,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {

  std::vector<NodeOutput*> input_nodes;
  std::vector<DML_TENSOR_DESC> input_tensor_descs;
  input_nodes.reserve(input_indices.size());
  input_tensor_descs.reserve(input_indices.size());

  for (uint64_t input_index : input_indices)
  {
    DCHECK(node_output_map_.contains(input_index));
    NodeOutput* input_node = node_output_map_[input_index].get();
    input_nodes.push_back(input_node);
    auto& input_tensor_desc = input_node->GetTensorDesc();
    input_tensor_descs.push_back(*input_tensor_desc.Get());
  }

  auto& output_dimensions = output_desc->dimensions;
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  DML_JOIN_OPERATOR_DESC operator_desc = {};
  operator_desc.InputCount = static_cast<uint32_t>(input_tensor_descs.size());
  operator_desc.InputTensors = input_tensor_descs.data();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.Axis = axis;

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_JOIN, &operator_desc);

  graph_desc_builder_->Connect(input_nodes, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddSlice(UINT64 input_index,
                            base::span<const uint32_t> starts,
                            base::span<const uint32_t> sizes,
                            OperandDescriptorPtr output_desc,
                            UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& output_dimensions = output_desc->dimensions;
  size_t output_rank = output_dimensions.size();
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  // WebNN v1 only supports steps of 1.
  std::vector<uint32_t> strides(output_rank, 1u);
  CHECK(starts.size() == output_rank);
  CHECK(sizes.size() == output_rank);

  DML_SLICE_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.DimensionCount = output_rank;
  operator_desc.Offsets = starts.data();
  operator_desc.Sizes = sizes.data();
  operator_desc.Strides = strides.data();

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_SLICE, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {node});
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddSplit(UINT64 input_index,
                            uint32_t axis,
                            base::span<const OperandDescriptorPtr> operands,
                            base::span<const uint64_t> output_indices
                            ) {
  std::vector<NodeOutput*> output_nodes;
  std::vector<TensorDesc> output_tensor_descs;
  std::vector<DML_TENSOR_DESC> output_tensor_dml_descs;
  output_nodes.reserve(output_indices.size());
  output_tensor_descs.reserve(output_indices.size());
  output_tensor_dml_descs.reserve(output_indices.size());

  DCHECK(node_output_map_.contains(input_index));
  NodeOutput* input_node = node_output_map_[input_index].get();
  TensorDesc& input_tensor_desc = input_node->GetTensorDesc();

  for (uint64_t output_index : output_indices)
  {
    OperandDescriptorPtr const& output_desc = operands[output_index];
    auto& output_dimensions = output_desc->dimensions;
    TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);
    output_tensor_descs.push_back(std::move(output_tensor_desc));
    output_tensor_dml_descs.push_back(*output_tensor_descs.back().Get());
  }

  DML_SPLIT_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputCount = static_cast<uint32_t>(output_tensor_dml_descs.size());
  operator_desc.OutputTensors = output_tensor_dml_descs.data();
  operator_desc.Axis = axis;

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_SPLIT, &operator_desc);

  graph_desc_builder_->Connect({input_node}, node);

  for (uint64_t i = 0, output_count = output_indices.size(); i < output_count; ++i)
  {
    std::unique_ptr<NodeOutput> node_output = graph_desc_builder_->CreateNodeOutput(
      node, i, std::move(output_tensor_descs[i]));
    uint64_t output_index = output_indices[i];
    node_output_map_[output_index] = std::move(node_output);
  }
}

void GraphDMLImpl::AddExpand(UINT64 input_index,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& output_dimensions = output_desc->dimensions;
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);
  TensorDesc input_tensor_desc =
      GetBroadcastedTensorDesc(input_node, output_dimensions);

  Node node;
  CREATE_UNARY_OPERATOR(ELEMENT_WISE_IDENTITY, input_tensor_desc.Get(), output_tensor_desc.Get(), node);

  graph_desc_builder_->Connect({input_node}, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddGather(UINT64 input_index,
                             UINT64 indices_index,
                             uint32_t axis,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  NodeOutput* indices_node = node_output_map_[indices_index].get();
  auto& indices_tensor_desc = indices_node->GetTensorDesc();
  auto& output_dimensions = output_desc->dimensions;

  size_t maximum_rank = std::max({input_tensor_desc.GetDimensions().size(),
                                  indices_tensor_desc.GetDimensions().size(),
                                  output_dimensions.size()});

  // Expand all tensor ranks to match ranks (which DML validation requires).
  TensorDesc input_expanded_desc(input_tensor_desc);
  TensorDesc indices_expanded_desc(indices_tensor_desc);
  TensorDesc output_expanded_desc(input_tensor_desc.GetDataType(), output_dimensions);
  input_expanded_desc.EnsureMinimumRank(maximum_rank, TensorDesc::Alignment::kTrailing);
  indices_expanded_desc.EnsureMinimumRank(maximum_rank, TensorDesc::Alignment::kTrailing);
  output_expanded_desc.EnsureMinimumRank(maximum_rank, TensorDesc::Alignment::kTrailing);

  DML_GATHER_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_expanded_desc.Get();
  operator_desc.IndicesTensor = indices_expanded_desc.Get();
  operator_desc.OutputTensor = output_expanded_desc.Get();
  operator_desc.IndexDimensions = indices_tensor_desc.GetDimensions().size();
  operator_desc.Axis = axis;
  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_GATHER, &operator_desc);

  graph_desc_builder_->Connect({input_node, indices_node}, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_expanded_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddInstanceNormalization(uint64_t input_index,
                                            uint64_t scale_index,
                                            uint64_t bias_index,
                                            float epsilon,
                                            InputOperandLayout operand_layout,
                                            OperandDescriptorPtr output_desc,
                                            UINT64 output_index) {
  NodeOutput* input_node = node_output_map_[input_index].get();
  TensorDesc input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc scale_tensor_desc;
  TensorDesc bias_tensor_desc;
  auto& output_dimensions = output_desc->dimensions;
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  // DirectML expects NCHW. So permute the dimensions and strides accordingly
  // if the input is anything else (e.g. NHWC).
  std::array<uint32_t, 4> tensor_dimensions_permutation = getLayoutToLayoutPermutation(operand_layout, InputOperandLayout::kNchw);
  input_tensor_desc.PermuteDimensions(tensor_dimensions_permutation, TensorDesc::Alignment::kTrailing);
  output_tensor_desc.PermuteDimensions(tensor_dimensions_permutation, TensorDesc::Alignment::kTrailing);

  // DirectML expects the channel dimension to be at NCHW.
  // So move the C dimension in 1D from XXXC to XCXX.
  const std::array<uint32_t, 4> scale_bias_dimensions_permutation = {0,3,1,2};

  std::vector<NodeOutput*> input_nodes;
  input_nodes.reserve(3);
  input_nodes.push_back(input_node);

  if (scale_index != std::numeric_limits<uint64_t>::max()) {
    NodeOutput* scale_node = node_output_map_[scale_index].get();
    scale_tensor_desc = scale_node->GetTensorDesc();
    scale_tensor_desc.PermuteDimensions(scale_bias_dimensions_permutation, TensorDesc::Alignment::kTrailing);
    input_nodes.push_back(scale_node);
  }
  if (bias_index != std::numeric_limits<uint64_t>::max()) {
    NodeOutput* bias_node = node_output_map_[bias_index].get();
    bias_tensor_desc = bias_node->GetTensorDesc();
    bias_tensor_desc.PermuteDimensions(scale_bias_dimensions_permutation, TensorDesc::Alignment::kTrailing);
    input_nodes.push_back(bias_node);
  }

  std::array<uint32_t, 2> axes = {2, 3};
  DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.ScaleTensor = scale_tensor_desc.Get();
  operator_desc.BiasTensor = bias_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.AxisCount = static_cast<uint32_t>(axes.size());
  operator_desc.Axes = axes.data();
  operator_desc.NormalizeVariance = true;
  operator_desc.Epsilon = epsilon;
  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operator_desc);

  graph_desc_builder_->Connect(input_nodes, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

#pragma optimize("", off) // TODO:::DELETE

void GraphDMLImpl::AddMeanVarianceNormalization(uint64_t input_index,
                                                uint64_t scale_index,
                                                uint64_t bias_index,
                                                float epsilon,
                                                base::span<const uint32_t> axes,
                                                OperandDescriptorPtr output_desc,
                                                UINT64 output_index) {
  NodeOutput* input_node = node_output_map_[input_index].get();
  TensorDesc input_tensor_desc = input_node->GetTensorDesc();
  TensorDesc scale_tensor_desc;
  TensorDesc bias_tensor_desc;
  auto& output_dimensions = output_desc->dimensions;
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  std::vector<NodeOutput*> input_nodes;
  input_nodes.reserve(3);
  input_nodes.push_back(input_node);

  if (scale_index != std::numeric_limits<uint64_t>::max()) {
    NodeOutput* scale_node = node_output_map_[scale_index].get();
    scale_tensor_desc = scale_node->GetTensorDesc();
    input_nodes.push_back(scale_node);
  }
  if (bias_index != std::numeric_limits<uint64_t>::max()) {
    NodeOutput* bias_node = node_output_map_[bias_index].get();
    bias_tensor_desc = bias_node->GetTensorDesc();
    input_nodes.push_back(bias_node);
  }

  DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.ScaleTensor = scale_tensor_desc.Get();
  operator_desc.BiasTensor = bias_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.AxisCount = static_cast<uint32_t>(axes.size());
  operator_desc.Axes = axes.data();
  operator_desc.NormalizeVariance = true;
  operator_desc.Epsilon = epsilon;
  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operator_desc);

  graph_desc_builder_->Connect(input_nodes, {node});

  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddPad(UINT64 input_index,
                          OperandDescriptorPtr output_desc,
                          UINT64 output_index) {
  // TODO:
}

void GraphDMLImpl::AddFillSequence(float start,
                                   float step,
                                   OperandDescriptorPtr output_desc,
                                   UINT64 output_index) {
  auto& output_dimensions = output_desc->dimensions;
  DML_TENSOR_DATA_TYPE dml_data_type =
      GetTensorDataType(output_desc->data_type);
  TensorDesc output_tensor_desc(dml_data_type, output_dimensions);

  DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC operator_desc = {};
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.ValueDataType = dml_data_type;
  operator_desc.ValueStart = GetScalarUnion(operator_desc.ValueDataType, start);
  operator_desc.ValueDelta = GetScalarUnion(operator_desc.ValueDataType, step);

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_FILL_VALUE_SEQUENCE, &operator_desc);

  graph_desc_builder_->Connect({}, {node});
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddReduce(UINT64 input_index,
                             OperatorType operator_type,
                             base::span<const uint32_t> axes,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& input_dimensions = input_tensor_desc.GetDimensions();

  // Determine output sizes. Ignore output_desc->dimensions for the dimensions,
  // since DirectML expects the output dimensions to have the same rank as the
  // input, and output_desc->dimensions may have removed dimensions if
  // keepDimensions was false.
  std::vector<uint32_t> output_dimensions = input_dimensions;
  for (uint32_t axis : axes)
  {
      DCHECK(axis < output_dimensions.size());
      output_dimensions[axis] = 1u;
  }

  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  DML_REDUCE_OPERATOR_DESC operator_desc = {};
  operator_desc.Function = MapOperatorTypeToReductionFuntion(operator_type);
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.AxisCount = static_cast<uint32_t>(axes.size());
  operator_desc.Axes = axes.data();

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_REDUCE, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {node});
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddResample2d(UINT64 input_index,
                                 ml::webnn::mojom::InterpolationMode interpolation_mode,
                                 base::span<const float> scales,
                                 base::span<const uint32_t> axes,
                                 OperandDescriptorPtr output_desc,
                                 UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));
  DCHECK(scales.size() == axes.size());

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& output_dimensions = output_desc->dimensions;
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type), output_dimensions);

  std::vector<float> full_scales(output_dimensions.size(), 1u);
  for (size_t i = 0; i < axes.size(); ++i)
  {
      auto axis = axes[i];
      DCHECK(axis < full_scales.size()); // The JS layer and mojom layer validated it.
      full_scales[axis] = scales[i];
  }

  static_assert(uint32_t(ml::webnn::mojom::InterpolationMode::kMaxValue) ==
                1);  // Update assert.
  static_assert(
      uint32_t(DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR) ==
      uint32_t(ml::webnn::mojom::InterpolationMode::kNearestNeighbor));
  static_assert(uint32_t(DML_INTERPOLATION_MODE_LINEAR) ==
                uint32_t(ml::webnn::mojom::InterpolationMode::kLinear));

  DML_RESAMPLE_OPERATOR_DESC operator_desc = {};
  operator_desc.InputTensor = input_tensor_desc.Get();
  operator_desc.OutputTensor = output_tensor_desc.Get();
  operator_desc.InterpolationMode = static_cast<DML_INTERPOLATION_MODE>(interpolation_mode);
  operator_desc.ScaleCount = static_cast<uint32_t>(full_scales.size());
  operator_desc.Scales = full_scales.data();

  Node node = graph_desc_builder_->CreateOperatorNode(
      DML_OPERATOR_RESAMPLE, &operator_desc);

  graph_desc_builder_->Connect({input_node}, {node});
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddTranspose(UINT64 input_index,
                                base::span<const uint32_t> permutation,
                                OperandDescriptorPtr output_desc,
                                UINT64 output_index) {
  DCHECK(node_output_map_.contains(input_index));

  NodeOutput* input_node = node_output_map_[input_index].get();
  auto& input_tensor_desc = input_node->GetTensorDesc();
  auto& input_dimensions = input_tensor_desc.GetDimensions();
  auto& output_dimensions = output_desc->dimensions;
  DCHECK(input_dimensions.size() == output_dimensions.size());
  input_tensor_desc.EnsureStridesExist();

  auto rearranged_input_strides =
      transposeStrides(*input_tensor_desc.GetStrides(), permutation);

  // Construct a new input tensor description based on the outputs's rearranged
  // dimensions, which is identical except for the remapped strides. Then both
  // input and output have the same sizes, just different memory mappings when
  // reading elements.
  TensorDesc remapped_input_tensor_desc(
      input_tensor_desc.GetDataType(), input_tensor_desc.GetFlags(),
      output_dimensions, rearranged_input_strides);
  TensorDesc output_tensor_desc(GetTensorDataType(output_desc->data_type),
                                output_dimensions);

  Node node;
  CREATE_UNARY_OPERATOR(ELEMENT_WISE_IDENTITY, remapped_input_tensor_desc.Get(),
                        output_tensor_desc.Get(), node);

  graph_desc_builder_->Connect({input_node}, {node});
  auto node_output = graph_desc_builder_->CreateNodeOutput(
      node, 0, std::move(output_tensor_desc));
  node_output_map_[output_index] = std::move(node_output);
}

void GraphDMLImpl::AddTriangularMatrix(UINT64 input_index,
                             OperandDescriptorPtr output_desc,
                             UINT64 output_index) {
  // TODO:
}

void GraphDMLImpl::AddOutput(const std::string& name, UINT64 index) {
  DCHECK(node_output_map_.find(index) != node_output_map_.end());
  auto* output_node = node_output_map_[index].get();
  DCHECK(output_node != nullptr);

  // Append identity to avoid directly using graph input as output, and
  // avoid lack of considering the impacts of strides if there are.
  auto node = output_node->GetNode();
  if (node.type == NodeType::kInput || node.type == NodeType::kConstant ||
      output_node->GetTensorDesc().GetStrides()) {
    auto& input_tensor = output_node->GetTensorDesc();

    TensorDesc output_tensor(input_tensor.GetDataType(),
                             input_tensor.GetDimensions());

    CREATE_UNARY_OPERATOR(ELEMENT_WISE_IDENTITY, input_tensor.Get(), output_tensor.Get(), node);

    graph_desc_builder_->Connect({output_node}, {node});
    std::unique_ptr<NodeOutput> identity_output_node =
        graph_desc_builder_->CreateNodeOutput(node, 0,
                                              std::move(output_tensor));
    graph_desc_builder_->AddOutputEdge(identity_output_node.get(), name);
  } else {
    graph_desc_builder_->AddOutputEdge(output_node, name);
  }
  return;
}

void GraphDMLImpl::Build(ModelInfoPtr model_info, BuildCallback callback) {
  BuildResult build_result = BuildResult::kOk;
  Build(std::move(model_info), /*out*/ &build_result); // Bool return ignored.
  std::move(callback).Run(build_result);
  return;
}

bool GraphDMLImpl::Build(ModelInfoPtr model_info, BuildResult* out_result) {
  // Add Input
  for (auto& input : model_info->inputs) {
    auto& operand_desc = model_info->operands[input->index];
    AddInput(std::move(input->name), std::move(operand_desc), input->index);
  }

  // Add Constant
  std::unique_ptr<UploadResource> uploader =
      std::make_unique<UploadResource>(execution_context_.get());
  ComPtr<gpgmm::d3d12::ResourceAllocation> constants_resource = nullptr;
  auto constants_info = std::move(model_info->constants);

  if (constants_info.get() != nullptr) {
    for (auto& [index, _] : constants_info->memory_info) {
      auto& operand_desc = model_info->operands[index];
      AddConstant(std::move(operand_desc), index);
    }
    // Upload the data to GPU so that the constant data are not saved as member
    // variable.
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        constants_info->shared_memory;
    size_t constants_byte_length = shared_memory_region.GetSize();
    ExecutionResources* execution_resources =
        execution_context_->GetExecutionResources();
    constants_resource = execution_resources->Allocate(constants_byte_length);
    uploader->UploadConstants(constants_resource->GetResource(),
                              constants_info);
  }

  // Add operations
  for (auto& operation : model_info->operations) {
    switch (operation->which()) {
      case OperationInfo::Tag::kClamp: {
        auto& clamp = operation->get_clamp();
        AddClamp(clamp->input_index, std::move(clamp->options),
                 clamp->output_index);
        break;
      }
      case OperationInfo::Tag::kConv2d: {
        auto& conv2d = operation->get_conv2d();
        auto& output_operand = model_info->operands[conv2d->output_index];
        AddConv2d(conv2d->input_index, conv2d->filter_index,
                  std::move(conv2d->options), std::move(output_operand),
                  conv2d->output_index);
        break;
      }
      case OperationInfo::Tag::kElementWiseUnary: {
        auto& mojom_operator = operation->get_element_wise_unary();
        auto& output_operand = model_info->operands[mojom_operator->output_index];
        AddElementWiseUnary(mojom_operator->input_index, mojom_operator->operator_type,
                            std::move(output_operand),
                            mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kElementWiseBinary: {
        auto& binary = operation->get_element_wise_binary();
        auto& output_operand = model_info->operands[binary->output_index];
        AddElementWiseBinary(binary->a_index, binary->b_index, binary->operator_type,
                             std::move(output_operand), binary->output_index);
        break;
      }
      case OperationInfo::Tag::kGemm: {
        // GEMM and MatMul.
        auto& gemm = operation->get_gemm();
        auto& output_operand = model_info->operands[gemm->output_index];
        AddGemm(gemm->a_index, gemm->b_index, std::move(gemm->options),
                std::move(output_operand), gemm->output_index);
        break;
      }
      case OperationInfo::Tag::kPool2d: {
        auto& pool2d = operation->get_pool2d();
        auto& output_operand = model_info->operands[pool2d->output_index];
        AddPool2d(pool2d->input_index, std::move(pool2d->options), pool2d->type,
                  std::move(output_operand), pool2d->output_index);
        break;
      }
      case OperationInfo::Tag::kRelu: {
        auto& relu = operation->get_relu();
        auto& output_operand = model_info->operands[relu->output_index];
        AddRelu(relu->input_index, std::move(output_operand),
                relu->output_index);
        break;
      }
      case OperationInfo::Tag::kReshape: {
        auto& reshape = operation->get_reshape();
        auto& output_operand = model_info->operands[reshape->output_index];
        AddReshape(reshape->input_index, std::move(output_operand),
                   reshape->output_index);
        break;
      }
      case OperationInfo::Tag::kSoftmax: {
        auto& softmax = operation->get_softmax();
        auto& output_operand = model_info->operands[softmax->output_index];
        AddSoftmax(softmax->input_index, std::move(output_operand),
                   softmax->output_index);
        break;
      }
      case OperationInfo::Tag::kArgMinMax: {
        auto& mojom_operator = operation->get_arg_min_max();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddArgMinMax(mojom_operator->operator_type, mojom_operator->input_index,
                     mojom_operator->axis,
                     mojom_operator->select_last_index,
                     std::move(output_operand), mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kCast: {
        auto& mojom_operator = operation->get_cast();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddCast(mojom_operator->input_index, mojom_operator->data_type,
                std::move(output_operand), mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kConcat: {
        auto& mojom_operator = operation->get_concat();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddConcat(mojom_operator->input_indices, mojom_operator->axis,
                  std::move(output_operand), mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kSlice: {
        auto& mojom_operator = operation->get_slice();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddSlice(mojom_operator->input_index, mojom_operator->starts,
                 mojom_operator->sizes, std::move(output_operand),
                 mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kSplit: {
        auto& mojom_operator = operation->get_split();
        AddSplit(mojom_operator->input_index, mojom_operator->axis,
                  model_info->operands, mojom_operator->output_indices);
        break;
      }
      case OperationInfo::Tag::kExpand: {
        auto& mojom_operator = operation->get_expand();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddExpand(mojom_operator->input_index, std::move(output_operand),
                  mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kGather: {
        auto& mojom_operator = operation->get_gather();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddGather(mojom_operator->input_index, mojom_operator->indices_index,
                  mojom_operator->axis, std::move(output_operand),
                  mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kInstanceNormalization: {
        auto& mojom_operator = operation->get_instance_normalization();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddInstanceNormalization(
            mojom_operator->input_index, mojom_operator->scale_index,
            mojom_operator->bias_index, mojom_operator->epsilon,
            mojom_operator->layout, std::move(output_operand),
            mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kMeanVarianceNormalization: {
        auto& mojom_operator = operation->get_mean_variance_normalization();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddMeanVarianceNormalization(
            mojom_operator->input_index, mojom_operator->scale_index,
            mojom_operator->bias_index, mojom_operator->epsilon,
            mojom_operator->axes, std::move(output_operand),
            mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kPad:
        DCHECK(false);
        break;
      case OperationInfo::Tag::kFillSequence: {
        auto& mojom_operator = operation->get_fill_sequence();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddFillSequence(mojom_operator->start, mojom_operator->delta,
                        std::move(output_operand),
                        mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kReduce: {
        auto& mojom_operator = operation->get_reduce();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddReduce(mojom_operator->input_index, mojom_operator->operator_type,
                  mojom_operator->axes, std::move(output_operand),
                  mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kResample2d: {
        auto& mojom_operator = operation->get_resample2d();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddResample2d(mojom_operator->input_index,
                      mojom_operator->interpolation_mode,
                      mojom_operator->scales, mojom_operator->axes,
                      std::move(output_operand), mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kTranspose: {
        auto& mojom_operator = operation->get_transpose();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddTranspose(mojom_operator->input_index, mojom_operator->permutation,
                     std::move(output_operand), mojom_operator->output_index);
        break;
      }
      case OperationInfo::Tag::kTriangularMatrix:
        DCHECK(false);
        break;
      case OperationInfo::Tag::kElementWiseIf: {
        auto& mojom_operator = operation->get_element_wise_if();
        auto& output_operand =
            model_info->operands[mojom_operator->output_index];
        AddElementWiseIf(
            mojom_operator->condition_index, mojom_operator->true_value_index,
            mojom_operator->false_value_index, std::move(output_operand),
            mojom_operator->output_index);
        break;
      }

      default:
        NOTREACHED();
    }
  }

  // Add Output with named operands.
  for (auto& output : model_info->outputs) {
    AddOutput(std::move(output->name), output->index);
  }

  // Finish the graph build.
  mCompiledOperator = graph_desc_builder_->Compile(DML_EXECUTION_FLAG_NONE);
  if (!mCompiledOperator) {
    *out_result = BuildResult::kUnknownError;
    return false;
  }

  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kConstant) {
      input_buffer_binding[i].Buffer = constants_resource->GetResource();
      auto& memory_info = constants_info->memory_info[input.object_id];
      input_buffer_binding[i].Offset = memory_info->byte_offset;
      input_buffer_binding[i].SizeInBytes = memory_info->byte_length;
    }
  }

  DML_BUFFER_ARRAY_BINDING input_buffer_array_binding = {};
  input_buffer_array_binding.BindingCount = input_buffer_binding.size();
  input_buffer_array_binding.Bindings = input_buffer_binding.data();
  DML_BINDING_DESC input_binding_desc{DML_BINDING_TYPE_BUFFER_ARRAY,
                                      &input_buffer_array_binding};

  execution_context_->InitializeGraph(this, mCompiledOperator.Get(),
                                      input_binding_desc);

  execution_context_->Flush();
  execution_context_->WaitForSignal();
  execution_context_->ReleaseCompletedResources();

  auto& named_outputs = graph_desc_builder_->GetNamedOutputs();
  HRESULT hr = output_resource_readback_->InitializeResource(named_outputs);
  if (FAILED(hr)) {
    *out_result = BuildResult::kUnknownError;
    return false;
  }
  *out_result = BuildResult::kOk;
  return true;
}

void GraphDMLImpl::Compute(NamedResourcesPtr named_inputs,
                           ComputeCallback callback) {
  ExecutionResources* execution_resources =
      execution_context_->GetExecutionResources();
  ID3D12Resource* inputs_resource =
      execution_resources->GetResource(this, ResourceType::kInput);
  if (inputs_resource == nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        named_inputs->shared_memory;
    DCHECK(shared_memory_region.IsValid());
    size_t inputs_byte_length = shared_memory_region.GetSize();
    inputs_resource = execution_resources->Allocate(ResourceType::kInput,
                                                    inputs_byte_length, this);
  }
  input_resource_uploader_->UploadInputs(inputs_resource, named_inputs);
  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  std::vector<DML_BINDING_DESC> input_binding_desc(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kInput) {
      input_buffer_binding[i].Buffer = inputs_resource;
      auto& memory_info = named_inputs->resources[input.name];
      input_buffer_binding[i].Offset = memory_info->byte_offset;
      input_buffer_binding[i].SizeInBytes = memory_info->byte_length;

      input_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                               &input_buffer_binding[i]};
    }
  }

  ID3D12Resource* outputs_resource =
      execution_resources->GetResource(this, ResourceType::kOutput);
  if (outputs_resource == nullptr) {
    size_t outputs_resource_size =
        output_resource_readback_->GetOutputsResourceSize();
    outputs_resource = execution_resources->Allocate(
        ResourceType::kOutput, outputs_resource_size, this);
  }
  auto& output_length_map = graph_desc_builder_->GetNamedOutputs();
  std::vector<DML_BINDING_DESC> output_binding_desc(output_length_map.size());
  // The sort of the outputs from Graph Compute is different from the
  // outputs from Graph Build, so the offset need to be found the correct output
  // with name to read back from GPU buffer.
  base::flat_map<std::string, DML_BUFFER_BINDING> output_buffer_binding;
  uint64_t aligned_offset = 0;
  size_t i = 0;
  for (auto& [name, byte_length] : output_length_map) {
    DML_BUFFER_BINDING buffer_binding;
    buffer_binding.Buffer = outputs_resource;
    buffer_binding.Offset = aligned_offset;
    buffer_binding.SizeInBytes = byte_length;
    output_buffer_binding[name] = buffer_binding;
    output_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                              &output_buffer_binding[name]};
    aligned_offset += Align(byte_length, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    ++i;
  }

  execution_context_->ExecuteGraph(this, mCompiledOperator.Get(),
                                   input_binding_desc, output_binding_desc);

  auto named_outputs = ml::webnn::mojom::NamedResources::New();
  HRESULT hr = output_resource_readback_->ReadResourceFromGpu(named_outputs,
                                                              outputs_resource);
  if (FAILED(hr)) {
    std::move(callback).Run(ComputeResult::kUnknownError, nullptr);
    return;
  }

  std::move(callback).Run(ComputeResult::kOk, std::move(named_outputs));
}

bool GraphDMLImpl::Compute(NamedResourcesPtr named_inputs,
                           ComputeResult* out_result,
                           NamedResourcesPtr* out_named_outputs) {
  ExecutionResources* execution_resources =
      execution_context_->GetExecutionResources();
  ID3D12Resource* inputs_resource =
      execution_resources->GetResource(this, ResourceType::kInput);
  if (inputs_resource == nullptr) {
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        named_inputs->shared_memory;
    DCHECK(shared_memory_region.IsValid());
    size_t inputs_byte_length = shared_memory_region.GetSize();
    inputs_resource = execution_resources->Allocate(ResourceType::kInput,
                                                    inputs_byte_length, this);
  }
  input_resource_uploader_->UploadInputs(inputs_resource, named_inputs);
  auto input_nodes = graph_desc_builder_->GetInputNodes();
  std::vector<DML_BUFFER_BINDING> input_buffer_binding(input_nodes.size());
  std::vector<DML_BINDING_DESC> input_binding_desc(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto input = input_nodes[i];
    if (input.type == NodeType::kInput) {
      input_buffer_binding[i].Buffer = inputs_resource;
      auto& memory_info = named_inputs->resources[input.name];
      input_buffer_binding[i].Offset = memory_info->byte_offset;
      input_buffer_binding[i].SizeInBytes = memory_info->byte_length;

      input_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                               &input_buffer_binding[i]};
    }
  }

  ID3D12Resource* outputs_resource =
      execution_resources->GetResource(this, ResourceType::kOutput);
  if (outputs_resource == nullptr) {
    size_t outputs_resource_size =
        output_resource_readback_->GetOutputsResourceSize();
    outputs_resource = execution_resources->Allocate(
        ResourceType::kOutput, outputs_resource_size, this);
  }
  auto& output_length_map = graph_desc_builder_->GetNamedOutputs();
  std::vector<DML_BINDING_DESC> output_binding_desc(output_length_map.size());
  // The sort of the outputs from Graph Compute is different from the
  // outputs from Graph Build, so the offset need to be found the correct output
  // with name to read back from GPU buffer.
  base::flat_map<std::string, DML_BUFFER_BINDING> output_buffer_binding;
  uint64_t aligned_offset = 0;
  size_t i = 0;
  for (auto& [name, byte_length] : output_length_map) {
    DML_BUFFER_BINDING buffer_binding;
    buffer_binding.Buffer = outputs_resource;
    buffer_binding.Offset = aligned_offset;
    buffer_binding.SizeInBytes = byte_length;
    output_buffer_binding[name] = buffer_binding;
    output_binding_desc[i] = {DML_BINDING_TYPE_BUFFER,
                              &output_buffer_binding[name]};
    aligned_offset += Align(byte_length, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    ++i;
  }

  execution_context_->ExecuteGraph(this, mCompiledOperator.Get(),
                                   input_binding_desc, output_binding_desc);

  *out_named_outputs = ml::webnn::mojom::NamedResources::New();
  HRESULT hr = output_resource_readback_->ReadResourceFromGpu(
      *out_named_outputs, outputs_resource);
  if (FAILED(hr)) {
    *out_result = ComputeResult::kUnknownError;
    *out_named_outputs = nullptr;
    return false;
  }
  *out_result = ComputeResult::kOk;
  return true;
}

}  // namespace content::webnn
