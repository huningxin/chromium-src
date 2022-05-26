// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#include <memory>

namespace blink {

namespace {

String DataTypeToString(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return "float32";
    case V8MLOperandType::Enum::kFloat16:
      return "float16";
    case V8MLOperandType::Enum::kInt32:
      return "int32";
    case V8MLOperandType::Enum::kUint32:
      return "uint32";
    case V8MLOperandType::Enum::kInt8:
      return "int8";
    case V8MLOperandType::Enum::kUint8:
      return "uint8";
  }
}

String OpKindToString(MLOperator::OpKind kind) {
  switch (kind) {
    case MLOperator::OpKind::kClamp:
      return "clamp";
    case MLOperator::OpKind::kConv2d:
      return "conv2d";
    case MLOperator::OpKind::kAdd:
      return "add";
    case MLOperator::OpKind::kGemm:
      return "gemm";
    case MLOperator::OpKind::kAveragePool2d:
      return "averagePool2d";
    case MLOperator::OpKind::kReshape:
      return "reshape";
    case MLOperator::OpKind::kSoftmax:
      return "softmax";
  }
}

String XnnStatusToString(xnn_status status) {
  switch (status) {
    case xnn_status_success:
      return "xnn_status_success";
    case xnn_status_uninitialized:
      return "xnn_status_uninitialized";
    case xnn_status_invalid_parameter:
      return "xnn_status_invalid_parameter";
    case xnn_status_invalid_state:
      return "xnn_status_invalid_state";
    case xnn_status_unsupported_parameter:
      return "xnn_status_unsupported_parameter";
    case xnn_status_unsupported_hardware:
      return "xnn_status_unsupported_hardware";
    case xnn_status_out_of_memory:
      return "xnn_status_out_of_memory";
  }
}

size_t GetBytesPerElement(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return 4;
    case V8MLOperandType::Enum::kFloat16:
      return 2;
    case V8MLOperandType::Enum::kInt32:
      return 4;
    case V8MLOperandType::Enum::kUint32:
      return 4;
    case V8MLOperandType::Enum::kInt8:
      return 1;
    case V8MLOperandType::Enum::kUint8:
      return 1;
  }
}

size_t GetByteLength(const MLOperand* operand) {
  size_t elements = 1;
  for (auto& d : operand->Dimensions()) {
    elements = elements * d;
  }
  return elements * GetBytesPerElement(operand->Type());
}

}  // namespace

MLGraphXnnpack::MLGraphXnnpack(MLContext* context)
    : MLGraph(context), runtime_(nullptr) {}

MLGraphXnnpack::~MLGraphXnnpack() {
  if (runtime_ != nullptr) {
    xnn_delete_runtime(runtime_);
  }
}

bool MLGraphXnnpack::BuildImpl(
    const MLNamedOperands& named_outputs,
    const std::vector<const MLOperand*>& inputs,
    const std::vector<const MLOperand*>& constants,
    const std::vector<const MLOperator*>& sorted_operators,
    ExceptionState& exception_state) {
  uint32_t externals_size =
      static_cast<uint32_t>(named_outputs.size() + inputs.size());
  xnn_subgraph_t subgraph_ptr = nullptr;
  xnn_status status;
  if ((status = xnn_create_subgraph(externals_size, 0, &subgraph_ptr)) !=
      xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to create XNNPACK subgraph: " + XnnStatusToString(status));
    return false;
  }

  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  std::unordered_map<const MLOperand*, uint32_t> tensors_map;
  uint32_t external_id = 0;
  for (auto* input : inputs) {
    uint32_t input_id = external_id++;
    tensors_map[input] = input_id;
    if (!DefineTensor(subgraph.get(), tensors_map, input, exception_state,
                      true)) {
      return false;
    }
    TensorValueInfo info = {0};
    info.id = input_id;
    info.byte_length = GetByteLength(input);
    inputs_info_.insert(input->Name(), std::move(info));
  }
  for (auto& named_output : named_outputs) {
    auto* output = named_output.second.Get();
    uint32_t output_id = external_id++;
    tensors_map[output] = output_id;
    if (!DefineTensor(subgraph.get(), tensors_map, output, exception_state,
                      true)) {
      return false;
    }
    TensorValueInfo info = {0};
    info.id = output_id;
    info.byte_length = GetByteLength(output);
    outputs_info_.insert(named_output.first, std::move(info));
  }
  for (auto* constant : constants) {
    if (!DefineTensor(subgraph.get(), tensors_map, constant, exception_state)) {
      return false;
    }
  }
  for (auto* op : sorted_operators) {
    switch (op->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* options =
            static_cast<const MLClampOptions*>(op->Options());
        if (!DefineClamp(subgraph.get(), tensors_map, op, options,
                         exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kConv2d: {
        const MLConv2dOptions* options =
            static_cast<const MLConv2dOptions*>(op->Options());
        if (!DefineConv2d(subgraph.get(), tensors_map, op, options,
                          exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kAdd: {
        if (!DefineBinary(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kGemm: {
        const MLGemmOptions* options =
            static_cast<const MLGemmOptions*>(op->Options());
        if (!DefineGemm(subgraph.get(), tensors_map, op, options,
                        exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kAveragePool2d: {
        const MLPool2dOptions* options =
            static_cast<const MLPool2dOptions*>(op->Options());
        if (!DefinePool2d(subgraph.get(), tensors_map, op, options,
                          exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kReshape: {
        if (!DefineReshape(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kSoftmax: {
        if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      default:
        exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                          "the operator (" +
                                              OpKindToString(op->Kind()) +
                                              ") is not supported");
        return false;
    }
  }
  uint32_t flags = XNN_FLAG_YIELD_WORKERS;
  if (xnn_create_runtime_v2(
          subgraph.get(),
          static_cast<MLContextXnnpack*>(ml_context_.Get())->Pthreadpool(),
          flags, &runtime_) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to create XNNPACK runtime");
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineTensor(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperand* operand,
    ExceptionState& exception_state,
    bool is_external) {
  xnn_datatype data_type = xnn_datatype_invalid;
  if (operand->Type() == V8MLOperandType::Enum::kFloat32) {
    data_type = xnn_datatype_fp32;
  } else {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "the data type (" +
                                          DataTypeToString(operand->Type()) +
                                          ") is not supported");
    return false;
  }
  std::vector<size_t> dims;
  for (auto& d : operand->Dimensions()) {
    if (d < 0) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kNotSupportedError,
          "the negative dimension is not supported");
      return false;
    }
    dims.push_back(static_cast<size_t>(d));
  }
  uint32_t flags = 0;
  uint32_t external_id = XNN_INVALID_VALUE_ID;
  std::unique_ptr<char> data;
  if (is_external) {
    DCHECK(tensors_map.find(operand) != tensors_map.end());
    external_id = tensors_map.at(operand);
    if (operand->Kind() == MLOperand::KindEnum::kInput) {
      flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
    } else {
      DCHECK(operand->Kind() == MLOperand::KindEnum::kOutput);
      flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    }
  } else {
    if (operand->Kind() == MLOperand::KindEnum::kConstant) {
      auto* array_buffer_view = operand->ArrayBufferView();
      data.reset(new char[array_buffer_view->byteLength()]);
      if (data.get() == nullptr) {
        exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                          "out of memory.");
        return false;
      }
      memcpy(data.get(),
             reinterpret_cast<char*>(array_buffer_view->BaseAddress()) +
                 array_buffer_view->byteOffset(),
             array_buffer_view->byteLength());
    }
  }
  uint32_t tensor_id;
  xnn_status status;
  if ((status = xnn_define_tensor_value(
           subgraph, data_type, dims.size(), dims.data(), data.get(),
           external_id, flags, &tensor_id)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define tensor value: " + XnnStatusToString(status));
    return false;
  }
  if (data) {
    constant_data_.push_back(std::move(data));
  }
  if (!is_external) {
    tensors_map.insert(std::make_pair(operand, tensor_id));
  }
  return true;
}

bool MLGraphXnnpack::DefineClamp(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* clamp,
    const MLClampOptions* options,
    ExceptionState& exception_state) {
  auto* input = clamp->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = clamp->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = options->hasMinValue()
                               ? options->minValue()
                               : -std::numeric_limits<float>::infinity();
  const float output_max = options->hasMaxValue()
                               ? options->maxValue()
                               : +std::numeric_limits<float>::infinity();
  xnn_status status;
  if ((status = xnn_define_clamp(subgraph, output_min, output_max, input_id,
                                 output_id, 0)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define clamp: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineConv2d(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* conv2d,
    const MLConv2dOptions* options,
    ExceptionState& exception_state) {
  return true;
}

bool MLGraphXnnpack::DefineBinary(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* binary,
    ExceptionState& exception_state) {
  auto* input0 = binary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input0) != tensors_map.end());
  uint32_t input0_id = tensors_map.at(input0);
  auto* input1 = binary->Inputs()[1].Get();
  DCHECK(tensors_map.find(input1) != tensors_map.end());
  uint32_t input1_id = tensors_map.at(input1);
  auto* output = binary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = +std::numeric_limits<float>::infinity();
  xnn_status status = xnn_status_success;
  switch (binary->Kind()) {
    case MLOperator::OpKind::kAdd: {
      status = xnn_define_add2(subgraph, output_min, output_max, input0_id,
                               input1_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to define " +
                                          OpKindToString(binary->Kind()) +
                                          ": " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineGemm(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* gemm,
    const MLGemmOptions* options,
    ExceptionState& exception_state) {
  return true;
}

bool MLGraphXnnpack::DefinePool2d(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* pool2d,
    const MLPool2dOptions* options,
    ExceptionState& exception_state) {
  return true;
}

bool MLGraphXnnpack::DefineReshape(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* reshape,
    ExceptionState& exception_state) {
  auto* input = reshape->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = reshape->Outputs()[0].Get();
  std::vector<size_t> new_sizes;
  for (auto& d : output->Dimensions()) {
    new_sizes.push_back(static_cast<size_t>(d));
  }
  if (new_sizes.size() > XNN_MAX_TENSOR_DIMS) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The rank of new shape is not supported.");
    return false;
  }
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  xnn_status status = xnn_status_success;
  if ((status = xnn_define_static_reshape(subgraph, new_sizes.size(),
                                          new_sizes.data(), input_id, output_id,
                                          0)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define reshape: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineUnary(
    xnn_subgraph_t subgraph,
    std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
    const MLOperator* unary,
    ExceptionState& exception_state) {
  auto* input = unary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = unary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  xnn_status status = xnn_status_success;
  switch (unary->Kind()) {
    case MLOperator::OpKind::kSoftmax: {
      status = xnn_define_softmax(subgraph, input_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to define " +
                                          OpKindToString(unary->Kind()) + ": " +
                                          XnnStatusToString(status));
    return false;
  }
  return true;
}

void MLGraphXnnpack::ComputeImpl(const MLNamedArrayInputs& inputs,
                                 const MLNamedArrayOutputs& outputs,
                                 ExceptionState& exception_state) {
  std::vector<xnn_external_value> external_values;
  if (inputs.size() != inputs_info_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of inputs is invalid");
    return;
  }
  if (outputs.size() != outputs_info_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of outputs is invalid");
    return;
  }
  for (const auto& input : inputs) {
    auto iter = inputs_info_.find(input.first);
    if (iter == inputs_info_.end()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "There is unknown input: " + input.first);
      return;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    DOMArrayBufferView* array_buffer_view = nullptr;
    if (input.second->IsArrayBufferView()) {
      array_buffer_view = input.second->GetAsArrayBufferView().Get();
    } else if (input.second->IsMLTensor()) {
      auto* ml_tensor = input.second->GetAsMLTensor();
      array_buffer_view = ml_tensor->data().Get();
    }
    DCHECK(array_buffer_view != nullptr);
    if (array_buffer_view->byteLength() < iter->value.byte_length) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The input (" + input.first + ") buffer length is invalid.");
      return;
    }
    value.data = reinterpret_cast<char*>(array_buffer_view->BaseAddress()) +
                 array_buffer_view->byteOffset();
    external_values.push_back(value);
  }
  for (const auto& output : outputs) {
    auto iter = outputs_info_.find(output.first);
    if (iter == outputs_info_.end()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "There is unknown output: " + output.first);
      return;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    if (output.second->IsArrayBufferView()) {
      DOMArrayBufferView* array_buffer_view =
          output.second->GetAsArrayBufferView().Get();
      if (array_buffer_view->byteLength() < iter->value.byte_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The output (" + output.first + ") buffer length is invalid.");
        return;
      }
      value.data = reinterpret_cast<char*>(array_buffer_view->BaseAddress()) +
                   array_buffer_view->byteOffset();
    } else if (output.second->IsArrayBuffer()) {
      DOMArrayBuffer* array_buffer = output.second->GetAsArrayBuffer();
      if (array_buffer->ByteLength() < iter->value.byte_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The output (" + output.first + ") buffer length is invalid.");
        return;
      }
      value.data = array_buffer->Data();
    }
    DCHECK(value.data);
    external_values.push_back(value);
  }
  if (xnn_setup_runtime(runtime_, external_values.size(),
                        external_values.data()) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to setup runtime");
    return;
  }
  if (xnn_invoke_runtime(runtime_) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to invoke runtime");
    return;
  }
}

}  // namespace blink
