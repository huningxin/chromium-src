// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

MLGraphXnnpack::MLGraphXnnpack(MLContext* context) : MLGraph(context) {}

bool MLGraphXnnpack::Build(
    const MLNamedOperands& named_outputs,
    const std::vector<const MLOperand*>& inputs,
    const std::vector<const MLOperand*>& constants,
    const std::vector<const MLOperator*>& sorted_operators,
    ExceptionState& exception_state) {
  size_t externals_size = named_outputs.Size();
  for (auto* input : inputs) {
    if (input->Kind() == MLOperand::KindEnum::kInput) {
      externals_size++;
    }
  }
  xnn_subgraph_t subgraph_ptr = nullptr;
  if (xnn_create_subgraph(externals_size, 0, &subgraph_ptr) !=
      xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to create XNNPACK subgraph");
    return false;
  }
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  std::unordered_map<const MLOperand*, uint32_t> tensors_map;
  uint32_t external_id = 0;
  for (auto* input : inputs) {
    uint32_t flags = 0;
    if (input->Kind() == MLOperand::KindEnum::kInput) {
      flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
      tensors_map[input] = external_id++;
    }
    if (!DefineTensor(subgraph.get(), tensors_map, input, flags,
                      exception_state)) {
      return false;
    }
  }
  for (auto& named_output : named_outputs) {
    uint32_t flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    auto* output = named_output.second;
    tensors_map[output] = external_id++;
    if (!DefineTensor(subgraph.get(), tensors_map, output, flags,
                      exception_state)) {
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
                                          "the operator is not supported");
        xnn_delete_subgraph(subgraph);
        return false;
    }
    if (xnn_create_runtime(subgraph.get(), &runtime_) != xnn_status_success) {
      exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                        "failed to create XNNPACK runtime.");
      return false;
    }
    return true;
  }

  bool MLGraphXnnpack::DefineTensor(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperand* operand, uint32_t flags,
      ExceptionState& exception_state) {
    return true;
  }

  bool MLGraphXnnpack::DefineClamp(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* clamp, const MLClampOptions* options,
      ExceptionState& exception_state) {
    return true;
  }

  bool MLGraphXnnpack::DefineConv2d(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* conv2d, const MLConv2dOptions* options,
      ExceptionState& exception_state) {
    return true;
  }
  bool MLGraphXnnpack::DefineBinary(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* binary, ExceptionState& exception_state) {
    return true;
  }
  bool MLGraphXnnpack::DefineGemm(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* gemm, const MLGemmOptions* options,
      ExceptionState& exception_state) {
    return true;
  }

  bool MLGraphXnnpack::DefineAveragePool2d(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* pool2d, const MLPool2dOptions* options,
      ExceptionState& exception_state) {
    return true;
  }

  bool MLGraphXnnpack::DefineReshape(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* reshape, ExceptionState& exception_state) {
    return true;
  }

  bool MLGraphXnnpack::DefineUnary(
      xnn_subgraph_t subgraph,
      const std::unordered_map<const MLOperand*, uint32_t>& tensors_map,
      const MLOperator* unary, ExceptionState& exception_state) {
    return true;
  }

}  // namespace blink
