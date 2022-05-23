// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

namespace blink {

namespace {

bool BroadcastShape(Vector<int32_t> dims_input0,
                    Vector<int32_t> dims_input1,
                    Vector<int32_t>& dims_output,
                    wtf_size_t skip_axes = 0) {
  // The rank of the output tensor is the maximum rank of the input tensors.
  auto rank_a = dims_input0.size(), rank_b = dims_input1.size();
  auto rank_c = rank_a >= rank_b ? rank_a : rank_b;
  dims_output.resize(rank_c);
  DCHECK(rank_a >= skip_axes && rank_b >= skip_axes);
  // For each dimension of the output tensor, its size is the maximum size along
  // that dimension of the input tensors.
  for (wtf_size_t i = 0; i < rank_c; ++i) {
    // Skip some axes from the right side when broadcasting.
    if (i >= skip_axes) {
      auto dim_a = i < rank_a ? dims_input0[rank_a - i - 1] : 1;
      auto dim_b = i < rank_b ? dims_input1[rank_b - i - 1] : 1;
      if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
        return false;
      }
      dims_output[rank_c - i - 1] = dim_a > dim_b ? dim_a : dim_b;
    }
  }
  return true;
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

MLOperand* MLGraphBuilder::input(String name,
                                 const MLOperandDescriptor* desc,
                                 ExceptionState& exception_state) {
  auto* input =
      MakeGarbageCollected<MLOperand>(this, MLOperand::KindEnum::kInput);
  input->SetName(name);
  input->SetType(desc->type().AsEnum());
  if (desc->hasDimensions()) {
    input->SetDimensions(desc->dimensions());
  }
  return input;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc,
                                    NotShared<DOMArrayBufferView> buffer_view,
                                    ExceptionState& exception_state) {
  auto* constant =
      MakeGarbageCollected<MLOperand>(this, MLOperand::KindEnum::kConstant);
  constant->SetType(desc->type().AsEnum());
  if (desc->hasDimensions()) {
    constant->SetDimensions(desc->dimensions());
  }
  constant->SetArrayBufferView(buffer_view.Get());
  return constant;
}

MLOperand* MLGraphBuilder::add(const MLOperand* a,
                               const MLOperand* b,
                               ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kConstraintError,
                                      "Input types are inconsistent.");
    return nullptr;
  }
  Vector<int32_t> dims_output;
  if (!BroadcastShape(a->Dimensions(), b->Dimensions(), dims_output)) {
    exception_state.ThrowDOMException(DOMExceptionCode::kConstraintError,
                                      "Input shapes are not incompatible.");
    return nullptr;
  }
  auto* add = MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kAdd);
  add->Inputs().resize(2);
  add->Inputs()[0] = a;
  add->Inputs()[1] = b;
  auto* c = MakeGarbageCollected<MLOperand>(this);
  c->SetType(a->Type());
  c->SetDimensions(std::move(dims_output));
  c->SetOperator(add);
  add->Outputs().resize(1);
  add->Outputs()[0] = c;
  return c;
}

MLOperand* MLGraphBuilder::clamp(const MLOperand* input,
                                 const MLClampOptions* options,
                                 ExceptionState& exception_state) {
  auto* clamp =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kClamp);
  clamp->Inputs().resize(1);
  clamp->Inputs()[0] = input;
  clamp->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(clamp);
  clamp->Outputs().resize(1);
  clamp->Outputs()[0] = output;
  return output;
}

MLOperator* MLGraphBuilder::clamp(const MLClampOptions* options,
                                  ExceptionState& exception_state) {
  auto* clamp =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kClamp);
  clamp->SetOptions(options);
  return clamp;
}

MLOperand* MLGraphBuilder::conv2d(const MLOperand* input,
                                  const MLOperand* filter,
                                  const MLConv2dOptions* options,
                                  ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Implement this on operating systems to access
  // hardware acceleration.
  NOTIMPLEMENTED();
  return MakeGarbageCollected<MLOperand>(this);
}

MLOperand* MLGraphBuilder::gemm(const MLOperand* a,
                                const MLOperand* b,
                                const MLGemmOptions* options,
                                ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Implement this on operating systems to access
  // hardware acceleration.
  NOTIMPLEMENTED();
  return MakeGarbageCollected<MLOperand>(this);
}

MLOperand* MLGraphBuilder::averagePool2d(const MLOperand* input,
                                         const MLPool2dOptions* options,
                                         ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Implement this on operating systems to access
  // hardware acceleration.
  NOTIMPLEMENTED();
  return MakeGarbageCollected<MLOperand>(this);
}

MLOperand* MLGraphBuilder::reshape(const MLOperand* input,
                                   const Vector<int32_t>& new_shape,
                                   ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Implement this on operating systems to access
  // hardware acceleration.
  NOTIMPLEMENTED();
  return MakeGarbageCollected<MLOperand>(this);
}

MLOperand* MLGraphBuilder::softmax(const MLOperand* input,
                                   ExceptionState& exception_state) {
  auto* softmax =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kSoftmax);
  softmax->Inputs().resize(1);
  softmax->Inputs()[0] = input;
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(softmax);
  softmax->Outputs().resize(1);
  softmax->Outputs()[0] = output;
  return output;
}

MLGraph* MLGraphBuilder::build(const MLNamedOperands& named_outputs,
                               ExceptionState& exception_state) {
  std::vector<const MLOperand*> inputs;
  std::vector<const MLOperand*> constants;
  std::vector<const MLOperator*> sorted_operators;

  std::stack<const MLOperator*> nodes_to_do;
  std::unordered_set<const MLOperator*> nodes_done;
  for (auto& output : named_outputs) {
    nodes_to_do.push(output.second->Operator());
  }
  while (nodes_to_do.size() > 0) {
    auto* node = nodes_to_do.top();
    if (nodes_done.count(node) == 0) {
      bool can_add = true;
      for (auto& dep : node->Inputs()) {
        if (dep->Operator()) {
          if (nodes_done.count(dep->Operator()) == 0) {
            can_add = false;
            nodes_to_do.push(dep->Operator());
          }
        } else {
          if (dep->Kind() == MLOperand::KindEnum::kInput) {
            inputs.push_back(dep);
          } else {
            DCHECK(dep->Kind() == MLOperand::KindEnum::kConstant);
            constants.push_back(dep);
          }
        }
      }
      if (can_add) {
        sorted_operators.push_back(node);
        nodes_to_do.pop();
        nodes_done.insert(node);
      }
    } else {
      nodes_to_do.pop();
    }
  }
  auto* graph = MakeGarbageCollected<MLGraphXnnpack>(ml_context_);
  if (!graph->Build(named_outputs, inputs, constants, sorted_operators,
                    exception_state)) {
    return nullptr;
  }
  return graph;
}

}  // namespace blink
