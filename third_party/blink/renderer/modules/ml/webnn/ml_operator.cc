// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

// static
String MLOperator::OperatorKindToString(MLOperator::OperatorKind kind) {
  static_assert(MLOperator::OperatorKind::kTotal == MLOperator::OperatorKind(50));

  switch (kind) {
    case MLOperator::OperatorKind::kClamp:
      return "clamp";
    case MLOperator::OperatorKind::kConv2d:
      return "conv2d";
    case MLOperator::OperatorKind::kAdd:
      return "add";
    case MLOperator::OperatorKind::kSub:
      return "sub";
    case MLOperator::OperatorKind::kMul:
      return "mul";
    case MLOperator::OperatorKind::kDiv:
      return "div";
    case MLOperator::OperatorKind::kMax:
      return "max";
    case MLOperator::OperatorKind::kMin:
      return "min";
    case MLOperator::OperatorKind::kGemm:
      return "gemm";
    case MLOperator::OperatorKind::kHardSwish:
      return "hardSwish";
    case MLOperator::OperatorKind::kAveragePool2d:
      return "averagePool2d";
    case MLOperator::OperatorKind::kMaxPool2d:
      return "maxPool2d";
    case MLOperator::OperatorKind::kRelu:
      return "relu";
    case MLOperator::OperatorKind::kReshape:
      return "reshape";
    case MLOperator::OperatorKind::kResample2d:
      return "resample2d";
    case MLOperator::OperatorKind::kSoftmax:
      return "softmax";
    case MLOperator::OperatorKind::kSigmoid:
      return "sigmoid";

    ////////////////////////////////////////////////////////////////////////////////
    // NEWOPS:::
    case MLOperator::OperatorKind::kArgMax:
      return "argMax";
    case MLOperator::OperatorKind::kArgMin:
      return "argMin";
    case MLOperator::OperatorKind::kCast:
      return "cast";
    case MLOperator::OperatorKind::kConcat:
      return "concat";
    case MLOperator::OperatorKind::kExpand:
      return "expand";
    case MLOperator::OperatorKind::kCos:
      return "cos";
    case MLOperator::OperatorKind::kEqual:
      return "equal";
    case MLOperator::OperatorKind::kErf:
      return "erf";
    case MLOperator::OperatorKind::kExp:
      return "exp";
    case MLOperator::OperatorKind::kFlattenTo2d:
      return "flattenTo2d";
    case MLOperator::OperatorKind::kGather:
      return "gather";
    case MLOperator::OperatorKind::kGreater:
      return "greater";
    case MLOperator::OperatorKind::kLesser:
      return "lesser";
    case MLOperator::OperatorKind::kIdentity:
      return "identity";
    case MLOperator::OperatorKind::kInstanceNormalization:
      return "instanceNormalization";
    case MLOperator::OperatorKind::kMatmul:
      return "matmul";
    case MLOperator::OperatorKind::kPad:
      return "pad";
    case MLOperator::OperatorKind::kPow:
      return "pow";
    case MLOperator::OperatorKind::kFillSequence:
      return "fillSequence";
    case MLOperator::OperatorKind::kReduceL2:
      return "reduceL2";
    case MLOperator::OperatorKind::kReduceMean:
      return "reduceMean";
    case MLOperator::OperatorKind::kReduceSum:
      return "reduceSum";
    case MLOperator::OperatorKind::kShape:
      return "shape";
    case MLOperator::OperatorKind::kSin:
      return "sin";
    case MLOperator::OperatorKind::kSlice:
      return "slice";
    case MLOperator::OperatorKind::kSqrt:
      return "sqrt";
    case MLOperator::OperatorKind::kTranspose:
      return "transpose";
    case MLOperator::OperatorKind::kTriangularMatrix:
      return "triangularMatrix";
    case MLOperator::OperatorKind::kTan:
      return "tan";
    case MLOperator::OperatorKind::kSqueeze:
      return "squeeze";
    case MLOperator::OperatorKind::kUnsqueeze:
      return "unsqueeze";
    case MLOperator::OperatorKind::kElementWiseIf:
      return "elementwiseIf";
    default:
      return "unknown";
  }
}

MLOperator::MLOperator(MLGraphBuilder* builder,
                       OperatorKind kind,
                       const bindings::DictionaryBase* options)
    : builder_(builder), kind_(kind), options_(options) {}

MLOperator::~MLOperator() = default;

void MLOperator::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(options_);
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
  ScriptWrappable::Trace(visitor);
}

MLOperator::OperatorKind MLOperator::Kind() const {
  return kind_;
}

const bindings::DictionaryBase* MLOperator::Options() const {
  return options_;
}

bool MLOperator::IsConnected() const {
  return is_connected_;
}

const HeapVector<Member<const MLOperand>>& MLOperator::Inputs() const {
  return inputs_;
}

const HeapVector<Member<const MLOperand>>& MLOperator::Outputs() const {
  return outputs_;
}

void MLOperator::Connect(HeapVector<Member<const MLOperand>> inputs,
                         HeapVector<Member<const MLOperand>> outputs) {
  DCHECK(!is_connected_);
  DCHECK(!inputs.empty());
  DCHECK(!outputs.empty());
  inputs_ = std::move(inputs);
  outputs_ = std::move(outputs);
  is_connected_ = true;
}

}  // namespace blink
