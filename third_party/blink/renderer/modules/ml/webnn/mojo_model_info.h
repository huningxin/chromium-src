// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_

#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_map.h"

namespace blink {

#if BUILDFLAG(IS_WIN)
static const uint32_t kBufferAlignment = 16;
#else
static const uint32_t kBufferAlignment = 1;
#endif
base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment);

using ml::webnn::mojom::blink::ModelInfoPtr;

class MojoModelInfo final : public GarbageCollected<MojoModelInfo> {
 public:
  MojoModelInfo();
  ~MojoModelInfo();

  MojoModelInfo(const MojoModelInfo&) = delete;
  MojoModelInfo& operator=(const MojoModelInfo&) = delete;

  void Trace(Visitor* visitor) const;

  void AddInput(const MLOperand* input);
  void AddConstant(const MLOperand* constant);
  void AddOutput(String name, const MLOperand* output);

  // The order of operations declaration is the same as spec.
  void AddClamp(const MLOperator* clamp);

  void AddConv2d(const MLOperator* conv2d);

  // Element-wise operations:
  // - Identity
  // - Sin
  // - Cos
  // - Tan
  // - Erf
  // - Exp
  // - Sqrt
  void AddElementWiseUnary(const MLOperator* ml_operator);

  // Elementwise binary operations:
  // - Add
  // - Sub
  // - Mul
  // - Div
  // - Pow
  void AddElementWiseBinary(const MLOperator* binary);

  // Gemm
  // MatMul
  void AddGemm(const MLOperator* gemm);

  // Pooling operations
  void AddPool2d(const MLOperator* pool2d);

  void AddRelu(const MLOperator* relu);

  void AddReshape(const MLOperator* reshape);

  void AddSoftmax(const MLOperator* softmax);

  ////////////////////////////////////////////////////////////////////////////////
  // NEWOPS:::

  // TODO::: Combine ArgMin and ArgMax?
  void AddArgMax(const MLOperator* ml_operator);
  void AddArgMin(const MLOperator* ml_operator);
  void AddCast(const MLOperator* ml_operator);
  void AddConcat(const MLOperator* ml_operator);
  void AddExpand(const MLOperator* ml_operator);
  void AddFlattenTo2d(const MLOperator* ml_operator);
  void AddGather(const MLOperator* ml_operator);
  void AddInstanceNormalization(const MLOperator* ml_operator);
  void AddPad(const MLOperator* ml_operator);
  void AddFillSequence(const MLOperator* ml_operator);
  void AddReduceL2(const MLOperator* ml_operator);
  void AddReduceMean(const MLOperator* ml_operator);
  void AddReduceSum(const MLOperator* ml_operator);
  void AddResample2d(const MLOperator* ml_operator);
  void AddShape(const MLOperator* ml_operator);
  void AddSlice(const MLOperator* ml_operator);
  void AddTranspose(const MLOperator* ml_operator);
  void AddTriangularMatrix(const MLOperator* ml_operator);
  void AddSqueeze(const MLOperator* ml_operator);
  void AddUnsqueeze(const MLOperator* ml_operator);
  void AddElementWiseIf(const MLOperator* ml_operator);

  void FillConstantsWithArrayBuffer();

  ModelInfoPtr GetModelInfo();

 private:
  // Add a operand to model which is output of the operation.
  size_t AddOperandToModel(const MLOperand* output);

  bool OperandsInIndexMap(const Member<const MLOperand>* operands, size_t operand_count) const;

  // Hold all operands of model to index the operand.
  HeapHashMap<Member<const MLOperand>, size_t> operand_index_map_;
  // All constant data will share a big shared memory, so hold the index of
  // constant temporarily.
  HeapHashMap<Member<const MLOperand>, size_t> constant_index_map_;
  ModelInfoPtr model_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_
