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
  void AddConvTranspose2d(const MLOperator* conv2d);

  // Element-wise unary operations:
  // - sin
  // - cos
  // - tan
  // - erf
  // - exp
  // - sqrt
  // - reciprocal
  // - logicalNot
  void AddElementWiseUnary(const MLOperator* ml_operator);

  // Elementwise binary operations:
  // - add
  // - sub
  // - mul
  // - div
  // - pow
  // - equal
  // - greater
  // - lesser
  void AddElementWiseBinary(const MLOperator* binary);

  // Dot product operators:
  // - gemm
  // - matMul
  void AddGemm(const MLOperator* gemm);

  // Pooling operations
  void AddPool2d(const MLOperator* pool2d);

  void AddRelu(const MLOperator* relu);

  // Reshaping operators (do not change the data, just interpretation):
  // - reshape
  // - squeeze
  // - unsqueeze
  // - flattenTo2d
  // - identity (included here because it's a no-op)
  void AddReshape(const MLOperator* reshape);

  void AddSoftmax(const MLOperator* softmax);

  void AddArgMinMax(const MLOperator* ml_operator);
  void AddCast(const MLOperator* ml_operator);
  void AddConcat(const MLOperator* ml_operator);
  void AddSlice(const MLOperator* ml_operator);
  void AddSplit(const MLOperator* ml_operator);
  void AddExpand(const MLOperator* ml_operator);
  void AddGather(const MLOperator* ml_operator);
  void AddInstanceNormalization(const MLOperator* ml_operator);
  void AddMeanVarianceNormalization(const MLOperator* ml_operator);
  void AddPad(const MLOperator* ml_operator);
  void AddFillSequence(const MLOperator* ml_operator);

  // Reduction operators:
  // - reduceL1
  // - reduceL2
  // - reduceLogSum
  // - reduceLogSumExp
  // - reduceMax
  // - reduceMean
  // - reduceMin
  // - reduceProduct
  // - reduceSum
  // - reduceSumSquare
  void AddReduce(const MLOperator* ml_operator);

  void AddResample2d(const MLOperator* ml_operator);
  void AddTranspose(const MLOperator* ml_operator);
  void AddTriangularMatrix(const MLOperator* ml_operator);
  void AddElementWiseIf(const MLOperator* ml_operator);

  void FillConstantsWithArrayBuffer();

  ModelInfoPtr GetModelInfo();

 private:
  // Add a operand to model which is output of the operation.
  size_t AddOperandToModel(const MLOperand* output);

  bool AreOperandsInIndexMap(const Member<const MLOperand>* operands, size_t operand_count) const;

  uint64_t GetOperandIndex(/*nullable*/ const MLOperand* operand) const;

  // Hold all operands of model to index the operand.
  HeapHashMap<Member<const MLOperand>, size_t> operand_index_map_;
  // All constant data will share a big shared memory, so hold the index of
  // constant temporarily.
  HeapHashMap<Member<const MLOperand>, size_t> constant_index_map_;
  ModelInfoPtr model_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_
