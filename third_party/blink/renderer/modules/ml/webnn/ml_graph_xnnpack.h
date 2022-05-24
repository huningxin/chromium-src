// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/xnnpack/src/include/xnnpack.h"

#include <unordered_map>

namespace blink {

class MLGraphXnnpack : public MLGraph {
 public:
  explicit MLGraphXnnpack(MLContext* context);

  bool Build(const MLNamedOperands& named_outputs,
             const std::vector<const MLOperand*>& inputs,
             const std::vector<const MLOperator*>& sorted_operators,
             ExceptionState& exception_state) override;

 private:
  bool DefineTensor(xnn_subgraph_t,
                    const std::unordered_map<const MLOperand*, uint32_t>&,
                    const MLOperand*,
										uint32_t,
                    ExceptionState&);
  bool DefineClamp(xnn_subgraph_t,
                   const std::unordered_map<const MLOperand*, uint32_t>&,
                   const MLOperator*,
                   const MLClampOptions*,
                   ExceptionState&);
  bool DefineConv2d(xnn_subgraph_t,
                    const std::unordered_map<const MLOperand*, uint32_t>&,
                    const MLOperator*,
                    const MLConv2dOptions*,
                    ExceptionState&);
  bool DefineBinary(xnn_subgraph_t,
                    const std::unordered_map<const MLOperand*, uint32_t>&,
                    const MLOperator*,
                    ExceptionState&);
  bool DefineGemm(xnn_subgraph_t,
                  const std::unordered_map<const MLOperand*, uint32_t>&,
                  const MLOperator*,
                  const MLGemmOptions*,
                  ExceptionState&);
  bool DefineAveragePool2d(
      xnn_subgraph_t,
      const std::unordered_map<const MLOperand*, uint32_t>&,
      const MLOperator*,
      const MLPool2dOptions*,
      ExceptionState&);
  bool DefineReshape(xnn_subgraph_t,
                     const std::unordered_map<const MLOperand*, uint32_t>&,
                     const MLOperator*,
                     ExceptionState&);
  bool DefineUnary(xnn_subgraph_t,
                   const std::unordered_map<const MLOperand*, uint32_t>&,
                   const MLOperator*,
                   ExceptionState&);

  xnn_runtime_t runtime_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_