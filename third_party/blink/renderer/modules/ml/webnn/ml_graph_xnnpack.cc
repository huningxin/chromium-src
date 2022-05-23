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
  for (auto* op : sorted_operators) {
    if (op->Kind() == MLOperator::OpKind::kClamp) {
      const MLClampOptions* options =
          static_cast<const MLClampOptions*>(op->Options());
      String max, min;
      if (options->hasMaxValue()) {
        max = "maxValue: " + String::Number(options->maxValue());
      }
      if (options->hasMinValue()) {
        min = "minValue: " + String::Number(options->minValue());
      }
      String message;
      message =
          "MLClampOptions = {" + max + (min.IsEmpty() ? "" : ", ") + min + "}";
      exception_state.ThrowDOMException(DOMExceptionCode::kConstraintError,
                                        message);
      return false;
    }
  }
  return true;
}

}  // namespace blink
