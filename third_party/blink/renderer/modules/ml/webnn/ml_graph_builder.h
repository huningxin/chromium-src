// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_

#include "third_party/abseil-cpp/absl/types/optional.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_auto_pad.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_type.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/modules_export.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class ExceptionState;
class MLContext;
class MLClampOptions;
class MLConv2dOptions;
class MLGemmOptions;
class MLGraph;
class MLPool2dOptions;
class MLConcatOptions;
class MLGatherOptions;
class MLResample2dOptions;
class MLArgMinMaxOptions;
class MLSqueezeOptions;
class MLSliceOptions;
class MLTransposeOptions;
class MLPadOptions;
class MLInstanceNormalizationOptions;
class MLFillSequenceOptions;
class MLTriangularMatrixOptions;
class MLReduceOptions;
class MLOperand;
class MLOperandDescriptor;
class ScriptPromiseResolver;
class ScriptPromise;

#if 0 // TODO::: Delete me?
// Additional options 
class MLInternalConcatOptions {
  public:
  static MLInternalConcatOptions* Create() {
    return MakeGarbageCollected<MLInternalConcatOptions>();
  }

  uint32_t axis_{0};
};

class MLInternalGatherOptions : public bindings::DictionaryBase {
  public:
  static MLInternalGatherOptions* Create() {
    return MakeGarbageCollected<MLInternalGatherOptions>();
  }

  uint32_t axis_{0};
};
#endif

typedef HeapVector<std::pair<String, Member<MLOperand>>> MLNamedOperands;

class MODULES_EXPORT MLGraphBuilder : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLGraphBuilder* Create(MLContext* context);

  explicit MLGraphBuilder(MLContext* context);

  MLGraphBuilder(const MLGraphBuilder&) = delete;
  MLGraphBuilder& operator=(const MLGraphBuilder&) = delete;

  ~MLGraphBuilder() override;

  void Trace(Visitor* visitor) const override;

  MLContext* GetContext() const;

  struct PaddingSizes {
    uint32_t begin;
    uint32_t end;
  };

  // Calculate the effective padding based on WebNN auto padding rules.
  //
  // TODO(crbug.com/1273291): Add the link to WebNN spec's algorithm once it is
  // defined, tracked by: https://github.com/webmachinelearning/webnn/issues/326
  static absl::optional<PaddingSizes> CalculatePaddingForAutoPad(
      V8MLAutoPad::Enum auto_pad,
      const uint32_t input_size,
      const uint32_t filter_size,
      const uint32_t stride,
      const uint32_t dilation);

  // ml_graph_builder.idl
  MLOperand* input(String name,
                   const MLOperandDescriptor* desc,
                   ExceptionState& exception_state);
  MLOperand* constant(const MLOperandDescriptor* desc,
                      NotShared<DOMArrayBufferView> buffer_view,
                      ExceptionState& exception_state);

  // The order of operations declaration is the same as spec.
  MLOperand* clamp(const MLOperand* input,
                   const MLClampOptions* options,
                   ExceptionState& exception_state);
  MLOperator* clamp(const MLClampOptions* options,
                    ExceptionState& exception_state);

  MLOperand* conv2d(const MLOperand* input,
                    const MLOperand* filter,
                    const MLConv2dOptions* options,
                    ExceptionState& exception_state);

  // Element-wise binary operations
  MLOperand* add(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* sub(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* mul(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* div(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* max(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* min(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);

  MLOperand* gemm(const MLOperand* a,
                  const MLOperand* b,
                  const MLGemmOptions* options,
                  ExceptionState& exception_state);

  MLOperand* hardSwish(const MLOperand* input, ExceptionState& exception_state);
  MLOperator* hardSwish(ExceptionState& exception_state);

  // Pooling operations
  MLOperand* averagePool2d(const MLOperand* input,
                           const MLPool2dOptions* options,
                           ExceptionState& exception_state);
  MLOperand* maxPool2d(const MLOperand* input,
                       const MLPool2dOptions* options,
                       ExceptionState& exception_state);

  MLOperand* relu(const MLOperand* input, ExceptionState& exception_state);
  MLOperator* relu(ExceptionState& exception_state);

  MLOperand* reshape(const MLOperand* input,
                     const Vector<int32_t>& new_shape,
                     ExceptionState& exception_state);

  MLOperand* resample2d(const MLOperand* input,
                        const MLResample2dOptions* options,
                        ExceptionState& exception_state);

  MLOperand* softmax(const MLOperand* input, ExceptionState& exception_state);

  MLOperand* sigmoid(const MLOperand* input, ExceptionState& exception_state);
  MLOperator* sigmoid(ExceptionState& exception_state);

  ////////////////////////////////////////////////////////////////////////////////
  // NEWOPS:::

  MLOperand* argMax(const MLOperand* input,
                    const MLArgMinMaxOptions* options,
                    ExceptionState& exception_state);
  MLOperand* argMin(const MLOperand* input,
                    const MLArgMinMaxOptions* options,
                    ExceptionState& exception_state);
  MLOperand* cast(const MLOperand* input,
                  V8MLOperandType data_type,
                  ExceptionState& exception_state);
  MLOperand* concat(const HeapVector<Member<MLOperand>>& inputs,
                    const MLConcatOptions* options,
                    ExceptionState& exception_state);
  MLOperand* expand(const MLOperand* input,
                    const Vector<uint32_t>& new_shape,
                    ExceptionState& exception_state);
  MLOperand* cos(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* equal(const MLOperand* a,
                   const MLOperand* b,
                   ExceptionState& exception_state);
  MLOperand* erf(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* exp(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* flattenTo2d(const MLOperand* input,
                         uint32_t axis,
                         ExceptionState& exception_state);
  MLOperand* gather(const MLOperand* input,
                    const MLOperand* indices,
                    const MLGatherOptions* options,
                    ExceptionState& exception_state);
  MLOperand* greater(const MLOperand* a,
                     const MLOperand* b,
                     ExceptionState& exception_state);
  MLOperand* identity(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* instanceNormalization(const MLOperand* input,
                                   const MLInstanceNormalizationOptions* options,
                                   ExceptionState& exception_state);
  MLOperand* lesser(const MLOperand* a,
                    const MLOperand* b,
                    ExceptionState& exception_state);
  MLOperand* matmul(const MLOperand* a,
                    const MLOperand* b,
                    ExceptionState& exception_state);
  MLOperand* pad(const MLOperand* input,
                 const Vector<uint32_t>& beginningPadding,
                 const Vector<uint32_t>& endingPadding,
                 const MLPadOptions* options,
                 ExceptionState& exception_state);
  MLOperand* pow(const MLOperand* a,
                 const MLOperand* b,
                 ExceptionState& exception_state);
  MLOperand* fillSequence(const MLOperand* input,
                          const MLFillSequenceOptions* options,
                          ExceptionState& exception_state);
  MLOperand* reduceL2(const MLOperand* input,
                      const MLReduceOptions* options,
                      ExceptionState& exception_state);
  MLOperand* reduceMean(const MLOperand* input,
                        const MLReduceOptions* options,
                        ExceptionState& exception_state);
  MLOperand* reduceSum(const MLOperand* input,
                       const MLReduceOptions* options,
                       ExceptionState& exception_state);
  MLOperand* shape(const MLOperand* input,
                   ExceptionState& exception_state);
  MLOperand* sin(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* slice(const MLOperand* input,
                   const Vector<int32_t>& starts,
                   const Vector<int32_t>& sizes,
                   const MLSliceOptions* options,
                   ExceptionState& exception_state);
  MLOperand* sqrt(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* tan(const MLOperand* input, ExceptionState& exception_state);
  MLOperand* transpose(const MLOperand* input,
                       const MLTransposeOptions* options,
                       ExceptionState& exception_state);
  MLOperand* triangularMatrix(const MLOperand* input,
                              const MLTriangularMatrixOptions* options,
                              ExceptionState& exception_state);
  MLOperand* squeeze(const MLOperand* input,
                     const MLSqueezeOptions* options,
                     ExceptionState& exception_state);
  MLOperand* unsqueeze(const MLOperand* input,
                       const MLSqueezeOptions* options,
                       ExceptionState& exception_state);

  MLOperand* elementwiseIf(const MLOperand* condition,
                           const MLOperand* true_value,
                           const MLOperand* false_value,
                           ExceptionState& exception_state);

  ScriptPromise build(ScriptState* script_state,
                      const MLNamedOperands& named_outputs,
                      ExceptionState& exception_state);
  static void SortOperators(
      const MLNamedOperands& named_outputs,
      HeapVector<Member<const MLOperand>>& inputs,
      HeapVector<Member<const MLOperand>>& constants,
      HeapVector<Member<const MLOperator>>& sorted_operators);

  // TODO(ningxin.hu@intel.com): Once the web-platform-tests are updated, add
  // MLGraphBuilder.buildSync() into ml_graph_builder.idl for dedicated worker
  // as WebNN spec: https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-buildsync
  MLGraph* buildSync(ScriptState* script_state,
                     const MLNamedOperands& named_outputs,
                     ExceptionState& exception_state);

  // The test cases can override the graph building behavior by implementing
  // this class and setting its instance by SetBackendForTesting().
  class BackendForTesting {
   public:
    virtual void BuildGraphAsyncImpl(MLContext* context,
                                     const MLNamedOperands& named_outputs,
                                     ScriptPromiseResolver* resolver) = 0;

    virtual MLGraph* BuildGraphSyncImpl(ScriptState* script_state,
                                        MLContext* context,
                                        const MLNamedOperands& named_outputs,
                                        ExceptionState& exception_state) = 0;
  };

  static void SetBackendForTesting(BackendForTesting* backend_for_testing);

 private:
  Member<MLContext> ml_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
