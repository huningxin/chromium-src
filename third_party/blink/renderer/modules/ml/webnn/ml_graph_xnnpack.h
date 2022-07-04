// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_

#include "base/allocator/partition_allocator/partition_root.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/platform/wtf/hash_map.h"
#include "third_party/blink/renderer/platform/wtf/text/string_hash.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"
#include "third_party/xnnpack/src/include/xnnpack.h"

namespace blink {

class ScriptPromiseResolver;

namespace {
class SharedXnnpackContext;
}

class MLGraphXnnpack final : public MLGraph {
 public:
  explicit MLGraphXnnpack(MLContext* context);
  ~MLGraphXnnpack() override;

  ScriptPromise BuildImpl(ScriptState* script_state,
                          const MLNamedOperands& named_outputs,
                          ExceptionState& exception_state) override;
  void BuildSyncImpl(const MLNamedOperands& named_outputs,
                     ExceptionState& exception_state) override;

  ScriptPromise ComputeImpl(ScriptState* script_state,
                            const MLNamedArrayInputs& inputs,
                            const MLNamedArrayOutputs& outputs,
                            ExceptionState& exception_state) override;
  void ComputeSyncImpl(const MLNamedArrayInputs& inputs,
                       const MLNamedArrayOutputs& outputs,
                       ExceptionState& exception_state) override;

 private:
  struct BuildRequest final : public GarbageCollected<BuildRequest> {
    BuildRequest(const MLNamedOperands& named_outputs);
    void Trace(Visitor*) const;

    MLNamedOperands outputs_;
    HeapVector<Member<const MLOperand>> inputs_;
    HeapVector<Member<const MLOperand>> constants_;
    HeapVector<Member<const MLOperator>> sorted_operators_;
  };

  struct ComputeRequest final : public GarbageCollected<ComputeRequest> {
    ComputeRequest(const MLNamedArrayInputs& inputs,
                   const MLNamedArrayOutputs& outputs);
    void Trace(Visitor*) const;

    MLNamedArrayInputs inputs_;
    MLNamedArrayOutputs outputs_;
  };

  // Perform the graph build off the main thread.
  static void BuildOnBackgroundThread(
      CrossThreadPersistent<MLGraphXnnpack> graph,
      CrossThreadPersistent<BuildRequest> request,
      CrossThreadPersistent<ScriptPromiseResolver> resolver,
      scoped_refptr<base::SequencedTaskRunner> resolver_task_runner);

  // Perform the post graph build on the main thread.
  void DidBuild(CrossThreadPersistent<ScriptPromiseResolver> resolver,
                xnn_status status,
                const String& error_message = String());

  // Perform the graph compute off the main thread.
  static void ComputeOnBackgroundThread(
      CrossThreadPersistent<MLGraphXnnpack> graph,
      CrossThreadPersistent<ComputeRequest> request,
      CrossThreadPersistent<ScriptPromiseResolver> resolver,
      scoped_refptr<base::SequencedTaskRunner> resolver_task_runner);

  // Perform the post graph compute on the main thread.
  void DidCompute(CrossThreadPersistent<ScriptPromiseResolver> resolver,
                  xnn_status status,
                  const String& error_message = String());

  // Methods for interaction with XNNPACK APIs
  xnn_status CreateRuntime(BuildRequest* request, String& error_message);
  xnn_status InvokeRuntime(ComputeRequest* request, String& error_message);

  xnn_status DefineTensor(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperand* operand,
      String& error_message,
      bool is_external = false);

  xnn_status DefineClamp(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* clamp,
      const MLClampOptions* options,
      String& error_message);

  xnn_status DefineConv2d(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* conv2d,
      const MLConv2dOptions* options,
      String& error_message);

  xnn_status DefineBinary(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* binary,
      String& error_message);

  xnn_status DefineGemm(xnn_subgraph_t subgraph,
                        HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
                        const MLOperator* gemm,
                        const MLGemmOptions* options,
                        String& error_message);

  xnn_status DefinePool2d(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* pool2d,
      const MLPool2dOptions* options,
      String& error_message);

  xnn_status DefineReshape(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* reshape,
      String& error_message);

  xnn_status DefineUnary(
      xnn_subgraph_t subgraph,
      HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
      const MLOperator* unary,
      String& error_message);

  // Members for graph compute, will be perserved in object life time.
  struct OnFree {
    void operator()(void* ptr) const {
      WTF::Partitions::BufferPartition()->Free(ptr);
    }
  };
  Vector<std::unique_ptr<uint8_t[], OnFree>> constant_data_;
  struct TensorValueInfo {
    uint32_t id;
    size_t byte_length;
  };
  HashMap<String, TensorValueInfo> inputs_info_;
  HashMap<String, TensorValueInfo> outputs_info_;
  xnn_runtime_t xnn_runtime_;
  scoped_refptr<SharedXnnpackContext> xnn_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
