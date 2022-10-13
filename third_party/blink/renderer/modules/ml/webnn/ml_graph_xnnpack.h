// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/platform/heap/persistent.h"
#include "third_party/xnnpack/src/include/xnnpack.h"

namespace blink {

class ScriptPromiseResolver;

namespace {
class SharedXnnpackContext;
}

class MLGraphXnnpack final : public MLGraph {
 public:
  // Create and build an MLGraphXnnpack object. Resolve the promise with
  // this concrete object if no errors.
  static void ValidateAndBuildAsync(MLContext* context,
                                    const MLNamedOperands& named_outputs,
                                    ScriptPromiseResolver* resolver);

  // The constructor shouldn't be called directly. The callers should use
  // ValidateAndBuildAsync() method instead.
  explicit MLGraphXnnpack(MLContext* context);

  ~MLGraphXnnpack() override;

  // Get the pthreadpool of SharedXnnpackContext for unit test.
  pthreadpool_t GetPthreadpoolForTesting() const;

 private:
  void BuildAsyncImpl(const MLNamedOperands& named_outputs,
                      ScriptPromiseResolver* resolver) override;

  // Perform the XNNPACK graph build off the main thread.
  static void BuildOnBackgroundThread(
      CrossThreadPersistent<MLGraphXnnpack> graph,
      CrossThreadPersistent<MLNamedOperands> named_outputs,
      CrossThreadPersistent<ScriptPromiseResolver> resolver,
      scoped_refptr<base::SequencedTaskRunner> resolver_task_runner);

  // Resolve the promise on the main thread.
  void OnBuildFinished(CrossThreadPersistent<ScriptPromiseResolver> resolver,
                       xnn_status status,
                       String error_message = String());

  scoped_refptr<SharedXnnpackContext> xnn_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
