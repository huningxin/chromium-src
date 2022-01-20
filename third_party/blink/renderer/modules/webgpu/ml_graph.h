// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_

#include "third_party/blink/renderer/modules/webgpu/dawn_object.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"

namespace blink {

class GPUDevice;
class MLBufferResourceView;

typedef HeapVector<std::pair<String, Member<MLBufferResourceView>>> MLNamedResources;

class MLGraph : public DawnObject<WGPUGraph> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLGraph(GPUDevice* device, WGPUGraph operand);

  MLGraph(const MLGraph&) = delete;
  MLGraph& operator=(const MLGraph&) = delete;

  void Trace(Visitor* visitor) const override;

  // ml_graph.idl
  void compute(const MLNamedResources& inputs, const MLNamedResources& outputs);

 private:
  WGPUNamedResources CreateAndPopulateDawnNamedResources(const MLNamedResources& resources);

};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_GRAPH_H_
