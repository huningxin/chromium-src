// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml_graph.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/ml_graph_builder.h"

namespace blink {

MLGraph::MLGraph(GPUDevice* device, WGPUGraph graph)
    : DawnObject<WGPUGraph>(device, graph) {
}

void MLGraph::Trace(Visitor* visitor) const {
  DawnObject<WGPUGraph>::Trace(visitor);
}

void MLGraph::compute(const MLNamedResources& inputs, const MLNamedResources& outputs) {
    WGPUNamedResources dawn_inputs = CreateAndPopulateDawnNamedResources(inputs);
    WGPUNamedResources dawn_outputs = CreateAndPopulateDawnNamedResources(outputs);
    GetProcs().graphCompute(GetHandle(), dawn_inputs, dawn_outputs);
}

WGPUNamedResources MLGraph::CreateAndPopulateDawnNamedResources(const MLNamedResources& resources) {
  WGPUNamedResources dawn_resources = GetProcs().graphCreateNamedResources(GetHandle());
  for (wtf_size_t i = 0; i < resources.size(); ++i) {
      std::string name = resources[i].first.Utf8();
      WGPUBufferResourceView dawn_buffer_view = AsDawnType(resources[i].second.Get());
      GetProcs().namedResourcesSet(dawn_resources, name.c_str(), &dawn_buffer_view);
  }
  return dawn_resources;
}

}  // namespace blink