// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_type.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/ml_context.h"
#include "third_party/blink/renderer/modules/webgpu/ml_graph.h"
#include "third_party/blink/renderer/modules/webgpu/ml_operand.h"

namespace blink {

//static
MLGraphBuilder* MLGraphBuilder::Create(const MLContext* context) {
  GPUDevice* device = context->GetDevice();
  WGPUGraphBuilder dawnBuilder = context->GetProcs().deviceCreateGraphBuilder(device->GetHandle());
  MLGraphBuilder* builder = MakeGarbageCollected<MLGraphBuilder>(device, dawnBuilder);
  return builder;
}

MLGraphBuilder::MLGraphBuilder(GPUDevice* deivce, WGPUGraphBuilder builder)
    : DawnObject<WGPUGraphBuilder>(deivce, builder) {
}

void MLGraphBuilder::Trace(Visitor* visitor) const {
  DawnObject<WGPUGraphBuilder>::Trace(visitor);
}

GPUDevice* MLGraphBuilder::GetDevice() const {
  return device_.Get();
}

MLOperand* MLGraphBuilder::input(String name, const MLOperandDescriptor* desc) {
  WGPUOperandDescriptor dawn_desc = AsDawnType(desc);
  std::string name_str = name.Utf8();
  WGPUOperand dawn_input = GetProcs().graphBuilderInput(GetHandle(), name_str.c_str(), &dawn_desc);
  EnsureFlush();
  MLOperand* input = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_input);
  return input;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc, const MLBufferResourceView* buffer_view) {
  WGPUOperandDescriptor dawn_desc = AsDawnType(desc);
  WGPUBufferResourceView dawn_buffer_view = AsDawnType(buffer_view);
  WGPUOperand dawn_constant = GetProcs().graphBuilderConstant(GetHandle(), &dawn_desc, &dawn_buffer_view);
  EnsureFlush();
  MLOperand* constant = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_constant);
  return constant;
}

MLOperand* MLGraphBuilder::relu(const MLOperand* input) {
  WGPUOperand dawn_output = GetProcs().graphBuilderRelu(GetHandle(), input->GetHandle());
  EnsureFlush();
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLGraph* MLGraphBuilder::build(const MLNamedOperands& outputs) {
  WGPUNamedOperands dawn_outputs = GetProcs().graphBuilderCreateNamedOperands(GetHandle());
  for (wtf_size_t i = 0; i < outputs.size(); ++i) {
      std::string name = outputs[i].first.Utf8();
      WGPUOperand dawn_operand = outputs[i].second->GetHandle();
      GetProcs().namedOperandsSet(dawn_outputs, name.c_str(), dawn_operand);
  }
  WGPUGraph dawn_graph = GetProcs().graphBuilderBuild(GetHandle(), dawn_outputs);
  EnsureFlush();
  MLGraph* graph = MakeGarbageCollected<MLGraph>(GetDevice(), dawn_graph);
  return graph;
}

WGPUOperandDescriptor AsDawnType(const MLOperandDescriptor* desc) {
  WGPUOperandDescriptor dawn_desc;
  switch(desc->type().AsEnum()) {
    case V8MLOperandType::Enum::kFloat32:
      dawn_desc.type = WGPUOperandType_Float32;
      break;
    case V8MLOperandType::Enum::kFloat16:
      dawn_desc.type = WGPUOperandType_Float16;
      break;
    case V8MLOperandType::Enum::kInt32:
      dawn_desc.type = WGPUOperandType_Int32;
      break;
    case V8MLOperandType::Enum::kUint32:
      dawn_desc.type = WGPUOperandType_Uint32;
      break;
    case V8MLOperandType::Enum::kInt8:
      dawn_desc.type = WGPUOperandType_Int8;
      break;
    case V8MLOperandType::Enum::kUint8:
      dawn_desc.type = WGPUOperandType_Uint8;
      break;
  }
  dawn_desc.dimensions = desc->dimensions().data();
  dawn_desc.dimensionsCount = desc->dimensions().size();
  return dawn_desc;
}

WGPUBufferResourceView AsDawnType(const MLBufferResourceView* buffer_view) {
  WGPUBufferResourceView dawn_buffer_view;
  dawn_buffer_view.resource = buffer_view->resource()->GetHandle();
  dawn_buffer_view.offset = buffer_view->offset();
  dawn_buffer_view.size = buffer_view->getSizeOr(0);
  return dawn_buffer_view;
}

}  // namespace blink