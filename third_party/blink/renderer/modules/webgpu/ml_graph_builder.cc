// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_buffer_resource_view.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_type.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_leaky_relu_options.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_device.h"
#include "third_party/blink/renderer/modules/webgpu/gpu_buffer.h"
#include "third_party/blink/renderer/modules/webgpu/ml_context.h"
#include "third_party/blink/renderer/modules/webgpu/ml_graph.h"
#include "third_party/blink/renderer/modules/webgpu/ml_operand.h"
#include "third_party/blink/renderer/modules/webgpu/ml_operator.h"

namespace blink {

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

WGPUAutoPad AsDawnType(const V8MLAutoPad auto_pad) {
  WGPUAutoPad dawn_auto_pad;
  switch(auto_pad.AsEnum()) {
    case V8MLAutoPad::Enum::kExplicit:
      dawn_auto_pad = WGPUAutoPad_Explicit;
      break;
    case V8MLAutoPad::Enum::kSameUpper:
      dawn_auto_pad = WGPUAutoPad_SameUpper;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      dawn_auto_pad = WGPUAutoPad_SameLower;
      break;
  }
  return dawn_auto_pad;
}

WGPUInputOperandLayout AsDawnType(const V8MLInputOperandLayout input_layout) {
  WGPUInputOperandLayout dawn_input_layout;
  switch(input_layout.AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNchw:
      dawn_input_layout = WGPUInputOperandLayout_Nchw;
      break;
    case V8MLInputOperandLayout::Enum::kNhwc:
      dawn_input_layout = WGPUInputOperandLayout_Nhwc;
      break;
  }
  return dawn_input_layout;
}

WGPUFilterOperandLayout AsDawnType(const V8MLConv2dFilterOperandLayout filter_layout) {
  WGPUFilterOperandLayout dawn_filter_layout;
  switch(filter_layout.AsEnum()) {
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      dawn_filter_layout = WGPUFilterOperandLayout_Oihw;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      dawn_filter_layout = WGPUFilterOperandLayout_Hwio;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      dawn_filter_layout = WGPUFilterOperandLayout_Ohwi;
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      dawn_filter_layout = WGPUFilterOperandLayout_Ihwo;
      break;
  }
  return dawn_filter_layout;
}

WGPUConv2dOptions AsDawnType(const MLConv2dOptions* conv2d_options) {
  WGPUConv2dOptions dawn_conv2d_options;
  dawn_conv2d_options.padding = conv2d_options->hasPadding() ? conv2d_options->padding().data() : nullptr;
  dawn_conv2d_options.paddingCount = conv2d_options->hasPadding() ? conv2d_options->padding().size(): 0;
  dawn_conv2d_options.strides = conv2d_options->hasStrides() ? conv2d_options->strides().data() : nullptr;
  dawn_conv2d_options.stridesCount = conv2d_options->hasStrides() ? conv2d_options->strides().size(): 0;
  dawn_conv2d_options.dilations = conv2d_options->hasDilations() ? conv2d_options->dilations().data() : nullptr;
  dawn_conv2d_options.dilationsCount = conv2d_options->hasDilations() ? conv2d_options->dilations().size(): 0;
  dawn_conv2d_options.autoPad = AsDawnType(conv2d_options->autoPad());
  dawn_conv2d_options.groups = conv2d_options->groups();
  dawn_conv2d_options.inputLayout = AsDawnType(conv2d_options->inputLayout());
  dawn_conv2d_options.filterLayout = AsDawnType(conv2d_options->filterLayout());
  dawn_conv2d_options.bias = conv2d_options->hasBias() ? conv2d_options->bias()->GetHandle() : nullptr;
  dawn_conv2d_options.activation = conv2d_options->hasActivation() ? conv2d_options->activation()->GetHandle() : nullptr;
  return dawn_conv2d_options;
}

WGPUGemmOptions AsDawnType(const MLGemmOptions* gemm_options) {
  WGPUGemmOptions dawn_gemm_options;
  dawn_gemm_options.c = gemm_options->hasC() ? gemm_options->c()->GetHandle() : nullptr;
  dawn_gemm_options.alpha = gemm_options->alpha();
  dawn_gemm_options.beta = gemm_options->beta();
  dawn_gemm_options.aTranspose = gemm_options->aTranspose();
  dawn_gemm_options.bTranspose = gemm_options->bTranspose();
  return dawn_gemm_options;
}

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
  MLOperand* input = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_input);
  return input;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc, const MLBufferResourceView* buffer_view) {
  WGPUOperandDescriptor dawn_desc = AsDawnType(desc);
  WGPUBufferResourceView dawn_buffer_view = AsDawnType(buffer_view);
  WGPUOperand dawn_constant = GetProcs().graphBuilderConstant(GetHandle(), &dawn_desc, &dawn_buffer_view);
  MLOperand* constant = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_constant);
  return constant;
}

MLOperand* MLGraphBuilder::add(const MLOperand* a, const MLOperand* b) {
  WGPUOperand dawn_output = GetProcs().graphBuilderAdd(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperand* MLGraphBuilder::clamp(const MLOperand* input, const MLClampOptions* options) {
  WGPUClampOptions dawn_clamp_options;
  dawn_clamp_options.minValue = options->getMinValueOr(std::numeric_limits<float>::lowest());
  dawn_clamp_options.maxValue = options->getMaxValueOr(std::numeric_limits<float>::max());
  WGPUOperand dawn_output = GetProcs().graphBuilderClamp(GetHandle(), input->GetHandle(), &dawn_clamp_options);
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperator* MLGraphBuilder::clamp(const MLClampOptions* options) {
  WGPUClampOptions dawn_clamp_options;
  dawn_clamp_options.minValue = options->getMinValueOr(std::numeric_limits<float>::lowest());
  dawn_clamp_options.maxValue = options->getMaxValueOr(std::numeric_limits<float>::max());
  WGPUFusionOperator dawn_operator = GetProcs().graphBuilderClampOperator(GetHandle(), &dawn_clamp_options);
  MLOperator* ml_operator = MakeGarbageCollected<MLOperator>(GetDevice(), dawn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::conv2d(const MLOperand* input, const MLOperand* filter, const MLConv2dOptions* options) {
  WGPUConv2dOptions dawn_conv2d_options = AsDawnType(options);
  WGPUOperand dawn_output = GetProcs().graphBuilderConv2d(GetHandle(), input->GetHandle(), filter->GetHandle(), &dawn_conv2d_options);
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperand* MLGraphBuilder::gemm(const MLOperand* a, const MLOperand* b, const MLGemmOptions* options) {
  WGPUGemmOptions dawn_gemm_options = AsDawnType(options);
  WGPUOperand dawn_output = GetProcs().graphBuilderGemm(GetHandle(), a->GetHandle(), b->GetHandle(), &dawn_gemm_options);
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperand* MLGraphBuilder::leakyRelu(const MLOperand* input, const MLLeakyReluOptions* options) {
  WGPULeakyReluOptions dawn_leaky_relu_options;
  dawn_leaky_relu_options.alpha = options->alpha();
  WGPUOperand dawn_output = GetProcs().graphBuilderLeakyRelu(GetHandle(), input->GetHandle(), &dawn_leaky_relu_options);
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperator* MLGraphBuilder::leakyRelu(const MLLeakyReluOptions* options) {
  WGPULeakyReluOptions dawn_leaky_relu_options;
  dawn_leaky_relu_options.alpha = options->alpha();
  WGPUFusionOperator dawn_operator = GetProcs().graphBuilderLeakyReluOperator(GetHandle(), &dawn_leaky_relu_options);
  MLOperator* ml_operator = MakeGarbageCollected<MLOperator>(GetDevice(), dawn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::matmul(const MLOperand* a, const MLOperand* b) {
  WGPUOperand dawn_output = GetProcs().graphBuilderMatmul(GetHandle(), a->GetHandle(), b->GetHandle());
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperand* MLGraphBuilder::relu(const MLOperand* input) {
  WGPUOperand dawn_output = GetProcs().graphBuilderRelu(GetHandle(), input->GetHandle());
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperator* MLGraphBuilder::relu() {
  WGPUFusionOperator dawn_operator = GetProcs().graphBuilderReluOperator(GetHandle());
  MLOperator* ml_operator = MakeGarbageCollected<MLOperator>(GetDevice(), dawn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::reshape(const MLOperand* input, const Vector<int32_t>& new_shape) {
  WGPUOperand dawn_output = GetProcs().graphBuilderReshape(GetHandle(), input->GetHandle(), new_shape.data(), static_cast<uint32_t>(new_shape.size()));
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperand* MLGraphBuilder::sigmoid(const MLOperand* input) {
  WGPUOperand dawn_output = GetProcs().graphBuilderSigmoid(GetHandle(), input->GetHandle());
  MLOperand* output = MakeGarbageCollected<MLOperand>(GetDevice(), dawn_output);
  return output;
}

MLOperator* MLGraphBuilder::sigmoid() {
  WGPUFusionOperator dawn_operator = GetProcs().graphBuilderSigmoidOperator(GetHandle());
  MLOperator* ml_operator = MakeGarbageCollected<MLOperator>(GetDevice(), dawn_operator);
  return ml_operator;
}

MLOperand* MLGraphBuilder::softmax(const MLOperand* input) {
  WGPUOperand dawn_output = GetProcs().graphBuilderSoftmax(GetHandle(), input->GetHandle());
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
  MLGraph* graph = MakeGarbageCollected<MLGraph>(GetDevice(), dawn_graph);
  return graph;
}

}  // namespace blink