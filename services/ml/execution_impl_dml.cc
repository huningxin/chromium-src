// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "services/ml/execution_impl_dml.h"

#include <utility>

#include "services/ml/common.h"
#include "services/ml/dml_d3dx12_utils.h"
#include "services/ml/public/mojom/constants.mojom.h"

namespace ml {

std::map<uint32_t, ExecutionImplDML*> ExecutionImplDML::instances_;

ExecutionImplDML::ExecutionImplDML(scoped_refptr<CompiledModelDML> dml,
                                   mojom::ExecutionInitParamsPtr params,
                                   uint32_t id)
    : params_(std::move(params)), dml_(dml) {
  ExecutionImplDML::instances_[id] = this;
  dml->CreateFormatData();
  input_resources_.resize(params_->inputs.size());
  output_resources_.resize(params_->outputs.size());
}

ExecutionImplDML::~ExecutionImplDML() = default;

ExecutionImplDML* ExecutionImplDML::getInstance(uint32_t id) {
  DCHECK(ExecutionImplDML::instances_[id]);
  return ExecutionImplDML::instances_[id];
}

void ExecutionImplDML::setInputD3D12Resource(
    const ComPtr<ID3D12Resource>& resource, uint32_t index) {
  input_resources_[index] = resource;
}

void ExecutionImplDML::setOutputD3D12Resource(
    const ComPtr<ID3D12Resource>& resource, uint32_t index) {
  output_resources_[index] = resource;
}

bool ExecutionImplDML::encodeToCommandList(ID3D12GraphicsCommandList* command_list) {
  uint32_t memory_offset = 0;
  HRESULT hr = S_OK;
  if (command_list == nullptr) {
    hr = UploadData(memory_offset);
    if (FAILED(hr)) {
      return false;
    }
  } else {
    for (size_t i = 0; i < params_->inputs.size(); ++i) {
      size_t input_index = params_->inputs[i]->index;
      ComPtr<ID3D12Resource> webgpu_buffer = input_resources_[i];
      ComPtr<ID3D12Resource> input_resource =
          dml_->operand_map_[input_index]->format_resource_;
      CD3DX12_RESOURCE_BARRIER resource_barrier =
          CD3DX12_RESOURCE_BARRIER::Transition(
              webgpu_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list->ResourceBarrier(1, &resource_barrier);
      command_list->CopyResource(input_resource.Get(),
                                 webgpu_buffer.Get());
      resource_barrier =
      CD3DX12_RESOURCE_BARRIER::Transition(
          input_resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      command_list->ResourceBarrier(1, &resource_barrier);
    }
  }

  // Format input data from NHWC to NCHW and float16 precision.
  dml_->FormatInputData(command_list);

  // Bind and execute the operators on the GPU.
  hr = ExecuteOperators(command_list);
  if (FAILED(hr)) {
    return false;
  }

  // Format input data from NCHW to NHWC and float32 precision.
  dml_->FormatOutputData(command_list);
  
  if (command_list == nullptr) {
    hr = ReadResultBack(memory_offset);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed reading result.";
      return false;
    }
  } else {
    for (size_t i = 0; i < params_->outputs.size(); ++i) {
      size_t output_index = params_->outputs[i]->index;
      ComPtr<ID3D12Resource> output_resource =
          dml_->operand_map_[output_index]->format_resource_;
      ComPtr<ID3D12Resource> webgpu_buffer = output_resources_[i];
      CD3DX12_RESOURCE_BARRIER resource_barrier =
          CD3DX12_RESOURCE_BARRIER::Transition(
              output_resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
              D3D12_RESOURCE_STATE_COPY_SOURCE);
      command_list->ResourceBarrier(1, &resource_barrier);
      command_list->CopyResource(webgpu_buffer.Get(),
                                 output_resource.Get());
      resource_barrier =
      CD3DX12_RESOURCE_BARRIER::Transition(
          webgpu_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      command_list->ResourceBarrier(1, &resource_barrier);
    }
  }
  return true;
}

void ExecutionImplDML::StartCompute(StartComputeCallback callback) {
  bool result = encodeToCommandList();
  if (!result) {
    std::move(callback).Run(mojom::OP_FAILED);
  }
  std::move(callback).Run(mojom::NOT_ERROR);
}

HRESULT ExecutionImplDML::UploadData(uint32_t& memory_offset) {
  HRESULT hr = S_OK;
  for (size_t i = 0; i < params_->inputs.size(); ++i) {
    const mojom::OperandInfoPtr& operand = params_->inputs[i];
    uint32_t offset = memory_offset;
    uint32_t length = GetRequiredSize(operand);
    memory_offset += length;
    auto mapping = params_->memory->MapAtOffset(length, offset);
    ComPtr<ID3D12Resource> upload_resource =
        dml_->operand_map_[operand->index]->upload_resource_;
    ComPtr<ID3D12Resource> format_resource =
        dml_->operand_map_[operand->index]->format_resource_;
    hr = UploadTensorResource(static_cast<void*>(mapping.get()), length,
                              upload_resource, format_resource,
                              dml_->command_list_);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed uploading tensor resource for inputs data.";
      return hr;
    }
  }
  return S_OK;
}

HRESULT ExecutionImplDML::ExecuteOperators(ID3D12GraphicsCommandList* cl) {
  HRESULT hr = S_OK;
  ID3D12GraphicsCommandList* command_list = dml_->command_list_.Get();
  if (cl) {
    command_list = cl;
  }

  ID3D12DescriptorHeap* d3D12_descriptor_heaps[] = {
      dml_->descriptor_heap_.Get()};
  command_list->SetDescriptorHeaps(ARRAYSIZE(d3D12_descriptor_heaps),
                                   d3D12_descriptor_heaps);
  for (size_t i = 0; i < dml_->operations_.size(); ++i) {
    hr = ExecuteCompiledOperator(dml_->operations_[i]->compiled_operator_.Get(),
                                 dml_->operations_[i], i, command_list);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed executing operator.";
      return hr;
    }
  }
  return hr;
}

HRESULT ExecutionImplDML::ExecuteCompiledOperator(
    IDMLCompiledOperator* compiled_operator,
    const std::unique_ptr<OperationDML>& operation,
    uint32_t operation_index,
    ID3D12GraphicsCommandList* command_list) {
  // Record execution of the compiled operator.
  dml_->command_recorder_->RecordDispatch(command_list,
                                          compiled_operator,
                                          operation->binding_table_.Get());
  CD3DX12_RESOURCE_BARRIER resource_barrier =
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  command_list->ResourceBarrier(1, &resource_barrier);

  return S_OK;
}

HRESULT ExecutionImplDML::ReadResultBack(uint32_t memory_offset) {
  // The output buffer now contains the result of the identity operator,
  // so read it back if you want the CPU to access it.
  HRESULT hr = S_OK;
  for (size_t i = 0; i < params_->outputs.size(); ++i) {
    size_t output_index = params_->outputs[i]->index;
    ComPtr<ID3D12Resource> output_resource =
        dml_->operand_map_[output_index]->format_resource_;
    ComPtr<ID3D12Resource> readback_buffer =
        dml_->operand_map_[output_index]->readback_resource_;
    CD3DX12_RESOURCE_BARRIER resource_barrier =
        CD3DX12_RESOURCE_BARRIER::Transition(
            output_resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    dml_->command_list_->ResourceBarrier(1, &resource_barrier);

    dml_->command_list_->CopyResource(readback_buffer.Get(),
                                      output_resource.Get());
  }

  // Reset command list only, keep the command in allocator.
  CloseExecuteResetWait(dml_->d3d12_device_, dml_->command_queue_,
                        dml_->command_allocator_, dml_->command_list_);

  for (size_t i = 0; i < params_->outputs.size(); ++i) {
    size_t output_index = params_->outputs[i]->index;
    ComPtr<ID3D12Resource> readback_buffer =
        dml_->operand_map_[output_index]->readback_resource_;
    const mojom::OperandInfoPtr& operand = params_->outputs[i];
    const uint32_t offset = memory_offset;
    const uint32_t output_buffer_size = GetRequiredSize(operand);
    memory_offset += output_buffer_size;
    auto mapping = params_->memory->MapAtOffset(output_buffer_size, offset);
    D3D12_RANGE tensor_buffer_range = {0, output_buffer_size};
    void* output_buffer_data = nullptr;
    hr = readback_buffer->Map(0, &tensor_buffer_range, &output_buffer_data);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed map buffer for reading result.";
      return hr;
    }
    memcpy(mapping.get(), output_buffer_data, output_buffer_size);

    D3D12_RANGE empty_range{0, 0};
    readback_buffer->Unmap(0, &empty_range);
  }

  return S_OK;
}

}  // namespace ml
