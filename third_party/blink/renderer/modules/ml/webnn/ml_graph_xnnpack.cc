// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

#include "base/numerics/checked_math.h"
#include "base/synchronization/lock.h"
#include "base/system/sys_info.h"
#include "base/task/sequenced_task_runner.h"
#include "base/task/thread_pool.h"
#include "base/thread_annotations.h"
#include "base/trace_event/typed_macros.h"
#include "build/buildflag.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/scheduler/public/post_cross_thread_task.h"
#include "third_party/blink/renderer/platform/scheduler/public/worker_pool.h"
#include "third_party/blink/renderer/platform/wtf/allocator/partitions.h"
#include "third_party/blink/renderer/platform/wtf/cross_thread_copier_base.h"
#include "third_party/blink/renderer/platform/wtf/cross_thread_copier_std.h"
#include "third_party/blink/renderer/platform/wtf/cross_thread_functional.h"
#include "third_party/blink/renderer/platform/wtf/thread_safe_ref_counted.h"

namespace blink {

namespace {

void* XnnAllocate(void* context, size_t size) {
  return WTF::Partitions::BufferPartition()->Alloc(size, "xnnpack");
}

void* XnnReallocate(void* context, void* pointer, size_t size) {
  return WTF::Partitions::BufferPartition()->TryRealloc(pointer, size,
                                                        "xnnpack");
}

void XnnDeallocate(void* context, void* pointer) {
  WTF::Partitions::BufferPartition()->Free(pointer);
}

void* XnnAlignedAllocate(void* context, size_t alignment, size_t size) {
  return WTF::Partitions::BufferPartition()->AlignedAllocWithFlags(0, alignment,
                                                                   size);
}

void XnnAlignedDeallocate(void* context, void* pointer) {
  WTF::Partitions::BufferFree(pointer);
}

class SharedXnnpackContext : public ThreadSafeRefCounted<SharedXnnpackContext> {
 public:
  static scoped_refptr<SharedXnnpackContext> GetInstance() {
    base::AutoLock auto_lock(SharedXnnpackContextLock());
    if (instance_ == nullptr) {
      return base::MakeRefCounted<SharedXnnpackContext>();
    } else {
      return base::WrapRefCounted(instance_);
    }
  }

  SharedXnnpackContext(const SharedXnnpackContext&) = delete;
  SharedXnnpackContext& operator=(const SharedXnnpackContext&) = delete;

  bool Initialize() {
    base::AutoLock auto_lock(lock_);
    if (initialized_) {
      return true;
    }
#if BUILDFLAG(IS_WIN)
    const struct xnn_allocator partition_allocator = {
        .allocate = XnnAllocate,
        .reallocate = XnnReallocate,
        .deallocate = XnnDeallocate,
        .aligned_allocate = XnnAlignedAllocate,
        .aligned_deallocate = XnnAlignedDeallocate,
    };
    if (xnn_initialize(&partition_allocator) != xnn_status_success) {
      return false;
    }
#endif

    // TODO: consider using base::PostJob in the future
    pthreadpool_ = pthreadpool_create(
        std::max(1, base::SysInfo::NumberOfProcessors() / 2));
    if (pthreadpool_ == nullptr) {
      return false;
    }
    initialized_ = true;
    return true;
  }

  pthreadpool_t Pthreadpool() {
    base::AutoLock auto_lock(lock_);
    return pthreadpool_;
  }

 private:
  friend class ThreadSafeRefCounted<SharedXnnpackContext>;
  template <typename T, typename... Args>
  friend scoped_refptr<T> base::MakeRefCounted(Args&&... args);

  SharedXnnpackContext() : initialized_(false) { instance_ = this; }
  ~SharedXnnpackContext() {
#if BUILDFLAG(IS_WIN)
    xnn_deinitialize();
#endif
    if (pthreadpool_ != nullptr) {
      pthreadpool_destroy(pthreadpool_);
    }
    instance_ = nullptr;
  }

  static base::Lock& SharedXnnpackContextLock() {
    DEFINE_THREAD_SAFE_STATIC_LOCAL(base::Lock, lock, ());
    return lock;
  }
  static SharedXnnpackContext* instance_;

  base::Lock lock_;
  bool initialized_ GUARDED_BY(lock_);
  pthreadpool_t pthreadpool_ GUARDED_BY(lock_);
};

SharedXnnpackContext* SharedXnnpackContext::instance_ = nullptr;

String DataTypeToString(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return "float32";
    case V8MLOperandType::Enum::kFloat16:
      return "float16";
    case V8MLOperandType::Enum::kInt32:
      return "int32";
    case V8MLOperandType::Enum::kUint32:
      return "uint32";
    case V8MLOperandType::Enum::kInt8:
      return "int8";
    case V8MLOperandType::Enum::kUint8:
      return "uint8";
  }
}

String OpKindToString(MLOperator::OpKind kind) {
  switch (kind) {
    case MLOperator::OpKind::kClamp:
      return "clamp";
    case MLOperator::OpKind::kConv2d:
      return "conv2d";
    case MLOperator::OpKind::kAdd:
      return "add";
    case MLOperator::OpKind::kGemm:
      return "gemm";
    case MLOperator::OpKind::kAveragePool2d:
      return "averagePool2d";
    case MLOperator::OpKind::kRelu:
      return "relu";
    case MLOperator::OpKind::kReshape:
      return "reshape";
    case MLOperator::OpKind::kSoftmax:
      return "softmax";
  }
}

String XnnStatusToString(xnn_status status) {
  switch (status) {
    case xnn_status_success:
      return "xnn_status_success";
    case xnn_status_uninitialized:
      return "xnn_status_uninitialized";
    case xnn_status_invalid_parameter:
      return "xnn_status_invalid_parameter";
    case xnn_status_invalid_state:
      return "xnn_status_invalid_state";
    case xnn_status_unsupported_parameter:
      return "xnn_status_unsupported_parameter";
    case xnn_status_unsupported_hardware:
      return "xnn_status_unsupported_hardware";
    case xnn_status_out_of_memory:
      return "xnn_status_out_of_memory";
  }
}

size_t GetBytesPerElement(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return 4;
    case V8MLOperandType::Enum::kFloat16:
      return 2;
    case V8MLOperandType::Enum::kInt32:
      return 4;
    case V8MLOperandType::Enum::kUint32:
      return 4;
    case V8MLOperandType::Enum::kInt8:
      return 1;
    case V8MLOperandType::Enum::kUint8:
      return 1;
  }
}

bool CalculateByteLength(const MLOperand* operand, size_t& byte_length) {
  base::CheckedNumeric<size_t> checked_byte_length = 1;
  for (auto& d : operand->Dimensions()) {
    checked_byte_length *= d;
  }
  checked_byte_length *= GetBytesPerElement(operand->Type());
  if (!checked_byte_length.AssignIfValid(&byte_length)) {
    return false;
  }
  return true;
}

bool CalculatePaddingForAutoPad(V8MLAutoPad::Enum autoPad,
                                size_t input_size,
                                uint32_t filter_size,
                                uint32_t stride,
                                uint32_t& padding_begin,
                                uint32_t& padding_end) {
  base::CheckedNumeric<size_t> output_size =
      base::ClampCeil<size_t>(base::checked_cast<double>(input_size) / stride);
  auto total_padding = (output_size - 1) * stride + filter_size - input_size;
  if (!total_padding.IsValid()) {
    return false;
  }
  base::CheckedNumeric<uint32_t> checked_padding_begin(0),
      checked_padding_end(0);
  switch (autoPad) {
    case V8MLAutoPad::Enum::kSameUpper:
      checked_padding_begin = total_padding / 2;
      checked_padding_end = (total_padding + 1) / 2;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      checked_padding_begin = (total_padding + 1) / 2;
      checked_padding_end = total_padding / 2;
      break;
    default:
      NOTREACHED();
  }
  if (!checked_padding_begin.AssignIfValid(&padding_begin) ||
      !checked_padding_end.AssignIfValid(&padding_end)) {
    return false;
  }
  return true;
}

DOMExceptionCode XnnStatusToDOMExceptionCode(xnn_status status) {
  switch (status) {
    case xnn_status_success:
      // This function should only be called with an error.
      NOTREACHED();
      return DOMExceptionCode::kNoError;
    case xnn_status_uninitialized:
      return DOMExceptionCode::kUnknownError;
    case xnn_status_invalid_parameter:
      return DOMExceptionCode::kDataError;
    case xnn_status_invalid_state:
      return DOMExceptionCode::kInvalidStateError;
    case xnn_status_unsupported_parameter:
    case xnn_status_unsupported_hardware:
      return DOMExceptionCode::kNotSupportedError;
    case xnn_status_out_of_memory:
      return DOMExceptionCode::kQuotaExceededError;
  }
  NOTREACHED();
  return DOMExceptionCode::kUnknownError;
}

}  // namespace

MLGraphXnnpack::BuildRequest::BuildRequest(const MLNamedOperands& named_outputs)
    : outputs_(std::move(named_outputs)) {
  MLGraphBuilder::SortOperators(outputs_, inputs_, constants_,
                                sorted_operators_);
}

void MLGraphXnnpack::BuildRequest::Trace(Visitor* visitor) const {
  visitor->Trace(outputs_);
  visitor->Trace(inputs_);
  visitor->Trace(constants_);
  visitor->Trace(sorted_operators_);
}

MLGraphXnnpack::ComputeRequest::ComputeRequest(
    const MLNamedArrayInputs& inputs,
    const MLNamedArrayOutputs& outputs)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

void MLGraphXnnpack::ComputeRequest::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
}

MLGraphXnnpack::MLGraphXnnpack(MLContext* context)
    : MLGraph(context),
      xnn_runtime_(nullptr),
      xnn_context_(SharedXnnpackContext::GetInstance()) {}

MLGraphXnnpack::~MLGraphXnnpack() {
  if (xnn_runtime_ != nullptr) {
    xnn_delete_runtime(xnn_runtime_);
  }
}

ScriptPromise MLGraphXnnpack::BuildImpl(ScriptState* script_state,
                                        const MLNamedOperands& named_outputs,
                                        ExceptionState& exception_state) {
  auto* request = MakeGarbageCollected<BuildRequest>(std::move(named_outputs));
  // TODO: Get a dedicated queue when the specification matures.
  scoped_refptr<base::SequencedTaskRunner> task_runner =
      ExecutionContext::From(script_state)
          ->GetTaskRunner(TaskType::kMiscPlatformAPI);
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  worker_pool::PostTask(FROM_HERE, {base::MayBlock()},
                        CrossThreadBindOnce(&BuildOnBackgroundThread,
                                            WrapCrossThreadPersistent(this),
                                            WrapCrossThreadPersistent(request),
                                            WrapCrossThreadPersistent(resolver),
                                            std::move(task_runner)));
  return resolver->Promise();
}

void MLGraphXnnpack::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                   ExceptionState& exception_state) {
  auto* request = MakeGarbageCollected<BuildRequest>(std::move(named_outputs));
  String error_message;
  xnn_status status = CreateRuntime(request, error_message);
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(XnnStatusToDOMExceptionCode(status),
                                      error_message);
  }
}

ScriptPromise MLGraphXnnpack::ComputeImpl(ScriptState* script_state,
                                          const MLNamedArrayInputs& inputs,
                                          const MLNamedArrayOutputs& outputs,
                                          ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  auto* request = MakeGarbageCollected<ComputeRequest>(std::move(inputs),
                                                       std::move(outputs));
  // TODO: Get a dedicated queue when the specification matures.
  scoped_refptr<base::SequencedTaskRunner> task_runner =
      ExecutionContext::From(script_state)
          ->GetTaskRunner(TaskType::kMiscPlatformAPI);
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  worker_pool::PostTask(FROM_HERE, {base::MayBlock()},
                        CrossThreadBindOnce(&ComputeOnBackgroundThread,
                                            WrapCrossThreadPersistent(this),
                                            WrapCrossThreadPersistent(request),
                                            WrapCrossThreadPersistent(resolver),
                                            std::move(task_runner)));
  return resolver->Promise();
}

void MLGraphXnnpack::ComputeSyncImpl(const MLNamedArrayInputs& inputs,
                                     const MLNamedArrayOutputs& outputs,
                                     ExceptionState& exception_state) {
  auto* request = MakeGarbageCollected<ComputeRequest>(std::move(inputs),
                                                       std::move(outputs));
  String error_message;
  xnn_status status = InvokeRuntime(request, error_message);
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(XnnStatusToDOMExceptionCode(status),
                                      error_message);
  }
}

// static
void MLGraphXnnpack::BuildOnBackgroundThread(
    CrossThreadPersistent<MLGraphXnnpack> graph,
    CrossThreadPersistent<BuildRequest> request,
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    scoped_refptr<base::SequencedTaskRunner> resolver_task_runner) {
  DCHECK(!IsMainThread());

  String error_message;
  xnn_status status = graph->CreateRuntime(request, error_message);
  PostCrossThreadTask(
      *resolver_task_runner, FROM_HERE,
      CrossThreadBindOnce(&MLGraphXnnpack::DidBuild, std::move(graph),
                          std::move(resolver), status, error_message));
}

void MLGraphXnnpack::DidBuild(
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    xnn_status status,
    const String& error_message) {
  if (status != xnn_status_success) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        XnnStatusToDOMExceptionCode(status), error_message));
    return;
  }
  resolver->Resolve(this);
}

// static
void MLGraphXnnpack::ComputeOnBackgroundThread(
    CrossThreadPersistent<MLGraphXnnpack> graph,
    CrossThreadPersistent<ComputeRequest> request,
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    scoped_refptr<base::SequencedTaskRunner> resolver_task_runner) {
  String error_message;
  xnn_status status = graph->InvokeRuntime(request, error_message);
  PostCrossThreadTask(
      *resolver_task_runner, FROM_HERE,
      CrossThreadBindOnce(&MLGraphXnnpack::DidCompute, std::move(graph),
                          std::move(resolver), status, error_message));
}

void MLGraphXnnpack::DidCompute(
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    xnn_status status,
    const String& error_message) {
  if (status != xnn_status_success) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        XnnStatusToDOMExceptionCode(status), error_message));
    return;
  }
  resolver->Resolve();
}

xnn_status MLGraphXnnpack::CreateRuntime(BuildRequest* request,
                                         String& error_message) {
  TRACE_EVENT("blink", "MLGraphXnnpack::CreateRuntime");
  if (!xnn_context_->Initialize()) {
    error_message = "Failed to initialize XNNPACK context.";
    return xnn_status_uninitialized;
  }
  uint32_t externals_size;
  if (!base::CheckAdd<wtf_size_t>(request->outputs_.size(),
                                  request->inputs_.size())
           .AssignIfValid(&externals_size)) {
    error_message = "Overflow occurred when calcuating externals size.";
    return xnn_status_invalid_parameter;
  }
  xnn_subgraph_t subgraph_ptr = nullptr;
  xnn_status status;
  if ((status = xnn_create_subgraph(externals_size, 0, &subgraph_ptr)) !=
      xnn_status_success) {
    error_message =
        "Failed to create XNNPACK subgraph: " + XnnStatusToString(status);
    return status;
  }

  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  HashMap<Member<const MLOperand>, uint32_t> tensors_map;
  uint32_t external_id = 0;
  for (const auto& input : request->inputs_) {
    uint32_t input_id = external_id++;
    tensors_map.insert(input, input_id);
    if ((status = DefineTensor(subgraph.get(), tensors_map, input,
                               error_message, true)) != xnn_status_success) {
      return status;
    }
    TensorValueInfo info = {0};
    info.id = input_id;
    if (!CalculateByteLength(input, info.byte_length)) {
      error_message =
          "Overflow occurred when calcuating byte length of input: " +
          input->Name();
      return xnn_status_invalid_parameter;
    }
    inputs_info_.insert(input->Name(), std::move(info));
  }
  for (const auto& named_output : request->outputs_) {
    auto* output = named_output.second.Get();
    uint32_t output_id = external_id++;
    tensors_map.insert(output, output_id);
    if ((status = DefineTensor(subgraph.get(), tensors_map, output,
                               error_message, true)) != xnn_status_success) {
      return status;
    }
    TensorValueInfo info = {0};
    info.id = output_id;
    if (!CalculateByteLength(output, info.byte_length)) {
      error_message =
          "Overflow occurred when calcuating byte length of output: " +
          named_output.first;
      return xnn_status_invalid_parameter;
    }
    outputs_info_.insert(named_output.first, std::move(info));
  }
  for (const auto& constant : request->constants_) {
    if ((status = DefineTensor(subgraph.get(), tensors_map, constant,
                               error_message)) != xnn_status_success) {
      return status;
    }
  }
  for (const auto& op : request->sorted_operators_) {
    switch (op->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* options =
            static_cast<const MLClampOptions*>(op->Options());
        if ((status = DefineClamp(subgraph.get(), tensors_map, op, options,
                                  error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kConv2d: {
        const MLConv2dOptions* options =
            static_cast<const MLConv2dOptions*>(op->Options());
        if ((status = DefineConv2d(subgraph.get(), tensors_map, op, options,
                                   error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kAdd: {
        if ((status = DefineBinary(subgraph.get(), tensors_map, op,
                                   error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kGemm: {
        const MLGemmOptions* options =
            static_cast<const MLGemmOptions*>(op->Options());
        if ((status = DefineGemm(subgraph.get(), tensors_map, op, options,
                                 error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kAveragePool2d: {
        const MLPool2dOptions* options =
            static_cast<const MLPool2dOptions*>(op->Options());
        if ((status = DefinePool2d(subgraph.get(), tensors_map, op, options,
                                   error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kRelu: {
        if ((status = DefineUnary(subgraph.get(), tensors_map, op,
                                  error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kReshape: {
        if ((status = DefineReshape(subgraph.get(), tensors_map, op,
                                    error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      case MLOperator::OpKind::kSoftmax: {
        if ((status = DefineUnary(subgraph.get(), tensors_map, op,
                                  error_message)) != xnn_status_success) {
          return status;
        }
        break;
      }
      default:
        error_message = "The operator (" + OpKindToString(op->Kind()) +
                        ") is not supported";
        return xnn_status_unsupported_parameter;
    }
  }
  uint32_t flags = XNN_FLAG_YIELD_WORKERS;
  if ((status =
           xnn_create_runtime_v2(subgraph.get(), xnn_context_->Pthreadpool(),
                                 flags, &xnn_runtime_)) != xnn_status_success) {
    error_message =
        "Failed to create XNNPACK runtime: " + XnnStatusToString(status);
    return status;
  }

  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineTensor(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperand* operand,
    String& error_message,
    bool is_external) {
  xnn_datatype data_type = xnn_datatype_invalid;
  if (operand->Type() == V8MLOperandType::Enum::kFloat32) {
    data_type = xnn_datatype_fp32;
  } else {
    error_message = "The data type (" + DataTypeToString(operand->Type()) +
                    ") is not supported";
    return xnn_status_unsupported_parameter;
  }
  Vector<size_t> dims;
  for (auto& d : operand->Dimensions()) {
    if (d < 0) {
      error_message = "The negative dimension is not supported";
      return xnn_status_unsupported_parameter;
    }
    dims.push_back(base::checked_cast<size_t>(d));
  }
  uint32_t flags = 0;
  uint32_t external_id = XNN_INVALID_VALUE_ID;
  std::unique_ptr<uint8_t[], OnFree> data;
  if (is_external) {
    DCHECK(tensors_map.find(operand) != tensors_map.end());
    external_id = tensors_map.at(operand);
    if (operand->Kind() == MLOperand::KindEnum::kInput) {
      flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
    } else {
      DCHECK(operand->Kind() == MLOperand::KindEnum::kOutput);
      flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    }
  } else {
    if (operand->Kind() == MLOperand::KindEnum::kConstant) {
      auto* array_buffer_view = operand->ArrayBufferView();
      data.reset(
          reinterpret_cast<uint8_t*>(WTF::Partitions::BufferPartition()->Alloc(
              array_buffer_view->byteLength(), "MLGraphXnnpack")));
      if (data.get() == nullptr) {
        error_message = "Out of memory.";
        return xnn_status_out_of_memory;
      }
      memcpy(data.get(), array_buffer_view->BaseAddressMaybeShared(),
             array_buffer_view->byteLength());
    }
  }
  uint32_t tensor_id;
  xnn_status status;
  if ((status = xnn_define_tensor_value(
           subgraph, data_type, dims.size(), dims.data(), data.get(),
           external_id, flags, &tensor_id)) != xnn_status_success) {
    error_message =
        "Failed to define tensor value: " + XnnStatusToString(status);
    return status;
  }
  if (data) {
    constant_data_.push_back(std::move(data));
  }
  if (!is_external) {
    tensors_map.insert(operand, tensor_id);
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineClamp(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* clamp,
    const MLClampOptions* options,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = clamp->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = clamp->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = options->hasMinValue()
                               ? options->minValue()
                               : -std::numeric_limits<float>::infinity();
  const float output_max = options->hasMaxValue()
                               ? options->maxValue()
                               : +std::numeric_limits<float>::infinity();
  if ((status = xnn_define_clamp(subgraph, output_min, output_max, input_id,
                                 output_id, 0)) != xnn_status_success) {
    error_message = "Failed to define clamp: " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineConv2d(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* conv2d,
    const MLConv2dOptions* options,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = conv2d->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* filter = conv2d->Inputs()[1].Get();
  DCHECK(tensors_map.find(filter) != tensors_map.end());
  uint32_t filter_id = tensors_map.at(filter);
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (conv2d->Inputs().size() == 3) {
    auto* bias = conv2d->Inputs()[2].Get();
    DCHECK(tensors_map.find(bias) != tensors_map.end());
    bias_id = tensors_map.at(bias);
  }
  auto* output = conv2d->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);

  uint32_t groups = base::checked_cast<uint32_t>(options->groups());
  uint32_t stride_height =
      options->hasStrides()
          ? base::checked_cast<uint32_t>(options->strides()[0])
          : 1;
  uint32_t stride_width =
      options->hasStrides()
          ? base::checked_cast<uint32_t>(options->strides()[1])
          : 1;
  uint32_t dilation_height =
      options->hasDilations()
          ? base::checked_cast<uint32_t>(options->dilations()[0])
          : 1;
  uint32_t dilation_width =
      options->hasDilations()
          ? base::checked_cast<uint32_t>(options->dilations()[1])
          : 1;
  size_t input_height, input_width;
  uint32_t filter_height, filter_width;
  size_t input_channels, output_channels;
  bool depthwise = false;
  if (options->inputLayout().AsEnum() == V8MLInputOperandLayout::Enum::kNhwc) {
    input_height = base::checked_cast<size_t>(input->Dimensions()[1]);
    input_width = base::checked_cast<size_t>(input->Dimensions()[2]);
    input_channels = base::checked_cast<size_t>(input->Dimensions()[3]);
    depthwise = (groups == input_channels);
    if (!depthwise) {
      // For regular conv2d, xnn pack expects weights layed out like (ohwi):
      //   [groups * group_output_channels, kernel_height, kernel_width,
      //   group_input_channels]
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kOhwi) {
        error_message = "The filter layout of conv2d is not supported.";
        return xnn_status_unsupported_parameter;
      }
    } else {
      // For depthwise conv2d, xnn pack expects weights layed out like (ihwo):
      //   [1, kernel_height, kernel_width, input_channels * depth_multiplier]
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kIhwo) {
        error_message =
            "The filter layout of depthwise conv2d is not supported.";
        return xnn_status_unsupported_parameter;
      }
    }
    filter_height = base::checked_cast<uint32_t>(filter->Dimensions()[1]);
    filter_width = base::checked_cast<uint32_t>(filter->Dimensions()[2]);
    output_channels = base::checked_cast<size_t>(output->Dimensions()[3]);
  } else {
    error_message = "The input layout of conv2d is not supported.";
    return xnn_status_unsupported_parameter;
  }
  size_t group_input_channels = input_channels / groups;
  size_t group_output_channels = output_channels / groups;

  uint32_t pad_top, pad_bottom, pad_left, pad_right;
  if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kExplicit) {
    // WebNN padding: [beginning_height, ending_height, beginning_width,
    // ending_width]
    pad_top = options->hasPadding()
                  ? base::checked_cast<uint32_t>(options->padding()[0])
                  : 0;
    pad_bottom = options->hasPadding()
                     ? base::checked_cast<uint32_t>(options->padding()[1])
                     : 0;
    pad_left = options->hasPadding()
                   ? base::checked_cast<uint32_t>(options->padding()[2])
                   : 0;
    pad_right = options->hasPadding()
                    ? base::checked_cast<uint32_t>(options->padding()[3])
                    : 0;
  } else {
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_height,
                                    filter_height, stride_height, pad_top,
                                    pad_bottom) ||
        !CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_width,
                                    filter_width, stride_width, pad_left,
                                    pad_right)) {
      error_message = "Overflow occurred when calcuating padding for autoPad.";
      return xnn_status_invalid_parameter;
    }
  }

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = +std::numeric_limits<float>::infinity();
  if (options->hasActivation()) {
    switch (options->activation()->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* clamp_options =
            static_cast<const MLClampOptions*>(
                options->activation()->Options());
        if (clamp_options->hasMinValue()) {
          output_min = clamp_options->minValue();
        }
        if (clamp_options->hasMaxValue()) {
          output_max = clamp_options->maxValue();
        }
        break;
      }
      case MLOperator::OpKind::kRelu:
        output_min = 0.0f;
        output_max = std::numeric_limits<float>::infinity();
        break;
      default:
        error_message =
            "Only clamp and relu fused operator are supported by conv2d.";
        return xnn_status_unsupported_parameter;
    }
  }
  if (depthwise) {
    status = xnn_define_depthwise_convolution_2d(
        subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
        filter_width, stride_height, stride_width, dilation_height,
        dilation_width, 1, input_channels, output_min, output_max, input_id,
        filter_id, bias_id, output_id, 0);
  } else {
    status = xnn_define_convolution_2d(
        subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
        filter_width, stride_height, stride_width, dilation_height,
        dilation_width, groups, group_input_channels, group_output_channels,
        output_min, output_max, input_id, filter_id, bias_id, output_id, 0);
  }
  if (status != xnn_status_success) {
    error_message = "Failed to define conv2d: " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineBinary(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* binary,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input0 = binary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input0) != tensors_map.end());
  uint32_t input0_id = tensors_map.at(input0);
  auto* input1 = binary->Inputs()[1].Get();
  DCHECK(tensors_map.find(input1) != tensors_map.end());
  uint32_t input1_id = tensors_map.at(input1);
  auto* output = binary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = +std::numeric_limits<float>::infinity();
  switch (binary->Kind()) {
    case MLOperator::OpKind::kAdd: {
      status = xnn_define_add2(subgraph, output_min, output_max, input0_id,
                               input1_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    error_message = "Failed to define element-wise binary op " +
                    OpKindToString(binary->Kind()) + ": " +
                    XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineGemm(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* gemm,
    const MLGemmOptions* options,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = gemm->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* filter = gemm->Inputs()[1].Get();
  if (filter->Kind() != MLOperand::KindEnum::kConstant) {
    error_message = "Only constant input b is supported.";
    return xnn_status_unsupported_parameter;
  }
  DCHECK(tensors_map.find(filter) != tensors_map.end());
  uint32_t filter_id = tensors_map.at(filter);
  auto* output = gemm->Outputs()[0].Get();
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (gemm->Inputs().size() == 3) {
    auto* bias = gemm->Inputs()[2].Get();
    if (bias->Dimensions().size() != 1 ||
        bias->Dimensions()[0] != output->Dimensions()[1]) {
      error_message = "The shape of bias is not supported.";
      return xnn_status_unsupported_parameter;
    }
    DCHECK(tensors_map.find(bias) != tensors_map.end());
    bias_id = tensors_map.at(bias);
  }
  if (fabs(options->alpha() - 1.0f) > std::numeric_limits<float>::epsilon()) {
    error_message = "The alpha is not supported.";
    return xnn_status_unsupported_parameter;
  }
  if (fabs(options->beta() - 1.0f) > std::numeric_limits<float>::epsilon()) {
    error_message = "The beta is not supported.";
    return xnn_status_unsupported_parameter;
  }
  if (options->aTranspose()) {
    error_message = "The aTranspose is not supported.";
    return xnn_status_unsupported_parameter;
  }
  uint32_t flags = 0;
  if (!options->bTranspose()) {
    flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
  }
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = +std::numeric_limits<float>::infinity();
  if ((status = xnn_define_fully_connected(
           subgraph, output_min, output_max, input_id, filter_id, bias_id,
           output_id, flags)) != xnn_status_success) {
    error_message = "Failed to define gemm: " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefinePool2d(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* pool2d,
    const MLPool2dOptions* options,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = pool2d->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = pool2d->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);

  uint32_t stride_height =
      options->hasStrides()
          ? base::checked_cast<uint32_t>(options->strides()[0])
          : 1;
  uint32_t stride_width =
      options->hasStrides()
          ? base::checked_cast<uint32_t>(options->strides()[1])
          : 1;
  uint32_t dilation_height =
      options->hasDilations()
          ? base::checked_cast<uint32_t>(options->dilations()[0])
          : 1;
  uint32_t dilation_width =
      options->hasDilations()
          ? base::checked_cast<uint32_t>(options->dilations()[1])
          : 1;
  size_t input_height, input_width;
  uint32_t filter_height, filter_width;
  bool global_pooling = false;
  if (options->layout().AsEnum() == V8MLInputOperandLayout::Enum::kNhwc) {
    input_height = base::checked_cast<size_t>(input->Dimensions()[1]);
    input_width = base::checked_cast<size_t>(input->Dimensions()[2]);
    if (options->hasWindowDimensions()) {
      filter_height =
          base::checked_cast<uint32_t>(options->windowDimensions()[0]);
      filter_width =
          base::checked_cast<uint32_t>(options->windowDimensions()[1]);
    } else {
      global_pooling = true;
    }
  } else {
    error_message = "The input layout of pool2d is not supported.";
    return xnn_status_unsupported_parameter;
  }

  uint32_t pad_top, pad_bottom, pad_left, pad_right;
  if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kExplicit) {
    // WebNN padding: [beginning_height, ending_height, beginning_width,
    // ending_width]
    pad_top = options->hasPadding()
                  ? base::checked_cast<uint32_t>(options->padding()[0])
                  : 0;
    pad_bottom = options->hasPadding()
                     ? base::checked_cast<uint32_t>(options->padding()[1])
                     : 0;
    pad_left = options->hasPadding()
                   ? base::checked_cast<uint32_t>(options->padding()[2])
                   : 0;
    pad_right = options->hasPadding()
                    ? base::checked_cast<uint32_t>(options->padding()[3])
                    : 0;
  } else {
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_height,
                                    filter_height, stride_height, pad_top,
                                    pad_bottom) ||
        !CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_width,
                                    filter_width, stride_width, pad_left,
                                    pad_right)) {
      error_message = "Overflow occurred when calcuating padding for autoPad.";
      return xnn_status_invalid_parameter;
    }
  }

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = +std::numeric_limits<float>::infinity();
  const uint32_t flags = 0;
  switch (pool2d->Kind()) {
    case MLOperator::OpKind::kAveragePool2d: {
      if (dilation_height != 1 || dilation_width != 1) {
        error_message = "The dilation for averagePool2d is not supported.";
        return xnn_status_unsupported_parameter;
      }
      if (global_pooling) {
        status = xnn_define_global_average_pooling_2d(
            subgraph, output_min, output_max, input_id, output_id, flags);
      } else {
        status = xnn_define_average_pooling_2d(
            subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
            filter_width, stride_height, stride_width, output_min, output_max,
            input_id, output_id, flags);
      }
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    error_message = "Failed to define " + OpKindToString(pool2d->Kind()) +
                    ": " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineReshape(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* reshape,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = reshape->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = reshape->Outputs()[0].Get();
  Vector<size_t> new_sizes;
  for (auto& d : output->Dimensions()) {
    new_sizes.push_back(base::checked_cast<size_t>(d));
  }
  if (new_sizes.size() > XNN_MAX_TENSOR_DIMS) {
    error_message = "The rank of new shape (" +
                    String::Number(new_sizes.size()) +
                    ") is greater than XNNPACK max tensor dims (" +
                    String::Number(XNN_MAX_TENSOR_DIMS) + ").";
    return xnn_status_unsupported_parameter;
  }
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  if ((status = xnn_define_static_reshape(subgraph, new_sizes.size(),
                                          new_sizes.data(), input_id, output_id,
                                          0)) != xnn_status_success) {
    error_message = "Failed to define reshape: " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::DefineUnary(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* unary,
    String& error_message) {
  xnn_status status = xnn_status_success;
  auto* input = unary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = unary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if ((status = DefineTensor(subgraph, tensors_map, output, error_message)) !=
        xnn_status_success) {
      return status;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  switch (unary->Kind()) {
    case MLOperator::OpKind::kRelu: {
      status = xnn_define_clamp(subgraph, 0.0f,
                                std::numeric_limits<float>::infinity(),
                                input_id, output_id, 0);
      break;
    }
    case MLOperator::OpKind::kSoftmax: {
      status = xnn_define_softmax(subgraph, input_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    error_message = "Failed to define element-wise unary op " +
                    OpKindToString(unary->Kind()) + ": " +
                    XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

xnn_status MLGraphXnnpack::InvokeRuntime(ComputeRequest* request,
                                         String& error_message) {
  TRACE_EVENT("blink", "MLGraphXnnpack::InvokeRuntime");
  if (request->inputs_.size() != inputs_info_.size()) {
    error_message = "The number of inputs doesn't match graph's expectation.";
    return xnn_status_invalid_parameter;
  }
  if (request->outputs_.size() != outputs_info_.size()) {
    error_message = "The number of outputs doesn't match graph's expectation.";
    return xnn_status_invalid_parameter;
  }
  Vector<xnn_external_value> external_values;
  external_values.ReserveCapacity(inputs_info_.size() + outputs_info_.size());
  for (const auto& input : request->inputs_) {
    auto iter = inputs_info_.find(input.first);
    if (iter == inputs_info_.end()) {
      error_message = "There is unknown input: " + input.first;
      return xnn_status_invalid_parameter;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    DOMArrayBufferView* array_buffer_view = nullptr;
    if (input.second->IsArrayBufferViewAllowShared()) {
      array_buffer_view = input.second->GetAsArrayBufferViewAllowShared().Get();
    } else if (input.second->IsMLTensor()) {
      auto* ml_tensor = input.second->GetAsMLTensor();
      array_buffer_view = ml_tensor->data().Get();
    }
    DCHECK(array_buffer_view != nullptr);
    if (array_buffer_view->byteLength() < iter->value.byte_length) {
      error_message = "Wrong size of input: " + input.first;
      return xnn_status_invalid_parameter;
    }
    value.data = array_buffer_view->BaseAddressMaybeShared();
    external_values.push_back(value);
  }
  for (const auto& output : request->outputs_) {
    auto iter = outputs_info_.find(output.first);
    if (iter == outputs_info_.end()) {
      error_message = "There is unknown output: " + output.first;
      return xnn_status_invalid_parameter;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    if (output.second->IsArrayBufferViewAllowShared()) {
      DOMArrayBufferView* array_buffer_view =
          output.second->GetAsArrayBufferViewAllowShared().Get();
      if (array_buffer_view->byteLength() < iter->value.byte_length) {
        error_message = "Wrong size of output: " + output.first;
        return xnn_status_invalid_parameter;
      }
      value.data = array_buffer_view->BaseAddressMaybeShared();
    } else if (output.second->IsArrayBufferAllowShared()) {
      DOMArrayBufferBase* array_buffer =
          output.second->GetAsArrayBufferAllowShared();
      if (array_buffer->ByteLength() < iter->value.byte_length) {
        error_message = "Wrong size of output: " + output.first;
        return xnn_status_invalid_parameter;
      }
      value.data = array_buffer->DataMaybeShared();
    }
    DCHECK(value.data);
    external_values.push_back(value);
  }
  xnn_status status = xnn_status_success;
  if ((status = xnn_setup_runtime(xnn_runtime_, external_values.size(),
                                  external_values.data())) !=
      xnn_status_success) {
    error_message =
        "Failed to setup XNNPACK runtime: " + XnnStatusToString(status);
    return status;
  }
  if ((status = xnn_invoke_runtime(xnn_runtime_)) != xnn_status_success) {
    error_message =
        "Failed to invoke XNNPACK runtime: " + XnnStatusToString(status);
    return status;
  }
  return xnn_status_success;
}

}  // namespace blink
