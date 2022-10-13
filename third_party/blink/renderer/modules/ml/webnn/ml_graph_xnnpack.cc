// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

#include "base/allocator/partition_allocator/partition_root.h"
#include "base/synchronization/lock.h"
#include "base/system/sys_info.h"
#include "base/task/sequenced_task_runner.h"
#include "base/task/thread_pool.h"
#include "base/thread_annotations.h"
#include "build/buildflag.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/platform/heap/cross_thread_persistent.h"
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
  return WTF::Partitions::BufferPartition()->Alloc(size, "xnnpack_WebNN");
}

void* XnnReallocate(void* context, void* pointer, size_t size) {
  return WTF::Partitions::BufferPartition()->TryRealloc(pointer, size,
                                                        "xnnpack_WebNN");
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
      const struct xnn_allocator partition_allocator = {
          .allocate = XnnAllocate,
          .reallocate = XnnReallocate,
          .deallocate = XnnDeallocate,
          .aligned_allocate = XnnAlignedAllocate,
          .aligned_deallocate = XnnAlignedDeallocate,
      };
      if (xnn_initialize(&partition_allocator) != xnn_status_success) {
        return nullptr;
      }
      // TODO(ningxin.hu@intel.com): consider using base::PostJob in the future
      pthreadpool_t pthreadpool_ptr = pthreadpool_create(
          std::max(1, base::SysInfo::NumberOfProcessors() / 2));
      if (pthreadpool_ptr == nullptr) {
        return nullptr;
      }
      return base::MakeRefCounted<SharedXnnpackContext>(pthreadpool_ptr);
    } else {
      return base::WrapRefCounted(instance_);
    }
  }

  SharedXnnpackContext(const SharedXnnpackContext&) = delete;
  SharedXnnpackContext& operator=(const SharedXnnpackContext&) = delete;

  pthreadpool_t Pthreadpool() { return pthreadpool_.get(); }

 private:
  friend class ThreadSafeRefCounted<SharedXnnpackContext>;
  template <typename T, typename... Args>
  friend scoped_refptr<T> base::MakeRefCounted(Args&&... args);

  explicit SharedXnnpackContext(pthreadpool_t pthreadpool_ptr)
      : pthreadpool_(pthreadpool_ptr, &pthreadpool_destroy) {
    instance_ = this;
  }
  ~SharedXnnpackContext() {
    base::AutoLock auto_lock(SharedXnnpackContextLock());
#if !(BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS))
    // For Linux and ChromeOS, cpuinfo needs to parse /proc/cpuinfo to
    // initialize in pre sandbox stage. Calling xnn_deinitialize() here will
    // deinitialize cpufino within sandbox and cannot access /proc/cpuinfo
    // again.
    // See https://chromium-review.googlesource.com/c/chromium/src/+/3907965 for
    // more details.
    xnn_deinitialize();
#endif
    DCHECK_EQ(this, instance_);
    instance_ = nullptr;
  }

  static base::Lock& SharedXnnpackContextLock() {
    DEFINE_THREAD_SAFE_STATIC_LOCAL(base::Lock, lock, ());
    return lock;
  }
  static SharedXnnpackContext* instance_ GUARDED_BY(SharedXnnpackContextLock());
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> pthreadpool_;
};

SharedXnnpackContext* SharedXnnpackContext::instance_ = nullptr;

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

// static
void MLGraphXnnpack::ValidateAndBuildAsync(MLContext* context,
                                  const MLNamedOperands& named_outputs,
                                  ScriptPromiseResolver* resolver) {
  auto* graph = MakeGarbageCollected<MLGraphXnnpack>(context);
  graph->BuildAsync(named_outputs, resolver);
}

MLGraphXnnpack::MLGraphXnnpack(MLContext* context) : MLGraph(context) {}

MLGraphXnnpack::~MLGraphXnnpack() = default;

pthreadpool_t MLGraphXnnpack::GetPthreadpoolForTesting() const {
  return xnn_context_->Pthreadpool();
}

void MLGraphXnnpack::BuildAsyncImpl(const MLNamedOperands& named_outputs,
                                    ScriptPromiseResolver* resolver) {
  // TODO(crbug.com/1273291): Get a dedicated queue when the specification
  // matures.
  scoped_refptr<base::SequencedTaskRunner> task_runner =
      ExecutionContext::From(resolver->GetScriptState())
          ->GetTaskRunner(TaskType::kMiscPlatformAPI);
  auto* on_heap_named_outputs =
      MakeGarbageCollected<MLNamedOperands>(named_outputs);
  worker_pool::PostTask(
      FROM_HERE, {base::MayBlock()},
      CrossThreadBindOnce(
          &BuildOnBackgroundThread, WrapCrossThreadPersistent(this),
          WrapCrossThreadPersistent(on_heap_named_outputs),
          WrapCrossThreadPersistent(resolver), std::move(task_runner)));
}

// static
void MLGraphXnnpack::BuildOnBackgroundThread(
    CrossThreadPersistent<MLGraphXnnpack> graph,
    CrossThreadPersistent<MLNamedOperands> named_outputs,
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    scoped_refptr<base::SequencedTaskRunner> resolver_task_runner) {
  DCHECK(!IsMainThread());
  DCHECK(!graph->xnn_context_);

  // Get or create the shared XNNPACK context.
  graph->xnn_context_ = SharedXnnpackContext::GetInstance();
  String error_message;
  xnn_status status;
  if (!graph->xnn_context_) {
    error_message = "Failed to get XNNPACK context.";
    status = xnn_status_uninitialized;
  }

  PostCrossThreadTask(*resolver_task_runner, FROM_HERE,
                      CrossThreadBindOnce(&MLGraphXnnpack::OnBuildFinished,
                                          std::move(graph), std::move(resolver),
                                          status, std::move(error_message)));
}

void MLGraphXnnpack::OnBuildFinished(
    CrossThreadPersistent<ScriptPromiseResolver> resolver,
    xnn_status status,
    String error_message) {
  if (status != xnn_status_success) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        XnnStatusToDOMExceptionCode(status), error_message));
    return;
  }
  resolver->Resolve(this);
}

}  // namespace blink
