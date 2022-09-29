// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

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
    // cpuinfo needs to parse /proc/cpuinfo that needs to be done pre sandbox.
    // xnn_deinitialize() will call cpuinfo_deinitialize().
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

}  // namespace

MLGraphXnnpack::MLGraphXnnpack(MLContext* context) : MLGraph(context) {}

MLGraphXnnpack::~MLGraphXnnpack() = default;

void MLGraphXnnpack::BuildAsyncImpl(BuildInfo* build_info,
                                    ScriptPromiseResolver* resolver) {
  // TODO(ningxin.hu@intel.com): Get a dedicated queue when the specification
  // matures.
  scoped_refptr<base::SequencedTaskRunner> task_runner =
      ExecutionContext::From(resolver->GetScriptState())
          ->GetTaskRunner(TaskType::kMiscPlatformAPI);
  worker_pool::PostTask(
      FROM_HERE, {base::MayBlock()},
      CrossThreadBindOnce(
          &BuildOnBackgroundThread, WrapCrossThreadPersistent(this),
          WrapCrossThreadPersistent(build_info),
          WrapCrossThreadPersistent(resolver), std::move(task_runner)));
}

}  // namespace blink
