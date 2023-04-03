// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Configuration header.
#include "threadpool-common.h"

// Public library header.
#include <pthreadpool.h>

// Internal library headers.
#include "threadpool-atomics.h"
#include "threadpool-object.h"
#include "threadpool-utils.h"

/* Chromium header */
#include "base/functional/bind.h"
#include "base/synchronization/lock.h"
#include "base/system/sys_info.h"
#include "base/task/post_job.h"
#include "base/task/task_traits.h"

class PthreadPoolJob {
 public:
  PthreadPoolJob(struct pthreadpool* threadpool, size_t num_work_items)
      : threadpool_(threadpool),
        num_work_items_(num_work_items),
        num_remaining_work_items_(num_work_items) {}
  ~PthreadPoolJob() = default;

  void Run(base::JobDelegate* delegate) {
    CHECK(delegate);
    while (!delegate->ShouldYield()) {
      size_t index = GetNextIndex();
      if (index >= num_work_items_) {
        return;
      }

      DoWork(index);

      if (CompleteWork()) {
        return;
      }
    }
  }

  size_t GetMaxConcurrency(size_t worker_count) const {
    // `num_remaining_work_items_` includes the units that other workers are
    // currently working on, so we can safely ignore the `worker_count` and just
    // return the current number of outstanding units.
    return num_remaining_work_items_.load(std::memory_order_relaxed);
  }

 private:
  void DoWork(size_t index) {
    struct thread_info* thread = &threadpool_->threads[index];
    const uint32_t flags =
        pthreadpool_load_relaxed_uint32_t(&threadpool_->flags);
    const thread_function_t thread_function =
        (thread_function_t)pthreadpool_load_relaxed_void_p(
            &threadpool_->thread_function);

    struct fpu_state saved_fpu_state = {0};
    if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
      saved_fpu_state = get_fpu_state();
      disable_fpu_denormals();
    }

    thread_function(threadpool_, thread);

    if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
      set_fpu_state(saved_fpu_state);
    }
  }

  // Returns the index of the next work item to process.
  size_t GetNextIndex() {
    // `index_` may exceeed `num_work_items_`, but only by the number of
    // workers at worst, thus it can't exceed 2 * |num_work_items_| and
    // overflow shouldn't happen.
    return index_.fetch_add(1, std::memory_order_relaxed);
  }

  // Returns true if the last work item was completed.
  bool CompleteWork() {
    size_t num_remaining_work_items =
        num_remaining_work_items_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GE(num_remaining_work_items, 1);
    return num_remaining_work_items == 1;
  }

  struct pthreadpool* threadpool_;
  const size_t num_work_items_;
  std::atomic<size_t> index_{0};
  std::atomic<size_t> num_remaining_work_items_;
};

struct pthreadpool* pthreadpool_create(size_t threads_count) {
  // When using Jobs API, the `threads_count` only means the number of work
  // items for each operation. Jobs API doesn't guarantee scheduling the number
  // of threads according to that.
  if (threads_count == 0) {
    // Ideally the number of work items would match the number of worker threads
    // of the thread pool. However, the
    // `base::ThreadPoolInstance::GetMaxConcurrentNonBlockedTasksWithTraitsDeprecated()`
    // is private and will be deprecated.
    //
    // TODO(crbug.com/1228275): Set a more reasonable threads count.
    threads_count = base::SysInfo::NumberOfProcessors();
    if (threads_count == 0) {
      return nullptr;
    }
  }

  struct pthreadpool* threadpool = pthreadpool_allocate(threads_count);
  if (threadpool == nullptr) {
    return nullptr;
  }

  threadpool->threads_count = fxdiv_init_size_t(threads_count);
  for (size_t tid = 0; tid < threads_count; tid++) {
    threadpool->threads[tid].thread_number = tid;
    threadpool->threads[tid].threadpool = threadpool;
  }

  threadpool->execution_mutex = new base::Lock();

  return threadpool;
}

PTHREADPOOL_INTERNAL void pthreadpool_parallelize(
    struct pthreadpool* threadpool,
    thread_function_t thread_function,
    const void* params,
    size_t params_size,
    void* task,
    void* context,
    size_t linear_range,
    uint32_t flags) {
  CHECK(threadpool);
  CHECK(thread_function);
  CHECK(task);
  CHECK_GT(linear_range, 1);

  // Protect the global threadpool structures.
  base::AutoLock auto_lock(
      *static_cast<base::Lock*>(threadpool->execution_mutex));

  // Setup global arguments.
  pthreadpool_store_relaxed_void_p(&threadpool->thread_function,
                                   (void*)thread_function);
  pthreadpool_store_relaxed_void_p(&threadpool->task, task);
  pthreadpool_store_relaxed_void_p(&threadpool->argument, context);
  pthreadpool_store_relaxed_uint32_t(&threadpool->flags, flags);

  const struct fxdiv_divisor_size_t threads_count = threadpool->threads_count;

  if (params_size != 0) {
    memcpy(&threadpool->params, params, params_size);
  }

  // Spread the work between threads.
  const struct fxdiv_result_size_t range_params =
      fxdiv_divide_size_t(linear_range, threads_count);
  size_t range_start = 0;
  for (size_t tid = 0; tid < threads_count.value; tid++) {
    struct thread_info* thread = &threadpool->threads[tid];
    const size_t range_length =
        range_params.quotient + (size_t)(tid < range_params.remainder);
    const size_t range_end = range_start + range_length;
    pthreadpool_store_relaxed_size_t(&thread->range_start, range_start);
    pthreadpool_store_relaxed_size_t(&thread->range_end, range_end);
    pthreadpool_store_relaxed_size_t(&thread->range_length, range_length);

    /* The next subrange starts where the previous ended */
    range_start = range_end;
  }

  auto job = std::make_unique<PthreadPoolJob>(threadpool, threads_count.value);
  auto handle = base::PostJob(
      FROM_HERE, {},
      base::BindRepeating(&PthreadPoolJob::Run, base::Unretained(job.get())),
      base::BindRepeating(&PthreadPoolJob::GetMaxConcurrency,
                          base::Unretained(job.get())));
  handle.Join();
}

void pthreadpool_destroy(struct pthreadpool* threadpool) {
  if (threadpool != nullptr) {
    // Release resources.
    delete static_cast<base::Lock*>(threadpool->execution_mutex);
    pthreadpool_deallocate(threadpool);
  }
}
