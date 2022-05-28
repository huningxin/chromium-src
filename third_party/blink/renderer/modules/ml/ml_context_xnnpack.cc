// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"

#include <thread>

namespace blink {

namespace {

class SharedXnnpackContext
    : public base::RefCountedThreadSafe<SharedXnnpackContext> {
 public:
  static SharedXnnpackContext* GetInstance(size_t num_threads) {
    if (instance_ == nullptr) {
      instance_ = new SharedXnnpackContext(num_threads);
    }
    return instance_;
  }

  SharedXnnpackContext(const SharedXnnpackContext&) = delete;
  SharedXnnpackContext& operator=(const SharedXnnpackContext&) = delete;

  ~SharedXnnpackContext() {
    base::AutoLock auto_lock(lock_);
    xnn_deinitialize();
    if (pthreadpool_ != nullptr) {
      pthreadpool_destroy(pthreadpool_);
    }
    instance_ = nullptr;
  }

  bool Initialize() {
    base::AutoLock auto_lock(lock_);
    if (initialized_) {
      return true;
    }
    if (xnn_initialize(NULL) != xnn_status_success) {
      return false;
    }

    uint32_t num_cores = std::thread::hardware_concurrency() / 2;
    size_t num_threads = num_threads_ == 0          ? num_cores
                         : num_threads_ > num_cores ? num_cores
                                                    : num_threads_;
    pthreadpool_ = pthreadpool_create(num_threads);
    if (pthreadpool_ == nullptr) {
      return false;
    }
    initialized_ = true;
    return true;
  }

  pthreadpool_t Pthreadpool() { return pthreadpool_; }

 private:
  explicit SharedXnnpackContext(size_t num_threads)
      : initialized_(false), num_threads_(num_threads) {}

  static SharedXnnpackContext* instance_;

  base::Lock lock_;
  bool initialized_;
  size_t num_threads_;
  pthreadpool_t pthreadpool_;
};

SharedXnnpackContext* SharedXnnpackContext::instance_ = nullptr;

}  // namespace

MLContextXnnpack::MLContextXnnpack(const unsigned int num_threads, ML* ml)
    : MLContext(V8MLDevicePreference(V8MLDevicePreference::Enum::kCpu),
                V8MLPowerPreference(V8MLPowerPreference::Enum::kAuto),
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
                num_threads,
                ml) {
  impl_ = base::subtle::AdoptRefIfNeeded(
      SharedXnnpackContext::GetInstance(num_threads),
      SharedXnnpackContext::kRefCountPreference);
}

MLContextXnnpack::~MLContextXnnpack() = default;

bool MLContextXnnpack::Initialize() {
  return impl_->Initialize();
}

pthreadpool_t MLContextXnnpack::Pthreadpool() const {
  return impl_->Pthreadpool();
}

}  // namespace blink