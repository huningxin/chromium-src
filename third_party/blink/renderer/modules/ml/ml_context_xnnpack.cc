// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"

#include "base/synchronization/lock.h"
#include "base/system/sys_info.h"
#include "base/thread_annotations.h"
#include "build/buildflag.h"
#include "third_party/blink/renderer/platform/wtf/thread_safe_ref_counted.h"

namespace blink {

namespace {

class SharedXnnpackContext : public ThreadSafeRefCounted<SharedXnnpackContext> {
 public:
  static scoped_refptr<SharedXnnpackContext> GetInstance() {
    if (instance_ == nullptr) {
      return base::MakeRefCounted<SharedXnnpackContext>();
    } else {
      return base::WrapRefCounted(instance_);
    }
  }

  explicit SharedXnnpackContext() : initialized_(false) { instance_ = this; }

  SharedXnnpackContext(const SharedXnnpackContext&) = delete;
  SharedXnnpackContext& operator=(const SharedXnnpackContext&) = delete;

  bool Initialize() {
    base::AutoLock auto_lock(lock_);
    if (initialized_) {
      return true;
    }
#if BUILDFLAG(IS_WIN)
    if (xnn_initialize(NULL) != xnn_status_success) {
      return false;
    }
#endif

    pthreadpool_ = pthreadpool_create(base::SysInfo::NumberOfProcessors() / 2);
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

  ~SharedXnnpackContext() {
#if BUILDFLAG(IS_WIN)
    xnn_deinitialize();
#endif
    if (pthreadpool_ != nullptr) {
      pthreadpool_destroy(pthreadpool_);
    }
    instance_ = nullptr;
  }

  static SharedXnnpackContext* instance_;

  base::Lock lock_;
  bool initialized_ GUARDED_BY(lock_);
  pthreadpool_t pthreadpool_ GUARDED_BY(lock_);
};

SharedXnnpackContext* SharedXnnpackContext::instance_ = nullptr;

}  // namespace

MLContextXnnpack::MLContextXnnpack(ML* ml)
    : MLContext(V8MLDevicePreference(V8MLDevicePreference::Enum::kCpu),
                V8MLPowerPreference(V8MLPowerPreference::Enum::kAuto),
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
                0,
                ml),
      impl_(SharedXnnpackContext::GetInstance()) {}

MLContextXnnpack::~MLContextXnnpack() = default;

bool MLContextXnnpack::Initialize() {
  return impl_->Initialize();
}

pthreadpool_t MLContextXnnpack::Pthreadpool() const {
  return impl_->Pthreadpool();
}

}  // namespace blink
