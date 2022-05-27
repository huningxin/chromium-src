// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"

#include <thread>

namespace blink {

MLContextXnnpack::MLContextXnnpack(const unsigned int num_threads, ML* ml)
    : MLContext(V8MLDevicePreference(V8MLDevicePreference::Enum::kCpu),
                V8MLPowerPreference(V8MLPowerPreference::Enum::kAuto),
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
                num_threads,
                ml),
      pthreadpool_(nullptr) {}

MLContextXnnpack::~MLContextXnnpack() {
  xnn_deinitialize();
  if (pthreadpool_ != nullptr) {
    pthreadpool_destroy(pthreadpool_);
  }
}

bool MLContextXnnpack::Initialize() {
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
  return true;
}

pthreadpool_t MLContextXnnpack::Pthreadpool() const {
  return pthreadpool_;
}

}  // namespace blink