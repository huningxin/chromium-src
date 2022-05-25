// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"

namespace blink {

MLContextXnnpack::MLContextXnnpack(const V8MLDevicePreference device_preference,
            const V8MLPowerPreference power_preference,
            const unsigned int num_threads,
            ML* ml) : MLContext(
                device_preference,
                power_preference,
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite), /*should move to model loader context*/
                num_threads, ml), pthreadpool_(nullptr) {}

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
  pthreadpool_ = pthreadpool_create(num_threads_);
  if (pthreadpool_ == nullptr) {
    return false;
  }
  return true;
}

pthreadpool_t MLContextXnnpack::Pthreadpool() const {
  return pthreadpool_;
}

}