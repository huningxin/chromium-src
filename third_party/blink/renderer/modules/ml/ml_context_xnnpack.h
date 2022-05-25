// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_XNNPACK_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_XNNPACK_H_

#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/xnnpack/src/include/xnnpack.h"

namespace blink {

class MLContextXnnpack : public MLContext {
 public:
  MLContextXnnpack(const V8MLDevicePreference device_preference,
            const V8MLPowerPreference power_preference,
            const unsigned int num_threads,
            ML* ml);
  ~MLContextXnnpack() override;

  bool Initialize();

  pthreadpool_t Pthreadpool() const;

 private:
  pthreadpool_t pthreadpool_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_XNNPACK_H_