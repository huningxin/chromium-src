// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_CONTEXT_H_

#include "third_party/blink/renderer/modules/webgpu/dawn_object.h"

namespace blink {

class MLContext : public DawnObjectImpl {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLContext(GPUDevice* device);

  MLContext(const MLContext&) = delete;
  MLContext& operator=(const MLContext&) = delete;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_CONTEXT_H_
