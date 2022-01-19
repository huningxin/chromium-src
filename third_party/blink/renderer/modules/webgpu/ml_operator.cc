// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml_operator.h"

namespace blink {

MLOperator::MLOperator(GPUDevice* device, WGPUFusionOperator op)
    : DawnObject<WGPUFusionOperator>(device, op) {
}

void MLOperator::Trace(Visitor* visitor) const {
  DawnObject<WGPUFusionOperator>::Trace(visitor);
}

}  // namespace blink