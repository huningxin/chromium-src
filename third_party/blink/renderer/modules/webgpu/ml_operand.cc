// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml_operand.h"

namespace blink {

MLOperand::MLOperand(GPUDevice* device, WGPUOperand operand)
    : DawnObject<WGPUOperand>(device, operand) {
}

void MLOperand::Trace(Visitor* visitor) const {
  DawnObject<WGPUOperand>::Trace(visitor);
}

}  // namespace blink