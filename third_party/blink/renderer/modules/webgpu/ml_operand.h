// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERAND_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERAND_H_

#include "third_party/blink/renderer/modules/webgpu/dawn_object.h"

namespace blink {

class MLOperand : public DawnObject<WGPUOperand> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLOperand(GPUDevice* device,
                     WGPUOperand operand);

  MLOperand(const MLOperand&) = delete;
  MLOperand& operator=(const MLOperand&) = delete;

  void Trace(Visitor* visitor) const override;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_OPERAND_H_
