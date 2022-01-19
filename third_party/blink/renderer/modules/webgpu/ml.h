// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_H_

#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/supplementable.h"

namespace blink {

class NavigatorBase;
class GPUDevice;
class MLContext;

class ML final : public ScriptWrappable,
                 public Supplement<NavigatorBase> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static const char kSupplementName[];

  // Getter for navigator.gpu
  static ML* ml(NavigatorBase&);

  explicit ML(NavigatorBase&);

  ML(const ML&) = delete;
  ML& operator=(const ML&) = delete;

  ~ML() override;

  // ScriptWrappable overrides
  void Trace(Visitor* visitor) const override;

  // gpu.idl
  MLContext* createContext(GPUDevice* device);
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBGPU_ML_H_
