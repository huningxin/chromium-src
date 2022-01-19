// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/webgpu/ml.h"

#include <utility>

#include "third_party/blink/renderer/core/execution_context/navigator_base.h"
#include "third_party/blink/renderer/modules/webgpu/ml_context.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"

namespace blink {

const char ML::kSupplementName[] = "ML";

// static
ML* ML::ml(NavigatorBase& navigator) {
  ML* ml = Supplement<NavigatorBase>::From<ML>(navigator);
  if (!ml) {
    ml = MakeGarbageCollected<ML>(navigator);
    ProvideTo(navigator, ml);
  }
  return ml;
}

ML::ML(NavigatorBase& navigator)
    : Supplement<NavigatorBase>(navigator) {}

ML::~ML() = default;

void ML::Trace(Visitor* visitor) const {
  ScriptWrappable::Trace(visitor);
  Supplement<NavigatorBase>::Trace(visitor);
}

MLContext* ML::createContext(GPUDevice* device) {
  DCHECK(device);

  MLContext* context = MakeGarbageCollected<MLContext>(device);
  return context;
}

}  // namespace blink
