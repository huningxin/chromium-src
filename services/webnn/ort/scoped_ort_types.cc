// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/scoped_ort_types.h"

namespace webnn::ort {

#define SCOPED_ORT_TYPE_DEFINITION(ort_type, ort_api)                          \
  ScopedOrt##ort_type::ScopedOrt##ort_type() {                                 \
    pptr_ = std::make_unique<Ort##ort_type*>(nullptr);                         \
  }                                                                            \
  ScopedOrt##ort_type::~ScopedOrt##ort_type() {                                \
    if (pptr_) {                                                               \
      Get##ort_api()->Release##ort_type(*pptr_);                               \
    }                                                                          \
  }                                                                            \
  ScopedOrt##ort_type::ScopedOrt##ort_type(ScopedOrt##ort_type&&) = default;   \
  ScopedOrt##ort_type& ScopedOrt##ort_type::operator=(ScopedOrt##ort_type&&) = \
      default;

SCOPED_ORT_TYPE_DEFINITION(Value, OrtApi)
SCOPED_ORT_TYPE_DEFINITION(MemoryInfo, OrtApi)
SCOPED_ORT_TYPE_DEFINITION(OpAttr, OrtApi)
SCOPED_ORT_TYPE_DEFINITION(TypeInfo, OrtApi)
SCOPED_ORT_TYPE_DEFINITION(TensorTypeAndShapeInfo, OrtApi)
SCOPED_ORT_TYPE_DEFINITION(ValueInfo, OrtModelBuilderApi)
SCOPED_ORT_TYPE_DEFINITION(Node, OrtModelBuilderApi)
SCOPED_ORT_TYPE_DEFINITION(Graph, OrtModelBuilderApi)
SCOPED_ORT_TYPE_DEFINITION(Model, OrtModelBuilderApi)

}  // namespace webnn::ort
