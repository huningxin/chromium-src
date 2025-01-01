// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_ORT_SCOPED_ORT_TYPES_H
#define SERVICES_WEBNN_ORT_SCOPED_ORT_TYPES_H

#include <memory>

#include "services/webnn/ort/utils_ort.h"
#include "third_party/onnxruntime_headers/src/include/onnxruntime/core/session/onnxruntime_c_api.h"

namespace webnn::ort {

#define SCOPED_ORT_TYPE_DECLARATION(ort_type)                            \
  class ScopedOrt##ort_type {                                            \
   public:                                                               \
    ScopedOrt##ort_type();                                               \
    ~ScopedOrt##ort_type();                                              \
    ScopedOrt##ort_type(const ScopedOrt##ort_type&) = delete;            \
    ScopedOrt##ort_type& operator=(const ScopedOrt##ort_type&) = delete; \
    ScopedOrt##ort_type(ScopedOrt##ort_type&&);                          \
    ScopedOrt##ort_type& operator=(ScopedOrt##ort_type&&);               \
    Ort##ort_type* get_ptr() const {                                     \
      return *pptr_;                                                     \
    }                                                                    \
    Ort##ort_type** get_pptr() const {                                   \
      return pptr_.get();                                                \
    }                                                                    \
                                                                         \
   private:                                                              \
    std::unique_ptr<Ort##ort_type*> pptr_;                               \
  };

SCOPED_ORT_TYPE_DECLARATION(Value)
SCOPED_ORT_TYPE_DECLARATION(MemoryInfo)
SCOPED_ORT_TYPE_DECLARATION(OpAttr)
SCOPED_ORT_TYPE_DECLARATION(TypeInfo)
SCOPED_ORT_TYPE_DECLARATION(TensorTypeAndShapeInfo)
SCOPED_ORT_TYPE_DECLARATION(ValueInfo)
SCOPED_ORT_TYPE_DECLARATION(Node)
SCOPED_ORT_TYPE_DECLARATION(Graph)
SCOPED_ORT_TYPE_DECLARATION(Model)

}  // namespace webnn::ort

#endif  // SERVICES_WEBNN_ORT_SCOPED_ORT_TYPES_H
