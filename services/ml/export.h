// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_EXPORT_H_
#define SERVICES_ML_EXPORT_H_

#if defined(COMPONENT_BUILD)

#if defined(WIN32)

#if defined(SERVICES_ML_IMPLEMENTATION)
#define ML_EXPORT __declspec(dllexport)
#else
#define ML_EXPORT __declspec(dllimport)
#endif

#else  // !defined(WIN32)

#if defined(SERVICES_ML_IMPLEMENTATION)
#define ML_EXPORT __attribute__((visibility("default")))
#else
#define ML_EXPORT
#endif

#endif

#else  // !defined(COMPONENT_BUILD)

#define ML_EXPORT

#endif

#endif  // SERVICES_ML_EXPORT_H_