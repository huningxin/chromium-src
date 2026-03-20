// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Custom main() for WebNN fuzz tests that initializes base::CommandLine
// with the real argc/argv before running tests.

#include "base/command_line.h"
#include "gtest/gtest.h"
#include "third_party/fuzztest/src/fuzztest/init_fuzztest.h"

int main(int argc, char** argv) {
  base::CommandLine::Init(argc, argv);

  testing::InitGoogleTest(&argc, argv);
  fuzztest::ParseAbslFlags(argc, argv);
  fuzztest::InitFuzzTest(&argc, &argv);
  return RUN_ALL_TESTS();
}
