// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_TENSOR_DESC_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_TENSOR_DESC_H_

#include <vector>

#include <wrl.h>
#include "DirectML.h"
#include "third_party/abseil-cpp/absl/types/optional.h"
#include "base/containers/span.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

class TensorDesc final {
 public:
  TensorDesc();
  TensorDesc(DML_TENSOR_DATA_TYPE data_type, std::vector<UINT> dimensions);
  TensorDesc(DML_TENSOR_DATA_TYPE data_type,
             DML_TENSOR_FLAGS flags,
             std::vector<UINT> dimensions);
  TensorDesc(DML_TENSOR_DATA_TYPE data_type,
             DML_TENSOR_FLAGS flags,
             std::vector<UINT> dimensions,
             absl::optional<std::vector<UINT>> strides);
  TensorDesc(const TensorDesc& other);
  TensorDesc(TensorDesc&& other);
  TensorDesc& operator=(TensorDesc&& other);
  TensorDesc& operator=(const TensorDesc& other);
  ~TensorDesc();

  DML_TENSOR_DESC* Get();
  DML_TENSOR_DATA_TYPE GetDataType() const;
  DML_TENSOR_FLAGS GetFlags() const;
  std::vector<UINT>& GetDimensions();

  // Returns the strides, or empty if none exist.
  absl::optional<std::vector<UINT>>& GetStrides();

  // Returns valid strides, either the explicit ones contained or the generated
  // ones (packed with no padding in descending order left-to-right) if empty.
  std::vector<UINT> GetStridesOrDefaultStrides() const;

  // Ensures strides are not empty, computing them from dimensions if needed.
  void EnsureStridesExist();

  // Ensures the rank is at least the minimum rank, filling the leading left side
  // with 1's if needed. e.g. [4,5] with minimum rank of 4 yields [1,1,4,5].
  void EnsureMinimumRankRightAligned(size_t minimum_rank);

  UINT64 GetTotalTensorSizeInBytes();

  // Static helper functions.
  static std::vector<uint32_t> ComputeDecreasingStrides(base::span<const uint32_t> dimensions);

 private:
  void Initialize(DML_TENSOR_DATA_TYPE data_type,
                  DML_TENSOR_FLAGS flags,
                  std::vector<UINT> dimensions,
                  absl::optional<std::vector<UINT>> strides);

  // DML_BUFFER_TENSOR_DESC only has a pointer to dimensions and strides,
  // which points to dimensions_ and strides_.
  std::vector<UINT> dimensions_;
  absl::optional<std::vector<UINT>> strides_;

  // Describes a tensor that will be stored in a Direct3D 12 buffer resource.
  DML_BUFFER_TENSOR_DESC buffer_desc_ = {};
  DML_TENSOR_DESC tensor_desc_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_TENSOR_DESC_H_
