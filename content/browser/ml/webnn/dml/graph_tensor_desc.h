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
  enum Alignment : uint32_t
  {
    kLeading,  // Align to leading/left edge. e.g. the 1 in [1,2,3]
    kTrailing, // Align to trailing/right edge. e.g. the 3 in [1,2,3]
  };

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

  // Ensures the rank is at least the minimum rank, filling the opposite side
  // (depending on alignment) with 1's when needed.
  // e.g. [5,6] with minimum rank of 4 yields [1,1,5,6].
  void EnsureMinimumRank(size_t minimum_rank, Alignment alignment);

  // Permute the original dimensions/strides to the given remapping.
  // e.g. dimensions [5,6,7,8] with permutation [3,2,0,1] yields dimensions
  // of [8,7,5,6]. All indices must be within [0, permutation.size() - 1].
  // A permutation larger than the current rank will increase the rank first.
  void PermuteDimensions(base::span<const uint32_t> permutation,
                         Alignment alignment);

  void BroadcastTo(base::span<const uint32_t> dimensions,
                   Alignment alignment,
                   size_t ignorable_tail_count = 0);

  UINT64 GetTotalTensorSizeInBytes();

  // Returns the default decreasing order packed strides for the given dimensions.
  // e.g. dimensions [1,2,3,4] yields strides [24,12,4,1].
  // See https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides.
  static std::vector<uint32_t> ComputeDecreasingStrides(
      base::span<const uint32_t> dimensions);

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
