// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_tensor_desc.h"

#include "base/check_op.h"
#include "base/numerics/checked_math.h"
#include "base/containers/span.h"

#pragma optimize("", off) // TODO:::DELETE

namespace content::webnn {

namespace {

size_t GetBytesOfDataType(DML_TENSOR_DATA_TYPE data_type) {
  static_assert(sizeof(float) == 4, "DirectML expects to run on machines with 4-byte floats.");
  static_assert(sizeof(double) == 8, "DirectML expects to run on machines with 8-byte doubles.");

  switch (data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return sizeof(float);
    case DML_TENSOR_DATA_TYPE_FLOAT64:
      return sizeof(double);
    case DML_TENSOR_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case DML_TENSOR_DATA_TYPE_INT8:
      return sizeof(int8_t);
    case DML_TENSOR_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case DML_TENSOR_DATA_TYPE_INT16:
      return sizeof(int16_t);
    case DML_TENSOR_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case DML_TENSOR_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case DML_TENSOR_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    case DML_TENSOR_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

absl::optional<UINT64> CalculateElementsNumber(
    const std::vector<UINT>& dimensions) {
  base::CheckedNumeric<UINT64> checked_elements_number = 1;
  for (auto& d : dimensions) {
    checked_elements_number *= d;
  }
  if (!checked_elements_number.IsValid()) {
    return absl::nullopt;
  }
  return checked_elements_number.ValueOrDie();
}

absl::optional<UINT64> CalculateElementsNumber(
    const std::vector<UINT>& dimensions,
    const absl::optional<std::vector<UINT>>& strides) {
  if (!strides) {
    return CalculateElementsNumber(dimensions);
  }

  base::CheckedNumeric<UINT64> checked_elements_number = 1;
  for (uint32_t i = 0; i < dimensions.size(); ++i) {
    // The specific dimension from broadcasting shouldn't increase the count of
    // elements.
    auto d = dimensions[i];
    if (strides.value()[i] == 0) {
      d = 1;
    }
    checked_elements_number *= d;
  }
  if (!checked_elements_number.IsValid()) {
    return absl::nullopt;
  }
  return checked_elements_number.ValueOrDie();
}

absl::optional<UINT64> TotalTensorSizeInBytes(
    DML_TENSOR_DATA_TYPE data_type,
    const std::vector<UINT>& dimensions,
    const absl::optional<std::vector<UINT>>& strides) {
  absl::optional<size_t> elements_num =
      CalculateElementsNumber(dimensions, strides);
  if (!elements_num) {
    return absl::nullopt;
  }

  base::CheckedNumeric<UINT64> checked_total_size_in_bytes =
      elements_num.value() * GetBytesOfDataType(data_type);
  if (!checked_total_size_in_bytes.IsValid()) {
    return absl::nullopt;
  }
  return checked_total_size_in_bytes.ValueOrDie();
}

void InsertPaddingOnes(/*inout*/ std::vector<uint32_t>& values,
                       size_t minimum_size,
                       TensorDesc::Alignment alignment) {
  // Insert's enough 1's to satisfy the minimum size.
  // If already large enough, no additional 1's are added.
  size_t old_size = values.size();
  size_t new_size = std::max(minimum_size, old_size);
  size_t filler_count = new_size - old_size;

  // Insert filler values on:
  // the leading edge if trailing aligned: [4,5] -> [1,1,1,4,5]
  // the trailing edge if leading aligned: [4,5] -> [4,5,1,1,1]
  auto insertion_point = (alignment == TensorDesc::Alignment::kTrailing)
                             ? values.begin()
                             : values.end();
  values.insert(insertion_point, filler_count, 1u);
}

}  // namespace

TensorDesc::TensorDesc() = default;

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       std::vector<UINT> dimensions) {
  Initialize(data_type, DML_TENSOR_FLAG_NONE, std::move(dimensions),
             absl::nullopt);
}

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       DML_TENSOR_FLAGS flags,
                       std::vector<UINT> dimensions) {
  Initialize(data_type, flags, std::move(dimensions), absl::nullopt);
}

TensorDesc::TensorDesc(DML_TENSOR_DATA_TYPE data_type,
                       DML_TENSOR_FLAGS flags,
                       std::vector<UINT> dimensions,
                       absl::optional<std::vector<UINT>> strides) {
  Initialize(data_type, flags, std::move(dimensions), std::move(strides));
}

void TensorDesc::Initialize(DML_TENSOR_DATA_TYPE data_type,
                            DML_TENSOR_FLAGS flags,
                            std::vector<UINT> dimensions,
                            absl::optional<std::vector<UINT>> strides) {
  DCHECK(dimensions.size() <= DML_TENSOR_DIMENSION_COUNT_MAX);
  DCHECK(!strides || dimensions.size() == strides->size());
  dimensions_ = std::move(dimensions);
  strides_ = std::move(strides);

  // DML (as of at least 1.11) requires dimension count to be at least 1
  // because otherwise validation during operator creation will complain with
  // E_INVALIDARG. So scalars must be conveyed with dimensions = [1].
  EnsureMinimumRank(1u, Alignment::kTrailing);

  // Round up to the nearest 4 bytes.
  // The buffer allocation already aligned chunks up to
  // DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT.
  uint64_t minimum_implied_size_in_bytes =
      TotalTensorSizeInBytes(data_type, dimensions_, strides_).value();
  minimum_implied_size_in_bytes = (minimum_implied_size_in_bytes + 3) & ~3ull;

  buffer_desc_.DimensionCount = dimensions_.size();
  buffer_desc_.Sizes = dimensions_.data();
  buffer_desc_.Strides = strides_ ? strides_.value().data() : nullptr;
  buffer_desc_.TotalTensorSizeInBytes = minimum_implied_size_in_bytes;
  buffer_desc_.GuaranteedBaseOffsetAlignment = 0;
  buffer_desc_.DataType = data_type;
  buffer_desc_.Flags = flags;
}

TensorDesc::TensorDesc(TensorDesc const& other)
    : dimensions_(other.dimensions_),
      strides_(other.strides_),
      buffer_desc_(other.buffer_desc_),
      tensor_desc_(other.tensor_desc_) {
  // Update the internal self-referential pointers.
  buffer_desc_.Sizes = dimensions_.data();
  buffer_desc_.Strides = strides_ ? strides_.value().data() : nullptr;
}

TensorDesc::TensorDesc(TensorDesc&& other) = default;

TensorDesc& TensorDesc::operator=(TensorDesc&& other) = default;

TensorDesc& TensorDesc::operator=(const TensorDesc& other) {
  dimensions_ = other.dimensions_;
  strides_ = other.strides_;
  buffer_desc_ = other.buffer_desc_;
  tensor_desc_ = other.tensor_desc_;

  // Update the internal self-referential pointers.
  buffer_desc_.Sizes = dimensions_.data();
  buffer_desc_.Strides = strides_ ? strides_.value().data() : nullptr;

  return *this;
}

TensorDesc::~TensorDesc() = default;

DML_TENSOR_DESC* TensorDesc::Get() {
  DCHECK(buffer_desc_.Sizes == dimensions_.data());
  DCHECK(buffer_desc_.Strides == nullptr || buffer_desc_.Strides == strides_.value().data());

  // Refresh the pointers to avoid them being stale after move
  // or copy construction.

  if (buffer_desc_.DataType == DML_TENSOR_DATA_TYPE_UNKNOWN) {
    return nullptr;
  }
  tensor_desc_ = DML_TENSOR_DESC{DML_TENSOR_TYPE_BUFFER, &buffer_desc_};
  return &tensor_desc_;
}

DML_TENSOR_DATA_TYPE TensorDesc::GetDataType() const {
  return buffer_desc_.DataType;
}

DML_TENSOR_FLAGS TensorDesc::GetFlags() const {
  return buffer_desc_.Flags;
}

std::vector<UINT>& TensorDesc::GetDimensions() {
  return dimensions_;
}

absl::optional<std::vector<UINT>>& TensorDesc::GetStrides() {
  return strides_;
}

std::vector<UINT> TensorDesc::GetStridesOrDefaultStrides() const {
  return strides_ ? *strides_ : ComputeDecreasingStrides(dimensions_);
}

void TensorDesc::EnsureStridesExist() {
  if (!strides_)
  {
    std::vector<uint32_t> new_strides = ComputeDecreasingStrides(dimensions_);
    strides_ = std::move(new_strides);
    buffer_desc_.Strides = strides_.value().data();
  }
}

UINT64 TensorDesc::GetTotalTensorSizeInBytes() {
  return buffer_desc_.TotalTensorSizeInBytes;
}

void TensorDesc::EnsureMinimumRank(size_t minimum_rank, Alignment alignment)
{
  if (dimensions_.size() < minimum_rank) {
    // Note this does not change the TotalTensorSizeInBytes, since leading 1's
    // make no difference, nor any other field.
    InsertPaddingOnes(/*inout*/ dimensions_, minimum_rank, alignment);
    buffer_desc_.DimensionCount = dimensions_.size();
    buffer_desc_.Sizes = dimensions_.data();

    if (strides_) {
      InsertPaddingOnes(/*inout*/ *strides_, minimum_rank, alignment);
      buffer_desc_.Strides = strides_.value().data();
    }
  }
}

std::vector<uint32_t> TensorDesc::ComputeDecreasingStrides(base::span<const uint32_t> dimensions)
{
  auto dimension_count = dimensions.size();
  std::vector<uint32_t> strides(dimension_count);

  uint32_t stride = 1;
  for (auto i = dimension_count; i-- > 0; )
  {
    strides[i] = stride;
    stride *= dimensions[i];
  }

  return strides;
}

void TensorDesc::PermuteDimensions(base::span<const uint32_t> permutation, Alignment alignment)
{
  size_t permutation_rank = permutation.size();

  // Ensure there are enough elements to apply the permutation by adding
  // leading 1's if necessary.
  EnsureMinimumRank(permutation_rank, alignment);

  // Compute strides *before* the reordering, since callers use this function
  // to rearrange the dimensions (depending on the affinity of the backend
  // for a certain preference - e.g. NHWC vs NCHW) and need the strides to
  // be adjusted accordingly. Otherwise strides would be computed using
  // the permuted dimensions and read the wrong elements.
  EnsureStridesExist();

  // If there are more dimensions than permutation the size (e.g. maybe
  // they were already padded out to some limit like 4D with leading 1's)
  // then the permutation only applies to the significant portion,
  // depending on alignment. Any additional leading/traling batch dimensions
  // are ignorable.

  size_t subset_offset = (alignment == Alignment::kLeading)
                             ? 0
                             : dimensions_.size() - permutation_rank;
  auto dimensions_span = base::span<uint32_t>(dimensions_);
  auto strides_span = base::span<uint32_t>(*strides_);
  dimensions_span = dimensions_span.subspan(subset_offset, permutation_rank);
  strides_span = strides_span.subspan(subset_offset, permutation_rank);

  auto permute = [](/*inout*/ base::span<uint32_t> values,
                    base::span<const uint32_t> permutation) {
    DCHECK(values.size() == permutation.size());

    // Gather the original values via the permutation.
    std::vector<uint32_t> temporary_values(values.begin(), values.end());
    for (size_t i = 0, count = values.size(); i < count; ++i) {
      values[i] = temporary_values[permutation[i]];
    }
  };

  permute(/*inout*/ dimensions_span, permutation);
  permute(/*inout*/ strides_span, permutation);
}

void TensorDesc::BroadcastTo(base::span<const uint32_t> broadcast_dimensions,
                             Alignment alignment,
                             size_t ignorable_tail_count) {
  size_t broadcast_rank = broadcast_dimensions.size();
  EnsureMinimumRank(broadcast_rank, alignment);
  EnsureStridesExist();

  // Determine the window of dimensions and strides that are to be modified.
  // e.g.
  //    Alignment            = trailing/right
  //    Ignorable tail count = 0
  //    Original dimensions  =   [2,1,4]
  //    Original strides     =   [4,4,1]
  //    Broadcast dimensions = [5,2,3,4]
  //    New dimensions       = [5,2,3,4]
  //    New strides          = [0,4,0,1]
  // e.g.
  //    Alignment            = trailing/right
  //    Ignorable tail count = 2
  //    Original dimensions  =   [2,1,4]
  //    Original strides     =   [4,4,1]
  //    Broadcast dimensions = [5,2,3,4]
  //    New dimensions       = [5,2,1,4]
  //    New strides          = [0,4,4,1]
  // e.g.
  //    Alignment            = leading/left
  //    Ignorable tail count = 0
  //    Original dimensions  = [3,1,4]
  //    Original strides     = [4,4,1]
  //    Broadcast dimensions = [3,2]
  //    New dimensions       = [3,2,4]
  //    New strides          = [4,0,1]

  size_t subset_offset = (alignment == Alignment::kLeading)
                             ? 0
                             : dimensions_.size() - broadcast_rank;
  auto dimensions_span = base::span<uint32_t>(dimensions_);
  auto strides_span = base::span<uint32_t>(*strides_);
  dimensions_span = dimensions_span.subspan(subset_offset, broadcast_rank);
  strides_span = strides_span.subspan(subset_offset, broadcast_rank);
  size_t clamped_rank =
      broadcast_rank - std::min(broadcast_rank, ignorable_tail_count);

  for (size_t i = 0; i < clamped_rank; ++i) {
    // Any 1-size dimensions get promoted to the target broadcast dimension
    // and have their stride set to 0 for projection.
    if (dimensions_span[i] == 1u) {
      dimensions_span[i] = broadcast_dimensions[i];
      strides_span[i] = 0;
    }
  }
}


}  // namespace content::webnn
