/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>

#include <cstdint>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data, const int gpu_device_id) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
  gpu_device_id_ = gpu_device_id >= 0 ? gpu_device_id : 0;
  SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  cuda_used_indices_ = nullptr;
  cuda_data_by_column_ = nullptr;
  cuda_column_bit_type_ = nullptr;
  cuda_feature_min_bin_ = nullptr;
  cuda_feature_max_bin_ = nullptr;
  cuda_feature_offset_ = nullptr;
  cuda_feature_most_freq_bin_ = nullptr;
  cuda_feature_default_bin_ = nullptr;
  cuda_feature_missing_is_zero_ = nullptr;
  cuda_feature_missing_is_na_ = nullptr;
  cuda_feature_mfb_is_zero_ = nullptr;
  cuda_feature_mfb_is_na_ = nullptr;
  cuda_feature_to_column_ = nullptr;
  data_by_column_.clear();
}

CUDAColumnData::~CUDAColumnData() {
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
  DeallocateCUDAMemory<void*>(&cuda_data_by_column_, __FILE__, __LINE__);
  // Free per-column allocations. If a compact view was set, data_by_column_
  // currently holds offsets into someone-else's buffer (NOT ours to free) —
  // free the snapshot in data_by_column_orig_ instead. If no compact view was
  // ever set, data_by_column_orig_ is empty and we free data_by_column_ directly.
  // If init_skipped_per_column_alloc_ was true, no per-column alloc happened
  // and both vectors are nullptr-only — skip the loop.
  if (!init_skipped_per_column_alloc_) {
    std::vector<void*>& to_free = data_by_column_orig_.empty() ? data_by_column_ : data_by_column_orig_;
    for (size_t i = 0; i < to_free.size(); ++i) {
      DeallocateCUDAMemory<void>(&to_free[i], __FILE__, __LINE__);
    }
  }
  DeallocateCUDAMemory<uint8_t>(&cuda_column_bit_type_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_min_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_max_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_offset_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_most_freq_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_default_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_missing_is_zero_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_missing_is_na_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_is_zero_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_is_na_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_feature_to_column_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
}

template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
void CUDAColumnData::InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, void** out_column_data_pointer) {
  BIN_TYPE* cuda_column_data = nullptr;
  if (!IS_SPARSE) {
    if (IS_4BIT) {
      std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
      const BIN_TYPE* in_column_data_reintrepreted = reinterpret_cast<const BIN_TYPE*>(in_column_data);
      for (data_size_t i = 0; i < num_data_; ++i) {
        expanded_column_data[i] = static_cast<BIN_TYPE>((in_column_data_reintrepreted[i >> 1] >> ((i & 1) << 2)) & 0xf);
      }
      InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                  expanded_column_data.data(),
                                                  static_cast<size_t>(num_data_),
                                                  __FILE__,
                                                  __LINE__);
    } else {
      InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                  reinterpret_cast<const BIN_TYPE*>(in_column_data),
                                                  static_cast<size_t>(num_data_),
                                                  __FILE__,
                                                  __LINE__);
    }
  } else {
    // need to iterate bin iterator
    std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
    for (data_size_t i = 0; i < num_data_; ++i) {
      expanded_column_data[i] = static_cast<BIN_TYPE>(bin_iterator->RawGet(i));
    }
    InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                expanded_column_data.data(),
                                                static_cast<size_t>(num_data_),
                                                __FILE__,
                                                __LINE__);
  }
  *out_column_data_pointer = reinterpret_cast<void*>(cuda_column_data);
}

void CUDAColumnData::Init(const int num_columns,
                          const std::vector<const void*>& column_data,
                          const std::vector<BinIterator*>& column_bin_iterator,
                          const std::vector<uint8_t>& column_bit_type,
                          const std::vector<uint32_t>& feature_max_bin,
                          const std::vector<uint32_t>& feature_min_bin,
                          const std::vector<uint32_t>& feature_offset,
                          const std::vector<uint32_t>& feature_most_freq_bin,
                          const std::vector<uint32_t>& feature_default_bin,
                          const std::vector<uint8_t>& feature_missing_is_zero,
                          const std::vector<uint8_t>& feature_missing_is_na,
                          const std::vector<uint8_t>& feature_mfb_is_zero,
                          const std::vector<uint8_t>& feature_mfb_is_na,
                          const std::vector<int>& feature_to_column) {
  num_columns_ = num_columns;
  column_bit_type_ = column_bit_type;
  feature_max_bin_ = feature_max_bin;
  feature_min_bin_ = feature_min_bin;
  feature_offset_ = feature_offset;
  feature_most_freq_bin_ = feature_most_freq_bin;
  feature_default_bin_ = feature_default_bin;
  feature_missing_is_zero_ = feature_missing_is_zero;
  feature_missing_is_na_ = feature_missing_is_na;
  feature_mfb_is_zero_ = feature_mfb_is_zero;
  feature_mfb_is_na_ = feature_mfb_is_na;
  data_by_column_.resize(num_columns_, nullptr);
  // Decide whether to skip the per-column GPU allocation. With a 32 GB GPU and
  // a 17 GB row matrix on top, we can't afford another 17 GB of per-column data.
  // Skip when total size would exceed 8 GB; the caller (tree learner) will provide
  // a compact column view per tree via SetCompactColumnView.
  size_t expected_total_bytes = 0;
  for (int c = 0; c < num_columns_; ++c) {
    const int8_t bt = column_bit_type[c];
    int bytes_per = (bt == 4 || bt == 8) ? 1 : (bt == 16 ? 2 : 4);
    expected_total_bytes += static_cast<size_t>(num_data_) * bytes_per;
  }
  init_skipped_per_column_alloc_ = (expected_total_bytes > (size_t)8 * 1024 * 1024 * 1024);
  if (init_skipped_per_column_alloc_) {
    Log::Warning("CUDAColumnData: skipping per-column allocation (would be %.2f GB). "
                 "Caller must invoke SetCompactColumnView per tree.", expected_total_bytes / 1e9);
  }
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const int8_t bit_type = column_bit_type[column_index];
      if (column_data[column_index] != nullptr) {
        // is dense column
        if (init_skipped_per_column_alloc_) {
          // Adjust column_bit_type_ for 4-bit case (which expanded to 8) and skip GPU alloc.
          if (bit_type == 4) {
            column_bit_type_[column_index] = 8;
          }
          data_by_column_[column_index] = nullptr;
        } else if (bit_type == 4) {
          column_bit_type_[column_index] = 8;
          InitOneColumnData<false, true, uint8_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
        } else if (bit_type == 8) {
          InitOneColumnData<false, false, uint8_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
        } else if (bit_type == 16) {
          InitOneColumnData<false, false, uint16_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
        } else if (bit_type == 32) {
          InitOneColumnData<false, false, uint32_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
        } else {
          Log::Fatal("Unknow column bit type %d", bit_type);
        }
      } else {
        // is sparse column
        if (bit_type == 8) {
          InitOneColumnData<true, false, uint8_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
        } else if (bit_type == 16) {
          InitOneColumnData<true, false, uint16_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
        } else if (bit_type == 32) {
          InitOneColumnData<true, false, uint32_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
        } else {
          Log::Fatal("Unknow column bit type %d", bit_type);
        }
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  feature_to_column_ = feature_to_column;
  InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_,
                                           data_by_column_.data(),
                                           data_by_column_.size(),
                                           __FILE__,
                                           __LINE__);
  InitColumnMetaInfo();
}

void CUDAColumnData::CopySubrow(
  const CUDAColumnData* full_set,
  const data_size_t* used_indices,
  const data_size_t num_used_indices) {
  num_threads_ = full_set->num_threads_;
  num_columns_ = full_set->num_columns_;
  column_bit_type_ = full_set->column_bit_type_;
  feature_min_bin_ = full_set->feature_min_bin_;
  feature_max_bin_ = full_set->feature_max_bin_;
  feature_offset_ = full_set->feature_offset_;
  feature_most_freq_bin_ = full_set->feature_most_freq_bin_;
  feature_default_bin_ = full_set->feature_default_bin_;
  feature_missing_is_zero_ = full_set->feature_missing_is_zero_;
  feature_missing_is_na_ = full_set->feature_missing_is_na_;
  feature_mfb_is_zero_ = full_set->feature_mfb_is_zero_;
  feature_mfb_is_na_ = full_set->feature_mfb_is_na_;
  feature_to_column_ = full_set->feature_to_column_;
  if (cuda_used_indices_ == nullptr) {
    // initialize the subset cuda column data
    const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
    AllocateCUDAMemory<data_size_t>(&cuda_used_indices_, num_used_indices_size, __FILE__, __LINE__);
    data_by_column_.resize(num_columns_, nullptr);
    OMP_INIT_EX();
    #pragma omp parallel num_threads(num_threads_)
    {
      SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
      #pragma omp for schedule(static)
      for (int column_index = 0; column_index < num_columns_; ++column_index) {
        OMP_LOOP_EX_BEGIN();
        const uint8_t bit_type = column_bit_type_[column_index];
        if (bit_type == 8) {
          uint8_t* column_data = nullptr;
          AllocateCUDAMemory<uint8_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
          data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
        } else if (bit_type == 16) {
          uint16_t* column_data = nullptr;
          AllocateCUDAMemory<uint16_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
          data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
        } else if (bit_type == 32) {
          uint32_t* column_data = nullptr;
          AllocateCUDAMemory<uint32_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
          data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
        }
        OMP_LOOP_EX_END();
      }
    }
    OMP_THROW_EX();
    InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
    InitColumnMetaInfo();
    cur_subset_buffer_size_ = num_used_indices;
  } else {
    if (num_used_indices > cur_subset_buffer_size_) {
      ResizeWhenCopySubrow(num_used_indices);
      cur_subset_buffer_size_ = num_used_indices;
    }
  }
  CopyFromHostToCUDADevice<data_size_t>(cuda_used_indices_, used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
  num_used_indices_ = num_used_indices;
  LaunchCopySubrowKernel(full_set->cuda_data_by_column());
}

void CUDAColumnData::SetCompactColumnView(const std::vector<int>& column_to_compact_slot,
                                          void* compact_buf,
                                          size_t bytes_per_col) {
  // Snapshot original pointers on first compact-view call so we can both
  // restore on demand and free them in the destructor (the compact-buf offsets
  // we write into data_by_column_ are NOT owned by us).
  if (data_by_column_orig_.empty() && !data_by_column_.empty()) {
    data_by_column_orig_ = data_by_column_;
  }
  for (int c = 0; c < num_columns_; ++c) {
    if (c < static_cast<int>(column_to_compact_slot.size()) && column_to_compact_slot[c] >= 0) {
      const size_t off = static_cast<size_t>(column_to_compact_slot[c]) * bytes_per_col;
      data_by_column_[c] = reinterpret_cast<uint8_t*>(compact_buf) + off;
    } else {
      data_by_column_[c] = nullptr;
    }
  }
  if (cuda_data_by_column_ == nullptr) {
    InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
  } else {
    CopyFromHostToCUDADevice<void*>(cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
  }
}

void CUDAColumnData::RestoreOriginalColumnView() {
  if (data_by_column_orig_.empty()) return;
  data_by_column_ = data_by_column_orig_;
  CopyFromHostToCUDADevice<void*>(cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
}

void CUDAColumnData::ResizeWhenCopySubrow(const data_size_t num_used_indices) {
  const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_used_indices_, num_used_indices_size, __FILE__, __LINE__);
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const uint8_t bit_type = column_bit_type_[column_index];
      if (bit_type == 8) {
        uint8_t* column_data = reinterpret_cast<uint8_t*>(data_by_column_[column_index]);
        DeallocateCUDAMemory<uint8_t>(&column_data, __FILE__, __LINE__);
        AllocateCUDAMemory<uint8_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 16) {
        uint16_t* column_data = reinterpret_cast<uint16_t*>(data_by_column_[column_index]);
        DeallocateCUDAMemory<uint16_t>(&column_data, __FILE__, __LINE__);
        AllocateCUDAMemory<uint16_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 32) {
        uint32_t* column_data = reinterpret_cast<uint32_t*>(data_by_column_[column_index]);
        DeallocateCUDAMemory<uint32_t>(&column_data, __FILE__, __LINE__);
        AllocateCUDAMemory<uint32_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  DeallocateCUDAMemory<void*>(&cuda_data_by_column_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
}

void CUDAColumnData::InitColumnMetaInfo() {
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_column_bit_type_,
                                       column_bit_type_.data(),
                                       column_bit_type_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_max_bin_,
                                         feature_max_bin_.data(),
                                         feature_max_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_min_bin_,
                                         feature_min_bin_.data(),
                                         feature_min_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_offset_,
                                         feature_offset_.data(),
                                         feature_offset_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_most_freq_bin_,
                                         feature_most_freq_bin_.data(),
                                         feature_most_freq_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_default_bin_,
                                         feature_default_bin_.data(),
                                         feature_default_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_zero_,
                                         feature_missing_is_zero_.data(),
                                         feature_missing_is_zero_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_na_,
                                        feature_missing_is_na_.data(),
                                        feature_missing_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_zero_,
                                        feature_mfb_is_zero_.data(),
                                        feature_mfb_is_zero_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_na_,
                                        feature_mfb_is_na_.data(),
                                        feature_mfb_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_feature_to_column_,
                                    feature_to_column_.data(),
                                    feature_to_column_.size(),
                                    __FILE__,
                                    __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
