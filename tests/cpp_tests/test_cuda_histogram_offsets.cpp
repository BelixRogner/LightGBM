/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>
#include <LightGBM/train_share_states.h>
#include <LightGBM/utils/log.h>

#include "../../src/treelearner/cuda/cuda_histogram_constructor.hpp"
#include "../../src/treelearner/cuda/cuda_leaf_splits.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace LightGBM {

namespace {

// 12000 rows uniformly drawn from [0, 4096) yield ~3700 distinct values for
// the wide column, which exceeds max_num_bin_per_partition = 3072 under
// gpu_use_dp=true and so triggers NumLargeBinPartition() > 0.
constexpr int kNumRows = 12000;
constexpr int kNumCols = 5;
constexpr int kNumWideUnique = 4096;
constexpr int kNumNarrowUnique = 16;
constexpr int kNumGradQuantBins = 30;
constexpr int32_t kPackedGradHessOne = 0x00010001;  // grad=1, hess=1 packed int16

// Dense kNumRows x kNumCols dataset: one wide column (>3072 distinct values)
// plus four narrow columns. The wide column forces a large-bin partition,
// which forces the global-memory kernel branch (USE_GLOBAL_MEM_BUFFER=true).
DatasetHandle BuildDataset(std::vector<float>* data_out, std::vector<float>* label_out) {
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> wide_dist(0, kNumWideUnique - 1);
  std::uniform_int_distribution<int> narrow_dist(0, kNumNarrowUnique - 1);

  data_out->resize(static_cast<size_t>(kNumRows) * kNumCols);
  label_out->assign(kNumRows, 0.0f);
  for (int r = 0; r < kNumRows; ++r) {
    (*data_out)[r * kNumCols + 0] = static_cast<float>(wide_dist(rng));
    for (int c = 1; c < kNumCols; ++c) {
      (*data_out)[r * kNumCols + c] = static_cast<float>(narrow_dist(rng));
    }
  }

  DatasetHandle dataset_handle = nullptr;
  // min_data_in_bin=1 keeps every distinct value as its own bin.
  const char* params = "max_bin=8192 min_data_in_bin=1 verbose=-1";
  EXPECT_EQ(0, LGBM_DatasetCreateFromMat(
      data_out->data(), C_API_DTYPE_FLOAT32, kNumRows, kNumCols,
      /*is_row_major=*/1, params, /*reference=*/nullptr, &dataset_handle))
      << LGBM_GetLastError();
  EXPECT_EQ(0, LGBM_DatasetSetField(
      dataset_handle, "label", label_out->data(), kNumRows, C_API_DTYPE_FLOAT32))
      << LGBM_GetLastError();
  return dataset_handle;
}

}  // namespace

// Drives CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory directly
// with synthetic packed gradients and verifies the histogram contents.
//
// Each row contributes a packed (grad_int16=1, hess_int16=1) to exactly one
// bin per feature, so summing the high 16 bits (grad) and low 16 bits (hess)
// of every histogram entry must yield kNumRows * kNumFeatures for both.
//
// Without the offset fix, partition 1 reads/writes its scratch buffer at
// global_hist_buffer + partition_column_start instead of
// global_hist_buffer + partition_hist_start. With more than one partition
// these only coincide for partition 0, so partition 1's gradient sums end up
// at an offset inside partition 0's scratch range. Partition 0's writeback
// then reads those (now-corrupt) entries on top of its own and feeds them
// back into hist_in_leaf, while partition 1's writeback reads the same
// scratch slots and writes them to its own correct hist_in_leaf range. The
// net effect is that partition 1's gradient sums are double-counted and the
// invariant breaks. With the fix the totals match exactly.
TEST(CUDAHistogramOffset, DiscretizedDenseGlobalMemoryUsesPartitionHistStart) {
  std::vector<float> data;
  std::vector<float> label;
  DatasetHandle dataset_handle = BuildDataset(&data, &label);
  Dataset* dataset = static_cast<Dataset*>(dataset_handle);

  // CUDA needs row-wise data; force_row_wise so share_state->GetRowWiseData()
  // returns a valid pointer when CUDARowData::Init reads it.
  std::vector<int8_t> is_feature_used(dataset->num_features(), 1);
  std::unique_ptr<TrainingShareStates> share_state(
      dataset->GetShareStates<false, 0>(
          /*gradients=*/nullptr, /*hessians=*/nullptr, is_feature_used,
          /*is_constant_hessian=*/true,
          /*force_col_wise=*/false, /*force_row_wise=*/true,
          kNumGradQuantBins));
  ASSERT_NE(share_state, nullptr);

  // gpu_use_dp=true drops max_num_bin_per_partition to shared_hist_size_/2 = 3072.
  // The wide column has > 3072 bins, so NumLargeBinPartition() > 0 and the
  // kernel will dispatch through USE_GLOBAL_MEM_BUFFER=true.
  CUDAHistogramConstructor constructor(
      dataset, /*num_leaves=*/2, /*num_threads=*/1,
      share_state->feature_hist_offsets(),
      /*min_data_in_leaf=*/1, /*min_sum_hessian_in_leaf=*/0.0,
      /*gpu_device_id=*/0, /*gpu_use_dp=*/true,
      /*use_quantized_grad=*/true, kNumGradQuantBins);
  constructor.Init(dataset, share_state.get());

  ASSERT_GT(constructor.cuda_row_data_internal()->NumLargeBinPartition(), 0)
      << "test scaffolding: large-bin partition wasn't triggered, so the "
         "global-memory kernel won't run";

  // One quantized gradient/hessian per row, packed (grad_int16=1, hess_int16=1).
  std::vector<int32_t> host_grad_hess(kNumRows, kPackedGradHessOne);
  int32_t* cuda_grad_hess = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&cuda_grad_hess, kNumRows * sizeof(int32_t)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(
      cuda_grad_hess, host_grad_hess.data(), kNumRows * sizeof(int32_t),
      cudaMemcpyHostToDevice));

  // data_indices_in_leaf = 0..kNumRows-1 (all rows in the root leaf).
  std::vector<data_size_t> host_indices(kNumRows);
  for (int i = 0; i < kNumRows; ++i) host_indices[i] = i;
  data_size_t* cuda_indices = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&cuda_indices, kNumRows * sizeof(data_size_t)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(
      cuda_indices, host_indices.data(), kNumRows * sizeof(data_size_t),
      cudaMemcpyHostToDevice));

  // Minimal CUDALeafSplitsStruct on device: only the three fields the
  // histogram kernel reads (num_data_in_leaf, data_indices_in_leaf, hist_in_leaf).
  CUDALeafSplitsStruct host_struct = {};
  host_struct.leaf_index = 0;
  host_struct.num_data_in_leaf = kNumRows;
  host_struct.data_indices_in_leaf = cuda_indices;
  host_struct.hist_in_leaf = constructor.cuda_hist_pointer();

  CUDALeafSplitsStruct* cuda_leaf_splits = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&cuda_leaf_splits, sizeof(CUDALeafSplitsStruct)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(
      cuda_leaf_splits, &host_struct, sizeof(CUDALeafSplitsStruct),
      cudaMemcpyHostToDevice));

  // BeforeTrain stores the gradient pointer and zeroes cuda_hist_. Reinterpret
  // int32_t* as score_t* to match BeforeTrain's float-grad signature; the
  // kernel reinterpret_casts it back to int32_t* when use_quantized_grad_=true.
  constructor.BeforeTrain(reinterpret_cast<const score_t*>(cuda_grad_hess), nullptr);
  // num_bits_in_histogram_bins=16 selects the int32-packed (USE_16BIT_HIST=true)
  // kernel branch -- the one with the offset bug.
  constructor.ConstructHistogramForLeaf(
      cuda_leaf_splits, /*cuda_larger_leaf_splits=*/cuda_leaf_splits,
      /*num_data_in_smaller_leaf=*/kNumRows, /*num_data_in_larger_leaf=*/0,
      /*sum_hessians_in_smaller_leaf=*/static_cast<double>(kNumRows),
      /*sum_hessians_in_larger_leaf=*/0.0,
      /*num_bits_in_histogram_bins=*/16);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  // For use_quantized_grad the kernel reinterprets cuda_hist_ as int32_t and
  // writes one packed int32 per bin (high 16 = grad, low 16 = hess). The first
  // num_total_bin int32s of cuda_hist_ contain the root leaf's histogram.
  const int num_total_bin = static_cast<int>(share_state->feature_hist_offsets().back());
  std::vector<int32_t> host_hist(num_total_bin);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(
      host_hist.data(), constructor.cuda_hist(),
      num_total_bin * sizeof(int32_t),
      cudaMemcpyDeviceToHost));

  int total_grad = 0;
  int total_hess = 0;
  for (int b = 0; b < num_total_bin; ++b) {
    const int32_t packed = host_hist[b];
    total_hess += packed & 0xffff;
    total_grad += (packed >> 16) & 0xffff;
  }
  const int expected_total = kNumRows * static_cast<int>(dataset->num_features());
  EXPECT_EQ(total_grad, expected_total)
      << "sum of grad-counts over all bins must equal kNumRows*kNumFeatures = "
      << expected_total
      << ". With the offset bug, partition 1's gradients are double-counted;"
      << " with the fix they appear exactly once.";
  EXPECT_EQ(total_hess, expected_total) << "sum of hess-counts over all bins";

  cudaFree(cuda_grad_hess);
  cudaFree(cuda_indices);
  cudaFree(cuda_leaf_splits);
  LGBM_DatasetFree(dataset_handle);
}

}  // namespace LightGBM

#endif  // USE_CUDA
