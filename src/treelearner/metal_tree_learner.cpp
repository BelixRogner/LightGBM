/*!
 * Copyright (c) 2026 ExaBoost authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "metal_tree_learner.h"

#ifdef USE_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace LightGBM {

namespace {

constexpr int kThreadsPerGroup = 256;
constexpr int kNumBins         = 256;

// Inlined Metal kernel source. Kept in sync with src/treelearner/metal/histogram256.metal;
// a regression test verifies the two strings match.
const char* kHistogramKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

#ifndef NUM_BINS
#define NUM_BINS 256u
#endif
#ifndef NUM_SUBHIST
#define NUM_SUBHIST 4u
#endif

inline void atomic_tg_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float cur = as_type<float>(expected);
        desired = as_type<uint>(cur + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

kernel void histogram_partial(
    device const uchar*  features      [[ buffer(0) ]],
    device const float*  gradients     [[ buffer(1) ]],
    device const float*  hessians      [[ buffer(2) ]],
    device float*        partial_hist  [[ buffer(3) ]],
    constant uint&       num_data      [[ buffer(4) ]],
    constant uint&       wg_per_feat   [[ buffer(5) ]],
    uint3 tid3      [[ thread_position_in_threadgroup ]],
    uint3 gid       [[ threadgroup_position_in_grid ]],
    uint3 tg_sz3    [[ threads_per_threadgroup ]],
    uint  sg_idx    [[ simdgroup_index_in_threadgroup ]])
{
    uint tid   = tid3.x;
    uint tg_sz = tg_sz3.x;
    threadgroup atomic_uint local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_uint local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_id    = gid.y;
    uint wg_in_feat = gid.x;
    uint per_wg     = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start      = wg_in_feat * per_wg;
    uint end        = min(start + per_wg, num_data);
    device const uchar* feat_col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint i = start + tid; i < end; i += tg_sz) {
        uint bin = (uint)feat_col[i];
        atomic_tg_add_f(&local_grad[sub][bin], gradients[i]);
        atomic_tg_add_f(&local_hess[sub][bin], hessians[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out =
        partial_hist + ((uint64_t)feat_id * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        float g = 0.0f, h = 0.0f;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g += as_type<float>(atomic_load_explicit(&local_grad[s][b], memory_order_relaxed));
            h += as_type<float>(atomic_load_explicit(&local_hess[s][b], memory_order_relaxed));
        }
        out[2 * b + 0] = g;
        out[2 * b + 1] = h;
    }
}

kernel void histogram_reduce(
    device const float*  partial_hist  [[ buffer(0) ]],
    device float*        out_hist      [[ buffer(1) ]],
    constant uint&       wg_per_feat   [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint gid [[ threadgroup_position_in_grid ]])
{
    uint feat_id = gid;
    uint bin     = tid;
    float g = 0.0f, h = 0.0f;
    for (uint w = 0; w < wg_per_feat; ++w) {
        device const float* p =
            partial_hist + ((uint64_t)feat_id * wg_per_feat + w) * NUM_BINS * 2;
        g += p[2 * bin + 0];
        h += p[2 * bin + 1];
    }
    device float* out = out_hist + (uint64_t)feat_id * NUM_BINS * 2;
    out[2 * bin + 0] = g;
    out[2 * bin + 1] = h;
}
)MSL";

}  // namespace

struct MetalTreeLearner::MetalState {
  MTL::Device*               device  = nullptr;
  MTL::CommandQueue*         queue   = nullptr;
  MTL::Library*              library = nullptr;
  MTL::ComputePipelineState* pso_partial = nullptr;
  MTL::ComputePipelineState* pso_reduce  = nullptr;

  // Persistent device-resident buffers (rebuilt on training-data changes).
  MTL::Buffer* feat_buf     = nullptr;  // uchar [num_metal_features * num_data]
  MTL::Buffer* grad_buf     = nullptr;  // float [num_data]
  MTL::Buffer* hess_buf     = nullptr;  // float [num_data]
  MTL::Buffer* partial_buf  = nullptr;  // float [num_metal_features * wg_per_feat * 2*NUM_BINS]
  MTL::Buffer* out_buf      = nullptr;  // float [num_metal_features * 2*NUM_BINS]

  int num_metal_features = 0;
  int num_data           = 0;
  int wg_per_feat        = 1;

  ~MetalState() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(feat_buf); rel(grad_buf); rel(hess_buf);
    rel(partial_buf); rel(out_buf);
    rel(pso_partial); rel(pso_reduce);
    rel(library); rel(queue); rel(device);
  }
};

MetalTreeLearner::MetalTreeLearner(const Config* tree_config)
    : SerialTreeLearner(tree_config),
      state_(std::make_unique<MetalState>()) {}

MetalTreeLearner::~MetalTreeLearner() = default;

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  InitMetal();
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                              bool is_constant_hessian,
                                              bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian,
                                            reset_multi_val_bin);
  // Force a rebuild of the dense feature buffer next time around.
  metal_ready_ = false;
  metal_feature_groups_.clear();
}

void MetalTreeLearner::InitMetal() {
  state_->device = MTL::CreateSystemDefaultDevice();
  if (!state_->device) {
    Log::Warning("Metal: no system default device found. Falling back to CPU path.");
    return;
  }
  Log::Info("Metal device: %s", state_->device->name()->utf8String());

  NS::Error* err = nullptr;
  auto src = NS::String::string(kHistogramKernelSrc, NS::UTF8StringEncoding);
  auto opts = MTL::CompileOptions::alloc()->init();
  state_->library = state_->device->newLibrary(src, opts, &err);
  opts->release();
  if (!state_->library) {
    Log::Warning("Metal: MSL compile failed: %s. Falling back to CPU path.",
                 err ? err->localizedDescription()->utf8String() : "(unknown)");
    return;
  }

  auto make_pso = [&](const char* name) -> MTL::ComputePipelineState* {
    auto nm = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* fn = state_->library->newFunction(nm);
    if (!fn) { Log::Warning("Metal: missing kernel function %s", name); return nullptr; }
    NS::Error* e = nullptr;
    MTL::ComputePipelineState* p = state_->device->newComputePipelineState(fn, &e);
    fn->release();
    if (!p) {
      Log::Warning("Metal: pipeline state creation failed (%s): %s", name,
                   e ? e->localizedDescription()->utf8String() : "(unknown)");
    }
    return p;
  };
  state_->pso_partial = make_pso("histogram_partial");
  state_->pso_reduce  = make_pso("histogram_reduce");
  if (!state_->pso_partial || !state_->pso_reduce) {
    Log::Warning("Metal: pipeline setup incomplete. Falling back to CPU path.");
    return;
  }

  state_->queue = state_->device->newCommandQueue();
  metal_ready_ = true;
  Log::Info("Metal: backend initialized (kernels compiled, queue ready). "
            "Histogram-acceleration wiring follows in subsequent commits.");
}

void MetalTreeLearner::TeardownMetal() {
  state_.reset();
  metal_ready_ = false;
}

bool MetalTreeLearner::BuildDenseFeatureBuffer() {
  // Phase 2.0 stub: returns false so ConstructHistograms always delegates to CPU.
  // Phase 2.1 will scan train_data_->FeatureGroup(i), select dense single-binsize
  // groups, materialize a packed uchar buffer in state_->feat_buf, and return true.
  return false;
}

void MetalTreeLearner::RunMetalHistogram(const score_t* /*gradients*/,
                                         const score_t* /*hessians*/,
                                         data_size_t /*num_data*/) {
  // Phase 2.0 stub. The dispatch is exercised in tools/metal_bench/; integration
  // follows in subsequent commits.
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  // Phase 2.0: route to the CPU implementation while the Metal pipeline is
  // being wired. This keeps device_type=metal numerically identical to CPU.
  SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
}

}  // namespace LightGBM

#endif  // USE_METAL
