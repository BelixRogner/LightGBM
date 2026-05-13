// 256-bin gradient/hessian histogram kernel for Apple Metal.
//
// This file is the canonical MSL source. It is also embedded in
// metal_tree_learner.cpp as a C++ raw string literal — keep the two in sync
// (a regression test verifies this).
//
// Tuning history: NUM_SUBHIST=4 was empirically pareto-optimal on M4 Pro;
// see tools/metal_bench/RESULTS.md.

#include <metal_stdlib>
using namespace metal;

#ifndef NUM_BINS
#define NUM_BINS 256u
#endif
#ifndef NUM_SUBHIST
#define NUM_SUBHIST 4u
#endif

// MSL on the system doesn't accept threadgroup atomic_float, so we mirror the
// LightGBM OpenCL pattern: store float histograms as uint bit-patterns in
// threadgroup memory and atomic-add via compare-exchange CAS loop.
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

// Builds a partial histogram for (feature, wg-in-feature) into partial_hist.
// Grid:        dispatchThreadgroups(MTL::Size(wg_per_feat, num_features, 1)).
// Threadgroup: (256, 1, 1). One feature column per gid.y; one row slice per gid.x.
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

// Sums partial_hist across the wg-per-feature dim into out_hist.
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
