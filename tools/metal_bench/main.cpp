// Phase 1 benchmark: gradient/hessian histogram construction.
// Compares Metal (Apple GPU) vs OpenMP CPU baseline. Single-file build.
//
// This mirrors the *essential* work of LightGBM's OpenCL histogram256 kernel
// without the full uchar4 packing / workgroup-per-feature tiling, so the
// per-iteration numbers here are a directional signal for the go/no-go gate,
// not a perf claim for the integrated path.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kNumBins = 256;
constexpr int kThreadsPerGroup = 256;

// Tuned kernel: per-SIMD-group sub-histograms + workgroups-per-feature tiling.
// - SUBHIST: N copies of the histogram in threadgroup memory, one per SIMD
//   group, cuts atomic contention N-fold. Final reduction sums the copies.
// - WG_PER_FEAT: launch multiple threadgroups per feature, each handling a
//   row-stride. Output goes to per-WG slot in partial_hist; a second kernel
//   sums across the WG dimension. Fixes GPU underfill on narrow datasets.
const char* kMetalKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint NUM_BINS    = 256;
#ifndef NUM_SUBHIST
#define NUM_SUBHIST 8u   // simdgroups_per_threadgroup at 256 threads
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

// Builds a partial histogram for (feature, wg-in-feature) into partial_hist.
// Grid layout: dispatchThreadgroups(MTL::Size(wg_per_feat, num_features, 1)).
// Threadgroup size: (256, 1, 1).
kernel void histogram_partial(
    device const uchar*  features      [[ buffer(0) ]],  // [num_features * num_data]
    device const float*  gradients     [[ buffer(1) ]],  // [num_data]
    device const float*  hessians      [[ buffer(2) ]],  // [num_data]
    device float*        partial_hist  [[ buffer(3) ]],  // [num_features * wg_per_feat * NUM_BINS * 2]
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

    // Zero all sub-histograms.
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint feat_id     = gid.y;
    uint wg_in_feat  = gid.x;

    // Row range for this WG. Each WG processes a contiguous slice.
    uint per_wg = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start  = wg_in_feat * per_wg;
    uint end    = min(start + per_wg, num_data);

    device const uchar* feat_col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;  // map each SIMD group to its own sub-histogram
    for (uint i = start + tid; i < end; i += tg_sz) {
        uint bin = (uint)feat_col[i];
        atomic_tg_add_f(&local_grad[sub][bin], gradients[i]);
        atomic_tg_add_f(&local_hess[sub][bin], hessians[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sub-histograms and write out.
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

// Sums partial_hist across the wg-in-feature dimension into out_hist.
// Grid: dispatchThreadgroups(MTL::Size(num_features, 1, 1)); tg_size NUM_BINS.
kernel void histogram_reduce(
    device const float*  partial_hist  [[ buffer(0) ]],  // [num_features * wg_per_feat * NUM_BINS * 2]
    device float*        out_hist      [[ buffer(1) ]],  // [num_features * NUM_BINS * 2]
    constant uint&       wg_per_feat   [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint gid [[ threadgroup_position_in_grid ]])
{
    uint feat_id = gid;
    uint bin     = tid;
    float g = 0.0f, h = 0.0f;
    for (uint w = 0; w < wg_per_feat; ++w) {
        device const float* p = partial_hist + ((uint64_t)feat_id * wg_per_feat + w) * NUM_BINS * 2;
        g += p[2 * bin + 0];
        h += p[2 * bin + 1];
    }
    device float* out = out_hist + (uint64_t)feat_id * NUM_BINS * 2;
    out[2 * bin + 0] = g;
    out[2 * bin + 1] = h;
}
)MSL";

void cpu_histogram(const uint8_t* features, const float* gradients, const float* hessians,
                   float* out_hist, int num_features, int num_data) {
    std::memset(out_hist, 0, sizeof(float) * 2 * kNumBins * (size_t)num_features);
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < num_features; ++f) {
        float* h = out_hist + (size_t)f * kNumBins * 2;
        const uint8_t* col = features + (size_t)f * num_data;
        for (int i = 0; i < num_data; ++i) {
            uint8_t b = col[i];
            h[2 * b + 0] += gradients[i];
            h[2 * b + 1] += hessians[i];
        }
    }
}

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    }
    return m;
}

double max_rel_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double denom = std::max((double)std::fabs(a[i]), 1e-6);
        m = std::max(m, (double)std::fabs(a[i] - b[i]) / denom);
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::printf("Usage: %s [num_data] [num_features] [iters] [wg_per_feat=auto] [num_subhist=8]\n"
                    "  Defaults: 1M rows, 64 features, 20 iters\n", argv[0]);
        return 0;
    }
    int num_data     = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    int num_features = (argc > 2) ? std::atoi(argv[2]) : 64;
    int iters        = (argc > 3) ? std::atoi(argv[3]) : 20;
    int wg_per_feat  = (argc > 4) ? std::atoi(argv[4]) : 0;  // 0 = auto
    int num_subhist  = (argc > 5) ? std::atoi(argv[5]) : 8;  // sub-hists per threadgroup

    if (wg_per_feat <= 0) {
        // Target ~512 threadgroups so a 20-core M-series GPU stays saturated.
        wg_per_feat = std::max(1, 512 / std::max(num_features, 1));
        wg_per_feat = std::min(wg_per_feat, 32);
    }

    std::printf("Config: num_data=%d, num_features=%d, num_bins=%d, iters=%d, wg_per_feat=%d, num_subhist=%d\n",
                num_data, num_features, kNumBins, iters, wg_per_feat, num_subhist);

    // ---- Synthetic data ----
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> bin_dist(0, kNumBins - 1);
    std::normal_distribution<float> grad_dist(0.0f, 1.0f);

    std::vector<uint8_t> features((size_t)num_features * num_data);
    std::vector<float>   gradients(num_data);
    std::vector<float>   hessians(num_data);
    for (auto& x : features)  x = (uint8_t)bin_dist(rng);
    for (auto& x : gradients) x = grad_dist(rng);
    for (auto& x : hessians)  x = std::fabs(grad_dist(rng)) + 0.1f;

    const size_t hist_elems = (size_t)num_features * kNumBins * 2;
    std::vector<float> cpu_hist(hist_elems);

    // ---- CPU benchmark ----
    {
        // Warmup
        cpu_histogram(features.data(), gradients.data(), hessians.data(),
                      cpu_hist.data(), num_features, num_data);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            cpu_histogram(features.data(), gradients.data(), hessians.data(),
                          cpu_hist.data(), num_features, num_data);
        }
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("CPU    : %8.3f ms/iter (total %.1f ms over %d iters)\n",
                    total_ms / iters, total_ms, iters);
    }

    // ---- Metal setup ----
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device.\n");
        return 1;
    }
    std::printf("Metal device: %s\n", device->name()->utf8String());

    NS::Error* err = nullptr;
    auto src_nsstring = NS::String::string(kMetalKernelSrc, NS::UTF8StringEncoding);
    auto compile_opts = MTL::CompileOptions::alloc()->init();
    {
        auto k_subhist = NS::String::string("NUM_SUBHIST", NS::UTF8StringEncoding);
        auto v_subhist = NS::Number::number((int)num_subhist);
        const NS::Object* values[1] = { v_subhist };
        const NS::Object* keys[1]   = { k_subhist };
        auto macros = NS::Dictionary::dictionary(values, keys, 1);
        compile_opts->setPreprocessorMacros(macros);
    }
    MTL::Library* library = device->newLibrary(src_nsstring, compile_opts, &err);
    compile_opts->release();
    if (!library) {
        std::fprintf(stderr, "MSL compile error: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(null)");
        return 1;
    }

    auto make_pso = [&](const char* name) -> MTL::ComputePipelineState* {
        auto nm = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* fn = library->newFunction(nm);
        if (!fn) { std::fprintf(stderr, "newFunction(%s) failed\n", name); std::exit(1); }
        NS::Error* e = nullptr;
        MTL::ComputePipelineState* p = device->newComputePipelineState(fn, &e);
        fn->release();
        if (!p) {
            std::fprintf(stderr, "PSO(%s) error: %s\n", name,
                         e ? e->localizedDescription()->utf8String() : "(null)");
            std::exit(1);
        }
        return p;
    };
    MTL::ComputePipelineState* pso_partial = make_pso("histogram_partial");
    MTL::ComputePipelineState* pso_reduce  = make_pso("histogram_reduce");

    MTL::CommandQueue* queue = device->newCommandQueue();

    // Shared-storage buffers — zero-copy on Apple silicon.
    MTL::Buffer* feat_buf = device->newBuffer(features.data(),  features.size(),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* grad_buf = device->newBuffer(gradients.data(), gradients.size() * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* hess_buf = device->newBuffer(hessians.data(),  hessians.size() * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* out_buf  = device->newBuffer(hist_elems * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    const size_t partial_elems = hist_elems * (size_t)wg_per_feat;
    MTL::Buffer* part_buf = device->newBuffer(partial_elems * sizeof(float),
                                              MTL::ResourceStorageModePrivate);
    uint32_t num_data_u   = (uint32_t)num_data;
    uint32_t wg_per_feat_u = (uint32_t)wg_per_feat;
    MTL::Buffer* n_buf  = device->newBuffer(&num_data_u,    sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
    MTL::Buffer* wg_buf = device->newBuffer(&wg_per_feat_u, sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);

    auto dispatch_one = [&]() {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        // Partial histograms.
        enc->setComputePipelineState(pso_partial);
        enc->setBuffer(feat_buf, 0, 0);
        enc->setBuffer(grad_buf, 0, 1);
        enc->setBuffer(hess_buf, 0, 2);
        enc->setBuffer(part_buf, 0, 3);
        enc->setBuffer(n_buf,    0, 4);
        enc->setBuffer(wg_buf,   0, 5);
        enc->dispatchThreadgroups(
            MTL::Size::Make(wg_per_feat, num_features, 1),
            MTL::Size::Make(kThreadsPerGroup, 1, 1));
        // Reduce across wg-per-feature dim.
        enc->setComputePipelineState(pso_reduce);
        enc->setBuffer(part_buf, 0, 0);
        enc->setBuffer(out_buf,  0, 1);
        enc->setBuffer(wg_buf,   0, 2);
        enc->dispatchThreadgroups(
            MTL::Size::Make(num_features, 1, 1),
            MTL::Size::Make(kNumBins, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    };

    // Warmup (also triggers shader cache).
    dispatch_one();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dispatch_one();
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Metal  : %8.3f ms/iter (total %.1f ms over %d iters)\n",
                total_ms / iters, total_ms, iters);

    // ---- Correctness ----
    const float* gpu_hist = static_cast<const float*>(out_buf->contents());
    double abs_d = max_abs_diff(cpu_hist.data(), gpu_hist, hist_elems);
    double rel_d = max_rel_diff(cpu_hist.data(), gpu_hist, hist_elems);
    std::printf("Diff   : max_abs=%.6f max_rel=%.6f\n", abs_d, rel_d);

    feat_buf->release(); grad_buf->release(); hess_buf->release();
    out_buf->release();  part_buf->release(); n_buf->release(); wg_buf->release();
    queue->release(); pso_partial->release(); pso_reduce->release();
    library->release(); device->release();
    pool->release();
    return 0;
}
