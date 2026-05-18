# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import platform

import numpy as np
import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("items_per_query", [50, 1500])
def test_cuda_lambdarank_deterministic_is_bit_identical_run_to_run(items_per_query):
    """Regression test for LambdaRank gradient kernel run-to-run determinism.

    The non-deterministic kernel scatters per-pair gradient and hessian
    contributions into shared memory via atomicAdd_block. Different runs
    interleave those atomics in different orders, and floating-point
    addition is non-associative, so the per-slot sums (and hence the
    boosted tree) are not bit-identical across runs.

    With deterministic=True the kernel switches to a CUB BlockReduce-based
    code path that accumulates each output slot in a fixed (i, j) order
    without atomics. Output should then be bit-identical across runs.

    Two parametrized values cover both code paths:
      - items_per_query=50  -> BitonicArgSort_1024 path
      - items_per_query=1500 -> BitonicArgSort_2048 path
    """
    rng = np.random.default_rng(7)
    n_queries = max(2, 600 // items_per_query)
    n = n_queries * items_per_query
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    coef = rng.standard_normal(n_features)
    score = X @ coef
    quantiles = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(score, quantiles).astype(np.int32)
    group = np.full(n_queries, items_per_query, dtype=np.int32)

    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "device_type": "cuda",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 7,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 8,
        "learning_rate": 0.1,
        "min_data_in_leaf": 3,
    }

    preds = []
    for _ in range(3):
        ds = lgb.Dataset(X, label=y, group=group, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train(base, ds, num_boost_round=5)
        preds.append(bst.predict(X, raw_score=True))

    # Bit-identical across all three runs.
    for i, p in enumerate(preds[1:], start=1):
        assert np.array_equal(p, preds[0]), (
            f"deterministic=True LambdaRank produced non-identical output between run 0 and run {i} "
            f"(max|Δ|={float(np.abs(p - preds[0]).max()):.3e}); "
            f"items_per_query={items_per_query}."
        )


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {"verbosity": -1, "num_leaves": 31, "objective": "binary", "device": "cpu"}
    cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
    cpu_score = log_loss(y, cpu_bst.predict(X))

    params_gpu = params_cpu.copy()
    params_gpu["device"] = "gpu"
    # Double-precision floats are only supported on x86_64 with PoCL
    params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
    assert cpu_score == pytest.approx(gpu_score, rel=rel)
    assert gpu_score < 0.242
