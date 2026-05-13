"""Quickstart for the ExaBoost Metal backend on Apple silicon.

Trains the same binary-classification model on CPU and Metal, reports the
wall-clock timing of each, and verifies they agree on AUC.

Usage:
    python examples/python-guide/metal_quickstart.py
"""

import platform
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("Metal backend requires macOS arm64 (Apple silicon). Skipping.")
        return

    print("Generating synthetic dataset: 500k rows x 256 features")
    X, y = make_classification(
        n_samples=500_000,
        n_features=256,
        n_informative=64,
        n_redundant=16,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    base_params = dict(
        objective="binary",
        num_leaves=63,
        learning_rate=0.05,
        verbosity=-1,
        deterministic=True,
        seed=42,
    )

    results = {}
    for device in ("cpu", "metal"):
        ds = lgb.Dataset(X_train, y_train)
        t0 = time.perf_counter()
        bst = lgb.train(dict(base_params, device_type=device), ds, num_boost_round=50)
        elapsed = time.perf_counter() - t0
        pred = bst.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        results[device] = {"time": elapsed, "auc": auc}
        print(f"  {device:>5}: {elapsed:6.2f}s  AUC={auc:.6f}")

    cpu_t, gpu_t = results["cpu"]["time"], results["metal"]["time"]
    print(f"\nMetal speedup vs CPU: {cpu_t / gpu_t:.2f}x")
    auc_diff = abs(results["metal"]["auc"] - results["cpu"]["auc"])
    print(f"AUC parity: cpu={results['cpu']['auc']:.6f} metal={results['metal']['auc']:.6f}  diff={auc_diff:.6f}")


if __name__ == "__main__":
    main()
