# coding: utf-8
"""Tests for the Metal backend (Apple silicon).

These tests train the same data on both CPU and Metal and assert that the
resulting models agree on the standard headline metrics (AUC, accuracy, F1,
log-loss). Tolerances are deliberately loose-but-not-trivial: Metal histogram
construction uses non-deterministic atomic-add ordering, which gives ULP-level
drift in histograms and slight differences in tree splits.

Skipped unless LIGHTGBM_TEST_METAL=1 is set in the environment (and only runs
on darwin/arm64 where a Metal-enabled build is realistic).
"""

import os
import platform

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb


_skip_reason = (
    "Set LIGHTGBM_TEST_METAL=1 and build with -DUSE_METAL=ON on macOS/arm64 to run."
)
_should_run = (
    os.environ.get("LIGHTGBM_TEST_METAL") == "1"
    and platform.system() == "Darwin"
    and platform.machine() == "arm64"
)
pytestmark = pytest.mark.skipif(not _should_run, reason=_skip_reason)


def _train_both(params_base, X_train, y_train, X_test, y_test, num_rounds=50):
    """Train identical models on cpu and metal devices; return (cpu_pred, metal_pred)."""
    rng_seed = 42
    common = dict(params_base, verbosity=-1, deterministic=True, seed=rng_seed)

    cpu_params = dict(common, device_type="cpu")
    cpu_ds = lgb.Dataset(X_train, y_train)
    cpu_bst = lgb.train(cpu_params, cpu_ds, num_boost_round=num_rounds)

    metal_params = dict(common, device_type="metal")
    metal_ds = lgb.Dataset(X_train, y_train)
    metal_bst = lgb.train(metal_params, metal_ds, num_boost_round=num_rounds)

    return cpu_bst.predict(X_test), metal_bst.predict(X_test)


def test_binary_classification_parity():
    """Binary task: AUC, accuracy, F1, log-loss agree between cpu and metal."""
    X, y = make_classification(
        n_samples=4_000, n_features=64, n_informative=24, n_redundant=8,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )

    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    cpu_lbl = (cpu_pred > 0.5).astype(int)
    metal_lbl = (metal_pred > 0.5).astype(int)

    cpu_acc = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1 = f1_score(y_test, cpu_lbl)
    metal_f1 = f1_score(y_test, metal_lbl)

    # Headline metrics should agree to ~1% relative or 0.01 absolute, whichever
    # is looser. Bumps to 2% / 0.02 if the Metal histograms produce different
    # but still high-quality splits.
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_acc == pytest.approx(cpu_acc, abs=0.02), (cpu_acc, metal_acc)
    assert metal_f1 == pytest.approx(cpu_f1, abs=0.02), (cpu_f1, metal_f1)
    # Both should be meaningful (not degenerate).
    assert metal_auc > 0.7
    assert metal_acc > 0.6


def test_multiclass_classification_parity():
    """Multiclass: macro-F1 agrees between cpu and metal."""
    X, y = make_classification(
        n_samples=3_000, n_features=48, n_informative=20, n_classes=4,
        n_clusters_per_class=2, random_state=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    cpu_pred_proba, metal_pred_proba = _train_both(
        {"objective": "multiclass", "num_class": 4, "num_leaves": 15,
         "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )
    cpu_lbl = cpu_pred_proba.argmax(axis=1)
    metal_lbl = metal_pred_proba.argmax(axis=1)

    cpu_acc = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1 = f1_score(y_test, cpu_lbl, average="macro")
    metal_f1 = f1_score(y_test, metal_lbl, average="macro")

    assert metal_acc == pytest.approx(cpu_acc, abs=0.03), (cpu_acc, metal_acc)
    assert metal_f1 == pytest.approx(cpu_f1, abs=0.03), (cpu_f1, metal_f1)


def test_regression_parity():
    """Regression: MSE agrees between cpu and metal."""
    X, y = make_regression(
        n_samples=3_000, n_features=32, n_informative=16, noise=0.5,
        random_state=2,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2
    )

    cpu_pred, metal_pred = _train_both(
        {"objective": "regression", "num_leaves": 31, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )

    cpu_mse = mean_squared_error(y_test, cpu_pred)
    metal_mse = mean_squared_error(y_test, metal_pred)
    # MSEs can drift more than per-prediction metrics; tolerate 5% relative.
    assert metal_mse == pytest.approx(cpu_mse, rel=0.05), (cpu_mse, metal_mse)


def test_categorical_features_parity():
    """Mix continuous and explicit categorical features. LightGBM has a
    separate categorical histogram path; verify Metal acceleration doesn't
    silently mishandle these (or falls back cleanly)."""
    rng = np.random.default_rng(7)
    n = 4_000
    # 32 continuous + 16 categoricals (each with 8 levels).
    cont = rng.normal(size=(n, 32))
    cat = rng.integers(low=0, high=8, size=(n, 16)).astype(float)
    X = np.concatenate([cont, cat], axis=1)
    coef = rng.normal(size=X.shape[1])
    y = (X @ coef + 0.5 * rng.normal(size=n) > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    cat_feature_idx = list(range(32, 48))  # which columns are categorical
    base = dict(
        objective="binary", num_leaves=31, learning_rate=0.1,
        verbosity=-1, deterministic=True, seed=42,
    )

    cpu_ds = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_idx)
    cpu_bst = lgb.train(dict(base, device_type="cpu"), cpu_ds, num_boost_round=40)
    metal_ds = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_idx)
    metal_bst = lgb.train(dict(base, device_type="metal"), metal_ds, num_boost_round=40)

    cpu_pred = cpu_bst.predict(X_test)
    metal_pred = metal_bst.predict(X_test)

    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.03), (cpu_auc, metal_auc)
    assert metal_auc > 0.6


def test_bagging_parity():
    """bagging_fraction != 1.0 subsamples rows each iteration; the Metal
    indexed-kernel path must handle this. Tight tolerance because bagging is
    a common LightGBM use case and silent drift here would hurt real users."""
    X, y = make_classification(
        n_samples=5_000, n_features=64, n_informative=20, random_state=11,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=11
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "bagging_fraction": 0.5, "bagging_freq": 1},
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_feature_fraction_parity():
    """feature_fraction < 1 randomly disables features per-tree. Metal must
    correctly honor is_feature_used in the per-feature write-back."""
    X, y = make_classification(
        n_samples=5_000, n_features=96, n_informative=24, random_state=12,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "feature_fraction": 0.5},
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_64bin_kernel_parity():
    """max_bin=60 forces the 64-bin Metal kernel variant. Verifies that
    pipeline state separate from the default 256-bin variant produces
    matching results."""
    X, y = make_classification(
        n_samples=4_000, n_features=64, n_informative=20, random_state=8,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "max_bin": 60},
        X_train, y_train, X_test, y_test,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_binary_classification_multifeature_group_parity():
    """Force LightGBM to pack features into multi-feature groups (low max_bin)
    and verify cpu vs metal still match. Exercises the multi-feature-group
    write-back path added in Phase 2.5."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, n_redundant=8,
        random_state=4,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=4
    )

    # max_bin=15 -> each feature has <=16 bins -> LightGBM packs 8 features
    # per uint32 feature group.
    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "max_bin": 15,
    }
    cpu_pred, metal_pred = _train_both(params, X_train, y_train, X_test, y_test)

    cpu_auc   = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    cpu_lbl   = (cpu_pred   > 0.5).astype(int)
    metal_lbl = (metal_pred > 0.5).astype(int)
    cpu_acc   = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1    = f1_score(y_test, cpu_lbl)
    metal_f1  = f1_score(y_test, metal_lbl)

    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_acc == pytest.approx(cpu_acc, abs=0.02), (cpu_acc, metal_acc)
    assert metal_f1  == pytest.approx(cpu_f1,  abs=0.02), (cpu_f1,  metal_f1)
    assert metal_auc > 0.7


def test_metal_init_smoke():
    """Smoke test: Metal device initializes and produces a non-degenerate model."""
    X, y = make_classification(n_samples=500, n_features=16, random_state=3)
    params = {
        "objective": "binary", "num_leaves": 7, "learning_rate": 0.1,
        "verbosity": -1, "deterministic": True, "device_type": "metal",
    }
    ds = lgb.Dataset(X, y)
    bst = lgb.train(params, ds, num_boost_round=10)
    pred = bst.predict(X)
    assert np.all(np.isfinite(pred))
    assert 0.0 <= pred.min() and pred.max() <= 1.0
    assert roc_auc_score(y, pred) > 0.6
