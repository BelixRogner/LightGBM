# Metal benchmarking tools

Two benches live here:

## `main.cpp` — standalone histogram kernel benchmark

Builds a self-contained Metal-vs-CPU comparison of the histogram kernel
on synthetic data. Useful for kernel-level perf work — isolates the
GPU kernel from tree-building overhead.

```bash
make
./main 1000000 64 20   # num_data, num_features, iters
./main --help
```

Reports per-iteration time for CPU (OpenMP) and Metal, plus correctness
diff between the two outputs.

## `train_bench.py` — end-to-end training benchmark

Trains real LightGBM models on CPU and Metal across several dataset
shapes, reports wall-clock times and AUCs. Run from the repo root:

```bash
python tools/metal_bench/train_bench.py
```

Pair with `LIGHTGBM_METAL_TIMING=1` for per-call dispatch / GPU /
write-back timing:

```bash
LIGHTGBM_METAL_TIMING=1 python tools/metal_bench/train_bench.py
```

## Tuning knobs

Pass via environment variables; documented in the [main README](../../README.md).

- `LIGHTGBM_METAL_MIN_FEATURES=N`
- `LIGHTGBM_METAL_WG_PER_FEAT=N`
- `LIGHTGBM_METAL_MIN_LEAF_ROWS=N`
- `LIGHTGBM_METAL_TIMING=1`
