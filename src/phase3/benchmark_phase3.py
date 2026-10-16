"""
benchmark_phase3.py — Phase 3 HPC Benchmark: C+OpenMP vs Sequential vs Parallel vs GPU
========================================================================================

Purpose
-------
In Phase 1 and Phase 2, we showed:
  - Phase 1: Python's joblib was slower than sequential because overhead dominated.
  - Phase 2: GPU scored 2.69x faster, but CPU parallelism still struggled with Python serialization.

In Phase 3, we test our compiled C + OpenMP kernel against all three previous methods
on the exact same workload:
  Evaluating fitness for 100 individuals (each with 20 genes) across 228 patients.

The 4 contestants:
  1. Sequential Python (1 CPU core)
  2. Parallel CPU (joblib, multi-core Python)
  3. C + OpenMP (compiled native C, OpenMP threads, zero-copy NumPy memory)
  4. GPU (PyTorch CUDA, RTX 4060)

Outputs:
  - results/phase3_hpc_benchmark_results.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import RANDOM_STATE, RESULTS_DIR
from src.phase2.genetic_search import (
    GA_POPULATION_SIZE,
    GA_SIGNATURE_SIZE,
    TORCH_AVAILABLE,
    _GPU_NAME,
    _fitness_sequential,
    _fitness_parallel,
    _fitness_gpu,
    _fitness_c,
)
from src.phase3.cpp_search import get_num_threads


def run_phase3_benchmark(X_train: np.ndarray,
                         y_train: np.ndarray,
                         n_repeats: int = 15) -> pd.DataFrame:
    """
    Time all 4 HPC implementations on real candidate expression data.

    Parameters
    ----------
    X_train   : expression matrix sliced to 500 candidates (228 samples x 500 genes)
    y_train   : training binary labels (228,)
    n_repeats : timing iterations to average (default 15 for statistical stability)

    Returns
    -------
    pd.DataFrame with columns: mode, mean_ms, std_ms, speedup
    Saved to results/phase3_hpc_benchmark_results.csv
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_cands = X_train.shape[1]

    # Generate a fixed population of 100 signatures (each with 20 genes)
    population = np.array([
        rng.choice(n_cands, size=GA_SIGNATURE_SIZE, replace=False)
        for _ in range(GA_POPULATION_SIZE)
    ], dtype=np.int32)

    n_threads = get_num_threads()

    modes = [
        ("Sequential (1-core Python)", lambda: _fitness_sequential(population, X_train, y_train)),
        ("Parallel CPU (joblib all cores)", lambda: _fitness_parallel(population, X_train, y_train, n_jobs=-1)),
        (f"C + OpenMP ({n_threads} threads)", lambda: _fitness_c(population, X_train, y_train)),
    ]
    if TORCH_AVAILABLE:
        modes.append((f"GPU ({_GPU_NAME})", lambda: _fitness_gpu(population, X_train, y_train)))

    print("\n" + "=" * 65)
    print("  Phase 3: 4-Way HPC Benchmark (GA Fitness Evaluation)")
    print("=" * 65)
    print(f"  Workload   : {GA_POPULATION_SIZE} signatures x {GA_SIGNATURE_SIZE} genes")
    print(f"  Data       : {X_train.shape[0]} patients x {X_train.shape[1]} candidate genes")
    print(f"  Repeats    : {n_repeats} per implementation\n")

    records = []
    for label, fn in modes:
        # Warmup run to eliminate cold caches / JIT effects
        fn()

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)

        mean_s = float(np.mean(times))
        std_s  = float(np.std(times))
        mean_ms = mean_s * 1000.0
        std_ms  = std_s * 1000.0

        records.append({
            "mode": label,
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "mean_s": round(mean_s, 6),
            "std_s": round(std_s, 6),
        })
        print(f"  {label:<35}: {mean_ms:7.2f} ms +/- {std_ms:5.2f} ms")

    df = pd.DataFrame(records)
    base_time = df.loc[0, "mean_ms"]   # sequential baseline
    df["speedup"] = (base_time / df["mean_ms"]).round(2)

    print("\n" + "-" * 65)
    print("  Speedup vs Sequential Baseline:")
    for _, row in df.iterrows():
        print(f"    {row['mode']:<35}: {row['speedup']:6.2f}x  ({row['mean_ms']:.2f} ms)")
    print("-" * 65)

    out_csv = RESULTS_DIR / "phase3_hpc_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv.name}\n")
    return df


def run_phase3_scaling_benchmark(X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 sizes: list = [5, 10, 20, 30],
                                 n_repeats: int = 10) -> pd.DataFrame:
    """
    Measure how all 4 HPC implementations scale as signature size increases.

    Evaluates signature sizes in `sizes` (e.g. 5, 10, 20, 30 genes) across:
      1. Sequential (1-core Python)
      2. Parallel CPU (joblib all cores)
      3. C + OpenMP (20 native threads)
      4. GPU CUDA (RTX 4060)

    Saves to results/phase3_scaling_benchmark_results.csv
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_cands = X_train.shape[1]
    n_threads = get_num_threads()

    print("\n" + "=" * 65)
    print("  Phase 3 HPC Scaling Benchmark Across Signature Sizes")
    print("=" * 65)
    print(f"  Sizes tested: {sizes}")
    print(f"  Repeats     : {n_repeats} per measurement\n")

    records = []

    for k in sizes:
        # Generate fixed population of 100 signatures of size k
        population = np.array([
            rng.choice(n_cands, size=k, replace=False)
            for _ in range(GA_POPULATION_SIZE)
        ], dtype=np.int32)

        modes = [
            ("Sequential (1-core Python)", lambda: _fitness_sequential(population, X_train, y_train)),
            ("Parallel CPU (joblib all cores)", lambda: _fitness_parallel(population, X_train, y_train, n_jobs=-1)),
            (f"C + OpenMP ({n_threads} threads)", lambda: _fitness_c(population, X_train, y_train)),
        ]
        if TORCH_AVAILABLE:
            modes.append((f"GPU ({_GPU_NAME})", lambda: _fitness_gpu(population, X_train, y_train)))

        # Baseline time for this size
        base_ms = 1.0

        for idx, (label, fn) in enumerate(modes):
            # Warmup
            fn()
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)

            mean_s = float(np.mean(times))
            std_s = float(np.std(times))
            mean_ms = mean_s * 1000.0
            std_ms = std_s * 1000.0

            if idx == 0:
                base_ms = mean_ms

            speedup = base_ms / mean_ms

            records.append({
                "signature_size": k,
                "engine": label,
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
                "speedup": round(speedup, 2),
            })
            print(f"  Size {k:>2d} | {label:<32}: {mean_ms:6.2f} ms ({speedup:5.2f}x)")

    df_scale = pd.DataFrame(records)
    out_scale_csv = RESULTS_DIR / "phase3_scaling_benchmark_results.csv"
    df_scale.to_csv(out_scale_csv, index=False)
    print(f"\n  Saved scaling benchmark: {out_scale_csv.name}\n")
    return df_scale


if __name__ == "__main__":
    from src.data_loader import get_data
    from src.preprocessing import preprocess
    from src.feature_selection import rank_genes

    print("Loading data for Phase 3 benchmark...")
    df_expr, df_meta = get_data()
    prep = preprocess(df_expr, df_meta)
    cand_ids, cand_idx, ranking_df = rank_genes(
        prep["X_train"], prep["y_train"], prep["gene_ids"]
    )

    X_train_cands = prep["X_train"][:, cand_idx]
    y_train = prep["y_train"]

    run_phase3_benchmark(X_train_cands, y_train)
    run_phase3_scaling_benchmark(X_train_cands, y_train)
