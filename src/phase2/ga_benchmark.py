# -*- coding: utf-8 -*-
"""
ga_benchmark.py -- HPC benchmark for the Genetic Algorithm fitness evaluation.

Purpose
-------
The most computationally expensive part of any GA is the FITNESS step:
scoring all 100 individuals every generation. Since each individual's score
is independent of the others, this step is "embarrassingly parallel."

We time three implementations of the exact same fitness evaluation:
  1. Sequential    -- pure Python loop over 100 individuals (1 CPU core)
  2. Parallel CPU  -- joblib distributes 100 individuals across all CPU cores
  3. GPU           -- ONE batched tensor operation scores all 100 at once

This is intentionally comparable to the Phase 1 benchmark (benchmark.py),
but with a more meaningful workload per task:
  - Phase 1:  score one gene against a projection  (~286 additions per task)
  - Phase 2:  score one 20-gene signature against all patients
              (20 gene columns x 228 samples = 4,560 additions per task)

This larger workload gives the CPU parallel version a MUCH better chance
to outperform sequential -- because the "work per task" is now big enough
to justify the thread-launch overhead.

Expected Result (on i7-13650HX + RTX 4060):
  - Sequential:   some baseline time
  - Parallel CPU: likely 2-5x faster (unlike Phase 1 where it was slower!)
  - GPU:          likely 5-15x faster (matrix multiply on 3,072 CUDA cores)

This contrast tells a rich story: "HPC strategy must match workload size."
"""

import time
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RESULTS_DIR
from src.phase2.genetic_search import (
    GA_POPULATION_SIZE,
    GA_SIGNATURE_SIZE,
    TORCH_AVAILABLE,
    _GPU_NAME,
    _fitness_sequential,
    _fitness_parallel,
    _fitness_gpu,
)


def run_ga_benchmark(X_train: np.ndarray,
                     y_train: np.ndarray,
                     n_repeats: int = 5) -> pd.DataFrame:
    """
    Time the three HPC fitness-evaluation strategies on real data.

    Parameters
    ----------
    X_train   : real training expression matrix (228 samples x 500 genes)
    y_train   : real training labels
    n_repeats : number of timing runs to average (reduces noise)

    Returns
    -------
    DataFrame with columns: mode, mean_s, std_s, speedup
    Saved to results/ga_benchmark_results.csv
    """
    rng  = np.random.default_rng(RANDOM_STATE)
    n_cands = X_train.shape[1]

    # Build a fixed random population for fair comparison
    population = np.array([
        rng.choice(n_cands, size=GA_SIGNATURE_SIZE, replace=False)
        for _ in range(GA_POPULATION_SIZE)
    ])

    # Define which modes to benchmark
    modes = [
        ("Sequential (1-core)",  lambda: _fitness_sequential(population, X_train, y_train)),
        ("Parallel CPU (all cores)", lambda: _fitness_parallel(population, X_train, y_train, n_jobs=-1)),
    ]
    if TORCH_AVAILABLE:
        modes.append((f"GPU ({_GPU_NAME})", lambda: _fitness_gpu(population, X_train, y_train)))

    print(f"\n  Benchmarking GA fitness evaluation")
    print(f"  Population: {GA_POPULATION_SIZE} individuals x {GA_SIGNATURE_SIZE} genes")
    print(f"  Data:       {X_train.shape[0]} samples x {X_train.shape[1]} candidates")
    print(f"  Repeats:    {n_repeats}\n")

    records = []
    for label, fn in modes:
        # Warm-up run (avoids JIT / caching effects)
        fn()
        times = []
        for _ in range(n_repeats):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)

        mean_t = float(np.mean(times))
        std_t  = float(np.std(times))
        records.append({"mode": label, "mean_s": round(mean_t, 6), "std_s": round(std_t, 6)})
        print(f"  {label:<35}: {mean_t:.4f}s +/- {std_t:.4f}s")

    df   = pd.DataFrame(records)
    base = df.loc[0, "mean_s"]                    # sequential is always first
    df["speedup"] = (base / df["mean_s"]).round(3)

    print(f"\n  Speedups vs sequential:")
    for _, row in df.iterrows():
        print(f"    {row['mode']:<35}: {row['speedup']:.2f}x")

    out_path = RESULTS_DIR / "ga_benchmark_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.name}")
    return df
