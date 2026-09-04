# -*- coding: utf-8 -*-
"""
run_pipeline.py -- Pipeline runner for all phases.

Usage
-----
    # Full Phase 1 run (downloads data, greedy search + benchmark):
    python run_pipeline.py

    # Quick run (fewer genes, skip benchmark -- good for testing):
    python run_pipeline.py --max-genes 15 --skip-benchmark

    # Force re-download even if local cache exists:
    python run_pipeline.py --force-download

Phase 1 Steps
-------------
    [1] Download & parse GSE2034 gene expression dataset
    [2] Preprocess (log-transform, filter, train/test split)
    [3] Rank genes by differential expression (statistical filter)
    [4] Sequential greedy search  <-- reference single-core (src/phase1/)
    [5] GPU greedy search         <-- HPC implementation   (src/phase1/)
    [6] HPC benchmark             <-- scaling study        (src/phase1/)
    [7] Save all results to results/

After this script finishes, run:
    python notebooks/make_figures.py
to generate the slide-ready PNG figures.
"""

import argparse
import os
import sys
import time
from pathlib import Path


import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config            import RESULTS_DIR, MAX_SIGNATURE_SIZE
from src.data_loader       import get_data
from src.preprocessing     import preprocess
from src.feature_selection import rank_genes
from src.phase1.signature_search  import run_sequential
from src.phase1.gpu_search        import run_gpu, TORCH_AVAILABLE
from src.phase1.benchmark         import run_benchmark
from src.phase2.genetic_search    import run_ga, GA_SIGNATURE_SIZE
from src.phase2.ga_benchmark      import run_ga_benchmark


def _banner(text: str) -> None:
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  Minimal Gene Signature Discovery -- Phase 1")
    print("=" * 60)
    t_total = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Data
    _banner("[1/6] Loading dataset")
    expr_df, meta_df = get_data(force_download=args.force_download)

    # [2] Preprocessing
    _banner("[2/6] Preprocessing")
    prep = preprocess(expr_df, meta_df)

    # Save dataset stats for figure generation
    pd.Series(prep["stats"]).to_csv(RESULTS_DIR / "dataset_stats.csv", header=["value"])

    # [3] Gene ranking
    _banner("[3/6] Ranking genes by differential expression")
    cand_ids, cand_idx, ranking_df = rank_genes(
        prep["X_train"], prep["y_train"], prep["gene_ids"]
    )
    ranking_df.to_csv(RESULTS_DIR / "gene_ranking.csv", index=False)
    print(f"  Saved: gene_ranking.csv  ({len(ranking_df)} genes ranked)")

    # [4] Sequential greedy search
    _banner(f"[4/6] Sequential greedy forward selection (up to {args.max_genes} genes)")
    seq_results = run_sequential(
        prep["X_train"], prep["y_train"],
        prep["X_test"],  prep["y_test"],
        cand_idx, max_genes=args.max_genes,
    )
    # Add readable gene names
    id_map = {i: gid for i, gid in enumerate(prep["gene_ids"])}
    seq_results["gene_id"] = seq_results["gene_index"].map(id_map)
    seq_results.to_csv(RESULTS_DIR / "sequential_search_results.csv", index=False)
    print(f"  Saved: sequential_search_results.csv")

    # [5] GPU greedy search
    _banner("[5/6] GPU greedy search")
    if TORCH_AVAILABLE:
        try:
            gpu_results = run_gpu(
                prep["X_train"], prep["y_train"],
                prep["X_test"],  prep["y_test"],
                cand_idx, max_genes=args.max_genes,
            )
            gpu_results["gene_id"] = gpu_results["gene_index"].map(id_map)
            gpu_results.to_csv(RESULTS_DIR / "gpu_search_results.csv", index=False)
            print(f"  Saved: gpu_search_results.csv")
        except Exception as exc:
            print(f"  GPU search failed: {exc}")
    else:
        print("  PyTorch not found -- GPU search skipped.")
        print("  Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # [6] Phase 1 HPC benchmark
    if args.skip_benchmark:
        _banner("[6/7] Phase 1 HPC benchmark SKIPPED  (--skip-benchmark flag)")
    else:
        _banner("[6/7] Phase 1 HPC benchmark -- greedy scoring scaling study")
        run_benchmark()

    # [7] Phase 2 -- Genetic Algorithm
    _banner(f"[7/7] Phase 2: Genetic Algorithm (sequential mode, {GA_SIGNATURE_SIZE} genes)")
    ga_results = run_ga(
        prep["X_train"], prep["y_train"],
        prep["X_test"],  prep["y_test"],
        cand_idx,
        mode="sequential",
    )
    ga_results.to_csv(RESULTS_DIR / "ga_results.csv", index=False)
    print(f"  Saved: ga_results.csv")

    if not args.skip_benchmark:
        _banner("  Phase 2: GA HPC benchmark (sequential vs parallel CPU vs GPU)")
        # Slice X_train to the 500-candidate columns -- same view the GA uses internally
        X_cands_train = prep["X_train"][:, cand_idx]
        ga_bench = run_ga_benchmark(X_cands_train, prep["y_train"])
        ga_bench.to_csv(RESULTS_DIR / "ga_benchmark_results.csv", index=False)

    # Summary
    best_auc_greedy = seq_results["val_auc"].max()
    best_n_greedy   = int(seq_results.loc[seq_results["val_auc"].idxmax(), "step"])
    best_auc_ga     = ga_results["best_val_auc"].max()
    total           = time.perf_counter() - t_total

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Dataset         : {prep['stats']['n_samples']} samples x "
          f"{prep['stats']['n_probes_raw']:,} probes")
    print(f"  Candidates      : {len(cand_idx)} genes")
    print(f"  Greedy best AUC : {best_auc_greedy:.4f}  ({best_n_greedy} genes)")
    print(f"  GA best AUC     : {best_auc_ga:.4f}  ({GA_SIGNATURE_SIZE} genes)")
    print(f"  Total time      : {total:.1f}s")
    print(f"\n  Outputs -> results/")
    print(f"\nNext: python notebooks/make_figures.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Gene Signature Discovery Pipeline")
    parser.add_argument(
        "--max-genes", type=int, default=MAX_SIGNATURE_SIZE,
        help=f"Maximum signature size to explore (default: {MAX_SIGNATURE_SIZE})",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip the HPC benchmark (much faster for quick runs)",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download the dataset even if local files already exist",
    )
    main(parser.parse_args())
