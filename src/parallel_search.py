"""
parallel_search.py — CPU-parallel greedy forward gene selection.

Identical algorithm to signature_search.py.

What is parallelised?
---------------------
At each greedy step we need to score every remaining candidate gene.
These scores are INDEPENDENT of each other — the score of gene A does
not affect or depend on the score of gene B.

This is called "embarrassingly parallel": the work can be divided
across CPU cores with no communication needed between cores.

Implementation: joblib.Parallel
--------------------------------
joblib is a Python library for parallel computing.  We distribute the
inner loop (one core per candidate gene) across n_jobs CPU cores.

  prefer="threads"
    We use threads rather than separate processes because:
    - NumPy releases Python's GIL (Global Interpreter Lock) during
      computation, so threads can run truly in parallel.
    - Threads share memory — no need to copy the large expression
      matrix (X_train) to each worker.  This is efficient.

Expected speedup
----------------
  Ideal:  speedup = n_jobs  (doubling cores halves the time)
  Real:   slightly less due to thread scheduling overhead.

  For 500 candidates on 14 cores:  ~10–12× speedup.
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MAX_SIGNATURE_SIZE
from src.utils  import auc_from_scores, final_lr_auc


# ─── Worker function (called once per candidate gene per step) ────────────────

def _score_one_gene(
    gene_idx : int,
    proj     : np.ndarray,   # (n_train,) current signature projection
    X_train  : np.ndarray,   # (n_train, n_genes) full expression matrix
    y_train  : np.ndarray,   # (n_train,) labels
) -> tuple[int, float]:
    """Score one candidate gene by computing AUC of (proj + gene_expression)."""
    combined = proj + X_train[:, gene_idx]
    return gene_idx, auc_from_scores(combined, y_train)


# ─── Public API ────────────────────────────────────────────────────────────────

def run_parallel(
    X_train           : np.ndarray,
    y_train           : np.ndarray,
    X_test            : np.ndarray,
    y_test            : np.ndarray,
    candidate_indices : List[int],
    max_genes         : int  = MAX_SIGNATURE_SIZE,
    n_jobs            : int  = -1,
    verbose           : bool = True,
) -> pd.DataFrame:
    """
    Greedy forward selection with multi-core CPU parallelism.

    Parameters
    ----------
    n_jobs : number of CPU worker threads.
             -1  = use all available logical cores (recommended).
             1   = single thread (same as sequential, for comparison).
             4   = 4 cores, etc.

    Returns
    -------
    Same column structure as signature_search.run_sequential.
    """
    remaining          = list(candidate_indices)
    selected           = []
    current_proj_train = np.zeros(len(y_train), dtype=np.float32)
    current_proj_test  = np.zeros(len(y_test),  dtype=np.float32)
    records            = []
    t0                 = time.perf_counter()

    for step in range(min(max_genes, len(remaining))):

        # ── Parallel scoring: one thread per candidate ────────────────────
        scored = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_one_gene)(idx, current_proj_train, X_train, y_train)
            for idx in remaining
        )
        # scored = [(gene_idx, auc), …] — one entry per remaining candidate
        best_gene, best_auc = max(scored, key=lambda t: t[1])

        selected.append(best_gene)
        remaining.remove(best_gene)
        current_proj_train += X_train[:, best_gene]
        current_proj_test  += X_test[:, best_gene]

        val_auc = final_lr_auc(
            X_train[:, selected], y_train,
            X_test[:, selected],  y_test,
        )

        elapsed = time.perf_counter() - t0
        records.append({
            "step"      : step + 1,
            "gene_index": best_gene,
            "search_auc": round(best_auc, 6),
            "val_auc"   : round(val_auc, 6),
            "n_jobs"    : n_jobs,
            "elapsed_s" : round(elapsed, 3),
        })

        if verbose:
            print(f"  Step {step+1:3d}  gene={best_gene:6d}  "
                  f"search_AUC={best_auc:.4f}  val_AUC={val_auc:.4f}  "
                  f"t={elapsed:.1f}s  (n_jobs={n_jobs})")

    return pd.DataFrame(records)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    from src.data_loader       import get_data
    from src.preprocessing     import preprocess
    from src.feature_selection import rank_genes

    n_cores = multiprocessing.cpu_count()
    print(f"=== Parallel greedy search ({n_cores} logical cores available) ===\n")

    expr_df, meta_df = get_data()
    prep = preprocess(expr_df, meta_df)
    _, cand_idx, _   = rank_genes(prep["X_train"], prep["y_train"], prep["gene_ids"])

    for workers in [1, 4, n_cores]:
        print(f"\n── {workers} worker(s) ──")
        t0 = time.perf_counter()
        results = run_parallel(
            prep["X_train"], prep["y_train"],
            prep["X_test"],  prep["y_test"],
            cand_idx, max_genes=5, n_jobs=workers, verbose=False,
        )
        print(f"  5 steps in {time.perf_counter()-t0:.2f}s  "
              f"val_AUC={results['val_auc'].iloc[-1]:.4f}")
