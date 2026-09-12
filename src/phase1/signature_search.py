"""
signature_search.py — Sequential (single-core) greedy forward gene selection.

This is the REFERENCE implementation.  The parallel (parallel_search.py) and
GPU (gpu_search.py) versions compute identical results using different hardware.

Algorithm — Greedy Forward Selection
--------------------------------------
  Intuition: build the signature one gene at a time.
  At each step, try adding every remaining candidate gene and keep whichever
  addition produces the best AUC score.

  Step 0 (empty signature):
      Score each gene individually → pick the best → add to signature.

  Step 1 (1 gene in signature):
      Score each remaining gene *given* that the first gene is already selected
      → pick the best → add.

  Step 2, 3, … up to MAX_SIGNATURE_SIZE.

  This is "greedy" because at each step we make the locally best decision.
  It is not guaranteed to find the globally optimal combination, but it is
  fast, interpretable, and works well in practice for gene selection.

Scoring function
----------------
  At each step k, the current signature has been reduced to a single
  "projection score" per sample:

      proj[i] = sum of expression values of selected genes for sample i

  For each candidate gene c, the combined score is:

      combined[i] = proj[i] + expression_of_c[i]

  We compute AUC of combined vs. the class labels.
  The candidate gene with highest AUC gets added to the signature.

  This is equivalent to a step of linear discriminant analysis.

Final evaluation
----------------
  The greedy search uses the fast AUC approximation above.
  After each gene is added, we ALSO train a proper Logistic Regression
  on the train set and evaluate on the held-out test set.
  This gives the 'val_auc' column in the results — the real performance metric.
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MAX_SIGNATURE_SIZE
from src.utils  import auc_from_scores, final_lr_auc


# ─── Public API ────────────────────────────────────────────────────────────────

def run_sequential(
    X_train           : np.ndarray,
    y_train           : np.ndarray,
    X_test            : np.ndarray,
    y_test            : np.ndarray,
    candidate_indices : List[int],
    max_genes         : int = MAX_SIGNATURE_SIZE,
    verbose           : bool = True,
) -> pd.DataFrame:
    """
    Run sequential greedy forward selection (single CPU core).

    Parameters
    ----------
    X_train, y_train   : training expression matrix and labels
    X_test,  y_test    : held-out test data for validation
    candidate_indices  : column indices of the candidate genes in X_train/X_test
    max_genes          : stop after selecting this many genes
    verbose            : print a line after each step

    Returns
    -------
    pd.DataFrame with columns:
        step        — which round of selection (1, 2, 3, …)
        gene_index  — column index of the selected gene
        search_auc  — AUC used during the search (training data only)
        val_auc     — held-out test AUC from a Logistic Regression on the signature so far
        elapsed_s   — cumulative wall-clock time in seconds
    """
    remaining          = list(candidate_indices)
    selected           = []
    current_proj_train = np.zeros(len(y_train), dtype=np.float32)
    current_proj_test  = np.zeros(len(y_test),  dtype=np.float32)
    records            = []
    t0                 = time.perf_counter()

    for step in range(min(max_genes, len(remaining))):

        # ── Inner loop: score every remaining candidate gene ─────────────
        best_auc  = -1.0
        best_gene = None

        for gene_idx in remaining:
            combined = current_proj_train + X_train[:, gene_idx]
            score    = auc_from_scores(combined, y_train)
            if score > best_auc:
                best_auc  = score
                best_gene = gene_idx

        # ── Add the best gene to the signature ────────────────────────────
        selected.append(best_gene)
        remaining.remove(best_gene)
        current_proj_train += X_train[:, best_gene]
        current_proj_test  += X_test[:, best_gene]

        # ── Official evaluation: Logistic Regression on the test set ──────
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
            "elapsed_s" : round(elapsed, 3),
        })

        if verbose:
            print(f"  Step {step+1:3d}  gene={best_gene:6d}  "
                  f"search_AUC={best_auc:.4f}  val_AUC={val_auc:.4f}  "
                  f"t={elapsed:.1f}s")

    return pd.DataFrame(records)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader       import get_data
    from src.preprocessing     import preprocess
    from src.feature_selection import rank_genes

    print("=== Sequential greedy forward selection ===\n")
    expr_df, meta_df = get_data()
    prep = preprocess(expr_df, meta_df)
    _, cand_idx, _   = rank_genes(prep["X_train"], prep["y_train"], prep["gene_ids"])

    results = run_sequential(
        prep["X_train"], prep["y_train"],
        prep["X_test"],  prep["y_test"],
        cand_idx, max_genes=15,
    )
    best_row = results.loc[results["val_auc"].idxmax()]
    print(f"\nBest val AUC = {best_row['val_auc']:.4f} at step {int(best_row['step'])}")
