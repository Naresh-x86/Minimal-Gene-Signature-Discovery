"""
preprocessing.py — Clean and prepare gene-expression data for analysis.

Why each step?
--------------
1. Label filter    Remove samples with unknown outcome (label = -1).
2. Log₂ transform  Microarray values span many orders of magnitude.
                   Log-transforming compresses the range and makes the
                   distribution more symmetric — standard in bioinformatics.
3. Z-score         Centre each gene to mean=0, std=1 across samples so
                   that genes measured in different ranges are comparable.
4. Variance filter Drop genes whose expression barely changes across samples.
                   Constant genes carry zero information about the outcome.
                   Removing the bottom 20% reduces computational load without
                   losing signal.
5. Train/test split Stratified split (same class ratio in train & test) to
                   avoid evaluating a model on an unbalanced subset.

Output shape example:
  Raw:    22,283 genes  →  after filter: ~17,800  →  candidates: 500
  Samples: 286 total   →  train: ~229,  test: ~57
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    LOG_TRANSFORM,
    VARIANCE_PERCENTILE,
    TEST_SIZE,
    RANDOM_STATE,
)


# ─── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    expr_df : pd.DataFrame  shape (n_probes, n_samples)  — raw expression from data_loader
    meta_df : pd.DataFrame  shape (n_samples, …)         — must have a 'label' column

    Returns
    -------
    dict with keys:
        X_train    (n_train, n_genes)   training expression matrix (numpy float32)
        X_test     (n_test,  n_genes)   test expression matrix
        y_train    (n_train,)           binary labels for training
        y_test     (n_test,)            binary labels for test
        gene_ids   list[str]            probe/gene IDs after variance filter
        stats      dict                 diagnostic counts for each pipeline stage
    """
    # ── 1. Keep only samples with known labels ─────────────────────────────
    valid_mask = meta_df["label"].isin([0, 1])
    valid_ids  = meta_df.index[valid_mask]
    y          = meta_df.loc[valid_ids, "label"].values.astype(int)

    # expr_df: rows = genes, columns = sample IDs
    # Subset to samples that have valid labels AND exist in expression matrix
    common = [s for s in valid_ids if s in expr_df.columns]
    if len(common) < len(valid_ids):
        missing = len(valid_ids) - len(common)
        print(f"  ⚠  {missing} sample IDs not found in expression matrix — dropped.")
    valid_ids = pd.Index(common)
    y = meta_df.loc[valid_ids, "label"].values.astype(int)

    # Subset columns (samples) → transpose → (n_samples, n_genes)
    X_raw    = expr_df[valid_ids].T.values.astype(np.float32)
    gene_ids = list(expr_df.index)

    stats = {
        "n_samples"   : len(valid_ids),
        "n_relapse"   : int((y == 1).sum()),
        "n_no_relapse": int((y == 0).sum()),
        "n_probes_raw": X_raw.shape[1],
    }
    print(f"\n[Preprocessing]")
    print(f"  Samples          : {stats['n_samples']}"
          f"  (relapse={stats['n_relapse']}, no-relapse={stats['n_no_relapse']})")
    print(f"  Raw probes       : {stats['n_probes_raw']:,}")

    # ── 2. Log₂ transform ──────────────────────────────────────────────────
    if LOG_TRANSFORM:
        # Suppress the numpy warning that fires when log2 evaluates the
        # 'false' branch of np.where on negative values (result is discarded).
        with np.errstate(invalid='ignore', divide='ignore'):
            X_raw = np.where(X_raw > 0, np.log2(X_raw + 1.0), 0.0)
        print(f"  Log2 transform   : done")

    # ── 3. Z-score per gene ────────────────────────────────────────────────
    mu    = X_raw.mean(axis=0, keepdims=True)
    sigma = X_raw.std(axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)   # avoid divide-by-zero for constant genes
    X_std = (X_raw - mu) / sigma

    # ── 4. Variance filter ─────────────────────────────────────────────────
    gene_var  = X_std.var(axis=0)
    threshold = np.percentile(gene_var, VARIANCE_PERCENTILE)
    keep      = gene_var >= threshold
    X_filt    = X_std[:, keep]
    gene_ids  = [gene_ids[i] for i in range(len(gene_ids)) if keep[i]]

    stats["n_probes_after_var_filter"] = int(X_filt.shape[1])
    print(f"  Variance filter  : {stats['n_probes_after_var_filter']:,} probes kept "
          f"(dropped bottom {VARIANCE_PERCENTILE}%)")

    # ── 5. Train / test split ──────────────────────────────────────────────
    (X_train, X_test,
     y_train, y_test) = train_test_split(
        X_filt, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train / test     : {len(y_train)} / {len(y_test)} samples")

    return {
        "X_train"  : X_train,
        "X_test"   : X_test,
        "y_train"  : y_train,
        "y_test"   : y_test,
        "gene_ids" : gene_ids,
        "stats"    : stats,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import get_data

    print("=== Preprocessing pipeline ===")
    expr_df, meta_df = get_data()
    result = preprocess(expr_df, meta_df)
    print(f"\nX_train : {result['X_train'].shape}")
    print(f"X_test  : {result['X_test'].shape}")
    print(f"Genes kept: {len(result['gene_ids']):,}")
