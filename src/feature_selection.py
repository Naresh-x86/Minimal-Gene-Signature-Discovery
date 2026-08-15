"""
feature_selection.py — Statistical ranking of candidate genes.

What we are doing here (in plain English)
-----------------------------------------
After preprocessing we still have ~17,000 genes. We can't search all
combinations — there are too many. So we first ask:

  "Which genes already look different between relapse and no-relapse patients
   when considered individually?"

We rank ALL genes using two statistics, then keep the top 500 as 'candidates'
for the harder combinatorial search.

Statistics used
---------------
Mann-Whitney U test
    A statistical test that asks: are the expression values of this gene
    generally *higher* or *lower* in relapse patients than in healthy ones?
    It makes no assumption about the distribution of gene expression values
    (non-parametric), which is good because biological data is rarely normal.

    Result: a p-value. A small p-value means the gene likely differs between
    groups (not just by random chance).

AUC (single-gene)
    For each gene, how well does its expression alone separate the two classes?
    AUC=0.5 → random; AUC=1.0 → perfect.

We rank by p-value and keep the top TOP_N_CANDIDATES genes.

Why not use these top genes directly as our signature?
    Because we are looking for *combinations*.  Gene A alone might be weak;
    Gene B alone might be weak; but A + B together might be very strong.
    The statistical ranking gives us a manageable starting pool for that search.
"""

import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import TOP_N_CANDIDATES
from src.utils  import auc_from_scores


# ─── Public API ────────────────────────────────────────────────────────────────

def rank_genes(
    X_train   : np.ndarray,
    y_train   : np.ndarray,
    gene_ids  : List[str],
    top_n     : int = TOP_N_CANDIDATES,
) -> Tuple[List[str], List[int], pd.DataFrame]:
    """
    Rank all genes by differential expression (relapse vs. no-relapse).

    Parameters
    ----------
    X_train  : (n_train, n_genes)  training expression matrix (post-preprocessing)
    y_train  : (n_train,)          binary labels (0/1)
    gene_ids : list[str]           gene/probe IDs corresponding to columns of X_train
    top_n    : int                 number of top-ranked candidates to return

    Returns
    -------
    candidate_ids     : list[str]  top_n gene IDs
    candidate_indices : list[int]  their column indices in X_train / X_test
    ranking_df        : pd.DataFrame  full ranking table (all genes, sorted by p-value)
    """
    n_genes = X_train.shape[1]
    p_vals  = np.empty(n_genes, dtype=float)
    aucs    = np.empty(n_genes, dtype=float)

    pos_mask = (y_train == 1)
    neg_mask = ~pos_mask
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()

    for i in range(n_genes):
        x_pos = X_train[pos_mask, i]
        x_neg = X_train[neg_mask, i]
        _, p   = mannwhitneyu(x_pos, x_neg, alternative="two-sided")
        p_vals[i] = p
        aucs[i]   = auc_from_scores(X_train[:, i], y_train)

    ranking_df = pd.DataFrame({
        "gene_id"     : gene_ids,
        "p_value"     : p_vals,
        "auc_single"  : aucs,
        "neg_log10_p" : -np.log10(p_vals + 1e-300),   # volcano-plot axis
    }).sort_values("p_value").reset_index(drop=True)

    top_df            = ranking_df.head(top_n)
    candidate_ids     = list(top_df["gene_id"])
    id_to_col         = {gid: i for i, gid in enumerate(gene_ids)}
    candidate_indices = [id_to_col[gid] for gid in candidate_ids]

    best = ranking_df.iloc[0]
    print(f"\n[Gene Ranking]")
    print(f"  Genes ranked     : {n_genes:,}")
    print(f"  Candidates kept  : {top_n}")
    print(f"  Best single gene : {best['gene_id']}"
          f"  (AUC={best['auc_single']:.3f}, p={best['p_value']:.2e})")

    return candidate_ids, candidate_indices, ranking_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader    import get_data
    from src.preprocessing  import preprocess

    expr_df, meta_df = get_data()
    prep = preprocess(expr_df, meta_df)
    ids, idx, df = rank_genes(prep["X_train"], prep["y_train"], prep["gene_ids"])
    print(f"\nTop 10 candidates:\n{df.head(10).to_string(index=False)}")
