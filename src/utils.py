"""
utils.py — Shared mathematical utilities used by all three search implementations.

Key function: auc_from_scores
-------------------------------
AUC stands for "Area Under the ROC Curve". It measures how well a score
separates two groups (relapse vs. no-relapse).

AUC = P(score for positive > score for negative)
    = the fraction of (relapse, no-relapse) sample pairs where the relapse
      sample has a higher score than the no-relapse sample.

AUC = 0.5 → random (no predictive power)
AUC = 1.0 → perfect separation

We compute this via the Mann-Whitney U statistic, which is equivalent to AUC
and needs no probability model or distributional assumptions.

All three implementations — sequential, parallel CPU, and GPU — use the same
mathematical definition so their results are directly comparable.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ─── Core AUC scorer (single candidate) ───────────────────────────────────────

def auc_from_scores(scores: np.ndarray, y: np.ndarray) -> float:
    """
    Compute exact AUC for a single score vector via the Mann-Whitney U statistic.

    We always return max(AUC, 1-AUC) so that a gene that perfectly separates
    the classes in *either* direction (high or low expression) scores 1.0.
    A truly random gene scores 0.5.

    Parameters
    ----------
    scores : (n_samples,) float array — combined signature score per sample
    y      : (n_samples,) int array  — binary labels (0 = no relapse, 1 = relapse)

    Returns
    -------
    auc : float in [0.5, 1.0]
    """
    pos = scores[y == 1]
    neg = scores[y == 0]
    n_pos, n_neg = len(pos), len(neg)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Vectorised pairwise comparison: wins[i,j] = (pos[i] > neg[j])
    # Shape: (n_pos, n_neg)
    wins = (pos[:, None] > neg[None, :]).sum()
    auc  = wins / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


# ─── Batch AUC scorer (all candidates at once, CPU) ───────────────────────────

def auc_from_scores_batch(
    proj: np.ndarray,        # (n_samples,)      current signature projection
    candidates: np.ndarray,  # (n_cands, n_samples) all candidate expressions
    y: np.ndarray,           # (n_samples,)      binary labels
) -> np.ndarray:
    """
    Compute AUC for ALL candidate genes simultaneously using NumPy broadcasting.

    For each candidate i:
        combined[i] = proj + candidates[i]
        AUC[i]      = auc_from_scores(combined[i], y)

    The inner pairwise comparison is vectorised across all candidates at once.
    This is the batch CPU version; the GPU version (gpu_search.py) does the
    same computation on CUDA tensors.

    Returns
    -------
    aucs : (n_cands,) float array
    """
    # combined: (n_cands, n_samples)
    combined   = proj[None, :] + candidates

    pos_mask   = (y == 1)
    neg_mask   = ~pos_mask
    pos_scores = combined[:, pos_mask]   # (n_cands, n_pos)
    neg_scores = combined[:, neg_mask]   # (n_cands, n_neg)
    n_pos      = pos_mask.sum()
    n_neg      = neg_mask.sum()

    # (n_cands, n_pos, n_neg) boolean pairwise comparison, then sum over (n_pos, n_neg)
    wins = (pos_scores[:, :, None] > neg_scores[:, None, :]).sum(axis=(1, 2))
    aucs = wins.astype(float) / (n_pos * n_neg)
    return np.maximum(aucs, 1.0 - aucs)


# ─── Final validation model ────────────────────────────────────────────────────

def final_lr_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Train a Logistic Regression on X_train and return test-set AUC.

    This is the 'official' result metric, separate from the greedy search score.
    It gives a proper out-of-sample estimate of the signature's predictive value.

    Parameters
    ----------
    X_train, X_test : (n_samples, n_selected_genes) — expression of selected genes
    y_train, y_test : (n_samples,) — binary labels

    Returns
    -------
    test_auc : float in [0, 1]
    """
    if X_train.shape[1] == 0:
        return 0.5

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


# Avoid circular import by inlining the constant here
RANDOM_STATE = 42
