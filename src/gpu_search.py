"""
gpu_search.py — GPU-accelerated greedy forward gene selection.

What moves to the GPU?
-----------------------
At each greedy step we need to score N_remaining candidate genes.
The sequential version does this in a Python loop (N_remaining iterations).
The parallel CPU version distributes that loop across CPU threads.

The GPU version eliminates the loop entirely by batching ALL candidate
evaluations into a single tensor operation:

  Sequential:   for gene in remaining: score(gene)        ← N iterations
  Parallel CPU: distribute loop across cores              ← N / n_cores iterations
  GPU:          score_all_candidates_gpu(all_at_once)     ← 1 tensor operation

GPU kernel (core computation)
-------------------------------
  Input:
    proj     : (N_samples,)        current projection of samples
    X_cands  : (N_remaining, N_samples)  expression of all candidates
    y_bool   : (N_samples,)        True = relapse, False = no relapse

  Step 1: combined[i] = proj + X_cands[i]             (broadcast add)
          → shape (N_remaining, N_samples)

  Step 2: pos_scores = combined[:, y_bool]             (index select)
          neg_scores = combined[:, ~y_bool]
          → shapes (N_remaining, n_pos) and (N_remaining, n_neg)

  Step 3: wins = (pos_scores[:,:,None] > neg_scores[:,None,:]).sum([1,2])
          → shape (N_remaining,)  ← pairwise comparison for all candidates at once

  Step 4: aucs = wins / (n_pos × n_neg)               (element-wise division)

  Memory footprint at N_remaining=500:
    Comparison tensor: 500 × 85 × 143 × 1 byte ≈ 6 MB   (fine on RTX 4060)
  At N_remaining=10,000:
    Comparison tensor: 10,000 × 85 × 143 × 1 byte ≈ 121 MB  (still fine)

Why is this faster than the CPU?
----------------------------------
  The RTX 4060 has 3,072 CUDA cores operating in parallel.
  A single matmul / comparison across thousands of candidates runs in
  microseconds on the GPU, compared to milliseconds in a CPU loop.
  The speedup grows with N_remaining — this is the key HPC result.

Requirements
------------
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  (or the appropriate CUDA version for your driver — check nvidia-smi)
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MAX_SIGNATURE_SIZE
from src.utils  import final_lr_auc

# ── PyTorch import (optional — graceful fallback) ─────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── GPU kernel ────────────────────────────────────────────────────────────────

def score_all_candidates_gpu(
    proj    : "torch.Tensor",   # (N_samples,)           on device
    X_cands : "torch.Tensor",   # (N_remaining, N_samples) on device
    y_bool  : "torch.Tensor",   # (N_samples,) bool       on device
) -> "torch.Tensor":
    """
    Compute exact Mann-Whitney AUC for every candidate gene in one GPU pass.

    This is the GPU kernel — the critical function that replaces the sequential
    inner loop.  Compute identical results to auc_from_scores() in utils.py,
    but for ALL candidates simultaneously.

    Returns
    -------
    aucs : (N_remaining,) tensor on the same device as inputs
    """
    # combined[i] = proj + X_cands[i]  →  shape (N_remaining, N_samples)
    combined   = proj.unsqueeze(0) + X_cands

    pos_scores = combined[:, y_bool]    # (N_remaining, n_pos)
    neg_scores = combined[:, ~y_bool]   # (N_remaining, n_neg)
    n_pos = y_bool.sum().item()
    n_neg = (~y_bool).sum().item()

    # Pairwise comparison: (N_remaining, n_pos, n_neg)
    # Memory: N_remaining × n_pos × n_neg × 1 byte (bool)
    wins = (pos_scores.unsqueeze(2) > neg_scores.unsqueeze(1)).sum(dim=[1, 2])

    aucs = wins.float() / (n_pos * n_neg)
    return torch.maximum(aucs, 1.0 - aucs)


# ─── Public API ────────────────────────────────────────────────────────────────

def run_gpu(
    X_train           : np.ndarray,
    y_train           : np.ndarray,
    X_test            : np.ndarray,
    y_test            : np.ndarray,
    candidate_indices : List[int],
    max_genes         : int  = MAX_SIGNATURE_SIZE,
    device            : str  = "cuda",
    verbose           : bool = True,
) -> pd.DataFrame:
    """
    Greedy forward selection with GPU-batched candidate scoring.

    Parameters
    ----------
    device : "cuda" to use the GPU (recommended), "cpu" to debug on CPU tensors.

    Returns
    -------
    Same column structure as signature_search.run_sequential.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. "
            "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠  CUDA not available — falling back to CPU tensors (no speedup).")
        device = "cpu"

    dev = torch.device(device)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU device : {gpu_name}")
    else:
        print(f"Using CPU tensors (debug mode).")

    # Move the full training matrix to device once (avoids repeated transfers)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    y_bool    = torch.tensor(y_train == 1, dtype=torch.bool, device=dev)

    remaining          = list(candidate_indices)
    selected           = []
    current_proj_train = torch.zeros(len(y_train), dtype=torch.float32, device=dev)
    current_proj_test  = np.zeros(len(y_test), dtype=np.float32)
    records            = []
    t0                 = time.perf_counter()

    for step in range(min(max_genes, len(remaining))):

        # Build candidate matrix: (N_remaining, N_samples)
        remaining_t = torch.tensor(remaining, dtype=torch.long)
        X_cands     = X_train_t[:, remaining_t].T   # index → (N_samples, N_rem).T

        # ── Single GPU operation: score all candidates at once ─────────
        aucs_t     = score_all_candidates_gpu(current_proj_train, X_cands, y_bool)
        best_local = aucs_t.argmax().item()   # position in remaining list
        best_auc   = aucs_t[best_local].item()
        best_gene  = remaining[best_local]

        # Update signature
        selected.append(best_gene)
        remaining.pop(best_local)             # O(1) pop by index
        current_proj_train += X_train_t[:, best_gene]
        current_proj_test  += X_test[:, best_gene]

        # Official evaluation on CPU (Logistic Regression)
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
            "device"    : device,
            "elapsed_s" : round(elapsed, 3),
        })

        if verbose:
            print(f"  Step {step+1:3d}  gene={best_gene:6d}  "
                  f"search_AUC={best_auc:.4f}  val_AUC={val_auc:.4f}  "
                  f"t={elapsed:.1f}s  ({device})")

    return pd.DataFrame(records)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader       import get_data
    from src.preprocessing     import preprocess
    from src.feature_selection import rank_genes

    print("=== GPU greedy search ===\n")
    expr_df, meta_df = get_data()
    prep = preprocess(expr_df, meta_df)
    _, cand_idx, _   = rank_genes(prep["X_train"], prep["y_train"], prep["gene_ids"])

    results = run_gpu(
        prep["X_train"], prep["y_train"],
        prep["X_test"],  prep["y_test"],
        cand_idx, max_genes=15,
    )
    best = results.loc[results["val_auc"].idxmax()]
    print(f"\nBest val AUC = {best['val_auc']:.4f} at step {int(best['step'])}")
