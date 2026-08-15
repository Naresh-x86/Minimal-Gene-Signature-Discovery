"""
config.py — Central configuration for the Minimal Gene Signature Discovery project.

Edit the constants here to tune the pipeline without touching any other file.
All other modules import from this file; nothing is hard-coded elsewhere.
"""

import multiprocessing
from pathlib import Path

# ─── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# ─── Dataset ───────────────────────────────────────────────────────────────────
# GSE2034 — breast-cancer relapse-free survival (Wang et al., 2005)
# 286 patient samples × 22,283 Affymetrix HG-U133A probes
# Labels: distant relapse within 5 years (1) vs. no relapse (0)
GEO_ACCESSION  = "GSE2034"
GEO_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/matrix/"
    "GSE2034_series_matrix.txt.gz"
)
ZENODO_RECORD  = "18125181"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# ─── Preprocessing ─────────────────────────────────────────────────────────────
LOG_TRANSFORM       = True   # Log₂-transform expression values (standard for arrays)
VARIANCE_PERCENTILE = 20     # Drop genes in the bottom 20% by variance (no signal)
TOP_N_CANDIDATES    = 500    # Final candidate pool for the signature search

# ─── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.2   # 20% held-out test set (stratified)
RANDOM_STATE = 42

# ─── Signature search ──────────────────────────────────────────────────────────
MAX_SIGNATURE_SIZE = 30   # Maximum number of genes to select

# ─── HPC benchmark ─────────────────────────────────────────────────────────────
# Problem: score N candidate genes simultaneously — the inner loop of greedy selection.
# We scale N_candidates from small → large to find where GPU advantage kicks in.
_n_cpu = multiprocessing.cpu_count()       # 20 logical threads on i7-13650HX
BENCHMARK_CPU_WORKERS   = sorted(set([1, 2, 4, _n_cpu // 2, _n_cpu]))
BENCHMARK_PROBLEM_SIZES = [100, 500, 1_000, 2_000, 5_000, 10_000]

# Keep benchmark data realistic: match real GSE2034 dimensions
BENCHMARK_N_SAMPLES = 286   # patients (samples)
BENCHMARK_N_POS     = 69    # relapse-positive samples in GSE2034 (bone metastasis)
BENCHMARK_N_NEG     = 217   # relapse-negative samples in GSE2034
BENCHMARK_REPEATS   = 3     # average over N independent runs for stable timing
