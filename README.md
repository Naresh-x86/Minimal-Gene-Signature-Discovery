# Minimal Gene Signature Discovery

> **AIML + Biological Data + HPC** — B.Tech Phase 1 Project

## Core Question

> *How few genes do we actually need to predict whether a breast cancer patient will relapse — and how fast can we find them?*

Using the **GSE2034** public dataset (286 breast cancer patients × 22,283 genes), we discover the smallest combination of genes that preserves predictive performance. The search is accelerated using parallel CPU cores and a GPU.

---

## Project Structure

```
├── src/
│   ├── config.py              # All tuneable parameters in one place
│   ├── data_loader.py         # Download + parse GSE2034 from NCBI GEO
│   ├── preprocessing.py       # Log-transform, variance filter, train/test split
│   ├── feature_selection.py   # Statistical gene ranking (Mann-Whitney U)
│   ├── signature_search.py    # Sequential greedy forward selection (baseline)
│   ├── parallel_search.py     # Multi-core CPU version (joblib)
│   ├── gpu_search.py          # GPU version (PyTorch) — the HPC contribution
│   ├── benchmark.py           # Scaling study: 1-core vs N-core vs GPU
│   └── utils.py               # Shared AUC computation
├── notebooks/
│   └── make_figures.py        # Generate all slide-ready plots → results/figures/
├── data/                      # Auto-downloaded dataset (created on first run)
├── results/                   # All outputs: CSVs + figures
├── run_pipeline.py            # One-command runner
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (check your CUDA version with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Run the full pipeline (downloads data automatically)
python run_pipeline.py

# 3. Generate slide-ready figures
python notebooks/make_figures.py
```

For a faster run (fewer genes, skip benchmark):
```bash
python run_pipeline.py --max-genes 15 --skip-benchmark
```

---

## What Each Module Does

| Module | Role |
|--------|------|
| `data_loader.py` | Downloads GSE2034 from NCBI GEO or Zenodo |
| `preprocessing.py` | Log₂ transform → Z-score → variance filter → train/test split |
| `feature_selection.py` | Ranks genes by Mann-Whitney U p-value, keeps top 500 |
| `signature_search.py` | Sequential greedy: add the best gene one step at a time |
| `parallel_search.py` | Same algorithm, inner loop parallelised across CPU cores |
| `gpu_search.py` | Same algorithm, inner loop replaced by one batched GPU tensor op |
| `benchmark.py` | Times all three on synthetic data at N=100…10,000 candidates |

---

## Dataset: GSE2034

- **Source:** NCBI GEO ([accession GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034))
- **Publication:** Wang et al., *Lancet* 2005
- **Content:** 286 breast tumour samples × 22,283 Affymetrix HG-U133A probes
- **Labels:** Distant metastasis within 5 years (106 positive, 180 negative)
- **Download:** Automated on first `python run_pipeline.py`

---

## Phase 2 (next month — not implemented yet)

- External validation on a second independent cohort
- Genetic / evolutionary algorithm search
- Gene interaction network (pairwise scoring)
- MPI / OpenMP C-level HPC
- Full written report with statistical analysis