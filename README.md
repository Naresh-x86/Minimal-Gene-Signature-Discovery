# Minimal Gene Signature Discovery

## Core Question

> *How few genes do we actually need to predict whether a breast cancer patient will relapse — and how fast can we find them?*

Using the **GSE2034** public dataset (286 breast cancer patients × 22,283 genes), we discover the smallest combination of genes that preserves predictive performance. The search is accelerated using parallel CPU cores and a GPU.

---

## Project Structure

```
src/
├── phase1/                     # Phase 1: Greedy Forward Selection + HPC benchmarking
│   ├── signature_search.py     # Sequential greedy (single-core baseline)
│   ├── parallel_search.py      # Greedy parallelised across CPU cores (joblib)
│   ├── gpu_search.py           # Greedy with GPU batching (PyTorch)
│   └── benchmark.py            # HPC scaling study: 1-core vs N-core vs GPU
├── phase2/                     # Phase 2: Genetic Algorithm + Biological Pathways
│   ├── genetic_search.py       # Genetic Algorithm search engine  [coming Sep 4]
│   └── pathway_graph.py        # Gene interaction graph scoring   [coming Sep 10]
├── config.py                   # All tuneable parameters in one place
├── data_loader.py              # Download + parse GSE2034 from Zenodo / NCBI GEO
├── feature_selection.py        # Statistical gene ranking (Mann-Whitney U)
├── preprocessing.py            # Log-transform, variance filter, train/test split
└── utils.py                    # Shared AUC computation helpers

notebooks/
└── make_figures.py             # Generate all slide-ready plots -> results/figures/

data/                           # Auto-downloaded dataset (created on first run)
results/                        # All outputs: CSVs + figures
run_pipeline.py                 # One-command runner
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

### Shared (both phases)

| Module | Role |
|--------|------|
| `data_loader.py` | Downloads GSE2034 from Zenodo or NCBI GEO |
| `preprocessing.py` | Log2 transform -> Z-score -> variance filter -> train/test split |
| `feature_selection.py` | Ranks genes by Mann-Whitney U p-value, keeps top 500 |
| `utils.py` | AUC calculation helpers used by all search modules |

### Phase 1 — Greedy Forward Selection (`src/phase1/`)

| Module | Role |
|--------|------|
| `signature_search.py` | Sequential greedy: add the best gene one step at a time |
| `parallel_search.py` | Same algorithm, inner loop parallelised across CPU cores |
| `gpu_search.py` | Same algorithm, inner loop replaced by one batched GPU tensor op |
| `benchmark.py` | Times all three on synthetic data at N=100...10,000 candidates |

### Phase 2 — Genetic Algorithm + Pathways (`src/phase2/`)

| Module | Role |
|--------|------|
| `genetic_search.py` | Genetic Algorithm: population-based search that escapes local minima |
| `pathway_graph.py` | Loads gene interaction graph; adds pathway-connectivity fitness bonus |

---

## Dataset: GSE2034

- **Source:** NCBI GEO ([accession GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034))
- **Publication:** Wang et al., *Lancet* 2005
- **Content:** 286 breast tumour samples x 22,283 Affymetrix HG-U133A probes
- **Labels:** Bone metastasis within 5 years (69 positive / 217 negative)
- **Download:** Automated on first `python run_pipeline.py`

---

## Phase 1 Results (complete)

- Best single gene: **CD44** (AUC = 0.721, p = 8.5e-7)
- Greedy signature peak: **AUC = 0.594** at 20 genes (held-out test set)
- GPU speedup: **up to 5.4x** faster than sequential at N=2,000 candidates
- CPU parallel (joblib): slower than sequential for this dataset size — demonstrates
  that HPC requires the right hardware for the right problem size

## Phase 2 (in progress)

- [x] Repository reorganization (`src/phase1/`, `src/phase2/`)
- [x] Genetic Algorithm search engine (completed Sep 4)
  - Best val AUC: **0.7727** vs Greedy 0.5942 (+30.1% improvement)
  - GPU fitness evaluation: **2.69x** faster than sequential
  - 200 generations x 100 individuals in 1.6s (sequential)
- [x] Biological pathway integration via gene interaction graph (completed Sep 10)
  - Curated 160+ breast cancer & immune gene interactions (KEGG/Literature)
  - Added vectorizable pathway connectivity scoring module
  - GA now selects biologically meaningful combinations while maintaining high AUC