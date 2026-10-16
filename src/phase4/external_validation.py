# -*- coding: utf-8 -*-
"""
external_validation.py — Phase 4: External Cohort Validation
============================================================

Scientific Goal
---------------
Test whether the minimal gene signatures discovered on the Rotterdam discovery
cohort (GSE2034, 286 patients, Wang et al.) generalize to predict metastatic relapse
in completely unseen breast cancer patients from independent medical centers:
  1. GSE11121 (Schmidt et al., 200 patients, Mainz University Medical Center, Germany)
  2. GSE7390  (Desmedt et al., 198 patients, TRANSBIG European Multicenter Cohort)

Both external cohorts used the identical microarray platform (Affymetrix HG-U133A,
GPL96) and recorded distant metastasis-free survival (e.dmfs).

Validation Protocol:
  - ZERO DATA LEAKAGE: Logistic regression models are trained strictly on GSE2034.
  - Frozen model weights are applied directly to external cohorts.
  - Per-cohort Z-score normalization prevents cross-center batch effects without leakage.
  - Statistical significance evaluated via 1,000 bootstrap iterations for 95% CIs.
"""

import gzip
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Force UTF-8 stdout/stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Add root directory to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DATA_DIR, RESULTS_DIR, RANDOM_STATE

EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# NCBI GEO FTP URLs
URL_GPL96_ANNOT = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"
URL_GSE11121_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11121/matrix/GSE11121_series_matrix.txt.gz"
URL_GSE7390_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE7nnn/GSE7390/matrix/GSE7390_series_matrix.txt.gz"


# ─── 1. Download & Caching Helpers ───────────────────────────────────────────

def download_file(url: str, dest_path: Path, chunk_size: int = 128 * 1024) -> Path:
    """Download file with progress bar if it doesn't already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return dest_path

    print(f"Downloading {dest_path.name} from NCBI GEO...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"✓ Saved: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
    return dest_path


# ─── 2. GPL96 Annotation Parser ──────────────────────────────────────────────

def load_gpl96_probe_map(annot_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Parse GPL96 annotation table to map Affymetrix Probe IDs to a list of Gene Symbols.
    Handles multi-symbol entries (e.g. 'LOC102725271///NTM') and historical aliases (TSTA3 -> GFUS).
    """
    if annot_path is None:
        annot_path = EXTERNAL_DATA_DIR / "GPL96.annot.gz"
    download_file(URL_GPL96_ANNOT, annot_path)

    print("Loading GPL96 annotation dictionary...")
    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            if line.startswith("!platform_table_begin"):
                break
        df = pd.read_csv(gz, sep="\t", low_memory=False)

    probe_to_genes: Dict[str, List[str]] = {}
    for _, row in df[["ID", "Gene symbol"]].dropna().iterrows():
        p_id = str(row["ID"]).strip()
        syms = [s.strip() for s in str(row["Gene symbol"]).split("///") if s.strip()]
        # Historical alias: TSTA3 is the former name of GFUS (GDP-L-fucose synthase, Entrez 7264)
        if "TSTA3" in syms and "GFUS" not in syms:
            syms.append("GFUS")
        if syms:
            probe_to_genes[p_id] = syms

    print(f"✓ Mapped {len(probe_to_genes):,} probes to gene symbols (including aliases).")
    return probe_to_genes


# ─── 3. GEO Series Matrix Parser ─────────────────────────────────────────────

def parse_geo_matrix(matrix_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parse a GEO series matrix file.
    Returns:
      expr_df: DataFrame (probes x samples)
      labels : Series (sample_id -> binary int: 1=relapse/metastasis, 0=no relapse)
    """
    print(f"Parsing {matrix_path.name}...")
    sample_ids = None
    char_rows = []
    probe_ids = []
    expr_matrix_rows = []
    in_table = False

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as gz:
        for raw in gz:
            line = raw.rstrip("\r\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                parts = line.replace('"', '').split("\t")
                if parts[0] == "ID_REF":
                    sample_ids = parts[1:]
                else:
                    probe_ids.append(parts[0])
                    expr_matrix_rows.append([float(x) if x.strip() else np.nan for x in parts[1:]])
                continue

            if line.startswith("!Sample_characteristics_ch1\t"):
                char_rows.append([v.strip().strip('"') for v in line.split("\t")[1:]])
            elif line.startswith("!Sample_geo_accession\t") and sample_ids is None:
                sample_ids = [v.strip().strip('"') for v in line.split("\t")[1:]]

    if not sample_ids or not expr_matrix_rows:
        raise ValueError(f"Failed to parse expression table from {matrix_path.name}")

    n_samples = len(sample_ids)
    expr_df = pd.DataFrame(expr_matrix_rows, index=probe_ids, columns=sample_ids)

    # Extract binary relapse label (prefer e.dmfs if available)
    labels = pd.Series(index=sample_ids, dtype=float)
    label_found = False

    for row in char_rows:
        for idx, cell in enumerate(row[:n_samples]):
            lower = cell.lower()
            if "e.dmfs:" in lower or "dmfs:" in lower:
                val = cell.split(":")[-1].strip()
                labels.iloc[idx] = 1.0 if val in {"1", "yes", "true"} else 0.0
                label_found = True
            elif not label_found and any(kw in lower for kw in ["relapse:", "event:", "metastasis:"]):
                val = cell.split(":")[-1].strip()
                labels.iloc[idx] = 1.0 if val in {"1", "yes", "true", "positive"} else 0.0

    if labels.isna().any():
        labels = labels.fillna(0.0)
    labels = labels.astype(int)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"✓ Parsed {matrix_path.name}: {expr_df.shape[0]:,} probes × {expr_df.shape[1]} samples | Relapses={n_pos}, No-Relapse={n_neg}")
    return expr_df, labels


def process_external_cohort(
    matrix_url: str,
    matrix_filename: str,
    probe_to_genes: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Download, parse, map probes to genes, check log2, and z-score standardise an external cohort.
    Returns:
      gene_expr: DataFrame (samples x genes), ready for model inference
      labels   : Series (samples -> binary 0/1)
    """
    local_path = EXTERNAL_DATA_DIR / matrix_filename
    download_file(matrix_url, local_path)
    probe_expr, labels = parse_geo_matrix(local_path)

    # Invert probe_to_genes to gene -> list of probes present in this cohort
    gene_to_probes: Dict[str, List[str]] = {}
    for probe, syms in probe_to_genes.items():
        if probe in probe_expr.index:
            for s in syms:
                gene_to_probes.setdefault(s, []).append(probe)

    print(f"  Summarizing {len(probe_expr):,} probes into {len(gene_to_probes):,} genes for {matrix_filename}...")
    
    # Vectorized summarization: average all probes belonging to each gene
    # Grouping indices for matrix multiplication / efficient slicing
    gene_rows = {}
    for gene, p_list in gene_to_probes.items():
        if len(p_list) == 1:
            gene_rows[gene] = probe_expr.loc[p_list[0]].values
        else:
            gene_rows[gene] = probe_expr.loc[p_list].values.mean(axis=0)

    gene_expr = pd.DataFrame(gene_rows, index=probe_expr.columns)

    # Check if raw linear scale; if so, apply log2(x + 1)
    max_val = gene_expr.values.max()
    if max_val > 50.0:
        print(f"  Detected raw linear scale (max={max_val:.1f}). Applying log2(x + 1)...")
        gene_expr = np.log2(np.maximum(gene_expr, 0) + 1.0)
    else:
        print(f"  Values appear already log2-transformed (max={max_val:.2f}).")

    # Per-cohort Z-score standardization across patients per gene
    means = gene_expr.mean(axis=0)
    stds = gene_expr.std(axis=0).replace(0, 1.0)
    gene_expr_z = (gene_expr - means) / stds

    return gene_expr_z, labels


# ─── 4. Model Training & Cross-Cohort Evaluation ──────────────────────────────

def clean_gene_name(name: str) -> str:
    """Strip duplicate probe suffixes like '_1' from discovery gene names."""
    return name.split("_")[0].strip()


def evaluate_signature_on_cohort(
    clf: LogisticRegression,
    genes: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cohort_name: str,
    n_bootstrap: int = 1000
) -> Dict:
    """
    Evaluate trained logistic regression classifier on an external cohort.
    Computes AUC, 95% bootstrap confidence interval, and risk stratification metrics.
    """
    clean_genes = [clean_gene_name(g) for g in genes]
    # Check gene coverage
    missing = [g for g in clean_genes if g not in X_test.columns]
    if missing:
        print(f"  ⚠ Warning: Missing genes in {cohort_name}: {missing}")
        avail_genes = [g for g in clean_genes if g in X_test.columns]
    else:
        avail_genes = clean_genes

    X_sub = X_test[avail_genes].values
    y_true = y_test.values

    # Predict relapse probabilities
    y_probs = clf.predict_proba(X_sub)[:, 1]

    # Compute ROC-AUC
    auc = float(roc_auc_score(y_true, y_probs))
    # If inverted due to sign flip, record standard AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # 1,000 Stratified Bootstrap iterations for 95% Confidence Interval
    rng = np.random.RandomState(RANDOM_STATE)
    boot_aucs = []
    n_samples = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    for _ in range(n_bootstrap):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        try:
            b_auc = roc_auc_score(y_true[b_idx], y_probs[b_idx])
            boot_aucs.append(b_auc)
        except ValueError:
            continue

    ci_low = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else auc
    ci_high = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else auc

    # Clinical Risk Stratification: Median Cutoff
    median_thresh = np.median(y_probs)
    high_risk_mask = y_probs >= median_thresh
    low_risk_mask = ~high_risk_mask

    rate_high = float(y_true[high_risk_mask].mean()) if high_risk_mask.sum() > 0 else 0.0
    rate_low = float(y_true[low_risk_mask].mean()) if low_risk_mask.sum() > 0 else 0.0

    # Odds Ratio & 2x2 Contingency Table
    #           Relapse   No Relapse
    # High Risk    a          b
    # Low Risk     c          d
    a = int((y_true[high_risk_mask] == 1).sum())
    b = int((y_true[high_risk_mask] == 0).sum())
    c = int((y_true[low_risk_mask] == 1).sum())
    d = int((y_true[low_risk_mask] == 0).sum())

    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])

    return {
        "cohort": cohort_name,
        "n_patients": n_samples,
        "n_events": int((y_true == 1).sum()),
        "event_rate": float((y_true == 1).mean()),
        "auc": auc,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "fpr": fpr,
        "tpr": tpr,
        "y_probs": y_probs,
        "y_true": y_true,
        "rate_high_risk": rate_high,
        "rate_low_risk": rate_low,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "contingency": [[a, b], [c, d]]
    }


# ─── 5. Main Validation Runner ────────────────────────────────────────────────

def run_external_validation() -> pd.DataFrame:
    """
    Execute complete Phase 4 external validation across all cohorts and signatures.
    """
    print("=" * 70)
    print("PHASE 4: EXTERNAL COHORT VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Load GPL96 Probe Annotation
    probe_map = load_gpl96_probe_map()

    # 2. Load and Preprocess Discovery Cohort (GSE2034)
    print("\nLoading Discovery Cohort (GSE2034, Wang et al.)...")
    disc_expr = pd.read_csv(DATA_DIR / "expression.csv", index_col=0)
    disc_meta = pd.read_csv(DATA_DIR / "metadata.csv", index_col=0)
    y_disc = disc_meta["label"].loc[disc_expr.columns]

    # Map discovery gene rows to clean gene symbols if needed
    disc_expr_clean = disc_expr.copy()
    disc_expr_clean.index = [clean_gene_name(g) for g in disc_expr_clean.index]
    # If duplicate rows exist after cleaning, average them
    disc_expr_clean = disc_expr_clean.groupby(disc_expr_clean.index).mean().T

    # Standardize discovery cohort
    disc_means = disc_expr_clean.mean(axis=0)
    disc_stds = disc_expr_clean.std(axis=0).replace(0, 1.0)
    X_disc = (disc_expr_clean - disc_means) / disc_stds

    print(f"✓ Discovery cohort ready: {X_disc.shape[0]} patients × {X_disc.shape[1]:,} genes.")

    # 3. Load External Cohorts
    print("\nLoading External Cohort 1: GSE11121 (Schmidt et al. Mainz, 200 patients)...")
    X_gse11121, y_gse11121 = process_external_cohort(
        URL_GSE11121_MATRIX, "GSE11121_series_matrix.txt.gz", probe_map
    )

    print("\nLoading External Cohort 2: GSE7390 (Desmedt et al. TRANSBIG, 198 patients)...")
    X_gse7390, y_gse7390 = process_external_cohort(
        URL_GSE7390_MATRIX, "GSE7390_series_matrix.txt.gz", probe_map
    )

    # Combined External Cohort (398 patients)
    common_ext_genes = list(set(X_gse11121.columns) & set(X_gse7390.columns))
    X_ext_comb = pd.concat([X_gse11121[common_ext_genes], X_gse7390[common_ext_genes]], axis=0)
    y_ext_comb = pd.concat([y_gse11121, y_gse7390], axis=0)
    print(f"\n✓ Combined External Cohort: {X_ext_comb.shape[0]} patients ({int((y_ext_comb==1).sum())} relapses)")

    # 4. Signatures to Evaluate
    signatures = {
        "GA-5 (Core Hub)": ["SLC35E3", "NTM", "RECQL4", "TNFRSF10C", "CPB1"],
        "GA-5 (Sweep k=5)": ["NEK4", "SURF2", "RECQL4", "CEACAM6", "PDGFRB"],
        "GA-22 (Peak k=22)": [
            "SURF2", "NTM", "ATP9A", "TANC2", "EEF1A2", "PLAT", "TMEM106C",
            "RECQL4", "POLR2J", "ZC3H11A", "TNFRSF10C", "PNRC1", "CPB1", "COX11",
            "CEACAM6", "SLC1A1", "ENSA", "GFUS", "NDUFA4", "ADAM18", "SMYD2", "TGFBI"
        ],
        "Greedy-5 Baseline": ["ADAM18", "FTCD", "TNFRSF10C", "NEFL", "NKG7"],
        "Single-Gene (RECQL4)": ["RECQL4"],
        "Single-Gene (CD44)": ["CD44"],
    }

    cohort_dict = {
        "Discovery (GSE2034)": (X_disc, y_disc),
        "External 1: Mainz (GSE11121)": (X_gse11121, y_gse11121),
        "External 2: TRANSBIG (GSE7390)": (X_gse7390, y_gse7390),
        "Combined External (398 pts)": (X_ext_comb, y_ext_comb),
    }

    # 5. Fit Model on Discovery and Evaluate Everywhere
    results_records = []
    detailed_results = {}

    print("\n" + "=" * 70)
    print("CROSS-COHORT MODEL EVALUATION")
    print("=" * 70)

    for sig_name, genes in signatures.items():
        print(f"\n--- Testing Signature: {sig_name} ({len(genes)} genes) ---")
        clean_genes = [clean_gene_name(g) for g in genes]
        
        # Train strictly on GSE2034
        X_train_sub = X_disc[clean_genes].values
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train_sub, y_disc.values)

        for c_name, (X_c, y_c) in cohort_dict.items():
            metrics = evaluate_signature_on_cohort(
                clf, genes, X_c, y_c, c_name, n_bootstrap=1000
            )
            detailed_results[(sig_name, c_name)] = metrics
            
            print(f"  {c_name:<30} | AUC = {metrics['auc']:.4f} (95% CI: {metrics['ci_lower']:.4f}–{metrics['ci_upper']:.4f}) | "
                  f"High vs Low Risk: {metrics['rate_high_risk']*100:.1f}% vs {metrics['rate_low_risk']*100:.1f}% "
                  f"(OR={metrics['odds_ratio']:.2f}, p={metrics['p_value']:.4f})")

            results_records.append({
                "signature": sig_name,
                "n_genes": len(genes),
                "cohort": c_name,
                "n_patients": metrics["n_patients"],
                "n_events": metrics["n_events"],
                "auc": metrics["auc"],
                "ci_lower": metrics["ci_lower"],
                "ci_upper": metrics["ci_upper"],
                "rate_high_risk": metrics["rate_high_risk"],
                "rate_low_risk": metrics["rate_low_risk"],
                "odds_ratio": metrics["odds_ratio"],
                "p_value": metrics["p_value"],
                "genes": ", ".join(genes)
            })

    results_df = pd.DataFrame(results_records)
    out_csv = RESULTS_DIR / "external_validation_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved validation metrics to: {out_csv}")

    # Also save detailed results dictionary for notebook figure generation
    import pickle
    pkl_path = RESULTS_DIR / "external_validation_detailed.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(detailed_results, f)
    print(f"✓ Saved detailed ROC curve data to: {pkl_path}")

    return results_df


if __name__ == "__main__":
    df = run_external_validation()
    print("\nSummary Table:")
    print(df[["signature", "cohort", "auc", "ci_lower", "ci_upper", "p_value"]].to_string())
