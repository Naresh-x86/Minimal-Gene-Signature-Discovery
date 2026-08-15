# -*- coding: utf-8 -*-
# Force UTF-8 output on Windows (avoids cp1252 errors with special chars)

"""
data_loader.py -- Download and parse the GSE2034 breast cancer dataset.

What is GSE2034?
----------------
A landmark 2005 study (Wang et al.) that measured gene activity across
286 breast tumour samples using an Affymetrix HG-U133A microarray chip.
Each row of the dataset is a patient tumour; the outcome is whether the
patient developed a distant metastasis within 5 years.

Zenodo file layout (primary source, fast and pre-cleaned):
  GSE2034.tsv.gz        — expression matrix
      rows   = genes  (columns: Dataset_ID, Entrez_Gene_ID, HGNC_Symbol,
                                Ensembl_Gene_ID, Chromosome, then sample cols)
      sample cols = GSM IDs (e.g. 'GSM36777', 'GSM36778', …)
      values = normalised expression (already processed, no log needed from raw)

  GSE2034_metadata.tsv  — sample metadata
      columns: Dataset_ID, Sample_ID, Platform_ID, bone_relapses_1_yes_0_no
      label: bone_relapses_1_yes_0_no  (1 = relapse, 0 = no relapse)

Fall-back: NCBI GEO FTP series-matrix if Zenodo is unreachable.

Both paths write two clean CSVs to data/:
  expression.csv   — rows = HGNC gene symbols, columns = GSM sample IDs
  metadata.csv     — rows = GSM sample IDs, columns = clinical attributes
                     Always includes a 'label' column: 1 = relapse, 0 = no relapse
"""

import gzip
import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, GEO_MATRIX_URL, ZENODO_API_URL

# Known Zenodo file names for GSE2034
_ZENODO_EXPR_FILE = "GSE2034.tsv.gz"
_ZENODO_META_FILE = "GSE2034_metadata.tsv"
# Gene-annotation columns in the expression TSV (not expression values)
_EXPR_ANNOT_COLS  = {"Dataset_ID", "Entrez_Gene_ID", "HGNC_Symbol",
                     "Ensembl_Gene_ID", "Chromosome"}


# ─── Public API ────────────────────────────────────────────────────────────────

def get_data(
    data_dir      : Path = DATA_DIR,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the GSE2034 expression matrix and sample metadata.

    Downloads on first call; uses cached CSVs afterwards.

    Returns
    -------
    expr_df : pd.DataFrame  shape (n_genes, n_samples)
              Index = HGNC gene symbols, columns = GSM sample IDs.
    meta_df : pd.DataFrame  shape (n_samples, n_features)
              Index = GSM sample IDs.  Always contains 'label': 1=relapse, 0=no relapse.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    expr_path = data_dir / "expression.csv"
    meta_path = data_dir / "metadata.csv"

    if expr_path.exists() and meta_path.exists() and not force_download:
        print("✓ Data already cached — loading from disk.")
        return _load_cached(expr_path, meta_path)

    # ── Try Zenodo first (pre-cleaned, fast) ──────────────────────────────────
    try:
        print("Downloading dataset from Zenodo …")
        expr_df, meta_df = _download_from_zenodo(data_dir)
        _save(expr_df, meta_df, expr_path, meta_path)
        return expr_df, meta_df
    except Exception as exc:
        print(f"  Zenodo unavailable ({exc}). Falling back to NCBI GEO …")

    # ── Fall back to NCBI GEO FTP ─────────────────────────────────────────────
    raw_gz = data_dir / "GSE2034_series_matrix.txt.gz"
    if not raw_gz.exists() or force_download:
        print("Downloading GSE2034 series matrix from NCBI GEO …")
        _download_file(GEO_MATRIX_URL, raw_gz)

    print("Parsing series matrix …")
    expr_df, meta_df = _parse_series_matrix(raw_gz)
    _save(expr_df, meta_df, expr_path, meta_path)
    return expr_df, meta_df


# ─── Zenodo download (primary path) ───────────────────────────────────────────

def _download_from_zenodo(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the two Zenodo files for GSE2034 and assemble clean DataFrames.

    Expression file  (GSE2034.tsv.gz)
      Rows = genes.  First 5 cols = gene annotation.  Rest = per-sample expression.

    Metadata file  (GSE2034_metadata.tsv)
      Rows = samples.  Includes 'bone_relapses_1_yes_0_no' as the label.
    """
    # ── Fetch file list from Zenodo API ───────────────────────────────────────
    api_resp = requests.get(ZENODO_API_URL, timeout=15)
    api_resp.raise_for_status()
    file_list = {f["key"]: f for f in api_resp.json().get("files", [])}

    if _ZENODO_EXPR_FILE not in file_list or _ZENODO_META_FILE not in file_list:
        raise ValueError(
            f"Expected {_ZENODO_EXPR_FILE} and {_ZENODO_META_FILE} in Zenodo record. "
            f"Found: {list(file_list.keys())}"
        )

    # ── Download expression file ──────────────────────────────────────────────
    expr_url  = file_list[_ZENODO_EXPR_FILE]["links"]["self"]
    expr_dest = data_dir / _ZENODO_EXPR_FILE
    print(f"  Downloading expression matrix ({file_list[_ZENODO_EXPR_FILE].get('size',0)//1024//1024} MB) …")
    _download_file(expr_url, expr_dest)

    # ── Download metadata file ────────────────────────────────────────────────
    meta_url  = file_list[_ZENODO_META_FILE]["links"]["self"]
    meta_dest = data_dir / _ZENODO_META_FILE
    print(f"  Downloading metadata …")
    _download_file(meta_url, meta_dest)

    # ── Parse expression TSV ──────────────────────────────────────────────────
    print("  Parsing expression matrix …")
    expr_raw = pd.read_csv(expr_dest, sep="\t", index_col=None)

    # Identify sample columns (GSM IDs) vs. annotation columns
    sample_cols = [c for c in expr_raw.columns if c not in _EXPR_ANNOT_COLS and c.startswith("GSM")]
    gene_col    = "HGNC_Symbol" if "HGNC_Symbol" in expr_raw.columns else expr_raw.columns[1]

    # Use HGNC symbol as row index; if duplicated, append a counter
    gene_ids = expr_raw[gene_col].fillna("unknown").astype(str)
    if gene_ids.duplicated().any():
        counts = gene_ids.groupby(gene_ids).cumcount()
        gene_ids = gene_ids + counts.map(lambda x: f"_{x}" if x > 0 else "")

    expr_df = expr_raw[sample_cols].copy()
    expr_df.index = gene_ids.values
    # Transpose: rows = genes, columns = samples (this is the convention we use)
    # (already in gene × sample format here)

    print(f"  Expression matrix: {expr_df.shape[0]:,} genes × {expr_df.shape[1]} samples")

    # ── Parse metadata ────────────────────────────────────────────────────────
    meta_raw = pd.read_csv(meta_dest, sep="\t")
    meta_raw = meta_raw.set_index("Sample_ID")

    # The label column is 'bone_relapses_1_yes_0_no'
    label_col = "bone_relapses_1_yes_0_no"
    if label_col not in meta_raw.columns:
        raise ValueError(f"Expected label column '{label_col}' in metadata. "
                         f"Got: {list(meta_raw.columns)}")

    meta_raw["label"] = meta_raw[label_col].astype(int)
    n1 = (meta_raw["label"] == 1).sum()
    n0 = (meta_raw["label"] == 0).sum()
    print(f"\n✓ Labels: Relapse (1) = {n1}  |  No relapse (0) = {n0}")

    # Keep only samples that exist in both the expression matrix and metadata
    common = list(set(sample_cols) & set(meta_raw.index))
    if len(common) < len(sample_cols):
        print(f"  ⚠  {len(sample_cols) - len(common)} samples dropped (not in metadata).")
    expr_df = expr_df[common]
    meta_df = meta_raw.loc[common]

    return expr_df, meta_df


# ─── NCBI GEO fall-back parser ────────────────────────────────────────────────

def _parse_series_matrix(gz_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series-matrix .txt.gz file.

    Series-matrix format:
      • Lines starting with ! are metadata.
      • Multiple !Sample_characteristics_ch1 lines: one attribute per line,
        one tab-separated value per sample.
      • Expression table lives between !series_matrix_table_begin / _end.
    """
    char_rows: list[list[str]] = []
    sample_ids: Optional[list[str]] = None
    probe_ids:  list[str]           = []
    expr_rows:  list[list[float]]   = []
    in_table = False

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            if line.startswith("!series_matrix_table_begin"):
                in_table = True;  continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False; continue

            if in_table:
                parts = line.replace('"', "").split("\t")
                if parts[0] == "ID_REF":
                    sample_ids = parts[1:]
                else:
                    probe_ids.append(parts[0])
                    try:
                        expr_rows.append(
                            [float(x) if x.strip() else np.nan for x in parts[1:]]
                        )
                    except ValueError:
                        pass
                continue

            if line.startswith("!Sample_characteristics_ch1\t"):
                char_rows.append([v.strip().strip('"') for v in line.split("\t")[1:]])
            elif line.startswith("!Sample_geo_accession\t") and sample_ids is None:
                sample_ids = [v.strip().strip('"') for v in line.split("\t")[1:]]

    if sample_ids is None or not expr_rows:
        raise RuntimeError("Could not parse series matrix — file may be corrupted.")

    n_samples = len(sample_ids)
    expr_df   = pd.DataFrame(expr_rows, index=probe_ids, columns=sample_ids[:len(expr_rows[0])])

    # Build metadata from characteristics
    per_sample: list[dict] = [{} for _ in range(n_samples)]
    for row in char_rows:
        for i, cell in enumerate(row[:n_samples]):
            if ":" in cell:
                k, v = cell.split(":", 1)
                per_sample[i][k.strip().lower().replace(" ", "_")] = v.strip()
    meta_df = pd.DataFrame(per_sample, index=sample_ids)
    meta_df = _extract_label_geo(meta_df)
    return expr_df, meta_df


def _extract_label_geo(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Find the relapse column in GEO metadata and encode as 0/1."""
    meta_df = meta_df.copy()
    for kw in ["relapse", "distant_metastasis", "metastasis", "event"]:
        hits = [c for c in meta_df.columns if kw in c.lower()]
        if hits:
            col = hits[0]
            meta_df["label"] = meta_df[col].apply(
                lambda v: 1 if str(v).strip().lower() in {"yes","1","true","positive"} else
                          0 if str(v).strip().lower() in {"no","0","false","negative"} else -1
            )
            n1 = (meta_df["label"] == 1).sum()
            n0 = (meta_df["label"] == 0).sum()
            print(f"\n✓ Labels from '{col}': Relapse={n1}  No-relapse={n0}")
            return meta_df

    # Diagnostic dump if label not found
    print("\n⚠  Could not auto-detect label. Available columns:")
    for c in meta_df.columns:
        print(f"   {c!r:<40}  e.g. {meta_df[c].iloc[0]!r}")
    meta_df["label"] = -1
    return meta_df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, chunk_size: int = 65_536) -> None:
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))
    print(f"  -> {dest.name}")


def _save(expr_df, meta_df, expr_path, meta_path):
    expr_df.to_csv(expr_path)
    meta_df.to_csv(meta_path)
    print(f"\n✓ Saved: expression.csv {expr_df.shape}  |  metadata.csv {meta_df.shape}")


def _load_cached(expr_path, meta_path):
    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    print(f"  Expression: {expr_df.shape}  |  Metadata: {meta_df.shape}")
    return expr_df, meta_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    expr, meta = get_data()
    print("\n── First 3 genes, first 3 samples ──")
    print(expr.iloc[:3, :3])
    print("\n── Label distribution ──")
    print(meta["label"].value_counts())
