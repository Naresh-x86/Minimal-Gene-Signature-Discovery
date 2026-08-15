"""
notebooks/make_figures.py — Generate all slide-ready PNG figures.

Run this AFTER run_pipeline.py has produced results/*.csv files.

Usage
-----
    python notebooks/make_figures.py

Output (in results/figures/)
------------------------------
    01_dataset_overview.png      — class distribution + expression heatmap
    02_preprocessing_funnel.png  — how many genes survive each filter stage
    03_auc_vs_signature_size.png — the key result: AUC vs. number of genes
    04_top_genes_progression.png — AUC gained at each greedy step
    05_hpc_speedup.png           — speedup chart: sequential vs CPU vs GPU
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global style ──────────────────────────────────────────────────────────────
BG      = "#0D1117"
CARD    = "#161B22"
BORDER  = "#21262D"
TEXT    = "#E6EDF3"
MUTED   = "#8B949E"
C0      = "#58A6FF"   # blue
C1      = "#3FB950"   # green
C2      = "#FF7B72"   # red/orange
C3      = "#D2A8FF"   # purple
C4      = "#FFA657"   # amber
C5      = "#F78166"   # salmon

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : CARD,
    "axes.edgecolor"    : BORDER,
    "axes.labelcolor"   : TEXT,
    "axes.titlecolor"   : TEXT,
    "text.color"        : TEXT,
    "xtick.color"       : MUTED,
    "ytick.color"       : MUTED,
    "grid.color"        : BORDER,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.6,
    "legend.facecolor"  : CARD,
    "legend.edgecolor"  : BORDER,
    "font.family"       : "sans-serif",
    "font.size"         : 12,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

def _save(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  ✓ {name}")


# ─── Figure 1: Dataset overview ────────────────────────────────────────────────

def fig_dataset_overview() -> None:
    print("\n[1] Dataset overview …")
    try:
        meta_df = pd.read_csv(DATA_DIR / "metadata.csv", index_col=0)
        expr_df = pd.read_csv(DATA_DIR / "expression.csv", index_col=0, nrows=60)
    except FileNotFoundError:
        print("  ⚠  data/ files not found — skipping.")
        return

    n0 = (meta_df["label"] == 0).sum()
    n1 = (meta_df["label"] == 1).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "GSE2034 — Breast Cancer Gene Expression Dataset",
        fontsize=16, fontweight="bold", color=TEXT, y=1.02,
    )

    # ── Left: class distribution bar chart ──
    bars = ax1.bar(["No Relapse\n(Control)", "Relapse\n(Case)"],
                   [n0, n1], color=[C1, C2], width=0.45, edgecolor="none")
    for b, n in zip(bars, [n0, n1]):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                 str(n), ha="center", fontsize=14, fontweight="bold", color=TEXT)
    ax1.set_ylabel("Number of Patients", fontsize=13)
    ax1.set_title("Sample Distribution", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(n0, n1) * 1.25)
    ax1.tick_params(bottom=False)
    ax1.text(0.5, -0.14,
             f"Total: {n0+n1} patients  ·  {n1/(n0+n1)*100:.0f}% developed distant metastasis",
             ha="center", transform=ax1.transAxes, fontsize=11, color=MUTED, style="italic")

    # ── Right: expression heatmap ──
    valid_samples = [s for s in meta_df.sort_values("label").index if s in expr_df.columns]
    hmap = expr_df[valid_samples].values[:60, :]
    hmap = np.clip(hmap, np.nanpercentile(hmap, 5), np.nanpercentile(hmap, 95))

    im = ax2.imshow(hmap, aspect="auto", cmap="RdBu_r",
                    vmin=hmap.min(), vmax=hmap.max())
    ax2.set_xlabel("Samples (sorted by outcome)", fontsize=13)
    ax2.set_ylabel("Top 60 Gene Probes", fontsize=13)
    ax2.set_title("Expression Heatmap", fontsize=14, fontweight="bold")
    ax2.set_yticks([])

    # Class separator line
    n0_sorted = sum(1 for s in valid_samples
                    if meta_df.loc[s, "label"] == 0)
    ax2.axvline(n0_sorted - 0.5, color=C4, linewidth=2, linestyle="--", alpha=0.9)
    ax2.text(n0_sorted / 2, -4, "No Relapse", ha="center",
             color=C1, fontsize=10, fontweight="bold")
    ax2.text(n0_sorted + (len(valid_samples) - n0_sorted) / 2, -4, "Relapse",
             ha="center", color=C2, fontsize=10, fontweight="bold")

    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label("Log₂ Expression", color=MUTED)

    plt.tight_layout()
    _save("01_dataset_overview.png")


# ─── Figure 2: Preprocessing funnel ────────────────────────────────────────────

def fig_preprocessing_funnel() -> None:
    print("\n[2] Preprocessing funnel …")
    try:
        stats = pd.read_csv(RESULTS_DIR / "dataset_stats.csv",
                            index_col=0, header=None, names=["value"])
        n_raw  = int(stats.loc["n_probes_raw", "value"])
        n_filt = int(stats.loc["n_probes_after_var_filter", "value"])
    except Exception:
        n_raw, n_filt = 22_283, 17_800
    n_cands = 500   # from config

    labels  = ["Raw probes\n(microarray chip)",
                "After variance\nfilter (−20%)",
                "Statistical\ncandidates (top 500)"]
    counts  = [n_raw, n_filt, n_cands]
    colors  = [C0, C1, C3]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.suptitle("Gene Candidate Reduction Pipeline",
                 fontsize=16, fontweight="bold", color=TEXT)

    bars = ax.barh(range(3), counts, color=colors, height=0.5, edgecolor="none")
    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + n_raw * 0.01, i,
                f"{n:,}", va="center", fontsize=14, fontweight="bold", color=TEXT)

    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Number of Gene Features", fontsize=13)
    ax.set_xlim(0, n_raw * 1.2)
    ax.invert_yaxis()

    # Reduction labels
    for y_mid, (a, b) in zip([0.5, 1.5], [(n_raw, n_filt), (n_filt, n_cands)]):
        pct = (a - b) / a * 100
        ax.text(n_raw * 0.6, y_mid, f"−{pct:.0f}%", ha="center", fontsize=12,
                color=MUTED, style="italic")

    plt.tight_layout()
    _save("02_preprocessing_funnel.png")


# ─── Figure 3: AUC vs. signature size (the KEY result) ────────────────────────

def fig_auc_vs_size() -> None:
    print("\n[3] AUC vs. signature size …")
    try:
        df = pd.read_csv(RESULTS_DIR / "sequential_search_results.csv")
    except FileNotFoundError:
        print("  ⚠  sequential_search_results.csv not found — skipping.")
        return

    n_genes = df["step"].values
    val_auc = df["val_auc"].values
    max_auc = val_auc.max()

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("How Few Genes Do We Actually Need?",
                 fontsize=17, fontweight="bold", color=TEXT)

    ax.plot(n_genes, val_auc, color=C0, linewidth=2.5, zorder=3)
    ax.fill_between(n_genes, val_auc, alpha=0.12, color=C0)
    ax.scatter(n_genes, val_auc, color=C0, s=60, zorder=4)

    # Reference line: full-feature performance (last data point)
    ax.axhline(max_auc, color=C4, linewidth=1.5, linestyle=":", alpha=0.85)
    ax.text(n_genes[-1] * 0.02, max_auc + 0.004,
            f"Best AUC = {max_auc:.3f}", color=C4, fontsize=11)

    # Mark the "95% of max" point
    threshold = 0.95 * max_auc
    hits = np.where(val_auc >= threshold)[0]
    if len(hits):
        ge_n, ge_auc = n_genes[hits[0]], val_auc[hits[0]]
        ax.scatter([ge_n], [ge_auc], s=250, marker="*", color=C1, zorder=5)
        ax.annotate(
            f"≥95% of max AUC\nwith just {ge_n} gene{'s' if ge_n > 1 else ''}",
            xy=(ge_n, ge_auc),
            xytext=(ge_n + max(1, len(n_genes) // 8), ge_auc - 0.025),
            color=C1, fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C1, lw=1.5),
        )

    ax.set_xlabel("Number of Genes in Signature", fontsize=13)
    ax.set_ylabel("Test AUC  (Logistic Regression, held-out set)", fontsize=13)
    ax.set_ylim(max(0.4, val_auc.min() - 0.06), min(1.02, max_auc + 0.06))
    ax.set_xticks(n_genes)
    ax.set_xticklabels(n_genes, fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.text(0.5, -0.12,
            "Each point = one more gene added to the signature via greedy forward selection.",
            ha="center", transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    plt.tight_layout()
    _save("03_auc_vs_signature_size.png")


# ─── Figure 4: Top genes progression ──────────────────────────────────────────

def fig_top_genes() -> None:
    print("\n[4] Top genes progression …")
    try:
        df = pd.read_csv(RESULTS_DIR / "sequential_search_results.csv")
    except FileNotFoundError:
        print("  ⚠  sequential_search_results.csv not found — skipping.")
        return

    n_show = min(20, len(df))
    df     = df.head(n_show)

    # Label: use gene_id if available, else "Gene #index"
    if "gene_id" in df.columns:
        labels = df["gene_id"].apply(lambda x: str(x)[:12])
    else:
        labels = [f"Gene {r.gene_index}" for _, r in df.iterrows()]

    # Colour by incremental improvement in val_auc
    increments = np.diff(df["val_auc"].values, prepend=df["val_auc"].values[0])
    norm_inc   = (increments - increments.min()) / max(increments.max() - increments.min(), 1e-6)
    colors     = plt.cm.YlOrRd(0.3 + 0.7 * norm_inc)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.suptitle("Greedy Selection: AUC as Each Gene Is Added",
                 fontsize=16, fontweight="bold", color=TEXT)

    x    = np.arange(n_show)
    bars = ax.bar(x, df["val_auc"].values, color=colors, width=0.7, edgecolor="none")
    for bar, auc in zip(bars, df["val_auc"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.3f}", ha="center", fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative Test AUC", fontsize=13)
    ax.set_xlabel("Gene Selected at Each Step", fontsize=13)
    lo = max(0.35, df["val_auc"].min() - 0.07)
    hi = min(1.02, df["val_auc"].max() + 0.06)
    ax.set_ylim(lo, hi)

    ax.text(0.5, -0.22,
            "Each bar is the test AUC achieved by the signature after that gene is added.",
            ha="center", transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

    plt.tight_layout()
    _save("04_top_genes_progression.png")


# ─── Figure 5: HPC speedup ─────────────────────────────────────────────────────

def fig_hpc_speedup() -> None:
    print("\n[5] HPC speedup …")
    bench_path = RESULTS_DIR / "benchmark_results.csv"
    if not bench_path.exists():
        print("  ⚠  benchmark_results.csv not found — run without --skip-benchmark.")
        return

    df  = pd.read_csv(bench_path)
    n_s = sorted(df["n_candidates"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "HPC Scaling: Candidate Scoring Benchmark\n"
        "i7-13650HX  ·  NVIDIA RTX 4060  ·  286 samples",
        fontsize=14, fontweight="bold", color=TEXT,
    )

    style = {
        "sequential"  : dict(color=MUTED,  linestyle="--", marker="o", lw=2),
        "parallel_cpu": dict(color=C0,     linestyle="-",  marker="s", lw=2),
        "gpu"         : dict(color=C2,     linestyle="-",  marker="^", lw=2.5),
    }

    cpu_workers_to_show = {1, 4, max(df.loc[df.implementation=="parallel_cpu","n_workers"])}

    for impl, grp in df.groupby("implementation"):
        s = style.get(impl, {})
        if impl == "parallel_cpu":
            for nw, sub in grp.groupby("n_workers"):
                if nw not in cpu_workers_to_show:
                    continue
                label = f"CPU {nw}-core"
                # Dim lower-core lines
                alpha = 0.55 if nw == 1 else 1.0
                ax1.plot(sub.n_candidates, sub.mean_s,
                         label=label, alpha=alpha, **s)
                ax2.plot(sub.n_candidates, sub.speedup,
                         label=label, alpha=alpha, **s)
        else:
            label = "Sequential (baseline)" if impl == "sequential" else f"GPU (RTX 4060)"
            ax1.plot(grp.n_candidates, grp.mean_s, label=label, **s)
            if impl != "sequential":
                ax2.plot(grp.n_candidates, grp.speedup, label=label, **s)

    for ax, ylabel, title in [
        (ax1, "Scoring Time (seconds)", "Runtime"),
        (ax2, "Speedup  (× faster than sequential)", "Speedup over Sequential"),
    ]:
        ax.set_xlabel("N Candidates Scored Simultaneously", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax1.set_yscale("log")
    ax2.axhline(1, color=MUTED, linewidth=1, linestyle=":", alpha=0.7)
    ax2.text(n_s[0] * 1.1, 1.05, "baseline (1×)", color=MUTED, fontsize=10)

    plt.tight_layout()
    _save("05_hpc_speedup.png")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Generating slide-ready figures …")
    print("=" * 55)

    fig_dataset_overview()
    fig_preprocessing_funnel()
    fig_auc_vs_size()
    fig_top_genes()
    fig_hpc_speedup()

    print(f"\n✓ All figures saved to: {FIG_DIR}")
    print("  Copy the PNGs directly into your presentation slides.")
