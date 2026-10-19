# -*- coding: utf-8 -*-
"""
model_benchmarks.py — Addressing Class Imbalance & Multi-Model Benchmarks
=========================================================================

Scientific Objectives:
1. Address Class Imbalance (76% No-Relapse vs. 24% Relapse):
   - Implement Cost-Sensitive Learning (class_weight='balanced') to equalize
     gradient loss contributions between majority and minority classes.
   - Evaluate Precision-Recall AUC (PR-AUC / Average Precision) alongside ROC-AUC.
   - Measure Balanced Accuracy, Sensitivity (minority relapse recall), and Specificity.
2. Multi-Model Algorithmic Benchmarking:
   - Evaluate the 5-gene minimal signature across 4 distinct model families:
     a) Logistic Regression (Balanced) — Cost-sensitive linear decision boundary
     b) K-Nearest Neighbors (KNN) — Instance-based non-linear metric learning in 5D
     c) Ridge Classifier — L2-regularized linear classification
     d) Random Forest — Non-linear bagging ensemble of decision trees
   - Test out-of-distribution generalizability on Discovery (GSE2034) and
     External Cohorts (GSE11121 + GSE7390, 398 patients).
"""

import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force UTF-8 stdout/stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DATA_DIR, RANDOM_STATE, RESULTS_DIR
from src.phase4.external_validation import (
    URL_GSE11121_MATRIX,
    URL_GSE7390_MATRIX,
    clean_gene_name,
    load_gpl96_probe_map,
    process_external_cohort,
)


def load_all_cohorts(genes: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load preprocessed Discovery and External datasets sliced to candidate genes.
    """
    clean_genes = [clean_gene_name(g) for g in genes]

    # 1. Discovery Cohort (GSE2034)
    print("Loading Discovery Cohort (GSE2034)...")
    disc_expr = pd.read_csv(DATA_DIR / "expression.csv", index_col=0)
    disc_meta = pd.read_csv(DATA_DIR / "metadata.csv", index_col=0)
    y_disc = disc_meta["label"].loc[disc_expr.columns].astype(int)

    disc_clean = disc_expr.copy()
    disc_clean.index = [clean_gene_name(g) for g in disc_clean.index]
    disc_clean = disc_clean.groupby(disc_clean.index).mean().T

    # Standardize
    disc_means = disc_clean.mean(axis=0)
    disc_stds = disc_clean.std(axis=0).replace(0, 1.0)
    X_disc = (disc_clean - disc_means) / disc_stds
    X_disc_sub = X_disc[clean_genes]

    # 2. External Cohorts
    print("Loading External Cohorts (GSE11121 & GSE7390)...")
    probe_map = load_gpl96_probe_map()
    X_11121, y_11121 = process_external_cohort(URL_GSE11121_MATRIX, "GSE11121_series_matrix.txt.gz", probe_map)
    X_7390, y_7390   = process_external_cohort(URL_GSE7390_MATRIX, "GSE7390_series_matrix.txt.gz", probe_map)

    # Combined External
    comm = list(set(X_11121.columns) & set(X_7390.columns) & set(clean_genes))
    X_ext_comb = pd.concat([X_11121[comm], X_7390[comm]], axis=0)
    y_ext_comb = pd.concat([y_11121, y_7390], axis=0).astype(int)

    return X_disc_sub, y_disc, X_11121[clean_genes], y_11121, X_ext_comb[clean_genes], y_ext_comb


def run_model_benchmarks() -> pd.DataFrame:
    """
    Execute multi-model evaluation and class imbalance experiments.
    """
    print("=" * 75)
    print("RUNNING MULTI-MODEL BENCHMARKS & CLASS IMBALANCE EVALUATION")
    print("=" * 75)

    genes_5 = ["SLC35E3", "NTM", "RECQL4", "TNFRSF10C", "CPB1"]
    X_disc, y_disc, X_11121, y_11121, X_comb, y_comb = load_all_cohorts(genes_5)

    # Define model suite
    models = {
        "Logistic Regression (Unweighted)": LogisticRegression(
            C=1.0, class_weight=None, random_state=RANDOM_STATE, max_iter=1000
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
        ),
        "K-Nearest Neighbors (KNN, k=5)": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
        "Ridge Classifier (Balanced)": RidgeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100, max_depth=3, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    cohorts = {
        "Discovery (GSE2034)": (X_disc, y_disc),
        "Combined External (398 pts)": (X_comb, y_comb),
    }

    records = []
    detailed_curves = {}

    for model_name, clf in models.items():
        print(f"\n--- Model: {model_name} ---")
        # Train strictly on Discovery cohort (zero leakage)
        clf.fit(X_disc.values, y_disc.values)

        for cohort_name, (X_eval, y_eval) in cohorts.items():
            y_true = y_eval.values
            X_mat = X_eval.values

            # Get predicted probabilities or decision scores
            if hasattr(clf, "predict_proba"):
                y_probs = clf.predict_proba(X_mat)[:, 1]
            elif hasattr(clf, "decision_function"):
                df_scores = clf.decision_function(X_mat)
                # Sigmoid scaling for decision function
                y_probs = 1.0 / (1.0 + np.exp(-df_scores))
            else:
                y_probs = clf.predict(X_mat).astype(float)

            # Binary predictions
            y_pred = (y_probs >= 0.5).astype(int)

            # Metrics
            auc = float(roc_auc_score(y_true, y_probs))
            pr_auc = float(average_precision_score(y_true, y_probs))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            # Compute Precision-Recall curve
            prec, rec, _ = precision_recall_curve(y_true, y_probs)
            fpr, tpr, _ = roc_curve(y_true, y_probs)

            detailed_curves[(model_name, cohort_name)] = {
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "tpr": tpr,
                "y_true": y_true,
                "y_probs": y_probs,
                "auc": auc,
                "pr_auc": pr_auc,
                "bal_acc": bal_acc,
                "sensitivity": sens,
                "specificity": spec,
            }

            print(
                f"  {cohort_name:<28} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | "
                f"Bal-Acc: {bal_acc:.4f} | Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}%"
            )

            records.append({
                "model": model_name,
                "cohort": cohort_name,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "balanced_accuracy": bal_acc,
                "sensitivity": sens,
                "specificity": spec,
                "f1_score": f1,
                "n_samples": len(y_true),
                "prevalence": float(y_true.mean()),
            })

    results_df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "model_benchmark_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved benchmark table to: {out_csv}")

    out_pkl = RESULTS_DIR / "model_benchmark_detailed.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(detailed_curves, f)
    print(f"✓ Saved curve details to: {out_pkl}")

    return results_df


if __name__ == "__main__":
    df = run_model_benchmarks()
    print("\nSummary Table:")
    print(df[["model", "cohort", "roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]].to_string())
