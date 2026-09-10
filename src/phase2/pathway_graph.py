# -*- coding: utf-8 -*-
"""
pathway_graph.py -- Biological Gene Interaction Graph for Phase 2.

What is this file doing?
------------------------
Genes don't work alone. Inside a cell, Gene A can turn on Gene B,
Gene B can block Gene C, and Gene C can activate Gene D. These
relationships form a network called a "biological pathway."

Example (from this dataset):
  HLA-DRA --presents-antigen-to--> CD4
  CD4     --activates--> LCK
  LCK     --phosphorylates--> FYN
  FYN     --activates--> downstream signaling

This network is called the "T-Cell Receptor Signaling Pathway."
All four genes (HLA-DRA, CD4, LCK, FYN) happen to be in our top
500 candidate genes! That's not a coincidence -- they are all
genuinely differentially expressed between cancer and normal breast.

Why does this matter for gene signature discovery?
---------------------------------------------------
Two gene signatures might have identical AUC:
  - Signature A: 20 random, disconnected genes
  - Signature B: 20 genes from 3 known cancer pathways

Signature B is MUCH more meaningful biologically. It tells a coherent
story: "These genes are all part of the immune response pathway, and
when they're dysregulated together, cancer escapes detection."

The pathway bonus in the GA nudges the search toward Signature B
without sacrificing predictive power. It adds a tiny bonus (up to 0.05
AUC points) to signatures whose genes are connected in known pathways.

The Graph (explained for beginners)
-------------------------------------
A "graph" in computer science is just:
  - A set of NODES (our case: gene names like "CD44", "LCK")
  - A set of EDGES (our case: known interactions, e.g. "CD4 interacts with LCK")

We represent this as a Python dictionary:
  adjacency = {
      "CD4":  {"LCK", "FYN", "CD3E", ...},
      "LCK":  {"CD4", "FYN", "CD8A", ...},
      ...
  }

adjacency["CD4"] is the SET of genes that CD4 is known to interact with.

Data Source
-----------
data/pathway_interactions.csv is a hand-curated file of ~160 gene-gene
interactions derived from KEGG pathway database and published literature.
All gene names are verified to exist in the GSE2034 top-500 candidate set.

Key pathways covered:
  - MHC Class II Antigen Presentation (KEGG hsa04612)
  - T-Cell Receptor Signaling         (KEGG hsa04660)
  - NK and Cytotoxic T-Cell Response  (KEGG hsa04650)
  - ECM-Receptor Interaction          (KEGG hsa04512)
  - Breast Cancer ER+ Signaling       (multiple publications)
  - PI3K-AKT Signaling                (KEGG hsa04151)
  - MAPK Signaling                    (KEGG hsa04010)
  - Apoptosis                         (KEGG hsa04210)
  - Mitochondrial Oxidative Phosphorylation (KEGG hsa00190)
"""

from pathlib import Path
from typing  import Dict, Set, List

import numpy  as np
import pandas as pd

# Path to the curated interaction table (shipped with the repo)
_DATA_DIR     = Path(__file__).parent.parent.parent / "data"
PATHWAY_CSV   = _DATA_DIR / "pathway_interactions.csv"
PATHWAY_BONUS = 0.05  # maximum AUC-point bonus for perfect pathway connectivity


# =============================================================================
# Build the gene interaction graph
# =============================================================================

def load_pathway_graph(csv_path: Path = PATHWAY_CSV) -> Dict[str, Set[str]]:
    """
    Read the pathway interaction CSV and build an adjacency dictionary.

    The graph is UNDIRECTED -- if Gene A interacts with Gene B,
    then B also interacts with A. This matches how gene interactions
    work: if LCK phosphorylates CD4, then CD4 is affected by LCK,
    and we count them as connected in both directions.

    Returns
    -------
    adjacency : dict[str, set[str]]
        adjacency[gene_name] = set of all genes that interact with gene_name

    Example
    -------
    >>> graph = load_pathway_graph()
    >>> "LCK" in graph["CD4"]
    True
    >>> "CD4" in graph["LCK"]
    True   # undirected: both directions are stored
    """
    if not csv_path.exists():
        print(f"  WARNING: Pathway file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    adjacency: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        a, b = str(row["gene_a"]).strip(), str(row["gene_b"]).strip()
        # Add both directions (undirected graph)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    return adjacency


def pathway_connectivity_score(gene_names: List[str],
                                adjacency: Dict[str, Set[str]]) -> float:
    """
    Compute what fraction of the selected genes are connected to at least one
    other selected gene in the pathway graph.

    Example (5-gene signature):
        selected = ["CD4", "LCK", "FYN", "SOD2", "ACBD3"]
        - CD4  connects to LCK (yes!) and FYN (yes!)  -> connected
        - LCK  connects to CD4 (yes!) and FYN (yes!)  -> connected
        - FYN  connects to CD4 (yes!) and LCK (yes!)  -> connected
        - SOD2 connects to NDUFA2 (not selected)      -> NOT connected
        - ACBD3 connects to nothing in our graph       -> NOT connected
        -> 3 out of 5 genes are connected = score 0.60

    Parameters
    ----------
    gene_names : list of gene name strings (the selected signature)
    adjacency  : the graph dict from load_pathway_graph()

    Returns
    -------
    float in [0.0, 1.0]
        0.0 = no selected genes have any pathway neighbors among selected genes
        1.0 = every selected gene has at least one pathway neighbor in the set
    """
    if not adjacency or not gene_names:
        return 0.0

    selected_set = set(gene_names)
    n_connected  = 0

    for gene in selected_set:
        neighbors = adjacency.get(gene, set())
        # Check if any neighbor is also in the selected signature
        if neighbors & selected_set:  # set intersection
            n_connected += 1

    return n_connected / len(selected_set)


def pathway_fitness_bonus(gene_names: List[str],
                          adjacency: Dict[str, Set[str]]) -> float:
    """
    Convert a pathway connectivity score into an AUC bonus.

    bonus = PATHWAY_BONUS * connectivity_score
          = 0.05 * (fraction of selected genes connected in pathways)

    Design rationale:
    - The bonus is intentionally SMALL (max 0.05 AUC points).
    - Biology informs but does not override the data signal (AUC).
    - A signature with AUC=0.77 and 0 pathway connections will always
      beat a signature with AUC=0.59 and perfect pathway connections.
    - Two signatures with similar AUC (~0.002 difference) will be
      broken in favour of the one with better pathway connectivity.

    Returns
    -------
    float: AUC bonus in [0.0, PATHWAY_BONUS]
    """
    score = pathway_connectivity_score(gene_names, adjacency)
    return PATHWAY_BONUS * score


# =============================================================================
# Summary stats for logging
# =============================================================================

def describe_graph(adjacency: Dict[str, Set[str]]) -> None:
    """Print a human-readable summary of the pathway graph."""
    if not adjacency:
        print("  Pathway graph: empty (no data loaded)")
        return

    n_genes = len(adjacency)
    n_edges = sum(len(v) for v in adjacency.values()) // 2  # undirected: count once
    degrees = [len(v) for v in adjacency.values()]

    print(f"\n  Pathway Graph Summary:")
    print(f"    Genes (nodes)      : {n_genes}")
    print(f"    Interactions (edges): {n_edges}")
    print(f"    Avg connections/gene: {np.mean(degrees):.1f}")
    print(f"    Most connected gene : "
          f"{max(adjacency, key=lambda g: len(adjacency[g]))} "
          f"({max(degrees)} connections)")


def score_signature_with_pathway(gene_names: List[str],
                                 adjacency: Dict[str, Set[str]]) -> None:
    """
    Print a detailed pathway analysis for a given signature.

    Useful for understanding WHICH genes in a final signature are part
    of known cancer pathways, and what their connections are.
    """
    if not adjacency or not gene_names:
        print("  No pathway data available.")
        return

    selected_set = set(gene_names)
    print(f"\n  Pathway analysis for {len(gene_names)}-gene signature:")
    print(f"  {'Gene':<20} {'In Graph?':<12} {'Connected Partners in Signature'}")
    print(f"  {'-'*70}")

    n_connected = 0
    for gene in sorted(gene_names):
        neighbors = adjacency.get(gene, set())
        in_graph  = gene in adjacency
        partners  = sorted(neighbors & selected_set - {gene})
        if partners:
            n_connected += 1
            partner_str = ", ".join(partners)
        else:
            partner_str = "(none in this signature)"
        flag = "*" if partners else " "
        print(f"  {flag} {gene:<19} {'Yes' if in_graph else 'No':<12} {partner_str}")

    connectivity = n_connected / len(gene_names)
    bonus        = PATHWAY_BONUS * connectivity
    print(f"\n  Pathway connectivity : {n_connected}/{len(gene_names)} genes "
          f"= {connectivity:.1%}")
    print(f"  Pathway bonus (max {PATHWAY_BONUS}): {bonus:.4f} AUC points")
