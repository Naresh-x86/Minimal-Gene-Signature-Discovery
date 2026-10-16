# -*- coding: utf-8 -*-
"""
genetic_search.py -- Genetic Algorithm for minimal gene signature discovery.

This is the Phase 2 upgrade to the Greedy Forward Selection from Phase 1.

Why a Genetic Algorithm?
------------------------
Greedy search commits permanently. Once it picks Gene X at Step 3, it is
stuck with Gene X forever, even if Gene X turns out to be a bad choice in
the context of the bigger signature. This is called a "local minimum" trap.

A Genetic Algorithm (GA) avoids this by working with a whole *population*
of candidate signatures at the same time and letting them evolve.

Think of it like a nature documentary about survival:
  - Phase 1 (Greedy): One mountain climber, always steps uphill.
    Gets permanently stuck on the first peak it reaches.
  - Phase 2 (GA):     100 climbers spread across the entire mountain range.
    Bad climbers don't survive. Good ones breed and share their best routes.
    The population slowly converges on the highest peak in the whole range.

The Algorithm (step by step)
-----------------------------
1. INITIALIZE  - Create 100 random gene signatures ("the population").
                 Each signature has exactly 20 genes chosen at random
                 from the 500 statistical candidates.

2. FITNESS     - Score every signature using AUC. This is the "survival
                 of the fittest" test -- higher AUC = better signature.

3. SELECTION   - Keep only the top 30 best-scoring signatures ("elites").
                 The bottom 70 are eliminated.

4. CROSSOVER   - Randomly pair up the survivors. Each pair produces a
                 "child" signature by mixing genes from both parents.

5. MUTATION    - With a 5% chance per gene, randomly swap one gene in a
                 child for a new one. This prevents the whole population
                 from becoming identical clones.

6. REPEAT      - New population = 30 survivors + 70 new children.
                 Go back to step 2. Repeat for 200 "generations".

7. REPORT      - Return the best signature ever seen across all generations.

HPC Comparison (the key teaching point)
-----------------------------------------
The FITNESS step (Step 2) is "embarrassingly parallel" -- scoring
individual #1 is 100% independent from scoring individual #2.
We implement three versions to compare:

  - Sequential:    Score individual 1, then 2, then 3... (1 CPU core)
  - Parallel CPU:  Spread 100 individuals across all 14 CPU cores
  - GPU:           Score ALL 100 individuals in ONE tensor operation

Unlike Phase 1, each fitness evaluation here involves ~20 genes x 228
patients of matrix math. This gives the CPU parallel version a much
fairer fight than Phase 1's tiny inner loop -- making this a richer
HPC comparison.
"""

import time
import numpy as np
import pandas as pd
from typing import List
from joblib import Parallel, delayed

from src.config            import RANDOM_STATE
from src.utils             import auc_from_scores, final_lr_auc
from src.phase2.pathway_graph import load_pathway_graph, pathway_fitness_bonus, describe_graph, score_signature_with_pathway

# ── GA hyperparameters ────────────────────────────────────────────────────────
# All these live here so they are easy to find and change.
GA_POPULATION_SIZE = 100    # Number of candidate signatures alive each generation
GA_GENERATIONS     = 200    # Number of rounds of evolution
GA_ELITE_FRACTION  = 0.30   # Top 30% survive each generation (30 out of 100)
GA_MUTATION_RATE   = 0.05   # 5% chance any individual gene gets randomly swapped
GA_SIGNATURE_SIZE  = 20     # Fixed number of genes in every signature

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        _GPU_NAME = torch.cuda.get_device_name(0)
except ImportError:
    TORCH_AVAILABLE = False
    _GPU_NAME = "N/A"


# =============================================================================
# STEP 2 HELPER: Score one individual
# =============================================================================

def _score_individual(gene_indices: np.ndarray,
                      X: np.ndarray,
                      y: np.ndarray) -> float:
    """
    Compute the AUC fitness score for a single gene signature.

    How it works:
    - gene_indices is a list of 20 column indices (each = one gene)
    - X[:, gene_indices] selects just those 20 gene columns from the expression matrix
    - .sum(axis=1) adds them together for each patient -> one "Risk Score" per patient
    - auc_from_scores() checks: do relapse patients tend to have higher Risk Scores?

    Example with 3 genes and 5 patients:
        X[:, genes] = [[1.2, 0.8, 2.1],    <- Patient 1
                       [0.1, 0.2, 0.3],    <- Patient 2
                       ...]
        Risk Scores = [4.1, 0.6, ...]
        AUC = how well these scores separate relapse vs no-relapse
    """
    proj = X[:, gene_indices].sum(axis=1)
    return auc_from_scores(proj, y)


# =============================================================================
# STEP 4: Crossover (breeding)
# =============================================================================

def _crossover(parent_a: np.ndarray,
               parent_b: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """
    Combine two parent signatures into one child signature.

    Strategy:
    - If signature size is 1, randomly inherit from parent A or parent B
    - Randomly decide split point between 1 and size-1
    - Fill remaining slots from Parent B, avoiding duplicates
    """
    size = len(parent_a)
    if size <= 1:
        return parent_a.copy() if rng.random() < 0.5 else parent_b.copy()

    split = int(rng.integers(1, size))                  # random split point
    child_set = set(parent_a[:split].tolist())
    # Fill remaining slots from parent_b (skipping genes already in child_set)
    for g in parent_b:
        if len(child_set) >= size:
            break
        child_set.add(int(g))
    # If still short (very rare edge case), pad from parent_a's remainder
    for g in parent_a[split:]:
        if len(child_set) >= size:
            break
        child_set.add(int(g))
    return np.array(sorted(child_set)[:size], dtype=np.int32)


# =============================================================================
# STEP 5: Mutation
# =============================================================================

def _mutate(individual: np.ndarray,
            n_cands: int,
            mutation_rate: float,
            rng: np.random.Generator) -> np.ndarray:
    """
    Randomly swap out genes in an individual with a small probability.

    Without mutation, the whole population would eventually become identical
    (because only the "winning" genes get passed to children). Mutation
    ensures that new, unexplored genes always have a chance to enter the pool.

    Parameters
    ----------
    individual    : array of gene indices (the signature to potentially mutate)
    n_cands       : total number of candidate genes available (500)
    mutation_rate : probability of mutating any single gene (5% = 0.05)
    rng           : random number generator for reproducibility
    """
    mutated     = individual.copy()
    current_set = set(mutated.tolist())

    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            # Pick a new gene that is NOT already in the signature
            available = list(set(range(n_cands)) - current_set)
            if not available:
                continue
            new_gene = int(rng.choice(available))
            current_set.discard(mutated[i])
            current_set.add(new_gene)
            mutated[i] = new_gene

    return mutated


# =============================================================================
# STEP 2 (HPC version A): Sequential fitness evaluation
# =============================================================================

def _fitness_sequential(population: np.ndarray,
                        X: np.ndarray,
                        y: np.ndarray) -> np.ndarray:
    """
    Score all individuals ONE BY ONE using a single CPU core.

    This is the baseline. No parallelism. Simple Python for-loop.
    100 individuals -> 100 separate function calls, one after the other.
    """
    return np.array([_score_individual(ind, X, y) for ind in population])


# =============================================================================
# STEP 2 (HPC version B): Parallel CPU fitness evaluation
# =============================================================================

def _fitness_parallel(population: np.ndarray,
                      X: np.ndarray,
                      y: np.ndarray,
                      n_jobs: int = -1) -> np.ndarray:
    """
    Score all individuals by spreading them across all CPU cores.

    Joblib distributes the 100 individuals across however many cores your
    CPU has. On the i7-13650HX (14 cores), about 7 individuals run on each
    core simultaneously.

    Unlike Phase 1, each individual here involves computing AUC over 20 genes
    x 228 samples -- which is more meaningful work per task. This gives the
    CPU parallel version a real chance to beat sequential.

    n_jobs = -1 means "use every available CPU core".
    """
    scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_score_individual)(ind, X, y)
        for ind in population
    )
    return np.array(scores)


# =============================================================================
# STEP 2 (HPC version C): GPU fitness evaluation
# =============================================================================

def _fitness_gpu(population: np.ndarray,
                 X: np.ndarray,
                 y: np.ndarray) -> np.ndarray:
    """
    Score ALL 100 individuals in a single GPU matrix operation.

    The Key Idea (don't panic -- read slowly):
    -----------
    Instead of 100 separate function calls, we build ONE big matrix:
      - A "mask" matrix of shape [100, 500]:
        mask[i, j] = 1  if gene j is in individual i's signature
        mask[i, j] = 0  otherwise
      - X (expression data) has shape [228 patients, 500 genes]

    One matrix multiply:
        mask @ X.T  ->  shape [100, 228]
    gives us the Risk Score for every patient, for every individual,
    all at once. That's 100 x 228 = 22,800 numbers computed in one shot.

    Then we compute AUC for all 100 rows simultaneously using broadcasting
    (the same trick Phase 1's GPU search used).

    Why the GPU wins here:
    This is pure matrix math -- exactly what GPUs are designed for.
    The RTX 4060's 3,072 cores can crunch the entire [100, 228] result
    matrix far faster than 14 CPU cores doing 100 separate loops.
    """
    import torch
    device   = torch.device("cuda")
    n_pop    = len(population)
    n_cands  = X.shape[1]

    # Build the binary mask on the GPU
    # mask[i, j] = 1 if gene j is in individual i's signature
    mask  = torch.zeros(n_pop, n_cands, dtype=torch.float32, device=device)
    pop_t = torch.from_numpy(population).long().to(device)
    mask.scatter_(1, pop_t, 1.0)

    # Expression matrix on GPU: [n_samples, n_cands]
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    # [n_pop, n_cands] @ [n_cands, n_samples] = [n_pop, n_samples]
    # scores_all[i, s] = risk score of patient s under individual i's signature
    scores_all = mask @ X_t.T

    # Batched AUC via broadcasting
    y_t  = torch.from_numpy(y.astype(np.float32)).to(device)
    pos  = scores_all[:, y_t == 1]   # [n_pop, n_pos]
    neg  = scores_all[:, y_t == 0]   # [n_pop, n_neg]

    # [n_pop, n_pos, 1] > [n_pop, 1, n_neg]  ->  [n_pop, n_pos, n_neg]
    wins = (pos.unsqueeze(2) > neg.unsqueeze(1)).float()
    auc  = wins.mean(dim=(1, 2))     # [n_pop]

    # auc_from_scores always returns max(auc, 1-auc) so mirror that here
    auc = torch.maximum(auc, 1.0 - auc)
    return auc.cpu().numpy()


# =============================================================================
# STEP 2D: C + OpenMP fitness evaluation (Phase 3)
# =============================================================================

def _fitness_c(population: np.ndarray,
               X: np.ndarray,
               y: np.ndarray) -> np.ndarray:
    """
    Score ALL 100 individuals using the compiled C + OpenMP kernel.
    Passes NumPy memory directly to C with zero copying.
    """
    from src.phase3.cpp_search import evaluate_population_c
    return evaluate_population_c(X, population, y)


# =============================================================================
# MAIN ENTRY POINT: run the full GA
# =============================================================================

# =============================================================================
# MAIN ENTRY POINT: run the full GA
# =============================================================================

def run_ga(X_train: np.ndarray,
           y_train: np.ndarray,
           X_test:  np.ndarray,
           y_test:  np.ndarray,
           cand_ids: List[str],
           sig_size: int = GA_SIGNATURE_SIZE,
           generations: int = GA_GENERATIONS,
           population_size: int = GA_POPULATION_SIZE,
           mode: str = "sequential",
           use_pathway: bool = False,
           verbose: bool = True,
           n_jobs: int = -1,
           seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Run the Genetic Algorithm to find the best gene signature of size `sig_size`.

    Parameters
    ----------
    X_train, y_train : training expression matrix + labels (already sliced to candidates)
                       Shape: (228 samples, 500 candidates)
    X_test, y_test   : held-out test data (58 samples, 500 candidates)
    cand_ids         : the 500 candidate gene names (strings)
    sig_size         : number of genes in the signature (default 20)
    generations      : number of evolutionary rounds (default 200)
    population_size  : number of signatures in the pool (default 100)
    mode             : 'sequential' | 'parallel' | 'gpu' | 'c_openmp'
    use_pathway      : whether to add biological pathway bonus (default False)
    verbose          : whether to print progress
    n_jobs           : CPU cores for parallel mode (-1 = all cores)
    seed             : random seed (set for reproducibility)

    Returns
    -------
    pd.DataFrame with generation history:
        generation, best_train_auc, best_val_auc, elapsed_s
    (Attributes df.attrs['best_individual'] and df.attrs['best_genes'] also populated)
    """
    rng     = np.random.default_rng(seed)
    n_cands = len(cand_ids)
    n_elite = max(2, int(population_size * GA_ELITE_FRACTION))
    
    # Load biological pathway graph if requested
    graph = None
    if use_pathway:
        graph = load_pathway_graph()
        if verbose:
            describe_graph(graph)

    # Choose the fitness evaluation function
    if mode in ("c_openmp", "cpp", "c"):
        try:
            from src.phase3.cpp_search import get_num_threads
            threads = get_num_threads()
            _evaluate    = lambda pop: _fitness_c(pop, X_train, y_train)
            device_label = f"C + OpenMP ({threads} threads)"
        except Exception as e:
            if verbose:
                print(f"  [WARNING] C+OpenMP failed ({e}), falling back to sequential.")
            _evaluate    = lambda pop: _fitness_sequential(pop, X_train, y_train)
            device_label = "Sequential (1-core fallback)"
    elif mode == "gpu" and TORCH_AVAILABLE:
        _evaluate    = lambda pop: _fitness_gpu(pop, X_train, y_train)
        device_label = f"GPU ({_GPU_NAME})"
    elif mode == "parallel":
        _evaluate    = lambda pop: _fitness_parallel(pop, X_train, y_train, n_jobs)
        device_label = f"Parallel CPU (n_jobs={n_jobs})"
    else:
        _evaluate    = lambda pop: _fitness_sequential(pop, X_train, y_train)
        device_label = "Sequential (1-core)"

    if verbose:
        print(f"\n  Mode        : {device_label}")
        print(f"  Population  : {population_size}  |  Generations: {generations}")
        print(f"  Signature   : {sig_size} genes  |  Candidates: {n_cands}")
        print(f"  Elite keep  : {n_elite} ({GA_ELITE_FRACTION*100:.0f}%)  "
              f"|  Mutation rate: {GA_MUTATION_RATE*100:.0f}%")

    # ── STEP 1: INITIALIZE ────────────────────────────────────────────────────
    population = np.array([
        rng.choice(n_cands, size=sig_size, replace=False)
        for _ in range(population_size)
    ], dtype=np.int32)

    best_individual = population[0].copy()
    best_train_auc  = 0.0
    records         = []
    t_start         = time.perf_counter()

    for gen in range(generations):
        t_gen = time.perf_counter()

        # ── STEP 2: FITNESS ───────────────────────────────────────────────────
        base_fitness = _evaluate(population)
        
        if graph:
            bonus = np.zeros(population_size)
            for i, ind in enumerate(population):
                gene_names = [cand_ids[idx] for idx in ind]
                bonus[i]   = pathway_fitness_bonus(gene_names, graph)
            fitness = base_fitness + bonus
        else:
            fitness = base_fitness

        # ── STEP 3: SELECTION ─────────────────────────────────────────────────
        ranked      = np.argsort(fitness)[::-1]
        elites      = population[ranked[:n_elite]]
        top_fitness = fitness[ranked[0]]

        if top_fitness > best_train_auc:
            best_train_auc  = top_fitness
            best_individual = elites[0].copy()

        # Evaluate best on held-out test set
        best_X_train = X_train[:, best_individual]
        best_X_test  = X_test[:, best_individual]
        val_auc = final_lr_auc(best_X_train, y_train, best_X_test, y_test)

        elapsed = time.perf_counter() - t_gen
        records.append({
            "generation"    : gen,
            "best_train_auc": round(float(best_train_auc), 6),
            "best_val_auc"  : round(float(val_auc),        6),
            "elapsed_s"     : round(elapsed,                4),
        })

        if verbose and (gen % 25 == 0 or gen == generations - 1):
            total_s = time.perf_counter() - t_start
            print(f"  Gen {gen:>3d}  best_train={best_train_auc:.4f}  "
                  f"val_AUC={val_auc:.4f}  elapsed={total_s:.1f}s")

        # ── STEPS 4 & 5: CROSSOVER + MUTATION ────────────────────────────────
        children = []
        while len(children) < population_size - n_elite:
            p_a, p_b = elites[rng.choice(n_elite, size=2, replace=False)]
            child    = _crossover(p_a, p_b, rng)
            child    = _mutate(child, n_cands, GA_MUTATION_RATE, rng)
            children.append(child)

        population = np.vstack([elites, np.array(children, dtype=np.int32)])

    best_gene_names = [cand_ids[i] for i in best_individual]
    if verbose:
        print(f"\n  Best signature genes: {best_gene_names}")
        print(f"  Best train AUC: {best_train_auc:.4f}  |  Best val AUC: {records[-1]['best_val_auc']:.4f}")
        if graph:
            score_signature_with_pathway(best_gene_names, graph)

    df_out = pd.DataFrame(records)
    df_out.attrs["best_individual"] = best_individual
    df_out.attrs["best_genes"] = best_gene_names
    return df_out


# =============================================================================
# MINIMAL GENE SIGNATURE DISCOVERY: Size Sweep (k = 1 to max_size)
# =============================================================================

def run_ga_size_sweep(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test:  np.ndarray,
                      y_test:  np.ndarray,
                      cand_ids: List[str],
                      max_size: int = 30,
                      generations_per_size: int = 150,
                      population_size: int = 100,
                      mode: str = "c_openmp",
                      seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Search for the optimal minimal gene signature across all sizes k from 1 to max_size.

    For each signature size k:
      - Runs Genetic Algorithm to explore the combinatorial space of signatures of size k
      - Computes out-of-sample Logistic Regression test-set AUC
      - Compares against Phase 1 Greedy Selection results

    Returns
    -------
    pd.DataFrame with columns:
        signature_size, ga_train_auc, ga_val_auc, greedy_val_auc, ga_advantage, genes, elapsed_s
    """
    from src.config import RESULTS_DIR
    greedy_csv = RESULTS_DIR / "sequential_search_results.csv"
    greedy_map = {}
    if greedy_csv.exists():
        gdf = pd.read_csv(greedy_csv)
        for _, row in gdf.iterrows():
            step_k = int(row.get("step", 0))
            if step_k > 0:
                greedy_map[step_k] = float(row.get("val_auc", 0.5))

    print("\n" + "=" * 65)
    print("  MINIMAL GENE SIGNATURE DISCOVERY: GA SIZE SWEEP (k = 1..30)")
    print("=" * 65)
    print(f"  Searching across sizes 1 to {max_size} using engine: {mode}")
    print(f"  Generations per size: {generations_per_size}  |  Population: {population_size}\n")

    results = []
    t_sweep_start = time.perf_counter()

    for k in range(1, max_size + 1):
        t0 = time.perf_counter()
        
        # Run GA for signature of size k (pure evolutionary search, no pathway bonus)
        df_ga = run_ga(
            X_train, y_train, X_test, y_test,
            cand_ids=cand_ids,
            sig_size=k,
            generations=generations_per_size,
            population_size=population_size,
            mode=mode,
            use_pathway=False,
            verbose=False,
            seed=seed + k,
        )
        
        elapsed_k = time.perf_counter() - t0
        best_ind = df_ga.attrs["best_individual"]
        best_genes = [cand_ids[i] for i in best_ind]
        
        # Test AUC via standard Logistic Regression
        val_auc = float(df_ga.iloc[-1]["best_val_auc"])
        train_auc = float(df_ga.iloc[-1]["best_train_auc"])
        greedy_auc = greedy_map.get(k, np.nan)
        adv = val_auc - greedy_auc if not np.isnan(greedy_auc) else 0.0

        adv_str = f"({adv:+.4f} vs Greedy)" if not np.isnan(greedy_auc) else ""
        print(f"  Size {k:>2d}/30: GA Val AUC = {val_auc:.4f}  | Train AUC = {train_auc:.4f}  "
              f"{adv_str:<18} [{elapsed_k*1000:5.0f} ms]")

        results.append({
            "signature_size": k,
            "ga_train_auc": round(train_auc, 6),
            "ga_val_auc": round(val_auc, 6),
            "greedy_val_auc": round(greedy_auc, 6) if not np.isnan(greedy_auc) else None,
            "ga_advantage": round(adv, 6),
            "genes": ", ".join(best_genes),
            "elapsed_s": round(elapsed_k, 4),
        })

    total_sweep_time = time.perf_counter() - t_sweep_start
    print("-" * 65)
    print(f"  Completed all {max_size} sizes in {total_sweep_time:.2f} seconds!")
    print("-" * 65)

    df_sweep = pd.DataFrame(results)
    out_path = RESULTS_DIR / "ga_size_sweep_results.csv"
    df_sweep.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}\n")
    return df_sweep
