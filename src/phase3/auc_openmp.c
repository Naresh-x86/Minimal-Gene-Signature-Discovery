/*
 * auc_openmp.c  —  Phase 3: C + OpenMP Fitness Evaluation Kernel
 * ================================================================
 *
 * PURPOSE
 * -------
 * This file is the "engine room" of Phase 3. It does one specific job
 * with maximum possible speed: given a population of 100 gene signatures,
 * compute the AUC fitness score for every single one of them simultaneously,
 * using all available CPU cores via OpenMP.
 *
 * WHY C, NOT PYTHON?
 * ------------------
 * Python has a safety mechanism called the GIL (Global Interpreter Lock).
 * The GIL is a mutex (a lock) that ensures only ONE Python thread runs
 * at a time — even if you have 14 CPU cores. This is why Python's joblib
 * was so slow: the threads existed, but only one ran at a time.
 *
 * C has no GIL. When we use OpenMP in C, all 14 cores genuinely run
 * simultaneously in hardware. Additionally, Python's joblib had to
 * "pickle" (serialize) the data to send it to each worker. Here, we
 * receive raw pointers directly to NumPy's memory — zero copying,
 * zero serialization.
 *
 * HOW AUC IS COMPUTED (Mann-Whitney U method)
 * --------------------------------------------
 * The Area Under the ROC Curve (AUC) has a beautiful mathematical
 * equivalence to the Mann-Whitney U statistic:
 *
 *   AUC = (number of (relapse, healthy) patient pairs where the
 *          relapse patient has a higher risk score than the healthy one)
 *         divided by
 *         (total number of such pairs)
 *
 * In math: AUC = U / (n_pos * n_neg)
 * where U = sum over all positive patients p of:
 *             (number of negative patients with score < score[p])
 *
 * This is O(n_pos * n_neg) per individual — about 69 * 217 ≈ 15,000
 * comparisons per signature. With 100 individuals, that's 1.5 million
 * comparisons per generation. On 14 cores, each core does ~107,000.
 * All in compiled machine code. It finishes in microseconds.
 *
 * COMPILATION (handled by build.bat):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC
 *       -o auc_openmp.dll auc_openmp.c
 *
 * EXPORTED FUNCTION SIGNATURE:
 *   void evaluate_population(
 *       const double* X,          // [n_patients × n_candidates] row-major
 *       const int*    population,  // [n_individuals × sig_size] row-major
 *       const int*    y,           // [n_patients] binary labels (1=relapse)
 *       int           n_patients,
 *       int           n_candidates,
 *       int           n_individuals,
 *       int           sig_size,
 *       double*       fitness_out  // [n_individuals] — written by this function
 *   );
 */

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─────────────────────────────────────────────────────────────────
 * compute_auc_mannwhitney
 * -----------------------
 * Computes AUC for a single risk-score vector using the Mann-Whitney
 * formula. No memory allocation — works entirely on the passed arrays.
 *
 * Parameters:
 *   scores      : risk score for each of n_patients patients
 *   y           : binary label (1 = relapse, 0 = no relapse) per patient
 *   n_patients  : total number of patients (228 in training)
 *
 * Returns:
 *   AUC as a double in [0.0, 1.0]
 * ───────────────────────────────────────────────────────────────── */
static double compute_auc_mannwhitney(const double* scores,
                                      const int*    y,
                                      int           n_patients)
{
    long long u_stat   = 0;
    long long n_pos    = 0;   /* number of relapse patients    (y == 1) */
    long long n_neg    = 0;   /* number of no-relapse patients (y == 0) */

    /* Count positives and negatives */
    for (int i = 0; i < n_patients; i++) {
        if (y[i] == 1) n_pos++;
        else           n_neg++;
    }

    /* Edge case: if one class is missing, AUC is undefined → return 0.5 */
    if (n_pos == 0 || n_neg == 0) return 0.5;

    /*
     * Mann-Whitney U:
     * For every (positive patient p, negative patient q) pair, check
     * if score[p] > score[q]. Count how many such pairs exist.
     *
     * This double loop is O(n_pos * n_neg) ≈ 69 × 217 ≈ 15,000 iterations.
     * Small, predictable, and the CPU branch predictor handles it well.
     */
    for (int i = 0; i < n_patients; i++) {
        if (y[i] != 1) continue;           /* skip non-relapse patients */
        double si = scores[i];
        for (int j = 0; j < n_patients; j++) {
            if (y[j] != 0) continue;       /* skip relapse patients */
            if      (si > scores[j]) u_stat += 2;   /* clear win   → +1 */
            else if (si == scores[j]) u_stat += 1;  /* tie         → +0.5 */
            /* si < scores[j]: no contribution */
        }
    }

    /*
     * u_stat was accumulated as 2× the integer part to handle 0.5 ties
     * without floating point in the inner loop. Now convert:
     *   AUC = (u_stat / 2.0) / (n_pos * n_neg)
     *
     * In this project, AUC is defined as max(auc, 1.0 - auc) so that a gene
     * or signature that separates classes in either direction scores identically.
     */
    double auc = (double)u_stat / (2.0 * (double)(n_pos * n_neg));
    if (auc < (1.0 - auc)) {
        auc = 1.0 - auc;
    }
    return auc;
}


/* ─────────────────────────────────────────────────────────────────
 * evaluate_population  (the exported function Python calls)
 * -----------------------------------------------------------
 * For each of the n_individuals gene signatures in `population`,
 * compute the sum of expression scores across the selected genes
 * for every patient, then compute AUC and write it to fitness_out.
 *
 * The OpenMP pragma below fires one thread per CPU core. Each thread
 * takes a slice of individuals and works independently — there is no
 * shared mutable state (each individual writes to its own fitness_out
 * index), so no mutexes or locks are needed.
 * ───────────────────────────────────────────────────────────────── */
#ifdef _WIN32
__declspec(dllexport)
#endif
void evaluate_population(const double* X,
                         const int*    population,
                         const int*    y,
                         int           n_patients,
                         int           n_candidates,
                         int           n_individuals,
                         int           sig_size,
                         double*       fitness_out)
{
    /*
     * OpenMP parallel for:
     * Each loop iteration processes one individual independently.
     * `schedule(static)` splits iterations into equal chunks upfront —
     * the best choice when every iteration takes roughly the same time.
     */
    #pragma omp parallel for schedule(static)
    for (int ind = 0; ind < n_individuals; ind++) {

        /* Pointer to the gene indices for this individual */
        const int* genes = population + (ind * sig_size);

        /*
         * Compute risk scores: for each patient, sum up the expression
         * values of all sig_size selected genes.
         *
         * X is stored row-major: X[patient_idx * n_candidates + gene_idx]
         * So X[p * n_candidates + genes[g]] = expression of gene g in patient p.
         */
        double* scores = (double*)malloc(n_patients * sizeof(double));
        if (!scores) {
            fitness_out[ind] = 0.0;   /* memory failure fallback */
            continue;
        }

        /* Initialize scores to zero */
        memset(scores, 0, n_patients * sizeof(double));

        /* Sum selected gene expressions for each patient */
        for (int g = 0; g < sig_size; g++) {
            int gene_col = genes[g];                 /* column index in X */
            for (int p = 0; p < n_patients; p++) {
                scores[p] += X[p * n_candidates + gene_col];
            }
        }

        /* Compute AUC using the Mann-Whitney formula */
        fitness_out[ind] = compute_auc_mannwhitney(scores, y, n_patients);

        free(scores);
    }
}


/* ─────────────────────────────────────────────────────────────────
 * get_num_threads  (utility — Python uses this to report thread count)
 * ───────────────────────────────────────────────────────────────── */
#ifdef _WIN32
__declspec(dllexport)
#endif
int get_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
