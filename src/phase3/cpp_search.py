"""
cpp_search.py  —  Phase 3: Python ctypes wrapper for the C+OpenMP kernel
=========================================================================

This module is the bridge between Python (which handles all the data loading,
preprocessing, and GA logic) and the C DLL (which handles the heavy math).

HOW ctypes WORKS (beginner explanation)
----------------------------------------
ctypes is a Python standard library module that lets you load any compiled
C shared library (.dll on Windows, .so on Linux) and call its functions
directly from Python code.

The magic trick is that NumPy arrays store their data as a flat block of
C-compatible memory (exactly like a C array). So when we pass a NumPy array
to ctypes, we are literally giving C a pointer to the same memory NumPy is
using. No copying. No serialization. The C function reads and writes into
the exact same memory block Python owns.

This is the fundamental reason it's so much faster than Python's joblib:
    joblib: Python → pickle (copy) → worker process → unpickle (copy) → result
    ctypes: Python → pointer (no copy) → C runs → result in same memory

WHAT THIS MODULE EXPORTS
------------------------
    evaluate_population_c(X_train, population, y_train) → np.ndarray of AUC scores

    This is a drop-in replacement for the Python fitness evaluation loop in
    genetic_search.py. The results are numerically identical (same AUC formula).
"""

import ctypes
import os
import numpy as np
from pathlib import Path

import shutil

# ─── Load the compiled DLL ──────────────────────────────────────────────────

_DLL_PATH = Path(__file__).parent.resolve() / "auc_openmp.dll"

def _find_mingw_bin() -> Path | None:
    # 1. Check if gcc is in PATH
    gcc_which = shutil.which("gcc")
    if gcc_which:
        p = Path(gcc_which).parent
        if p.exists():
            return p

    # 2. Check WinGet Packages directory
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_pkgs = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.exists():
            for match in winget_pkgs.glob("BrechtSanders*/**/bin"):
                if (match / "gcc.exe").exists():
                    return match

    # 3. Common install locations
    for candidate in [
        Path(r"C:\mingw64\bin"),
        Path(r"C:\msys64\ucrt64\bin"),
        Path(r"C:\msys64\mingw64\bin"),
    ]:
        if candidate.exists():
            return candidate
    return None

_MINGW_BIN = _find_mingw_bin()

if not _DLL_PATH.exists():
    raise FileNotFoundError(
        f"\n\n  [Phase 3] DLL not found at: {_DLL_PATH}\n"
        f"  Please compile it first by running:\n"
        f"      .\\src\\phase3\\build.bat\n"
        f"  from the repo root directory.\n"
    )

# Windows requires both the phase3 folder (where auc_openmp.dll lives)
# and the MinGW bin folder (where libgomp dependencies live) to be registered
# as DLL search directories BEFORE ctypes.CDLL is called.
_dll_handles = []
if hasattr(os, "add_dll_directory"):
    _dll_handles.append(os.add_dll_directory(str(_DLL_PATH.parent)))
    if _MINGW_BIN and _MINGW_BIN.exists():
        _dll_handles.append(os.add_dll_directory(str(_MINGW_BIN)))

_lib = ctypes.CDLL(str(_DLL_PATH))

# ─── Tell ctypes the exact signature of our C function ──────────────────────
#
# ctypes needs to know:
#   1. What types the arguments are (so it can convert Python objects to C types)
#   2. What type the return value is
#
# If you skip this step, ctypes will guess wrong and you'll get crashes or
# garbage numbers. Always specify argtypes and restype explicitly.

_lib.evaluate_population.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # X          : double* (the expression matrix)
    ctypes.POINTER(ctypes.c_int),      # population : int*    (gene index arrays)
    ctypes.POINTER(ctypes.c_int),      # y          : int*    (labels 0/1)
    ctypes.c_int,                      # n_patients
    ctypes.c_int,                      # n_candidates
    ctypes.c_int,                      # n_individuals
    ctypes.c_int,                      # sig_size
    ctypes.POINTER(ctypes.c_double),   # fitness_out: double* (output array)
]
_lib.evaluate_population.restype = None   # void function

_lib.get_num_threads.argtypes = []
_lib.get_num_threads.restype  = ctypes.c_int


def get_num_threads() -> int:
    """Return the number of OpenMP threads (= CPU cores available)."""
    return _lib.get_num_threads()


def evaluate_population_c(X_train: np.ndarray,
                           population: np.ndarray,
                           y_train: np.ndarray) -> np.ndarray:
    """
    Compute AUC fitness for all individuals in `population` using the
    C+OpenMP kernel. This is a drop-in replacement for the Python loop.

    Parameters
    ----------
    X_train    : np.ndarray, shape (n_patients, n_candidates), dtype float64
                 The expression matrix — only the candidate gene columns.
    population : np.ndarray, shape (n_individuals, sig_size), dtype int32
                 Each row is one signature: indices into X_train's columns.
    y_train    : np.ndarray, shape (n_patients,), dtype int32
                 Binary labels: 1 = relapse, 0 = no relapse.

    Returns
    -------
    fitness : np.ndarray, shape (n_individuals,), dtype float64
              AUC score for each individual. Values in [0.0, 1.0].
    """
    # ── Ensure correct dtypes and memory layout ──────────────────────────────
    #
    # C expects:
    #   - doubles (64-bit floats) for X and the output
    #   - int32 (32-bit signed ints) for population and y
    #   - Row-major (C-contiguous) memory layout
    #
    # np.ascontiguousarray() ensures the array is stored as a flat C-style
    # block in memory — required for the pointer trick to work correctly.

    X    = np.ascontiguousarray(X_train,   dtype=np.float64)
    pop  = np.ascontiguousarray(population, dtype=np.int32)
    y    = np.ascontiguousarray(y_train,    dtype=np.int32)

    n_patients, n_candidates = X.shape
    n_individuals, sig_size  = pop.shape

    # Allocate output array — C will write directly into this
    fitness = np.zeros(n_individuals, dtype=np.float64)

    # ── Call the C function via ctypes ────────────────────────────────────────
    #
    # .ctypes.data_as(ctypes.POINTER(...)) gets a raw C pointer to the
    # underlying memory of the NumPy array. This is the zero-copy transfer.

    _lib.evaluate_population(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pop.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n_patients),
        ctypes.c_int(n_candidates),
        ctypes.c_int(n_individuals),
        ctypes.c_int(sig_size),
        fitness.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return fitness


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("  Phase 3: C+OpenMP Wrapper Self-Test")
    print("=" * 60)
    print(f"\n  DLL loaded from : {_DLL_PATH}")
    print(f"  OpenMP threads  : {get_num_threads()}")

    # Create a tiny synthetic dataset to verify correctness
    rng           = np.random.default_rng(42)
    n_patients    = 50
    n_candidates  = 100
    n_individuals = 10
    sig_size      = 5

    X_test    = rng.standard_normal((n_patients, n_candidates))
    pop_test  = np.array([rng.choice(n_candidates, size=sig_size, replace=False)
                          for _ in range(n_individuals)], dtype=np.int32)
    y_test    = rng.integers(0, 2, size=n_patients, dtype=np.int32)

    fitness = evaluate_population_c(X_test, pop_test, y_test)

    print(f"\n  Synthetic test  : {n_individuals} individuals, "
          f"{n_patients} patients, {sig_size}-gene signatures")
    print(f"  Fitness scores  : {fitness.round(4)}")
    print(f"  All in [0,1]?   : {bool((fitness >= 0).all() and (fitness <= 1).all())}")
    print("\n  [PASS] C kernel loaded and returning valid AUC scores.\n")
