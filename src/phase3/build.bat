@echo off
:: build.bat — Compile auc_openmp.c into a Windows DLL
:: =====================================================
:: Run from the REPO ROOT: .\src\phase3\build.bat
:: After running, you'll have: src\phase3\auc_openmp.dll
::
:: Flags explained:
::   -O3           : Maximum speed optimisation (unroll loops, vectorise, etc.)
::   -march=native : Use all CPU instructions available on THIS machine
::                   (AVX2, FMA, etc. on the i7-13650HX). Never do this for
::                   a binary you'll distribute — it won't run on older CPUs.
::   -fopenmp      : Enable OpenMP multi-threading support
::   -shared       : Build a shared library (.dll) instead of an executable
::   -fPIC         : Position-Independent Code — required for shared libraries
::   -o ...        : Output file path

where gcc >nul 2>nul
if %ERRORLEVEL% == 0 (
    set GCC=gcc
) else (
    set GCC=C:\Users\NonoNegative\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gcc.exe
)
set SRC=src\phase3\auc_openmp.c
set OUT=src\phase3\auc_openmp.dll

echo.
echo [Phase 3] Compiling %SRC% with GCC + OpenMP...
echo.

"%GCC%" -O3 -march=native -fopenmp -shared -fPIC -o "%OUT%" "%SRC%"

if %ERRORLEVEL% == 0 (
    echo [OK] Compiled successfully: %OUT%
) else (
    echo [ERROR] Compilation failed. Check the error messages above.
    exit /b 1
)

echo.
echo You can now run: python src\phase3\cpp_search.py
echo.
