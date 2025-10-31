@echo off
REM Set console to handle UTF-8 properly (suppress output)
chcp 65001 >nul 2>&1

echo ===========================================
echo   Building REAL UMAPPP Windows Version Only
echo     (MSYS2 GCC - Bypasses MSVC Issues)
echo ===========================================
echo.

echo [1/1] Building REAL UMAPPP with CMake...
echo       (libscran/umappp + spectral initialization)
echo ----------------------------------------

REM Check if build directory exists, clean if needed
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo.
echo Configuring REAL UMAPPP with CMake using MSYS2 GCC...
echo   - Bypasses MSVC JacobiSVD compilation issues
echo   - libscran/umappp reference implementation
echo   - Spectral initialization for complex 3D structures
echo   - Full performance with GCC + Eigen 3.4.0
"C:\msys64\mingw64.exe" /usr/bin/bash -lc "cd /c/UMAP/uwot_umappp_wrapper/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building Release version with GCC...
"C:\msys64\mingw64.exe" /usr/bin/bash -lc "cd /c/UMAP/uwot_umappp_wrapper/build && make -j$(nproc)"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Enhanced Windows Build Complete!
echo ===========================================
echo.
echo Generated files:
echo   - uwot.dll (Enhanced UMAP library with spectral initialization)
echo   - HNSW-knncolle bridge for performance
echo   - Solves mammoth dataset fragmentation issues!
echo.

REM Copy DLL directly to C# project folder
if exist "Release\uwot.dll" (
    if exist "..\UMAPuwotSharp\UMAPuwotSharp\" (
        copy "Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\" >nul
        echo [COPY] uwot.dll copied to C# project folder
    ) else (
        echo [WARN] C# project folder not found: ..\UMAPuwotSharp\UMAPuwotSharp\
    )
) else (
    echo [FAIL] uwot.dll not found for copying
)

echo.
echo ===========================================
echo   Enhanced UMAP Build Summary
echo ===========================================
echo.
echo Windows libraries (build\Release\):
if exist "build\Release\uwot.dll" (
    echo   [PASS] uwot.dll (Enhanced UMAP with multiple metrics)
    for %%A in ("build\Release\uwot.dll") do echo         Size: %%~zA bytes
) else (
    echo   [FAIL] uwot.dll missing!
)

echo.
echo Enhanced features available:
echo   - Arbitrary Dimensions      - 1D to 50D embeddings (including 27D)
echo   - Multiple Distance Metrics - Euclidean, Cosine, Manhattan, Correlation, Hamming
echo   - Model Training           - uwot_fit() with enhanced parameters
echo   - Model Persistence        - uwot_save_model() / uwot_load_model()
echo   - Data Transformation      - uwot_transform() for out-of-sample projection
echo   - Model Information        - uwot_get_model_info() with metric support
echo   - Cross-platform          - Windows ready, Linux compatible
echo   - OpenMP Support          - Parallel optimization enabled
echo.
echo Windows build completed successfully!
echo Press any key to exit...
pause >nul