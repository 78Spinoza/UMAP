@echo off
setlocal enabledelayedexpansion
echo ==========================================
echo   Building REAL UMAPPP Linux Version Only
echo     (Docker-based Cross-Platform Build)
echo ==========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

echo [1/1] Building Enhanced Linux version with Docker...
echo       (libscran/umappp + spectral initialization)
echo ----------------------------------------

if exist build-linux (
    rmdir /s /q build-linux
)

REM Run Docker build with Ubuntu 22.04 and manual CMake installation
docker run --rm -v "%cd%":/src -w /src ubuntu:22.04 bash -c "apt-get update && apt-get install -y build-essential wget git libstdc++-11-dev && wget -q -O /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh && chmod +x /tmp/cmake.sh && /tmp/cmake.sh --skip-license --prefix=/usr/local && cd /src && mkdir -p build-linux && cd build-linux && /usr/local/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make -j4 && echo Build completed && ls -la && if [ -f libuwot.so ]; then cp libuwot.so libuwot_backup.so && echo Library backup created; fi && echo Linux build finished successfully"

if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker Linux build failed!
    exit /b 1
)

echo [PASS] Linux build completed

echo.
echo ==========================================
echo   Enhanced Linux Build Summary
echo ==========================================
echo.
echo Linux libraries (build-linux/):
if exist build-linux (
    dir build-linux\libuwot.so* build-linux\*.so /B 2>nul | findstr /R \".*\"
    if !ERRORLEVEL! EQU 0 (
        echo   [PASS] Linux .so files found:
        for %%F in (build-linux\libuwot.so* build-linux\*.so) do (
            if exist "%%F" (
                for %%A in ("%%F") do echo         %%~nxF (%%~zA bytes)
            )
        )
    ) else (
        echo   [FAIL] No Linux .so files found
    )
) else (
    echo   [FAIL] build-linux directory not found
)

echo.
echo Cross-platform libraries ready:
if exist build-linux\libuwot.so (
    echo   [PASS] Linux library ready: libuwot.so
) else (
    echo   [WARN] Linux library missing
)

echo.
echo ==========================================
echo   Enhanced UMAP Features Available
echo ==========================================
echo.
echo Your Enhanced UMAPuwotSharp library now supports:
echo   - Arbitrary embedding dimensions (1D to 50D, including 27D)
echo   - Multiple distance metrics (Euclidean, Cosine, Manhattan, Correlation, Hamming)
echo   - Complete model save/load functionality
echo   - True out-of-sample projection (transform new data)
echo   - Progress reporting with callback support
echo   - Based on proven umappp algorithms with spectral initialization
echo   - Cross-platform support (Windows + Linux)
echo.
echo ==========================================
echo   Linux Build Complete!
echo ==========================================
echo.
pause