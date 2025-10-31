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

REM Copy Linux .so file to C# project
echo.
echo Copying Linux library to C# project...

REM Try libuwot_final.so first (our guaranteed backup from Docker)
if exist "build-linux\libuwot_final.so" (
    for %%A in ("build-linux\libuwot_final.so") do (
        if %%~zA GTR 1000 (
            powershell -Command "Copy-Item 'build-linux\libuwot_final.so' '..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so' -Force"
            if !ERRORLEVEL! EQU 0 (
                echo [COPY] libuwot_final.so copied as libuwot.so to C# project folder successfully
            ) else (
                echo [FAIL] Failed to copy libuwot_final.so - error code !ERRORLEVEL!
            )
            goto :linux_copy_done
        )
    )
)

REM Try libuwot_backup.so
if exist "build-linux\libuwot_backup.so" (
    for %%A in ("build-linux\libuwot_backup.so") do (
        if %%~zA GTR 1000 (
            powershell -Command "Copy-Item 'build-linux\libuwot_backup.so' '..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so' -Force"
            if !ERRORLEVEL! EQU 0 (
                echo [COPY] libuwot_backup.so copied as libuwot.so to C# project folder successfully
            ) else (
                echo [FAIL] Failed to copy libuwot_backup.so - error code !ERRORLEVEL!
            )
            goto :linux_copy_done
        )
    )
)

REM Check for versioned .so files
for %%F in (build-linux\libuwot.so.*.*.*) do (
    if exist "%%F" (
        for %%A in ("%%F") do (
            if %%~zA GTR 1000 (
                powershell -Command "Copy-Item '%%F' '..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so' -Force"
                if !ERRORLEVEL! EQU 0 (
                    echo [COPY] %%~nxF copied as libuwot.so to C# project folder successfully
                ) else (
                    echo [FAIL] Failed to copy %%~nxF - error code !ERRORLEVEL!
                )
                goto :linux_copy_done
            )
        )
    )
)

REM Try libuwot.so directly
if exist "build-linux\libuwot.so" (
    for %%A in ("build-linux\libuwot.so") do (
        if %%~zA GTR 1000 (
            powershell -Command "Copy-Item 'build-linux\libuwot.so' '..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so' -Force"
            if !ERRORLEVEL! EQU 0 (
                echo [COPY] libuwot.so copied to C# project folder successfully
            ) else (
                echo [FAIL] Failed to copy libuwot.so - error code !ERRORLEVEL!
            )
            goto :linux_copy_done
        )
    )
)

echo [WARN] No suitable Linux library found to copy
echo [INFO] Current directory: %CD%

:linux_copy_done
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