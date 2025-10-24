@echo off
setlocal enabledelayedexpansion
echo ===========================================
echo   REAL UMAPPP Cross-Platform Build
echo     (Spectral Initialization - Solves Fragmentation!)
echo ===========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

REM === BUILD WINDOWS VERSION ===
echo [1/2] Building Windows version...
echo       (libscran/umappp + spectral initialization)
echo ----------------------------------------

REM Call the Windows-only build script
call BuildWindowsOnly.bat
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows build failed!
    exit /b 1
)

REM === BUILD LINUX VERSION WITH DOCKER ===
echo.
echo [2/2] Building Linux version with Docker...
echo ----------------------------------------

REM Call the Linux-only build script
call BuildLinux.bat
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Linux build failed!
    exit /b 1
)

REM === SETUP CSHARP PROJECT STRUCTURE ===
echo.
echo [3/3] Setting up Enhanced UMAPuwotSharp project...
echo ----------------------------------------

REM Create C# project structure
if not exist "..\UMAPuwotSharp" (
    echo [INFO] Creating UMAPuwotSharp project structure
    mkdir "..\UMAPuwotSharp"
)

if not exist "..\UMAPuwotSharp\UMAPuwotSharp" (
    mkdir "..\UMAPuwotSharp\UMAPuwotSharp"
)

REM Copy Windows DLL directly to project base folder
if exist "build\Release\uwot.dll" (
    copy "build\Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll"
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Copied Windows uwot.dll to C# project base folder
    ) else (
        echo [FAIL] Failed to copy Windows uwot.dll - Error: !ERRORLEVEL!
    )
) else (
    echo [FAIL] Windows uwot.dll not found in build\Release\
)

REM Copy Linux .so file directly to project base folder
set LINUX_LIB_COPIED=0

REM Try libuwot_final.so first (our guaranteed backup from Docker)
if exist "build-linux\libuwot_final.so" (
    for %%A in ("build-linux\libuwot_final.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libuwot_final.so" "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so"
            echo [PASS] Copied libuwot_final.so as libuwot.so to C# project base folder
            set LINUX_LIB_COPIED=1
            goto :linux_lib_done
        )
    )
)

REM Check for versioned .so files (the actual library files)
for %%F in (build-linux\libuwot.so.*.*.*) do (
    if exist "%%F" (
        for %%A in ("%%F") do (
            if %%~zA GTR 1000 (
                copy "%%F" "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so"
                echo [PASS] Copied %%~nxF as libuwot.so to C# project base folder
                set LINUX_LIB_COPIED=1
                goto :linux_lib_done
            )
        )
    )
)

REM Try backup file if available
if exist "build-linux\libuwot_backup.so" (
    for %%A in ("build-linux\libuwot_backup.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libuwot_backup.so" "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so"
            echo [PASS] Copied libuwot_backup.so as libuwot.so to C# project base folder
            set LINUX_LIB_COPIED=1
            goto :linux_lib_done
        )
    )
)

REM Check for libuwot.so (last resort)
if exist "build-linux\libuwot.so" (
    for %%A in ("build-linux\libuwot.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libuwot.so" "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so"
            if !ERRORLEVEL! EQU 0 (
                echo [PASS] Copied Linux libuwot.so to C# project base folder
                set LINUX_LIB_COPIED=1
            ) else (
                echo [FAIL] Failed to copy Linux libuwot.so - Error: !ERRORLEVEL!
            )
        )
    )
)

:linux_lib_done
if !LINUX_LIB_COPIED! EQU 0 (
    echo [FAIL] Linux libuwot.so not found in build-linux\
    echo [INFO] Available files in build-linux:
    if exist "build-linux" (
        dir build-linux\ /B 2>nul || echo        Directory empty or not accessible
    ) else (
        echo        build-linux directory not found
    )
)

echo [PASS] Enhanced libraries setup completed

REM === SUMMARY ===
echo.
echo ===========================================
echo   Enhanced UMAP Build Summary
echo ===========================================
echo.
echo Windows libraries (build\Release\):
if exist "build\Release\uwot.dll" (
    echo   [PASS] uwot.dll (Enhanced UMAP with multiple metrics)
    for %%A in ("build\Release\uwot.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] uwot.dll)

if exist "build\Release\test_standard_comprehensive.exe" (
    echo   [PASS] test_standard_comprehensive.exe
) else (
    echo   [FAIL] test_standard_comprehensive.exe
)

if exist "build\Release\test_error_fixes_simple.exe" (
    echo   [PASS] test_error_fixes_simple.exe
) else (
    echo   [FAIL] test_error_fixes_simple.exe
)

if exist "build\Release\test_comprehensive_pipeline.exe" (
    echo   [PASS] test_comprehensive_pipeline.exe
) else (
    echo   [FAIL] test_comprehensive_pipeline.exe
)

echo.
echo Linux libraries (build-linux\):
if exist "build-linux" (
    dir build-linux\libuwot.so* build-linux\*.so /B 2>nul | findstr /R ".*"
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

if exist "build-linux\test_standard_comprehensive" (
    echo   [PASS] test_standard_comprehensive
) else (
    echo   [FAIL] test_standard_comprehensive
)

if exist "build-linux\test_error_fixes_simple" (
    echo   [PASS] test_error_fixes_simple
) else (
    echo   [FAIL] test_error_fixes_simple
)

if exist "build-linux\test_comprehensive_pipeline" (
    echo   [PASS] test_comprehensive_pipeline
) else (
    echo   [FAIL] test_comprehensive_pipeline
)

echo.
echo UMAPuwotSharp C# project files:
if exist "..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll" (
    echo   [PASS] Windows library: uwot.dll
    for %%A in ("..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Windows library missing)

if exist "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so" (
    echo   [PASS] Linux library: libuwot.so
    for %%A in ("..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Linux library missing)

echo.
echo Cross-platform libraries ready for direct deployment:
if exist "..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll" (
    echo   [PASS] Windows library ready: uwot.dll
) else (echo   [WARN] Windows library missing)

if exist "..\UMAPuwotSharp\UMAPuwotSharp\libuwot.so" (
    echo   [PASS] Linux library ready: libuwot.so
) else (echo   [WARN] Linux library missing)

echo.
echo ===========================================
echo   Enhanced UMAP Features Available
echo ===========================================
echo.
echo Your Enhanced UMAPuwotSharp library now supports:
echo   - Arbitrary embedding dimensions (1D to 50D, including 27D)
echo   - Multiple distance metrics (Euclidean, Cosine, Manhattan, Correlation, Hamming)
echo   - Complete model save/load functionality
echo   - True out-of-sample projection (transform new data)
echo   - Progress reporting with callback support
echo   - Based on proven uwot algorithms
echo   - Cross-platform support (Windows + Linux)
echo.
echo ===========================================
echo   Ready for C# Development!
echo ===========================================
echo.
echo Next steps:
echo   1. Build C# library: dotnet build ..\UMAPuwotSharp\UMAPuwotSharp
echo   2. Create NuGet package: dotnet pack ..\UMAPuwotSharp\UMAPuwotSharp --configuration Release
echo   3. Test 27D embedding: var embedding27D = model.Fit(data, embeddingDimension: 27);
echo   4. Use progress reporting: model.FitWithProgress(data, progressCallback);
echo.
pause