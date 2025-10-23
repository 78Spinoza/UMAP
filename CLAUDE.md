# UMAP C++ Implementation with C# Wrapper - Claude Guide

## 🎯 Project Overview
High-performance UMAP implementation with enhanced features:
- **HNSW Optimization**: Fast transforms with memory reduction
- **Production Safety**: Multi-level outlier detection and confidence scoring
- **Multi-dimensional embeddings**: 1D to 50D support
- **Multiple distance metrics**: Euclidean, Cosine, Manhattan, Correlation, Hamming
- **Model persistence**: Save/load trained models
- **Progress reporting**: Real-time training feedback
- **Transform capability**: Project new data using existing models with safety analysis

## 🚀 Critical Build Protocol

**ALWAYS navigate to correct folder FIRST before running batch files:**
```bash
cd uwot_pure_cpp                  # ALWAYS go to the folder first!
./BuildDockerLinuxWindows.bat     # THEN run the batch file
```

**Why this is critical:**
- Running from wrong directory causes path resolution issues
- Libraries get copied to wrong locations
- NuGet packages contain incorrect binaries

## 📁 Correct DLL Location

**⚠️ CRITICAL: DLL goes directly to UMAPuwotSharp project root**
```bash
# CORRECT - Copy DLL to project root
cp "C:\UMAP\uwot_pure_cpp\build\Release\uwot.dll" "C:\UMAP\UMAPuwotSharp\UMAPuwotSharp\uwot.dll"

# ❌ WRONG - Do NOT use runtimes folder
# "C:\UMAP\UMAPuwotSharp\UMAPuwotSharp\runtimes\win-x64\native\uwot.dll"  # DOES NOT EXIST
```

## 🛠️ Build Commands

### C# Library (Main Development)
```bash
cd UMAPuwotSharp
dotnet build                       # Build library and example
dotnet run --project UMAPuwotSharp.Example  # Run demo
dotnet test                        # Run comprehensive test suite
```

### C++ Native Library Development
```bash
cd uwot_pure_cpp
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest                              # Run C++ validation tests
```

## 🧪 Testing Guidelines

**✅ ALWAYS verify build success before running tests**
**✅ NEVER run tests if compilation failed** - fix compilation first
**✅ Test ONLY on freshly compiled binaries** that include all recent code changes

## 🔧 Version Management

**Binary Version Checking:**
1. **C++ Version**: Set in `uwot_simple_wrapper.h` - `UWOT_WRAPPER_VERSION_STRING`
2. **C# Version**: Set in `UMapModel.cs` - `EXPECTED_DLL_VERSION` constant
3. **Both must match exactly** or constructor throws detailed error

## 📊 Current Status

**✅ COMPLETED ACHIEVEMENTS:**
- ✅ NuGet Package v3.18.0: Phase 4 Large Dataset Support & Unified API
- ✅ Performance optimization with HNSW integration
- ✅ Production safety features with 5-level outlier detection
- ✅ AI/ML integration ready with data validation
- ✅ 15/15 C# Tests Passing: All validation tests green
- ✅ Perfect pipeline consistency: Training embeddings match transform results

## 🚨 Build Timeouts

**CRITICAL TIMEOUT GUIDELINES:**
- **Docker builds**: 30+ minutes (1800000ms minimum)
- **C# example tests**: 5-10 minutes (300000-600000ms)
- **C++ comprehensive tests**: 5-10 minutes
- **Regular builds**: 2-3 minutes (120000-180000ms)
- **Never use default 2-minute timeouts for complex operations**

## ✅ Clean Compilation Standards

**Always maintain:**
- Zero build errors
- Zero nullability warnings
- Zero unused code/structs/functions
- Professional commit messages without AI attribution