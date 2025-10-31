# UMAPuwotSharp Version History

Complete changelog and release notes for UMAPuwotSharp C# wrapper.

## Version 3.40.0 (2025-10-31)

### 🎯 Major Changes: Initialization API Enhancement

**Theme**: Improved API clarity with InitializationMethod enum and spectral default

### New Features

#### InitializationMethod Enum
- **NEW**: Type-safe `InitializationMethod` enum for explicit initialization control
  - `Spectral = 1`: High-quality manifold-aware initialization (DEFAULT)
  - `Auto = -1`: Automatic size-based selection (≤20k→Spectral, >20k→Random)
  - `Random = 0`: Fast random initialization
- **API Clarity**: Replaced confusing `AlwaysUseSpectral` boolean with clearer `InitMethod` property
- **Backward Compatibility**: `AlwaysUseSpectral` property maintained as obsolete for existing code

```csharp
// NEW v3.40.0 API
var model = new UMapModel();
model.InitMethod = InitializationMethod.Spectral;  // Clear and explicit

// OLD API (still works)
model.AlwaysUseSpectral = true;  // Marked [Obsolete]
```

#### Spectral Initialization Default
- **BREAKING CHANGE**: Spectral initialization is now the **default** (was Auto in v3.39.0)
- **Rationale**: Best quality embeddings out-of-the-box for most use cases
- **Impact**: Users get better quality without explicit configuration
- **Migration**: Set `InitMethod = InitializationMethod.Auto` if you need old behavior

#### Dynamic Metadata Extraction
- **Enhanced**: All visualizations now use model-extracted parameters
- **Removed**: Hardcoded parameter strings in demo visualization titles
- **Benefit**: Guarantees accuracy between actual model parameters and visualization metadata

### Improvements

#### Clean Compilation
- **Fixed**: All unused variable warnings in C++ code
- **Fixed**: Type conversion warnings in uwot_persistence.cpp
- **Fixed**: Unsigned comparison warnings in hnswalg.h
- **Result**: Zero compiler warnings across entire codebase

#### Enhanced Demo
- **Updated**: Hairy mammoth bandwidth experiments with optimal parameters
  - 11 bandwidth values: 3.0 to 5.0 (increments of 0.2)
  - local_connectivity = 2.0
  - spread = 2.0
  - n_neighbors = 60
- **Dynamic Titles**: Use `BuildVisualizationTitle()` for accurate metadata

#### Code Quality
- **Removed**: Unused variables in uwot_persistence.cpp
- **Cleaned**: Dead code paths and unreachable statements
- **Improved**: Error messages and validation throughout

### Migration Guide

#### From v3.39.0 to v3.40.0

**Breaking Changes**:
```csharp
// v3.39.0: Auto was default
var model = new UMapModel();
var embedding = model.Fit(data);  // Used Auto mode

// v3.40.0: Spectral is now default
var model = new UMapModel();
var embedding = model.Fit(data);  // Uses Spectral mode (better quality!)

// To get old Auto behavior:
model.InitMethod = InitializationMethod.Auto;
```

**Recommended Migration**:
```csharp
// Replace this:
model.AlwaysUseSpectral = true;
// With this:
model.InitMethod = InitializationMethod.Spectral;

// Replace this:
model.AlwaysUseSpectral = false;
// With this:
model.InitMethod = InitializationMethod.Auto;
```

---

## Version 3.39.0 (2025-10-27)

### New Features
- **AlwaysUseSpectral Property**: Force spectral initialization for any dataset size
- **Eigen Compilation Fix**: Resolved MSVC compilation errors
- **Hyperparameter Integration**: LocalConnectivity & Bandwidth exposed in API
- **Bandwidth Sweep Testing**: Comprehensive experiments for parameter discovery

---

## Version 3.37.0 (2025-10-26)

### New Features
- **OpenMP Parallelization**: 4-5x faster multi-point transforms
- **Single-Point Optimization**: 12-15x speedup with stack allocation
- **Stringstream Persistence**: No temp files for HNSW serialization
- **Windows DLL Stability**: Proper OpenMP cleanup

---

## Version 3.33.1 (2025-10-25)

### New Features
- **Dual-Mode k-NN**: Both HNSW and exact computation
- **CPU Core Reporting**: Real-time parallel processing feedback
- **Complete Parameter Propagation**: All C# parameters reach C++

---

## See README.md for complete version history

