# UMAPuwotSharp Version History

Complete changelog and release notes for UMAPuwotSharp C# wrapper.

## Version 3.41.0 (2025-11-07)

### 🐛 Critical Fixes: Model Persistence & Large File Support

**Theme**: Fixed access violation on load + 20x larger file support for production reliability

### Critical Bug Fixes

#### Fixed Access Violation on Load + TransformWithSafety
- **CRITICAL FIX**: Resolved crash when loading models and calling `TransformWithSafety()`
- **Root Cause**: HNSW `original_space_index` failed to load due to 100MB size limit
- **Impact**: Models with >100MB HNSW indices (large datasets) crashed on load
- **Solution**:
  - Increased HNSW size limit from 100MB to 2GB (20x increase)
  - Changed HNSW load errors from silent warnings to fail-fast exceptions
  - Added comprehensive validation ensuring all indices ready after load
  - Improved error messages identifying exact failure point

```csharp
// Now works reliably with large models
var model = UMapModel.Load("large_model.umap");  // 112MB model, 106k vertices
var results = model.TransformWithSafety(newData);  // ✅ Works perfectly!
```

### New Features

#### Increased File Size Limits
- **HNSW Indices**: 100MB → 2000MB (2GB limit) - 20x increase
- **Embedding Data**: 100MB → 2000MB (2GB limit) - 20x increase
- **LZ4 Overflow Protection**: Added safety checks for LZ4 int limit (2GB max)
- **Backward Compatible**: Maintains uint32_t format for existing files
- **Technical**: Changed limits in `uwot_constants.hpp`, added overflow validation

#### LoadWithCallbacks() API
- **NEW**: Static method for monitoring load progress and errors
- **Callback Support**: Receive warnings/errors during model load
- **Use Case**: Production pipelines needing visibility into load failures

```csharp
var model = UMapModel.LoadWithCallbacks("model.umap", (phase, current, total, percent, message) => {
    Console.WriteLine($"[{phase}] {message}");
});
```

### Improvements

#### Fail-Fast Validation
- **HNSW Load Errors**: Now throw exceptions instead of silent failure
- **Index Validation**: Comprehensive checks ensure all indices ready for TransformWithSafety
- **Error Messages**: Distinguish between corruption, missing data, and reconstruction failure
- **Production Safety**: If load succeeds, TransformWithSafety is guaranteed to work

#### LZ4 Overflow Protection
- **HNSW Compression**: Added checks for LZ4 int limit before compression
- **Embedding Compression**: Same overflow protection for embedding data
- **Safety First**: Prevents integer overflow for models approaching 2GB size
- **Clear Errors**: Helpful messages when data exceeds safe compression limits

#### Enhanced Testing
- **NEW Test**: `Test_Persistence_With_TransformWithSafety`
  - Validates save → load → TransformWithSafety workflow
  - Ensures HNSW indices properly persisted and reconstructed
  - Verifies callback support during load
- **Test Count**: Maintained 15/15 tests passing
- **Coverage**: All persistence scenarios validated

### Technical Details

#### Files Modified
- `uwot_constants.hpp:27-28`: Increased size limits to 2GB
- `uwot_persistence.cpp:18`: Added `LZ4_MAX_SIZE` constant
- `uwot_persistence.cpp:38-39,258-260,477-483`: LZ4 overflow checks
- `uwot_persistence.cpp:535-546`: Fail-fast HNSW load error handling
- `uwot_persistence.cpp:96-103`: Backward compatible uint32_t format
- `UMapModel.cs:343`: Updated EXPECTED_DLL_VERSION to 3.41.0
- `UMapModel.cs:471`: Added LoadWithCallbacks() static method
- `UMapModelTests.cs:437`: Added Test_Persistence_With_TransformWithSafety

#### Bug Details
- **Symptom**: Access violation (0xc0000005) when calling TransformWithSafety after load
- **Cause**: `original_space_index->searchKnn()` called on NULL pointer
- **Trigger**: Models with HNSW indices >100MB failed to load, left index NULL
- **Fix**: Fail-fast validation + 20x larger limits + overflow protection

### Performance Impact
- **Load**: No performance change for existing models
- **Save**: No performance change (still uses uint32_t format)
- **Large Models**: Now supported up to 2GB HNSW indices (previously 100MB)
- **Memory**: No changes to runtime memory usage

### Migration Guide

#### From v3.40.0 to v3.41.0

**No Breaking Changes** - Fully backward compatible!

```csharp
// Existing code works unchanged
var model = UMapModel.Load("model.umap");
var results = model.TransformWithSafety(data);

// NEW: Optional callback monitoring
var model2 = UMapModel.LoadWithCallbacks("large_model.umap", (phase, current, total, percent, message) => {
    if (!string.IsNullOrEmpty(message)) {
        Console.WriteLine($"Load: {message}");
    }
});
```

**When to Use LoadWithCallbacks()**:
- Production pipelines needing load failure visibility
- Debugging load issues with large models
- Monitoring HNSW reconstruction progress
- Capturing detailed error messages

### Validation Results
- ✅ **15/15 tests passing** (all existing + 1 new test)
- ✅ **112MB model load**: Successfully loads 106,346 vertex model
- ✅ **TransformWithSafety**: Works correctly after load
- ✅ **HNSW reconstruction**: Embedding space index rebuilt from data
- ✅ **Zero compilation warnings**: Clean codebase

---

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

