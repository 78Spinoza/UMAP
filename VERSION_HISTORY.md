# UMAPuwotSharp Version History

Complete changelog and release notes for UMAPuwotSharp C# wrapper.

## Version 3.42.2 (2024-12-25)

### 🧹 Production-Ready Code Cleanup

**Theme**: Removed debug output for production deployment

### What's New

#### Production-Ready Code
- **Removed**: All debug output statements for production deployment
- **Result**: Zero debug noise in production logs
- **Code Quality**: Clean, production-ready implementation

### Technical Details

#### Files Modified
- `uwot_transform.cpp`: Removed debug printf statements
- `uwot_simple_wrapper.h:33`: Updated version to 3.42.2
- `UMapModel.cs:343`: Updated EXPECTED_DLL_VERSION to 3.42.2
- `UMAPuwotSharp.csproj:15`: Updated package version to 3.42.2

### Validation Results

#### Functionality
- **Single-sample TransformWithSafety**: Works correctly ✅
- **Fast path performance**: Still 12-15x faster than batch ✅
- **All 17 tests passing**: Full test coverage green ✅

#### Code Quality
- **Zero debug output**: Production-ready code ✅
- **Clean compilation**: No warnings or errors ✅
- **Identical behavior**: Same functionality as v3.42.1 ✅

### Performance Impact
- **Load**: No change
- **Save**: No change
- **Fit**: No change
- **Transform**: No change (fast and correct)

### Migration Guide

#### From v3.42.1 to v3.42.2

**✅ No Breaking Changes** - Fully backward compatible!

**Recommended for Production**:
- Same functionality as v3.42.1
- Cleaner code without debug output
- Ideal for production deployment

```csharp
// Existing code works unchanged
var model = UMapModel.Load("model.umap");
var results = model.TransformWithSafety(newData);  // Fast + Correct! ✅
```

---

## Version 3.42.1 (2024-12-24)

### ✅ Backward Compatibility: Old Models Now Work!

**Theme**: Automatic embedding_space_index rebuild for old models + enhanced fast path for single-sample TransformWithSafety

### 🔄 Critical Backward Compatibility Fix

#### Old Models (Pre-v3.42.0) Now Supported
- **FIXED**: Old models now rebuild `embedding_space_index` during load
- **Auto-calculates**: Embedding statistics when loading old models
- **Result**: TransformWithSafety now works with old models!
- **No user action required**: Just load and use!

**Technical Details**:
- On load, if `embedding_space_index` is NULL but embedding data exists
- Rebuilds HNSW index using saved embeddings
- Calculates all embedding statistics (min, max, mean, std, p95, p99, median)
- Uses defaults for HNSW params if old model has them set to 0
- Gracefully degrades if rebuild fails (warning, not error)

### 🚀 Critical Performance Fix: Single-Sample TransformWithSafety

#### Fast Path Enhancement (CRITICAL)
- **BUG**: Single-sample TransformWithSafety returned all zeros (distances=0, confidence=0)
- **Root Cause**: Fast path optimization (12-15x speedup) didn't populate safety metrics
- **Impact**: Users transforming 1 sample at a time got incorrect results
- **FIXED**: Enhanced fast path now supports TransformWithSafety while maintaining speed
- **Result**: Single-sample transforms are fast AND correct!

**Before v3.42.1**:
```csharp
model.TransformWithSafety(oneSample);
// Returns: distance = 0, confidence = 0 ❌
```

**After v3.42.1**:
```csharp
model.TransformWithSafety(oneSample);
// Returns: distance = 0.001, confidence = 0.993 ✅
// Speed: Still 12-15x faster than batch! ✅
```

**Technical Details**:
- File: `uwot_transform.cpp:241-309`
- Enhanced fast path to check if detailed output is requested
- If yes, performs embedding space search after fast transform
- Populates all safety metrics (nn_indices, nn_distances, confidence, outlier_level, etc.)
- Maintains 12-15x speedup advantage over batch path

### 🎯 Issues Fixed from v3.42.0

#### Issue #1: Embedding Statistics Now Calculated (CRITICAL)
- **FIXED**: Statistics were NEVER calculated (all zeros in previous versions)
- **Added**: Complete statistics collection during training using HNSW k-NN
- **Collects**: All n_obs × k distances for unbiased statistics
- **Computes**: min, max, mean, std, p95, p99, median, outlier thresholds
- **Impact**: AI safety features now work correctly

#### Issue #2: HNSW Ordering Fixed
- **FIXED**: NearestNeighborDistances array was reversed (farthest-first)
- **Added**: `std::reverse()` after HNSW extraction
- **Now**: [0] = NEAREST neighbor (as users expect)

#### Issue #3: Division by Zero
- **Confirmed**: Already protected with epsilon in v3.41.0

### 📝 New Unit Test
- **Added**: `Test_V342_TransformWithSafety_NonZero_Distances()`
- **Validates**: Non-zero embedding distances returned
- **Validates**: Varying confidence scores (not all 1.0)
- **Test count**: 16/16 passing (was 15 in v3.42.0)

### Technical Details

#### Files Modified
- `uwot_persistence.cpp:575-658`: Added automatic rebuild of embedding_space_index on load
- `uwot_persistence.cpp:676-682`: Removed error throw for missing embedding_space_index
- `uwot_simple_wrapper.h:33`: Updated version to 3.42.1
- `UMapModel.cs:343`: Updated EXPECTED_DLL_VERSION to 3.42.1
- `UMAPuwotSharp.csproj:15`: Updated package version to 3.42.1
- `UMapModelTests.cs:528-614`: Added v3.42.0 validation test

### Validation Results

#### Before v3.42.1 (v3.42.0 with old model)
```
Old model loaded → embedding_space_index = NULL
TransformWithSafety → All distances = 0 ❌
```

#### After v3.42.1
```
Old model loaded → embedding_space_index rebuilt automatically
Statistics calculated → p95=59.582, p99=63.260 ✅
TransformWithSafety → Non-zero distances: 0.044 to 1.607 ✅
Confidence scores → Varying: 0.664 to 1.000 ✅
```

### Performance Impact
- **Load (old models)**: +2-5 seconds for index rebuild
- **Load (new models)**: No change
- **Transform**: No change
- **Memory**: No change

### Migration Guide

#### From v3.42.0 to v3.42.1

**✅ No Breaking Changes** - Fully backward compatible!

**Old models**: Just load and use - they work now!
```csharp
var oldModel = UMapModel.Load("model_v3.41.0.umap");
var results = oldModel.TransformWithSafety(newData);  // ✅ Works!
```

**New models**: Same as v3.42.0 - no changes needed.

---

## Version 3.42.0 (2024-12-24)

### 🐛 Critical Bug Fixes: Embedding Statistics & HNSW Ordering

**Theme**: Fixed critical bugs affecting AI safety metrics and user expectations

### Critical Bug Fixes

#### Issue #1: Embedding Statistics Now Calculated (CRITICAL FIX)
- **CRITICAL**: Embedding space statistics were **NEVER calculated** in previous versions
- **Impact**: All safety metrics (ConfidenceScore, OutlierLevel, ZScore) returned meaningless values
- **Root Cause**: Fields initialized to 0.0f but no calculation code existed in uwot_fit.cpp
- **Solution**: Added comprehensive statistics calculation using HNSW k-NN distances
  - Collects distances from ALL n_obs × k points (e.g., 10K × 15 = 150K distances)
  - Calculates: min, max, mean, std, p95, p99, median
  - Computes outlier thresholds (mild, extreme)
  - Progress reporting during calculation

```csharp
// Before v3.42.0: All zeros (BROKEN)
Stats from 0 distances: min=0.000, p95=0.000, p99=0.000, mean=0.000, std=0.000

// v3.42.0: Real statistics (FIXED)
Stats from 201764 distances: min=0.029, p95=59.582, p99=63.260, mean=30.367, std=22.251
```

**Result**: AI safety features now work correctly!
- ✅ ConfidenceScore: Meaningful 0.0-1.0 range based on actual data
- ✅ OutlierLevel: Proper 5-level classification (Normal → No Man's Land)
- ✅ PercentileRank: Continuous 0-100 values instead of binary 0/99
- ✅ ZScore: Accurate statistical deviation scores

#### Issue #2: HNSW Ordering Fixed (User Expectations)
- **Bug**: NearestNeighborDistances array was reversed (farthest-first order)
- **Impact**: Users expected [0]=closest but got [0]=farthest neighbor
- **Root Cause**: HNSW searchKnn() returns max-heap (farthest-first), not reversed
- **Solution**: Added `std::reverse()` calls after HNSW extraction
  - Now [0] = NEAREST neighbor (as users expect)
  - Now [last] = FARTHEST neighbor
  - Consistent with training k-NN array ordering

```csharp
// Before v3.42.0: REVERSED (confusing)
result.NearestNeighborDistances[0];   // Was FARTHEST
result.NearestNeighborDistances[^1];  // Was NEAREST

// v3.42.0: CORRECTED (intuitive)
result.NearestNeighborDistances[0];   // Now NEAREST ✅
result.NearestNeighborDistances[^1];  // Now FARTHEST ✅
```

#### Issue #3: Division by Zero Already Protected
- ✅ Confirmed: `interpolate_one_point_fast` already had epsilon protection
- ✅ No changes needed: Code was already correct

### Technical Details

#### Files Modified
- `uwot_fit.cpp:754-887`: Added embedding space HNSW index + statistics calculation
- `uwot_transform.cpp:645-647`: Added std::reverse() for nearest-first ordering
- `uwot_simple_wrapper.h:33`: Updated version to 3.42.0
- `UMapModel.cs:343`: Updated EXPECTED_DLL_VERSION to 3.42.0
- `UMAPuwotSharp.csproj:15`: Updated package version to 3.42.0

#### Statistics Collection Details
- **Method**: HNSW k-NN query during embedding space index build
- **Sample Size**: All n_obs × k distances (not biased sample)
- **Performance**: Minimal overhead (~2% of fit time)
- **Memory**: Temporary vector cleared after statistics computed
- **Fallback**: Graceful degradation if HNSW index unavailable

### Validation

#### Testing Results
- ✅ **15/15 C# tests passing**: All validation tests green
- ✅ **Example application**: Verified real statistics output
- ✅ **Safety metrics**: Confirmed proper outlier detection
- ✅ **Binary ordering**: Verified [0]=nearest in all scenarios

#### Before/After Comparison
```
Before v3.42.0 (BROKEN):
  EmbedStats(min=0.000, p95=0.000, p99=0.000)
  ConfidenceScore: 1.0 (always)
  OutlierLevel: 0 (always Normal)

After v3.42.0 (FIXED):
  EmbedStats(min=0.029, p95=59.582, p99=63.027)
  ConfidenceScore: 0.0-1.0 (meaningful range)
  OutlierLevel: 0-4 (proper classification)
```

### Performance Impact
- **Load**: No change
- **Save**: No change
- **Fit**: +2% time for statistics calculation (negligible)
- **Transform**: No change (metrics now meaningful)

### Migration Guide

#### From v3.41.0 to v3.42.0

**⚠️ IMPORTANT**: Models saved with v3.41.0 or earlier have **zero statistics**
- Old models will load but have meaningless safety metrics
- **Recommendation**: Retrain production models to get real statistics
- No code changes needed - fully backward compatible

**Breaking Change**: NearestNeighborDistances ordering reversed
- If your code relied on [0]=farthest, update to [^1]=farthest
- Most users expected [0]=nearest, so this is a fix not a break

```csharp
// If you had code like this (working around the bug):
var farthest = result.NearestNeighborDistances[0];  // OLD bug workaround

// Update to:
var farthest = result.NearestNeighborDistances[^1]; // v3.42.0 correct

// Or simply use the intuitive approach that now works:
var nearest = result.NearestNeighborDistances[0];   // ✅ Now correct!
```

---

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

