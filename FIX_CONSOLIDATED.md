# UMAP & PaCMAP C++ Implementation - Comprehensive Issue Analysis

**Date**: 2024-12-24
**UMAP Versions Affected**: All versions up to v3.41.0
**PaCMAP Versions Affected**: All versions up to v2.8.34
**Severity**: HIGH (affects training quality, confidence scores, and user expectations)

---

## 🚨 CRITICAL: UMAP Codebase Analysis

**Investigation Date**: December 24, 2024
**Project**: UMAP C++ Implementation with C# Wrapper (uwot_umappp_wrapper)

### Status of Issues in UMAP

| Issue | PaCMAP Status | UMAP Status | Action Required |
|-------|---------------|-------------|-----------------|
| **1. Embedding Statistics** | BUG: Biased sampling | ❌ **WORSE: Never calculated!** | **CRITICAL FIX NEEDED** |
| **2. HNSW Ordering** | BUG: Reversed arrays | ❌ **Same bug exists** | **FIX NEEDED** |
| **3. Division by Zero** | BUG: No epsilon | ✅ **Already fixed** | None |
| **4. Documentation** | Missing/Wrong | ⚠️ **Needs update** | Low priority |

---

## UMAP-Specific Findings

### Issue 1: Embedding Statistics NEVER CALCULATED ❌❌❌

**Severity**: **CRITICAL**
**Location**: `uwot_umappp_wrapper/uwot_fit.cpp` - MISSING implementation
**Impact**: All safety metrics return WRONG values (divide by zero protected but meaningless)

#### Current State
```cpp
// In uwot_model.h lines 87-97 - Fields are defined:
float min_embedding_distance;
float max_embedding_distance;
float mean_embedding_distance;
float std_embedding_distance;
float p95_embedding_distance;      // ← Used in confidence calculation
float p99_embedding_distance;      // ← Used in outlier detection
float mild_embedding_outlier_threshold;
float extreme_embedding_outlier_threshold;
float median_embedding_distance;
```

```cpp
// In uwot_model.h lines 132-136 - Constructor initialization:
min_embedding_distance(0.0f), max_embedding_distance(0.0f),
mean_embedding_distance(0.0f), std_embedding_distance(0.0f),
p95_embedding_distance(0.0f), p99_embedding_distance(0.0f),  // ← STAYS 0.0!
mild_embedding_outlier_threshold(0.0f),
extreme_embedding_outlier_threshold(0.0f),
median_embedding_distance(0.0f),
```

#### The Problem
**NO CODE IN uwot_fit.cpp CALCULATES THESE VALUES!**

They are:
1. Declared in the model struct ✅
2. Initialized to 0.0f ✅
3. **NEVER COMPUTED** ❌❌❌
4. Persisted as 0.0f to saved models ❌
5. Loaded as 0.0f from saved models ❌

#### Impact on Transform Safety Metrics

In `uwot_transform.cpp:660-710`, safety metrics use these values:

```cpp
// Line 660: Confidence Score
float denom = std::max(EPS, model->p95_embedding_distance - model->min_embedding_distance);
// If p95=0.0 and min=0.0, then denom=EPS (1e-8), making confidence meaningless

// Line 667-681: Outlier Level
if (min_distance <= model->p95_embedding_distance) {  // Always TRUE if p95=0.0!
    outlier_level[i] = 0; // Normal
}
// ALL points classified as "Normal" because p95=0.0!

// Line 709: Z-Score
z_score[i] = (min_distance - model->mean_embedding_distance) / denom_z;
// If mean=0.0 and std=0.0, z_score = min_distance / EPS = HUGE VALUE
```

**Result**: All transformed points get:
- ConfidenceScore ≈ 1.0 (meaningless)
- OutlierLevel = 0 (always "Normal")
- PercentileRank = 0 or 99 (binary, not continuous)
- ZScore = extremely large positive value (useless)

### Issue 2: HNSW Ordering Bug - Same as PaCMAP ❌

**Severity**: **MEDIUM**
**Location**: `uwot_umappp_wrapper/uwot_transform.cpp:632-641`
**Impact**: User expectations violated, inconsistent with training

#### The Bug
```cpp
// Line 629: HNSW search returns "further first" (max-heap)
auto embedding_search_result = model->embedding_space_index->searchKnn(
    new_embedding_point, model->n_neighbors);

// Lines 632-641: Extract without reversing
while (!embedding_search_result.empty()) {
    auto pair = embedding_search_result.top();  // FARTHEST neighbor
    embedding_search_result.pop();

    int neighbor_idx = static_cast<int>(pair.second);
    float distance = std::sqrt(std::max(0.0f, pair.first));

    embedding_neighbors.push_back(neighbor_idx);  // [0] = FARTHEST
    embedding_distances.push_back(distance);      // [0] = FARTHEST
}
// NO REVERSE! Arrays stored with [0]=farthest, [last]=nearest
```

#### Consequences
1. **C# API returns reversed arrays**: `TransformResult.NearestNeighborDistances[0]` is FARTHEST, not nearest
2. **User confusion**: Documentation doesn't mention ordering
3. **Inconsistent with training**: During fit (uwot_fit.cpp:835-836), arrays ARE sorted nearest-first
4. **Metrics still work**: `std::min_element()` finds minimum regardless of order

#### Required Fix
```cpp
// After line 641, add:
std::reverse(embedding_neighbors.begin(), embedding_neighbors.end());
std::reverse(embedding_distances.begin(), embedding_distances.end());
```

### Issue 3: Division by Zero - Already Fixed ✅

**Location**: `uwot_transform.cpp:77`
**Status**: **ALREADY CORRECT**

```cpp
// Line 77: Proper epsilon protection
sum_weights = std::max(sum_weights, 1e-8f);
for (int d = 0; d < emb_dim; ++d) {
    out_embedding[d] /= sum_weights;
}
```

No action needed.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [UMAP Implementation Plan](#umap-implementation-plan)
3. [Issue 1: Biased Embedding Statistics (PaCMAP)](#issue-1-biased-embedding-statistics)
4. [Issue 2: HNSW Neighbor Ordering Confusion](#issue-2-hnsw-neighbor-ordering-confusion)
5. [Issue 3: Documentation Errors](#issue-3-documentation-errors)
6. [Implementation Plan (PaCMAP)](#implementation-plan)
7. [Testing Strategy](#testing-strategy)

---

## UMAP Implementation Plan

### Priority 1: Implement Embedding Statistics Calculation (CRITICAL)

**Estimated Effort**: 4-6 hours
**Complexity**: Medium
**Risk**: Medium (new calculation code)

#### Recommended Approach: HNSW-Based Collection

**Location**: `uwot_fit.cpp` - Add after embedding optimization completes (around line 740)

**Strategy**: Collect k-NN distances during embedding space HNSW index construction

```cpp
// After line 740 in uwot_fit.cpp (after embedding optimization, before model finalization)

// ===== CRITICAL: Calculate Embedding Space Statistics for Transform Safety =====
if (wrapped_callback) {
    wrapped_callback("Calculating embedding statistics", 96, 100, 96.0f,
                     "Computing safety metrics from embedding space k-NN distances");
}

try {
    if (model->embedding_space_index) {
        // Collect k-NN distances from ALL points in embedding space
        std::vector<float> all_embedding_knn_distances;
        all_embedding_knn_distances.reserve(static_cast<size_t>(n_obs) * n_neighbors);

        for (int i = 0; i < n_obs; i++) {
            const float* point = &embedding_vec[static_cast<size_t>(i) * embedding_dim];

            // Query k+1 neighbors (includes self)
            auto knn_result = model->embedding_space_index->searchKnn(point, n_neighbors + 1);

            // Extract distances, skipping self-match
            while (!knn_result.empty()) {
                auto pair = knn_result.top();
                knn_result.pop();

                int neighbor_id = static_cast<int>(pair.second);
                if (neighbor_id != i) {  // Skip self
                    float dist = std::sqrt(std::max(0.0f, pair.first));
                    all_embedding_knn_distances.push_back(dist);
                }
            }
        }

        // Sort distances for percentile calculation
        std::sort(all_embedding_knn_distances.begin(), all_embedding_knn_distances.end());

        if (!all_embedding_knn_distances.empty()) {
            size_t n_distances = all_embedding_knn_distances.size();

            // Calculate percentiles
            model->min_embedding_distance = all_embedding_knn_distances[0];
            model->max_embedding_distance = all_embedding_knn_distances[n_distances - 1];
            model->p95_embedding_distance = all_embedding_knn_distances[static_cast<size_t>(0.95 * n_distances)];
            model->p99_embedding_distance = all_embedding_knn_distances[static_cast<size_t>(0.99 * n_distances)];
            model->median_embedding_distance = all_embedding_knn_distances[n_distances / 2];

            // Calculate mean
            double sum = 0.0;
            for (float d : all_embedding_knn_distances) {
                sum += d;
            }
            model->mean_embedding_distance = static_cast<float>(sum / n_distances);

            // Calculate standard deviation
            double variance = 0.0;
            for (float d : all_embedding_knn_distances) {
                double diff = d - model->mean_embedding_distance;
                variance += diff * diff;
            }
            model->std_embedding_distance = static_cast<float>(std::sqrt(variance / n_distances));

            // Calculate outlier thresholds
            model->mild_embedding_outlier_threshold = model->mean_embedding_distance + 2.5f * model->std_embedding_distance;
            model->extreme_embedding_outlier_threshold = model->mean_embedding_distance + 4.0f * model->std_embedding_distance;
            model->exact_embedding_match_threshold = model->min_embedding_distance * 0.1f;  // 10% of min distance

            if (wrapped_callback) {
                char stats_msg[256];
                snprintf(stats_msg, sizeof(stats_msg),
                         "Embedding stats from %zu distances: min=%.3f, p95=%.3f, p99=%.3f, mean=%.3f, std=%.3f",
                         n_distances, model->min_embedding_distance, model->p95_embedding_distance,
                         model->p99_embedding_distance, model->mean_embedding_distance,
                         model->std_embedding_distance);
                wrapped_callback("Statistics ready", 97, 100, 97.0f, stats_msg);
            }
        }
    } else {
        // Fallback: If no HNSW index, use simple pairwise sampling
        // (This path should rarely/never execute in production UMAP)
        if (wrapped_callback) {
            wrapped_callback("Warning", 96, 100, 96.0f,
                           "Embedding space HNSW index not available, statistics will be approximate");
        }
    }
} catch (const std::exception& e) {
    if (wrapped_callback) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Failed to calculate embedding statistics: %s", e.what());
        wrapped_callback("Warning", 96, 100, 96.0f, err_msg);
    }
    // Don't fail fit if stats calculation fails - just leave as 0.0
}
```

**Benefits**:
- Uses all n_obs × k distances (e.g., 100K × 15 = 1.5M distances!)
- No additional time overhead (single pass)
- Representative sample from actual k-NN graph
- Scales automatically with dataset size
- No bias from ordered data

### Priority 2: Fix HNSW Ordering (MEDIUM)

**Estimated Effort**: 30 minutes
**Complexity**: Low
**Risk**: Very Low (simple array reversal)

#### Fix Location
`uwot_transform.cpp:641` - Add two lines after extracting neighbors

```cpp
// After line 641 (after while loop), add:
// Reverse arrays so that [0] = NEAREST, [last] = FARTHEST (user-expected ordering)
std::reverse(embedding_neighbors.begin(), embedding_neighbors.end());
std::reverse(embedding_distances.begin(), embedding_distances.end());
```

#### Update Documentation
`UMAPuwotSharp/UMAPuwotSharp/UMapModel.cs` - Fix XML comments

```csharp
/// <summary>
/// Gets the distances to nearest neighbors in the EMBEDDING space.
/// Ordered from NEAREST to FARTHEST: [0] = closest neighbor.
/// Used for confidence scoring and outlier detection.
/// </summary>
public double[] NearestNeighborDistances { get; }
```

---

## Executive Summary (PaCMAP)

Three critical issues have been identified in the PaCMAP C++ implementation:

1. **Biased Embedding Statistics** (Severity: HIGH)
   - Non-random sampling (first 1000 points only)
   - Hardcoded sample size regardless of dataset
   - Affects confidence scores and outlier detection

2. **HNSW Ordering Inconsistency** (Severity: MEDIUM)
   - Different ordering conventions in fit vs transform
   - Confusion about "farthest first" vs "nearest first"
   - Correct implementation in `build_knn_graph_hnsw` but reversed in `transform`

3. **Documentation Errors** (Severity: MEDIUM)
   - API documentation says "original space" when it's "embedding space"
   - No documentation of array ordering
   - Example code has bugs based on wrong assumptions

---

## Issue 1: Biased Embedding Statistics

### Problem Description

**Location**: `src/pacmap_pure_cpp/pacmap_fit.cpp:505-537`

During model training, embedding space statistics (p95, p99, mean, std) are calculated from pairwise distances. These statistics are used for:
- `ConfidenceScore` calculation during transform
- Outlier detection thresholds
- Quality metrics

**Current Implementation**:
```cpp
// Compute embedding statistics for transform safety
std::vector<float> embedding_distances;
int sample_size = std::min(n_obs, 1000);  // ← HARDCODED 1000
for (int i = 0; i < sample_size; i++) {   // ← Takes FIRST 1000 points, NOT random!
    for (int j = i + 1; j < sample_size; j++) {
        float dist = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            float diff = embedding[i * embedding_dim + d] - embedding[j * embedding_dim + d];
            dist += diff * diff;
        }
        embedding_distances.push_back(std::sqrt(dist));
    }
}
```

### Critical Flaws

1. **Non-Random Sampling**
   - Always takes points 0-999, never random
   - If data is sorted (by class, time, spatial location), statistics are biased
   - For 1M points, only samples 0.1% from the **beginning**

2. **Hardcoded Sample Size**
   - 1000 points regardless of dataset size
   - For 100K points: 1% sample
   - For 1M points: 0.1% sample
   - For 10M points: 0.01% sample

3. **Computational Reason**
   - Pairwise distances: O(n²) complexity
   - 1000 points = 499,500 distance calculations
   - Capped for performance, but should be adaptive

### Impact

**User-Reported Example**:
```
EmbedStats(min=2.953, p95=2.997, p99=3.027)
```

Range of only 0.074 units is suspicious and suggests:
- Biased sampling from tightly clustered subset
- All points in narrow range (unlikely)
- Statistics don't represent full embedding

**Downstream Effects**:
1. **Wrong ConfidenceScore**: `e^(-min_dist / p95_dist)` uses biased p95
2. **Wrong Outlier Detection**: Thresholds based on biased mean/std
3. **Persisted Forever**: Once saved, model carries biased stats
4. **No User Recourse**: Can't fix without retraining

### Solution

**Recommended Approach**: Use HNSW embedding index during construction

```cpp
// During HNSW embedding index building (pacmap_fit.cpp:540-580)
std::vector<float> embedding_knn_distances;
embedding_knn_distances.reserve(n_obs * model->n_neighbors);

for (int i = 0; i < n_obs; i++) {
    // Add point to index
    model->embedding_space_index->addPoint(&embedding_vec[i * embedding_dim], i);

    // Query k-NN for this point (uses already-inserted points)
    if (i >= model->n_neighbors) {  // Need enough points for k-NN
        auto knn_result = model->embedding_space_index->searchKnn(
            &embedding_vec[i * embedding_dim], model->n_neighbors);

        while (!knn_result.empty()) {
            float dist_sq = knn_result.top().first;
            float dist = std::sqrt(std::max(0.0f, dist_sq));
            embedding_knn_distances.push_back(dist);
            knn_result.pop();
        }
    }
}

// Calculate statistics from ALL k-NN distances (n_obs * k distances)
std::sort(embedding_knn_distances.begin(), embedding_knn_distances.end());
model->p95_embedding_distance = embedding_knn_distances[static_cast<size_t>(0.95 * embedding_knn_distances.size())];
model->p99_embedding_distance = embedding_knn_distances[static_cast<size_t>(0.99 * embedding_knn_distances.size())];
// ... etc
```

**Benefits**:
- Uses all points (n_obs × k distances, e.g., 100K × 40 = 4M distances!)
- No additional time/memory overhead (done during index construction anyway)
- Representative sample from actual k-NN graph
- Scales with dataset size

**Alternative**: Reservoir sampling with adaptive sample size (for when HNSW disabled)

---

## Issue 2: HNSW Neighbor Ordering Confusion

### HNSW Library Behavior (GROUND TRUTH)

**From `hnswlib.h:209`**:
```cpp
// here searchKnn returns the result in the order of further first
auto ret = searchKnn(query_data, k, isIdAllowed);
```

**`searchKnn()` returns `std::priority_queue`**:
- It's a **max-heap** by distance
- `top()` returns **FARTHEST** neighbor (largest distance)
- Popping drains from **farthest to nearest**

**Verification**: `searchKnnCloserFirst()` implementation (line 214-216):
```cpp
while (!ret.empty()) {
    result[--sz] = ret.top();  // ← Fills BACKWARDS to reverse order
    ret.pop();
}
```

If it were already "nearest first", this reversal wouldn't be needed!

### Current Implementation Analysis

#### ✅ CORRECT: `build_knn_graph_hnsw()` (pacmap_hnsw_utils.cpp:376-386)

```cpp
while (!result.empty()) {
    auto& top = result.top();
    int neighbor_id = static_cast<int>(top.second);
    if (neighbor_id != i) {
        neighbors.emplace_back(top.first, neighbor_id);
    }
    result.pop();
}

// Reverse to get nearest first
std::reverse(neighbors.begin(), neighbors.end());  // ← CORRECT!
```

**Result**: `nn_indices[0]` = nearest, `nn_indices[39]` = farthest

**Used for**: Triplet sampling and optimization (NEEDS nearest-first for correct algorithm)

#### ❌ NEEDS CLARIFICATION: `internal_pacmap_transform_detailed()` (pacmap_transform.cpp:203-222)

```cpp
// Extract embedding space neighbors and distances for AI inference
while (!embedding_search_result.empty()) {
    auto pair = embedding_search_result.top();
    embedding_search_result.pop();

    int neighbor_idx = static_cast<int>(pair.second);
    float distance = std::sqrt(std::max(0.0f, pair.first));

    embedding_neighbors.push_back(neighbor_idx);
    embedding_distances.push_back(distance);
}

// Store EMBEDDING SPACE neighbor information
for (size_t k = 0; k < embedding_neighbors.size(); k++) {
    size_t out_idx = i * model->n_neighbors + k;
    nn_indices[out_idx] = embedding_neighbors[k];
    nn_distances[out_idx] = embedding_distances[k];  // NO REVERSE!
}
```

**Result**: `nn_distances[0]` = **FARTHEST**, `nn_distances[39]` = **NEAREST**

**Question**: Is this intentional?

### Analysis of Current State

**What Actually Works**:
1. ✅ Confidence/outlier metrics are **CORRECT** because they use `std::min_element`:
   ```cpp
   float min_dist = *std::min_element(embedding_distances.begin(), embedding_distances.end());
   confidence_score[i] = std::exp(-min_dist / (model->p95_embedding_distance + 1e-8f));
   ```

2. ✅ Transform projection is **CORRECT** (weighted interpolation is order-invariant)

**What's Confusing**:
1. ❓ User-facing API returns `NearestNeighborDistances` with unclear ordering
2. ❓ Documentation says nothing about ordering
3. ❌ Example code **ASSUMES** `[0]` is nearest (BUG in Program.cs:390)
4. ✅ MnistDemo **WORKS AROUND IT** by explicitly sorting (line 1157)

### Decision Point

**Option A: Keep as-is (farthest-first) and document it**
- Pros: No breaking change, metrics already correct
- Cons: Counter-intuitive for users, inconsistent with fit

**Option B: Reverse to nearest-first (recommended)**
- Pros: Matches fit ordering, intuitive, fixes example code
- Cons: Breaking change for anyone relying on current order

**Recommendation**: **Option B** - Add reversal in transform for consistency

```cpp
// After populating embedding_neighbors and embedding_distances
std::reverse(embedding_neighbors.begin(), embedding_neighbors.end());
std::reverse(embedding_distances.begin(), embedding_distances.end());
```

Then update documentation to reflect nearest-first ordering.

---

## Issue 3: Documentation Errors

### Error 1: Wrong Space Mentioned

**Current** (`PacMapModel.cs:90-92`):
```csharp
/// <summary>
/// Gets the distances to nearest neighbors in the original feature space  ← WRONG!
/// </summary>
public double[] NearestNeighborDistances { get; }
```

**Fix**:
```csharp
/// <summary>
/// Gets the distances to nearest neighbors in the EMBEDDING space.
/// Ordered from nearest to farthest: [0] = closest, [Length-1] = farthest.
/// Used for confidence scoring and outlier detection.
/// </summary>
public double[] NearestNeighborDistances { get; }
```

### Error 2: No Ordering Documentation

**Add to all relevant places**:
- `NearestNeighborIndices`: Same ordering as distances
- C++ header comments
- Example code comments

### Error 3: Bug in Example Code

**Current** (`Program.cs:390-391`):
```csharp
var nearestDist = result.NearestNeighborDistances[0];
Console.WriteLine($"    Closest training point distance: {nearestDist:F3}");  // WRONG if [0]=farthest
```

**Fix** (if we keep farthest-first):
```csharp
var nearestDist = result.NearestNeighborDistances[^1];  // Last = nearest
Console.WriteLine($"    Closest training point distance: {nearestDist:F3}");
```

**Or Fix** (if we reverse to nearest-first):
```csharp
var nearestDist = result.NearestNeighborDistances[0];  // First = nearest
Console.WriteLine($"    Closest training point distance: {nearestDist:F3}");
```

---

## Implementation Plan

### Phase 1: Fix Embedding Statistics (Priority 1)

**Files to Modify**:
1. `src/pacmap_pure_cpp/pacmap_fit.cpp` (lines 505-537)

**Changes**:
- Replace pairwise distance calculation with HNSW-based approach
- Collect distances during embedding index construction
- Fallback to reservoir sampling if `force_exact_knn = true`

**Testing**:
- Compare old vs new statistics on known datasets
- Verify biased vs unbiased with sorted data
- Check performance impact

### Phase 2: Fix/Document Ordering (Priority 2)

**Files to Modify**:
1. `src/pacmap_pure_cpp/pacmap_transform.cpp` (lines 203-222)
2. `src/PACMAPCSharp/PACMAPCSharp/PacMapModel.cs` (lines 85-143)
3. `src/PACMAPCSharp/PACMAPCSharp.Example/Program.cs` (line 390)
4. `src/pacmap_pure_cpp/pacmap_simple_wrapper.h` (line 133)

**Decision Required**:
- Keep farthest-first and document, OR
- Reverse to nearest-first for consistency

**If reversing**:
```cpp
// In pacmap_transform.cpp after line 212
std::reverse(embedding_neighbors.begin(), embedding_neighbors.end());
std::reverse(embedding_distances.begin(), embedding_distances.end());
```

### Phase 3: Documentation Updates (Priority 3)

**Files to Update**:
- All C# XML documentation
- All C++ header comments
- README if applicable

**Add**:
- Clear ordering specification
- Space clarification (embedding vs original)
- Examples of correct usage

---

## Testing Strategy

### 1. Embedding Statistics Validation

**Test**: Biased vs Unbiased
```cpp
// Create sorted dataset: first 1000 from class A, rest from class B
// Old: Stats only from class A
// New: Stats from both classes
```

**Test**: Consistency with Fixed Seed
```cpp
// Run fit twice with same seed
// Verify stats match (proves no random variance issues)
```

**Test**: Small Dataset Correctness
```cpp
// < 1000 points: compare against exhaustive calculation
```

### 2. Ordering Validation

**Test**: Verify Nearest is Where Expected
```cpp
// Transform a training point
// Verify it finds itself as nearest neighbor
// Check correct index ([0] or [last] depending on decision)
```

**Test**: Cross-check with Manual Search
```cpp
// Compute distances manually
// Verify ordering matches expectation
```

### 3. Regression Testing

**Test**: Example Code Runs Without Errors
```cpp
// Run Program.cs example
// Run MnistDemo
// Verify correct classification results
```

---

## Statistics to Track

### Embedding Space Statistics (from training)

| Statistic | Purpose | Current Calculation | Proposed Fix |
|-----------|---------|---------------------|--------------|
| `min_embedding_distance` | Minimum pairwise distance | First 1000 pairs | All k-NN distances |
| `p95_embedding_distance` | 95th percentile | First 1000 pairs | All k-NN distances |
| `p99_embedding_distance` | 99th percentile | First 1000 pairs | All k-NN distances |
| `mean_embedding_distance` | Mean distance | First 1000 pairs | All k-NN distances |
| `std_embedding_distance` | Std deviation | First 1000 pairs | All k-NN distances |
| `mild_embedding_outlier_threshold` | mean + 2.5*std | Biased | Unbiased |
| `extreme_embedding_outlier_threshold` | mean + 4.0*std | Biased | Unbiased |

### Transform Safety Metrics (calculated during transform)

| Metric | Formula | Input | Ordering Dependency |
|--------|---------|-------|---------------------|
| `ConfidenceScore` | `e^(-min_dist / p95)` | min of k-NN distances | ✅ Uses min_element (order-independent) |
| `OutlierLevel` | Threshold classification | min of k-NN distances | ✅ Uses min_element (order-independent) |
| `PercentileRank` | `(min_dist / p95) * 95` | min of k-NN distances | ✅ Uses min_element (order-independent) |
| `ZScore` | `(min_dist - mean) / std` | min of k-NN distances | ✅ Uses min_element (order-independent) |
| `NearestNeighborDistances` | Array of k-NN distances | From HNSW search | ❌ Ordering matters for API users |
| `NearestNeighborIndices` | Array of k-NN indices | From HNSW search | ❌ Ordering matters for API users |

---

## Version Impact

**Minimum Version for Fixes**: 2.8.35

**Breaking Changes**:
- If we reverse ordering: Breaking change for code relying on current order
- If we keep ordering: No breaking change, just documentation

**Model Compatibility**:
- Old models will load with old (biased) statistics
- New models will have correct (unbiased) statistics
- Recommend retraining for production use
- Add warning when loading old models

---

## References

### Source Code Locations

**Issue 1 (Statistics)**:
- Calculation: `src/pacmap_pure_cpp/pacmap_fit.cpp:505-537`
- Persistence: `src/pacmap_pure_cpp/pacmap_persistence.cpp:460-466, 1083-1089`
- Usage: `src/pacmap_pure_cpp/pacmap_transform.cpp:231, 236-258`

**Issue 2 (Ordering)**:
- HNSW library: `src/pacmap_pure_cpp/hnswlib.h:209-210`
- Fit (correct): `src/pacmap_pure_cpp/pacmap_hnsw_utils.cpp:376-386`
- Transform (needs decision): `src/pacmap_pure_cpp/pacmap_transform.cpp:203-222`

**Issue 3 (Documentation)**:
- C# API: `src/PACMAPCSharp/PACMAPCSharp/PacMapModel.cs:85-143`
- Bug in example: `src/PACMAPCSharp/PACMAPCSharp.Example/Program.cs:390-391`
- Workaround: `src/PacMapDemo/MnistDemo.cs:1154-1159`

### External References

- User Issue: EmbedStats(min=2.953, p95=2.997, p99=3.027) - suspicious narrow range
- HNSW Documentation: Comment at hnswlib.h:209 "further first"
- fix2.txt: Analysis suggesting transform ordering issue
- User appended note: Conflicting analysis about HNSW ordering

---

## Final Recommendations

### Must-Fix (High Priority)

1. ✅ **Fix biased embedding statistics** using HNSW-based collection
   - Severity: HIGH
   - Impact: Confidence scores, outlier detection
   - Effort: 4-8 hours

2. ✅ **Reverse transform ordering to nearest-first**
   - Severity: MEDIUM
   - Impact: User expectations, API consistency
   - Effort: 1-2 hours

3. ✅ **Fix documentation and example code**
   - Severity: MEDIUM
   - Impact: User confusion, bugs in user code
   - Effort: 1-2 hours

### Should-Have (Medium Priority)

4. ⚠️ **Add helper properties** for common operations
   - `ClosestNeighborDistance` property
   - `FarthestNeighborDistance` property
   - Effort: 30 minutes

5. ⚠️ **Add migration strategy** for old models
   - Warning when loading models with biased stats
   - Optional recalculation on load
   - Effort: 2-4 hours

### Nice-to-Have (Low Priority)

6. 📝 **Add comprehensive unit tests**
   - Test statistics calculation
   - Test ordering conventions
   - Test edge cases
   - Effort: 4-8 hours

---

## Conclusion

The PaCMAP C++ implementation has three interconnected issues:
1. Biased statistics from non-random sampling
2. Unclear/inconsistent neighbor ordering
3. Missing/incorrect documentation

All three can be fixed with modest effort and will significantly improve the reliability and usability of the library. The metrics calculations themselves are correct (they use `min_element`), so the core algorithm works—it's the peripheral features and documentation that need attention.

**Recommended Action**: Implement all high-priority fixes in v2.8.35, test thoroughly, and document breaking changes clearly in release notes.
