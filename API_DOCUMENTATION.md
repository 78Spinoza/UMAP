# UMAPuwotSharp API Documentation

## Table of Contents
- [Overview](#overview)
- [Core Classes](#core-classes)
- [Enumerations](#enumerations)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)

## Overview

UMAPuwotSharp provides a production-ready C# wrapper for high-performance UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction with advanced features including HNSW optimization, safety analysis, and comprehensive configuration options.

## Core Classes

### UMapModel

Main class for UMAP model training, transformation, and persistence.

```csharp
public class UMapModel : IDisposable
{
    // Properties
    public InitializationMethod InitMethod { get; set; }
    public ModelInfo ModelInfo { get; }

    [Obsolete("Use InitMethod property instead")]
    public bool AlwaysUseSpectral { get; set; }

    // Methods
    public float[,] Fit(float[,] data, ...);
    public float[,] FitWithProgress(float[,] data, ...);
    public float[,] Transform(float[,] data);
    public TransformResult[] TransformWithSafety(float[,] data);
    public void SaveModel(string filePath);
    public static UMapModel LoadModel(string filePath);
}
```

### ModelInfo

Contains comprehensive metadata about a fitted UMAP model.

```csharp
public class ModelInfo
{
    public int TrainingSamples { get; }
    public int InputDimension { get; }
    public int OutputDimension { get; }
    public string MetricName { get; }
    public int Neighbors { get; }
    public float MinimumDistance { get; }
    public float Spread { get; }
    public float LocalConnectivity { get; }
    public float Bandwidth { get; }
    public bool IsHNSWEnabled { get; }
}
```

### TransformResult

Represents the result of a transform operation with safety analysis.

```csharp
public class TransformResult
{
    public float[] Embedding { get; }
    public float ConfidenceScore { get; }
    public OutlierLevel OutlierSeverity { get; }
    public string QualityAssessment { get; }
    public bool IsProductionReady { get; }
    public int[] NearestNeighborIndices { get; }
    public float[] NearestNeighborDistances { get; }
}
```

## Enumerations

### InitializationMethod

**NEW in v3.40.0**: Controls UMAP initialization strategy.

```csharp
public enum InitializationMethod
{
    Auto = -1,      // Automatic selection: ≤20k samples→Spectral, >20k→Random
    Random = 0,     // Fast random initialization (lower quality for large datasets)
    Spectral = 1    // High-quality manifold-aware initialization (DEFAULT)
}
```

**Default Value**: `InitializationMethod.Spectral` (since v3.40.0)

**Usage**:
```csharp
var model = new UMapModel();

// Explicit spectral (high quality) - this is the default
model.InitMethod = InitializationMethod.Spectral;

// Auto mode (size-based selection)
model.InitMethod = InitializationMethod.Auto;

// Random (fast but lower quality)
model.InitMethod = InitializationMethod.Random;
```

### DistanceMetric

Specifies the distance metric for UMAP computation.

```csharp
public enum DistanceMetric
{
    Euclidean,      // General-purpose (HNSW accelerated)
    Cosine,         // High-dimensional sparse data (HNSW accelerated)
    Manhattan,      // Outlier-robust (HNSW accelerated)
    Correlation,    // Time series, correlated features (exact fallback)
    Hamming         // Binary, categorical data (exact fallback)
}
```

### OutlierLevel

Five-level outlier classification for safety analysis.

```csharp
public enum OutlierLevel
{
    Normal = 0,         // ≤95th percentile - similar to training data
    Unusual = 1,        // 95-99th percentile - noteworthy but acceptable
    MildOutlier = 2,    // 99th percentile to 2.5σ - moderate deviation
    ExtremeOutlier = 3, // 2.5σ to 4σ - significant deviation
    NoMansLand = 4      // >4σ - completely outside training distribution
}
```

## API Reference

### Training Methods

#### Fit

Basic training method without progress reporting.

```csharp
public float[,] Fit(
    float[,] data,
    int embeddingDimension = 2,
    int nNeighbors = 15,
    float minDist = 0.1f,
    float spread = 1.0f,
    int nEpochs = 200,
    DistanceMetric metric = DistanceMetric.Euclidean,
    bool forceExactKnn = false,
    bool autoHNSWParam = true,
    int randomSeed = 42,
    float localConnectivity = 1.0f,
    float bandwidth = 1.0f
)
```

**Parameters**:
- `data`: Input data matrix [samples × features]
- `embeddingDimension`: Output dimensionality (1-50)
- `nNeighbors`: Number of nearest neighbors (5-100)
- `minDist`: Minimum distance in embedding (0.0-0.99)
- `spread`: Scale of embedded points (0.5-5.0)
- `nEpochs`: Training iterations (50-1000)
- `metric`: Distance metric
- `forceExactKnn`: Use exact k-NN instead of HNSW
- `autoHNSWParam`: Auto-optimize HNSW parameters
- `randomSeed`: Random seed for reproducibility
- `localConnectivity`: Fuzzy simplicial set connectivity (0.5-3.0)
- `bandwidth`: Kernel density estimation bandwidth (0.5-5.0)

**Returns**: Embedding matrix [samples × embeddingDimension]

#### FitWithProgress

Training with real-time progress callbacks.

```csharp
public float[,] FitWithProgress(
    float[,] data,
    Action<ProgressInfo> progressCallback,
    // ... same parameters as Fit()
)
```

**Progress Callback**:
```csharp
public class ProgressInfo
{
    public int CurrentEpoch { get; }
    public int TotalEpochs { get; }
    public float PercentComplete { get; }
    public string Message { get; }
    public string Phase { get; }
}
```

### Transform Methods

#### Transform

Fast transformation of new data to embedding space.

```csharp
public float[,] Transform(float[,] data)
```

**Parameters**:
- `data`: New data matrix [samples × features] (must match training feature count)

**Returns**: Embedding matrix [samples × embeddingDimension]

**Performance**: 1-3ms per sample with HNSW optimization

#### TransformWithSafety

Transform with comprehensive safety analysis and outlier detection.

```csharp
public TransformResult[] TransformWithSafety(float[,] data)
```

**Returns**: Array of `TransformResult` objects with:
- Embedding coordinates
- Confidence scores (0.0-1.0)
- Outlier severity level
- Production readiness flag
- Nearest neighbor information

**Performance**: 3-5ms per sample (includes safety analysis)

### Persistence Methods

#### SaveModel

Save trained model to disk with compression and integrity validation.

```csharp
public void SaveModel(string filePath)
```

**Format**: Binary format with:
- Model parameters
- Training statistics
- HNSW indices (both original and embedding space)
- CRC32 checksums for corruption detection

#### LoadModel

Load previously saved model from disk.

```csharp
public static UMapModel LoadModel(string filePath)
```

**Validation**: Automatic CRC32 integrity checking

## Usage Examples

### Basic Training

```csharp
using UMAPuwotSharp;

// Create model (spectral initialization is default)
using var model = new UMapModel();

// Train with default settings
var embedding = model.Fit(
    data: trainingData,
    embeddingDimension: 2
);
```

### Advanced Configuration

```csharp
using var model = new UMapModel();

// Explicit initialization control
model.InitMethod = InitializationMethod.Spectral;  // High quality (default)

var embedding = model.FitWithProgress(
    data: trainingData,
    progressCallback: progress =>
        Console.WriteLine($"{progress.Phase}: {progress.PercentComplete:F1}%"),
    embeddingDimension: 20,
    nNeighbors: 30,
    minDist: 0.1f,
    spread: 2.0f,
    metric: DistanceMetric.Cosine,
    localConnectivity: 1.3f,
    bandwidth: 3.2f
);
```

### Production Inference with Safety

```csharp
// Transform new data with safety checks
var results = model.TransformWithSafety(newData);

foreach (var result in results)
{
    Console.WriteLine($"Confidence: {result.ConfidenceScore:F3}");
    Console.WriteLine($"Quality: {result.QualityAssessment}");

    if (result.OutlierSeverity >= OutlierLevel.ExtremeOutlier)
    {
        Console.WriteLine("⚠️  Extreme outlier - AI predictions may be unreliable");
    }

    if (result.IsProductionReady)
    {
        // Use embedding for downstream ML tasks
        ProcessEmbedding(result.Embedding);
    }
}
```

### Model Persistence

```csharp
// Train and save
using (var model = new UMapModel())
{
    var embedding = model.Fit(trainingData);
    model.SaveModel("production_model.umap");
}

// Load and use
using var loadedModel = UMapModel.LoadModel("production_model.umap");

// Access model information
var info = loadedModel.ModelInfo;
Console.WriteLine($"Training samples: {info.TrainingSamples}");
Console.WriteLine($"Input → Output: {info.InputDimension}D → {info.OutputDimension}D");
Console.WriteLine($"Metric: {info.MetricName}");
Console.WriteLine($"Local connectivity: {info.LocalConnectivity}");
Console.WriteLine($"Bandwidth: {info.Bandwidth}");

// Transform new data
var transformed = loadedModel.Transform(newData);
```

### Backward Compatibility

```csharp
// OLD API (v3.39.0 and earlier) - still works but obsolete
var model = new UMapModel();
model.AlwaysUseSpectral = true;  // ← Obsolete but functional

// NEW API (v3.40.0+) - recommended
var model = new UMapModel();
model.InitMethod = InitializationMethod.Spectral;  // ← Clear and explicit
```

## Performance Guidelines

### Memory Requirements

| Dataset Size | Typical Memory | With HNSW Optimization |
|--------------|----------------|------------------------|
| 1,000 samples | ~50MB | ~15MB (70% reduction) |
| 10,000 samples | ~240MB | ~45MB (81% reduction) |
| 50,000 samples | ~1.2GB | ~180MB (85% reduction) |
| 100,000 samples | ~2.4GB | ~360MB (85% reduction) |

### Speed Benchmarks

| Operation | Without HNSW | With HNSW | Speedup |
|-----------|--------------|-----------|---------|
| Training (10k samples) | 12 min | 8.1s | 89x |
| Transform (single) | 89ms | 2.3ms | 38x |
| Transform (batch 100) | 8.9s | 230ms | 38x |

### Initialization Time Comparison

| Dataset Size | Random Init | Spectral Init | Quality Difference |
|--------------|-------------|---------------|-------------------|
| 1,000 samples | 0.5s | 1.2s | Minor |
| 10,000 samples | 2s | 30s | Moderate |
| 50,000 samples | 10s | 5-10 min | Significant |
| 100,000 samples | 30s | 20-40 min | Critical |

**Recommendation**: Use spectral initialization (default) for best quality. For speed-critical applications with large datasets, consider `InitMethod = InitializationMethod.Auto`.

## Thread Safety

- `UMapModel` instances are **not thread-safe** for concurrent modifications
- Multiple read operations (Transform) on a fitted model are safe
- Use separate model instances for parallel processing
- Progress callbacks are invoked from the native thread

## Error Handling

All API methods throw appropriate exceptions:

```csharp
try
{
    var embedding = model.Fit(data);
}
catch (ArgumentException ex)
{
    // Invalid parameters (e.g., wrong dimensions)
}
catch (InvalidOperationException ex)
{
    // Model not fitted, or other state errors
}
catch (Exception ex)
{
    // Native library errors
}
```

## Version Compatibility

- **v3.40.0+**: `InitializationMethod` enum, spectral default
- **v3.39.0+**: `AlwaysUseSpectral` property (now obsolete)
- **v3.37.0+**: OpenMP parallelization, stringstream persistence
- **v3.33.0+**: Dual-mode exact k-NN support
- **v3.14.0+**: Dual HNSW architecture, safety features

Models saved in newer versions can be loaded by older versions (forward compatibility), but newer features may not be available.
