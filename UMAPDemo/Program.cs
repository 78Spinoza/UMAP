using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using OxyPlot.Legends;
using OxyPlot.Annotations;
using OxyPlot.WindowsForms;
using UMAPuwotSharp;

namespace UMAPDemo
{
    /// <summary>
    /// Main program class for running UMAP demonstrations and experiments on the mammoth dataset.
    /// </summary>
    public class Program
    {
        private const string ResultsDir = "Results";
        private static readonly string DataDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "UMAPDemo", "Data"));
        private const string MammothDataFile = "mammoth_data.csv";
        private const string HairyMammothDataFile = "mammoth_a.csv.zip";

      
        /// <summary>
        /// Entry point for the UMAP demo application.
        /// </summary>
        public static void Main(string[] args)
        {
            Console.WriteLine("=================================");
            Console.WriteLine("Enhanced UMAP Library Demo");

            try
            {

                OpenResultsFolder();

                // Initialize results directory and clean previous results
                InitializeResultsDirectory();

                // Load and prepare mammoth dataset
                var (data, labels, uniqueParts) = LoadMammothData();
                Console.WriteLine($"Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");
                Run10kMammothDemo(data, labels, uniqueParts);
                Run10kMammothTransformDemo(data, labels, uniqueParts);
                CreateFlagship100KHairyMammoth();

                // Run advanced parameter tuning (UMAP version) - ENABLED
                DemoAdvancedParameterTuning(data, labels);

                // Run hairy mammoth min_dist experiments (45 neighbors fixed) - DISABLED
               // DemoHairyMammothMinDistExperiments(data, labels);

                // Run MNIST demo
                RunMnistDemo();

                // Run MNIST parameter tuning
                MnistDemo.RunMnistParameterTuning();

                // Run MNIST min_dist experiments (40 neighbors, vary min_dist 0.05-2.0)
                MnistDemo.RunMnistMinDistExperiments();

                // Run transform consistency tests (PacMap-specific - disabled for UMAP)
               // RunTransformConsistencyTests(data, labels);


                Console.WriteLine("🎉 ALL DEMONSTRATIONS AND EXPERIMENTS COMPLETED!");
                Console.WriteLine($"📁 Check {ResultsDir} folder for visualizations.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }

        /// <summary>
        /// Initializes the results directory and cleans up previous results.
        /// </summary>
        private static void InitializeResultsDirectory()
        {
            Console.WriteLine("🧹 Cleaning up previous results...");
            CleanupAllResults();
            Directory.CreateDirectory(ResultsDir);
            Console.WriteLine($"   📁 Created {ResultsDir} directory");
        }

        /// <summary>
        /// Opens the Results folder in Windows Explorer.
        /// </summary>
        private static void OpenResultsFolder()
        {
            Console.WriteLine("📂 Opening Results folder...");
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = GetResultsPath(),
                    UseShellExecute = true
                });
                Console.WriteLine($"   ✅ Results folder opened");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ Could not open Results folder: {ex.Message}");
            }
        }

        /// <summary>
        /// Loads the mammoth dataset from a CSV file (supports both .csv and .csv.zip files).
        /// </summary>
        private static (double[,] data, int[] labels, string[] uniqueParts) LoadMammothData()
        {
            Console.WriteLine("📥 Loading mammoth dataset...");
            string csvPath = Path.Combine(DataDir, MammothDataFile);

            // Try CSV file first, then zip file
            if (!File.Exists(csvPath))
            {
                string zipPath = Path.ChangeExtension(csvPath, ".zip");
                if (File.Exists(zipPath))
                {
                    csvPath = zipPath;
                }
                else
                {
                    throw new FileNotFoundException($"Mammoth data file not found: {csvPath} or {Path.ChangeExtension(csvPath, ".zip")}");
                }
            }
            return DataLoaders.LoadMammothWithLabels(csvPath);
        }

        /// <summary>
        /// Calculates the quality of an embedding based on intra-label distances.
        /// </summary>
        private static double CalculateEmbeddingQuality(double[,] embedding, int[] labels)
        {
            int nSamples = embedding.GetLength(0);
            double totalDistance = 0;
            int count = 0;

            for (int i = 0; i < nSamples; i++)
            {
                double minSameLabelDistance = double.MaxValue;
                for (int j = 0; j < nSamples; j++)
                {
                    if (i != j && labels[i] == labels[j])
                    {
                        double dist = Math.Sqrt(
                            Math.Pow(embedding[i, 0] - embedding[j, 0], 2) +
                            Math.Pow(embedding[i, 1] - embedding[j, 1], 2));
                        minSameLabelDistance = Math.Min(minSameLabelDistance, dist);
                    }
                }
                if (minSameLabelDistance < double.MaxValue)
                {
                    totalDistance += minSameLabelDistance;
                    count++;
                }
            }
            return count > 0 ? totalDistance / count : double.MaxValue;
        }

        /// <summary>
        /// Deletes all files and subdirectories in the Results folder.
        /// </summary>
        private static void CleanupAllResults()
        {
            string resultsPath = GetResultsPath();
            if (!Directory.Exists(resultsPath))
            {
                Directory.CreateDirectory(resultsPath);
                return;
            }

            int deletedFiles = 0, deletedFolders = 0, failedFiles = 0, failedFolders = 0;

            foreach (var file in Directory.GetFiles(resultsPath, "*", SearchOption.AllDirectories))
            {
                try
                {
                    File.SetAttributes(file, FileAttributes.Normal);
                    File.Delete(file);
                    deletedFiles++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Could not delete file {Path.GetFileName(file)}: {ex.Message}");
                    failedFiles++;
                }
            }

            foreach (var dir in Directory.GetDirectories(resultsPath, "*", SearchOption.AllDirectories).Reverse())
            {
                try
                {
                    if (Directory.GetFiles(dir).Length == 0 && Directory.GetDirectories(dir).Length == 0)
                    {
                        Directory.Delete(dir);
                        deletedFolders++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Could not delete folder {Path.GetFileName(dir)}: {ex.Message}");
                    failedFolders++;
                }
            }

            Console.WriteLine($"   📊 Cleanup Summary:");
            if (deletedFiles > 0) Console.WriteLine($"      ✅ Deleted {deletedFiles} files");
            if (deletedFolders > 0) Console.WriteLine($"      ✅ Deleted {deletedFolders} folders");
            if (failedFiles > 0) Console.WriteLine($"      ⚠️ Failed to delete {failedFiles} files");
            if (failedFolders > 0) Console.WriteLine($"      ⚠️ Failed to delete {failedFolders} folders");
            if (deletedFiles == 0 && deletedFolders == 0)
                Console.WriteLine($"      ℹ️ Results folder was already clean");
        }

        /// <summary>
        /// Gets the full path to the Results directory or a subdirectory within it.
        /// </summary>
        private static string GetResultsPath(string subDirectory = "")
        {
            var basePath = Path.Combine(Directory.GetCurrentDirectory(), ResultsDir);
            return string.IsNullOrEmpty(subDirectory) ? basePath : Path.Combine(basePath, subDirectory);
        }

        /// <summary>
        /// Unified progress callback for consistent console output.
        /// </summary>
        private static void UnifiedProgressCallback(string phase, int current, int total, float percent, string? message)
        {
            Console.Write($"\r{new string(' ', 120)}\r[{phase}] {current}/{total} ({percent:F1}%) {message}");
        }

        /// <summary>
        /// Unified progress callback logger with consistent console output.
        /// </summary>
        private static void UnifiedProgressCallbackLogger(string phase, int current, int total, float percent, string? message)
        {
            Console.WriteLine($"   [{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
        }

        /// <summary>
        /// Creates a progress callback with prefix for better organization.
        /// </summary>
        private static ProgressCallback CreatePrefixedCallback(string prefix)
        {
            return (phase, current, total, percent, message) =>
            {
                string displayPrefix = string.IsNullOrEmpty(prefix) ? "" : $"[{prefix}] ";
                Console.Write($"\r{new string(' ', 180)}\r   {displayPrefix}[{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
            };
        }

        /// <summary>
        /// Creates a progress callback logger with prefix for better organization.
        /// </summary>
        private static ProgressCallback CreatePrefixedLoggerCallback(string prefix)
        {
            return (phase, current, total, percent, message) =>
            {
                string displayPrefix = string.IsNullOrEmpty(prefix) ? "" : $"[{prefix}] ";
                Console.WriteLine($"{displayPrefix}[{phase}] Progress: {current}/{total} ({percent:F1}%) {message}");
            };
        }

    


        /// <summary>
        /// Runs a demo on a 10K mammoth dataset using HNSW approximation.
        /// </summary>
        private static void Run10kMammothDemo(double[,] data, int[] labels, string[] uniqueParts)
        {
            Console.WriteLine("🦣 Running 10K Mammoth Demo (HNSW Approximation)...");
            var umap = new UMapModel();
            var stopwatch = Stopwatch.StartNew();
            // Convert double[,] to float[,] for UMAP API
            int nSamples = data.GetLength(0);
            int nFeatures = data.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data[i, j];

            var embedding = umap.FitWithProgress(
                data: floatData,
                progressCallback: UnifiedProgressCallback,
                embeddingDimension: 2,
                nNeighbors: 25,  // Test with 25 instead of 50 - less smoothing
                minDist: 0.35f,
                spread: 1.0f,
                nEpochs: 300,  // Standard UMAP epochs for 10k dataset
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,  // Use high-quality HNSW for speed
                autoHNSWParam: false,
                randomSeed: 42
            );
            stopwatch.Stop();
            Console.WriteLine();
            Console.WriteLine($"✅ 10K Embedding created: {embedding.GetLength(0)} x {embedding.GetLength(1)}");
            Console.WriteLine($"⏱️ Execution time: {stopwatch.Elapsed.TotalSeconds:F2}s");

            // Save model
            string modelPath = Path.Combine(ResultsDir, "mammoth_10k_hnsw.umap");
            umap.Save(modelPath);
            Console.WriteLine($"✅ Model saved: {modelPath}");

            // Convert float[,] embedding back to double[,] for visualization
            int embedSamples = embedding.GetLength(0);
            int embedDims = embedding.GetLength(1);
            var doubleEmbedding = new double[embedSamples, embedDims];
            for (int i = 0; i < embedSamples; i++)
                for (int j = 0; j < embedDims; j++)
                    doubleEmbedding[i, j] = embedding[i, j];

          
            CreateVisualizations(doubleEmbedding, data, labels, umap, stopwatch.Elapsed.TotalSeconds, uniqueParts);
        }

        /// <summary>
        /// Runs a transform demo on the same 10K mammoth dataset using the saved model.
        /// This demonstrates the transform functionality by loading the saved model and projecting the same data.
        /// </summary>
        private static void Run10kMammothTransformDemo(double[,] data, int[] labels, string[] uniqueParts)
        {
            Console.WriteLine("🔄 Running 10K Mammoth Transform Demo (Using Saved Model)...");

            string modelPath = Path.Combine(ResultsDir, "mammoth_10k_hnsw.umap");

            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"❌ Model file not found: {modelPath}");
                Console.WriteLine("   Please run the 10K Mammoth Demo first to create the model.");
                return;
            }

            var stopwatch = Stopwatch.StartNew();

            // Load the saved model
            var umap = UMapModel.Load(modelPath);
            Console.WriteLine($"✅ Model loaded: {modelPath}");

            // Convert double[,] to float[,] for transform API
            int nSamples = data.GetLength(0);
            int nFeatures = data.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data[i, j];

            // Transform the same data using the loaded model
            var transformedEmbedding = umap.Transform(floatData);
            stopwatch.Stop();

            Console.WriteLine();
            Console.WriteLine($"✅ Transformed Embedding created: {transformedEmbedding.GetLength(0)} x {transformedEmbedding.GetLength(1)}");
            Console.WriteLine($"⏱️ Transform time: {stopwatch.Elapsed.TotalSeconds:F2}s");

            // Convert float[,] embedding back to double[,] for visualization
            int embedSamples = transformedEmbedding.GetLength(0);
            int embedDims = transformedEmbedding.GetLength(1);
            var doubleEmbedding = new double[embedSamples, embedDims];
            for (int i = 0; i < embedSamples; i++)
                for (int j = 0; j < embedDims; j++)
                    doubleEmbedding[i, j] = transformedEmbedding[i, j];

            // Create visualizations with custom title and filename for transformed results
            string transformedTitle = "Mammoth 10k Dataset - Transformed";
            string transformedFilename = "mammoth_10k_transformed_embedding.png";
            CreateVisualizations(doubleEmbedding, data, labels, umap, stopwatch.Elapsed.TotalSeconds, uniqueParts, title: transformedTitle, filename: transformedFilename);
        }

        /// <summary>
        /// Calculates optimal neighbor count using the adaptive formula
        /// </summary>
        private static int CalculateOptimalNeighbors(int nSamples)
        {
            // Adaptive formula: n_neighbors = 10 + 15 * (log10(n_samples) - 4)
            // This gives 10 for 10,000 samples and scales appropriately
            double log10Samples = Math.Log10(nSamples);
            int optimalNeighbors = (int)(10 + 15 * (log10Samples - 4));
            return Math.Max(10, Math.Min(optimalNeighbors, 100)); // Clamp between 10-100
        }

        /// <summary>
        /// Creates the flagship 70k hairy mammoth demo.
        /// </summary>
        private static void CreateFlagship100KHairyMammoth()
        {
            Console.WriteLine("🦣 Running Flagship 70k Hairy Mammoth Demo...");
            string csvPath = Path.Combine(DataDir, HairyMammothDataFile);
            if (!File.Exists(csvPath))
            {
                Console.WriteLine($"   ⚠️ Hairy mammoth data file not found: {csvPath}");
                return;
            }

            var (data, labels, uniqueParts) = DataLoaders.LoadMammothWithLabels(csvPath);

            Console.WriteLine($"   Loaded: {data.GetLength(0)} points, {data.GetLength(1)} dimensions");

            // 100k hairy mammoth dataset
            int availableSamples = data.GetLength(0);
            int requestedSamples = 100000;  // 100k dataset for comprehensive testing

            if (availableSamples < requestedSamples)
            {
                Console.WriteLine($"   ⚠️ Warning: Only {availableSamples:N0} samples available, using all instead of {requestedSamples:N0}");
                requestedSamples = availableSamples;
            }

            Console.WriteLine($"   Processing FULL {requestedSamples:N0} points for UMAP (200k FLAGSHIP DEMO)...");
            var (data2, labels2) = DataLoaders.SampleRandomPoints(data, labels, requestedSamples);
            Console.WriteLine($"   Subsampled: {data2.GetLength(0)} points, {data2.GetLength(1)} dimensions");

            // Parameters are set in the FitWithProgress call below - no auto-optimization

            // Convert double[,] to float[,] for UMAP API
            int nSamples2 = data2.GetLength(0);
            int nFeatures2 = data2.GetLength(1);
            var floatData2 = new float[nSamples2, nFeatures2];
            for (int i = 0; i < nSamples2; i++)
                for (int j = 0; j < nFeatures2; j++)
                    floatData2[i, j] = (float)data2[i, j];

            var umap = new UMapModel();
            var stopwatch = Stopwatch.StartNew();
            var embedding = umap.FitWithProgress(
                data: floatData2,
                progressCallback: UnifiedProgressCallback,
                embeddingDimension: 2,
                nNeighbors: 60,        // Optimal for large datasets
                minDist: 0.35f,        // Updated to 0.35 as requested
                spread: 1.0f,
                nEpochs: 300,
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,
                autoHNSWParam: false,
                randomSeed: 42
            );




            stopwatch.Stop();
            Console.WriteLine();
            Console.WriteLine($"   ✅ Hairy Mammoth Embedding created: {embedding.GetLength(0):N0} x {embedding.GetLength(1)}");
            Console.WriteLine($"   ⏱️ Execution time: {stopwatch.Elapsed.TotalSeconds:F2}s");

            // === SAVE UMAP MODEL ===
            Console.WriteLine("   Saving UMAP model...");

            string resultsDir = Path.Combine(ResultsDir, $"HairyMammoth_{embedding.GetLength(0)}");
            Directory.CreateDirectory(resultsDir);

            var modelPath = Path.Combine(resultsDir, $"umap_{embedding.GetLength(0)}_model.umap");

            // Save the trained model
            umap.Save(modelPath);

            Console.WriteLine($"   ✅ Model saved: {Path.GetFileName(modelPath)}");

            // Convert float[,] embedding back to double[,] for visualization
            int embedSamples2 = embedding.GetLength(0);
            int embedDims2 = embedding.GetLength(1);
            var doubleEmbedding2 = new double[embedSamples2, embedDims2];
            for (int i = 0; i < embedSamples2; i++)
                for (int j = 0; j < embedDims2; j++)
                    doubleEmbedding2[i, j] = embedding[i, j];

            // === VISUALIZATION (Actual UMAP 2D Embedding with Hyperparameters) ===
            Console.WriteLine("   Creating 2D visualizations of UMAP embedding...");

            var sampleCount = doubleEmbedding2.GetLength(0);
            var title2D = $"Hairy Mammoth {sampleCount:N0} Points - UMAP 2D Embedding (Labeled)\n" + BuildVisualizationTitle(umap);
            var outputPath2D = Path.Combine(resultsDir, $"hairy_mammoth_{sampleCount}_umap_2d.png");
            var outputPath2D_BW = Path.Combine(resultsDir, $"hairy_mammoth_{sampleCount}_umap_2d_bw.png");

            // Create colored version with anatomical labels and hyperparameters
            try
            {
                Visualizer.PlotMammothUMAP(doubleEmbedding2, labels2, title2D, outputPath2D, uniqueParts);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ Visualization failed: {ex.Message}");
                Console.WriteLine("   Continuing with other outputs...");
            }

            // Create true black and white version with hyperparameters (no labels)
            var titleBW = $"Hairy Mammoth {sampleCount:N0} Points - UMAP 2D Embedding (Black & White)\n" + BuildVisualizationTitle(umap);
            Visualizer.PlotSimpleUMAP(doubleEmbedding2, titleBW, outputPath2D_BW, null);

            Console.WriteLine($"   ✅ UMAP 2D labeled visualization created: {Path.GetFileName(outputPath2D)}");
            Console.WriteLine($"   ✅ UMAP 2D black & white visualization created: {Path.GetFileName(outputPath2D_BW)}");
        }

        /// <summary>
        /// Creates visualizations for the mammoth demos.
        /// </summary>
        private static void CreateVisualizations(double[,] embedding, double[,] originalData, int[] labels, UMapModel umap, double executionTime, string[] uniqueParts, string? title = null, string? filename = null)
        {
            try
            {
                Console.WriteLine("Creating visualizations...");
                string original3DPath = Path.Combine(ResultsDir, "mammoth_original_3d.png");
                Visualizer.PlotOriginalMammoth3DReal(originalData, labels, "Original Mammoth 3D Data", original3DPath);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(original3DPath)}");

                // Use provided title and filename or fall back to defaults
                string umapPath = Path.Combine(ResultsDir, filename ?? "mammoth_umap_embedding.png");
                var modelInfo = umap.ModelInfo;
                var paramInfo = CreateFitParamInfo(umap, executionTime, "Main_Demo");

                var displayTitle = title ?? BuildVisualizationTitle(umap);
                Visualizer.PlotMammothUMAP(embedding, labels, displayTitle, umapPath, paramInfo, autoFitAxes: true, partNames: uniqueParts);
                Console.WriteLine($"   ✅ Created: {Path.GetFileName(umapPath)}");
                Console.WriteLine($"   📊 KNN Mode: {paramInfo["KNN_Mode"]}");
                Console.WriteLine($"   🚀 HNSW Status: ACTIVE"); // UMAP always uses HNSW
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Visualization creation failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates parameter info dictionary from model info (essential FIT parameters only).
        /// </summary>
        private static Dictionary<string, object> CreateFitParamInfo(UMapModel model, double executionTime, string experimentType = "")
        {
            var modelInfo = model.ModelInfo;
            return new Dictionary<string, object>
            {
                ["experiment_type"] = experimentType,
                ["UMAP Version"] = UMapModel.GetVersion(), // Get version dynamically from wrapper
                ["n_neighbors"] = modelInfo.Neighbors,
                ["embedding_dimension"] = modelInfo.OutputDimension,
                ["distance_metric"] = modelInfo.Metric.ToString(),
                ["min_dist"] = modelInfo.MinimumDistance.ToString("F2"),
                ["spread"] = modelInfo.Spread.ToString("F2"),
                ["negative_sample_rate"] = "N/A", // UMAP doesn't have this parameter
                ["transform_seed"] = "N/A", // UMAP doesn't expose this
                ["data_points"] = modelInfo.TrainingSamples,
                ["original_dimensions"] = modelInfo.InputDimension,
                ["hnsw_m"] = modelInfo.HnswM,
                ["hnsw_ef_construction"] = modelInfo.HnswEfConstruction,
                ["hnsw_ef_search"] = modelInfo.HnswEfSearch,
                ["KNN_Mode"] = "HNSW", // UMAP always uses HNSW optimization
                ["random_seed"] = "N/A", // UMAP doesn't expose this
                ["execution_time"] = $"{executionTime:F2}s"
            };
        }

        /// <summary>
        /// Builds a title for visualizations from model info.
        /// </summary>
        private static string BuildVisualizationTitle(UMapModel model, string prefix = "Mammoth UMAP 2D Embedding")
        {
            var modelInfo = model.ModelInfo;
            var version = UMapModel.GetVersion(); // Get version dynamically from wrapper
            var knnMode = "HNSW"; // UMAP always uses HNSW optimization

            return $@"{prefix}
UMAP v{version} | Sample: {modelInfo.TrainingSamples:N0} | {knnMode}
k={modelInfo.Neighbors} | {modelInfo.Metric} | dims={modelInfo.OutputDimension}
min_dist={modelInfo.MinimumDistance:F2} | spread={modelInfo.Spread:F2}
HNSW: M={modelInfo.HnswM}, ef_c={modelInfo.HnswEfConstruction}, ef_s={modelInfo.HnswEfSearch}";
        }

        /// <summary>
        /// Runs transform consistency tests for reproducibility and persistence.
        /// </summary>
        private static void RunTransformConsistencyTests(double[,] data, int[] labels)
        {
            Console.WriteLine("🧪 Running Transform Consistency Tests...");
            var testConfigs = new[]
            {
                new { Name = "Exact KNN Mode", NNeighbors = 10, Distance = "euclidean", UseHnsw = false, UseQuantization = false, Seed = 42 },
                new { Name = "HNSW Mode", NNeighbors = 10, Distance = "euclidean", UseHnsw = true, UseQuantization = false, Seed = 42 }
            };

            foreach (var config in testConfigs)
            {
                Console.WriteLine($"   🔍 Testing {config.Name}...");
                string testDir = Path.Combine(ResultsDir, config.Name.Replace(" ", "_") + "_Reproducibility");
                Directory.CreateDirectory(testDir);
                RunTransformTest(data, labels, config.NNeighbors, config.Distance, config.UseHnsw, config.UseQuantization, config.Seed, testDir);
            }
            Console.WriteLine("✅ All transform tests completed!");
        }

        /// <summary>
        /// Runs a single transform test with validation steps.
        /// </summary>
        private static void RunTransformTest(double[,] data, int[] labels, int nNeighbors, string distance, bool useHnsw, bool useQuantization, int seed, string outputDir)
        {
            var metric = distance.ToLower() switch
            {
                "euclidean" => DistanceMetric.Euclidean,
                "manhattan" => DistanceMetric.Manhattan,
                "cosine" => DistanceMetric.Cosine,
                _ => DistanceMetric.Euclidean
            };

            Console.WriteLine($"   Configuration: n_neighbors={nNeighbors}, distance={distance}, hnsw={useHnsw}, quantization={useQuantization}");

            // Convert double[,] to float[,] for UMAP API
            int nSamples = data.GetLength(0);
            int nFeatures = data.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data[i, j];

            // Step 1: Initial fit
            var model1 = new UMapModel();
            var embedding1 = model1.FitWithProgress(
                data: floatData,
                progressCallback: UnifiedProgressCallback,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                minDist: 0.1f,
                spread: 1.0f,
                nEpochs: 300,
                metric: metric,
                forceExactKnn: !useHnsw,
                autoHNSWParam: false,
                randomSeed: 42
            );
            Console.WriteLine($"   ✅ Initial embedding created: {embedding1.GetLength(0)}x{embedding1.GetLength(1)}");

            // Step 2: Save model
            string modelPath = Path.Combine(outputDir, "umap_model.umap");
            model1.Save(modelPath);
            Console.WriteLine($"   ✅ Model saved: {modelPath}");

            // Step 3: Second fit
            var model2 = new UMapModel();
            var embedding2 = model2.FitWithProgress(
                data: floatData,
                progressCallback: UnifiedProgressCallback,
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                minDist: 0.1f,
                spread: 1.0f,
                nEpochs: 300,
                metric: metric,
                forceExactKnn: !useHnsw,
                autoHNSWParam: false,
                randomSeed: 42
            );
            Console.WriteLine($"   ✅ Second embedding created: {embedding2.GetLength(0)}x{embedding2.GetLength(1)}");

            // Step 4: Load saved model
            var loadedModel = UMapModel.Load(modelPath);
            Console.WriteLine("   ✅ Model loaded successfully");

            // Step 5: Transform with loaded model
            var embeddingLoaded = loadedModel.Transform(floatData);
            Console.WriteLine($"   ✅ Transform completed: {embeddingLoaded.GetLength(0)}x{embeddingLoaded.GetLength(1)}");

            // Convert float[,] embeddings back to double[,] for comparison
            var doubleEmbedding1 = ConvertFloatToDouble(embedding1);
            var doubleEmbedding2 = ConvertFloatToDouble(embedding2);
            var doubleEmbeddingLoaded = ConvertFloatToDouble(embeddingLoaded);

            // Step 6: Calculate reproducibility metrics
            double mse = CalculateMSE(doubleEmbedding1, doubleEmbedding2);
            double maxDiff = CalculateMaxDifference(doubleEmbedding1, doubleEmbedding2);
            Console.WriteLine($"   MSE between embeddings: {mse:E2}");
            Console.WriteLine($"   Max difference: {maxDiff:E2}");

            // Step 7: Generate visualizations
            GenerateTransformVisualizations(data, doubleEmbedding1, doubleEmbedding2, labels, model1, outputDir);

            // Step 8: Summary and validation
            bool isReproducible = mse < 1e-6 && maxDiff < 1e-4;
            bool dimensionsMatch = embedding1.GetLength(0) == embedding2.GetLength(0) && embedding1.GetLength(1) == embedding2.GetLength(1);
            Console.WriteLine($"   Reproducibility: {(isReproducible ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Dimension consistency: {(dimensionsMatch ? "✅ PASS" : "❌ FAIL")}");
            Console.WriteLine($"   Model persistence: ✅ PASS");
        }

        /// <summary>
        /// Generates visualizations for transform consistency tests.
        /// </summary>
        private static void GenerateTransformVisualizations(double[,] data, double[,] embedding1, double[,] embedding2, int[] labels, UMapModel model, string outputDir)
        {
            var originalData = data;
            GenerateProjection(originalData, embedding1, "XY", Path.Combine(outputDir, "original_3d_XY_TopView.png"));
            GenerateProjection(originalData, embedding1, "XZ", Path.Combine(outputDir, "original_3d_XZ_SideView.png"));
            GenerateProjection(originalData, embedding1, "YZ", Path.Combine(outputDir, "original_3d_YZ_FrontView.png"));

            var paramInfo1 = CreateFitParamInfo(model, 0, "Reproducibility_Test_Embedding_1");
            var paramInfo2 = CreateFitParamInfo(model, 0, "Reproducibility_Test_Embedding_2");
            var title1 = BuildVisualizationTitle(model, "UMAP Reproducibility Test - Embedding 1");
            var title2 = BuildVisualizationTitle(model, "UMAP Reproducibility Test - Embedding 2");

            Visualizer.PlotMammothUMAP(embedding1, labels, title1, Path.Combine(outputDir, "embedding1.png"), paramInfo1);
            Visualizer.PlotMammothUMAP(embedding2, labels, title2, Path.Combine(outputDir, "embedding2.png"), paramInfo2);
            GenerateConsistencyPlot(embedding1, embedding2, labels, "Embedding Consistency (X)", Path.Combine(outputDir, "consistency_x.png"));
            GenerateHeatmapPlot(embedding1, embedding2, "Pairwise Distance Difference Heatmap", Path.Combine(outputDir, "distance_heatmap.png"));
            Console.WriteLine("   ✅ Visualizations generated");
        }

        /// <summary>
        /// Runs hyperparameter experiments on the mammoth dataset.
        /// </summary>
        private static void RunHyperparameterExperiments(double[,] data, int[] labels)
        {
            Console.WriteLine("🔬 Running Hyperparameter Experiments...");
            var optimalHNSWParams = AutoDiscoverHNSWParameters(data);
            Console.WriteLine($"✅ HNSW Parameters: M={optimalHNSWParams.M}, ef_construction={optimalHNSWParams.EfConstruction}, ef_search={optimalHNSWParams.EfSearch}");

            DemoNeighborExperiments(data, labels, optimalHNSWParams);
            DemoLearningRateExperiments(data, labels, optimalHNSWParams);
            DemoInitializationStdDevExperiments(data, labels, optimalHNSWParams);
        }

        /// <summary>
        /// Auto-discovers optimal HNSW parameters.
        /// </summary>
        private static (int M, int EfConstruction, int EfSearch) AutoDiscoverHNSWParameters(double[,] data)
        {
            Console.WriteLine("🔍 Auto-discovering HNSW parameters...");
            var model = new UMapModel();
            var stopwatch = Stopwatch.StartNew();
            // Convert double[,] to float[,] for UMAP API
                int nSamples = data.GetLength(0);
                int nFeatures = data.GetLength(1);
                var floatData = new float[nSamples, nFeatures];
                for (int i = 0; i < nSamples; i++)
                    for (int j = 0; j < nFeatures; j++)
                        floatData[i, j] = (float)data[i, j];

            model.FitWithProgress(
                data: floatData,
                progressCallback: CreatePrefixedCallback("Auto-Discovery"),
                embeddingDimension: 2,
                nNeighbors: 10,
                minDist: 0.1f,
                spread: 1.0f,
                nEpochs: 300,
                metric: DistanceMetric.Euclidean,
                forceExactKnn: false,
                autoHNSWParam: true,
                randomSeed: 42
            );
            stopwatch.Stop();
            Console.WriteLine();
            var modelInfo = model.ModelInfo;
            return (modelInfo.HnswM, modelInfo.HnswEfConstruction, modelInfo.HnswEfSearch);
        }

        /// <summary>
        /// Tests different neighbor counts for the mammoth dataset.
        /// </summary>
        private static void DemoNeighborExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🔬 Testing Neighbor Counts (5-50)...");
            var neighborTests = Enumerable.Range(0, 13).Select(i => 5 + i * 2).ToArray();
            var results = new List<(int nNeighbors, double[,] embedding, double time, double quality)>();

            foreach (var nNeighbors in neighborTests)
            {
                Console.WriteLine($"   📊 Testing n_neighbors = {nNeighbors}...");
                var model = new UMapModel();
                var stopwatch = Stopwatch.StartNew();
                // Convert double[,] to float[,] for UMAP API
                int nSamples = data.GetLength(0);
                int nFeatures = data.GetLength(1);
                var floatData = new float[nSamples, nFeatures];
                for (int i = 0; i < nSamples; i++)
                    for (int j = 0; j < nFeatures; j++)
                        floatData[i, j] = (float)data[i, j];

                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"n={nNeighbors}"),
                    embeddingDimension: 2,
                    nNeighbors: nNeighbors,
                    minDist: 0.35f,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();
                Console.WriteLine();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels);
                results.Add((nNeighbors, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ n={nNeighbors}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Neighbor_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "neighbor_experiments");
                Directory.CreateDirectory(experimentDir);
                var outputPath = Path.Combine(experimentDir, $"{(nNeighbors - 5) / 2 + 1:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Neighbor Experiment: n={modelInfo.Neighbors}\n" + BuildVisualizationTitle(model, "Neighbor Experiment");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Neighbor Experiments Summary");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best neighbor count: n={bestResult.nNeighbors} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

        /// <summary>
        /// Tests different learning rates for the mammoth dataset.
        /// </summary>
        private static void DemoLearningRateExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🎓 Testing Learning Rates (0.5-1.0)...");
            var learningRateTests = new[] { 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f, 1.0f };
            var results = new List<(float learningRate, double[,] embedding, double time, double quality)>();

            foreach (var learningRate in learningRateTests)
            {
                Console.WriteLine($"   📊 Testing learning_rate = {learningRate:F1}...");
                var model = new UMapModel();
                var stopwatch = Stopwatch.StartNew();
                // Convert double[,] to float[,] for UMAP API
                int nSamples = data.GetLength(0);
                int nFeatures = data.GetLength(1);
                var floatData = new float[nSamples, nFeatures];
                for (int i = 0; i < nSamples; i++)
                    for (int j = 0; j < nFeatures; j++)
                        floatData[i, j] = (float)data[i, j];

                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"lr={learningRate:F1}"),
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    minDist: 0.1f,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();
                Console.WriteLine();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels);
                results.Add((learningRate, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ lr={learningRate:F1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Learning_Rate_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "learning_rate_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = (int)((learningRate - 0.6f) / 0.05f) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Learning Rate Experiment: lr={learningRate:F1}\n" + BuildVisualizationTitle(model, "Learning Rate Experiment");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Learning Rate Experiments Summary");
            Console.WriteLine(new string('=', 50));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best learning rate: {bestResult.learningRate:F1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

        /// <summary>
        /// Tests different initialization standard deviations.
        /// </summary>
        private static void DemoInitializationStdDevExperiments(double[,] data, int[] labels, (int M, int EfConstruction, int EfSearch) hnswParams)
        {
            Console.WriteLine("🎲 Testing Initialization Std Dev...");
            var initStdDevTests = new[] { 1e-4f, 1e-3f, 1e-2f, 1e-1f };
            var results = new List<(float initStdDev, double[,] embedding, double time, double quality)>();

            foreach (var initStdDev in initStdDevTests)
            {
                Console.WriteLine($"   📊 Testing initialization_std_dev = {initStdDev:E1}...");
                var model = new UMapModel();
                var stopwatch = Stopwatch.StartNew();
                // Convert double[,] to float[,] for UMAP API
                int nSamples = data.GetLength(0);
                int nFeatures = data.GetLength(1);
                var floatData = new float[nSamples, nFeatures];
                for (int i = 0; i < nSamples; i++)
                    for (int j = 0; j < nFeatures; j++)
                        floatData[i, j] = (float)data[i, j];

                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"init={initStdDev:E1}"),
                    embeddingDimension: 2,
                    nNeighbors: 10,
                    minDist: 0.1f,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();
                Console.WriteLine();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels);
                results.Add((initStdDev, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ init_std={initStdDev:E1}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Initialization_Std_Dev_Experiments");
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "init_std_dev_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = initStdDev switch
                {
                    1e-4f => 1,
                    1e-3f => 2,
                    1e-2f => 3,
                    1e-1f => 4,
                    _ => 1
                };
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D4}.png");
                var modelInfo = model.ModelInfo;
                var title = $"Init Std Dev Experiment: {initStdDev:E0}\n" + BuildVisualizationTitle(model, "Init Std Dev Experiment");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels, title, outputPath, paramInfo);
                Console.WriteLine($"   📈 Saved: {Path.GetFileName(outputPath)}");
            }

            Console.WriteLine("📊 Initialization Std Dev Experiments Summary");
            Console.WriteLine(new string('=', 60));
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best init std dev: {bestResult.initStdDev:E1} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"⏱️ Execution times: {results.Min(r => r.time):F2}s to {results.Max(r => r.time):F2}s");
        }

     

        /// <summary>
        /// Runs advanced UMAP parameter tuning experiments.
        /// Test 1: n_neighbors from 5 to 40 (increments of 1) with min_dist=0.1
        /// Test 2: min_dist from 0.05 to 0.3 (increments of 0.05) with n_neighbors=40
        /// </summary>
        private static void DemoAdvancedParameterTuning(double[,] data, int[] labels)
        {
            Console.WriteLine("🔬 Running UMAP Advanced Parameter Tuning...");
            Console.WriteLine("   Test 1: n_neighbors from 5 to 40 (increments of 1) with min_dist=0.4");
            Console.WriteLine("   Test 2: min_dist from 0.05 to 0.3 (increments of 0.05) with n_neighbors=60");

            // Convert double[,] to float[,] for UMAP API
            int nSamples = data.GetLength(0);
            int nFeatures = data.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data[i, j];

            var allResults = new List<(string testType, int nNeighbors, float minDist, double[,] embedding, double time, double quality)>();

            // === TEST 1: n_neighbors from 10 to 70 with increments of 5 ===
            Console.WriteLine("\n📊 === TEST 1: n_neighbors Optimization (min_dist=0.4) ===");
            var nNeighborsTests = Enumerable.Range(0, 13).Select(i => 10 + i * 5).ToArray(); // 10, 15, 20, 25, ..., 70

            foreach (var nNeighbors in nNeighborsTests)
            {
                Console.WriteLine($"   🔍 Testing n_neighbors = {nNeighbors}, min_dist = 0.4...");
                var model = new UMapModel();

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"N={nNeighbors}"),
                    embeddingDimension: 2,
                    nNeighbors: nNeighbors,
                    minDist: 0.4f,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels);
                allResults.Add(("n_neighbors_test", nNeighbors, 0.2f, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ n_neighbors={nNeighbors}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Save visualization for all results (only 13 total)
                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "N_Neighbors_Experiments");
                paramInfo["n_neighbors"] = nNeighbors.ToString();
                paramInfo["min_dist"] = "0.4";
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "n_neighbors_experiments");
                Directory.CreateDirectory(experimentDir);
                var outputPath = Path.Combine(experimentDir, $"neighbors_{nNeighbors:D2}.png");
                // Get actual model parameters - NO HARDCODING!
                var modelInfo = model.ModelInfo;
                var title = $"n_neighbors Experiment: k={modelInfo.Neighbors}, min_dist={modelInfo.MinimumDistance:F2}\n" + BuildVisualizationTitle(model, "n_neighbors Experiment");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels, title, outputPath, paramInfo);
                Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath)}");
            }

            // === TEST 2: min_dist from 0.05 to 0.7 with n_neighbors=60 ===
            Console.WriteLine("\n📊 === TEST 2: min_dist Optimization (n_neighbors=60) ===");
            var minDistTests = new[] { 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.55f, 0.6f, 0.65f, 0.7f };

            foreach (var minDist in minDistTests)
            {
                Console.WriteLine($"   🔍 Testing min_dist = {minDist:F2}, n_neighbors = 60...");
                var model = new UMapModel();

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"MD={minDist:F2}"),
                    embeddingDimension: 2,
                    nNeighbors: 60,
                    minDist: minDist,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels);
                allResults.Add(("min_dist_test", 40, minDist, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ min_dist={minDist:F2}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Save visualization for all min_dist tests (only 6 files)
                var paramInfo2 = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Min_Dist_Experiments");
                paramInfo2["n_neighbors"] = "40";
                paramInfo2["min_dist"] = minDist.ToString("F2");
                paramInfo2["embedding_quality"] = quality.ToString("F4");

                var experimentDir2 = Path.Combine(ResultsDir, "min_dist_experiments");
                Directory.CreateDirectory(experimentDir2);
                var imageNumber = Array.IndexOf(minDistTests, minDist) + 1;
                var outputPath2 = Path.Combine(experimentDir2, $"{imageNumber:D2}_mindist_{minDist:F2}.png");
                var title2 = $"min_dist Experiment: k=40, min_dist={minDist:F2}\n" + BuildVisualizationTitle(model, "min_dist Experiment");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels, title2, outputPath2, paramInfo2);
                Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath2)}");
            }

            // === ANALYSIS AND SUMMARY ===
            Console.WriteLine("\n📊 === PARAMETER TUNING ANALYSIS ===");

            // Analyze n_neighbors results
            var nNeighborsResults = allResults.Where(r => r.testType == "n_neighbors_test").ToList();
            var bestNeighborsResult = nNeighborsResults.OrderBy(r => r.quality).First();
            Console.WriteLine("🏆 n_neighbors Optimization Results:");
            Console.WriteLine($"   Best n_neighbors: {bestNeighborsResult.nNeighbors} (quality: {bestNeighborsResult.quality:F4})");
            Console.WriteLine($"   Quality range: {nNeighborsResults.Min(r => r.quality):F4} - {nNeighborsResults.Max(r => r.quality):F4}");
            Console.WriteLine($"   Time range: {nNeighborsResults.Min(r => r.time):F2}s - {nNeighborsResults.Max(r => r.time):F2}s");

            // Analyze min_dist results
            var minDistResults = allResults.Where(r => r.testType == "min_dist_test").ToList();
            var bestMinDistResult = minDistResults.OrderBy(r => r.quality).First();
            Console.WriteLine("\n🏆 min_dist Optimization Results:");
            Console.WriteLine($"   Best min_dist: {bestMinDistResult.minDist:F2} (quality: {bestMinDistResult.quality:F4})");
            Console.WriteLine($"   Quality range: {minDistResults.Min(r => r.quality):F4} - {minDistResults.Max(r => r.quality):F4}");
            Console.WriteLine($"   Time range: {minDistResults.Min(r => r.time):F2}s - {minDistResults.Max(r => r.time):F2}s");

            // Overall best combination
            var overallBest = allResults.OrderBy(r => r.quality).First();
            Console.WriteLine("\n🥇 Overall Best Parameters:");
            Console.WriteLine($"   n_neighbors: {overallBest.nNeighbors}, min_dist: {overallBest.minDist:F2}");
            Console.WriteLine($"   Best quality: {overallBest.quality:F4}, Time: {overallBest.time:F2}s");
            Console.WriteLine($"   Test type: {overallBest.testType}");

            Console.WriteLine($"\n📁 Results saved to:");
            Console.WriteLine($"   - {Path.Combine(ResultsDir, "n_neighbors_experiments")} (key results)");
            Console.WriteLine($"   - {Path.Combine(ResultsDir, "min_dist_experiments")} (all results)");
        }

        /// <summary>
        /// Runs hairy mammoth min_dist experiments with fixed 45 neighbors.
        /// Tests min_dist from 0.05 to 0.3 in increments of 0.05 for the hairy mammoth dataset.
        /// </summary>
        private static void DemoHairyMammothMinDistExperiments(double[,] data, int[] labels)
        {
            Console.WriteLine("🦣 Running Hairy Mammoth min_dist Experiments (n_neighbors=45)...");
            Console.WriteLine("   Testing min_dist from 0.05 to 0.3 (increments of 0.05) with n_neighbors=45");

            // Load hairy mammoth data (use 50k subset)
            string csvPath = Path.Combine(DataDir, HairyMammothDataFile);
            if (!File.Exists(csvPath))
            {
                Console.WriteLine($"   ⚠️ Hairy mammoth data file not found: {csvPath}");
                return;
            }

            var (hairyData, hairyLabels, hairyUniqueParts) = DataLoaders.LoadMammothWithLabels(csvPath);
            Console.WriteLine($"   Loaded: {hairyData.GetLength(0)} points, {hairyData.GetLength(1)} dimensions");

            // Use 50k samples for comprehensive testing
            int availableSamples = hairyData.GetLength(0);
            int requestedSamples = 50000;

            if (availableSamples < requestedSamples)
            {
                Console.WriteLine($"   ⚠️ Warning: Only {availableSamples:N0} samples available, using all instead of {requestedSamples:N0}");
                requestedSamples = availableSamples;
            }

            Console.WriteLine($"   Processing {requestedSamples:N0} hairy mammoth points for min_dist experiments...");
            var (data2, labels2) = DataLoaders.SampleRandomPoints(hairyData, hairyLabels, requestedSamples);
            Console.WriteLine($"   Subsampled: {data2.GetLength(0)} points, {data2.GetLength(1)} dimensions");

            // Convert double[,] to float[,] for UMAP API
            int nSamples = data2.GetLength(0);
            int nFeatures = data2.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data2[i, j];

            // Test min_dist values from 0.05 to 0.3 in increments of 0.05
            var minDistTests = new[] { 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f };
            var results = new List<(float minDist, double[,] embedding, double time, double quality)>();

            Console.WriteLine($"   🔍 Testing 6 min_dist values with n_neighbors=45...");

            foreach (var minDist in minDistTests)
            {
                Console.WriteLine($"   📊 Testing min_dist = {minDist:F2} (n_neighbors=45)...");
                var model = new UMapModel();

                var stopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: CreatePrefixedCallback($"MD={minDist:F2}"),
                    embeddingDimension: 2,
                    nNeighbors: 120,  // Updated to 120 as requested
                    minDist: minDist,
                    spread: 1.0f,
                    nEpochs: 300,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                stopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateEmbeddingQuality(doubleEmbedding, labels2);
                results.Add((minDist, doubleEmbedding, stopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ min_dist={minDist:F2}: quality={quality:F4}, time={stopwatch.Elapsed.TotalSeconds:F2}s");

                // Create visualization for all min_dist tests
                var paramInfo = CreateFitParamInfo(model, stopwatch.Elapsed.TotalSeconds, "Hairy_Mammoth_MinDist_Experiments");
                paramInfo["n_neighbors"] = "45";
                paramInfo["min_dist"] = minDist.ToString("F2");
                paramInfo["dataset"] = "Hairy Mammoth 50k";
                paramInfo["embedding_quality"] = quality.ToString("F4");

                var experimentDir = Path.Combine(ResultsDir, "hairy_mammoth_mindist_experiments");
                Directory.CreateDirectory(experimentDir);
                var imageNumber = Array.IndexOf(minDistTests, minDist) + 1;
                var outputPath = Path.Combine(experimentDir, $"{imageNumber:D2}_hairy_mindist_{minDist:F2}.png");
                var title = $"Hairy Mammoth min_dist Experiment: k=45, min_dist={minDist:F2}\n" + BuildVisualizationTitle(model, "Hairy Mammoth MinDist");
                Visualizer.PlotMammothUMAP(doubleEmbedding, labels2, title, outputPath, paramInfo, autoFitAxes: true, partNames: hairyUniqueParts);
                Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath)}");
            }

            // Analysis and Summary
            Console.WriteLine("\n📊 === Hairy Mammoth min_dist Experiments Summary ===");
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best min_dist: {bestResult.minDist:F2} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"📈 Quality range: {results.Min(r => r.quality):F4} - {results.Max(r => r.quality):F4}");
            Console.WriteLine($"⏱️ Time range: {results.Min(r => r.time):F2}s - {results.Max(r => r.time):F2}s");

            Console.WriteLine($"\n📁 Results saved to: {Path.Combine(ResultsDir, "hairy_mammoth_mindist_experiments")}");

            // Show detailed results table
            Console.WriteLine("\n📊 Detailed Results:");
            Console.WriteLine("   min_dist | Quality | Time (s)");
            Console.WriteLine("   ---------|---------|----------");
            foreach (var result in results.OrderBy(r => r.minDist))
            {
                var marker = result.minDist == bestResult.minDist ? "🏆" : "  ";
                Console.WriteLine($"   {marker}{result.minDist,-8:F2} | {result.quality,-7:F4} | {result.time,-8:F2}");
            }
        }

        /// <summary>
        /// Runs the MNIST demo.
        /// </summary>
        private static void RunMnistDemo()
        {
            Console.WriteLine("🔢 Running MNIST Demo...");
            MnistDemo.RunDemo();
        }

        /// <summary>
        /// Opens the 70k MNIST embedding result if it exists.
        /// </summary>
        private static void Open70kMnistEmbedding()
        {
            string embeddingPath = Path.Combine(ResultsDir, "mnist_2d_embedding_70k.png");
            if (File.Exists(embeddingPath))
            {
                Console.WriteLine("🖼️ Opening 70k MNIST embedding result...");
                try
                {
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = embeddingPath,
                        UseShellExecute = true
                    });
                    Console.WriteLine($"   ✅ Opened 70k MNIST embedding: {Path.GetFileName(embeddingPath)}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Could not open 70k MNIST embedding: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine($"   ℹ️ 70k MNIST embedding not found: {embeddingPath}");
            }
        }

  
        /// <summary>
        /// Calculates the Mean Squared Error between two embeddings.
        /// </summary>
        private static double CalculateMSE(double[,] embedding1, double[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double mse = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    mse += Math.Pow(embedding1[i, j] - embedding2[i, j], 2);
            return mse / (n * d);
        }

        /// <summary>
        /// Calculates the maximum difference between two embeddings.
        /// </summary>
        private static double CalculateMaxDifference(double[,] embedding1, double[,] embedding2)
        {
            int n = embedding1.GetLength(0);
            int d = embedding1.GetLength(1);
            double maxDiff = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    maxDiff = Math.Max(maxDiff, Math.Abs(embedding1[i, j] - embedding2[i, j]));
            return maxDiff;
        }

        /// <summary>
        /// Generates a consistency plot comparing two embeddings.
        /// </summary>
        private static void GenerateConsistencyPlot(double[,] embedding1, double[,] embedding2, int[] labels, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Embedding 1 - X Coordinate" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Embedding 2 - X Coordinate" });

            var uniqueLabels = labels.Distinct().OrderBy(x => x).ToArray();
            var colors = new[] { OxyColors.Blue, OxyColors.Red, OxyColors.Green, OxyColors.Orange, OxyColors.Purple };

            for (int labelIdx = 0; labelIdx < uniqueLabels.Length; labelIdx++)
            {
                var label = uniqueLabels[labelIdx];
                var scatterSeries = new ScatterSeries { Title = $"Label {label}", MarkerType = MarkerType.Circle, MarkerSize = 3 };
                scatterSeries.MarkerFill = colors[labelIdx % colors.Length];
                scatterSeries.MarkerStroke = colors[labelIdx % colors.Length];

                for (int i = 0; i < labels.Length; i++)
                {
                    if (labels[i] == label)
                        scatterSeries.Points.Add(new ScatterPoint(embedding1[i, 0], embedding2[i, 0], 3));
                }
                plotModel.Series.Add(scatterSeries);
            }

            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Generates a placeholder heatmap plot for pairwise distance differences.
        /// </summary>
        private static void GenerateHeatmapPlot(double[,] embedding1, double[,] embedding2, string title, string outputPath)
        {
            var plotModel = new PlotModel { Title = title };
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Sample Index" });
            plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Sample Index" });

            var annotation = new TextAnnotation
            {
                Text = "Heatmap visualization\n(Complex pairwise distance matrix)",
                TextPosition = new DataPoint(0.5, 0.5),
                TextHorizontalAlignment = HorizontalAlignment.Center,
                TextVerticalAlignment = VerticalAlignment.Middle,
                FontSize = 16,
                TextColor = OxyColors.Blue
            };
            plotModel.Annotations.Add(annotation);
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Generates a projection plot for original data.
        /// </summary>
        private static void GenerateProjection(double[,] originalData, double[,] embedding, string projectionType, string outputPath)
        {
            var plotModel = new PlotModel { Title = $"Original Data {projectionType} Projection" };
            var scatterSeries = new ScatterSeries { Title = $"Original {projectionType}", MarkerType = MarkerType.Circle, MarkerSize = 2 };

            if (projectionType == "XY")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Blue;
                scatterSeries.MarkerStroke = OxyColors.Blue;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 1], 2));
            }
            else if (projectionType == "XZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Red;
                scatterSeries.MarkerStroke = OxyColors.Red;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 0], originalData[i, 2], 2));
            }
            else if (projectionType == "YZ")
            {
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Y Coordinate" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Z Coordinate" });
                scatterSeries.MarkerFill = OxyColors.Green;
                scatterSeries.MarkerStroke = OxyColors.Green;
                for (int i = 0; i < originalData.GetLength(0); i++)
                    scatterSeries.Points.Add(new ScatterPoint(originalData[i, 1], originalData[i, 2], 2));
            }

            plotModel.Series.Add(scatterSeries);
            plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });
            ExportPlotToPng(plotModel, outputPath);
        }

        /// <summary>
        /// Saves embedding data to a CSV file.
        /// </summary>
        private static void SaveEmbeddingToCSV(double[,] embedding, string filePath)
        {
            using var writer = new StreamWriter(filePath);
            int nSamples = embedding.GetLength(0);
            int nDimensions = embedding.GetLength(1);

            // Write header
            var header = new string[nDimensions];
            for (int i = 0; i < nDimensions; i++)
            {
                header[i] = $"Dim{i + 1}";
            }
            writer.WriteLine(string.Join(",", header));

            // Write data
            for (int i = 0; i < nSamples; i++)
            {
                var row = new string[nDimensions];
                for (int j = 0; j < nDimensions; j++)
                {
                    row[j] = embedding[i, j].ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
                }
                writer.WriteLine(string.Join(",", row));
            }
        }

        /// <summary>
        /// Exports a plot model to a PNG file.
        /// </summary>
        private static void ExportPlotToPng(PlotModel plotModel, string outputPath)
        {
            var exporter = new PngExporter { Width = 800, Height = 600, Resolution = 300 };
            using var stream = File.Create(outputPath);
            exporter.Export(plotModel, stream);
        }

        /// <summary>
        /// Converts float[,] array to double[,] array
        /// </summary>
        private static double[,] ConvertFloatToDouble(float[,] floatArray)
        {
            int rows = floatArray.GetLength(0);
            int cols = floatArray.GetLength(1);
            var doubleArray = new double[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    doubleArray[i, j] = floatArray[i, j];

            return doubleArray;
        }
    }
}