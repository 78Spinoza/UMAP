using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
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
    /// MNIST Demo Program
    /// Demonstrates loading and using MNIST data with UMAP
    /// </summary>
    public class MnistDemo
    {
        /// <summary>
        /// Run MNIST demonstration
        /// </summary>
        public static void RunDemo()
        {
            Console.WriteLine("🔢 MNIST Binary Reader Demo");
            Console.WriteLine("=========================");

            // Create and open Results folder before starting
            var resultsDir = "Results";
            Directory.CreateDirectory(resultsDir);

            try
            {
                // Open Results folder in Windows Explorer
                Process.Start(new ProcessStartInfo
                {
                    FileName = Path.GetFullPath(resultsDir),
                    UseShellExecute = true
                });
                Console.WriteLine($"📂 Opened Results folder: {Path.GetFullPath(resultsDir)}");
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Could not open Results folder: {ex.Message}");
                Console.WriteLine();
            }

            try
            {
                // Path to the binary MNIST file
                string dataPath = Path.Combine("Data", "mnist_binary.dat.zip");

                if (!File.Exists(dataPath))
                {
                    Console.WriteLine($"❌ MNIST binary file not found: {dataPath}");
                    Console.WriteLine("   Please run the Python converter first:");
                    Console.WriteLine("   cd Data && python mnist_converter.py");
                    return;
                }

                Console.WriteLine($"📁 Loading MNIST data from: {dataPath}");

                // Load MNIST data
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var mnistData = MnistReader.Read(dataPath);
                stopwatch.Stop();

                Console.WriteLine($"✅ Loaded in {stopwatch.Elapsed.TotalMilliseconds:F1} ms");
                Console.WriteLine();

                // Print dataset information
                MnistReader.PrintInfo(mnistData);
                Console.WriteLine();

                // Demonstrate data access
                Console.WriteLine("🔍 Data Access Examples:");
                Console.WriteLine("=====================");

                // Show some sample images and labels
                var samples = MnistReader.GetRandomSamples(mnistData, samplesPerDigit: 3, seed: 42);
                Console.WriteLine($"Random samples (showing first {Math.Min(10, samples.Length)}):");

                for (int i = 0; i < Math.Min(10, samples.Length); i++)
                {
                    var index = samples[i];
                    var label = mnistData.Labels?[index] ?? 0;
                    Console.WriteLine($"   Sample {i + 1}: Index {index:D5}, Label: {label}");
                }

                Console.WriteLine();

                // Demonstrate conversion to float array for UMAP
                Console.WriteLine("🔄 Data Conversion for UMAP:");
                Console.WriteLine("===============================");

                // Use full dataset for demo (70,000 samples)
                var subsetSize = Math.Min(70000, mnistData.NumImages);
                var doubleData = mnistData.GetDoubleArray(0, subsetSize);
                var labels = mnistData.Labels?.Take(subsetSize).ToArray() ?? Array.Empty<byte>();

                Console.WriteLine($"   Using MNIST subset: {doubleData.GetLength(0):N0} images of {mnistData.NumImages:N0} total");
                Console.WriteLine($"   Shape: [{doubleData.GetLength(0):N0}, {doubleData.GetLength(1)}]");
                Console.WriteLine($"   Memory: {doubleData.Length * 8.0 / 1024 / 1024:F1} MB");
                Console.WriteLine($"   ✅ Data loaded as double pixels [0-255] - no conversion needed!");
                Console.WriteLine();

                // Show actual label distribution from subset
                Console.WriteLine("📊 Subset Label Distribution:");
                Console.WriteLine("============================");
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }
                for (int digit = 0; digit < 10; digit++)
                {
                    var percentage = (labelCounts[digit] * 100.0) / labels.Length;
                    Console.WriteLine($"   Digit {digit}: {labelCounts[digit]:D4} samples ({percentage:F1}%)");
                }
                Console.WriteLine();

                // Show some statistics about the double data
                Console.WriteLine("📊 Double Data Statistics:");
                Console.WriteLine("========================");

                double minVal = doubleData.Cast<double>().Min();
                double maxVal = doubleData.Cast<double>().Max();
                double meanVal = doubleData.Cast<double>().Average();

                Console.WriteLine($"   Value range: [{minVal:F3}, {maxVal:F3}]");
                Console.WriteLine($"   Mean value: {meanVal:F3}");
                Console.WriteLine($"   Expected range: [0, 255] (raw pixel values)");
                Console.WriteLine();

                // Create MNIST sample visualization first
                CreateMnistSampleVisualization(mnistData);

                // Print file paths that will be created
                Console.WriteLine();
                Console.WriteLine("📁 Files that will be created:");
                var samplePath = Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_samples_visualization.png");
                Console.WriteLine($"   🎨 MNIST sample visualization: {samplePath}");
                Console.WriteLine($"   📊 2D UMAP embedding: {Path.Combine(Directory.GetCurrentDirectory(), "Results", "mnist_2d_embedding.png")}");
                Console.WriteLine();

                // Create 2D embedding using helper function and save model + timing
                var (umap, fitTime) = CreateMnistEmbeddingWithModel(doubleData, labels, nNeighbors: 40, minDist: 0.05f, spread: 1.0f,
                    name: "mnist_2d_embedding", folderName: "", directKNN: false);

                Console.WriteLine();
                Console.WriteLine("🔄 UMAP embedding completed successfully!");
                Console.WriteLine("===============================================");
                Console.WriteLine("   Note: This is a basic UMAP demonstration with essential features only");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// Creates a visualization of actual MNIST digit images (0-9) using OxyPlot
        /// </summary>
        private static void CreateMnistSampleVisualization(MnistReader.MnistData mnistData)
        {
            Console.WriteLine("🎨 Creating MNIST Sample Visualization...");

            try
            {
                // Get random samples - one for each digit 0-9
                var random = new Random(42);
                var selectedSamples = new Dictionary<int, int>();

                for (int digit = 0; digit < 10; digit++)
                {
                    var digitIndices = new List<int>();
                    for (int i = 0; i < (mnistData.Labels?.Length ?? 0); i++)
                    {
                        if (mnistData.Labels != null && mnistData.Labels[i] == digit)
                            digitIndices.Add(i);
                    }

                    if (digitIndices.Count > 0)
                    {
                        var randomIndex = digitIndices[random.Next(digitIndices.Count)];
                        selectedSamples[digit] = randomIndex;
                        Console.WriteLine($"   Digit {digit}: Sample index {randomIndex:D5}");
                    }
                }

                // Create plot model for displaying actual digit images
                var plotModel = new PlotModel
                {
                    Title = "MNIST Digit Samples (0-9) - Actual Images",
                    Background = OxyColors.White
                };

                // Create image annotations for each digit
                for (int row = 0; row < 2; row++)
                {
                    for (int col = 0; col < 5; col++)
                    {
                        int digit = row * 5 + col;
                        if (digit >= 10) break;

                        if (selectedSamples.ContainsKey(digit))
                        {
                            var sampleIndex = selectedSamples[digit];
                            var image = mnistData.GetImageAsByteArray(sampleIndex);

                            // Create simple scatter plot representation of the digit pixels
                            CreateDigitPixelScatter(plotModel, image, digit, col, 1 - row);

                            // Add label below the image
                            var labelAnnotation = new TextAnnotation
                            {
                                Text = $"Label: {digit}",
                                TextPosition = new DataPoint(col, 0.7 - row),
                                TextHorizontalAlignment = HorizontalAlignment.Center,
                                TextVerticalAlignment = VerticalAlignment.Middle,
                                FontSize = 12,
                                FontWeight = FontWeights.Bold,
                                TextColor = OxyColors.Black
                            };
                            plotModel.Annotations.Add(labelAnnotation);
                        }
                    }
                }

                // Configure axes
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = -0.5, Maximum = 4.5, Title = "" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -0.5, Maximum = 1.5, Title = "" });

                // Save to file
                var outputPath = Path.Combine("Results", "mnist_samples_visualization.png");
                Directory.CreateDirectory("Results");

                var exporter = new PngExporter { Width = 1200, Height = 500, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ MNIST sample visualization saved: {outputPath}");
                Console.WriteLine($"   📊 Sample file: {Path.Combine(Directory.GetCurrentDirectory(), outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating sample visualization: {ex.Message}");
            }
        }

        /// <summary>
        /// Creates a scatter plot representation of MNIST digit pixels
        /// </summary>
        private static void CreateDigitPixelScatter(PlotModel plotModel, byte[,] image, int digit, double centerX, double centerY)
        {
            var scatterSeries = new ScatterSeries
            {
                Title = $"Digit {digit}",
                MarkerType = MarkerType.Square,
                MarkerSize = 1.0,  // Large for clear digit visualization
                MarkerFill = OxyColors.Black,
                MarkerStroke = OxyColors.Black
            };

            int height = image.GetLength(0);
            int width = image.GetLength(1);

            // Scale and center the digit within the plot area
            double scale = 0.4; // Scale to fit within the 0.8x0.8 area
            double offsetX = centerX - scale / 2;
            double offsetY = centerY - scale / 2;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Use original pixel value (0 = white background, 255 = black ink)
                    var pixelValue = image[h, w];

                    // Only plot dark pixels (ink) - MNIST has black ink on white background
                    if (pixelValue > 128) // High values = black ink
                    {
                        double x = offsetX + (w / (double)width) * scale;
                        // Flip Y-axis to display digits upright
                        double y = offsetY + ((height - h) / (double)height) * scale;

                        scatterSeries.Points.Add(new ScatterPoint(x, y, 1.0));
                    }
                }
            }

            if (scatterSeries.Points.Count > 0)
                plotModel.Series.Add(scatterSeries);
        }

        /// <summary>
        /// Helper function to create MNIST embedding with specified parameters and return the fitted model + fit time
        /// </summary>
        private static (UMapModel model, double fitTime) CreateMnistEmbeddingWithModel(double[,] data, byte[] labels, int nNeighbors, float minDist, float spread, string name, string folderName = "", bool directKNN = false)
        {
            string knnType = directKNN ? "Direct KNN" : "HNSW";
            Console.WriteLine($"🚀 Creating {name} embedding (k={nNeighbors}, min_dist={minDist:F2}, spread={spread:F2}, KNN={knnType})...");

            // Convert double[,] to float[,] for UMAP API
            int nSamples = data.GetLength(0);
            int nFeatures = data.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            for (int i = 0; i < nSamples; i++)
                for (int j = 0; j < nFeatures; j++)
                    floatData[i, j] = (float)data[i, j];

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            var umap = new UMapModel();
            var embedding = umap.FitWithProgress(
                data: floatData,
                progressCallback: (phase, current, total, percent, message) =>
                {
                    Console.Write($"\r[{phase}] {current}/{total} ({percent:F1}%) {message}".PadRight(180));
                },
                embeddingDimension: 2,
                nNeighbors: nNeighbors,
                minDist: minDist,
                spread: spread,
                nEpochs: 600,
                metric: DistanceMetric.Euclidean,
                forceExactKnn: directKNN,
                autoHNSWParam: false,  // Use default HNSW values
                randomSeed: 42
            );

            stopwatch.Stop();
            Console.WriteLine($"✅ {name} embedding completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine($"   Shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");

            // Convert float[,] embedding back to double[,] for visualization
            int embedSamples = embedding.GetLength(0);
            int embedDims = embedding.GetLength(1);
            var doubleEmbedding = new double[embedSamples, embedDims];
            for (int i = 0; i < embedSamples; i++)
                for (int j = 0; j < embedDims; j++)
                    doubleEmbedding[i, j] = embedding[i, j];

            // Create 2D visualization with colored labels and model info
            Create2DScatterPlot(doubleEmbedding, labels, umap, stopwatch.Elapsed.TotalSeconds, name, folderName);

            return (umap, stopwatch.Elapsed.TotalSeconds); // Return fitted model + fit time for reuse in Transform
        }

        /// <summary>
        /// Creates 2D scatter plot with colored labels, counts, and hyperparameters
        /// </summary>
        private static void Create2DScatterPlot(double[,] embedding, byte[] labels, UMapModel umap, double executionTime, string name = "mnist_2d_embedding", string folderName = "Results")
        {
            Console.WriteLine("🎨 Creating 2D Embedding Visualization...");

            try
            {
                // Count labels for title
                var labelCounts = new int[10];
                for (int i = 0; i < labels.Length; i++)
                {
                    labelCounts[labels[i]]++;
                }

                // Build title with hyperparameters like mammoth visualizations
                var modelInfo = umap.ModelInfo;
                var version = UMapModel.GetVersion(); // Get version dynamically from wrapper
                var knnMode = "HNSW"; // UMAP always uses HNSW optimization

                var timeUnit = executionTime >= 60 ? $"{executionTime / 60.0:F1}m" : $"{executionTime:F1}s";
                var title = $@"MNIST 2D Embedding (UMAP)
UMAP v{version} | Sample: {embedding.GetLength(0):N0} | {knnMode} | Time: {timeUnit}
k={modelInfo.Neighbors} | {modelInfo.Metric} | dims={modelInfo.OutputDimension} | seed=42
min_dist={modelInfo.MinimumDistance:F2} | spread={modelInfo.Spread:F2} | epochs=300 | HNSW: M={modelInfo.HnswM}, ef_c={modelInfo.HnswEfConstruction}, ef_s={modelInfo.HnswEfSearch}";

                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White
                };

                // Define digit groups with different markers and better colors
                var digitConfigs = new[]
                {
                    new { Digit = 0, Color = OxyColors.Red, Marker = MarkerType.Triangle, Name = "0-Triangle" },
                    new { Digit = 1, Color = OxyColors.Blue, Marker = MarkerType.Diamond, Name = "1-Diamond" },
                    new { Digit = 2, Color = OxyColors.Green, Marker = MarkerType.Circle, Name = "2-Circle" },
                    new { Digit = 3, Color = OxyColors.Orange, Marker = MarkerType.Square, Name = "3-Square" },
                    new { Digit = 4, Color = OxyColors.Purple, Marker = MarkerType.Plus, Name = "4-Plus" },
                    new { Digit = 5, Color = OxyColors.Cyan, Marker = MarkerType.Star, Name = "5-Star" },
                    new { Digit = 6, Color = OxyColors.Magenta, Marker = MarkerType.Cross, Name = "6-Cross" },
                    new { Digit = 7, Color = OxyColors.Brown, Marker = MarkerType.Diamond, Name = "7-Diamond" },
                    new { Digit = 8, Color = OxyColors.Pink, Marker = MarkerType.Square, Name = "8-Square" },
                    new { Digit = 9, Color = OxyColors.Gray, Marker = MarkerType.Square, Name = "9-Square" }
                };

                // Create scatter series for each digit
                foreach (var config in digitConfigs)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {config.Digit} ({labelCounts[config.Digit]:D4}) - {config.Name}",
                        MarkerType = config.Marker,
                        MarkerSize = 4,
                        MarkerFill = config.Color,
                        MarkerStroke = config.Color,
                        MarkerStrokeThickness = 0.5
                    };

                    // Add points for this digit
                    for (int i = 0; i < embedding.GetLength(0); i++)
                    {
                        if (labels[i] == config.Digit)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1], 4));
                        }
                    }

                    if (scatterSeries.Points.Count > 0)
                        plotModel.Series.Add(scatterSeries);
                }

                // Calculate min/max for proper axis scaling
                double minX = embedding[0, 0], maxX = embedding[0, 0];
                double minY = embedding[0, 1], maxY = embedding[0, 1];

                for (int i = 1; i < embedding.GetLength(0); i++)
                {
                    if (embedding[i, 0] < minX) minX = embedding[i, 0];
                    if (embedding[i, 0] > maxX) maxX = embedding[i, 0];
                    if (embedding[i, 1] < minY) minY = embedding[i, 1];
                    if (embedding[i, 1] > maxY) maxY = embedding[i, 1];
                }

                // Add 20% padding to right side of X axis to fit labels
                double xPadding = (maxX - minX) * 0.2;

                // Configure axes with proper min/max - only add padding to right side
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Bottom,
                    Title = "X Coordinate",
                    Minimum = minX,
                    Maximum = maxX + xPadding
                });
                plotModel.Axes.Add(new LinearAxis
                {
                    Position = AxisPosition.Left,
                    Title = "Y Coordinate",
                    Minimum = minY,
                    Maximum = maxY
                });

                // Add legend
                plotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight });

                // Determine output path based on folder parameter
                string outputDir = string.IsNullOrEmpty(folderName) ? "Results" : Path.Combine("Results", folderName);
                Directory.CreateDirectory(outputDir);
                var outputPath = Path.Combine(outputDir, $"{name}.png");
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 1200, Height = 900, Resolution = 300 };
                using var stream = File.Create(outputPath);
                exporter.Export(plotModel, stream);

                Console.WriteLine($"✅ 2D embedding visualization saved: {outputPath}");
                Console.WriteLine($"   Resolution: 1200x900, Points: {embedding.GetLength(0):N0}");
                Console.WriteLine($"   📊 Full path: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating 2D plot: {ex.Message}");
            }
        }

        /// <summary>
        /// Runs advanced UMAP parameter tuning experiments on MNIST data.
        /// Uses 20k MNIST samples for comprehensive testing.
        /// Test 1: n_neighbors from 5 to 40 (increments of 1) with min_dist=0.2
        /// Test 2: min_dist from 0.05 to 0.3 (increments of 0.05) with n_neighbors=40
        /// </summary>
        public static void RunMnistParameterTuning()
        {
            Console.WriteLine("🔬 Running MNIST Advanced Parameter Tuning...");
            Console.WriteLine("   Using 20,000 MNIST samples for comprehensive testing");
            Console.WriteLine("   Test 1: n_neighbors from 5 to 40 (increments of 1) with min_dist=0.2");
            Console.WriteLine("   Test 2: min_dist from 0.05 to 0.3 (increments of 0.05) with n_neighbors=40");

            // Path to the binary MNIST file
            string dataPath = Path.Combine("Data", "mnist_binary.dat.zip");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"❌ MNIST binary file not found: {dataPath}");
                Console.WriteLine("   Please run the Python converter first:");
                Console.WriteLine("   cd Data && python mnist_converter.py");
                return;
            }

            Console.WriteLine($"📁 Loading MNIST data from: {dataPath}");

            // Load MNIST data
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var mnistData = MnistReader.Read(dataPath);
            stopwatch.Stop();

            Console.WriteLine($"✅ Loaded in {stopwatch.Elapsed.TotalMilliseconds:F1} ms");

            // Use 20,000 samples for parameter tuning (subsetSize)
            var subsetSize = Math.Min(20000, mnistData.NumImages);
            var doubleData = mnistData.GetDoubleArray(0, subsetSize);
            var labels = mnistData.Labels?.Take(subsetSize).ToArray() ?? Array.Empty<byte>();

            Console.WriteLine($"   Using MNIST subset: {doubleData.GetLength(0):N0} images of {mnistData.NumImages:N0} total");
            Console.WriteLine($"   Shape: [{doubleData.GetLength(0):N0}, {doubleData.GetLength(1)}]");
            Console.WriteLine($"   Memory: {doubleData.Length * 8.0 / 1024 / 1024:F1} MB");
            Console.WriteLine();

            // Show label distribution for the 20k subset
            Console.WriteLine("📊 20k Subset Label Distribution:");
            Console.WriteLine("===============================");
            var labelCounts = new int[10];
            for (int i = 0; i < labels.Length; i++)
            {
                labelCounts[labels[i]]++;
            }
            for (int digit = 0; digit < 10; digit++)
            {
                var percentage = (labelCounts[digit] * 100.0) / labels.Length;
                Console.WriteLine($"   Digit {digit}: {labelCounts[digit]:D4} samples ({percentage:F1}%)");
            }
            Console.WriteLine();

            // Convert double[,] to float[,] for UMAP API and byte[] to int[]
            int nSamples = doubleData.GetLength(0);
            int nFeatures = doubleData.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            var intLabels = new int[labels.Length];

            for (int i = 0; i < nSamples; i++)
            {
                intLabels[i] = labels[i];
                for (int j = 0; j < nFeatures; j++)
                {
                    floatData[i, j] = (float)doubleData[i, j];
                }
            }

            var allResults = new List<(string testType, int nNeighbors, float minDist, double[,] embedding, double time, double quality)>();

            // === TEST 1: n_neighbors from 5 to 40 with min_dist=0.2 ===
            Console.WriteLine("\n📊 === TEST 1: n_neighbors Optimization (min_dist=0.2) ===");
            var nNeighborsTests = Enumerable.Range(5, 36).ToArray(); // 5 to 40 inclusive

            foreach (var nNeighbors in nNeighborsTests)
            {
                Console.WriteLine($"   🔍 Testing n_neighbors = {nNeighbors}, min_dist = 0.2...");
                var model = new UMapModel();

                var testStopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: (phase, current, total, percent, message) => {
                        if (current % 50 == 0 || current == total)
                            Console.WriteLine($"      N={nNeighbors} [{phase}] {current}/{total} ({percent:F0}%) {message}");
                    },
                    embeddingDimension: 2,
                    nNeighbors: nNeighbors,
                    minDist: 0.001f,
                    spread: 1.0f,
                    nEpochs: 600,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                testStopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateMnistEmbeddingQuality(doubleEmbedding, intLabels);
                allResults.Add(("n_neighbors_test", nNeighbors, 0.2f, doubleEmbedding, testStopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ n_neighbors={nNeighbors}: quality={quality:F4}, time={testStopwatch.Elapsed.TotalSeconds:F2}s");

                // Save visualization for key results only (every 5th neighbor to avoid too many files)
                if (nNeighbors % 5 == 0 || nNeighbors <= 10)
                {
                    var experimentDir = Path.Combine("Results", "mnist_n_neighbors_experiments");
                    Directory.CreateDirectory(experimentDir);
                    var outputPath = Path.Combine(experimentDir, $"mnist_neighbors_{nNeighbors:D2}.png");
                    var title = $"MNIST n_neighbors Experiment: k={nNeighbors}, min_dist=0.2\nUMAP MNIST {doubleData.GetLength(0):N0} samples | k={nNeighbors} | min_dist=0.2";
                    CreateMnistParameterPlot(doubleEmbedding, intLabels, title, outputPath, nNeighbors, 0.2f, quality, testStopwatch.Elapsed.TotalSeconds);
                    Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath)}");
                }
            }

            // === TEST 2: min_dist from 0.05 to 0.3 with n_neighbors=40 ===
            Console.WriteLine("\n📊 === TEST 2: min_dist Optimization (n_neighbors=40) ===");
            var minDistTests = new[] { 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f };

            foreach (var minDist in minDistTests)
            {
                Console.WriteLine($"   🔍 Testing min_dist = {minDist:F2}, n_neighbors = 40...");
                var model = new UMapModel();

                var testStopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: (phase, current, total, percent, message) => {
                        if (current % 50 == 0 || current == total)
                            Console.WriteLine($"      MD={minDist:F2} [{phase}] {current}/{total} ({percent:F0}%) {message}");
                    },
                    embeddingDimension: 2,
                    nNeighbors: 40,
                    minDist: minDist,
                    spread: 1.0f,
                    nEpochs: 600,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                testStopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateMnistEmbeddingQuality(doubleEmbedding, intLabels);
                allResults.Add(("min_dist_test", 40, minDist, doubleEmbedding, testStopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ min_dist={minDist:F2}: quality={quality:F4}, time={testStopwatch.Elapsed.TotalSeconds:F2}s");

                // Save visualization for all min_dist tests (only 6 files)
                var experimentDir2 = Path.Combine("Results", "mnist_min_dist_experiments");
                Directory.CreateDirectory(experimentDir2);
                var imageNumber = Array.IndexOf(minDistTests, minDist) + 1;
                var outputPath2 = Path.Combine(experimentDir2, $"{imageNumber:D2}_mnist_mindist_{minDist:F2}.png");
                var title2 = $"MNIST min_dist Experiment: k=40, min_dist={minDist:F2}\nUMAP MNIST {doubleData.GetLength(0):N0} samples | k=40 | min_dist={minDist:F2}";
                CreateMnistParameterPlot(doubleEmbedding, intLabels, title2, outputPath2, 40, minDist, quality, testStopwatch.Elapsed.TotalSeconds);
                Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath2)}");
            }

            // === ANALYSIS AND SUMMARY ===
            Console.WriteLine("\n📊 === MNIST PARAMETER TUNING ANALYSIS ===");

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
            Console.WriteLine("\n🥇 Overall Best MNIST Parameters:");
            Console.WriteLine($"   n_neighbors: {overallBest.nNeighbors}, min_dist: {overallBest.minDist:F2}");
            Console.WriteLine($"   Best quality: {overallBest.quality:F4}, Time: {overallBest.time:F2}s");
            Console.WriteLine($"   Test type: {overallBest.testType}");

            Console.WriteLine($"\n📁 MNIST Results saved to:");
            Console.WriteLine($"   - Results/mnist_n_neighbors_experiments/ (key results)");
            Console.WriteLine($"   - Results/mnist_min_dist_experiments/ (all results)");
        }

        /// <summary>
        /// Calculates embedding quality for MNIST data based on label separation.
        /// Lower quality score indicates better separation (like the mammoth version).
        /// </summary>
        private static double CalculateMnistEmbeddingQuality(double[,] embedding, int[] labels)
        {
            if (embedding.GetLength(0) != labels.Length)
                throw new ArgumentException("Embedding and labels must have same number of samples");

            int nSamples = embedding.GetLength(0);
            double totalIntraClusterDistance = 0;
            int intraClusterPairs = 0;
            double totalInterClusterDistance = 0;
            int interClusterPairs = 0;

            // Sample subset for efficiency (max 1000 points)
            var sampleSize = Math.Min(1000, nSamples);
            var random = new Random(42);
            var sampleIndices = Enumerable.Range(0, nSamples).OrderBy(x => random.Next()).Take(sampleSize).ToArray();

            for (int i = 0; i < sampleIndices.Length; i++)
            {
                for (int j = i + 1; j < sampleIndices.Length; j++)
                {
                    int idx1 = sampleIndices[i];
                    int idx2 = sampleIndices[j];

                    double distance = Math.Sqrt(
                        Math.Pow(embedding[idx1, 0] - embedding[idx2, 0], 2) +
                        Math.Pow(embedding[idx1, 1] - embedding[idx2, 1], 2)
                    );

                    if (labels[idx1] == labels[idx2])
                    {
                        // Same cluster - want small distances
                        totalIntraClusterDistance += distance;
                        intraClusterPairs++;
                    }
                    else
                    {
                        // Different clusters - want large distances
                        totalInterClusterDistance += distance;
                        interClusterPairs++;
                    }
                }
            }

            double avgIntra = intraClusterPairs > 0 ? totalIntraClusterDistance / intraClusterPairs : 0;
            double avgInter = interClusterPairs > 0 ? totalInterClusterDistance / interClusterPairs : 0;

            // Quality score: lower is better (small intra-cluster, large inter-cluster distances)
            // Normalize by dividing by inter-cluster distance to make it scale-invariant
            return avgInter > 0 ? avgIntra / avgInter : double.MaxValue;
        }

        /// <summary>
        /// Creates visualization for MNIST parameter tuning results
        /// </summary>
        private static void CreateMnistParameterPlot(double[,] embedding, int[] labels, string title, string outputPath, int nNeighbors, float minDist, double quality, double executionTime)
        {
            try
            {
                var model = new PlotModel { Title = title, Background = OxyColors.White };

                // Add axes
                model.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "UMAP 1" });
                model.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "UMAP 2" });

                // Create series for each digit (0-9)
                var digitColors = new[]
                {
                    OxyColors.Red, OxyColors.Blue, OxyColors.Green, OxyColors.Orange, OxyColors.Purple,
                    OxyColors.Cyan, OxyColors.Magenta, OxyColors.Brown, OxyColors.Pink, OxyColors.Lime
                };

                for (int digit = 0; digit < 10; digit++)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"Digit {digit}",
                        MarkerType = MarkerType.Circle,
                        MarkerSize = 2,
                        MarkerFill = digitColors[digit]
                    };

                    // Add points for this digit
                    int pointCount = 0;
                    for (int i = 0; i < labels.Length; i++)
                    {
                        if (labels[i] == digit)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1]));
                            pointCount++;
                        }
                    }

                    if (pointCount > 0)
                    {
                        scatterSeries.Title = $"Digit {digit} ({pointCount})";
                        model.Series.Add(scatterSeries);
                    }
                }

                // Add parameter info as annotation
                var annotation = new TextAnnotation
                {
                    Text = $"k={nNeighbors}, min_dist={minDist:F2}\nQuality: {quality:F4}\nTime: {executionTime:F2}s",
                    TextPosition = new DataPoint(0.02, 0.98),
                    TextHorizontalAlignment = HorizontalAlignment.Left,
                    TextVerticalAlignment = VerticalAlignment.Top,
                    FontSize = 10,
                    Background = OxyColor.FromArgb(200, 255, 255, 255),
                    TextColor = OxyColors.Black
                };
                model.Annotations.Add(annotation);

                // Add legend
                var legend = new Legend
                {
                    LegendPosition = LegendPosition.RightTop,
                    LegendBackground = OxyColor.FromArgb(200, 255, 255, 255),
                    LegendBorder = OxyColors.Black
                };
                model.Legends.Add(legend);

                // Save the plot
                var exporter = new PngExporter { Width = 1200, Height = 900 };
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(model, stream);
                }

                Console.WriteLine($"   📊 Resolution: 1200x900, Points: {embedding.GetLength(0):N0}");
                Console.WriteLine($"   📊 Full path: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating MNIST parameter plot: {ex.Message}");
            }
        }

        /// <summary>
        /// Runs MNIST min_dist experiments with fixed 40 neighbors.
        /// Tests min_dist from 0.05 to 2.0 in increments of 0.05 for the MNIST dataset.
        /// </summary>
        public static void RunMnistMinDistExperiments()
        {
            Console.WriteLine("🔢 Running MNIST min_dist Experiments (n_neighbors=40)...");
            Console.WriteLine("   Testing min_dist from 0.05 to 0.3 (increments of 0.2) with n_neighbors=40");

            // Path to the binary MNIST file
            string dataPath = Path.Combine("Data", "mnist_binary.dat.zip");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"❌ MNIST binary file not found: {dataPath}");
                Console.WriteLine("   Please run the Python converter first:");
                Console.WriteLine("   cd Data && python mnist_converter.py");
                return;
            }

            Console.WriteLine($"📁 Loading MNIST data from: {dataPath}");

            // Load MNIST data
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var mnistData = MnistReader.Read(dataPath);
            stopwatch.Stop();

            Console.WriteLine($"✅ Loaded in {stopwatch.Elapsed.TotalMilliseconds:F1} ms");

            // Use 20,000 samples for comprehensive testing (same as other MNIST experiments)
            var subsetSize = Math.Min(20000, mnistData.NumImages);
            var doubleData = mnistData.GetDoubleArray(0, subsetSize);
            var labels = mnistData.Labels?.Take(subsetSize).ToArray() ?? Array.Empty<byte>();

            Console.WriteLine($"   Using MNIST subset: {doubleData.GetLength(0):N0} images of {mnistData.NumImages:N0} total");
            Console.WriteLine($"   Shape: [{doubleData.GetLength(0):N0}, {doubleData.GetLength(1)}]");
            Console.WriteLine($"   Memory: {doubleData.Length * 8.0 / 1024 / 1024:F1} MB");
            Console.WriteLine();

            // Convert double[,] to float[,] for UMAP API and byte[] to int[]
            int nSamples = doubleData.GetLength(0);
            int nFeatures = doubleData.GetLength(1);
            var floatData = new float[nSamples, nFeatures];
            var intLabels = new int[labels.Length];

            for (int i = 0; i < nSamples; i++)
            {
                intLabels[i] = labels[i];
                for (int j = 0; j < nFeatures; j++)
                {
                    floatData[i, j] = (float)doubleData[i, j];
                }
            }

            // Test min_dist values from 0.05 to 2.0 in increments of 0.05 (40 tests total)
            var minDistTests = new List<float>();
            for (float dist = 0.05f; dist <= 0.3f; dist += 0.2f)
            {
                minDistTests.Add(dist);
            }

            var results = new List<(float minDist, double[,] embedding, double time, double quality)>();

            Console.WriteLine($"   🔍 Testing {minDistTests.Count} min_dist values with n_neighbors=40...");

            foreach (var minDist in minDistTests)
            {
                Console.WriteLine($"   📊 Testing min_dist = {minDist:F2} (n_neighbors=40)...");
                var model = new UMapModel();

                var testStopwatch = Stopwatch.StartNew();
                var embedding = model.FitWithProgress(
                    data: floatData,
                    progressCallback: (phase, current, total, percent, message) => {
                        if (current % 50 == 0 || current == total)
                            Console.WriteLine($"      MD={minDist:F2} [{phase}] {current}/{total} ({percent:F0}%) {message}");
                    },
                    embeddingDimension: 2,
                    nNeighbors: 40,  // Fixed to 40 as requested
                    minDist: minDist,
                    spread: 1.0f,
                    nEpochs: 600,
                    metric: DistanceMetric.Euclidean,
                    forceExactKnn: false,
                    autoHNSWParam: false,
                    randomSeed: 42
                );
                testStopwatch.Stop();

                // Convert float[,] embedding back to double[,] for compatibility
                int embedSamples = embedding.GetLength(0);
                int embedDims = embedding.GetLength(1);
                var doubleEmbedding = new double[embedSamples, embedDims];
                for (int i = 0; i < embedSamples; i++)
                    for (int j = 0; j < embedDims; j++)
                        doubleEmbedding[i, j] = embedding[i, j];

                double quality = CalculateMnistEmbeddingQuality(doubleEmbedding, intLabels);
                results.Add((minDist, doubleEmbedding, testStopwatch.Elapsed.TotalSeconds, quality));
                Console.WriteLine($"   ✅ min_dist={minDist:F2}: quality={quality:F4}, time={testStopwatch.Elapsed.TotalSeconds:F2}s");

                // Save visualization for key results only (every 10th value to avoid too many files)
                if (minDistTests.IndexOf(minDist) % 10 == 0)
                {
                    var experimentDir = Path.Combine("Results", "mnist_mindist_experiments");
                    Directory.CreateDirectory(experimentDir);
                    var imageNumber = (minDistTests.IndexOf(minDist) / 10) + 1;
                    var outputPath = Path.Combine(experimentDir, $"{imageNumber:D2}_mnist_mindist_{minDist:F2}.png");
                    var title = $"MNIST min_dist Experiment: k=40, min_dist={minDist:F2}\nUMAP MNIST {doubleData.GetLength(0):N0} samples | k=40 | min_dist={minDist:F2}";
                    CreateMnistParameterPlot(doubleEmbedding, intLabels, title, outputPath, 40, minDist, quality, testStopwatch.Elapsed.TotalSeconds);
                    Console.WriteLine($"      📈 Saved: {Path.GetFileName(outputPath)}");
                }
            }

            // Analysis and Summary
            Console.WriteLine("\n📊 === MNIST min_dist Experiments Summary ===");
            var bestResult = results.OrderBy(r => r.quality).First();
            Console.WriteLine($"🏆 Best min_dist: {bestResult.minDist:F2} (quality: {bestResult.quality:F4})");
            Console.WriteLine($"📈 Quality range: {results.Min(r => r.quality):F4} - {results.Max(r => r.quality):F4}");
            Console.WriteLine($"⏱️ Time range: {results.Min(r => r.time):F2}s - {results.Max(r => r.time):F2}s");

            Console.WriteLine($"\n📁 Results saved to: {Path.Combine("Results", "mnist_mindist_experiments")} (key results only)");

            // Show detailed results table for key results
            Console.WriteLine("\n📊 Key Results (every 10th value):");
            Console.WriteLine("   min_dist | Quality | Time (s)");
            Console.WriteLine("   ---------|---------|----------");
            for (int i = 0; i < results.Count; i += 10)
            {
                var result = results[i];
                var marker = result.minDist == bestResult.minDist ? "🏆" : "  ";
                Console.WriteLine($"   {marker}{result.minDist,-8:F2} | {result.quality,-7:F4} | {result.time,-8:F2}");
            }

            // Show best and worst extremes
            var worstResult = results.OrderBy(r => r.quality).Last();
            Console.WriteLine($"\n🔍 Extremes:");
            Console.WriteLine($"   Best:  min_dist={bestResult.minDist:F2}, quality={bestResult.quality:F4}");
            Console.WriteLine($"   Worst: min_dist={worstResult.minDist:F2}, quality={worstResult.quality:F4}");
        }
    }
}