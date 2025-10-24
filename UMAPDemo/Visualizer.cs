using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.WindowsForms;
using OxyPlot.Legends;
using OxyPlot.Annotations;

namespace UMAPDemo
{
    public static class Visualizer
    {
        /// <summary>
        /// Assign anatomical parts to mammoth points based on 3D coordinates
        /// </summary>
        public static string[] AssignMammothParts(double[,] originalData)
        {
            int numPoints = originalData.GetLength(0);
            var parts = new string[numPoints];

            // Compute coordinate ranges
            double minZ = double.MaxValue, maxZ = double.MinValue;
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minZ = Math.Min(minZ, z);
                maxZ = Math.Max(maxZ, z);
            }

            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double zRange = maxZ - minZ;

            // Initialize all points as "body" first
            for (int i = 0; i < numPoints; i++)
            {
                parts[i] = "body";
            }

            // 1. FEET: Bottom 8% (very bottom)
            double feetZThreshold = minZ + zRange * 0.08;
            for (int i = 0; i < numPoints; i++)
            {
                double z = originalData[i, 2];
                if (z < feetZThreshold)
                    parts[i] = "feet";
            }

            // 2. LEGS: Bottom 28% but excluding feet (8% to 28% Z range)
            double legZThreshold = minZ + zRange * 0.28;
            for (int i = 0; i < numPoints; i++)
            {
                double z = originalData[i, 2];
                if (z >= feetZThreshold && z < legZThreshold)
                    parts[i] = "legs";
            }

            // 3. HEAD: Middle 28% X + First 41.5% Y + Top 50% Z (optimized from testing)
            double firstXThreshold = minX + xRange * 0.36;  // First 36% of X
            double lastXThreshold = maxX - xRange * 0.36;   // Last 36% of X (middle 28%)
            double yThreshold = minY + yRange * 0.415; // First 41.5% of Y direction
            double zThreshold = minZ + zRange * 0.5; // Top 50% of Z (upper half)

            for (int i = 0; i < numPoints; i++)
            {
                double x = originalData[i, 0];
                double y = originalData[i, 1];
                double z = originalData[i, 2];

                if (x >= firstXThreshold && x <= lastXThreshold && y < yThreshold && z > zThreshold)
                    parts[i] = "head";
            }

            // 4. TRUNK: Inverse of X (First 36% + Last 36%) + First 41.5% Y (extremities)
            for (int i = 0; i < numPoints; i++)
            {
                if (parts[i] == "body") // Don't override feet, legs, or head
                {
                    double x = originalData[i, 0];
                    double y = originalData[i, 1];

                    if ((x < firstXThreshold || x > lastXThreshold) && y < yThreshold)
                        parts[i] = "trunk";
                }
            }

            // 5. TUSKS: Very forward + very high + very narrow (refine from trunk)
            double tuskXThreshold = minX + xRange * 0.9; // Ultra forward
            double tuskZThreshold = minZ + zRange * 0.85; // Very high
            double tuskYWidth = yRange * 0.1; // Ultra narrow
            double yCenter = (minY + maxY) / 2;

            for (int i = 0; i < numPoints; i++)
            {
                if (parts[i] == "trunk") // Can refine trunk to tusks
                {
                    double x = originalData[i, 0];
                    double y = originalData[i, 1];
                    double z = originalData[i, 2];
                    double yDist = Math.Abs(y - yCenter);

                    if (x > tuskXThreshold && z > tuskZThreshold && yDist < tuskYWidth)
                        parts[i] = "tusks";
                }
            }

            return parts;
        }

        /// <summary>
        /// Create real 3D visualization using OxyPlot with anatomical part coloring
        /// </summary>
        public static void PlotOriginalMammoth3DReal(double[,] originalData, int[] labels, string title, string outputPath)
        {
            try
            {
                Console.WriteLine($"Creating real 3D mammoth plot: {title}");

                int numPoints = originalData.GetLength(0);

                // Convert integer labels to part names using the SAME mapping as PlotMammothUMAP
                var partNames = new[] { "body", "head", "feet", "tusks", "legs" };
                var parts = new string[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    int labelIndex = labels[i];
                    if (labelIndex >= 0 && labelIndex < partNames.Length)
                    {
                        parts[i] = partNames[labelIndex];
                    }
                    else
                    {
                        parts[i] = "body"; // Fallback
                    }
                }

                var partColors = new Dictionary<string, OxyColor>
                {
                    { "body", OxyColors.ForestGreen },   // Green
                    { "head", OxyColors.Blue },          // Blue
                    { "legs", OxyColors.Brown },         // Brown
                    { "tusks", OxyColors.Gray },          // Gray (something)
                    { "trunk", OxyColors.Red },          // Red
                    { "feet", OxyColors.Gold }           // Yellow (if needed)
                };

                // Create a composite plot with 3 views (XY, XZ, YZ)
                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Configure for side-by-side layout - we'll create separate plots
                // This main plot will show XY view (top view)
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "X Coordinate (Left-Right)" });
                plotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "Y Coordinate (Front-Back)" });

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y, List<double> z)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(originalData[i, 0]);
                    partGroups[part].y.Add(originalData[i, 1]);
                    partGroups[part].z.Add(originalData[i, 2]);
                }

                // Calculate Z bounds for depth mapping
                double minZ = partGroups.Values.SelectMany(g => g.z).Min();
                double maxZ = partGroups.Values.SelectMany(g => g.z).Max();
                double zRange = maxZ - minZ;

                // Add scatter series for each part
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints, zPoints) = kvp.Value;

                    if (xPoints.Count > 0)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                            MarkerType = MarkerType.Circle,
                            MarkerFill = partColors[part],
                            MarkerStroke = partColors[part]
                        };

                        // Add XY view points (top view)
                        for (int i = 0; i < xPoints.Count; i++)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(xPoints[i], yPoints[i], 3));
                        }

                        plotModel.Series.Add(scatterSeries);
                    }
                }

                // Display statistics
                var partCounts = partGroups.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.x.Count);
                Console.WriteLine("Anatomical part distribution:");
                foreach (var part in partCounts.OrderByDescending(p => p.Value))
                {
                    double percentage = (part.Value * 100.0) / numPoints;
                    Console.WriteLine($"   {char.ToUpper(part.Key[0]) + part.Key.Substring(1)}: {part.Value} points ({percentage:F1}%)");
                }

                // Export XY view (top view)
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 800, Height = 600, Resolution = 300 };
                string xyPath = outputPath.Replace(".png", "_XY_TopView.png");
                using (var stream = File.Create(xyPath))
                {
                    exporter.Export(plotModel, stream);
                }

                // Create XZ view (side view)
                var xzPlotModel = CreateViewPlot(partGroups, partColors, "XZ View (Side)", "X Coordinate (Left-Right)", "Z Coordinate (Height)", "xz");
                string xzPath = outputPath.Replace(".png", "_XZ_SideView.png");
                using (var stream = File.Create(xzPath))
                {
                    exporter.Export(xzPlotModel, stream);
                }

                // Create YZ view (front view)
                var yzPlotModel = CreateViewPlot(partGroups, partColors, "YZ View (Front)", "Y Coordinate (Front-Back)", "Z Coordinate (Height)", "yz");
                string yzPath = outputPath.Replace(".png", "_YZ_FrontView.png");
                using (var stream = File.Create(yzPath))
                {
                    exporter.Export(yzPlotModel, stream);
                }

                // Also create the expected single file (use XY view as the main one)
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(plotModel, stream);
                }

                Console.WriteLine($"SUCCESS: Multiple view mammoth plots saved:");
                Console.WriteLine($"  Main: {Path.GetFullPath(outputPath)}");
                Console.WriteLine($"  XY (Top): {Path.GetFullPath(xyPath)}");
                Console.WriteLine($"  XZ (Side): {Path.GetFullPath(xzPath)}");
                Console.WriteLine($"  YZ (Front): {Path.GetFullPath(yzPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create 3D plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create PacMAP 2D embedding visualization with anatomical part coloring
        /// </summary>
        public static void PlotMammothPacMAP(double[,] embedding, int[] labels, string title, string outputPath)
        {
            PlotMammothPacMAP(embedding, labels, title, outputPath, null, true);
        }

        /// <summary>
        /// Create PacMAP 2D embedding visualization with anatomical part coloring (with correct part names)
        /// </summary>
        public static void PlotMammothPacMAP(double[,] embedding, int[] labels, string title, string outputPath, string[] partNames)
        {
            PlotMammothPacMAP(embedding, labels, title, outputPath, null, true, partNames);
        }

        /// <summary>
        /// Create UMAP 2D embedding visualization with anatomical part coloring
        /// </summary>
        public static void PlotMammothUMAP(double[,] embedding, int[] labels, string title, string outputPath)
        {
            PlotMammothUMAP(embedding, labels, title, outputPath, null, true);
        }

        /// <summary>
        /// Create UMAP 2D embedding visualization with anatomical part coloring (with correct part names)
        /// </summary>
        public static void PlotMammothUMAP(double[,] embedding, int[] labels, string title, string outputPath, string[] partNames)
        {
            PlotMammothUMAP(embedding, labels, title, outputPath, null, true, partNames);
        }

        /// <summary>
        /// Create UMAP 2D embedding visualization with anatomical part coloring and parameters
        /// </summary>
        public static void PlotMammothUMAP(double[,] embedding, int[] labels, string title, string outputPath, Dictionary<string, object>? paramInfo, bool autoFitAxes = true, string[]? partNames = null)
        {
            try
            {
                Console.WriteLine($"   Creating UMAP embedding plot: {title}");

                int numPoints = embedding.GetLength(0);

                // Map integer labels back to anatomical part names
                var uniqueLabels = labels.Distinct().ToArray();
                // Use provided partNames, or fall back to default ordering
                if (partNames == null)
                {
                    partNames = new[] { "body", "head", "feet", "tusks", "legs" }; // Default ordering from AssignMammothParts
                }
                var parts = new string[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    int labelIndex = labels[i];
                    if (labelIndex >= 0 && labelIndex < partNames.Length)
                    {
                        parts[i] = partNames[labelIndex];
                    }
                    else
                    {
                        parts[i] = "body"; // Fallback
                    }
                }

                var partColors = new Dictionary<string, OxyColor>
                {
                    { "body", OxyColors.ForestGreen },   // Green
                    { "head", OxyColors.Blue },          // Blue
                    { "legs", OxyColors.Brown },         // Brown
                    { "tusks", OxyColors.Gray },          // Gray (something)
                    { "trunk", OxyColors.Red },          // Red
                    { "feet", OxyColors.Gold }           // Yellow (if needed)
                };

                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Calculate data bounds for auto-fitting
                double minX = embedding[0, 0], maxX = embedding[0, 0];
                double minY = embedding[0, 1], maxY = embedding[0, 1];

                for (int i = 0; i < embedding.GetLength(0); i++)
                {
                    minX = Math.Min(minX, embedding[i, 0]);
                    maxX = Math.Max(maxX, embedding[i, 0]);
                    minY = Math.Min(minY, embedding[i, 1]);
                    maxY = Math.Max(maxY, embedding[i, 1]);
                }

                // Add padding for better visualization (10% of data range)
                double xRange = maxX - minX;
                double yRange = maxY - minY;
                double xPadding = xRange * 0.1;
                double yPadding = yRange * 0.1;

                // Add extra space on the right for legend (+20 units)
                double legendPadding = 20.0;

                if (autoFitAxes)
                {
                    // Auto-fit axes to data with padding
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Bottom,
                        Title = "X Coordinate (UMAP Dimension 1)",
                        Minimum = minX - xPadding,
                        Maximum = maxX + xPadding + legendPadding,
                        MajorStep = CalculateNiceStep(xRange + 2 * xPadding + legendPadding),
                        MinorStep = CalculateNiceStep(xRange + 2 * xPadding + legendPadding) / 2
                    });
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Left,
                        Title = "Y Coordinate (UMAP Dimension 2)",
                        Minimum = minY - yPadding,
                        Maximum = maxY + yPadding,
                        MajorStep = CalculateNiceStep(yRange + 2 * yPadding),
                        MinorStep = CalculateNiceStep(yRange + 2 * yPadding) / 2
                    });

                    Console.WriteLine($"   Auto-fitted axes: X=[{minX - xPadding:F2}, {maxX + xPadding + legendPadding:F2}], Y=[{minY - yPadding:F2}, {maxY + yPadding:F2}]");
                }
                else
                {
                    // Use fixed ranges (original behavior)
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Bottom,
                        Title = "X Coordinate (UMAP Dimension 1)",
                        Minimum = -25,
                        Maximum = 40 + legendPadding,  // Add space for legend on the right
                        MajorStep = 10,
                        MinorStep = 5
                    });
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Left,
                        Title = "Y Coordinate (UMAP Dimension 2)",
                        Minimum = -30,
                        Maximum = 30,
                        MajorStep = 10,
                        MinorStep = 5
                    });
                }

                // Add legend with standard configuration
                plotModel.Legends.Add(new Legend
                {
                    LegendTitle = "Mammoth Anatomy",
                    LegendPlacement = LegendPlacement.Inside,
                    LegendPosition = LegendPosition.TopRight,
                    LegendBackground = OxyColors.White,
                    LegendBorder = OxyColors.Black,
                    LegendTextColor = OxyColors.Black
                });

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(embedding[i, 0]);
                    partGroups[part].y.Add(embedding[i, 1]);
                }

                // Add scatter series for each part
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints) = kvp.Value;

                    if (xPoints.Count > 0)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                            MarkerType = MarkerType.Circle,
                            MarkerFill = partColors[part],
                            MarkerStroke = partColors[part],
                            MarkerSize = 1  // Smaller points for better visualization
                        };

                        for (int i = 0; i < xPoints.Count; i++)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(xPoints[i], yPoints[i]));
                        }

                        plotModel.Series.Add(scatterSeries);
                    }
                }

                // Export
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 800, Height = 600, Resolution = 300 };
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(plotModel, stream);
                }

                Console.WriteLine($"SUCCESS: UMAP plot saved to: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create UMAP plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Plot simple UMAP embedding with black dots for parameter experiments
        /// </summary>
        public static void PlotSimpleUMAP(double[,] embedding, string title, string outputPath, Dictionary<string, object>? paramInfo)
        {
            var plotModel = new PlotModel { Title = title, Background = OxyColors.White };

            var scatterSeries = new ScatterSeries
            {
                Title = "UMAP Embedding",
                MarkerType = MarkerType.Circle,
                MarkerSize = 1,
                MarkerFill = OxyColors.Black,
                MarkerStroke = OxyColors.Black
            };

            for (int i = 0; i < embedding.GetLength(0); i++)
            {
                scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1]));
            }

            plotModel.Series.Add(scatterSeries);
            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Bottom,
                Title = "X",
                Minimum = -35,
                Maximum = 35,
                MajorStep = 10,
                MinorStep = 5,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black
            });
            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Left,
                Title = "Y",
                Minimum = -35,
                Maximum = 35,
                MajorStep = 10,
                MinorStep = 5,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black
            });

            // Set plot area background to white
            plotModel.PlotAreaBackground = OxyColors.White;
            plotModel.Background = OxyColors.White;

            // Export to PNG
            var exporter = new PngExporter { Width = 1200, Height = 1200, Resolution = 300 };
            using (var stream = File.Create(outputPath))
            {
                exporter.Export(plotModel, stream);
            }

            Console.WriteLine($"   📊 Simple plot saved: {Path.GetFullPath(outputPath)}");
        }

        /// <summary>
        /// Create PacMAP 2D embedding visualization with anatomical part coloring and parameters
        /// </summary>
        public static void PlotMammothPacMAP(double[,] embedding, int[] labels, string title, string outputPath, Dictionary<string, object>? paramInfo, bool autoFitAxes = true, string[]? partNames = null)
        {
            try
            {
                Console.WriteLine($"   Creating PacMAP embedding plot: {title}");

                int numPoints = embedding.GetLength(0);

                // Map integer labels back to anatomical part names
                var uniqueLabels = labels.Distinct().ToArray();
                // Use provided partNames, or fall back to default ordering
                if (partNames == null)
                {
                    partNames = new[] { "body", "head", "feet", "tusks", "legs" }; // Default ordering from AssignMammothParts
                }
                var parts = new string[numPoints];

                for (int i = 0; i < numPoints; i++)
                {
                    int labelIndex = labels[i];
                    if (labelIndex >= 0 && labelIndex < partNames.Length)
                    {
                        parts[i] = partNames[labelIndex];
                    }
                    else
                    {
                        parts[i] = "body"; // Fallback
                    }
                }

                var partColors = new Dictionary<string, OxyColor>
                {
                    { "body", OxyColors.ForestGreen },   // Green
                    { "head", OxyColors.Blue },          // Blue
                    { "legs", OxyColors.Brown },         // Brown
                    { "tusks", OxyColors.Gray },          // Gray (something)
                    { "trunk", OxyColors.Red },          // Red
                    { "feet", OxyColors.Gold }           // Yellow (if needed)
                };

                var plotModel = new PlotModel
                {
                    Title = title,
                    Background = OxyColors.White,
                    PlotAreaBorderColor = OxyColors.Black
                };

                // Calculate data bounds for auto-fitting
                double minX = embedding[0, 0], maxX = embedding[0, 0];
                double minY = embedding[0, 1], maxY = embedding[0, 1];

                for (int i = 0; i < embedding.GetLength(0); i++)
                {
                    minX = Math.Min(minX, embedding[i, 0]);
                    maxX = Math.Max(maxX, embedding[i, 0]);
                    minY = Math.Min(minY, embedding[i, 1]);
                    maxY = Math.Max(maxY, embedding[i, 1]);
                }

                // Add padding for better visualization (10% of data range)
                double xRange = maxX - minX;
                double yRange = maxY - minY;
                double xPadding = xRange * 0.1;
                double yPadding = yRange * 0.1;

                // Add extra space on the right for legend (+20 units)
                double legendPadding = 20.0;

                if (autoFitAxes)
                {
                    // Auto-fit axes to data with padding
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Bottom,
                        Title = "X Coordinate (PacMAP Dimension 1)",
                        Minimum = minX - xPadding,
                        Maximum = maxX + xPadding + legendPadding,
                        MajorStep = CalculateNiceStep(xRange + 2 * xPadding + legendPadding),
                        MinorStep = CalculateNiceStep(xRange + 2 * xPadding + legendPadding) / 2
                    });
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Left,
                        Title = "Y Coordinate (PacMAP Dimension 2)",
                        Minimum = minY - yPadding,
                        Maximum = maxY + yPadding,
                        MajorStep = CalculateNiceStep(yRange + 2 * yPadding),
                        MinorStep = CalculateNiceStep(yRange + 2 * yPadding) / 2
                    });

                    Console.WriteLine($"   Auto-fitted axes: X=[{minX - xPadding:F2}, {maxX + xPadding + legendPadding:F2}], Y=[{minY - yPadding:F2}, {maxY + yPadding:F2}]");
                }
                else
                {
                    // Use fixed ranges (original behavior)
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Bottom,
                        Title = "X Coordinate (PacMAP Dimension 1)",
                        Minimum = -25,
                        Maximum = 40 + legendPadding,  // Add space for legend on the right
                        MajorStep = 10,
                        MinorStep = 5
                    });
                    plotModel.Axes.Add(new LinearAxis {
                        Position = AxisPosition.Left,
                        Title = "Y Coordinate (PacMAP Dimension 2)",
                        Minimum = -30,
                        Maximum = 30,
                        MajorStep = 10,
                        MinorStep = 5
                    });
                }

                // Add legend with standard configuration
                plotModel.Legends.Add(new Legend
                {
                    LegendTitle = "Mammoth Anatomy",
                    LegendPlacement = LegendPlacement.Inside,
                    LegendPosition = LegendPosition.TopRight,
                    LegendBackground = OxyColors.White,
                    LegendBorder = OxyColors.Black,
                    LegendTextColor = OxyColors.Black
                });

                // Group points by anatomical part
                var partGroups = new Dictionary<string, (List<double> x, List<double> y)>();
                foreach (var part in partColors.Keys)
                {
                    partGroups[part] = (new List<double>(), new List<double>());
                }

                for (int i = 0; i < numPoints; i++)
                {
                    string part = parts[i];
                    partGroups[part].x.Add(embedding[i, 0]);
                    partGroups[part].y.Add(embedding[i, 1]);
                }

                // Add scatter series for each part
                foreach (var kvp in partGroups)
                {
                    var part = kvp.Key;
                    var (xPoints, yPoints) = kvp.Value;

                    if (xPoints.Count > 0)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                            MarkerType = MarkerType.Circle,
                            MarkerFill = partColors[part],
                            MarkerStroke = partColors[part],
                            MarkerSize = 1  // Smaller points for better visualization
                        };

                        for (int i = 0; i < xPoints.Count; i++)
                        {
                            scatterSeries.Points.Add(new ScatterPoint(xPoints[i], yPoints[i]));
                        }

                        plotModel.Series.Add(scatterSeries);
                    }
                }

                // Export
                var exporter = new OxyPlot.WindowsForms.PngExporter { Width = 800, Height = 600, Resolution = 300 };
                using (var stream = File.Create(outputPath))
                {
                    exporter.Export(plotModel, stream);
                }

                Console.WriteLine($"SUCCESS: PacMAP plot saved to: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to create PacMAP plot: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Save embedding data as CSV
        /// </summary>
        public static void SaveEmbeddingAsCSV(double[,] embedding, string[]? labels, string outputPath)
        {
            try
            {
                int numPoints = embedding.GetLength(0);

                using (var writer = new StreamWriter(outputPath))
                {
                    // Write header
                    if (labels != null)
                        writer.WriteLine("x,y,label");
                    else
                        writer.WriteLine("x,y");

                    // Write data
                    for (int i = 0; i < numPoints; i++)
                    {
                        if (labels != null)
                            writer.WriteLine($"{embedding[i, 0]},{embedding[i, 1]},{labels[i]}");
                        else
                            writer.WriteLine($"{embedding[i, 0]},{embedding[i, 1]}");
                    }
                }

                Console.WriteLine($"SUCCESS: Embedding saved as CSV: {Path.GetFullPath(outputPath)} ({numPoints} points)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: Failed to save embedding as CSV: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Create a specific view plot (XY, XZ, or YZ)
        /// </summary>
        private static PlotModel CreateViewPlot(
            Dictionary<string, (List<double> x, List<double> y, List<double> z)> partGroups,
            Dictionary<string, OxyColor> partColors,
            string title,
            string xAxisTitle,
            string yAxisTitle,
            string viewType)
        {
            // Calculate data bounds for auto-fitting
            double minX = double.MaxValue, maxX = double.MinValue;
            double minY = double.MaxValue, maxY = double.MinValue;

            foreach (var kvp in partGroups)
            {
                var (xPoints, yPoints, zPoints) = kvp.Value;
                for (int i = 0; i < xPoints.Count; i++)
                {
                    double x, y;
                    switch (viewType.ToLower())
                    {
                        case "xy":
                            x = xPoints[i];
                            y = yPoints[i];
                            break;
                        case "xz":
                            x = xPoints[i];
                            y = zPoints[i];
                            break;
                        case "yz":
                            x = yPoints[i];
                            y = zPoints[i];
                            break;
                        default:
                            x = xPoints[i];
                            y = yPoints[i];
                            break;
                    }
                    minX = Math.Min(minX, x);
                    maxX = Math.Max(maxX, x);
                    minY = Math.Min(minY, y);
                    maxY = Math.Max(maxY, y);
                }
            }

            // Add padding for better visualization
            double xRange = maxX - minX;
            double yRange = maxY - minY;
            double xPadding = xRange * 0.1;
            double yPadding = yRange * 0.1;
            double legendPadding = 10.0; // Extra space for legend

            var plotModel = new PlotModel
            {
                Title = title,
                Background = OxyColors.White,
                PlotAreaBorderColor = OxyColors.Black
            };

            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Bottom,
                Title = xAxisTitle,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black,
                Minimum = minX - xPadding,
                Maximum = maxX + xPadding + legendPadding
            });
            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Left,
                Title = yAxisTitle,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black,
                Minimum = minY - yPadding,
                Maximum = maxY + yPadding
            });

            // Ensure plot area is white
            plotModel.PlotAreaBackground = OxyColors.White;

            // Add legend
            plotModel.Legends.Add(new Legend
            {
                LegendTitle = "Mammoth Anatomy",
                LegendPlacement = LegendPlacement.Inside,
                LegendPosition = LegendPosition.TopRight,
                LegendBackground = OxyColors.White,
                LegendBorder = OxyColors.Black,
                LegendTextColor = OxyColors.Black
            });

            foreach (var kvp in partGroups)
            {
                var part = kvp.Key;
                var (xPoints, yPoints, zPoints) = kvp.Value;

                if (xPoints.Count > 0)
                {
                    var scatterSeries = new ScatterSeries
                    {
                        Title = $"{char.ToUpper(part[0]) + part.Substring(1)} ({xPoints.Count} points)",
                        MarkerType = MarkerType.Circle,
                        MarkerFill = partColors[part],
                        MarkerStroke = partColors[part],
                        MarkerSize = 3
                    };

                    for (int i = 0; i < xPoints.Count; i++)
                    {
                        double x, y;
                        switch (viewType.ToLower())
                        {
                            case "xy":
                                x = xPoints[i];
                                y = yPoints[i];
                                break;
                            case "xz":
                                x = xPoints[i];
                                y = zPoints[i];
                                break;
                            case "yz":
                                x = yPoints[i];
                                y = zPoints[i];
                                break;
                            default:
                                x = xPoints[i];
                                y = yPoints[i];
                                break;
                        }

                        scatterSeries.Points.Add(new ScatterPoint(x, y));
                    }

                    plotModel.Series.Add(scatterSeries);
                }
            }

            return plotModel;
        }

  
  
        /// <summary>
        /// Plot simple PACMAP embedding with black dots for parameter experiments
        /// </summary>
        public static void PlotSimplePacMAP(double[,] embedding, string title, string outputPath, Dictionary<string, object>? paramInfo)
        {
            var plotModel = new PlotModel { Title = title, Background = OxyColors.White };

            var scatterSeries = new ScatterSeries
            {
                Title = "PACMAP Embedding",
                MarkerType = MarkerType.Circle,
                MarkerSize = 1,
                MarkerFill = OxyColors.Black,
                MarkerStroke = OxyColors.Black
            };

            for (int i = 0; i < embedding.GetLength(0); i++)
            {
                scatterSeries.Points.Add(new ScatterPoint(embedding[i, 0], embedding[i, 1]));
            }

            plotModel.Series.Add(scatterSeries);
            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Bottom,
                Title = "X",
                Minimum = -35,
                Maximum = 35,
                MajorStep = 10,
                MinorStep = 5,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black
            });
            plotModel.Axes.Add(new LinearAxis {
                Position = AxisPosition.Left,
                Title = "Y",
                Minimum = -35,
                Maximum = 35,
                MajorStep = 10,
                MinorStep = 5,
                AxislineColor = OxyColors.Black,
                TicklineColor = OxyColors.Black,
                TextColor = OxyColors.Black
            });

            // Set plot area background to white
            plotModel.PlotAreaBackground = OxyColors.White;
            plotModel.Background = OxyColors.White;

            // Export to PNG
            var exporter = new PngExporter { Width = 1200, Height = 1200, Resolution = 300 };
            using (var stream = File.Create(outputPath))
            {
                exporter.Export(plotModel, stream);
            }

            Console.WriteLine($"   📊 Simple plot saved: {Path.GetFullPath(outputPath)}");
        }

        /// <summary>
        /// Create detailed parameter information string for image annotation
        /// </summary>
        public static string CreateParameterInfo(Dictionary<string, object> paramInfo)
        {
            if (paramInfo == null) return "";

            var info = new System.Text.StringBuilder();
            foreach (var kvp in paramInfo)
            {
                info.AppendLine($"{kvp.Key}: {kvp.Value}");
            }
            return info.ToString();
        }

        private static OxyColor GetPartColor(string part)
        {
            return part switch
            {
                "feet" => OxyColors.Orange,
                "legs" => OxyColors.Blue,
                "body" => OxyColors.Green,
                "head" => OxyColors.Purple,
                "tusks" => OxyColors.Yellow,
                "trunk" => OxyColors.Red,
                _ => OxyColors.Gray
            };
        }

        /// <summary>
        /// Calculate a nice step size for axis ticks based on data range
        /// </summary>
        private static double CalculateNiceStep(double range)
        {
            if (range <= 0) return 1.0;

            // Target number of tick marks (approximately 5-10)
            double targetTickCount = 8.0;
            double roughStep = range / targetTickCount;

            // Find nice round numbers (1, 2, 5, 10, 20, 50, etc.)
            double exponent = Math.Floor(Math.Log10(roughStep));
            double baseValue = Math.Pow(10, exponent);

            double[] niceValues = { 1.0, 2.0, 5.0 };
            double niceStep = niceValues.FirstOrDefault(v => v * baseValue >= roughStep, niceValues.Last()) * baseValue;

            // Fallback if FirstOrDefault still fails
            if (double.IsNaN(niceStep) || niceStep <= 0)
            {
                niceStep = 1.0;
            }

            // Ensure minimum step size to avoid too many ticks
            return Math.Max(niceStep, 0.1);
        }

        }
}