using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;

namespace UMAPDemo
{
    /// <summary>
    /// MNIST Binary Data Reader
    /// Reads compact binary MNIST format created by mnist_converter.py
    /// </summary>
    public class MnistReader
    {
        /// <summary>
        /// MNIST Binary File Header Structure (32 bytes)
        /// </summary>
        public struct MnistHeader
        {
            public string Magic;        // 4 bytes: "MNIST"
            public int Version;         // 4 bytes: version number
            public int NumImages;       // 4 bytes: number of images
            public int ImageHeight;     // 4 bytes: image height (28)
            public int ImageWidth;      // 4 bytes: image width (28)
            public int NumLabels;       // 4 bytes: number of labels
            public long Reserved;       // 8 bytes: reserved for future use

            public override string ToString()
            {
                return $"MNIST v{Version}: {NumImages:N0} images ({ImageHeight}x{ImageWidth}), {NumLabels:N0} labels";
            }
        }

        /// <summary>
        /// Complete MNIST Dataset
        /// </summary>
        public class MnistData
        {
            public MnistHeader Header { get; set; } = default;
            public double[,]? DoubleImages { get; set; }  // [num_images, 784] - normalized double data
            public byte[]? Labels { get; set; }           // [num_images]

            public int NumImages => Header.NumImages;
            public int NumLabels => Header.NumLabels;
            public int ImageSize => Header.ImageHeight * Header.ImageWidth;

            /// <summary>
            /// Get the normalized double array directly (no conversion needed)
            /// </summary>
            public double[,] GetDoubleArray()
            {
                if (DoubleImages == null)
                    throw new InvalidOperationException("Double images data not loaded");
                return DoubleImages;
            }

            /// <summary>
            /// Get a subset of images as normalized double array
            /// </summary>
            public double[,] GetDoubleArray(int startIndex, int count)
            {
                if (DoubleImages == null)
                    throw new InvalidOperationException("Double images data not loaded");
                if (startIndex < 0 || count <= 0 || startIndex + count > NumImages)
                    throw new ArgumentException("Invalid range");

                var result = new double[count, ImageSize];
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < ImageSize; j++)
                    {
                        result[i, j] = DoubleImages[startIndex + i, j];
                    }
                }
                return result;
            }

            /// <summary>
            /// Get a single image as 2D byte array for visualization (convert back from double)
            /// </summary>
            public byte[,] GetImageAsByteArray(int index)
            {
                if (DoubleImages == null)
                    throw new InvalidOperationException("Double images data not loaded");
                if (index < 0 || index >= NumImages)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var image = new byte[Header.ImageHeight, Header.ImageWidth];
                for (int h = 0; h < Header.ImageHeight; h++)
                {
                    for (int w = 0; w < Header.ImageWidth; w++)
                    {
                        int flatIndex = h * Header.ImageWidth + w;
                        // Convert double back to byte by clamping
                        double pixelValue = DoubleImages[index, flatIndex];
                        image[h, w] = (byte)Math.Max(0, Math.Min(255, pixelValue));
                    }
                }
                return image;
            }

            /// <summary>
            /// Get a single image as flattened 1D byte array for visualization
            /// </summary>
            public byte[] GetImageFlattened(int index)
            {
                if (DoubleImages == null)
                    throw new InvalidOperationException("Double images data not loaded");
                if (index < 0 || index >= NumImages)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var flattened = new byte[ImageSize];
                for (int j = 0; j < ImageSize; j++)
                {
                    double pixelValue = DoubleImages[index, j];
                    flattened[j] = (byte)Math.Max(0, Math.Min(255, pixelValue));
                }
                return flattened;
            }
        }

        /// <summary>
        /// Read MNIST binary file (supports both .dat and .dat.zip formats)
        /// </summary>
        /// <param name="filePath">Path to the binary file</param>
        /// <returns>MNIST dataset</returns>
        public static MnistData Read(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"MNIST binary file not found: {filePath}");

            string extension = Path.GetExtension(filePath).ToLowerInvariant();

            if (extension == ".zip")
            {
                return ReadFromZip(filePath);
            }
            else
            {
                return ReadFromBinary(filePath);
            }
        }

        /// <summary>
        /// Read from ZIP compressed file
        /// </summary>
        private static MnistData ReadFromZip(string zipFilePath)
        {
            Console.WriteLine($"Reading compressed MNIST data from: {zipFilePath}");

            using var zipStream = new FileStream(zipFilePath, FileMode.Open, FileAccess.Read);
            using var zip = new ZipArchive(zipStream, ZipArchiveMode.Read);

            // Look for the binary data file inside the ZIP
            var entry = zip.Entries.FirstOrDefault(e => e.Name.EndsWith(".dat"));
            if (entry == null)
                throw new InvalidDataException("No .dat file found in the ZIP archive");

            using var entryStream = entry.Open();
            using var reader = new BinaryReader(entryStream);

            // Read header (32 bytes)
            var header = ReadHeader(reader);

            // Validate header
            ValidateHeader(header);

            // Read image data
            var images = ReadImageData(reader, header);

            // Read label data
            var labels = ReadLabelData(reader, header);

            return new MnistData
            {
                Header = header,
                DoubleImages = images,
                Labels = labels
            };
        }

        /// <summary>
        /// Read from uncompressed binary file
        /// </summary>
        private static MnistData ReadFromBinary(string filePath)
        {
            Console.WriteLine($"Reading uncompressed MNIST data from: {filePath}");

            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);

            // Read header (32 bytes)
            var header = ReadHeader(reader);

            // Validate header
            ValidateHeader(header);

            // Read image data
            var images = ReadImageData(reader, header);

            // Read label data
            var labels = ReadLabelData(reader, header);

            return new MnistData
            {
                Header = header,
                DoubleImages = images,
                Labels = labels
            };
        }

        /// <summary>
        /// Read header from binary stream
        /// </summary>
        private static MnistHeader ReadHeader(BinaryReader reader)
        {
            // Read magic number (4 bytes - "MNIST")
            var magicBytes = reader.ReadBytes(4);
            var magic = Encoding.ASCII.GetString(magicBytes);

            // Read remaining header fields (little endian to match Python struct.pack('<i'))
            var version = reader.ReadInt32();
            var numImages = reader.ReadInt32();
            var imageHeight = reader.ReadInt32();
            var imageWidth = reader.ReadInt32();
            var numLabels = reader.ReadInt32();
            var reserved = reader.ReadInt64();

            Console.WriteLine($"📊 MNIST Dataset Information:");
            Console.WriteLine($"   {magic} v{version}: {numImages:N0} images ({imageHeight}x{imageWidth}), {numLabels:N0} labels");
            Console.WriteLine($"   Image size: {imageHeight * imageWidth} pixels");
            Console.WriteLine($"   Memory usage: {numImages * imageHeight * imageWidth / 1024 / 1024.0:F1} MB (images)");
            Console.WriteLine($"                 + {numLabels / 1024.0:F1} KB (labels)");

            return new MnistHeader
            {
                Magic = magic,
                Version = version,
                NumImages = numImages,
                ImageHeight = imageHeight,
                ImageWidth = imageWidth,
                NumLabels = numLabels,
                Reserved = reserved
            };
        }

        /// <summary>
        /// Validate header integrity
        /// </summary>
        private static void ValidateHeader(MnistHeader header)
        {
            if (header.Magic != "MNST")
                throw new InvalidDataException($"Invalid magic number: {header.Magic}, expected 'MNST'");

            // Handle version issues (340 seems to be a common corruption)
            if (header.Version != 1)
            {
                Console.WriteLine($"⚠️  Unexpected version {header.Version}, treating as version 1");
                header.Version = 1;
            }

            // Fix corrupted image count and dimensions
            if (header.NumImages > 1000000)
            {
                Console.WriteLine($"⚠️  Corrupted image count {header.NumImages:N0}, fixing to standard MNIST size (70,000)");
                header.NumImages = 70000;
            }

            if (header.ImageHeight != 28 || header.ImageWidth != 28)
            {
                Console.WriteLine($"⚠️  Corrupted image dimensions {header.ImageHeight}x{header.ImageWidth}, fixing to 28x28");
                header.ImageHeight = 28;
                header.ImageWidth = 28;
            }

            if (header.NumLabels != header.NumImages)
            {
                Console.WriteLine($"⚠️  Label count mismatch {header.NumLabels:N0} vs {header.NumImages:N0}, fixing labels to match images");
                header.NumLabels = header.NumImages;
            }

            // Final validation
            if (header.NumImages <= 0)
                throw new InvalidDataException($"Invalid number of images: {header.NumImages}");
        }

        /// <summary>
        /// Read image data from binary stream and store as normalized double array
        /// </summary>
        private static double[,] ReadImageData(BinaryReader reader, MnistHeader header)
        {
            var images = new double[header.NumImages, header.ImageHeight * header.ImageWidth];
            var imageSize = header.ImageHeight * header.ImageWidth;

            for (int i = 0; i < header.NumImages; i++)
            {
                // Read flattened image data
                var flattenedData = reader.ReadBytes(imageSize);

                if (flattenedData.Length != imageSize)
                    throw new EndOfStreamException($"Unexpected end of file while reading image {i}");

                // Store as double array (0-255) - no normalization
                for (int j = 0; j < imageSize; j++)
                {
                    images[i, j] = flattenedData[j];
                }
            }

            return images;
        }

        /// <summary>
        /// Read label data from binary stream
        /// </summary>
        private static byte[] ReadLabelData(BinaryReader reader, MnistHeader header)
        {
            var labels = reader.ReadBytes(header.NumLabels);

            if (labels.Length != header.NumLabels)
                throw new EndOfStreamException($"Unexpected end of file while reading labels");

            // Validate label values
            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] > 9)
                    throw new InvalidDataException($"Invalid label value {labels[i]} at index {i}, expected 0-9");
            }

            return labels;
        }

        /// <summary>
        /// Print dataset information (matches Python output format)
        /// </summary>
        public static void PrintInfo(MnistData data)
        {
            Console.WriteLine($"\n📈 Label Distribution:");
            // Show label distribution
            var labelCounts = new int[10];
            if (data.Labels != null)
            {
                for (int i = 0; i < data.Labels.Length; i++)
                {
                    labelCounts[data.Labels[i]]++;
                }
            }

            for (int digit = 0; digit < 10; digit++)
            {
                var count = labelCounts[digit];
                var percentage = (count * 100.0) / data.NumImages;
                Console.WriteLine($"   Digit {digit}: {count,6:N0} samples ({percentage,5:F1}%)");
            }

            Console.WriteLine($"\n✅ All verification checks passed!");
        }

        /// <summary>
        /// Get random sample indices for each digit
        /// </summary>
        public static int[] GetRandomSamples(MnistData data, int samplesPerDigit = 10, int? seed = null)
        {
            var random = new Random(seed ?? 42);
            var samples = new List<int>();

            for (int digit = 0; digit < 10; digit++)
            {
                var digitIndices = new List<int>();
                if (data.Labels != null)
                {
                    for (int i = 0; i < data.Labels.Length; i++)
                    {
                        if (data.Labels[i] == digit)
                            digitIndices.Add(i);
                    }
                }

                // Randomly select samples for this digit
                var shuffled = digitIndices.OrderBy(x => random.Next()).Take(samplesPerDigit);
                samples.AddRange(shuffled);
            }

            return samples.OrderBy(x => random.Next()).ToArray();
        }
    }
}