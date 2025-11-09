#ifndef UWOT_CONSTANTS_HPP
#define UWOT_CONSTANTS_HPP

#include <cstdint>

/**
 * @file uwot_constants.hpp
 * @brief Centralized magic constants and configuration values for UMAP
 *
 * This header consolidates all magic numbers, configuration limits, and
 * protocol constants that were previously scattered throughout the codebase.
 */

namespace uwot {
namespace constants {

    // File format constants
    constexpr uint32_t UMAP_FILE_MAGIC = 0x554D4150;        // "UMAP"
    constexpr uint32_t UMOD_FILE_MAGIC = 0x5A4D4F44;        // "UMOD"
    constexpr uint32_t VERSION_STRING_LENGTH = 16;          // Length of version string
    constexpr uint32_t INITIAL_CRC32 = 0xFFFFFFFF;          // CRC32 initialization value
    constexpr uint32_t FINAL_CRC32_XOR = 0xFFFFFFFF;        // CRC32 final XOR value

    // File size limits (for safety)
    // Note: Using size_t (64-bit) to avoid overflow, but LZ4 uses int (32-bit signed = 2GB max)
    // So we cap at 2GB to stay within LZ4's safe range
    constexpr size_t MAX_DECOMPRESSED_SIZE = static_cast<size_t>(2000) * 1024 * 1024;  // 2000MB (2GB) limit - safe for LZ4
    constexpr size_t MAX_COMPRESSED_SIZE = static_cast<size_t>(1500) * 1024 * 1024;    // 1500MB limit - safe for LZ4

    // Dataset size thresholds for algorithm selection
    constexpr int LARGE_DATASET_THRESHOLD = 20000;           // Switch to random init
    constexpr int VERY_LARGE_DATASET_THRESHOLD = 50000;      // HNSW parameter scaling
    constexpr int HNSW_PARALLEL_THRESHOLD = 5000;           // Parallel point addition
    constexpr int OPENMP_THRESHOLD = 1000;                  // General OpenMP threshold
    constexpr int PROGRESS_REPORTING_THRESHOLD = 10000;     // Fine-grained progress

    // HNSW configuration defaults
    constexpr int DEFAULT_M_SMALL = 32;                      // Default M for <20k points
    constexpr int DEFAULT_M_MEDIUM = 48;                     // Default M for 20k-50k points
    constexpr int DEFAULT_M_LARGE = 64;                      // Default M for >50k points
    constexpr int DEFAULT_EF_SMALL = 300;                    // Default ef for <20k points
    constexpr int DEFAULT_EF_MEDIUM = 400;                   // Default ef for 20k-50k points
    constexpr int DEFAULT_EF_LARGE = 500;                    // Default ef for >50k points
    constexpr int DEFAULT_BASE_EF_SMALL = 100;               // Base ef search for <20k
    constexpr int DEFAULT_BASE_EF_MEDIUM = 200;              // Base ef search for 20k-50k
    constexpr int DEFAULT_BASE_EF_LARGE = 300;               // Base ef search for >50k

    // Transform optimization limits
    constexpr int MAX_FAST_TRANSFORM_NEIGHBORS = 128;        // Stack safety limit
    constexpr int FAST_TRANSFORM_CHECK_COUNT = 10;           // Validation sample size
    constexpr int SMOOTH_KNN_MAX_ITER = 64;                  // Binary search iterations

    // Progress reporting constants
    constexpr float PROGRESS_START = 0.0f;
    constexpr float PROGRESS_COMPLETE = 100.0f;
    constexpr int PROGRESS_INTERVAL_SMALL = 20;              // 5% for small datasets
    constexpr int PROGRESS_INTERVAL_MEDIUM = 50;             // 2% for medium datasets
    constexpr int PROGRESS_INTERVAL_LARGE = 100;             // 1% for large datasets

    // Distance and outlier detection thresholds
    constexpr float PERCENTILE_GOOD = 95.0f;                 // Good data threshold
    constexpr float PERCENTILE_BAD = 99.0f;                  // Bad data threshold
    constexpr float DISTANCE_WARNING_BANDWIDTH = 1.0f;       // Bandwidth warning threshold
    constexpr float LARGE_DISTANCE_PENALTY = 1000.0f;        // Penalty for missing neighbors

    // Data validation sampling limits
    constexpr int MAX_OBSERVATIONS_TO_SAMPLE = 1000;         // Max observations for validation
    constexpr int MAX_FEATURES_TO_SAMPLE = 50;               // Max features for validation
    constexpr int BINARY_CHECK_LIMIT = 10;                   // Early stop for binary detection
    constexpr int BINARY_VARIANCE_THRESHOLD = 10;            // Non-binary count threshold
    constexpr int CONSTANT_FEATURE_THRESHOLD = 50;           // 50% constant features warning

    // Fast transform optimization
    constexpr int FAST_TRANSFORM_ITERATIONS = 32;            // Transform optimization iterations
    constexpr int SCHEDULE_CHUNK_SIZE = 100;                 // OpenMP dynamic schedule chunk

    // CRC32 and hash constants
    constexpr uint32_t FNV1A_HASH_INIT = 5381;               // FNV-1a hash initial value
    constexpr int BUFFER_SIZE = 8192;                        // File I/O buffer size

    // Time constants for progress formatting
    constexpr int SECONDS_PER_MINUTE = 60;
    constexpr int SECONDS_PER_HOUR = 3600;

    // Random seed defaults
    constexpr uint32_t DEFAULT_INITIALIZE_SEED = 42;         // Default initialization seed
    constexpr uint32_t DEFAULT_OPTIMIZE_SEED = 43;           // Default optimization seed

} // namespace constants
} // namespace uwot

#endif // UWOT_CONSTANTS_HPP