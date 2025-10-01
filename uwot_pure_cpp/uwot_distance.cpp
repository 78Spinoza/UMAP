#include "uwot_distance.h"
#include <cstdio>  // For fprintf warnings
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>

namespace distance_metrics {

    float euclidean_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    float cosine_distance(const float* a, const float* b, int dim) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        // Enhanced zero-norm handling with informative behavior
        const float ZERO_TOLERANCE = 1e-12f;

        if (norm_a < ZERO_TOLERANCE && norm_b < ZERO_TOLERANCE) {
            // Both vectors are zero - they are identical in direction
            return 0.0f;  // Zero distance (maximum similarity)
        } else if (norm_a < ZERO_TOLERANCE || norm_b < ZERO_TOLERANCE) {
            // One vector is zero, the other is not - maximum dissimilarity
            return 1.0f;  // Maximum distance (minimum similarity)
        }

        float cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

        return 1.0f - cosine_sim;
    }

    float manhattan_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    float correlation_distance(const float* a, const float* b, int dim) {
        float mean_a = 0.0f, mean_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a /= static_cast<float>(dim);
        mean_b /= static_cast<float>(dim);

        float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff_a = a[i] - mean_a;
            float diff_b = b[i] - mean_b;
            num += diff_a * diff_b;
            den_a += diff_a * diff_a;
            den_b += diff_b * diff_b;
        }

        // Enhanced zero-variance handling with informative behavior
        const float ZERO_VARIANCE_TOLERANCE = 1e-12f;

        if (den_a < ZERO_VARIANCE_TOLERANCE && den_b < ZERO_VARIANCE_TOLERANCE) {
            // Both vectors have zero variance (all elements identical)
            // If they have the same mean, they are perfectly correlated
            if (std::abs(mean_a - mean_b) < 1e-12f) {
                return 0.0f;  // Zero distance (perfect correlation)
            } else {
                return 2.0f;  // Maximum distance for correlation (1 - (-1))
            }
        } else if (den_a < ZERO_VARIANCE_TOLERANCE || den_b < ZERO_VARIANCE_TOLERANCE) {
            // One vector has zero variance, the other doesn't - cannot correlate
            return 1.0f;  // Maximum distance (zero correlation)
        }

        float correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0f, std::min(1.0f, correlation));

        return 1.0f - correlation;
    }

    float hamming_distance(const float* a, const float* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6f) {
                different++;
            }
        }
        return static_cast<float>(different) / static_cast<float>(dim);
    }

    float compute_distance(const float* a, const float* b, int dim, UwotMetric metric) {
        switch (metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case UWOT_METRIC_COSINE:
            return cosine_distance(a, b, dim);
        case UWOT_METRIC_MANHATTAN:
            return manhattan_distance(a, b, dim);
        case UWOT_METRIC_CORRELATION:
            return correlation_distance(a, b, dim);
        case UWOT_METRIC_HAMMING:
            return hamming_distance(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
        }
    }

    // Data validation functions for specific metrics
    bool validate_hamming_data(const float* data, int n_obs, int n_dim) {
        int non_binary_count = 0;
        const int MAX_NON_BINARY_TO_CHECK = std::min(1000, n_obs); // Sample validation
        const int MAX_FEATURES_TO_CHECK = std::min(50, n_dim);     // Sample features

        for (int i = 0; i < MAX_NON_BINARY_TO_CHECK; i++) {
            for (int j = 0; j < MAX_FEATURES_TO_CHECK; j++) {
                float val = data[i * n_dim + j];
                // Check if value is approximately 0 or 1 (allowing small floating point errors)
                if (!(std::abs(val) < 1e-6f || std::abs(val - 1.0f) < 1e-6f)) {
                    non_binary_count++;
                    if (non_binary_count > 10) { // Stop early if clearly not binary
                        return false;
                    }
                }
            }
        }

        // Consider data binary if less than 5% non-binary values in sample
        float non_binary_ratio = static_cast<float>(non_binary_count) / (MAX_NON_BINARY_TO_CHECK * MAX_FEATURES_TO_CHECK);
        return non_binary_ratio < 0.05f;
    }

    bool validate_correlation_data(const float* data, int n_obs, int n_dim) {
        // Check if data has sufficient variance for meaningful correlation
        if (n_dim < 2) return false; // Correlation needs at least 2 dimensions

        // Sample a few features to check for constant values (zero variance)
        const int MAX_FEATURES_TO_CHECK = std::min(10, n_dim);
        int constant_features = 0;

        for (int feature = 0; feature < MAX_FEATURES_TO_CHECK; feature++) {
            float min_val = data[feature];
            float max_val = data[feature];

            // Sample values to check variance
            int sample_size = std::min(100, n_obs);
            for (int i = 0; i < sample_size; i++) {
                float val = data[i * n_dim + feature];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            // Check if feature has essentially no variance
            if ((max_val - min_val) < 1e-10f) {
                constant_features++;
            }
        }

        // Warn if more than 50% of sampled features are constant
        return constant_features < (MAX_FEATURES_TO_CHECK / 2);
    }

    // Zero-norm vector detection for cosine and correlation metrics
    bool detect_zero_norm_vectors(const float* data, int n_obs, int n_dim, UwotMetric metric) {
        const int MAX_VECTORS_TO_CHECK = std::min(1000, n_obs);
        int zero_norm_count = 0;

        for (int i = 0; i < MAX_VECTORS_TO_CHECK; i++) {
            float norm = 0.0f;

            if (metric == UWOT_METRIC_COSINE) {
                // Check for zero Euclidean norm
                for (int j = 0; j < n_dim; j++) {
                    norm += data[i * n_dim + j] * data[i * n_dim + j];
                }
                norm = std::sqrt(norm);
            } else if (metric == UWOT_METRIC_CORRELATION) {
                // Check for zero variance (constant vector)
                float mean = 0.0f;
                for (int j = 0; j < n_dim; j++) {
                    mean += data[i * n_dim + j];
                }
                mean /= static_cast<float>(n_dim);

                float variance = 0.0f;
                for (int j = 0; j < n_dim; j++) {
                    float diff = data[i * n_dim + j] - mean;
                    variance += diff * diff;
                }
                norm = std::sqrt(variance);
            }

            if (norm < 1e-12f) {
                zero_norm_count++;
                if (zero_norm_count > 10) {
                    // Found enough zero-norm vectors to be significant
                    return true;
                }
            }
        }

        // Consider it problematic if >5% of checked vectors have zero norm
        float zero_norm_ratio = static_cast<float>(zero_norm_count) / static_cast<float>(MAX_VECTORS_TO_CHECK);
        return zero_norm_ratio > 0.05f;
    }

    // Main validation function that issues warnings for inappropriate data
    void validate_metric_data(const float* data, int n_obs, int n_dim, UwotMetric metric) {
        switch (metric) {
            case UWOT_METRIC_HAMMING:
                if (!validate_hamming_data(data, n_obs, n_dim)) {
                    fprintf(stderr, "WARNING: Hamming metric expects binary data (0/1 values). "
                                   "Non-binary data detected - results may be meaningless.\n");
                }
                break;

            case UWOT_METRIC_CORRELATION:
                if (!validate_correlation_data(data, n_obs, n_dim)) {
                    fprintf(stderr, "WARNING: Correlation metric expects data with meaningful variance. "
                                   "Constant or near-constant features detected - results may be unreliable.\n");
                }

                // Additional check for zero-variance vectors
                if (detect_zero_norm_vectors(data, n_obs, n_dim, UWOT_METRIC_CORRELATION)) {
                    fprintf(stderr, "WARNING: Zero-variance vectors detected for correlation metric. "
                                   "These vectors cannot be meaningfully correlated and may affect results.\n");
                }
                break;

            case UWOT_METRIC_COSINE:
                // Check for zero-norm vectors which cause issues with cosine similarity
                if (detect_zero_norm_vectors(data, n_obs, n_dim, UWOT_METRIC_COSINE)) {
                    fprintf(stderr, "WARNING: Zero-norm vectors detected for cosine metric. "
                                   "These vectors have no direction and may affect cosine similarity calculations.\n");
                }
                break;

            case UWOT_METRIC_EUCLIDEAN:
            case UWOT_METRIC_MANHATTAN:
            default:
                // These metrics are generally robust to different data types
                break;
        }
    }

    // Helper functions for common distance conversion patterns

    // Find k nearest neighbors from a query point to all dataset points (exact search)
    void find_knn_exact(const float* query_point, const float* dataset, int n_obs, int n_dim,
                       UwotMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index) {
        neighbors_out.clear();
        neighbors_out.reserve(n_obs - 1);

        // Compute distances from query point to all other points
        for (int j = 0; j < n_obs; j++) {
            // Skip if this is the same point (by index if provided, otherwise by pointer comparison)
            if (query_index >= 0 && j == query_index) {
                continue;
            }

            const float* dataset_point = &dataset[static_cast<size_t>(j) * static_cast<size_t>(n_dim)];

            // If no query index provided, try pointer comparison as fallback
            if (query_index < 0 && query_point == dataset_point) {
                continue;
            }

            float dist = compute_distance(query_point, dataset_point, n_dim, metric);
            neighbors_out.emplace_back(dist, j);
        }

        // Sort by distance and keep only k nearest neighbors
        std::sort(neighbors_out.begin(), neighbors_out.end());
        int max_neighbors = std::min(k_neighbors, static_cast<int>(neighbors_out.size()));
        neighbors_out.resize(max_neighbors);
    }

    // Compare two neighbor lists and calculate recall (intersection / exact_neighbors)
    float calculate_recall(const std::vector<std::pair<float, int>>& exact_neighbors,
                          const int* hnsw_neighbor_indices, int k_neighbors) {
        if (exact_neighbors.empty()) {
            return 1.0f; // Perfect recall if no exact neighbors found
        }

        // Build set of HNSW neighbor indices for fast lookup
        std::unordered_set<int> hnsw_neighbors;
        for (int k = 0; k < k_neighbors; k++) {
            int neighbor_idx = hnsw_neighbor_indices[k];
            if (neighbor_idx >= 0) {
                hnsw_neighbors.insert(neighbor_idx);
            }
        }

        // Count matches between exact and HNSW neighbors
        int matches = 0;
        for (const auto& exact_neighbor : exact_neighbors) {
            if (hnsw_neighbors.count(exact_neighbor.second) > 0) {
                matches++;
            }
        }

        return static_cast<float>(matches) / exact_neighbors.size();
    }

    // Build distance matrix with progress reporting (all-to-all distances)
    void build_distance_matrix(const float* data, int n_obs, int n_dim, UwotMetric metric,
                              float* distance_matrix, uwot_progress_callback_v2 progress_callback,
                              int current_obs, int total_obs) {
        auto start_time = std::chrono::steady_clock::now();

        for (int i = 0; i < n_obs; i++) {
            const float* point_i = &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)];

            for (int j = 0; j < n_obs; j++) {
                if (i == j) {
                    // Distance to self is zero
                    distance_matrix[static_cast<size_t>(i) * static_cast<size_t>(n_obs) + static_cast<size_t>(j)] = 0.0f;
                    continue;
                }

                const float* point_j = &data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)];
                float dist = compute_distance(point_i, point_j, n_dim, metric);

                // Store distance in symmetric matrix
                distance_matrix[static_cast<size_t>(i) * static_cast<size_t>(n_obs) + static_cast<size_t>(j)] = dist;
            }

            // Progress reporting every 5%
            if (progress_callback && i % (n_obs / 20 + 1) == 0) {
                float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                auto elapsed_sec = std::chrono::duration<double>(elapsed).count();

                // Estimate remaining time
                double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);
                int remaining_min = static_cast<int>(remaining_sec / 60.0);
                int remaining_sec_int = static_cast<int>(remaining_sec) % 60;

                char progress_msg[256];
                snprintf(progress_msg, sizeof(progress_msg),
                    "Computing distance matrix (%d/%d points) - ETA: %dm %ds",
                    i, n_obs, remaining_min, remaining_sec_int);

                progress_callback("Distance Matrix", current_obs + i, total_obs, percent, progress_msg);
            }
        }
    }

}