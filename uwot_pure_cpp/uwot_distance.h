#pragma once

#include "uwot_model.h"
#include <cmath>
#include <algorithm>

// Distance metric implementations for UMAP
namespace distance_metrics {

    // Individual distance metric functions
    float euclidean_distance(const float* a, const float* b, int dim);
    float cosine_distance(const float* a, const float* b, int dim);
    float manhattan_distance(const float* a, const float* b, int dim);
    float correlation_distance(const float* a, const float* b, int dim);
    float hamming_distance(const float* a, const float* b, int dim);

    // Unified distance computation based on metric type
    float compute_distance(const float* a, const float* b, int dim, UwotMetric metric);

    // Data validation for metric-specific requirements
    void validate_metric_data(const float* data, int n_obs, int n_dim, UwotMetric metric);

    // Zero-norm vector detection for cosine and correlation metrics
    bool detect_zero_norm_vectors(const float* data, int n_obs, int n_dim, UwotMetric metric);

    // Helper functions for common distance conversion patterns

    // Find k nearest neighbors from a query point to all dataset points (exact search)
    void find_knn_exact(const float* query_point, const float* dataset, int n_obs, int n_dim,
                       UwotMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index = -1);

    // Compare two neighbor lists and calculate recall (intersection / union)
    float calculate_recall(const std::vector<std::pair<float, int>>& exact_neighbors,
                          const int* hnsw_neighbor_indices, int k_neighbors);

    // Build distance matrix with progress reporting (all-to-all distances)
    void build_distance_matrix(const float* data, int n_obs, int n_dim, UwotMetric metric,
                              float* distance_matrix, uwot_progress_callback_v2 progress_callback = nullptr,
                              int current_obs = 0, int total_obs = 0);

}