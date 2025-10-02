#include "uwot_transform.h"
#include "uwot_simple_wrapper.h"
#include "uwot_quantization.h"
#include "uwot_crc32.h"
#include "uwot_progress_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace transform_utils {

    // TEMPORARY: Use exact working version from git commit 65abd80
    int uwot_transform_detailed(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score
    ) {
        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // Transform operation starting

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Apply EXACT same normalization as training using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }

                // Raw point data collected

                // Use stored normalization mode from training
                hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                    raw_point, normalized_point, 1, n_dim,
                    model->feature_means, model->feature_stds,
                    model->normalization_mode);

                // Point normalization completed

                // CRITICAL FIX: Apply quantization if model uses it (must match training data space)
                std::vector<float> search_point = normalized_point; // Default: use normalized point

                if (model->use_quantization && !model->pq_centroids.empty()) {
                    try {
                        // Step 1: Quantize the normalized point using saved PQ centroids
                        std::vector<uint8_t> point_codes;
                        int subspace_dim = n_dim / model->pq_m;
                        point_codes.resize(model->pq_m);

                        // Encode single point using existing centroids
                        for (int sub = 0; sub < model->pq_m; sub++) {
                            float min_dist = std::numeric_limits<float>::max();
                            uint8_t best_code = 0;

                            // Find closest centroid in this subspace
                            for (int c = 0; c < 256; c++) {
                                float dist = 0.0f;
                                for (int d = 0; d < subspace_dim; d++) {
                                    int point_idx = sub * subspace_dim + d;
                                    int centroid_idx = sub * 256 * subspace_dim + c * subspace_dim + d;
                                    float diff = normalized_point[point_idx] - model->pq_centroids[centroid_idx];
                                    dist += diff * diff;
                                }
                                if (dist < min_dist) {
                                    min_dist = dist;
                                    best_code = c;
                                }
                            }
                            point_codes[sub] = best_code;
                        }

                        // Step 2: Reconstruct quantized point for HNSW search
                        std::vector<float> quantized_point;
                        pq_utils::reconstruct_vector(point_codes, 0, model->pq_m,
                                                   model->pq_centroids, subspace_dim, quantized_point);
                        search_point = quantized_point;

                    } catch (...) {
                        // Quantization failed - fall back to normalized point
                        search_point = normalized_point;
                    }
                }

                // Note: Embedding space HNSW index is optional for basic transform (only needed for AI inference)

                // CRITICAL FIX: Exact coordinate preservation for identical training points
                // If this point is identical to a training point, return exact fitted coordinates
                // This ensures perfect consistency between fit and transform for training data

                bool found_exact_match = false;
                int exact_match_idx = -1;
                const float EXACT_MATCH_TOLERANCE = 1e-3f; // 1e-3 euclidean distance tolerance

                // Save original ef value for restoration
                size_t original_ef = model->original_space_index->ef_;

                // Check if this point is identical to any training point (fast path for training data)
                if (!model->embedding.empty() && model->n_vertices > 0) {
                    // Use exact k-NN search with very small radius to find identical points
                    // This is much faster than brute force O(n) check
                    model->original_space_index->setEf(model->n_neighbors * 8); // Higher ef for exactness
                    auto exact_search = model->original_space_index->searchKnn(normalized_point.data(), 1);
                    model->original_space_index->setEf(original_ef);

                    if (!exact_search.empty()) {
                        auto pair = exact_search.top();
                        float distance = pair.first;
                        int neighbor_idx = static_cast<int>(pair.second);

                        // Apply metric-specific distance conversion
                        switch (model->metric) {
                        case UWOT_METRIC_EUCLIDEAN:
                            // L2Space returns squared distance - convert to actual Euclidean distance
                            distance = std::sqrt(std::max(0.0f, distance));
                            break;
                        case UWOT_METRIC_COSINE:
                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                            break;
                        case UWOT_METRIC_MANHATTAN:
                            distance = std::max(0.0f, distance);
                            break;
                        default:
                            distance = std::max(0.0f, distance);
                            break;
                        }

                        if (distance < EXACT_MATCH_TOLERANCE) {
                            found_exact_match = true;
                            exact_match_idx = neighbor_idx;
                        }
                    }
                }

                // STEP 1: Transform new data point to embedding space
                if (found_exact_match) {
                    // PERFECT: Return exact fitted coordinates for identical training point
                    const float* exact_embedding = &model->embedding[static_cast<size_t>(exact_match_idx) * static_cast<size_t>(model->embedding_dim)];
                    for (int d = 0; d < model->embedding_dim; d++) {
                        new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = exact_embedding[d];
                    }
                } else {
                    // APPROXIMATE: Use HNSW approximation for truly new points
                    // This follows the traditional UMAP transform method to get embedding coordinates

                    // Find neighbors in ORIGINAL space (for transform weights calculation)
                size_t boosted_ef = static_cast<size_t>(model->n_neighbors * 32);
                boosted_ef = std::min(boosted_ef, static_cast<size_t>(400));
                model->original_space_index->setEf(std::max(original_ef, boosted_ef));

                auto original_search_result = model->original_space_index->searchKnn(search_point.data(), model->n_neighbors);
                model->original_space_index->setEf(original_ef);

                // Calculate transform weights using original space distances
                std::vector<int> original_neighbors;
                std::vector<float> original_distances;
                std::vector<float> weights;
                float total_weight = 0.0f;

                while (!original_search_result.empty()) {
                    auto pair = original_search_result.top();
                    original_search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = pair.first;

                    // Convert HNSW distance based on metric
                    switch (model->metric) {
                    case UWOT_METRIC_EUCLIDEAN:
                        // HNSW already returns actual Euclidean distance - no conversion needed
                        // distance = distance;  // No-op
                        break;
                    case UWOT_METRIC_COSINE:
                        distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                        break;
                    case UWOT_METRIC_MANHATTAN:
                        distance = std::max(0.0f, distance);
                        break;
                    default:
                        distance = std::max(0.0f, distance);
                        break;
                    }

                    original_neighbors.push_back(neighbor_idx);
                    original_distances.push_back(distance);

                    // Calculate transform weights
                    float median_dist = model->median_original_distance > 0.0f ? model->median_original_distance : model->mean_original_distance;
                    float base_bandwidth = std::max(1e-4f, 0.5f * median_dist);
                    float adaptive_bandwidth = base_bandwidth;
                    if (distance > base_bandwidth * 2.0f) {
                        adaptive_bandwidth = distance * 0.3f;
                    }
                    float weight = std::exp(-distance * distance / (2.0f * adaptive_bandwidth * adaptive_bandwidth));
                    weight = std::max(weight, 1e-6f);
                    weights.push_back(weight);
                    total_weight += weight;
                }

                // Normalize weights
                if (total_weight > 0.0f) {
                    for (float& w : weights) {
                        w /= total_weight;
                    }
                }

                // DEBUG: Check if embedding array has data
                if (i == 0) {
                    bool embedding_has_data = false;
                    size_t check_count = std::min(static_cast<size_t>(10), model->embedding.size());
                    for (size_t idx = 0; idx < check_count; idx++) {
                        if (std::abs(model->embedding[idx]) > 1e-6f) {
                            embedding_has_data = true;
                            break;
                        }
                    }
                    std::cout << "[DEBUG] Transform: Model embedding array has data: " << (embedding_has_data ? "YES" : "NO") << std::endl;
                    std::cout << "[DEBUG] Transform: Embedding array size: " << model->embedding.size() << std::endl;
                }

                // Calculate new embedding coordinates as weighted average of neighbor embeddings
                // TEMPORARY: Use embedding array directly until HNSW extraction is fully reliable
                for (int d = 0; d < model->embedding_dim; d++) {
                    float coord = 0.0f;
                    for (size_t k = 0; k < original_neighbors.size(); k++) {
                        size_t embed_idx = static_cast<size_t>(original_neighbors[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d);
                        if (embed_idx < model->embedding.size()) {
                            coord += model->embedding[embed_idx] * weights[k];
                        } else {
                            std::cout << "[DEBUG] Transform: Embedding index out of bounds: " << embed_idx << " >= " << model->embedding.size() << std::endl;
                        }
                    }
                    new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                }

                // STEP 2: CRITICAL FOR AI - Find neighbors in EMBEDDING space for AI inference (optional)
                // This answers: "Which learned patterns are similar to this new data?"
                // NOTE: This is only used for AI inference/outlier detection, NOT for basic transform

                std::vector<int> embedding_neighbors;
                std::vector<float> embedding_distances;

                // Only perform embedding space search if index exists AND caller requested detailed info
                if (model->embedding_space_index && (nn_indices || confidence_score || outlier_level)) {
                    // Embedding space search is only for AI inference features, not basic transform
                    const float* new_embedding_point = &new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)];

                    size_t embedding_ef = model->embedding_space_index->ef_;
                    size_t boosted_embedding_ef = static_cast<size_t>(model->n_neighbors * 32);
                    boosted_embedding_ef = std::min(boosted_embedding_ef, static_cast<size_t>(400));
                    model->embedding_space_index->setEf(std::max(embedding_ef, boosted_embedding_ef));

                    auto embedding_search_result = model->embedding_space_index->searchKnn(new_embedding_point, model->n_neighbors);
                    model->embedding_space_index->setEf(embedding_ef);

                    // Extract embedding space neighbors and distances for AI inference
                    while (!embedding_search_result.empty()) {
                        auto pair = embedding_search_result.top();
                        embedding_search_result.pop();

                        int neighbor_idx = static_cast<int>(pair.second);
                        float distance = std::sqrt(std::max(0.0f, pair.first)); // L2Space returns squared distance

                        embedding_neighbors.push_back(neighbor_idx);
                        embedding_distances.push_back(distance);
                    }
                }

                // Store EMBEDDING SPACE neighbor information (this is what AI needs)
                if (nn_indices && nn_distances) {
                    for (size_t k = 0; k < embedding_neighbors.size() && k < static_cast<size_t>(model->n_neighbors); k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = embedding_neighbors[k];
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = embedding_distances[k];
                    }
                }

                // Calculate AI inference safety metrics using EMBEDDING space distances
                if (!embedding_distances.empty()) {
                    float min_distance = *std::min_element(embedding_distances.begin(), embedding_distances.end());
                    float mean_distance = std::accumulate(embedding_distances.begin(), embedding_distances.end(), 0.0f) / embedding_distances.size();

                    // AI confidence score based on embedding space neighbor distances
                    if (confidence_score) {
                        const float EPS = 1e-8f;
                        float denom = std::max(EPS, model->p95_embedding_distance - model->min_embedding_distance);
                        float normalized_dist = (min_distance - model->min_embedding_distance) / denom;
                        confidence_score[i] = std::clamp(1.0f - normalized_dist, 0.0f, 1.0f);
                    }

                    // AI outlier level assessment based on embedding space
                    if (outlier_level) {
                        if (min_distance <= model->p95_embedding_distance) {
                            outlier_level[i] = 0; // Normal - AI has seen similar patterns
                        }
                        else if (min_distance <= model->p99_embedding_distance) {
                            outlier_level[i] = 1; // Unusual but acceptable
                        }
                        else if (min_distance <= model->mild_embedding_outlier_threshold) {
                            outlier_level[i] = 2; // Mild outlier - AI extrapolating
                        }
                        else if (min_distance <= model->extreme_embedding_outlier_threshold) {
                            outlier_level[i] = 3; // Extreme outlier - AI uncertain
                        }
                        else {
                            outlier_level[i] = 4; // No man's land - AI should not trust
                        }
                    }

                    // Percentile rank in embedding space
                    if (percentile_rank) {
                        const float EPS = 1e-8f;
                        if (min_distance <= model->min_embedding_distance) {
                            percentile_rank[i] = 0.0f;
                        }
                        else if (min_distance >= model->p99_embedding_distance) {
                            percentile_rank[i] = 99.0f;
                        }
                        else {
                            float p95_range = std::max(EPS, model->p95_embedding_distance - model->min_embedding_distance);
                            if (min_distance <= model->p95_embedding_distance) {
                                percentile_rank[i] = 95.0f * (min_distance - model->min_embedding_distance) / p95_range;
                            }
                            else {
                                float p99_range = std::max(EPS, model->p99_embedding_distance - model->p95_embedding_distance);
                                percentile_rank[i] = 95.0f + 4.0f * (min_distance - model->p95_embedding_distance) / p99_range;
                            }
                        }
                    }

                    // Z-score in embedding space
                    if (z_score) {
                        const float EPS = 1e-8f;
                        float denom_z = std::max(EPS, model->std_embedding_distance);
                        z_score[i] = (min_distance - model->mean_embedding_distance) / denom_z;
                    }
                }
                } // Close the else block for approximate transformation
            }

            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy
            size_t expected = static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim);
            if (new_embedding.size() < expected) {
                return UWOT_ERROR_MEMORY;
            }
            for (size_t i = 0; i < expected; ++i) {
                embedding[i] = new_embedding[i];
            }

            return UWOT_SUCCESS;

        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            return UWOT_ERROR_MEMORY;
        }
    }

    // Minimal uwot_transform that just calls uwot_transform_detailed
    int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    ) {
        return transform_utils::uwot_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}