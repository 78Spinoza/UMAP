#include "uwot_transform.h"
#include "uwot_simple_wrapper.h"
#include "uwot_crc32.h"
#include "uwot_progress_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration for smooth_knn helper function
void apply_smooth_knn_to_point(const int* indices, const float* distances, int k, float* weights);

// Forward declaration for spectral interpolation helper function
static void interpolate_initial_embedding(
    const float* query,
    const float* train_data,
    const float* train_embedding,
    int n_train, int n_dim, int emb_dim,
    float* out_init
);

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
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        if (!model->is_fitted) {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        if (!new_data) {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        if (!embedding) {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        if (n_new_obs <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        if (n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // FAST TRANSFORM OPTIMIZATION: Backend validation (error4c.txt)
        // Transform must use the same k-NN backend as fit for rho/sigma to be valid
        if (model->has_fast_transform_data) {
            bool backend_mismatch = false;
            const char* error_msg = nullptr;

            if (model->knn_backend == UwotModel::KnnBackend::HNSW && model->force_exact_knn) {
                backend_mismatch = true;
                error_msg = "Model was fitted with HNSW but transform is using exact k-NN. "
                           "Set force_exact_knn=false during transform for HNSW models.";
            } else if (model->knn_backend == UwotModel::KnnBackend::EXACT && !model->force_exact_knn && model->original_space_index) {
                backend_mismatch = true;
                error_msg = "Model was fitted with exact k-NN but transform is using HNSW. "
                           "Set force_exact_knn=true during transform for exact k-NN models.";
            }

            if (backend_mismatch) {
                std::cerr << "BACKEND MISMATCH ERROR: " << error_msg << std::endl;
                return UWOT_ERROR_INVALID_PARAMS;
            }
        }

        // Transform operation starting
        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            // Pre-validate model before entering parallel region to avoid returns inside OpenMP
            size_t element_count = static_cast<size_t>(model->n_vertices);
            if (element_count == 0) {
                return UWOT_ERROR_INVALID_PARAMS;
            }

            // Report OpenMP thread count for transform (helps verify parallelization is working)
            #ifdef _OPENMP
            int transform_threads = omp_get_max_threads();
            // Only report if transforming multiple points (where parallelization helps)
            if (n_new_obs > 4) {
                // Report via error callback if available (similar to fit function)
                std::string thread_msg = "\n[INFO] Transform using " + std::to_string(transform_threads) +
                                        " OpenMP threads for " + std::to_string(n_new_obs) + " queries\n";
                hnsw_utils::report_hnsw_error(thread_msg);
            }
            #endif

            // Thread-safe error tracking for parallel region
            std::atomic<int> parallel_error_code(UWOT_SUCCESS);

            // Parallelize transform loop for significant speedup (4-5x for large batches)
            // Only parallelize if we have multiple queries to avoid OpenMP overhead
            #pragma omp parallel for schedule(dynamic, 4) if(n_new_obs > 4)
            for (int i = 0; i < n_new_obs; i++) {
                // Skip processing if an earlier thread encountered an error
                if (parallel_error_code.load() != UWOT_SUCCESS) {
                    continue;
                }

                // Apply EXACT same normalization as training using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }

                // CRITICAL: Only apply normalization if model was trained with it
                if (model->use_normalization) {
                    hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                        raw_point, normalized_point, 1, n_dim,
                        model->feature_means, model->feature_stds,
                        model->normalization_mode);
                } else {
                    // Skip normalization - use raw data directly
                    normalized_point = raw_point;
                }
                // Point normalization completed

                // Use normalized point directly for HNSW search (quantization removed - no benefit with HNSW)
                std::vector<float> search_point = normalized_point;

                // Note: Embedding space HNSW index is optional for basic transform (only needed for AI inference)

                // CRITICAL FIX: Exact coordinate preservation for identical training points
                // If this point is identical to a training point, return exact fitted coordinates
                // This ensures perfect consistency between fit and transform for training data

                bool found_exact_match = false;
                int exact_match_idx = -1;
                // Metric-specific tolerance for exact match detection
                const float EXACT_MATCH_TOLERANCE = (model->metric == UWOT_METRIC_COSINE) ? 1e-4f : 1e-3f;

                // Save original ef value for restoration
                size_t original_ef = 0;
                if (model->original_space_index) {
                    try {
                        original_ef = model->original_space_index->ef_;
                    } catch (const std::exception& e) {
                        // HNSW is corrupted - disable it for this operation and report error
                        hnsw_utils::report_hnsw_error("HNSW corruption detected: " + std::string(e.what()));
                        original_ef = 0;
                    } catch (...) {
                        // HNSW is corrupted - disable it for this operation and report error
                        hnsw_utils::report_hnsw_error("HNSW corruption detected: unknown error");
                        original_ef = 0;
                    }
                }

                // Check if this point is identical to any training point (fast path for training data)
                if (!model->embedding.empty() && model->n_vertices > 0 && model->original_space_index && original_ef > 0) {
                    // Use exact k-NN search with very small radius to find identical points
                    // NOTE: searchKnn is thread-safe, we just avoid setEf in parallel regions
                    try {
                        // Use current ef setting (no setEf for thread-safety in parallel transform)
                        auto exact_search = model->original_space_index->searchKnn(normalized_point.data(), 1);

                        if (!exact_search.empty()) {
                            auto pair = exact_search.top();
                            float distance = pair.first;
                            int neighbor_idx = static_cast<int>(pair.second);

                            // Apply metric-specific distance conversion
                            switch (model->metric) {
                            case UWOT_METRIC_EUCLIDEAN:
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
                    } catch (const std::exception& e) {
                        // HNSW search failed - set error and continue to skip processing
                        hnsw_utils::report_hnsw_error("HNSW search failed: " + std::string(e.what()));
                        parallel_error_code.store(UWOT_ERROR_INVALID_PARAMS);
                        continue;
                    } catch (...) {
                        // HNSW search failed - set error and continue to skip processing
                        hnsw_utils::report_hnsw_error("HNSW search failed: unknown error");
                        parallel_error_code.store(UWOT_ERROR_INVALID_PARAMS);
                        continue;
                    }
                }

                // STEP 1: Transform new data point to embedding space
                if (found_exact_match) {
                    // PERFECT: Return exact fitted coordinates for identical training point
                    size_t embed_start_idx = static_cast<size_t>(exact_match_idx) * static_cast<size_t>(model->embedding_dim);
                    size_t new_embed_start_idx = static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim);

                    // Bounds checking before accessing model embedding
                    if (embed_start_idx + model->embedding_dim <= model->embedding.size() &&
                        new_embed_start_idx + model->embedding_dim <= new_embedding.size()) {
                        const float* exact_embedding = &model->embedding[embed_start_idx];
                        for (int d = 0; d < model->embedding_dim; d++) {
                            new_embedding[new_embed_start_idx + static_cast<size_t>(d)] = exact_embedding[d];
                        }
                    } else {
                        parallel_error_code.store(UWOT_ERROR_MEMORY);
                        continue;
                    }
                } else {
                    // APPROXIMATE: Use HNSW approximation for truly new points
                    // This follows the traditional UMAP transform method to get embedding coordinates

                    // Find neighbors in ORIGINAL space (for transform weights calculation)
                std::vector<int> original_neighbors;
                std::vector<float> original_distances;
                std::vector<float> weights;
                float total_weight = 0.0f;

                // Check if force_exact_knn flag is set (use exact k-NN instead of HNSW)
                if (model->force_exact_knn) {
                    // Use saved exact k-NN data with smooth_knn fuzzy weighting
                    int actual_neighbors = std::min(model->n_neighbors, static_cast<int>(model->nn_indices.size()));

                    if (actual_neighbors > 0) {
                        // FAST TRANSFORM OPTIMIZATION: Use pre-computed rho/sigma instead of smooth_knn (20x faster)
                        std::vector<float> fuzzy_weights(actual_neighbors);

                        if (model->has_fast_transform_data && !model->rho.empty() && !model->sigma.empty()) {
                            // Fast path: Use pre-computed rho/sigma - no binary search needed!
                            for (int k = 0; k < actual_neighbors; k++) {
                                int neighbor_idx = model->nn_indices[k];
                                if (neighbor_idx >= 0 && neighbor_idx < static_cast<int>(model->rho.size())) {
                                    float rho_j = model->rho[neighbor_idx];
                                    float sigma_j = model->sigma[neighbor_idx];
                                    float distance = model->nn_distances[k];

                                    // Fast fuzzy weight computation (no binary search)
                                    float val = (distance - rho_j) / sigma_j;
                                    fuzzy_weights[k] = (val <= 0) ? 1.0f : std::exp(-val);
                                } else {
                                    fuzzy_weights[k] = 0.0f; // Invalid neighbor index
                                }
                            }
                        } else {
                            // Fallback: Use smooth_knn (shouldn't happen if model was fitted properly)
                            apply_smooth_knn_to_point(model->nn_indices.data(), model->nn_distances.data(),
                                                    actual_neighbors, fuzzy_weights.data());
                        }

                        for (int k = 0; k < actual_neighbors; k++) {
                            int neighbor_idx = model->nn_indices[k];
                            float distance = model->nn_distances[k];
                            float weight = fuzzy_weights[k];

                            // Only include neighbors with significant weights
                            if (weight > 1e-6f) {
                                original_neighbors.push_back(neighbor_idx);
                                original_distances.push_back(distance);
                                weights.push_back(weight);
                                total_weight += weight;
                            }
                        }
                    }
                } else {
                    // Use HNSW for fast neighbor search (default case - 99.9% of the time)
                    // NOTE: For thread-safety in parallel transform, we use the HNSW index's current ef
                    // (set during model fitting). No dynamic setEf() calls in parallel region.

                    try {
                        // Thread-safe: searchKnn can be called from multiple threads
                        auto original_search_result = model->original_space_index->searchKnn(search_point.data(), model->n_neighbors);

                        // Collect all neighbors first, then apply smooth_knn for consistent fuzzy weighting
                        std::vector<std::pair<int, float>> neighbors_and_distances;

                        while (!original_search_result.empty()) {
                            auto pair = original_search_result.top();
                            original_search_result.pop();

                            int neighbor_idx = static_cast<int>(pair.second);
                            float distance = pair.first;

                            // Convert HNSW distance based on metric
                            switch (model->metric) {
                            case UWOT_METRIC_EUCLIDEAN:
                                // HNSW already returns actual Euclidean distance - no conversion needed
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

                            neighbors_and_distances.emplace_back(neighbor_idx, distance);
                        }

                        // Apply smooth_knn to get fuzzy weights (consistent with fitting)
                        if (!neighbors_and_distances.empty()) {
                            // Separate indices and distances for smooth_knn
                            std::vector<int> neighbor_indices;
                            std::vector<float> neighbor_distances;

                            for (const auto& nd : neighbors_and_distances) {
                                neighbor_indices.push_back(nd.first);
                                neighbor_distances.push_back(nd.second);
                            }

                            // FAST TRANSFORM OPTIMIZATION: Use pre-computed rho/sigma instead of smooth_knn (20x faster)
                            std::vector<float> fuzzy_weights(neighbor_indices.size());

                            if (model->has_fast_transform_data && !model->rho.empty() && !model->sigma.empty()) {
                                // Fast path: Use pre-computed rho/sigma - no binary search needed!
                                for (size_t j = 0; j < neighbor_indices.size(); ++j) {
                                    int neighbor_idx = neighbor_indices[j];
                                    if (neighbor_idx >= 0 && neighbor_idx < static_cast<int>(model->rho.size())) {
                                        float rho_j = model->rho[neighbor_idx];
                                        float sigma_j = model->sigma[neighbor_idx];
                                        float distance = neighbor_distances[j];

                                        // Fast fuzzy weight computation (no binary search)
                                        float val = (distance - rho_j) / sigma_j;
                                        fuzzy_weights[j] = (val <= 0) ? 1.0f : std::exp(-val);
                                    } else {
                                        fuzzy_weights[j] = 0.0f; // Invalid neighbor index
                                    }
                                }
                            } else {
                                // Fallback: Use smooth_knn (shouldn't happen if model was fitted properly)
                                apply_smooth_knn_to_point(neighbor_indices.data(), neighbor_distances.data(),
                                                         static_cast<int>(neighbor_indices.size()),
                                                         fuzzy_weights.data());
                            }

                            // Build the final neighbor lists with fuzzy weights
                            for (size_t j = 0; j < neighbor_indices.size(); ++j) {
                                if (fuzzy_weights[j] > 1e-6f) {  // Filter out very weak connections
                                    original_neighbors.push_back(neighbor_indices[j]);
                                    original_distances.push_back(neighbor_distances[j]);
                                    weights.push_back(fuzzy_weights[j]);
                                    total_weight += fuzzy_weights[j];
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        // HNSW failed - report error and fall back to exact k-NN
                        hnsw_utils::report_hnsw_error("HNSW k-NN search failed: " + std::string(e.what()) + " - using exact k-NN fallback");
                        for (int k = 0; k < model->n_neighbors && k < static_cast<int>(model->nn_indices.size()); k++) {
                            int neighbor_idx = model->nn_indices[k];
                            float distance = model->nn_distances[k];

                            original_neighbors.push_back(neighbor_idx);
                            original_distances.push_back(distance);
                            weights.push_back(1.0f / model->n_neighbors);  // Uniform weights for fallback
                            total_weight += 1.0f / model->n_neighbors;
                        }
                    } catch (...) {
                        // HNSW failed - report error and fall back to exact k-NN
                        hnsw_utils::report_hnsw_error("HNSW k-NN search failed: unknown error - using exact k-NN fallback");
                        for (int k = 0; k < model->n_neighbors && k < static_cast<int>(model->nn_indices.size()); k++) {
                            int neighbor_idx = model->nn_indices[k];
                            float distance = model->nn_distances[k];

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
                    }
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
                    #if 0
std::cout << "[DEBUG] Transform: Model embedding array has data: " << (embedding_has_data ? "YES" : "NO") << std::endl;
#endif
                    #if 0
std::cout << "[DEBUG] Transform: Embedding array size: " << model->embedding.size() << std::endl;
#endif
                }

                // SPECTRAL INITIALIZATION OPTIMIZATION: Use spectral interpolation for better initial embedding (error4d.txt)
                // Get initial coordinates using spectral interpolation from training embeddings
                float* current_embedding = &new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)];

                if (!model->initial_embedding.empty() && !model->embedding.empty()) {
                    // Use spectral interpolation from initial training embedding
                    interpolate_initial_embedding(
                        normalized_point.data(),
                        model->embedding.data(), // Use training embeddings for interpolation
                        model->initial_embedding.data(), // Interpolate from spectral initialization
                        model->n_vertices, model->n_dim, model->embedding_dim,
                        current_embedding
                    );
                } else {
                    // Fallback: Use weighted average of neighbor embeddings (original method)
                    for (int d = 0; d < model->embedding_dim; d++) {
                        current_embedding[d] = 0.0f;
                        for (size_t k = 0; k < original_neighbors.size(); k++) {
                            size_t embed_idx = static_cast<size_t>(original_neighbors[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d);
                            if (embed_idx < model->embedding.size()) {
                                current_embedding[d] += model->embedding[embed_idx] * weights[k];
                            }
                        }
                    }
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

                    // Thread-safe: Use current ef setting, no dynamic setEf in parallel region
                    auto embedding_search_result = model->embedding_space_index->searchKnn(new_embedding_point, model->n_neighbors);

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
                    // Removed unused mean_distance variable for clean build

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
            } // End of parallel for loop

            // Check if any thread encountered an error during parallel processing
            int error_from_parallel = parallel_error_code.load();
            if (error_from_parallel != UWOT_SUCCESS) {
                return error_from_parallel;
            }

            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy
            size_t expected = static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim);
            if (new_embedding.size() < expected) {
                return UWOT_ERROR_MEMORY;
            }

            // CRITICAL: Additional bounds checking to prevent buffer overflow
            // Ensure we don't write past the output embedding buffer
            if (!embedding) {
                return UWOT_ERROR_INVALID_PARAMS;
            }

            // Safe element-wise copy with bounds checking
            for (size_t i = 0; i < expected; ++i) {
                if (i < new_embedding.size()) {
                    embedding[i] = new_embedding[i];
                } else {
                    // This should never happen with proper bounds checking above
                    return UWOT_ERROR_MEMORY;
                }
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

// SPECTRAL INITIALIZATION OPTIMIZATION: Interpolate initial embedding for better transform consistency (error4d.txt)
static void interpolate_initial_embedding(
    const float* query,
    const float* train_data,
    const float* train_embedding,
    int n_train, int n_dim, int emb_dim,
    float* out_init
) {
    // Find 5 nearest neighbors in training data
    const int k_neighbors = 5;
    std::vector<int> idx(k_neighbors);
    std::vector<float> dist(k_neighbors);

    // Simple brute-force search for nearest training points
    std::vector<std::pair<float, int>> distances;
    distances.reserve(n_train);

    for (int i = 0; i < n_train; ++i) {
        float d = 0.0f;
        const float* train_point = train_data + i * n_dim;
        for (int d_idx = 0; d_idx < n_dim; ++d_idx) {
            float diff = query[d_idx] - train_point[d_idx];
            d += diff * diff;
        }
        distances.emplace_back(d, i);
    }

    // Find k nearest neighbors
    std::partial_sort(distances.begin(), distances.begin() + k_neighbors, distances.end());

    for (int k = 0; k < k_neighbors; ++k) {
        idx[k] = distances[k].second;
        dist[k] = distances[k].first;
    }

    // Weighted average in embedding space
    std::fill(out_init, out_init + emb_dim, 0.0f);
    float sum_w = 0.0f;
    for (int k = 0; k < k_neighbors; ++k) {
        float w = 1.0f / (1e-8f + dist[k]);
        const float* src = train_embedding + idx[k] * emb_dim;
        for (int d = 0; d < emb_dim; ++d) {
            out_init[d] += w * src[d];
        }
        sum_w += w;
    }
    for (int d = 0; d < emb_dim; ++d) {
        out_init[d] /= sum_w;
    }
}

// Implementation of smooth_knn helper function for transform operations
void apply_smooth_knn_to_point(const int* indices, const float* distances, int k, float* weights) {
    if (k <= 0) return;

    // Convert distances to double for smooth_knn processing
    std::vector<double> dist_double(k);
    std::vector<double> weights_double(k);

    for (int i = 0; i < k; ++i) {
        dist_double[i] = static_cast<double>(distances[i]);
    }

    // Apply simplified smooth_knn algorithm for single point
    try {
        // Find rho (smallest non-zero distance)
        double rho = 0.0;
        for (int i = 0; i < k; ++i) {
            if (dist_double[i] > 0.0) {
                rho = dist_double[i];
                break;
            }
        }

        // Find sigma using binary search to achieve target sum
        double target_sum = static_cast<double>(k);  // Target sum is k neighbors
        double sigma = 1.0;
        constexpr int max_iter = 64;
        constexpr double tol = 1e-6;

        for (int iter = 0; iter < max_iter; ++iter) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) {
                double r = dist_double[i] - rho;
                sum += (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
            }

            if (std::abs(sum - target_sum) < tol) break;

            if (sum > target_sum) {
                sigma *= 0.5;
            } else {
                sigma *= 2.0;
            }
        }

        // Compute final fuzzy weights
        double weight_sum = 0.0;
        for (int i = 0; i < k; ++i) {
            double r = dist_double[i] - rho;
            weights_double[i] = (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
            weight_sum += weights_double[i];
        }

        // Normalize and convert back to float
        if (weight_sum > 0.0) {
            for (int i = 0; i < k; ++i) {
                weights[i] = static_cast<float>(weights_double[i] / weight_sum);
            }
        } else {
            // Fallback: uniform weights
            for (int i = 0; i < k; ++i) {
                weights[i] = 1.0f / static_cast<float>(k);
            }
        }

    } catch (...) {
        // Fallback: simple exponential decay
        double weight_sum = 0.0;
        for (int i = 0; i < k; ++i) {
            weights_double[i] = std::exp(-dist_double[i]);
            weight_sum += weights_double[i];
        }

        if (weight_sum > 0.0) {
            for (int i = 0; i < k; ++i) {
                weights[i] = static_cast<float>(weights_double[i] / weight_sum);
            }
        } else {
            // Final fallback: uniform weights
            for (int i = 0; i < k; ++i) {
                weights[i] = 1.0f / static_cast<float>(k);
            }
        }
    }
}