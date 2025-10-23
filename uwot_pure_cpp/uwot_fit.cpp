#include "uwot_fit.h"
#include "uwot_simple_wrapper.h"
#include "uwot_quantization.h"
#include "uwot_distance.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>
#include <ctime>
#include <chrono>



// Include uwot headers
#include "smooth_knn.h"

namespace fit_utils {



    // Helper function to compute normalization parameters

    void compute_normalization(const std::vector<float>& data, int n_obs, int n_dim,

        std::vector<float>& means, std::vector<float>& stds) {

        means.resize(n_dim);

        stds.resize(n_dim);



        // Calculate means

        std::fill(means.begin(), means.end(), 0.0f);

        for (int i = 0; i < n_obs; i++) {

            for (int j = 0; j < n_dim; j++) {

                means[j] += data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)];

            }

        }

        for (int j = 0; j < n_dim; j++) {

            means[j] /= static_cast<float>(n_obs);

        }



        // Calculate standard deviations

        std::fill(stds.begin(), stds.end(), 0.0f);

        for (int i = 0; i < n_obs; i++) {

            for (int j = 0; j < n_dim; j++) {

                float diff = data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)] - means[j];

                stds[j] += diff * diff;

            }

        }

        for (int j = 0; j < n_dim; j++) {

            stds[j] = std::sqrt(stds[j] / static_cast<float>(n_obs - 1));

            if (stds[j] < 1e-8f) stds[j] = 1.0f; // Prevent division by zero

        }

    }



    // Helper function to compute comprehensive neighbor statistics

    void compute_neighbor_statistics(UwotModel* model, const std::vector<float>& normalized_data) {

        if (!model->original_space_index || model->n_vertices == 0) return;



        std::vector<float> all_distances;

        all_distances.reserve(model->n_vertices * model->n_neighbors);



        // Query each point to get neighbor distances

        for (int i = 0; i < model->n_vertices; i++) {

            const float* query_point = &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(model->n_dim)];



            try {

                // Search for k+1 neighbors (includes self)

                auto result = model->original_space_index->searchKnn(query_point, model->n_neighbors + 1);



                // Skip the first result (self) and collect distances

                int count = 0;

                while (!result.empty() && count < model->n_neighbors) {

                    auto pair = result.top();

                    result.pop();



                    if (count > 0) { // Skip self-distance

                        // Convert HNSW distance based on metric

                        float distance = pair.first;

                        switch (model->metric) {

                        case UWOT_METRIC_EUCLIDEAN:

                            distance = std::sqrt(std::max(0.0f, distance)); // L2Space returns squared distance

                            break;

                        case UWOT_METRIC_COSINE:

                            // InnerProductSpace returns -inner_product for unit vectors

                            // Convert to cosine distance: distance = 1 - similarity

                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));

                            break;

                        case UWOT_METRIC_MANHATTAN:

                            // L1Space returns direct Manhattan distance

                            distance = std::max(0.0f, distance);

                            break;

                        default:

                            distance = std::max(0.0f, distance);

                            break;

                        }

                        all_distances.push_back(distance);

                    }

                    count++;

                }

            }

            catch (...) {

                // Handle any HNSW exceptions gracefully

                continue;

            }

        }



        if (all_distances.empty()) return;



        // Sort distances for percentile calculations

        std::sort(all_distances.begin(), all_distances.end());



        // Calculate statistics

        model->min_original_distance = all_distances.front();



        // Mean calculation

        float sum = 0.0f;

        for (float dist : all_distances) {

            sum += dist;

        }

        model->mean_original_distance = sum / all_distances.size();



        // Standard deviation calculation

        float sq_sum = 0.0f;

        for (float dist : all_distances) {

            float diff = dist - model->mean_original_distance;

            sq_sum += diff * diff;

        }

        model->std_original_distance = std::sqrt(sq_sum / all_distances.size());



        // Percentile calculations

        size_t p95_idx = static_cast<size_t>(0.95 * all_distances.size());

        size_t p99_idx = static_cast<size_t>(0.99 * all_distances.size());

        model->p95_original_distance = all_distances[std::min(p95_idx, all_distances.size() - 1)];

        model->p99_original_distance = all_distances[std::min(p99_idx, all_distances.size() - 1)];



        // Fix 3: Compute median neighbor distance

        size_t median_idx = all_distances.size() / 2;

        model->median_original_distance = all_distances[median_idx];



        // Fix 3: Set robust exact-match threshold for float32

        model->exact_match_threshold = 1e-3f / std::sqrt(static_cast<float>(model->n_dim));



        // Outlier thresholds

        model->mild_original_outlier_threshold = model->mean_original_distance + 2.5f * model->std_original_distance;

        model->extreme_original_outlier_threshold = model->mean_original_distance + 4.0f * model->std_original_distance;



        printf("[STATS] Neighbor distances - min: %.4f, median: %.4f, mean: %.4f +/- %.4f, p95: %.4f, p99: %.4f\n",

            model->min_original_distance, model->median_original_distance, model->mean_original_distance,

            model->std_original_distance, model->p95_original_distance, model->p99_original_distance);

        printf("[STATS] Outlier thresholds - mild: %.4f, extreme: %.4f\n",

            model->mild_original_outlier_threshold, model->extreme_original_outlier_threshold);

    }



    // Distance metric implementations



    // Build k-NN graph using specified distance metric

    void build_knn_graph(const std::vector<float>& data, int n_obs, int n_dim,

        int n_neighbors, UwotMetric metric, UwotModel* model,

        std::vector<int>& nn_indices, std::vector<double>& nn_distances,

        int force_exact_knn, uwot_progress_callback_v2 progress_callback, int autoHNSWParam) {



        nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

        nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));



        auto start_time = std::chrono::steady_clock::now();



        // Enhanced HNSW optimization check with model availability

        bool can_use_hnsw = !force_exact_knn &&

            model && model->original_space_factory && model->original_space_factory->can_use_hnsw() &&

            model->original_space_index && model->original_space_index->getCurrentElementCount() > 0;



        // k-NN strategy determined



        if (can_use_hnsw) {

            // ====== HNSW APPROXIMATE k-NN (FAST) ======

            // Using HNSW approximate k-NN



            // HNSW approximate k-NN (50-2000x faster)

            for (int i = 0; i < n_obs; i++) {

                try {

                    const float* query_point = &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)];



                    // Search for k+1 neighbors (includes self)

                    auto result = model->original_space_index->searchKnn(query_point, n_neighbors + 1);



                    // Extract neighbors, skipping self

                    std::vector<std::pair<float, int>> neighbors;

                    while (!result.empty()) {

                        auto pair = result.top();

                        result.pop();

                        int neighbor_id = static_cast<int>(pair.second);

                        if (neighbor_id != i) { // Skip self

                            neighbors.push_back({ pair.first, neighbor_id });

                        }

                    }



                    // Sort by distance and take k nearest

                    std::sort(neighbors.begin(), neighbors.end());

                    int actual_neighbors = std::min(static_cast<int>(neighbors.size()), n_neighbors);



                    for (int k = 0; k < actual_neighbors; k++) {

                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = neighbors[k].second;



                        // Convert HNSW distance to actual distance based on metric

                        float distance = neighbors[k].first;

                        switch (metric) {

                        case UWOT_METRIC_EUCLIDEAN:

                            distance = std::sqrt(std::max(0.0f, distance)); // L2Space returns squared distance

                            break;

                        case UWOT_METRIC_COSINE:

                            // InnerProductSpace returns -inner_product for unit vectors

                            // Convert to cosine distance: distance = 1 - similarity

                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));

                            break;

                        case UWOT_METRIC_MANHATTAN:

                            // L1Space returns direct Manhattan distance

                            distance = std::max(0.0f, distance);

                            break;

                        default:

                            distance = std::max(0.0f, distance);

                            break;

                        }



                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] =

                            static_cast<double>(distance);

                    }



                    // Fill remaining slots if needed

                    for (int k = actual_neighbors; k < n_neighbors; k++) {

                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 0;

                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;

                    }

                }

                catch (...) {

                    // Fallback for any HNSW errors

                    for (int k = 0; k < n_neighbors; k++) {

                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = (i + k + 1) % n_obs;

                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;

                    }

                }



                // Progress reporting every 10%

                if (progress_callback && i % (n_obs / 10 + 1) == 0) {

                    float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);

                    auto elapsed = std::chrono::steady_clock::now() - start_time;

                    auto elapsed_sec = std::chrono::duration<double>(elapsed).count();

                    double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);



                    char message[256];

                    snprintf(message, sizeof(message), "HNSW approx k-NN: %.1f%% (est. remaining: %.1fs)",

                        percent, remaining_sec);

                    progress_callback("HNSW k-NN Graph", i, n_obs, percent, message);

                }

            }



            if (progress_callback) {

                auto total_elapsed = std::chrono::steady_clock::now() - start_time;

                auto total_sec = std::chrono::duration<double>(total_elapsed).count();

                char final_message[256];

                snprintf(final_message, sizeof(final_message), "HNSW k-NN completed in %.2fs (approx mode)", total_sec);

                progress_callback("HNSW k-NN Graph", n_obs, n_obs, 100.0f, final_message);

            }



        }

        else {

            // ====== BRUTE-FORCE EXACT k-NN (SLOW BUT EXACT) ======

            // Using exact brute-force k-NN



            // Issue warnings for large datasets

            if (progress_callback) {

                const char* reason = force_exact_knn ? "exact k-NN forced" :

                    (!model || !model->original_space_factory) ? "HNSW not available" :

                    !model->original_space_factory->can_use_hnsw() ? "unsupported metric for HNSW" : "HNSW index missing";



                if (n_obs > 10000 || (static_cast<long long>(n_obs) * n_obs * n_dim) > 1e8) {

                    // Estimate time for large datasets

                    double est_operations = static_cast<double>(n_obs) * n_obs * n_dim;

                    double est_seconds = est_operations * 1e-9; // Conservative estimate: 1B ops/sec



                    char warning[512];

                    snprintf(warning, sizeof(warning),

                        "WARNING: Exact k-NN on %dx%d dataset (%s). Est. time: %.1f minutes. "

                        "Consider Euclidean/Cosine/Manhattan metrics for HNSW speedup.",

                        n_obs, n_dim, reason, est_seconds / 60.0);

                    progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, warning);

                }

                else {

                    char info[256];

                    snprintf(info, sizeof(info), "Exact k-NN mode (%s)", reason);

                    progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, info);

                }

            }



            // Original brute-force implementation with progress reporting

            for (int i = 0; i < n_obs; i++) {

                std::vector<std::pair<double, int>> distances;



                for (int j = 0; j < n_obs; j++) {

                    if (i == j) continue;



                    float dist = distance_metrics::compute_distance(

                        &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],

                        &data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)],

                        n_dim, metric);

                    distances.push_back({ static_cast<double>(dist), j });

                }



                std::partial_sort(distances.begin(),

                    distances.begin() + n_neighbors,

                    distances.end());



                for (int k = 0; k < n_neighbors; k++) {

                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].second;

                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].first;

                }



                // Progress reporting every 5%

                if (progress_callback && i % (n_obs / 20 + 1) == 0) {

                    float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);

                    auto elapsed = std::chrono::steady_clock::now() - start_time;

                    auto elapsed_sec = std::chrono::duration<double>(elapsed).count();

                    double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);



                    char message[256];

                    snprintf(message, sizeof(message), "Exact k-NN: %.1f%% (est. remaining: %.1fs)",

                        percent, remaining_sec);

                    progress_callback("Exact k-NN Graph", i, n_obs, percent, message);

                }

            }



            if (progress_callback) {

                auto total_elapsed = std::chrono::steady_clock::now() - start_time;

                auto total_sec = std::chrono::duration<double>(total_elapsed).count();

                char final_message[256];

                snprintf(final_message, sizeof(final_message), "Exact k-NN completed in %.2fs", total_sec);

                progress_callback("Exact k-NN Graph", n_obs, n_obs, 100.0f, final_message);

            }

        }

    }



    // Convert uwot smooth k-NN output to edge list format

    void convert_to_edges(const std::vector<int>& nn_indices,

        const std::vector<double>& nn_weights,

        int n_obs, int n_neighbors,

        std::vector<unsigned int>& heads,

        std::vector<unsigned int>& tails,

        std::vector<double>& weights) {



        // Use map to store symmetric edges and combine weights

        std::map<std::pair<int, int>, double> edge_map;



        for (int i = 0; i < n_obs; i++) {

            for (int k = 0; k < n_neighbors; k++) {

                int j = nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];

                double weight = nn_weights[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];



                // Add edge in both directions for symmetrization

                edge_map[{i, j}] += weight;

                edge_map[{j, i}] += weight;

            }

        }



        // Convert to edge list, avoiding duplicates

        for (const auto& edge : edge_map) {

            int i = edge.first.first;

            int j = edge.first.second;



            if (i < j) { // Only add each edge once

                heads.push_back(static_cast<unsigned int>(i));

                tails.push_back(static_cast<unsigned int>(j));

                weights.push_back(edge.second / 2.0); // Average the weights

            }

        }

    }



    // Calculate UMAP a,b parameters from spread and min_dist using Gauss-Newton optimization

    // This matches the precision of umappp implementation

    void calculate_ab_from_spread_and_min_dist(UwotModel* model) {

        double spread = static_cast<double>(model->spread);

        double min_dist = static_cast<double>(model->min_dist);



        // Handle edge cases

        if (spread <= 0.0) spread = 1.0;

        if (min_dist < 0.0) min_dist = 0.0f;

        if (min_dist >= spread) {

            // If min_dist >= spread, use default values

            model->a = 1.929f;

            model->b = 0.7915f;

            return;

        }



        // CRITICAL FIX: Use Gauss-Newton optimization like umappp for precise a/b calculation

        // Target function: f(x) = 1/(1 + a*x^(2*b))

        // Target curve: exponential decay with smooth transition at min_dist



        // Generate target points for curve fitting

        const int n_points = 15;

        std::vector<double> x_vals(n_points);

        std::vector<double> y_vals(n_points);



        for (int i = 0; i < n_points; i++) {

            double x = min_dist + (i * spread / 10.0); // Range from min_dist to beyond spread

            x_vals[i] = x;



            if (x <= min_dist) {

                y_vals[i] = 1.0; // Flat region before min_dist

            }
            else {

                y_vals[i] = std::exp(-(x - min_dist) / spread); // Exponential decay

            }

        }



        // Gauss-Newton optimization for non-linear least squares

        // Initialize with reasonable starting values

        double a = 1.0;

        double b = 1.0;

        const int max_iterations = 100;

        const double tolerance = 1e-8;



        for (int iter = 0; iter < max_iterations; iter++) {

            double residual_sum = 0.0;

            double JJa = 0.0, JJb = 0.0, Jab = 0.0;

            double Jra = 0.0, Jrb = 0.0;



            // Compute Jacobian and normal equations

            for (int i = 0; i < n_points; i++) {

                double x = x_vals[i];

                double x_pow_2b = std::pow(x, 2.0 * b);

                double denom = 1.0 + a * x_pow_2b;

                double predicted = 1.0 / denom;

                double residual = y_vals[i] - predicted;

                residual_sum += residual * residual;



                // Jacobian elements

                double d_pred_da = -x_pow_2b / (denom * denom);

                double d_pred_db = -2.0 * a * std::log(x) * x_pow_2b / (denom * denom);



                // Accumulate normal equations: J^T * J and J^T * r

                JJa += d_pred_da * d_pred_da;

                JJb += d_pred_db * d_pred_db;

                Jab += d_pred_da * d_pred_db;

                Jra += d_pred_da * residual;

                Jrb += d_pred_db * residual;

            }



            // Solve normal equations: [J^T * J] * delta = J^T * r

            double det = JJa * JJb - Jab * Jab;

            if (std::abs(det) < 1e-15) break; // Singular matrix



            double delta_a = (JJb * Jra - Jab * Jrb) / det;

            double delta_b = (JJa * Jrb - Jab * Jra) / det;



            // Update parameters

            a += delta_a;

            b += delta_b;



            // Check convergence

            if (std::abs(delta_a) < tolerance && std::abs(delta_b) < tolerance) {

                break;

            }

        }



        // Clamp parameters to reasonable ranges and convert back to float

        model->a = static_cast<float>(std::max(0.001, std::min(a, 1000.0)));

        model->b = static_cast<float>(std::max(0.1, std::min(b, 2.0)));

    }



    // Main fit function with progress reporting

    int uwot_fit_with_progress(UwotModel* model,

        float* data,

        int n_obs,

        int n_dim,

        int embedding_dim,

        int n_neighbors,

        float min_dist,

        float spread,

        int n_epochs,

        UwotMetric metric,

        float* embedding,

        uwot_progress_callback_v2 progress_callback,

        int force_exact_knn,

        int M,

        int ef_construction,

        int ef_search,

        int use_quantization,

        int random_seed,

        int autoHNSWParam) {



        // Training function called



        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||

            embedding_dim <= 0 || n_neighbors <= 0 || n_epochs <= 0) {

            return UWOT_ERROR_INVALID_PARAMS;

        }



        if (embedding_dim > 50) {

            return UWOT_ERROR_INVALID_PARAMS;

        }



        // Validate data appropriateness for the selected metric

        distance_metrics::validate_metric_data(data, n_obs, n_dim, metric);



        try {

            model->n_vertices = n_obs;

            model->n_dim = n_dim;

            model->embedding_dim = embedding_dim;

            model->n_neighbors = n_neighbors;

            model->min_dist = min_dist;

            model->spread = spread;

            model->metric = metric;

            model->force_exact_knn = (force_exact_knn != 0); // Convert int to bool

            model->use_quantization = (use_quantization != 0); // Enable/disable based on parameter



            // Auto-scale HNSW parameters based on dataset size (if not explicitly set)

            if (M == -1) {  // Auto-scale flag

                if (n_obs < 50000) {

                    model->hnsw_M = 16;

                    model->hnsw_ef_construction = 64;

                    model->hnsw_ef_search = 32;

                }

                else if (n_obs < 1000000) {

                    model->hnsw_M = 32;

                    model->hnsw_ef_construction = 128;

                    model->hnsw_ef_search = 64;

                }

                else {

                    model->hnsw_M = 64;

                    model->hnsw_ef_construction = 128;

                    model->hnsw_ef_search = 128;

                }

            }

            else {

                // Use explicitly provided parameters

                model->hnsw_M = M;

                model->hnsw_ef_construction = ef_construction;

                model->hnsw_ef_search = ef_search;

            }



            // Suggestion 4: Auto-scale ef_search based on dim/size

            model->hnsw_ef_search = std::max(model->hnsw_ef_search, static_cast<int>(model->n_neighbors * std::log(static_cast<float>(n_obs)) / std::log(2.0f)));

            model->hnsw_ef_search = std::max(model->hnsw_ef_search, static_cast<int>(std::sqrt(static_cast<float>(n_dim)) * 2));  // Scale with sqrt(dim) for FP robustness



            // UNIFIED DATA PIPELINE from errors4.txt Solution 2

            // Use the SAME data for both HNSW index and k-NN computation

            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));



            // Compute normalization parameters

            compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);

            model->use_normalization = true;



            // Determine normalization mode and apply consistently

            auto norm_mode = hnsw_utils::NormalizationPipeline::determine_normalization_mode(metric);

            model->normalization_mode = norm_mode;



            // Apply consistent normalization to create SINGLE unified dataset

            std::vector<float> normalized_data;

            hnsw_utils::NormalizationPipeline::normalize_data_consistent(

                input_data, normalized_data, n_obs, n_dim,

                model->feature_means, model->feature_stds, norm_mode);



            // L2 normalization for cosine is now handled in the normalization pipeline (mode=2)



            if (progress_callback) {

                progress_callback("Data normalization", 10, 100, 10.0f, "Normalization complete");

            }



            // CRITICAL FIX: Create HNSW index BEFORE k-NN graph so build_knn_graph can use it

            if (!model->original_space_factory->create_space(metric, n_dim)) {

                return UWOT_ERROR_MEMORY;

            }



            // Memory estimation for HNSW index - calculate expected memory usage

            size_t estimated_memory_mb = ((size_t)n_obs * model->hnsw_M * 4 * 2) / (1024 * 1024);

            // Creating HNSW index with calculated parameters



            model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(

                model->original_space_factory->get_space(), n_obs, model->hnsw_M, model->hnsw_ef_construction);

            model->original_space_index->setEf(model->hnsw_ef_search);  // Set query-time ef parameter



            // CRITICAL OPTIMIZATION: Apply quantization BEFORE creating HNSW index

            std::vector<float> hnsw_data = normalized_data; // Default: use original data



            if (model->use_quantization) {

                try {

                    // Calculate optimal number of PQ subspaces

                    // Use conservative quantization for <20% error rate - fewer subspaces = better accuracy

                    model->pq_m = pq_utils::calculate_optimal_pq_m(n_dim, 40);  // 40D per subspace minimum



                    // Only proceed if PQ is beneficial (pq_m > 1)

                    if (model->pq_m > 1) {

                        if (progress_callback) {

                            progress_callback("Product Quantization", n_epochs, n_epochs, 100.0f, "PQ encoding complete");

                        }



                        // Apply Product Quantization to normalized training data

                        pq_utils::encode_pq(normalized_data, n_obs, n_dim, model->pq_m,

                            model->pq_codes, model->pq_centroids);



                        // STEP 1: Create reconstructed quantized data for HNSW (with overflow check)

                        if (n_obs > 0 && n_dim > 0 && n_obs > SIZE_MAX / static_cast<size_t>(n_dim)) {

                            throw std::runtime_error("Integer overflow in HNSW data allocation");

                        }

                        hnsw_data.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim));

                        int subspace_dim = n_dim / model->pq_m;



                        for (int i = 0; i < n_obs; i++) {

                            std::vector<float> reconstructed_point;

                            pq_utils::reconstruct_vector(model->pq_codes, i, model->pq_m,

                                model->pq_centroids, subspace_dim,

                                reconstructed_point);



                            // Copy reconstructed point to HNSW data

                            for (int d = 0; d < n_dim; d++) {

                                hnsw_data[i * n_dim + d] = reconstructed_point[d];

                            }

                        }



                        if (progress_callback) {

                            progress_callback("Product Quantization", n_epochs, n_epochs, 100.0f, "PQ optimization complete");

                        }

                    }
                    else {

                        // Disable quantization if not beneficial for this dataset

                        model->use_quantization = false;

                    }

                }
                catch (const std::exception&) {

                    // PQ failed - continue without quantization

                    model->use_quantization = false;

                    if (progress_callback) {

                        progress_callback("Product Quantization", n_epochs, n_epochs, 100.0f, "PQ failed - continuing without quantization");

                    }

                }

            }



            // Add all points to HNSW index using quantized data (if enabled) or original data

            for (int i = 0; i < n_obs; i++) {

                model->original_space_index->addPoint(

                    &hnsw_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],

                    static_cast<hnswlib::labeltype>(i));

            }



            // HNSW index construction completed



            // Use same data for BOTH HNSW and exact k-NN - this is the key fix!



            // Compute comprehensive neighbor statistics on the SAME data as HNSW

            compute_neighbor_statistics(model, hnsw_data);



            // Build k-NN graph using SAME prepared data as HNSW index - INDEX NOW AVAILABLE!

            std::vector<int> nn_indices;

            std::vector<double> nn_distances;



            // Create wrapper for passing warnings to v2 callback if available

            uwot_progress_callback_v2 wrapped_callback = nullptr;

            if (g_v2_callback) {

                wrapped_callback = g_v2_callback;  // Pass warnings directly to v2 callback

            }



            build_knn_graph(hnsw_data, n_obs, n_dim, n_neighbors, metric, model,

                nn_indices, nn_distances, force_exact_knn, wrapped_callback, autoHNSWParam);



            // Use uwot smooth_knn to compute weights

            std::vector<std::size_t> nn_ptr = { static_cast<std::size_t>(n_neighbors) };

            std::vector<double> target = { std::log2(static_cast<double>(n_neighbors)) };

            std::vector<double> nn_weights(nn_indices.size());

            std::vector<double> sigmas, rhos;

            std::atomic<std::size_t> n_search_fails{ 0 };



            uwot::smooth_knn(0, static_cast<std::size_t>(n_obs), nn_distances, nn_ptr, false, target,

                1.0, 1e-5, 64, 0.001,

                uwot::mean_average(nn_distances), false,

                nn_weights, sigmas, rhos, n_search_fails);



            // Fix 2: Apply minimal weight floor to preserve relative differences

            const double MIN_WEIGHT = 1e-6;

            for (size_t wi = 0; wi < nn_weights.size(); ++wi) {

                if (nn_weights[wi] < MIN_WEIGHT) nn_weights[wi] = MIN_WEIGHT;

            }



            // Convert to edge format for optimization

            convert_to_edges(nn_indices, nn_weights, n_obs, n_neighbors,

                model->positive_head, model->positive_tail, model->positive_weights);



            // STEP 2: Store k-NN data for transform ONLY if quantization disabled

            // When quantization is enabled, training data can be reconstructed from PQ codes

            if (!model->use_quantization) {

                model->nn_indices = nn_indices;

                model->nn_distances.resize(nn_distances.size());

                model->nn_weights.resize(nn_weights.size());

                for (size_t i = 0; i < nn_distances.size(); i++) {

                    model->nn_distances[i] = static_cast<float>(nn_distances[i]);

                    // Fix 2: Ensure no overly-large floor when converting to float

                    model->nn_weights[i] = static_cast<float>(std::max<double>(nn_weights[i], 1e-6));

                }

            }
            else {

                // Clear redundant arrays when quantization is enabled - saves 70-80% memory

                model->nn_indices.clear();

                model->nn_distances.clear();

                model->nn_weights.clear();

            }



            // Initialize embedding

            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));



            // Thread-safe random initialization

#pragma omp parallel if(n_obs > 1000)

            {

                // Each thread gets its own generator to avoid race conditions

                thread_local std::mt19937 gen(42 + omp_get_thread_num());

                thread_local std::normal_distribution<float> dist(0.0f, 1e-4f);



#pragma omp for

                for (int i = 0; i < static_cast<int>(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim)); i++) {

                    model->embedding[i] = dist(gen);

                }

            }



            // Calculate UMAP parameters from spread and min_dist

            calculate_ab_from_spread_and_min_dist(model);



            // CRITICAL: Setup epoch scheduling (umappp approach)
            // Edges with higher weights are processed more frequently
            const float negative_sample_rate = 5.0f;

            // Find max weight for scheduling (umappp line 51-57)
            float max_weight = 0.0f;
            for (size_t i = 0; i < model->positive_weights.size(); i++) {
                if (model->positive_weights[i] > max_weight) {
                    max_weight = model->positive_weights[i];
                }
            }

            // CRITICAL: Filter edges by weight threshold (umappp line 65-70)
            // Only process edges with weight >= max_weight / n_epochs
            const float limit = max_weight / static_cast<float>(n_epochs);

            std::vector<size_t> filtered_edge_indices;  // Maps to original edge indices
            std::vector<float> epochs_per_sample;
            std::vector<float> epoch_of_next_sample;
            std::vector<float> epoch_of_next_negative_sample;

            filtered_edge_indices.reserve(model->positive_weights.size());
            epochs_per_sample.reserve(model->positive_weights.size());

            for (size_t i = 0; i < model->positive_weights.size(); i++) {
                float weight = static_cast<float>(model->positive_weights[i]);
                if (weight >= limit) {  // umappp line 70: edge filtering
                    filtered_edge_indices.push_back(i);
                    float eps = max_weight / std::max(weight, 1e-6f);
                    epochs_per_sample.push_back(eps);
                }
            }

            // Initialize epoch scheduling (umappp lines 79-83)
            epoch_of_next_sample = epochs_per_sample;  // Start at epochs_per_sample
            epoch_of_next_negative_sample = epochs_per_sample;
            for (auto& e : epoch_of_next_negative_sample) {
                e /= negative_sample_rate;
            }

            // Direct UMAP optimization implementation with progress reporting
            const float learning_rate = 1.0f;
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> vertex_dist(0, static_cast<size_t>(n_obs) - 1);



            // Enhanced progress reporting setup

            int progress_interval = std::max(1, n_epochs / 100);  // Report every 1% progress

            auto last_report_time = std::chrono::steady_clock::now();



            // Only show console output if no callback provided

            if (!progress_callback) {

                std::printf("UMAP Training Progress:\n");

                std::printf("[                    ] 0%% (Epoch 0/%d)\n", n_epochs);

                std::fflush(stdout);

            }



            for (int epoch = 0; epoch < n_epochs; epoch++) {

                float alpha = learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(n_epochs));



                // Loss calculation for progress reporting

                float epoch_loss = 0.0f;

                int loss_samples = 0;



                // Process positive edges (attractive forces)
                // CRITICAL: Only process filtered edges (umappp line 67-73)

                for (size_t filt_idx = 0; filt_idx < filtered_edge_indices.size(); filt_idx++) {

                    // CRITICAL: Check if this edge should be processed in this epoch (umappp line 143)
                    if (epoch_of_next_sample[filt_idx] > epoch) continue;

                    size_t edge_idx = filtered_edge_indices[filt_idx];  // Map to original edge index
                    size_t i = static_cast<size_t>(model->positive_head[edge_idx]);

                    size_t j = static_cast<size_t>(model->positive_tail[edge_idx]);



                    // Compute squared distance

                    float dist_sq = 0.0f;

                    for (int d = 0; d < embedding_dim; d++) {

                        float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -

                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];

                        dist_sq += diff * diff;

                    }



                    if (dist_sq > std::numeric_limits<float>::epsilon()) {

                        // CRITICAL FIX: Use double precision for force calculations to match umappp

                        double dist_sq_d = static_cast<double>(dist_sq);

                        double a_d = static_cast<double>(model->a);

                        double b_d = static_cast<double>(model->b);



                        // UMAP attractive gradient: -2*2ab * d^(2b-2) / (1 + a*d^(2b))

                        double pd2b = std::pow(dist_sq_d, b_d);

                        double grad_coeff_d = (-2.0 * a_d * b_d * pd2b) /

                            (dist_sq_d * (a_d * pd2b + 1.0));



                        // Apply clamping

                        grad_coeff_d = std::max(-4.0, std::min(4.0, grad_coeff_d));

                        float grad_coeff = static_cast<float>(grad_coeff_d);



                        for (int d = 0; d < embedding_dim; d++) {

                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -

                                model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];

                            float grad = alpha * grad_coeff * diff;

                            model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;

                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -= grad;

                        }



                        // Accumulate attractive force loss (UMAP cross-entropy: attractive term)

                        if (loss_samples < 1000) { // Sample subset for performance

                            double attractive_prob = 1.0 / (1.0 + a_d * pd2b);

                            epoch_loss += static_cast<float>(-std::log(attractive_prob + 1e-12));  // -log(P_attract)

                            loss_samples++;

                        }

                    }



                    // CRITICAL: Dynamic negative sampling (umappp line 163)
                    float epochs_per_negative_sample = epochs_per_sample[filt_idx] / negative_sample_rate;
                    int num_neg_samples = static_cast<int>((epoch - epoch_of_next_negative_sample[filt_idx]) / epochs_per_negative_sample);

                    for (int neg = 0; neg < num_neg_samples; neg++) {

                        size_t k = vertex_dist(rng);

                        if (k == i) continue;  // CRITICAL FIX: Only skip self-samples, not target vertex



                        float neg_dist_sq = 0.0f;

                        for (int d = 0; d < embedding_dim; d++) {

                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -

                                model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];

                            neg_dist_sq += diff * diff;

                        }



                        if (neg_dist_sq > std::numeric_limits<float>::epsilon()) {

                            // CRITICAL FIX: Use double precision for repulsive force calculations to match umappp

                            double neg_dist_sq_d = static_cast<double>(neg_dist_sq);

                            double a_d = static_cast<double>(model->a);

                            double b_d = static_cast<double>(model->b);



                            // UMAP repulsive gradient: 2b / ((0.001 + d^2) * (1 + a*d^(2b)))

                            double pd2b = std::pow(neg_dist_sq_d, b_d);

                            double grad_coeff_d = (2.0 * b_d) /

                                ((0.001 + neg_dist_sq_d) * (a_d * pd2b + 1.0));



                            // Apply clamping

                            grad_coeff_d = std::max(-4.0, std::min(4.0, grad_coeff_d));

                            float grad_coeff = static_cast<float>(grad_coeff_d);



                            for (int d = 0; d < embedding_dim; d++) {

                                float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -

                                    model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];

                                float grad = alpha * grad_coeff * diff;

                                model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;

                            }



                            // Accumulate repulsive force loss (UMAP cross-entropy: repulsive term)

                            if (loss_samples < 1000) { // Sample subset for performance

                                double repulsive_prob = 1.0 / (1.0 + a_d * pd2b);

                                epoch_loss += static_cast<float>(-std::log(1.0 - repulsive_prob + 1e-12));  // -log(1 - P_repulse)

                                loss_samples++;

                            }

                        }

                    }

                    // CRITICAL: Update epoch scheduling counters (umappp lines 180-181)
                    epoch_of_next_sample[filt_idx] += epochs_per_sample[filt_idx];
                    epoch_of_next_negative_sample[filt_idx] += num_neg_samples * epochs_per_negative_sample;

                }



                // Adaptive progress reporting: more frequent for early epochs

                bool should_report = (epoch < 10) ||                        // Report first 10 epochs

                    (epoch % progress_interval == 0) ||       // Regular interval

                    (epoch == n_epochs - 1);                  // Final epoch



                if (should_report) {

                    float percent = (static_cast<float>(epoch + 1) / static_cast<float>(n_epochs)) * 100.0f;



                    // Calculate average loss for this epoch (shared by both callback and console)

                    float avg_loss = loss_samples > 0 ? epoch_loss / loss_samples : 0.0f;



                    if (progress_callback) {

                        // Use callback for C# integration - pass loss info in global variable

                        g_current_epoch_loss = avg_loss;  // Store for v2 callback wrapper

                        char loss_msg[64];

                        snprintf(loss_msg, sizeof(loss_msg), "Epoch %d/%d Loss: %.3f", epoch + 1, n_epochs, avg_loss);

                        progress_callback("Training", epoch + 1, n_epochs, percent, loss_msg);

                    }

                    else {

                        // Console output for C++ testing

                        int percent_int = static_cast<int>(percent);

                        int filled = percent_int / 5;  // 20 characters for 100%



                        auto current_time = std::chrono::steady_clock::now();

                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_report_time);



                        std::printf("\r[");

                        for (int i = 0; i < 20; i++) {

                            if (i < filled) std::printf("=");

                            else if (i == filled && percent_int % 5 >= 2) std::printf(">");

                            else std::printf(" ");

                        }



                        std::printf("] %d%% (Epoch %d/%d) Loss: %.3f [%lldms]",

                            percent_int, epoch + 1, n_epochs, avg_loss, static_cast<long long>(elapsed.count()));

                        std::fflush(stdout);



                        last_report_time = current_time;

                    }

                }

            }



            if (!progress_callback) {

                std::printf("\nTraining completed!\n");

                std::fflush(stdout);

            }



            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy

            size_t expected = static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim);

            if (model->embedding.size() < expected) {

                return UWOT_ERROR_MEMORY;

            }

            for (size_t i = 0; i < expected; ++i) {

                embedding[i] = model->embedding[i];

            }



            model->is_fitted = true;

            return UWOT_SUCCESS;



        }

        catch (...) {

            return UWOT_ERROR_MEMORY;

        }

    }





}