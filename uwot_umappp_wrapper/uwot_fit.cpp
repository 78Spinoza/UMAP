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
#include <thread>



// Include umappp headers for core algorithms
#include "umappp.hpp"
#include "initialize.hpp"
#include "optimize_layout.hpp"
#include "Options.hpp"
#include "Status.hpp"
#include "NeighborList.hpp"
#include "uwot_hnsw_knncolle.hpp"

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

    // OLD IMPLEMENTATION REMOVED - Use uwot_fit_with_umappp_hnsw instead
    // Alternative umappp + HNSW implementation
    int uwot_fit_with_umappp_hnsw(UwotModel* model,
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
        int random_seed,
        int M,
        int ef_construction,
        int ef_search) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 || embedding_dim <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            // Create wrapped callback for progress reporting
            auto wrapped_callback = progress_callback;

            if (wrapped_callback) {
                wrapped_callback("🚀 UMAPPP+HNSW CODE PATH ACTIVE 🚀", 0, 100, 0.0f, "Using proven umappp reference implementation with HNSW k-NN search");
            }

            // Create umappp Options from parameters
            umappp::Options options;
            options.num_neighbors = n_neighbors;
            options.min_dist = min_dist;
            options.spread = spread;
            options.num_epochs = n_epochs;
            options.local_connectivity = 1.0;
            options.bandwidth = 1.0;
            options.mix_ratio = 1.0;
            options.repulsion_strength = 1.0;
            options.learning_rate = 1.0;
            options.negative_sample_rate = 5.0;
            // Use all available CPU cores for maximum performance
            options.num_threads = std::thread::hardware_concurrency();
            options.parallel_optimization = true;
            // Use RANDOM initialization for large datasets (>20k) to avoid spectral initialization hanging
            // Spectral initialization requires computing eigenvectors which is O(n^3) and very slow for large n
            options.initialize_method = (n_obs > 20000) ? umappp::InitializeMethod::RANDOM : umappp::InitializeMethod::SPECTRAL;
            options.initialize_random_on_spectral_fail = true;
            options.initialize_seed = (random_seed > 0) ? random_seed : 42;
            options.optimize_seed = (random_seed > 0) ? random_seed + 1 : 43;

            if (wrapped_callback) {
                wrapped_callback("Creating HNSW builder", 10, 100, 0.0f, "Setting up HNSW with Euclidean distance");
            }

            // Use default HNSW parameters if not specified (-1 means use default)
            // Use high-quality HNSW parameters for accurate k-NN approximation
            // Scale with dataset size to balance accuracy and speed
            int actual_M = (M <= 0) ? ((n_obs > 50000) ? 64 : (n_obs > 20000) ? 48 : 32) : M;
            int actual_ef_construction = (ef_construction <= 0) ? ((n_obs > 50000) ? 500 : (n_obs > 20000) ? 400 : 300) : ef_construction;
            int actual_ef_search = (ef_search <= 0) ? ((n_obs > 50000) ? 300 : (n_obs > 20000) ? 200 : 100) : ef_search;

            // Create HNSW builder for Euclidean distance
            auto hnsw_builder = std::make_unique<uwot::HnswEuclideanBuilder<int, float>>(actual_M, actual_ef_construction, actual_ef_search);

            if (wrapped_callback) {
                wrapped_callback("Building HNSW index", 20, 100, 0.0f, "Indexing data points with HNSW");
            }

            // Create matrix wrapper for the data using SimpleMatrix
            knncolle::SimpleMatrix<int, float> matrix(n_dim, n_obs, data);

            if (wrapped_callback) {
                wrapped_callback("Building HNSW structure", 30, 100, 0.0f, "Constructing HNSW graph");
            }

            // Build HNSW index
            auto hnsw_prebuilt = hnsw_builder->build_unique(matrix);

            if (wrapped_callback) {
                wrapped_callback("Initializing umappp", 50, 100, 0.0f, "Setting up umappp optimization with HNSW neighbors");
            }

            // Initialize umappp with HNSW prebuilt index
            auto status = umappp::initialize<int, float, float>(*hnsw_prebuilt, embedding_dim, embedding, std::move(options));

            if (wrapped_callback) {
                wrapped_callback("Optimizing layout", 60, 100, 0.0f, "Running umappp optimization with HNSW neighbors");
            }

            // Run optimization with progress reporting
            int total_epochs = status.num_epochs();
            for (int epoch = 0; epoch < total_epochs; ++epoch) {
                status.run(embedding, epoch + 1);

                // Update progress callback (60% to 95%)
                if (wrapped_callback) {
                    float progress = 60.0f + (95.0f - 60.0f) * (static_cast<float>(epoch + 1) / static_cast<float>(total_epochs));
                    char msg[256];
                    snprintf(msg, sizeof(msg), "Epoch %d/%d - umappp optimization with HNSW", epoch + 1, total_epochs);
                    wrapped_callback("Optimizing layout", epoch + 1, total_epochs, progress, msg);
                }
            }

            if (wrapped_callback) {
                wrapped_callback("Finalizing", 100, 100, 0.0f, "umappp with HNSW completed successfully");
            }

            // Update model with results
            model->is_fitted = true;
            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;
            model->spread = spread;
            model->metric = metric;
            model->hnsw_M = actual_M;
            model->hnsw_ef_construction = actual_ef_construction;
            model->hnsw_ef_search = actual_ef_search;

            return UWOT_SUCCESS;

        } catch (const std::exception& e) {
            if (progress_callback) {
                progress_callback("Error", 0, 100, 0.0f, e.what());
            }
            return UWOT_ERROR_MEMORY;
        } catch (...) {
            if (progress_callback) {
                progress_callback("Error", 0, 100, 0.0f, "Unknown error in umappp with HNSW");
            }
            return UWOT_ERROR_MEMORY;
        }
    }

}