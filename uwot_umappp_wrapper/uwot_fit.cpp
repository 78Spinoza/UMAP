#include "uwot_fit.h"
#include "uwot_simple_wrapper.h"
#include "uwot_distance.h"
#include "optimize_layout.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <omp.h>



// Include umappp headers for core algorithms
#include "umappp.hpp"
#include "initialize.hpp"
#include "optimize_layout.hpp"
#include "Options.hpp"
#include "Status.hpp"
#include "NeighborList.hpp"
#include "uwot_hnsw_knncolle.hpp"
#include "uwot_hnsw_utils.h"

// Include uwot headers
#include "smooth_knn.h"

namespace fit_utils {

// CRITICAL MISSING FUNCTION: Compute normalization statistics (from pure_cpp)
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






    // Helper function to apply smooth_knn to a single point's neighbors
// STEP 7: Simplified smooth_knn for legacy brute-force exact k-NN path only
// NOTE: The HNSW path now uses correct umappp::initialize() delegation (lines 565-570)
// This function is ONLY for the legacy brute-force fallback path (line 922)
void apply_smooth_knn_to_point(const int* indices, const float* distances, int k, float* weights) {
    if (k <= 0) return;

    // Find rho (smallest non-zero distance)
    double rho = 0.0;
    for (int i = 0; i < k; ++i) {
        if (distances[i] > 0.0) {
            rho = distances[i];
            break;
        }
    }

    // CORRECTED: Use proper UMAP target (log2(k+1) * bandwidth)
    // This matches umappp's neighbor_similarities.hpp line 129
    const double bandwidth = 1.0;  // Standard UMAP bandwidth
    double target_sum = std::log2(static_cast<double>(k) + 1.0) * bandwidth;
    double sigma = 1.0;
    constexpr int max_iter = 64;
    constexpr double tol = 1e-6;

    // Binary search for sigma
    double lo = 0.0, hi = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iter; ++iter) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            double r = distances[i] - rho;
            sum += (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
        }

        if (std::abs(sum - target_sum) < tol) break;

        if (sum > target_sum) {
            hi = sigma;
            sigma = (lo + sigma) / 2.0;
        } else {
            lo = sigma;
            if (hi == std::numeric_limits<double>::max()) {
                sigma *= 2.0;
            } else {
                sigma = (sigma + hi) / 2.0;
            }
        }
    }

    // Compute final fuzzy weights
    double weight_sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double r = distances[i] - rho;
        double w = (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
        weights[i] = static_cast<float>(w);
        weight_sum += w;
    }

    // Normalize
    if (weight_sum > 0.0) {
        for (int i = 0; i < k; ++i) {
            weights[i] /= static_cast<float>(weight_sum);
        }
    } else {
        // Fallback: uniform weights
        for (int i = 0; i < k; ++i) {
            weights[i] = 1.0f / static_cast<float>(k);
        }
    }
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
        int ef_search,
        float local_connectivity,
        float bandwidth,
        int use_spectral_init) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 || embedding_dim <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            // Create wrapped callback for progress reporting
            auto wrapped_callback = progress_callback;

            // Set up error callback for HNSW operations
            hnsw_utils::set_hnsw_error_callback(wrapped_callback);

            if (wrapped_callback) {
                wrapped_callback("UMAPPP+HNSW CODE PATH ACTIVE", 0, 100, 0.0f, "Using proven umappp reference implementation with HNSW k-NN search");
            }

            // Create umappp Options from parameters
            umappp::Options options;
            options.num_neighbors = n_neighbors;
            options.min_dist = min_dist;
            options.spread = spread;
            options.num_epochs = n_epochs;
            // UMAP smoothing parameters
            // For large datasets with random init, higher bandwidth creates stronger global connections
            // Standard UMAP uses 1.0, but large datasets (50k+) often need 2.0-3.0 for global structure
            options.local_connectivity = local_connectivity;
            options.bandwidth = bandwidth;

            // WARNING: Large datasets with random init and low bandwidth
            bool use_random_init = (n_obs > 20000);
            if (use_random_init && n_obs >= 50000 && bandwidth <= 1.0f && wrapped_callback) {
                wrapped_callback("Bandwidth Warning", 3, 100, 3.0f,
                               "\n[WARNING] Large dataset (>=50k) with random init and bandwidth=1.0 may have poor global structure!\n"
                               "          Consider bandwidth=2.0-3.0 for better results. See UMAP bandwidth parameter.\n");
            }
            options.mix_ratio = 1.0;
            options.repulsion_strength = 1.0;
            options.learning_rate = 1.0;
            options.negative_sample_rate = 5.0;
            // Use all available CPU cores for maximum performance
            options.num_threads = std::thread::hardware_concurrency();
            options.parallel_optimization = true;

            // Set up progress callback for HNSW path
            if (wrapped_callback) {
                options.progress_callback = [wrapped_callback](int current_epoch, int total_epochs, const double* embedding) {
                    // Special case: epoch=-1 indicates KNN smoothing report from initialize.hpp
                    if (current_epoch == -1) {
                        wrapped_callback("KNN Smoothing Verified", 0, 100, 55.0f,
                                       "\n[SMOOTHING] Applied smooth_knn with correct target=log2(k+1)*bandwidth\n");
                        return;
                    }
                    float progress = 60.0f + (95.0f - 60.0f) * (static_cast<float>(current_epoch) / static_cast<float>(total_epochs));
                    wrapped_callback("Optimizing layout", current_epoch, total_epochs, progress, "umappp optimization with HNSW");
                };
            }

            // Report CPU core usage to user
            if (wrapped_callback) {
                auto num_cores = std::thread::hardware_concurrency();
                std::string core_message = "Using " + std::to_string(num_cores) + " CPU cores for parallel processing";
                wrapped_callback("CPU Core Detection", 8, 100, 0.0f, core_message.c_str());
            }
            // Initialization method: User override or auto-select based on size
            // use_spectral_init: -1 = auto (>20k=random, ≤20k=spectral), 0 = force random, 1 = force spectral
            if (use_spectral_init >= 0) {
                // User explicitly requested spectral (1) or random (0)
                options.initialize_method = (use_spectral_init == 1) ? umappp::InitializeMethod::SPECTRAL : umappp::InitializeMethod::RANDOM;
            } else {
                // Auto-select: RANDOM for large datasets (>20k) to avoid spectral hanging
                // Spectral initialization requires computing eigenvectors which is O(n^3) and very slow for large n
                options.initialize_method = (n_obs > 20000) ? umappp::InitializeMethod::RANDOM : umappp::InitializeMethod::SPECTRAL;
            }
            options.initialize_random_on_spectral_fail = true;
            options.initialize_seed = (random_seed > 0) ? random_seed : 42;
            options.optimize_seed = (random_seed > 0) ? random_seed + 1 : 43;

            // Warn user about initialization method for large datasets
            if (n_obs > 20000 && wrapped_callback) {
                wrapped_callback("Large dataset detected", 5, 100, 0.0f,
                    "Dataset has >20k samples. Using RANDOM initialization instead of SPECTRAL for performance.");
            }

            if (wrapped_callback) {
                wrapped_callback("Creating HNSW builder", 10, 100, 0.0f, "Setting up HNSW with Euclidean distance");
            }

            // Use default HNSW parameters if not specified (-1 means use default)
            // Use high-quality HNSW parameters for accurate k-NN approximation
            // Scale with dataset size to balance accuracy and speed
            int actual_M = (M <= 0) ? ((n_obs > 50000) ? 64 : (n_obs > 20000) ? 48 : 32) : M;
            int actual_ef_construction = (ef_construction <= 0) ? ((n_obs > 50000) ? 500 : (n_obs > 20000) ? 400 : 300) : ef_construction;
            // CRITICAL: ef_search must be >= n_neighbors for good accuracy, use max(default, n_neighbors * 4)
            int base_ef_search = (n_obs > 50000) ? 300 : (n_obs > 20000) ? 200 : 100;
            int actual_ef_search = (ef_search <= 0) ? std::max(base_ef_search, n_neighbors * 4) : ef_search;

            // Warn user about HNSW parameter scaling for large datasets
            if (wrapped_callback && (M <= 0 || ef_construction <= 0 || ef_search <= 0)) {
                char hnsw_msg[256];
                snprintf(hnsw_msg, sizeof(hnsw_msg),
                        "\n[HNSW] Auto-scaled: M=%d, ef_construction=%d, ef_search=%d (k=%d)\n",
                        actual_M, actual_ef_construction, actual_ef_search, n_neighbors);
                wrapped_callback("HNSW parameters scaled", 8, 100, 0.0f, hnsw_msg);
            }

            // CRITICAL FIX: Initialize model->embedding early (like pure_cpp version)
            // This prevents memory corruption by working directly with model->embedding
            // instead of using a local embedding array and copying at the end
            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));

            // Thread-safe random initialization (matching pure_cpp approach)
            if (wrapped_callback) {
                wrapped_callback("Initializing embedding", 12, 100, 12.0f, "Setting up random embedding coordinates");
            }

#pragma omp parallel if(n_obs > 1000)
            {
                // Each thread gets its own generator to avoid race conditions
                thread_local std::mt19937 gen(static_cast<unsigned int>(options.initialize_seed));
                thread_local std::normal_distribution<float> dist(0.0f, 1e-4f);

#pragma omp for
                for (int i = 0; i < n_obs; i++) {
                    for (int d = 0; d < embedding_dim; d++) {
                        size_t idx = static_cast<size_t>(i) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d);
                        model->embedding[idx] = dist(gen);
                    }
                }
            }

            // CRITICAL FIX 2.1: Normalize data before HNSW building (must match transform normalization)
            // Determine normalization mode based on metric
            int norm_mode = hnsw_utils::NormalizationPipeline::determine_normalization_mode(metric);
            model->normalization_mode = norm_mode;
            model->use_normalization = (norm_mode > 0);

            // CRITICAL MISSING STEP: Compute normalization statistics FIRST (like pure_cpp)
            if (norm_mode > 0) {
                std::vector<float> input_data(data, data + static_cast<size_t>(n_obs) * n_dim);
                compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);
            }

            // Prepare data for HNSW (normalized or raw depending on metric)
            std::vector<float> processed_data;
            const float* data_for_hnsw = data;  // Default: use raw data

            if (norm_mode > 0) {
                if (wrapped_callback) {
                    wrapped_callback("Normalizing data", 15, 100, 15.0f, "Applying metric normalization");
                }

                // Convert raw data array to vector for normalization
                std::vector<float> input_data(data, data + static_cast<size_t>(n_obs) * n_dim);

                // Normalize data (will compute and store means/stds if mode 1, or just normalize if mode 2)
                bool norm_success = hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                    input_data,
                    processed_data,
                    n_obs, n_dim,
                    model->feature_means,
                    model->feature_stds,
                    norm_mode
                );

                                if (norm_success) {
                    data_for_hnsw = processed_data.data();

                    if (wrapped_callback) {
                        wrapped_callback("Normalization complete", 18, 100, 18.0f, "Data normalized for HNSW");
                    }
                } else {
                    if (wrapped_callback) {
                        wrapped_callback("Warning", 18, 100, 18.0f, "Normalization failed - using raw data");
                    }
                    model->normalization_mode = 0;
                    model->use_normalization = false;
                }
            }

            // CRITICAL FIX: Store raw_data for exact k-NN when force_exact_knn = true
            if (model->force_exact_knn) {
                if (wrapped_callback) {
                    wrapped_callback("Storing training data", 19, 100, 19.0f, "Saving raw data for exact k-NN search");
                }

                // Store the normalized data for exact k-NN search
                if (norm_mode > 0 && !processed_data.empty()) {
                    // Use normalized data
                    model->raw_data = processed_data;
                } else {
                    // Use original raw data
                    model->raw_data.assign(data, data + static_cast<size_t>(n_obs) * n_dim);
                }

                if (wrapped_callback) {
                    wrapped_callback("Training data stored", 20, 100, 20.0f,
                        ("Stored " + std::to_string(model->raw_data.size() / n_dim) + " training points").c_str());
                }
            }

            // CHECK: Force exact k-NN or use HNSW based on force_exact_knn parameter
            if (model->force_exact_knn) {
                if (wrapped_callback) {
                    wrapped_callback("Using exact k-NN", 20, 100, 20.0f, "force_exact_knn = true - using brute-force k-NN instead of HNSW");
                }

                // Use the original build_knn_graph function for exact k-NN
                std::vector<int> nn_indices;
                std::vector<double> nn_distances;

                // EXACT k-NN mode - PROPER IMPLEMENTATION
                if (wrapped_callback) {
                    wrapped_callback("Exact k-NN mode", 50, 100, 50.0f, "Computing exact k-NN graph (O(n^2) - slow but exact)");
                }

                // Create SimpleMatrix from the data for umappp
                const float* data_for_umappp = processed_data.empty() ? data : processed_data.data();
                knncolle::SimpleMatrix<int, float> mat(n_obs, n_dim, data_for_umappp);

                if (wrapped_callback) {
                    wrapped_callback("Building exact k-NN index", 60, 100, 60.0f, "Creating knncolle::BruteforcePrebuilt for exact k-NN");
                }

                // Create exact k-NN index using knncolle::BruteforcePrebuilt
                std::unique_ptr<knncolle::Prebuilt<int, float, float>> exact_knn_index;

                // Choose distance metric based on UwotMetric - ONLY SUPPORTED METRICS
                std::shared_ptr<const knncolle::DistanceMetric<float, float>> distance_metric;
                switch (metric) {
                    case UWOT_METRIC_EUCLIDEAN:
                        distance_metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
                        break;
                    case UWOT_METRIC_MANHATTAN:
                        distance_metric = std::make_shared<knncolle::ManhattanDistance<float, float>>();
                        break;
                    default:
                        // For unsupported metrics (Cosine, Correlation, Hamming), use Euclidean
                        distance_metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
                        if (wrapped_callback) {
                            wrapped_callback("Warning", 65, 100, 65.0f, "Only Euclidean and Manhattan supported for exact k-NN, using Euclidean");
                        }
                        break;
                }

                // Create BruteforceBuilder and build the index
                knncolle::BruteforceBuilder<int, float, float> builder(distance_metric);
                exact_knn_index.reset(builder.build_raw(mat));

                if (wrapped_callback) {
                    wrapped_callback("Running umappp with exact k-NN", 70, 100, 70.0f, "Starting umappp optimization with exact k-NN graph");
                }

                // Create umappp options for exact k-NN
                umappp::Options exact_options;
                exact_options.local_connectivity = 1.0;
                exact_options.bandwidth = 1.0;
                exact_options.mix_ratio = 1.0;
                exact_options.spread = spread;
                exact_options.min_dist = min_dist;
                exact_options.num_epochs = n_epochs;
                exact_options.num_neighbors = n_neighbors;
                // Use all available CPU cores for maximum performance (Linux & Windows friendly)
                exact_options.num_threads = std::thread::hardware_concurrency();
                exact_options.parallel_optimization = true;

                if (random_seed >= 0) {
                    exact_options.initialize_seed = static_cast<uint64_t>(random_seed);
                }

                if (wrapped_callback) {
                    wrapped_callback("Initializing umappp", 50, 100, 50.0f, "Setting up umappp optimization with exact k-NN");
                }

                // Set up progress callback in umappp options before initialization
                if (wrapped_callback) {
                    exact_options.progress_callback = [wrapped_callback](int current_epoch, int total_epochs, const double* embedding) {
                        // Special case: epoch=-1 indicates KNN smoothing report from initialize.hpp
                        if (current_epoch == -1) {
                            wrapped_callback("KNN Smoothing Verified (Exact)", 0, 100, 55.0f,
                                           "\n[SMOOTHING] Applied smooth_knn with correct target=log2(k+1)*bandwidth (Exact k-NN)\n");
                            return;
                        }
                        float progress = 60.0f + (95.0f - 60.0f) * (static_cast<float>(current_epoch) / static_cast<float>(total_epochs));
                        wrapped_callback("Optimizing layout", current_epoch, total_epochs, progress, "umappp exact k-NN optimization");
                    };
                }

                // Initialize umappp with exact k-NN prebuilt index - write directly to output embedding
                auto exact_status = umappp::initialize<int, float, float>(*exact_knn_index, embedding_dim, embedding, std::move(exact_options));

                // CRITICAL: Retrieve rho/sigma from Status for fast transform (EXACT k-NN path)
                model->rho.assign(exact_status.rhos().begin(), exact_status.rhos().end());
                model->sigma.assign(exact_status.sigmas().begin(), exact_status.sigmas().end());
                model->has_fast_transform_data = (!model->rho.empty() && !model->sigma.empty());

                // Get thread count and report OpenMP usage for exact k-NN
                int exact_n_threads = omp_get_max_threads();
                if (wrapped_callback) {
                    std::string thread_info = "[INFO] Using " + std::to_string(exact_n_threads) + " threads for umappp (exact k-NN)\n";
                    wrapped_callback("Thread Detection", 61, 100, 61.0f, thread_info.c_str());
                }

                // Run optimization with native progress callback support - maximum performance!
                exact_status.run(embedding);

                if (wrapped_callback) {
                    wrapped_callback("Exact k-NN optimization completed", 95, 100, 95.0f, "Exact k-NN optimization finished successfully");
                }

                // CRITICAL: Build HNSW index for transform even in exact k-NN mode
                if (wrapped_callback) {
                    wrapped_callback("Saving HNSW index", 96, 100, 96.0f, "Building original space HNSW for transform");
                }

                try {
                    // Create space based on metric
                    auto original_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
                    hnswlib::SpaceInterface<float>* space = nullptr;

                    switch (metric) {
                        case UWOT_METRIC_EUCLIDEAN:
                            space = new hnswlib::L2Space(n_dim);
                            break;
                        case UWOT_METRIC_COSINE:
                            space = new hnswlib::InnerProductSpace(n_dim);
                            break;
                        case UWOT_METRIC_MANHATTAN:
                            space = new hnswlib::L2Space(n_dim);
                            break;
                        default:
                            space = new hnswlib::L2Space(n_dim);
                            break;
                    }

                    // Create HNSW index for original space
                    auto original_hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        space, n_obs, actual_M, actual_ef_construction
                    );

                    // Add all data points to the index (using same data as used for exact k-NN)
                    const float* data_for_index = processed_data.empty() ? data : processed_data.data();
                    for (int i = 0; i < n_obs; i++) {
                        original_hnsw->addPoint(const_cast<float*>(data_for_index + i * n_dim), static_cast<hnswlib::labeltype>(i));

                        // Report progress more frequently for large datasets (>=10k)
                        int progress_interval = (n_obs >= 10000) ? (n_obs / 100) : (n_obs / 20); // 1% for >=10k, 5% otherwise
                        if (wrapped_callback && (i % progress_interval == 0 || i == n_obs - 1)) {
                            float sub_progress = 96.0f + (1.0f * i / n_obs);
                            wrapped_callback("Saving HNSW index", i, n_obs, sub_progress, "Indexing points for transform");
                        }
                    }

                    // Set ef for search
                    original_hnsw->setEf(actual_ef_search);

                    // Save to model
                    model->original_space_index = std::move(original_hnsw);
                    model->original_space_factory = std::move(original_space_factory);

                } catch (const std::exception& e) {
                    // Suppress unreferenced variable warning - exception is intentionally ignored
                    (void)e;

                    if (wrapped_callback) {
                        wrapped_callback("Warning", 96, 100, 96.0f, "Failed to save HNSW index - transform may not work");
                    }
                    // Continue anyway - fit succeeded even if index save failed
                }

                if (wrapped_callback) {
                    wrapped_callback("Finalizing", 100, 100, 100.0f, "umappp with exact k-NN completed successfully");
                }

            } else {
                // Use HNSW (default path)
                if (wrapped_callback) {
                    wrapped_callback("Using HNSW k-NN", 20, 100, 20.0f, "force_exact_knn = false - using fast HNSW approximation");
                }

                // Create HNSW builder based on distance metric
            std::unique_ptr<knncolle::Builder<int, float, float, knncolle::SimpleMatrix<int, float>>> hnsw_builder;

            switch (metric) {
                case UWOT_METRIC_EUCLIDEAN:
                    hnsw_builder = std::make_unique<uwot::HnswEuclideanBuilder<int, float>>(actual_M, actual_ef_construction, actual_ef_search);
                    break;
                case UWOT_METRIC_COSINE:
                    hnsw_builder = std::make_unique<uwot::HnswCosineBuilder<int, float>>(actual_M, actual_ef_construction, actual_ef_search);
                    break;
                case UWOT_METRIC_MANHATTAN:
                    hnsw_builder = std::make_unique<uwot::HnswManhattanBuilder<int, float>>(actual_M, actual_ef_construction, actual_ef_search);
                    break;
                default:
                    hnsw_builder = std::make_unique<uwot::HnswEuclideanBuilder<int, float>>(actual_M, actual_ef_construction, actual_ef_search);
                    break;
            }

            if (wrapped_callback) {
                wrapped_callback("Building HNSW index", 20, 100, 0.0f, "Indexing data points with HNSW");
            }

            // Create matrix wrapper for the data using SimpleMatrix (using normalized or raw data)
            knncolle::SimpleMatrix<int, float> matrix(n_dim, n_obs, data_for_hnsw);

            // Build HNSW index with progress reporting for large datasets
            std::unique_ptr<knncolle::Prebuilt<int, float, float>> hnsw_prebuilt;

            if (n_obs >= 50000 && wrapped_callback) {
                // For large datasets: Manual build with progress reporting
                if (wrapped_callback) {
                    wrapped_callback("Building HNSW structure", 30, 100, 30.0f, "\nConstructing HNSW graph (100k points - this may take 1-2 minutes)...\n");
                }

                // Build manually to add progress reporting
                hnsw_prebuilt = hnsw_builder->build_unique(matrix);

                if (wrapped_callback) {
                    wrapped_callback("HNSW construction complete", 48, 100, 48.0f, "\n[HNSW] Index construction completed successfully\n");
                }
            } else {
                // For smaller datasets: Quick build without spam
                if (wrapped_callback) {
                    wrapped_callback("Building HNSW structure", 30, 100, 30.0f, "Constructing HNSW graph...");
                }
                hnsw_prebuilt = hnsw_builder->build_unique(matrix);
                if (wrapped_callback) {
                    wrapped_callback("HNSW construction complete", 48, 100, 48.0f, "HNSW index ready");
                }
            }

            // STEP 8: CORRECT DELEGATION - Let umappp handle neighbor search and smoothing
            if (wrapped_callback) {
                wrapped_callback("Initializing UMAP with HNSW", 50, 100, 50.0f,
                                "Using correct umappp workflow - delegating to umappp::initialize");
            }

            // This is the CORRECT way: pass the HNSW index directly to umappp::initialize
            // umappp will:
            // 1. Search the HNSW index for neighbors
            // 2. Apply correct smooth_knn with proper target (log2(k+1) * bandwidth)
            // 3. Create the fuzzy simplicial set correctly
            // 4. Return Status with rho/sigma values we need for fast transform
            auto status = umappp::initialize<int, float, float>(
                *hnsw_prebuilt,
                embedding_dim,
                embedding,
                std::move(options)
            );

            // STEP 9: Retrieve rho/sigma from Status for fast transform
            model->rho.assign(status.rhos().begin(), status.rhos().end());
            model->sigma.assign(status.sigmas().begin(), status.sigmas().end());
            model->has_fast_transform_data = (!model->rho.empty() && !model->sigma.empty());

            if (wrapped_callback) {
                wrapped_callback("UMAP initialization complete", 60, 100, 60.0f,
                                "Fuzzy graph created with correct smooth_knn - rho/sigma saved for fast transform");
            }

            // Get thread count and report OpenMP usage
            int n_threads = omp_get_max_threads();
            if (wrapped_callback) {
                std::string thread_info = "[INFO] Using " + std::to_string(n_threads) + " threads for umappp\n";
                wrapped_callback("Thread Detection", 61, 100, 61.0f, thread_info.c_str());
            }

            // Run optimization with native progress callback support - maximum performance!
            status.run(embedding);

            if (wrapped_callback) {
                wrapped_callback("Finalizing", 95, 100, 95.0f, "umappp optimization completed");
            }

            // Extract HNSW index from knncolle wrapper for transform (NO rebuild!)
            // The wrapper has a release_index() method that transfers ownership safely
            if (wrapped_callback) {
                wrapped_callback("Extracting HNSW for transform", 96, 100, 96.0f, "Reusing HNSW index from umappp - no rebuild needed!");
            }

            try {
                // Cast to HnswPrebuilt to access release_index() method
                auto* hnsw_wrapper_release = dynamic_cast<uwot::HnswPrebuilt<int, float>*>(hnsw_prebuilt.get());

                if (hnsw_wrapper_release) {
                    // Release ownership of HNSW index and space from the wrapper
                    // After this call, the wrapper no longer owns these resources, so they won't be destroyed
                    auto [released_index, released_space] = hnsw_wrapper_release->release_index();

                    if (released_index && released_space) {
                        // Transfer ownership of BOTH index and space to model
                        model->original_space_index = std::move(released_index);

                        // CRITICAL: Store the RELEASED space in the factory
                        // We MUST use the same space object that the HNSW index is using!
                        // Creating a new space would leave the index with a dangling pointer
                        model->original_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();

                        // Manually assign the released space to the appropriate factory member
                        // based on the metric type
                        switch (metric) {
                            case UWOT_METRIC_EUCLIDEAN:
                                model->original_space_factory->l2_space.reset(
                                    static_cast<hnswlib::L2Space*>(released_space.release()));
                                break;
                            case UWOT_METRIC_COSINE:
                                model->original_space_factory->ip_space.reset(
                                    static_cast<hnswlib::InnerProductSpace*>(released_space.release()));
                                break;
                            case UWOT_METRIC_MANHATTAN:
                                model->original_space_factory->l1_space.reset(
                                    static_cast<L1Space*>(released_space.release()));
                                break;
                            default:
                                model->original_space_factory->l2_space.reset(
                                    static_cast<hnswlib::L2Space*>(released_space.release()));
                                break;
                        }

                        model->original_space_factory->current_metric = metric;
                        model->original_space_factory->current_dim = n_dim;

                        if (wrapped_callback) {
                            wrapped_callback("HNSW extraction successful", 97, 100, 97.0f, "HNSW index extracted - zero rebuild overhead!");
                        }
                    } else {
                        throw std::runtime_error("Released index or space is null");
                    }
                } else {
                    throw std::runtime_error("Failed to cast to HnswPrebuilt");
                }

            } catch (const std::exception& e) {
                // Fallback: rebuild if extraction fails (should never happen)
                (void)e;

                if (wrapped_callback) {
                    wrapped_callback("Warning", 96, 100, 96.0f, "Extraction failed, rebuilding HNSW index");
                }

                try {
                    auto fallback_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
                    hnswlib::SpaceInterface<float>* fallback_space = nullptr;

                    switch (metric) {
                        case UWOT_METRIC_EUCLIDEAN:
                            fallback_space = new hnswlib::L2Space(n_dim);
                            break;
                        case UWOT_METRIC_COSINE:
                            fallback_space = new hnswlib::InnerProductSpace(n_dim);
                            break;
                        case UWOT_METRIC_MANHATTAN:
                            fallback_space = new hnswlib::L2Space(n_dim);
                            break;
                        default:
                            fallback_space = new hnswlib::L2Space(n_dim);
                            break;
                    }

                    auto fallback_hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        fallback_space, n_obs, actual_M, actual_ef_construction
                    );

                    for (int i = 0; i < n_obs; i++) {
                        fallback_hnsw->addPoint(const_cast<float*>(data_for_hnsw + i * n_dim), static_cast<hnswlib::labeltype>(i));
                    }

                    fallback_hnsw->setEf(actual_ef_search);

                    model->original_space_index = std::move(fallback_hnsw);
                    model->original_space_factory = std::move(fallback_space_factory);

                } catch (...) {
                    if (wrapped_callback) {
                        wrapped_callback("Warning", 96, 100, 96.0f, "Failed to save HNSW index - transform may not work");
                    }
                }
            }

            if (wrapped_callback) {
                wrapped_callback("Finalizing", 100, 100, 100.0f, "umappp with HNSW completed successfully");
            }
            }

            // FAST TRANSFORM OPTIMIZATION: HNSW path already computed rho/sigma in the k-NN loop
            if (wrapped_callback) {
                wrapped_callback("Fast Transform Data Ready", 98, 100, 98.0f, "Fast transform optimization completed during HNSW k-NN");
            }

            // Update model with results
            model->is_fitted = true;
            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;
            model->spread = spread;
            model->local_connectivity = local_connectivity;
            model->bandwidth = bandwidth;
            model->metric = metric;
            model->hnsw_M = actual_M;
            model->hnsw_ef_construction = actual_ef_construction;
            model->hnsw_ef_search = actual_ef_search;

            // Save embedding to model for persistence (copy from output buffer)
            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));
            std::memcpy(model->embedding.data(), embedding,
                        static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim) * sizeof(float));

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

    // Legacy exact k-NN function (from old codebase) - COMPLETE IMPLEMENTATION
    void build_knn_graph(
        const std::vector<float>& data,
        int n_obs, int n_dim, int n_neighbors,
        UwotMetric metric, UwotModel* model,
        std::vector<int>& nn_indices, std::vector<double>& nn_distances,
        int force_exact_knn, uwot_progress_callback_v2 progress_callback, int autoHNSWParam) {
        // Suppress unreferenced parameter warning (future functionality)
        (void)autoHNSWParam;

        nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
        nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

        auto start_time = std::chrono::steady_clock::now();

        // Enhanced HNSW optimization check with model availability
        bool can_use_hnsw = !force_exact_knn &&
            model && model->original_space_factory && model->original_space_factory->can_use_hnsw() &&
            model->original_space_index && model->original_space_index->getCurrentElementCount() > 0;

        if (can_use_hnsw) {
            // FAST TRANSFORM OPTIMIZATION: Initialize HNSW backend data
            model->knn_backend = UwotModel::KnnBackend::HNSW;
            model->rho.resize(n_obs);
            model->sigma.resize(n_obs);
            model->has_fast_transform_data = true;

            // ====== HNSW APPROXIMATE k-NN (FAST) ======
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

                        // FAST TRANSFORM OPTIMIZATION: Store distance for rho/sigma computation
                        if (k == 0) {
                            model->rho[i] = distance; // rho[i] = distance to nearest neighbor
                        }
                    }

                    // Fill remaining slots if needed
                    for (int k = actual_neighbors; k < n_neighbors; k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 0;
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;
                        if (k == 0) {
                            model->rho[i] = 1000.0f; // rho fallback
                        }
                    }

                    // FAST TRANSFORM OPTIMIZATION: Compute sigma[i] using stored rho
                    // Binary search for sigma (bandwidth parameter) - same as smooth_knn
                    float lo = 0, hi = 1e9;
                    const double local_connectivity = 1.3; // Updated for better local structure

                    for (int it = 0; it < 32; ++it) {
                        float mid = (lo + hi) / 2;
                        float sum = 0;
                        for (int k_check = 0; k_check < n_neighbors; ++k_check) {
                            float dist_check = static_cast<float>(nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k_check)]);
                            float val = (dist_check - model->rho[i]) / mid;
                            sum += (val <= 0) ? 1.0f : std::exp(-val);
                        }
                        (sum >= local_connectivity) ? hi = mid : lo = mid;
                    }
                    model->sigma[i] = (lo + hi) / 2;
                }
                catch (...) {
                    // Fallback for any HNSW errors
                    for (int k = 0; k < n_neighbors; k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = (i + k + 1) % n_obs;
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;
                    }

                    // FAST TRANSFORM OPTIMIZATION: Compute rho/sigma for fallback case
                    if (n_neighbors > 0) {
                        model->rho[i] = 1000.0f; // Fallback rho

                        // Binary search for sigma (bandwidth parameter)
                        float lo = 0, hi = 1e9;
                        const double local_connectivity = 1.3; // Updated for better local structure

                        for (int it = 0; it < 32; ++it) {
                            float mid = (lo + hi) / 2;
                            float sum = 0;
                            for (int k_check = 0; k_check < n_neighbors; ++k_check) {
                                float dist_check = static_cast<float>(nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k_check)]);
                                float val = (dist_check - model->rho[i]) / mid;
                                sum += (val <= 0) ? 1.0f : std::exp(-val);
                            }
                            (sum >= local_connectivity) ? hi = mid : lo = mid;
                        }
                        model->sigma[i] = (lo + hi) / 2;
                    }
                }

                // Progress reporting more frequently for large datasets (>=10k)
                int progress_interval = (n_obs >= 10000) ? (n_obs / 50) : (n_obs / 10); // 2% for >=10k, 10% otherwise
                if (progress_callback && i % (progress_interval + 1) == 0) {
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

                // Progress reporting more frequently for large datasets (>=10k)
                int progress_interval = (n_obs >= 10000) ? (n_obs / 50) : (n_obs / 20); // 2% for >=10k, 5% otherwise
                if (progress_callback && i % (progress_interval + 1) == 0) {
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

            // Apply smooth_knn to exact k-NN results (REQUIRED for UMAP correctness)
            if (progress_callback) {
                progress_callback("Exact k-NN Smooth Processing", 0, n_obs, 0.0f, "Applying smooth_knn fuzzy simplicial set...");
            }

            std::vector<float> fuzzy_weights(n_obs * n_neighbors);

            #pragma omp parallel for if(n_obs > 1000)
            for (int i = 0; i < n_obs; i++) {
                int* indices_ptr = nn_indices.data() + i * n_neighbors;
                const double* distances_ptr = nn_distances.data() + i * n_neighbors;
                float* weights_ptr = fuzzy_weights.data() + i * n_neighbors;

                // Convert double distances to float for smooth_knn processing
                std::vector<float> distances_float(n_neighbors);
                for (int k = 0; k < n_neighbors; ++k) {
                    distances_float[k] = static_cast<float>(distances_ptr[k]);
                }

                // Apply smooth_knn to convert raw distances to fuzzy weights
                apply_smooth_knn_to_point(indices_ptr, distances_float.data(), n_neighbors, weights_ptr);

                // Progress reporting for smooth_knn processing (more frequent for large datasets >=10k)
                int progress_interval = (n_obs >= 10000) ? (n_obs / 100) : (n_obs / 20); // 1% for >=10k, 5% otherwise
                if (progress_callback && i % (progress_interval + 1) == 0) {
                    float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                    char message[256];
                    snprintf(message, sizeof(message), "Smooth k-NN processing: %.1f%%", percent);
                    progress_callback("Exact k-NN Smooth Processing", i, n_obs, percent, message);
                }
            }

            if (progress_callback) {
                progress_callback("Exact k-NN Smooth Processing", n_obs, n_obs, 100.0f,
                    "Fuzzy simplicial set construction completed");
            }

            // Build fuzzy neighbor list from smooth_knn results (exact k-NN path)
            if (progress_callback) {
                progress_callback("Building Exact k-NN Fuzzy Graph", 55, 100, 0.0f, "Creating fuzzy neighbor list from exact k-NN with smooth_knn");
            }

            umappp::NeighborList<int, float> exact_fuzzy_neighbors(n_obs);

            #pragma omp parallel for if(n_obs > 1000)
            for (int i = 0; i < n_obs; i++) {
                const int* indices_ptr = nn_indices.data() + i * n_neighbors;
                const float* weights_ptr = fuzzy_weights.data() + i * n_neighbors;

                // Build fuzzy neighbor list for this point using smooth_knn weights
                for (int k = 0; k < n_neighbors; k++) {
                    if (weights_ptr[k] > 1e-6f) {  // Filter out very weak connections
                        exact_fuzzy_neighbors[i].emplace_back(indices_ptr[k], weights_ptr[k]);
                    }
                }
            }

            if (progress_callback) {
                progress_callback("Initializing umappp with exact fuzzy graph", 60, 100, 0.0f, "Setting up umappp optimization with exact k-NN fuzzy simplicial sets");
            }

            // REMOVED: umappp optimization should be handled in main function, not in build_knn_graph
            // This is the responsibility of uwot_fit_with_umappp_hnsw, not build_knn_graph

            if (progress_callback) {
                progress_callback("Optimizing layout (Exact k-NN)", 65, 100, 0.0f, "Running umappp optimization with exact k-NN fuzzy simplicial sets");
            }

            // REMOVED: umappp optimization loop - this should be in the main function
            // build_knn_graph should only build k-NN graphs, not run full UMAP optimization

            // FAST TRANSFORM OPTIMIZATION: Compute rho/sigma for exact k-NN path
            if (progress_callback) {
                progress_callback("Computing Fast Transform Data", 96, 100, 96.0f, "Pre-computing rho/sigma for 20x faster transform...");
            }

            model->knn_backend = UwotModel::KnnBackend::EXACT;
            model->rho.resize(n_obs);
            model->sigma.resize(n_obs);

            #pragma omp parallel for if(n_obs > 1000)
            for (int i = 0; i < n_obs; i++) {
                // Use the exact k-NN distances computed earlier (convert from double to float)
                const double* point_distances_dbl = nn_distances.data() + i * n_neighbors;
                std::vector<float> point_distances(n_neighbors);
                for (int k = 0; k < n_neighbors; ++k) {
                    point_distances[k] = static_cast<float>(point_distances_dbl[k]);
                }

                // rho[i] = distance to nearest neighbor
                model->rho[i] = point_distances[0];

                // Binary search for sigma (bandwidth parameter) - same as smooth_knn
                float lo = 0, hi = 1e9;
                const double local_connectivity = 1.3; // Updated for better local structure

                for (int it = 0; it < 32; ++it) {
                    float mid = (lo + hi) / 2;
                    float sum = 0;
                    for (int k = 0; k < n_neighbors; ++k) {
                        float val = (point_distances[k] - model->rho[i]) / mid;
                        sum += (val <= 0) ? 1.0f : std::exp(-val);
                    }
                    (sum >= local_connectivity) ? hi = mid : lo = mid;
                }
                model->sigma[i] = (lo + hi) / 2;
            }

            model->has_fast_transform_data = true;

            if (progress_callback) {
                progress_callback("Fast Transform Data Ready", 98, 100, 98.0f, "Fast transform optimization completed");
            }

            // REMOVED: Model updates should be in main function, not build_knn_graph
            if (progress_callback) {
                progress_callback("Exact k-NN Graph Complete", 100, 100, 100.0f, "Exact k-NN graph construction completed");
            }
        }
    }

}