#pragma once

#include "uwot_model.h"
#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include "uwot_distance.h"
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fit_utils {

    // Import distance metrics from dedicated module
    using namespace distance_metrics;

    // Calculate UMAP parameters from spread and min_dist
    void calculate_ab_from_spread_and_min_dist(UwotModel* model);

    // Calculate epochs per sample for intelligent edge scheduling
    std::vector<float> make_epochs_per_sample(const std::vector<double>& weights, int n_epochs);

    // umappp + HNSW implementation
    int uwot_fit_with_umappp_hnsw(
        UwotModel* model,
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
        int random_seed = -1,
        int M = 16,
        int ef_construction = 200,
        int ef_search = 50,
        int useQuantization = 0
    );

    // Helper functions for uwot_fit refactoring
    namespace fit_helpers {
        // HNSW recall validation and auto-tuning
        bool validate_hnsw_recall(UwotModel* model, const float* data, int n_obs, int n_dim,
                                  int n_neighbors, UwotMetric metric, uwot_progress_callback_v2 progress_callback);

        // Auto-tune ef_search parameter based on recall measurement
        bool auto_tune_ef_search(UwotModel* model, const float* data, int n_obs, int n_dim,
                                 int n_neighbors, UwotMetric metric, uwot_progress_callback_v2 progress_callback);

        // Initialize random number generators with seed
        void initialize_random_generators(UwotModel* model);

        // Legacy exact k-NN function (from old codebase)
        void build_knn_graph(
            const std::vector<float>& data,
            int n_obs, int n_dim, int n_neighbors,
            UwotMetric metric, UwotModel* model,
            std::vector<int>& nn_indices,
            std::vector<double>& nn_distances,
            int force_exact_knn,
            uwot_progress_callback_v2 progress_callback = nullptr,
            int autoHNSWParam = 1
        );
    }
}