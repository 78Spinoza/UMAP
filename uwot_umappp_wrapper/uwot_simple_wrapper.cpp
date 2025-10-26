#include "uwot_simple_wrapper.h"
#include "uwot_progress_utils.h"
#include "uwot_hnsw_utils.h"
#include "uwot_model.h"
#include "uwot_persistence.h"
#include "uwot_fit.h"
#include "uwot_transform.h"

#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Note: g_v2_callback is already declared in uwot_progress_utils.h

extern "C" {

    UWOT_API UwotModel* uwot_create() {
        return model_utils::create_model();
    }

    UWOT_API void uwot_destroy(UwotModel* model) {
        model_utils::destroy_model(model);
    }

    UWOT_API int uwot_fit_with_progress_v2(
        UwotModel* model,
        float* data,
        int64_t n_obs,        // LARGE DATASET SUPPORT (up to 2B observations)
        int64_t n_dim,         // Dimension support (max 50D)
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
        int random_seed,
        int autoHNSWParam
    ) {
        // Suppress unreferenced parameter warning (future functionality)
        (void)autoHNSWParam;

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 || embedding_dim <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // Overflow checks for n_obs and n_dim
        if (n_obs > INT_MAX || n_dim > INT_MAX) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // Check for overflow in n_obs * n_dim calculation
        if (n_dim > 0 && n_obs > SIZE_MAX / static_cast<size_t>(n_dim)) {
            return UWOT_ERROR_MEMORY;
        }

        // Cast to int for internal processing (still supports up to 2.1B points)
        int n_obs_int = static_cast<int>(n_obs);
        int n_dim_int = static_cast<int>(n_dim);

        // Apply C# parameters to model
        model->random_seed = random_seed;
        model->force_exact_knn = (force_exact_knn != 0);

        // Use the new umappp + HNSW implementation for better embeddings
        int result = fit_utils::uwot_fit_with_umappp_hnsw(model, data, n_obs_int, n_dim_int, embedding_dim, n_neighbors,
            min_dist, spread, n_epochs, metric, embedding, progress_callback,
            random_seed, M, ef_construction, ef_search);

        // CRITICAL: Auto-cleanup OpenMP threads after fit completes
        // This prevents segfault during DLL unload by ensuring all threads are terminated
        #ifdef _OPENMP
        omp_set_num_threads(1);
        omp_set_nested(0);
        omp_set_dynamic(0);
        // Force thread pool shutdown
        #pragma omp parallel
        {
            // Single-threaded region forces OpenMP runtime cleanup
        }
        #endif

        return result;
    }

    UWOT_API int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    ) {
        if (!model || !new_data || !embedding || n_new_obs <= 0 || n_dim <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (!model->is_fitted) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // Overflow check for n_new_obs * n_dim
        if (n_dim > 0 && static_cast<size_t>(n_new_obs) > SIZE_MAX / static_cast<size_t>(n_dim)) {
            return UWOT_ERROR_MEMORY;
        }
        return transform_utils::uwot_transform(model, new_data, n_new_obs, n_dim, embedding);
    }

    UWOT_API int uwot_transform_detailed(
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
        if (!model || !new_data || !embedding || n_new_obs <= 0 || n_dim <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // Overflow check for n_new_obs * n_dim
        if (n_dim > 0 && static_cast<size_t>(n_new_obs) > SIZE_MAX / static_cast<size_t>(n_dim)) {
            return UWOT_ERROR_MEMORY;
        }

        return transform_utils::uwot_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nn_indices, nn_distances, confidence_score, outlier_level, percentile_rank, z_score);
    }

    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        float* spread,
        UwotMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) {
            // Fix C4244: Explicit cast with bounds check
            if (model->n_vertices > std::numeric_limits<int>::max()) {
                *n_vertices = std::numeric_limits<int>::max();  // Clamp to max int
            } else {
                *n_vertices = static_cast<int>(model->n_vertices);
            }
        }
        if (n_dim) {
            // Fix C4244: Explicit cast with bounds check
            if (model->n_dim > std::numeric_limits<int>::max()) {
                *n_dim = std::numeric_limits<int>::max();  // Clamp to max int
            } else {
                *n_dim = static_cast<int>(model->n_dim);
            }
        }
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (spread) *spread = model->spread;
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_M;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        return UWOT_SUCCESS;
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        if (!model || !filename || filename[0] == '\0') {
            return UWOT_ERROR_INVALID_PARAMS;
        }
        return persistence_utils::save_model(model, filename);
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        if (!filename || filename[0] == '\0') {
            return nullptr;
        }
        return persistence_utils::load_model(filename);
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        return model_utils::get_error_message(error_code);
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        return model_utils::get_metric_name(metric);
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        if (!model) return -1;
        return model_utils::get_embedding_dim(model);
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        if (!model) return -1;
        return model_utils::get_n_vertices(model);
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        if (!model) return 0;
        return model_utils::is_fitted(model);
    }

    UWOT_API const char* uwot_get_version() {
        return model_utils::get_version();
    }

    UWOT_API void uwot_set_always_save_embedding_data(UwotModel* model, bool always_save) {
        if (model) {
            model->always_save_embedding_data = always_save;
        }
    }

    UWOT_API bool uwot_get_always_save_embedding_data(UwotModel* model) {
        if (!model) return false;
        return model->always_save_embedding_data;
    }

    UWOT_API int uwot_get_model_info_v2(
        UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        float* spread,
        UwotMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search,
        uint32_t* original_crc,
        uint32_t* embedding_crc,
        uint32_t* version_crc,
        float* hnsw_recall_percentage
    ) {
        return model_utils::get_model_info_v2(model, n_vertices, n_dim, embedding_dim,
            n_neighbors, min_dist, spread, metric, hnsw_M, hnsw_ef_construction, hnsw_ef_search,
            original_crc, embedding_crc, version_crc, hnsw_recall_percentage);
    }

    UWOT_API void uwot_set_global_callback(uwot_progress_callback_v2 callback) {
        g_v2_callback = callback;
    }

    UWOT_API void uwot_clear_global_callback() {
        g_v2_callback = nullptr;
    }

    // OpenMP cleanup function to prevent segfault on DLL unload
    UWOT_API void uwot_cleanup() {
        // Clean up OpenMP threads to prevent segfault on DLL unload
        #ifdef _OPENMP
        // CRITICAL: Force immediate shutdown of ALL OpenMP activity
        // This prevents any lingering threads from causing segfault

        // Step 1: Disable all parallelism immediately
        omp_set_num_threads(1);
        omp_set_nested(0);
        omp_set_dynamic(0);

        // Step 2: Execute a dummy parallel region to force thread pool shutdown
        // This ensures all worker threads are terminated before DLL unload
        #pragma omp parallel
        {
            // This single-threaded region forces OpenMP runtime to clean up
            // the thread pool and terminate worker threads
        }
        #endif
    }

} // extern "C"

// DLL process detach handler for clean OpenMP shutdown
#ifdef _WIN32
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    (void)hModule;      // Suppress unused parameter warning
    (void)lpReserved;   // Suppress unused parameter warning

    switch (ul_reason_for_call) {
    case DLL_PROCESS_DETACH:
        // Clean up OpenMP threads before DLL unload to prevent segfault
        #ifdef _OPENMP
        // Force complete OpenMP shutdown
        omp_set_num_threads(1);
        // Reset thread pool completely
        omp_set_nested(0);
        // Additional safety: disable dynamic thread adjustment
        omp_set_dynamic(0);
        #endif
        break;
    }
    return TRUE;
}
#endif