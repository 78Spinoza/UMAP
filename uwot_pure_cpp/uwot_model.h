#pragma once

#include "uwot_simple_wrapper.h"
#include "uwot_hnsw_utils.h"
#include <memory>
#include <vector>

// Core UMAP model structure
struct UwotModel {
    // Model parameters
    int n_vertices;
    int n_dim;
    int embedding_dim;
    int n_neighbors;
    float min_dist;
    float spread; // UMAP spread parameter (controls global scale)
    UwotMetric metric;
    float a, b; // UMAP curve parameters
    bool is_fitted;
    bool force_exact_knn; // Override flag to force brute-force k-NN

    // Dual HNSW Indices for fast neighbor search
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> original_space_index;  // For fitting/training
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> embedding_space_index; // For AI inference
    std::unique_ptr<hnsw_utils::SpaceFactory> original_space_factory;       // Original metric space
    std::unique_ptr<hnsw_utils::SpaceFactory> embedding_space_factory;      // Always L2 space for embeddings

    // Normalization parameters (moved from C#)
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool use_normalization;
    int normalization_mode;

    // Graph structure using uwot types
    std::vector<unsigned int> positive_head;
    std::vector<unsigned int> positive_tail;
    std::vector<double> positive_weights;  // uwot uses double for weights

    // Final embedding
    std::vector<float> embedding;

    // k-NN structure for transformation (uwot format)
    std::vector<int> nn_indices;      // flattened indices
    std::vector<float> nn_distances;  // flattened distances
    std::vector<float> nn_weights;    // flattened weights for transform

    // Dual distance statistics for safety detection
    // Original space statistics (for fitting)
    float min_original_distance;
    float max_original_distance;
    float mean_original_distance;
    float std_original_distance;
    float p95_original_distance;
    float p99_original_distance;
    float median_original_distance;
    float mild_original_outlier_threshold;     // 2.5 std deviations
    float extreme_original_outlier_threshold;  // 4.0 std deviations
    float exact_match_threshold;               // Legacy exact match threshold
    float hnsw_recall_percentage;              // HNSW accuracy vs exact k-NN (0-100%)

    // Embedding space statistics (for AI inference)
    float min_embedding_distance;
    float max_embedding_distance;
    float mean_embedding_distance;
    float std_embedding_distance;
    float p95_embedding_distance;
    float p99_embedding_distance;
    float mild_embedding_outlier_threshold;      // 2.5 std deviations
    float extreme_embedding_outlier_threshold;   // 4.0 std deviations
    float median_embedding_distance;            // For robust bandwidth scaling
    float exact_embedding_match_threshold;       // Robust exact-match detection threshold

    // HNSW hyperparameters
    int hnsw_M;                        // Graph degree parameter (16-64)
    int hnsw_ef_construction;          // Build quality parameter (64-256)
    int hnsw_ef_search;                // Query quality parameter (32-128)
    bool use_quantization;             // Enable Product Quantization

    // Reproducibility parameters
    int random_seed;                   // Random seed for reproducible training (-1 = random)

    // Product Quantization data structures
    std::vector<uint8_t> pq_codes;     // Quantized vector codes (n_vertices * pq_m bytes)
    std::vector<float> pq_centroids;   // PQ codebook (pq_m * 256 * subspace_dim floats)
    int pq_m;                          // Number of subspaces (default: 4)

    // Embedding data preservation option
    bool always_save_embedding_data;    // When true, save original embeddings and rebuild HNSW on load

    // Model integrity validation
    uint32_t original_space_crc;       // CRC32 of original space HNSW index
    uint32_t embedding_space_crc;      // CRC32 of embedding space HNSW index
    uint32_t model_version_crc;        // CRC32 of model structure version

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), spread(1.0f), metric(UWOT_METRIC_EUCLIDEAN), a(1.929f), b(0.7915f),
        is_fitted(false), force_exact_knn(false), use_normalization(false),
        // Original space statistics
        min_original_distance(0.0f), max_original_distance(0.0f), mean_original_distance(0.0f),
        std_original_distance(0.0f), p95_original_distance(0.0f), p99_original_distance(0.0f),
        median_original_distance(0.0f), mild_original_outlier_threshold(0.0f),
        extreme_original_outlier_threshold(0.0f), exact_match_threshold(1e-3f), hnsw_recall_percentage(0.0f),
        // Embedding space statistics
        min_embedding_distance(0.0f), max_embedding_distance(0.0f), mean_embedding_distance(0.0f),
        std_embedding_distance(0.0f), p95_embedding_distance(0.0f),
        p99_embedding_distance(0.0f), mild_embedding_outlier_threshold(0.0f),
        extreme_embedding_outlier_threshold(0.0f), median_embedding_distance(0.0f),
        exact_embedding_match_threshold(1e-3f), hnsw_M(32), hnsw_ef_construction(128),
        hnsw_ef_search(64), use_quantization(true), random_seed(-1), pq_m(4), normalization_mode(1),
        // Embedding data preservation
        always_save_embedding_data(false),
        // CRC32 validation
        original_space_crc(0), embedding_space_crc(0), model_version_crc(0x5A4D4F44) { // "UMOD" hex

        original_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
        embedding_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
    }
};

// Model lifecycle functions
namespace model_utils {
    UwotModel* create_model();
    void destroy_model(UwotModel* model);

    // Model information functions
    int get_model_info(UwotModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, UwotMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search);

    // Enhanced model information with dual HNSW indices
    int get_model_info_v2(UwotModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, UwotMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search,
        uint32_t* original_crc, uint32_t* embedding_crc, uint32_t* version_crc,
        float* hnsw_recall_percentage);

    // Utility functions
    int get_embedding_dim(UwotModel* model);
    int get_n_vertices(UwotModel* model);
    int is_fitted(UwotModel* model);

    // Error handling
    const char* get_error_message(int error_code);
    const char* get_metric_name(UwotMetric metric);
    const char* get_version();
}