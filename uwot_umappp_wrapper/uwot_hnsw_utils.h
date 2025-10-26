#pragma once

#include "uwot_simple_wrapper.h"
#include "hnswlib.h"
#include "hnswalg.h"
#include "space_l2.h"
#include "space_ip.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>

// Custom L1Space implementation for Manhattan distance in HNSW
class L1Space : public hnswlib::SpaceInterface<float> {
    hnswlib::DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    L1Space(size_t dim) : dim_(dim) {
        fstdistfunc_ = L1Sqr;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void* get_dist_func_param() override {
        return &dim_;
    }

    ~L1Space() {}

    static float L1Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;
        size_t qty = *((size_t*)qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = pVect1[i] - pVect2[i];
            res += std::abs(t);
        }
        return res;
    }
};

// CRITICAL FIX: Custom Cosine Space for HNSW
// InnerProductSpace returns dot product (higher = more similar), but HNSW expects smaller distances = closer
// We need to negate the dot product so larger similarity = smaller "distance"
class CosineSpace : public hnswlib::SpaceInterface<float> {
private:
    size_t dim_;

public:
    CosineSpace(size_t dim) : dim_(dim) {}

    size_t get_data_size() override {
        return dim_ * sizeof(float);
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return [](const void* p1, const void* p2, const void* dim_ptr) -> float {
            const float* x = static_cast<const float*>(p1);
            const float* y = static_cast<const float*>(p2);
            const size_t dim = *static_cast<const size_t*>(dim_ptr);

            // Calculate dot product
            float dot = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                dot += x[i] * y[i];
            }

            // KEY FIX: Return negative dot product
            // Higher similarity (larger dot) = smaller distance
            return -dot;
        };
    }

    void* get_dist_func_param() override {
        return &dim_;
    }

    ~CosineSpace() {}
};

// HNSW space factory and management utilities
namespace hnsw_utils {

    // Space factory - creates appropriate space based on metric type
    struct SpaceFactory {
        std::unique_ptr<hnswlib::L2Space> l2_space;
        std::unique_ptr<hnswlib::InnerProductSpace> ip_space;
        std::unique_ptr<L1Space> l1_space;
        std::unique_ptr<CosineSpace> cosine_space;
        UwotMetric current_metric;
        int current_dim;

        SpaceFactory() : current_metric(UWOT_METRIC_EUCLIDEAN), current_dim(0) {}

        bool create_space(UwotMetric metric, int n_dim);
        hnswlib::SpaceInterface<float>* get_space();
        bool can_use_hnsw() const;
    };

    // HNSW stream utilities
    namespace hnsw_stream_utils {
        std::string generate_unique_temp_filename(const std::string& prefix);
        void save_hnsw_to_stream(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index);
        void load_hnsw_from_stream(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
            hnswlib::SpaceInterface<float>* space, size_t hnsw_size);
    }

    // HNSW compression utilities
    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index);
    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space);

    // HNSW k-NN query utilities
    void build_knn_graph_hnsw(const std::vector<float>& data, int n_obs, int n_dim, int n_neighbors,
        hnswlib::HierarchicalNSW<float>* hnsw_index, std::vector<int>& nn_indices,
        std::vector<double>& nn_distances);

    // HNSW index creation and management
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> create_hnsw_index(
        hnswlib::SpaceInterface<float>* space, int n_obs, int hnsw_M, int hnsw_ef_construction, int hnsw_ef_search);

    void add_points_to_hnsw(hnswlib::HierarchicalNSW<float>* hnsw_index,
        const std::vector<float>& normalized_data, int n_obs, int n_dim);

    // Error callback management for HNSW operations
    void set_hnsw_error_callback(uwot_progress_callback_v2 callback);
    void report_hnsw_error(const std::string& error_message);

    // Temporary normalization utilities (will be moved to separate module later)
    namespace NormalizationPipeline {
        int determine_normalization_mode(UwotMetric metric);
        bool normalize_data_consistent(std::vector<float>& input_data, std::vector<float>& output_data,
            int n_obs, int n_dim, std::vector<float>& means, std::vector<float>& stds, int mode);
    }
}