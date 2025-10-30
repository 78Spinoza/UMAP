#ifndef UWOT_HNSW_KNNCOLLE_HPP
#define UWOT_HNSW_KNNCOLLE_HPP

#include "hnswlib.h"
#include "knncolle/knncolle.hpp"
#include "smooth_knn.h"
#include "NeighborList.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <cmath>
#include <omp.h>

namespace uwot {

// Forward declaration
template<typename Index_, typename Float_>
class HnswPrebuilt;

/**
 * Searcher implementation for HNSW
 */
template<typename Index_, typename Float_>
class HnswSearcher : public knncolle::Searcher<Index_, Float_, Float_> {
private:
    const HnswPrebuilt<Index_, Float_>* parent_;

public:
    HnswSearcher(const HnswPrebuilt<Index_, Float_>* parent) : parent_(parent) {}

    void search(Index_ i, Index_ k, std::vector<Index_>* indices, std::vector<Float_>* distances) override {
        // Search for k-nearest neighbors using HNSW
        auto result = parent_->hnsw_index_->searchKnn(
            const_cast<void*>(static_cast<const void*>(parent_->get_observation(i))),
            static_cast<size_t>(k)
        );

        if (indices) indices->clear();
        if (distances) distances->clear();

        // Priority queue returns results in reverse order (farthest first)
        std::vector<std::pair<Float_, Index_>> temp;
        while (!result.empty()) {
            temp.push_back(result.top());
            result.pop();
        }

        // Reverse to get nearest first
        for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
            if (indices) indices->push_back(static_cast<Index_>(it->second));
            if (distances) distances->push_back(it->first);
        }
    }

    void search(const Float_* query, Index_ k, std::vector<Index_>* indices, std::vector<Float_>* distances) override {
        auto result = parent_->hnsw_index_->searchKnn(const_cast<Float_*>(query), static_cast<size_t>(k));

        if (indices) indices->clear();
        if (distances) distances->clear();

        std::vector<std::pair<Float_, Index_>> temp;
        while (!result.empty()) {
            temp.push_back(result.top());
            result.pop();
        }

        for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
            if (indices) indices->push_back(static_cast<Index_>(it->second));
            if (distances) distances->push_back(it->first);
        }
    }

    bool can_search_all() const override {
        return true;
    }

    Index_ search_all(Index_ i, Float_ threshold, std::vector<Index_>* indices, std::vector<Float_>* distances) override {
        const Index_ large_k = std::min(static_cast<Index_>(100), parent_->num_observations());
        auto result = parent_->hnsw_index_->searchKnn(
            const_cast<void*>(static_cast<const void*>(parent_->get_observation(i))),
            static_cast<size_t>(large_k)
        );

        if (indices) indices->clear();
        if (distances) distances->clear();

        Index_ count = 0;
        while (!result.empty()) {
            auto pair = result.top();
            result.pop();
            if (pair.first <= threshold) {
                if (indices) indices->push_back(static_cast<Index_>(pair.second));
                if (distances) distances->push_back(pair.first);
                count++;
            }
        }
        return count;
    }
};

/**
 * Custom knncolle Prebuilt interface that wraps HNSW for use with umappp
 */
template<typename Index_, typename Float_>
class HnswPrebuilt : public knncolle::Prebuilt<Index_, Float_, Float_> {
public:
    std::unique_ptr<hnswlib::HierarchicalNSW<Float_>> hnsw_index_;
    std::unique_ptr<hnswlib::SpaceInterface<Float_>> space_;

private:
    Index_ num_obs_;
    std::size_t num_dim_;

public:
    HnswPrebuilt(
        hnswlib::HierarchicalNSW<Float_>* hnsw_index,
        hnswlib::SpaceInterface<Float_>* space,
        Index_ num_obs,
        std::size_t num_dim
    ) : hnsw_index_(hnsw_index), space_(space), num_obs_(num_obs), num_dim_(num_dim) {}

    virtual ~HnswPrebuilt() = default;

    Index_ num_observations() const override {
        return num_obs_;
    }

    std::size_t num_dimensions() const override {
        return num_dim_;
    }

    std::unique_ptr<knncolle::Searcher<Index_, Float_, Float_>> initialize() const override {
        return std::make_unique<HnswSearcher<Index_, Float_>>(this);
    }

    // Method to get the raw data pointer for observation i
    virtual const Float_* get_observation(Index_ i) const = 0;

    // Release ownership of HNSW index and space for external use (e.g., for transform)
    // After calling this, the wrapper no longer owns these resources
    std::pair<std::unique_ptr<hnswlib::HierarchicalNSW<Float_>>, std::unique_ptr<hnswlib::SpaceInterface<Float_>>> release_index() {
        return std::make_pair(std::move(hnsw_index_), std::move(space_));
    }

    // Friend class to allow searcher access to private members
    friend class HnswSearcher<Index_, Float_>;
};

/**
 * Builder class for creating HNSW-based indices
 */
template<typename Index_, typename Float_>
class HnswBuilder : public knncolle::Builder<Index_, Float_, Float_, knncolle::SimpleMatrix<Index_, Float_>> {
protected:
    int M_;
    int ef_construction_;
    int ef_search_;

public:
    HnswBuilder(int M = 16, int ef_construction = 200, int ef_search = 50)
        : M_(M), ef_construction_(ef_construction), ef_search_(ef_search) {}

    virtual ~HnswBuilder() = default;

    // The actual HNSW index creation will be done by the derived classes
    // that know the specific distance metric and data layout
};

/**
 * Specialized builder for Euclidean distance
 */
template<typename Index_, typename Float_>
class HnswEuclideanBuilder : public HnswBuilder<Index_, Float_> {
public:
    HnswEuclideanBuilder(int M = 16, int ef_construction = 200, int ef_search = 50)
        : HnswBuilder<Index_, Float_>(M, ef_construction, ef_search) {}

    knncolle::Prebuilt<Index_, Float_, Float_>* build_raw(const knncolle::SimpleMatrix<Index_, Float_>& matrix) const override {
        const std::size_t num_dim = matrix.num_dimensions();
        const Index_ num_obs = matrix.num_observations();

        hnswlib::SpaceInterface<Float_>* dist_space = nullptr;
        hnswlib::HierarchicalNSW<Float_>* hnsw_graph = nullptr;

        dist_space = new hnswlib::L2Space(num_dim);
        hnsw_graph = new hnswlib::HierarchicalNSW<Float_>(dist_space, num_obs, this->M_, this->ef_construction_);

        // Extract data using the new iterator pattern
        auto extractor = matrix.new_extractor();
        for (Index_ i = 0; i < num_obs; ++i) {
            const Float_* data_ptr = extractor->next();
            hnsw_graph->addPoint(const_cast<Float_*>(data_ptr), static_cast<hnswlib::labeltype>(i));
        }

        // Set ef_search parameter
        hnsw_graph->setEf(this->ef_search_);

        EuclideanHnswPrebuilt* result = new EuclideanHnswPrebuilt(hnsw_graph, dist_space, num_obs, num_dim, matrix);
        return result;
    }

private:
    // Concrete implementation that stores the matrix data
    class EuclideanHnswPrebuilt : public HnswPrebuilt<Index_, Float_> {
    private:
        const Float_* data_ptr_;  // Store pointer to original data (no copy)
        Index_ num_obs_;
        std::size_t num_dim_;

    public:
        EuclideanHnswPrebuilt(
            hnswlib::HierarchicalNSW<Float_>* hnsw_index,
            hnswlib::SpaceInterface<Float_>* space,
            Index_ num_obs,
            std::size_t num_dim,
            const knncolle::SimpleMatrix<Index_, Float_>& matrix
        ) : HnswPrebuilt<Index_, Float_>(hnsw_index, space, num_obs, num_dim),
            num_obs_(num_obs), num_dim_(num_dim) {
            // Get pointer to original data - no copying needed
            auto extractor = matrix.new_extractor();
            data_ptr_ = extractor->next();
        }

        const Float_* get_observation(Index_ i) const override {
            return data_ptr_ + i * num_dim_;
        }
    };
};

/**
 * Specialized builder for Cosine distance (using Inner Product space)
 */
template<typename Index_, typename Float_>
class HnswCosineBuilder : public HnswBuilder<Index_, Float_> {
public:
    HnswCosineBuilder(int M = 16, int ef_construction = 200, int ef_search = 50)
        : HnswBuilder<Index_, Float_>(M, ef_construction, ef_search) {}

    knncolle::Prebuilt<Index_, Float_, Float_>* build_raw(const knncolle::SimpleMatrix<Index_, Float_>& matrix) const override {
        const std::size_t num_dim = matrix.num_dimensions();
        const Index_ num_obs = matrix.num_observations();

        hnswlib::SpaceInterface<Float_>* dist_space = nullptr;
        hnswlib::HierarchicalNSW<Float_>* hnsw_graph = nullptr;

        // Use custom CosineSpace for correct cosine distance calculation
        // CosineSpace returns negative dot product so HNSW can minimize it
        dist_space = new CosineSpace(num_dim);
        hnsw_graph = new hnswlib::HierarchicalNSW<Float_>(dist_space, num_obs, this->M_, this->ef_construction_);

        // Extract data using the new iterator pattern
        auto extractor = matrix.new_extractor();
        for (Index_ i = 0; i < num_obs; ++i) {
            const Float_* data_ptr = extractor->next();
            hnsw_graph->addPoint(const_cast<Float_*>(data_ptr), static_cast<hnswlib::labeltype>(i));
        }

        // Set ef_search parameter
        hnsw_graph->setEf(this->ef_search_);

        CosineHnswPrebuilt* result = new CosineHnswPrebuilt(hnsw_graph, dist_space, num_obs, num_dim, matrix);
        return result;
    }

private:
    // Concrete implementation for Cosine distance
    class CosineHnswPrebuilt : public HnswPrebuilt<Index_, Float_> {
    private:
        const Float_* data_ptr_;  // Store pointer to original data (no copy)
        Index_ num_obs_;
        std::size_t num_dim_;

    public:
        CosineHnswPrebuilt(
            hnswlib::HierarchicalNSW<Float_>* hnsw_index,
            hnswlib::SpaceInterface<Float_>* space,
            Index_ num_obs,
            std::size_t num_dim,
            const knncolle::SimpleMatrix<Index_, Float_>& matrix
        ) : HnswPrebuilt<Index_, Float_>(hnsw_index, space, num_obs, num_dim),
            num_obs_(num_obs), num_dim_(num_dim) {
            // Get pointer to original data - no copying needed
            auto extractor = matrix.new_extractor();
            data_ptr_ = extractor->next();
        }

        const Float_* get_observation(Index_ i) const override {
            return data_ptr_ + i * num_dim_;
        }
    };
};

/**
 * Specialized builder for Manhattan distance (L1 norm)
 */
template<typename Index_, typename Float_>
class HnswManhattanBuilder : public HnswBuilder<Index_, Float_> {
public:
    HnswManhattanBuilder(int M = 16, int ef_construction = 200, int ef_search = 50)
        : HnswBuilder<Index_, Float_>(M, ef_construction, ef_search) {}

    knncolle::Prebuilt<Index_, Float_, Float_>* build_raw(const knncolle::SimpleMatrix<Index_, Float_>& matrix) const override {
        const std::size_t num_dim = matrix.num_dimensions();
        const Index_ num_obs = matrix.num_observations();

        hnswlib::SpaceInterface<Float_>* dist_space = nullptr;
        hnswlib::HierarchicalNSW<Float_>* hnsw_graph = nullptr;

        // Use custom L1Space for correct Manhattan distance calculation
        // L1Space computes sum of absolute differences (L1 norm)
        dist_space = new L1Space(num_dim);
        hnsw_graph = new hnswlib::HierarchicalNSW<Float_>(dist_space, num_obs, this->M_, this->ef_construction_);

        // Extract data using the new iterator pattern
        auto extractor = matrix.new_extractor();
        for (Index_ i = 0; i < num_obs; ++i) {
            const Float_* data_ptr = extractor->next();
            hnsw_graph->addPoint(const_cast<Float_*>(data_ptr), static_cast<hnswlib::labeltype>(i));
        }

        // Set ef_search parameter
        hnsw_graph->setEf(this->ef_search_);

        ManhattanHnswPrebuilt* result = new ManhattanHnswPrebuilt(hnsw_graph, dist_space, num_obs, num_dim, matrix);
        return result;
    }

private:
    // Concrete implementation for Manhattan distance
    class ManhattanHnswPrebuilt : public HnswPrebuilt<Index_, Float_> {
    private:
        const Float_* data_ptr_;  // Store pointer to original data (no copy)
        Index_ num_obs_;
        std::size_t num_dim_;

    public:
        ManhattanHnswPrebuilt(
            hnswlib::HierarchicalNSW<Float_>* hnsw_index,
            hnswlib::SpaceInterface<Float_>* space,
            Index_ num_obs,
            std::size_t num_dim,
            const knncolle::SimpleMatrix<Index_, Float_>& matrix
        ) : HnswPrebuilt<Index_, Float_>(hnsw_index, space, num_obs, num_dim),
            num_obs_(num_obs), num_dim_(num_dim) {
            // Get pointer to original data - no copying needed
            auto extractor = matrix.new_extractor();
            data_ptr_ = extractor->next();
        }

        const Float_* get_observation(Index_ i) const override {
            return data_ptr_ + i * num_dim_;
        }
    };
};

/**
 * Fuzzy k-NN Searcher that applies smooth_knn preprocessing
 * This wrapper transforms raw HNSW distances into fuzzy simplicial sets
 * compatible with umappp's expected input format
 */
template<typename Index_, typename Float_>
class FuzzyKnnSearcher {
private:
    std::unique_ptr<HnswSearcher<Index_, Float_>> base_searcher_;
    int n_neighbors_;
    double local_connectivity_;
    bool parallel_;

public:
    FuzzyKnnSearcher(std::unique_ptr<HnswSearcher<Index_, Float_>> base_searcher,
                    int n_neighbors = 15,
                    double local_connectivity = 1.0,
                    bool parallel = true)
        : base_searcher_(std::move(base_searcher)),
          n_neighbors_(n_neighbors),
          local_connectivity_(local_connectivity),
          parallel_(parallel) {}

    ~FuzzyKnnSearcher() = default;

    // Search for a single observation and return fuzzy neighbors
    void search_fuzzy(Index_ query_idx, std::vector<Index_>* indices, std::vector<Float_>* weights) {
        if (!indices || !weights) return;

        indices->clear();
        weights->clear();

        // 1. Get raw k-NN from HNSW
        std::vector<Index_> raw_indices;
        std::vector<Float_> raw_distances;
        base_searcher_->search(query_idx, n_neighbors_, &raw_indices, &raw_distances);

        if (raw_indices.empty()) return;

        // 2. Apply smooth_knn to convert distances to fuzzy weights
        indices->resize(raw_indices.size());
        weights->resize(raw_indices.size());

        apply_smooth_knn(raw_indices.data(), raw_distances.data(),
                        static_cast<Index_>(raw_indices.size()),
                        indices->data(), weights->data());
    }

    // Search for multiple observations (batch processing with optional parallelization)
    umappp::NeighborList<Index_, Float_> search_batch(const Float_* query_data, Index_ n_queries, Index_ n_dim) {
        umappp::NeighborList<Index_, Float_> result(n_queries);

        if (parallel_ && n_queries > 1) {
            // Parallel processing for multiple queries
            #pragma omp parallel for if(n_queries > 4)
            for (Index_ q = 0; q < n_queries; ++q) {
                std::vector<Index_> indices;
                std::vector<Float_> weights;

                // Search using query data point
                search_fuzzy_query(&query_data[q * n_dim], n_dim, &indices, &weights);

                // Build neighbor list for this query
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (weights[i] > 1e-6f) {  // Filter out very weak connections
                        result[q].emplace_back(indices[i], weights[i]);
                    }
                }
            }
        } else {
            // Sequential processing
            for (Index_ q = 0; q < n_queries; ++q) {
                std::vector<Index_> indices;
                std::vector<Float_> weights;

                search_fuzzy_query(&query_data[q * n_dim], n_dim, &indices, &weights);

                for (size_t i = 0; i < indices.size(); ++i) {
                    if (weights[i] > 1e-6f) {
                        result[q].emplace_back(indices[i], weights[i]);
                    }
                }
            }
        }

        return result;
    }

private:
    // Search for a query point by coordinate (instead of index)
    void search_fuzzy_query(const Float_* query, Index_ n_dim, std::vector<Index_>* indices, std::vector<Float_>* weights) {
        if (!indices || !weights) return;

        indices->clear();
        weights->clear();

        // 1. Get raw k-NN from HNSW
        std::vector<Index_> raw_indices;
        std::vector<Float_> raw_distances;
        base_searcher_->search(query, n_neighbors_, &raw_indices, &raw_distances);

        if (raw_indices.empty()) return;

        // 2. Apply smooth_knn to convert distances to fuzzy weights
        indices->resize(raw_indices.size());
        weights->resize(raw_indices.size());

        apply_smooth_knn(raw_indices.data(), raw_distances.data(),
                        static_cast<Index_>(raw_indices.size()),
                        indices->data(), weights->data());
    }

    // Apply smooth_knn to convert raw distances to fuzzy weights
    void apply_smooth_knn(const Index_* indices, const Float_* distances, Index_ k,
                         Index_* out_indices, Float_* out_weights) {

        // Convert distances to double for smooth_knn
        std::vector<double> dist_double(k);
        std::vector<double> weights_double(k);

        for (Index_ i = 0; i < k; ++i) {
            dist_double[i] = static_cast<double>(distances[i]);
            out_indices[i] = indices[i];
        }

        // Apply smooth_knn algorithm
        constexpr double tol = 1e-6;
        constexpr size_t n_iter = 64;
        constexpr double min_k_dist_scale = 1.0;
        std::vector<double> nn_ptr = { static_cast<size_t>(k) };  // Fixed k neighbors per point
        std::vector<double> target = { static_cast<double>(n_neighbors_) };
        std::vector<double> sigmas(k);
        std::vector<double> rhos(k);
        std::atomic_size_t n_search_fails{0};

        // smooth_knn expects the entire distance matrix, but we're processing one point at a time
        // So we need to adapt the interface
        try {
            // Create fuzzy weights using simplified smooth_knn for single point
            compute_fuzzy_weights_single_point(dist_double.data(), k, weights_double.data());
        } catch (...) {
            // Fallback: use simple exponential decay if smooth_knn fails
            for (Index_ i = 0; i < k; ++i) {
                weights_double[i] = std::exp(-dist_double[i]);
            }
        }

        // Convert back to float and normalize
        double weight_sum = 0.0;
        for (Index_ i = 0; i < k; ++i) {
            weight_sum += weights_double[i];
        }

        if (weight_sum > 0.0) {
            for (Index_ i = 0; i < k; ++i) {
                out_weights[i] = static_cast<Float_>(weights_double[i] / weight_sum);
            }
        } else {
            // Fallback: uniform weights
            for (Index_ i = 0; i < k; ++i) {
                out_weights[i] = 1.0f / static_cast<Float_>(k);
            }
        }
    }

    // Simplified fuzzy weight computation for single point
    void compute_fuzzy_weights_single_point(const double* distances, Index_ k, double* weights) {
        // Find rho (smallest non-zero distance)
        double rho = 0.0;
        for (Index_ i = 0; i < k; ++i) {
            if (distances[i] > 0.0) {
                rho = distances[i];
                break;
            }
        }

        // Find sigma using binary search to achieve target sum
        double target_sum = static_cast<double>(n_neighbors_);
        double sigma = 1.0;
        constexpr int max_iter = 64;
        constexpr double tol = 1e-6;

        for (int iter = 0; iter < max_iter; ++iter) {
            double sum = 0.0;
            for (Index_ i = 0; i < k; ++i) {
                double r = distances[i] - rho;
                sum += (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
            }

            if (std::abs(sum - target_sum) < tol) break;

            if (sum > target_sum) {
                sigma *= 0.5;
            } else {
                sigma *= 2.0;
            }
        }

        // Compute final weights
        for (Index_ i = 0; i < k; ++i) {
            double r = distances[i] - rho;
            weights[i] = (r <= 0.0) ? 1.0 : std::exp(-r / sigma);
        }
    }
};

} // namespace uwot

#endif // UWOT_HNSW_KNNCOLLE_HPP