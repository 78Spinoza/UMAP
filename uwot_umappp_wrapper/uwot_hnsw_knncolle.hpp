#ifndef UWOT_HNSW_KNNCOLLE_HPP
#define UWOT_HNSW_KNNCOLLE_HPP

#include "hnswlib.h"
#include "knncolle/knncolle.hpp"
#include <vector>
#include <memory>

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
        std::vector<Float_> data_;  // Store copy of the data
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
            // Copy data from matrix using extractor
            data_.resize(static_cast<size_t>(num_obs) * num_dim);
            auto extractor = matrix.new_extractor();
            for (Index_ i = 0; i < num_obs; ++i) {
                const Float_* obs = extractor->next();
                std::copy(obs, obs + num_dim, data_.begin() + i * num_dim);
            }
        }

        const Float_* get_observation(Index_ i) const override {
            return data_.data() + i * num_dim_;
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

        // Use InnerProductSpace for cosine similarity
        // Note: hnswlib returns -dot_product, so distances are negative
        dist_space = new hnswlib::InnerProductSpace(num_dim);
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
        std::vector<Float_> data_;
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
            // Copy data from matrix
            data_.resize(static_cast<size_t>(num_obs) * num_dim);
            auto extractor = matrix.new_extractor();
            for (Index_ i = 0; i < num_obs; ++i) {
                const Float_* obs = extractor->next();
                std::copy(obs, obs + num_dim, data_.begin() + i * num_dim);
            }
        }

        const Float_* get_observation(Index_ i) const override {
            return data_.data() + i * num_dim_;
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

        // Use L1 space for Manhattan distance
        // Note: hnswlib doesn't have built-in L1, so we fall back to L2 for now
        // TODO: Implement custom L1Space if needed
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

        ManhattanHnswPrebuilt* result = new ManhattanHnswPrebuilt(hnsw_graph, dist_space, num_obs, num_dim, matrix);
        return result;
    }

private:
    // Concrete implementation for Manhattan distance
    class ManhattanHnswPrebuilt : public HnswPrebuilt<Index_, Float_> {
    private:
        std::vector<Float_> data_;
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
            // Copy data from matrix
            data_.resize(static_cast<size_t>(num_obs) * num_dim);
            auto extractor = matrix.new_extractor();
            for (Index_ i = 0; i < num_obs; ++i) {
                const Float_* obs = extractor->next();
                std::copy(obs, obs + num_dim, data_.begin() + i * num_dim);
            }
        }

        const Float_* get_observation(Index_ i) const override {
            return data_.data() + i * num_dim_;
        }
    };
};

} // namespace uwot

#endif // UWOT_HNSW_KNNCOLLE_HPP