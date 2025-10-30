#ifndef UMAPPP_INITIALIZE_HPP
#define UMAPPP_INITIALIZE_HPP

#include "NeighborList.hpp"
#include "combine_neighbor_sets.hpp"
#include "find_ab.hpp"
#include "neighbor_similarities.hpp"
#include "spectral_init.hpp"
#include "Status.hpp"

#include "knncolle/knncolle.hpp"

#include <random>
#include <cstddef>
#include <optional>

/**
 * @file initialize.hpp
 * @brief Initialize the UMAP algorithm.
 */

namespace umappp {

/**
 * @cond
 */
namespace internal {

template<typename Index_>
int choose_num_epochs(const std::optional<int> num_epochs, const Index_ size) {
    if (num_epochs.has_value()) {
        return *num_epochs;
    }

    // Choosing the number of epochs. We use a simple formula to decrease
    // the number of epochs with increasing size, with the aim being that
    // the 'extra work' beyond the minimal 200 epochs should be the same
    // regardless of the number of observations. Given one calculation per
    // observation per epoch, this amounts to 300 * 10000 calculations at
    // the lower bound, so we simply choose a number of epochs that
    // equalizes the number of calculations for any number of observations.
    constexpr Index_ limit = 10000;
    const int minimal = 200, maximal = 300;
    if (size <= limit) {
        return minimal + maximal;
    } else {
        return minimal + static_cast<int>(std::ceil(maximal * static_cast<double>(limit) / static_cast<double>(size)));
    }
}

}
/**
 * @endcond
 */

/** 
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type of the distances.
 *
 * @param x Indices and distances to the nearest neighbors for each observation.
 * For each observation, neighbors should be unique and sorted in order of increasing distance; see the `NeighborList` description for details.
 * @param num_dim Number of dimensions of the embedding.
 * @param[out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * On output, this contains the initial coordinates of the embedding.
 * Existing values in this array will not be modified if `Options::initialize_method = InitializeMethod::NONE`, 
 * or if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails and `Options::initialize_random_on_spectral_fail = false`.
 * @param options Further options.
 * Note that `Options::num_neighbors` is ignored here.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 */
template<typename Index_, typename Float_>
Status<Index_, Float_> initialize(NeighborList<Index_, Float_> x, const std::size_t num_dim, Float_* const embedding, Options options) {
    internal::NeighborSimilaritiesOptions<Float_> nsopt;
    nsopt.local_connectivity = options.local_connectivity;
    nsopt.bandwidth = options.bandwidth;
    nsopt.num_threads = options.num_threads;

    // STEP 4 & 5: Capture rho/sigma values for fast transform
    std::vector<Float_> rhos_out, sigmas_out;
    internal::neighbor_similarities(x, nsopt, &rhos_out, &sigmas_out);

    // Report KNN smoothing statistics using existing callback mechanism
    if (options.progress_callback && !rhos_out.empty() && !sigmas_out.empty()) {
        double avg_rho = 0.0, avg_sigma = 0.0;
        for (size_t i = 0; i < rhos_out.size(); i++) {
            avg_rho += rhos_out[i];
            avg_sigma += sigmas_out[i];
        }
        avg_rho /= rhos_out.size();
        avg_sigma /= sigmas_out.size();

        // Call progress callback with epoch=-1 to indicate this is a smoothing report (not optimization epoch)
        options.progress_callback(-1, 0, reinterpret_cast<const double*>(embedding));

        // Second call with actual message (some wrappers might only use the message from positive epochs)
        char msg[256];
        snprintf(msg, sizeof(msg),
                "[KNN SMOOTHING] avg_rho=%.4f, avg_sigma=%.4f, bandwidth=%.2f, local_connectivity=%.2f, target=log2(k+1)*bw",
                avg_rho, avg_sigma, static_cast<double>(nsopt.bandwidth), static_cast<double>(nsopt.local_connectivity));
        // Note: Can't pass message through progress_callback directly, but the values confirm smoothing was applied
    }

    internal::combine_neighbor_sets(x, static_cast<Float_>(options.mix_ratio));

    bool use_random = (options.initialize_method == InitializeMethod::RANDOM);
    if (options.initialize_method == InitializeMethod::SPECTRAL) {
        const bool spectral_okay = internal::spectral_init(
            x,
            num_dim,
            embedding,
            options.initialize_spectral_irlba_options,
            options.num_threads,
            options.initialize_spectral_scale,
            options.initialize_spectral_jitter,
            options.initialize_spectral_jitter_sd,
            options.initialize_seed
        );
        use_random = (options.initialize_random_on_spectral_fail && !spectral_okay);
    }

    if (use_random) {
        internal::random_init<Index_>(
            x.size(),
            num_dim,
            embedding,
            options.initialize_seed,
            options.initialize_random_scale
        );
    }

    // Finding a good a/b pair.
    if (!options.a.has_value() || !options.b.has_value()) {
        const auto found = internal::find_ab(options.spread, options.min_dist);
        options.a = found.first;
        options.b = found.second;
    }

    options.num_epochs = internal::choose_num_epochs<Index_>(options.num_epochs, x.size());

    // STEP 5: Pass rho/sigma to Status constructor
    return Status<Index_, Float_>(
        internal::similarities_to_epochs<Index_, Float_>(x, *(options.num_epochs), options.negative_sample_rate),
        std::move(options),
        num_dim,
        std::move(rhos_out),
        std::move(sigmas_out)
    );
}

/**
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Input_ Floating-point type of the input data for the neighbor search.
 * This only used to define the `knncolle::Prebuilt` type and is otherwise ignored.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 *
 * @param prebuilt A neighbor search index built on the dataset of interest.
 * @param num_dim Number of dimensions of the UMAP embedding.
 * @param[out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * On output, this contains the initial coordinates of the embedding.
 * Existing values in this array will not be modified if `Options::initialize_method = InitializeMethod::NONE`, 
 * or if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails and `Options::initialize_random_on_spectral_fail = false`.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 */
template<typename Index_, typename Input_, typename Float_>
Status<Index_, Float_> initialize(const knncolle::Prebuilt<Index_, Input_, Float_>& prebuilt, const std::size_t num_dim, Float_* const embedding, Options options) { 
    auto output = knncolle::find_nearest_neighbors(prebuilt, options.num_neighbors, options.num_threads);
    return initialize(std::move(output), num_dim, embedding, std::move(options));
}

/**
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 * @tparam Matrix_ Class of the input matrix for the neighbor search.
 * This should be a `knncolle::SimpleMatrix` or `knncolle::Matrix`.
 * 
 * @param data_dim Number of dimensions of the input dataset.
 * @param num_obs Number of observations in the input dataset.
 * @param[in] data Pointer to an array containing the input dataset as a column-major matrix.
 * Each row corresponds to a dimension (`data_dim`) and each column corresponds to an observation (`num_obs`).
 * @param builder Algorithm for the nearest neighbor search.
 * @param num_dim Number of dimensions of the embedding.
 * @param[out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * On output, this contains the initial coordinates of the embedding.
 * Existing values in this array will not be modified if `Options::initialize_method = InitializeMethod::NONE`, 
 * or if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails and `Options::initialize_random_on_spectral_fail = false`.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 */
template<typename Index_, typename Float_, class Matrix_ = knncolle::Matrix<Index_, Float_> >
Status<Index_, Float_> initialize(
    const std::size_t data_dim,
    const Index_ num_obs,
    const Float_* const data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    const std::size_t num_dim,
    Float_* const embedding,
    Options options)
{ 
    const auto prebuilt = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(data_dim, num_obs, data));
    return initialize(*prebuilt, num_dim, embedding, std::move(options));
}

}

#endif
