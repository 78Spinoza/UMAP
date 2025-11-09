#include "uwot_persistence.h"
#include "uwot_progress_utils.h"
#include "uwot_endian_utils.hpp"
#include "uwot_constants.hpp"
#include "lz4.h"
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <limits>

namespace persistence_utils {
    using namespace uwot::endian_utils;
    using namespace uwot::constants;

    // LZ4 uses 'int' which is 32-bit signed, so max safe size is INT_MAX
    static const size_t LZ4_MAX_SIZE = 2147483647; // INT_MAX = 2^31 - 1


    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        if (!hnsw_index) {
            throw std::runtime_error("HNSW index is null");
        }

        try {
            // ✅ STRINGSTREAM APPROACH (PacMap pattern) - No temp files!
            // Save HNSW index to stringstream to get uncompressed data in memory
            std::stringstream hnsw_data_stream;
            hnsw_index->saveIndex(hnsw_data_stream);

            // Get the HNSW data as string
            std::string hnsw_data = hnsw_data_stream.str();
            size_t uncompressed_size = hnsw_data.size();

            // LZ4 uses 'int' which is 32-bit signed (max ~2GB on most platforms)
            // Check for overflow before calling LZ4
            if (uncompressed_size > LZ4_MAX_SIZE) {
                throw std::runtime_error("HNSW data too large for LZ4 compression (exceeds 2GB int limit)");
            }

            // Compress with LZ4
            int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_size));
            if (max_compressed_size <= 0) {
                throw std::runtime_error("LZ4_compressBound failed - data too large");
            }
            std::vector<char> compressed_data(max_compressed_size);

            int compressed_size = LZ4_compress_default(
                hnsw_data.data(), compressed_data.data(),
                static_cast<int>(uncompressed_size), max_compressed_size);

            if (compressed_size <= 0) {
                throw std::runtime_error("LZ4 compression failed for HNSW data");
            }

            size_t final_compressed_size = static_cast<size_t>(compressed_size);

            // Write headers to output stream using endian-safe functions
            // Using uint32_t for backward compatibility (fits within 2GB LZ4 limit anyway)
            uint32_t uncompressed_size_32 = static_cast<uint32_t>(uncompressed_size);
            uint32_t final_compressed_size_32 = static_cast<uint32_t>(final_compressed_size);
            uwot::endian_utils::write_value(output, uncompressed_size_32);
            uwot::endian_utils::write_value(output, final_compressed_size_32);

            // Write compressed data to output stream
            output.write(compressed_data.data(), final_compressed_size);

            // Check if write was successful
            if (!output.good()) {
                throw std::runtime_error("Failed to write compressed HNSW data to stream");
            }

            // Flush the stream to ensure data is written
            output.flush();
        }
        catch (const std::exception& e) {
            std::string error_msg = std::string("HNSW Save (stringstream): ") + e.what();
            send_error_to_callback(error_msg.c_str());
            throw;
        }
    }


    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space) {
        try {
            // Check stream state first
            if (!input.good()) {
                throw std::runtime_error("Input stream is in bad state before reading");
            }

            // Read compression headers using endian-safe functions
            // OLD FORMAT (v3.40.0 and earlier): uint32_t sizes
            // NEW FORMAT (v3.41.0+): uint64_t sizes
            // For backward compatibility, we read as uint32_t (old format still in use)
            uint32_t uncompressed_size_32 = 0, compressed_size_32 = 0;
            if (!uwot::endian_utils::read_value(input, uncompressed_size_32) ||
                !uwot::endian_utils::read_value(input, compressed_size_32)) {
                throw std::runtime_error("Failed to read HNSW compression headers - stream error or EOF");
            }

            uint64_t uncompressed_size_64 = uncompressed_size_32;
            uint64_t compressed_size_64 = compressed_size_32;

            // Validate sizes - zero is allowed for empty HNSW data
            if (uncompressed_size_64 == 0 && compressed_size_64 == 0) {
                return; // Successfully loaded empty HNSW
            }

            // Security validation
            if (uncompressed_size_64 > MAX_DECOMPRESSED_SIZE) {
                std::string error = "HNSW uncompressed size too large: " + std::to_string(uncompressed_size_64) +
                    " bytes (" + std::to_string(uncompressed_size_64 / 1024.0 / 1024.0) + " MB) exceeds limit of " +
                    std::to_string(MAX_DECOMPRESSED_SIZE) + " bytes (" +
                    std::to_string(MAX_DECOMPRESSED_SIZE / 1024.0 / 1024.0) + " MB). " +
                    "This model requires increasing MAX_DECOMPRESSED_SIZE in uwot_constants.hpp.";
                throw std::runtime_error(error);
            }
            if (compressed_size_64 > MAX_COMPRESSED_SIZE) {
                std::string error = "HNSW compressed size too large: " + std::to_string(compressed_size_64) +
                    " bytes (" + std::to_string(compressed_size_64 / 1024.0 / 1024.0) + " MB) exceeds limit of " +
                    std::to_string(MAX_COMPRESSED_SIZE) + " bytes (" +
                    std::to_string(MAX_COMPRESSED_SIZE / 1024.0 / 1024.0) + " MB). " +
                    "This model requires increasing MAX_COMPRESSED_SIZE in uwot_constants.hpp.";
                throw std::runtime_error(error);
            }

            // LZ4 uses 'int' for sizes - check for overflow
            if (uncompressed_size_64 > LZ4_MAX_SIZE) {
                throw std::runtime_error("HNSW uncompressed size exceeds LZ4 int limit (2GB)");
            }
            if (compressed_size_64 > LZ4_MAX_SIZE) {
                throw std::runtime_error("HNSW compressed size exceeds LZ4 int limit (2GB)");
            }

            size_t uncompressed_size = static_cast<size_t>(uncompressed_size_64);
            size_t compressed_size = static_cast<size_t>(compressed_size_64);

            // Read compressed data into buffer
            std::vector<char> compressed_data(compressed_size);
            input.read(compressed_data.data(), compressed_size);

            if (!input.good() || input.gcount() != static_cast<std::streamsize>(compressed_size)) {
                throw std::runtime_error("Failed to read HNSW compressed data - unexpected EOF");
            }

            // Decompress with LZ4
            std::vector<char> decompressed_data(uncompressed_size);
            int decompressed_result = LZ4_decompress_safe(
                compressed_data.data(), decompressed_data.data(),
                static_cast<int>(compressed_size), static_cast<int>(uncompressed_size));

            if (decompressed_result != static_cast<int>(uncompressed_size)) {
                throw std::runtime_error("LZ4 decompression failed for HNSW data");
            }

            // ✅ STRINGSTREAM APPROACH (PacMap pattern) - No temp files!
            // Create stringstream from decompressed data for loading
            std::stringstream data_stream;
            data_stream.write(decompressed_data.data(), uncompressed_size);

            // Load HNSW from decompressed data stream
            hnsw_index->loadIndex(data_stream, space);

        } catch (const std::exception& e) {
            std::string error_msg = std::string("HNSW Load (stringstream): ") + e.what();
            send_error_to_callback(error_msg.c_str());
            throw;
        }
    }


    int save_model(UwotModel* model, const char* filename) {
        if (!model || !model->is_fitted || !filename) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return UWOT_ERROR_FILE_IO;
        }

        try {
            // Magic number and version
            constexpr uint32_t MAGIC = 0x554D4150; // "UMAP"
            constexpr uint32_t FORMAT_VERSION = 1;
            uwot::endian_utils::write_value(file, MAGIC);
            uwot::endian_utils::write_value(file, FORMAT_VERSION);
            file.write(UWOT_WRAPPER_VERSION_STRING, 16);

            // Basic parameters
            uwot::endian_utils::write_value(file, model->n_vertices);
            uwot::endian_utils::write_value(file, model->n_dim);
            uwot::endian_utils::write_value(file, model->embedding_dim);
            uwot::endian_utils::write_value(file, model->n_neighbors);
            uwot::endian_utils::write_value(file, model->min_dist);
            uwot::endian_utils::write_value(file, model->spread);
            uwot::endian_utils::write_value(file, model->local_connectivity);
            uwot::endian_utils::write_value(file, model->bandwidth);
            uwot::endian_utils::write_value(file, static_cast<int>(model->metric));

            // HNSW parameters
            uwot::endian_utils::write_value(file, model->hnsw_M);
            uwot::endian_utils::write_value(file, model->hnsw_ef_construction);
            uwot::endian_utils::write_value(file, model->hnsw_ef_search);

            // Fast transform data
            uwot::endian_utils::write_value(file, static_cast<int>(model->knn_backend));
            uwot::endian_utils::write_value(file, model->has_fast_transform_data);

            if (model->has_fast_transform_data) {
                const size_t rho_size = model->rho.size();
                uwot::endian_utils::write_value(file, rho_size);
                for (float v : model->rho) uwot::endian_utils::write_value(file, v);

                const size_t sigma_size = model->sigma.size();
                uwot::endian_utils::write_value(file, sigma_size);
                for (float v : model->sigma) uwot::endian_utils::write_value(file, v);
            }
            else {
                constexpr size_t zero = 0;
                uwot::endian_utils::write_value(file, zero);
                uwot::endian_utils::write_value(file, zero);
            }

            // Neighbor statistics
            uwot::endian_utils::write_value(file, model->mean_original_distance);
            uwot::endian_utils::write_value(file, model->std_original_distance);
            uwot::endian_utils::write_value(file, model->min_original_distance);
            uwot::endian_utils::write_value(file, model->p95_original_distance);
            uwot::endian_utils::write_value(file, model->p99_original_distance);
            uwot::endian_utils::write_value(file, model->mild_original_outlier_threshold);
            uwot::endian_utils::write_value(file, model->extreme_original_outlier_threshold);
            uwot::endian_utils::write_value(file, model->median_original_distance);
            uwot::endian_utils::write_value(file, model->exact_match_threshold);
            uwot::endian_utils::write_value(file, model->hnsw_recall_percentage);

            // Normalization
            const bool has_normalization = !model->feature_means.empty() && !model->feature_stds.empty();
            uwot::endian_utils::write_value(file, has_normalization);
            if (has_normalization) {
                for (float v : model->feature_means) uwot::endian_utils::write_value(file, v);
                for (float v : model->feature_stds) uwot::endian_utils::write_value(file, v);
                uwot::endian_utils::write_value(file, model->normalization_mode);
            }

            // Embedding storage (currently always saved for reliability)
            const size_t embedding_size = model->embedding.size();
            const bool save_embedding = true;
            uwot::endian_utils::write_value(file, embedding_size);
            uwot::endian_utils::write_value(file, save_embedding);

            if (save_embedding && embedding_size > 0) {
                const size_t uncompressed_bytes = embedding_size * sizeof(float);

                // Check for LZ4 int overflow
                if (uncompressed_bytes > LZ4_MAX_SIZE) {
                    throw std::runtime_error("Embedding data too large for LZ4 compression (exceeds 2GB int limit)");
                }

                const int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
                if (max_compressed_size <= 0) {
                    throw std::runtime_error("LZ4_compressBound failed for embedding data");
                }
                std::vector<char> compressed_data(max_compressed_size);

                const int compressed_bytes = LZ4_compress_default(
                    reinterpret_cast<const char*>(model->embedding.data()),
                    compressed_data.data(),
                    static_cast<int>(uncompressed_bytes),
                    max_compressed_size);

                uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                uint32_t comp_size = (compressed_bytes > 0) ? static_cast<uint32_t>(compressed_bytes) : 0;

                uwot::endian_utils::write_value(file, uncompressed_size);
                uwot::endian_utils::write_value(file, comp_size);

                if (comp_size > 0) {
                    file.write(compressed_data.data(), compressed_bytes);
                }
                else {
                    for (float v : model->embedding) {
                        uwot::endian_utils::write_value(file, v);
                    }
                }
            }
            else {
                constexpr uint32_t zero = 0;
                uwot::endian_utils::write_value(file, zero);
                uwot::endian_utils::write_value(file, zero);
            }

            // k-NN data
            constexpr bool needs_knn = true;
            uwot::endian_utils::write_value(file, needs_knn);

            if (needs_knn) {
                const auto write_vec = [&file](const auto& vec) {
                    uwot::endian_utils::write_value(file, vec.size());
                    for (const auto& v : vec) uwot::endian_utils::write_value(file, v);
                    };

                write_vec(model->nn_indices);
                write_vec(model->nn_distances);
                write_vec(model->nn_weights);
            }

            // Raw data
            uwot::endian_utils::write_value(file, model->force_exact_knn);
            const bool has_raw_data = model->force_exact_knn && !model->raw_data.empty();
            uwot::endian_utils::write_value(file, has_raw_data);
            if (has_raw_data) {
                uwot::endian_utils::write_value(file, model->raw_data.size());
                for (float v : model->raw_data) uwot::endian_utils::write_value(file, v);
            }

            // HNSW indices
            const bool save_original_index = model->original_space_index != nullptr;
            const bool save_embedding_index = model->embedding_space_index != nullptr && !model->always_save_embedding_data;

            uwot::endian_utils::write_value(file, save_original_index);
            uwot::endian_utils::write_value(file, save_embedding_index);

            auto save_hnsw = [&file](const auto& index, const char* name) {
                try {
                    save_hnsw_to_stream_compressed(file, index.get());
                }
                catch (...) {
                    constexpr size_t zero = 0;
                    uwot::endian_utils::write_value(file, zero);
                    send_warning_to_callback((std::string(name) + " HNSW save failed").c_str());
                }
                };

            if (save_original_index) {
                save_hnsw(model->original_space_index, "Original space");
            }
            else {
                constexpr size_t zero = 0;
                uwot::endian_utils::write_value(file, zero);
            }

            if (save_embedding_index) {
                save_hnsw(model->embedding_space_index, "Embedding space");
            }
            else {
                constexpr size_t zero = 0;
                uwot::endian_utils::write_value(file, zero);
            }

            // CRCs
            uwot::endian_utils::write_value(file, model->original_space_crc);
            uwot::endian_utils::write_value(file, model->embedding_space_crc);
            uwot::endian_utils::write_value(file, model->model_version_crc);

            file.close();
            return UWOT_SUCCESS;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            uwot_set_global_callback(nullptr);
            return UWOT_ERROR_FILE_IO;
        }
    }


    UwotModel* load_model(const char* filename) {
        if (!filename) return nullptr;

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return nullptr;

        UwotModel* model = model_utils::create_model();
        if (!model) return nullptr;

        try {
            // Magic and version
            uint32_t magic = 0, format_version = 0;
            if (!uwot::endian_utils::read_value(file, magic) || magic != 0x554D4150) {
                throw std::runtime_error("Invalid magic number");
            }
            uwot::endian_utils::read_value(file, format_version);

            char version[17] = { 0 };
            file.read(version, 16);
            if (strcmp(version, UWOT_WRAPPER_VERSION_STRING) != 0) {
                send_warning_to_callback("Model version mismatch");
            }

            // Basic parameters
            auto read = [&file](auto& v) { return uwot::endian_utils::read_value(file, v); };
            if (!read(model->n_vertices) || !read(model->n_dim) || !read(model->embedding_dim) ||
                !read(model->n_neighbors) || !read(model->min_dist) || !read(model->spread) ||
                !read(model->local_connectivity) || !read(model->bandwidth)) {
                throw std::runtime_error("Failed to read basic parameters");
            }

            int metric_val = 0;
            if (!read(metric_val)) throw std::runtime_error("Failed to read metric");
            model->metric = static_cast<UwotMetric>(metric_val);

            if (!read(model->hnsw_M) || !read(model->hnsw_ef_construction) || !read(model->hnsw_ef_search)) {
                throw std::runtime_error("Failed to read HNSW parameters");
            }

            // Fast transform
            int knn_backend_val = 0;
            if (!read(knn_backend_val) || !read(model->has_fast_transform_data)) {
                throw std::runtime_error("Failed to read fast transform metadata");
            }
            model->knn_backend = static_cast<UwotModel::KnnBackend>(knn_backend_val);

            auto read_vec_float = [&file, &read](std::vector<float>& vec, size_t size) {
                vec.resize(size);
                for (float& v : vec) if (!read(v)) throw std::runtime_error("Failed to read vector");
                };
            auto read_vec_int = [&file, &read](std::vector<int>& vec, size_t size) {
                vec.resize(size);
                for (int& v : vec) if (!read(v)) throw std::runtime_error("Failed to read vector");
                };

            if (model->has_fast_transform_data) {
                // CRITICAL: Read rho_size, then rho values, then sigma_size, then sigma values
                // Must match the SAVE format (not reading both sizes together!)
                size_t rho_size = 0;
                if (!read(rho_size)) throw std::runtime_error("Failed to read rho size");
                read_vec_float(model->rho, rho_size);

                size_t sigma_size = 0;
                if (!read(sigma_size)) throw std::runtime_error("Failed to read sigma size");
                read_vec_float(model->sigma, sigma_size);
            }
            else {
                size_t dummy = 0;
                read(dummy); read(dummy);
                model->rho.clear();
                model->sigma.clear();
            }

            // Neighbor stats
            if (!read(model->mean_original_distance) || !read(model->std_original_distance) ||
                !read(model->min_original_distance) || !read(model->p95_original_distance) ||
                !read(model->p99_original_distance) || !read(model->mild_original_outlier_threshold) ||
                !read(model->extreme_original_outlier_threshold) || !read(model->median_original_distance) ||
                !read(model->exact_match_threshold) || !read(model->hnsw_recall_percentage)) {
                throw std::runtime_error("Failed to read neighbor stats");
            }

            // Normalization
            bool has_normalization = false;
            if (!read(has_normalization)) throw std::runtime_error("Failed to read normalization flag");

            model->feature_means.assign(model->n_dim, 0.0f);
            model->feature_stds.assign(model->n_dim, 1.0f);
            model->use_normalization = false;
            model->normalization_mode = 0;

            if (has_normalization) {
                for (float& v : model->feature_means) if (!read(v)) throw std::runtime_error("Failed to read means");
                for (float& v : model->feature_stds) if (!read(v)) throw std::runtime_error("Failed to read stds");
                if (!read(model->normalization_mode)) throw std::runtime_error("Failed to read normalization mode");
                model->use_normalization = true;
            }

            // Embedding
            size_t embedding_size = 0;
            bool save_embedding = false;
            if (!read(embedding_size) || !read(save_embedding)) throw std::runtime_error("Failed to read embedding metadata");
            model->embedding.resize(embedding_size);

            if (save_embedding && embedding_size > 0) {
                uint32_t uncompressed_size = 0, compressed_size = 0;
                if (!read(uncompressed_size) || !read(compressed_size)) throw std::runtime_error("Failed to read embedding headers");

                // Validate sizes against LZ4 limits
                if (uncompressed_size > LZ4_MAX_SIZE) {
                    throw std::runtime_error("Embedding uncompressed size exceeds LZ4 int limit (2GB)");
                }
                if (compressed_size > LZ4_MAX_SIZE) {
                    throw std::runtime_error("Embedding compressed size exceeds LZ4 int limit (2GB)");
                }

                if (compressed_size > 0) {
                    std::vector<char> compressed_data(compressed_size);
                    file.read(compressed_data.data(), compressed_size);
                    if (file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                        throw std::runtime_error("Failed to read compressed embedding");
                    }

                    const int decompressed = LZ4_decompress_safe(
                        compressed_data.data(),
                        reinterpret_cast<char*>(model->embedding.data()),
                        static_cast<int>(compressed_size),
                        static_cast<int>(uncompressed_size));

                    if (decompressed != static_cast<int>(uncompressed_size)) {
                        throw std::runtime_error("Embedding decompression failed");
                    }
                }
                else {
                    for (float& v : model->embedding) if (!read(v)) throw std::runtime_error("Failed to read embedding");
                }
            }
            else if (!save_embedding) {
                uint32_t dummy1 = 0, dummy2 = 0;
                read(dummy1); read(dummy2);
            }

            // k-NN data
            bool needs_knn = false;
            if (!read(needs_knn)) throw std::runtime_error("Failed to read k-NN flag");
            if (needs_knn) {
                size_t idx_size = 0, dist_size = 0, w_size = 0;
                if (!read(idx_size) || !read(dist_size) || !read(w_size)) throw std::runtime_error("Failed to read k-NN sizes");
                read_vec_int(model->nn_indices, idx_size);
                read_vec_float(model->nn_distances, dist_size);
                read_vec_float(model->nn_weights, w_size);
            }

            // Raw data
            if (!read(model->force_exact_knn)) throw std::runtime_error("Failed to read force_exact_knn");
            bool has_raw_data = false;
            if (!read(has_raw_data)) throw std::runtime_error("Failed to read raw_data flag");
            if (has_raw_data) {
                size_t raw_size = 0;
                if (!read(raw_size)) throw std::runtime_error("Failed to read raw_data size");
                read_vec_float(model->raw_data, raw_size);
            }

            // HNSW indices
            bool has_original_index = false, has_embedding_index = false;
            if (!read(has_original_index) || !read(has_embedding_index)) {
                throw std::runtime_error("Failed to read HNSW flags");
            }

            auto load_hnsw = [&file, model](bool flag, std::unique_ptr<hnswlib::HierarchicalNSW<float>>& index_ptr, std::unique_ptr<hnsw_utils::SpaceFactory>& factory_ptr, UwotMetric metric, int dim, const char* name) {
                if (!flag) {
                    size_t zero = 0;
                    uwot::endian_utils::read_value(file, zero);
                    return;
                }

                try {
                    if (!factory_ptr) factory_ptr = std::make_unique<hnsw_utils::SpaceFactory>();
                    if (!factory_ptr->create_space(metric, dim)) throw std::runtime_error("Failed to create space");

                    index_ptr.reset(new hnswlib::HierarchicalNSW<float>(
                        factory_ptr->get_space(), model->n_vertices, model->hnsw_M, model->hnsw_ef_construction));
                    index_ptr->setEf(model->hnsw_ef_search);

                    load_hnsw_from_stream_compressed(file, index_ptr.get(), factory_ptr->get_space());
                }
                catch (const std::exception& e) {
                    index_ptr.reset();
                    std::string error_msg = std::string(name) + " HNSW load failed: " + e.what();
                    send_error_to_callback(error_msg.c_str());
                    throw std::runtime_error(error_msg);
                }
                catch (...) {
                    index_ptr.reset();
                    std::string error_msg = std::string(name) + " HNSW load failed with unknown exception";
                    send_error_to_callback(error_msg.c_str());
                    throw std::runtime_error(error_msg);
                }
                };

            load_hnsw(has_original_index, model->original_space_index, model->original_space_factory,
                model->metric, model->n_dim, "Original space");

            load_hnsw(has_embedding_index, model->embedding_space_index, model->embedding_space_factory,
                UWOT_METRIC_EUCLIDEAN, model->embedding_dim, "Embedding space");

            // Rebuild embedding HNSW if needed and possible
            if (!model->embedding_space_index && !model->embedding.empty() && model->n_vertices > 0) {
                try {
                    if (!model->embedding_space_factory) {
                        model->embedding_space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
                    }
                    if (!model->embedding_space_factory->create_space(UWOT_METRIC_EUCLIDEAN, model->embedding_dim)) {
                        throw std::runtime_error("Failed to create embedding space");
                    }

                    model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        model->embedding_space_factory->get_space(), model->n_vertices,
                        model->hnsw_M, model->hnsw_ef_construction);
                    model->embedding_space_index->setEf(model->hnsw_ef_search);

                    for (int i = 0; i < model->n_vertices; ++i) {
                        const float* ptr = &model->embedding[i * model->embedding_dim];
                        model->embedding_space_index->addPoint(ptr, i);
                    }
                }
                catch (const std::exception& e) {
                    model->embedding_space_index = nullptr;
                    std::string error_msg = std::string("CRITICAL: Embedding HNSW reconstruction failed: ") + e.what();
                    send_error_to_callback(error_msg.c_str());
                    throw std::runtime_error(error_msg);
                }
            }

            // CRCs
            uwot::endian_utils::read_value(file, model->original_space_crc);
            uwot::endian_utils::read_value(file, model->embedding_space_crc);
            uwot::endian_utils::read_value(file, model->model_version_crc);

            // CRITICAL VALIDATION: Ensure ALL required components for TransformWithSafety are present

            // 1. Check original_space_index (required for basic transform)
            if (!model->original_space_index) {
                throw std::runtime_error("CRITICAL: Model load failed - original_space_index is NULL. Transform operations will not work.");
            }
            if (model->original_space_index->cur_element_count == 0 ||
                model->original_space_index->max_elements_ == 0) {
                throw std::runtime_error("CRITICAL: Model load failed - original_space_index is uninitialized (no elements). Transform operations will not work.");
            }

            // 2. Check embedding_space_index (required for TransformWithSafety)
            if (!model->embedding_space_index) {
                if (!model->embedding.empty()) {
                    throw std::runtime_error("CRITICAL: Model load failed - embedding_space_index is NULL but embedding data exists. "
                        "HNSW index reconstruction failed. Check if embedding data is corrupted or model was saved with incompatible settings. "
                        "TransformWithSafety will not work.");
                } else {
                    throw std::runtime_error("CRITICAL: Model load failed - embedding_space_index is NULL and no embedding data to rebuild from. "
                        "Model file may be corrupted or incomplete. TransformWithSafety will not work.");
                }
            }
            if (model->embedding_space_index->cur_element_count == 0 ||
                model->embedding_space_index->max_elements_ == 0) {
                throw std::runtime_error("CRITICAL: Model load failed - embedding_space_index is uninitialized (no elements loaded). "
                    "TransformWithSafety will not work. Check model file integrity.");
            }

            // 3. Check embedding data (required for transform)
            if (model->embedding.empty()) {
                throw std::runtime_error("CRITICAL: Model load failed - embedding array is empty. Transform operations will not work.");
            }
            if (model->embedding.size() != static_cast<size_t>(model->n_vertices) * static_cast<size_t>(model->embedding_dim)) {
                throw std::runtime_error("CRITICAL: Model load failed - embedding array size mismatch. Expected " +
                    std::to_string(model->n_vertices * model->embedding_dim) + " but got " +
                    std::to_string(model->embedding.size()));
            }

            model->is_fitted = true;
            file.close();
            return model;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            uwot_set_global_callback(nullptr);
            model_utils::destroy_model(model);
            return nullptr;
        }
    }

} // namespace persistence_utils
