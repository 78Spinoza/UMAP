#include "uwot_persistence.h"
#include "uwot_progress_utils.h"
#include "lz4.h"
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

namespace persistence_utils {

    // Endian-safe serialization utilities
    namespace endian_utils {

        // Check if system is little-endian
        inline bool is_little_endian() {
            uint16_t test = 0x1234;
            return *reinterpret_cast<uint8_t*>(&test) == 0x34;
        }

        // Convert to/from little-endian (same operation: swap if big-endian)
        template<typename T>
        inline void convert_endian(T& value) {
            if (!is_little_endian()) {
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
                for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
            }
        }

        // Safe write with endian conversion
        template<typename T>
        inline void write_value(std::ostream& stream, const T& value) {
            T temp = value;
            convert_endian(temp);
            stream.write(reinterpret_cast<const char*>(&temp), sizeof(T));
        }

        // Safe read with endian conversion
        template<typename T>
        inline bool read_value(std::istream& stream, T& value) {
            if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T))) {
                return false;
            }
            convert_endian(value);
            return true;
        }
    } // namespace endian_utils


    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        const std::string temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_compressed");

        try {
            // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file
            std::ifstream temp_file(temp_filename, std::ios::binary | std::ios::ate);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to open temporary HNSW file for compression");
            }

            const std::streamsize file_size = temp_file.tellg();
            temp_file.seekg(0, std::ios::beg);

            std::vector<char> uncompressed_data(file_size);
            if (!temp_file.read(uncompressed_data.data(), file_size)) {
                throw std::runtime_error("Failed to read HNSW temporary file");
            }
            temp_file.close();

            // Compress with LZ4
            const int max_compressed_size = LZ4_compressBound(static_cast<int>(file_size));
            std::vector<char> compressed_data(max_compressed_size);

            const int compressed_size = LZ4_compress_default(
                uncompressed_data.data(), compressed_data.data(),
                static_cast<int>(file_size), max_compressed_size);

            if (compressed_size <= 0) {
                throw std::runtime_error("LZ4 compression failed for HNSW data");
            }

            // Write sizes and compressed data
            const uint32_t original_size = static_cast<uint32_t>(file_size);
            const uint32_t comp_size = static_cast<uint32_t>(compressed_size);

            endian_utils::write_value(output, original_size);
            endian_utils::write_value(output, comp_size);
            output.write(compressed_data.data(), compressed_size);

            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }


    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space) {
        std::string temp_filename;

        try {
            // Read compression headers
            uint32_t original_size = 0, compressed_size = 0;
            if (!endian_utils::read_value(input, original_size) ||
                !endian_utils::read_value(input, compressed_size)) {
                throw std::runtime_error("Failed to read LZ4 compression headers");
            }

            // Security validation
            constexpr uint32_t MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024; // 100MB
            constexpr uint32_t MAX_COMPRESSED_SIZE = 80 * 1024 * 1024;    // 80MB

            if (original_size > MAX_DECOMPRESSED_SIZE || compressed_size > MAX_COMPRESSED_SIZE ||
                original_size == 0 || compressed_size == 0) {
                throw std::runtime_error("Invalid HNSW compressed data size");
            }

            // Read compressed data
            std::vector<char> compressed_data(compressed_size);
            input.read(compressed_data.data(), compressed_size);
            if (!input.good() || input.gcount() != compressed_size) {
                throw std::runtime_error("Failed to read HNSW compressed data");
            }

            // Decompress
            std::vector<char> decompressed_data(original_size);
            const int decompressed_size = LZ4_decompress_safe(
                compressed_data.data(), decompressed_data.data(),
                static_cast<int>(compressed_size), static_cast<int>(original_size));

            if (decompressed_size <= 0 || decompressed_size != static_cast<int>(original_size)) {
                throw std::runtime_error("LZ4 decompression failed for HNSW data");
            }

            // Write to temporary file and load
            temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_decomp");
            {
                std::ofstream temp_file(temp_filename, std::ios::binary);
                if (!temp_file.is_open()) {
                    throw std::runtime_error("Failed to create temporary file for HNSW decompression");
                }
                temp_file.write(decompressed_data.data(), original_size);
            }

            hnsw_index->loadIndex(temp_filename, space, hnsw_index->getCurrentElementCount());
            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            if (!temp_filename.empty()) {
                temp_utils::safe_remove_file(temp_filename);
            }
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
            endian_utils::write_value(file, MAGIC);
            endian_utils::write_value(file, FORMAT_VERSION);
            file.write(UWOT_WRAPPER_VERSION_STRING, 16);

            // Basic parameters
            endian_utils::write_value(file, model->n_vertices);
            endian_utils::write_value(file, model->n_dim);
            endian_utils::write_value(file, model->embedding_dim);
            endian_utils::write_value(file, model->n_neighbors);
            endian_utils::write_value(file, model->min_dist);
            endian_utils::write_value(file, model->spread);
            endian_utils::write_value(file, static_cast<int>(model->metric));

            // HNSW parameters
            endian_utils::write_value(file, model->hnsw_M);
            endian_utils::write_value(file, model->hnsw_ef_construction);
            endian_utils::write_value(file, model->hnsw_ef_search);

            // Fast transform data
            endian_utils::write_value(file, static_cast<int>(model->knn_backend));
            endian_utils::write_value(file, model->has_fast_transform_data);

            if (model->has_fast_transform_data) {
                const size_t rho_size = model->rho.size();
                endian_utils::write_value(file, rho_size);
                for (float v : model->rho) endian_utils::write_value(file, v);

                const size_t sigma_size = model->sigma.size();
                endian_utils::write_value(file, sigma_size);
                for (float v : model->sigma) endian_utils::write_value(file, v);
            }
            else {
                constexpr size_t zero = 0;
                endian_utils::write_value(file, zero);
                endian_utils::write_value(file, zero);
            }

            // Neighbor statistics
            endian_utils::write_value(file, model->mean_original_distance);
            endian_utils::write_value(file, model->std_original_distance);
            endian_utils::write_value(file, model->min_original_distance);
            endian_utils::write_value(file, model->p95_original_distance);
            endian_utils::write_value(file, model->p99_original_distance);
            endian_utils::write_value(file, model->mild_original_outlier_threshold);
            endian_utils::write_value(file, model->extreme_original_outlier_threshold);
            endian_utils::write_value(file, model->median_original_distance);
            endian_utils::write_value(file, model->exact_match_threshold);
            endian_utils::write_value(file, model->hnsw_recall_percentage);

            // Normalization
            const bool has_normalization = !model->feature_means.empty() && !model->feature_stds.empty();
            endian_utils::write_value(file, has_normalization);
            if (has_normalization) {
                for (float v : model->feature_means) endian_utils::write_value(file, v);
                for (float v : model->feature_stds) endian_utils::write_value(file, v);
                endian_utils::write_value(file, model->normalization_mode);
            }

            // Embedding storage (currently always saved for reliability)
            const size_t embedding_size = model->embedding.size();
            const bool save_embedding = true;
            endian_utils::write_value(file, embedding_size);
            endian_utils::write_value(file, save_embedding);

            if (save_embedding && embedding_size > 0) {
                const size_t uncompressed_bytes = embedding_size * sizeof(float);
                const int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
                std::vector<char> compressed_data(max_compressed_size);

                const int compressed_bytes = LZ4_compress_default(
                    reinterpret_cast<const char*>(model->embedding.data()),
                    compressed_data.data(),
                    static_cast<int>(uncompressed_bytes),
                    max_compressed_size);

                uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                uint32_t comp_size = (compressed_bytes > 0) ? static_cast<uint32_t>(compressed_bytes) : 0;

                endian_utils::write_value(file, uncompressed_size);
                endian_utils::write_value(file, comp_size);

                if (comp_size > 0) {
                    file.write(compressed_data.data(), compressed_bytes);
                }
                else {
                    for (float v : model->embedding) {
                        endian_utils::write_value(file, v);
                    }
                }
            }
            else {
                constexpr uint32_t zero = 0;
                endian_utils::write_value(file, zero);
                endian_utils::write_value(file, zero);
            }

            // k-NN data
            constexpr bool needs_knn = true;
            endian_utils::write_value(file, needs_knn);

            if (needs_knn) {
                const auto write_vec = [&file](const auto& vec) {
                    endian_utils::write_value(file, vec.size());
                    for (const auto& v : vec) endian_utils::write_value(file, v);
                    };

                write_vec(model->nn_indices);
                write_vec(model->nn_distances);
                write_vec(model->nn_weights);
            }

            // Raw data
            endian_utils::write_value(file, model->force_exact_knn);
            const bool has_raw_data = model->force_exact_knn && !model->raw_data.empty();
            endian_utils::write_value(file, has_raw_data);
            if (has_raw_data) {
                endian_utils::write_value(file, model->raw_data.size());
                for (float v : model->raw_data) endian_utils::write_value(file, v);
            }

            // HNSW indices
            const bool save_original_index = model->original_space_index != nullptr;
            const bool save_embedding_index = model->embedding_space_index != nullptr && !model->always_save_embedding_data;

            endian_utils::write_value(file, save_original_index);
            endian_utils::write_value(file, save_embedding_index);

            auto save_hnsw = [&file](const auto& index, const char* name) {
                try {
                    save_hnsw_to_stream_compressed(file, index.get());
                }
                catch (...) {
                    constexpr size_t zero = 0;
                    endian_utils::write_value(file, zero);
                    send_warning_to_callback((std::string(name) + " HNSW save failed").c_str());
                }
                };

            if (save_original_index) {
                save_hnsw(model->original_space_index, "Original space");
            }
            else {
                constexpr size_t zero = 0;
                endian_utils::write_value(file, zero);
            }

            if (save_embedding_index) {
                save_hnsw(model->embedding_space_index, "Embedding space");
            }
            else {
                constexpr size_t zero = 0;
                endian_utils::write_value(file, zero);
            }

            // CRCs
            endian_utils::write_value(file, model->original_space_crc);
            endian_utils::write_value(file, model->embedding_space_crc);
            endian_utils::write_value(file, model->model_version_crc);

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
            if (!endian_utils::read_value(file, magic) || magic != 0x554D4150) {
                throw std::runtime_error("Invalid magic number");
            }
            endian_utils::read_value(file, format_version);

            char version[17] = { 0 };
            file.read(version, 16);
            if (strcmp(version, UWOT_WRAPPER_VERSION_STRING) != 0) {
                send_warning_to_callback("Model version mismatch");
            }

            // Basic parameters
            auto read = [&file](auto& v) { return endian_utils::read_value(file, v); };
            if (!read(model->n_vertices) || !read(model->n_dim) || !read(model->embedding_dim) ||
                !read(model->n_neighbors) || !read(model->min_dist) || !read(model->spread)) {
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
                size_t rho_size = 0, sigma_size = 0;
                if (!read(rho_size) || !read(sigma_size)) throw std::runtime_error("Failed to read rho/sigma sizes");
                read_vec_float(model->rho, rho_size);
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

                if (compressed_size > 0) {
                    std::vector<char> compressed_data(compressed_size);
                    file.read(compressed_data.data(), compressed_size);
                    if (file.gcount() != compressed_size) throw std::runtime_error("Failed to read compressed embedding");

                    const int decompressed = LZ4_decompress_safe(
                        compressed_data.data(),
                        reinterpret_cast<char*>(model->embedding.data()),
                        compressed_size, uncompressed_size);

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
                    endian_utils::read_value(file, zero);
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
                catch (...) {
                    index_ptr.reset();
                    send_warning_to_callback((std::string(name) + " HNSW load failed").c_str());
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
                catch (...) {
                    model->embedding_space_index = nullptr;
                    send_warning_to_callback("Embedding HNSW reconstruction failed");
                }
            }

            // CRCs
            endian_utils::read_value(file, model->original_space_crc);
            endian_utils::read_value(file, model->embedding_space_crc);
            endian_utils::read_value(file, model->model_version_crc);

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
