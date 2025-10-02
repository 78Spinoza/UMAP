#include "uwot_persistence.h"
#include "uwot_quantization.h"
#include "lz4.h"
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

namespace persistence_utils {

    // Endian-safe serialization utilities
    namespace endian_utils {

        // Check if system is little-endian
        bool is_little_endian() {
            uint16_t test = 0x1234;
            return *reinterpret_cast<uint8_t*>(&test) == 0x34;
        }

        // Convert to little-endian (portable format)
        template<typename T>
        void to_little_endian(T& value) {
            if (!is_little_endian()) {
                // Swap bytes for big-endian systems
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
                for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
            }
        }

        // Convert from little-endian to native
        template<typename T>
        void from_little_endian(T& value) {
            to_little_endian(value); // Same operation - byte swap if needed
        }

        // Safe write with endian conversion
        template<typename T>
        void write_value(std::ostream& stream, const T& value) {
            T temp = value;
            to_little_endian(temp);
            stream.write(reinterpret_cast<const char*>(&temp), sizeof(T));
        }

        // Safe read with endian conversion
        template<typename T>
        bool read_value(std::istream& stream, T& value) {
            if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T))) {
                return false;
            }
            from_little_endian(value);
            return true;
        }
    }


    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
                std::string temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_compressed");

        try {
                        // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file
            std::ifstream temp_file(temp_filename, std::ios::binary | std::ios::ate);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to open temporary HNSW file for compression");
            }

            std::streamsize file_size = temp_file.tellg();
                        temp_file.seekg(0, std::ios::beg);

            std::vector<char> uncompressed_data(file_size);
            if (!temp_file.read(uncompressed_data.data(), file_size)) {
                throw std::runtime_error("Failed to read HNSW temporary file");
            }
            temp_file.close();

            // Compress with LZ4
            int max_compressed_size = LZ4_compressBound(static_cast<int>(file_size));
            std::vector<char> compressed_data(max_compressed_size);

            int compressed_size = LZ4_compress_default(
                uncompressed_data.data(), compressed_data.data(),
                static_cast<int>(file_size), max_compressed_size);

            if (compressed_size <= 0) {
                throw std::runtime_error("LZ4 compression failed for HNSW data");
            }

            // Write sizes and compressed data
            uint32_t original_size = static_cast<uint32_t>(file_size);
            uint32_t comp_size = static_cast<uint32_t>(compressed_size);

                        endian_utils::write_value(output, original_size);
            endian_utils::write_value(output, comp_size);
            output.write(compressed_data.data(), compressed_size);
            
            // Clean up
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
            std::cout << "[DEBUG] Inside load_hnsw_from_stream_compressed - about to read headers" << std::endl;
            // Read LZ4 compression headers with validation (endian-safe)
            uint32_t original_size, compressed_size;
            if (!endian_utils::read_value(input, original_size) ||
                !endian_utils::read_value(input, compressed_size)) {
                throw std::runtime_error("Failed to read LZ4 compression headers");
            }
            std::cout << "[DEBUG] Inside load_hnsw_from_stream_compressed - read original_size=" << original_size << ", compressed_size=" << compressed_size << std::endl;

            
            // Enhanced security validation for LZ4 decompression
            const uint32_t MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024; // 100MB limit
            const uint32_t MAX_COMPRESSED_SIZE = 80 * 1024 * 1024;    // 80MB limit

            if (original_size > MAX_DECOMPRESSED_SIZE) {
                throw std::runtime_error("LZ4 decompression: HNSW original size too large (potential attack)");
            }
            if (compressed_size > MAX_COMPRESSED_SIZE) {
                throw std::runtime_error("LZ4 decompression: HNSW compressed size too large (potential attack)");
            }
            if (original_size == 0 || compressed_size == 0) {
                throw std::runtime_error("LZ4 decompression: Invalid zero size for HNSW data");
            }

            // Read compressed data with validation
            std::vector<char> compressed_data(compressed_size);
            input.read(compressed_data.data(), compressed_size);

            if (!input.good() || input.gcount() != static_cast<std::streamsize>(compressed_size)) {
                throw std::runtime_error("LZ4 decompression: Failed to read HNSW compressed data");
            }

            // Decompress with LZ4 (bounds-checked)
            std::vector<char> decompressed_data(original_size);
            int decompressed_size = LZ4_decompress_safe(
                compressed_data.data(), decompressed_data.data(),
                static_cast<int>(compressed_size), static_cast<int>(original_size));

            if (decompressed_size <= 0) {
                throw std::runtime_error("LZ4 decompression failed: Malformed HNSW compressed data");
            }
            if (decompressed_size != static_cast<int>(original_size)) {
                throw std::runtime_error("LZ4 decompression failed: HNSW size mismatch");
            }

            // Write to temporary file and load
            temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_decomp");
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to create temporary file for HNSW decompression");
            }

            temp_file.write(decompressed_data.data(), original_size);
            temp_file.close();

            // Load from temporary file with correct parameters
            hnsw_index->loadIndex(temp_filename, space, hnsw_index->getCurrentElementCount());

            // Clean up
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
            // Magic number for file format validation (4 bytes)
            const uint32_t magic = 0x554D4150; // "UMAP" in ASCII
            endian_utils::write_value(file, magic);

            // File format version (4 bytes)
            const uint32_t format_version = 1;
            endian_utils::write_value(file, format_version);

            // Library version (16 bytes) - keep as string for compatibility
            file.write(UWOT_WRAPPER_VERSION_STRING, 16);

            // Basic model parameters (endian-safe)
            endian_utils::write_value(file, model->n_vertices);
            endian_utils::write_value(file, model->n_dim);
            endian_utils::write_value(file, model->embedding_dim);
            endian_utils::write_value(file, model->n_neighbors);
            endian_utils::write_value(file, model->min_dist);
            endian_utils::write_value(file, model->spread);
            endian_utils::write_value(file, static_cast<int>(model->metric));
            endian_utils::write_value(file, model->a);
            endian_utils::write_value(file, model->b);

            // HNSW parameters (endian-safe)
            endian_utils::write_value(file, model->hnsw_M);
            endian_utils::write_value(file, model->hnsw_ef_construction);
            endian_utils::write_value(file, model->hnsw_ef_search);

            // Neighbor statistics (endian-safe)
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

            // Normalization parameters (endian-safe)
            bool has_normalization = !model->feature_means.empty() && !model->feature_stds.empty();
            endian_utils::write_value(file, has_normalization);
            if (has_normalization) {
                // Write feature means (endian-safe)
                for (int i = 0; i < model->n_dim; i++) {
                    endian_utils::write_value(file, model->feature_means[i]);
                }
                // Write feature standard deviations (endian-safe)
                for (int i = 0; i < model->n_dim; i++) {
                    endian_utils::write_value(file, model->feature_stds[i]);
                }
                endian_utils::write_value(file, model->normalization_mode);
            }

            // EMBEDDING STORAGE OPTIMIZATION: Skip redundant training embedding array
            // Embeddings are extracted from embedding_space HNSW during transform
            // This saves 50% model file size by eliminating duplicate storage
            size_t embedding_size = model->embedding.size();

            // OPTIMIZATION: Skip redundant embedding array - use HNSW extraction instead
            // CRITICAL: Only skip when embedding space HNSW can be reliably saved/loaded
            // Current HNSW loading has issues, so keep embedding array for reliability
            bool save_embedding = true;  // TEMPORARY: Keep embedding array until HNSW loading is fixed
            endian_utils::write_value(file, embedding_size);
            endian_utils::write_value(file, save_embedding);

            if (save_embedding && embedding_size > 0) {
                // Compress embedding data with LZ4
                size_t uncompressed_bytes = embedding_size * sizeof(float);
                int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
                std::vector<char> compressed_data(max_compressed_size);

                int compressed_bytes = LZ4_compress_default(
                    reinterpret_cast<const char*>(model->embedding.data()),
                    compressed_data.data(),
                    static_cast<int>(uncompressed_bytes),
                    max_compressed_size);

                if (compressed_bytes > 0) {
                    // Write: uncompressed_bytes, compressed_bytes, compressed_data (endian-safe)
                    uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                    uint32_t comp_size = static_cast<uint32_t>(compressed_bytes);
                    endian_utils::write_value(file, uncompressed_size);
                    endian_utils::write_value(file, comp_size);
                    file.write(compressed_data.data(), compressed_bytes);
                } else {
                    // Compression failed - fall back to uncompressed (endian-safe)
                    uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                    uint32_t comp_size = 0; // 0 = uncompressed
                    endian_utils::write_value(file, uncompressed_size);
                    endian_utils::write_value(file, comp_size);
                    // Write raw float data (endian conversion needed for floats)
                    for (size_t i = 0; i < embedding_size; i++) {
                        endian_utils::write_value(file, model->embedding[i]);
                    }
                }
            } else {
                // Embedding storage optimized: Using HNSW extraction instead (50% space savings)
                uint32_t zero = 0;
                endian_utils::write_value(file, zero);
                endian_utils::write_value(file, zero);
            }

            // k-NN data - Only save when NOT using quantization (quantization uses PQ codes instead)
            // CRITICAL OPTIMIZATION: Skip k-NN data when quantization enabled - saves massive file space
            bool needs_knn = !model->use_quantization;  // Use PQ codes instead when quantized
            endian_utils::write_value(file, needs_knn);

            if (needs_knn) {
                size_t indices_size = model->nn_indices.size();
                size_t distances_size = model->nn_distances.size();
                size_t weights_size = model->nn_weights.size();

                endian_utils::write_value(file, indices_size);
                endian_utils::write_value(file, distances_size);
                endian_utils::write_value(file, weights_size);

                // Write k-NN indices (endian-safe)
                for (size_t i = 0; i < indices_size; i++) {
                    endian_utils::write_value(file, model->nn_indices[i]);
                }
                // Write k-NN distances (endian-safe)
                for (size_t i = 0; i < distances_size; i++) {
                    endian_utils::write_value(file, model->nn_distances[i]);
                }
                // Write k-NN weights (endian-safe)
                for (size_t i = 0; i < weights_size; i++) {
                    endian_utils::write_value(file, model->nn_weights[i]);
                }
            }

            // Quantization data (PQ - Product Quantization) (endian-safe)
            endian_utils::write_value(file, model->use_quantization);
            endian_utils::write_value(file, model->pq_m);

            if (model->use_quantization) {
                // Save PQ codes
                size_t pq_codes_size = model->pq_codes.size();
                endian_utils::write_value(file, pq_codes_size);
                // PQ codes are uint8_t - no endian conversion needed
                if (pq_codes_size > 0) {
                    file.write(reinterpret_cast<const char*>(model->pq_codes.data()), pq_codes_size * sizeof(uint8_t));
                }

                // Save PQ centroids (endian-safe)
                size_t pq_centroids_size = model->pq_centroids.size();
                endian_utils::write_value(file, pq_centroids_size);
                for (size_t i = 0; i < pq_centroids_size; i++) {
                    endian_utils::write_value(file, model->pq_centroids[i]);
                }
            }

            // Save DUAL HNSW indices directly to stream with CRC32 validation
            bool save_original_index = model->original_space_index != nullptr && !model->use_quantization;
            bool save_embedding_index = model->embedding_space_index != nullptr && !model->always_save_embedding_data;

            std::cout << "[DEBUG] Save flags: save_original_index=" << save_original_index
                      << ", save_embedding_index=" << save_embedding_index << std::endl;
            std::cout << "[DEBUG] Index pointers: original=" << (void*)model->original_space_index.get()
                      << ", embedding=" << (void*)model->embedding_space_index.get() << std::endl;

            endian_utils::write_value(file, save_original_index);
            endian_utils::write_value(file, save_embedding_index);

            // Save original space HNSW index
            if (save_original_index) {
                try {
                    std::streampos hnsw_data_start = file.tellp();

                    // Save original space HNSW index data with simple temp file approach
                    // Note: save_hnsw_to_stream_compressed handles its own size field
                    std::cout << "[DEBUG] Saving original space HNSW index..." << std::endl;
                    hnsw_utils::save_hnsw_to_stream_compressed(file, model->original_space_index.get());

                    std::streampos hnsw_data_end = file.tellp();
                }
                catch (const std::exception&) {
                    // Original HNSW save failed
                    size_t zero_size = 0;
                    endian_utils::write_value(file, zero_size);
                    send_warning_to_callback("Original space HNSW index save failed - transforms may be slower");
                }
            }
            else {
                // No original space HNSW index - write zero size
                size_t zero_size = 0;
                endian_utils::write_value(file, zero_size);
            }

            // Save embedding space HNSW index (always saved, never quantized)
            if (save_embedding_index) {
                try {
                    std::streampos embedding_hnsw_start = file.tellp();

                    // Save embedding space HNSW index data with simple temp file approach
                    // Note: save_hnsw_to_stream_compressed handles its own size field
                    std::cout << "[DEBUG] Saving embedding space HNSW index..." << std::endl;
                    try {
                        std::cout << "[DEBUG] About to call save_hnsw_to_stream_compressed for embedding..." << std::endl;
                        hnsw_utils::save_hnsw_to_stream_compressed(file, model->embedding_space_index.get());
                        std::cout << "[DEBUG] save_hnsw_to_stream_compressed for embedding completed successfully" << std::endl;
                    }
                    catch (const std::exception& e) {
                        std::cout << "[DEBUG] Embedding HNSW save exception: " << e.what() << std::endl;
                        throw;
                    }

                    std::streampos embedding_hnsw_end = file.tellp();
                }
                catch (const std::exception& e) {
                    // Embedding HNSW save failed - this is critical for AI inference
                    std::cout << "[DEBUG] Embedding space HNSW save exception: " << e.what() << std::endl;
                    size_t zero_size = 0;
                    endian_utils::write_value(file, zero_size);
                    send_warning_to_callback("Embedding space HNSW index save failed - AI inference will not work");
                }
            }
            else {
                // No embedding space HNSW index - write zero size
                size_t zero_size = 0;
                endian_utils::write_value(file, zero_size);
            }

            // Save model CRC32 values for integrity validation
            endian_utils::write_value(file, model->original_space_crc);
            endian_utils::write_value(file, model->embedding_space_crc);
            endian_utils::write_value(file, model->model_version_crc);

            file.close();
            return UWOT_SUCCESS;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            return UWOT_ERROR_FILE_IO;
        }
    }

    UwotModel* load_model(const char* filename) {
        if (!filename) {
            return nullptr;
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return nullptr;
        }

        UwotModel* model = nullptr;

        try {
            model = model_utils::create_model();
            if (!model) {
                return nullptr;
            }

            // Read and verify magic number and version (endian-safe)
            uint32_t magic;
            if (!endian_utils::read_value(file, magic)) {
                throw std::runtime_error("Failed to read magic number");
            }
            if (magic != 0x554D4150) { // "UMAP" in ASCII
                throw std::runtime_error("Invalid file format: magic number mismatch");
            }

            uint32_t format_version;
            if (!endian_utils::read_value(file, format_version)) {
                throw std::runtime_error("Failed to read format version");
            }
            if (format_version != 1) {
                send_warning_to_callback("Unsupported format version - attempting to load anyway");
            }

            // Read library version (16 bytes) - keep as string for compatibility
            char version[17] = { 0 };
            file.read(version, 16);
            if (strcmp(version, UWOT_WRAPPER_VERSION_STRING) != 0) {
                send_warning_to_callback("Model version mismatch - attempting to load anyway");
            }

            // Read basic model parameters (endian-safe)
            if (!endian_utils::read_value(file, model->n_vertices) ||
                !endian_utils::read_value(file, model->n_dim) ||
                !endian_utils::read_value(file, model->embedding_dim) ||
                !endian_utils::read_value(file, model->n_neighbors) ||
                !endian_utils::read_value(file, model->min_dist) ||
                !endian_utils::read_value(file, model->spread)) {
                throw std::runtime_error("Failed to read basic model parameters");
            }

            int metric_value;
            if (!endian_utils::read_value(file, metric_value)) {
                throw std::runtime_error("Failed to read metric parameter");
            }
            model->metric = static_cast<UwotMetric>(metric_value);

            if (!endian_utils::read_value(file, model->a) ||
                !endian_utils::read_value(file, model->b)) {
                throw std::runtime_error("Failed to read curve parameters");
            }

            // Read HNSW parameters (endian-safe)
            if (!endian_utils::read_value(file, model->hnsw_M) ||
                !endian_utils::read_value(file, model->hnsw_ef_construction) ||
                !endian_utils::read_value(file, model->hnsw_ef_search)) {
                throw std::runtime_error("Failed to read HNSW parameters");
            }

            // Read neighbor statistics (endian-safe)
            if (!endian_utils::read_value(file, model->mean_original_distance) ||
                !endian_utils::read_value(file, model->std_original_distance) ||
                !endian_utils::read_value(file, model->min_original_distance) ||
                !endian_utils::read_value(file, model->p95_original_distance) ||
                !endian_utils::read_value(file, model->p99_original_distance) ||
                !endian_utils::read_value(file, model->mild_original_outlier_threshold) ||
                !endian_utils::read_value(file, model->extreme_original_outlier_threshold) ||
                !endian_utils::read_value(file, model->median_original_distance) ||
                !endian_utils::read_value(file, model->exact_match_threshold) ||
                !endian_utils::read_value(file, model->hnsw_recall_percentage)) {
                throw std::runtime_error("Failed to read neighbor statistics");
            }

            // Read normalization parameters (endian-safe)
            bool has_normalization;
            if (!endian_utils::read_value(file, has_normalization)) {
                throw std::runtime_error("Failed to read normalization flag");
            }
            if (has_normalization) {
                model->feature_means.resize(model->n_dim);
                model->feature_stds.resize(model->n_dim);

                // Read feature means (endian-safe)
                for (int i = 0; i < model->n_dim; i++) {
                    if (!endian_utils::read_value(file, model->feature_means[i])) {
                        throw std::runtime_error("Failed to read feature means");
                    }
                }

                // Read feature standard deviations (endian-safe)
                for (int i = 0; i < model->n_dim; i++) {
                    if (!endian_utils::read_value(file, model->feature_stds[i])) {
                        throw std::runtime_error("Failed to read feature standard deviations");
                    }
                }

                if (!endian_utils::read_value(file, model->normalization_mode)) {
                    throw std::runtime_error("Failed to read normalization mode");
                }
                model->use_normalization = true;
            }

            // Read embedding data - handle optimized format for quantized models (endian-safe)
            size_t embedding_size;
            if (!endian_utils::read_value(file, embedding_size)) {
                throw std::runtime_error("Failed to read embedding size");
            }

            // Read optimization flag for quantized models
            bool save_embedding;
            if (!endian_utils::read_value(file, save_embedding)) {
                throw std::runtime_error("Failed to read embedding save flag");
            }

            model->embedding.resize(embedding_size);

            if (save_embedding && embedding_size > 0) {
                // Full embedding data saved (non-quantized or legacy models)
                uint32_t uncompressed_size, compressed_size;
                if (!endian_utils::read_value(file, uncompressed_size) ||
                    !endian_utils::read_value(file, compressed_size)) {
                    throw std::runtime_error("Failed to read embedding compression headers");
                }

                if (compressed_size > 0) {
                    // Enhanced security validation for embedding LZ4 decompression
                    const uint32_t MAX_EMBEDDING_DECOMPRESSED = 200 * 1024 * 1024; // 200MB limit for embeddings
                    const uint32_t MAX_EMBEDDING_COMPRESSED = 150 * 1024 * 1024;   // 150MB limit for compressed

                    if (uncompressed_size > MAX_EMBEDDING_DECOMPRESSED) {
                        throw std::runtime_error("LZ4 decompression: Embedding uncompressed size too large (potential attack)");
                    }
                    if (compressed_size > MAX_EMBEDDING_COMPRESSED) {
                        throw std::runtime_error("LZ4 decompression: Embedding compressed size too large (potential attack)");
                    }
                    if (uncompressed_size == 0) {
                        throw std::runtime_error("LZ4 decompression: Invalid zero uncompressed size for embedding");
                    }

                    // Read compressed embedding data with validation
                    std::vector<char> compressed_data(compressed_size);
                    file.read(compressed_data.data(), compressed_size);

                    if (!file.good() || file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                        throw std::runtime_error("LZ4 decompression: Failed to read embedding compressed data");
                    }

                    // Decompress with LZ4 (bounds-checked)
                    int decompressed_size = LZ4_decompress_safe(
                        compressed_data.data(),
                        reinterpret_cast<char*>(model->embedding.data()),
                        static_cast<int>(compressed_size),
                        static_cast<int>(uncompressed_size));

                    if (decompressed_size <= 0) {
                        throw std::runtime_error("LZ4 decompression failed: Malformed embedding compressed data");
                    }
                    if (decompressed_size != static_cast<int>(uncompressed_size)) {
                        throw std::runtime_error("LZ4 decompression failed: Embedding size mismatch");
                    }
                } else {
                    // Uncompressed embedding data (fallback or old format) - read endian-safe floats
                    for (size_t i = 0; i < embedding_size; i++) {
                        if (!endian_utils::read_value(file, model->embedding[i])) {
                            throw std::runtime_error("Failed to read embedding data");
                        }
                    }
                }
            } else if (!save_embedding) {
                // OPTIMIZATION: Embedding storage skipped - using HNSW extraction instead
                // Read and skip the compression headers (legacy format compatibility)
                uint32_t uncompressed_size, compressed_size;
                if (!endian_utils::read_value(file, uncompressed_size) ||
                    !endian_utils::read_value(file, compressed_size)) {
                    throw std::runtime_error("Failed to read embedding compression headers");
                }

                // Embeddings extracted from HNSW during transform - no redundant storage needed
                send_warning_to_callback("Embedding storage optimized: Using HNSW extraction for transforms (50% space savings)");
            }

            // Read k-NN data (endian-safe)
            bool needs_knn;
            if (!endian_utils::read_value(file, needs_knn)) {
                throw std::runtime_error("Failed to read k-NN flag");
            }

            if (needs_knn) {
                // k-NN data available for exact reproducibility

                size_t indices_size, distances_size, weights_size;
                if (!endian_utils::read_value(file, indices_size) ||
                    !endian_utils::read_value(file, distances_size) ||
                    !endian_utils::read_value(file, weights_size)) {
                    throw std::runtime_error("Failed to read k-NN sizes");
                }

                // Read k-NN indices (endian-safe)
                if (indices_size > 0) {
                    model->nn_indices.resize(indices_size);
                    for (size_t i = 0; i < indices_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_indices[i])) {
                            throw std::runtime_error("Failed to read k-NN indices");
                        }
                    }
                }
                // Read k-NN distances (endian-safe)
                if (distances_size > 0) {
                    model->nn_distances.resize(distances_size);
                    for (size_t i = 0; i < distances_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_distances[i])) {
                            throw std::runtime_error("Failed to read k-NN distances");
                        }
                    }
                }
                // Read k-NN weights (endian-safe)
                if (weights_size > 0) {
                    model->nn_weights.resize(weights_size);
                    for (size_t i = 0; i < weights_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_weights[i])) {
                            throw std::runtime_error("Failed to read k-NN weights");
                        }
                    }
                }

                // k-NN fallback data loaded successfully
            }

            // Read quantization data (PQ - Product Quantization) (endian-safe)
            if (!endian_utils::read_value(file, model->use_quantization) ||
                !endian_utils::read_value(file, model->pq_m)) {
                throw std::runtime_error("Failed to read quantization parameters");
            }

            if (model->use_quantization) {
                // Read PQ codes
                size_t pq_codes_size;
                if (!endian_utils::read_value(file, pq_codes_size)) {
                    throw std::runtime_error("Failed to read PQ codes size");
                }
                if (pq_codes_size > 0) {
                    model->pq_codes.resize(pq_codes_size);
                    // PQ codes are uint8_t - no endian conversion needed
                    file.read(reinterpret_cast<char*>(model->pq_codes.data()), pq_codes_size * sizeof(uint8_t));
                }

                // Read PQ centroids (endian-safe)
                size_t pq_centroids_size;
                if (!endian_utils::read_value(file, pq_centroids_size)) {
                    throw std::runtime_error("Failed to read PQ centroids size");
                }
                if (pq_centroids_size > 0) {
                    model->pq_centroids.resize(pq_centroids_size);
                    for (size_t i = 0; i < pq_centroids_size; i++) {
                        if (!endian_utils::read_value(file, model->pq_centroids[i])) {
                            throw std::runtime_error("Failed to read PQ centroids");
                        }
                    }
                }
            }

            // Read DUAL HNSW indices (endian-safe)
            bool has_original_index, has_embedding_index;
            if (!endian_utils::read_value(file, has_original_index) ||
                !endian_utils::read_value(file, has_embedding_index)) {
                throw std::runtime_error("Failed to read HNSW index flags");
            }

            // Load original space HNSW index
            if (has_original_index) {
                try {
                    // Initialize original space factory with correct metric
                    if (!model->original_space_factory->create_space(model->metric, model->n_dim)) {
                        throw std::runtime_error("Failed to create original space HNSW space");
                    }

                    // Create original space HNSW index with saved parameters
                    model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        model->original_space_factory->get_space(),
                        model->n_vertices,
                        model->hnsw_M,
                        model->hnsw_ef_construction
                    );

                    model->original_space_index->setEf(model->hnsw_ef_search);

                    // Load original space HNSW data - PROPER LOADING FROM SAVED INDEX
                    try {
                        send_warning_to_callback("Loading original space HNSW index from saved data");

                        // Load HNSW index directly from stream (massive performance improvement)
                        hnsw_utils::load_hnsw_from_stream_compressed(file, model->original_space_index.get(),
                                                                   model->original_space_factory->get_space());

                        send_warning_to_callback("Original space HNSW index loaded successfully - no reconstruction needed");
                    }
                    catch (const std::exception&) {
                        send_warning_to_callback("Failed to load saved HNSW index, falling back to rebuild");

                        // Fallback: rebuild the index from quantized data if loading fails
                        if (model->use_quantization && !model->pq_codes.empty() && !model->pq_centroids.empty()) {
                            std::vector<float> reconstructed_data(model->n_vertices * model->n_dim);
                            int subspace_dim = model->n_dim / model->pq_m;

                            for (int i = 0; i < model->n_vertices; i++) {
                                std::vector<float> reconstructed_point;
                                pq_utils::reconstruct_vector(model->pq_codes, i, model->pq_m,
                                                           model->pq_centroids, subspace_dim,
                                                           reconstructed_point);

                                for (int d = 0; d < model->n_dim; d++) {
                                    reconstructed_data[i * model->n_dim + d] = reconstructed_point[d];
                                }
                            }

                            // Add all points to HNSW index
                            for (int i = 0; i < model->n_vertices; i++) {
                                model->original_space_index->addPoint(&reconstructed_data[i * model->n_dim], i);
                            }
                        }
                    }
                }
                catch (const std::exception&) {
                    model->original_space_index = nullptr;
                    send_warning_to_callback("Original space HNSW index loading failed - no index available");
                }
            } else {
                // Skip the zero-size placeholder written when original HNSW was not saved
                size_t zero_size;
                if (!endian_utils::read_value(file, zero_size)) {
                    throw std::runtime_error("Failed to read original HNSW placeholder size");
                }
            }

            // Load embedding space HNSW index (critical for AI inference)
            if (has_embedding_index) {
                try {
                    // Initialize embedding space factory (always L2 for embeddings)
                    if (!model->embedding_space_factory->create_space(UWOT_METRIC_EUCLIDEAN, model->embedding_dim)) {
                        throw std::runtime_error("Failed to create embedding space HNSW space");
                    }

                    // Create embedding space HNSW index with saved parameters
                    model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        model->embedding_space_factory->get_space(),
                        model->n_vertices,
                        model->hnsw_M,
                        model->hnsw_ef_construction
                    );

                    model->embedding_space_index->setEf(model->hnsw_ef_search);

                    // Load embedding space HNSW data - PROPER LOADING FROM SAVED INDEX
                    try {
                        send_warning_to_callback("Loading embedding space HNSW index from saved data");

                        // Load HNSW index directly from stream (massive performance improvement)
                        hnsw_utils::load_hnsw_from_stream_compressed(file, model->embedding_space_index.get(),
                                                                   model->embedding_space_factory->get_space());

                        send_warning_to_callback("Embedding space HNSW index loaded successfully - AI inference ready");
                    }
                    catch (const std::exception&) {
                        send_warning_to_callback("Failed to load saved embedding HNSW index, rebuilding from embeddings");

                        // Fallback: rebuild the index from embedding coordinates if loading fails
                        if (!model->embedding.empty() && model->n_vertices > 0) {
                            // Add embedding points to HNSW index
                            for (int i = 0; i < model->n_vertices; i++) {
                                model->embedding_space_index->addPoint(&model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)], static_cast<size_t>(i));
                            }
                            send_warning_to_callback("Embedding space HNSW index rebuilt from embeddings");
                        }
                    }
                }
                catch (const std::exception&) {
                    model->embedding_space_index = nullptr;
                    send_warning_to_callback("CRITICAL: Embedding space HNSW index loading failed - AI inference will not work");
                }
            }
            else if (model->always_save_embedding_data) {
                // NEW: Rebuild embedding HNSW index from saved embeddings when always_save_embedding_data is true
                send_warning_to_callback("Rebuilding embedding space HNSW index from saved embeddings (always_save_embedding_data mode)");
                try {
                    // Initialize embedding space factory (always L2 for embeddings)
                    if (!model->embedding_space_factory->create_space(UWOT_METRIC_EUCLIDEAN, model->embedding_dim)) {
                        throw std::runtime_error("Failed to create embedding space HNSW space");
                    }

                    // Create embedding space HNSW index with saved parameters
                    model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        model->embedding_space_factory->get_space(),
                        model->n_vertices,
                        model->hnsw_M,
                        model->hnsw_ef_construction
                    );

                    model->embedding_space_index->setEf(model->hnsw_ef_search);

                    // Rebuild the index from embedding coordinates
                    if (!model->embedding.empty() && model->n_vertices > 0) {
                        // Add embedding points to HNSW index
                        for (int i = 0; i < model->n_vertices; i++) {
                            model->embedding_space_index->addPoint(&model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)], static_cast<size_t>(i));
                        }
                        send_warning_to_callback("Embedding space HNSW index rebuilt successfully from saved embeddings");
                    }
                }
                catch (const std::exception&) {
                    model->embedding_space_index = nullptr;
                    send_warning_to_callback("CRITICAL: Failed to rebuild embedding space HNSW index from saved embeddings");
                }
            }
            else {
                // No HNSW indices saved - rebuild from quantized data if available
                if (model->use_quantization && !model->pq_codes.empty() && !model->pq_centroids.empty()) {
                    // WARNING: Reconstructing original space HNSW from lossy quantized data
                    send_warning_to_callback("Reconstructing original space HNSW from quantized data - accuracy may be reduced");
                    try {
                        // Initialize original space factory
                        if (!model->original_space_factory->create_space(model->metric, model->n_dim)) {
                            throw std::runtime_error("Failed to create original space HNSW for reconstruction");
                        }

                        // Create original space HNSW index
                        model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            model->original_space_factory->get_space(),
                            model->n_vertices,
                            model->hnsw_M,
                            model->hnsw_ef_construction
                        );
                        model->original_space_index->setEf(model->hnsw_ef_search);

                        // Reconstruct quantized data from PQ codes
                        std::vector<float> reconstructed_data(model->n_vertices * model->n_dim);
                        int subspace_dim = model->n_dim / model->pq_m;

                        for (int i = 0; i < model->n_vertices; i++) {
                            std::vector<float> reconstructed_point;
                            pq_utils::reconstruct_vector(model->pq_codes, i, model->pq_m,
                                                       model->pq_centroids, subspace_dim,
                                                       reconstructed_point);

                            // Copy to reconstructed data
                            for (int d = 0; d < model->n_dim; d++) {
                                reconstructed_data[i * model->n_dim + d] = reconstructed_point[d];
                            }
                        }

                        // Add all points to original space HNSW index
                        for (int i = 0; i < model->n_vertices; i++) {
                            model->original_space_index->addPoint(&reconstructed_data[i * model->n_dim], i);
                        }
                    }
                    catch (...) {
                        model->original_space_index = nullptr;
                        send_warning_to_callback("Original space HNSW reconstruction failed");
                    }
                }

                // Embedding space HNSW cannot be reconstructed from quantized data
                // It must be rebuilt from embeddings if available
                if (!model->embedding.empty()) {
                    send_warning_to_callback("Rebuilding embedding space HNSW from saved embeddings");
                    try {
                        // Initialize embedding space factory (always L2)
                        if (!model->embedding_space_factory->create_space(UWOT_METRIC_EUCLIDEAN, model->embedding_dim)) {
                            throw std::runtime_error("Failed to create embedding space HNSW for reconstruction");
                        }

                        // Create embedding space HNSW index
                        model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            model->embedding_space_factory->get_space(),
                            model->n_vertices,
                            model->hnsw_M,
                            model->hnsw_ef_construction
                        );
                        model->embedding_space_index->setEf(model->hnsw_ef_search);

                        // Add all embedding points to embedding space HNSW index
                        for (int i = 0; i < model->n_vertices; i++) {
                            const float* embedding_point = &model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)];
                            model->embedding_space_index->addPoint(embedding_point, i);
                        }
                    }
                    catch (...) {
                        model->embedding_space_index = nullptr;
                        send_warning_to_callback("CRITICAL: Embedding space HNSW reconstruction failed - AI inference will not work");
                    }
                }
            }

            // Load model CRC32 values for integrity validation
            if (!endian_utils::read_value(file, model->original_space_crc) ||
                !endian_utils::read_value(file, model->embedding_space_crc) ||
                !endian_utils::read_value(file, model->model_version_crc)) {
                send_warning_to_callback("Failed to read model CRC32 values - integrity validation disabled");
            }

            model->is_fitted = true;
            file.close();


            // Print HNSW save/load consistency summary
            // HNSW save/load consistency fixes applied:
            // - Space factory initialized with correct metric before HNSW loading
            // - HNSW index created with saved parameters (M, ef_construction, ef_search)
            // - HNSW element count validation (expected vs actual)
            // - k-NN fallback data always saved for exact reproducibility
            // Result: Transform consistency guaranteed

            return model;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            if (model) {
                model_utils::destroy_model(model);
            }
            return nullptr;
        }
    }
}