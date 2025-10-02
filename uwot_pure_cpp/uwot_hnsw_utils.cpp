#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include "uwot_crc32.h"
#include "lz4.h"

// Endian-safe serialization utilities (inline for HNSW)
namespace endian_utils {
    bool is_little_endian() {
        uint16_t test = 0x1234;
        return *reinterpret_cast<uint8_t*>(&test) == 0x34;
    }

    template<typename T>
    void to_little_endian(T& value) {
        if (!is_little_endian()) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }
    }

    template<typename T>
    void from_little_endian(T& value) {
        to_little_endian(value); // Same operation - byte swap if needed
    }

    template<typename T>
    void write_value(std::ostream& output, const T& value) {
        T little_endian_value = value;
        to_little_endian(little_endian_value);
        output.write(reinterpret_cast<const char*>(&little_endian_value), sizeof(T));
    }

    template<typename T>
    bool read_value(std::istream& input, T& value) {
        input.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (input.good()) {
            from_little_endian(value);
            return true;
        }
        return false;
    }
}
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <filesystem>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// HNSW space factory implementation
namespace hnsw_utils {

    bool SpaceFactory::create_space(UwotMetric metric, int n_dim) {
        current_metric = metric;
        current_dim = n_dim;

        // Clean up existing spaces
        l2_space.reset();
        ip_space.reset();
        l1_space.reset();

        try {
            switch (metric) {
            case UWOT_METRIC_EUCLIDEAN:
                l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                return true;

            case UWOT_METRIC_COSINE:
                ip_space = std::make_unique<hnswlib::InnerProductSpace>(n_dim);
                return true;

            case UWOT_METRIC_MANHATTAN:
                l1_space = std::make_unique<L1Space>(n_dim);
                return true;

            default:
                return false; // Unsupported metric for HNSW
            }
        }
        catch (...) {
            return false;
        }
    }

    hnswlib::SpaceInterface<float>* SpaceFactory::get_space() {
        switch (current_metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return l2_space.get();
        case UWOT_METRIC_COSINE:
            return ip_space.get();
        case UWOT_METRIC_MANHATTAN:
            return l1_space.get();
        default:
            return nullptr;
        }
    }

    bool SpaceFactory::can_use_hnsw() const {
        return current_metric == UWOT_METRIC_EUCLIDEAN ||
               current_metric == UWOT_METRIC_COSINE ||
               current_metric == UWOT_METRIC_MANHATTAN;
    }

    // HNSW stream utilities implementation
    namespace hnsw_stream_utils {

        std::string generate_unique_temp_filename(const std::string& prefix) {
            try {
                // Use std::filesystem for secure temp directory
                std::filesystem::path temp_dir = std::filesystem::temp_directory_path();

                // Generate cryptographically secure filename
                std::random_device rd;
                std::mt19937_64 gen(rd()); // Use 64-bit generator for better entropy
                std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

                // Create multiple random components to prevent prediction
                uint64_t random1 = dis(gen);
                uint64_t random2 = dis(gen);
                auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

                std::ostringstream oss;
                oss << prefix << "_" << std::hex << timestamp << "_" << random1 << "_" << random2 << ".tmp";

                std::filesystem::path temp_file = temp_dir / oss.str();
                return temp_file.string();
            }
            catch (...) {
                // Fallback to current directory if temp dir access fails
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
                uint64_t random = dis(gen);

                std::ostringstream oss;
                oss << prefix << "_fallback_" << std::hex << random << ".tmp";
                return oss.str();
            }
        }

        void save_hnsw_to_stream(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
            std::string temp_filename = generate_unique_temp_filename("hnsw_save");

            try {
                // Save HNSW index to temporary file
                hnsw_index->saveIndex(temp_filename);

                // Read the temporary file and stream it directly
                std::ifstream temp_file(temp_filename, std::ios::binary);
                if (!temp_file.is_open()) {
                    throw std::runtime_error("Failed to open temporary HNSW file for reading");
                }

                // Stream the file contents
                output << temp_file.rdbuf();
                temp_file.close();

                // Clean up temporary file
                temp_utils::safe_remove_file(temp_filename);
            }
            catch (...) {
                // Ensure cleanup on error
                temp_utils::safe_remove_file(temp_filename);
                throw;
            }
        }

        void load_hnsw_from_stream(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
            hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
            std::string temp_filename = generate_unique_temp_filename("hnsw_load");

            try {
                // Write stream data to temporary file
                std::ofstream temp_file(temp_filename, std::ios::binary);
                if (!temp_file.is_open()) {
                    throw std::runtime_error("Failed to create temporary HNSW file");
                }

                // Copy specified amount of data from stream to file
                std::vector<char> buffer(8192);
                size_t remaining = hnsw_size;
                while (remaining > 0 && input.good()) {
                    size_t to_read = std::min(remaining, buffer.size());
                    input.read(buffer.data(), to_read);
                    size_t actually_read = input.gcount();
                    if (actually_read > 0) {
                        temp_file.write(buffer.data(), actually_read);
                        remaining -= actually_read;
                    }
                    else {
                        break;
                    }
                }
                temp_file.close();

                // Load from temporary file
                hnsw_index->loadIndex(temp_filename, space);

                // Clean up temporary file
                temp_utils::safe_remove_file(temp_filename);
            }
            catch (...) {
                // Ensure cleanup on error
                temp_utils::safe_remove_file(temp_filename);
                throw;
            }
        }
    }

    // STREAM-ONLY APPROACH WITH SIZE HEADERS AND CRC32 - Enhanced HNSW stream serialization
    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        std::cout << "[STREAM] HNSW Save: Starting stream-only approach with size headers and CRC32..." << std::endl;

        if (!hnsw_index) {
            throw std::runtime_error("HNSW index is null");
        }

        try {
            // Use stringstream to capture HNSW data for CRC computation
            std::stringstream hnsw_data_stream;
            hnsw_index->saveIndex(hnsw_data_stream);

            // Get the HNSW data as string for CRC computation
            std::string hnsw_data = hnsw_data_stream.str();
            uint32_t actual_size = static_cast<uint32_t>(hnsw_data.size());

            // Compute CRC32 of the HNSW data
            uint32_t data_crc32 = crc_utils::compute_crc32(hnsw_data.data(), actual_size);

            // Write headers to output stream using endian-safe functions
            std::cout << "[STREAM] HNSW Save: Writing headers - size: " << actual_size
                      << ", CRC32: " << std::hex << data_crc32 << std::dec << std::endl;
            endian_utils::write_value(output, actual_size);
            endian_utils::write_value(output, actual_size);
            endian_utils::write_value(output, data_crc32);

            // Check if header write was successful
            if (!output.good()) {
                throw std::runtime_error("Failed to write HNSW headers to stream");
            }

            std::cout << "[STREAM] HNSW Save: Headers written successfully" << std::endl;

            // Write HNSW data to output stream
            output.write(hnsw_data.data(), actual_size);

            // Check if data write was successful
            if (!output.good()) {
                throw std::runtime_error("Failed to write HNSW data to stream");
            }

            // Flush the stream to ensure data is written
            output.flush();

            std::cout << "[STREAM] HNSW Save: Stream saveIndex() with " << actual_size
                      << " bytes, CRC32: " << std::hex << data_crc32 << std::dec
                      << " completed successfully" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "[STREAM] HNSW Save: Exception: " << e.what() << std::endl;
            throw;
        }
    }

    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space) {
        try {
            std::cout << "[STREAM] HNSW Load: Starting stream-only approach with CRC32 validation..." << std::endl;

            // Check stream state first
            if (!input.good()) {
                throw std::runtime_error("Input stream is in bad state before reading headers");
            }

            // Check current stream position
            std::streampos current_pos = input.tellg();
            std::cout << "[STREAM] HNSW Load: Current stream position: " << current_pos << std::endl;

            // Read headers using endian-safe functions for compatibility with existing format
            uint32_t original_size, compressed_size, expected_crc32;
            if (!endian_utils::read_value(input, original_size) ||
                !endian_utils::read_value(input, compressed_size) ||
                !endian_utils::read_value(input, expected_crc32)) {
                throw std::runtime_error("Failed to read HNSW headers - stream error or EOF");
            }

            std::cout << "[STREAM] HNSW Load: Read headers - original: " << original_size
                      << ", compressed: " << compressed_size
                      << ", CRC32: " << std::hex << expected_crc32 << std::dec << std::endl;

            // Validate sizes - allow zero if this is a marker for no data
            if (original_size == 0 && compressed_size == 0) {
                std::cout << "[STREAM] HNSW Load: Zero size headers detected - this might indicate empty HNSW data" << std::endl;
                throw std::runtime_error("Invalid HNSW size headers - both sizes are zero");
            }

            // Read HNSW data into buffer for CRC validation
            std::vector<char> data_buffer(original_size);
            input.read(data_buffer.data(), original_size);

            if (!input.good() || input.gcount() != static_cast<std::streamsize>(original_size)) {
                throw std::runtime_error("Failed to read HNSW data");
            }

            // Compute and validate CRC32 before loading
            uint32_t computed_crc32 = crc_utils::compute_crc32(data_buffer.data(), original_size);

            std::cout << "[STREAM] HNSW Load: Computed CRC32: " << std::hex << computed_crc32
                      << ", Expected: " << expected_crc32 << std::dec << std::endl;

            if (computed_crc32 != expected_crc32) {
                throw std::runtime_error("HNSW data CRC32 validation failed - index corruption detected!");
            }

            // Create stringstream from validated data for loading
            std::stringstream data_stream;
            data_stream.write(data_buffer.data(), original_size);

            // Load HNSW from validated data stream
            hnsw_index->loadIndex(data_stream, space);

            std::cout << "[STREAM] HNSW Load: ✅ CRC32 validation passed - Stream loadIndex() completed successfully" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "[STREAM] HNSW Load: Exception: " << e.what() << std::endl;
            throw;
        }
    }

    // HNSW k-NN query utilities
    void build_knn_graph_hnsw(const std::vector<float>& data, int n_obs, int n_dim, int n_neighbors,
        hnswlib::HierarchicalNSW<float>* hnsw_index, std::vector<int>& nn_indices,
        std::vector<double>& nn_distances) {

        nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
        nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

        // Use HNSW for fast approximate k-NN queries
#pragma omp parallel for if(n_obs > 1000)
        for (int i = 0; i < n_obs; i++) {
            // Query HNSW index for k+1 neighbors (includes self)
            std::vector<float> query_data(data.begin() + static_cast<size_t>(i) * static_cast<size_t>(n_dim),
                data.begin() + static_cast<size_t>(i + 1) * static_cast<size_t>(n_dim));

            // CRITICAL SAFETY CHECK: Ensure HNSW index is valid
            if (!hnsw_index) {
                continue; // Skip this iteration if no index
            }

            try {
                // Query for k+1 neighbors to account for self-match
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                    hnsw_index->searchKnn(query_data.data(), n_neighbors + 1);

                // Extract results, skipping self-match
                std::vector<std::pair<float, int>> neighbors;
                neighbors.reserve(n_neighbors + 1);

                while (!result.empty()) {
                    auto& top = result.top();
                    int neighbor_id = static_cast<int>(top.second);
                    if (neighbor_id != i) { // Skip self-match
                        neighbors.emplace_back(top.first, neighbor_id);
                    }
                    result.pop();
                }

                // Reverse to get nearest first, and take only n_neighbors
                std::reverse(neighbors.begin(), neighbors.end());
                if (neighbors.size() > static_cast<size_t>(n_neighbors)) {
                    neighbors.resize(n_neighbors);
                }

                // Fill the arrays
                for (int j = 0; j < n_neighbors && j < static_cast<int>(neighbors.size()); j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = neighbors[j].second;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = static_cast<double>(neighbors[j].first);
                }

                // Fill remaining slots with -1 and high distance if not enough neighbors
                for (int j = static_cast<int>(neighbors.size()); j < n_neighbors; j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = -1;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = 1e6;
                }
            }
            catch (...) {
                // Handle HNSW query errors gracefully
                for (int j = 0; j < n_neighbors; j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = -1;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = 1e6;
                }
            }
        }
    }

    // HNSW index creation and management
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> create_hnsw_index(
        hnswlib::SpaceInterface<float>* space, int n_obs, int hnsw_M, int hnsw_ef_construction, int hnsw_ef_search) {

        // Memory estimation for HNSW index
        size_t estimated_memory_mb = ((size_t)n_obs * hnsw_M * 4 * 2) / (1024 * 1024);
        // Removed debug output for production build

        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space, n_obs, hnsw_M, hnsw_ef_construction);
        index->setEf(hnsw_ef_search);  // Set query-time ef parameter

        return index;
    }

    void add_points_to_hnsw(hnswlib::HierarchicalNSW<float>* hnsw_index,
        const std::vector<float>& normalized_data, int n_obs, int n_dim) {

        // Add all points to HNSW index using the normalized data
        // Use parallel point addition for large datasets (>5000 points)
        if (n_obs > 5000) {
#ifdef _OPENMP
            // Parallel addition for large datasets - HNSW index handles thread safety
            #pragma omp parallel for schedule(dynamic, 100)
#endif
            for (int i = 0; i < n_obs; i++) {
                hnsw_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }
        } else {
            // Sequential addition for smaller datasets to avoid OpenMP overhead
            for (int i = 0; i < n_obs; i++) {
                hnsw_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }
        }

        // Removed debug output for production build
    }

    // Temporary normalization utilities (will be moved to separate module later)
    namespace NormalizationPipeline {
        int determine_normalization_mode(UwotMetric metric) {
            // Enhanced logic for proper HNSW compatibility
            if (metric == UWOT_METRIC_COSINE) {
                return 2; // L2 normalization for cosine (HNSW InnerProductSpace requires unit vectors)
            }
            else if (metric == UWOT_METRIC_CORRELATION) {
                return 0; // No normalization for correlation
            }
            return 1; // Use z-score normalization for other metrics
        }

        bool normalize_data_consistent(std::vector<float>& input_data, std::vector<float>& output_data,
            int n_obs, int n_dim, std::vector<float>& means, std::vector<float>& stds, int mode) {

            // Resize output to match input
            output_data.resize(input_data.size());

            if (mode == 1) {
                // Check if this is training mode (large n_obs) or transform mode (n_obs = 1)
                if (n_obs > 1) {
                    // TRAINING MODE: Compute new means and stds from the data
                    means.assign(n_dim, 0.0f);
                    stds.assign(n_dim, 1.0f);

                    // Compute means
                    for (int i = 0; i < n_obs; i++) {
                        for (int j = 0; j < n_dim; j++) {
                            means[j] += input_data[static_cast<size_t>(i) * n_dim + j];
                        }
                    }
                    for (int j = 0; j < n_dim; j++) {
                        means[j] /= n_obs;
                    }

                    // Compute stds
                    for (int i = 0; i < n_obs; i++) {
                        for (int j = 0; j < n_dim; j++) {
                            float diff = input_data[static_cast<size_t>(i) * n_dim + j] - means[j];
                            stds[j] += diff * diff;
                        }
                    }
                    for (int j = 0; j < n_dim; j++) {
                        stds[j] = std::sqrt(stds[j] / (n_obs - 1));
                        if (stds[j] < 1e-8f) stds[j] = 1.0f; // Avoid division by zero
                    }
                }
                // TRANSFORM MODE: Use existing means and stds (don't overwrite them!)

                // Apply normalization to output using current means/stds
                for (int i = 0; i < n_obs; i++) {
                    for (int j = 0; j < n_dim; j++) {
                        size_t idx = static_cast<size_t>(i) * n_dim + j;
                        output_data[idx] = (input_data[idx] - means[j]) / stds[j];
                    }
                }
            }
            else if (mode == 2) {
                // Mode 2: L2 normalization (unit vectors for cosine HNSW)
                for (int i = 0; i < n_obs; i++) {
                    // Calculate L2 norm for this vector
                    float norm = 0.0f;
                    for (int j = 0; j < n_dim; j++) {
                        size_t idx = static_cast<size_t>(i) * n_dim + j;
                        float v = input_data[idx];
                        norm += v * v;
                    }
                    norm = std::sqrt(norm);

                    // Avoid division by zero
                    if (norm < 1e-8f) {
                        norm = 1.0f;
                    }

                    // Normalize to unit length
                    for (int j = 0; j < n_dim; j++) {
                        size_t idx = static_cast<size_t>(i) * n_dim + j;
                        output_data[idx] = input_data[idx] / norm;
                    }
                }
            }
            else {
                // Mode 0 or other: just copy data
                output_data = input_data;
            }

            return true;
        }
    }
}