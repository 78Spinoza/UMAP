#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

namespace crc_utils {

    // CRC32 table for fast computation
    extern const uint32_t crc32_table[256];

    // Initialize CRC32 table (called automatically)
    void init_crc32_table();

    // Compute CRC32 of data
    uint32_t compute_crc32(const void* data, size_t length) noexcept;

    // Compute CRC32 of file
    uint32_t compute_file_crc32(const std::string& filename) noexcept;

    // Compute CRC32 of vector
    template<typename T>
    uint32_t compute_vector_crc32(const std::vector<T>& data) noexcept {
        return compute_crc32(data.data(), data.size() * sizeof(T));
    }

    // Validate CRC32 with detailed error reporting
    bool validate_crc32(const void* data, size_t length, uint32_t expected_crc) noexcept;
    bool validate_file_crc32(const std::string& filename, uint32_t expected_crc) noexcept;

    // CRC32 manipulation utilities
    uint32_t combine_crc32(uint32_t crc1, uint32_t crc2, size_t len2) noexcept;
    std::string crc32_to_string(uint32_t crc) noexcept;
    uint32_t string_to_crc32(const std::string& str) noexcept;

} // namespace crc_utils