#ifndef UWOT_ENDIAN_UTILS_HPP
#define UWOT_ENDIAN_UTILS_HPP

#include <cstdint>
#include <cstdint>
#include <algorithm>
#include <iostream>

/**
 * @file uwot_endian_utils.hpp
 * @brief Unified endian-safe serialization utilities for UMAP
 *
 * This header consolidates all endian-related utilities that were previously
 * duplicated across multiple source files. Provides consistent cross-platform
 * binary serialization support.
 */

namespace uwot {
namespace endian_utils {

    /**
     * @brief Check if the current system is little-endian
     * @return true if system is little-endian, false if big-endian
     */
    inline bool is_little_endian() noexcept {
        uint16_t test = 0x1234;
        return *reinterpret_cast<uint8_t*>(&test) == 0x34;
    }

    /**
     * @brief Convert value to/from little-endian (same operation: swap if big-endian)
     * @tparam T Type of value to convert (must be trivially copyable)
     * @param value Reference to value to convert in-place
     */
    template<typename T>
    inline void convert_endian(T& value) noexcept {
        if (!is_little_endian()) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }
    }

    /**
     * @brief Convert value to little-endian format
     * @tparam T Type of value to convert
     * @param value Reference to value to convert in-place
     */
    template<typename T>
    inline void to_little_endian(T& value) noexcept {
        convert_endian(value);
    }

    /**
     * @brief Convert value from little-endian format
     * @tparam T Type of value to convert
     * @param value Reference to value to convert in-place
     */
    template<typename T>
    inline void from_little_endian(T& value) noexcept {
        convert_endian(value); // Same operation - byte swap if needed
    }

    /**
     * @brief Write value to stream with little-endian conversion
     * @tparam T Type of value to write
     * @param stream Output stream to write to
     * @param value Value to write
     */
    template<typename T>
    inline void write_value(std::ostream& stream, const T& value) {
        T temp = value;
        convert_endian(temp);
        stream.write(reinterpret_cast<const char*>(&temp), sizeof(T));
    }

    /**
     * @brief Read value from stream with little-endian conversion
     * @tparam T Type of value to read
     * @param stream Input stream to read from
     * @param value Reference to store read value
     * @return true if read was successful, false otherwise
     */
    template<typename T>
    inline bool read_value(std::istream& stream, T& value) {
        if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T))) {
            return false;
        }
        convert_endian(value);
        return true;
    }

} // namespace endian_utils
} // namespace uwot

#endif // UWOT_ENDIAN_UTILS_HPP