#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// Endian-safe serialization utilities
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
        to_little_endian(value);
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

int main() {
    std::cout << "=== Simple HNSW Stream Debug Test ===" << std::endl;

    // Create test data
    std::string test_data = "Hello, HNSW World! This is test data for stream serialization.";
    uint32_t test_size = static_cast<uint32_t>(test_data.size());
    uint32_t test_crc = 0x12345678;

    std::cout << "Original data: " << test_data << std::endl;
    std::cout << "Data size: " << test_size << std::endl;
    std::cout << "Test CRC: " << std::hex << test_crc << std::dec << std::endl;
    std::cout << "System is little-endian: " << endian_utils::is_little_endian() << std::endl;

    // Test 1: Write to stringstream
    std::cout << "\n--- Test 1: StringStream ---" << std::endl;
    std::stringstream ss;

    std::cout << "Writing headers..." << std::endl;
    endian_utils::write_value(ss, test_size);
    endian_utils::write_value(ss, test_size);
    endian_utils::write_value(ss, test_crc);

    std::cout << "Writing data..." << std::endl;
    ss.write(test_data.data(), test_size);

    std::cout << "Stream position after writing: " << ss.tellg() << std::endl;

    // Reset stream position
    ss.seekg(0, std::ios::beg);
    std::cout << "Stream position after seek: " << ss.tellg() << std::endl;

    // Read back
    uint32_t read_size1, read_size2, read_crc;
    std::cout << "Reading headers..." << std::endl;
    if (endian_utils::read_value(ss, read_size1) &&
        endian_utils::read_value(ss, read_size2) &&
        endian_utils::read_value(ss, read_crc)) {
        std::cout << "✅ Read successful!" << std::endl;
        std::cout << "Read size1: " << read_size1 << std::endl;
        std::cout << "Read size2: " << read_size2 << std::endl;
        std::cout << "Read CRC: " << std::hex << read_crc << std::dec << std::endl;

        if (read_size1 == test_size && read_size2 == test_size && read_crc == test_crc) {
            std::cout << "✅ StringStream test PASSED!" << std::endl;
        } else {
            std::cout << "❌ StringStream test FAILED - data mismatch!" << std::endl;
        }
    } else {
        std::cout << "❌ StringStream test FAILED - read error!" << std::endl;
    }

    // Test 2: Write to file
    std::cout << "\n--- Test 2: File Stream ---" << std::endl;
    const char* filename = "debug_hnsw_stream.bin";

    {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "❌ Failed to create file" << std::endl;
            return 1;
        }

        std::cout << "Writing headers to file..." << std::endl;
        endian_utils::write_value(file, test_size);
        endian_utils::write_value(file, test_size);
        endian_utils::write_value(file, test_crc);

        std::cout << "Writing data to file..." << std::endl;
        file.write(test_data.data(), test_size);

        std::cout << "File position after writing: " << file.tellp() << std::endl;
        file.close();
    }

    // Read from file
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "❌ Failed to open file for reading" << std::endl;
            return 1;
        }

        std::cout << "File position before reading: " << file.tellg() << std::endl;

        uint32_t file_size1, file_size2, file_crc;
        std::cout << "Reading headers from file..." << std::endl;
        if (endian_utils::read_value(file, file_size1) &&
            endian_utils::read_value(file, file_size2) &&
            endian_utils::read_value(file, file_crc)) {
            std::cout << "✅ File read successful!" << std::endl;
            std::cout << "Read size1: " << file_size1 << std::endl;
            std::cout << "Read size2: " << file_size2 << std::endl;
            std::cout << "Read CRC: " << std::hex << file_crc << std::dec << std::endl;

            if (file_size1 == test_size && file_size2 == test_size && file_crc == test_crc) {
                std::cout << "✅ File stream test PASSED!" << std::endl;
            } else {
                std::cout << "❌ File stream test FAILED - data mismatch!" << std::endl;
            }
        } else {
            std::cout << "❌ File stream test FAILED - read error!" << std::endl;
        }

        file.close();
    }

    // Cleanup
    std::remove(filename);

    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}