#include <iostream>
#include <vector>
#include <cmath>
#include "uwot_distance.h"

// Include the implementation directly for testing
#include "uwot_distance.cpp"

// Stub implementation for missing callback function used by distance validation
void send_warning_to_callback(const char* message) {
    // For testing, just print the warning to stdout
    std::cout << "WARNING: " << message << std::endl;
}

void test_find_knn_exact() {
    std::cout << "=== Testing find_knn_exact Helper Function ===" << std::endl;

    // Create a simple test dataset
    const int n_obs = 5;
    const int n_dim = 2;
    float data[n_obs * n_dim] = {
        1.0f, 2.0f,  // Point 0
        1.1f, 2.1f,  // Point 1 (close to point 0)
        5.0f, 6.0f,  // Point 2 (far from point 0)
        1.2f, 1.9f,  // Point 3 (close to point 0)
        10.0f, 12.0f // Point 4 (very far from point 0)
    };

    std::vector<std::pair<float, int>> neighbors_out;

    // Test with query point 0, find 3 nearest neighbors
    const float* query_point = &data[0];
    distance_metrics::find_knn_exact(query_point, data, n_obs, n_dim, UWOT_METRIC_EUCLIDEAN, 3, neighbors_out, 0);

    std::cout << "Query point: [1.0, 2.0]" << std::endl;
    std::cout << "3 nearest neighbors:" << std::endl;
    for (size_t i = 0; i < neighbors_out.size(); i++) {
        std::cout << "  Neighbor " << neighbors_out[i].second << ": distance = " << neighbors_out[i].first << std::endl;
    }

    // Verify the closest neighbor should be point 1, then point 3, then point 2
    std::vector<int> expected_order = {1, 3, 2};
    bool test_passed = true;
    for (size_t i = 0; i < neighbors_out.size(); i++) {
        if (neighbors_out[i].second != expected_order[i]) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "✅ PASS: find_knn_exact returned correct nearest neighbors" << std::endl;
    } else {
        std::cout << "❌ FAIL: find_knn_exact returned incorrect nearest neighbors" << std::endl;
    }
}

void test_calculate_recall() {
    std::cout << "\n=== Testing calculate_recall Helper Function ===" << std::endl;

    // Create exact neighbors (sorted by distance)
    std::vector<std::pair<float, int>> exact_neighbors = {
        {0.5f, 1}, {0.8f, 3}, {2.0f, 2}, {5.0f, 4}
    };

    // Create HNSW neighbors (some overlap with exact)
    int hnsw_neighbors[4] = {3, 1, 5, 6};  // Points 3 and 1 overlap with exact

    float recall = distance_metrics::calculate_recall(exact_neighbors, hnsw_neighbors, 4);

    std::cout << "Exact neighbors: [1, 3, 2, 4]" << std::endl;
    std::cout << "HNSW neighbors: [3, 1, 5, 6]" << std::endl;
    std::cout << "Recall: " << recall << " (expected: 0.5)" << std::endl;

    if (std::abs(recall - 0.5f) < 0.001f) {
        std::cout << "✅ PASS: calculate_recall computed correct recall value" << std::endl;
    } else {
        std::cout << "❌ FAIL: calculate_recall computed incorrect recall value" << std::endl;
    }
}

void test_build_distance_matrix() {
    std::cout << "\n=== Testing build_distance_matrix Helper Function ===" << std::endl;

    // Create a simple test dataset
    const int n_obs = 3;
    const int n_dim = 2;
    float data[n_obs * n_dim] = {
        0.0f, 0.0f,  // Point 0
        3.0f, 4.0f,  // Point 1 (distance 5 from point 0)
        6.0f, 8.0f   // Point 2 (distance 10 from point 0)
    };

    float distance_matrix[n_obs * n_obs];

    // Build distance matrix
    distance_metrics::build_distance_matrix(data, n_obs, n_dim, UWOT_METRIC_EUCLIDEAN, distance_matrix);

    std::cout << "Distance matrix:" << std::endl;
    for (int i = 0; i < n_obs; i++) {
        for (int j = 0; j < n_obs; j++) {
            std::cout << "  " << distance_matrix[i * n_obs + j];
        }
        std::cout << std::endl;
    }

    // Verify key distances
    bool test_passed = true;

    // Diagonal should be zero
    if (std::abs(distance_matrix[0 * n_obs + 0]) > 0.001f ||
        std::abs(distance_matrix[1 * n_obs + 1]) > 0.001f ||
        std::abs(distance_matrix[2 * n_obs + 2]) > 0.001f) {
        test_passed = false;
    }

    // Distance from 0 to 1 should be 5
    if (std::abs(distance_matrix[0 * n_obs + 1] - 5.0f) > 0.001f) {
        test_passed = false;
    }

    // Distance from 0 to 2 should be 10
    if (std::abs(distance_matrix[0 * n_obs + 2] - 10.0f) > 0.001f) {
        test_passed = false;
    }

    if (test_passed) {
        std::cout << "✅ PASS: build_distance_matrix computed correct distances" << std::endl;
    } else {
        std::cout << "❌ FAIL: build_distance_matrix computed incorrect distances" << std::endl;
    }
}

int main() {
    std::cout << "🧪 Testing Distance Conversion Helper Functions" << std::endl;

    test_find_knn_exact();
    test_calculate_recall();
    test_build_distance_matrix();

    std::cout << "\n🎉 Helper function tests completed!" << std::endl;
    std::cout << "\n📋 Summary of Improvements:" << std::endl;
    std::cout << "✅ Extracted find_knn_exact: Eliminates duplicate k-NN search code" << std::endl;
    std::cout << "✅ Extracted calculate_recall: Eliminates duplicate recall computation" << std::endl;
    std::cout << "✅ Extracted build_distance_matrix: Eliminates duplicate matrix building" << std::endl;
    std::cout << "✅ Improved code maintainability and reduced duplication" << std::endl;

    return 0;
}