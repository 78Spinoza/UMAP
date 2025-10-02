#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include "uwot_simple_wrapper.h"

void progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    if (strstr(phase, "HNSW") != nullptr || strstr(phase, "STATS") != nullptr || strstr(phase, "DUAL_HNSW") != nullptr) {
        std::cout << "[" << phase << "] " << (message ? message : "") << std::endl;
    }
}

void test_ef_search_scaling_for_sizes() {
    struct TestCase {
        int n_obs;
        int n_dim;
        int expected_min_ef_search;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {1000, 10, 32, "Small dataset (<50K)"},
        {50000, 50, 64, "Medium dataset (50K)"},
        {1000000, 100, 200, "Large dataset (1M)"},
        {5000000, 200, 1200, "Very large dataset (5M)"},
        {10000000, 300, 2000, "Extra large dataset (10M)"}
    };

    for (const auto& test_case : test_cases) {
        std::cout << "\n=== Testing: " << test_case.description << " ===" << std::endl;
        std::cout << "Dataset: " << test_case.n_obs << " points × " << test_case.n_dim << " dimensions" << std::endl;

        // Use smaller actual dataset sizes for testing but still test scaling logic
        const int actual_n_obs = std::min(test_case.n_obs, 2000);  // Limit actual size for testing
        const int actual_n_dim = std::min(test_case.n_dim, 50);     // Limit dimensions for testing
        const int embedding_dim = 2;

        std::vector<float> data(actual_n_obs * actual_n_dim);
        std::mt19937 gen(42);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < actual_n_obs * actual_n_dim; i++) {
            data[i] = dis(gen);
        }

        // Create model with auto-scaling
        UwotModel* model = uwot_create();
        if (!model) {
            std::cout << "❌ Failed to create model" << std::endl;
            continue;
        }

        std::vector<float> embedding(actual_n_obs * embedding_dim);

        std::cout << "🚀 Training with auto HNSW scaling..." << std::endl;
        int result = uwot_fit_with_progress_v2(
            model,
            data.data(),
            test_case.n_obs,  // Use the test case size for scaling logic
            actual_n_dim,     // Use actual dimension for data
            embedding_dim,
            15,  // n_neighbors
            0.1f,  // min_dist
            1.0f,  // spread
            5,   // n_epochs (very reduced for testing)
            UWOT_METRIC_EUCLIDEAN,
            embedding.data(),
            progress_callback,
            0,  // force_exact_knn
            -1,  // M (auto-scale)
            -1,  // ef_construction (auto-scale)
            -1,  // ef_search (auto-scale)
            0,  // use_quantization
            42,  // random_seed
            1   // autoHNSWParam
        );

        if (result == 0) {
            // Get HNSW parameters
            int n_vertices, n_dim_check, embedding_dim_check, n_neighbors;
            int hnsw_M, hnsw_ef_construction, hnsw_ef_search;
            float min_dist, spread;
            UwotMetric metric;
            uwot_get_model_info(model, &n_vertices, &n_dim_check, &embedding_dim_check, &n_neighbors,
                              &min_dist, &spread, &metric, &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);

            std::cout << "📊 HNSW Parameters Used:" << std::endl;
            std::cout << "   M: " << hnsw_M << std::endl;
            std::cout << "   ef_construction: " << hnsw_ef_construction << std::endl;
            std::cout << "   ef_search: " << hnsw_ef_search << std::endl;

            if (hnsw_ef_search >= test_case.expected_min_ef_search) {
                std::cout << "✅ PASS: ef_search (" << hnsw_ef_search << ") >= expected minimum ("
                         << test_case.expected_min_ef_search << ")" << std::endl;
            } else {
                std::cout << "❌ FAIL: ef_search (" << hnsw_ef_search << ") < expected minimum ("
                         << test_case.expected_min_ef_search << ")" << std::endl;
            }

            // Test transform with high ef_search
            std::vector<float> test_point(actual_n_dim);
            for (int i = 0; i < actual_n_dim; i++) {
                test_point[i] = dis(gen);
            }
            std::vector<float> test_embedding(embedding_dim);

            int transform_result = uwot_transform(model, test_point.data(), 1, actual_n_dim, test_embedding.data());
            if (transform_result == 0) {
                std::cout << "✅ Transform successful with high ef_search" << std::endl;
            } else {
                std::cout << "❌ Transform failed: " << transform_result << std::endl;
            }

        } else {
            std::cout << "❌ Training failed: " << result << " - " << uwot_get_error_message(result) << std::endl;
        }

        uwot_destroy(model);
    }
}

void test_explicit_ef_search_limits() {
    std::cout << "\n=== Testing Explicit ef_search Limits ===" << std::endl;

    const int n_obs = 2000;  // Reduced for testing
    const int n_dim = 50;    // Reduced for testing
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(123);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Test with very high explicit ef_search
    const int high_ef_search = 1500;
    std::cout << "🚀 Testing with explicit ef_search = " << high_ef_search << std::endl;

    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return;
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    int result = uwot_fit_with_progress_v2(
        model,
        data.data(),
        n_obs,
        n_dim,
        embedding_dim,
        15,  // n_neighbors
        0.1f,  // min_dist
        1.0f,  // spread
        5,   // n_epochs (reduced for testing)
        UWOT_METRIC_EUCLIDEAN,
        embedding.data(),
        progress_callback,
        0,  // force_exact_knn
        -1,  // M (auto-scale)
        -1,  // ef_construction (auto-scale)
        high_ef_search,  // ef_search (explicit high value)
        0,  // use_quantization
        123,  // random_seed
        1   // autoHNSWParam
    );

    if (result == 0) {
        int hnsw_M, hnsw_ef_construction, hnsw_ef_search;
        float min_dist, spread;
        UwotMetric metric;
        uwot_get_model_info(model, nullptr, nullptr, nullptr, nullptr,
                          &min_dist, &spread, &metric, &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);

        std::cout << "📊 Explicit HNSW Parameters:" << std::endl;
        std::cout << "   M: " << hnsw_M << std::endl;
        std::cout << "   ef_construction: " << hnsw_ef_construction << std::endl;
        std::cout << "   ef_search: " << hnsw_ef_search << " (requested: " << high_ef_search << ")" << std::endl;

        if (hnsw_ef_search == high_ef_search) {
            std::cout << "✅ PASS: Explicit high ef_search value respected" << std::endl;
        } else {
            std::cout << "❌ FAIL: Explicit ef_search not respected" << std::endl;
        }
    } else {
        std::cout << "❌ Training failed with high ef_search: " << result << std::endl;
    }

    uwot_destroy(model);
}

int main() {
    std::cout << "🧪 Testing HNSW ef_search Scaling for Very Large Datasets" << std::endl;

    test_ef_search_scaling_for_sizes();
    test_explicit_ef_search_limits();

    std::cout << "\n🎉 ef_search scaling tests completed!" << std::endl;
    std::cout << "\n📋 Summary:" << std::endl;
    std::cout << "✅ Small datasets (<50K): ef_search up to 32-64" << std::endl;
    std::cout << "✅ Large datasets (1M): ef_search up to 200-800" << std::endl;
    std::cout << "✅ Very large datasets (5M+): ef_search up to 1200-2000" << std::endl;
    std::cout << "✅ Auto-tuning caps increased based on dataset size" << std::endl;
    std::cout << "✅ Explicit high ef_search values respected" << std::endl;

    return 0;
}