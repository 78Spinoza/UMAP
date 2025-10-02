#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include "uwot_simple_wrapper.h"

void progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    if (strstr(phase, "WARNING") != nullptr) {
        std::cout << "⚠️  " << phase << ": " << (message ? message : "") << std::endl;
    } else {
        std::cout << "[" << phase << "] " << current << "/" << total << " (" << percent << "%)";
        if (message) std::cout << " - " << message;
        std::cout << std::endl;
    }
}

void test_cosine_with_zero_vectors() {
    std::cout << "\n=== Testing Cosine Metric with Zero Vectors ===" << std::endl;

    // Create dataset with zero vectors
    const int n_obs = 50;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);

    // Fill with normal data first
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Insert some zero vectors (every 10th vector)
    for (int i = 0; i < n_obs; i += 10) {
        for (int j = 0; j < n_dim; j++) {
            data[i * n_dim + j] = 0.0f;
        }
    }

    std::cout << "🔍 Created dataset with " << (n_obs / 10) << " zero vectors for cosine metric testing" << std::endl;

    // Create model and test with cosine metric
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return;
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    std::cout << "🚀 Training with cosine metric (should show zero-norm warnings)..." << std::endl;
    int result = uwot_fit_with_progress_v2(
        model,
        data.data(),
        n_obs,
        n_dim,
        embedding_dim,
        15,  // n_neighbors
        0.1f,  // min_dist
        1.0f,  // spread
        20,  // n_epochs (reduced for quick test)
        UWOT_METRIC_COSINE,
        embedding.data(),
        progress_callback,
        0,  // force_exact_knn
        -1,  // M
        -1,  // ef_construction
        -1,  // ef_search
        0,  // use_quantization
        42,  // random_seed
        1   // autoHNSWParam
    );

    if (result == 0) {
        std::cout << "✅ Cosine training completed successfully despite zero vectors" << std::endl;
    } else {
        std::cout << "❌ Cosine training failed: " << result << " - " << uwot_get_error_message(result) << std::endl;
    }

    uwot_destroy(model);
}

void test_correlation_with_constant_vectors() {
    std::cout << "\n=== Testing Correlation Metric with Constant Vectors ===" << std::endl;

    // Create dataset with constant vectors
    const int n_obs = 40;
    const int n_dim = 8;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);

    // Fill with normal data first
    std::mt19937 gen(123);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Insert some constant vectors (every 8th vector)
    for (int i = 0; i < n_obs; i += 8) {
        float constant_value = 5.0f + (i / 8) * 2.0f;  // Different constant values
        for (int j = 0; j < n_dim; j++) {
            data[i * n_dim + j] = constant_value;
        }
    }

    std::cout << "🔍 Created dataset with " << (n_obs / 8) << " constant vectors for correlation metric testing" << std::endl;

    // Create model and test with correlation metric
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return;
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    std::cout << "🚀 Training with correlation metric (should show zero-variance warnings)..." << std::endl;
    int result = uwot_fit_with_progress_v2(
        model,
        data.data(),
        n_obs,
        n_dim,
        embedding_dim,
        15,  // n_neighbors
        0.1f,  // min_dist
        1.0f,  // spread
        20,  // n_epochs (reduced for quick test)
        UWOT_METRIC_CORRELATION,
        embedding.data(),
        progress_callback,
        0,  // force_exact_knn
        -1,  // M
        -1,  // ef_construction
        -1,  // ef_search
        0,  // use_quantization
        123,  // random_seed
        1   // autoHNSWParam
    );

    if (result == 0) {
        std::cout << "✅ Correlation training completed successfully despite constant vectors" << std::endl;
    } else {
        std::cout << "❌ Correlation training failed: " << result << " - " << uwot_get_error_message(result) << std::endl;
    }

    uwot_destroy(model);
}

void test_normal_data_control() {
    std::cout << "\n=== Control Test: Normal Data (No Warnings Expected) ===" << std::endl;

    // Create normal dataset without zero/constant vectors
    const int n_obs = 30;
    const int n_dim = 6;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);

    std::mt19937 gen(456);
    std::normal_distribution<float> dis(1.0f, 2.0f);  // Non-zero mean to avoid accidental zeros
    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
        // Ensure no exact zeros
        if (std::abs(data[i]) < 0.1f) {
            data[i] = data[i] < 0 ? -0.1f : 0.1f;
        }
    }

    std::cout << "🔍 Created normal dataset without zero or constant vectors" << std::endl;

    // Create model and test with cosine metric
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return;
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    std::cout << "🚀 Training with cosine metric (no warnings expected)..." << std::endl;
    int result = uwot_fit_with_progress_v2(
        model,
        data.data(),
        n_obs,
        n_dim,
        embedding_dim,
        15,  // n_neighbors
        0.1f,  // min_dist
        1.0f,  // spread
        20,  // n_epochs (reduced for quick test)
        UWOT_METRIC_COSINE,
        embedding.data(),
        progress_callback,
        0,  // force_exact_knn
        -1,  // M
        -1,  // ef_construction
        -1,  // ef_search
        0,  // use_quantization
        456,  // random_seed
        1   // autoHNSWParam
    );

    if (result == 0) {
        std::cout << "✅ Normal data training completed successfully" << std::endl;
    } else {
        std::cout << "❌ Normal data training failed: " << result << " - " << uwot_get_error_message(result) << std::endl;
    }

    uwot_destroy(model);
}

int main() {
    std::cout << "🧪 Testing Zero-Norm Handling Improvements via UMAP API" << std::endl;

    test_cosine_with_zero_vectors();
    test_correlation_with_constant_vectors();
    test_normal_data_control();

    std::cout << "\n🎉 Zero-norm handling tests completed!" << std::endl;
    std::cout << "\n📋 Summary:" << std::endl;
    std::cout << "✅ Zero-norm vectors should now be handled gracefully" << std::endl;
    std::cout << "✅ Zero-variance vectors should produce warnings but not crash" << std::endl;
    std::cout << "✅ Normal data should work without warnings" << std::endl;

    return 0;
}