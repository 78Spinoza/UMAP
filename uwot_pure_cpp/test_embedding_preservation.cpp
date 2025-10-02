#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "uwot_simple_wrapper.h"

void progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    if (percent == 100.0f) {
        printf("%s: %s\n", phase, message ? message : "");
    }
}

int main() {
    std::cout << "=== TESTING EMBEDDING DATA PRESERVATION OPTION ===" << std::endl;

    // Create test dataset
    const int N_SAMPLES = 100;
    const int N_DIM = 10;
    const int EMBEDDING_DIM = 2;

    std::vector<float> data(N_SAMPLES * N_DIM);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N_SAMPLES * N_DIM; i++) {
        data[i] = dist(gen);
    }

    std::cout << "Created test dataset: " << N_SAMPLES << " x " << N_DIM << std::endl;

    // Test 1: Traditional approach (save both HNSW indices)
    std::cout << "\n--- Test 1: Traditional Approach (save both HNSW indices) ---" << std::endl;

    UwotModel* model1 = uwot_create();
    std::vector<float> fit_embedding1(N_SAMPLES * EMBEDDING_DIM);

    int result = uwot_fit_with_progress_v2(model1, data.data(), N_SAMPLES, N_DIM, EMBEDDING_DIM,
        15, 0.1f, 1.0f, 100, UWOT_METRIC_EUCLIDEAN, fit_embedding1.data(),
        progress_callback, 1, -1, -1, -1, 0, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << result << std::endl;
        uwot_destroy(model1);
        return 1;
    }

    std::cout << "✅ Training completed" << std::endl;

    // Save with traditional approach
    result = uwot_save_model(model1, "test_traditional.umap");
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << result << std::endl;
        uwot_destroy(model1);
        return 1;
    }
    std::cout << "✅ Saved with traditional approach" << std::endl;
    uwot_destroy(model1);

    // Test 2: New approach (always save embedding data, rebuild HNSW)
    std::cout << "\n--- Test 2: New Approach (always save embedding data) ---" << std::endl;

    UwotModel* model2 = uwot_create();
    std::vector<float> fit_embedding2(N_SAMPLES * EMBEDDING_DIM);

    // Enable the new option
    uwot_set_always_save_embedding_data(model2, true);
    std::cout << "✅ Enabled always_save_embedding_data option" << std::endl;

    result = uwot_fit_with_progress_v2(model2, data.data(), N_SAMPLES, N_DIM, EMBEDDING_DIM,
        15, 0.1f, 1.0f, 100, UWOT_METRIC_EUCLIDEAN, fit_embedding2.data(),
        progress_callback, 1, -1, -1, -1, 0, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << result << std::endl;
        uwot_destroy(model2);
        return 1;
    }

    std::cout << "✅ Training completed" << std::endl;

    // Save with new approach
    result = uwot_save_model(model2, "test_preserve.umap");
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << result << std::endl;
        uwot_destroy(model2);
        return 1;
    }
    std::cout << "✅ Saved with embedding data preservation" << std::endl;
    uwot_destroy(model2);

    // Test 3: Load both models and compare
    std::cout << "\n--- Test 3: Loading and Comparing ---" << std::endl;

    // Load traditional model
    UwotModel* loaded1 = uwot_load_model("test_traditional.umap");
    if (!loaded1) {
        std::cout << "❌ Failed to load traditional model" << std::endl;
        return 1;
    }
    std::cout << "✅ Loaded traditional model" << std::endl;

    // Load preserved model
    UwotModel* loaded2 = uwot_load_model("test_preserve.umap");
    if (!loaded2) {
        std::cout << "❌ Failed to load preserved model" << std::endl;
        return 1;
    }
    std::cout << "✅ Loaded preserved model" << std::endl;

    // Check if both have the embedding HNSW index via model info
    uint32_t traditional_embedding_crc = 0, preserved_embedding_crc = 0;
    uint32_t traditional_original_crc = 0, preserved_original_crc = 0;
    uint32_t version_crc1 = 0, version_crc2 = 0;
    float hnsw_recall1 = 0.0f, hnsw_recall2 = 0.0f;

    uwot_get_model_info_v2(loaded1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, &traditional_original_crc, &traditional_embedding_crc,
        &version_crc1, &hnsw_recall1);

    uwot_get_model_info_v2(loaded2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, &preserved_original_crc, &preserved_embedding_crc,
        &version_crc2, &hnsw_recall2);

    bool traditional_has_embedding = (traditional_embedding_crc != 0);
    bool preserved_has_embedding = (preserved_embedding_crc != 0);

    std::cout << "Traditional model has embedding HNSW: " << (traditional_has_embedding ? "YES" : "NO") << std::endl;
    std::cout << "Preserved model has embedding HNSW: " << (preserved_has_embedding ? "YES" : "NO") << std::endl;

    // Transform a test point with both models
    std::vector<float> test_point(N_DIM);
    for (int i = 0; i < N_DIM; i++) {
        test_point[i] = dist(gen);
    }

    std::vector<float> transform1(EMBEDDING_DIM);
    std::vector<float> transform2(EMBEDDING_DIM);

    result = uwot_transform_detailed(loaded1, test_point.data(), 1, N_DIM, transform1.data(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform with traditional model failed: " << result << std::endl;
    } else {
        std::cout << "✅ Transform with traditional model succeeded" << std::endl;
    }

    result = uwot_transform_detailed(loaded2, test_point.data(), 1, N_DIM, transform2.data(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform with preserved model failed: " << result << std::endl;
    } else {
        std::cout << "✅ Transform with preserved model succeeded" << std::endl;
    }

    // Compare transform results
    double max_diff = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        double diff = std::abs(transform1[i] - transform2[i]);
        max_diff = std::max(max_diff, diff);
    }
    std::cout << "Maximum difference between approaches: " << max_diff << std::endl;

    if (max_diff < 1e-4) {
        std::cout << "✅ EXCELLENT: Results nearly identical" << std::endl;
    } else if (max_diff < 1e-2) {
        std::cout << "✅ GOOD: Results very similar" << std::endl;
    } else if (max_diff < 0.1) {
        std::cout << "⚠️  ACCEPTABLE: Results similar but some differences" << std::endl;
    } else {
        std::cout << "❌ CONCERNING: Large differences between approaches" << std::endl;
    }

    std::cout << "\n=== EMBEDDING PRESERVATION TEST COMPLETED ===" << std::endl;

    uwot_destroy(loaded1);
    uwot_destroy(loaded2);
    return 0;
}