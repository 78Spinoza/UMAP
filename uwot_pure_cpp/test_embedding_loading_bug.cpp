#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>

void test_embedding_loading_bug() {
    std::cout << "=== EMBEDDING LOADING BUG DIAGNOSTICS ===" << std::endl;

    // Create test data
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }

    // Create and fit model
    UwotModel* model = uwot_create();
    std::vector<float> embedding(n_obs * embedding_dim);

    std::cout << "1. Training model..." << std::endl;
    int result = uwot_fit_with_progress_v2(model, data.data(), n_obs, n_dim, embedding_dim,
        15, 0.1f, 1.0f, 200, UWOT_METRIC_EUCLIDEAN, embedding.data(), nullptr,
        0, -1, -1, -1, 1, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << uwot_get_error_message(result) << std::endl;
        return;
    }
    std::cout << "✅ Training successful" << std::endl;

    // Test transform with original model
    std::vector<float> test_point(1 * n_dim);
    for (int i = 0; i < n_dim; i++) {
        test_point[i] = dis(gen);
    }

    std::vector<float> original_transform(1 * embedding_dim);
    std::cout << "2. Testing transform with original model..." << std::endl;
    result = uwot_transform(model, test_point.data(), 1, n_dim, original_transform.data());

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Original transform failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return;
    }

    std::cout << "✅ Original transform result: ["
              << std::fixed << std::setprecision(6)
              << original_transform[0] << ", " << original_transform[1] << "]" << std::endl;

    // Check if embedding array has data
    bool embedding_has_data = false;
    for (size_t i = 0; i < embedding.size(); i++) {
        if (std::abs(embedding[i]) > 1e-6f) {
            embedding_has_data = true;
            break;
        }
    }
    std::cout << "3. Original model embedding array has data: " << (embedding_has_data ? "YES" : "NO") << std::endl;

    // Save model
    const char* filename = "test_embedding_bug.umap";
    std::cout << "4. Saving model to " << filename << "..." << std::endl;
    result = uwot_save_model(model, filename);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return;
    }
    std::cout << "✅ Save successful" << std::endl;

    // Check file size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size_t file_size = file.tellg();
        file.close();
        std::cout << "   Model file size: " << file_size << " bytes" << std::endl;
    }

    uwot_destroy(model);

    // Load model
    std::cout << "5. Loading model from " << filename << "..." << std::endl;
    UwotModel* loaded_model = uwot_load_model(filename);

    if (!loaded_model) {
        std::cout << "❌ Load failed: nullptr returned" << std::endl;
        return;
    }
    std::cout << "✅ Load successful" << std::endl;

    // Get loaded model info
    int loaded_n_vertices, loaded_n_dim, loaded_embedding_dim;
    uwot_get_model_info(loaded_model, &loaded_n_vertices, &loaded_n_dim, &loaded_embedding_dim,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    std::cout << "   Loaded model: " << loaded_n_vertices << " vertices, "
              << loaded_n_dim << " dims, " << loaded_embedding_dim << " embedding dims" << std::endl;

    // Test transform with loaded model
    std::vector<float> loaded_transform(1 * embedding_dim);
    std::cout << "6. Testing transform with loaded model..." << std::endl;
    result = uwot_transform(loaded_model, test_point.data(), 1, n_dim, loaded_transform.data());

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Loaded transform failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(loaded_model);
        return;
    }

    std::cout << "✅ Loaded transform result: ["
              << std::fixed << std::setprecision(6)
              << loaded_transform[0] << ", " << loaded_transform[1] << "]" << std::endl;

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        float diff = std::abs(original_transform[i] - loaded_transform[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "7. COMPARISON:" << std::endl;
    std::cout << "   Max difference: " << std::scientific << max_diff << std::fixed << std::endl;

    if (max_diff < 0.001f) {
        std::cout << "✅ TRANSFORM CONSISTENCY: PASSED" << std::endl;
    } else {
        std::cout << "❌ TRANSFORM CONSISTENCY: FAILED" << std::endl;
        std::cout << "   Original: [" << original_transform[0] << ", " << original_transform[1] << "]" << std::endl;
        std::cout << "   Loaded:   [" << loaded_transform[0] << ", " << loaded_transform[1] << "]" << std::endl;

        // Check if loaded model produces zero embeddings
        bool both_zero = (std::abs(loaded_transform[0]) < 1e-6f && std::abs(loaded_transform[1]) < 1e-6f);
        if (both_zero) {
            std::cout << "🚨 CRITICAL BUG: Loaded model produces ZERO embeddings!" << std::endl;
        }
    }

    // Test HNSW index status
    std::cout << "8. HNSW Index Status:" << std::endl;
    bool is_fitted = uwot_is_fitted(loaded_model) != 0;
    std::cout << "   Is fitted: " << (is_fitted ? "YES" : "NO") << std::endl;

    uwot_destroy(loaded_model);

    // Cleanup
    std::remove(filename);

    std::cout << "=== DIAGNOSTIC COMPLETE ===" << std::endl;
}

int main() {
    test_embedding_loading_bug();
    return 0;
}