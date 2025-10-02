#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    std::cout << "=== Debug Quantization Save/Load Issue ===" << std::endl;

    // Create simple test data
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Create and train model WITH quantization
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return 1;
    }

    std::vector<float> embedding(n_obs * embedding_dim);
    int result = uwot_fit_with_progress_v2(
        model, data.data(), n_obs, n_dim, embedding_dim, 15,
        0.1f, 1.0f, 50, UWOT_METRIC_EUCLIDEAN, embedding.data(),
        nullptr, 0, -1, -1, -1, 1, 42, 1 // use_quantization = 1
    );

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Training completed with quantization" << std::endl;

    // Save model
    const char* filename = "debug_quantization_save_load.umap";
    result = uwot_save_model(model, filename);
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Quantized model saved" << std::endl;
    uwot_destroy(model);

    // Load model
    model = uwot_load_model(filename);
    if (!model) {
        std::cout << "❌ Load failed" << std::endl;
        return 1;
    }

    std::cout << "✅ Quantized model loaded successfully" << std::endl;

    // Try transform
    std::vector<float> transform_embedding(n_obs * embedding_dim);
    result = uwot_transform(model, data.data(), n_obs, n_dim, transform_embedding.data());
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Transform completed successfully" << std::endl;

    uwot_destroy(model);
    std::remove(filename);

    std::cout << "✅ Quantization save/load test PASSED" << std::endl;
    return 0;
}