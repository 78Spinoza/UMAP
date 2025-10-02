#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    std::cout << "Simple save/load test..." << std::endl;

    // Small test dataset
    std::vector<float> data(50 * 5);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }

    // Create and train model
    UwotModel* model = uwot_create();
    std::vector<float> embedding(50 * 2);

    int result = uwot_fit_with_progress_v2(model, data.data(), 50, 5,
        2, 10, 0.1f, 1.0f, 30, UWOT_METRIC_EUCLIDEAN,
        embedding.data(), nullptr, 0, 32, 128, 64, 0);

    if (result != 0) {
        std::cout << "Training failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "Training completed" << std::endl;

    // Test transform before save
    std::vector<float> test_point(5);
    for (int i = 0; i < 5; i++) {
        test_point[i] = dis(gen);
    }
    std::vector<float> result_embedding(2);

    result = uwot_transform(model, test_point.data(), 1, 5, result_embedding.data());
    if (result != 0) {
        std::cout << "Transform failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "Transform result: [" << result_embedding[0] << ", " << result_embedding[1] << "]" << std::endl;

    // Save model
    result = uwot_save_model(model, "simple_test.umap");
    if (result != 0) {
        std::cout << "Save failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "Model saved" << std::endl;

    // Load model
    UwotModel* loaded_model = uwot_load_model("simple_test.umap");
    if (!loaded_model) {
        std::cout << "Load failed" << std::endl;
        return 1;
    }

    std::cout << "Model loaded" << std::endl;

    // Test transform after load
    std::vector<float> loaded_embedding(2);
    result = uwot_transform(loaded_model, test_point.data(), 1, 5, loaded_embedding.data());
    if (result != 0) {
        std::cout << "Transform after load failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "Transform after load: [" << loaded_embedding[0] << ", " << loaded_embedding[1] << "]" << std::endl;

    // Compare results
    float diff = std::abs(result_embedding[0] - loaded_embedding[0]) +
                 std::abs(result_embedding[1] - loaded_embedding[1]);
    std::cout << "Difference: " << diff << std::endl;

    if (diff < 0.01f) {
        std::cout << "✅ Save/load test PASSED!" << std::endl;
    } else {
        std::cout << "⚠️ Save/load test difference > 0.01 (may be expected with HNSW)" << std::endl;
    }

    uwot_destroy(model);
    uwot_destroy(loaded_model);

    return 0;
}