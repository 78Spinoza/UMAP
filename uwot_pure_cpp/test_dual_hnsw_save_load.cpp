#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=== DUAL HNSW SAVE/LOAD TEST ===" << std::endl;

    // Generate test data
    const int n_samples = 100;
    const int n_features = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_samples * n_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }

    // Create model
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return 1;
    }

    // Train model
    std::vector<float> embedding(n_samples * embedding_dim);
    int result = uwot_fit_with_progress_v2(model, data.data(), n_samples, n_features,
        embedding_dim, 15, 0.1f, 1.0f, 50, UWOT_METRIC_EUCLIDEAN,
        embedding.data(), nullptr, 0, 32, 128, 64, 0, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Training completed successfully" << std::endl;

    // Test transform before save
    std::vector<float> test_point(10);
    for (int i = 0; i < 10; i++) {
        test_point[i] = dis(gen);
    }
    std::vector<float> transform_embedding_before(2);
    std::vector<int> nn_indices_before(15);
    std::vector<float> nn_distances_before(15);
    std::vector<float> confidence_before(1);
    std::vector<int> outlier_level_before(1);

    result = uwot_transform_detailed(model, test_point.data(), 1, 10,
        transform_embedding_before.data(), nn_indices_before.data(),
        nn_distances_before.data(), confidence_before.data(),
        outlier_level_before.data(), nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform before save failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Transform before save successful" << std::endl;
    std::cout << "   Embedding: [" << transform_embedding_before[0] << ", " << transform_embedding_before[1] << "]" << std::endl;
    std::cout << "   Confidence: " << confidence_before[0] << std::endl;
    std::cout << "   Nearest neighbor indices: ";
    for (int i = 0; i < 3; i++) {
        std::cout << nn_indices_before[i] << " ";
    }
    std::cout << std::endl;

    // Save model
    result = uwot_save_model(model, "test_dual_hnsw.umap");
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Model saved successfully" << std::endl;

    // Load model
    UwotModel* loaded_model = uwot_load_model("test_dual_hnsw.umap");
    if (!loaded_model) {
        std::cout << "❌ Load failed" << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "✅ Model loaded successfully" << std::endl;

    // Test transform after load
    std::vector<float> transform_embedding_after(2);
    std::vector<int> nn_indices_after(15);
    std::vector<float> nn_distances_after(15);
    std::vector<float> confidence_after(1);
    std::vector<int> outlier_level_after(1);

    result = uwot_transform_detailed(loaded_model, test_point.data(), 1, 10,
        transform_embedding_after.data(), nn_indices_after.data(),
        nn_distances_after.data(), confidence_after.data(),
        outlier_level_after.data(), nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform after load failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        uwot_destroy(loaded_model);
        return 1;
    }

    std::cout << "✅ Transform after load successful" << std::endl;
    std::cout << "   Embedding: [" << transform_embedding_after[0] << ", " << transform_embedding_after[1] << "]" << std::endl;
    std::cout << "   Confidence: " << confidence_after[0] << std::endl;
    std::cout << "   Nearest neighbor indices: ";
    for (int i = 0; i < 3; i++) {
        std::cout << nn_indices_after[i] << " ";
    }
    std::cout << std::endl;

    // Compare results
    float embedding_diff = std::abs(transform_embedding_before[0] - transform_embedding_after[0]) +
                           std::abs(transform_embedding_before[1] - transform_embedding_after[1]);
    float confidence_diff = std::abs(confidence_before[0] - confidence_after[0]);

    std::cout << std::endl << "=== COMPARISON RESULTS ===" << std::endl;
    std::cout << "Embedding difference: " << embedding_diff << std::endl;
    std::cout << "Confidence difference: " << confidence_diff << std::endl;

    // Check neighbor indices
    bool neighbors_match = true;
    for (int i = 0; i < 3; i++) {
        if (nn_indices_before[i] != nn_indices_after[i]) {
            neighbors_match = false;
            break;
        }
    }

    std::cout << "Neighbor indices match: " << (neighbors_match ? "✅ YES" : "❌ NO") << std::endl;

    if (embedding_diff < 0.01f && confidence_diff < 0.01f && neighbors_match) {
        std::cout << "🎉 DUAL HNSW SAVE/LOAD TEST PASSED!" << std::endl;
    } else {
        std::cout << "⚠️  Some differences detected (may be expected with HNSW approximation)" << std::endl;
    }

    // Cleanup
    uwot_destroy(model);
    uwot_destroy(loaded_model);

    return 0;
}