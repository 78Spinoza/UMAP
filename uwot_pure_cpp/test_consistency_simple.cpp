#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=== Testing Training vs Transform Consistency ===" << std::endl;

    // Create test data
    const int n_obs = 100;
    const int n_dim = 5;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Create model
    UwotModel* model = uwot_create();

    // Train without quantization
    std::vector<float> training_embedding(n_obs * embedding_dim);
    int result = uwot_fit_with_progress_v2(
        model, data.data(), n_obs, n_dim, embedding_dim, 15,
        0.1f, 1.0f, 50, UWOT_METRIC_EUCLIDEAN, training_embedding.data(),
        nullptr, // No progress callback
        0, // Don't force exact KNN
        -1, -1, -1, // Auto HNSW params
        0, // No quantization
        42, // Fixed seed
        1   // Auto HNSW
    );

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "✅ Training completed" << std::endl;
    std::cout << "First 3 training embeddings:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  Point " << i << ": [";
        for (int d = 0; d < embedding_dim; d++) {
            std::cout << std::fixed << std::setprecision(6)
                     << training_embedding[i * embedding_dim + d];
            if (d < embedding_dim - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Transform the same training data
    std::vector<float> transform_embedding(n_obs * embedding_dim);
    result = uwot_transform(model, data.data(), n_obs, n_dim, transform_embedding.data());

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << uwot_get_error_message(result) << std::endl;
        return 1;
    }

    std::cout << "\n✅ Transform completed" << std::endl;
    std::cout << "First 3 transform embeddings:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  Point " << i << ": [";
        for (int d = 0; d < embedding_dim; d++) {
            std::cout << std::fixed << std::setprecision(6)
                     << transform_embedding[i * embedding_dim + d];
            if (d < embedding_dim - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Compare results
    std::cout << "\n🔍 COMPARISON:" << std::endl;
    double max_diff = 0.0;
    int mismatch_count = 0;
    const double tolerance = 1e-6;

    for (int i = 0; i < n_obs; i++) {
        for (int d = 0; d < embedding_dim; d++) {
            double train_val = training_embedding[i * embedding_dim + d];
            double trans_val = transform_embedding[i * embedding_dim + d];
            double diff = std::abs(train_val - trans_val);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                mismatch_count++;
                if (mismatch_count <= 5) { // Show first 5 mismatches
                    std::cout << "❌ Mismatch at point " << i << ", dim " << d
                             << ": Train=" << train_val << ", Trans=" << trans_val
                             << ", Diff=" << diff << std::endl;
                }
            }
        }
    }

    std::cout << "\n📊 RESULTS:" << std::endl;
    std::cout << "  Max difference: " << std::scientific << max_diff << std::endl;
    std::cout << "  Mismatch count: " << mismatch_count << "/" << n_obs * embedding_dim << std::endl;

    if (max_diff < tolerance) {
        std::cout << "✅ CONSISTENCY CHECK PASSED" << std::endl;
    } else {
        std::cout << "❌ CONSISTENCY CHECK FAILED" << std::endl;
        std::cout << "Training and Transform produce different results for the same data!" << std::endl;
    }

    uwot_destroy(model);
    return 0;
}