#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

void progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    if (percent == 100.0f) {
        printf("%s: %s\n", phase, message ? message : "");
    }
}

int main() {
    std::cout << "=== QUICK EUCLIDEAN TEST ===" << std::endl;

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

    // Test with EXACT k-NN
    UwotModel* model = uwot_create();
    std::vector<float> fit_embedding(N_SAMPLES * EMBEDDING_DIM);
    std::vector<float> transform_embedding(N_SAMPLES * EMBEDDING_DIM);

    int result = uwot_fit_with_progress_v2(model, data.data(), N_SAMPLES, N_DIM, EMBEDDING_DIM,
        15, 0.1f, 1.0f, 100, UWOT_METRIC_EUCLIDEAN, fit_embedding.data(),
        progress_callback, 1, -1, -1, -1, 0, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << result << std::endl;
        uwot_destroy(model);
        return 1;
    }
    std::cout << "✅ Training completed" << std::endl;

    // Transform same data with EXACT k-NN
    result = uwot_transform_detailed(model, data.data(), N_SAMPLES, N_DIM, transform_embedding.data(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << result << std::endl;
        uwot_destroy(model);
        return 1;
    }
    std::cout << "✅ Transform completed" << std::endl;

    // Compare results
    double max_diff = 0.0;
    double total_diff = 0.0;
    int significant_diffs = 0;

    for (int i = 0; i < N_SAMPLES * EMBEDDING_DIM; i++) {
        double diff = std::abs(fit_embedding[i] - transform_embedding[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        if (diff > 0.02) { // 2% threshold
            significant_diffs++;
        }
    }

    double mse = total_diff * total_diff / (N_SAMPLES * EMBEDDING_DIM);
    double error_rate = (double)significant_diffs / (N_SAMPLES * EMBEDDING_DIM) * 100.0;

    std::cout << "Results:" << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;
    std::cout << "  Significant differences (>2%): " << significant_diffs << "/" << (N_SAMPLES * EMBEDDING_DIM) << " (" << error_rate << "%)" << std::endl;

    if (error_rate > 10.0) {
        std::cout << "❌ HIGH ERROR RATE" << std::endl;
    } else {
        std::cout << "✅ ACCEPTABLE ERROR RATE" << std::endl;
    }

    // Show sample differences
    std::cout << "\nSample comparisons (first 5 points):" << std::endl;
    for (int i = 0; i < 5 && i < N_SAMPLES; i++) {
        std::cout << "  Point " << i << ":" << std::endl;
        std::cout << "    Fit:      [" << fit_embedding[i*2] << ", " << fit_embedding[i*2+1] << "]" << std::endl;
        std::cout << "    Transform:[" << transform_embedding[i*2] << ", " << transform_embedding[i*2+1] << "]" << std::endl;
        std::cout << "    Diff:     " << std::abs(fit_embedding[i*2] - transform_embedding[i*2]) << std::endl;
    }

    uwot_destroy(model);
    return 0;
}