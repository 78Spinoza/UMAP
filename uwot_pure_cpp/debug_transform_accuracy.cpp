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
    std::cout << "=== DEBUGGING TRANSFORM ACCURACY ===" << std::endl;

    // Create a very simple test dataset
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

    // Train model with EXACT k-NN (no HNSW)
    std::cout << "Training with EXACT k-NN (force_exact_knn=1)..." << std::endl;

    UwotModel* model = uwot_create();
    std::vector<float> fit_embedding(N_SAMPLES * EMBEDDING_DIM);

    int result = uwot_fit_with_progress_v2(model, data.data(), N_SAMPLES, N_DIM, EMBEDDING_DIM,
        15, 0.1f, 1.0f, 100, UWOT_METRIC_EUCLIDEAN, fit_embedding.data(),
        progress_callback, 1, -1, -1, -1, 0, 42, 1);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << result << std::endl;
        return 1;
    }

    std::cout << "✅ Training completed" << std::endl;

    // Transform the SAME data (should get identical results with exact k-NN)
    std::cout << "Transforming same data with EXACT k-NN..." << std::endl;

    std::vector<float> transform_embedding(N_SAMPLES * EMBEDDING_DIM);

    result = uwot_transform_detailed(model, data.data(), N_SAMPLES, N_DIM,
        transform_embedding.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << result << std::endl;
        return 1;
    }

    std::cout << "✅ Transform completed" << std::endl;

    // Compare fit vs transform results - should be NEARLY IDENTICAL with exact k-NN
    std::cout << "Comparing fit vs transform results..." << std::endl;

    double total_diff = 0.0;
    double max_diff = 0.0;
    int significant_diff_count = 0;
    const double SIGNIFICANT_THRESHOLD = 1e-6;

    for (int i = 0; i < N_SAMPLES * EMBEDDING_DIM; i++) {
        double diff = std::abs(fit_embedding[i] - transform_embedding[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);

        if (diff > SIGNIFICANT_THRESHOLD) {
            significant_diff_count++;
        }
    }

    double avg_diff = total_diff / (N_SAMPLES * EMBEDDING_DIM);
    double mse = total_diff * total_diff / (N_SAMPLES * EMBEDDING_DIM);

    std::cout << "Results:" << std::endl;
    std::cout << "  Average difference: " << avg_diff << std::endl;
    std::cout << "  Maximum difference: " << max_diff << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    std::cout << "  Significant differences (>1e-6): " << significant_diff_count << "/" << (N_SAMPLES * EMBEDDING_DIM) << std::endl;
    std::cout << "  Percentage with significant differences: " << (100.0 * significant_diff_count / (N_SAMPLES * EMBEDDING_DIM)) << "%" << std::endl;

    // Show some sample values
    std::cout << std::endl << "Sample point comparisons (first 5 points):" << std::endl;
    for (int i = 0; i < 5 && i < N_SAMPLES; i++) {
        std::cout << "  Point " << i << ":" << std::endl;
        std::cout << "    Fit:      [" << fit_embedding[i * EMBEDDING_DIM] << ", " << fit_embedding[i * EMBEDDING_DIM + 1] << "]" << std::endl;
        std::cout << "    Transform:[" << transform_embedding[i * EMBEDDING_DIM] << ", " << transform_embedding[i * EMBEDDING_DIM + 1] << "]" << std::endl;
        std::cout << "    Diff:     [" << std::abs(fit_embedding[i * EMBEDDING_DIM] - transform_embedding[i * EMBEDDING_DIM]) << ", " << std::abs(fit_embedding[i * EMBEDDING_DIM + 1] - transform_embedding[i * EMBEDDING_DIM + 1]) << "]" << std::endl;
    }

    // Assessment
    std::cout << std::endl << "=== ASSESSMENT ===" << std::endl;
    if (max_diff < 1e-6) {
        std::cout << "✅ PERFECT: Fit and Transform are identical (as expected with exact k-NN)" << std::endl;
    } else if (max_diff < 1e-4) {
        std::cout << "⚠️  MINOR: Small differences detected (might be acceptable)" << std::endl;
    } else if (max_diff < 1e-2) {
        std::cout << "❌ PROBLEMATIC: Significant differences detected" << std::endl;
    } else {
        std::cout << "🚨 CRITICAL: Major differences - definitely a bug" << std::endl;
    }

    uwot_destroy(model);
    return 0;
}