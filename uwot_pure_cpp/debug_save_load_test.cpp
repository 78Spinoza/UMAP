#include <iostream>
#include <vector>
#include <random>
#include "uwot_simple_wrapper.h"

void progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    std::cout << "[" << phase << "] " << current << "/" << total << " (" << percent << "%)";
    if (message) std::cout << " - " << message;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Debug Save/Load Issue ===" << std::endl;

    // Create a small simple dataset
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    // Create model
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return -1;
    }

    // Train model with autoHNSWParam enabled
    std::vector<float> embedding(n_obs * embedding_dim);

    std::cout << "🚀 Training model..." << std::endl;
    int result = uwot_fit_with_progress_v2(
        model,
        data.data(),
        n_obs,
        n_dim,
        embedding_dim,
        15,  // n_neighbors
        0.1f,  // min_dist
        1.0f,  // spread
        50,  // n_epochs
        UWOT_METRIC_EUCLIDEAN,
        embedding.data(),
        progress_callback,
        0,  // force_exact_knn
        -1,  // M
        -1,  // ef_construction
        -1,  // ef_search
        0,  // use_quantization
        42,  // random_seed
        1   // autoHNSWParam (enabled)
    );

    if (result != 0) {
        std::cout << "❌ Training failed: " << result << std::endl;
        return -1;
    }
    std::cout << "✅ Training completed" << std::endl;

    // Check model info before save
    int n_vertices, n_dim_check, embedding_dim_check, n_neighbors;
    uwot_get_model_info(model, &n_vertices, &n_dim_check, &embedding_dim_check, &n_neighbors, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    std::cout << "📊 Model info before save: " << n_vertices << " points, " << n_dim_check << "->" << embedding_dim_check << " dims" << std::endl;

    // Save model
    const char* filename = "debug_save_load_test.umap";
    std::cout << "💾 Saving model to " << filename << "..." << std::endl;
    result = uwot_save_model(model, filename);
    if (result != 0) {
        std::cout << "❌ Save failed: " << result << std::endl;
        uwot_destroy(model);
        return -1;
    }
    std::cout << "✅ Model saved" << std::endl;

    // Destroy original model
    uwot_destroy(model);
    std::cout << "🗑️ Original model destroyed" << std::endl;

    // Load model
    std::cout << "📂 Loading model from " << filename << "..." << std::endl;
    UwotModel* loaded_model = uwot_load_model(filename);
    if (!loaded_model) {
        std::cout << "❌ Load failed: null model returned" << std::endl;
        return -1;
    }
    std::cout << "✅ Model loaded" << std::endl;

    // Check model info after load
    int n_vertices_loaded, n_dim_check_loaded, embedding_dim_check_loaded, n_neighbors_loaded;
    uwot_get_model_info(loaded_model, &n_vertices_loaded, &n_dim_check_loaded, &embedding_dim_check_loaded, &n_neighbors_loaded, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    std::cout << "📊 Model info after load: " << n_vertices_loaded << " points, " << n_dim_check_loaded << "->" << embedding_dim_check_loaded << " dims" << std::endl;

    // Check if model is fitted
    int is_fitted = uwot_is_fitted(loaded_model);
    std::cout << "🔧 Is fitted: " << (is_fitted ? "YES" : "NO") << std::endl;

    if (is_fitted) {
        // Try transform
        std::vector<float> new_data(1 * n_dim);
        for (int i = 0; i < n_dim; i++) {
            new_data[i] = dis(gen);
        }
        std::vector<float> new_embedding(1 * embedding_dim);

        std::cout << "🔄 Testing transform..." << std::endl;
        result = uwot_transform(loaded_model, new_data.data(), 1, n_dim, new_embedding.data());
        if (result == 0) {
            std::cout << "✅ Transform succeeded: [";
            for (int i = 0; i < embedding_dim; i++) {
                std::cout << new_embedding[i];
                if (i < embedding_dim - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "❌ Transform failed: " << result << " - " << uwot_get_error_message(result) << std::endl;
        }
    } else {
        std::cout << "❌ Cannot test transform - model not fitted after load" << std::endl;
    }

    // Cleanup
    uwot_destroy(loaded_model);

    std::cout << "🎉 Debug test completed" << std::endl;
    return 0;
}