/**
 * @file batch_inference_callback.hpp
 * @brief Abstract batch inference callback interface (no Python dependencies)
 *
 * This header defines the pure C++ interface for batch inference callbacks.
 * It has no pybind11 dependencies, allowing it to be used in the mcts_core
 * static library.
 */

#pragma once

#include "../utils/igamestate.h"
#include <vector>
#include <utility>

namespace mcts {

/**
 * @brief Abstract batch inference callback interface
 *
 * Allows C++ simulation runner to request batched neural network inference.
 * Batching reduces GIL crossings from N (per simulation) to 1 (per batch).
 *
 * This is a pure C++ interface - concrete implementations may use Python
 * (via PyBatchInferenceCallback) or native C++ inference backends.
 *
 * **T018g Optimization**: Supports pre-extracted features to eliminate state
 * cloning overhead (418μs → ~10μs per simulation).
 */
class BatchInferenceCallback {
public:
    virtual ~BatchInferenceCallback() = default;

    /**
     * @brief Request neural network inference for a batch of game states (legacy)
     *
     * @param states Vector of game state pointers to evaluate
     * @return Vector of (policy vector, value scalar) pairs
     *
     * Thread safety: Implementation must be thread-safe if called from
     * multiple threads (e.g., in BatchInferenceCoordinator background thread).
     */
    virtual std::vector<std::pair<std::vector<float>, float>>
    batch_inference(const std::vector<const IGameState*>& states) = 0;

    /**
     * @brief Request neural network inference with pre-extracted features (T018g)
     *
     * Optimized path that accepts pre-extracted feature tensors, eliminating
     * the need for state cloning and Python-side feature extraction.
     *
     * Default implementation falls back to batch_inference (not optimized).
     * Derived classes should override this for optimal performance.
     *
     * @param features_batch Vector of flattened feature tensors (C×H×W each)
     * @param board_sizes Vector of board sizes (for reshaping in Python)
     * @param num_planes_list Vector of feature plane counts
     * @return Vector of (policy, value) pairs
     */
    virtual std::vector<std::pair<std::vector<float>, float>>
    batch_inference_features(const std::vector<std::vector<float>>& features_batch,
                              const std::vector<int>& board_sizes,
                              const std::vector<int>& num_planes_list) {
        // Default fallback: not implemented
        // Derived classes (PyBatchInferenceCallback) should override this
        throw std::runtime_error("batch_inference_features not implemented - use PyBatchInferenceCallback");
        (void)features_batch;  // Suppress unused warnings
        (void)board_sizes;
        (void)num_planes_list;
    }
};

} // namespace mcts
