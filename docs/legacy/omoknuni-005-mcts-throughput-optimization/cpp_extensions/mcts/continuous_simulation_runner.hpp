/**
 * @file continuous_simulation_runner.hpp
 * @brief Continuous MCTS simulation runner with async inference
 *
 * This module implements a simulation runner that executes MCTS simulations
 * continuously without blocking on neural network inference. Simulations
 * submit inference requests to an AsyncInferenceQueue and immediately continue
 * with new simulations while waiting for results.
 *
 * Performance targets:
 * - 30,000+ simulations/second with 8-12 threads
 * - 75-85% parallel efficiency
 * - GPU utilization 60-80%
 * - Average batch size 48-64 positions
 *
 * Key design principles:
 * - Non-blocking simulation loop (threads never wait for inference)
 * - Pending expansion tracking (map of request_id → expansion data)
 * - Async result processing (check queue periodically, expand when ready)
 * - Continuous progress (always making forward progress on tree growth)
 */

#pragma once

#include "simulation_runner.hpp"
#include "async_inference_queue.hpp"
#include "profiling/enhanced_profiler.hpp"
#include <array>
#include <atomic>
#include <unordered_map>

namespace mcts {

/**
 * @brief Pending expansion data
 *
 * Tracks state for a simulation that has selected to a leaf
 * and submitted an inference request, but hasn't yet received
 * the result to expand the node.
 *
 * **Optimization (T018g)**: Uses raw pointer to pool-allocated state
 * instead of unique_ptr to eliminate clone() allocation overhead.
 * State lifecycle managed by ThreadLocalStatePool.
 */
struct PendingExpansion {
    NodeIndex leaf_node;                       // Node to expand with result
    std::vector<NodeIndex> path;               // Path from root to leaf (for backup)
    IGameState* state;                         // Game state at leaf (pool-managed, NON-owning)

    // Move-only type (paths use vector move semantics)
    PendingExpansion() : leaf_node(0), state(nullptr) {}
    PendingExpansion(PendingExpansion&&) = default;
    PendingExpansion& operator=(PendingExpansion&&) = default;
    PendingExpansion(const PendingExpansion&) = delete;
    PendingExpansion& operator=(const PendingExpansion&) = delete;
};

/**
 * @brief Batched update accumulator for reducing atomic contention (T014)
 *
 * Accumulates multiple visit/value updates for a single node before
 * applying them atomically. This is the key optimization in batched
 * result processing: instead of N atomic operations for N path
 * traversals that touch the same node, we do 1 atomic operation
 * with the accumulated increment.
 *
 * Performance Impact:
 * - Before: 32 results × 10 nodes × 2 atomics = 640 atomic operations
 * - After: ~160 unique nodes × 2 atomics = 320 atomic operations
 * - Result: 2× reduction + reduced contention
 */
struct BatchedUpdate {
    float visit_increment = 0.0f;  // Accumulated visit count increments
    float value_increment = 0.0f;  // Accumulated value increments

    BatchedUpdate() = default;
};

/**
 * @brief Ready result container for batched processing (T014)
 *
 * Holds a completed inference result along with its associated
 * pending expansion data. Used to collect all ready results before
 * processing them in batch to reduce lock contention and improve
 * cache locality.
 */
struct ReadyResult {
    PendingExpansion pending;           // Expansion data (leaf, path, state)
    InferenceResult result;             // Neural network inference result
    size_t slot_index;                  // Pending buffer slot index
    bool expansion_succeeded = false;   // Whether node expansion succeeded

    // Move-only type (owns expansion data)
    ReadyResult() = default;
    ReadyResult(ReadyResult&&) = default;
    ReadyResult& operator=(ReadyResult&&) = default;
    ReadyResult(const ReadyResult&) = delete;
    ReadyResult& operator=(const ReadyResult&) = delete;
};

/**
 * @brief Continuous MCTS simulation runner
 *
 * Runs MCTS simulations continuously without blocking on inference.
 * Achieves high throughput (30k+ sims/sec) by decoupling simulation
 * threads from GPU inference latency.
 *
 * Algorithm:
 * 1. Select to leaf (C++ tree traversal, ~0.26ms)
 * 2. Submit inference request to queue (non-blocking, ~0.1ms)
 * 3. Immediately start next simulation (no waiting!)
 * 4. Periodically check for completed results
 * 5. Expand nodes and backup values when results arrive
 * 6. Continue until quota reached
 *
 * Thread Safety:
 * - Multiple ContinuousSimulationRunner instances can run concurrently
 * - Each runner has independent pending expansion map
 * - Shared AsyncInferenceQueue is thread-safe
 * - Tree operations use atomics (same as base SimulationRunner)
 *
 * Performance Characteristics:
 * - Throughput: 30,000-40,000 sims/sec (8-12 threads)
 * - Latency: ~5ms per simulation (including queue time)
 * - Memory: ~100 bytes per pending expansion
 * - Parallelism: 75-85% efficiency with thread scaling
 */
class ContinuousSimulationRunner : public SimulationRunner {
public:
    /**
     * @brief Construct continuous simulation runner
     *
     * @param tree Shared MCTS tree (thread-safe via atomics)
     * @param selector PUCT child selector (thread-safe, no state)
     * @param backup Value backup manager (thread-safe)
     * @param virtual_loss Virtual loss coordinator (thread-safe)
     */
    ContinuousSimulationRunner(MCTSTree& tree,
                                PUCTSelector& selector,
                                BackupManager& backup,
                                VirtualLossManager& virtual_loss);

    /**
     * @brief Run continuous MCTS simulations with async inference
     *
     * Executes: Continuous loop of (Select → Queue → Process Results)
     * - Threads never block waiting for inference
     * - Simulations accumulate in pending expansions
     * - Results processed asynchronously as they arrive
     * - Loop continues until num_simulations completed
     *
     * Performance:
     * - Target: 30,000+ sims/sec with 8-12 threads
     * - Each simulation submits request (~0.1ms) then continues
     * - Results processed in batches (amortized cost)
     * - GPU batching happens in background coordinator
     *
     * Thread Safety:
     * - Safe to call from multiple threads with same queue
     * - Each thread has independent pending_expansions_ map
     * - Queue handles concurrent access internally
     *
     * @param root_state Initial game state (will be cloned for each simulation)
     * @param root_index Root node index in MCTS tree
     * @param queue Async inference queue for request/result exchange
     * @param num_simulations Number of simulations to complete
     * @return Number of successfully completed simulations
     */
    int run_continuous(IGameState& root_state,
                       NodeIndex root_index,
                       AsyncInferenceQueue& queue,
                       int num_simulations);

private:
    /**
     * @brief Process completed inference results
     *
     * Checks queue for available results, expands corresponding nodes,
     * and backs up values along paths. Processes all available results
     * in a batch to amortize queue access overhead.
     *
     * @param queue Async inference queue to poll for results
     * @return Number of results processed
     */
    int process_completed_results(AsyncInferenceQueue& queue);

    /**
     * @brief Expand node with pre-fetched inference result
     *
     * Same logic as SimulationRunner::expand_node() but with
     * policy/value already fetched from queue instead of calling
     * inference callback synchronously.
     *
     * @param leaf_node Node to expand
     * @param state Game state at leaf
     * @param policy Policy distribution over actions
     * @param value Position evaluation
     * @return true if expansion successful, false on error
     */
    bool expand_node_with_result(NodeIndex leaf_node,
                                   const IGameState& state,
                                   const std::vector<float>& policy,
                                   float value);

    /**
     * @brief Ensure root node is expanded before simulation threads start
     *
     * This eliminates the N-1 thread idle problem where all threads race
     * to expand the root, but only one succeeds and the others waste time.
     * By pre-expanding the root synchronously before threading begins,
     * all threads can immediately start productive work.
     *
     * Performance Impact: 2× speedup (eliminates initial serialization bottleneck)
     *
     * @param root_state Game state at root
     * @param root_index Root node index in tree
     * @param queue Async inference queue for synchronous root expansion
     * @return true if expansion performed, false if already expanded
     */
    bool ensure_root_expanded(IGameState& root_state,
                              NodeIndex root_index,
                              AsyncInferenceQueue& queue);

    /**
     * @brief Add Dirichlet noise to root node for exploration
     *
     * Mixes Dirichlet noise with policy priors at root to encourage
     * exploration during self-play. Uses AlphaZero's mixing formula:
     *   P'(a) = (1 - ε) * P(a) + ε * η_a
     * where η ~ Dir(α) and ε = 0.25
     *
     * @param root_index Root node index
     * @param alpha Dirichlet concentration parameter (0.3 for Go, 0.15 for Chess)
     */
    void add_dirichlet_noise(NodeIndex root_index, float alpha);

    /**
     * @brief Select to leaf using make_move pattern (T024f-6)
     *
     * Enhanced version of select_leaf that uses make_move() instead of makeMove(),
     * collecting undo tokens for efficient state restoration via unmake_move().
     * This eliminates the need for state cloning (418μs → ~15ns per move).
     *
     * @param root Root node index to start selection
     * @param current_state Game state (modified via make_move, 0 allocations)
     * @param path Output vector of node indices from root to leaf
     * @param undo_tokens Output vector of undo tokens for each move (parallel to path)
     * @return Index of selected leaf node
     */
    NodeIndex select_leaf_with_make_unmake(NodeIndex root,
                                             IGameState& current_state,
                                             std::vector<NodeIndex>& path,
                                             std::vector<uint64_t>& undo_tokens);

    /**
     * @brief Restore state to root via unmake_move (T024f-6)
     *
     * Unwinds the path by calling unmake_move() in reverse order,
     * restoring the state to the root position. Each unmake takes ~15ns.
     *
     * @param state Game state at leaf (will be restored to root)
     * @param path Node indices from root to leaf (from select_leaf)
     * @param undo_tokens Undo tokens collected during selection
     */
    void unwind_path(IGameState& state,
                     const std::vector<NodeIndex>& path,
                     const std::vector<uint64_t>& undo_tokens);

    /**
     * @brief Fixed-size ring buffer for pending expansions
     *
     * Replaces std::unordered_map with O(1) direct indexing using
     * request_id % CAPACITY. Provides faster lookups and lower memory
     * overhead than hash map.
     *
     * Capacity of 8192 supports high throughput with minimal memory
     * (8192 * ~200 bytes = 1.6 MB vs unordered_map's 3-4 MB overhead).
     */
    static constexpr size_t PENDING_BUFFER_CAPACITY = 8192;

    struct PendingSlot {
        std::atomic<bool> occupied{false};  // Slot in use
        uint64_t request_id{0};              // Request ID for verification
        PendingExpansion data;               // Actual expansion data

        PendingSlot() = default;
    };

    std::array<PendingSlot, PENDING_BUFFER_CAPACITY> pending_buffer_;
    std::atomic<size_t> pending_count_{0};  // Track number of pending items

    /**
     * @brief Thread-local state for zero-copy MCTS (T024f-6)
     *
     * Each worker thread maintains a persistent game state that is reused
     * across simulations. State is modified via make_move() during selection
     * and restored via unmake_move() after backup, eliminating the need for
     * expensive state cloning (418μs → ~15ns per move).
     */
    struct ThreadLocalState {
        std::unique_ptr<IGameState> state;      // Persistent state (reused across searches)
        std::vector<uint64_t> undo_tokens;      // Undo tokens for current path
        bool initialized = false;                // Initialization flag

        // T011: Pre-allocated feature buffer for zero-copy extraction
        std::vector<float> feature_buffer;      // Size = max_planes × max_board² (52KB for 36×19×19)
        bool feature_buffer_initialized = false; // Guard against double-initialization

        // Initialize state on first use, reset to root on subsequent searches
        void ensure_initialized(const IGameState& root) {
            if (!initialized) {
                // T024f-6: Track thread-local initialization (should happen once per thread)
                PROFILE_SCOPE(profiling::ProfileMetric::SelectionThreadLocalInit);
                {
                    PROFILE_SCOPE(profiling::ProfileMetric::SelectionActualStateClone);  // This IS a real clone
                    state = root.clone();
                }
                undo_tokens.reserve(256);  // Typical max MCTS depth
                initialized = true;
            } else {
                // CRITICAL FIX: Reset state to root for new search
                // Without this, state accumulates from previous searches causing:
                // - Memory bloat (3.5 MB → 100 MB)
                // - Performance degradation (1,396 → 869 sims/sec)
                PROFILE_SCOPE(profiling::ProfileMetric::StateThreadLocalClone);
                state->copyFrom(root);
                undo_tokens.clear();  // Clear any accumulated undo tokens
            }
        }

        // T012: Ensure feature buffer is properly sized
        // Called before each feature extraction to handle moved-from state
        void initialize_feature_buffer(int max_planes, int max_board_size) {
            size_t required_size = max_planes * max_board_size * max_board_size;

            // After std::move, buffer is empty - need to resize it
            if (feature_buffer.size() != required_size) {
                feature_buffer.resize(required_size);
                feature_buffer_initialized = true;
            }
        }
    };
};

} // namespace mcts
