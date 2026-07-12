// cpp_extensions/mcts/state_pool.hpp
// T018b: ThreadLocalStatePool Implementation (PERFORMANCE FIX v2)
// Lock-free ring buffer with lazy allocation

#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include "../utils/igamestate.h"

namespace mcts {

using alphazero::core::IGameState;

// Forward declaration (defined in dlpack_bridge.hpp)
enum class GameType;

/**
 * @brief Thread-local state pool with lock-free ring buffer + lazy allocation
 *
 * **DESIGN v2 (Performance + Memory Fix)**:
 * - Lock-free ring buffer (no mutex contention!)
 * - Lazy allocation (allocate on first access to slot)
 * - No pre-allocation (starts with 0 memory)
 * - Fixed size (prevents unbounded growth)
 * - No-op release() (ring buffer automatically reuses)
 *
 * **Why this works**:
 * - Peak concurrent simulations determines how many slots get allocated
 * - Once allocated, slots are reused via wraparound
 * - No mutex = no performance regression (same speed as old ring buffer)
 * - Lazy allocation = memory efficiency (only allocates what's used)
 *
 * **Performance**:
 * - `acquire()`: O(1), ~5ns (atomic increment + conditional alloc)
 * - `release()`: O(1), ~2ns (no-op, for API compat)
 * - No mutex contention!
 *
 * **Memory** (example):
 * - 2000 sims, 8 threads, ~100 peak concurrent per thread
 * - Allocates 100 slots × 120KB = **12MB per thread** (96MB total)
 * - vs 0MB pre-allocation, 3.9GB with old pre-allocated ring buffer!
 *
 * **Ring size**: Set to accommodate peak concurrent + safety margin
 * - Typical: 256-512 per thread
 * - If wraparound with in-use state → illegal move (detectable)
 */
class ThreadLocalStatePool {
public:
    /**
     * @brief Construct pool with lazy-allocated ring buffer
     *
     * @param game_type Type of game (GOMOKU, CHESS, GO)
     * @param ring_size Ring buffer capacity (default: 512)
     *
     * Ring size should be >= peak concurrent simulations per thread.
     * Larger = more memory reserved (but not allocated upfront).
     * Smaller = risk of wraparound if peak concurrent > ring_size.
     */
    explicit ThreadLocalStatePool(GameType game_type, size_t ring_size = 512);

    /**
     * @brief Destructor
     */
    ~ThreadLocalStatePool() = default;

    // Non-copyable, non-movable (thread-local singleton pattern)
    ThreadLocalStatePool(const ThreadLocalStatePool&) = delete;
    ThreadLocalStatePool& operator=(const ThreadLocalStatePool&) = delete;
    ThreadLocalStatePool(ThreadLocalStatePool&&) = delete;
    ThreadLocalStatePool& operator=(ThreadLocalStatePool&&) = delete;

    /**
     * @brief Acquire a state from the pool (lock-free, O(1))
     *
     * Gets next slot from ring buffer. If slot is null (first access),
     * allocates state lazily. Otherwise returns existing state.
     *
     * **Performance**: ~5ns if slot exists, ~1μs if allocating
     *
     * @return Pointer to IGameState (never null, always valid)
     */
    IGameState* acquire();

    /**
     * @brief Release a state back to the pool (no-op, O(1))
     *
     * Ring buffer automatically reuses states via wraparound.
     * This is a no-op for performance, included for API compat.
     *
     * **Performance**: ~2ns (updates stats only)
     *
     * @param state Pointer to state (ignored)
     */
    void release(IGameState* state);

    /**
     * @brief Get pool statistics
     *
     * Returns usage statistics for monitoring and debugging.
     *
     * @return Stats struct with counters
     */
    struct Stats {
        size_t total_acquires;     ///< Total acquire() calls
        size_t total_releases;     ///< Total release() calls
        size_t slots_allocated;    ///< Number of ring slots actually allocated
        size_t ring_size;          ///< Total ring buffer capacity
        size_t peak_usage;         ///< Peak concurrent usage estimate
    };
    Stats get_stats() const;

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

private:
    GameType game_type_;  ///< Game type for state creation
    size_t ring_size_;    ///< Ring buffer capacity

    // Ring buffer (fixed size, lazily allocated)
    std::vector<std::unique_ptr<IGameState>> ring_;  ///< Ring buffer slots

    // Lock-free ring buffer index
    std::atomic<size_t> next_idx_{0};

    // Statistics (relaxed atomics for minimal overhead)
    std::atomic<size_t> total_acquires_{0};
    std::atomic<size_t> total_releases_{0};
    std::atomic<size_t> peak_usage_{0};
};

/**
 * @brief Get thread-local state pool (lazy initialization)
 *
 * Returns a pointer to the thread-local pool singleton. Creates pool
 * on first access. Pool persists for thread lifetime.
 *
 * **Thread Safety**: Each thread has its own pool (no contention).
 *
 * @param game_type Type of game
 * @param ring_size Ring buffer capacity (default: 512, only used on first call)
 * @return Pointer to thread-local pool (never null)
 */
ThreadLocalStatePool* get_thread_state_pool(GameType game_type, size_t ring_size = 512);

} // namespace mcts
