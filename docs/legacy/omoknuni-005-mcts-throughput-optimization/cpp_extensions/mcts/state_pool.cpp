// cpp_extensions/mcts/state_pool.cpp
// T018b: ThreadLocalStatePool Implementation (PERFORMANCE FIX v2)
// Lock-free ring buffer with lazy allocation

#include "state_pool.hpp"
#include "dlpack_bridge.hpp"  // For GameType definition
#include "../games/gomoku/gomoku_state.h"
#include "../games/chess/chess_state.h"
#include "../games/go/go_state.h"
#include <thread>
#include <stdexcept>

namespace mcts {

using alphazero::games::gomoku::GomokuState;
using alphazero::games::chess::ChessState;
using alphazero::games::go::GoState;

// Helper: Create state for game type
static std::unique_ptr<IGameState> create_state_for_game(GameType game_type) {
    switch (game_type) {
        case GameType::GOMOKU:
            return std::make_unique<GomokuState>();
        case GameType::CHESS:
            return std::make_unique<ChessState>();
        case GameType::GO:
            return std::make_unique<GoState>();
        default:
            throw std::invalid_argument("ThreadLocalStatePool: unsupported game type");
    }
}

ThreadLocalStatePool::ThreadLocalStatePool(GameType game_type, size_t ring_size)
    : game_type_(game_type), ring_size_(ring_size) {

    if (ring_size == 0) {
        throw std::invalid_argument("ThreadLocalStatePool: ring_size must be > 0");
    }

    // Reserve slots (but don't allocate states yet - lazy allocation!)
    ring_.resize(ring_size);  // All slots initially null
}

IGameState* ThreadLocalStatePool::acquire() {
    // Lock-free atomic increment
    total_acquires_.fetch_add(1, std::memory_order_relaxed);
    size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed);

    // Wrap around ring buffer
    size_t slot = idx % ring_size_;

    // Lazy allocation: allocate state on first access to this slot
    if (!ring_[slot]) {
        ring_[slot] = create_state_for_game(game_type_);
    }

    // Update peak usage tracking
    size_t current_usage = (idx / ring_size_) + 1;
    size_t peak = peak_usage_.load(std::memory_order_relaxed);
    while (current_usage > peak) {
        if (peak_usage_.compare_exchange_weak(peak, current_usage,
                                              std::memory_order_relaxed)) {
            break;
        }
    }

    return ring_[slot].get();
}

void ThreadLocalStatePool::release(IGameState* state) {
    // No-op: Ring buffer automatically reuses states
    // Update stats for monitoring
    total_releases_.fetch_add(1, std::memory_order_relaxed);
    (void)state;  // Suppress unused warning
}

ThreadLocalStatePool::Stats ThreadLocalStatePool::get_stats() const {
    // Count allocated slots
    size_t allocated = 0;
    for (const auto& slot : ring_) {
        if (slot) {
            ++allocated;
        }
    }

    return Stats{
        .total_acquires = total_acquires_.load(std::memory_order_relaxed),
        .total_releases = total_releases_.load(std::memory_order_relaxed),
        .slots_allocated = allocated,
        .ring_size = ring_size_,
        .peak_usage = peak_usage_.load(std::memory_order_relaxed)
    };
}

void ThreadLocalStatePool::reset_stats() {
    total_acquires_.store(0, std::memory_order_relaxed);
    total_releases_.store(0, std::memory_order_relaxed);
    peak_usage_.store(0, std::memory_order_relaxed);
}

// Thread-local singleton accessor
ThreadLocalStatePool* get_thread_state_pool(GameType game_type, size_t ring_size) {
    // Thread-local storage: each thread has its own pool (no contention)
    // Lazy initialization: created on first call per thread
    // Lifetime: persists for thread duration
    thread_local std::unique_ptr<ThreadLocalStatePool> pool;

    if (!pool) {
        pool = std::make_unique<ThreadLocalStatePool>(game_type, ring_size);
    }

    return pool.get();
}

} // namespace mcts
