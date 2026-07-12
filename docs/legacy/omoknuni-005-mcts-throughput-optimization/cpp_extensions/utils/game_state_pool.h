// include/core/game_state_pool.h
#ifndef ALPHAZERO_CORE_GAME_STATE_POOL_H
#define ALPHAZERO_CORE_GAME_STATE_POOL_H

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include "igamestate.h"

namespace alphazero {
namespace core {

/**
 * @brief Thread-safe object pool for game states
 * 
 * Provides efficient memory management by reusing game state objects
 * instead of allocating/deallocating them repeatedly.
 */
template<typename GameStateType>
class GameStatePool {
public:
    /**
     * @brief Constructor
     * 
     * @param initial_size Initial pool size
     * @param max_size Maximum pool size (0 = unlimited)
     */
    explicit GameStatePool(size_t initial_size = 1000, size_t max_size = 10000)
        : max_size_(max_size), total_created_(0) {
        // Pre-allocate initial pool
        pool_.reserve(initial_size);
        for (size_t i = 0; i < initial_size; ++i) {
            pool_.push_back(createNew());
        }
    }

    /**
     * @brief Get a game state from the pool
     * 
     * @return Unique pointer to a game state
     */
    std::unique_ptr<GameStateType> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!pool_.empty()) {
            auto state = std::move(pool_.back());
            pool_.pop_back();
            return state;
        }
        
        // Pool is empty, create new if under limit
        if (max_size_ == 0 || total_created_ < max_size_) {
            return createNew();
        }
        
        // At capacity, fallback to regular allocation
        return std::make_unique<GameStateType>();
    }

    /**
     * @brief Return a game state to the pool
     * 
     * @param state The state to return
     */
    void release(std::unique_ptr<GameStateType> state) {
        if (!state) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Only keep if under max pool size
        if (max_size_ == 0 || pool_.size() < max_size_) {
            pool_.push_back(std::move(state));
        }
        // Otherwise let it be destroyed
    }

    /**
     * @brief Get current pool size
     * 
     * @return Number of states currently in pool
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

    /**
     * @brief Get total number of states created
     * 
     * @return Total states created
     */
    size_t totalCreated() const {
        return total_created_.load();
    }

private:
    std::unique_ptr<GameStateType> createNew() {
        total_created_++;
        return std::make_unique<GameStateType>();
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<GameStateType>> pool_;
    size_t max_size_;
    std::atomic<size_t> total_created_;
};

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_GAME_STATE_POOL_H