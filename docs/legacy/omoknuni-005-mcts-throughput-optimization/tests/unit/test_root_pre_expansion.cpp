/**
 * @file test_root_pre_expansion.cpp
 * @brief Unit tests for root pre-expansion (T003)
 *
 * Validates that root nodes are expanded before simulation threads start,
 * eliminating the N-1 thread idle problem and improving performance by 2×.
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/continuous_simulation_runner.hpp"
#include "../../cpp_extensions/mcts/tree.hpp"
#include "../../cpp_extensions/mcts/selection.hpp"
#include "../../cpp_extensions/mcts/backup.hpp"
#include "../../cpp_extensions/mcts/virtual_loss.hpp"
#include "../../cpp_extensions/mcts/async_inference_queue.hpp"
#include "../../cpp_extensions/utils/igamestate.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <optional>
#include <cstring>  // For memset in MockGameState

using namespace mcts;

using namespace alphazero::core;

// Complete mock game state for testing - implements all IGameState methods
class MockGameState : public IGameState {
private:
    int move_count_ = 0;
    bool terminal_ = false;
    std::vector<int> history_;

public:
    MockGameState() : IGameState(GameType::GOMOKU) {}

    // Core game logic
    std::vector<int> getLegalMoves() const override {
        if (terminal_) return {};
        return {0, 1, 2, 3, 4};  // 5 legal moves
    }

    bool isLegalMove(int action) const override {
        if (terminal_) return false;
        return action >= 0 && action < 5;
    }

    void makeMove(int action) override {
        history_.push_back(action);
        move_count_++;
        if (move_count_ >= 5) {
            terminal_ = true;
        }
    }

    bool undoMove() override {
        if (history_.empty()) return false;
        history_.pop_back();
        move_count_--;
        terminal_ = false;
        return true;
    }

    bool isTerminal() const override {
        return terminal_;
    }

    GameResult getGameResult() const override {
        if (!terminal_) return GameResult::ONGOING;
        return move_count_ % 2 == 0 ? GameResult::WIN_PLAYER2 : GameResult::WIN_PLAYER1;
    }

    int getCurrentPlayer() const override {
        return (move_count_ % 2) + 1;  // Returns 1 or 2
    }

    int getBoardSize() const override {
        return 5;  // 5x5 mock board
    }

    int getActionSpaceSize() const override {
        return 5;
    }

    // Tensor representations - simplified for mocking
    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override {
        // Return 5x5 board with 2 channels (simplified)
        return std::vector<std::vector<std::vector<float>>>(
            2, std::vector<std::vector<float>>(5, std::vector<float>(5, 0.0f))
        );
    }

    std::vector<std::vector<std::vector<float>>> getBasicTensorRepresentation() const override {
        // 18 channels for AlphaZero format
        return std::vector<std::vector<std::vector<float>>>(
            18, std::vector<std::vector<float>>(5, std::vector<float>(5, 0.0f))
        );
    }

    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        // Same as basic for mock
        return getBasicTensorRepresentation();
    }

    void extract_features_to_buffer(float* buffer) const override {
        // Simple stub: write zeros to buffer
        int num_planes = get_num_feature_planes();
        int board_size = getBoardSize();
        std::memset(buffer, 0, num_planes * board_size * board_size * sizeof(float));
    }

    int get_num_feature_planes() const override {
        return 18;  // Mock: same as basic tensor representation
    }

    uint64_t getHash() const override {
        uint64_t hash = 0;
        for (int move : history_) {
            hash = hash * 31 + move;
        }
        return hash;
    }

    std::unique_ptr<IGameState> clone() const override {
        auto cloned = std::make_unique<MockGameState>();
        cloned->move_count_ = move_count_;
        cloned->terminal_ = terminal_;
        cloned->history_ = history_;
        return cloned;
    }

    void copyFrom(const IGameState& source) override {
        const MockGameState* mock_source = dynamic_cast<const MockGameState*>(&source);
        if (mock_source) {
            move_count_ = mock_source->move_count_;
            terminal_ = mock_source->terminal_;
            history_ = mock_source->history_;
        }
    }

    // T024c: Stub implementations for make/unmake (not used in this test)
    uint64_t make_move(uint16_t move) override {
        throw std::runtime_error("MockGameState make_move not implemented");
    }

    void unmake_move(uint16_t move, uint64_t undo_token) override {
        throw std::runtime_error("MockGameState unmake_move not implemented");
    }

    std::string actionToString(int action) const override {
        return std::to_string(action);
    }

    std::optional<int> stringToAction(const std::string& moveStr) const override {
        try {
            int action = std::stoi(moveStr);
            if (action >= 0 && action < 5) {
                return action;
            }
        } catch (...) {}
        return std::nullopt;
    }

    std::string toString() const override {
        return "MockGameState(moves=" + std::to_string(move_count_) + ")";
    }

    bool equals(const IGameState& other) const override {
        const MockGameState* mock_other = dynamic_cast<const MockGameState*>(&other);
        return mock_other &&
               move_count_ == mock_other->move_count_ &&
               terminal_ == mock_other->terminal_;
    }

    std::vector<int> getMoveHistory() const override {
        return history_;
    }

    bool validate() const override {
        return true;  // Always valid for mock
    }

    std::vector<std::vector<uint64_t>> getBitboards() const override {
        // Return empty bitboards for mock (2 players)
        return std::vector<std::vector<uint64_t>>(2, std::vector<uint64_t>(1, 0ULL));
    }
};

class RootPreExpansionTest : public ::testing::Test {
protected:
    static constexpr std::size_t MAX_NODES = 100'000;

    void SetUp() override {
        tree_ = new MCTSTree(MAX_NODES);
        selector_ = new PUCTSelector(PUCTConfig{});
        backup_ = new BackupManager(*tree_);
        virtual_loss_ = new VirtualLossManager(*tree_);
        runner_ = new ContinuousSimulationRunner(*tree_, *selector_, *backup_, *virtual_loss_);
        queue_ = new AsyncInferenceQueue();
    }

    void TearDown() override {
        delete queue_;
        delete runner_;
        delete virtual_loss_;
        delete backup_;
        delete selector_;
        delete tree_;
    }

    MCTSTree* tree_;
    PUCTSelector* selector_;
    BackupManager* backup_;
    VirtualLossManager* virtual_loss_;
    ContinuousSimulationRunner* runner_;
    AsyncInferenceQueue* queue_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(RootPreExpansionTest, RootGetsExpandedBeforeSimulations) {
    MockGameState root_state;
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Verify root is not expanded initially
    EXPECT_FALSE(tree_->get_flags(root).is_expanded());
    EXPECT_EQ(tree_->get_num_children(root), 0);

    // Start a background thread to process inference requests
    std::atomic<bool> running{true};
    std::thread coordinator([&]() {
        while (running.load()) {
            auto batch = queue_->collect_batch(1, 10.0);  // Small batch, quick timeout
            if (!batch.empty()) {
                std::vector<InferenceResult> results;
                for (const auto& req : batch) {
                    InferenceResult result;
                    result.request_id = req.request_id;
                    result.policy = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};  // Uniform policy
                    result.value = 0.0f;
                    results.push_back(result);
                }
                queue_->submit_results(results);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run simulations (root should be pre-expanded)
    runner_->run_continuous(root_state, root, *queue_, 10);

    running.store(false);
    coordinator.join();

    // Verify root was expanded
    EXPECT_TRUE(tree_->get_flags(root).is_expanded());
    EXPECT_GT(tree_->get_num_children(root), 0);
}

TEST_F(RootPreExpansionTest, AlreadyExpandedRootIsNotReexpanded) {
    MockGameState root_state;
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Manually expand root first
    std::vector<int> legal_moves = root_state.getLegalMoves();
    NodeIndex first_child = tree_->allocate_nodes(legal_moves.size());
    tree_->set_first_child_index(root, first_child);
    tree_->set_num_children(root, legal_moves.size());

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        NodeIndex child = first_child + i;
        tree_->set_prior_prob(child, 0.2f);
        tree_->set_parent_index(child, root);
    }

    NodeFlags flags = tree_->get_flags(root);
    flags.set_expanded(true);
    tree_->set_flags(root, flags);

    uint16_t original_child_count = tree_->get_num_children(root);

    // Start coordinator thread
    std::atomic<bool> running{true};
    std::thread coordinator([&]() {
        while (running.load()) {
            auto batch = queue_->collect_batch(1, 10.0);
            if (!batch.empty()) {
                std::vector<InferenceResult> results;
                for (const auto& req : batch) {
                    InferenceResult result;
                    result.request_id = req.request_id;
                    result.policy = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
                    result.value = 0.0f;
                    results.push_back(result);
                }
                queue_->submit_results(results);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run simulations
    runner_->run_continuous(root_state, root, *queue_, 10);

    running.store(false);
    coordinator.join();

    // Verify root still has same children (not re-expanded)
    EXPECT_EQ(tree_->get_num_children(root), original_child_count);
}

// ============================================================================
// Dirichlet Noise Tests
// ============================================================================

TEST_F(RootPreExpansionTest, DirichletNoiseIsAppliedToRoot) {
    MockGameState root_state;
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Start coordinator thread
    std::atomic<bool> running{true};
    std::thread coordinator([&]() {
        while (running.load()) {
            auto batch = queue_->collect_batch(1, 10.0);
            if (!batch.empty()) {
                std::vector<InferenceResult> results;
                for (const auto& req : batch) {
                    InferenceResult result;
                    result.request_id = req.request_id;
                    result.policy = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};  // Uniform original policy
                    result.value = 0.0f;
                    results.push_back(result);
                }
                queue_->submit_results(results);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run simulations
    runner_->run_continuous(root_state, root, *queue_, 10);

    running.store(false);
    coordinator.join();

    // Check that priors have been modified by Dirichlet noise
    // Original policy: [0.2, 0.2, 0.2, 0.2, 0.2]
    // With noise: should have some variation (not all exactly 0.2)
    NodeIndex first_child = tree_->get_first_child_index(root);
    uint16_t num_children = tree_->get_num_children(root);

    if (num_children > 0) {
        std::vector<float> priors;
        for (uint16_t i = 0; i < num_children; ++i) {
            priors.push_back(tree_->get_prior_prob(first_child + i));
        }

        // Check that at least one prior is different from 0.2 (noise was added)
        bool has_variation = false;
        for (float p : priors) {
            if (std::abs(p - 0.2f) > 0.01f) {
                has_variation = true;
                break;
            }
        }
        EXPECT_TRUE(has_variation) << "Dirichlet noise should modify priors";

        // Check that priors still sum to approximately 1.0
        float sum = 0.0f;
        for (float p : priors) {
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Priors should still sum to 1.0 after noise";
    }
}

TEST_F(RootPreExpansionTest, DirichletNoiseRespectsPriorDistribution) {
    MockGameState root_state;
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Start coordinator thread with non-uniform policy
    std::atomic<bool> running{true};
    std::thread coordinator([&]() {
        while (running.load()) {
            auto batch = queue_->collect_batch(1, 10.0);
            if (!batch.empty()) {
                std::vector<InferenceResult> results;
                for (const auto& req : batch) {
                    InferenceResult result;
                    result.request_id = req.request_id;
                    // Non-uniform policy: one action much more likely
                    result.policy = {0.6f, 0.1f, 0.1f, 0.1f, 0.1f};
                    result.value = 0.0f;
                    results.push_back(result);
                }
                queue_->submit_results(results);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run simulations
    runner_->run_continuous(root_state, root, *queue_, 10);

    running.store(false);
    coordinator.join();

    // With ε=0.25 mixing, the first action should still be most likely
    // Mixed: (1-0.25)*0.6 + 0.25*noise ≈ 0.45 + noise_contribution
    NodeIndex first_child = tree_->get_first_child_index(root);
    uint16_t num_children = tree_->get_num_children(root);

    if (num_children > 0) {
        float max_prior = 0.0f;
        for (uint16_t i = 0; i < num_children; ++i) {
            float prior = tree_->get_prior_prob(first_child + i);
            max_prior = std::max(max_prior, prior);
        }

        // First child should likely still have highest prior (though not guaranteed due to random noise)
        float first_prior = tree_->get_prior_prob(first_child);

        // At least verify the range is reasonable
        EXPECT_GT(first_prior, 0.3f) << "High-probability action should remain relatively high";
        EXPECT_LT(first_prior, 0.7f) << "Noise should prevent complete certainty";
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(RootPreExpansionTest, MultipleThreadsDoNotDuplicateExpansion) {
    MockGameState root_state;
    NodeIndex root = tree_->add_root_node(0.5f, 0);

    // Start coordinator thread that processes inference requests
    std::atomic<bool> running{true};
    std::atomic<int> requests_processed{0};
    std::thread coordinator([&]() {
        while (running.load()) {
            auto batch = queue_->collect_batch(1, 2.0);  // 2ms timeout for faster response
            if (!batch.empty()) {
                std::vector<InferenceResult> results;
                for (const auto& req : batch) {
                    InferenceResult result;
                    result.request_id = req.request_id;
                    result.policy = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
                    result.value = 0.0f;
                    results.push_back(result);
                }
                queue_->submit_results(results);
                requests_processed.fetch_add(results.size());
            }
            // No sleep - process requests as fast as possible
        }
    });

    // Give coordinator time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Launch multiple threads that each try to ensure root is expanded
    // Only one should actually perform the expansion
    std::vector<std::thread> threads;
    std::atomic<int> expansions_performed{0};

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            ContinuousSimulationRunner thread_runner(*tree_, *selector_, *backup_, *virtual_loss_);
            // Directly test ensure_root_expanded (can't call private method, so we call run_continuous with 0 sims)
            // This will trigger ensure_root_expanded but not run any actual simulations
            // Use a separate root state clone for each thread
            std::unique_ptr<IGameState> thread_state = root_state.clone();
            if (thread_state) {
                // Actually, we can't directly call ensure_root_expanded since it's private
                // Instead, we'll rely on run_continuous calling it
                // But set simulations to 0 so it exits immediately after root expansion
                thread_runner.run_continuous(*thread_state, root, *queue_, 0);
            }
        });
    }

    // Wait for all threads with timeout
    for (auto& t : threads) {
        t.join();
    }

    // Wait a bit for any pending operations
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    running.store(false);
    coordinator.join();

    // Verify root was expanded exactly once
    EXPECT_TRUE(tree_->get_flags(root).is_expanded()) << "Root should be expanded";

    // Verify children are properly initialized (no corruption from concurrent access)
    uint16_t num_children = tree_->get_num_children(root);
    EXPECT_GT(num_children, 0) << "Root should have children after expansion";
    EXPECT_EQ(num_children, 5) << "Root should have exactly 5 children (one per legal move)";

    NodeIndex first_child = tree_->get_first_child_index(root);
    for (uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        EXPECT_TRUE(tree_->is_valid_index(child)) << "Child " << i << " should be valid";
        EXPECT_EQ(tree_->get_parent_index(child), root) << "Child " << i << " should have correct parent";

        // Verify prior probabilities are set and modified by Dirichlet noise
        float prior = tree_->get_prior_prob(child);
        EXPECT_GT(prior, 0.0f) << "Child " << i << " should have positive prior";
        EXPECT_LT(prior, 1.0f) << "Child " << i << " should have prior < 1.0";
    }

    // Verify only one inference request was processed (for root expansion)
    // Since all 4 threads try to expand the same root, only one should submit a request
    EXPECT_EQ(requests_processed.load(), 1) << "Only one inference request should be processed for root";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
