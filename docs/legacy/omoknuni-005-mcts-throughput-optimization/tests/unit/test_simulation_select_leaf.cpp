/**
 * @file test_simulation_select_leaf.cpp
 * @brief Unit tests for SimulationRunner::select_leaf() method
 *
 * Tests the selection phase of MCTS simulation using a deterministic game fixture.
 *
 * Compile with:
 * g++ -std=c++17 -O2 -I../../cpp_extensions -o test_simulation_select_leaf \
 *     test_simulation_select_leaf.cpp \
 *     ../../cpp_extensions/mcts/tree.cpp \
 *     ../../cpp_extensions/mcts/selection.cpp \
 *     ../../cpp_extensions/mcts/backup.cpp \
 *     ../../cpp_extensions/mcts/virtual_loss.cpp \
 *     ../../cpp_extensions/mcts/simulation_runner.cpp \
 *     ../../cpp_extensions/utils/igamestate.cpp
 *
 * Or build with CMake test suite.
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include "mcts/simulation_runner.hpp"
#include "mcts/tree.hpp"
#include "mcts/selection.hpp"
#include "mcts/backup.hpp"
#include "mcts/virtual_loss.hpp"
#include "utils/igamestate.h"

using namespace mcts;
using namespace alphazero::core;

/**
 * @brief Simple deterministic game state for testing
 *
 * This creates a simple linear game tree where each position has 2-3 legal moves.
 * Used to test selection logic without complex game rules.
 */
class TestGameState : public IGameState {
public:
    TestGameState() : IGameState(GameType::GOMOKU), depth_(0), terminal_(false) {}

    std::vector<int> getLegalMoves() const override {
        if (terminal_) return {};
        if (depth_ >= 5) return {};  // Max depth 5
        return {0, 1, 2};  // Always 3 legal moves
    }

    bool isLegalMove(int action) const override {
        if (terminal_ || depth_ >= 5) return false;
        return action >= 0 && action <= 2;
    }

    void makeMove(int action) override {
        if (!isLegalMove(action)) {
            throw std::runtime_error("Illegal move");
        }
        move_history_.push_back(action);
        depth_++;
        // Terminate at depth 5
        if (depth_ >= 5) {
            terminal_ = true;
        }
    }

    bool undoMove() override {
        if (move_history_.empty()) return false;
        move_history_.pop_back();
        depth_--;
        terminal_ = false;
        return true;
    }

    bool isTerminal() const override {
        return terminal_;
    }

    GameResult getGameResult() const override {
        if (!terminal_) return GameResult::ONGOING;
        // Deterministic result based on move history
        return (move_history_.size() % 2 == 0) ? GameResult::WIN_PLAYER1 : GameResult::WIN_PLAYER2;
    }

    int getCurrentPlayer() const override {
        return (depth_ % 2) + 1;
    }

    int getBoardSize() const override {
        return 3;  // Dummy size
    }

    int getActionSpaceSize() const override {
        return 3;
    }

    std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const override {
        // Dummy implementation
        return std::vector<std::vector<std::vector<float>>>(1,
            std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));
    }

    std::vector<std::vector<std::vector<float>>> getBasicTensorRepresentation() const override {
        return getTensorRepresentation();
    }

    std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const override {
        return getTensorRepresentation();
    }

    uint64_t getHash() const override {
        return static_cast<uint64_t>(depth_);
    }

    std::unique_ptr<IGameState> clone() const override {
        auto copy = std::make_unique<TestGameState>();
        copy->depth_ = depth_;
        copy->terminal_ = terminal_;
        copy->move_history_ = move_history_;
        return copy;
    }

    void copyFrom(const IGameState& source) override {
        const TestGameState* src = dynamic_cast<const TestGameState*>(&source);
        if (src) {
            depth_ = src->depth_;
            terminal_ = src->terminal_;
            move_history_ = src->move_history_;
        }
    }

    std::string actionToString(int action) const override {
        return "move" + std::to_string(action);
    }

    std::vector<int> getMoveHistory() const override {
        return move_history_;
    }

    const std::vector<int>& getMoveHistoryRef() const {
        return move_history_;
    }

    std::optional<int> stringToAction(const std::string& moveStr) const override {
        if (moveStr == "move0") return 0;
        if (moveStr == "move1") return 1;
        if (moveStr == "move2") return 2;
        return std::nullopt;
    }

    std::string toString() const override {
        return "TestGameState(depth=" + std::to_string(depth_) + ")";
    }

    bool equals(const IGameState& other) const override {
        const TestGameState* other_state = dynamic_cast<const TestGameState*>(&other);
        if (!other_state) return false;
        return depth_ == other_state->depth_ && terminal_ == other_state->terminal_;
    }

    bool validate() const override {
        return depth_ >= 0 && depth_ <= 5;
    }

    std::vector<std::vector<uint64_t>> getBitboards() const override {
        return std::vector<std::vector<uint64_t>>();
    }

private:
    int depth_;
    bool terminal_;
    std::vector<int> move_history_;
};

void test_select_leaf_unexpanded_root() {
    std::cout << "Testing select_leaf with unexpanded root..." << std::endl;

    // Create MCTS components
    MCTSTree tree(1000);
    PUCTSelector selector;
    BackupManager backup(tree);
    VirtualLossManager vl(tree);

    // Create root node (unexpanded)
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Create game state
    TestGameState state;

    // Create simulation runner
    SimulationRunner runner(tree, selector, backup, vl);

    // Select leaf
    std::vector<NodeIndex> path;
    NodeIndex leaf = runner.select_leaf_public(root, state, path);

    // Verify
    assert(leaf == root && "Should select root as leaf when unexpanded");
    assert(path.size() == 1 && "Path should contain only root");
    assert(path[0] == root && "Path should start with root");
    assert(state.getMoveHistoryRef().empty() && "No moves should be applied");

    std::cout << "✓ Unexpanded root test passed" << std::endl;
}

void test_select_leaf_expanded_tree() {
    std::cout << "Testing select_leaf with expanded tree..." << std::endl;

    // Create MCTS components
    MCTSTree tree(1000);
    PUCTSelector selector;
    BackupManager backup(tree);
    VirtualLossManager vl(tree);

    // Create root node
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Expand root with 3 children
    NodeIndex first_child = tree.allocate_nodes(3);
    tree.set_first_child_index(root, first_child);
    tree.set_num_children(root, 3);

    // Set child move indices
    for (int i = 0; i < 3; ++i) {
        tree.set_move(first_child + i, static_cast<uint16_t>(i));
        tree.set_prior_prob(first_child + i, 0.33f);
        tree.set_visit_count(first_child + i, 0.0f);
        tree.set_parent_index(first_child + i, root);
    }

    // Mark root as expanded
    NodeFlags flags;
    flags.set_expanded(true);
    tree.set_flags(root, flags);
    tree.set_visit_count(root, 10.0f);

    // Create game state
    TestGameState state;

    // Create simulation runner
    SimulationRunner runner(tree, selector, backup, vl);

    // Select leaf
    std::vector<NodeIndex> path;
    NodeIndex leaf = runner.select_leaf_public(root, state, path);

    // Verify
    assert(leaf != root && "Should not select root when expanded");
    assert(leaf >= first_child && leaf < first_child + 3 && "Leaf should be one of the children");
    assert(path.size() == 2 && "Path should be root → child");
    assert(path[0] == root && "Path should start with root");
    assert(path[1] == leaf && "Path should end with leaf");
    assert(state.getMoveHistoryRef().size() == 1 && "One move should be applied");

    // Virtual loss should be applied to leaf
    float vl_value = tree.get_virtual_loss(leaf);
    assert(vl_value > 0.0f && "Virtual loss should be applied to selected leaf");

    std::cout << "✓ Expanded tree test passed" << std::endl;
}

void test_select_leaf_deep_tree() {
    std::cout << "Testing select_leaf with deeper tree..." << std::endl;

    // Create MCTS components
    MCTSTree tree(1000);
    PUCTSelector selector;
    BackupManager backup(tree);
    VirtualLossManager vl(tree);

    // Create a tree: root → child1 → grandchild (3 levels)
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Expand root
    NodeIndex child1 = tree.allocate_nodes(3);
    tree.set_first_child_index(root, child1);
    tree.set_num_children(root, 3);

    for (int i = 0; i < 3; ++i) {
        tree.set_move(child1 + i, static_cast<uint16_t>(i));
        tree.set_prior_prob(child1 + i, 0.33f);
        tree.set_visit_count(child1 + i, static_cast<float>(i + 1));  // Different visit counts
        tree.set_parent_index(child1 + i, root);
    }

    NodeFlags root_flags;
    root_flags.set_expanded(true);
    tree.set_flags(root, root_flags);
    tree.set_visit_count(root, 10.0f);

    // Expand first child
    NodeIndex grandchild = tree.allocate_nodes(3);
    tree.set_first_child_index(child1, grandchild);
    tree.set_num_children(child1, 3);

    for (int i = 0; i < 3; ++i) {
        tree.set_move(grandchild + i, static_cast<uint16_t>(i));
        tree.set_prior_prob(grandchild + i, 0.33f);
        tree.set_visit_count(grandchild + i, 0.0f);
        tree.set_parent_index(grandchild + i, child1);
    }

    NodeFlags child_flags;
    child_flags.set_expanded(true);
    tree.set_flags(child1, child_flags);
    tree.set_visit_count(child1, 5.0f);

    // Create game state
    TestGameState state;

    // Create simulation runner
    SimulationRunner runner(tree, selector, backup, vl);

    // Select leaf
    std::vector<NodeIndex> path;
    NodeIndex leaf = runner.select_leaf_public(root, state, path);

    // Verify - should traverse down to grandchild level
    // Note: The actual leaf depends on PUCT selection
    assert(path.size() >= 2 && "Path should have at least root → child");
    assert(path[0] == root && "Path should start with root");

    // Check if it went deep (to grandchild level) or stayed at child level
    if (path.size() == 3) {
        // Went to grandchild - this is expected if child1 was expanded
        assert(leaf >= grandchild && leaf < grandchild + 3 && "Leaf should be a grandchild");
        assert(state.getMoveHistoryRef().size() == 2 && "Two moves should be applied");
    } else {
        // Stopped at child level (child2 or child3 which are unexpanded)
        assert(path.size() == 2 && "Path should be root → child");
        assert(leaf >= child1 && leaf < child1 + 3 && "Leaf should be a child");
        assert(state.getMoveHistoryRef().size() == 1 && "One move should be applied");
    }

    std::cout << "✓ Deep tree test passed" << std::endl;
}

void test_select_leaf_terminal_node() {
    std::cout << "Testing select_leaf with terminal game state..." << std::endl;

    // Create MCTS components
    MCTSTree tree(1000);
    PUCTSelector selector;
    BackupManager backup(tree);
    VirtualLossManager vl(tree);

    // Create root
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Create game state at terminal position
    TestGameState state;
    // Make moves to reach terminal state
    for (int i = 0; i < 5; ++i) {
        state.makeMove(i % 3);
    }
    assert(state.isTerminal() && "State should be terminal");

    // Reset to near-terminal for selection test
    while (!state.getMoveHistoryRef().empty()) {
        state.undoMove();
    }

    // Make 4 moves (one away from terminal)
    for (int i = 0; i < 4; ++i) {
        state.makeMove(i % 3);
    }

    // Expand root and path to this position
    NodeIndex current = root;
    NodeFlags flags;
    flags.set_expanded(true);
    tree.set_flags(current, flags);
    tree.set_visit_count(current, 10.0f);

    // Add one child that will be terminal after move
    NodeIndex child = tree.allocate_node();
    tree.set_first_child_index(current, child);
    tree.set_num_children(current, 1);
    tree.set_move(child, 0);
    tree.set_prior_prob(child, 1.0f);
    tree.set_parent_index(child, current);

    // Create fresh game state for selection
    TestGameState fresh_state;
    for (int i = 0; i < 4; ++i) {
        fresh_state.makeMove(i % 3);
    }

    // Create simulation runner
    SimulationRunner runner(tree, selector, backup, vl);

    // Select leaf
    std::vector<NodeIndex> path;
    NodeIndex leaf = runner.select_leaf_public(root, fresh_state, path);

    // Verify - should select the child and detect terminal
    assert(leaf == child && "Should select child node");
    assert(fresh_state.isTerminal() && "State should be terminal after move");

    std::cout << "✓ Terminal node test passed" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Running SimulationRunner::select_leaf Tests" << std::endl;
    std::cout << "======================================" << std::endl << std::endl;

    try {
        test_select_leaf_unexpanded_root();
        test_select_leaf_expanded_tree();
        test_select_leaf_deep_tree();
        test_select_leaf_terminal_node();

        std::cout << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        std::cout << "======================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
