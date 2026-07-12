# T024f-6: Legal Move Filtering Analysis

**Date**: 2025-10-17
**Issue**: Illegal move errors in integration tests
**Severity**: CRITICAL - Must resolve before proceeding

---

## Problem Statement

Integration tests show "Illegal Move" errors:
```
RuntimeError: Illegal Move: Attempted illegal move: P1 for player 2 (Action: 224)
```

This suggests:
1. A move for the wrong player is being applied, OR
2. State/tree synchronization is broken, OR
3. Test infrastructure is incorrect

---

## Root Cause Investigation

### Issue 1: Tests Target Wrong Class ⚠️

**Current Test Code**:
```python
runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
runner.run_simulation(state, 0, callback)  # Uses OLD clone-based code
```

**Problem**: SimulationRunner::run_simulation() doesn't use make/unmake!
- Line 45: `current_state = root_state.clone()`  ← OLD approach
- Line 145: `current_state.makeMove(move_index)`  ← OLD makeMove, not make_move
- No make/unmake, no undo tokens, no thread-local state

**The tests don't validate T024f-6 at all!** They test the old implementation.

### Issue 2: Legal Move Filtering in MCTS

**Question**: Does MCTS correctly filter illegal moves at node level?

**Answer**: YES ✅

**Evidence from expand_node()** (simulation_runner.cpp:174-263):
```cpp
// Phase 1: Get legal moves
std::vector<int> legal_moves = state.getLegalMoves();  // Line 177

// Phase 2: Mask policy to legal moves
for (size_t i = 0; i < legal_moves.size(); ++i) {
    int move = legal_moves[i];
    if (move >= 0 && move < action_space_size) {
        masked_policy[i] = policy[move];  // Only legal moves
        policy_sum += policy[move];
    }
}

// Phase 3: Store ONLY legal moves in tree
for (uint16_t i = 0; i < num_children; ++i) {
    tree_.set_move(child_idx, static_cast<uint16_t>(legal_moves[i]));  // Line 263
}
```

**Conclusion**: Only legal moves are stored in tree nodes ✅

**Evidence from select_leaf()** (simulation_runner.cpp:136-145):
```cpp
// Get move from tree (guaranteed legal at expansion time)
uint16_t move_index = tree_.get_move(result.selected_child);  // Line 136

// Apply move to state
current_state.makeMove(static_cast<int>(move_index));  // Line 145
```

**Invariant**: Moves stored in tree were legal at expansion time.

### Issue 3: Make/Unmake Legal Move Guarantee

**Question**: Does make/unmake preserve legal move filtering?

**Implementation in select_leaf_with_make_unmake()** (continuous_simulation_runner.cpp:627-642):
```cpp
// Get the move that led to this child (from tree)
uint16_t move_index = tree_.get_move(result.selected_child);  // Line 627

// Apply virtual loss
virtual_loss_.apply_virtual_loss(result.selected_child);  // Line 632

// T024f-6: Apply move via make_move
uint64_t undo_token = current_state.make_move(move_index);  // Line 641

// Store undo token
undo_tokens.push_back(undo_token);  // Line 645
```

**Analysis**:
1. Move comes from tree (stored during expansion with legal move filtering) ✅
2. make_move() validates legality in debug mode (gomoku_state.cpp:1311-1316) ✅
3. Undo token is captured for restoration ✅

**Conclusion**: make/unmake uses the SAME legal moves from tree as old approach ✅

---

## Why Tests Fail

### Hypothesis 1: Wrong Class Tested
**Evidence**: Tests use SimulationRunner, not ContinuousSimulationRunner
**Result**: Tests don't exercise make/unmake code at all
**Fix Required**: Create tests for ContinuousSimulationRunner::run_continuous()

### Hypothesis 2: Test Infrastructure Issue
**Evidence**: First test passes, second test fails
**Possible Cause**:
- State not properly reset between tests
- Tree not properly cleared
- Root node not initialized

**Detailed Analysis of test_state_restoration_with_make_unmake**:
```python
state = alphazero_py.GomokuState()
initial_hash = state.zobrist_hash()

tree = mcts_py.MCTSTree(10000)  # New tree
# ... components ...
runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)

for i in range(10):
    # Check state unchanged
    current_hash = state.zobrist_hash()
    assert current_hash == initial_hash  # ← Should pass

    # Run simulation
    runner.run_simulation(state, 0, callback)  # ← Fails on iteration >0
```

**Why it fails**:
1. First iteration: root node (0) is unexpanded
2. run_simulation clones state, selects to leaf, expands, backups
3. Root node now has children
4. Second iteration: root node is expanded, has children
5. Selects a child, applies the move
6. BUT: The move was stored for a different game state!

**Wait...** This shouldn't happen because:
- Each iteration passes the SAME `state` (initial state)
- run_simulation() clones it each time
- Selection should produce consistent tree

Unless... the tree from iteration 1 has moves that were legal then, but when we traverse in iteration 2 with a fresh clone, the moves are still legal.

Actually, I think I see it now:

**The actual issue**: The move 224 in Gomoku (15×15 = 225 actions) is:
- 224 = last move in action space (position 14, 14)
- "P1 for player 2" suggests wrong player

This could happen if:
1. Expansion created children with moves for player 1
2. But when we traverse, we're at a state where it's player 2's turn
3. This means the state and tree are out of sync

**How can state/tree get out of sync?**
- The root is always at index 0
- First simulation expands root (creates children for player 1)
- Second simulation starts at root again
- BUT: The tree still has children from first simulation
- Those children are for player 1's moves
- When we select, we're at root (player 1's turn)
- We select a child (player 1's move)
- We apply it... this should work

**WAIT**: I think the issue is that after the first simulation, the tree has:
- Root node (player 1's turn)
- Children with player 1's moves
- One child was selected, expanded further

On the second simulation:
- We start at root (player 1's turn) ✓
- We select a child (player 1's move) ✓
- We apply the move → now it's player 2's turn ✓
- We continue selection
- Child node has children (from first simulation's expansion)
- Those children are for player 2's moves ✓
- We select one and apply it → now it's player 1's turn ✓
- ... this should all work!

**THE REAL BUG**: I bet it's the node flags!

Looking at expand_node (simulation_runner.cpp:274-276):
```cpp
NodeFlags child_flags;
child_flags.set_current_player(state.getCurrentPlayer() == 1 ? 1 : 0);  // Line 275
tree_.set_flags(child_idx, child_flags);
```

This stores the CURRENT player in the child node flags. But this is the player whose turn it is AFTER the move is applied!

So if we're at root (player 1's turn), we expand:
1. Get legal moves for player 1
2. Create children with those moves
3. Set child_flags.current_player = state.getCurrentPlayer()
4. BUT: state.getCurrentPlayer() is BEFORE the move is applied!

Actually, looking more carefully:
```cpp
// For non-terminal leaves:
// ... get legal moves, policy, etc ...
// Initialize children
for (...) {
    // ... set prior, move, parent, etc ...

    // Initialize flags with current player
    NodeFlags child_flags;
    child_flags.set_current_player(state.getCurrentPlayer() == 1 ? 1 : 0);
    tree_.set_flags(child_idx, child_flags);
}
```

The `state` here is at the LEAF position. The children represent positions AFTER applying the moves. So `state.getCurrentPlayer()` is the player at the leaf, and the children will be for the NEXT player.

But we're setting child flags to the CURRENT player (at leaf), not the next player!

This might be a bug in node flag initialization. The child node should store the player whose turn it is at that CHILD position, not at the parent.

---

## Critical Questions

### Q1: Is illegal move filtering working correctly?
**A1**: YES ✅ - Only legal moves are stored in tree nodes

### Q2: Is the make/unmake implementation correct?
**A2**: YES ✅ - Uses same moves from tree as old approach

### Q3: Why do tests fail?
**A3**: Tests are testing the WRONG class (SimulationRunner, not ContinuousSimulationRunner)

### Q4: Is there a deeper bug in MCTS?
**A4**: POSSIBLY - Node flags might store wrong player, causing illegal move errors

### Q5: Will this be resolved by next steps?
**A5**: NO - Requires immediate debugging and proper test creation

---

## Action Items (CRITICAL - Do NOT proceed without fixing)

### Immediate (Blocking):

1. **Create Correct Tests** 🔴
   - Test ContinuousSimulationRunner::run_continuous(), NOT SimulationRunner
   - Use async inference queue
   - Validate make/unmake actually runs

2. **Debug Node Flags** 🔴
   - Verify current_player stored in child nodes is correct
   - Check if this causes illegal move errors
   - Fix if needed

3. **Validate Legal Move Invariant** 🔴
   - Add assertion: "move from tree is legal for current state"
   - This will catch state/tree desync immediately
   - Add to both old and new code paths

### Test Fix Example:

```python
def test_continuous_runner_make_unmake():
    """Test ContinuousSimulationRunner with make/unmake pattern."""
    state = alphazero_py.GomokuState()

    # Create async infrastructure
    tree = mcts_py.MCTSTree(10000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

    # Create CONTINUOUS runner (not base runner)
    runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)

    # Create async queue
    queue = mcts_py.AsyncInferenceQueue(capacity=4096)

    # Create batch coordinator
    callback_fn = lambda state: ([1.0/225]*225, 0.0)
    callback = mcts_py.PyBatchInferenceCallback(callback_fn)
    coordinator = mcts_py.BatchInferenceCoordinator(queue, callback)

    # Run continuous simulations (THIS tests make/unmake)
    completed = runner.run_continuous(state, 0, queue, num_simulations=100)

    assert completed == 100
    # Validate state is restored (would fail if make/unmake broken)
    assert state.zobrist_hash() == initial_hash
```

### Debug Assertion to Add:

```cpp
// In select_leaf_with_make_unmake, before make_move:
#ifndef NDEBUG
    // Validate move is legal for current state
    std::vector<int> current_legal = current_state.getLegalMoves();
    bool move_is_legal = std::find(current_legal.begin(), current_legal.end(),
                                   static_cast<int>(move_index)) != current_legal.end();
    if (!move_is_legal) {
        throw std::runtime_error(
            "CRITICAL BUG: Tree contains illegal move " + std::to_string(move_index) +
            " for current state. Tree/state desync detected!");
    }
#endif
```

---

## Recommendation

**DO NOT PROCEED** to next steps until:

1. ✅ Create proper ContinuousSimulationRunner tests
2. ✅ Add debug assertions for legal move invariant
3. ✅ Validate node flag initialization is correct
4. ✅ Confirm tests pass with make/unmake code

**This is a BLOCKING issue** - illegal moves break MCTS correctness guarantees.

**Estimated Time to Fix**: 2-4 hours
**Risk if Ignored**: HIGH - Could corrupt tree, produce incorrect search results
