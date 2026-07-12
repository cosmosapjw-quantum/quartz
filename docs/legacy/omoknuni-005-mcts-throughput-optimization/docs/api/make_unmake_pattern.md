# make/unmake Pattern for Zero-Copy MCTS

**Document Version**: 1.0
**Date**: 2025-10-16
**Authority**: T024b make/unmake API Design (spec/004)

---

## Overview

The make/unmake pattern enables zero-copy MCTS by applying moves in-place rather than cloning game states. This eliminates the state cloning bottleneck (418μs, 86.6% of execution time) with fast in-place updates (~15ns per move).

**Performance Impact**:
- State cloning: 418μs → State reconstruction: ~600ns (make/unmake)
- Speedup: 697× faster per simulation
- Expected throughput: 15,000-25,000 sims/sec (5-10× improvement)

---

## API Contract

### make_move(move) → undo_token

Applies move to current state **in-place** and returns an undo token.

**Signature**:
```cpp
virtual uint64_t make_move(uint16_t move) = 0;
```

**Behavior**:
- Modifies board state in-place (place/remove piece)
- Updates current player (flip)
- Updates game result (if terminal)
- Updates move count and other metadata
- Updates Zobrist hash incrementally (XOR)
- Returns 64-bit undo token encoding modified fields

**Performance Requirements**:
- Target: ≤15ns per move
- NO heap allocations
- Thread-local usage only (NOT thread-safe)
- Deterministic (same state + move → same result)

**Example (Gomoku)**:
```cpp
uint64_t GomokuState::make_move(uint16_t move) {
    // Save state for undo token
    uint8_t prev_row = last_move_row_;
    uint8_t prev_col = last_move_col_;
    uint8_t prev_result = static_cast<uint8_t>(game_result_);
    uint8_t prev_count = move_count_;

    // Apply move
    board_[move] = current_player_;
    last_move_row_ = move / 15;
    last_move_col_ = move % 15;
    move_count_++;

    // Update Zobrist hash (incremental XOR)
    zobrist_hash_ ^= zobrist_table_[move][current_player_];

    // Check terminal condition
    if (check_five_in_row(move)) {
        game_result_ = (current_player_ == 1) ?
            GameResult::WIN_PLAYER1 : GameResult::WIN_PLAYER2;
    } else if (move_count_ == 225) {
        game_result_ = GameResult::DRAW;
    }

    // Flip player
    current_player_ = 3 - current_player_;

    // Pack undo token (64 bits)
    return (static_cast<uint64_t>(prev_row) << 8) |
           (static_cast<uint64_t>(prev_col) << 0) |
           (static_cast<uint64_t>(prev_result) << 16) |
           (static_cast<uint64_t>(prev_count) << 24);
}
```

---

### unmake_move(move, undo_token)

Reverses move applied by make_move using the undo token.

**Signature**:
```cpp
virtual void unmake_move(uint16_t move, uint64_t undo_token) = 0;
```

**Behavior**:
- Restores exact state before make_move
- Reverses all modifications (board, player, result, metadata)
- Restores Zobrist hash incrementally (same XOR)
- Must be called in LIFO order (reverse of make_move calls)

**Performance Requirements**:
- Target: ≤15ns per move (symmetric with make_move)
- NO heap allocations
- Thread-local usage only (NOT thread-safe)
- Bit-exact restoration (no state drift)

**Example (Gomoku)**:
```cpp
void GomokuState::unmake_move(uint16_t move, uint64_t undo_token) {
    // Extract fields from undo token
    uint8_t prev_row = (undo_token >> 8) & 0xFF;
    uint8_t prev_col = (undo_token >> 0) & 0xFF;
    uint8_t prev_result = (undo_token >> 16) & 0xFF;
    uint8_t prev_count = (undo_token >> 24) & 0xFF;

    // Restore board (remove stone)
    board_[move] = EMPTY;

    // Restore metadata
    last_move_row_ = prev_row;
    last_move_col_ = prev_col;
    game_result_ = static_cast<GameResult>(prev_result);
    move_count_ = prev_count;

    // Restore player (flip back)
    current_player_ = 3 - current_player_;

    // Restore Zobrist hash (XOR same as make_move)
    zobrist_hash_ ^= zobrist_table_[move][current_player_];
}
```

---

## Undo Token Design

Game-specific 64-bit encoding of modified fields.

### Gomoku (Minimal Undo)

**Fields to restore**:
- last_move_row (8 bits, 0-14)
- last_move_col (8 bits, 0-14)
- game_result (8 bits, 0-3)
- move_count (8 bits, 0-225)

**Bit layout**:
```
[63:32] unused (32 bits)
[31:24] move_count (8 bits)
[23:16] game_result (8 bits)
[15:8]  last_move_row (8 bits)
[7:0]   last_move_col (8 bits)
```

**Packing**:
```cpp
return (last_move_row << 8) | last_move_col |
       (game_result << 16) | (move_count << 24);
```

**Unpacking**:
```cpp
uint8_t prev_row = (undo_token >> 8) & 0xFF;
uint8_t prev_col = (undo_token >> 0) & 0xFF;
uint8_t prev_result = (undo_token >> 16) & 0xFF;
uint8_t prev_count = (undo_token >> 24) & 0xFF;
```

---

### Chess (Complex Undo)

**Fields to restore**:
- captured_piece (4 bits, 0-11: 6 pieces × 2 colors)
- castling_rights (4 bits, KQkq flags)
- en_passant_square (8 bits, 0-63 or 255=none)
- halfmove_clock (8 bits, 0-100)
- game_result (8 bits, 0-3)

**Bit layout**:
```
[63:32] unused (32 bits)
[31:24] game_result (8 bits)
[23:16] halfmove_clock (8 bits)
[15:8]  en_passant_square (8 bits)
[7:4]   castling_rights (4 bits)
[3:0]   captured_piece (4 bits)
```

**Packing**:
```cpp
return (captured_piece << 0) |
       (castling_rights << 4) |
       (en_passant_square << 8) |
       (halfmove_clock << 16) |
       (game_result << 24);
```

**Unpacking**:
```cpp
uint8_t captured = (undo_token >> 0) & 0x0F;
uint8_t castling = (undo_token >> 4) & 0x0F;
uint8_t ep_square = (undo_token >> 8) & 0xFF;
uint8_t halfmove = (undo_token >> 16) & 0xFF;
uint8_t result = (undo_token >> 24) & 0xFF;
```

**Captured Piece Encoding**:
```cpp
enum Piece {
    PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5,
    NONE=15
};
uint8_t encoded = (piece << 1) | color;  // 4 bits
```

---

### Go (Moderate Undo)

**Fields to restore**:
- ko_position (16 bits, 0-360 or 65535=none)
- captured_stones_mask (32 bits, bitboard of captured stones)
- passes (8 bits, 0-2 consecutive passes)
- game_result (8 bits, 0-3)

**Bit layout**:
```
[63:56] game_result (8 bits)
[55:48] passes (8 bits)
[47:16] captured_stones_mask (32 bits)
[15:0]  ko_position (16 bits)
```

**Packing**:
```cpp
return (ko_position << 0) |
       (static_cast<uint64_t>(captured_stones_mask) << 16) |
       (passes << 48) |
       (game_result << 56);
```

**Unpacking**:
```cpp
uint16_t ko = (undo_token >> 0) & 0xFFFF;
uint32_t captured = (undo_token >> 16) & 0xFFFFFFFF;
uint8_t pass_count = (undo_token >> 48) & 0xFF;
uint8_t result = (undo_token >> 56) & 0xFF;
```

**Captured Stones Restoration**:
```cpp
// Restore captured stones (up to 32 stones)
for (int i = 0; i < 32; ++i) {
    if (captured & (1u << i)) {
        board_[captured_indices_[i]] = opponent_color;
    }
}
```

---

## Thread Safety

### NOT Thread-Safe

make/unmake modify state in-place → NOT thread-safe.

**Requirements**:
- Each thread MUST maintain its own IGameState instance
- Recommended: `thread_local std::unique_ptr<IGameState> worker_state`
- No synchronization overhead (true thread-local)

**Example**:
```cpp
// Worker thread context
thread_local std::unique_ptr<IGameState> worker_state;

void worker_init(GameType game_type) {
    // Initialize worker state (once per thread)
    switch (game_type) {
        case GameType::GOMOKU:
            worker_state = std::make_unique<GomokuState>();
            break;
        case GameType::CHESS:
            worker_state = std::make_unique<ChessState>();
            break;
        case GameType::GO:
            worker_state = std::make_unique<GoState>();
            break;
    }
}

void run_simulation(TinyNode* root, const std::vector<TinyNode*>& path) {
    // Reset to root state
    worker_state->copyFrom(*root_state);

    // Traverse path
    std::vector<uint64_t> undo_stack;
    for (TinyNode* node : path) {
        uint64_t undo = worker_state->make_move(node->move);
        undo_stack.push_back(undo);
    }

    // Inference
    auto [policy, value] = infer(*worker_state);

    // Unwind
    for (int i = path.size() - 1; i >= 0; --i) {
        worker_state->unmake_move(path[i]->move, undo_stack[i]);
    }
}
```

---

## Usage Pattern

### MCTS Path Traversal

```cpp
// Thread-local state (one per worker)
thread_local std::unique_ptr<IGameState> worker_state;

// Traverse path in MCTS tree
std::vector<uint64_t> undo_stack;
undo_stack.reserve(path.size());  // Avoid reallocations

for (TinyNode* node : path) {
    uint64_t undo = worker_state->make_move(node->move);
    undo_stack.push_back(undo);
}

// Neural network inference at leaf
auto [policy, value] = infer(*worker_state);

// Unwind path (LIFO order)
for (int i = path.size() - 1; i >= 0; --i) {
    worker_state->unmake_move(path[i]->move, undo_stack[i]);
}

// Verify restoration (debug mode)
#ifndef NDEBUG
assert(worker_state->getHash() == root_state->getHash());
#endif
```

### LIFO Order Enforcement

unmake_move MUST be called in reverse order of make_move:

```cpp
// Correct LIFO order
undo1 = state.make_move(move1);
undo2 = state.make_move(move2);
undo3 = state.make_move(move3);

state.unmake_move(move3, undo3);  // Reverse order
state.unmake_move(move2, undo2);
state.unmake_move(move1, undo1);

// Wrong order - undefined behavior
state.unmake_move(move1, undo1);  // ERROR: move2, move3 not undone
state.unmake_move(move2, undo2);
state.unmake_move(move3, undo3);
```

---

## Correctness Validation

### Bit-Exact Restoration

make/unmake must restore bit-exact state:

```cpp
void test_make_unmake_equivalence(IGameState& state, uint16_t move) {
    // Save original state
    uint64_t original_hash = state.getHash();
    int original_player = state.getCurrentPlayer();
    std::string original_string = state.toString();

    // Apply and reverse move
    uint64_t undo = state.make_move(move);
    state.unmake_move(move, undo);

    // Verify bit-exact restoration
    assert(state.getHash() == original_hash);
    assert(state.getCurrentPlayer() == original_player);
    assert(state.toString() == original_string);
}
```

### Zobrist Hash Consistency

Zobrist hash must be consistent with incremental updates:

```cpp
void test_zobrist_consistency(IGameState& state, uint16_t move) {
    // Clone state for comparison
    auto cloned = state.clone();

    // Apply move to both
    state.make_move(move);
    cloned->makeMove(move);

    // Zobrist hashes must match
    assert(state.zobrist_hash() == cloned->getHash());
}
```

### Deep Path Validation

Test deep paths (>50 moves) for state drift:

```cpp
void test_deep_path(IGameState& state, const std::vector<uint16_t>& moves) {
    std::vector<uint64_t> undo_stack;
    undo_stack.reserve(moves.size());

    // Apply all moves
    for (uint16_t move : moves) {
        undo_stack.push_back(state.make_move(move));
    }

    // Save intermediate state
    uint64_t intermediate_hash = state.getHash();

    // Unwind all moves
    for (int i = moves.size() - 1; i >= 0; --i) {
        state.unmake_move(moves[i], undo_stack[i]);
    }

    // Verify restoration to root
    assert(state.getHash() == root_hash);

    // Re-apply moves
    for (uint16_t move : moves) {
        state.make_move(move);
    }

    // Verify same intermediate state
    assert(state.getHash() == intermediate_hash);
}
```

---

## Performance Benchmarking

### make/unmake Microbenchmark

```cpp
void benchmark_make_unmake(IGameState& state, uint16_t move, int iterations) {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        uint64_t undo = state.make_move(move);
        state.unmake_move(move, undo);
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
    ).count();

    double ns_per_op = elapsed_ns / (2.0 * iterations);  // make + unmake
    std::cout << "make/unmake: " << ns_per_op << " ns/op\n";

    // Target: <15ns per operation
    assert(ns_per_op < 15.0);
}
```

### Path Traversal Benchmark

```cpp
void benchmark_path_traversal(IGameState& state,
                              const std::vector<uint16_t>& path,
                              int iterations) {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        std::vector<uint64_t> undo_stack;
        undo_stack.reserve(path.size());

        // Apply moves
        for (uint16_t move : path) {
            undo_stack.push_back(state.make_move(move));
        }

        // Unwind moves
        for (int j = path.size() - 1; j >= 0; --j) {
            state.unmake_move(path[j], undo_stack[j]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
    ).count();

    double ns_per_sim = elapsed_ns / static_cast<double>(iterations);
    std::cout << "Path traversal: " << ns_per_sim << " ns/simulation\n";
    std::cout << "Depth: " << path.size() << " moves\n";

    // Target: <600ns for depth=20
    double target_ns = path.size() * 30.0;  // 15ns make + 15ns unmake
    assert(ns_per_sim < target_ns);
}
```

---

## Prior Art

### Stockfish (Chess)

**make/unmake with 64-bit undo tokens**:
```cpp
struct UndoInfo {
    Piece captured;
    CastlingRights castling;
    Square ep_square;
    int halfmove_clock;
    uint64_t zobrist_key;
};

void Position::do_move(Move m, UndoInfo& undo) {
    // Save state
    undo.captured = piece_on(to_sq(m));
    undo.castling = castling_rights();
    // ... apply move
}

void Position::undo_move(Move m, const UndoInfo& undo) {
    // Restore state
    put_piece(undo.captured, to_sq(m));
    set_castling_rights(undo.castling);
    // ... reverse move
}
```

**Performance**: 200M nodes/sec search speed

### KataGo (Go)

**Zero-copy MCTS with thread-local reconstruction**:
```cpp
struct BoardState {
    uint64_t zobrist_hash;
    std::vector<int> captured_stones;
    int ko_position;
    int passes;
};

void Board::make_move(Move m, BoardState& state) {
    state.captured_stones = capture_stones(m);
    state.ko_position = compute_ko(m);
    // ... apply move
}

void Board::unmake_move(Move m, const BoardState& state) {
    restore_stones(state.captured_stones);
    ko_position_ = state.ko_position;
    // ... reverse move
}
```

**Performance**: 80k playouts/sec on GPU

### Leela Zero (Go/Chess)

**AlphaZero-style with make/unmake**:
- Per-thread node arenas
- Thread-local state reconstruction
- Zobrist transposition tables

---

**Document Status**: ✅ Complete (T024b)
**Next**: T024c (Gomoku make/unmake Implementation)
