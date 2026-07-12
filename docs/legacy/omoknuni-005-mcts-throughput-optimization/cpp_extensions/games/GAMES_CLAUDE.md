# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the games module in this repository.

## GOLDEN RULE: This project follows Spec-Driven Development (SDD)

**Always maintain synchronization between code and specifications:**
- Code changes MUST reflect updates in `/specs/001-goal-create-spec/`
- API contracts in `/specs/001-goal-create-spec/contracts/` define the interface
- Data models in `/specs/001-goal-create-spec/data-model.md` define structures
- This CLAUDE.md must stay current with actual implementations

## Games Module Overview

The games module provides high-performance C++ implementations for board games (Chess, Gomoku/Omok, Go) with unified Python bindings. It follows a clean separation between game logic, state management, and rule engines.

**Architecture Principles:**
- **Game-agnostic interface**: All games implement `IGameState` for uniform MCTS integration
- **Performance-first**: In-place move application, bitboard representations, zero-allocation hot paths
- **Memory efficiency**: Structure-of-Arrays patterns, pre-allocated pools, cache-optimized layouts
- **Thread safety**: Immutable game rules, atomic state operations where needed

## Core Architecture

### Inheritance Hierarchy
```
IGameState (abstract base)
├── ChessState (with Chess960 support)
├── GomokuState (with Renju/Omok variants)
└── GoState (with Chinese/Japanese/Korean rule sets)
```

### Module Organization
```
games/
├── core/              # Base interfaces and game registry
│   ├── igamestate.cpp     # Abstract game state base class
│   └── game_export.cpp    # Game factory and registration system
├── games/             # Specific game implementations
│   ├── chess/             # Chess with 960 support
│   │   ├── chess_state.cpp    # Main state and move logic
│   │   ├── chess_rules.cpp    # Legal move generation
│   │   └── chess960.cpp       # Starting position generation
│   ├── gomoku/            # Gomoku/Omok/Renju variations
│   │   ├── gomoku_state.cpp   # Bitboard-based state
│   │   └── gomoku_rules.cpp   # Win condition checking
│   └── go/                # Go with multiple rule sets
│       ├── go_state.cpp       # Board state and captures
│       └── go_rules.cpp       # Territory and scoring
├── utils/             # Shared utilities
│   ├── zobrist_hash.cpp   # Position hashing for transposition tables
│   ├── logger.cpp         # Structured logging
│   └── attack_defense_module.h  # Neural network feature extraction
└── python/            # Python bindings
    └── bindings.cpp       # pybind11 interface for Python integration
```

## Function Glossary

### IGameState Interface (Pure Virtual)
**Core contract that all games must implement:**

- `apply_move_inplace(action)` → void
  - **Purpose**: Apply move directly to current state without copying
  - **Performance**: Zero-allocation, in-place modification for MCTS efficiency
  - **Error handling**: Throws exception for illegal moves

- `get_legal_moves()` → vector<int>
  - **Purpose**: Return all legal actions in current position
  - **Caching**: Results cached until state changes (dirty flag pattern)
  - **Format**: Integer action indices matching neural network output

- `is_terminal()` → bool
  - **Purpose**: Check if game has reached terminal state (win/loss/draw)
  - **Optimization**: Cached result with lazy evaluation
  - **MCTS integration**: Used to stop tree expansion

- `get_terminal_value()` → float
  - **Purpose**: Return game outcome from current player's perspective
  - **Range**: [-1.0, 1.0] where 1.0=win, 0.0=draw, -1.0=loss
  - **Precondition**: Only call when is_terminal() returns true

- `extract_features()` → vector<vector<vector<float>>>
  - **Purpose**: Generate neural network input features (C×H×W tensor)
  - **Performance**: Optimized for GPU batch processing
  - **Game-specific**: Different feature planes per game type

- `get_current_player()` → int
  - **Purpose**: Return active player (0 or 1 for two-player games)
  - **MCTS usage**: Critical for value sign flipping during backup

- `copy()` → unique_ptr<IGameState>
  - **Purpose**: Create deep copy for MCTS simulation
  - **Memory**: Uses object pooling where possible
  - **Thread safety**: Each thread gets independent copy

### ChessState Specific Functions

- `initializeChess960Position(position_number)` → void
  - **Purpose**: Set up Chess960 starting position (0-959)
  - **Validation**: Fischer random chess rules for piece placement
  - **State**: Updates castling rights based on rook positions

- `generateLegalMoves(current_player, castling_rights, en_passant)` → vector<ChessMove>
  - **Purpose**: Generate all legal chess moves including special moves
  - **Performance**: Pseudo-legal generation followed by king safety filter
  - **Special cases**: Castling, en passant, pawn promotion

- `isInCheck(player, king_position)` → bool
  - **Purpose**: Determine if player's king is under attack
  - **Algorithm**: Reverse move generation from king position
  - **Optimization**: Early termination on first attack found

- `makeMove(move)` → void
  - **Purpose**: Execute chess move with state updates
  - **Side effects**: Updates castling rights, en passant, move counters
  - **Zobrist**: Incrementally updates position hash

### GomokuState Specific Functions

- `checkWinCondition(player, last_move)` → bool
  - **Purpose**: Detect 5-in-a-row win condition after move
  - **Algorithm**: Directional scanning from last move position
  - **Optimization**: Only checks affected lines, not full board

- `applyRenjuRestrictions()` → vector<int>
  - **Purpose**: Filter moves based on Renju tournament rules
  - **Rules**: Forbids certain opening patterns for black player
  - **Usage**: Optional rule variant for competitive play

- `getBitboard(player)` → vector<uint64_t>
  - **Purpose**: Access internal bitboard representation
  - **Performance**: 64-bit words for efficient operations
  - **Thread safety**: Read-only access for evaluation

### GoState Specific Functions

- `playMove(x, y, player)` → bool
  - **Purpose**: Place stone and handle captures/ko
  - **Algorithm**: Flood fill for group connectivity and captures
  - **Ko detection**: Prevents immediate recapture of single stones

- `calculateScore(rule_set)` → float
  - **Purpose**: Compute final game score based on territory
  - **Rule sets**: Chinese (area), Japanese (territory), Korean variants
  - **Komi**: Accounts for first-player advantage compensation

- `isEye(x, y, player)` → bool
  - **Purpose**: Detect if point is an "eye" (surrounded territory)
  - **Algorithm**: Check all adjacent points and diagonals
  - **Usage**: Prevents self-capture in most cases

- `floodFill(x, y, target_color)` → vector<pair<int,int>>
  - **Purpose**: Find connected group of stones
  - **Algorithm**: Breadth-first search on board graph
  - **Performance**: Reuses buffer to avoid allocations

### Utility Functions

- `ZobristHash::getPositionHash()` → uint64_t
  - **Purpose**: Generate unique hash for position
  - **Usage**: Transposition table lookups, repetition detection
  - **Incremental**: Updates efficiently with move/undo operations

- `coordsToAction(row, col)` → int
  - **Purpose**: Convert 2D coordinates to flat action index
  - **Formula**: `row * board_size + col`
  - **Bounds checking**: Validates coordinates are within board

- `actionToCoords(action)` → pair<int,int>
  - **Purpose**: Convert flat action index to 2D coordinates
  - **Formula**: `{action / board_size, action % board_size}`
  - **Usage**: User interface and debugging

## Performance Characteristics

### Memory Layout
- **Bitboard games** (Gomoku): ~64 bytes per state, O(1) operations
- **Array games** (Chess/Go): ~200-500 bytes per state, O(N) for some operations
- **Zobrist hashing**: 8 bytes per position, O(1) updates

### Computational Complexity
- **Legal move generation**: O(N) where N = board positions
- **Move application**: O(1) for most games, O(N) for captures in Go
- **Terminal detection**: O(1) with caching, O(N) without
- **Feature extraction**: O(N×C) where C = feature channels

### Optimization Techniques
- **Dirty flags**: Avoid recalculation of cached values
- **Pre-allocated buffers**: Reuse memory for move generation
- **SIMD operations**: Vectorized bitboard operations where possible
- **Branch prediction**: Hot paths optimized for common cases

## Development Guidelines

### Adding New Games
1. **Inherit from IGameState**: Implement all pure virtual functions
2. **Follow SDD**: Update contracts in `/specs/` before coding
3. **Register game type**: Add to `GameRegistry` in `game_export.cpp`
4. **Add Python bindings**: Expose in `bindings.cpp` with pybind11
5. **Write tests**: Contract tests must pass before implementation

### Performance Requirements
- **No allocations** in `apply_move_inplace()` hot path
- **Cache legal moves** until state changes (dirty flag pattern)
- **Incremental hashing** for position signatures
- **Memory pools** for temporary objects in complex calculations

### Testing Patterns
```cpp
// Contract test pattern
void test_apply_move_inplace() {
    auto state = createTestState();
    auto legal_moves = state->get_legal_moves();

    // Test valid move
    state->apply_move_inplace(legal_moves[0]);
    assert(state->get_current_player() != original_player);

    // Test invalid move throws
    assertThrows([&]() { state->apply_move_inplace(-1); });
}
```

### Memory Management
```cpp
// Use RAII and smart pointers
std::unique_ptr<IGameState> state = GameRegistry::createGame(CHESS);

// Avoid raw pointers in game logic
std::shared_ptr<GameRules> rules = std::make_shared<ChessRules>();

// Use stack allocation for temporary objects
std::array<int, 64> move_buffer;  // Instead of vector for fixed-size data
```

## Integration Points

### MCTS Engine Integration
- Games provide `IGameState` interface for tree search
- Move application is in-place for performance
- Feature extraction optimized for neural network batching
- Terminal detection cached to avoid repeated computation

### Python Binding Integration
- All game states exposed through pybind11 in `bindings.cpp`
- NumPy array conversion for features and legal moves
- Exception translation from C++ to Python
- Memory management handled by pybind11 smart pointer integration

### Neural Network Integration
- Feature tensors formatted as (C, H, W) for PyTorch
- Legal move masking applied before policy normalization
- Value targets adjusted for current player perspective
- Batch processing optimized for GPU inference

## Common Pitfalls

### Performance Traps
- **Unnecessary copying**: Always use references for large objects
- **Allocations in hot paths**: Pre-allocate buffers and reuse
- **Missing dirty flags**: Cache expensive computations
- **String operations**: Use integer IDs instead of string comparisons

### Correctness Issues
- **Move validation**: Always check legality before application
- **Player perspective**: Value signs must flip with player changes
- **State consistency**: Update all derived state (hash, caches) with moves
- **Terminal conditions**: Handle all win/loss/draw cases

### Threading Concerns
- **Shared state**: Game rules can be shared, game states cannot
- **Const correctness**: Mark read-only methods as const
- **Atomic operations**: Use for any shared counters or flags
- **Local copies**: Each MCTS thread needs independent game state copy

---

**Remember**: This games module is the foundation for the entire AlphaZero engine. Changes here affect MCTS performance, neural network training, and game outcome accuracy. Always validate against the specifications and maintain backward compatibility for the Python interface.