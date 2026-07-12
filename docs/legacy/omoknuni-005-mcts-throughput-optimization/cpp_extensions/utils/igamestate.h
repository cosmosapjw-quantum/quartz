// include/core/igamestate.h
#ifndef ALPHAZERO_CORE_IGAMESTATE_H
#define ALPHAZERO_CORE_IGAMESTATE_H

#include <vector>
#include <string>
#include <optional>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include "export_macros.h"
#include "illegal_move_exception.h"

namespace alphazero {
namespace core {

// Game types
enum class GameType {
    UNKNOWN,
    CHESS,
    GO,
    GOMOKU
    // Can be extended with more game types
};

// Game results
enum class GameResult {
    ONGOING,
    WIN_PLAYER1,
    WIN_PLAYER2,
    DRAW,
    NO_RESULT  // For Japanese rules: triple ko, quadruple ko, eternal life
};

/**
 * @brief Interface for game state
 * 
 * This interface defines the operations that all game implementations
 * must provide. It's used by the MCTS algorithm to interact with
 * different games in a uniform way.
 */
class ALPHAZERO_API IGameState {
public:
    /**
     * @brief Constructor
     * 
     * @param type Game type
     */
    explicit IGameState(GameType type);

    /**
     * @brief Virtual destructor
     */
    virtual ~IGameState() = default;

    /**
     * @brief Get all legal moves in the current state
     * 
     * @return Vector of legal actions
     */
    virtual std::vector<int> getLegalMoves() const = 0;

    /**
     * @brief Check if a specific move is legal
     * 
     * @param action The action to check
     * @return true if legal, false otherwise
     */
    virtual bool isLegalMove(int action) const = 0;

    /**
     * @brief Execute a move
     * 
     * Updates the game state by applying the specified action.
     * The action is assumed to be legal.
     * 
     * @param action The action to execute
     * @throws std::runtime_error if the action is illegal
     */
    virtual void makeMove(int action) = 0;

    /**
     * @brief Undo the last move
     * 
     * Reverts the game state to what it was before the last move.
     * 
     * @return true if a move was undone, false if no moves to undo
     */
    virtual bool undoMove() = 0;

    /**
     * @brief Check if the game state is terminal
     * 
     * A terminal state is one where the game is over (win, loss, draw).
     * 
     * @return true if terminal, false otherwise
     */
    virtual bool isTerminal() const = 0;

    /**
     * @brief Get the result of the game
     * 
     * Should return ONGOING if the game is not terminal.
     * 
     * @return Game result
     */
    virtual GameResult getGameResult() const = 0;

    /**
     * @brief Get the current player
     * 
     * @return Current player (1 for player 1, 2 for player 2)
     */
    virtual int getCurrentPlayer() const = 0;

    /**
     * @brief Get the board size
     * 
     * @return Board size (typically width/height)
     */
    virtual int getBoardSize() const = 0;

    /**
     * @brief Get the action space size
     * 
     * The total number of possible actions, including illegal ones.
     * 
     * @return Size of the action space
     */
    virtual int getActionSpaceSize() const = 0;

    /**
     * @brief Get tensor representation for neural network
     * 
     * Creates a 3D tensor representation of the game state suitable
     * for input to a neural network. The format is:
     * [num_planes][height][width]
     * 
     * @return 3D tensor with basic features
     */
    virtual std::vector<std::vector<std::vector<float>>> getTensorRepresentation() const = 0;

    /**
     * @brief Get basic tensor representation (AlphaZero format)
     * 
     * Creates an 18-channel representation following the standard AlphaZero format:
     * - Channel 0: Current player's pieces
     * - Channel 1: Opponent's pieces
     * - Channels 2-17: Previous 8 board states for each player (16 channels)
     * - All channels normalized with current player indicator
     * 
     * @return 3D tensor with 18 channels
     */
    virtual std::vector<std::vector<std::vector<float>>> getBasicTensorRepresentation() const = 0;

    /**
     * @brief Get enhanced tensor representation with additional features
     *
     * Similar to getTensorRepresentation, but with additional planes
     * for features like move history, legal moves, game-specific data.
     *
     * @return 3D tensor with enhanced features
     */
    virtual std::vector<std::vector<std::vector<float>>> getEnhancedTensorRepresentation() const = 0;

    /**
     * @brief Extract features directly to pre-allocated buffer (T007e)
     *
     * Writes tensor representation to pre-allocated buffer in row-major layout.
     * This is a zero-copy optimization for batch inference, avoiding intermediate
     * vector allocations.
     *
     * Layout: [num_planes, height, width] with contiguous memory
     * Buffer size: num_planes * height * width * sizeof(float)
     *
     * @param buffer Float buffer of sufficient size (caller must allocate)
     *
     * Requirements:
     * - No heap allocations during extraction
     * - Thread-safe (read-only state access)
     * - Performance target: <10μs per state
     * - Deterministic output (same state → same features)
     *
     * Example for Gomoku (36 planes, 15×15):
     *   float buffer[36 * 15 * 15];
     *   state->extract_features_to_buffer(buffer);
     *   // buffer[0..224] = plane 0 (current player stones)
     *   // buffer[225..449] = plane 1 (opponent stones)
     *   // etc.
     */
    virtual void extract_features_to_buffer(float* buffer) const = 0;

    /**
     * @brief Get number of feature planes for this game type
     *
     * Returns the number of feature planes in the enhanced tensor representation.
     * Used to calculate buffer size for extract_features_to_buffer().
     *
     * @return Number of feature planes (Gomoku=36, Chess=30, Go=25)
     */
    virtual int get_num_feature_planes() const = 0;

    /**
     * @brief Get hash for transposition table
     *
     * Returns a Zobrist hash of the current state for efficient
     * lookups in transposition tables.
     *
     * @return 64-bit hash
     */
    virtual uint64_t getHash() const = 0;

    /**
     * @brief Clone the current state
     * 
     * Creates a deep copy of the current game state.
     * 
     * @return Unique pointer to a new copy
     */
    virtual std::unique_ptr<IGameState> clone() const = 0;
    
    /**
     * @brief Batch clone the current state multiple times
     * 
     * Creates multiple deep copies of the current game state efficiently.
     * This is optimized for cloning the same state multiple times,
     * sharing setup costs and potentially using vectorized operations.
     * 
     * @param count Number of clones to create
     * @return Vector of unique pointers to new copies
     */
    virtual std::vector<std::unique_ptr<IGameState>> batchClone(int count) const {
        // Default implementation - derived classes should override for better performance
        std::vector<std::unique_ptr<IGameState>> clones;
        clones.reserve(count);
        for (int i = 0; i < count; ++i) {
            clones.push_back(clone());
        }
        return clones;
    }
    
    /**
     * @brief Copy the state from another game state instance (T018a - State Pooling)
     *
     * Copies all relevant fields from the source state to this state.
     * This allows reusing existing state objects from a pool rather than
     * allocating new ones, eliminating heap allocations during MCTS simulations.
     *
     * **Performance Requirements** (profiling-validated):
     * - Target: 20μs per copy (vs 418μs for clone())
     * - NO heap allocations during copy operation
     * - Use memcpy() for fixed-size arrays (optimal cache efficiency)
     * - Shallow copy for primitive fields
     * - Bit-exact semantic equivalence with clone()
     *
     * **Thread Safety**:
     * - Read-only access to 'source' (thread-safe)
     * - Write access to 'this' (caller must ensure exclusivity)
     * - No shared mutable state accessed
     *
     * **Implementation Guidelines**:
     * ```cpp
     * void ConcreteState::copyFrom(const IGameState& other) {
     *     auto& src = static_cast<const ConcreteState&>(other);
     *
     *     // Fast memcpy for fixed-size arrays
     *     memcpy(board_, src.board_, sizeof(board_));
     *     memcpy(history_, src.history_, sizeof(history_));
     *
     *     // Primitive field copies
     *     current_player_ = src.current_player_;
     *     move_count_ = src.move_count_;
     *     // ... other primitive fields
     * }
     * ```
     *
     * @param source The source state to copy from
     * @throws std::runtime_error if the game types don't match
     *
     * @see docs/api/state_pooling.md for complete API contract and examples
     */
    virtual void copyFrom(const IGameState& source) = 0;

    /**
     * @brief Get estimated size of this state in bytes (T018a - Pool Sizing)
     *
     * Returns the total memory footprint of this game state instance,
     * including all dynamically allocated memory. Used for thread-local
     * state pool sizing and memory tracking.
     *
     * **Calculation**:
     * ```cpp
     * return sizeof(*this) +
     *        move_history_.capacity() * sizeof(int) +
     *        other_dynamic_allocations;
     * ```
     *
     * @return Estimated memory usage in bytes
     *
     * @see ThreadLocalStatePool for pool sizing based on this value
     */
    virtual size_t estimated_size_bytes() const {
        // Default implementation - derived classes should override
        return estimateMemoryUsage();
    }

    //
    // T024b: make/unmake API for Zero-Copy MCTS Architecture
    //

    /**
     * @brief Apply move in-place and return undo token (T024b - Zero-Copy MCTS)
     *
     * Applies the specified move to the current state in-place, modifying
     * the board, current player, and other game-specific fields. Returns
     * an opaque 64-bit undo token that can be used with unmake_move() to
     * restore the exact state before this move.
     *
     * **Zero-Copy Pattern**:
     * This replaces the clone() + makeMove() pattern used in T018, eliminating
     * state cloning overhead (418μs) with fast in-place updates (~15ns).
     *
     * **Performance Requirements**:
     * - Target: ≤15ns per move (vs 418μs for clone())
     * - NO heap allocations during move application
     * - Thread-local usage only (NOT thread-safe)
     * - Deterministic (same state + move → same result)
     *
     * **Undo Token Design**:
     * Game-specific 64-bit value encoding modified fields:
     *
     * Gomoku (minimal):
     * ```cpp
     * return (last_move_row << 8) | last_move_col |
     *        (game_result << 16) | (move_count << 24);
     * ```
     *
     * Chess (complex):
     * ```cpp
     * return (captured_piece << 0) | (castling_rights << 4) |
     *        (en_passant_square << 8) | (halfmove_clock << 16) |
     *        (game_result << 24);
     * ```
     *
     * Go (moderate):
     * ```cpp
     * return (ko_position << 0) | (captured_stones_mask << 16) |
     *        (passes << 48) | (game_result << 56);
     * ```
     *
     * **Usage Pattern**:
     * ```cpp
     * // Traverse MCTS path
     * std::vector<uint64_t> undo_stack;
     * for (TinyNode* node : path) {
     *     uint64_t undo = state.make_move(node->move);
     *     undo_stack.push_back(undo);
     * }
     *
     * // Inference at leaf
     * auto [policy, value] = infer(state);
     *
     * // Unwind path (LIFO)
     * for (int i = path.size() - 1; i >= 0; --i) {
     *     state.unmake_move(path[i]->move, undo_stack[i]);
     * }
     * ```
     *
     * **Thread Safety**:
     * - NOT thread-safe (modifies state in-place)
     * - Each thread MUST use its own IGameState instance
     * - Recommended: thread_local IGameState per worker
     *
     * **Zobrist Hash Update**:
     * Implementations should update Zobrist hash incrementally during make_move
     * via XOR operations for efficient transposition table lookups.
     *
     * @param move The move to apply (game-specific encoding)
     * @return Opaque 64-bit undo token for unmake_move()
     * @throws std::runtime_error if move is illegal
     *
     * @see unmake_move() to reverse this move
     * @see docs/api/make_unmake_pattern.md for complete API contract
     */
    virtual uint64_t make_move(uint16_t move) = 0;

    /**
     * @brief Reverse move using undo token (T024b - Zero-Copy MCTS)
     *
     * Restores the exact game state before the corresponding make_move() call
     * by using the undo token returned by make_move(). This undoes all changes
     * to the board, current player, game result, and other game-specific fields.
     *
     * **Performance Requirements**:
     * - Target: ≤15ns per move (symmetric with make_move)
     * - NO heap allocations during move reversal
     * - Thread-local usage only (NOT thread-safe)
     * - Bit-exact restoration (no state drift)
     *
     * **Correctness Requirements**:
     * ```cpp
     * // Save original state
     * auto original_hash = state.getHash();
     * auto original_player = state.getCurrentPlayer();
     *
     * // Apply and reverse move
     * uint64_t undo = state.make_move(move);
     * state.unmake_move(move, undo);
     *
     * // Verify bit-exact restoration
     * assert(state.getHash() == original_hash);
     * assert(state.getCurrentPlayer() == original_player);
     * ```
     *
     * **Implementation Guidelines**:
     * ```cpp
     * void ConcreteState::unmake_move(uint16_t move, uint64_t undo_token) {
     *     // Extract fields from undo token
     *     uint8_t last_row = (undo_token >> 8) & 0xFF;
     *     uint8_t last_col = (undo_token >> 0) & 0xFF;
     *     uint8_t prev_result = (undo_token >> 16) & 0xFF;
     *     uint8_t prev_count = (undo_token >> 24) & 0xFF;
     *
     *     // Restore board (remove stone)
     *     board_[move] = EMPTY;
     *
     *     // Restore metadata
     *     last_move_row_ = last_row;
     *     last_move_col_ = last_col;
     *     game_result_ = prev_result;
     *     move_count_ = prev_count;
     *
     *     // Restore player (flip)
     *     current_player_ = 3 - current_player_;
     *
     *     // Restore Zobrist hash (XOR with move)
     *     zobrist_hash_ ^= zobrist_table_[move][current_player_];
     * }
     * ```
     *
     * **Thread Safety**:
     * - NOT thread-safe (modifies state in-place)
     * - Must be called on same thread as make_move()
     * - Undo token is opaque (only valid for same state + move)
     *
     * **LIFO Order**:
     * unmake_move() must be called in reverse order of make_move() calls:
     * ```cpp
     * // Correct LIFO order
     * undo1 = state.make_move(move1);
     * undo2 = state.make_move(move2);
     * state.unmake_move(move2, undo2);  // Reverse order
     * state.unmake_move(move1, undo1);
     *
     * // Wrong order - undefined behavior
     * state.unmake_move(move1, undo1);  // ERROR
     * state.unmake_move(move2, undo2);
     * ```
     *
     * @param move The move to reverse (must match make_move call)
     * @param undo_token Undo token returned by make_move()
     *
     * @see make_move() to apply a move
     * @see docs/api/make_unmake_pattern.md for complete API contract
     */
    virtual void unmake_move(uint16_t move, uint64_t undo_token) = 0;

    /**
     * @brief Get Zobrist hash for transposition tables (T024b)
     *
     * Returns an incremental Zobrist hash of the current game state.
     * The hash should be updated during make_move/unmake_move via XOR
     * operations for O(1) incremental updates.
     *
     * **Zobrist Hashing**:
     * ```cpp
     * // Initialization (once per game type)
     * zobrist_table_[position][piece_type] = random_64bit();
     *
     * // Incremental update in make_move
     * zobrist_hash_ ^= zobrist_table_[move][piece];
     *
     * // Incremental restore in unmake_move
     * zobrist_hash_ ^= zobrist_table_[move][piece];  // Same XOR
     * ```
     *
     * **Properties**:
     * - Same position → same hash (deterministic)
     * - Different positions → different hash (high probability)
     * - O(1) incremental updates (XOR only)
     * - Used for transposition table lookups in DAG tree
     *
     * @return 64-bit Zobrist hash of current position
     *
     * @see getHash() for existing hash implementation (may differ)
     */
    virtual uint64_t zobrist_hash() const {
        // Default implementation delegates to getHash()
        // Game-specific implementations should override if Zobrist hashing
        // is not the primary hash function
        return getHash();
    }

    /**
     * @brief Convert action to string representation
     * 
     * Useful for human-readable move notation (e.g., "e2e4" in chess).
     * 
     * @param action The action to convert
     * @return String representation
     */
    virtual std::string actionToString(int action) const = 0;

    /**
     * @brief Convert string representation to action
     * 
     * The inverse of actionToString.
     * 
     * @param moveStr String representation
     * @return Optional action (nullopt if invalid)
     */
    virtual std::optional<int> stringToAction(const std::string& moveStr) const = 0;

    /**
     * @brief Get string representation of the state
     * 
     * Creates a human-readable representation of the entire game state.
     * 
     * @return String representation
     */
    virtual std::string toString() const = 0;

    /**
     * @brief Check equality with another game state
     * 
     * Two states are equal if they represent the same game position.
     * 
     * @param other The other game state
     * @return true if equal, false otherwise
     */
    virtual bool equals(const IGameState& other) const = 0;

    /**
     * @brief Get the history of moves
     * 
     * Returns the sequence of actions that led to the current state.
     * 
     * @return Vector of actions
     */
    virtual std::vector<int> getMoveHistory() const = 0;

    /**
     * @brief Validate the game state for consistency
     *
     * Checks if the current state is valid according to game rules.
     *
     * @return true if valid, false otherwise
     */
    virtual bool validate() const = 0;

    /**
     * @brief Estimate memory usage of this game state
     *
     * Provides a rough estimate of the memory used by this game state
     * in bytes. Used for memory tracking and debugging.
     *
     * @return Estimated memory usage in bytes
     */
    virtual size_t estimateMemoryUsage() const {
        // Default implementation - derived classes should override
        // for more accurate accounting
        return sizeof(*this) +
               getMoveHistory().capacity() * sizeof(int);
    }
    
    /**
     * @brief Get bitboard representation of the game state
     * 
     * Returns the internal bitboard representation for efficient operations.
     * Each player's pieces are represented as a vector of uint64_t words.
     * 
     * @return Vector of bitboards for each player. Each inner vector contains uint64_t words.
     *         For 2-player games: [player1_bitboards, player2_bitboards]
     */
    virtual std::vector<std::vector<uint64_t>> getBitboards() const = 0;

    /**
     * @brief Get the game type
     *
     * @return Game type
     */
    GameType getGameType() const;

protected:
    GameType type_;
};

// Free functions
ALPHAZERO_API std::string gameTypeToString(GameType type);
ALPHAZERO_API GameType stringToGameType(const std::string& str);

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_CORE_IGAMESTATE_H