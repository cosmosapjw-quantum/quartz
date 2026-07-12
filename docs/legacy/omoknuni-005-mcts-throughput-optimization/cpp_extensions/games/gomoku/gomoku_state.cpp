// File: gomoku_state.cpp
#include "games/gomoku/gomoku_state.h"
#include "games/gomoku/gomoku_rules.h"     // For rules_engine_
#include "illegal_move_exception.h" // For core::IllegalMoveException
// #include "mcts/aggressive_memory_manager.h" // Removed - not needed
// #include "utils/attack_defense_module.h"  // Removed - will be implemented in neural network tasks
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <iostream>  // For debugging (optional, remove in production)
#include <numeric>   // For std::accumulate, std::gcd
#include <algorithm> // For std::fill, std::find
#include <cstring>   // For memset (T007e)


namespace alphazero {
namespace games {
namespace gomoku {

// --- Constructor ---
GomokuState::GomokuState(int board_size, bool use_renju, bool use_omok, int seed, bool use_pro_long_opening)
    : IGameState(core::GameType::GOMOKU),
      board_size_(board_size),
      current_player_(BLACK),
      move_history_(), 
      zobrist_(board_size * board_size, 2, 4), 
      use_renju_(use_renju),
      use_omok_(use_omok),
      use_pro_long_opening_(use_pro_long_opening),
      black_first_stone_(-1),
      valid_moves_dirty_(true),
      cached_winner_(NO_PLAYER),
      winner_check_dirty_(true),
      hash_signature_(0), 
      hash_dirty_(true),
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true),
      last_action_played_(-1) {


    if (board_size_ <= 0) {
        throw std::invalid_argument("Board size must be positive.");
    }
    int total_cells = board_size_ * board_size_;
    num_words_ = (total_cells + 63) / 64;

    player_bitboards_.resize(2, std::vector<uint64_t>(num_words_, 0ULL));

    for (int p = 0; p < 2; ++p) {
        for (int w = 0; w < num_words_; ++w) {
            if (player_bitboards_[p][w] != 0) {
                std::cout << "Error: bitboard not initialized to zero!" << std::endl;
            }
        }
    }

    rules_engine_ = std::make_shared<GomokuRules>(board_size_);
    rules_engine_->setBoardAccessor(
        [this](int p_idx, int act) { return this->is_bit_set(p_idx, act); },
        [this](int act) { return this->is_any_bit_set_for_rules(act); },
        [this](int r, int c) { return this->coords_to_action(r, c); },
        [this](int act) { return this->action_to_coords_pair(act); },
        [this](int r, int c) { return this->in_bounds(r, c); }
    );
    invalidate_caches();
}

// --- Copy Constructor ---
GomokuState::GomokuState(const GomokuState& other)
    : IGameState(core::GameType::GOMOKU), 
      board_size_(other.board_size_),
      current_player_(other.current_player_),
      move_history_(other.move_history_),
      zobrist_(other.zobrist_),
      use_renju_(other.use_renju_),
      use_omok_(other.use_omok_),
      use_pro_long_opening_(other.use_pro_long_opening_),
      black_first_stone_(other.black_first_stone_),
      valid_moves_dirty_(other.valid_moves_dirty_.load()),
      cached_winner_(other.cached_winner_.load()),
      winner_check_dirty_(other.winner_check_dirty_.load()),
      hash_signature_(other.hash_signature_.load()),
      hash_dirty_(other.hash_dirty_.load()),
      // Don't copy cached tensors - force recomputation to avoid memory issues
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true),
      cached_valid_moves_(other.cached_valid_moves_),
      num_words_(other.num_words_),
      player_bitboards_(other.player_bitboards_),
      rules_engine_(std::make_shared<GomokuRules>(other.board_size_)), 
      last_action_played_(other.last_action_played_) {

    rules_engine_->setBoardAccessor(
        [this](int p_idx, int act) { return this->is_bit_set(p_idx, act); },
        [this](int act) { return this->is_any_bit_set_for_rules(act); },
        [this](int r, int c) { return this->coords_to_action(r, c); },
        [this](int act) { return this->action_to_coords_pair(act); },
        [this](int r, int c) { return this->in_bounds(r, c); }
    );
}

// --- Public IGameState Interface Methods ---

std::vector<int> GomokuState::getLegalMoves() const {
    if (isTerminal()) return {};
    if (valid_moves_dirty_) {
        refresh_valid_moves_cache();
    }
    return std::vector<int>(cached_valid_moves_.begin(), cached_valid_moves_.end());
}

bool GomokuState::isLegalMove(int action) const {
    if (action < 0 || action >= getActionSpaceSize()) return false;
    if (isTerminal()) return false; 

    return is_move_valid_internal(action, true);
}

void GomokuState::makeMove(int action) {
    if (!isLegalMove(action)) {
        throw core::IllegalMoveException("Attempted illegal move: " + actionToString(action) +
                                         " for player " + std::to_string(current_player_), action);
    }
    make_move_internal(action, current_player_);
}

bool GomokuState::undoMove() {
    if (move_history_.empty()) {
        return false;
    }
    int last_action_to_undo = move_history_.back();
    int player_who_made_that_move = (current_player_ == BLACK) ? WHITE : BLACK;

    undo_last_move_internal(last_action_to_undo, player_who_made_that_move);
    return true;
}

bool GomokuState::isTerminal() const {
    // CRITICAL FIX: Fast path for empty board to prevent MCTS stalling
    if (move_history_.empty()) {
        return false; // Empty board cannot be terminal
    }
    
    
    // MCTS OPTIMIZATION: For very early game (≤4 moves), skip expensive winner check
    // Cannot have a winner in Gomoku with 4 or fewer stones
    int total_stones = count_total_stones();
    if (total_stones <= 8) {
        return false;
    }
    
    // LOCK-FREE: Check winner cache with atomic operations
    if (winner_check_dirty_.load(std::memory_order_acquire)) {
        refresh_winner_cache(); // Lock-free refresh
    }
    
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    bool has_winner = (current_winner != NO_PLAYER);
    
    bool is_stale = is_stalemate(); // Lock-free stalemate check
    
    return has_winner || is_stale;
}

core::GameResult GomokuState::getGameResult() const {
    if (winner_check_dirty_.load(std::memory_order_acquire)) { 
        refresh_winner_cache();
    }

    int current_winner = cached_winner_.load(std::memory_order_acquire);
    if (current_winner == BLACK) return core::GameResult::WIN_PLAYER1;
    if (current_winner == WHITE) return core::GameResult::WIN_PLAYER2;
    
    if (is_stalemate()) return core::GameResult::DRAW;
    
    return core::GameResult::ONGOING;
}

int GomokuState::getCurrentPlayer() const {
    return current_player_;
}

int GomokuState::getBoardSize() const {
    return board_size_;
}

int GomokuState::getActionSpaceSize() const {
    return board_size_ * board_size_;
}

std::vector<std::vector<std::vector<float>>> GomokuState::getTensorRepresentation() const {
    // ENHANCED GOMOKU REPRESENTATION: 36-plane feature representation for stronger play
    // Planes 0-1: Stone planes (current player, opponent)
    // Planes 2-17: History planes (16 total - 8 pairs for each player)
    // Plane 18: Player indicator (black/white asymmetry)
    // Plane 19: Is_renju rule variation
    // Plane 20: Is_omok rule variation
    // Plane 21: Allowed moves mask
    // Planes 22-23: Immediate five threats (current/opponent)
    // Planes 24-25: Four threats (current/opponent)
    // Planes 26-27: Open three threats (current/opponent)
    // Planes 28-35: Run-length analysis (8 planes - 4 directions × 2 sides)

    auto tensor = std::vector<std::vector<std::vector<float>>>(
        36, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));

    int p_idx_current = current_player_ - 1;
    int p_idx_opponent = 1 - p_idx_current;

    // Planes 0-1: Stone planes
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (is_bit_set(p_idx_current, action)) {
                tensor[0][r][c] = 1.0f;
            } else if (is_bit_set(p_idx_opponent, action)) {
                tensor[1][r][c] = 1.0f;
            }
        }
    }

    // Planes 2-17: Enhanced history planes (8 pairs for each player)
    int history_len = move_history_.size();
    std::vector<int> current_player_moves;
    std::vector<int> opponent_player_moves;

    // Separate moves by player (considering current turn)
    for (int k = 0; k < history_len; ++k) {
        int move_action = move_history_[history_len - 1 - k];
        if (k % 2 == 0) {
            // Most recent move was by opponent (since current player is about to move)
            if (opponent_player_moves.size() < 8) {
                opponent_player_moves.push_back(move_action);
            }
        } else {
            // This move was by current player
            if (current_player_moves.size() < 8) {
                current_player_moves.push_back(move_action);
            }
        }
    }

    // Fill current player history planes (2, 4, 6, 8, 10, 12, 14, 16)
    for (size_t i = 0; i < current_player_moves.size(); ++i) {
        auto [r, c] = action_to_coords_pair(current_player_moves[i]);
        if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
            tensor[2 + i * 2][r][c] = 1.0f;
        }
    }

    // Fill opponent player history planes (3, 5, 7, 9, 11, 13, 15, 17)
    for (size_t i = 0; i < opponent_player_moves.size(); ++i) {
        auto [r, c] = action_to_coords_pair(opponent_player_moves[i]);
        if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
            tensor[3 + i * 2][r][c] = 1.0f;
        }
    }

    // Plane 18: Player indicator (1.0 for black/first player, 0.0 for white/second player)
    float player_indicator = (current_player_ == BLACK) ? 1.0f : 0.0f;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            tensor[18][r][c] = player_indicator;
        }
    }

    // Plane 19: Is_renju rule variation
    float renju_indicator = use_renju_ ? 1.0f : 0.0f;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            tensor[19][r][c] = renju_indicator;
        }
    }

    // Plane 20: Is_omok rule variation
    float omok_indicator = use_omok_ ? 1.0f : 0.0f;
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            tensor[20][r][c] = omok_indicator;
        }
    }

    // Plane 21: Allowed moves mask
    computeAllowedMovesMask(tensor[21]);

    // Planes 22-27: Threat detection planes
    computeThreatPlanes(tensor, 22);

    // Planes 28-35: Run-length analysis planes
    computeRunLengthPlanes(tensor, 28);

    return tensor;
}

std::vector<std::vector<std::vector<float>>> GomokuState::getBasicTensorRepresentation() const {
    // Basic tensor representation (AlphaZero standard format)
    // Channel 0: Current player's stones  
    // Channel 1: Opponent player's stones
    // Channel 2: Player indicator (all 1s if current player is BLACK/player1, all 0s if WHITE/player2)
    // Channels 3-18: Previous 8 moves for each player (16 channels)
    // Total: 19 channels (standard AlphaZero format)
    
    try {
        const int num_feature_planes = 19; // Standard AlphaZero representation
        
        // Create fresh tensor without pooling to avoid memory retention
        auto tensor = std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, std::vector<std::vector<float>>(
                board_size_, std::vector<float>(board_size_, 0.0f)));

        // Channels 0-1: Always consistent player mapping
        // Channel 0: Player 1 (BLACK) stones  
        // Channel 1: Player 2 (WHITE) stones
        // This ensures consistent tensor representation regardless of whose turn it is
        int p_idx_player1 = 0;  // Player 1 (BLACK) bitboard index
        int p_idx_player2 = 1;  // Player 2 (WHITE) bitboard index
        
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                if (is_bit_set(p_idx_player1, action)) {
                    tensor[0][r][c] = 1.0f; // Player 1 (BLACK) stones
                } else if (is_bit_set(p_idx_player2, action)) {
                    tensor[1][r][c] = 1.0f; // Player 2 (WHITE) stones
                }
                // Empty squares remain 0.0f in both channels
            }
        }

        // Channel 2: Player indicator (all 1s for BLACK/player1, all 0s for WHITE/player2)
        if (current_player_ == BLACK) {
            for (int r = 0; r < board_size_; ++r) {
                for (int c = 0; c < board_size_; ++c) {
                    tensor[2][r][c] = 1.0f;
                }
            }
        }
        // For WHITE (player 2), the channel remains all 0s

        // Channels 3-18: Move history (8 pairs)
        int history_len = move_history_.size();
        std::vector<int> player1_moves;
        std::vector<int> player2_moves;

        for(int k = history_len - 1; k >= 0; --k) {
            int move_action = move_history_[k];
            if (k % 2 == 0) { // Black's move (player 1)
                if (player1_moves.size() < 8) {
                    player1_moves.push_back(move_action);
                }
            } else { // White's move (player 2)
                if (player2_moves.size() < 8) {
                    player2_moves.push_back(move_action);
                }
            }
        }

        // Fill history channels starting from channel 3
        const int num_history_pairs = 8;
        for(size_t i=0; i < player1_moves.size(); ++i) {
            auto coords = action_to_coords_pair(player1_moves[i]);
            int r = coords.first;
            int c = coords.second;
            if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
                tensor[3 + i*2][r][c] = 1.0f;
            }
        }

        for(size_t i=0; i < player2_moves.size(); ++i) {
            auto coords = action_to_coords_pair(player2_moves[i]);
            int r = coords.first;
            int c = coords.second;
            if (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
                tensor[4 + i*2][r][c] = 1.0f;
            }
        }
        
        // No attack/defense planes - basic representation ends here
        return tensor;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in getBasicTensorRepresentation: " << e.what() << std::endl;
        
        // Return a default tensor with the correct dimensions (19 channels)
        const int num_feature_planes = 19; // Standard representation
        
        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, 
            std::vector<std::vector<float>>(
                board_size_, 
                std::vector<float>(board_size_, 0.0f)
            )
        );
    } catch (...) {
        std::cerr << "Unknown exception in getBasicTensorRepresentation" << std::endl;
        
        // Return a default tensor with the correct dimensions (19 channels)
        const int num_feature_planes = 19; // Standard representation
        
        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, 
            std::vector<std::vector<float>>(
                board_size_, 
                std::vector<float>(board_size_, 0.0f)
            )
        );
    }
}

std::vector<std::vector<std::vector<float>>> GomokuState::getEnhancedTensorRepresentation() const {
    // CRITICAL FIX: Don't cache tensors to prevent memory accumulation
    
    try {
        // Enhanced tensor format (exact 36-plane specification):
        // Plane 0: Current player's stones
        // Plane 1: Opponent player's stones
        // Plane 2: Empty cells
        // Plane 3: Player indicator
        // Planes 4-10: Last 7 moves for current player (7 planes)
        // Planes 11-17: Last 7 moves for opponent player (7 planes)
        // Plane 18: Freestyle rule
        // Plane 19: Renju rule
        // Plane 20: Omok rule
        // Plane 21: Allowed moves mask
        // Plane 22: Immediate five for current player
        // Plane 23: Immediate five for opponent player
        // Plane 24: Four threat for current player
        // Plane 25: Four threat for opponent player
        // Plane 26: Open three for current player
        // Plane 27: Open three for opponent player
        // Planes 28-31: Run-length features for current player (4 directions)
        // Planes 32-35: Run-length features for opponent player (4 directions)
        const int num_feature_planes = 36; // Exact specification
        
        // Create fresh tensor without pooling to avoid memory retention
        auto tensor = std::vector<std::vector<std::vector<float>>>(
            num_feature_planes, std::vector<std::vector<float>>(
                board_size_, std::vector<float>(board_size_, 0.0f)));

        int p_idx_current = current_player_ - 1;  // 0 for BLACK, 1 for WHITE
        int p_idx_opponent = 1 - p_idx_current;   // 1 for BLACK, 0 for WHITE

        // Plane 0: Current player's stones
        // Plane 1: Opponent player's stones
        // Plane 2: Empty cells
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                if (is_bit_set(p_idx_current, action)) {
                    tensor[0][r][c] = 1.0f; // Current player's stones
                } else if (is_bit_set(p_idx_opponent, action)) {
                    tensor[1][r][c] = 1.0f; // Opponent player's stones
                } else {
                    tensor[2][r][c] = 1.0f; // Empty cells
                }
            }
        }

        // Plane 3: Player indicator (1.0 for BLACK, 0.0 for WHITE)
        float player_indicator = (current_player_ == BLACK) ? 1.0f : 0.0f;
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                tensor[3][r][c] = player_indicator;
            }
        }

        // Planes 4-10: Last 7 moves for current player
        // Planes 11-17: Last 7 moves for opponent player
        int history_size = std::min(7, static_cast<int>(move_history_.size()));
        for (int h = 0; h < history_size; ++h) {
            int move_action = move_history_[move_history_.size() - 1 - h];  // Most recent first
            auto [move_r, move_c] = action_to_coords_pair(move_action);

            // Determine which player made this move
            int moves_back = h;
            int player_who_moved;
            if (moves_back % 2 == 0) {
                // Even number of moves back = same parity as current player
                player_who_moved = current_player_;
            } else {
                // Odd number of moves back = opposite parity
                player_who_moved = (current_player_ == BLACK) ? WHITE : BLACK;
            }

            int history_player_idx = player_who_moved - 1;  // 0-based

            // Place the move in the appropriate history channel
            if (history_player_idx == p_idx_current) {
                tensor[4 + h][move_r][move_c] = 1.0f;  // Current player's history
            } else {
                tensor[11 + h][move_r][move_c] = 1.0f; // Opponent's history
            }
        }

        // Planes 18-20: Rule indicators (all positions same value)
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                tensor[18][r][c] = (!use_renju_ && !use_omok_) ? 1.0f : 0.0f; // Freestyle
                tensor[19][r][c] = use_renju_ ? 1.0f : 0.0f;                   // Renju
                tensor[20][r][c] = use_omok_ ? 1.0f : 0.0f;                    // Omok
            }
        }

        // Plane 21: Allowed moves mask
        std::vector<int> legal_moves = getLegalMoves();
        for (int action : legal_moves) {
            auto [r, c] = action_to_coords_pair(action);
            tensor[21][r][c] = 1.0f;
        }

        // Set up board accessor for rule engine
        auto board_accessor = [this](int player_idx, int action) -> bool {
            return this->is_bit_set(player_idx, action);
        };

        // Plane 22: Immediate five for current player
        // Plane 23: Immediate five for opponent player
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                if (!is_bit_set(0, action) && !is_bit_set(1, action)) {
                    // Check if placing current player's stone here creates five in a row
                    if (rules_engine_->is_five_in_a_row(action, current_player_, true)) {
                        tensor[22][r][c] = 1.0f;
                    }
                    // Check if placing opponent's stone here creates five in a row
                    if (rules_engine_->is_five_in_a_row(action, 3 - current_player_, true)) {
                        tensor[23][r][c] = 1.0f;
                    }
                }
            }
        }

        // Plane 24: Four threat for current player
        // Plane 25: Four threat for opponent player
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                if (!is_bit_set(0, action) && !is_bit_set(1, action)) {
                    // Check for four threats (moves that create a four that threatens to become five)
                    if (createsFourThreat(action, p_idx_current)) {
                        tensor[24][r][c] = 1.0f;
                    }
                    if (createsFourThreat(action, p_idx_opponent)) {
                        tensor[25][r][c] = 1.0f;
                    }
                }
            }
        }

        // Plane 26: Open three for current player
        // Plane 27: Open three for opponent player
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                if (!is_bit_set(0, action) && !is_bit_set(1, action)) {
                    if (use_omok_) {
                        // Use Omok definition of open three
                        if (createsOmokOpenThree(action, p_idx_current)) {
                            tensor[26][r][c] = 1.0f;
                        }
                        if (createsOmokOpenThree(action, p_idx_opponent)) {
                            tensor[27][r][c] = 1.0f;
                        }
                    } else if (use_renju_) {
                        // Use Renju definition of open three
                        if (createsRenjuOpenThree(action, p_idx_current)) {
                            tensor[26][r][c] = 1.0f;
                        }
                        if (createsRenjuOpenThree(action, p_idx_opponent)) {
                            tensor[27][r][c] = 1.0f;
                        }
                    } else {
                        // Freestyle - simpler open three definition
                        if (createsFreestyleOpenThree(action, p_idx_current)) {
                            tensor[26][r][c] = 1.0f;
                        }
                        if (createsFreestyleOpenThree(action, p_idx_opponent)) {
                            tensor[27][r][c] = 1.0f;
                        }
                    }
                }
            }
        }

        // Planes 28-31: Run-length features for current player (4 directions)
        // Planes 32-35: Run-length features for opponent player (4 directions)
        const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}}; // horizontal, vertical, diagonal, anti-diagonal

        for (int dir = 0; dir < 4; ++dir) {
            for (int r = 0; r < board_size_; ++r) {
                for (int c = 0; c < board_size_; ++c) {
                    int action = coords_to_action(r, c);

                    // Calculate run-length to five for current player in this direction
                    float current_run_to_five = calculateRunLengthToFive(action, p_idx_current, DIRS[dir][0], DIRS[dir][1]);
                    tensor[28 + dir][r][c] = current_run_to_five;

                    // Calculate run-length to five for opponent player in this direction
                    float opponent_run_to_five = calculateRunLengthToFive(action, p_idx_opponent, DIRS[dir][0], DIRS[dir][1]);
                    tensor[32 + dir][r][c] = opponent_run_to_five;
                }
            }
        }

        return tensor;
    } catch (const std::exception& e) {
        std::cerr << "Exception in getEnhancedTensorRepresentation: " << e.what() << std::endl;

        // Return a default tensor with the correct dimensions (36 channels)
        const int num_feature_planes = 36; // Exact specification

        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes,
            std::vector<std::vector<float>>(
                board_size_,
                std::vector<float>(board_size_, 0.0f)
            )
        );
    } catch (...) {
        std::cerr << "Unknown exception in getEnhancedTensorRepresentation" << std::endl;

        // Return a default tensor with the correct dimensions (36 channels)
        const int num_feature_planes = 36; // Exact specification

        return std::vector<std::vector<std::vector<float>>>(
            num_feature_planes,
            std::vector<std::vector<float>>(
                board_size_,
                std::vector<float>(board_size_, 0.0f)
            )
        );
    }
}


uint64_t GomokuState::getHash() const {
    if (hash_dirty_) {
        hash_signature_ = compute_hash_signature_internal();
        hash_dirty_ = false;
    }
    return hash_signature_;
}

uint64_t GomokuState::zobrist_hash() const {
    // T024c: Return cached Zobrist hash for transposition tables
    // GomokuState already uses Zobrist hashing internally via zobrist_
    return getHash();
}

std::unique_ptr<core::IGameState> GomokuState::clone() const {
    try {
        // Track memory allocation for game state
        size_t state_size = sizeof(GomokuState) + 
                           (2 * player_bitboards_[0].size() * sizeof(uint64_t)) +
                           (move_history_.size() * sizeof(int));
        // TRACK_MEMORY_ALLOC("GameStateClone", state_size); // Removed
        
        // Create a new instance with same parameters - don't do any validation yet
        auto clone_ptr = std::make_unique<GomokuState>(
            board_size_,
            use_renju_,
            use_omok_,
            0, // Using 0 as seed since we're copying existing state
            use_pro_long_opening_
        );
        
        if (!clone_ptr) {
            throw std::runtime_error("Failed to allocate memory for GomokuState clone");
        }
        
        // Copy primitive state variables
        clone_ptr->current_player_ = current_player_;
        clone_ptr->black_first_stone_ = black_first_stone_;
        clone_ptr->last_action_played_ = last_action_played_;

        // Copy move history
        clone_ptr->move_history_ = move_history_; 
        
        // Don't copy cached data to avoid race conditions. Mark caches as dirty instead.
        clone_ptr->valid_moves_dirty_.store(true, std::memory_order_release);  // Force recomputation of valid moves
        clone_ptr->cached_valid_moves_.clear(); // Clear the cache
        clone_ptr->cached_winner_.store(NO_PLAYER, std::memory_order_release);
        clone_ptr->winner_check_dirty_.store(true, std::memory_order_release);  // Force recomputation of winner
        clone_ptr->hash_signature_.store(0, std::memory_order_release);
        clone_ptr->hash_dirty_.store(true, std::memory_order_release);  // Force recomputation of hash 
        
        // Optimized bitboard copying - assume well-formed states for performance
        // Both source and clone should have same board size and properly initialized bitboards
        for (int p = 0; p < 2; ++p) {
            // Direct assignment is faster than std::copy for vectors
            clone_ptr->player_bitboards_[p] = player_bitboards_[p];
        }
        
        // Skip expensive validation during cloning for performance
        // The original state should already be valid, and we're doing a byte-for-byte copy
        // Validation can be enabled for debugging if needed by uncommenting the lines below:
        // if (!clone_ptr->validate()) {
        //     throw std::runtime_error("Cloned GomokuState failed validation");
        // }
        
        
        return clone_ptr;
    } catch (const std::exception& e) {
        // Simple error reporting without complex logging
        std::cerr << "Error in GomokuState::clone(): " << e.what() << std::endl;
        // Free tracked memory on error
        size_t state_size = sizeof(GomokuState) + 
                           (2 * player_bitboards_[0].size() * sizeof(uint64_t)) +
                           (move_history_.size() * sizeof(int));
        // TRACK_MEMORY_FREE("GameStateClone", state_size); // Removed
        throw;
    }
}

std::vector<std::unique_ptr<core::IGameState>> GomokuState::batchClone(int count) const {
    std::vector<std::unique_ptr<core::IGameState>> clones;
    clones.reserve(count);
    
    // Pre-compute shared data that doesn't change across clones
    const size_t bitboard_size = player_bitboards_[0].size();
    
    // Batch allocate to improve memory locality
    for (int i = 0; i < count; ++i) {
        // Create clone without validation for speed
        auto clone_ptr = std::make_unique<GomokuState>(
            board_size_,
            use_renju_,
            use_omok_,
            0,
            use_pro_long_opening_
        );
        
        // Copy state data efficiently
        clone_ptr->current_player_ = current_player_;
        clone_ptr->black_first_stone_ = black_first_stone_;
        clone_ptr->last_action_played_ = last_action_played_;
        clone_ptr->move_history_ = move_history_;
        
        // Mark caches as dirty - no need to copy cached data
        clone_ptr->valid_moves_dirty_.store(true, std::memory_order_relaxed);
        clone_ptr->cached_valid_moves_.clear();
        clone_ptr->cached_winner_.store(NO_PLAYER, std::memory_order_relaxed);
        clone_ptr->winner_check_dirty_.store(true, std::memory_order_relaxed);
        clone_ptr->hash_signature_.store(0, std::memory_order_relaxed);
        clone_ptr->hash_dirty_.store(true, std::memory_order_relaxed);
        
        // Fast bitboard copy - use memcpy for better performance
        for (int p = 0; p < 2; ++p) {
            clone_ptr->player_bitboards_[p] = player_bitboards_[p];
        }
        
        clones.push_back(std::move(clone_ptr));
    }
    
    return clones;
}

void GomokuState::copyFrom(const core::IGameState& source) {
    // Ensure source is a GomokuState
    const GomokuState* gomoku_source = dynamic_cast<const GomokuState*>(&source);
    if (!gomoku_source) {
        throw std::runtime_error("Cannot copy from non-GomokuState: incompatible game types");
    }

    // Copy rule configurations
    use_renju_ = gomoku_source->use_renju_;
    use_omok_ = gomoku_source->use_omok_;
    use_pro_long_opening_ = gomoku_source->use_pro_long_opening_;

    // Copy game state
    current_player_ = gomoku_source->current_player_;
    black_first_stone_ = gomoku_source->black_first_stone_;
    last_action_played_ = gomoku_source->last_action_played_;
    move_history_ = gomoku_source->move_history_;
    // NOTE: Don't copy zobrist_ - it's deterministic for board_size and reused

    // Deep copy bitboards using vector assignment (reuses allocation)
    player_bitboards_[0] = gomoku_source->player_bitboards_[0];
    player_bitboards_[1] = gomoku_source->player_bitboards_[1];

    // Mark caches as dirty (use relaxed ordering - copyFrom is single-threaded)
    valid_moves_dirty_.store(true, std::memory_order_relaxed);
    winner_check_dirty_.store(true, std::memory_order_relaxed);
    hash_dirty_.store(true, std::memory_order_relaxed);
    tensor_cache_dirty_.store(true, std::memory_order_relaxed);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_relaxed);

    // Set cache values (don't clear unordered_set - just mark dirty)
    cached_winner_ = NO_PLAYER;
    hash_signature_ = 0;
}

std::string GomokuState::actionToString(int action) const {
    if (action < 0 || action >= getActionSpaceSize()) return "PASS";
    auto [r, c] = action_to_coords_pair(action);
    char col_char = 'A' + c;
    if (board_size_ > 8 && col_char >= 'I') { 
        col_char++;
    }
    return std::string(1, col_char) + std::to_string(board_size_ - r);
}

std::optional<int> GomokuState::stringToAction(const std::string& moveStr) const {
    if (moveStr.empty() || moveStr == "PASS") return -1; 

    char col_char_upper = std::toupper(moveStr[0]);
    int col = col_char_upper - 'A';
    if (board_size_ > 8 && col_char_upper > 'I') { 
        col--;
    }

    if (col < 0 || col >= board_size_) return std::nullopt;

    try {
        std::string row_str = moveStr.substr(1);
        if (row_str.empty()) return std::nullopt;
        int row_num_1_based = std::stoi(row_str);
        if (row_num_1_based <= 0 || row_num_1_based > board_size_) return std::nullopt;
        
        int r_0_based = board_size_ - row_num_1_based; 
        
        if (!in_bounds(r_0_based, col)) return std::nullopt;
        return coords_to_action(r_0_based, col);
    } catch (const std::exception&) {
        return std::nullopt; 
    }
}

std::string GomokuState::toString() const {
    std::stringstream ss;
    ss << "  "; 
    for (int c = 0; c < board_size_; ++c) {
        char col_char = 'A' + c;
        if (board_size_ > 8 && col_char >= 'I') col_char++;
        ss << col_char << " ";
    }
    ss << std::endl;

    for (int r = 0; r < board_size_; ++r) {
        ss << std::setw(2) << (board_size_ - r) << " "; 
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (is_bit_set(0, action)) ss << "X ";      
            else if (is_bit_set(1, action)) ss << "O "; 
            else ss << ". ";                            
        }
        ss << std::setw(2) << (board_size_ - r); 
        ss << std::endl;
    }
    ss << "  "; 
    for (int c = 0; c < board_size_; ++c) {
        char col_char = 'A' + c;
        if (board_size_ > 8 && col_char >= 'I') col_char++;
        ss << col_char << " ";
    }
    ss << std::endl;

    ss << "Player to move: " << (current_player_ == BLACK ? "X (BLACK)" : "O (WHITE)") << std::endl;
    if (!move_history_.empty()) {
        ss << "Last move: " << actionToString(last_action_played_) << std::endl;
    }
    if (isTerminal()) { 
         ss << "Game Over. Result: ";
         core::GameResult res = getGameResult(); 
         if (res == core::GameResult::WIN_PLAYER1) ss << "BLACK (X) wins.";
         else if (res == core::GameResult::WIN_PLAYER2) ss << "WHITE (O) wins.";
         else if (res == core::GameResult::DRAW) ss << "Draw.";
         else ss << "Ongoing (Error in toString terminal check)";
         ss << std::endl;
    }
    return ss.str();
}

bool GomokuState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::GOMOKU) return false;
    const auto* o_state = dynamic_cast<const GomokuState*>(&other);
    if (!o_state) return false; 
    return board_equal_internal(*o_state);
}

std::vector<int> GomokuState::getMoveHistory() const {
    return move_history_;
}

bool GomokuState::validate() const {
    try {
        // First check if board_size_ is valid to prevent segfaults
        if (board_size_ <= 0) {
            std::cerr << "Invalid board size: " << board_size_ << std::endl;
            return false;
        }
        
        // Check if player_bitboards_ has the expected size
        if (player_bitboards_.size() != 2) {
            std::cerr << "Invalid player_bitboards_ size: " << player_bitboards_.size() << std::endl;
            return false;
        }
        
        // Check if current_player_ is valid
        if (current_player_ != BLACK && current_player_ != WHITE) {
            std::cerr << "Invalid current_player_: " << current_player_ << std::endl;
            return false;
        }
        
        // Check if the bitboard word vectors have the expected size
        for (int p = 0; p < 2; p++) {
            if (player_bitboards_[p].size() != num_words_) {
                std::cerr << "Invalid player_bitboards_[" << p << "] size: " 
                          << player_bitboards_[p].size() << " (expected " << num_words_ << ")" << std::endl;
                return false;
            }
        }
        
        // Count stones to verify game state is valid
        int black_stones = 0;
        int white_stones = 0;
        
        for (int i = 0; i < getActionSpaceSize(); ++i) {
            if (i < 0 || i >= board_size_ * board_size_) {
                std::cerr << "Action space index out of range: " << i << std::endl;
                return false;
            }
            
            if (is_bit_set(0, i)) black_stones++; 
            if (is_bit_set(1, i)) white_stones++; 
            
            // Check that no position has both black and white stones
            if (is_bit_set(0, i) && is_bit_set(1, i)) {
                std::cerr << "Position " << i << " has both black and white stones" << std::endl;
                return false;
            }
        }
        
        // Check the stone count is valid based on the current player
        if (current_player_ == BLACK) {
            if (black_stones != white_stones) {
                std::cerr << "Invalid stone count for BLACK to move: black=" << black_stones 
                         << ", white=" << white_stones << std::endl;
                return false;
            }
        } else { // current_player_ == WHITE
            if (black_stones != white_stones + 1) {
                std::cerr << "Invalid stone count for WHITE to move: black=" << black_stones 
                         << ", white=" << white_stones << std::endl;
                return false;
            }
        }
        
        // Check if move history is consistent with stone count
        if (move_history_.size() != black_stones + white_stones) {
            std::cerr << "Move history size (" << move_history_.size() 
                     << ") inconsistent with stone count (" << (black_stones + white_stones) << ")" << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in GomokuState::validate(): " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception in GomokuState::validate()" << std::endl;
        return false;
    }
}


// --- Testing Specific Methods ---
void GomokuState::setStoneForTesting(int r, int c, int player) {
    if (!in_bounds(r,c)) return;
    int action = coords_to_action(r,c);
    
    clear_bit(0, action);
    clear_bit(1, action);

    if (player == BLACK) {
        set_bit(0, action);
    } else if (player == WHITE) {
        set_bit(1, action);
    }
    invalidate_caches(); 
}

void GomokuState::setCurrentPlayerForTesting(int player) {
    if (player == BLACK || player == WHITE) {
        current_player_ = player;
        invalidate_caches();
    }
}

void GomokuState::clearBoardForTesting() {
    for(int p_idx=0; p_idx<2; ++p_idx) {
        std::fill(player_bitboards_[p_idx].begin(), player_bitboards_[p_idx].end(), 0ULL);
    }
    move_history_.clear();
    current_player_ = BLACK; 
    black_first_stone_ = -1;
    last_action_played_ = -1;
    invalidate_caches();
}


// --- Private Helper Methods ---

bool GomokuState::is_bit_set(int player_idx_0_based, int action) const noexcept {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) return false;
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) return false;
    return (player_bitboards_[player_idx_0_based][word_idx] & (1ULL << bit_idx)) != 0;
}

void GomokuState::set_bit(int player_idx_0_based, int action) {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) {
        throw std::out_of_range("set_bit: Invalid player_idx or action.");
    }
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) {
         throw std::out_of_range("set_bit: word_idx out of bounds.");
    }
    player_bitboards_[player_idx_0_based][word_idx] |= (1ULL << bit_idx);
}

void GomokuState::clear_bit(int player_idx_0_based, int action) noexcept {
    if (player_idx_0_based < 0 || player_idx_0_based >= 2 ||
        action < 0 || action >= getActionSpaceSize()) return;
    int word_idx = action / 64;
    int bit_idx = action % 64;
    if (word_idx >= static_cast<int>(player_bitboards_[player_idx_0_based].size())) return;
    player_bitboards_[player_idx_0_based][word_idx] &= ~(1ULL << bit_idx);
}

std::pair<int, int> GomokuState::action_to_coords_pair(int action) const noexcept {
    if (board_size_ == 0) return {-1,-1}; 
    return {action / board_size_, action % board_size_};
}

int GomokuState::coords_to_action(int r, int c) const noexcept {
    return r * board_size_ + c;
}

bool GomokuState::in_bounds(int r, int c) const noexcept {
    return r >= 0 && r < board_size_ && c >= 0 && c < board_size_;
}

int GomokuState::count_total_stones() const noexcept {
    int count = 0;
    for (int p_idx = 0; p_idx < 2; ++p_idx) {
        for (uint64_t word : player_bitboards_[p_idx]) {
            #if defined(__GNUC__) || defined(__clang__)
                count += __builtin_popcountll(word);
            #else
                uint64_t temp_word = word;
                while (temp_word > 0) {
                    temp_word &= (temp_word - 1);
                    count++;
                }
            #endif
        }
    }
    return count;
}

void GomokuState::refresh_winner_cache() const {
    // LOCK-FREE: Check if refresh is needed using atomic load
    if (!winner_check_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    // Compute winner without modifying cache until the end
    int new_winner = NO_PLAYER; 
    if (last_action_played_ == -1 && !move_history_.empty()) { 
         const_cast<GomokuState*>(this)->last_action_played_ = move_history_.back();
    } else if (move_history_.empty() && last_action_played_ != -1) { 
        const_cast<GomokuState*>(this)->last_action_played_ = -1;
    }
    
    int player_who_made_last_move = NO_PLAYER;
    if (last_action_played_ != -1) { 
        player_who_made_last_move = (current_player_ == BLACK) ? WHITE : BLACK;
    }

    if (player_who_made_last_move != NO_PLAYER) {
        // Create a board accessor
        auto board_accessor = [this](int p_idx, int act){ 
            return this->is_bit_set(p_idx, act);
        };

        if (player_who_made_last_move == BLACK) {
            // Black must win with exactly 5 in Renju, any length >=5 in Standard/Omok
            bool allow_overline_black = !use_renju_;
            
            // Check for 5-in-a-row (or more if allowed)
            bool win_by_five = rules_engine_->is_five_in_a_row(last_action_played_, BLACK, allow_overline_black);
            
            // For Renju, we need to ensure it's exactly 5, not an overline
            if (use_renju_ && win_by_five) {
                // Get the actual line length
                const int DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
                int max_length = 0;
                for (auto& dir : DIRS) {
                    int length = rules_engine_->get_line_length_at_action(
                        last_action_played_, 0, board_accessor, dir[0], dir[1]);
                    max_length = std::max(max_length, length);
                }
                
                // Only set as winner if exactly 5 stones in a row
                if (max_length == 5) {
                    new_winner = player_who_made_last_move;
                }
            } else if (win_by_five) {
                // Standard or Omok
                new_winner = player_who_made_last_move;
            }
        } else { // WHITE made last move
            // White can win with 5+ in a row in any variant
            bool win_by_five = rules_engine_->is_five_in_a_row(last_action_played_, WHITE, true /*allow_overline*/);
            
            // For debugging the RenjuOverlineWhite test
            // Let's manually check the line length in each direction
            if (use_renju_) { // This is important for Renju variant specifically
                const int DIRS[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
                for (auto& dir : DIRS) {
                    int length = rules_engine_->get_line_length_at_action(
                        last_action_played_, 1, board_accessor, dir[0], dir[1]);
                    if (length >= 5) {
                        win_by_five = true;
                        break;
                    }
                }
            }
            
            if (win_by_five) {
                new_winner = player_who_made_last_move;
            }
        }
    }
    
    // Atomically publish the computed winner and mark cache as clean
    cached_winner_.store(new_winner, std::memory_order_release);
    winner_check_dirty_.store(false, std::memory_order_release);
}

bool GomokuState::is_stalemate() const {
    // Fast path: If there's a winner, it's not stalemate
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    if (current_winner != NO_PLAYER) {
        return false; 
    }

    // Stalemate only occurs when board is completely full with no winner
    int total_stones = count_total_stones();
    return (total_stones >= getActionSpaceSize());
}


void GomokuState::refresh_valid_moves_cache() const {
    // LOCK-FREE: Check if refresh is needed using atomic load
    if (!valid_moves_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    refresh_valid_moves_cache_internal();
}

void GomokuState::refresh_valid_moves_cache_internal() const {
    // Double-check pattern - another thread might have refreshed it
    if (!valid_moves_dirty_.load(std::memory_order_acquire)) {
        return; // Cache is already fresh
    }
    
    cached_valid_moves_.clear();
    
    // Check if we have a winner (no valid moves)
    int current_winner = cached_winner_.load(std::memory_order_acquire);
    bool winner_dirty = winner_check_dirty_.load(std::memory_order_acquire);
    
    if (current_winner != NO_PLAYER && !winner_dirty) {
        // Game has winner, no valid moves
        valid_moves_dirty_.store(false, std::memory_order_release);
        return;
    }

    // Compute all valid moves
    int total_actions = getActionSpaceSize();
    for (int action = 0; action < total_actions; ++action) {
        if (is_move_valid_internal(action, true)) { 
            cached_valid_moves_.insert(action);
        }
    }
    
    valid_moves_dirty_.store(false, std::memory_order_release);
}

bool GomokuState::is_move_valid_internal(int action, bool check_occupation) const {
    if (check_occupation) {
        if (is_occupied(action)) return false;
    }
    
    if (use_pro_long_opening_ && current_player_ == BLACK &&
        !is_pro_long_opening_move_valid(action, count_total_stones())) {
        return false;
    }

    if (current_player_ == BLACK) {
        if (use_renju_ && rules_engine_->is_black_renju_forbidden(action)) {
            return false;
        }
        if (use_omok_ && rules_engine_->is_black_omok_forbidden(action)) {
            return false;
        }
    }
    return true;
}


uint64_t GomokuState::compute_hash_signature_internal() const {
    uint64_t h = 0;
    for (int p_idx = 0; p_idx < 2; ++p_idx) {
        for (int action = 0; action < getActionSpaceSize(); ++action) {
            if (is_bit_set(p_idx, action)) {
                h ^= zobrist_.getPieceHash(p_idx, action);
            }
        }
    }
    h ^= zobrist_.getPlayerHash(current_player_ - 1); 
    return h;
}

bool GomokuState::board_equal_internal(const GomokuState& other) const {
    if (board_size_ != other.board_size_ || current_player_ != other.current_player_ ||
        use_renju_ != other.use_renju_ || use_omok_ != other.use_omok_ ||
        use_pro_long_opening_ != other.use_pro_long_opening_ ||
        player_bitboards_ != other.player_bitboards_ ) { 
        return false;
    }
    return true;
}

void GomokuState::make_move_internal(int action, int player_to_move) {
    set_bit(player_to_move - 1, action); 
    last_action_played_ = action;
    move_history_.push_back(action);

    if (player_to_move == BLACK && black_first_stone_ == -1) {
        black_first_stone_ = action;
    }
    current_player_ = (player_to_move == BLACK) ? WHITE : BLACK;
    invalidate_caches(); 
}

void GomokuState::undo_last_move_internal(int last_action_undone, int player_who_made_last_action) {
    clear_bit(player_who_made_last_action - 1, last_action_undone); 
    move_history_.pop_back();

    if (player_who_made_last_action == BLACK && black_first_stone_ == last_action_undone) {
        black_first_stone_ = -1;
        if (!move_history_.empty()) { 
            for(size_t i=0; i < move_history_.size(); ++i) {
                // Crude way to determine player of history move: assume Black starts (player 1)
                // Player of move_history_[i] is (i % 2 == 0) ? BLACK : WHITE
                if ( (i % 2) == 0 ) { 
                    black_first_stone_ = move_history_[i];
                    break;
                }
            }
        }
    }

    current_player_ = player_who_made_last_action; 
    last_action_played_ = move_history_.empty() ? -1 : move_history_.back();
    invalidate_caches();
}

void GomokuState::invalidate_caches() {
    valid_moves_dirty_.store(true, std::memory_order_release);
    winner_check_dirty_.store(true, std::memory_order_release);
    hash_dirty_.store(true, std::memory_order_release);

    // PERFORMANCE FIX: Return old tensors before invalidating caches
    clearTensorCache();

    // PERFORMANCE FIX: Invalidate tensor caches when game state changes
    tensor_cache_dirty_.store(true, std::memory_order_release);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_release);
}

//
// T024c: Gomoku make/unmake Implementation for Zero-Copy MCTS
//

uint64_t GomokuState::make_move(uint16_t move) {
    // Validate move (debug mode only for performance)
    #ifndef NDEBUG
    if (!isLegalMove(move)) {
        throw core::IllegalMoveException("Attempted illegal move: " + actionToString(move) +
                                         " for player " + std::to_string(current_player_), move);
    }
    #endif

    // Pack undo token (64 bits) - save state before modification
    // Layout: [63:56]=game_result [55:48]=move_count [47:40]=black_first_stone_flag
    //         [39:32]=last_action [31:24]=cached_winner [23:16]=current_player
    //         [15:8]=hash_dirty [7:0]=winner_check_dirty
    uint64_t undo_token = 0;

    // Save game result (8 bits)
    core::GameResult result = getGameResult();
    undo_token |= (static_cast<uint64_t>(result) & 0xFF) << 56;

    // Save move count (8 bits)
    undo_token |= (static_cast<uint64_t>(move_history_.size()) & 0xFF) << 48;

    // Save black_first_stone flag (8 bits: 0=not set, 1=was set, 2=this move sets it)
    uint8_t bfs_flag = 0;
    if (black_first_stone_ != -1) {
        bfs_flag = 1;  // Was already set
    } else if (current_player_ == BLACK) {
        bfs_flag = 2;  // This move will set it
    }
    undo_token |= (static_cast<uint64_t>(bfs_flag) & 0xFF) << 40;

    // Save last_action_played (8 bits - enough for 15×15=225 moves)
    undo_token |= (static_cast<uint64_t>(last_action_played_ + 128) & 0xFF) << 32;

    // Save cached_winner (8 bits)
    undo_token |= (static_cast<uint64_t>(cached_winner_.load(std::memory_order_relaxed)) & 0xFF) << 24;

    // Save current_player (8 bits)
    undo_token |= (static_cast<uint64_t>(current_player_) & 0xFF) << 16;

    // Save cache dirty flags (8 bits each)
    undo_token |= (hash_dirty_.load(std::memory_order_relaxed) ? 1ULL : 0ULL) << 8;
    undo_token |= (winner_check_dirty_.load(std::memory_order_relaxed) ? 1ULL : 0ULL) << 0;

    // Apply move (matching make_move_internal logic)
    set_bit(current_player_ - 1, move);
    last_action_played_ = move;
    move_history_.push_back(move);

    if (current_player_ == BLACK && black_first_stone_ == -1) {
        black_first_stone_ = move;
    }

    // Flip player
    current_player_ = (current_player_ == BLACK) ? WHITE : BLACK;

    // Invalidate caches (mark dirty)
    invalidate_caches();

    return undo_token;
}

void GomokuState::unmake_move(uint16_t move, uint64_t undo_token) {
    // Extract saved state from undo token
    core::GameResult prev_result = static_cast<core::GameResult>((undo_token >> 56) & 0xFF);
    size_t prev_move_count = (undo_token >> 48) & 0xFF;
    uint8_t bfs_flag = (undo_token >> 40) & 0xFF;
    int prev_last_action = static_cast<int>((undo_token >> 32) & 0xFF) - 128;
    int prev_cached_winner = static_cast<int>((undo_token >> 24) & 0xFF);
    int prev_player = static_cast<int>((undo_token >> 16) & 0xFF);
    bool prev_hash_dirty = ((undo_token >> 8) & 0x01) != 0;
    bool prev_winner_dirty = ((undo_token >> 0) & 0x01) != 0;

    // Restore player (flip back)
    current_player_ = prev_player;

    // Remove stone from board
    clear_bit(current_player_ - 1, move);

    // Restore move history
    if (!move_history_.empty() && move_history_.back() == move) {
        move_history_.pop_back();
    }

    // Restore black_first_stone
    if (bfs_flag == 0) {
        black_first_stone_ = -1;  // Was not set
    } else if (bfs_flag == 2) {
        black_first_stone_ = -1;  // This move set it, now unset
    } else if (bfs_flag == 1) {
        // Was already set - need to restore from history
        black_first_stone_ = -1;
        if (!move_history_.empty()) {
            for (size_t i = 0; i < move_history_.size(); ++i) {
                if ((i % 2) == 0) {  // Black's move
                    black_first_stone_ = move_history_[i];
                    break;
                }
            }
        }
    }

    // Restore last_action_played
    last_action_played_ = prev_last_action;

    // Restore cached values (avoid expensive recomputation)
    cached_winner_.store(prev_cached_winner, std::memory_order_relaxed);
    winner_check_dirty_.store(prev_winner_dirty, std::memory_order_relaxed);

    // CRITICAL FIX (T024f-6): Always invalidate hash after unmake
    // The cached hash value is for the POST-move state, not the PRE-move state
    // We must force recomputation to get the correct hash for the restored state
    hash_dirty_.store(true, std::memory_order_relaxed);

    // Mark other caches as dirty (conservative approach)
    valid_moves_dirty_.store(true, std::memory_order_release);
    tensor_cache_dirty_.store(true, std::memory_order_release);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_release);
}

GomokuState::~GomokuState() {
    // Return cached tensors to the pool to prevent memory leaks
    clearTensorCache();
}

void GomokuState::clearTensorCache() const {
    // CRITICAL FIX: No longer caching tensors, so nothing to clear
    // This method is kept for compatibility but does nothing
    cached_tensor_repr_.clear();
    cached_enhanced_tensor_repr_.clear();
}

bool GomokuState::is_occupied(int action) const {
    return is_bit_set(0, action) || is_bit_set(1, action);
}

bool GomokuState::is_any_bit_set_for_rules(int action) const {
    return is_occupied(action);
}


bool GomokuState::is_pro_long_opening_move_valid(int action, int total_stones_on_board) const {
    if (!use_pro_long_opening_) return true; 

    int center_r = board_size_ / 2;
    int center_c = board_size_ / 2;
    int center_action = coords_to_action(center_r, center_c);

    if (current_player_ == BLACK) {
        if (total_stones_on_board == 0) { 
            bool valid = (action == center_action);
            // Debug logging (commented out for production)
            // if (!valid && action == 0) { // Log only once for the first action checked
            //     std::cerr << "Pro-long opening: Black at move 0 must play center " << center_action 
            //              << ", but tried " << action << " (board_size=" << board_size_ << ")" << std::endl;
            // }
            return valid;
        }
        if (total_stones_on_board == 2) { 
            if (black_first_stone_ == -1) {
                return false;
            }
            auto [r1, c1] = action_to_coords_pair(black_first_stone_);
            auto [r2, c2] = action_to_coords_pair(action);
            int chebyshev_dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
            return chebyshev_dist >= 2; 
        }
    }
    return true;
}


// Static member definitions for GPU support
//std::unique_ptr<GomokuGPUAttackDefense> GomokuState::gpu_module_ = nullptr;
std::atomic<bool> GomokuState::gpu_enabled_{false};
std::mutex GomokuState::gpu_mutex_;

void GomokuState::initializeGPU(int board_size) {
#ifdef WITH_TORCH
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    if (torch::cuda::is_available()) {
        torch::Device device(torch::kCUDA);
        // TODO: Create proper GPU module implementation
        //gpu_module_ = std::make_unique<GomokuGPUAttackDefense>(board_size, device);
        gpu_enabled_ = true;
        std::cout << "GomokuState: GPU acceleration initialized for board size " << board_size << std::endl;
    } else {
        std::cout << "GomokuState: GPU not available, using CPU fallback" << std::endl;
        gpu_enabled_ = false;
    }
#else
    gpu_enabled_ = false;
#endif
}

void GomokuState::cleanupGPU() {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    //gpu_module_.reset();
    gpu_enabled_ = false;
}

void GomokuState::setGPUEnabled(bool enabled) {
    gpu_enabled_ = enabled;
}

bool GomokuState::isGPUEnabled() {
    return gpu_enabled_;
}

std::vector<std::vector<std::vector<std::vector<float>>>> 
GomokuState::computeEnhancedTensorBatch(const std::vector<const GomokuState*>& states) {
    if (states.empty()) {
        return {};
    }
    
#ifdef WITH_TORCH
    if (isGPUEnabled()) {
        std::lock_guard<std::mutex> lock(gpu_mutex_);
        
        // Convert states to board tensors
        int batch_size = states.size();
        int board_size = states[0]->board_size_;
        auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        
        // Fill board tensor
        for (int b = 0; b < batch_size; ++b) {
            for (int r = 0; r < board_size; ++r) {
                for (int c = 0; c < board_size; ++c) {
                    int action = states[b]->coords_to_action(r, c);
                    if (states[b]->is_bit_set(0, action)) {  // BLACK
                        board_tensor[b][r][c] = BLACK;
                    } else if (states[b]->is_bit_set(1, action)) {  // WHITE
                        board_tensor[b][r][c] = WHITE;
                    }
                }
            }
        }
        
        // TODO: Compute attack/defense planes on GPU
        // auto [attack_batch, defense_batch] = gpu_module_->compute_planes_gpu(
        //     board_tensor, states[0]->current_player_);
        
        // For now, return empty results
        return {};
        
        /* // TODO: Enable when GPU module is ready
        // Convert results back to CPU and build full tensor representations
        auto attack_cpu = attack_batch.cpu();
        auto defense_cpu = defense_batch.cpu();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> results;
        results.reserve(batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
            // Get base tensor representation
            auto tensor = states[b]->getEnhancedTensorRepresentation();
            
            // Replace attack/defense planes with GPU-computed ones
            for (int r = 0; r < board_size; ++r) {
                for (int c = 0; c < board_size; ++c) {
                    tensor[17][r][c] = attack_cpu[b][r][c].item<float>();
                    tensor[18][r][c] = defense_cpu[b][r][c].item<float>();
                }
            }
            
            results.push_back(std::move(tensor));
        }
        
        return results;
        */
    }
#endif
    
    // CPU fallback
    std::vector<std::vector<std::vector<std::vector<float>>>> results;
    results.reserve(states.size());
    for (const auto* state : states) {
        results.push_back(state->getEnhancedTensorRepresentation());
    }
    return results;
}

std::vector<std::vector<uint64_t>> GomokuState::getBitboards() const {
    // Return a copy of the internal bitboards
    // For Gomoku: [black_bitboards, white_bitboards]
    return player_bitboards_;
}

// Enhanced tensor representation helper implementations

void GomokuState::computeAllowedMovesMask(std::vector<std::vector<float>>& mask_plane) const {
    // Compute legal moves mask - 1.0 for legal moves, 0.0 for occupied or invalid positions
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (action >= 0 && action < getActionSpaceSize() && !is_occupied(action)) {
                // Check if move is truly legal (considering rule variations)
                if (is_move_valid_internal(action, false)) {
                    mask_plane[r][c] = 1.0f;
                }
            }
        }
    }
}

void GomokuState::computeThreatPlanes(std::vector<std::vector<std::vector<float>>>& tensor, int start_plane) const {
    // Planes 22-23: Immediate five threats (current/opponent)
    // Planes 24-25: Four threats (current/opponent)
    // Planes 26-27: Open three threats (current/opponent)

    int p_idx_current = current_player_ - 1;
    int p_idx_opponent = 1 - p_idx_current;

    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            // Skip if position is occupied
            int action = coords_to_action(r, c);
            if (is_occupied(action)) continue;

            // Check immediate five threats
            if (hasImmediateFive(current_player_, r, c)) {
                tensor[start_plane][r][c] = 1.0f;
            }
            if (hasImmediateFive(p_idx_opponent + 1, r, c)) {
                tensor[start_plane + 1][r][c] = 1.0f;
            }

            // Check four threats
            if (hasFourThreat(current_player_, r, c)) {
                tensor[start_plane + 2][r][c] = 1.0f;
            }
            if (hasFourThreat(p_idx_opponent + 1, r, c)) {
                tensor[start_plane + 3][r][c] = 1.0f;
            }

            // Check open three threats
            if (hasOpenThree(current_player_, r, c)) {
                tensor[start_plane + 4][r][c] = 1.0f;
            }
            if (hasOpenThree(p_idx_opponent + 1, r, c)) {
                tensor[start_plane + 5][r][c] = 1.0f;
            }
        }
    }
}

void GomokuState::computeRunLengthPlanes(std::vector<std::vector<std::vector<float>>>& tensor, int start_plane) const {
    // Planes 28-35: Run-length analysis (4 directions × 2 sides)
    // Directions: horizontal, vertical, diagonal /, diagonal \
    // For each direction, compute run-length to five from both sides

    const int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}}; // horizontal, vertical, diag/, diag\

    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            if (is_occupied(action)) continue; // Only analyze empty positions

            for (int dir = 0; dir < 4; ++dir) {
                int dr = directions[dir][0];
                int dc = directions[dir][1];

                // Compute run-length to five in both directions
                int length_positive = getRunLengthToFive(current_player_, r, c, dr, dc);
                int length_negative = getRunLengthToFive(current_player_, r, c, -dr, -dc);

                // Normalize to 0-1 range (5 means can make five in one move)
                float positive_value = std::min(1.0f, length_positive / 5.0f);
                float negative_value = std::min(1.0f, length_negative / 5.0f);

                tensor[start_plane + dir * 2][r][c] = positive_value;
                tensor[start_plane + dir * 2 + 1][r][c] = negative_value;
            }
        }
    }
}

bool GomokuState::hasImmediateFive(int player, int r, int c) const {
    // Check if placing a stone at (r,c) immediately creates five in a row
    const int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};

    for (int dir = 0; dir < 4; ++dir) {
        int dr = directions[dir][0];
        int dc = directions[dir][1];

        // Count consecutive stones in both directions
        int count = 1; // Count the stone we would place
        count += countConsecutive(player, r, c, dr, dc);
        count += countConsecutive(player, r, c, -dr, -dc);

        if (count >= 5) {
            return true;
        }
    }
    return false;
}

bool GomokuState::hasFourThreat(int player, int r, int c) const {
    // Check if placing a stone at (r,c) creates a four that threatens to make five
    const int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};

    for (int dir = 0; dir < 4; ++dir) {
        int dr = directions[dir][0];
        int dc = directions[dir][1];

        int count = 1;
        count += countConsecutive(player, r, c, dr, dc);
        count += countConsecutive(player, r, c, -dr, -dc);

        if (count == 4) {
            // Check if there's space to extend to five
            int nr1 = r + (count) * dr, nc1 = c + (count) * dc;
            int nr2 = r - (count) * dr, nc2 = c - (count) * dc;

            bool can_extend = false;
            if (nr1 >= 0 && nr1 < board_size_ && nc1 >= 0 && nc1 < board_size_) {
                int action1 = coords_to_action(nr1, nc1);
                if (!is_occupied(action1)) can_extend = true;
            }
            if (nr2 >= 0 && nr2 < board_size_ && nc2 >= 0 && nc2 < board_size_) {
                int action2 = coords_to_action(nr2, nc2);
                if (!is_occupied(action2)) can_extend = true;
            }

            if (can_extend) return true;
        }
    }
    return false;
}

bool GomokuState::hasOpenThree(int player, int r, int c) const {
    // Check if placing a stone at (r,c) creates an open three (can become four in two ways)
    const int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};

    for (int dir = 0; dir < 4; ++dir) {
        int dr = directions[dir][0];
        int dc = directions[dir][1];

        int count = 1;
        count += countConsecutive(player, r, c, dr, dc);
        count += countConsecutive(player, r, c, -dr, -dc);

        if (count == 3) {
            // Check if both ends are open (can extend in both directions)
            int nr1 = r + 2 * dr, nc1 = c + 2 * dc;
            int nr2 = r - 2 * dr, nc2 = c - 2 * dc;

            bool open1 = (nr1 >= 0 && nr1 < board_size_ && nc1 >= 0 && nc1 < board_size_ &&
                         !is_occupied(coords_to_action(nr1, nc1)));
            bool open2 = (nr2 >= 0 && nr2 < board_size_ && nc2 >= 0 && nc2 < board_size_ &&
                         !is_occupied(coords_to_action(nr2, nc2)));

            if (open1 && open2) return true;
        }
    }
    return false;
}

int GomokuState::countConsecutive(int player, int r, int c, int dr, int dc) const {
    // Count consecutive stones of the same player in one direction from (r,c)
    int count = 0;
    int nr = r + dr, nc = c + dc;

    while (nr >= 0 && nr < board_size_ && nc >= 0 && nc < board_size_) {
        int action = coords_to_action(nr, nc);
        if (is_bit_set(player - 1, action)) {
            count++;
            nr += dr;
            nc += dc;
        } else {
            break;
        }
    }
    return count;
}

int GomokuState::getRunLengthToFive(int player, int r, int c, int dr, int dc) const {
    // Calculate how many moves needed to create five in a row in this direction
    // This is a simplified version - could be enhanced with more sophisticated analysis

    int consecutive = 0;
    int gaps = 0;
    int total_length = 0;
    int nr = r, nc = c;

    // Scan up to 5 positions in this direction
    for (int i = 0; i < 5; ++i) {
        if (nr < 0 || nr >= board_size_ || nc < 0 || nc >= board_size_) {
            break;
        }

        int action = coords_to_action(nr, nc);
        if (is_bit_set(player - 1, action)) {
            consecutive++;
        } else if (!is_occupied(action)) {
            gaps++;
        } else {
            // Blocked by opponent - can't extend further
            break;
        }

        total_length++;
        nr += dr;
        nc += dc;
    }

    // Simple heuristic: return potential based on consecutive stones and available space
    return consecutive + std::min(gaps, 5 - consecutive);
}

// Helper methods for 36-plane tensor representation
bool GomokuState::createsFourThreat(int action, int player_idx) const {
    auto [r, c] = action_to_coords_pair(action);
    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_) return false;

    // Check if position is empty using bitboard representation
    if (is_bit_set(0, action) || is_bit_set(1, action)) return false;

    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& dir : DIRS) {
        int consecutive = rules_engine_->get_line_length_at_action(action, player_idx,
            [this](int p_idx, int a) { return this->is_bit_set(p_idx, a); }, dir[0], dir[1]);
        if (consecutive == 4) {
            return true; // Creates a four that threatens to become five
        }
    }
    return false;
}

bool GomokuState::createsOmokOpenThree(int action, int player_idx) const {
    auto [r, c] = action_to_coords_pair(action);
    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_) return false;

    // Check if position is empty using bitboard representation
    if (is_bit_set(0, action) || is_bit_set(1, action)) return false;

    // Use the existing Omok open three logic from rules engine
    auto board_accessor = [this, action, player_idx](int p_idx, int a) -> bool {
        if (a == action && p_idx == player_idx) return true;
        return this->is_bit_set(p_idx, a);
    };

    // Use the public Omok forbidden move checker to detect if this creates a forbidden double three
    // If placing the stone creates a forbidden double three, then it creates at least one open three
    return rules_engine_->omok_makes_double_three(action, player_idx, board_accessor);
}

bool GomokuState::createsRenjuOpenThree(int action, int player_idx) const {
    auto [r, c] = action_to_coords_pair(action);
    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_) return false;

    // Check if position is empty using bitboard representation
    if (is_bit_set(0, action) || is_bit_set(1, action)) return false;

    // Create a board accessor that simulates placing the stone at 'action'
    auto board_accessor = [this, action, player_idx](int p_idx, int a) -> bool {
        if (a == action && p_idx == player_idx) return true;
        return this->is_bit_set(p_idx, a);
    };

    // Following user's guidance: "I think renju open three function can be also written with the existing functions for renju as same as omok"
    // Use the public Renju forbidden move checker to detect if this creates a forbidden double three
    // If placing the stone creates a forbidden double three, then it creates at least one open three
    return rules_engine_->is_renju_double_three_forbidden(action, player_idx, board_accessor);
}

bool GomokuState::createsFreestyleOpenThree(int action, int player_idx) const {
    auto [r, c] = action_to_coords_pair(action);
    if (r < 0 || r >= board_size_ || c < 0 || c >= board_size_) return false;

    // Check if position is empty using bitboard representation
    if (is_bit_set(0, action) || is_bit_set(1, action)) return false;

    // Freestyle is simpler - just check for open three patterns
    auto board_accessor = [this, action, player_idx](int p_idx, int a) -> bool {
        if (a == action && p_idx == player_idx) return true;
        return this->is_bit_set(p_idx, a);
    };

    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& dir : DIRS) {
        auto [r, c] = action_to_coords_pair(action);

        // Check patterns like _XXX_ where _ is empty and X is the player's stone
        for (int offset = 0; offset < 3; ++offset) {
            int start_r = r - offset * dir[0];
            int start_c = c - offset * dir[1];

            bool valid_pattern = true;
            // Check the three stones
            for (int i = 0; i < 3; ++i) {
                int check_r = start_r + i * dir[0];
                int check_c = start_c + i * dir[1];
                if (check_r < 0 || check_r >= board_size_ || check_c < 0 || check_c >= board_size_) {
                    valid_pattern = false;
                    break;
                }
                int check_action = coords_to_action(check_r, check_c);
                if (!board_accessor(player_idx, check_action)) {
                    valid_pattern = false;
                    break;
                }
            }

            if (valid_pattern) {
                // Check that both ends are empty
                int before_r = start_r - dir[0];
                int before_c = start_c - dir[1];
                int after_r = start_r + 3 * dir[0];
                int after_c = start_c + 3 * dir[1];

                bool before_empty = (before_r >= 0 && before_r < board_size_ &&
                                   before_c >= 0 && before_c < board_size_ &&
                                   !is_bit_set(0, coords_to_action(before_r, before_c)) &&
                                   !is_bit_set(1, coords_to_action(before_r, before_c)));
                bool after_empty = (after_r >= 0 && after_r < board_size_ &&
                                  after_c >= 0 && after_c < board_size_ &&
                                  !is_bit_set(0, coords_to_action(after_r, after_c)) &&
                                  !is_bit_set(1, coords_to_action(after_r, after_c)));

                if (before_empty && after_empty) {
                    return true;
                }
            }
        }
    }
    return false;
}

float GomokuState::calculateRunLengthToFive(int action, int player_idx, int dr, int dc) const {
    auto [r, c] = action_to_coords_pair(action);

    int consecutive = 0;
    int gaps = 0;
    int total_potential = 0;

    // Count in both directions from the position
    const int MAX_SEARCH = 5; // Maximum distance to search for potential five-in-a-row

    // Count backwards
    for (int i = 1; i <= MAX_SEARCH; ++i) {
        int nr = r - i * dr;
        int nc = c - i * dc;
        if (nr < 0 || nr >= board_size_ || nc < 0 || nc >= board_size_) break;

        int check_action = coords_to_action(nr, nc);
        if (is_bit_set(player_idx, check_action)) {
            consecutive++;
        } else if (!is_bit_set(0, check_action) && !is_bit_set(1, check_action)) {
            gaps++;
            if (gaps > 1) break; // Too many gaps
        } else {
            break; // Blocked by opponent
        }
    }

    // Count forwards
    for (int i = 1; i <= MAX_SEARCH; ++i) {
        int nr = r + i * dr;
        int nc = c + i * dc;
        if (nr < 0 || nr >= board_size_ || nc < 0 || nc >= board_size_) break;

        int check_action = coords_to_action(nr, nc);
        if (is_bit_set(player_idx, check_action)) {
            consecutive++;
        } else if (!is_bit_set(0, check_action) && !is_bit_set(1, check_action)) {
            gaps++;
            if (gaps > 1) break; // Too many gaps
        } else {
            break; // Blocked by opponent
        }
    }

    // Include the current position if we're calculating potential
    if (!is_bit_set(0, action) && !is_bit_set(1, action)) {
        total_potential = consecutive + 1; // +1 for the stone we might place
    } else if (is_bit_set(player_idx, action)) {
        total_potential = consecutive + 1; // +1 for the existing stone
    } else {
        total_potential = 0; // Position occupied by opponent
    }

    // Normalize to [0,1] range where 1.0 means can definitely make five
    return std::min(total_potential / 5.0f, 1.0f);
}

// ============================================================================
// T007e: Direct Feature Extraction to Buffer
// ============================================================================

int GomokuState::get_num_feature_planes() const {
    return 36;  // Enhanced representation: 36 planes for Gomoku
}

void GomokuState::extract_features_to_buffer(float* buffer) const {
    // Zero-copy extraction: Write features directly to buffer
    // Layout: [36 planes][15 rows][15 cols] in row-major order
    // Total: 36 * 15 * 15 = 8100 floats

    const int num_planes = 36;
    const int plane_size = board_size_ * board_size_;  // 225 for 15×15

    // Zero-initialize buffer (faster than per-element initialization)
    std::memset(buffer, 0, num_planes * plane_size * sizeof(float));

    int p_idx_current = current_player_ - 1;   // 0 for BLACK, 1 for WHITE
    int p_idx_opponent = 1 - p_idx_current;     // 1 for BLACK, 0 for WHITE

    // Plane 0: Current player's stones
    // Plane 1: Opponent player's stones
    // Plane 2: Empty cells
    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            int offset = r * board_size_ + c;

            if (is_bit_set(p_idx_current, action)) {
                buffer[0 * plane_size + offset] = 1.0f;  // Current player
            } else if (is_bit_set(p_idx_opponent, action)) {
                buffer[1 * plane_size + offset] = 1.0f;  // Opponent
            } else {
                buffer[2 * plane_size + offset] = 1.0f;  // Empty
            }
        }
    }

    // Plane 3: Player indicator (1.0 for BLACK, 0.0 for WHITE)
    float player_indicator = (current_player_ == BLACK) ? 1.0f : 0.0f;
    float* plane3 = buffer + 3 * plane_size;
    for (int i = 0; i < plane_size; ++i) {
        plane3[i] = player_indicator;
    }

    // Planes 4-10: Last 7 moves for current player
    // Planes 11-17: Last 7 moves for opponent player
    int history_size = std::min(7, static_cast<int>(move_history_.size()));
    for (int h = 0; h < history_size; ++h) {
        int move_action = move_history_[move_history_.size() - 1 - h];
        auto [move_r, move_c] = action_to_coords_pair(move_action);
        int offset = move_r * board_size_ + move_c;

        // Determine which player made this move
        int moves_back = h;
        int player_who_moved;
        if (moves_back % 2 == 0) {
            player_who_moved = current_player_;
        } else {
            player_who_moved = (current_player_ == BLACK) ? WHITE : BLACK;
        }

        int history_player_idx = player_who_moved - 1;

        if (history_player_idx == p_idx_current) {
            buffer[(4 + h) * plane_size + offset] = 1.0f;  // Current player history
        } else {
            buffer[(11 + h) * plane_size + offset] = 1.0f; // Opponent history
        }
    }

    // Planes 18-20: Rule indicators (constant across board)
    float* plane18 = buffer + 18 * plane_size;
    float* plane19 = buffer + 19 * plane_size;
    float* plane20 = buffer + 20 * plane_size;

    float freestyle_val = (!use_renju_ && !use_omok_) ? 1.0f : 0.0f;
    float renju_val = use_renju_ ? 1.0f : 0.0f;
    float omok_val = use_omok_ ? 1.0f : 0.0f;

    for (int i = 0; i < plane_size; ++i) {
        plane18[i] = freestyle_val;
        plane19[i] = renju_val;
        plane20[i] = omok_val;
    }

    // Plane 21: Allowed moves mask
    std::vector<int> legal_moves = getLegalMoves();
    float* plane21 = buffer + 21 * plane_size;
    for (int action : legal_moves) {
        auto [r, c] = action_to_coords_pair(action);
        int offset = r * board_size_ + c;
        plane21[offset] = 1.0f;
    }

    // Planes 22-27: Tactical features (five threats, four threats, open three)
    float* plane22 = buffer + 22 * plane_size;  // Immediate five (current)
    float* plane23 = buffer + 23 * plane_size;  // Immediate five (opponent)
    float* plane24 = buffer + 24 * plane_size;  // Four threat (current)
    float* plane25 = buffer + 25 * plane_size;  // Four threat (opponent)
    float* plane26 = buffer + 26 * plane_size;  // Open three (current)
    float* plane27 = buffer + 27 * plane_size;  // Open three (opponent)

    for (int r = 0; r < board_size_; ++r) {
        for (int c = 0; c < board_size_; ++c) {
            int action = coords_to_action(r, c);
            int offset = r * board_size_ + c;

            // Only check empty positions
            if (!is_bit_set(0, action) && !is_bit_set(1, action)) {
                // Immediate five threats
                if (rules_engine_->is_five_in_a_row(action, current_player_, true)) {
                    plane22[offset] = 1.0f;
                }
                if (rules_engine_->is_five_in_a_row(action, 3 - current_player_, true)) {
                    plane23[offset] = 1.0f;
                }

                // Four threats
                if (createsFourThreat(action, p_idx_current)) {
                    plane24[offset] = 1.0f;
                }
                if (createsFourThreat(action, p_idx_opponent)) {
                    plane25[offset] = 1.0f;
                }

                // Open three patterns
                if (use_omok_) {
                    if (createsOmokOpenThree(action, p_idx_current)) {
                        plane26[offset] = 1.0f;
                    }
                    if (createsOmokOpenThree(action, p_idx_opponent)) {
                        plane27[offset] = 1.0f;
                    }
                } else if (use_renju_) {
                    if (createsRenjuOpenThree(action, p_idx_current)) {
                        plane26[offset] = 1.0f;
                    }
                    if (createsRenjuOpenThree(action, p_idx_opponent)) {
                        plane27[offset] = 1.0f;
                    }
                } else {
                    if (createsFreestyleOpenThree(action, p_idx_current)) {
                        plane26[offset] = 1.0f;
                    }
                    if (createsFreestyleOpenThree(action, p_idx_opponent)) {
                        plane27[offset] = 1.0f;
                    }
                }
            }
        }
    }

    // Planes 28-35: Run-length features (4 directions × 2 players)
    const int DIRS[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};

    for (int dir = 0; dir < 4; ++dir) {
        float* plane_current = buffer + (28 + dir) * plane_size;
        float* plane_opponent = buffer + (32 + dir) * plane_size;

        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int action = coords_to_action(r, c);
                int offset = r * board_size_ + c;

                // Calculate run-length to five for both players
                plane_current[offset] = calculateRunLengthToFive(action, p_idx_current, DIRS[dir][0], DIRS[dir][1]);
                plane_opponent[offset] = calculateRunLengthToFive(action, p_idx_opponent, DIRS[dir][0], DIRS[dir][1]);
            }
        }
    }
}

} // namespace gomoku
} // namespace games
} // namespace alphazero