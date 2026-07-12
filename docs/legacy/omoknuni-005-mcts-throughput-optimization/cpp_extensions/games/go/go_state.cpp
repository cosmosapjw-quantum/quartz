// src/games/go/go_state.cpp
#include "games/go/go_state.h"
// #include "utils/attack_defense_module.h"  // Removed - will be implemented in neural network tasks
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace alphazero {
namespace games {
namespace go {

// Constructor with rule set
GoState::GoState(int board_size, RuleSet rule_set, float custom_komi)
    : IGameState(core::GameType::GO),
      board_size_(board_size),
      current_player_(1),  // Black goes first
      rule_set_(rule_set),
      ko_point_(-1),
      consecutive_passes_(0),
      hash_dirty_(true),
      zobrist_(board_size, 2, 2)  // board_size, 2 piece types (black and white), 2 players
{
    // Set default komi based on rule set
    if (custom_komi < 0) {
        switch (rule_set_) {
            case RuleSet::CHINESE:
                komi_ = 7.5f;
                break;
            case RuleSet::JAPANESE:
            case RuleSet::KOREAN:
                komi_ = 6.5f;
                break;
        }
    } else {
        komi_ = custom_komi;
    }
    
    // Set chinese_rules_ for backward compatibility
    chinese_rules_ = (rule_set_ == RuleSet::CHINESE);
    
    // Determine if we should enforce superko
    bool enforce_superko = (rule_set_ == RuleSet::CHINESE);
    
    // Validate board size
    if (board_size != 9 && board_size != 13 && board_size != 19) {
        board_size_ = 19;  // Default to standard 19x19 if invalid
    }

    // Initialize board with empty intersections
    board_.resize(board_size_ * board_size_, 0);
    
    // Initialize capture counts
    captured_stones_.resize(3, 0);  // Index 0 unused, 1=Black, 2=White
    
    // Initialize Zobrist hash
    hash_ = 0;
    hash_dirty_ = true;
    
    // Initialize repetitive cycle detection
    has_repetitive_cycle_ = false;
    no_result_reason_ = "";
    
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, enforce_superko);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );

    // Add named features for hash calculation
    zobrist_.addFeature("ko_point", board_size_ * board_size_ + 1);  // All positions + none
    zobrist_.addFeature("rules", 2);          // Chinese or Japanese rules
    zobrist_.addFeature("komi", 16);          // Discretized komi values
    
    // Initialize position frequency with initial position
    if (rule_set_ != RuleSet::CHINESE) {
        position_frequency_[getHash()] = 1;
    }
}

// Legacy constructor
GoState::GoState(int board_size, float komi, bool chinese_rules, bool enforce_superko)
    : IGameState(core::GameType::GO),
      board_size_(board_size),
      current_player_(1),  // Black goes first
      komi_(komi),
      chinese_rules_(chinese_rules),
      rule_set_(chinese_rules ? RuleSet::CHINESE : RuleSet::JAPANESE),
      ko_point_(-1),
      consecutive_passes_(0),
      hash_dirty_(true),
      zobrist_(board_size, 2, 2)  // board_size, 2 piece types (black and white), 2 players
{
    // Validate board size
    if (board_size != 9 && board_size != 13 && board_size != 19) {
        board_size_ = 19;  // Default to standard 19x19 if invalid
    }

    // Initialize board with empty intersections
    board_.resize(board_size_ * board_size_, 0);
    
    // Initialize capture counts
    captured_stones_.resize(3, 0);  // Index 0 unused, 1=Black, 2=White
    
    // Initialize Zobrist hash
    hash_ = 0;
    hash_dirty_ = true;
    
    // Initialize repetitive cycle detection
    has_repetitive_cycle_ = false;
    no_result_reason_ = "";
    
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, enforce_superko);
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );

    // Add named features for hash calculation
    zobrist_.addFeature("ko_point", board_size_ * board_size_ + 1);  // All positions + none
    zobrist_.addFeature("rules", 2);          // Chinese or Japanese rules
    zobrist_.addFeature("komi", 16);          // Discretized komi values
    
    // Initialize position frequency with initial position
    if (!chinese_rules_) {
        position_frequency_[getHash()] = 1;
    }
}

// Copy constructor
GoState::GoState(const GoState& other)
    : IGameState(core::GameType::GO),
      board_size_(other.board_size_),
      current_player_(other.current_player_),
      board_(other.board_),
      komi_(other.komi_),
      chinese_rules_(other.chinese_rules_),
      rule_set_(other.rule_set_),
      ko_point_(other.ko_point_),
      captured_stones_(other.captured_stones_),
      consecutive_passes_(other.consecutive_passes_),
      move_history_(other.move_history_),
      position_history_(other.position_history_),
      full_move_history_(other.full_move_history_),
      dead_stones_(other.dead_stones_),
      zobrist_(other.zobrist_),
      hash_(other.hash_),
      hash_dirty_(other.hash_dirty_),
      has_repetitive_cycle_(other.has_repetitive_cycle_),
      no_result_reason_(other.no_result_reason_),
      position_frequency_(other.position_frequency_),
      // Don't copy cached tensors - force recomputation to avoid memory issues
      tensor_cache_dirty_(true),
      enhanced_tensor_cache_dirty_(true),
      groups_cache_dirty_(true)
{
    // Initialize rules
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, other.rules_->isSuperkoenforced());
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );
    
    // Ensure a fresh cache for this copy
    rules_->invalidateCache();
}

// Assignment operator
GoState& GoState::operator=(const GoState& other) {
    if (this != &other) {
        board_size_ = other.board_size_;
        current_player_ = other.current_player_;
        board_ = other.board_;
        komi_ = other.komi_;
        chinese_rules_ = other.chinese_rules_;
        rule_set_ = other.rule_set_;
        ko_point_ = other.ko_point_;
        captured_stones_ = other.captured_stones_;
        consecutive_passes_ = other.consecutive_passes_;
        move_history_ = other.move_history_;
        position_history_ = other.position_history_;
        full_move_history_ = other.full_move_history_;
        dead_stones_ = other.dead_stones_;
        hash_ = other.hash_;
        hash_dirty_ = other.hash_dirty_;
        has_repetitive_cycle_ = other.has_repetitive_cycle_;
        no_result_reason_ = other.no_result_reason_;
        position_frequency_ = other.position_frequency_;
        
        // Clear old tensor caches before assignment
        clearTensorCache();
        
        // Don't copy cached tensors - force recomputation
        tensor_cache_dirty_ = true;
        enhanced_tensor_cache_dirty_ = true;
        groups_cache_dirty_ = true;
        
        // Reinitialize rules
        rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, other.rules_->isSuperkoenforced());
        
        // Set up board accessor functions for rules
        rules_->setBoardAccessor(
            [this](int pos) { return this->getStone(pos); },
            [this](int pos) { return this->isInBounds(pos); },
            [this](int pos) { return this->getAdjacentPositions(pos); }
        );
        
        // Ensure a fresh cache for this copy
        rules_->invalidateCache();
    }
    return *this;
}

// IGameState interface implementation
std::vector<int> GoState::getLegalMoves() const {
    std::vector<int> legalMoves;
    
    // Add pass move (-1)
    legalMoves.push_back(-1);
    
    // Check all board positions
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isValidMove(pos)) {
            // If we're enforcing superko, check that too
            if (rules_->isSuperkoenforced()) {
                // Create a temporary copy to test for superko
                GoState tempState(*this);
                
                // Apply the move without updating history
                tempState.setStone(pos, tempState.current_player_);
                
                // Process any captures
                std::vector<StoneGroup> opponentGroups = tempState.rules_->findGroups(3 - tempState.current_player_);
                for (const auto& group : opponentGroups) {
                    if (group.liberties.empty()) {
                        tempState.captureGroup(group);
                    }
                }
                
                // Check for superko
                uint64_t newHash = tempState.getHash();
                if (!checkForSuperko(newHash)) {
                    legalMoves.push_back(pos);
                }
            } else {
                // If superko is not enforced, add all valid moves
                legalMoves.push_back(pos);
            }
        }
    }
    
    return legalMoves;
}

void GoState::makeMove(int action) {
    if (!isLegalMove(action)) {
        throw std::runtime_error("Illegal move attempted");
    }
    
    // Create a record for this move
    MoveRecord record;
    record.action = action;
    record.ko_point = ko_point_;
    record.consecutive_passes = consecutive_passes_;
    
    // Handle pass
    if (action == -1) {
        consecutive_passes_++;
        ko_point_ = -1;  // Clear ko point on pass
        
        // Record move
        move_history_.push_back(action);
        full_move_history_.push_back(record);
    } else {
        // Reset consecutive passes
        consecutive_passes_ = 0;
        
        // CRITICAL FIX: Always clear ko point for any non-pass move
        ko_point_ = -1;
        
        // Place stone
        setStone(action, current_player_);
        
        // Explicitly invalidate cache before finding groups for capture processing
        rules_->invalidateCache(); 

        // Check for captures
        std::vector<StoneGroup> opponentGroups = rules_->findGroups(3 - current_player_);
        std::vector<StoneGroup> capturedGroups;
        int capturedStones = 0;
        
        for (const auto& group : opponentGroups) {
            if (group.liberties.empty()) {
                capturedGroups.push_back(group);
                capturedStones += group.stones.size();
                // Save captured positions for undo
                for (int pos : group.stones) {
                    record.captured_positions.push_back(pos);
                }
            }
        }
        
        // Process captures
        for (const auto& group : capturedGroups) {
            captureGroup(group);
        }
        
        // ONLY set a ko point if exactly one stone was captured
        if (capturedGroups.size() == 1 && capturedGroups[0].stones.size() == 1) {
            ko_point_ = *capturedGroups[0].stones.begin();
        }
        
        // Update capture count
        captured_stones_[current_player_] += capturedStones;
        
        // Record move
        move_history_.push_back(action);
        full_move_history_.push_back(record);
        
        // Record position for ko/superko detection
        position_history_.push_back(getHash());
    }
    
    // Switch players
    current_player_ = 3 - current_player_;
    
    // Invalidate hash after all state changes
    invalidateHash();
    
    // Track position frequency for Japanese/Korean repetitive cycle detection
    if (rule_set_ != RuleSet::CHINESE && !has_repetitive_cycle_) {
        uint64_t currentHash = getHash();
        position_frequency_[currentHash]++;
        
        // Check for triple ko or similar (position repeated 3+ times)
        if (position_frequency_[currentHash] >= 3) {
            has_repetitive_cycle_ = true;
            if (rule_set_ == RuleSet::JAPANESE) {
                no_result_reason_ = "Triple repetition detected (possible triple ko) - No Result";
            } else if (rule_set_ == RuleSet::KOREAN) {
                no_result_reason_ = "Triple repetition detected (possible triple ko) - Draw";
            }
        }
    }
}

bool GoState::isLegalMove(int action) const {
    if (action == -1) {
        return true;  // Pass is always legal
    }
    
    // First check basic validity
    if (!isValidMove(action)) {
        return false;
    }
    
    // Check basic ko rule
    if (action == ko_point_) {
        return false;  // Ko violation
    }
    
    // If not enforcing superko, we're done
    if (!rules_->isSuperkoenforced()) {
        return true;
    }
    
    // Create a temporary copy to test for superko
    GoState tempState(*this);
    
    // CRITICAL FIX: Clear ko point first, just like in makeMove
    tempState.ko_point_ = -1;
    
    // Apply the move to tempState
    tempState.setStone(action, tempState.current_player_);
    
    // Process captures exactly as makeMove does
    std::vector<StoneGroup> opponentGroups = tempState.rules_->findGroups(3 - tempState.current_player_);
    std::vector<StoneGroup> capturedGroups;
    
    for (const auto& group : opponentGroups) {
        if (group.liberties.empty()) {
            capturedGroups.push_back(group);
            for (int pos : group.stones) {
                tempState.setStone(pos, 0);
            }
        }
    }
    
    // ONLY set a ko point if exactly one stone was captured
    if (capturedGroups.size() == 1 && capturedGroups[0].stones.size() == 1) {
        tempState.ko_point_ = *capturedGroups[0].stones.begin();
    }
    
    // CRITICAL FIX: Calculate hash BEFORE switching players (matching makeMove's timing)
    tempState.invalidateHash();
    uint64_t newHash = tempState.getHash();
    
    // Check for superko violation
    for (uint64_t hash : position_history_) {
        if (hash == newHash) {
            return false;  // Superko violation
        }
    }
    
    return true;
}

bool GoState::undoMove() {
    if (full_move_history_.empty()) {
        return false;
    }
    
    // Get last move record
    MoveRecord lastMove = full_move_history_.back();
    full_move_history_.pop_back();
    
    // Remove from move history
    if (!move_history_.empty()) {
        move_history_.pop_back();
    }
    
    // Remove last position from history
    if (!position_history_.empty()) {
        position_history_.pop_back();
    }
    
    // Update position frequency tracking
    if (!chinese_rules_) {
        uint64_t currentHash = getHash();
        if (position_frequency_[currentHash] > 0) {
            position_frequency_[currentHash]--;
            if (position_frequency_[currentHash] == 0) {
                position_frequency_.erase(currentHash);
            }
        }
        
        // Reset repetitive cycle if we've undone enough moves
        if (has_repetitive_cycle_) {
            // Check if any position still has frequency >= 3
            bool stillHasTripleRep = false;
            for (const auto& [hash, freq] : position_frequency_) {
                if (freq >= 3) {
                    stillHasTripleRep = true;
                    break;
                }
            }
            if (!stillHasTripleRep) {
                has_repetitive_cycle_ = false;
                no_result_reason_ = "";
            }
        }
    }
    
    // Switch back to previous player
    current_player_ = 3 - current_player_;
    
    // Restore ko point
    ko_point_ = lastMove.ko_point;
    
    // Restore consecutive passes
    consecutive_passes_ = lastMove.consecutive_passes;
    
    // If it was a pass, we're done
    if (lastMove.action == -1) {
        // Invalidate hash
        invalidateHash();
        return true;
    }
    
    // Remove the stone
    setStone(lastMove.action, 0);
    
    // Restore captured stones
    for (int pos : lastMove.captured_positions) {
        setStone(pos, 3 - current_player_);  // Opponent's color
    }
    
    // Update captured stones count
    captured_stones_[current_player_] -= lastMove.captured_positions.size();
    
    // Invalidate hash
    invalidateHash();
    
    // Invalidate rules cache
    rules_->invalidateCache();

    return true;
}

// T024e: Zero-copy make/unmake implementation for Go
uint64_t GoState::make_move(uint16_t action) {
    // Apply move using existing makeMove(int) which stores MoveRecord in full_move_history_
    // This leverages existing infrastructure for complex Go logic (captures, ko, superko)

    // Handle pass move: uint16_t(-1) = 65535, convert back to -1
    int move_action = (action == 65535) ? -1 : static_cast<int>(action);
    makeMove(move_action);

    // Return move history size as undo token for LIFO validation
    // (Similar to Chess approach - leverages existing move_history_ infrastructure)
    return static_cast<uint64_t>(full_move_history_.size());
}

void GoState::unmake_move(uint16_t action, uint64_t undo_token) {
    // Use existing undoMove() which handles all Go-specific logic
    // (captures restoration, ko point, superko, position history, etc.)
    if (!undoMove()) {
        throw std::runtime_error("Go unmake_move: no move to undo");
    }

    // Verify undo_token matches expected move_history_ size for correctness
    // After undoing, size should be undo_token - 1
    if (full_move_history_.size() != undo_token - 1) {
        throw std::runtime_error("Go unmake_move: move history size mismatch (LIFO violation)");
    }
}

bool GoState::isTerminal() const {
    // Game ends when both players pass consecutively
    if (consecutive_passes_ >= 2) {
        return true;
    }
    
    // For Japanese/Korean rules, game also ends if repetitive cycle detected
    if (rule_set_ != RuleSet::CHINESE && has_repetitive_cycle_) {
        return true;
    }
    
    return false;
}

core::GameResult GoState::getGameResult() const {
    if (!isTerminal()) {
        return core::GameResult::ONGOING;
    }
    
    // Check for repetitive cycle handling based on rule set
    if (has_repetitive_cycle_) {
        if (rule_set_ == RuleSet::JAPANESE) {
            // Japanese rules: triple ko leads to no result
            return core::GameResult::NO_RESULT;
        } else if (rule_set_ == RuleSet::KOREAN) {
            // Korean rules: triple ko leads to draw
            return core::GameResult::DRAW;
        }
        // Chinese rules: continue to normal scoring (triple ko allowed)
    }
    
    // Calculate scores
    auto [blackScore, whiteScore] = calculateScore();
    
    if (blackScore > whiteScore) {
        return core::GameResult::WIN_PLAYER1;  // Black wins
    } else if (whiteScore > blackScore) {
        return core::GameResult::WIN_PLAYER2;  // White wins
    } else {
        return core::GameResult::DRAW;  // Draw
    }
}

int GoState::getCurrentPlayer() const {
    return current_player_;
}

int GoState::getBoardSize() const {
    return board_size_;
}

int GoState::getActionSpaceSize() const {
    return board_size_ * board_size_ + 1;  // +1 for pass
}

std::vector<std::vector<std::vector<float>>> GoState::getTensorRepresentation() const {
    // T022 SPEC COMPLIANCE: 25-plane representation for Go (enhanced with proper move history)
    // Plane 0: Current player stones
    // Plane 1: Opponent stones
    // Plane 2: Ko position (if any)
    // Planes 3-18: Move history (16 total - 8 pairs for each player)
    // Planes 19-22: Capture patterns (liberties 1, 2, 3, 4+)
    // Plane 23: Legal move indicator
    // Plane 24: Player turn indicator

    auto tensor = std::vector<std::vector<std::vector<float>>>(
        25, std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f)));

    // Planes 0-1: Current player and opponent stones
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);

            if (stone == current_player_) {
                tensor[0][y][x] = 1.0f;  // Current player stones
            } else if (stone == (3 - current_player_)) {
                tensor[1][y][x] = 1.0f;  // Opponent stones
            }
        }
    }

    // Plane 2: Ko position
    if (ko_point_ >= 0 && ko_point_ < board_size_ * board_size_) {
        int ko_y = ko_point_ / board_size_;
        int ko_x = ko_point_ % board_size_;
        if (ko_y >= 0 && ko_y < board_size_ && ko_x >= 0 && ko_x < board_size_) {
            tensor[2][ko_y][ko_x] = 1.0f;
        }
    }

    // Planes 3-18: Move history (16 total - 8 pairs for each player)
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

    // Fill current player history planes (3, 5, 7, 9, 11, 13, 15, 17)
    for (size_t i = 0; i < current_player_moves.size(); ++i) {
        int action = current_player_moves[i];
        if (action >= 0 && action < board_size_ * board_size_) { // Skip pass moves
            int y = action / board_size_;
            int x = action % board_size_;
            if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                tensor[3 + i * 2][y][x] = 1.0f;
            }
        }
    }

    // Fill opponent player history planes (4, 6, 8, 10, 12, 14, 16, 18)
    for (size_t i = 0; i < opponent_player_moves.size(); ++i) {
        int action = opponent_player_moves[i];
        if (action >= 0 && action < board_size_ * board_size_) { // Skip pass moves
            int y = action / board_size_;
            int x = action % board_size_;
            if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                tensor[4 + i * 2][y][x] = 1.0f;
            }
        }
    }

    // Planes 19-22: Capture patterns (groups with 1, 2, 3, 4+ liberties)
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            if (stone != 0) { // Has a stone
                // Find all groups for this player and check if current position belongs to any
                std::vector<StoneGroup> groups = rules_->findGroups(stone);
                for (const auto& group : groups) {
                    // Check if current position is in this group
                    if (group.stones.find(pos) != group.stones.end()) {
                        int liberties = group.liberties.size();
                        if (liberties == 1) {
                            tensor[19][y][x] = 1.0f; // 1 liberty (atari)
                        } else if (liberties == 2) {
                            tensor[20][y][x] = 1.0f; // 2 liberties
                        } else if (liberties == 3) {
                            tensor[21][y][x] = 1.0f; // 3 liberties
                        } else if (liberties >= 4) {
                            tensor[22][y][x] = 1.0f; // 4+ liberties
                        }
                        break; // Found the group containing this position
                    }
                }
            }
        }
    }

    // Plane 23: Legal move indicator
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            if (isLegalMove(pos)) { // Use the GoState method
                tensor[23][y][x] = 1.0f;
            }
        }
    }

    // Plane 24: Player turn indicator
    float playerValue = (current_player_ == 1) ? 1.0f : 0.0f;
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            tensor[24][y][x] = playerValue;
        }
    }

    return tensor;
}

std::vector<std::vector<std::vector<float>>> GoState::getBasicTensorRepresentation() const {
    // Standard AlphaZero format: 19 channels
    // Channel 0: Current player's stones
    // Channel 1: Opponent player's stones
    // Channel 2: Player indicator (all 1s for player 1/black, all 0s for player 2/white)
    // Channels 3-18: Previous 8 moves for each player (16 channels)
    
    const int num_feature_planes = 19;
    
    // Create tensor
    auto tensor = std::vector<std::vector<std::vector<float>>>(
        num_feature_planes, 
        std::vector<std::vector<float>>(board_size_, std::vector<float>(board_size_, 0.0f))
    );
    
    // Channels 0-1: Current and opponent player stones
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            
            if (stone == current_player_) {
                tensor[0][y][x] = 1.0f;  // Current player's stones
            } else if (stone == (3 - current_player_)) {
                tensor[1][y][x] = 1.0f;  // Opponent player's stones
            }
            // Empty = 0.0 (default)
        }
    }
    
    // Channel 2: Player indicator (all 1s for player 1/black, all 0s for player 2/white)
    if (current_player_ == 1) {
        for (int y = 0; y < board_size_; ++y) {
            for (int x = 0; x < board_size_; ++x) {
                tensor[2][y][x] = 1.0f;
            }
        }
    }
    // For player 2 (white), the channel remains all 0s
    
    // Channels 3-18: Move history (8 pairs)
    int history_len = move_history_.size();
    std::vector<int> current_player_moves_in_history;
    std::vector<int> opponent_player_moves_in_history;

    // Go through history backwards and separate moves by player
    for(int k = 0; k < history_len; ++k) {
        int move_action = move_history_[history_len - 1 - k];
        // Note: Go includes pass moves (-1) in history
        if (k % 2 == 0) { 
            // Most recent move was by opponent
            opponent_player_moves_in_history.push_back(move_action);
        } else { 
            // Second most recent move was by current player
            current_player_moves_in_history.push_back(move_action);
        }
    }

    // Fill history channels starting from channel 3
    const int num_history_pairs = 8;
    for(int i = 0; i < num_history_pairs && i < current_player_moves_in_history.size(); ++i) {
        int move = current_player_moves_in_history[i];
        if (move >= 0) { // Not a pass move
            int y = move / board_size_;
            int x = move % board_size_;
            if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                tensor[3 + i*2][y][x] = 1.0f;  // Channels 3, 5, 7, ..., 17
            }
        }
    }
    
    for(int i = 0; i < num_history_pairs && i < opponent_player_moves_in_history.size(); ++i) {
        int move = opponent_player_moves_in_history[i];
        if (move >= 0) { // Not a pass move
            int y = move / board_size_;
            int x = move % board_size_;
            if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                tensor[4 + i*2][y][x] = 1.0f;  // Channels 4, 6, 8, ..., 18
            }
        }
    }
    
    return tensor;
}

std::vector<std::vector<std::vector<float>>> GoState::getEnhancedTensorRepresentation() const {
    // PERFORMANCE FIX: Use cached enhanced tensor if available and not dirty
    if (!enhanced_tensor_cache_dirty_.load(std::memory_order_relaxed) && !cached_enhanced_tensor_repr_.empty()) {
        return cached_enhanced_tensor_repr_;
    }
    
    try {
        // Enhanced format (consistent with basic representation):
        // Channel 0: Current player's stones
        // Channel 1: Opponent player's stones
        // Channel 2: Player indicator (all 1s for player 1/black, all 0s for player 2/white)
        // Channels 3-18: Previous 8 moves for each player (16 channels)
        // Channels 19-20: Attack/defense planes (optional)
        const int num_feature_planes = 21; // Total channels (19 standard + 2 enhanced)
        
        // Create tensor with 21 channels
        std::vector<std::vector<std::vector<float>>> tensor(
            num_feature_planes, 
            std::vector<std::vector<float>>(
                board_size_, 
                std::vector<float>(board_size_, 0.0f)
            )
        );
        
        // Channels 0-1: Current and opponent player stones
        for (int y = 0; y < board_size_; ++y) {
            for (int x = 0; x < board_size_; ++x) {
                int pos = y * board_size_ + x;
                int stone = board_[pos];
                if (stone == current_player_) {
                    tensor[0][y][x] = 1.0f;  // Current player's stones
                } else if (stone == (3 - current_player_)) {
                    tensor[1][y][x] = 1.0f;  // Opponent player's stones
                }
                // Empty squares remain 0.0f
            }
        }
        
        // Channel 2: Player indicator (all 1s for player 1/black, all 0s for player 2/white)
        if (current_player_ == 1) {
            for (int y = 0; y < board_size_; ++y) {
                for (int x = 0; x < board_size_; ++x) {
                    tensor[2][y][x] = 1.0f;
                }
            }
        }
        // For player 2 (white), the channel remains all 0s
        
        // Channels 3-18: Move history (8 pairs)
        int history_len = move_history_.size();
        std::vector<int> current_player_moves;
        std::vector<int> opponent_moves;
        
        // Go through history backwards to get most recent moves first
        for (int k = 0; k < history_len; ++k) {
            int move_action = move_history_[history_len - 1 - k];
            if (move_action >= 0) { // Skip pass moves
                if (k % 2 == 0) {
                    // Most recent move was by opponent (since current player is about to move)
                    opponent_moves.push_back(move_action);
                } else {
                    // This move was by current player
                    current_player_moves.push_back(move_action);
                }
            }
        }
        
        // Fill history channels starting from channel 3
        const int num_history_pairs = 8;
        for (int i = 0; i < num_history_pairs && i < static_cast<int>(current_player_moves.size()); ++i) {
            int action = current_player_moves[i];
            if (action >= 0 && action < board_size_ * board_size_) {
                int y = action / board_size_;
                int x = action % board_size_;
                if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                    tensor[3 + i * 2][y][x] = 1.0f;  // Channels 3, 5, 7, ..., 17
                }
            }
        }
        
        for (int i = 0; i < num_history_pairs && i < static_cast<int>(opponent_moves.size()); ++i) {
            int action = opponent_moves[i];
            if (action >= 0 && action < board_size_ * board_size_) {
                int y = action / board_size_;
                int x = action % board_size_;
                if (y >= 0 && y < board_size_ && x >= 0 && x < board_size_) {
                    tensor[4 + i * 2][y][x] = 1.0f;  // Channels 4, 6, 8, ..., 18
                }
            }
        }
        
        // Channels 19-20: Attack and Defense planes (GPU-accelerated if available)
        computeAttackDefensePlanes(tensor);
        
        // PERFORMANCE FIX: Cache the computed enhanced tensor
        cached_enhanced_tensor_repr_ = tensor;
        enhanced_tensor_cache_dirty_.store(false, std::memory_order_relaxed);
        
        return tensor;
    } catch (const std::exception& e) {
        std::cerr << "Exception in GoState::getEnhancedTensorRepresentation: " << e.what() << std::endl;
        
        // Return a default tensor with the correct dimensions
        // Enhanced format: 21 channels
        const int num_planes = 21;
        
        return std::vector<std::vector<std::vector<float>>>(
            num_planes,
            std::vector<std::vector<float>>(
                board_size_,
                std::vector<float>(board_size_, 0.0f)
            )
        );
    } catch (...) {
        std::cerr << "Unknown exception in GoState::getEnhancedTensorRepresentation" << std::endl;
        
        // Return a default tensor with the correct dimensions
        const int num_planes = 21;
        
        return std::vector<std::vector<std::vector<float>>>(
            num_planes,
            std::vector<std::vector<float>>(
                board_size_,
                std::vector<float>(board_size_, 0.0f)
            )
        );
    }
}

uint64_t GoState::getHash() const {
    if (hash_dirty_) {
        updateHash();
    }
    return hash_;
}

std::unique_ptr<core::IGameState> GoState::clone() const {
    return std::make_unique<GoState>(*this);
}

std::vector<std::unique_ptr<core::IGameState>> GoState::batchClone(int count) const {
    std::vector<std::unique_ptr<core::IGameState>> clones;
    clones.reserve(count);
    
    // Use copy constructor for efficient cloning
    for (int i = 0; i < count; ++i) {
        clones.push_back(std::make_unique<GoState>(*this));
    }
    
    return clones;
}

void GoState::copyFrom(const core::IGameState& source) {
    // Ensure source is a GoState
    const GoState* go_source = dynamic_cast<const GoState*>(&source);
    if (!go_source) {
        throw std::runtime_error("Cannot copy from non-GoState: incompatible game types");
    }
    
    // Copy all member variables
    board_size_ = go_source->board_size_;
    current_player_ = go_source->current_player_;
    board_ = go_source->board_;
    komi_ = go_source->komi_;
    chinese_rules_ = go_source->chinese_rules_;
    rule_set_ = go_source->rule_set_;
    ko_point_ = go_source->ko_point_;
    captured_stones_ = go_source->captured_stones_;
    consecutive_passes_ = go_source->consecutive_passes_;
    move_history_ = go_source->move_history_;
    position_history_ = go_source->position_history_;
    full_move_history_ = go_source->full_move_history_;
    dead_stones_ = go_source->dead_stones_;
    zobrist_ = go_source->zobrist_;
    hash_ = go_source->hash_;
    hash_dirty_ = go_source->hash_dirty_;
    has_repetitive_cycle_ = go_source->has_repetitive_cycle_;
    no_result_reason_ = go_source->no_result_reason_;
    position_frequency_ = go_source->position_frequency_;
    
    // Re-create rules with proper configuration
    rules_ = std::make_shared<GoRules>(board_size_, chinese_rules_, go_source->rules_->isSuperkoenforced());
    
    // Set up board accessor functions for rules
    rules_->setBoardAccessor(
        [this](int pos) { return this->getStone(pos); },
        [this](int pos) { return this->isInBounds(pos); },
        [this](int pos) { return this->getAdjacentPositions(pos); }
    );
    
    // Ensure fresh cache
    rules_->invalidateCache();
}

std::string GoState::actionToString(int action) const {
    if (action == -1) {
        return "pass";
    }
    
    if (action < 0 || action >= board_size_ * board_size_) {
        return "invalid";
    }
    
    std::pair<int, int> coords = actionToCoord(action);
    int x = coords.first;
    int y = coords.second;
    
    // Convert to Go coordinates (A-T, skipping I, 1-19)
    char colChar = 'A' + x;
    if (colChar >= 'I') {
        colChar++;  // Skip 'I'
    }
    
    return std::string(1, colChar) + std::to_string(board_size_ - y);
}

std::optional<int> GoState::stringToAction(const std::string& moveStr) const {
    if (moveStr == "pass" || moveStr == "PASS" || moveStr == "Pass") {
        return -1;
    }
    
    if (moveStr.length() < 2) {
        return std::nullopt;
    }
    
    char colChar = std::toupper(moveStr[0]);
    
    // Skip 'I' as it's not used in Go notation
    if (colChar == 'I') {
        return std::nullopt;
    }
    
    // Adjust for 'I' being skipped
    int x;
    if (colChar >= 'J') {
        x = colChar - 'A' - 1;
    } else {
        x = colChar - 'A';
    }
    
    // Parse row
    int y;
    try {
        y = board_size_ - std::stoi(moveStr.substr(1));
    } catch (...) {
        return std::nullopt;
    }
    
    if (x < 0 || x >= board_size_ || y < 0 || y >= board_size_) {
        return std::nullopt;
    }
    
    return coordToAction(x, y);
}

std::string GoState::toString() const {
    std::stringstream ss;
    
    // Print column headers
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print board
    for (int y = 0; y < board_size_; ++y) {
        ss << std::setw(2) << (board_size_ - y) << " ";
        
        for (int x = 0; x < board_size_; ++x) {
            int pos = y * board_size_ + x;
            int stone = getStone(pos);
            
            if (stone == 0) {
                // Check if this is a ko point
                if (pos == ko_point_) {
                    ss << "k ";
                } else if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "d ";  // Mark dead stones
                } else {
                    ss << ". ";
                }
            } else if (stone == 1) {
                if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "x ";  // Dead black stone
                } else {
                    ss << "X ";  // Black
                }
            } else if (stone == 2) {
                if (dead_stones_.find(pos) != dead_stones_.end()) {
                    ss << "o ";  // Dead white stone
                } else {
                    ss << "O ";  // White
                }
            }
        }
        
        ss << (board_size_ - y) << std::endl;
    }
    
    // Print column headers again
    ss << "   ";
    for (int x = 0; x < board_size_; ++x) {
        char colChar = 'A' + x;
        if (colChar >= 'I') {
            colChar++;  // Skip 'I'
        }
        ss << colChar << " ";
    }
    ss << std::endl;
    
    // Print game info
    ss << "Current player: " << (current_player_ == 1 ? "Black" : "White") << std::endl;
    ss << "Captures - Black: " << captured_stones_[1] << ", White: " << captured_stones_[2] << std::endl;
    ss << "Komi: " << komi_ << std::endl;
    ss << "Rules: ";
    switch (rule_set_) {
        case RuleSet::CHINESE: ss << "Chinese"; break;
        case RuleSet::JAPANESE: ss << "Japanese"; break;
        case RuleSet::KOREAN: ss << "Korean"; break;
    }
    ss << std::endl;
    ss << "Superko enforcement: " << (rules_->isSuperkoenforced() ? "Yes" : "No") << std::endl;
    
    if (isTerminal()) {
        auto [blackScore, whiteScore] = calculateScore();
        
        ss << "Game over!" << std::endl;
        ss << "Final score - Black: " << blackScore << ", White: " << whiteScore 
           << " (with komi " << komi_ << ")" << std::endl;
        
        if (blackScore > whiteScore) {
            ss << "Black wins by " << (blackScore - whiteScore) << " points" << std::endl;
        } else if (whiteScore > blackScore) {
            ss << "White wins by " << (whiteScore - blackScore) << " points" << std::endl;
        } else {
            ss << "Game ended in a draw" << std::endl;
        }
    }
    
    return ss.str();
}

bool GoState::equals(const core::IGameState& other) const {
    if (other.getGameType() != core::GameType::GO) {
        return false;
    }
    
    try {
        const GoState& otherGo = dynamic_cast<const GoState&>(other);
        
        if (board_size_ != otherGo.board_size_ || 
            current_player_ != otherGo.current_player_ ||
            ko_point_ != otherGo.ko_point_ ||
            komi_ != otherGo.komi_ ||
            chinese_rules_ != otherGo.chinese_rules_ ||
            rule_set_ != otherGo.rule_set_ ||
            consecutive_passes_ != otherGo.consecutive_passes_ ||
            captured_stones_ != otherGo.captured_stones_ ||
            dead_stones_ != otherGo.dead_stones_) {
            return false;
        }
        
        // Compare board positions
        return board_ == otherGo.board_;
    } catch (const std::bad_cast&) {
        return false;
    }
}

std::vector<int> GoState::getMoveHistory() const {
    return move_history_;
}

bool GoState::validate() const {
    // Check board size
    if (board_size_ != 9 && board_size_ != 13 && board_size_ != 19) {
        return false;
    }
    
    // Check current player
    if (current_player_ != 1 && current_player_ != 2) {
        return false;
    }
    
    // Check ko point
    if (ko_point_ >= board_size_ * board_size_) {
        return false;
    }
    
    // Check captured stones
    if (captured_stones_.size() != 3) {
        return false;
    }
    
    // Count stones of each color on the board
    int black_count = 0;
    int white_count = 0;
    
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        if (isInBounds(pos)) {
            int stone = getStone(pos);
            if (stone == 1) black_count++;
            else if (stone == 2) white_count++;
        }
    }
    
    // Check if stone counts make sense in relation to captures
    // Black goes first, so if it's black's turn, white should have placed equal stones
    // If it's white's turn, black should have one more stone
    if ((current_player_ == 1 && black_count != white_count + captured_stones_[1] - captured_stones_[2]) ||
        (current_player_ == 2 && black_count != white_count + 1 + captured_stones_[1] - captured_stones_[2])) {
        return false;
    }
    
    return true;
}

// Go-specific methods
int GoState::getStone(int pos) const {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return 0;  // Out of bounds, return empty
    }
    return board_[pos];
}

int GoState::getStone(int x, int y) const {
    if (!isInBounds(x, y)) {
        return 0;  // Out of bounds, return empty
    }
    return board_[y * board_size_ + x];
}

void GoState::setStone(int pos, int stone) {
    if (pos < 0 || pos >= board_size_ * board_size_) {
        return;  // Out of bounds, do nothing
    }
    board_[pos] = stone;
    invalidateHash();
    rules_->invalidateCache();
}

void GoState::setStone(int x, int y, int stone) {
    if (!isInBounds(x, y)) {
        return;  // Out of bounds, do nothing
    }
    board_[y * board_size_ + x] = stone;
    invalidateHash();
    rules_->invalidateCache();
}

int GoState::getCapturedStones(int player) const {
    if (player != 1 && player != 2) {
        return 0;
    }
    return captured_stones_[player];
}

float GoState::getKomi() const {
    return komi_;
}

bool GoState::isChineseRules() const {
    return chinese_rules_;
}

GoState::RuleSet GoState::getRuleSet() const {
    return rule_set_;
}

bool GoState::isEnforcingSuperko() const {
    return rules_->isSuperkoenforced();
}

std::pair<int, int> GoState::actionToCoord(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return {-1, -1};
    }
    
    int y = action / board_size_;
    int x = action % board_size_;
    
    return {x, y};
}

int GoState::coordToAction(int x, int y) const {
    if (!isInBounds(x, y)) {
        return -1;
    }
    
    return y * board_size_ + x;
}

int GoState::getKoPoint() const {
    return ko_point_;
}

std::vector<int> GoState::getTerritoryOwnership(const std::unordered_set<int>& dead_stones) const {
    // Combine local dead stones with any provided
    std::unordered_set<int> all_dead_stones = dead_stones;
    all_dead_stones.insert(dead_stones_.begin(), dead_stones_.end());
    
    return rules_->getTerritoryOwnership(all_dead_stones);
}

bool GoState::isInsideTerritory(int pos, int player, const std::unordered_set<int>& dead_stones) const {
    // Combine local dead stones with any provided
    std::unordered_set<int> all_dead_stones = dead_stones;
    all_dead_stones.insert(dead_stones_.begin(), dead_stones_.end());
    
    std::vector<int> territory = rules_->getTerritoryOwnership(all_dead_stones);
    if (pos < 0 || pos >= static_cast<int>(territory.size())) {
        return false;
    }
    return territory[pos] == player;
}

void GoState::markDeadStones(const std::unordered_set<int>& positions) {
    dead_stones_ = positions;
    invalidateHash();
    rules_->invalidateCache();
}

const std::unordered_set<int>& GoState::getDeadStones() const {
    return dead_stones_;
}

void GoState::clearDeadStones() {
    dead_stones_.clear();
    invalidateHash();
    rules_->invalidateCache();
}

std::pair<float, float> GoState::calculateScore() const {
    return rules_->calculateScores(captured_stones_, komi_, dead_stones_);
}

std::vector<int> GoState::findDamePoints() const {
    std::vector<int> damePoints;
    
    // Get territory ownership
    std::vector<int> territory = getTerritoryOwnership(dead_stones_);
    
    // Dame points are empty positions with neutral territory (0)
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (getStone(pos) == 0 && territory[pos] == 0) {
            damePoints.push_back(pos);
        }
    }
    
    return damePoints;
}

bool GoState::areAllDameFilled() const {
    // For Chinese rules, check if there are any unfilled dame
    if (!chinese_rules_) {
        return true;  // Japanese rules don't require dame filling
    }
    
    std::vector<int> damePoints = findDamePoints();
    return damePoints.empty();
}

bool GoState::hasRepetitiveCycle() const {
    return has_repetitive_cycle_;
}

std::string GoState::getNoResultReason() const {
    return no_result_reason_;
}

// Helper methods
std::vector<int> GoState::getAdjacentPositions(int pos) const {
    std::vector<int> adjacentPositions;
    int x, y;
    std::tie(x, y) = actionToCoord(pos);
    
    // Check orthogonally adjacent positions
    for (const auto& direction : std::vector<std::pair<int, int>>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
        int newX = x + direction.first;
        int newY = y + direction.second;
        
        if (isInBounds(newX, newY)) {
            adjacentPositions.push_back(coordToAction(newX, newY));
        }
    }
    
    return adjacentPositions;
}

bool GoState::isInBounds(int x, int y) const {
    return x >= 0 && x < board_size_ && y >= 0 && y < board_size_;
}

bool GoState::isInBounds(int pos) const {
    return pos >= 0 && pos < board_size_ * board_size_;
}

void GoState::invalidateHash() {
    hash_dirty_ = true;
    
    // PERFORMANCE FIX: Return old tensors before invalidating caches
    clearTensorCache();
    
    // PERFORMANCE FIX: Invalidate tensor and group caches when game state changes
    tensor_cache_dirty_.store(true, std::memory_order_release);
    enhanced_tensor_cache_dirty_.store(true, std::memory_order_release);
    groups_cache_dirty_.store(true, std::memory_order_release);
}

GoState::~GoState() {
    // Return cached tensors to the pool to prevent memory leaks
    clearTensorCache();
}

void GoState::clearTensorCache() const {
    // Simply clear the cached tensors
    cached_tensor_repr_.clear();
    cached_enhanced_tensor_repr_.clear();
}

void GoState::captureGroup(const StoneGroup& group) {
    // Remove all stones in the group
    for (int pos : group.stones) {
        setStone(pos, 0);  // This should be correctly removing stones
    }
    
    // Explicitly invalidate the rules cache after removing stones
    rules_->invalidateCache();
}

void GoState::captureStones(const std::unordered_set<int>& positions) {
    for (int pos : positions) {
        setStone(pos, 0);
    }
    // Note: setStone already invalidates the cache
}

bool GoState::isValidMove(int action) const {
    if (action < 0 || action >= board_size_ * board_size_) {
        return false;
    }
    
    // Check if the intersection is empty
    if (getStone(action) != 0) {
        return false;
    }
    
    // Check for suicide rule
    if (rules_->isSuicidalMove(action, current_player_)) {
        return false;
    }
    
    return true;
}

bool GoState::checkForSuperko(uint64_t new_hash) const {
    // Check if this position has appeared before
    for (uint64_t hash : position_history_) {
        if (hash == new_hash) {
            return true;  // Position repetition found
        }
    }
    return false;
}

void GoState::updateHash() const {
    hash_ = 0;
    
    // Hash board position
    for (int pos = 0; pos < board_size_ * board_size_; pos++) {
        if (!isInBounds(pos)) continue;
        
        int stone = getStone(pos);
        if (stone != 0) {
            int pieceIdx = stone - 1;  // Convert to 0-based index
            hash_ ^= zobrist_.getPieceHash(pieceIdx, pos);
        }
    }
    
    // Hash current player only for non-Chinese rules
    // Chinese rules use Positional Superko (PSK) - no player in hash
    // AGA/other rules use Situational Superko (SSK) - include player in hash
    // Japanese rules don't use superko but we include player for consistency
    if (!chinese_rules_) {
        hash_ ^= zobrist_.getPlayerHash(current_player_ - 1);
    }
    
    // Hash ko point
    if (ko_point_ >= 0) {
        hash_ ^= zobrist_.getFeatureHash("ko_point", ko_point_);
    }
    
    // Hash the rule variant
    if (chinese_rules_) {
        hash_ ^= zobrist_.getFeatureHash("rules", 1);  // Chinese rules
    } else {
        hash_ ^= zobrist_.getFeatureHash("rules", 0);  // Japanese rules
    }
    
    // Hash komi value (discretized)
    int komi_int = static_cast<int>(komi_ * 2);  // Convert to half-points
    hash_ ^= zobrist_.getFeatureHash("komi", komi_int & 0xF);  // Use lower 4 bits
    
    hash_dirty_ = false;
}

// Attack/Defense plane computation
void GoState::computeAttackDefensePlanes(std::vector<std::vector<std::vector<float>>>& tensor) const {
    // Initialize attack and defense planes
    std::vector<std::vector<float>> captureAttackPlane(board_size_, std::vector<float>(board_size_, 0.0f));
    std::vector<std::vector<float>> libertyDefensePlane(board_size_, std::vector<float>(board_size_, 0.0f));
    
#ifdef WITH_TORCH
    // Try GPU-accelerated computation if available
    // TODO: Re-implement GPU attack/defense computation using new interface
    // Need to implement GoGPUAttackDefense class that:
    // 1. Efficiently computes liberty counts using GPU convolutions
    // 2. Detects captures and atari situations in parallel
    // 3. Evaluates eye potential and territory
    // 4. Uses GPU flood-fill for group detection
    // Example implementation:
    // auto gpu_module = createGPUAttackDefenseModule(GameType::GO, board_size_, torch::kCUDA);
    // auto board_tensor = convertGoBoardToTensor(board_);
    // auto [attack_gpu, defense_gpu] = gpu_module->compute_planes_gpu(board_tensor, current_player_);
    
    // GPU functions commented out - no longer available
    // if (utils::AttackDefenseModule::isGPUAvailable()) {
    //     try {
    //         // Create a batch with just this state
    //         std::vector<const GoState*> states = {this};
    //         
    //         // Call GPU batch computation
    //         auto gpu_result = utils::AttackDefenseModule::computeGoAttackDefenseGPU(states);
    //         
    //         if (gpu_result.size(0) > 0) {
    //             // Extract attack and defense tensors
    //             auto capture_tensor = gpu_result[0][0];
    //             auto defense_tensor = gpu_result[0][1];
    //             
    //             // Convert torch tensors to std::vector
    //             auto capture_accessor = capture_tensor.accessor<float, 2>();
    //             auto defense_accessor = defense_tensor.accessor<float, 2>();
    //             
    //             for (int i = 0; i < board_size_; ++i) {
    //                 for (int j = 0; j < board_size_; ++j) {
    //                     captureAttackPlane[i][j] = capture_accessor[i][j];
    //                     libertyDefensePlane[i][j] = defense_accessor[i][j];
    //                 }
    //             }
    //             
    //             tensor.push_back(captureAttackPlane);
    //             tensor.push_back(libertyDefensePlane);
    //             return;
    //         }
    //     } catch (const std::exception& e) {
    //         // Fall back to CPU computation
    //         std::cerr << "GPU attack/defense computation failed: " << e.what() << std::endl;
    //     }
    // }
#endif
    
    // CPU fallback: compute capture potential and liberty-based defense
    auto legal_moves = getLegalMoves();
    
    // Analyze each legal move
    for (int move : legal_moves) {
        if (move == -1) continue;  // Skip pass
        
        int y = move / board_size_;
        int x = move % board_size_;
        
        // Calculate capture potential (attack)
        float capture_score = 0.0f;
        
        // Check adjacent positions for potential captures
        auto adjacents = getAdjacentPositions(move);
        for (int adj : adjacents) {
            int stone = getStone(adj);
            if (stone != 0 && stone != current_player_) {
                // Check if this enemy group can be captured
                // Simple heuristic: groups with fewer liberties are more capturable
                
                // Find the group this stone belongs to
                std::vector<int> group;
                std::unordered_set<int> visited;
                std::vector<int> stack = {adj};
                
                while (!stack.empty()) {
                    int pos = stack.back();
                    stack.pop_back();
                    
                    if (visited.count(pos) > 0) continue;
                    visited.insert(pos);
                    
                    if (getStone(pos) == stone) {
                        group.push_back(pos);
                        auto adj_positions = getAdjacentPositions(pos);
                        for (int p : adj_positions) {
                            if (visited.count(p) == 0) {
                                stack.push_back(p);
                            }
                        }
                    }
                }
                
                // Count liberties of this group
                std::unordered_set<int> liberties;
                for (int g : group) {
                    auto adj_positions = getAdjacentPositions(g);
                    for (int p : adj_positions) {
                        if (getStone(p) == 0 && p != move) {
                            liberties.insert(p);
                        }
                    }
                }
                
                // If placing stone at 'move' would capture this group
                if (liberties.size() == 0) {
                    capture_score += group.size() * 1.0f;
                } else if (liberties.size() == 1) {
                    capture_score += group.size() * 0.5f;  // Atari
                }
            }
        }
        
        // Calculate liberty defense score
        float defense_score = 0.0f;
        
        // Check if this move connects to friendly groups and increases liberties
        for (int adj : adjacents) {
            int stone = getStone(adj);
            if (stone == current_player_) {
                // Count current liberties of the friendly group
                std::vector<int> group;
                std::unordered_set<int> visited;
                std::vector<int> stack = {adj};
                
                while (!stack.empty()) {
                    int pos = stack.back();
                    stack.pop_back();
                    
                    if (visited.count(pos) > 0) continue;
                    visited.insert(pos);
                    
                    if (getStone(pos) == stone) {
                        group.push_back(pos);
                        auto adj_positions = getAdjacentPositions(pos);
                        for (int p : adj_positions) {
                            if (visited.count(p) == 0) {
                                stack.push_back(p);
                            }
                        }
                    }
                }
                
                // Count liberties before and after the move
                std::unordered_set<int> liberties_before;
                for (int g : group) {
                    auto adj_positions = getAdjacentPositions(g);
                    for (int p : adj_positions) {
                        if (getStone(p) == 0) {
                            liberties_before.insert(p);
                        }
                    }
                }
                
                // If group has few liberties, this move helps defend it
                if (liberties_before.size() <= 2) {
                    defense_score += (3.0f - liberties_before.size()) * 0.5f;
                }
            }
        }
        
        // Also check for eye-making potential (defensive)
        int empty_adjacents = 0;
        int friendly_adjacents = 0;
        for (int adj : adjacents) {
            if (getStone(adj) == 0) empty_adjacents++;
            else if (getStone(adj) == current_player_) friendly_adjacents++;
        }
        
        if (friendly_adjacents >= 2 && empty_adjacents <= 1) {
            defense_score += 0.3f;  // Potential eye point
        }
        
        // Normalize and assign scores
        captureAttackPlane[y][x] = std::min(1.0f, capture_score / 5.0f);
        libertyDefensePlane[y][x] = std::min(1.0f, defense_score);
    }
    
    // Assign to channels 18 and 19 (already allocated)
    if (tensor.size() >= 20) {
        tensor[18] = captureAttackPlane;
        tensor[19] = libertyDefensePlane;
    } else {
        // Fallback for old code that might expect push_back
        tensor.push_back(captureAttackPlane);
        tensor.push_back(libertyDefensePlane);
    }
}

// Static batch computation for multiple states (GPU-accelerated)
std::vector<std::vector<std::vector<std::vector<float>>>> 
GoState::computeBatchEnhancedTensorRepresentations(const std::vector<const GoState*>& states) {
    std::vector<std::vector<std::vector<std::vector<float>>>> results;
    results.reserve(states.size());
    
#ifdef WITH_TORCH
    // Try GPU batch computation for attack/defense planes
    // GPU functions commented out - no longer available
    // if (utils::AttackDefenseModule::isGPUAvailable() && states.size() > 1) {
    if (false) {  // Disabled GPU path
        try {
            // auto gpu_results = utils::AttackDefenseModule::computeGoAttackDefenseGPU(states);
            torch::Tensor gpu_results = torch::zeros({0, 0, 0});  // Empty tensor as placeholder
            
            // Process each state with GPU results
            for (size_t i = 0; i < states.size(); ++i) {
                auto tensor = states[i]->getEnhancedTensorRepresentation();
                
                // Add GPU-computed attack/defense planes
                if (i < gpu_results.size(0)) {
                    auto capture_tensor = gpu_results[i][0];
                    auto defense_tensor = gpu_results[i][1];
                    
                    auto capture_accessor = capture_tensor.accessor<float, 2>();
                    auto defense_accessor = defense_tensor.accessor<float, 2>();
                    
                    int board_size = states[i]->getBoardSize();
                    
                    // Add capture attack plane
                    std::vector<std::vector<float>> capture_plane(board_size, std::vector<float>(board_size, 0.0f));
                    std::vector<std::vector<float>> defense_plane(board_size, std::vector<float>(board_size, 0.0f));
                    
                    for (int r = 0; r < board_size; ++r) {
                        for (int c = 0; c < board_size; ++c) {
                            capture_plane[r][c] = capture_accessor[r][c];
                            defense_plane[r][c] = defense_accessor[r][c];
                        }
                    }
                    
                    tensor.push_back(capture_plane);
                    tensor.push_back(defense_plane);
                }
                
                results.push_back(tensor);
            }
            return results;
        } catch (const std::exception& e) {
            // Fall back to CPU computation
            std::cerr << "Batch GPU computation failed: " << e.what() << std::endl;
        }
    }
#endif
    
    // CPU fallback
    for (const auto* state : states) {
        results.push_back(state->getEnhancedTensorRepresentation());
    }
    return results;
}

std::vector<std::vector<uint64_t>> GoState::getBitboards() const {
    // Convert Go board representation to bitboards
    // Go uses 3 states: empty (0), black (1), white (2)
    // Format: 3 bitboards
    // [0]: Black stones (player 1)
    // [1]: White stones (player 2)
    // [2]: All occupied positions (black | white) - useful for legal move checking
    
    int total_positions = board_size_ * board_size_;
    int num_words = (total_positions + 63) / 64;
    
    // Initialize bitboards
    std::vector<std::vector<uint64_t>> bitboards(3);
    for (int i = 0; i < 3; ++i) {
        bitboards[i].resize(num_words, 0ULL);
    }
    
    // Fill bitboards based on board state
    for (int pos = 0; pos < total_positions; ++pos) {
        int stone = board_[pos];
        if (stone == 1 || stone == 2) {  // Black or White
            int word_idx = pos / 64;
            int bit_idx = pos % 64;
            uint64_t bit = 1ULL << bit_idx;
            
            int player_idx = stone - 1;  // 0 for black, 1 for white
            bitboards[player_idx][word_idx] |= bit;
            bitboards[2][word_idx] |= bit;  // All occupied positions
        }
    }
    
    return bitboards;
}

// ============================================================================
// T007e: Direct Feature Extraction to Buffer
// ============================================================================

int GoState::get_num_feature_planes() const {
    // Go enhanced tensor has 21 planes (verified empirically)
    // This matches the actual implementation in getEnhancedTensorRepresentation()
    return 21;
}

void GoState::extract_features_to_buffer(float* buffer) const {
    // Simplified implementation: Use existing getEnhancedTensorRepresentation()
    // and copy to buffer. This can be optimized later for zero-copy.
    //
    // Future optimization (T007e follow-up): Direct write like Gomoku implementation
    auto tensor = getEnhancedTensorRepresentation();

    const int num_planes = static_cast<int>(tensor.size());  // Use actual tensor size
    const int height = board_size_;  // 19 for standard Go
    const int width = board_size_;
    const int plane_size = height * width;

    // Copy tensor data to buffer in row-major layout
    for (int p = 0; p < num_planes; ++p) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                buffer[p * plane_size + r * width + c] = tensor[p][r][c];
            }
        }
    }
}

} // namespace go
} // namespace games
} // namespace alphazero