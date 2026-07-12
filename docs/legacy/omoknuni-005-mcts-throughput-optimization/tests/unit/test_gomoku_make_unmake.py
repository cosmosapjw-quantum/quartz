"""
Unit tests for Gomoku make/unmake equivalence (T024c)

Tests bit-exact restoration and correctness of make_move/unmake_move pattern.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
import alphazero_py


class TestGomokuMakeUnmake:
    """Test make/unmake pattern for Gomoku zero-copy MCTS"""

    def test_make_unmake_single_move(self):
        """Test make/unmake restores exact state for single move"""
        state = alphazero_py.GomokuState(15)

        # Save original state
        original_hash = state.get_hash()
        original_player = state.get_current_player()
        original_string = state.to_string()
        original_history_len = len(state.get_move_history())

        # Apply and reverse move (center position)
        move = 112  # 7*15 + 7 (center of 15x15)
        undo_token = state.make_move(move)
        state.unmake_move(move, undo_token)

        # Verify bit-exact restoration
        assert state.get_hash() == original_hash, "Hash mismatch after make/unmake"
        assert state.get_current_player() == original_player, "Player mismatch after make/unmake"
        assert state.to_string() == original_string, "Board state mismatch after make/unmake"
        assert len(state.get_move_history()) == original_history_len, "History length mismatch"

    def test_make_unmake_multiple_moves(self):
        """Test make/unmake for sequence of moves"""
        state = alphazero_py.GomokuState(15)

        # Save original state
        original_hash = state.get_hash()

        # Apply multiple moves
        moves = [112, 113, 127, 128, 142]  # Diagonal pattern
        undo_tokens = []

        for move in moves:
            undo = state.make_move(move)
            undo_tokens.append(undo)

        # Unwind in LIFO order
        for i in range(len(moves) - 1, -1, -1):
            state.unmake_move(moves[i], undo_tokens[i])

        # Verify restoration
        assert state.get_hash() == original_hash, "Hash mismatch after multiple make/unmake"
        assert state.get_current_player() == 1, "Player should be BLACK after full unwind"
        assert len(state.get_move_history()) == 0, "History should be empty"

    def test_zobrist_hash_consistency(self):
        """Test Zobrist hash is consistent with clone()"""
        state = alphazero_py.GomokuState(15)

        # Clone state for comparison
        cloned = state.clone()

        # Apply move to both
        move = 112
        state.make_move(move)
        cloned.make_move(move)

        # Zobrist hashes must match
        assert state.zobrist_hash() == cloned.get_hash(), "Zobrist hash mismatch with clone"

    def test_deep_path_no_drift(self):
        """Test deep paths (>20 moves) for state drift"""
        state = alphazero_py.GomokuState(15)

        # Generate long move sequence
        moves = []
        for row in range(5):
            for col in range(5):
                moves.append(row * 15 + col)

        undo_tokens = []

        # Apply all moves
        for move in moves:
            undo_tokens.append(state.make_move(move))

        # Save intermediate state
        intermediate_hash = state.get_hash()
        intermediate_player = state.get_current_player()

        # Unwind all moves
        for i in range(len(moves) - 1, -1, -1):
            state.unmake_move(moves[i], undo_tokens[i])

        # Verify restoration to root
        assert state.get_current_player() == 1, "Player should be BLACK after full unwind"
        assert len(state.get_move_history()) == 0, "History should be empty"

        # Re-apply moves
        for move in moves:
            state.make_move(move)

        # Verify same intermediate state
        assert state.get_hash() == intermediate_hash, "Hash mismatch on re-application"
        assert state.get_current_player() == intermediate_player, "Player mismatch on re-application"

    def test_player_flip(self):
        """Test player correctly flips after make_move"""
        state = alphazero_py.GomokuState(15)

        # Black starts
        assert state.get_current_player() == 1

        # Black moves
        undo1 = state.make_move(112)
        assert state.get_current_player() == 2, "Should be White after Black's move"

        # White moves
        undo2 = state.make_move(113)
        assert state.get_current_player() == 1, "Should be Black after White's move"

        # Unwind
        state.unmake_move(113, undo2)
        assert state.get_current_player() == 2, "Should be White after unmake"

        state.unmake_move(112, undo1)
        assert state.get_current_player() == 1, "Should be Black after full unwind"

    def test_legal_moves_consistency(self):
        """Test legal moves are consistent after make/unmake"""
        state = alphazero_py.GomokuState(15)

        # Get legal moves
        original_moves = set(state.get_legal_moves())

        # Apply and reverse move
        move = 112
        undo = state.make_move(move)
        state.unmake_move(move, undo)

        # Legal moves should be unchanged
        restored_moves = set(state.get_legal_moves())
        assert original_moves == restored_moves, "Legal moves changed after make/unmake"

    def test_terminal_state_handling(self):
        """Test make/unmake near terminal state"""
        state = alphazero_py.GomokuState(15)

        # Create near-winning position (4 in a row)
        # Black: 112, 113, 114, 115 (row 7, cols 7-10) - needs 116 to win
        # White: 97, 98, 99 (row 6, cols 7-9) - blocking moves
        moves = [112, 97, 113, 98, 114, 99, 115, 100]  # 8 moves total
        undo_tokens = []

        for move in moves:
            undo_tokens.append(state.make_move(move))

        # Not terminal yet (Black has 4 in a row but it's Black's turn)
        assert not state.is_terminal()

        # Winning move - Black completes 5 in a row
        winning_move = 116  # row 7, col 11 - completes Black's 5 in a row
        undo_win = state.make_move(winning_move)

        # Should be terminal now
        assert state.is_terminal()

        # Unwind winning move
        state.unmake_move(winning_move, undo_win)

        # Should not be terminal after unmake
        assert not state.is_terminal()

        # Full unwind
        for i in range(len(moves) - 1, -1, -1):
            state.unmake_move(moves[i], undo_tokens[i])

        assert not state.is_terminal()
        assert state.get_current_player() == 1

    def test_board_occupation_correctness(self):
        """Test board cells are correctly occupied/unoccupied after make/unmake"""
        state = alphazero_py.GomokuState(15)

        move = 112

        # Cell should be empty
        assert state.is_legal_move(move)

        # Apply move
        undo = state.make_move(move)

        # Cell should be occupied
        assert not state.is_legal_move(move), "Move should be illegal (occupied)"

        # Reverse move
        state.unmake_move(move, undo)

        # Cell should be empty again
        assert state.is_legal_move(move), "Move should be legal after unmake"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
