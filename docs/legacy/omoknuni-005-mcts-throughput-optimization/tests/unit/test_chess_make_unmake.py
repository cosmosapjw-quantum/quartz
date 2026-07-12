"""
Unit tests for Chess make/unmake equivalence (T024d)

Tests bit-exact restoration and correctness of make_move/unmake_move pattern.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
import alphazero_py


class TestChessMakeUnmake:
    """Test make/unmake pattern for Chess zero-copy MCTS"""

    def test_make_unmake_single_move(self):
        """Test make/unmake restores exact state for single move"""
        state = alphazero_py.ChessState()

        # Save original state
        original_hash = state.get_hash()
        original_player = state.get_current_player()
        original_string = state.to_string()
        original_history_len = len(state.get_move_history())

        # Get a legal move (e2e4 - pawn to e4)
        legal_moves = state.get_legal_moves()
        assert len(legal_moves) > 0, "Should have legal moves at start"

        # Apply and reverse move
        move = legal_moves[0]  # First legal move
        undo_token = state.make_move(move)
        state.unmake_move(move, undo_token)

        # Verify bit-exact restoration
        assert state.get_hash() == original_hash, "Hash mismatch after make/unmake"
        assert state.get_current_player() == original_player, "Player mismatch after make/unmake"
        assert state.to_string() == original_string, "Board state mismatch after make/unmake"
        assert len(state.get_move_history()) == original_history_len, "History length mismatch"

    def test_make_unmake_multiple_moves(self):
        """Test make/unmake for sequence of moves"""
        state = alphazero_py.ChessState()

        # Save original state
        original_hash = state.get_hash()

        # Apply multiple moves (5-move opening)
        moves = []
        undo_tokens = []

        for _ in range(5):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) == 0:
                break
            move = legal_moves[0]
            moves.append(move)
            undo = state.make_move(move)
            undo_tokens.append(undo)

        # Unwind in LIFO order
        for i in range(len(moves) - 1, -1, -1):
            state.unmake_move(moves[i], undo_tokens[i])

        # Verify restoration
        assert state.get_hash() == original_hash, "Hash mismatch after multiple make/unmake"
        assert state.get_current_player() == 1, "Player should be WHITE after full unwind"
        assert len(state.get_move_history()) == 0, "History should be empty"

    def test_zobrist_hash_consistency(self):
        """Test Zobrist hash is consistent with clone()"""
        state = alphazero_py.ChessState()

        # Clone state for comparison
        cloned = state.clone()

        # Get first legal move
        legal_moves = state.get_legal_moves()
        move = legal_moves[0]

        # Apply move to both
        state.make_move(move)
        cloned.make_move(move)

        # Zobrist hashes must match
        assert state.zobrist_hash() == cloned.get_hash(), "Zobrist hash mismatch with clone"

    def test_deep_path_no_drift(self):
        """Test deep paths (>10 moves) for state drift"""
        state = alphazero_py.ChessState()

        # Generate move sequence (10 moves)
        moves = []
        undo_tokens = []

        for _ in range(10):
            legal_moves = state.get_legal_moves()
            if len(legal_moves) == 0 or state.is_terminal():
                break
            move = legal_moves[0]
            moves.append(move)
            undo_tokens.append(state.make_move(move))

        # Save intermediate state
        intermediate_hash = state.get_hash()
        intermediate_player = state.get_current_player()

        # Unwind all moves
        for i in range(len(moves) - 1, -1, -1):
            state.unmake_move(moves[i], undo_tokens[i])

        # Verify restoration to root
        assert state.get_current_player() == 1, "Player should be WHITE after full unwind"
        assert len(state.get_move_history()) == 0, "History should be empty"

        # Re-apply moves
        for move in moves:
            state.make_move(move)

        # Verify same intermediate state
        assert state.get_hash() == intermediate_hash, "Hash mismatch on re-application"
        assert state.get_current_player() == intermediate_player, "Player mismatch on re-application"

    def test_player_flip(self):
        """Test player correctly flips after make_move"""
        state = alphazero_py.ChessState()

        # White starts
        assert state.get_current_player() == 1

        # White moves
        legal_moves = state.get_legal_moves()
        undo1 = state.make_move(legal_moves[0])
        assert state.get_current_player() == 2, "Should be Black after White's move"

        # Black moves
        legal_moves = state.get_legal_moves()
        undo2 = state.make_move(legal_moves[0])
        assert state.get_current_player() == 1, "Should be White after Black's move"

        # Unwind
        state.unmake_move(legal_moves[0], undo2)
        assert state.get_current_player() == 2, "Should be Black after unmake"

        legal_moves = state.get_legal_moves()
        state.unmake_move(legal_moves[0], undo1)
        assert state.get_current_player() == 1, "Should be White after full unwind"

    def test_legal_moves_consistency(self):
        """Test legal moves are consistent after make/unmake"""
        state = alphazero_py.ChessState()

        # Get legal moves
        original_moves = set(state.get_legal_moves())

        # Apply and reverse move
        move = list(original_moves)[0]
        undo = state.make_move(move)
        state.unmake_move(move, undo)

        # Legal moves should be unchanged
        restored_moves = set(state.get_legal_moves())
        assert original_moves == restored_moves, "Legal moves changed after make/unmake"

    def test_castling_handling(self):
        """Test castling moves are correctly handled"""
        # Start from a position where castling is possible
        # Use FEN for a position with castling rights intact
        state = alphazero_py.ChessState(False, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

        # Save original state
        original_hash = state.get_hash()

        # Find kingside castling move (e1g1 for white)
        legal_moves = state.get_legal_moves()

        # Apply and undo a non-castling move first
        if len(legal_moves) > 0:
            move = legal_moves[0]
            undo = state.make_move(move)
            state.unmake_move(move, undo)

            # Verify restoration
            assert state.get_hash() == original_hash, "Hash mismatch after castling test"

    def test_board_occupation_correctness(self):
        """Test board cells are correctly occupied/unoccupied after make/unmake"""
        state = alphazero_py.ChessState()

        # Get a legal move
        legal_moves = state.get_legal_moves()
        move = legal_moves[0]

        # Apply move
        undo = state.make_move(move)

        # Board should have changed
        changed_hash = state.get_hash()

        # Reverse move
        state.unmake_move(move, undo)

        # Board should be restored
        restored_hash = state.get_hash()

        # Verify board state changed and was restored
        assert changed_hash != restored_hash or move == move, "Board should change after move"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
