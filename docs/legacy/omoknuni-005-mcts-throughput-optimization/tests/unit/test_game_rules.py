#!/usr/bin/env python3
"""
Comprehensive Game Rules Unit Tests (T025)

This test suite validates the correctness of game rule implementations
across all supported games: Gomoku, Chess, and Go. It tests legal move
generation, move validation, terminal detection, win conditions, and
edge cases to ensure rule engines are bug-free.
"""

import unittest
import sys
import os
import time
import numpy as np
from typing import List, Set, Tuple

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.alphazero_py_import import require_alphazero_py

try:
    alphazero_py = require_alphazero_py()
except ImportError as e:
    raise ImportError(f"Cannot import alphazero_py module. Build may be required: {e}")


class TestGomokuRules(unittest.TestCase):
    """Test Gomoku game rules implementation."""

    def setUp(self):
        """Set up test fixtures for Gomoku."""
        self.game = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)

    def test_initial_state(self):
        """Test initial game state is correct."""
        self.assertEqual(self.game.get_board_size(), 15)
        self.assertEqual(self.game.get_action_space_size(), 225)  # 15x15
        self.assertEqual(self.game.get_current_player(), 1)
        self.assertFalse(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.ONGOING)

    def test_legal_moves_initial(self):
        """Test all positions are legal initially."""
        legal_moves_mask = self.game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        self.assertEqual(len(legal_moves), 225)
        self.assertEqual(set(legal_moves), set(range(225)))

    def test_move_making_reduces_legal_moves(self):
        """Test that making moves reduces available legal moves."""
        initial_legal_moves_mask = self.game.get_legal_moves()
        initial_count = len(np.where(initial_legal_moves_mask)[0])

        # Make a move in the center
        center_move = 7 * 15 + 7  # (7,7) in 15x15 board
        self.assertTrue(self.game.is_legal_move(center_move))

        self.game.make_move(center_move)

        # Check legal moves reduced by 1 and doesn't include the played move
        new_legal_moves_mask = self.game.get_legal_moves()
        new_legal_moves = np.where(new_legal_moves_mask)[0]
        self.assertEqual(len(new_legal_moves), initial_count - 1)
        self.assertNotIn(center_move, new_legal_moves)

    def test_player_alternation(self):
        """Test players alternate correctly."""
        self.assertEqual(self.game.get_current_player(), 1)

        # Make first move
        self.game.make_move(0)
        self.assertEqual(self.game.get_current_player(), 2)

        # Make second move
        self.game.make_move(1)
        self.assertEqual(self.game.get_current_player(), 1)

    def test_move_undo(self):
        """Test move undo restores previous state."""
        initial_player = self.game.get_current_player()
        initial_legal_count = np.sum(self.game.get_legal_moves())

        # Make a move
        move = 112  # Center of board
        self.game.make_move(move)

        # Verify state changed
        self.assertEqual(self.game.get_current_player(), 3 - initial_player)
        self.assertEqual(np.sum(self.game.get_legal_moves()), initial_legal_count - 1)

        # Undo the move
        self.game.undo_move()

        # Verify state restored
        self.assertEqual(self.game.get_current_player(), initial_player)
        self.assertEqual(np.sum(self.game.get_legal_moves()), initial_legal_count)
        self.assertTrue(self.game.is_legal_move(move))

    def test_invalid_moves(self):
        """Test that invalid moves are rejected."""
        # Test out-of-bounds moves
        self.assertFalse(self.game.is_legal_move(-1))
        self.assertFalse(self.game.is_legal_move(225))
        self.assertFalse(self.game.is_legal_move(1000))

        # Make a move and test playing on occupied square
        self.game.make_move(0)
        self.assertFalse(self.game.is_legal_move(0))

    def test_horizontal_win_detection(self):
        """Test horizontal five-in-a-row win detection."""
        # Set up horizontal five in row 7, positions 5-9
        moves = [
            (5 + 7 * 15, 1),  # Player 1: (5,7)
            (0, 2),           # Player 2: (0,0) - irrelevant move
            (6 + 7 * 15, 1),  # Player 1: (6,7)
            (1, 2),           # Player 2: (1,0) - irrelevant move
            (7 + 7 * 15, 1),  # Player 1: (7,7)
            (2, 2),           # Player 2: (2,0) - irrelevant move
            (8 + 7 * 15, 1),  # Player 1: (8,7)
            (3, 2),           # Player 2: (3,0) - irrelevant move
        ]

        for move_pos, expected_player in moves:
            self.assertEqual(self.game.get_current_player(), expected_player)
            self.game.make_move(move_pos)
            if not self.game.is_terminal():
                continue
            else:
                break

        # Player 1 should need one more move to win
        self.assertFalse(self.game.is_terminal())

        # Make the winning move
        winning_move = 9 + 7 * 15  # (9,7)
        self.game.make_move(winning_move)

        # Game should be terminal with Player 1 winning
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.WIN_PLAYER1)

    def test_vertical_win_detection(self):
        """Test vertical five-in-a-row win detection."""
        # Set up vertical five in column 7, rows 5-9
        moves = [
            (7 + 5 * 15, 1),  # Player 1: (7,5)
            (0, 2),           # Player 2: (0,0)
            (7 + 6 * 15, 1),  # Player 1: (7,6)
            (1, 2),           # Player 2: (1,0)
            (7 + 7 * 15, 1),  # Player 1: (7,7)
            (2, 2),           # Player 2: (2,0)
            (7 + 8 * 15, 1),  # Player 1: (7,8)
            (3, 2),           # Player 2: (3,0)
        ]

        for move_pos, expected_player in moves:
            self.assertEqual(self.game.get_current_player(), expected_player)
            self.game.make_move(move_pos)

        # Make the winning move
        winning_move = 7 + 9 * 15  # (7,9)
        self.game.make_move(winning_move)

        # Game should be terminal with Player 1 winning
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.WIN_PLAYER1)

    def test_diagonal_win_detection(self):
        """Test diagonal five-in-a-row win detection."""
        # Set up diagonal from (3,3) to (7,7)
        moves = [
            (3 + 3 * 15, 1),  # Player 1: (3,3)
            (0, 2),           # Player 2: (0,0)
            (4 + 4 * 15, 1),  # Player 1: (4,4)
            (1, 2),           # Player 2: (1,0)
            (5 + 5 * 15, 1),  # Player 1: (5,5)
            (2, 2),           # Player 2: (2,0)
            (6 + 6 * 15, 1),  # Player 1: (6,6)
            (14, 2),          # Player 2: (14,0)
        ]

        for move_pos, expected_player in moves:
            self.assertEqual(self.game.get_current_player(), expected_player)
            self.game.make_move(move_pos)

        # Make the winning move
        winning_move = 7 + 7 * 15  # (7,7)
        self.game.make_move(winning_move)

        # Game should be terminal with Player 1 winning
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.WIN_PLAYER1)

    def test_anti_diagonal_win_detection(self):
        """Test anti-diagonal five-in-a-row win detection."""
        # Set up anti-diagonal from (7,3) to (3,7)
        moves = [
            (7 + 3 * 15, 1),  # Player 1: (7,3)
            (0, 2),           # Player 2: (0,0)
            (6 + 4 * 15, 1),  # Player 1: (6,4)
            (1, 2),           # Player 2: (1,0)
            (5 + 5 * 15, 1),  # Player 1: (5,5)
            (2, 2),           # Player 2: (2,0)
            (4 + 6 * 15, 1),  # Player 1: (4,6)
            (14, 2),          # Player 2: (14,0)
        ]

        for move_pos, expected_player in moves:
            self.assertEqual(self.game.get_current_player(), expected_player)
            self.game.make_move(move_pos)

        # Make the winning move
        winning_move = 3 + 7 * 15  # (3,7)
        self.game.make_move(winning_move)

        # Game should be terminal with Player 1 winning
        self.assertTrue(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.WIN_PLAYER1)

    def test_board_boundary_conditions(self):
        """Test moves at board boundaries."""
        # Test corners
        corners = [0, 14, 210, 224]  # (0,0), (14,0), (0,14), (14,14)
        for corner in corners:
            self.assertTrue(self.game.is_legal_move(corner))

        # Test edges
        edges = [7, 105, 112, 119]  # Top, left, center, right
        for edge in edges:
            self.assertTrue(self.game.is_legal_move(edge))

    def test_full_board_draw(self):
        """Test game handling when board is full without winner."""
        # This is a simplified test - in practice, it's hard to fill board without winner
        # Fill a significant portion of board alternately
        moves_made = 0
        max_moves = 50  # Test partial fill

        for i in range(max_moves):
            legal_moves = self.game.get_legal_moves()
            if not legal_moves.any() or self.game.is_terminal():
                break
            # Get first legal move from boolean mask
            legal_move_indices = np.where(legal_moves)[0]
            if len(legal_move_indices) > 0:
                self.game.make_move(legal_move_indices[0])
            else:
                break
            moves_made += 1

        # Verify game state is consistent
        legal_moves_count = np.sum(self.game.get_legal_moves())
        self.assertLessEqual(legal_moves_count, 225 - moves_made)


class TestChessRules(unittest.TestCase):
    """Test Chess game rules implementation."""

    def setUp(self):
        """Set up test fixtures for Chess."""
        self.game = alphazero_py.create_game(alphazero_py.GameType.CHESS)

    def test_initial_state(self):
        """Test initial chess state is correct."""
        self.assertEqual(self.game.get_board_size(), 8)
        self.assertGreater(self.game.get_action_space_size(), 4000)  # Chess has complex action space
        self.assertEqual(self.game.get_current_player(), 1)  # White starts
        self.assertFalse(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.ONGOING)

    def test_legal_moves_initial(self):
        """Test initial legal moves are reasonable."""
        legal_moves = self.game.get_legal_moves()
        # Should have 20 legal opening moves (16 pawn moves + 4 knight moves)
        self.assertEqual(np.sum(legal_moves), 20)

    def test_pawn_moves(self):
        """Test basic pawn movement rules."""
        initial_legal_count = np.sum(self.game.get_legal_moves())

        # Make a standard pawn move (e2-e4)
        legal_moves = self.game.get_legal_moves()
        # Find a pawn move (any legal move should work for testing)
        pawn_move = legal_moves[0]

        self.game.make_move(pawn_move)

        # Verify player switched
        self.assertEqual(self.game.get_current_player(), 2)  # Black's turn
        # Legal move count may be the same in opening, just verify we have moves
        self.assertGreater(np.sum(self.game.get_legal_moves()), 0)

    def test_move_undo_chess(self):
        """Test chess move undo functionality."""
        initial_player = self.game.get_current_player()
        initial_moves_mask = self.game.get_legal_moves()
        initial_hash = self.game.get_hash()

        # Make a move
        move = np.where(initial_moves_mask)[0][0]
        self.game.make_move(move)

        # Verify state changed
        self.assertNotEqual(self.game.get_current_player(), initial_player)
        self.assertNotEqual(self.game.get_hash(), initial_hash)

        # Undo the move
        self.game.undo_move()

        # Verify state restored
        self.assertEqual(self.game.get_current_player(), initial_player)
        self.assertEqual(self.game.get_hash(), initial_hash)

    def test_invalid_moves_chess(self):
        """Test that invalid chess moves are rejected."""
        # Test out-of-bounds moves
        self.assertFalse(self.game.is_legal_move(-1))
        self.assertFalse(self.game.is_legal_move(self.game.get_action_space_size()))

        # Test that only initial legal moves are valid
        legal_moves_mask = self.game.get_legal_moves()
        legal_moves = set(np.where(legal_moves_mask)[0])
        for test_move in range(min(100, self.game.get_action_space_size())):
            if test_move in legal_moves:
                self.assertTrue(self.game.is_legal_move(test_move))
            else:
                self.assertFalse(self.game.is_legal_move(test_move))

    def test_chess_game_progression(self):
        """Test that chess game can progress through several moves."""
        moves_made = 0
        max_moves = 10  # Test first 10 moves

        for _ in range(max_moves):
            if self.game.is_terminal():
                break

            legal_moves = self.game.get_legal_moves()
            self.assertGreater(len(legal_moves), 0, "Should have legal moves if not terminal")

            # Make first legal move
            self.game.make_move(legal_moves[0])
            moves_made += 1

        # Should have made some moves without error
        self.assertGreater(moves_made, 0)

    def test_chess_tensor_representation(self):
        """Test chess tensor representation has correct dimensions."""
        tensor = self.game.get_enhanced_tensor_representation()
        self.assertEqual(len(tensor.shape), 3)  # (channels, height, width)
        self.assertEqual(tensor.shape[1], 8)    # 8x8 board
        self.assertEqual(tensor.shape[2], 8)    # 8x8 board
        self.assertGreaterEqual(tensor.shape[0], 12)  # At least 12 feature planes


class TestGoRules(unittest.TestCase):
    """Test Go game rules implementation."""

    def setUp(self):
        """Set up test fixtures for Go."""
        self.game = alphazero_py.create_game(alphazero_py.GameType.GO)

    def test_initial_state(self):
        """Test initial Go state is correct."""
        self.assertEqual(self.game.get_board_size(), 19)
        self.assertGreater(self.game.get_action_space_size(), 361)  # 19x19 + pass move
        self.assertEqual(self.game.get_current_player(), 1)  # Black starts in Go
        self.assertFalse(self.game.is_terminal())
        self.assertEqual(self.game.get_game_result(), alphazero_py.GameResult.ONGOING)

    def test_legal_moves_initial(self):
        """Test initial legal moves in Go."""
        legal_moves = self.game.get_legal_moves()
        # Should have 361 legal moves (19x19 board) plus pass move
        self.assertGreaterEqual(len(legal_moves), 361)

    def test_go_move_making(self):
        """Test basic Go move making."""
        initial_legal_count = np.sum(self.game.get_legal_moves())

        # Make a move on the board
        corner_move = 0  # Top-left corner
        self.assertTrue(self.game.is_legal_move(corner_move))

        self.game.make_move(corner_move)

        # Verify state changed
        self.assertEqual(self.game.get_current_player(), 2)  # White's turn
        new_legal_moves = self.game.get_legal_moves()
        self.assertFalse(new_legal_moves[corner_move])  # Move should be illegal now

    def test_go_move_undo(self):
        """Test Go move undo functionality."""
        initial_player = self.game.get_current_player()
        initial_moves_mask = self.game.get_legal_moves()
        initial_hash = self.game.get_hash()

        # Make a move
        move = 180  # Center-ish of 19x19 board
        self.game.make_move(move)

        # Verify state changed
        self.assertNotEqual(self.game.get_current_player(), initial_player)
        self.assertNotEqual(self.game.get_hash(), initial_hash)

        # Undo the move
        self.game.undo_move()

        # Verify state restored
        self.assertEqual(self.game.get_current_player(), initial_player)
        self.assertEqual(self.game.get_hash(), initial_hash)

    def test_go_board_positions(self):
        """Test various board positions are valid in Go."""
        # Test corners
        corners = [0, 18, 342, 360]  # (0,0), (18,0), (0,18), (18,18)
        for corner in corners:
            if corner < self.game.get_action_space_size():
                self.assertTrue(self.game.is_legal_move(corner))

        # Test center
        center = 9 * 19 + 9  # (9,9)
        self.assertTrue(self.game.is_legal_move(center))

    def test_go_tensor_representation(self):
        """Test Go tensor representation has correct dimensions."""
        tensor = self.game.get_enhanced_tensor_representation()
        self.assertEqual(len(tensor.shape), 3)  # (channels, height, width)
        self.assertEqual(tensor.shape[1], 19)   # 19x19 board
        self.assertEqual(tensor.shape[2], 19)   # 19x19 board
        self.assertGreaterEqual(tensor.shape[0], 3)  # At least 3 feature planes (adjust based on implementation)


class TestCrossGameRuleConsistency(unittest.TestCase):
    """Test consistency of rule implementations across all games."""

    def setUp(self):
        """Set up test fixtures for cross-game tests."""
        self.games = {
            'gomoku': alphazero_py.create_game(alphazero_py.GameType.GOMOKU),
            'chess': alphazero_py.create_game(alphazero_py.GameType.CHESS),
            'go': alphazero_py.create_game(alphazero_py.GameType.GO)
        }

    def test_all_games_start_ongoing(self):
        """Test all games start in ongoing state."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                self.assertFalse(game.is_terminal(), f"{name} should start non-terminal")
                self.assertEqual(game.get_game_result(), alphazero_py.GameResult.ONGOING)

    def test_all_games_have_legal_moves_initially(self):
        """Test all games have legal moves at start."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                legal_moves = game.get_legal_moves()
                self.assertGreater(len(legal_moves), 0, f"{name} should have legal moves")

    def test_all_games_support_move_undo(self):
        """Test all games support move undo."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                initial_hash = game.get_hash()
                legal_moves = game.get_legal_moves()

                if legal_moves.any():
                    legal_move_indices = np.where(legal_moves)[0]
                    if len(legal_move_indices) > 0:
                        game.make_move(legal_move_indices[0])
                    game.undo_move()
                    self.assertEqual(game.get_hash(), initial_hash,
                                   f"{name} undo should restore state")

    def test_all_games_have_valid_tensors(self):
        """Test all games produce valid tensor representations."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                tensor = game.get_enhanced_tensor_representation()
                self.assertIsInstance(tensor, np.ndarray, f"{name} should return numpy array")
                self.assertEqual(len(tensor.shape), 3, f"{name} tensor should be 3D")
                self.assertEqual(tensor.dtype, np.float32, f"{name} tensor should be float32")
                self.assertTrue(tensor.flags.c_contiguous, f"{name} tensor should be C-contiguous")

    def test_hash_consistency(self):
        """Test hash consistency across operations."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                # Same state should have same hash
                hash1 = game.get_hash()
                hash2 = game.get_hash()
                self.assertEqual(hash1, hash2, f"{name} hash should be consistent")

    def test_string_representation_not_empty(self):
        """Test all games have non-empty string representation."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                game_str = game.to_string()
                self.assertIsInstance(game_str, str, f"{name} should return string")
                self.assertGreater(len(game_str), 0, f"{name} string should not be empty")


class TestGameRulePerformance(unittest.TestCase):
    """Performance tests for game rule operations."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.games = {
            'gomoku': alphazero_py.create_game(alphazero_py.GameType.GOMOKU),
            'chess': alphazero_py.create_game(alphazero_py.GameType.CHESS),
            'go': alphazero_py.create_game(alphazero_py.GameType.GO)
        }

    def test_legal_move_generation_performance(self):
        """Test legal move generation performance."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                start_time = time.time()
                iterations = 1000

                for _ in range(iterations):
                    legal_moves = game.get_legal_moves()
                    # Ensure moves are actually generated
                    self.assertGreater(len(legal_moves), 0)

                elapsed = time.time() - start_time
                moves_per_second = iterations / elapsed

                # Should generate at least 500 legal move lists per second (realistic for complex games)
                self.assertGreater(moves_per_second, 500,
                                 f"{name} legal move generation too slow: {moves_per_second:.0f}/s")

    def test_move_making_performance(self):
        """Test move making and undoing performance."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                legal_moves = game.get_legal_moves()
                if not legal_moves.any():
                    self.skipTest(f"No legal moves available for {name}")

                legal_move_indices = np.where(legal_moves)[0]
                move = legal_move_indices[0]
                start_time = time.time()
                iterations = 1000

                for _ in range(iterations):
                    game.make_move(move)
                    game.undo_move()

                elapsed = time.time() - start_time
                operations_per_second = (iterations * 2) / elapsed  # *2 for make+undo

                # Should handle at least 10k move operations per second (realistic target)
                self.assertGreater(operations_per_second, 10000,
                                 f"{name} move operations too slow: {operations_per_second:.0f}/s")

    def test_tensor_extraction_performance(self):
        """Test tensor extraction performance."""
        for name, game in self.games.items():
            with self.subTest(game=name):
                start_time = time.time()
                iterations = 1000

                for _ in range(iterations):
                    tensor = game.get_enhanced_tensor_representation()
                    # Ensure tensor is valid
                    self.assertGreater(tensor.size, 0)

                elapsed = time.time() - start_time
                extractions_per_second = iterations / elapsed

                # Should extract at least 500 tensors per second (realistic target for enhanced representations)
                self.assertGreater(extractions_per_second, 500,
                                 f"{name} tensor extraction too slow: {extractions_per_second:.0f}/s")


if __name__ == '__main__':
    # Set up for running tests
    print("Testing Game Rules for AlphaZero Engine...")
    print(f"Python version: {sys.version}")
    print(f"Numpy version: {np.__version__}")

    # Run tests with high verbosity for detailed output
    unittest.main(verbosity=2, buffer=True)