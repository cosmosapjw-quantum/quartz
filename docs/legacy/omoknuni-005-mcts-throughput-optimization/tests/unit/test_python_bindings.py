#!/usr/bin/env python3
"""
Test suite for Python bindings (T024)

This test validates the pybind11 bindings for game implementations,
ensuring Python can interact with C++ game classes and numpy arrays
work correctly.
"""

import unittest
import sys
import os
import numpy as np

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.alphazero_py_import import require_alphazero_py

try:
    alphazero_py = require_alphazero_py()
except ImportError as e:
    raise ImportError(f"Cannot import alphazero_py module. Build may be required: {e}")


class TestPythonBindings(unittest.TestCase):
    """Test Python bindings for game implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = alphazero_py

    def test_module_import(self):
        """Test that the module imports successfully."""
        self.assertIsNotNone(self.module)
        self.assertTrue(hasattr(self.module, 'GameType'))
        self.assertTrue(hasattr(self.module, 'create_game'))

    def test_game_types(self):
        """Test GameType enum functionality."""
        # Test enum values exist
        self.assertTrue(hasattr(self.module.GameType, 'CHESS'))
        self.assertTrue(hasattr(self.module.GameType, 'GO'))
        self.assertTrue(hasattr(self.module.GameType, 'GOMOKU'))
        self.assertTrue(hasattr(self.module.GameType, 'UNKNOWN'))

        # Test game type conversion utilities
        self.assertEqual(self.module.game_type_to_string(self.module.GameType.CHESS), "chess")
        self.assertEqual(self.module.game_type_to_string(self.module.GameType.GO), "go")
        self.assertEqual(self.module.game_type_to_string(self.module.GameType.GOMOKU), "gomoku")

        self.assertEqual(self.module.string_to_game_type("chess"), self.module.GameType.CHESS)
        self.assertEqual(self.module.string_to_game_type("go"), self.module.GameType.GO)
        self.assertEqual(self.module.string_to_game_type("gomoku"), self.module.GameType.GOMOKU)
        self.assertEqual(self.module.string_to_game_type("invalid"), self.module.GameType.UNKNOWN)

    def test_game_creation(self):
        """Test creating game instances."""
        # Test creating each game type
        gomoku = self.module.create_game(self.module.GameType.GOMOKU)
        self.assertIsNotNone(gomoku)
        self.assertEqual(gomoku.get_board_size(), 15)

        chess = self.module.create_game(self.module.GameType.CHESS)
        self.assertIsNotNone(chess)
        self.assertEqual(chess.get_board_size(), 8)

        go = self.module.create_game(self.module.GameType.GO)
        self.assertIsNotNone(go)
        self.assertEqual(go.get_board_size(), 19)

    def test_game_interface_methods(self):
        """Test common game interface methods."""
        game = self.module.create_game(self.module.GameType.GOMOKU)

        # Test basic properties
        self.assertEqual(game.get_board_size(), 15)
        self.assertEqual(game.get_action_space_size(), 225)  # 15x15 board
        self.assertEqual(game.get_current_player(), 1)
        self.assertFalse(game.is_terminal())

        # Test legal moves
        legal_moves_mask = game.get_legal_moves()
        self.assertIsInstance(legal_moves_mask, np.ndarray)
        self.assertEqual(legal_moves_mask.dtype, bool)
        self.assertEqual(len(legal_moves_mask), 225)  # All positions initially legal
        legal_moves = np.where(legal_moves_mask)[0]
        self.assertEqual(len(legal_moves), 225)  # All positions initially legal
        self.assertTrue(all(isinstance(move, (int, np.integer)) for move in legal_moves))

        # Test move validation
        self.assertTrue(game.is_legal_move(0))
        self.assertTrue(game.is_legal_move(224))  # Last position

    def test_move_making_and_undo(self):
        """Test making and undoing moves."""
        game = self.module.create_game(self.module.GameType.GOMOKU)

        initial_player = game.get_current_player()
        legal_moves_mask = game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        first_move = legal_moves[0]

        # Make a move
        game.make_move(first_move)
        self.assertEqual(game.get_current_player(), 3 - initial_player)  # Player switched
        self.assertFalse(game.get_legal_moves()[first_move])  # Move no longer legal

        # Undo the move
        game.undo_move()
        self.assertEqual(game.get_current_player(), initial_player)  # Player restored
        self.assertTrue(game.get_legal_moves()[first_move])  # Move legal again

    def test_numpy_tensor_integration(self):
        """Test numpy array compatibility for neural network features."""
        game = self.module.create_game(self.module.GameType.GOMOKU)

        # Test tensor representation
        tensor = game.get_enhanced_tensor_representation()
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(len(tensor.shape), 3)  # (channels, height, width)
        self.assertEqual(tensor.shape[1], 15)  # Board height
        self.assertEqual(tensor.shape[2], 15)  # Board width

        # Test basic tensor representation
        basic_tensor = game.get_basic_tensor_representation()
        self.assertIsInstance(basic_tensor, np.ndarray)
        self.assertEqual(basic_tensor.dtype, np.float32)

        # Test enhanced tensor representation
        enhanced_tensor = game.get_enhanced_tensor_representation()
        self.assertIsInstance(enhanced_tensor, np.ndarray)
        self.assertEqual(enhanced_tensor.dtype, np.float32)

        # Tensors should have valid values (0 or 1 for game state encoding)
        self.assertTrue(np.all((tensor >= 0) & (tensor <= 1)))

    def test_game_specific_classes(self):
        """Test game-specific class instantiation."""
        # Test Gomoku-specific constructor
        gomoku = self.module.GomokuState()
        self.assertEqual(gomoku.get_board_size(), 15)

        gomoku_custom = self.module.GomokuState(19)
        self.assertEqual(gomoku_custom.get_board_size(), 19)

        # Test Chess-specific constructor
        chess = self.module.ChessState()
        self.assertEqual(chess.get_board_size(), 8)

        # Test Go-specific constructor
        go = self.module.GoState()
        self.assertEqual(go.get_board_size(), 19)

        go_small = self.module.GoState(9)
        self.assertEqual(go_small.get_board_size(), 9)

    def test_different_game_types(self):
        """Test that different games have appropriate characteristics."""
        gomoku = self.module.create_game(self.module.GameType.GOMOKU)
        chess = self.module.create_game(self.module.GameType.CHESS)
        go = self.module.create_game(self.module.GameType.GO)

        # Different board sizes
        self.assertEqual(gomoku.get_board_size(), 15)
        self.assertEqual(chess.get_board_size(), 8)
        self.assertEqual(go.get_board_size(), 19)

        # Different action space sizes
        self.assertEqual(gomoku.get_action_space_size(), 225)  # 15x15
        self.assertGreater(chess.get_action_space_size(), 0)   # Chess has complex action space
        self.assertGreater(go.get_action_space_size(), 300)    # 19x19 + pass move

        # All should start with player 1
        self.assertEqual(gomoku.get_current_player(), 1)
        self.assertEqual(chess.get_current_player(), 1)
        self.assertEqual(go.get_current_player(), 1)

    def test_game_string_conversion(self):
        """Test string conversion methods."""
        game = self.module.create_game(self.module.GameType.GOMOKU)

        # Test toString method
        game_str = game.to_string()
        self.assertIsInstance(game_str, str)
        self.assertGreater(len(game_str), 0)

        # Test action to string conversion
        legal_moves_mask = game.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        if len(legal_moves) > 0:
            action_str = game.action_to_string(legal_moves[0])
            self.assertIsInstance(action_str, str)

            # Test string to action conversion (round trip)
            action_back = game.string_to_action(action_str)
            self.assertEqual(action_back, legal_moves[0])

    def test_game_hash(self):
        """Test game state hashing."""
        game1 = self.module.create_game(self.module.GameType.GOMOKU)
        game2 = self.module.create_game(self.module.GameType.GOMOKU)

        # Initial states should have same hash
        hash1 = game1.get_hash()
        hash2 = game2.get_hash()
        self.assertEqual(hash1, hash2)

        # After different moves, hashes should differ
        legal_moves_mask = game1.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]
        if len(legal_moves) >= 2:
            game1.make_move(legal_moves[0])
            game2.make_move(legal_moves[1])
            self.assertNotEqual(game1.get_hash(), game2.get_hash())

    def test_game_cloning(self):
        """Test game state cloning."""
        original = self.module.create_game(self.module.GameType.GOMOKU)
        legal_moves_mask = original.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]

        if len(legal_moves) > 0:
            original.make_move(legal_moves[0])

            # Test clone
            clone = original.clone()
            self.assertIsNotNone(clone)
            self.assertEqual(original.get_current_player(), clone.get_current_player())
            self.assertEqual(original.get_hash(), clone.get_hash())

    def test_serialization_interface(self):
        """Test game serialization interface."""
        game = self.module.create_game(self.module.GameType.GOMOKU)

        # Test serialization
        serialized = self.module.serialize_game(game)
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 0)

        # Test deserialization
        deserialized = self.module.deserialize_game(serialized)
        self.assertIsNotNone(deserialized)
        self.assertEqual(game.get_hash(), deserialized.get_hash())

    def test_memory_management(self):
        """Test that objects can be created and destroyed without memory issues."""
        # Create many game instances to test memory management
        games = []
        for i in range(100):
            game_type = [self.module.GameType.GOMOKU, self.module.GameType.CHESS, self.module.GameType.GO][i % 3]
            game = self.module.create_game(game_type)
            games.append(game)

        # Use games to ensure they're not optimized away
        total_actions = sum(game.get_action_space_size() for game in games)
        self.assertGreater(total_actions, 0)


if __name__ == '__main__':
    # Set up for running tests
    print("Testing Python bindings for AlphaZero games...")
    print(f"Python version: {sys.version}")
    print(f"Numpy version: {np.__version__}")

    # Run tests
    unittest.main(verbosity=2)