"""
Real Implementation Tests for Game Interface
===========================================

Tests all game implementations (Gomoku, Chess, Go) with actual C++ code.
No mocks - validates production game logic and tensor extraction.
"""

import pytest
import numpy as np
import alphazero_py
from src.games.game_state import create_game_state, GameStateWrapper


class TestRealGameInterface:
    """Test real C++ game implementations."""

    def test_gomoku_real_implementation(self):
        """Test real Gomoku implementation."""
        game = create_game_state('gomoku')

        # Test basic properties
        assert isinstance(game, GameStateWrapper)
        assert game.action_space_size == 225  # 15x15
        assert game.get_current_player() in [1, 2]
        assert not game.is_terminal()

        # Test legal moves
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 225  # All positions legal at start
        assert all(0 <= move < 225 for move in legal_moves)

        # Test tensor representation
        features = game.get_features()
        assert features.shape == (36, 15, 15)  # Gomoku features
        assert features.dtype == np.float32

        # Test move application
        center_move = 112  # Center position (7, 7)
        new_game = game.make_move(center_move)
        assert new_game.get_current_player() != game.get_current_player()
        assert len(new_game.get_legal_moves()) == 224  # One less legal move

    def test_chess_real_implementation(self):
        """Test real Chess implementation."""
        game = create_game_state('chess')

        # Test basic properties
        assert isinstance(game, GameStateWrapper)
        assert game.action_space_size == 20480  # Chess action space
        assert game.get_current_player() in [1, 2]
        assert not game.is_terminal()

        # Test legal moves (starting position should have ~20 legal moves)
        legal_moves = game.get_legal_moves()
        assert 15 <= len(legal_moves) <= 25  # Reasonable range for chess start
        assert all(0 <= move < 20480 for move in legal_moves)

        # Test tensor representation
        features = game.get_features()
        assert features.shape == (30, 8, 8)  # Chess features
        assert features.dtype == np.float32

        # Test move application
        first_move = legal_moves[0]
        new_game = game.make_move(first_move)
        assert new_game.get_current_player() != game.get_current_player()

    def test_go_real_implementation(self):
        """Test real Go implementation."""
        game = create_game_state('go', board_size=9)

        # Test basic properties
        assert isinstance(game, GameStateWrapper)
        assert game.action_space_size == 82  # 9x9 + pass move
        assert game.get_current_player() in [1, 2]
        assert not game.is_terminal()

        # Test legal moves (9x9 Go should have 81 positions + pass)
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) <= 82
        assert all(-1 <= move < 82 for move in legal_moves)  # -1 is pass move

        # Test tensor representation
        features = game.get_features()
        assert features.shape == (25, 9, 9)  # Go features
        assert features.dtype == np.float32

        # Test move application (use a valid position, not pass)
        valid_moves = [m for m in legal_moves if m >= 0]
        if valid_moves:
            corner_move = valid_moves[0]  # First valid position
            new_game = game.make_move(corner_move)
            assert new_game.get_current_player() != game.get_current_player()
            assert len(new_game.get_legal_moves()) <= len(legal_moves)

    def test_gomoku_game_progression(self):
        """Test a complete game progression in Gomoku."""
        game = create_game_state('gomoku')

        moves_played = []
        current_game = game

        # Play several moves
        for i in range(10):
            legal_moves = current_game.get_legal_moves()
            assert len(legal_moves) > 0, f"Should have legal moves on turn {i}"

            # Make a move (choose first legal move for consistency)
            move = legal_moves[0]
            moves_played.append(move)
            current_game = current_game.make_move(move)

            # Verify game state progression
            assert len(current_game.get_legal_moves()) == len(legal_moves) - 1
            assert current_game.get_current_player() != (1 if i % 2 == 0 else 2)

        # Game should not be terminal after 10 moves (unlikely to win)
        assert not current_game.is_terminal()

    def test_chess_opening_moves(self):
        """Test Chess opening moves work correctly."""
        game = create_game_state('chess')

        # Test e2-e4 equivalent (find pawn push move)
        legal_moves = game.get_legal_moves()

        # Make first move (any legal move)
        current_game = game.make_move(legal_moves[0])
        assert current_game.get_current_player() == 2  # Should be black's turn

        # Make second move
        black_legal_moves = current_game.get_legal_moves()
        current_game = current_game.make_move(black_legal_moves[0])
        assert current_game.get_current_player() == 1  # Back to white

        # Verify game is progressing normally
        assert not current_game.is_terminal()
        assert len(current_game.get_legal_moves()) > 0

    def test_go_capture_mechanics(self):
        """Test Go basic mechanics (placing stones)."""
        game = create_game_state('go', board_size=9)

        # Place stone in corner
        corner_game = game.make_move(0)  # Top-left corner
        assert corner_game.get_current_player() == 2  # Should be white's turn

        # Place adjacent stone
        adjacent_move = 1  # Adjacent to corner
        if adjacent_move in corner_game.get_legal_moves():
            final_game = corner_game.make_move(adjacent_move)
            assert final_game.get_current_player() == 1  # Back to black

    def test_game_state_cloning(self):
        """Test game state cloning works correctly."""
        original = create_game_state('gomoku')

        # Make some moves on original
        move1 = original.make_move(112)  # Center
        move2 = move1.make_move(113)     # Adjacent

        # Clone at different points
        original_clone = original.clone()
        move1_clone = move1.clone()
        move2_clone = move2.clone()

        # Clones should be independent
        assert original_clone.get_current_player() == original.get_current_player()
        assert move1_clone.get_current_player() == move1.get_current_player()
        assert move2_clone.get_current_player() == move2.get_current_player()

        # Clones should have same legal moves
        assert len(original_clone.get_legal_moves()) == len(original.get_legal_moves())
        assert len(move1_clone.get_legal_moves()) == len(move1.get_legal_moves())

    def test_cross_game_interface_consistency(self):
        """Test that all games implement the same interface consistently."""
        games = [
            create_game_state('gomoku'),
            create_game_state('chess'),
            create_game_state('go', board_size=9)
        ]

        for game in games:
            # All games should implement core interface
            assert hasattr(game, 'get_legal_moves')
            assert hasattr(game, 'make_move')
            assert hasattr(game, 'is_terminal')
            assert hasattr(game, 'get_current_player')
            assert hasattr(game, 'get_features')
            assert hasattr(game, 'clone')

            # Basic interface behavior should be consistent
            assert isinstance(game.get_legal_moves(), list)
            assert len(game.get_legal_moves()) > 0
            assert game.get_current_player() in [1, 2]
            assert not game.is_terminal()  # New games shouldn't be terminal
            assert isinstance(game.get_features(), np.ndarray)
            assert game.action_space_size > 0

    def test_feature_tensor_properties(self):
        """Test that feature tensors have correct properties."""
        games_and_shapes = [
            (create_game_state('gomoku'), (36, 15, 15)),
            (create_game_state('chess'), (30, 8, 8)),
            (create_game_state('go', board_size=9), (25, 9, 9))
        ]

        for game, expected_shape in games_and_shapes:
            features = game.get_features()

            # Check shape
            assert features.shape == expected_shape

            # Check data type
            assert features.dtype == np.float32

            # Check values are reasonable (not all zeros, not all ones)
            assert not np.all(features == 0)
            assert np.all(np.isfinite(features))  # No NaN or inf values

    def test_game_result_detection(self):
        """Test game result detection for terminal states."""
        # Create games and check initial state
        for game_type in ['gomoku', 'chess']:
            game = create_game_state(game_type)

            # New game should not be terminal
            assert not game.is_terminal()
            assert game.get_result() is None

            # After some moves, still shouldn't be terminal (statistically unlikely)
            current_game = game
            for _ in range(3):
                legal_moves = current_game.get_legal_moves()
                if legal_moves:
                    current_game = current_game.make_move(legal_moves[0])

            # Should still not be terminal after 3 moves
            assert not current_game.is_terminal()

    def test_alphazero_py_direct_access(self):
        """Test direct access to alphazero_py functionality."""
        # Test creating games directly through C++ interface
        gomoku_cpp = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)
        chess_cpp = alphazero_py.create_game(alphazero_py.GameType.CHESS)
        go_cpp = alphazero_py.create_game(alphazero_py.GameType.GO)

        # All should be valid C++ game states
        assert hasattr(gomoku_cpp, 'get_legal_moves')
        assert hasattr(chess_cpp, 'get_legal_moves')
        assert hasattr(go_cpp, 'get_legal_moves')

        # Test basic C++ functionality
        gomoku_moves = gomoku_cpp.get_legal_moves()
        chess_moves = chess_cpp.get_legal_moves()
        go_moves = go_cpp.get_legal_moves()

        # For boolean masks, count True values; for lists, use length
        gomoku_count = sum(gomoku_moves) if hasattr(gomoku_moves, '__iter__') and len(gomoku_moves) > 100 else len(gomoku_moves)
        chess_count = sum(chess_moves) if hasattr(chess_moves, '__iter__') and len(chess_moves) > 100 else len(chess_moves)
        go_count = sum(go_moves) if hasattr(go_moves, '__iter__') and len(go_moves) > 100 else len(go_moves)

        assert gomoku_count == 225  # All empty positions on 15x15 board
        assert 15 <= chess_count <= 25  # Reasonable number of chess moves
        assert go_count <= 362  # 19x19 + pass

    def test_game_factory_functions(self):
        """Test game creation with different parameters."""
        # Test Gomoku variants
        standard_gomoku = create_game_state('gomoku')
        renju_gomoku = create_game_state('gomoku', use_renju=True)

        assert standard_gomoku.action_space_size == 225
        assert renju_gomoku.action_space_size == 225

        # Test Chess variants
        standard_chess = create_game_state('chess')
        chess960 = create_game_state('chess', chess960=True)

        assert standard_chess.action_space_size == 20480
        assert chess960.action_space_size == 20480

        # Test Go variants
        small_go = create_game_state('go', board_size=9)
        standard_go = create_game_state('go', board_size=19)

        assert small_go.action_space_size == 82   # 9x9 + pass
        assert standard_go.action_space_size == 362  # 19x19 + pass


class TestGameStateWrapperSpecific:
    """Test GameStateWrapper specific functionality."""

    def test_wrapper_cpp_state_access(self):
        """Test access to underlying C++ state."""
        game = create_game_state('gomoku')

        # Should have access to C++ state
        cpp_state = game.cpp_state
        assert cpp_state is not None

        # C++ state should have same properties
        assert len(cpp_state.get_legal_moves()) == len(game.get_legal_moves())
        assert cpp_state.get_current_player() == game.get_current_player()

    def test_result_conversion(self):
        """Test game result conversion from C++ to Python."""
        game = create_game_state('gomoku')

        # New game should have no result
        assert game.get_result() is None

        # Terminal detection should work
        assert not game.is_terminal()

    def test_move_validation_integration(self):
        """Test move validation works with real C++ implementation."""
        game = create_game_state('gomoku')
        legal_moves = game.get_legal_moves()

        # All legal moves should be in valid range
        for move in legal_moves:
            assert 0 <= move < 225

        # Making legal moves should work
        if legal_moves:
            new_game = game.make_move(legal_moves[0])
            assert isinstance(new_game, GameStateWrapper)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])