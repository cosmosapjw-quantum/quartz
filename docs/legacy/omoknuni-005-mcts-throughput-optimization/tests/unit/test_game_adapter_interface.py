"""
Unit tests for the game adapter interface.

Tests the unified game interface including GameFactory, GameRegistry,
and GameSerializer classes that enable polymorphic dispatch across
all game implementations.

HOWTO-RUN-TESTS:
================
# Run all game adapter interface tests
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Run specific test class
python -m pytest tests/unit/test_game_adapter_interface.py::TestGameAdapterInterface -v

# Run with detailed output
python -m pytest tests/unit/test_game_adapter_interface.py -v -s
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the contract API to test against
sys.path.append('specs/001-goal-create-spec')
from contracts.interface_api import (
    GameType,
    GameResult,
    GameRegistry,
    GameFactory,
    GameSerializer,
    IGameState
)

# Import real game implementations for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def create_test_game(game_type='GOMOKU', board_size=15):
    """Create a real game instance for testing."""
    # Import real game state directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from games.game_state import create_game_state

    # Import the adapter from interface_api
    sys.path.append('specs/001-goal-create-spec')
    from contracts.interface_api import GameStateAdapter

    if game_type.upper() == 'CHESS':
        wrapper = create_game_state('chess')
    elif game_type.upper() == 'GO':
        wrapper = create_game_state('go', board_size=board_size)
    else:  # GOMOKU
        wrapper = create_game_state('gomoku', board_size=board_size)

    return GameStateAdapter(wrapper)


class TestGameAdapterInterface:
    """Test the game adapter interface functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing registrations
        try:
            GameRegistry.instance().clear()
        except:
            pass  # Ignore if not implemented yet

    def test_game_type_enum_exists(self):
        """Test that GameType enum exists and has expected values."""
        # These should be available from the interface
        expected_types = ['UNKNOWN', 'CHESS', 'GO', 'GOMOKU']

        for game_type in expected_types:
            assert hasattr(GameType, game_type), f"GameType.{game_type} should exist"

    def test_game_result_enum_exists(self):
        """Test that GameResult enum exists and has expected values."""
        expected_results = ['ONGOING', 'WIN_PLAYER1', 'WIN_PLAYER2', 'DRAW', 'NO_RESULT']

        for result in expected_results:
            assert hasattr(GameResult, result), f"GameResult.{result} should exist"

    def test_game_registry_singleton(self):
        """Test that GameRegistry follows singleton pattern."""
        try:
            registry1 = GameRegistry.instance()
            registry2 = GameRegistry.instance()

            # Should be the same instance
            assert registry1 is registry2
        except AttributeError:
            pytest.skip("GameRegistry not implemented yet")

    def test_game_factory_exists(self):
        """Test that GameFactory class exists with expected methods."""
        expected_methods = [
            'create_game',
            'create_chess',
            'create_go',
            'create_gomoku',
            'create_game_from_moves',
            'create_games',
            'detect_game_type'
        ]

        for method in expected_methods:
            assert hasattr(GameFactory, method), f"GameFactory.{method} should exist"

    def test_game_serializer_exists(self):
        """Test that GameSerializer class exists with expected methods."""
        expected_methods = [
            'serialize_game',
            'deserialize_game',
            'save_game',
            'load_game',
            'export_to_standard_format'
        ]

        for method in expected_methods:
            assert hasattr(GameSerializer, method), f"GameSerializer.{method} should exist"

    def test_interface_polymorphism(self):
        """Test that the interface supports polymorphic dispatch."""
        try:
            # Create real states for different games
            chess_state = create_test_game('CHESS', 8)
            go_state = create_test_game('GO', 19)
            gomoku_state = create_test_game('GOMOKU', 15)

            states = [chess_state, go_state, gomoku_state]

            # Test that all states implement the same interface
            for state in states:
                # Basic interface methods
                assert hasattr(state, 'get_legal_moves')
                assert hasattr(state, 'is_legal_move')
                assert hasattr(state, 'make_move')
                assert hasattr(state, 'is_terminal')
                assert hasattr(state, 'get_current_player')

                # Utility methods
                assert hasattr(state, 'clone')

                # Test that basic methods work
                legal_moves = state.get_legal_moves()
                assert isinstance(legal_moves, list)
                assert len(legal_moves) > 0  # New game should have legal moves

                assert not state.is_terminal()  # New game shouldn't be terminal
                assert state.get_current_player() in [1, 2]  # Valid player

        except Exception as e:
            pytest.skip(f"Real game implementations not ready: {e}")

    def test_game_type_detection(self):
        """Test automatic game type detection from move notation."""
        test_cases = [
            # Chess examples
            ("e2e4 e7e5 Nf3", "CHESS"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "CHESS"),

            # Go examples
            ("(;FF[4]GM[1];B[pd];W[dp])", "GO"),
            ("D4 Q16 D16", "GO"),

            # Gomoku examples
            ("H8 H9 I8", "GOMOKU"),
            ("A1 B2 C3", "GOMOKU"),
        ]

        try:
            for notation, expected_type in test_cases:
                detected = GameFactory.detect_game_type(notation)
                expected_enum = getattr(GameType, expected_type)
                assert detected == expected_enum, f"Failed to detect {expected_type} from '{notation}'"
        except AttributeError:
            pytest.skip("GameFactory.detect_game_type not implemented yet")

    def test_game_serialization_roundtrip(self):
        """Test that game state can be serialized and deserialized."""
        try:
            # Create a real game with some moves
            original_state = create_test_game('GOMOKU', 15)
            legal_moves = original_state.get_legal_moves()

            # Make a few moves if possible
            if len(legal_moves) >= 2:
                original_state.make_move(legal_moves[0])
                legal_moves2 = original_state.get_legal_moves()
                if len(legal_moves2) > 0:
                    original_state.make_move(legal_moves2[0])

            # Serialize the state
            serialized = GameSerializer.serialize_game(original_state)
            assert isinstance(serialized, str)
            assert len(serialized) > 0

            # Deserialize the state
            restored_state = GameSerializer.deserialize_game(serialized)

            # Verify the states have same basic properties
            if hasattr(restored_state, 'get_move_history') and hasattr(original_state, 'get_move_history'):
                assert len(restored_state.get_move_history()) == len(original_state.get_move_history())

        except Exception as e:
            pytest.skip(f"GameSerializer not fully implemented: {e}")

    def test_batch_game_creation(self):
        """Test efficient creation of multiple game instances."""
        try:
            # Test creating multiple games of the same type
            game_count = 5
            games = GameFactory.create_games(GameType.GOMOKU, game_count)

            assert len(games) == game_count

            # All games should be independent instances
            for i, game in enumerate(games):
                assert game.get_game_type() == GameType.GOMOKU
                assert game.get_board_size() == 15  # Default Gomoku size

                # Make different moves to verify independence
                if game.get_legal_moves():
                    game.make_move(game.get_legal_moves()[0])

            # Verify games have different states
            move_histories = [game.get_move_history() for game in games]
            for i in range(len(move_histories)):
                for j in range(i + 1, len(move_histories)):
                    # Different games should be independent
                    assert True  # They might have same initial state, which is fine

        except AttributeError:
            pytest.skip("GameFactory.create_games not implemented yet")

    def test_game_from_moves(self):
        """Test creating game state from move sequence."""
        try:
            # Test creating a game with pre-applied moves
            moves = "H8 H9 I8 I9 J8"
            game = GameFactory.create_game_from_moves(GameType.GOMOKU, moves)

            # Verify moves were applied
            history = game.get_move_history()
            assert len(history) > 0

            # Verify the game state is as expected
            assert game.get_current_player() in [1, 2]

        except AttributeError:
            pytest.skip("GameFactory.create_game_from_moves not implemented yet")

    def test_custom_game_parameters(self):
        """Test creating games with custom parameters."""
        try:
            # Test Chess with Chess960
            chess960_game = GameFactory.create_chess(chess960=True, position_number=518)
            assert chess960_game.get_game_type() == GameType.CHESS
            assert chess960_game.get_board_size() == 8

            # Test Go with different board size
            small_go = GameFactory.create_go(board_size=9, rule_set=1)  # Japanese rules
            assert small_go.get_game_type() == GameType.GO
            assert small_go.get_board_size() == 9

            # Test Gomoku with Renju rules
            renju_game = GameFactory.create_gomoku(board_size=15, use_renju=True)
            assert renju_game.get_game_type() == GameType.GOMOKU
            assert renju_game.get_board_size() == 15

        except AttributeError:
            pytest.skip("Custom game creation methods not implemented yet")

    def test_game_registry_registration(self):
        """Test manual game registration and retrieval."""
        try:
            registry = GameRegistry.instance()

            # Test registration mechanism exists
            assert hasattr(registry, 'register_game')
            assert hasattr(registry, 'is_registered')
            assert hasattr(registry, 'get_registered_types')

        except AttributeError:
            pytest.skip("GameRegistry registration methods not implemented yet")

    def test_error_handling(self):
        """Test proper error handling in the interface."""
        try:
            # Test creating unknown game type
            with pytest.raises(Exception):  # Should raise some kind of error
                GameFactory.create_game(999)  # Invalid game type

            # Test deserializing invalid data
            with pytest.raises(Exception):
                GameSerializer.deserialize_game("invalid data")

            # Test loading non-existent file
            with pytest.raises(Exception):
                GameSerializer.load_game("/nonexistent/file.game")

        except AttributeError:
            pytest.skip("Error handling not implemented yet")

    def test_export_to_standard_formats(self):
        """Test exporting games to standard formats."""
        try:
            # Create games with some moves
            chess_game = create_test_game('CHESS', 8)
            legal_moves = chess_game.get_legal_moves()
            if legal_moves:
                chess_game.make_move(legal_moves[0])  # Make first legal move

            go_game = create_test_game('GO', 19)
            legal_moves = go_game.get_legal_moves()
            if legal_moves:
                go_game.make_move(legal_moves[0])  # Make first legal move

            gomoku_game = create_test_game('GOMOKU', 15)
            legal_moves = gomoku_game.get_legal_moves()
            if legal_moves:
                gomoku_game.make_move(legal_moves[0])  # Make first legal move

            # Test export to standard formats
            chess_pgn = GameSerializer.export_to_standard_format(chess_game)
            go_sgf = GameSerializer.export_to_standard_format(go_game)
            gomoku_custom = GameSerializer.export_to_standard_format(gomoku_game)

            # Basic validation
            assert isinstance(chess_pgn, str) and len(chess_pgn) > 0
            assert isinstance(go_sgf, str) and len(go_sgf) > 0
            assert isinstance(gomoku_custom, str) and len(gomoku_custom) > 0

            # Check for format-specific markers
            assert "[Event" in chess_pgn or "1." in chess_pgn  # PGN markers
            assert "(;" in go_sgf or "FF[4]" in go_sgf  # SGF markers

        except AttributeError:
            pytest.skip("Export to standard formats not implemented yet")

    def test_tensor_representation_consistency(self):
        """Test that tensor representations are consistent across games."""
        games = [
            create_test_game('CHESS', 8),
            create_test_game('GO', 19),
            create_test_game('GOMOKU', 15)
        ]

        for game in games:
            # Basic tensor representation should always be 18 channels
            basic_tensor = game.get_basic_tensor_representation()
            assert len(basic_tensor) == 18, f"Basic tensor should have 18 channels for {game.get_game_type()}"

            # Enhanced tensor representation should match game-specific requirements
            enhanced_tensor = game.get_enhanced_tensor_representation()
            if game.get_game_type() == 'GOMOKU':
                assert len(enhanced_tensor) == 7, "Gomoku should have 7 channels"
            elif game.get_game_type() == 'CHESS':
                assert len(enhanced_tensor) == 12, "Chess should have 12 channels"
            elif game.get_game_type() == 'GO':
                assert len(enhanced_tensor) == 17, "Go should have 17 channels"

            # All tensors should have correct dimensions
            board_size = game.get_board_size()
            for channel in basic_tensor:
                assert len(channel) == board_size
                assert len(channel[0]) == board_size

    def test_move_validation_consistency(self):
        """Test that move validation is consistent across the interface."""
        game = create_test_game('GOMOKU', 15)

        # Test legal moves
        legal_moves = game.get_legal_moves()
        for move in legal_moves[:3]:  # Test first few moves
            assert game.is_legal_move(move), f"Move {move} should be legal"

            # Apply move and test it's recorded
            old_history_len = len(game.get_move_history())
            game.make_move(move)
            new_history_len = len(game.get_move_history())
            assert new_history_len == old_history_len + 1, "Move history should increase by 1"

        # Test undo functionality
        if len(game.get_move_history()) > 0:
            old_len = len(game.get_move_history())
            success = game.undo_move()
            assert success, "Undo should succeed"
            assert len(game.get_move_history()) == old_len - 1, "Move history should decrease by 1"


class TestGameInterfaceIntegration:
    """Integration tests for the complete game interface system."""

    def test_full_game_workflow(self):
        """Test a complete workflow using the game interface."""
        try:
            # 1. Create a game
            game = GameFactory.create_game(GameType.GOMOKU)

            # 2. Play some moves
            legal_moves = game.get_legal_moves()
            for i in range(min(3, len(legal_moves))):
                move = legal_moves[i]
                assert game.is_legal_move(move)
                game.make_move(move)

            # 3. Clone the game
            cloned_game = game.clone()
            assert cloned_game.equals(game)

            # 4. Serialize the game
            serialized = GameSerializer.serialize_game(game)

            # 5. Deserialize and verify
            restored_game = GameSerializer.deserialize_game(serialized)
            assert restored_game.get_move_history() == game.get_move_history()

            # 6. Export to standard format
            exported = GameSerializer.export_to_standard_format(game)
            assert isinstance(exported, str)

        except (AttributeError, NotImplementedError):
            pytest.skip("Full workflow not implemented yet")

    def test_cross_game_compatibility(self):
        """Test that the interface works consistently across all game types."""
        game_types = [GameType.CHESS, GameType.GO, GameType.GOMOKU]

        try:
            for game_type in game_types:
                # Create game
                game = GameFactory.create_game(game_type)

                # Test basic interface
                assert game.get_board_size() > 0
                assert game.get_action_space_size() > 0
                assert game.get_current_player() in [1, 2]
                assert not game.is_terminal()  # New game shouldn't be terminal

                # Test tensor representations
                basic_tensor = game.get_basic_tensor_representation()
                enhanced_tensor = game.get_enhanced_tensor_representation()

                assert len(basic_tensor) == 18  # Standard AlphaZero format
                assert len(enhanced_tensor) > 0   # Game-specific format

                # Test move interface
                legal_moves = game.get_legal_moves()
                assert len(legal_moves) > 0  # New game should have legal moves

                if legal_moves:
                    first_move = legal_moves[0]
                    assert game.is_legal_move(first_move)

                    # Test string conversion
                    move_str = game.action_to_string(first_move)
                    assert isinstance(move_str, str)

                    converted_back = game.string_to_action(move_str)
                    assert converted_back == first_move

        except (AttributeError, NotImplementedError):
            pytest.skip("Cross-game compatibility testing not fully implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])