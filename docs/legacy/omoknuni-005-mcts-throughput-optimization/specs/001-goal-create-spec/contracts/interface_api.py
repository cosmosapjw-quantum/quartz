"""
Game Adapter Interface API Contract

This file defines the contract for the unified game interface that enables
polymorphic dispatch across all game implementations (Chess, Go, Gomoku).

The interface provides:
- GameFactory: Creates game instances with type detection
- GameRegistry: Manages game type registration
- GameSerializer: Handles state serialization/deserialization
- Unified GameState interface for all games

This contract must be implemented to enable the MCTS algorithm to work
with any game without knowing specific implementation details.
"""

from enum import Enum
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod


class GameType(Enum):
    """Enumeration of supported game types."""
    UNKNOWN = 0
    CHESS = 1
    GO = 2
    GOMOKU = 3


class GameResult(Enum):
    """Enumeration of possible game results."""
    ONGOING = 0
    WIN_PLAYER1 = 1
    WIN_PLAYER2 = 2
    DRAW = 3
    NO_RESULT = 4  # For Japanese Go rules: triple ko, etc.


class IGameState(ABC):
    """
    Abstract interface for all game state implementations.

    This interface defines the operations that all game implementations
    must provide for use with the MCTS algorithm.
    """

    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """
        Get all legal moves in the current state.

        Returns:
            List of legal action integers
        """
        raise NotImplementedError()

    @abstractmethod
    def is_legal_move(self, action: int) -> bool:
        """
        Check if a specific move is legal.

        Args:
            action: The action to check

        Returns:
            True if legal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def make_move(self, action: int) -> None:
        """
        Execute a move, updating the game state.

        Args:
            action: The action to execute

        Raises:
            ValueError: If the action is illegal
        """
        raise NotImplementedError()

    @abstractmethod
    def undo_move(self) -> bool:
        """
        Undo the last move.

        Returns:
            True if a move was undone, False if no moves to undo
        """
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the game state is terminal.

        Returns:
            True if terminal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_game_result(self) -> GameResult:
        """
        Get the result of the game.

        Returns:
            Game result (should be ONGOING if not terminal)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Get the current player.

        Returns:
            Current player (1 or 2)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_board_size(self) -> int:
        """
        Get the board size.

        Returns:
            Board size (typically width/height)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get the action space size.

        Returns:
            Total number of possible actions
        """
        raise NotImplementedError()

    @abstractmethod
    def get_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get tensor representation for neural network.

        Returns:
            3D tensor: [channels][height][width]
        """
        raise NotImplementedError()

    @abstractmethod
    def get_basic_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get basic 18-channel AlphaZero tensor representation.

        Returns:
            3D tensor with 18 channels: [18][height][width]
        """
        raise NotImplementedError()

    @abstractmethod
    def get_enhanced_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get enhanced tensor representation with additional features.

        Returns:
            3D tensor with game-specific enhanced features
        """
        raise NotImplementedError()

    @abstractmethod
    def get_hash(self) -> int:
        """
        Get hash for transposition table.

        Returns:
            64-bit hash of current state
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> 'IGameState':
        """
        Create a deep copy of the current state.

        Returns:
            New copy of the game state
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_clone(self, count: int) -> List['IGameState']:
        """
        Create multiple deep copies efficiently.

        Args:
            count: Number of clones to create

        Returns:
            List of game state clones
        """
        raise NotImplementedError()

    @abstractmethod
    def copy_from(self, source: 'IGameState') -> None:
        """
        Copy state from another game state instance.

        Args:
            source: The source state to copy from

        Raises:
            ValueError: If game types don't match
        """
        raise NotImplementedError()

    @abstractmethod
    def action_to_string(self, action: int) -> str:
        """
        Convert action to string representation.

        Args:
            action: The action to convert

        Returns:
            String representation (e.g., "e2e4" in chess)
        """
        raise NotImplementedError()

    @abstractmethod
    def string_to_action(self, move_str: str) -> Optional[int]:
        """
        Convert string representation to action.

        Args:
            move_str: String representation

        Returns:
            Action integer, or None if invalid
        """
        raise NotImplementedError()

    @abstractmethod
    def to_string(self) -> str:
        """
        Get string representation of the state.

        Returns:
            Human-readable representation
        """
        raise NotImplementedError()

    @abstractmethod
    def equals(self, other: 'IGameState') -> bool:
        """
        Check equality with another game state.

        Args:
            other: The other game state

        Returns:
            True if equal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_move_history(self) -> List[int]:
        """
        Get the history of moves.

        Returns:
            List of actions that led to current state
        """
        raise NotImplementedError()

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the game state for consistency.

        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_bitboards(self) -> List[List[int]]:
        """
        Get bitboard representation.

        Returns:
            List of bitboards for each player
        """
        raise NotImplementedError()

    @abstractmethod
    def get_game_type(self) -> GameType:
        """
        Get the game type.

        Returns:
            Game type enum value
        """
        raise NotImplementedError()


class GameStateAdapter(IGameState):
    """Adapter to bridge GameStateWrapper to IGameState interface."""

    def __init__(self, wrapper):
        """Initialize adapter with GameStateWrapper instance."""
        self._wrapper = wrapper
        self._move_history = []

    def get_legal_moves(self) -> List[int]:
        """Get all legal moves in the current state."""
        return self._wrapper.get_legal_moves()

    def is_legal_move(self, action: int) -> bool:
        """Check if a specific move is legal."""
        return action in self.get_legal_moves()

    def make_move(self, action: int) -> None:
        """Execute a move, updating the game state."""
        if not self.is_legal_move(action):
            raise ValueError(f"Illegal move: {action}")
        new_wrapper = self._wrapper.make_move(action)
        self._wrapper = new_wrapper
        self._move_history.append(action)

    def undo_move(self) -> bool:
        """Undo the last move."""
        if not self._move_history:
            return False

        # Recreate game state from scratch without last move
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
        from games.game_state import create_game_state

        # Detect game type
        game_type_str = 'gomoku'  # Default
        if hasattr(self._wrapper._state, 'get_game_type'):
            game_type_str = self._wrapper._state.get_game_type().lower()

        # Recreate with moves except last one
        new_wrapper = create_game_state(game_type_str, board_size=self.get_board_size())
        for move in self._move_history[:-1]:
            new_wrapper = new_wrapper.make_move(move)

        self._wrapper = new_wrapper
        self._move_history.pop()
        return True

    def is_terminal(self) -> bool:
        """Check if the game state is terminal."""
        return self._wrapper.is_terminal()

    def get_game_result(self) -> GameResult:
        """Get the result of the game."""
        if not self.is_terminal():
            return GameResult.ONGOING

        result = self._wrapper.get_result()
        if result is None:
            return GameResult.ONGOING
        elif result == 0.0:
            return GameResult.DRAW
        elif result == 1.0:
            return GameResult.WIN_PLAYER1 if self.get_current_player() == 1 else GameResult.WIN_PLAYER2
        else:  # result == -1.0
            return GameResult.WIN_PLAYER2 if self.get_current_player() == 1 else GameResult.WIN_PLAYER1

    def get_current_player(self) -> int:
        """Get the current player."""
        cpp_player = self._wrapper.get_current_player()
        return 1 if cpp_player == 0 else 2  # Convert 0-indexed to 1-indexed (1 or 2 only)

    def get_board_size(self) -> int:
        """Get the board size."""
        action_space = self._wrapper.action_space_size

        # Game-specific board size detection
        if action_space == 20480:  # Chess full action space
            return 8
        elif action_space == 225:  # 15x15 gomoku
            return 15
        elif action_space == 361 or action_space == 362:  # 19x19 go (with pass move)
            return 19
        elif action_space == 81 or action_space == 82:  # 9x9 go (with pass move)
            return 9
        elif action_space == 169:  # 13x13 go
            return 13
        elif action_space == 64:  # Chess simple or 8x8 board
            return 8

        # Try to infer from C++ state if available
        if hasattr(self._wrapper, '_state'):
            cpp_state = self._wrapper._state
            if hasattr(cpp_state, '__class__'):
                class_name = cpp_state.__class__.__name__.lower()
                if 'chess' in class_name:
                    return 8
                elif 'go' in class_name:
                    # For Go, try to infer from action space
                    if action_space > 300:
                        return 19
                    elif action_space > 150:
                        return 13
                    else:
                        return 9

        # Square board fallback
        return int(action_space ** 0.5)

    def get_action_space_size(self) -> int:
        """Get the action space size."""
        return self._wrapper.action_space_size

    def get_tensor_representation(self) -> List[List[List[float]]]:
        """Get tensor representation for neural network."""
        features = self._wrapper.get_features()
        # Ensure proper 3D shape: [channels, height, width]
        if len(features.shape) == 3:
            return features.tolist()
        elif len(features.shape) == 2:
            # Single channel case
            return [features.tolist()]
        else:
            # Flatten and reshape to board dimensions
            board_size = self.get_board_size()
            channels = features.shape[0]
            reshaped = features.reshape(channels, board_size, board_size)
            return reshaped.tolist()

    def get_basic_tensor_representation(self) -> List[List[List[float]]]:
        """Get basic 18-channel AlphaZero tensor representation."""
        features = self._wrapper.get_features()
        board_size = self.get_board_size()

        # Reshape features to proper dimensions
        if len(features.shape) == 3:
            reshaped = features
        else:
            channels = features.shape[0]
            reshaped = features.reshape(channels, board_size, board_size)

        # Take first 18 channels or pad if fewer
        if reshaped.shape[0] >= 18:
            return reshaped[:18].tolist()
        else:
            # Pad with zeros
            import numpy as np
            padded = np.zeros((18, board_size, board_size))
            padded[:reshaped.shape[0]] = reshaped
            return padded.tolist()

    def get_enhanced_tensor_representation(self) -> List[List[List[float]]]:
        """Get enhanced tensor representation with additional features."""
        features = self._wrapper.get_features()
        board_size = self.get_board_size()

        # Reshape features to proper dimensions
        if len(features.shape) == 3:
            reshaped = features
        else:
            channels = features.shape[0]
            reshaped = features.reshape(channels, board_size, board_size)

        # Return game-specific number of channels
        game_type = self.get_game_type()
        if game_type == GameType.GOMOKU:
            target_channels = min(7, reshaped.shape[0])
            return reshaped[:target_channels].tolist()
        elif game_type == GameType.CHESS:
            target_channels = min(12, reshaped.shape[0])
            return reshaped[:target_channels].tolist()
        elif game_type == GameType.GO:
            target_channels = min(17, reshaped.shape[0])
            return reshaped[:target_channels].tolist()
        else:
            return reshaped.tolist()

    def get_hash(self) -> int:
        """Get hash for transposition table."""
        # Simple hash based on move history
        return hash(tuple(self._move_history))

    def clone(self) -> 'GameStateAdapter':
        """Create a deep copy of the current state."""
        new_wrapper = self._wrapper.clone()
        new_adapter = GameStateAdapter(new_wrapper)
        new_adapter._move_history = self._move_history.copy()
        return new_adapter

    def batch_clone(self, count: int) -> List['GameStateAdapter']:
        """Create multiple deep copies efficiently."""
        return [self.clone() for _ in range(count)]

    def copy_from(self, source: 'IGameState') -> None:
        """Copy state from another game state instance."""
        if not isinstance(source, GameStateAdapter):
            raise ValueError("Can only copy from GameStateAdapter instances")
        self._wrapper = source._wrapper.clone()
        self._move_history = source._move_history.copy()

    def action_to_string(self, action: int) -> str:
        """Convert action to string representation."""
        game_type = self.get_game_type()

        if game_type == GameType.CHESS:
            # For chess, use simple index representation since chess moves are complex
            return str(action)
        elif game_type == GameType.GO and action == -1:
            # Handle Go pass move
            return "PASS"
        else:
            # Simple coordinate notation for Go/Gomoku
            board_size = self.get_board_size()
            if action < 0:
                return f"SPECIAL_{action}"
            row = action // board_size
            col = action % board_size
            return f"{chr(ord('A') + col)}{row + 1}"

    def string_to_action(self, move_str: str) -> Optional[int]:
        """Convert string representation to action."""
        # Handle special moves
        if move_str.upper() == "PASS":
            if -1 in self.get_legal_moves():
                return -1
            return None

        if move_str.startswith("SPECIAL_"):
            try:
                action = int(move_str[8:])  # Remove "SPECIAL_" prefix
                if action in self.get_legal_moves():
                    return action
            except ValueError:
                pass

        try:
            # Try coordinate notation
            if len(move_str) >= 2:
                col = ord(move_str[0].upper()) - ord('A')
                row = int(move_str[1:]) - 1
                board_size = self.get_board_size()
                if 0 <= row < board_size and 0 <= col < board_size:
                    action = row * board_size + col
                    # Verify action is actually legal
                    if action in self.get_legal_moves():
                        return action
                    # If not legal, try different coordinate systems
                    # Try column-major ordering
                    action = col * board_size + row
                    if action in self.get_legal_moves():
                        return action
        except (ValueError, IndexError):
            pass

        # Try numeric parsing as fallback
        try:
            action = int(move_str)
            if action in self.get_legal_moves():
                return action
        except ValueError:
            pass

        return None

    def to_string(self) -> str:
        """Get string representation of the state."""
        return f"GameState(type={self.get_game_type()}, moves={len(self._move_history)}, player={self.get_current_player()})"

    def equals(self, other: 'IGameState') -> bool:
        """Check equality with another game state."""
        if not isinstance(other, GameStateAdapter):
            return False
        return (self.get_game_type() == other.get_game_type() and
                self._move_history == other._move_history)

    def get_move_history(self) -> List[int]:
        """Get the history of moves."""
        return self._move_history.copy()

    def validate(self) -> bool:
        """Validate the game state for consistency."""
        return True  # Assume wrapper handles validation

    def get_bitboards(self) -> List[List[int]]:
        """Get bitboard representation."""
        # Simplified bitboard representation
        return [[0] * 64, [0] * 64]  # Two 64-bit boards for two players

    def get_game_type(self) -> GameType:
        """Get the game type."""
        action_space = self.get_action_space_size()

        # Precise action space detection based on actual values
        if action_space == 20480:  # Chess full action space
            return GameType.CHESS
        elif action_space in [361, 362, 81, 82, 169]:  # Go (various sizes with/without pass)
            return GameType.GO
        elif action_space == 225:  # Gomoku 15x15
            return GameType.GOMOKU
        else:
            # Check using C++ state type info if available
            if hasattr(self._wrapper, '_state'):
                cpp_state = self._wrapper._state
                if hasattr(cpp_state, '__class__'):
                    class_name = cpp_state.__class__.__name__.lower()
                    if 'chess' in class_name:
                        return GameType.CHESS
                    elif 'go' in class_name:
                        return GameType.GO
                    elif 'gomoku' in class_name:
                        return GameType.GOMOKU

            # Default based on action space range
            if action_space > 10000:  # Very large action spaces (Chess-like)
                return GameType.CHESS
            elif action_space >= 300:  # Large boards (Go-like)
                return GameType.GO
            else:  # Medium boards (Gomoku-like)
                return GameType.GOMOKU


class GameRegistry:
    """
    Singleton registry for game types and their factories.

    Manages registration of game types and provides centralized
    game instance creation without tight coupling.
    """

    _instance = None
    _factories = {}

    @classmethod
    def instance(cls) -> 'GameRegistry':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_game(self, game_type: GameType, factory_func) -> None:
        """
        Register a game type with its factory function.

        Args:
            game_type: Game type to register
            factory_func: Function that creates new instances
        """
        self._factories[game_type] = factory_func

    def is_registered(self, game_type: GameType) -> bool:
        """
        Check if a game type is registered.

        Args:
            game_type: Game type to check

        Returns:
            True if registered, False otherwise
        """
        return game_type in self._factories

    def get_factory(self, game_type: GameType):
        """
        Get the factory function for a game type.

        Args:
            game_type: Game type

        Returns:
            Factory function

        Raises:
            ValueError: If type is not registered
        """
        if game_type not in self._factories:
            raise ValueError(f"Game type {game_type} is not registered")
        return self._factories[game_type]

    def get_registered_types(self) -> List[GameType]:
        """
        Get all registered game types.

        Returns:
            List of registered game types
        """
        return list(self._factories.keys())

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._factories.clear()


class GameFactory:
    """
    Factory for creating game instances.

    Provides static methods to create game instances using the registry
    or directly with specific parameters.
    """

    @staticmethod
    def create_game(game_type: Union[GameType, str]) -> IGameState:
        """
        Create a game instance of the specified type.

        Args:
            game_type: Game type to create (enum or string)

        Returns:
            Game instance

        Raises:
            ValueError: If type is not registered
        """
        if isinstance(game_type, str):
            game_type = GameType[game_type.upper()]

        # Direct factory dispatch instead of registry
        if game_type == GameType.CHESS:
            return GameFactory.create_chess()
        elif game_type == GameType.GO:
            return GameFactory.create_go()
        elif game_type == GameType.GOMOKU:
            return GameFactory.create_gomoku()
        else:
            raise ValueError(f"Unsupported game type: {game_type}")

    @staticmethod
    def create_chess(
        chess960: bool = False,
        fen: str = "",
        position_number: int = -1
    ) -> IGameState:
        """
        Create a chess game with specific options.

        Args:
            chess960: Whether to use Chess960 rules
            fen: Optional FEN string for initial position
            position_number: Chess960 position number (0-959)

        Returns:
            Chess game instance
        """
        # Import real game state implementations
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
        from games.game_state import create_game_state

        wrapper = create_game_state('chess', chess960=chess960, fen=fen, position_number=position_number)
        return GameStateAdapter(wrapper)

    @staticmethod
    def create_go(
        board_size: int = 19,
        rule_set: int = 0,  # 0=Chinese, 1=Japanese, 2=Korean
        custom_komi: float = -1.0
    ) -> IGameState:
        """
        Create a Go game with specific options.

        Args:
            board_size: Board size (9, 13, or 19)
            rule_set: Rule set to use
            custom_komi: Optional custom komi value

        Returns:
            Go game instance
        """
        # Import real game state implementations
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
        from games.game_state import create_game_state

        # Convert rule_set integer to string
        rule_set_map = {0: 'chinese', 1: 'japanese', 2: 'korean'}
        rule_set_str = rule_set_map.get(rule_set, 'chinese')
        komi = custom_komi if custom_komi >= 0 else 7.5

        wrapper = create_game_state('go', board_size=board_size, rule_set=rule_set_str, komi=komi)
        return GameStateAdapter(wrapper)

    @staticmethod
    def create_gomoku(
        board_size: int = 15,
        use_renju: bool = False,
        use_omok: bool = False,
        seed: int = 0,
        use_pro_long_opening: bool = False
    ) -> IGameState:
        """
        Create a Gomoku game with specific options.

        Args:
            board_size: Board size (typically 15)
            use_renju: Whether to use Renju rules
            use_omok: Whether to use Omok rules
            seed: Random seed for initialization
            use_pro_long_opening: Whether to use pro-long opening restrictions

        Returns:
            Gomoku game instance
        """
        # Import real game state implementations
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
        from games.game_state import create_game_state

        wrapper = create_game_state('gomoku', board_size=board_size, use_renju=use_renju, use_omok=use_omok, seed=seed, use_pro_long_opening=use_pro_long_opening)
        return GameStateAdapter(wrapper)

    @staticmethod
    def create_game_from_moves(
        game_type: GameType,
        moves: str
    ) -> IGameState:
        """
        Create a game instance from a sequence of moves.

        Args:
            game_type: Game type
            moves: String containing move sequence

        Returns:
            Game instance with moves applied

        Raises:
            ValueError: If moves are invalid
        """
        # Create empty game first
        if game_type == GameType.CHESS:
            game = GameFactory.create_chess()
        elif game_type == GameType.GO:
            game = GameFactory.create_go()
        elif game_type == GameType.GOMOKU:
            game = GameFactory.create_gomoku()
        else:
            raise ValueError(f"Unsupported game type: {game_type}")

        # Parse and apply moves
        if moves.strip():
            move_list = moves.strip().split()
            for move_str in move_list:
                try:
                    action = game.string_to_action(move_str)
                    if action is not None and game.is_legal_move(action):
                        game.make_move(action)
                except:
                    # Skip invalid moves
                    continue

        return game

    @staticmethod
    def create_games(game_type: GameType, count: int) -> List[IGameState]:
        """
        Create multiple game instances efficiently.

        Args:
            game_type: Game type
            count: Number of instances to create

        Returns:
            List of game instances
        """
        return [GameFactory.create_game(game_type) for _ in range(count)]

    @staticmethod
    def detect_game_type(input_str: str) -> GameType:
        """
        Detect game type from state or move notation.

        Args:
            input_str: String containing game state or moves

        Returns:
            Detected game type
        """
        input_str = input_str.strip().lower()

        # Chess detection patterns
        if any(pattern in input_str for pattern in [
            'rnbqkbnr', 'kqkq', 'e2e4', 'nf3', 'qd4', '[event', 'pgn', '1.'
        ]):
            return GameType.CHESS

        # Go detection patterns
        if any(pattern in input_str for pattern in [
            '(;ff[4]', 'gm[1]', ';b[', ';w[', 'sgf', 'q16', 'd4 q16'
        ]):
            return GameType.GO

        # Gomoku detection patterns (default for simple coordinate moves)
        if any(pattern in input_str for pattern in [
            'h8', 'i8', 'j8', 'a1 b2', 'h8 h9'
        ]) or (len(input_str.split()) > 0 and
               all(len(move) <= 3 and move[0].isalpha() for move in input_str.split()[:3])):
            return GameType.GOMOKU

        # Default to Gomoku for unrecognized patterns
        return GameType.GOMOKU


class GameSerializer:
    """
    Game state serialization and deserialization.

    Handles saving and loading game states to/from various formats.
    """

    @staticmethod
    def serialize_game(game: IGameState) -> str:
        """
        Serialize a game state to string.

        Args:
            game: Game state to serialize

        Returns:
            Serialized string representation
        """
        import json

        # Create serializable representation
        game_type = game.get_game_type() if hasattr(game, 'get_game_type') else GameType.GOMOKU
        game_type_str = game_type.name if hasattr(game_type, 'name') else str(game_type)

        data = {
            'game_type': game_type_str,
            'move_history': game.get_move_history() if hasattr(game, 'get_move_history') else [],
            'current_player': game.get_current_player() if hasattr(game, 'get_current_player') else 1,
            'board_size': game.get_board_size() if hasattr(game, 'get_board_size') else 15,
            'is_terminal': game.is_terminal() if hasattr(game, 'is_terminal') else False
        }

        return json.dumps(data)

    @staticmethod
    def deserialize_game(data: str) -> IGameState:
        """
        Deserialize a game state from string.

        Args:
            data: Serialized string representation

        Returns:
            Deserialized game state

        Raises:
            ValueError: If deserialization fails
        """
        import json

        try:
            parsed_data = json.loads(data)
            game_type_str = parsed_data.get('game_type', 'GOMOKU')
            move_history = parsed_data.get('move_history', [])

            # Create game of appropriate type
            if game_type_str == 'CHESS':
                game = GameFactory.create_chess()
            elif game_type_str == 'GO':
                game = GameFactory.create_go()
            else:  # Default to GOMOKU
                game = GameFactory.create_gomoku()

            # Apply moves from history
            for move in move_history:
                if hasattr(game, 'is_legal_move') and game.is_legal_move(move):
                    game.make_move(move)

            return game

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            raise ValueError(f"Failed to deserialize game: {e}")

    @staticmethod
    def save_game(game: IGameState, filename: str) -> None:
        """
        Save game state to file.

        Args:
            game: Game state to save
            filename: Output filename

        Raises:
            IOError: If file cannot be written
        """
        try:
            serialized_data = GameSerializer.serialize_game(game)
            with open(filename, 'w') as f:
                f.write(serialized_data)
        except Exception as e:
            raise IOError(f"Failed to save game to {filename}: {e}")

    @staticmethod
    def load_game(filename: str) -> IGameState:
        """
        Load game state from file.

        Args:
            filename: Input filename

        Returns:
            Loaded game state

        Raises:
            IOError: If file cannot be read
            ValueError: If file cannot be parsed
        """
        try:
            with open(filename, 'r') as f:
                data = f.read()
            return GameSerializer.deserialize_game(data)
        except FileNotFoundError:
            raise IOError(f"File not found: {filename}")
        except Exception as e:
            raise IOError(f"Failed to load game from {filename}: {e}")

    @staticmethod
    def export_to_standard_format(game: IGameState) -> str:
        """
        Export game to standard format (PGN/SGF/custom).

        Args:
            game: Game state to export

        Returns:
            String in standard format
        """
        game_type = game.get_game_type() if hasattr(game, 'get_game_type') else GameType.GOMOKU
        game_type_str = game_type.name if hasattr(game_type, 'name') else str(game_type)
        move_history = game.get_move_history() if hasattr(game, 'get_move_history') else []

        if game_type == GameType.CHESS or game_type_str == 'CHESS':
            # Basic PGN format
            moves_str = ' '.join([f"{i//2 + 1}." if i % 2 == 0 else "" + game.action_to_string(move)
                                 for i, move in enumerate(move_history) if hasattr(game, 'action_to_string')])
            return f'[Event "Game"]\n[Site "AlphaZero"]\n[Date "2024.01.01"]\n[Round "1"]\n[White "Player1"]\n[Black "Player2"]\n\n{moves_str or "1."}'

        elif game_type == GameType.GO or game_type_str == 'GO':
            # Basic SGF format
            move_strs = []
            for i, move in enumerate(move_history):
                player = 'B' if i % 2 == 0 else 'W'
                move_str = game.action_to_string(move) if hasattr(game, 'action_to_string') else f"[{move}]"
                move_strs.append(f";{player}[{move_str.lower()}]")
            return f"(;FF[4]GM[1]SZ[19]{''.join(move_strs)})"

        else:  # GOMOKU
            # Custom format
            moves_str = ' '.join([game.action_to_string(move) if hasattr(game, 'action_to_string') else str(move)
                                 for move in move_history])
            return f"Gomoku Game: {moves_str or 'No moves'}"


# Utility functions
def game_type_to_string(game_type: GameType) -> str:
    """Convert GameType enum to string."""
    return game_type.name


def string_to_game_type(type_str: str) -> GameType:
    """Convert string to GameType enum."""
    try:
        return GameType[type_str.upper()]
    except KeyError:
        return GameType.UNKNOWN


# Game adapter utilities
def are_states_equivalent(state1: IGameState, state2: IGameState) -> bool:
    """
    Check if two game states are equivalent.

    Args:
        state1: First game state
        state2: Second game state

    Returns:
        True if equivalent, False otherwise
    """
    return (
        state1.get_game_type() == state2.get_game_type() and
        state1.get_hash() == state2.get_hash() and
        state1.equals(state2)
    )


def get_game_statistics(game: IGameState) -> Dict[str, float]:
    """
    Get game statistics.

    Args:
        game: Game state to analyze

    Returns:
        Dictionary of statistic names to values
    """
    return {
        'move_count': len(game.get_move_history()),
        'legal_moves': len(game.get_legal_moves()),
        'board_size': game.get_board_size(),
        'action_space_size': game.get_action_space_size(),
        'current_player': game.get_current_player(),
        'is_terminal': 1.0 if game.is_terminal() else 0.0
    }


def validate_move_sequence(game: IGameState, moves: List[int]) -> bool:
    """
    Validate a sequence of moves.

    Args:
        game: Initial game state (will be cloned)
        moves: List of actions to validate

    Returns:
        True if all moves are legal in sequence, False otherwise
    """
    test_game = game.clone()

    for move in moves:
        if not test_game.is_legal_move(move):
            return False
        try:
            test_game.make_move(move)
        except ValueError:
            return False

    return True


def convert_action_format(
    game: IGameState,
    action: int,
    format_type: str
) -> str:
    """
    Convert between different action representations.

    Args:
        game: Game state for context
        action: Action to convert
        format_type: Target format ("string", "coordinate", "index")

    Returns:
        Converted action representation
    """
    if format_type == "string":
        return game.action_to_string(action)
    elif format_type == "index":
        return str(action)
    elif format_type == "coordinate":
        # Convert to coordinate notation (row, col)
        board_size = game.get_board_size()
        row = action // board_size
        col = action % board_size
        return f"({row},{col})"
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_game_complexity(game_type: GameType) -> Dict[str, float]:
    """
    Get game complexity metrics.

    Args:
        game_type: Game type

    Returns:
        Dictionary of complexity metrics
    """
    complexity_data = {
        GameType.CHESS: {
            'branching_factor': 35.0,
            'average_game_length': 40.0,
            'state_space_complexity': 47.0,  # log10
            'game_tree_complexity': 123.0   # log10
        },
        GameType.GO: {
            'branching_factor': 250.0,
            'average_game_length': 150.0,
            'state_space_complexity': 171.0,  # log10
            'game_tree_complexity': 360.0    # log10
        },
        GameType.GOMOKU: {
            'branching_factor': 200.0,
            'average_game_length': 30.0,
            'state_space_complexity': 105.0,  # log10
            'game_tree_complexity': 70.0     # log10
        }
    }

    return complexity_data.get(game_type, {
        'branching_factor': 0.0,
        'average_game_length': 0.0,
        'state_space_complexity': 0.0,
        'game_tree_complexity': 0.0
    })