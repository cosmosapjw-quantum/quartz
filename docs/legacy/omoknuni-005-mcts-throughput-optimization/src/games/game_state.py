"""
Game State Implementation
========================

Real game state implementation using C++ bindings for high-performance game logic.
Provides unified Python interface to C++ game implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple, Union
import numpy as np
import alphazero_py


class IGameState(ABC):
    """Interface for game state management.

    This interface defines the contract for game state implementations
    used by the MCTS search algorithm.
    """

    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices from current position.

        Returns:
            List[int]: List of legal action indices
        """
        pass

    @abstractmethod
    def make_move(self, move: int) -> 'IGameState':
        """Apply a move and return new game state."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        pass

    @abstractmethod
    def get_result(self) -> Optional[float]:
        """Get game result from current player's perspective."""
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """Get neural network features for current position."""
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Get current player to move."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of possible actions in this game."""
        pass


class GameStateWrapper(IGameState):
    """Wrapper for C++ game state implementations.

    This class provides a unified Python interface to the C++ game implementations
    while maintaining compatibility with the existing Python codebase.
    """

    def __init__(self, cpp_game_state: alphazero_py.IGameState):
        """Initialize wrapper with C++ game state.

        Args:
            cpp_game_state: C++ game state instance from alphazero_py
        """
        self._state = cpp_game_state

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices from current position."""
        legal_moves_mask = self._state.get_legal_moves()
        return np.where(legal_moves_mask)[0].tolist()

    def get_legal_moves_mask(self) -> np.ndarray:
        """Get boolean mask of legal moves (for performance-critical code)."""
        return self._state.get_legal_moves()

    def make_move(self, move: int) -> 'GameStateWrapper':
        """Apply a move and return new game state."""
        new_cpp_state = self._state.clone()
        new_cpp_state.make_move(move)
        return GameStateWrapper(new_cpp_state)

    def is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        return self._state.is_terminal()

    def get_result(self) -> Optional[float]:
        """Get game result from current player's perspective."""
        if not self.is_terminal():
            return None

        result = self._state.get_game_result()
        current_player = self.get_current_player()

        # Convert C++ game result to player perspective
        if result == alphazero_py.GameResult.DRAW:
            return 0.0
        elif result == alphazero_py.GameResult.WIN_PLAYER1:
            return 1.0 if current_player == 0 else -1.0
        elif result == alphazero_py.GameResult.WIN_PLAYER2:
            return 1.0 if current_player == 1 else -1.0
        else:
            return None

    def get_features(self) -> np.ndarray:
        """Get neural network features for current position."""
        return self._state.get_tensor_representation()

    def get_current_player(self) -> int:
        """Get current player to move."""
        return self._state.get_current_player()

    @property
    def action_space_size(self) -> int:
        """Number of possible actions in this game."""
        return self._state.get_action_space_size()

    @property
    def cpp_state(self) -> alphazero_py.IGameState:
        """Access to underlying C++ state for performance-critical operations."""
        return self._state

    def clone(self) -> 'GameStateWrapper':
        """Create a deep copy of this game state."""
        return GameStateWrapper(self._state.clone())

    # Additional methods to fully leverage C++ capabilities

    def apply_move_inplace(self, move: int) -> None:
        """Apply a move in-place to this game state."""
        self._state.apply_move_inplace(move)

    def undo_move(self) -> bool:
        """Undo the last move. Returns True if successful."""
        return self._state.undo_move()

    def is_legal_move(self, move: int) -> bool:
        """Check if a move is legal in the current position."""
        return self._state.is_legal_move(move)

    def get_board_size(self) -> int:
        """Get the board size for this game."""
        return self._state.get_board_size()

    def get_hash(self) -> int:
        """Get a hash value for the current position."""
        return self._state.get_hash()

    def get_move_history(self) -> List[int]:
        """Get the history of moves played."""
        return self._state.get_move_history()

    def action_to_string(self, action: int) -> str:
        """Convert an action index to human-readable string."""
        return self._state.action_to_string(action)

    def string_to_action(self, move_str: str) -> Optional[int]:
        """Convert a move string to action index."""
        result = self._state.string_to_action(move_str)
        return result if result >= 0 else None

    def to_string(self) -> str:
        """Get a string representation of the current board."""
        return self._state.to_string()

    def get_basic_tensor_representation(self) -> np.ndarray:
        """Get basic neural network features (19 channels for Gomoku)."""
        return self._state.get_basic_tensor_representation()

    def get_enhanced_tensor_representation(self) -> np.ndarray:
        """Get enhanced neural network features (36 channels for Gomoku)."""
        return self._state.get_enhanced_tensor_representation()

    def get_tensor_representation(self) -> np.ndarray:
        """Get default neural network features."""
        return self._state.get_tensor_representation()

    def batch_clone(self, count: int) -> List['GameStateWrapper']:
        """Create multiple clones efficiently."""
        cpp_clones = self._state.batch_clone(count)
        return [GameStateWrapper(clone) for clone in cpp_clones]

    def copy_from(self, other: 'GameStateWrapper') -> None:
        """Copy state from another game state."""
        self._state.copy_from(other._state)

    # Game-specific methods (will only work for appropriate game types)
    def get_renju_rules(self) -> bool:
        """Get whether Renju rules are enabled (Gomoku only)."""
        if hasattr(self._state, 'get_renju_rules'):
            return self._state.get_renju_rules()
        return False

    def get_omok_rules(self) -> bool:
        """Get whether Omok rules are enabled (Gomoku only)."""
        if hasattr(self._state, 'get_omok_rules'):
            return self._state.get_omok_rules()
        return False

    def get_pro_long_opening(self) -> bool:
        """Get whether pro-long opening is enabled (Gomoku only)."""
        if hasattr(self._state, 'get_pro_long_opening'):
            return self._state.get_pro_long_opening()
        return False


def create_game_state(game_type: str, **kwargs) -> GameStateWrapper:
    """Factory function to create game states.

    Args:
        game_type: Game type ('gomoku', 'chess', 'go')
        **kwargs: Game-specific configuration options

    Returns:
        Wrapped game state instance

    Raises:
        ValueError: If game_type is not supported
    """
    game_type = game_type.lower()

    if game_type == 'gomoku':
        board_size = kwargs.get('board_size', 15)
        use_renju = kwargs.get('use_renju', False)
        use_omok = kwargs.get('use_omok', False)
        seed = kwargs.get('seed', 0)
        use_pro_long_opening = kwargs.get('use_pro_long_opening', False)

        cpp_state = alphazero_py.GomokuState(
            board_size, use_renju, use_omok, seed, use_pro_long_opening
        )

    elif game_type == 'chess':
        chess960 = kwargs.get('chess960', False)
        fen = kwargs.get('fen', '')
        position_number = kwargs.get('position_number', -1)

        if position_number >= 0:
            cpp_state = alphazero_py.ChessState(chess960, fen, position_number)
        elif fen:
            cpp_state = alphazero_py.ChessState(chess960, fen)
        else:
            cpp_state = alphazero_py.ChessState(chess960)

    elif game_type == 'go':
        board_size = kwargs.get('board_size', 19)
        komi = kwargs.get('komi', 7.5)
        rule_set = kwargs.get('rule_set', 'chinese')

        if rule_set.lower() == 'chinese':
            cpp_rule_set = alphazero_py.GoRuleSet.CHINESE
        elif rule_set.lower() == 'japanese':
            cpp_rule_set = alphazero_py.GoRuleSet.JAPANESE
        elif rule_set.lower() == 'korean':
            cpp_rule_set = alphazero_py.GoRuleSet.KOREAN
        else:
            cpp_rule_set = alphazero_py.GoRuleSet.CHINESE

        cpp_state = alphazero_py.GoState(board_size, cpp_rule_set, komi)

    else:
        raise ValueError(f"Unsupported game type: {game_type}")

    return GameStateWrapper(cpp_state)


class CppGameStateWrapper(IGameState):
    """Wrapper for C++ game state implementations to ensure proper interface compatibility."""

    def __init__(self, cpp_state):
        """Initialize wrapper with C++ game state.

        Args:
            cpp_state: C++ game state from alphazero_py
        """
        self._state = cpp_state

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices from current position."""
        # C++ states return boolean mask, convert to integer indices
        legal_moves_mask = self._state.get_legal_moves()
        return np.where(legal_moves_mask)[0].tolist()

    def make_move(self, move: int) -> 'CppGameStateWrapper':
        """Apply a move and return new game state."""
        new_state = self._state.clone()
        new_state.make_move(move)
        return CppGameStateWrapper(new_state)

    def clone(self) -> 'CppGameStateWrapper':
        """Create a deep copy of the game state."""
        return CppGameStateWrapper(self._state.clone())

    def is_terminal(self) -> bool:
        """Check if game is finished."""
        return self._state.is_terminal()

    def get_result(self) -> Optional[float]:
        """Get game result from current player's perspective."""
        if not self.is_terminal():
            return None

        result = self._state.get_game_result()
        current_player = self._state.get_current_player()

        # Convert C++ game result to float from current player's perspective
        if hasattr(result, 'WIN_PLAYER1'):
            # Result enum values
            if result == result.WIN_PLAYER1:
                return 1.0 if current_player == 1 else -1.0
            elif result == result.WIN_PLAYER2:
                return 1.0 if current_player == 2 else -1.0
            else:  # DRAW
                return 0.0
        else:
            # Result is integer (1, 2, or 0 for draw)
            if result == 1:
                return 1.0 if current_player == 1 else -1.0
            elif result == 2:
                return 1.0 if current_player == 2 else -1.0
            else:  # 0 = draw
                return 0.0

    def get_features(self) -> np.ndarray:
        """Get neural network features for current position."""
        return self._state.get_enhanced_tensor_representation()

    def get_current_player(self) -> int:
        """Get current player to move."""
        return self._state.get_current_player()

    @property
    def action_space_size(self) -> int:
        """Number of possible actions in this game."""
        return self._state.action_space_size


class GameStateAdapter(IGameState):
    """Adapter to bridge contract GameState to IGameState interface.

    This adapter allows contract-defined GameState objects to work
    with the IGameState interface expected by AlphaZero MCTS.
    """

    def __init__(self, game_state):
        """Initialize adapter with contract GameState.

        Args:
            game_state: GameState object from contract API
        """
        self._state = game_state

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices from current position."""
        legal_moves_mask = self._state.get_legal_moves()
        return np.where(legal_moves_mask)[0].tolist()

    def make_move(self, move: int) -> 'GameStateAdapter':
        """Apply a move and return new game state."""
        new_state = self._state.copy()
        new_state.apply_move_inplace(move)
        return GameStateAdapter(new_state)

    def is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        return self._state.is_terminal()

    def get_result(self) -> Optional[float]:
        """Get game result from current player's perspective."""
        if not self.is_terminal():
            return None
        return self._state.get_terminal_value()

    def get_features(self) -> np.ndarray:
        """Get neural network features for current position."""
        return self._state.extract_features()

    def get_current_player(self) -> int:
        """Get current player to move."""
        return self._state.get_current_player()

    @property
    def action_space_size(self) -> int:
        """Number of possible actions in this game."""
        # Infer from legal moves array size
        legal_moves = self._state.get_legal_moves()
        return len(legal_moves)

    def clone(self) -> 'GameStateAdapter':
        """Create a deep copy of this game state."""
        return GameStateAdapter(self._state.copy())


