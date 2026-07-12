#!/usr/bin/env python3
"""
Python Bindings Demo for AlphaZero Engine

This script demonstrates how to use the Python bindings to interact
with the C++ game implementations from Python.
"""

import sys
import os
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'cpp_extensions', 'games'))

import alphazero_py


def demo_game_creation():
    """Demonstrate creating different game types."""
    print("=== Game Creation Demo ===")

    # Create different games
    games = {
        'Gomoku': alphazero_py.create_game(alphazero_py.GameType.GOMOKU),
        'Chess': alphazero_py.create_game(alphazero_py.GameType.CHESS),
        'Go': alphazero_py.create_game(alphazero_py.GameType.GO)
    }

    for name, game in games.items():
        print(f"{name}:")
        print(f"  Board size: {game.get_board_size()}")
        print(f"  Action space: {game.get_action_space_size()}")
        print(f"  Current player: {game.get_current_player()}")
        print()


def demo_numpy_integration():
    """Demonstrate numpy array integration for neural networks."""
    print("=== Numpy Integration Demo ===")

    game = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)

    # Get tensor representation
    tensor = game.get_tensor_representation()
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Memory usage: {tensor.nbytes} bytes")
    print(f"Is C-contiguous: {tensor.flags.c_contiguous}")

    # Demonstrate that this is a real numpy array
    print(f"Mean value: {np.mean(tensor):.4f}")
    print(f"Non-zero values: {np.count_nonzero(tensor)}")
    print()


def demo_game_playing():
    """Demonstrate playing a simple game."""
    print("=== Game Playing Demo ===")

    game = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)
    moves_made = []

    print("Making some random moves...")
    for i in range(5):
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        # Make the first legal move (in practice, you'd use MCTS or neural network)
        move = legal_moves[0]
        moves_made.append(move)

        print(f"Player {game.get_current_player()} makes move {move}")
        game.make_move(move)

        if game.is_terminal():
            result = game.get_game_result()
            print(f"Game ended with result: {result}")
            break

    print(f"Made {len(moves_made)} moves")
    print(f"Current player: {game.get_current_player()}")
    print(f"Remaining legal moves: {len(game.get_legal_moves())}")
    print()


def demo_serialization():
    """Demonstrate game state serialization."""
    print("=== Serialization Demo ===")

    # Create a game and make some moves
    game = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)
    legal_moves = game.get_legal_moves()

    if len(legal_moves) >= 3:
        for i in range(3):
            game.make_move(legal_moves[i])

    # Serialize the game state
    serialized = alphazero_py.serialize_game(game)
    print(f"Serialized game state ({len(serialized)} characters):")
    print(serialized[:200] + "..." if len(serialized) > 200 else serialized)

    # Deserialize and verify
    deserialized = alphazero_py.deserialize_game(serialized)
    print(f"Original hash: {game.get_hash()}")
    print(f"Deserialized hash: {deserialized.get_hash()}")
    print(f"Hashes match: {game.get_hash() == deserialized.get_hash()}")
    print()


def demo_performance():
    """Demonstrate performance characteristics."""
    print("=== Performance Demo ===")

    import time

    # Test game creation performance
    start_time = time.time()
    games = [alphazero_py.create_game(alphazero_py.GameType.GOMOKU) for _ in range(1000)]
    creation_time = time.time() - start_time
    print(f"Created 1000 games in {creation_time:.3f}s ({1000/creation_time:.0f} games/s)")

    # Test tensor extraction performance
    game = games[0]
    start_time = time.time()
    tensors = [game.get_tensor_representation() for _ in range(1000)]
    tensor_time = time.time() - start_time
    print(f"Extracted 1000 tensors in {tensor_time:.3f}s ({1000/tensor_time:.0f} tensors/s)")

    # Test move making performance
    start_time = time.time()
    for _ in range(1000):
        legal_moves = game.get_legal_moves()
        if legal_moves:
            game.make_move(legal_moves[0])
            game.undo_move()
    move_time = time.time() - start_time
    print(f"Made/undid 1000 moves in {move_time:.3f}s ({1000/move_time:.0f} moves/s)")
    print()


def main():
    """Run all demonstrations."""
    print("Python Bindings Demo for AlphaZero Engine")
    print("=" * 50)
    print()

    try:
        demo_game_creation()
        demo_numpy_integration()
        demo_game_playing()
        demo_serialization()
        demo_performance()

        print("All demonstrations completed successfully!")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()