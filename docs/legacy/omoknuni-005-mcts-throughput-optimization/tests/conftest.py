"""
Pytest Configuration and Fixtures for Real Implementation Testing
================================================================

This file provides common fixtures and configuration for testing with
real C++ implementations instead of mocks.
"""

import pytest
import numpy as np
import tempfile
import torch
import os
from concurrent.futures import Future
from src.games.game_state import create_game_state
from src.neural.cpu_inference import CPUInferenceWorker
from src.neural.model import create_model_for_game


@pytest.fixture
def real_gomoku_game():
    """Create a real Gomoku game state for testing."""
    return create_game_state('gomoku')


@pytest.fixture
def real_chess_game():
    """Create a real Chess game state for testing."""
    return create_game_state('chess')


@pytest.fixture
def real_go_game():
    """Create a real Go game state (9x9) for testing."""
    return create_game_state('go', board_size=9)


@pytest.fixture(params=['gomoku', 'chess', 'go'])
def test_model_path(request):
    """Create a temporary test model file for different games."""
    game_type = request.param

    # Create temporary model file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model_path = f.name

    try:
        # Create a test model for the specific game
        model = create_model_for_game(game_type)

        # Get appropriate board size for each game
        if game_type == 'gomoku':
            board_size = 15
        elif game_type == 'chess':
            board_size = 8
        else:  # go
            board_size = 19

        # Initialize the model with a forward pass to create lazy layers
        with torch.no_grad():
            dummy_input = torch.randn(1, model.input_channels, board_size, board_size)
            _ = model(dummy_input)

        # Save the model
        torch.save(model.state_dict(), model_path)

        yield model_path, game_type
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)


@pytest.fixture
def real_inference_worker(test_model_path):
    """Create a properly configured CPU inference worker with real model."""
    model_path, game_type = test_model_path
    return CPUInferenceWorker(
        model_path=model_path,
        device='cpu',
        batch_size=32,
        timeout_ms=1000.0
    ), game_type


@pytest.fixture
def mock_inference_function():
    """Create a mock inference function for MCTS testing."""
    def inference_fn(game_state):
        """Mock inference that returns uniform policy and random value."""
        future = Future()

        # Generate mock policy and value
        action_space = game_state.action_space_size
        legal_moves_mask = game_state.get_legal_moves()
        legal_moves = np.where(legal_moves_mask)[0]

        policy = np.zeros(action_space, dtype=np.float32)
        if len(legal_moves) > 0:
            for move in legal_moves:
                policy[move] = 1.0 / len(legal_moves)

        value = np.random.uniform(-0.5, 0.5)

        # Set result immediately (synchronous for testing)
        future.set_result((policy, value))
        return future

    return inference_fn


class TestCompatibilityLayer:
    """Compatibility layer for tests to work with both mock and real implementations."""

    @staticmethod
    def create_game_state(game_type='gomoku', **kwargs):
        """Create real game state with fallback compatibility."""
        return create_game_state(game_type, **kwargs)

    @staticmethod
    def create_inference_worker(**kwargs):
        """Create properly configured inference worker."""
        # Create a temporary model for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_model_path = f.name

        # Create a simple test model
        model = create_model_for_game('gomoku')
        with torch.no_grad():
            dummy_input = torch.randn(1, model.input_channels, 15, 15)
            _ = model(dummy_input)
        torch.save(model.state_dict(), temp_model_path)

        defaults = {
            'model_path': temp_model_path,
            'device': 'cpu',
            'batch_size': 32,
            'timeout_ms': 1000.0
        }
        defaults.update(kwargs)
        return CPUInferenceWorker(**defaults)