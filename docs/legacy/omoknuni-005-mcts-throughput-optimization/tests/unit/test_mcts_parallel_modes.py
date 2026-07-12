"""Unit tests for AlphaZeroMCTS parallel mode configuration."""

from concurrent.futures import Future

import numpy as np
import pytest

import alphazero_py
from src.core.mcts import AlphaZeroMCTS


@pytest.fixture
def gomoku_state():
    return alphazero_py.GomokuState(board_size=15)


def basic_inference(state):
    policy = np.ones(state.get_action_space_size(), dtype=np.float32) / state.get_action_space_size()
    future = Future()
    future.set_result((policy, 0.0))
    return future


def test_virtual_loss_free_disables_virtual_loss(gomoku_state):
    engine = AlphaZeroMCTS(
        inference_fn=basic_inference,
        num_threads=2,
        use_async_inference=False,
        parallel_mode="virtual_loss_free",
    )
    try:
        config = engine.virtual_loss_manager.get_config()
        assert config.enable_virtual_loss is False

        engine.search(gomoku_state, simulations=8)
        stats = engine.get_statistics()
        assert stats['parallel_mode'] == 'virtual_loss_free'
        assert stats['virtual_loss_enabled'] is False
    finally:
        engine.close()


def test_switching_parallel_mode_restores_virtual_loss(gomoku_state):
    engine = AlphaZeroMCTS(
        inference_fn=basic_inference,
        num_threads=2,
        use_async_inference=False,
        parallel_mode="virtual_loss_free",
    )
    try:
        engine.set_parallel_mode("shared")
        config = engine.virtual_loss_manager.get_config()
        assert config.enable_virtual_loss is True

        engine.search(gomoku_state, simulations=4)
        stats = engine.get_statistics()
        assert stats['parallel_mode'] == 'shared'
        assert stats['virtual_loss_enabled'] is True
    finally:
        engine.close()


def test_invalid_parallel_mode_raises():
    with pytest.raises(ValueError):
        AlphaZeroMCTS(
            inference_fn=basic_inference,
            num_threads=1,
            use_async_inference=False,
            parallel_mode="not_a_mode",
        )
