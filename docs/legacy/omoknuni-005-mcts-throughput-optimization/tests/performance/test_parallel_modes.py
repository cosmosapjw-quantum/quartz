"""Performance sanity checks for different parallel modes."""

import pytest
import numpy as np

import alphazero_py
from src.core.mcts import AlphaZeroMCTS


@pytest.fixture(scope="module")
def gomoku_state():
    return alphazero_py.GomokuState(board_size=15)


def constant_inference(state):
    policy = np.ones(state.get_action_space_size(), dtype=np.float32) / state.get_action_space_size()
    future = Future()
    future.set_result((policy, 0.0))
    return future


from concurrent.futures import Future  # pylint: disable=wrong-import-position


@pytest.mark.parametrize("mode", ["shared", "virtual_loss_free"])
def test_parallel_mode_instrumentation(mode, gomoku_state):
    engine = AlphaZeroMCTS(
        inference_fn=constant_inference,
        num_threads=2,
        use_async_inference=False,
        enable_instrumentation=True,
        parallel_mode=mode,
    )
    try:
        engine.search(gomoku_state, simulations=32)
        stats = engine.get_statistics()
        instrumentation = stats.get('instrumentation', {})
        assert instrumentation, "Instrumentation data should be present"
        assert stats['parallel_mode'] == mode
        if mode == 'virtual_loss_free':
            assert stats['virtual_loss_enabled'] is False
        else:
            assert stats['virtual_loss_enabled'] is True
    finally:
        engine.close()
