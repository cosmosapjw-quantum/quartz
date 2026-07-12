"""Tests ensuring AlphaZeroMCTS reuses its thread pool between searches."""

import numpy as np
from concurrent.futures import Future

import alphazero_py
from src.core.mcts import AlphaZeroMCTS


def _noop_inference(state):
    policy = np.ones(state.get_action_space_size(), dtype=np.float32)
    policy /= policy.sum()
    future: Future = Future()
    future.set_result((policy, 0.0))
    return future


def test_thread_pool_reuse():
    engine = AlphaZeroMCTS(
        inference_fn=_noop_inference,
        num_threads=2,
        use_async_inference=False,
        enable_instrumentation=False,
    )

    game = alphazero_py.GomokuState(board_size=15)

    first_executor = engine._executor  # pylint: disable=protected-access
    assert first_executor is not None

    engine.search(game, simulations=16)
    second_executor = engine._executor  # pylint: disable=protected-access
    assert second_executor is first_executor

    engine.search(game, simulations=8)
    third_executor = engine._executor  # pylint: disable=protected-access
    assert third_executor is first_executor

    engine.close()
