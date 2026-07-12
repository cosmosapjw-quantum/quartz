"""
Unit tests for Persistent Coordinator Lifecycle (T011a)

Validates that:
1. Coordinator created once in __init__, not per-search
2. Same coordinator instance reused across multiple searches
3. Coordinator state management (started/stopped flags)
4. Clean shutdown via close() method
5. __del__ fallback if close() not called
"""

import pytest
import sys
from pathlib import Path
from concurrent.futures import Future
from typing import Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mcts_py
    from src.core.mcts import AlphaZeroMCTS
    import alphazero_py
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"Components not available: {e}")


def create_mock_inference_fn():
    """Create mock inference function that returns Future."""
    def mock_inference(state):
        """Mock inference returning uniform policy and zero value."""
        future = Future()
        action_space = state.get_action_space_size()
        policy = np.ones(action_space, dtype=np.float32) / action_space
        value = 0.0
        future.set_result((policy, value))
        return future
    return mock_inference


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
class TestCoordinatorLifecycle:
    """Tests for persistent coordinator lifecycle (T011a)."""

    def test_coordinator_created_in_init(self):
        """Verify coordinator is created once in __init__, not None."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            # T011a: Coordinator should exist as instance variable
            assert hasattr(mcts, '_coordinator'), "Should have _coordinator attribute"
            assert mcts._coordinator is not None, "Coordinator should be created in __init__"
            assert hasattr(mcts, '_coordinator_started'), "Should have _coordinator_started attribute"
            assert mcts._coordinator_started is False, "Coordinator should not be started yet"
        finally:
            mcts.close()

    def test_coordinator_started_on_first_search(self):
        """Verify coordinator is started on first search call."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Before first search
            assert mcts._coordinator_started is False

            # Run first search
            mcts.search(state, simulations=10)

            # After first search
            assert mcts._coordinator_started is True, "Coordinator should be started after first search"
        finally:
            mcts.close()

    def test_coordinator_reused_across_searches(self):
        """Verify same coordinator instance is reused for 3 consecutive searches."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=1,  # Single thread for stability
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Capture coordinator reference after first search
            mcts.search(state, simulations=5)
            coordinator_ref = mcts._coordinator

            # Run 3 consecutive searches
            for i in range(3):
                # Reset tree to clear state
                mcts.reset()
                mcts.search(state, simulations=5)

                # T011a: Verify same coordinator instance (not recreated)
                assert mcts._coordinator is coordinator_ref, \
                    f"Coordinator should be same instance on search {i+1}"
                assert mcts._coordinator_started is True, \
                    f"Coordinator should remain started on search {i+1}"

        finally:
            mcts.close()

    def test_coordinator_state_management(self):
        """Verify coordinator state transitions correctly."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Initial state: not_started
            assert mcts._coordinator_started is False

            # After first search: started
            mcts.search(state, simulations=10)
            assert mcts._coordinator_started is True

            # After multiple searches: still started
            mcts.search(state, simulations=10)
            mcts.search(state, simulations=10)
            assert mcts._coordinator_started is True

            # After close: stopped
            mcts.close()
            assert mcts._coordinator_started is False

        except Exception:
            mcts.close()
            raise

    def test_close_stops_coordinator(self):
        """Verify close() method stops coordinator and sets flag."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Start coordinator
            mcts.search(state, simulations=10)
            assert mcts._coordinator_started is True

            # Close should stop coordinator
            mcts.close()
            assert mcts._coordinator_started is False

        except Exception:
            mcts.close()
            raise

    def test_close_idempotent(self):
        """Verify close() can be called multiple times safely."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()
            mcts.search(state, simulations=10)

            # Call close() multiple times
            mcts.close()
            mcts.close()
            mcts.close()

            # Should not raise exception
            assert mcts._coordinator_started is False

        except Exception:
            mcts.close()
            raise

    def test_coordinator_survives_search_exception(self):
        """Verify coordinator is preserved when search encounters exception."""
        def failing_inference_fn(state):
            """Mock inference that fails after first call."""
            if not hasattr(failing_inference_fn, 'call_count'):
                failing_inference_fn.call_count = 0
            failing_inference_fn.call_count += 1

            if failing_inference_fn.call_count > 5:
                raise RuntimeError("Simulated inference failure")

            future = Future()
            action_space = state.get_action_space_size()
            policy = np.ones(action_space, dtype=np.float32) / action_space
            value = 0.0
            future.set_result((policy, value))
            return future

        mcts = AlphaZeroMCTS(
            inference_fn=failing_inference_fn,
            num_threads=1,
            use_async_inference=True,
            async_batch_size=2,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # First search succeeds
            mcts.search(state, simulations=3)
            coordinator_ref = mcts._coordinator
            assert mcts._coordinator_started is True

            # Second search may fail but coordinator should survive
            try:
                mcts.search(state, simulations=10)
            except Exception:
                pass  # Expected failure

            # T011a: Coordinator should still be the same instance and started
            assert mcts._coordinator is coordinator_ref, "Coordinator should survive exception"
            assert mcts._coordinator_started is True, "Coordinator should remain started after exception"

        finally:
            mcts.close()

    def test_sync_mode_no_coordinator(self):
        """Verify synchronous mode doesn't create coordinator."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=False  # Synchronous mode
        )

        try:
            # Sync mode should have coordinator set to None
            assert mcts._coordinator is None
            assert mcts._coordinator_started is False

            state = alphazero_py.GomokuState()
            mcts.search(state, simulations=10)

            # Should still be None after search
            assert mcts._coordinator is None
            assert mcts._coordinator_started is False

        finally:
            mcts.close()

    def test_no_per_search_coordinator_creation(self):
        """Verify coordinator is NOT created per-search (regression test)."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=1,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Track coordinator object ID across searches
            mcts.search(state, simulations=5)
            coordinator_id_1 = id(mcts._coordinator)

            for i in range(3):  # Reduced to 3 for stability
                mcts.reset()
                mcts.search(state, simulations=5)
                coordinator_id = id(mcts._coordinator)

                # T011a: Object ID should remain constant (same instance)
                assert coordinator_id == coordinator_id_1, \
                    f"Coordinator recreated on search {i+1} (regression: per-search creation detected)"

        finally:
            mcts.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
