"""
Integration tests for Coordinator Persistence Across Searches (T011b)

Validates that:
1. Single coordinator handles 1000+ searches without restart
2. Coordinator survives search errors gracefully
3. Metrics confirm no per-search coordinator recreation
4. No memory leaks from coordinator accumulation
5. Health checks and defensive restart logic work correctly
"""

import pytest
import sys
from pathlib import Path
from concurrent.futures import Future
from typing import Tuple
import numpy as np
import gc
import tracemalloc

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
class TestCoordinatorPersistence:
    """Tests for coordinator persistence across searches (T011b)."""

    def test_coordinator_handles_1000_searches(self):
        """Verify single coordinator handles 1000 consecutive searches without restart."""
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
            initial_coordinator_id = id(coordinator_ref)

            # Run 1000 consecutive searches
            for i in range(1000):
                mcts.reset()
                mcts.search(state, simulations=5)

                # T011b: Verify same coordinator instance throughout
                assert mcts._coordinator is coordinator_ref, \
                    f"Coordinator recreated on search {i+1} (object changed)"
                assert id(mcts._coordinator) == initial_coordinator_id, \
                    f"Coordinator object ID changed on search {i+1}"
                assert mcts._coordinator_started is True, \
                    f"Coordinator should remain started on search {i+1}"

                # Check metrics every 100 searches
                if (i + 1) % 100 == 0:
                    stats = mcts.get_statistics()
                    assert stats['coordinator_searches'] == i + 2, \
                        f"Expected {i+2} searches, got {stats['coordinator_searches']}"
                    assert stats['coordinator_started'] is True

            # Final verification
            stats = mcts.get_statistics()
            assert stats['coordinator_searches'] == 1001, \
                "Should have 1001 total searches (1 initial + 1000 in loop)"

        finally:
            mcts.close()

    def test_coordinator_survives_search_exception(self):
        """Verify coordinator persists when search encounters exception."""
        call_count = 0

        def failing_inference_fn(state):
            """Mock inference that fails after 10 calls."""
            nonlocal call_count
            call_count += 1

            if call_count > 10:
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
            assert mcts._coordinator_searches == 1

            # Second search may fail but coordinator should survive
            try:
                mcts.search(state, simulations=10)
            except Exception:
                pass  # Expected failure

            # T011b: Coordinator should still be same instance and started
            assert mcts._coordinator is coordinator_ref, \
                "Coordinator should survive exception (same object)"
            assert mcts._coordinator_started is True, \
                "Coordinator should remain started after exception"

            # Reset call count and verify recovery
            call_count = 0
            mcts.reset()
            mcts.search(state, simulations=3)

            # Coordinator should still be same instance
            assert mcts._coordinator is coordinator_ref, \
                "Coordinator should remain same after recovery"
            assert mcts._coordinator_searches >= 2, \
                "Search counter should have incremented for successful searches"

        finally:
            mcts.close()

    def test_no_coordinator_recreation_between_searches(self):
        """Verify coordinator is NOT recreated between searches (regression test)."""
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

            # Track coordinator object ID across 100 searches
            coordinator_ids = []

            for i in range(100):
                mcts.reset()
                mcts.search(state, simulations=5)
                coordinator_ids.append(id(mcts._coordinator))

            # T011b: All coordinator IDs should be identical
            unique_ids = set(coordinator_ids)
            assert len(unique_ids) == 1, \
                f"Coordinator recreated {len(unique_ids)} times (should be 1 instance)"

            # Verify metrics match
            stats = mcts.get_statistics()
            assert stats['coordinator_searches'] == 100, \
                f"Expected 100 searches, got {stats['coordinator_searches']}"

        finally:
            mcts.close()

    def test_coordinator_metrics_accuracy(self):
        """Verify coordinator lifetime metrics are accurate."""
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

            # Run varying numbers of searches and check metrics
            for expected_count in [1, 5, 10, 25, 50]:
                while mcts._coordinator_searches < expected_count:
                    mcts.reset()
                    mcts.search(state, simulations=5)

                stats = mcts.get_statistics()
                assert stats['coordinator_searches'] == expected_count, \
                    f"Metric mismatch: expected {expected_count}, got {stats['coordinator_searches']}"
                assert stats['coordinator_started'] is True

        finally:
            mcts.close()

    def test_no_memory_leaks_over_1000_searches(self):
        """Verify no memory leaks from coordinator over 1000 searches."""
        inference_fn = create_mock_inference_fn()

        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=1,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Run 1000 searches
            for _ in range(1000):
                mcts.reset()
                mcts.search(state, simulations=5)

            # Force garbage collection
            gc.collect()

            # Take snapshot after searches
            snapshot_end = tracemalloc.take_snapshot()

            # Compare memory usage
            top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')

            # Calculate total memory increase
            total_increase = sum(stat.size_diff for stat in top_stats)

            # T011b: Memory increase should be minimal (< 10MB for 1000 searches)
            max_allowed_increase = 10 * 1024 * 1024  # 10MB
            assert total_increase < max_allowed_increase, \
                f"Memory leak detected: {total_increase / 1024 / 1024:.2f}MB increase " \
                f"(limit: {max_allowed_increase / 1024 / 1024:.2f}MB)"

            # Verify coordinator counter
            assert mcts._coordinator_searches == 1000

        finally:
            mcts.close()
            tracemalloc.stop()

    def test_coordinator_defensive_restart_logic(self):
        """Verify coordinator restart logic works when coordinator stops externally."""
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

            # First search - starts coordinator
            mcts.search(state, simulations=5)
            assert mcts._coordinator_started is True
            assert mcts._coordinator_searches == 1

            # Simulate external coordinator stop (edge case)
            mcts._coordinator.stop()
            mcts._coordinator_started = False

            # T011b: Next search should detect stopped coordinator and restart
            mcts.reset()
            mcts.search(state, simulations=5)

            # Coordinator should be started again
            assert mcts._coordinator_started is True, \
                "Coordinator should be restarted after external stop"
            assert mcts._coordinator_searches == 2, \
                "Search counter should increment after restart"

        finally:
            mcts.close()

    def test_sync_mode_no_coordinator_metrics(self):
        """Verify synchronous mode doesn't create coordinator or track metrics."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=False  # Synchronous mode
        )

        try:
            state = alphazero_py.GomokuState()

            # Run searches
            mcts.search(state, simulations=10)
            mcts.reset()
            mcts.search(state, simulations=10)

            # Verify no coordinator
            assert mcts._coordinator is None
            assert mcts._coordinator_started is False

            # Verify stats don't include coordinator metrics
            stats = mcts.get_statistics()
            assert 'coordinator_searches' not in stats
            assert 'coordinator_started' not in stats

        finally:
            mcts.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
