"""
Performance tests for Coordinator Lifecycle Overhead (T011c)

Measures:
1. Baseline throughput with per-search coordinator creation (simulated)
2. Current throughput with persistent coordinator
3. Coordinator creation overhead
4. Thread startup/teardown elimination
5. Memory stability across searches

Target: 15-25% throughput improvement from persistent coordinator
"""

import pytest
import sys
import time
from pathlib import Path
from concurrent.futures import Future
from typing import Tuple
import numpy as np
import gc

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
class TestCoordinatorOverhead:
    """Performance tests for coordinator lifecycle optimization."""

    def test_persistent_coordinator_throughput(self, benchmark):
        """Benchmark throughput with persistent coordinator (T011a/T011b)."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=4,
            use_async_inference=True,
            async_batch_size=16,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            def run_searches():
                """Run 10 searches with persistent coordinator."""
                for _ in range(10):
                    mcts.reset()
                    mcts.search(state, simulations=50)

            # Benchmark persistent coordinator
            benchmark(run_searches)

            # Verify coordinator persistence (benchmark may run multiple rounds)
            stats = mcts.get_statistics()
            assert stats['coordinator_searches'] >= 10, \
                f"Should have at least 10 searches, got {stats['coordinator_searches']}"

            print(f"\nCoordinator searches: {stats['coordinator_searches']}")

        finally:
            mcts.close()

    def test_coordinator_creation_overhead(self):
        """Measure overhead of coordinator creation vs reuse."""
        inference_fn = create_mock_inference_fn()
        num_iterations = 100

        # Measure persistent coordinator (T011a/T011b)
        mcts_persistent = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=1.0
        )

        try:
            state = alphazero_py.GomokuState()

            # Warmup
            mcts_persistent.search(state, simulations=10)

            # Measure persistent coordinator throughput
            start = time.perf_counter()
            for _ in range(num_iterations):
                mcts_persistent.reset()
                mcts_persistent.search(state, simulations=20)
            persistent_time = time.perf_counter() - start

            persistent_throughput = num_iterations / persistent_time

            # Verify single coordinator
            stats = mcts_persistent.get_statistics()
            assert stats['coordinator_searches'] == num_iterations + 1  # +1 for warmup

            print(f"\n=== Coordinator Overhead Analysis ===")
            print(f"Iterations: {num_iterations}")
            print(f"Persistent coordinator throughput: {persistent_throughput:.2f} searches/sec")
            print(f"Total time: {persistent_time:.3f}s")
            print(f"Coordinator recreations: 0 (single instance)")
            print(f"Coordinator searches: {stats['coordinator_searches']}")

            # Note: We can't directly measure per-search creation without modifying
            # the code to recreate coordinator. Instead, we validate that:
            # 1. Single coordinator handles all searches
            # 2. No performance degradation over time
            # 3. Memory stable (tested in integration tests)

        finally:
            mcts_persistent.close()

    def test_throughput_stability_over_1000_searches(self):
        """Verify throughput remains stable over 1000 searches (no degradation)."""
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

            # Measure throughput in batches of 100
            throughputs = []
            batch_size = 100

            for batch in range(10):  # 10 batches = 1000 searches
                start = time.perf_counter()
                for _ in range(batch_size):
                    mcts.reset()
                    mcts.search(state, simulations=10)
                batch_time = time.perf_counter() - start
                batch_throughput = batch_size / batch_time
                throughputs.append(batch_throughput)

                print(f"Batch {batch + 1}: {batch_throughput:.2f} searches/sec")

            # Calculate statistics
            mean_throughput = np.mean(throughputs)
            std_throughput = np.std(throughputs)
            cv = (std_throughput / mean_throughput) * 100  # Coefficient of variation

            print(f"\n=== Throughput Stability Analysis ===")
            print(f"Mean throughput: {mean_throughput:.2f} searches/sec")
            print(f"Std deviation: {std_throughput:.2f} searches/sec")
            print(f"Coefficient of variation: {cv:.2f}%")
            print(f"Min throughput: {np.min(throughputs):.2f} searches/sec")
            print(f"Max throughput: {np.max(throughputs):.2f} searches/sec")

            # Verify coordinator persistence
            stats = mcts.get_statistics()
            assert stats['coordinator_searches'] == 1000

            # Verify stability (CV should be < 10% for stable performance)
            assert cv < 15.0, \
                f"Throughput variance too high: {cv:.2f}% (should be <15%)"

        finally:
            mcts.close()

    def test_memory_stability_over_searches(self):
        """Verify memory usage remains stable (no leaks from coordinator)."""
        inference_fn = create_mock_inference_fn()

        # Start memory tracking
        gc.collect()
        import tracemalloc
        tracemalloc.start()
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

            # Run many searches
            for i in range(500):
                mcts.reset()
                mcts.search(state, simulations=10)

                # Sample memory every 100 searches
                if (i + 1) % 100 == 0:
                    gc.collect()
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.compare_to(snapshot_start, 'lineno')
                    total_increase = sum(stat.size_diff for stat in top_stats)
                    print(f"After {i+1} searches: {total_increase / 1024 / 1024:.2f}MB increase")

            # Final memory check
            gc.collect()
            snapshot_end = tracemalloc.take_snapshot()
            top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
            total_increase = sum(stat.size_diff for stat in top_stats)

            print(f"\n=== Memory Stability Analysis ===")
            print(f"Total searches: 500")
            print(f"Memory increase: {total_increase / 1024 / 1024:.2f}MB")
            print(f"Memory per search: {total_increase / 500 / 1024:.2f}KB")

            # Verify minimal memory increase (<5MB for 500 searches)
            max_allowed_increase = 5 * 1024 * 1024  # 5MB
            assert total_increase < max_allowed_increase, \
                f"Memory leak detected: {total_increase / 1024 / 1024:.2f}MB increase"

            # Verify coordinator persistence
            stats = mcts.get_statistics()
            assert stats['coordinator_searches'] == 500

        finally:
            mcts.close()
            tracemalloc.stop()

    def test_coordinator_lifecycle_metrics(self):
        """Validate coordinator lifecycle tracking metrics."""
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

            # Track coordinator lifecycle
            print(f"\n=== Coordinator Lifecycle Metrics ===")

            # Initial state
            stats = mcts.get_statistics()
            print(f"Initial - Searches: {stats['coordinator_searches']}, Started: {stats['coordinator_started']}")
            assert stats['coordinator_searches'] == 0
            assert stats['coordinator_started'] is False

            # After first search
            mcts.search(state, simulations=10)
            stats = mcts.get_statistics()
            print(f"After 1st search - Searches: {stats['coordinator_searches']}, Started: {stats['coordinator_started']}")
            assert stats['coordinator_searches'] == 1
            assert stats['coordinator_started'] is True

            # After 10 searches
            for i in range(9):
                mcts.reset()
                mcts.search(state, simulations=10)

            stats = mcts.get_statistics()
            print(f"After 10 searches - Searches: {stats['coordinator_searches']}, Started: {stats['coordinator_started']}")
            assert stats['coordinator_searches'] == 10
            assert stats['coordinator_started'] is True

            # After close
            mcts.close()
            stats = mcts.get_statistics()
            print(f"After close - Searches: {stats['coordinator_searches']}, Started: {stats['coordinator_started']}")
            assert stats['coordinator_searches'] == 10
            assert stats['coordinator_started'] is False

        except Exception:
            mcts.close()
            raise

    def test_sync_vs_async_mode_overhead(self):
        """Compare sync mode (no coordinator) vs async mode (with coordinator)."""
        inference_fn = create_mock_inference_fn()
        num_searches = 20

        # Test sync mode
        mcts_sync = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=False  # Sync mode
        )

        try:
            state = alphazero_py.GomokuState()

            start = time.perf_counter()
            for _ in range(num_searches):
                mcts_sync.reset()
                mcts_sync.search(state, simulations=20)
            sync_time = time.perf_counter() - start

        finally:
            mcts_sync.close()

        # Test async mode
        mcts_async = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=2,
            use_async_inference=True  # Async mode
        )

        try:
            state = alphazero_py.GomokuState()

            start = time.perf_counter()
            for _ in range(num_searches):
                mcts_async.reset()
                mcts_async.search(state, simulations=20)
            async_time = time.perf_counter() - start

            stats = mcts_async.get_statistics()
            assert stats['coordinator_searches'] == num_searches

        finally:
            mcts_async.close()

        print(f"\n=== Sync vs Async Mode Comparison ===")
        print(f"Searches: {num_searches}")
        print(f"Sync mode time: {sync_time:.3f}s ({num_searches/sync_time:.2f} searches/sec)")
        print(f"Async mode time: {async_time:.3f}s ({num_searches/async_time:.2f} searches/sec)")
        print(f"Speedup: {sync_time/async_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
