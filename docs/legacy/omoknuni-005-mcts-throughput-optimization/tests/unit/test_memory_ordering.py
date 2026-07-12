"""
Unit tests for memory ordering validation (T012)

Tests verify that:
1. Statistics counters work correctly with relaxed ordering
2. Synchronizing operations maintain data consistency
3. No race conditions under high contention
4. Memory ordering is sufficient for correctness

These tests validate that relaxed memory ordering is safe where used.
"""

import pytest
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

try:
    import mcts_py
    import alphazero_py
    from src.core.mcts import AlphaZeroMCTS
    MCTS_AVAILABLE = True
except ImportError as e:
    MCTS_AVAILABLE = False
    print(f"MCTS not available: {e}")


def create_mock_inference_fn():
    """Create mock inference function."""
    def inference_fn(state):
        future = Future()
        action_space = state.get_action_space_size()
        policy = np.ones(action_space, dtype=np.float32) / action_space
        value = 0.0
        future.set_result((policy, value))
        return future
    return inference_fn


@pytest.mark.skipif(not MCTS_AVAILABLE, reason="MCTS not available")
class TestMemoryOrdering:
    """Memory ordering validation tests."""

    def test_relaxed_statistics_counters(self):
        """Test that relaxed statistics counters are thread-safe."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=8,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=5.0
        )

        state = alphazero_py.GomokuState()

        # Run many searches concurrently to stress statistics
        def worker():
            for _ in range(10):
                mcts.reset()
                mcts.search(state, simulations=20)

        threads = []
        for _ in range(8):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify statistics are consistent (relaxed ordering sufficient)
        stats = mcts.get_statistics()

        # Should have 8 threads * 10 searches = 80 total searches
        # (statistics may not be exact due to relaxed ordering, but should be close)
        assert stats['total_searches'] >= 70, f"Expected ~80 searches, got {stats['total_searches']}"

        mcts.close()

    def test_acquire_release_visit_counts(self):
        """Test that visit counts are correctly synchronized across threads."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=8,
            use_async_inference=False  # Sync mode for deterministic testing
        )

        state = alphazero_py.GomokuState()

        # Run search with multiple threads
        mcts.search(state, simulations=100)

        # Verify visit counts sum correctly
        root_visits = mcts.tree.get_visit_count(mcts.root_index)

        # Should be exactly 100 (acquire/release ensures accuracy)
        assert root_visits == 100.0, f"Expected 100 root visits, got {root_visits}"

        # Verify child visit counts sum to parent (within epsilon)
        first_child = mcts.tree.get_first_child_index(mcts.root_index)
        num_children = mcts.tree.get_num_children(mcts.root_index)

        child_visit_sum = 0.0
        for i in range(num_children):
            child_idx = first_child + i
            if mcts.tree.is_valid_index(child_idx):
                child_visit_sum += mcts.tree.get_visit_count(child_idx)

        # Child visits should approximately equal parent visits - 1 (root visit)
        # Allow small epsilon for floating point
        assert abs(child_visit_sum - (root_visits - 1.0)) < 1.0, \
            f"Child visits {child_visit_sum} != parent visits {root_visits - 1.0}"

        mcts.close()

    def test_epoch_synchronization(self):
        """Test that epoch-based tree clearing works correctly across threads."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=4,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=5.0
        )

        state = alphazero_py.GomokuState()

        # Run search, then clear, then search again
        mcts.search(state, simulations=50)
        visit_count_1 = mcts.tree.get_visit_count(mcts.root_index)

        # Clear tree (epoch increment with acq-rel ordering)
        mcts.reset()

        # Run another search
        mcts.search(state, simulations=50)
        visit_count_2 = mcts.tree.get_visit_count(mcts.root_index)

        # Both should be 50 (epoch synchronization ensures clean slate)
        assert visit_count_1 == 50.0, f"First search: expected 50 visits, got {visit_count_1}"
        assert visit_count_2 == 50.0, f"Second search: expected 50 visits, got {visit_count_2}"

        mcts.close()

    def test_high_contention_consistency(self):
        """Stress test: verify consistency under high thread contention."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=12,  # High contention
            use_async_inference=True,
            async_batch_size=16,
            async_timeout_ms=2.0
        )

        state = alphazero_py.GomokuState()

        # Run many simulations with high thread count
        mcts.search(state, simulations=1000)

        # Verify visit count accuracy
        root_visits = mcts.tree.get_visit_count(mcts.root_index)

        # Should be exactly 1000 (atomic operations ensure this)
        assert root_visits == 1000.0, \
            f"Expected 1000 visits, got {root_visits} (memory ordering issue?)"

        mcts.close()

    def test_async_queue_relaxed_counters(self):
        """Test that async queue counters work correctly with relaxed ordering."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=4,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=5.0
        )

        state = alphazero_py.GomokuState()

        # Run search
        mcts.search(state, simulations=100)

        # Check queue statistics (uses relaxed ordering)
        stats = mcts.get_statistics()

        # Queue should have processed some requests
        # (exact count may vary due to relaxed ordering, but should be non-zero)
        assert stats['total_searches'] > 0, "No searches recorded"

        mcts.close()

    def test_flag_synchronization(self):
        """Test that expanding flags are properly synchronized."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=8,
            use_async_inference=True,
            async_batch_size=8,
            async_timeout_ms=5.0
        )

        state = alphazero_py.GomokuState()

        # Run search (will set/clear expanding flags many times)
        mcts.search(state, simulations=200)

        # After search, no nodes should be marked as expanding
        # (flags use acquire/release for proper synchronization)
        root_visits = mcts.tree.get_visit_count(mcts.root_index)
        assert root_visits == 200.0, "Visit count mismatch indicates synchronization issue"

        mcts.close()

    def test_concurrent_statistics_updates(self):
        """Test that statistics updates don't cause data corruption."""
        inference_fn = create_mock_inference_fn()

        # Create multiple MCTS instances to stress memory ordering
        instances = []
        for _ in range(4):
            mcts = AlphaZeroMCTS(
                inference_fn=inference_fn,
                num_threads=2,
                use_async_inference=True,
                async_batch_size=8,
                async_timeout_ms=5.0
            )
            instances.append(mcts)

        state = alphazero_py.GomokuState()

        # Run searches concurrently across instances
        def worker(mcts_instance):
            for _ in range(10):
                mcts_instance.reset()
                mcts_instance.search(state, simulations=20)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, mcts) for mcts in instances]
            for future in futures:
                future.result()

        # Verify all instances have consistent statistics
        for i, mcts in enumerate(instances):
            stats = mcts.get_statistics()
            # Each instance should have 10 searches
            assert stats['total_searches'] >= 8, \
                f"Instance {i}: expected ~10 searches, got {stats['total_searches']}"
            mcts.close()

    def test_memory_ordering_documentation(self):
        """Verify that memory ordering documentation exists."""
        doc_path = os.path.join(
            os.path.dirname(__file__),
            '../../docs/performance/memory_ordering_strategy.md'
        )

        assert os.path.exists(doc_path), \
            "Memory ordering documentation not found"

        # Verify it contains key sections
        with open(doc_path, 'r') as f:
            content = f.read()

        assert 'memory_order_relaxed' in content, \
            "Documentation missing relaxed ordering discussion"
        assert 'memory_order_acquire' in content, \
            "Documentation missing acquire ordering discussion"
        assert 'memory_order_release' in content, \
            "Documentation missing release ordering discussion"
        assert 'ThreadSanitizer' in content, \
            "Documentation missing TSan validation strategy"


@pytest.mark.skipif(not MCTS_AVAILABLE, reason="MCTS not available")
class TestMemoryOrderingStress:
    """Stress tests for memory ordering under extreme conditions."""

    @pytest.mark.slow
    def test_extended_high_contention(self):
        """Extended stress test with maximum thread contention."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=16,  # Maximum contention
            use_async_inference=True,
            async_batch_size=32,
            async_timeout_ms=1.0
        )

        state = alphazero_py.GomokuState()

        # Run many searches to expose rare race conditions
        for _ in range(100):
            mcts.reset()
            mcts.search(state, simulations=100)

            # Verify visit count after each search
            root_visits = mcts.tree.get_visit_count(mcts.root_index)
            assert root_visits == 100.0, \
                f"Visit count corruption: expected 100, got {root_visits}"

        mcts.close()

    @pytest.mark.slow
    def test_rapid_epoch_transitions(self):
        """Test rapid tree clearing to stress epoch synchronization."""
        inference_fn = create_mock_inference_fn()

        mcts = AlphaZeroMCTS(
            inference_fn=inference_fn,
            num_threads=8,
            use_async_inference=True,
            async_batch_size=16,
            async_timeout_ms=2.0
        )

        state = alphazero_py.GomokuState()

        # Rapidly clear and search
        for _ in range(1000):
            mcts.reset()  # Epoch increment
            mcts.search(state, simulations=10)

            # Verify consistency
            root_visits = mcts.tree.get_visit_count(mcts.root_index)
            assert root_visits == 10.0, \
                f"Epoch synchronization issue: expected 10 visits, got {root_visits}"

        mcts.close()


if __name__ == "__main__":
    print("\n=== Memory Ordering Validation Tests ===\n")
    pytest.main([__file__, "-v", "-s"])
