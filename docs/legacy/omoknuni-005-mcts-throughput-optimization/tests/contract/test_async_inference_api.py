"""
Contract Tests for AsyncInferenceQueue API

Tests Python bindings for AsyncInferenceQueue to validate API surface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import mcts_py
import alphazero_py


class TestAsyncInferenceQueueAPI:
    """Contract tests for AsyncInferenceQueue Python bindings."""

    def test_queue_instantiation(self):
        """Test AsyncInferenceQueue can be instantiated."""
        queue = mcts_py.AsyncInferenceQueue()
        assert queue is not None
        assert isinstance(queue, mcts_py.AsyncInferenceQueue)

    def test_queue_initial_state(self):
        """Test queue starts empty."""
        queue = mcts_py.AsyncInferenceQueue()

        assert queue.pending_count() == 0
        assert queue.results_count() == 0
        assert not queue.has_results()

    def test_submit_request(self):
        """Test submitting inference request."""
        queue = mcts_py.AsyncInferenceQueue()
        state = alphazero_py.GomokuState()

        # Submit request
        req_id = queue.submit_request(state, 0, [0])

        assert isinstance(req_id, int)
        assert req_id == 0  # First request ID
        assert queue.pending_count() == 1

    def test_unique_request_ids(self):
        """Test request IDs are unique and sequential."""
        queue = mcts_py.AsyncInferenceQueue()

        ids = []
        for i in range(10):
            state = alphazero_py.GomokuState()
            req_id = queue.submit_request(state, i, [i])
            ids.append(req_id)

        # All unique
        assert len(set(ids)) == 10
        # Sequential starting from 0
        assert ids == list(range(10))
        assert queue.pending_count() == 10

    def test_collect_batch_size_trigger(self):
        """Test batch collection on size trigger."""
        queue = mcts_py.AsyncInferenceQueue()

        # Submit 32 requests
        for i in range(32):
            state = alphazero_py.GomokuState()
            queue.submit_request(state, i, [i])

        # Collect batch (min_size=32, timeout=1000ms)
        batch = queue.collect_batch(32, 1000.0)

        assert len(batch) == 32
        assert queue.pending_count() == 0

    def test_collect_batch_timeout_trigger(self):
        """Test batch collection on timeout trigger."""
        import time
        queue = mcts_py.AsyncInferenceQueue()

        # Submit only 10 requests (less than min_size=32)
        for i in range(10):
            state = alphazero_py.GomokuState()
            queue.submit_request(state, i, [i])

        # Collect batch with short timeout
        start = time.time()
        batch = queue.collect_batch(32, 10.0)  # 10ms timeout
        elapsed = (time.time() - start) * 1000

        assert len(batch) == 10
        assert queue.pending_count() == 0
        assert 9 <= elapsed <= 20  # Timeout ~10ms

    def test_inference_result_creation(self):
        """Test creating InferenceResult."""
        result = mcts_py.InferenceResult()
        result.request_id = 42
        result.policy = [0.01] * 225
        result.value = 0.5

        assert result.request_id == 42
        assert len(result.policy) == 225
        assert result.value == 0.5

    def test_submit_and_retrieve_results(self):
        """Test submitting and retrieving results."""
        queue = mcts_py.AsyncInferenceQueue()

        # Create and submit result
        result = mcts_py.InferenceResult()
        result.request_id = 100
        result.policy = [0.01] * 225
        result.value = 0.7

        queue.submit_results([result])

        # Verify available
        assert queue.has_results()
        assert queue.results_count() == 1

        # Retrieve result
        retrieved = queue.try_get_result(100)
        assert retrieved is not None
        assert retrieved.request_id == 100
        assert abs(retrieved.value - 0.7) < 0.0001

        # Verify consumed
        assert not queue.has_results()
        assert queue.results_count() == 0

    def test_try_get_nonexistent_result(self):
        """Test retrieving non-existent result returns None."""
        queue = mcts_py.AsyncInferenceQueue()

        result = queue.try_get_result(999)
        assert result is None

    def test_multiple_results(self):
        """Test submitting multiple results."""
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch of results
        results = []
        for i in range(10):
            result = mcts_py.InferenceResult()
            result.request_id = i
            result.policy = [0.01] * 225
            result.value = float(i) * 0.1
            results.append(result)

        queue.submit_results(results)

        # Verify all available
        assert queue.results_count() == 10

        # Retrieve in different order
        r5 = queue.try_get_result(5)
        assert r5.request_id == 5
        assert abs(r5.value - 0.5) < 0.0001

        r0 = queue.try_get_result(0)
        assert r0.request_id == 0

        assert queue.results_count() == 8

    def test_result_consumed_after_retrieval(self):
        """Test result is consumed after retrieval."""
        queue = mcts_py.AsyncInferenceQueue()

        result = mcts_py.InferenceResult()
        result.request_id = 200
        result.policy = [0.01] * 225
        result.value = 0.3
        queue.submit_results([result])

        # Retrieve once
        first = queue.try_get_result(200)
        assert first is not None

        # Try again - should be None (consumed)
        second = queue.try_get_result(200)
        assert second is None

    def test_memory_usage(self):
        """Test memory usage reporting."""
        queue = mcts_py.AsyncInferenceQueue()

        initial_mem = queue.get_memory_usage()

        # Submit 100 requests
        for i in range(100):
            state = alphazero_py.GomokuState()
            queue.submit_request(state, i, [i])

        mem_with_requests = queue.get_memory_usage()
        assert mem_with_requests > initial_mem

    def test_empty_batch_on_timeout(self):
        """Test collect_batch returns empty if no requests and timeout."""
        queue = mcts_py.AsyncInferenceQueue()

        # Don't submit any requests
        batch = queue.collect_batch(32, 10.0)

        assert len(batch) == 0

    def test_docstrings_present(self):
        """Test API has docstrings."""
        queue = mcts_py.AsyncInferenceQueue

        assert queue.__doc__ is not None
        assert "async inference queue" in queue.__doc__.lower()

        # Check method docstrings
        assert queue.submit_request.__doc__ is not None
        assert queue.collect_batch.__doc__ is not None
        assert queue.try_get_result.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
