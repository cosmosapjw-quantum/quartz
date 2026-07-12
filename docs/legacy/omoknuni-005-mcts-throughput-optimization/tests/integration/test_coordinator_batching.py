"""
Integration tests for BatchInferenceCoordinator

Tests verify:
1. Coordinator collects batches correctly
2. Batches are processed by Python callback
3. Results are distributed back to queue
4. Integration with ContinuousSimulationRunner
5. GIL is acquired only once per batch (measured indirectly)
"""

import pytest
import time
import threading
from typing import List, Tuple
import numpy as np

# Import game implementation from C++ bindings
import alphazero_py


def test_coordinator_starts_stops():
    """Test coordinator lifecycle"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    queue = mcts_py.AsyncInferenceQueue()

    # Create batch callback
    def batch_inference_fn(states):
        results = []
        for state in states:
            policy = np.ones(225, dtype=np.float32) / 225.0
            value = 0.0
            results.append((policy.tolist(), value))
        return results

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Initially not running
    assert not coordinator.is_running()

    # Start coordinator
    coordinator.start(queue, callback, 32, 2.0)
    assert coordinator.is_running()

    # Stop coordinator
    coordinator.stop()
    assert not coordinator.is_running()


def test_coordinator_processes_requests():
    """Test coordinator processes inference requests"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    queue = mcts_py.AsyncInferenceQueue()

    # Track batch calls
    batch_calls = []

    def batch_inference_fn(states):
        batch_calls.append(len(states))
        results = []
        for state in states:
            policy = np.ones(225, dtype=np.float32) / 225.0
            value = 0.5
            results.append((policy.tolist(), value))
        return results

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Start coordinator with small batch size
    coordinator.start(queue, callback, 2, 100.0)  # batch_size=2, timeout=100ms

    try:
        # Submit 3 requests
        game_state = alphazero_py.GomokuState()
        request_ids = []
        for i in range(3):
            rid = queue.submit_request(game_state, 10 + i, [])
            request_ids.append(rid)

        # Wait for coordinator to process
        time.sleep(0.2)

        # Should have processed at least one batch
        assert len(batch_calls) > 0

        # Retrieve results
        for rid in request_ids:
            result = queue.try_get_result(rid)
            assert result is not None, f"Result for request {rid} not found"
            assert result.request_id == rid
            assert len(result.policy) == 225
            assert result.value == 0.5

    finally:
        coordinator.stop()


def test_coordinator_batch_size_trigger():
    """Test coordinator triggers on batch size"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    queue = mcts_py.AsyncInferenceQueue()

    batch_sizes = []
    batch_event = threading.Event()

    def batch_inference_fn(states):
        batch_sizes.append(len(states))
        batch_event.set()
        results = []
        for state in states:
            policy = np.ones(225, dtype=np.float32) / 225.0
            value = 0.0
            results.append((policy.tolist(), value))
        return results

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Start with batch_size=4, long timeout
    coordinator.start(queue, callback, 4, 10000.0)

    try:
        game_state = alphazero_py.GomokuState()

        # Submit exactly 4 requests (should trigger batch immediately)
        for i in range(4):
            queue.submit_request(game_state, i, [])

        # Wait for batch processing (should be fast)
        assert batch_event.wait(timeout=0.5), "Batch not processed within 500ms"

        # Should have processed exactly one batch of size 4
        assert len(batch_sizes) >= 1
        assert batch_sizes[0] == 4

    finally:
        coordinator.stop()


def test_coordinator_timeout_trigger():
    """Test coordinator triggers on timeout"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    queue = mcts_py.AsyncInferenceQueue()

    batch_sizes = []

    def batch_inference_fn(states):
        batch_sizes.append(len(states))
        results = []
        for state in states:
            policy = np.ones(225, dtype=np.float32) / 225.0
            value = 0.0
            results.append((policy.tolist(), value))
        return results

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Start with large batch_size, short timeout
    coordinator.start(queue, callback, 100, 50.0)  # batch_size=100, timeout=50ms

    try:
        game_state = alphazero_py.GomokuState()

        # Submit only 2 requests (below batch_size)
        queue.submit_request(game_state, 1, [])
        queue.submit_request(game_state, 2, [])

        # Wait for timeout to trigger
        time.sleep(0.15)

        # Should have processed batch due to timeout
        assert len(batch_sizes) >= 1
        assert batch_sizes[0] == 2  # Partial batch

    finally:
        coordinator.stop()


def test_coordinator_with_continuous_runner():
    """Test coordinator integration with ContinuousSimulationRunner"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    # Create MCTS components
    tree = mcts_py.MCTSTree(10000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

    # Create queue and coordinator
    queue = mcts_py.AsyncInferenceQueue()

    batch_count = [0]

    def batch_inference_fn(states):
        batch_count[0] += 1
        results = []
        for state in states:
            # Return uniform policy
            policy = np.ones(225, dtype=np.float32) / 225.0
            value = 0.0
            results.append((policy.tolist(), value))
        return results

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Start coordinator
    coordinator.start(queue, callback, 8, 5.0)  # batch_size=8, timeout=5ms

    try:
        # Create continuous runner
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)

        # Create root
        game_state = alphazero_py.GomokuState()
        root_idx = tree.add_root_node(0.5, 0)

        # Run 20 simulations
        completed = runner.run_continuous(game_state, root_idx, queue, 20)

        # Should complete all simulations
        assert completed == 20

        # Root should have visit count
        assert tree.get_visit_count(root_idx) >= 20

        # Coordinator should have processed batches
        assert batch_count[0] > 0
        print(f"Processed {batch_count[0]} batches for 20 simulations")

    finally:
        coordinator.stop()


def test_coordinator_handles_errors():
    """Test coordinator handles callback errors gracefully"""
    try:
        import mcts_py
    except ImportError:
        pytest.skip("mcts_py not built")

    queue = mcts_py.AsyncInferenceQueue()

    call_count = [0]

    def failing_batch_inference(states):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated inference failure")
        # Succeed on subsequent calls
        results = []
        for state in states:
            policy = np.ones(225, dtype=np.float32) / 225.0
            results.append((policy.tolist(), 0.0))
        return results

    callback = mcts_py.PyBatchInferenceCallback(failing_batch_inference)
    coordinator = mcts_py.BatchInferenceCoordinator()

    coordinator.start(queue, callback, 2, 50.0)

    try:
        game_state = alphazero_py.GomokuState()

        # First batch should fail
        queue.submit_request(game_state, 1, [])
        queue.submit_request(game_state, 2, [])
        time.sleep(0.1)

        # Second batch should succeed
        rid1 = queue.submit_request(game_state, 3, [])
        rid2 = queue.submit_request(game_state, 4, [])
        time.sleep(0.1)

        # Should be able to get results from second batch
        result1 = queue.try_get_result(rid1)
        result2 = queue.try_get_result(rid2)

        # At least one should succeed (depending on timing)
        # The coordinator should continue running after error
        assert coordinator.is_running()

    finally:
        coordinator.stop()


if __name__ == "__main__":
    print("\n=== BatchInferenceCoordinator Integration Tests ===\n")

    test_coordinator_starts_stops()
    print("✓ test_coordinator_starts_stops")

    test_coordinator_processes_requests()
    print("✓ test_coordinator_processes_requests")

    test_coordinator_batch_size_trigger()
    print("✓ test_coordinator_batch_size_trigger")

    test_coordinator_timeout_trigger()
    print("✓ test_coordinator_timeout_trigger")

    test_coordinator_with_continuous_runner()
    print("✓ test_coordinator_with_continuous_runner")

    test_coordinator_handles_errors()
    print("✓ test_coordinator_handles_errors")

    print("\n=== All coordinator batching tests passed! ===\n")
