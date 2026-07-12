"""
Integration tests for SearchCoordinator shutdown logic.

Tests that the SearchCoordinator shuts down cleanly without thread leaks,
properly cancels active searches, stops the inference worker, and joins
background threads.

HOWTO-RUN-TESTS:
===============
# Run coordinator shutdown tests
python -m pytest tests/integration/test_coordinator_shutdown.py -v

# Run with verbose output
python -m pytest tests/integration/test_coordinator_shutdown.py -v -s

# Run specific test
python -m pytest tests/integration/test_coordinator_shutdown.py::TestCoordinatorShutdown::test_clean_shutdown -v
"""

import pytest
import sys
from pathlib import Path
import threading
import time
from unittest.mock import Mock, MagicMock
from concurrent.futures import Future

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.search_coordinator import SearchCoordinator, SearchRequest
from src.neural.inference_worker import GPUInferenceWorker


class MockInferenceWorker:
    """Mock inference worker for testing."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.warmup_completed = False
        self._warmup_completed = False

    def start(self):
        """Mock start."""
        self.started = True

    def batch_inference(self, features_batch):
        """Mock batch inference."""
        import numpy as np
        batch_size = len(features_batch)
        action_space = 225  # Gomoku 15x15
        policy_batch = np.ones((batch_size, action_space), dtype=np.float32) / action_space
        value_batch = np.zeros(batch_size, dtype=np.float32)
        return policy_batch, value_batch

    def warmup(self, input_shape):
        """Mock warmup."""
        self._warmup_completed = True
        self.warmup_completed = True

    def stop(self):
        """Mock stop."""
        self.stopped = True


class TestCoordinatorShutdown:
    """Test SearchCoordinator shutdown logic."""

    @pytest.fixture
    def mock_inference_worker(self):
        """Create mock inference worker."""
        return MockInferenceWorker()

    @pytest.fixture
    def coordinator(self, mock_inference_worker):
        """Create SearchCoordinator for testing."""
        return SearchCoordinator(
            inference_worker=mock_inference_worker,
            max_threads=4,
            max_queue_size=100,
            monitoring_interval=0.1
        )

    def test_clean_shutdown(self, coordinator):
        """Test that coordinator shuts down cleanly."""
        # Start coordinator
        coordinator.start()
        assert coordinator.running is True

        # Get initial thread count
        initial_thread_count = threading.active_count()

        # Stop coordinator
        coordinator.stop()

        # Verify coordinator stopped
        assert coordinator.running is False
        assert coordinator.shutdown_event.is_set()

        # Wait a bit for threads to finish
        time.sleep(0.5)

        # Verify no thread leak
        final_thread_count = threading.active_count()
        assert final_thread_count <= initial_thread_count, \
            f"Thread leak detected: {final_thread_count} threads vs {initial_thread_count} initial"

    def test_stop_cancels_active_searches(self, coordinator):
        """Test that stop() cancels all active searches."""
        coordinator.start()

        # Create mock search request
        mock_game_state = Mock()
        mock_game_state.get_tensor_representation.return_value = [[0] * 225]  # Mock features

        request = SearchRequest(
            request_id="test-1",
            game_state=mock_game_state,
            simulations=100
        )

        # Submit search (it will be pending/running)
        future = coordinator.submit_search(request)

        # Verify search is active
        assert len(coordinator.active_searches) > 0

        # Stop coordinator
        coordinator.stop()

        # Verify search was cancelled
        assert len(coordinator.active_searches) == 0
        # Future might be cancelled or completed, both are acceptable
        assert future.done() or future.cancelled()

    def test_stop_inference_worker(self, coordinator, mock_inference_worker):
        """Test that stop() stops the inference worker."""
        coordinator.start()

        # Verify worker not stopped initially
        assert mock_inference_worker.stopped is False

        # Stop coordinator
        coordinator.stop()

        # Verify inference worker was stopped
        assert mock_inference_worker.stopped is True

    def test_thread_pool_shutdown(self, coordinator):
        """Test that thread pool shuts down properly."""
        coordinator.start()

        # Verify thread pool exists and is running
        assert hasattr(coordinator, 'thread_pool')
        assert coordinator.thread_pool._shutdown is False

        # Stop coordinator
        coordinator.stop()

        # Verify thread pool is shut down
        assert coordinator.thread_pool._shutdown is True

    def test_background_threads_join(self, coordinator):
        """Test that background threads are joined."""
        coordinator.start()

        # Get references to background threads
        inference_thread = coordinator.inference_coordinator_thread
        metrics_thread = coordinator.metrics_monitor_thread

        # Verify threads are alive
        assert inference_thread.is_alive()
        assert metrics_thread.is_alive()

        # Stop coordinator
        coordinator.stop()

        # Verify threads have stopped
        assert not inference_thread.is_alive(), "Inference coordinator thread still alive"
        assert not metrics_thread.is_alive(), "Metrics monitor thread still alive"

    def test_repeated_start_stop(self, coordinator):
        """Test starting and stopping coordinator multiple times."""
        initial_thread_count = threading.active_count()

        for i in range(3):
            # Start coordinator
            coordinator.start()
            assert coordinator.running is True

            # Stop coordinator
            coordinator.stop()
            assert coordinator.running is False

            # Wait for threads to finish
            time.sleep(0.2)

        # Verify no thread accumulation
        final_thread_count = threading.active_count()
        assert final_thread_count <= initial_thread_count + 2, \
            f"Thread accumulation detected: {final_thread_count} threads after 3 cycles"

    def test_stop_when_not_running(self, coordinator):
        """Test that stop() is safe when coordinator not running."""
        # Don't start coordinator
        assert coordinator.running is False

        # Stop should be no-op and not raise
        coordinator.stop()

        # Verify still not running
        assert coordinator.running is False

    def test_double_stop(self, coordinator):
        """Test that calling stop() twice is safe."""
        coordinator.start()
        assert coordinator.running is True

        # First stop
        coordinator.stop()
        assert coordinator.running is False

        # Second stop should be no-op
        coordinator.stop()
        assert coordinator.running is False

    def test_shutdown_with_pending_inference(self, coordinator):
        """Test shutdown with pending inference requests."""
        coordinator.start()

        # Create mock game state
        mock_game_state = Mock()
        mock_game_state.get_tensor_representation.return_value = [[0] * 225]

        # Submit search that will trigger inference
        request = SearchRequest(
            request_id="test-inf",
            game_state=mock_game_state,
            simulations=10
        )
        future = coordinator.submit_search(request)

        # Give it a moment to start
        time.sleep(0.1)

        # Stop coordinator
        coordinator.stop()

        # Verify clean shutdown even with pending work
        assert coordinator.running is False
        assert len(coordinator.active_searches) == 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
