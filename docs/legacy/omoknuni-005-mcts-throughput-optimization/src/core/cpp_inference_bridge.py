"""
C++ Inference Bridge
====================

Bridge between C++ MCTS simulation runner and Python GPU inference worker.
Provides a simple callable interface for C++ code to request neural network
inference asynchronously.

This module is part of T016 (Inference bridge) in Phase 3 of the C++ MCTS
simulation runner implementation (spec 002-cpp-simulation-runner).
"""

import numpy as np
import logging
import time
from typing import Tuple, Callable, List
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from threading import Lock

from src.neural.inference_worker import GPUInferenceWorker
from src.utils.errors import InferenceError


class CppInferenceBridge:
    """Bridge between C++ MCTS runner and Python GPU inference worker.

    Provides a callable interface that:
    1. Extracts features from game state
    2. Submits to GPU inference worker
    3. Returns Future for asynchronous result retrieval
    4. Handles timeouts and error propagation
    5. Routes to CPU fallback when appropriate

    Usage:
        bridge = CppInferenceBridge(gpu_worker)
        future = bridge(game_state)
        policy, value = future.result(timeout=1.0)

    Args:
        inference_worker: GPU inference worker instance
        default_timeout: Default timeout for inference requests (seconds)
        enable_cpu_fallback: Enable automatic CPU fallback on GPU errors
    """

    def __init__(self,
                 inference_worker: GPUInferenceWorker,
                 default_timeout: float = 1.0,
                 enable_cpu_fallback: bool = True):
        self.inference_worker = inference_worker
        self.default_timeout = default_timeout
        self.enable_cpu_fallback = enable_cpu_fallback

        # Metrics tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._timeout_requests = 0
        self._cpu_fallback_requests = 0
        self._batch_requests = 0
        self._batch_failures = 0
        self._metrics_lock = Lock()

        # Logger
        self.logger = logging.getLogger(__name__)

    def __call__(self, game_state) -> Future[Tuple[np.ndarray, float]]:
        """Request neural network inference for a game state.

        This method is the main entry point for C++ code to request inference.
        It extracts features, submits to the inference worker, and returns a
        Future that will contain the (policy, value) result.

        Args:
            game_state: IGameState instance to evaluate

        Returns:
            Future[Tuple[np.ndarray, float]]: Future containing (policy, value)
                policy: Action probabilities (numpy array)
                value: Position evaluation (-1.0 to 1.0)

        Raises:
            InferenceError: If feature extraction fails
            ValueError: If game state is invalid
        """
        with self._metrics_lock:
            self._total_requests += 1

        # Create future for async result
        result_future = Future()

        try:
            # Extract features from game state
            features = self._extract_features(game_state)

            # Submit to inference worker
            # Note: GPUInferenceWorker.batch_inference() is synchronous, so we
            # wrap it in a way that allows async consumption
            self._submit_inference_request(features, result_future, game_state)

        except Exception as e:
            # Feature extraction failed, set exception on future
            self.logger.error(f"Feature extraction failed: {e}")
            result_future.set_exception(
                InferenceError(f"Feature extraction failed: {e}")
            )
            with self._metrics_lock:
                self._failed_requests += 1

        return result_future

    def batch_inference(self, positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run batched inference directly through the underlying worker.

        Args:
            positions: List of preprocessed feature tensors

        Returns:
            Tuple of (policies, values) numpy arrays
        """
        with self._metrics_lock:
            self._batch_requests += 1

        try:
            policies, values = self.inference_worker.batch_inference(positions)
            return policies, values
        except Exception as error:
            with self._metrics_lock:
                self._batch_failures += 1

            if self.enable_cpu_fallback and self._should_use_cpu_fallback(error):
                try:
                    return self._cpu_batch_fallback(positions)
                except Exception as fallback_error:
                    raise InferenceError(
                        f"CPU batch fallback failed: {fallback_error}"
                    ) from fallback_error

            raise

    def _extract_features(self, game_state) -> np.ndarray:
        """Extract feature tensor from game state.

        Args:
            game_state: IGameState instance

        Returns:
            Feature tensor (C, H, W) as numpy array

        Raises:
            ValueError: If features are invalid
        """
        # Try standard method first
        if hasattr(game_state, 'get_tensor_representation'):
            raw_features = game_state.get_tensor_representation()
        elif hasattr(game_state, 'extract_features'):
            raw_features = game_state.extract_features()
        else:
            raise ValueError(
                f"Game state {type(game_state)} does not have "
                "get_tensor_representation() or extract_features() method"
            )

        # Ensure correct format
        features = np.array(raw_features, dtype=np.float32, copy=True)

        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D feature tensor (C, H, W), got shape {features.shape}"
            )

        return features

    def _submit_inference_request(self,
                                  features: np.ndarray,
                                  result_future: Future,
                                  game_state) -> None:
        """Submit inference request to worker.

        This method handles the actual submission to the inference worker and
        populates the result future with the outcome.

        Args:
            features: Feature tensor (C, H, W)
            result_future: Future to populate with result
            game_state: Original game state (for fallback)
        """
        try:
            # Call inference worker synchronously
            # Note: The inference worker handles batching internally
            policy_batch, value_batch = self.inference_worker.batch_inference([features])

            # Extract single result from batch
            policy = policy_batch[0]
            value = value_batch[0] if value_batch.ndim > 0 else float(value_batch)

            # Set result on future
            result_future.set_result((policy, value))

            with self._metrics_lock:
                self._successful_requests += 1

        except TimeoutError as e:
            # Inference timeout
            self.logger.warning(f"Inference timeout: {e}")
            result_future.set_exception(
                InferenceError(f"Inference timeout: {e}")
            )
            with self._metrics_lock:
                self._timeout_requests += 1
                self._failed_requests += 1

        except Exception as e:
            # Check if we should try CPU fallback
            if self.enable_cpu_fallback and self._should_use_cpu_fallback(e):
                self.logger.info(f"GPU inference failed, using CPU fallback: {e}")
                try:
                    result = self._cpu_fallback_inference(features, game_state)
                    result_future.set_result(result)
                    with self._metrics_lock:
                        self._cpu_fallback_requests += 1
                        self._successful_requests += 1
                    return
                except Exception as fallback_error:
                    self.logger.error(f"CPU fallback also failed: {fallback_error}")
                    result_future.set_exception(
                        InferenceError(f"Both GPU and CPU inference failed: {fallback_error}")
                    )
                    with self._metrics_lock:
                        self._failed_requests += 1
                    return

            # No fallback or fallback not applicable
            self.logger.error(f"Inference failed: {e}")
            result_future.set_exception(
                InferenceError(f"Inference failed: {e}")
            )
            with self._metrics_lock:
                self._failed_requests += 1

    def _should_use_cpu_fallback(self, error: Exception) -> bool:
        """Determine if error should trigger CPU fallback.

        Args:
            error: Exception from GPU inference

        Returns:
            True if CPU fallback should be used
        """
        # Check for OOM errors
        error_str = str(error).lower()
        if 'out of memory' in error_str or 'oom' in error_str:
            return True

        # Check for CUDA errors
        if 'cuda' in error_str or 'gpu' in error_str:
            return True

        # Check if worker has flagged fallback
        if hasattr(self.inference_worker, '_fallback_triggered'):
            if self.inference_worker._fallback_triggered:
                return True

        return False

    def _cpu_fallback_inference(self,
                               features: np.ndarray,
                               game_state) -> Tuple[np.ndarray, float]:
        """Perform inference using CPU fallback.

        Args:
            features: Feature tensor (C, H, W)
            game_state: Original game state

        Returns:
            (policy, value) tuple

        Raises:
            InferenceError: If CPU inference fails
        """
        # Check if inference worker has CPU fallback capability
        if hasattr(self.inference_worker, '_cpu_fallback_worker'):
            cpu_worker = self.inference_worker._cpu_fallback_worker
            if cpu_worker is not None:
                policy_batch, value_batch = cpu_worker.batch_inference([features])
                policy = policy_batch[0]
                value = value_batch[0] if value_batch.ndim > 0 else float(value_batch)
                return (policy, value)

        # No CPU worker available, return uniform policy
        self.logger.warning("No CPU worker available, using uniform policy fallback")
        return self._uniform_policy_fallback(game_state)

    def _uniform_policy_fallback(self, game_state) -> Tuple[np.ndarray, float]:
        """Generate uniform policy as last-resort fallback.

        Args:
            game_state: Game state to evaluate

        Returns:
            (policy, value) tuple with uniform policy
        """
        # Get legal moves
        if hasattr(game_state, 'get_legal_moves'):
            legal_moves = game_state.get_legal_moves()
        else:
            legal_moves = []

        # Get action space size
        if hasattr(game_state, 'action_space_size'):
            action_space = game_state.action_space_size
        else:
            # Default to Gomoku 15x15
            action_space = 225

        # Create uniform policy over legal moves
        policy = np.zeros(action_space, dtype=np.float32)
        if len(legal_moves) > 0:
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                policy[move] = prob
        else:
            # No legal moves, uniform over all
            policy.fill(1.0 / action_space)

        # Return neutral value
        value = 0.0

        return (policy, value)

    def _cpu_batch_fallback(self, positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run batch inference on the CPU fallback worker when available."""
        cpu_worker = getattr(self.inference_worker, '_cpu_fallback_worker', None)
        if cpu_worker is None:
            raise RuntimeError("CPU fallback worker unavailable for batch inference")

        policies, values = cpu_worker.batch_inference(positions)

        with self._metrics_lock:
            self._cpu_fallback_requests += len(positions)
            self._successful_requests += len(positions)

        return policies, values

    def get_metrics(self) -> dict:
        """Get inference bridge metrics.

        Returns:
            Dictionary with metrics:
                - total_requests: Total inference requests
                - successful_requests: Successfully completed requests
                - failed_requests: Failed requests
                - timeout_requests: Requests that timed out
                - cpu_fallback_requests: Requests using CPU fallback
                - success_rate: Success rate (0.0 to 1.0)
        """
        with self._metrics_lock:
            success_rate = (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            )

            return {
                'total_requests': self._total_requests,
                'successful_requests': self._successful_requests,
                'failed_requests': self._failed_requests,
                'timeout_requests': self._timeout_requests,
                'cpu_fallback_requests': self._cpu_fallback_requests,
                'batch_requests': self._batch_requests,
                'batch_failures': self._batch_failures,
                'success_rate': success_rate,
            }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self._metrics_lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._timeout_requests = 0
            self._cpu_fallback_requests = 0
