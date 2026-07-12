"""
CPU Fallback Inference Implementation (T018)
============================================

CPU-only neural network inference worker for fallback scenarios.
Provides automatic fallback when GPU inference fails or is unavailable.

Key features:
- Single-threaded CPU inference for reliability
- Automatic fallback on GPU failures (CUDA OOM, device errors)
- Performance monitoring and degradation tracking
- Compatible interface with GPUInferenceWorker
"""

import torch
import numpy as np
import time
import threading
import logging
import psutil
import os
import math
from typing import List, Dict, Optional, Tuple, Any
from queue import Queue, Empty
from collections import deque
from dataclasses import dataclass

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import InferenceWorker, InferenceRequest, InferenceResult

# Import neural network model
from src.neural.model import AlphaZeroNet, create_model_for_game


@dataclass
class CPUInferenceMetrics:
    """CPU inference performance metrics."""
    total_inferences: int = 0
    total_inference_time: float = 0.0
    average_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    fallback_count: int = 0
    last_updated: float = 0.0


class CPUInferenceWorker(InferenceWorker):
    """CPU-only inference worker for fallback scenarios.

    Provides reliable CPU-based neural network inference when GPU
    inference fails or is unavailable. Designed for robustness
    over performance.
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 batch_size: int = 1,  # CPU typically processes smaller batches
                 timeout_ms: float = 10.0,  # More lenient timeout for CPU
                 use_mixed_precision: bool = False):  # CPU doesn't benefit from fp16
        """Initialize CPU inference worker.

        Args:
            model_path: Path to trained PyTorch model
            device: Force to 'cpu' for CPU inference
            batch_size: Smaller batch size for CPU (typically 1-4)
            timeout_ms: Batch timeout - more lenient for CPU
            use_mixed_precision: Disabled for CPU (no benefit)
        """
        self.model_path = model_path
        self.device = 'cpu'  # Force CPU device
        self.batch_size = min(batch_size, 4)  # Limit CPU batch size
        self.timeout_ms = timeout_ms
        self.use_mixed_precision = False  # No mixed precision on CPU

        # Performance tracking
        self.metrics = CPUInferenceMetrics()
        self.performance_history = deque(maxlen=100)

        # Threading and queues
        self.worker_thread = None
        self.stop_event = threading.Event()
        self._warmup_completed = False

        # Game-agnostic model info
        self._model_info = None

        # Logging
        self.logger = logging.getLogger(f'CPUInferenceWorker')
        self.logger.setLevel(logging.INFO)

        # Load model
        self._load_model()

    def _initialize_model_layers(self, input_shape: Tuple[int, int, int]) -> None:
        """Initialize all model layers with dummy forward pass."""
        if self.model is None:
            return

        dummy_input = torch.zeros(1, *input_shape, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)

    def _derive_input_shape(self) -> Tuple[int, int, int]:
        """Derive an appropriate input shape based on detected model info."""
        if not self._model_info:
            return (36, 15, 15)

        channels = self._model_info.get('input_channels', 36)
        num_actions = self._model_info.get('num_actions', 225)

        if num_actions in (225, 324):  # Gomoku/15x15 variants
            board_dim = 15
        elif num_actions in (361, 400):  # Go 19x19 or similar
            board_dim = 19
        elif num_actions in (82, 100):  # Smaller Go boards (9x9, 10x10)
            board_dim = int(round(math.sqrt(num_actions)))
        elif num_actions in (64, 4096, 20480):  # Chess board representations
            board_dim = 8
        else:
            approx_dim = int(round(math.sqrt(num_actions)))
            board_dim = approx_dim if approx_dim * approx_dim == num_actions else 15

        return (channels, board_dim, board_dim)

    def _load_model(self) -> None:
        """Load neural network model on CPU device with game-agnostic approach."""
        try:
            self.logger.info(f"Loading model from {self.model_path} on CPU")

            # Try to load the model directly first (full model save)
            try:
                self.model = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.model.to(self.device)
                self.model.eval()
                self._extract_model_info()
                self.logger.info("Model loaded successfully (full model)")
                return
            except Exception as e:
                self.logger.info(f"Full model loading failed, trying state_dict: {e}")

            # Fallback to state dict loading - create a flexible model
            loaded_data = torch.load(self.model_path, map_location='cpu', weights_only=False)

            # Handle both state dict and full model formats
            if hasattr(loaded_data, 'state_dict'):
                # Full model was saved, extract state dict
                state_dict = loaded_data.state_dict()
            elif isinstance(loaded_data, dict):
                # State dict was saved directly
                state_dict = loaded_data
            else:
                # Try to use as model directly and extract state dict
                state_dict = loaded_data.state_dict() if hasattr(loaded_data, 'state_dict') else loaded_data

            self._model_info = self._analyze_state_dict(state_dict)

            # Create model based on detected architecture
            self.model = self._create_flexible_model(self._model_info)

            # Handle PolicyHead lazy initialization
            expected_input_shape = self._derive_input_shape()

            if 'policy_head.fc.weight' in state_dict:
                # PolicyHead is already initialized in saved model - initialize ours too before loading
                self._initialize_model_layers(expected_input_shape)
                self.model.load_state_dict(state_dict)
            else:
                # PolicyHead not initialized in saved model - load first then initialize
                self.model.load_state_dict(state_dict)
                self._initialize_model_layers(expected_input_shape)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Model loaded successfully on CPU (channels: {self._model_info['input_channels']}, "
                           f"actions: {self._model_info['num_actions']})")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"CPU model loading failed: {e}")

    def _analyze_state_dict(self, state_dict: dict) -> dict:
        """Analyze state dict to extract model architecture information."""
        info = {
            'input_channels': 36,  # Default to Gomoku
            'num_actions': 225,    # Default to Gomoku
            'num_blocks': 20,      # Default blocks
            'hidden_channels': 256 # Default channels
        }

        # Extract input channels from initial conv layer
        if 'initial_conv.weight' in state_dict:
            weight_shape = state_dict['initial_conv.weight'].shape
            info['input_channels'] = weight_shape[1]  # [out_channels, in_channels, h, w]
            info['hidden_channels'] = weight_shape[0]

        # Extract number of actions from policy head
        if 'policy_head.fc.weight' in state_dict:
            policy_shape = state_dict['policy_head.fc.weight'].shape
            info['num_actions'] = policy_shape[0]  # [num_actions, hidden_size]

        # Count residual blocks
        max_block_idx = -1
        for key in state_dict.keys():
            if 'residual_blocks.' in key:
                parts = key.split('.')
                if len(parts) >= 2:
                    try:
                        block_idx = int(parts[1])
                        max_block_idx = max(max_block_idx, block_idx)
                    except ValueError:
                        pass

        if max_block_idx >= 0:
            info['num_blocks'] = max_block_idx + 1

        return info

    def _create_flexible_model(self, model_info: dict) -> torch.nn.Module:
        """Create model with flexible architecture based on detected parameters."""
        from src.neural.model import AlphaZeroNet

        return AlphaZeroNet(
            input_channels=model_info['input_channels'],
            num_actions=model_info['num_actions'],
            num_blocks=model_info['num_blocks'],
            hidden_channels=model_info['hidden_channels']
        )

    def _extract_model_info(self) -> None:
        """Extract model information from loaded model."""
        if self.model is None:
            return

        # Try to extract from model attributes
        info = {
            'input_channels': getattr(self.model, 'input_channels', 36),
            'num_actions': getattr(self.model, 'num_actions', 225),
            'num_blocks': getattr(self.model, 'num_blocks', 20),
            'hidden_channels': getattr(self.model, 'hidden_channels', 256)
        }

        # Try to infer from first layer if attributes not available
        if hasattr(self.model, 'initial_conv') and hasattr(self.model.initial_conv, 'weight'):
            weight_shape = self.model.initial_conv.weight.shape
            info['input_channels'] = weight_shape[1]
            info['hidden_channels'] = weight_shape[0]

        if hasattr(self.model, 'policy_head') and hasattr(self.model.policy_head, 'fc'):
            policy_shape = self.model.policy_head.fc.weight.shape
            info['num_actions'] = policy_shape[0]

        self._model_info = info

    def warmup(self, input_shape: Tuple[int, int, int]) -> None:
        """Warmup CPU inference with dummy calls.

        Args:
            input_shape: (channels, height, width) for input tensors
        """
        if self._warmup_completed:
            return

        self.logger.info("Warming up CPU inference")
        self.input_shape = input_shape

        try:
            # Single dummy inference call
            dummy_input = torch.randn(1, *input_shape, dtype=torch.float32)

            start_time = time.time()
            with torch.no_grad():
                policy_logits, values = self.model(dummy_input)
            warmup_time = time.time() - start_time

            self.logger.info(f"CPU warmup completed in {warmup_time*1000:.2f}ms")
            self._warmup_completed = True

        except Exception as e:
            self.logger.error(f"CPU warmup failed: {e}")
            raise RuntimeError(f"CPU warmup failed: {e}")

    def inference_loop(self,
                      input_queue: Queue,
                      output_queues: List[Queue]) -> None:
        """Main CPU inference loop for processing requests.

        Processes inference requests one by one (no batching for simplicity
        and reliability on CPU).

        Args:
            input_queue: Queue of inference requests
            output_queues: List of result queues, one per search thread
        """
        self.logger.info("Starting CPU inference loop")

        while not self.stop_event.is_set():
            try:
                # Get request with timeout
                try:
                    request = input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Process single request
                start_time = time.time()

                try:
                    # Create input tensor
                    input_tensor = torch.tensor(
                        request.features.reshape(1, *request.features.shape),
                        dtype=torch.float32,
                        device=self.device
                    )

                    # Run inference
                    with torch.no_grad():
                        policy_logits, values = self.model(input_tensor)

                    # Convert to numpy
                    policy = policy_logits.cpu().numpy().squeeze(0)
                    value = values.cpu().numpy().squeeze()

                    processing_time = (time.time() - start_time) * 1000

                    # Create result
                    result = InferenceResult(
                        node_id=request.leaf_node_id,
                        policy=policy,
                        value=float(value),
                        path=request.path,
                        processing_time_ms=processing_time
                    )

                    # Send result to appropriate output queue
                    if request.thread_id < len(output_queues):
                        output_queues[request.thread_id].put(result)

                    # Update metrics
                    self._update_metrics(processing_time)

                except Exception as e:
                    self.logger.error(f"CPU inference failed for request {request.leaf_node_id}: {e}")
                    # Put error result (Gomoku 15x15 = 225 actions)
                    action_count = self._model_info['num_actions'] if self._model_info else 225
                    error_result = InferenceResult(
                        node_id=request.leaf_node_id,
                        policy=np.zeros(action_count),  # Safe fallback aligned with model
                        value=0.0,
                        path=request.path,
                        processing_time_ms=0.0
                    )
                    if request.thread_id < len(output_queues):
                        output_queues[request.thread_id].put(error_result)

                input_queue.task_done()

            except Exception as e:
                self.logger.error(f"CPU inference loop error: {e}")
                continue

        self.logger.info("CPU inference loop stopped")

    def batch_inference(self,
                       positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process batch of positions through CPU neural network.

        CPU implementation processes positions sequentially for reliability.

        Args:
            positions: List of feature tensors, each (C, H, W)

        Returns:
            tuple: (policies, values)
                policies: Policy probabilities (batch_size, num_actions)
                values: Position values (batch_size,)
        """
        if not self._warmup_completed:
            raise RuntimeError("CPU inference worker not warmed up")

        batch_size = len(positions)
        start_time = time.time()

        try:
            # Process positions sequentially for CPU reliability
            policies = []
            values = []

            for position in positions:
                # Create input tensor
                input_tensor = torch.tensor(
                    position.reshape(1, *position.shape),
                    dtype=torch.float32,
                    device=self.device
                )

                # Run inference
                with torch.no_grad():
                    policy_logits, value = self.model(input_tensor)

                # Apply softmax to convert logits to normalized probabilities
                policy_probs = torch.softmax(policy_logits, dim=1)

                # Convert and collect results
                policies.append(policy_probs.cpu().numpy().squeeze(0))
                values.append(value.cpu().numpy().squeeze())

            # Stack results
            policies_np = np.stack(policies)
            values_np = np.array(values)

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time / batch_size)  # Per-position time

            return policies_np, values_np

        except Exception as e:
            self.logger.error(f"CPU batch inference failed: {e}")
            # Return safe fallback results with dynamic action space
            num_actions = self._model_info['num_actions'] if self._model_info else 225
            policies_np = np.zeros((batch_size, num_actions))
            values_np = np.zeros(batch_size)
            return policies_np, values_np

    def _update_metrics(self, processing_time_ms: float) -> None:
        """Update CPU inference performance metrics.

        Args:
            processing_time_ms: Processing time for current inference
        """
        self.metrics.total_inferences += 1
        self.metrics.total_inference_time += processing_time_ms / 1000
        self.metrics.average_latency_ms = (
            self.metrics.total_inference_time * 1000 / self.metrics.total_inferences
        )
        self.metrics.last_updated = time.time()

        # Update memory usage
        try:
            process = psutil.Process(os.getpid())
            self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        # Track performance history
        self.performance_history.append({
            'latency_ms': processing_time_ms,
            'timestamp': time.time(),
            'memory_mb': self.metrics.memory_usage_mb
        })

    def get_metrics(self) -> Dict[str, float]:
        """Get CPU inference performance metrics.

        Returns:
            dict: Metrics including latency, throughput, and memory usage
        """
        current_time = time.time()

        # Calculate recent throughput (last 10 seconds)
        recent_inferences = [
            entry for entry in self.performance_history
            if current_time - entry['timestamp'] <= 10.0
        ]

        recent_throughput = len(recent_inferences) / 10.0 if recent_inferences else 0.0

        return {
            'device': 'cpu',
            'inference_type': 'cpu_fallback',
            'total_inferences': self.metrics.total_inferences,
            'average_latency_ms': self.metrics.average_latency_ms,
            'recent_throughput_per_sec': recent_throughput,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'fallback_active': True,
            'fallback_count': self.metrics.fallback_count,
            'last_updated': self.metrics.last_updated
        }

    def start_worker(self,
                    input_queue: Queue,
                    output_queues: List[Queue]) -> None:
        """Start CPU inference worker thread.

        Args:
            input_queue: Queue for incoming inference requests
            output_queues: Output queues for results per search thread
        """
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.logger.warning("CPU inference worker already running")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self.inference_loop,
            args=(input_queue, output_queues),
            name="CPUInferenceWorker"
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()

        self.logger.info("CPU inference worker started")

    def stop_worker(self) -> None:
        """Stop CPU inference worker thread."""
        if self.worker_thread is None:
            return

        self.logger.info("Stopping CPU inference worker")
        self.stop_event.set()

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                self.logger.warning("CPU inference worker did not stop gracefully")

        self.worker_thread = None
        self.logger.info("CPU inference worker stopped")

    @property
    def running(self) -> bool:
        """Running property for SearchCoordinator compatibility."""
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def is_running(self) -> bool:
        """Check if worker thread is running."""
        return self.running

    def start(self) -> None:
        """Start the inference worker with default queues for SearchCoordinator compatibility."""
        if self.running:
            return

        # Create default queues for SearchCoordinator interface
        from queue import Queue
        input_queue = Queue(maxsize=1000)
        output_queues = [Queue() for _ in range(8)]  # Default to 8 output queues

        # Store queues for internal use
        self._input_queue = input_queue
        self._output_queues = output_queues

        # Start the worker
        self.start_worker(input_queue, output_queues)

    def stop(self) -> None:
        """Stop the inference worker for SearchCoordinator compatibility."""
        self.stop_worker()


class CPUFallbackInference:
    """CPU fallback for GPU inference failures.

    Implements the contract API for single-position CPU inference
    as a fallback mechanism when GPU inference fails.
    """

    def __init__(self, model_path: str):
        """Initialize CPU fallback inference backend.

        Args:
            model_path: Path to PyTorch model
        """
        self.model_path = model_path
        self.device = 'cpu'
        self.model = None
        self.logger = logging.getLogger('CPUFallbackInference')

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load neural network model for CPU inference."""
        try:
            self.logger.info(f"Loading CPU fallback model from {self.model_path}")

            # Try to load the model directly first (full model save)
            try:
                self.model = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.model.to(self.device)
                self.model.eval()
                self.logger.info("CPU fallback model loaded (full model)")
                return
            except Exception as e:
                self.logger.info(f"Full model loading failed, trying state_dict: {e}")

            # Fallback to state dict loading
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)

            # Infer game type from model structure
            first_conv_weight = None
            for key, tensor in state_dict.items():
                if 'conv' in key.lower() and 'weight' in key:
                    first_conv_weight = tensor
                    break

            if first_conv_weight is not None:
                input_channels = first_conv_weight.shape[1]
                if input_channels == 7:
                    game_type = 'gomoku'
                elif input_channels == 12:
                    game_type = 'chess'
                elif input_channels == 17:
                    game_type = 'go'
                else:
                    game_type = 'gomoku'
            else:
                game_type = 'gomoku'

            self.model = create_model_for_game(game_type)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"CPU fallback model loaded (game: {game_type})")

        except Exception as e:
            self.logger.error(f"Failed to load CPU fallback model: {e}")
            raise RuntimeError(f"CPU fallback model loading failed: {e}")

    def inference(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single position inference on CPU.

        Args:
            features: Position features (C, H, W)

        Returns:
            tuple: (policy, value) from neural network
        """
        if self.model is None:
            raise RuntimeError("CPU fallback model not loaded")

        try:
            # Create input tensor
            input_tensor = torch.tensor(
                features.reshape(1, *features.shape),
                dtype=torch.float32,
                device=self.device
            )

            # Run inference
            with torch.no_grad():
                policy_logits, value = self.model(input_tensor)

            # Convert to numpy
            policy = policy_logits.cpu().numpy().squeeze(0)
            value = float(value.cpu().numpy().squeeze())

            return policy, value

        except Exception as e:
            self.logger.error(f"CPU fallback inference failed: {e}")
            # Return safe fallback (Gomoku 15x15 = 225 actions)
            policy = np.zeros(225)  # Safe default
            value = 0.0
            return policy, value


def create_cpu_fallback_worker(model_path: str, **kwargs) -> CPUInferenceWorker:
    """Factory function to create CPU fallback inference worker.

    Args:
        model_path: Path to trained PyTorch model
        **kwargs: Additional worker configuration

    Returns:
        CPUInferenceWorker: Configured CPU inference worker
    """
    return CPUInferenceWorker(model_path=model_path, **kwargs)


def detect_gpu_failure() -> bool:
    """Detect if GPU inference is likely to fail.

    Returns:
        bool: True if GPU inference should fallback to CPU
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        return True

    # Check GPU memory availability
    try:
        if torch.cuda.device_count() > 0:
            # Try to allocate a small tensor to test GPU
            test_tensor = torch.zeros(1, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            return False
    except Exception:
        return True

    return True


def should_fallback_to_cpu(error: Exception) -> bool:
    """Determine if an error should trigger CPU fallback.

    Args:
        error: Exception from GPU inference

    Returns:
        bool: True if should fallback to CPU
    """
    error_str = str(error).lower()

    # CUDA out of memory
    if 'out of memory' in error_str:
        return True

    # CUDA device errors
    if any(phrase in error_str for phrase in [
        'cuda', 'cudnn', 'cublas', 'device-side assert',
        'illegal memory access', 'unspecified launch failure'
    ]):
        return True

    # Generic GPU-related errors
    if any(phrase in error_str for phrase in [
        'gpu', 'tensor not on cuda', 'expected cuda'
    ]):
        return True

    return False
