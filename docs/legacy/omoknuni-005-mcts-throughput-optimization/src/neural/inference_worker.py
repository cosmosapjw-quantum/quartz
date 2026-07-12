"""
GPU Inference Worker Implementation
==================================

Threaded GPU inference worker for batched neural network evaluation.
Optimized for RTX 3060 Ti with queue-based communication and dynamic batching.

The worker runs in a dedicated thread, consuming inference requests from a shared
input queue and distributing results to thread-specific output queues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from queue import Queue, Empty, Full
from collections import deque
import logging
import psutil
import os
from dataclasses import dataclass

# GPU monitoring (optional import)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import (
    InferenceWorker,
    InferenceRequest,
    InferenceResult
)

# Import neural network model
from src.neural.model import AlphaZeroNet, create_model_for_game

# Import CPU fallback functionality (T018)
from src.neural.cpu_inference import CPUInferenceWorker, should_fallback_to_cpu, detect_gpu_failure


class GPUInferenceWorker(InferenceWorker):
    """GPU-based inference worker with batched processing.

    Runs neural network inference on GPU with dynamic batching for optimal
    throughput and GPU utilization.

    Args:
        model_path: Path to trained PyTorch model
        device: Device for inference ('cuda:0' or 'cpu')
        batch_size: Maximum batch size for GPU inference
        timeout_ms: Batch timeout in milliseconds
        use_mixed_precision: Enable fp16 inference
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda:0',
                 batch_size: int = 64,
                 timeout_ms: float = 3.0,
                 use_mixed_precision: bool = True):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        self.use_mixed_precision = use_mixed_precision

        # Dynamic micro-batching parameters
        self.min_batch_size = max(1, min(32, batch_size))  # Target ≥32 for efficiency
        self.max_timeout_ms = min(3.0, timeout_ms) / 1000.0  # Target ≤3ms for responsiveness
        self.target_gpu_utilization = 0.80  # Target >80% GPU utilization

        # Adaptive batching state
        self._performance_history = deque(maxlen=100)  # Track recent performance
        self._current_optimal_batch = self.min_batch_size
        self._gpu_handle = None

        # Mixed precision state
        self._mixed_precision_enabled = False
        self._mixed_precision_fallback_count = 0
        self._baseline_memory_usage = None

        # Pinned memory buffers for optimized H2D/D2H transfers
        self._pinned_input_buffer = None
        self._pinned_output_buffers = {}
        self._current_buffer_capacity = 0
        self._use_pinned_memory = str(self.device).startswith('cuda') and torch.cuda.is_available()

        # CPU fallback mechanism (T018)
        self._cpu_fallback_worker = None
        self._fallback_enabled = True
        self._fallback_triggered = False
        self._fallback_failure_count = 0
        self._last_gpu_attempt = 0.0

        # OOM recovery mechanisms (T050)
        self._oom_recovery_enabled = True
        self._original_batch_size = batch_size
        self._min_batch_size = max(1, batch_size // 16)  # Minimum batch size (1/16 of original)
        self._oom_count = 0
        self._consecutive_oom_count = 0
        self._last_oom_time = 0.0
        self._batch_size_reduction_factor = 0.5  # Reduce batch size by half on OOM
        self._oom_recovery_cooldown = 60.0  # 60 seconds before attempting size increase
        self._last_successful_batch_size = batch_size
        self._oom_memory_threshold = 0.9  # Consider memory usage >90% as high risk

        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._is_running = False

        # Model and computation
        self.model = None
        self.input_shape = None

        # Performance tracking
        self._metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'total_inference_time': 0.0,
            'batch_sizes': deque(maxlen=1000),  # Recent batch sizes
            'inference_times': deque(maxlen=1000),  # Recent inference times
            'nn_evaluations': 0,  # Total NN forward passes (for cache hit rate analysis)
            'states_evaluated': 0,  # Total states processed (for cache hit rate analysis)
        }
        self._metrics_lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(f'InferenceWorker[{device}]')

        # Initialize GPU monitoring
        self._init_gpu_monitoring()

        # Initialize model
        self._load_model()

        # Initialize CPU fallback worker if needed (T018)
        self._init_cpu_fallback()

    def _load_model(self) -> None:
        """Load and initialize the neural network model."""
        try:
            # Initialize game type, action space, and board size (will be set after model load)
            self.game_type = None
            self.num_actions = None
            self.board_size = None

            if self.model_path is None:
                self.logger.warning("No model_path provided. Initializing default AlphaZeroNet weights for benchmark use.")
                game_type = 'gomoku'
                input_shape = (36, 15, 15)
                self.model = create_model_for_game(game_type)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.game_type = game_type
                self.num_actions = 225  # Gomoku 15×15
                self.board_size = 15
            else:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model not found: {self.model_path}")

                load_device = self.device
                if str(self.device).startswith('cuda') and not torch.cuda.is_available():
                    load_device = 'cpu'
                    self.logger.warning(f"CUDA device {self.device} requested but unavailable, loading on CPU")

                model_data = torch.load(self.model_path, map_location=load_device, weights_only=False)

                if hasattr(model_data, 'state_dict'):
                    self.model = model_data.to(self.device)
                    self.model.eval()
                    # Detect game type and board size from model attributes
                    self.num_actions = getattr(self.model, 'num_actions', 225)
                    self.board_size = None

                    if self.num_actions == 225:
                        self.game_type = 'gomoku'
                        self.board_size = 15
                    elif self.num_actions == 4096:
                        self.game_type = 'chess'
                        self.board_size = 8
                    elif self.num_actions in [82, 170, 362]:
                        # Go with pass move: 82 (9×9+1), 170 (13×13+1), 362 (19×19+1)
                        self.game_type = 'go'
                        if self.num_actions == 82:
                            self.board_size = 9
                        elif self.num_actions == 170:
                            self.board_size = 13
                        else:
                            self.board_size = 19
                    else:
                        self.game_type = 'unknown'
                        self.board_size = None

                    board_info = f" (board {self.board_size}×{self.board_size})" if self.board_size else ""
                    self.logger.info(f"Model loaded successfully (full model, game: {self.game_type}{board_info}, actions: {self.num_actions})")
                elif isinstance(model_data, dict):
                    first_conv_weight = next((tensor for key, tensor in model_data.items()
                                              if 'conv' in key.lower() and 'weight' in key), None)
                    if first_conv_weight is not None:
                        input_channels = first_conv_weight.shape[1]
                        if input_channels == 36:
                            game_type = 'gomoku'
                            input_shape = (36, 15, 15)
                            self.board_size = 15
                        elif input_channels == 30:
                            game_type = 'chess'
                            input_shape = (30, 8, 8)
                            self.board_size = 8
                        elif input_channels == 25:
                            # Go: detect board size from spatial dimensions or filename
                            game_type = 'go'
                            spatial_dim = first_conv_weight.shape[2] if len(first_conv_weight.shape) > 2 else 19

                            # Try to infer from filename first
                            filename_lower = str(self.model_path).lower()
                            if 'go9' in filename_lower or '9x9' in filename_lower:
                                spatial_dim = 9
                            elif 'go13' in filename_lower or '13x13' in filename_lower:
                                spatial_dim = 13

                            # Use detected spatial dimension
                            if spatial_dim == 9:
                                self.board_size = 9
                                input_shape = (25, 9, 9)
                            elif spatial_dim == 13:
                                self.board_size = 13
                                input_shape = (25, 13, 13)
                            else:
                                self.board_size = 19
                                input_shape = (25, 19, 19)
                        else:
                            game_type = 'gomoku'
                            input_shape = (36, 15, 15)
                            self.board_size = 15
                    else:
                        game_type = 'gomoku'
                        input_shape = (36, 15, 15)
                        self.board_size = 15

                    self.model = create_model_for_game(game_type)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    self._initialize_model_layers(input_shape)
                    self.model.load_state_dict(model_data)
                    self.game_type = game_type

                    # Calculate num_actions from game type and board size
                    if game_type == 'gomoku':
                        self.num_actions = 225  # 15×15
                    elif game_type == 'chess':
                        self.num_actions = 4096
                    elif game_type == 'go':
                        # Go includes pass move: board_size² + 1
                        self.num_actions = self.board_size * self.board_size + 1
                    else:
                        self.num_actions = 225  # Default

                    board_info = f" (board {self.board_size}×{self.board_size})" if self.board_size else ""
                    self.logger.info(f"Model loaded successfully (state_dict, game: {game_type}{board_info}, actions: {self.num_actions})")
                else:
                    self.model = model_data.to(self.device)
                    self.model.eval()
                    # Detect game type and board size from model attributes
                    self.num_actions = getattr(self.model, 'num_actions', 225)
                    self.board_size = None

                    if self.num_actions == 225:
                        self.game_type = 'gomoku'
                        self.board_size = 15
                    elif self.num_actions == 4096:
                        self.game_type = 'chess'
                        self.board_size = 8
                    elif self.num_actions in [82, 170, 362]:
                        # Go with pass move: 82 (9×9+1), 170 (13×13+1), 362 (19×19+1)
                        self.game_type = 'go'
                        if self.num_actions == 82:
                            self.board_size = 9
                        elif self.num_actions == 170:
                            self.board_size = 13
                        else:
                            self.board_size = 19
                    else:
                        self.game_type = 'unknown'
                        self.board_size = None

                    board_info = f" (board {self.board_size}×{self.board_size})" if self.board_size else ""
                    self.logger.info(f"Model loaded successfully (direct model, game: {self.game_type}{board_info}, actions: {self.num_actions})")

            # Ensure num_actions and board_size are set (fallback to 225 for Gomoku)
            if self.num_actions is None:
                self.num_actions = 225
                self.game_type = 'gomoku'
                self.board_size = 15
                self.logger.warning(f"Could not detect num_actions, defaulting to {self.num_actions} (Gomoku 15×15)")
            if self.board_size is None:
                # Infer board_size from num_actions if possible
                if self.num_actions == 225:
                    self.board_size = 15
                elif self.num_actions == 4096:
                    self.board_size = 8
                elif self.num_actions in [82, 170, 362]:
                    if self.num_actions == 82:
                        self.board_size = 9
                    elif self.num_actions == 170:
                        self.board_size = 13
                    else:
                        self.board_size = 19
                else:
                    self.board_size = 15  # Default fallback

            self._setup_mixed_precision()
            self.logger.info(f"Model ready on {self.device} (game: {self.game_type}, actions: {self.num_actions})")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _initialize_model_layers(self, input_shape: Tuple[int, int, int]) -> None:
        """Initialize lazy layers (like PolicyHead.fc) with dummy forward pass.

        Args:
            input_shape: (channels, height, width) for dummy input
        """
        try:
            # Create dummy input tensor with batch size 1
            dummy_input = torch.zeros(1, *input_shape, device=self.device)

            # Do dummy forward pass to initialize lazy layers
            with torch.no_grad():
                self.model(dummy_input)

            self.logger.debug("Model layers initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model layers: {e}")
            # Continue anyway - the loading might still work

    def _init_cpu_fallback(self) -> None:
        """Initialize CPU fallback worker for GPU failure scenarios (T018)."""
        if not self._fallback_enabled or not str(self.device).startswith('cuda'):
            return

        try:
            # Check if GPU fallback is likely needed
            if detect_gpu_failure():
                self.logger.warning("GPU failure detected, initializing CPU fallback immediately")
                self._enable_cpu_fallback()
            else:
                self.logger.info("GPU available, CPU fallback on standby")

        except Exception as e:
            self.logger.warning(f"Could not initialize CPU fallback: {e}")

    def _enable_cpu_fallback(self) -> None:
        """Enable CPU fallback worker (T018)."""
        if self._cpu_fallback_worker is not None:
            return  # Already enabled

        try:
            self.logger.info("Enabling CPU fallback inference worker")
            self._cpu_fallback_worker = CPUInferenceWorker(
                model_path=self.model_path,
                device='cpu',
                batch_size=min(4, self.batch_size),
                timeout_ms=max(10.0, self.timeout_ms * 1000),
                use_mixed_precision=False
            )

            if self.input_shape is not None:
                self._cpu_fallback_worker.warmup(self.input_shape)

            self._fallback_triggered = True
            self.logger.info("CPU fallback worker enabled successfully")

        except Exception as e:
            self.logger.error(f"Failed to enable CPU fallback: {e}")
            self._cpu_fallback_worker = None

    def _should_attempt_gpu_retry(self) -> bool:
        """Check if we should retry GPU inference after fallback (T018)."""
        if not self._fallback_triggered:
            return True

        # Allow GPU retry every 30 seconds
        current_time = time.time()
        if current_time - self._last_gpu_attempt > 30.0:
            return True

        return False

    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring for utilization tracking."""
        if not NVML_AVAILABLE or not str(self.device).startswith('cuda'):
            self.logger.info("GPU monitoring not available (no pynvml or CPU device)")
            return

        try:
            pynvml.nvmlInit()
            device_str = str(self.device)
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.logger.info(f"GPU monitoring initialized for device {device_id}")
        except Exception as e:
            self.logger.warning(f"Could not initialize GPU monitoring: {e}")
            self._gpu_handle = None

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization as percentage (0.0-1.0), or 0.0 if unavailable
        """
        if not self._gpu_handle:
            return 0.0

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def _setup_mixed_precision(self) -> None:
        """Setup and validate mixed precision inference."""
        if not self.use_mixed_precision:
            self.logger.info("Mixed precision disabled by configuration")
            return

        if not str(self.device).startswith('cuda'):
            self.logger.info("Mixed precision not available on CPU device, using fp32")
            self.use_mixed_precision = False
            return

        try:
            # Check CUDA and device capability
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, disabling mixed precision")
                self.use_mixed_precision = False
                return

            device_str = str(self.device)
            device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
            device_capability = torch.cuda.get_device_capability(device_idx)

            # Tensor cores available on compute capability 7.0+ (V100, RTX 20/30/40 series)
            if device_capability[0] < 7:
                self.logger.warning(
                    f"Device compute capability {device_capability[0]}.{device_capability[1]} "
                    f"may not benefit from mixed precision. Tensor cores require 7.0+"
                )

            # Apply mixed precision optimizations to model
            from src.neural.model import enable_mixed_precision
            self.model = enable_mixed_precision(self.model)

            # Record baseline memory usage for efficiency monitoring
            torch.cuda.empty_cache()
            self._baseline_memory_usage = torch.cuda.memory_allocated(device_idx)

            self._mixed_precision_enabled = True
            self.logger.info(
                f"Mixed precision enabled on {self.device} "
                f"(compute capability: {device_capability[0]}.{device_capability[1]})"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup mixed precision: {e}")
            self.use_mixed_precision = False
            self._mixed_precision_enabled = False

    def _setup_pinned_memory_buffers(self, batch_size: int, input_shape: Tuple[int, int, int]) -> None:
        """Setup pinned memory buffers for optimized GPU transfers.

        Args:
            batch_size: Maximum batch size to allocate for
            input_shape: (channels, height, width) shape for input tensors
        """
        if not self._use_pinned_memory:
            return

        # Add safety margin for dynamic batching
        buffer_capacity = int(batch_size * 1.5)

        if buffer_capacity <= self._current_buffer_capacity:
            return  # Current buffers are sufficient

        try:
            # Free existing buffers
            self._cleanup_pinned_buffers()

            # Allocate pinned input buffer
            input_buffer_shape = (buffer_capacity, *input_shape)
            self._pinned_input_buffer = torch.empty(
                input_buffer_shape,
                dtype=torch.float32,
                pin_memory=True
            )

            # Pre-allocate common output buffer sizes
            # Policy head output: (batch_size, num_actions) - game-specific (Task #4)
            # Gomoku: 225, Chess: 4096, Go: 361
            num_actions = getattr(self, 'num_actions', 225)  # Use detected num_actions or default
            policy_buffer_shape = (buffer_capacity, num_actions)
            self._pinned_output_buffers['policy'] = torch.empty(
                policy_buffer_shape,
                dtype=torch.float32,
                pin_memory=True
            )

            # Value head output: (batch_size, 1)
            value_buffer_shape = (buffer_capacity, 1)
            self._pinned_output_buffers['value'] = torch.empty(
                value_buffer_shape,
                dtype=torch.float32,
                pin_memory=True
            )

            self._current_buffer_capacity = buffer_capacity
            game_info = getattr(self, 'game_type', 'unknown')
            self.logger.info(
                f"Pinned memory buffers allocated for batch capacity: {buffer_capacity} "
                f"(game: {game_info}, policy size: {num_actions})"
            )

        except Exception as e:
            self.logger.warning(f"Failed to allocate pinned memory buffers: {e}")
            self._use_pinned_memory = False
            self._cleanup_pinned_buffers()

    def _cleanup_pinned_buffers(self) -> None:
        """Cleanup pinned memory buffers."""
        try:
            if self._pinned_input_buffer is not None:
                del self._pinned_input_buffer
                self._pinned_input_buffer = None

            for key in list(self._pinned_output_buffers.keys()):
                del self._pinned_output_buffers[key]
            self._pinned_output_buffers.clear()

            self._current_buffer_capacity = 0

        except Exception as e:
            self.logger.warning(f"Error during pinned buffer cleanup: {e}")

    def _create_batch_tensor_optimized(self, positions: List[np.ndarray]) -> torch.Tensor:
        """Create batch tensor using pinned memory optimization if available.

        Args:
            positions: List of numpy arrays representing game positions

        Returns:
            Batch tensor ready for inference
        """
        batch_size = len(positions)

        if (self._use_pinned_memory and
            self._pinned_input_buffer is not None and
            batch_size <= self._current_buffer_capacity):

            # Use pinned memory buffer for optimized transfer
            try:
                # Copy data to pinned buffer (on host)
                batch_data = np.stack(positions)
                input_slice = self._pinned_input_buffer[:batch_size]
                input_slice.copy_(torch.from_numpy(batch_data))

                # Transfer to GPU using pinned memory (faster H2D transfer)
                batch_tensor = input_slice.to(self.device, non_blocking=True)
                return batch_tensor

            except Exception as e:
                self.logger.warning(f"Pinned memory transfer failed, falling back to standard: {e}")

        # Fallback to standard tensor creation
        return torch.tensor(
            np.stack(positions),
            dtype=torch.float32,
            device=self.device
        )

    def _transfer_outputs_optimized(self, policy_logits: torch.Tensor, values: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Transfer outputs from GPU using pinned memory optimization if available.

        Args:
            policy_logits: Policy output tensor from model
            values: Value output tensor from model

        Returns:
            Tuple of (policies_np, values_np) as numpy arrays
        """
        batch_size = policy_logits.size(0)

        if (self._use_pinned_memory and
            'policy' in self._pinned_output_buffers and
            'value' in self._pinned_output_buffers and
            batch_size <= self._current_buffer_capacity):

            try:
                # Use pinned buffers for optimized D2H transfer
                policy_buffer = self._pinned_output_buffers['policy'][:batch_size, :policy_logits.size(1)]
                value_buffer = self._pinned_output_buffers['value'][:batch_size]

                # Transfer to pinned memory (faster D2H transfer)
                policy_buffer.copy_(policy_logits, non_blocking=True)
                value_buffer.copy_(values, non_blocking=True)

                # Convert to numpy (no copy needed, already on host)
                policies_np = policy_buffer.numpy()
                values_np = value_buffer.numpy().squeeze(-1)  # Squeeze after numpy conversion

                return policies_np, values_np

            except Exception as e:
                self.logger.warning(f"Pinned memory D2H transfer failed, falling back to standard: {e}")

        # Fallback to standard transfer
        policies_np = policy_logits.cpu().numpy()
        values_np = values.cpu().numpy().squeeze(-1)

        return policies_np, values_np

    def _run_inference_with_precision(self, batch_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference with mixed precision and automatic fallback.

        Args:
            batch_tensor: Input tensor batch

        Returns:
            Tuple of (policy_logits, values)
        """
        if self._mixed_precision_enabled and str(self.device).startswith('cuda'):
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    policy_logits, values = self.model(batch_tensor)
                return policy_logits, values

            except RuntimeError as e:
                # Handle mixed precision failures (e.g., unsupported operations)
                if "autocast" in str(e).lower() or "half" in str(e).lower():
                    self._mixed_precision_fallback_count += 1
                    self.logger.warning(
                        f"Mixed precision failed, falling back to fp32: {e} "
                        f"(fallback count: {self._mixed_precision_fallback_count})"
                    )

                    # Disable mixed precision after too many failures
                    if self._mixed_precision_fallback_count >= 3:
                        self.logger.warning("Disabling mixed precision due to repeated failures")
                        self._mixed_precision_enabled = False
                        self.use_mixed_precision = False
                else:
                    raise  # Re-raise non-precision related errors

        # Fallback to fp32
        # Ensure tensor is on the same device as the model if there's a device mismatch
        try:
            model_device = next(self.model.parameters()).device
            if batch_tensor.device != model_device:
                batch_tensor = batch_tensor.to(model_device)
        except StopIteration:
            # Model has no parameters (e.g., in tests), skip device check
            pass

        policy_logits, values = self.model(batch_tensor)
        return policy_logits, values

    def _get_memory_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate memory efficiency metrics for mixed precision.

        Returns:
            Dictionary with memory efficiency statistics
        """
        metrics = {
            'mixed_precision_active': self._mixed_precision_enabled,
            'mixed_precision_fallback_count': self._mixed_precision_fallback_count
        }

        if str(self.device).startswith('cuda') and torch.cuda.is_available():
            device_str = str(self.device)
            device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
            current_memory = torch.cuda.memory_allocated(device_idx)
            max_memory = torch.cuda.max_memory_allocated(device_idx)

            metrics['current_memory_mb'] = current_memory / (1024**2)
            metrics['max_memory_mb'] = max_memory / (1024**2)

            if self._baseline_memory_usage is not None:
                memory_ratio = current_memory / self._baseline_memory_usage
                metrics['memory_efficiency_ratio'] = memory_ratio
                metrics['memory_reduction_achieved'] = memory_ratio < 0.7  # Target ~50% reduction

        # Add pinned memory optimization metrics
        metrics['pinned_memory_enabled'] = self._use_pinned_memory
        metrics['pinned_buffer_capacity'] = self._current_buffer_capacity
        if self._pinned_input_buffer is not None:
            # Calculate approximate memory usage of pinned buffers
            input_buffer_size = self._pinned_input_buffer.numel() * self._pinned_input_buffer.element_size()
            output_buffer_size = sum(
                buf.numel() * buf.element_size()
                for buf in self._pinned_output_buffers.values()
            )
            total_pinned_mb = (input_buffer_size + output_buffer_size) / (1024**2)
            metrics['pinned_memory_usage_mb'] = total_pinned_mb

        # Add OOM recovery metrics (T050)
        metrics['oom_recovery_enabled'] = self._oom_recovery_enabled
        metrics['oom_total_count'] = self._oom_count
        metrics['oom_consecutive_count'] = self._consecutive_oom_count
        metrics['original_batch_size'] = self._original_batch_size
        metrics['oom_min_batch_size'] = self._min_batch_size
        metrics['last_successful_batch_size'] = self._last_successful_batch_size
        metrics['batch_size_reduction_factor'] = self._batch_size_reduction_factor
        metrics['oom_memory_threshold'] = self._oom_memory_threshold

        # Add current memory usage fraction for monitoring
        memory_fraction = self._get_memory_usage_fraction()
        metrics['memory_usage_fraction'] = memory_fraction
        metrics['memory_usage_high_risk'] = memory_fraction > self._oom_memory_threshold

        return metrics

    def warmup(self, input_shape: Tuple[int, int, int]) -> None:
        """Warmup GPU with dummy inference calls.

        Critical for consistent latency measurements. Must be called
        before starting inference loop.

        Args:
            input_shape: (channels, height, width) for input tensors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.input_shape = input_shape
        self.logger.info(f"Warming up GPU with input shape {input_shape}")

        # Setup pinned memory buffers for the maximum batch size we'll use
        self._setup_pinned_memory_buffers(self.batch_size, input_shape)

        # Warm up with different batch sizes
        warmup_batches = [1, 8, 16, 32, min(64, self.batch_size)]

        with torch.no_grad():
            for batch_size in warmup_batches:
                dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

                # Warmup runs
                for _ in range(3):
                    _ = self._run_inference_with_precision(dummy_input)

                # Synchronize GPU
                if str(self.device).startswith('cuda'):
                    torch.cuda.synchronize()

        self.logger.info("GPU warmup completed")

    def start_worker(self, input_queue: Queue, output_queues: List[Queue]) -> None:
        """Start the inference worker thread.

        Args:
            input_queue: Queue of inference requests
            output_queues: List of result queues, one per search thread
        """
        if self._is_running:
            raise RuntimeError("Worker already running")

        self.logger.info("Starting inference worker thread")

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self.inference_loop,
            args=(input_queue, output_queues),
            name="InferenceWorker",
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True

    def stop_worker(self, timeout: float = 5.0) -> None:
        """Stop the inference worker thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._is_running:
            return

        self.logger.info("Stopping inference worker thread")

        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

            if self._worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
            else:
                self.logger.info("Worker thread stopped successfully")

        self._is_running = False
        self._worker_thread = None

        # Cleanup pinned memory buffers
        self._cleanup_pinned_buffers()

        # Cleanup CPU fallback worker (T018)
        if self._cpu_fallback_worker is not None:
            try:
                self._cpu_fallback_worker.stop_worker()
                self._cpu_fallback_worker = None
                self.logger.info("CPU fallback worker stopped and cleaned up")
            except Exception as e:
                self.logger.warning(f"Error stopping CPU fallback worker: {e}")

    def inference_loop(self,
                      input_queue: Queue,
                      output_queues: List[Queue]) -> None:
        """Main inference loop for batched processing.

        Runs in dedicated thread, consuming requests from input_queue
        and distributing results to thread-specific output_queues.

        Args:
            input_queue: Queue of inference requests
            output_queues: List of result queues, one per search thread
        """
        self.logger.info("Inference loop started")

        try:
            while not self._stop_event.is_set():
                # Collect batch of requests
                batch_requests = self._collect_batch(input_queue)

                if not batch_requests:
                    continue

                # Process batch
                start_time = time.time()
                batch_results = self._process_batch(batch_requests)
                inference_time = time.time() - start_time

                # Distribute results to output queues
                self._distribute_results(batch_results, output_queues)

                # Update metrics
                self._update_metrics(len(batch_requests), inference_time)

        except Exception as e:
            self.logger.error(f"Error in inference loop: {e}")
            raise
        finally:
            self.logger.info("Inference loop ended")

    def _collect_batch(self, input_queue: Queue) -> List[InferenceRequest]:
        """Collect a batch of requests with dynamic micro-batching.

        Uses sophisticated count-based (≥32) OR timeout-based (≤3ms) batching
        to optimize for target >80% GPU utilization.

        Args:
            input_queue: Queue of inference requests

        Returns:
            List of inference requests (may be empty)
        """
        batch = []
        start_time = time.time()

        # Determine optimal batch size based on recent performance
        target_batch_size = self._get_optimal_batch_size()

        # Try to get first request with micro-timeout
        try:
            first_request = input_queue.get(timeout=self.max_timeout_ms)
            batch.append(first_request)
        except Empty:
            return batch

        # Phase 1: Quick collection to target batch size
        # Collect aggressively for first few requests
        quick_timeout = min(0.001, self.max_timeout_ms / 4)  # 1ms or quarter of max

        while len(batch) < target_batch_size:
            elapsed = time.time() - start_time

            # If we're approaching max timeout, be more aggressive
            if elapsed > self.max_timeout_ms * 0.8:
                break

            try:
                request = input_queue.get(timeout=quick_timeout)
                batch.append(request)
            except Empty:
                break

        # Phase 2: Smart timeout-based collection
        # If we haven't reached min efficient batch size, wait a bit longer
        if len(batch) < self.min_batch_size:
            remaining_timeout = max(0, self.max_timeout_ms - (time.time() - start_time))

            while len(batch) < self.min_batch_size and remaining_timeout > 0:
                try:
                    request = input_queue.get(timeout=remaining_timeout)
                    batch.append(request)

                    # Update remaining timeout
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, self.max_timeout_ms - elapsed)

                except Empty:
                    break

        # Phase 3: Opportunistic collection
        # If we have time left and haven't hit max batch size, collect more
        remaining_time = self.max_timeout_ms - (time.time() - start_time)
        if remaining_time > 0 and len(batch) < self.batch_size:
            # Use very short timeout for opportunistic collection
            opportunistic_timeout = min(0.0005, remaining_time)  # 0.5ms max

            while len(batch) < self.batch_size and remaining_time > 0:
                try:
                    request = input_queue.get(timeout=opportunistic_timeout)
                    batch.append(request)
                    remaining_time = self.max_timeout_ms - (time.time() - start_time)
                except Empty:
                    break

        return batch

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on recent performance.

        Returns:
            Optimal batch size for current conditions
        """
        # If no performance history, start with minimum efficient size
        if not self._performance_history:
            return self.min_batch_size

        # Get recent GPU utilization if available
        gpu_util = self._get_gpu_utilization()

        # Analyze recent performance
        recent_perf = list(self._performance_history)[-10:]  # Last 10 batches
        if not recent_perf:
            return self._current_optimal_batch

        avg_throughput = sum(p['throughput'] for p in recent_perf) / len(recent_perf)
        avg_batch_size = sum(p['batch_size'] for p in recent_perf) / len(recent_perf)

        # Adaptive logic based on GPU utilization and throughput
        if gpu_util > 0:  # GPU monitoring available
            if gpu_util < self.target_gpu_utilization * 0.9:  # Below 72%
                # Increase batch size to improve GPU utilization
                self._current_optimal_batch = min(
                    self.batch_size,
                    int(self._current_optimal_batch * 1.1)
                )
            elif gpu_util > self.target_gpu_utilization * 1.1:  # Above 88%
                # Decrease batch size to avoid overload
                self._current_optimal_batch = max(
                    self.min_batch_size,
                    int(self._current_optimal_batch * 0.9)
                )
        else:
            # No GPU monitoring - use throughput-based adaptation
            if len(recent_perf) >= 5:
                # Check if throughput is improving with larger batches
                recent_5 = recent_perf[-5:]
                throughput_trend = (recent_5[-1]['throughput'] - recent_5[0]['throughput']) / 5

                if throughput_trend > 0 and avg_batch_size < self.batch_size * 0.8:
                    # Throughput improving, try larger batches
                    self._current_optimal_batch = min(
                        self.batch_size,
                        int(self._current_optimal_batch * 1.05)
                    )
                elif throughput_trend < 0 and avg_batch_size > self.min_batch_size * 1.2:
                    # Throughput declining, try smaller batches
                    self._current_optimal_batch = max(
                        self.min_batch_size,
                        int(self._current_optimal_batch * 0.95)
                    )

        return self._current_optimal_batch

    def _process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process a batch of inference requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        if not requests:
            return []

        # Extract features from requests
        positions = [req.features for req in requests]

        # Run batch inference
        start_time = time.time()
        policies, values = self.batch_inference(positions)
        processing_time_ms = (time.time() - start_time) * 1000

        # Create results
        results = []
        for i, request in enumerate(requests):
            result = InferenceResult(
                node_id=request.leaf_node_id,
                policy=policies[i],
                value=values[i].item(),
                path=request.path,
                processing_time_ms=processing_time_ms / len(requests)  # Per-sample time
            )
            # Preserve originating thread information for distribution
            if hasattr(request, 'thread_id'):
                result.thread_id = request.thread_id
            results.append(result)

        return results

    def _distribute_results(self,
                          results: List[InferenceResult],
                          output_queues: List[Queue]) -> None:
        """Distribute results to appropriate output queues.

        Args:
            results: List of inference results
            output_queues: List of result queues, one per thread
        """
        for result in results:
            queue_idx = getattr(result, 'thread_id', None)
            if queue_idx is None or queue_idx >= len(output_queues):
                queue_idx = result.node_id % len(output_queues)

            try:
                output_queues[queue_idx].put(result, timeout=1.0)
            except Full:
                self.logger.warning(f"Output queue {queue_idx} full, dropping result")

    def batch_inference(self,
                       positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process batch of positions through neural network.

        Args:
            positions: List of feature tensors, each (C, H, W)

        Returns:
            tuple: (policies, values)
                policies: Policy probabilities (batch_size, num_actions)
                values: Position values (batch_size,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Record start time for metrics
        start_time = time.time()

        if positions:
            sample_shape = tuple(positions[0].shape)
            if self.input_shape is None:
                self.input_shape = sample_shape
                self._setup_pinned_memory_buffers(self.batch_size, sample_shape)

        # Try GPU inference first if available and not permanently failed (T018)
        if self._should_attempt_gpu_retry():
            try:
                self._last_gpu_attempt = time.time()

                # Create batch tensor using pinned memory optimization
                batch_tensor = self._create_batch_tensor_optimized(positions)

                # Run inference with enhanced mixed precision
                with torch.no_grad():
                    policy_logits, values = self._run_inference_with_precision(batch_tensor)

                    # Convert to probabilities
                    policies = torch.softmax(policy_logits, dim=1)
                    values = torch.tanh(values)

                # Transfer outputs using pinned memory optimization
                policies_np, values_np = self._transfer_outputs_optimized(policies, values)

                # GPU inference succeeded, reset OOM and fallback state (T050)
                self._reset_oom_recovery_state()
                if self._fallback_triggered:
                    self._fallback_failure_count = 0
                    self.logger.info("GPU inference recovered, continuing with GPU")

                # Attempt to gradually increase batch size if conditions are favorable
                self._attempt_batch_size_increase()

                # Update metrics (T015)
                inference_time = time.time() - start_time
                self._update_metrics(len(positions), inference_time)

                return policies_np, values_np

            except Exception as e:
                # Handle OOM errors specifically (T050)
                if self._is_oom_error(e) and self._oom_recovery_enabled:
                    self.logger.warning(f"CUDA OOM detected: {e}")

                    # Attempt OOM recovery
                    if self._handle_oom_recovery():
                        # OOM recovery succeeded, try again with smaller batch size
                        self.logger.info(f"Retrying inference with reduced batch size: {self.batch_size}")

                        # Split the batch if it's still too large
                        if len(positions) > self.batch_size:
                            # Process in smaller sub-batches
                            return self._process_batch_chunks(positions)
                        else:
                            # Retry with the same batch but reduced internal batch size
                            try:
                                return self.batch_inference(positions)  # Recursive retry
                            except Exception as retry_e:
                                if self._is_oom_error(retry_e):
                                    # Still getting OOM, fallback to CPU
                                    self.logger.error("OOM persists after recovery attempt, falling back to CPU")
                                    self._fallback_failure_count += 1
                                    self._enable_cpu_fallback()
                                else:
                                    raise retry_e
                    else:
                        # OOM recovery failed, fallback to CPU
                        self.logger.error("OOM recovery failed, falling back to CPU")
                        self._fallback_failure_count += 1
                        self._enable_cpu_fallback()

                # Check if this error should trigger CPU fallback
                elif should_fallback_to_cpu(e):
                    self._fallback_failure_count += 1
                    self.logger.warning(f"GPU inference failed (attempt {self._fallback_failure_count}), "
                                      f"falling back to CPU: {e}")

                    # Enable CPU fallback if not already enabled
                    if self._cpu_fallback_worker is None:
                        self._enable_cpu_fallback()
                else:
                    # Non-fallback error, re-raise
                    raise

        # Use CPU fallback if GPU failed or fallback is already active (T018)
        if self._cpu_fallback_worker is not None:
            try:
                result = self._cpu_fallback_worker.batch_inference(positions)
                # Update metrics even for CPU fallback (T015)
                inference_time = time.time() - start_time
                self._update_metrics(len(positions), inference_time)
                return result
            except Exception as fallback_error:
                self.logger.error(f"CPU fallback also failed: {fallback_error}")
                # Return safe default values as last resort
                batch_size = len(positions)
                num_actions = getattr(self, 'num_actions', 225)  # Game-specific action space
                policies_np = np.ones((batch_size, num_actions)) / num_actions  # Uniform distribution
                values_np = np.zeros(batch_size)
                # Update metrics even for fallback (T015)
                inference_time = time.time() - start_time
                self._update_metrics(batch_size, inference_time)
                return policies_np, values_np

        # No fallback available and GPU failed
        raise RuntimeError("Both GPU and CPU inference failed")

    def _update_metrics(self, batch_size: int, inference_time: float) -> None:
        """Update performance metrics with GPU utilization tracking.

        Args:
            batch_size: Size of processed batch
            inference_time: Time taken for inference (seconds)
        """
        positions_per_second = batch_size / inference_time if inference_time > 0 else 0
        gpu_util = self._get_gpu_utilization()

        with self._metrics_lock:
            if 'start_time' not in self._metrics:
                self._metrics['start_time'] = time.time()
            self._metrics['total_requests'] += batch_size
            self._metrics['total_batches'] += 1
            self._metrics['total_inference_time'] += inference_time
            self._metrics['batch_sizes'].append(batch_size)
            self._metrics['inference_times'].append(inference_time)
            self._metrics['last_update_time'] = time.time()

            # Track NN evaluations for cache hit rate analysis
            self._metrics['nn_evaluations'] += 1  # One forward pass
            self._metrics['states_evaluated'] += batch_size  # Number of states in this batch

            # Add GPU utilization metrics
            if 'gpu_utilization_samples' not in self._metrics:
                self._metrics['gpu_utilization_samples'] = deque(maxlen=100)

            if gpu_util > 0:
                self._metrics['gpu_utilization_samples'].append(gpu_util)

        # Store performance data for adaptive batching
        perf_data = {
            'batch_size': batch_size,
            'inference_time': inference_time,
            'throughput': positions_per_second,
            'gpu_utilization': gpu_util,
            'timestamp': time.time()
        }
        self._performance_history.append(perf_data)

        # Enhanced logging with GPU utilization
        gpu_info = f", GPU: {gpu_util*100:.1f}%" if gpu_util > 0 else ""
        self.logger.debug(
            f"Batch processed: {batch_size} positions in {inference_time:.3f}s "
            f"({positions_per_second:.1f} pos/s{gpu_info})"
        )

        # Log performance summary periodically
        if self._metrics['total_batches'] % 50 == 0:
            with self._metrics_lock:
                avg_gpu = 0
                if 'gpu_utilization_samples' in self._metrics and self._metrics['gpu_utilization_samples']:
                    avg_gpu = sum(self._metrics['gpu_utilization_samples']) / len(self._metrics['gpu_utilization_samples']) * 100

            gpu_status = "" if avg_gpu == 0 else f", avg GPU util: {avg_gpu:.1f}%"
            avg_throughput = self._metrics['total_requests'] / self._metrics['total_inference_time'] if self._metrics['total_inference_time'] > 0 else 0
            self.logger.info(
                f"Performance summary: {avg_throughput:.1f} pos/s, "
                f"avg batch: {self._metrics['total_requests'] / self._metrics['total_batches']:.1f}{gpu_status}"
            )

    def _calculate_batch_size_distribution(self, batch_sizes: List[int]) -> Dict[str, float]:
        """Calculate batch size distribution statistics (T015).

        Args:
            batch_sizes: List of recent batch sizes

        Returns:
            Dictionary with batch size distribution metrics
        """
        if not batch_sizes:
            return {
                'batch_size_min': 0.0,
                'batch_size_max': 0.0,
                'batch_size_median': 0.0,
                'batch_size_p50': 0.0,
                'batch_size_p90': 0.0,
                'batch_size_p95': 0.0,
                'batch_size_p99': 0.0,
                'batch_size_std': 0.0,
            }

        batch_array = np.array(batch_sizes)
        return {
            'batch_size_min': float(np.min(batch_array)),
            'batch_size_max': float(np.max(batch_array)),
            'batch_size_median': float(np.median(batch_array)),
            'batch_size_p50': float(np.percentile(batch_array, 50)),
            'batch_size_p90': float(np.percentile(batch_array, 90)),
            'batch_size_p95': float(np.percentile(batch_array, 95)),
            'batch_size_p99': float(np.percentile(batch_array, 99)),
            'batch_size_std': float(np.std(batch_array)),
        }

    def _calculate_timeout_compliance(self, inference_times: List[float]) -> Dict[str, Any]:
        """Calculate timeout compliance metrics (T015).

        Args:
            inference_times: List of recent inference times in seconds

        Returns:
            Dictionary with timeout compliance metrics
        """
        if not inference_times:
            return {
                'timeout_compliance_rate': 1.0,
                'timeout_violations': 0,
                'inference_time_p50_ms': 0.0,
                'inference_time_p90_ms': 0.0,
                'inference_time_p95_ms': 0.0,
                'inference_time_p99_ms': 0.0,
                'inference_time_max_ms': 0.0,
            }

        times_array = np.array(inference_times)
        times_ms = times_array * 1000  # Convert to milliseconds

        # Count timeout violations (times > max_timeout_ms)
        violations = np.sum(times_array > self.max_timeout_ms)
        compliance_rate = 1.0 - (violations / len(inference_times))

        return {
            'timeout_compliance_rate': float(compliance_rate),
            'timeout_violations': int(violations),
            'inference_time_p50_ms': float(np.percentile(times_ms, 50)),
            'inference_time_p90_ms': float(np.percentile(times_ms, 90)),
            'inference_time_p95_ms': float(np.percentile(times_ms, 95)),
            'inference_time_p99_ms': float(np.percentile(times_ms, 99)),
            'inference_time_max_ms': float(np.max(times_ms)),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get enhanced inference performance metrics including micro-batching data.

        Returns:
            dict: Comprehensive metrics including GPU utilization, adaptive batching info,
                 and performance targets
        """
        with self._metrics_lock:
            # Calculate averages
            recent_batch_sizes = list(self._metrics['batch_sizes'])
            recent_inference_times = list(self._metrics['inference_times'])

            avg_batch_size = np.mean(recent_batch_sizes) if recent_batch_sizes else 0.0
            avg_inference_time = np.mean(recent_inference_times) if recent_inference_times else 0.0

            # Calculate batch size distribution (T015)
            batch_size_metrics = self._calculate_batch_size_distribution(recent_batch_sizes)

            # Calculate inference rate
            if avg_inference_time > 0:
                inference_rate = avg_batch_size / avg_inference_time
            else:
                inference_rate = 0.0

            # GPU utilization metrics
            current_gpu_util = self._get_gpu_utilization()
            avg_gpu_util = 0.0
            samples = self._metrics.get('gpu_utilization_samples')
            if samples:
                avg_gpu_util = sum(samples) / len(samples)
            else:
                start_time = self._metrics.get('start_time')
                last_time = self._metrics.get('last_update_time', start_time)
                if start_time is not None and last_time is not None and last_time > start_time:
                    elapsed = last_time - start_time
                    busy = self._metrics.get('total_inference_time', 0.0)
                    if elapsed > 0:
                        avg_gpu_util = min(1.0, max(0.0, busy / elapsed))

            memory_usage_gb = self._get_memory_usage()

            # Timeout compliance metrics (T015)
            timeout_metrics = self._calculate_timeout_compliance(recent_inference_times)

            metrics = {
                # Core performance metrics
                'gpu_utilization': current_gpu_util,
                'avg_gpu_utilization': avg_gpu_util,
                'average_batch_size': avg_batch_size,
                'inference_rate': inference_rate,
                'memory_usage_gb': memory_usage_gb,
                'total_requests': self._metrics['total_requests'],
                'total_batches': self._metrics['total_batches'],
                'total_inference_time': self._metrics['total_inference_time'],

                # NN evaluation tracking (cache hit rate analysis)
                'nn_evaluations': self._metrics['nn_evaluations'],
                'states_evaluated': self._metrics['states_evaluated'],

                # Micro-batching configuration
                'current_optimal_batch': self._current_optimal_batch,
                'min_batch_size': self.min_batch_size,
                'max_timeout_ms': self.max_timeout_ms * 1000,
                'target_gpu_utilization': self.target_gpu_utilization,

                # Performance targets status
                'meets_batch_target': avg_batch_size >= self.min_batch_size,
                'meets_gpu_target': avg_gpu_util >= self.target_gpu_utilization,
                'timeout_compliance': avg_inference_time <= self.max_timeout_ms
            }

            # Add batch size distribution metrics (T015)
            metrics.update(batch_size_metrics)

            # Add timeout compliance metrics (T015)
            metrics.update(timeout_metrics)

            # Add mixed precision metrics
            memory_metrics = self._get_memory_efficiency_metrics()
            metrics.update(memory_metrics)

            # Add CPU fallback metrics (T018)
            fallback_metrics = self._get_cpu_fallback_metrics()
            metrics.update(fallback_metrics)

            return metrics

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage (enhanced version).

        Returns:
            GPU utilization as percentage (0.0-1.0), or 0.0 if unavailable
        """
        # Use nvidia-ml-py if available and initialized
        if self._gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                return util.gpu / 100.0
            except Exception:
                pass

        # Fallback to memory-based estimation
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                device_str = str(self.device)
                device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                return (memory_used / memory_total)  # Return as 0.0-1.0 not percentage
            else:
                return 0.0
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current VRAM usage in GB."""
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                device_str = str(self.device)
                device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                return memory_used / (1024**3)  # Convert to GB
            else:
                # For CPU, return system memory usage of current process
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024**3)
        except Exception:
            return 0.0

    def _get_memory_usage_fraction(self) -> float:
        """Get current GPU memory usage as fraction of total memory."""
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                device_str = str(self.device)
                device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                return memory_used / memory_total if memory_total > 0 else 0.0
        except Exception:
            pass
        return 0.0

    def _is_oom_error(self, exception: Exception) -> bool:
        """Check if exception is a CUDA out-of-memory error."""
        error_msg = str(exception).lower()
        oom_indicators = [
            'out of memory',
            'cuda out of memory',
            'cuda error: out of memory',
            'cuda runtime error: out of memory',
            'allocation failure',
            'memory exhausted'
        ]
        return any(indicator in error_msg for indicator in oom_indicators)

    def _handle_oom_recovery(self) -> bool:
        """Handle OOM recovery by reducing batch size and clearing cache.

        Returns:
            bool: True if recovery should be attempted, False if fallback to CPU
        """
        current_time = time.time()
        self._oom_count += 1
        self._consecutive_oom_count += 1
        self._last_oom_time = current_time

        self.logger.warning(f"CUDA OOM detected (#{self._oom_count}, consecutive: {self._consecutive_oom_count})")

        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cleared CUDA cache")

        # If we've had too many consecutive OOM errors, fallback to CPU
        if self._consecutive_oom_count >= 3:
            self.logger.error("Too many consecutive OOM errors, falling back to CPU permanently")
            return False

        # Calculate new batch size
        new_batch_size = max(
            self._min_batch_size,
            int(self.batch_size * self._batch_size_reduction_factor)
        )

        if new_batch_size < self.batch_size:
            old_batch_size = self.batch_size
            self.batch_size = new_batch_size
            self.logger.warning(f"Reduced batch size from {old_batch_size} to {new_batch_size} due to OOM")

            # Update batching parameters
            self._current_optimal_batch = min(self._current_optimal_batch, new_batch_size)

            # Recreate pinned memory buffers with smaller size
            if self._use_pinned_memory:
                try:
                    self._setup_pinned_memory_buffers(new_batch_size)
                    self.logger.info("Recreated pinned memory buffers with reduced size")
                except Exception as e:
                    self.logger.warning(f"Failed to recreate pinned buffers: {e}")

            return True
        else:
            # Already at minimum batch size, fallback to CPU
            self.logger.error(f"Already at minimum batch size {self._min_batch_size}, falling back to CPU")
            return False

    def _can_increase_batch_size(self) -> bool:
        """Check if batch size can be increased after successful operations."""
        current_time = time.time()

        # Don't increase if we've had recent OOM errors
        if current_time - self._last_oom_time < self._oom_recovery_cooldown:
            return False

        # Don't increase if memory usage is high
        memory_usage = self._get_memory_usage_fraction()
        if memory_usage > self._oom_memory_threshold:
            return False

        # Don't increase beyond original batch size
        if self.batch_size >= self._original_batch_size:
            return False

        return True

    def _attempt_batch_size_increase(self) -> None:
        """Gradually increase batch size if conditions are favorable."""
        if not self._can_increase_batch_size():
            return

        # Conservative increase (25% increase or +4, whichever is smaller)
        increase = min(4, max(1, int(self.batch_size * 0.25)))
        new_batch_size = min(self._original_batch_size, self.batch_size + increase)

        if new_batch_size > self.batch_size:
            old_batch_size = self.batch_size
            self.batch_size = new_batch_size
            self._current_optimal_batch = new_batch_size
            self.logger.info(f"Gradually increased batch size from {old_batch_size} to {new_batch_size}")

            # Recreate pinned memory buffers with larger size
            if self._use_pinned_memory:
                try:
                    self._setup_pinned_memory_buffers(new_batch_size)
                except Exception as e:
                    # If buffer recreation fails, revert batch size
                    self.batch_size = old_batch_size
                    self._current_optimal_batch = old_batch_size
                    self.logger.warning(f"Failed to increase buffer size, reverted to {old_batch_size}: {e}")

    def _reset_oom_recovery_state(self) -> None:
        """Reset OOM recovery state after successful operations."""
        self._consecutive_oom_count = 0
        self._last_successful_batch_size = self.batch_size

    def _process_batch_chunks(self, positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process large batch in smaller chunks to avoid OOM.

        Args:
            positions: List of position arrays to process

        Returns:
            Tuple of (policies, values) numpy arrays
        """
        chunk_size = self.batch_size
        total_positions = len(positions)
        all_policies = []
        all_values = []

        self.logger.info(f"Processing batch of {total_positions} in chunks of {chunk_size}")

        for start_idx in range(0, total_positions, chunk_size):
            end_idx = min(start_idx + chunk_size, total_positions)
            chunk = positions[start_idx:end_idx]

            try:
                # Process chunk
                chunk_tensor = self._create_batch_tensor_optimized(chunk)
                with torch.no_grad():
                    policy_logits, values = self._run_inference_with_precision(chunk_tensor)
                    policies = torch.softmax(policy_logits, dim=1)

                # Transfer chunk results
                policies_np, values_np = self._transfer_outputs_optimized(policies, values)

                all_policies.append(policies_np)
                all_values.append(values_np)

            except Exception as chunk_e:
                if self._is_oom_error(chunk_e):
                    # Even chunks are too big, reduce chunk size further
                    self.logger.warning(f"OOM in chunk processing, reducing chunk size further")
                    if self._handle_oom_recovery():
                        # Retry this chunk with even smaller size
                        return self._process_batch_chunks(positions)  # Start over with smaller chunks
                    else:
                        # Give up on GPU, use CPU fallback
                        raise chunk_e
                else:
                    raise chunk_e

        # Combine all chunk results
        combined_policies = np.concatenate(all_policies, axis=0)
        combined_values = np.concatenate(all_values, axis=0)

        self.logger.info(f"Successfully processed {total_positions} positions in {len(all_policies)} chunks")
        return combined_policies, combined_values

    def _get_cpu_fallback_metrics(self) -> Dict[str, Any]:
        """Get CPU fallback performance metrics (T018).

        Returns:
            dict: CPU fallback status and performance metrics
        """
        metrics = {
            'cpu_fallback_enabled': self._fallback_enabled,
            'cpu_fallback_active': self._fallback_triggered,
            'cpu_fallback_failure_count': self._fallback_failure_count,
            'cpu_fallback_available': self._cpu_fallback_worker is not None,
            'last_gpu_attempt': self._last_gpu_attempt
        }

        # Add CPU fallback worker metrics if available
        if self._cpu_fallback_worker is not None:
            try:
                cpu_metrics = self._cpu_fallback_worker.get_metrics()
                # Prefix CPU metrics to avoid conflicts
                for key, value in cpu_metrics.items():
                    metrics[f'cpu_{key}'] = value
            except Exception as e:
                self.logger.warning(f"Failed to get CPU fallback metrics: {e}")

        return metrics

    def get_mixed_precision_metrics(self) -> Dict[str, float]:
        """Get mixed precision performance metrics.

        Returns:
            Dictionary with mixed precision efficiency statistics
        """
        return self._get_memory_efficiency_metrics()

    def is_running(self) -> bool:
        """Check if worker thread is running."""
        return self._is_running

    @property
    def running(self) -> bool:
        """Running property for SearchCoordinator compatibility."""
        return self._is_running

    def start(self) -> None:
        """Start the inference worker with default queues for SearchCoordinator compatibility."""
        if self._is_running:
            return

        # Create default queues for SearchCoordinator interface
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure clean shutdown."""
        self.stop_worker()


def create_inference_worker(model_path: str,
                           device: str = 'cuda:0',
                           **kwargs) -> InferenceWorker:
    """Factory function to create inference worker.

    Args:
        model_path: Path to trained PyTorch model
        device: Inference device
        **kwargs: Additional worker configuration

    Returns:
        InferenceWorker: Configured inference worker instance
    """
    return GPUInferenceWorker(model_path, device=device, **kwargs)
