"""
GPU device management and optimization for the AlphaZero engine.

Provides CUDA detection, GPU warmup, batch size estimation, and RTX 3060 Ti
specific optimizations for optimal neural network inference performance.
"""

import time
import math
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from src.telemetry import get_logger


@dataclass
class DeviceInfo:
    """Information about detected GPU device."""

    name: str
    memory_total_mb: float
    memory_free_mb: float
    compute_capability: Tuple[int, int]
    device_id: int
    is_cuda_available: bool
    optimal_batch_size: Optional[int] = None
    warmup_time_ms: Optional[float] = None


class DummyModel(nn.Module):
    """Lightweight model for GPU warmup and batch size estimation."""

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int = 361):
        """
        Initialize dummy model mimicking AlphaZero architecture.

        Args:
            input_shape: (channels, height, width) for input tensors
            num_actions: Number of possible actions (e.g., 361 for 19x19 Go)
        """
        super().__init__()
        self.input_shape = input_shape
        channels, height, width = input_shape

        # Lightweight ResNet-like backbone
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [self._make_residual_block(128) for _ in range(4)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _make_residual_block(self, channels: int) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value predictions."""
        # Backbone
        x = self.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = self.relu(x + residual)

        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)

        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class DeviceManager:
    """
    GPU device manager for AlphaZero neural network inference.

    Handles CUDA detection, GPU warmup, batch size optimization,
    and RTX 3060 Ti specific performance tuning.
    """

    def __init__(self, memory_fraction: float = 0.85, warmup_iterations: int = 10):
        """
        Initialize device manager.

        Args:
            memory_fraction: Maximum fraction of GPU memory to use
            warmup_iterations: Number of warmup inference calls
        """
        self.memory_fraction = memory_fraction
        self.warmup_iterations = warmup_iterations
        self.logger = get_logger("device_manager")

        self.device_info: Optional[DeviceInfo] = None
        self.device: torch.device = torch.device("cpu")
        self.gpu_warmed_up: bool = False

        # RTX 3060 Ti specific optimizations
        self.rtx_3060_ti_memory_gb = 8.0
        self.rtx_3060_ti_compute_capability = (8, 6)

    def detect_device(self) -> DeviceInfo:
        """
        Detect and initialize GPU device.

        Returns:
            DeviceInfo with detailed device information
        """
        self.logger.info("Starting GPU device detection")

        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return DeviceInfo(
                name="CPU",
                memory_total_mb=0,
                memory_free_mb=0,
                compute_capability=(0, 0),
                device_id=-1,
                is_cuda_available=False,
            )

        # Use first GPU device
        device_id = 0
        self.device = torch.device(f"cuda:{device_id}")

        # Get device properties
        props = torch.cuda.get_device_properties(device_id)
        memory_total_mb = props.total_memory / 1024 / 1024
        memory_free_mb = (
            (
                torch.cuda.get_device_properties(device_id).total_memory
                - torch.cuda.memory_allocated(device_id)
            )
            / 1024
            / 1024
        )

        device_info = DeviceInfo(
            name=props.name,
            memory_total_mb=memory_total_mb,
            memory_free_mb=memory_free_mb,
            compute_capability=(props.major, props.minor),
            device_id=device_id,
            is_cuda_available=True,
        )

        self.device_info = device_info

        self.logger.info(
            f"GPU detected: {device_info.name}, "
            f"{device_info.memory_total_mb:.0f}MB total, "
            f"compute capability {device_info.compute_capability}"
        )

        # Apply RTX 3060 Ti specific optimizations
        if self._is_rtx_3060_ti(device_info):
            self._apply_rtx_3060_ti_optimizations()

        return device_info

    def _is_rtx_3060_ti(self, device_info: DeviceInfo) -> bool:
        """Check if device is RTX 3060 Ti based on memory and compute capability."""
        memory_match = (
            abs(device_info.memory_total_mb - self.rtx_3060_ti_memory_gb * 1024) < 512
        )
        compute_match = (
            device_info.compute_capability == self.rtx_3060_ti_compute_capability
        )
        name_match = (
            "RTX 3060" in device_info.name or "GeForce RTX 3060" in device_info.name
        )

        return memory_match and (compute_match or name_match)

    def _apply_rtx_3060_ti_optimizations(self) -> None:
        """Apply RTX 3060 Ti specific optimizations."""
        self.logger.info("Applying RTX 3060 Ti optimizations")

        # Enable TensorFloat-32 for better performance on Ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark mode for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Reduce memory fragmentation
        torch.cuda.empty_cache()

        self.logger.info("RTX 3060 Ti optimizations applied")

    def warmup(self, input_shape: Tuple[int, int, int]) -> float:
        """
        Warmup GPU with dummy inference calls.

        Critical for consistent latency measurements and CUDA initialization.

        Args:
            input_shape: (channels, height, width) for input tensors

        Returns:
            Average warmup time per inference in milliseconds
        """
        if not self.device_info or not self.device_info.is_cuda_available:
            self.logger.warning("No GPU available for warmup")
            return 0.0

        self.logger.info(
            f"Starting GPU warmup with {self.warmup_iterations} iterations"
        )

        # Create dummy model
        dummy_model = DummyModel(input_shape).to(self.device)
        dummy_model.eval()

        # Create dummy input
        batch_size = 8  # Small batch for warmup
        dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

        # Warmup iterations
        warmup_times = []

        with torch.no_grad():
            for i in range(self.warmup_iterations):
                start_time = time.perf_counter()

                policy, value = dummy_model(dummy_input)
                torch.cuda.synchronize()  # Ensure completion

                end_time = time.perf_counter()
                iteration_time_ms = (end_time - start_time) * 1000
                warmup_times.append(iteration_time_ms)

                if i == 0:
                    self.logger.debug(
                        f"First warmup iteration: {iteration_time_ms:.2f}ms"
                    )

        # Calculate average warmup time
        avg_warmup_time = sum(warmup_times[1:]) / max(
            1, len(warmup_times) - 1
        )  # Skip first iteration

        if self.device_info:
            self.device_info.warmup_time_ms = avg_warmup_time

        self.logger.info(
            f"GPU warmup completed: {avg_warmup_time:.2f}ms average inference time"
        )

        # Mark GPU as warmed up
        self.gpu_warmed_up = True

        # Clean up
        del dummy_model, dummy_input
        torch.cuda.empty_cache()

        return avg_warmup_time

    def estimate_batch_size(
        self, input_shape: Tuple[int, int, int], max_batch_size: int = 256
    ) -> int:
        """
        Estimate optimal batch size using binary search within memory constraints.

        Args:
            input_shape: (channels, height, width) for input tensors
            max_batch_size: Maximum batch size to test

        Returns:
            Optimal batch size that fits within memory constraints
        """
        if not self.device_info or not self.device_info.is_cuda_available:
            self.logger.warning("No GPU available for batch size estimation")
            return 1

        self.logger.info("Estimating optimal batch size using binary search")

        # Create model for testing
        test_model = DummyModel(input_shape).to(self.device)
        test_model.eval()

        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal_batch_size = 1

        try:
            while low <= high:
                mid = (low + high) // 2

                if self._test_batch_size(test_model, input_shape, mid):
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1

            # Reduce by safety margin (10%) to account for inference worker overhead
            optimal_batch_size = max(1, int(optimal_batch_size * 0.9))

        except Exception as e:
            self.logger.error(f"Error during batch size estimation: {e}")
            optimal_batch_size = 32  # Safe default for RTX 3060 Ti

        finally:
            # Clean up
            del test_model
            torch.cuda.empty_cache()

        if self.device_info:
            self.device_info.optimal_batch_size = optimal_batch_size

        self.logger.info(f"Optimal batch size estimated: {optimal_batch_size}")

        return optimal_batch_size

    def _test_batch_size(
        self, model: nn.Module, input_shape: Tuple[int, int, int], batch_size: int
    ) -> bool:
        """
        Test if a given batch size fits within memory constraints.

        Args:
            model: Test model
            input_shape: Input tensor shape
            batch_size: Batch size to test

        Returns:
            True if batch size fits, False otherwise
        """
        try:
            # Clear cache before test
            torch.cuda.empty_cache()

            # Create test input
            test_input = torch.randn(batch_size, *input_shape, device=self.device)

            # Test forward pass
            with torch.no_grad():
                policy, value = model(test_input)
                torch.cuda.synchronize()

            # Check memory usage
            memory_used = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # MB
            memory_limit = self.device_info.memory_total_mb * self.memory_fraction

            success = memory_used <= memory_limit

            if not success:
                self.logger.debug(
                    f"Batch size {batch_size} failed: {memory_used:.0f}MB > {memory_limit:.0f}MB"
                )

            # Clean up test tensors
            del test_input, policy, value
            torch.cuda.empty_cache()

            return success

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.debug(f"Batch size {batch_size} failed: OOM")
                return False
            else:
                raise e

    def get_device_info(self) -> Optional[DeviceInfo]:
        """Get current device information."""
        return self.device_info

    def get_device(self) -> torch.device:
        """Get PyTorch device object."""
        return self.device

    def initialize(self, input_shape: Tuple[int, int, int]) -> DeviceInfo:
        """
        Complete device initialization: detect, warmup, and optimize.

        Args:
            input_shape: (channels, height, width) for input tensors

        Returns:
            DeviceInfo with all optimization results
        """
        start_time = time.perf_counter()

        # Step 1: Detect device
        device_info = self.detect_device()

        if device_info.is_cuda_available:
            # Step 2: Warmup GPU
            warmup_time = self.warmup(input_shape)

            # Step 3: Estimate optimal batch size
            optimal_batch_size = self.estimate_batch_size(input_shape)

            total_time = time.perf_counter() - start_time

            self.logger.info(
                f"Device initialization completed in {total_time:.2f}s: "
                f"warmup {warmup_time:.2f}ms, batch size {optimal_batch_size}"
            )
        else:
            self.logger.warning("GPU initialization skipped - CUDA not available")

        return device_info

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if not self.device_info or not self.device_info.is_cuda_available:
            return {}

        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024  # MB

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": self.device_info.memory_total_mb,
            "utilization_percent": (allocated / self.device_info.memory_total_mb) * 100,
        }


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def initialize_device(input_shape: Tuple[int, int, int]) -> DeviceInfo:
    """
    Initialize global device manager with given input shape.

    Args:
        input_shape: (channels, height, width) for neural network inputs

    Returns:
        DeviceInfo with initialization results
    """
    device_manager = get_device_manager()
    return device_manager.initialize(input_shape)
