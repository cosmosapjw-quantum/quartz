"""
GPU Utilization Monitoring
===========================

Lightweight GPU monitoring using NVML (NVIDIA Management Library).
Provides real-time GPU utilization for adaptive batching decisions.

Usage:
    monitor = GPUMonitor()
    util = monitor.get_utilization()  # Returns 0.0-1.0

Reference: comments.md Section 3, Issue #3C (Adaptive batching)
"""

import logging
from typing import Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class GPUMonitor:
    """Monitor GPU utilization using NVML.

    Provides real-time GPU utilization percentage for adaptive batching.
    Falls back to 0.5 (50%) if NVML not available or on CPU.

    Thread-safe: Can be called from multiple threads.
    """

    def __init__(self, device_index: int = 0):
        """Initialize GPU monitor.

        Args:
            device_index: CUDA device index to monitor (default: 0)
        """
        self.device_index = device_index
        self.handle = None
        self.logger = logging.getLogger(__name__)

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                device_name = pynvml.nvmlDeviceGetName(self.handle)
                self.logger.info(f"GPU monitor initialized: {device_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}. Using fallback.")
                self.handle = None
        else:
            self.logger.warning(
                "pynvml not available. Install with: pip install nvidia-ml-py3"
            )

    def get_utilization(self) -> float:
        """Get current GPU utilization.

        Returns:
            float: GPU utilization in range [0.0, 1.0]
                   0.0 = idle, 1.0 = fully utilized
                   Falls back to 0.5 if monitoring unavailable
        """
        if self.handle is None:
            return 0.5  # Fallback: assume 50% utilization

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            # util.gpu is percentage (0-100), convert to fraction
            return util.gpu / 100.0
        except Exception as e:
            self.logger.warning(f"Failed to get GPU utilization: {e}")
            return 0.5  # Fallback

    def get_memory_info(self) -> dict:
        """Get GPU memory information.

        Returns:
            dict: Memory info with keys 'used', 'free', 'total' (in bytes)
                  Empty dict if monitoring unavailable
        """
        if self.handle is None:
            return {}

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                'used': mem.used,
                'free': mem.free,
                'total': mem.total,
                'utilization': mem.used / mem.total
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory info: {e}")
            return {}

    def shutdown(self):
        """Shutdown NVML (call on program exit)."""
        if HAS_NVML and self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class AdaptiveBatchController:
    """Adaptive batching controller for dynamic timeout adjustment.

    Adjusts batch timeout (2-10ms) based on GPU utilization to maximize throughput
    while maintaining GPU saturation.

    Strategy:
    - High GPU util (>80%): Use shorter timeout (2-4ms) to keep GPU fed
    - Medium GPU util (50-80%): Use medium timeout (4-7ms)
    - Low GPU util (<50%): Use longer timeout (7-10ms) to fill batches better

    Includes smoothing to prevent oscillation.

    Reference: comments.md Section 3, Issue #3C
    """

    def __init__(
        self,
        gpu_monitor: Optional[GPUMonitor] = None,
        min_timeout_ms: float = 2.0,
        max_timeout_ms: float = 10.0,
        smoothing_factor: float = 0.7
    ):
        """Initialize adaptive batch controller.

        Args:
            gpu_monitor: GPUMonitor instance (creates one if None)
            min_timeout_ms: Minimum batch timeout in milliseconds (default: 2.0)
            max_timeout_ms: Maximum batch timeout in milliseconds (default: 10.0)
            smoothing_factor: Exponential smoothing factor [0,1] (default: 0.7)
                              Higher = more smoothing, slower adaptation
        """
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        self.smoothing_factor = smoothing_factor

        # State tracking
        self.current_timeout_ms = (min_timeout_ms + max_timeout_ms) / 2  # Start at midpoint
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Adaptive batch controller initialized: "
            f"timeout range [{min_timeout_ms}, {max_timeout_ms}] ms, "
            f"smoothing={smoothing_factor}"
        )

    def get_timeout(self) -> float:
        """Compute adaptive timeout based on current GPU utilization.

        Returns:
            float: Timeout in milliseconds, in range [min_timeout_ms, max_timeout_ms]
        """
        # Get current GPU utilization (0.0-1.0)
        gpu_util = self.gpu_monitor.get_utilization()

        # Compute target timeout based on utilization
        # Strategy from comments.md:
        # base = 2.0 + (1.0 - min(util, 0.9)) * 8.0
        # This gives:
        # - util=0.0 → timeout=10ms (low util, wait longer to fill batches)
        # - util=0.5 → timeout=6ms
        # - util=0.9 → timeout=2.8ms (high util, keep GPU fed)
        # - util=1.0 → timeout=2.0ms (saturated)

        clamped_util = min(gpu_util, 0.9)
        target_timeout_ms = self.min_timeout_ms + (1.0 - clamped_util) * (
            self.max_timeout_ms - self.min_timeout_ms
        )

        # Apply exponential smoothing to prevent oscillation
        # new = alpha * target + (1-alpha) * current
        alpha = 1.0 - self.smoothing_factor
        smoothed_timeout_ms = (
            alpha * target_timeout_ms +
            self.smoothing_factor * self.current_timeout_ms
        )

        # Clamp to valid range
        smoothed_timeout_ms = max(
            self.min_timeout_ms,
            min(self.max_timeout_ms, smoothed_timeout_ms)
        )

        # Update state
        self.current_timeout_ms = smoothed_timeout_ms

        return smoothed_timeout_ms

    def get_stats(self) -> dict:
        """Get controller statistics.

        Returns:
            dict: Stats with keys 'gpu_utilization', 'current_timeout_ms'
        """
        return {
            'gpu_utilization': self.gpu_monitor.get_utilization(),
            'current_timeout_ms': self.current_timeout_ms,
            'min_timeout_ms': self.min_timeout_ms,
            'max_timeout_ms': self.max_timeout_ms
        }

    def shutdown(self):
        """Shutdown GPU monitor."""
        self.gpu_monitor.shutdown()
