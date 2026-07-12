"""
Telemetry and metrics collection for the AlphaZero engine.

Provides Prometheus-compatible metrics collection with GPU utilization monitoring,
memory usage tracking, and performance metrics for simulations/sec.
"""

import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import defaultdict, deque

import psutil
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    simulations_per_second: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Prometheus-compatible metrics collector for AlphaZero engine performance monitoring.

    Tracks:
    - Simulations per second
    - GPU utilization and memory usage
    - System memory and CPU usage
    - Performance counters for different operations
    """

    def __init__(self, collect_interval: float = 1.0):
        """
        Initialize metrics collector.

        Args:
            collect_interval: How often to collect GPU/system metrics (seconds)
        """
        self.collect_interval = collect_interval
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Prometheus registry and metrics
        self.registry = CollectorRegistry()

        # Performance counters
        self.simulations_total = Counter(
            "alphazero_simulations_total",
            "Total number of MCTS simulations performed",
            ["game_type"],
            registry=self.registry,
        )

        self.inference_requests_total = Counter(
            "alphazero_inference_requests_total",
            "Total number of neural network inference requests",
            registry=self.registry,
        )

        # Performance histograms
        self.simulation_duration = Histogram(
            "alphazero_simulation_duration_seconds",
            "Time taken for individual MCTS simulations",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        self.inference_duration = Histogram(
            "alphazero_inference_duration_seconds",
            "Time taken for neural network inference",
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            registry=self.registry,
        )

        self.batch_size = Histogram(
            "alphazero_inference_batch_size",
            "Neural network inference batch sizes",
            buckets=[1, 8, 16, 32, 48, 64, 96, 128],
            registry=self.registry,
        )

        # System resource gauges
        self.gpu_utilization = Gauge(
            "alphazero_gpu_utilization_percent",
            "GPU utilization percentage",
            registry=self.registry,
        )

        self.gpu_memory_used = Gauge(
            "alphazero_gpu_memory_used_mb",
            "GPU memory used in megabytes",
            registry=self.registry,
        )

        self.gpu_memory_total = Gauge(
            "alphazero_gpu_memory_total_mb",
            "Total GPU memory in megabytes",
            registry=self.registry,
        )

        self.system_memory_used = Gauge(
            "alphazero_system_memory_used_mb",
            "System memory used in megabytes",
            registry=self.registry,
        )

        self.system_memory_total = Gauge(
            "alphazero_system_memory_total_mb",
            "Total system memory in megabytes",
            registry=self.registry,
        )

        self.cpu_percent = Gauge(
            "alphazero_cpu_percent",
            "CPU utilization percentage",
            registry=self.registry,
        )

        self.simulations_per_second = Gauge(
            "alphazero_simulations_per_second",
            "Current simulations per second rate",
            registry=self.registry,
        )

        # Performance tracking
        self._simulation_times: deque = deque(maxlen=1000)
        self._simulation_counts = defaultdict(int)
        self._last_simulation_reset = time.time()

        # GPU monitoring setup
        self._gpu_handle = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                        0
                    )  # Use first GPU
            except pynvml.NVMLError:
                self._gpu_handle = None

    def start_collection(self) -> None:
        """Start background metrics collection."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._collector_thread = threading.Thread(
                target=self._collection_loop, daemon=True, name="MetricsCollector"
            )
            self._collector_thread.start()

    def stop_collection(self) -> None:
        """Stop background metrics collection."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._collector_thread:
                self._collector_thread.join(timeout=2.0)
                self._collector_thread = None

    def _collection_loop(self) -> None:
        """Background thread for collecting system metrics."""
        while self._running:
            try:
                self._update_system_metrics()
                time.sleep(self.collect_interval)
            except Exception as e:
                # Log error but continue collection
                print(f"Metrics collection error: {e}")
                time.sleep(self.collect_interval)

    def _update_system_metrics(self) -> None:
        """Update system resource metrics."""
        # System memory
        memory = psutil.virtual_memory()
        self.system_memory_used.set(memory.used / 1024 / 1024)  # MB
        self.system_memory_total.set(memory.total / 1024 / 1024)  # MB

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_percent.set(cpu_percent)

        # GPU metrics
        if self._gpu_handle:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                self.gpu_utilization.set(util.gpu)

                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self.gpu_memory_used.set(memory_info.used / 1024 / 1024)  # MB
                self.gpu_memory_total.set(memory_info.total / 1024 / 1024)  # MB

            except pynvml.NVMLError:
                # GPU might not be available
                pass

        # Update simulations per second
        self._update_simulation_rate()

    def _update_simulation_rate(self) -> None:
        """Update the simulations per second metric."""
        current_time = time.time()
        time_window = 10.0  # 10-second window

        # Remove old simulation times
        cutoff_time = current_time - time_window
        while self._simulation_times and self._simulation_times[0] < cutoff_time:
            self._simulation_times.popleft()

        # Calculate rate
        if len(self._simulation_times) > 0:
            rate = len(self._simulation_times) / time_window
            self.simulations_per_second.set(rate)

    def record_simulation(self, game_type: str, duration: float) -> None:
        """
        Record a completed MCTS simulation.

        Args:
            game_type: Type of game (e.g., 'gomoku', 'chess', 'go')
            duration: Time taken for the simulation in seconds
        """
        current_time = time.time()

        # Update counters and histograms
        self.simulations_total.labels(game_type=game_type).inc()
        self.simulation_duration.observe(duration)

        # Track for rate calculation
        self._simulation_times.append(current_time)

    def record_inference(self, batch_size: int, duration: float) -> None:
        """
        Record a neural network inference request.

        Args:
            batch_size: Number of positions in the batch
            duration: Time taken for inference in seconds
        """
        self.inference_requests_total.inc()
        self.inference_duration.observe(duration)
        self.batch_size.observe(batch_size)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        # Get latest system metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)

        # GPU metrics
        gpu_util = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0

        if self._gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = util.gpu

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_memory_used = memory_info.used / 1024 / 1024
                gpu_memory_total = memory_info.total / 1024 / 1024
            except pynvml.NVMLError:
                pass

        # Calculate current simulation rate
        current_time = time.time()
        time_window = 10.0
        cutoff_time = current_time - time_window
        recent_simulations = sum(1 for t in self._simulation_times if t >= cutoff_time)
        sim_rate = recent_simulations / time_window if recent_simulations > 0 else 0.0

        return PerformanceMetrics(
            simulations_per_second=sim_rate,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            system_memory_used_mb=memory.used / 1024 / 1024,
            system_memory_total_mb=memory.total / 1024 / 1024,
            cpu_percent=cpu_percent,
            timestamp=current_time,
        )

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def reset_counters(self) -> None:
        """Reset all performance counters."""
        self._simulation_times.clear()
        self._simulation_counts.clear()
        self._last_simulation_reset = time.time()


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        _global_collector.start_collection()
    return _global_collector


def cleanup_metrics() -> None:
    """Clean up the global metrics collector."""
    global _global_collector
    if _global_collector:
        _global_collector.stop_collection()
        _global_collector = None
