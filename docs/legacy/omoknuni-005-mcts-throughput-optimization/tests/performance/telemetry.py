"""
Performance telemetry data structures for MCTS benchmark harness.

This module provides comprehensive telemetry collection for performance validation
according to spec.md v2.0 requirements.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
import time
import os
import subprocess
from datetime import datetime


@dataclass
class Telemetry:
    """Performance telemetry for single benchmark run."""

    # PRIMARY KPIs (spec.md G1-G8)
    throughput: float = 0.0  # simulations/sec (target: ≥8,000)
    gpu_util_percent: float = 0.0  # 0-100 (target: 80-95%)
    cpu_util_percent: float = 0.0  # 0-100
    avg_batch_size: float = 0.0  # positions per batch (target: 32-64)
    batch_timeout_ms: float = 0.0  # milliseconds (target: ≤3ms)

    # CPU BREAKDOWN (review.txt analysis)
    feature_extraction_ms: float = 0.0  # per batch-64 (baseline: 7.5ms, target: <1.0ms)
    selection_time_ms: float = 0.0  # total selection phase time
    expansion_time_ms: float = 0.0  # total expansion phase time
    backup_time_ms: float = 0.0  # total backup phase time
    queue_wait_ms: float = 0.0  # time waiting for inference results

    # THREAD METRICS (review.txt thread analysis)
    num_threads: int = 0  # MCTS worker threads
    thread_efficiency: float = 0.0  # vs linear scaling (0-1)
    thread_idle_percent: float = 0.0  # 0-100 (baseline: 60%, target: <1%)
    thread_contention_count: int = 0  # atomic contention events

    # MEMORY (spec.md REQ-PERF-004)
    memory_rss_mb: float = 0.0  # Resident set size
    memory_peak_mb: float = 0.0  # Peak usage (target: <1GB for 10M nodes)
    tree_size_nodes: int = 0  # Total MCTS nodes allocated
    bytes_per_node: float = 0.0  # Memory efficiency (target: <64 bytes)

    # GPU METRICS (spec.md G2)
    gpu_inference_ms: float = 0.0  # Total GPU inference time
    gpu_memory_mb: float = 0.0  # GPU memory usage
    batches_submitted: int = 0  # Total batches sent to GPU
    batches_per_second: float = 0.0  # Batch submission rate

    # CACHE METRICS (if enabled, Phase 3 optional)
    cache_hit_rate: float = 0.0  # 0-1 (Phase 3 NN-eval cache)
    cache_size_entries: int = 0  # Current cache entries
    cache_evictions: int = 0  # Total evictions

    # COLLISION METRICS (spec.md T001 virtual loss)
    virtual_loss_collisions: int = 0  # Busy-edge masking events
    expansion_races: int = 0  # Thread expansion conflicts
    root_preexpansion_enabled: bool = False  # Root pre-expansion (T003)

    # TIMING BREAKDOWN
    total_time_sec: float = 0.0  # Wall-clock time for benchmark
    mcts_overhead_percent: float = 0.0  # CPU time percentage
    gpu_overhead_percent: float = 0.0  # GPU time percentage

    # METADATA
    timestamp: str = ""  # ISO 8601 timestamp
    git_commit: str = ""  # Git commit hash
    git_branch: str = ""  # Git branch name
    config: Dict = field(default_factory=dict)  # Benchmark configuration
    hostname: str = ""  # System hostname
    python_version: str = ""  # Python version
    pytorch_version: str = ""  # PyTorch version
    cuda_version: str = ""  # CUDA version
    cpu_model: str = ""  # CPU model string
    gpu_model: str = ""  # GPU model string

    # OPTIMIZATION FLAGS (feature flags from T003)
    openmp_enabled: bool = False  # OpenMP parallelization
    state_pooling_enabled: bool = False  # State reuse (Phase 1 T007-T009)
    condition_vars_enabled: bool = False  # CV synchronization (Phase 1 T010-T011)
    node_allocator_optimized: bool = False  # Arena allocation (Phase 1 T012-T013)
    nn_cache_enabled: bool = False  # NN-eval cache (Phase 3)
    fp16_enabled: bool = False  # Mixed precision (validated in Phase 5)

    def __post_init__(self):
        """Initialize metadata fields."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.git_commit:
            self.git_commit = self._get_git_commit()
        if not self.git_branch:
            self.git_branch = self._get_git_branch()
        if not self.hostname:
            self.hostname = os.uname().nodename
        if not self.cpu_model:
            self.cpu_model = self._get_cpu_model()

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return "unknown"

    @staticmethod
    def _get_git_branch() -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return "unknown"

    @staticmethod
    def _get_cpu_model() -> str:
        """Get CPU model string."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (IOError, IndexError):
            pass
        return "unknown"

    def to_dict(self) -> Dict:
        """Convert telemetry to dictionary for serialization."""
        return asdict(self)

    def to_csv_row(self) -> Dict:
        """Convert telemetry to flat CSV row (exclude nested config)."""
        data = self.to_dict()
        # Flatten config into top-level fields with config_ prefix
        if "config" in data:
            config = data.pop("config")
            for key, value in config.items():
                data[f"config_{key}"] = value
        return data

    def compute_derived_metrics(self):
        """Compute derived metrics from raw measurements."""
        # Thread efficiency (actual speedup / linear speedup)
        if self.num_threads > 1 and self.throughput > 0:
            # Assume single-threaded baseline is stored in config
            baseline_throughput = self.config.get("baseline_throughput_single_thread", 0)
            if baseline_throughput > 0:
                actual_speedup = self.throughput / baseline_throughput
                linear_speedup = self.num_threads
                self.thread_efficiency = actual_speedup / linear_speedup

        # Bytes per node
        if self.tree_size_nodes > 0 and self.memory_rss_mb > 0:
            self.bytes_per_node = (self.memory_rss_mb * 1024 * 1024) / self.tree_size_nodes

        # Batches per second
        if self.total_time_sec > 0 and self.batches_submitted > 0:
            self.batches_per_second = self.batches_submitted / self.total_time_sec

        # MCTS vs GPU overhead percentage
        if self.total_time_sec > 0:
            gpu_fraction = self.gpu_inference_ms / (self.total_time_sec * 1000)
            self.gpu_overhead_percent = gpu_fraction * 100
            self.mcts_overhead_percent = (1 - gpu_fraction) * 100


@dataclass
class BenchmarkStatistics:
    """Statistical summary of multiple benchmark runs."""

    # Mean values
    mean_throughput: float = 0.0
    mean_gpu_util: float = 0.0
    mean_batch_size: float = 0.0
    mean_thread_idle: float = 0.0

    # Standard deviation
    std_throughput: float = 0.0
    std_gpu_util: float = 0.0
    std_batch_size: float = 0.0

    # Coefficient of variation (std/mean)
    cv_throughput: float = 0.0
    cv_gpu_util: float = 0.0

    # Min/Max
    min_throughput: float = 0.0
    max_throughput: float = 0.0

    # Sample size
    num_runs: int = 0

    # Pass/Fail against targets
    meets_throughput_target: bool = False  # ≥8,000 sims/sec
    meets_gpu_target: bool = False  # 80-95% utilization
    meets_memory_target: bool = False  # <1GB for 10M nodes

    @classmethod
    def from_telemetry_list(cls, telemetry_list: List[Telemetry]) -> 'BenchmarkStatistics':
        """Compute statistics from list of telemetry measurements."""
        if not telemetry_list:
            return cls()

        import statistics as stats

        throughputs = [t.throughput for t in telemetry_list]
        gpu_utils = [t.gpu_util_percent for t in telemetry_list]
        batch_sizes = [t.avg_batch_size for t in telemetry_list]
        thread_idles = [t.thread_idle_percent for t in telemetry_list]

        mean_throughput = stats.mean(throughputs)
        mean_gpu_util = stats.mean(gpu_utils)
        mean_batch_size = stats.mean(batch_sizes)
        mean_thread_idle = stats.mean(thread_idles)

        std_throughput = stats.stdev(throughputs) if len(throughputs) > 1 else 0.0
        std_gpu_util = stats.stdev(gpu_utils) if len(gpu_utils) > 1 else 0.0
        std_batch_size = stats.stdev(batch_sizes) if len(batch_sizes) > 1 else 0.0

        cv_throughput = (std_throughput / mean_throughput) if mean_throughput > 0 else 0.0
        cv_gpu_util = (std_gpu_util / mean_gpu_util) if mean_gpu_util > 0 else 0.0

        # Check against targets from spec.md
        meets_throughput = mean_throughput >= 8000  # G1 target
        meets_gpu = 80 <= mean_gpu_util <= 95  # G2 target
        meets_memory = all(t.memory_peak_mb < 1024 for t in telemetry_list)  # REQ-PERF-004

        return cls(
            mean_throughput=mean_throughput,
            mean_gpu_util=mean_gpu_util,
            mean_batch_size=mean_batch_size,
            mean_thread_idle=mean_thread_idle,
            std_throughput=std_throughput,
            std_gpu_util=std_gpu_util,
            std_batch_size=std_batch_size,
            cv_throughput=cv_throughput,
            cv_gpu_util=cv_gpu_util,
            min_throughput=min(throughputs),
            max_throughput=max(throughputs),
            num_runs=len(telemetry_list),
            meets_throughput_target=meets_throughput,
            meets_gpu_target=meets_gpu,
            meets_memory_target=meets_memory,
        )

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary."""
        return asdict(self)
