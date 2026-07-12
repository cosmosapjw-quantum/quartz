"""
GPU Profiling System for Neural Network Inference
================================================

Comprehensive GPU profiling for PyTorch inference pipeline with RTX 3060 Ti optimization.
Provides detailed CUDA metrics, kernel timing, memory analysis, and performance visualization.

Architecture:
- CUDA event timing for precise kernel measurements
- NVML integration for hardware metrics (SM occupancy, power, thermal)
- torch.profiler integration for kernel/memory analysis
- CUPTI-based metrics via torch.profiler
- TensorBoard export for visualization
- Real-time monitoring thread for sustained profiling

Target Hardware: RTX 3060 Ti (GA104, 8GB GDDR6, 4864 CUDA cores, 152 Tensor Cores)

Performance Targets:
- GPU utilization: 80-92%
- Batch sizes: 32-64 (optimal for 8GB VRAM)
- Inference latency: <10ms/batch @ FP32, <5ms/batch @ FP16
- Memory bandwidth utilization: >70%
- Tensor Core utilization: >50% (FP16 mode)

Usage:
    >>> profiler = GPUProfiler(device='cuda:0', log_dir='runs/profile')
    >>> profiler.start_profiling(trace_memory=True, profile_tensorboard=True)
    >>> # Run inference workload
    >>> for batch in batches:
    >>>     with profiler.profile_batch():
    >>>         model(batch)
    >>> profiler.stop_profiling()
    >>> report = profiler.generate_report()
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from contextlib import contextmanager
import statistics

import torch
import numpy as np

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    from torch.profiler import tensorboard_trace_handler
    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False


@dataclass
class CUDAMetrics:
    """CUDA-level metrics from NVML."""
    timestamp: float
    gpu_utilization: float  # 0-100%
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization: float  # 0-100%
    sm_clock_mhz: int
    memory_clock_mhz: int
    power_draw_watts: float
    power_limit_watts: float
    temperature_c: int
    fan_speed_percent: int
    pcie_throughput_mbps: float
    compute_mode: str
    persistence_mode: bool


@dataclass
class KernelMetrics:
    """PyTorch kernel-level metrics."""
    kernel_name: str
    count: int
    total_time_us: float
    avg_time_us: float
    min_time_us: float
    max_time_us: float
    cuda_time_us: float
    cpu_time_us: float
    self_cuda_time_us: float
    occupancy: Optional[float] = None  # 0-1, if available


@dataclass
class MemoryMetrics:
    """GPU memory metrics."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    num_allocations: int
    num_deallocations: int
    allocation_rate: float  # allocs/sec
    deallocation_rate: float  # deallocs/sec
    fragmentation_ratio: float  # 0-1


@dataclass
class InferenceBatchMetrics:
    """Per-batch inference metrics."""
    batch_id: int
    batch_size: int
    timestamp: float

    # End-to-end timing
    total_time_ms: float
    h2d_transfer_ms: float
    inference_ms: float
    d2h_transfer_ms: float

    # GPU state during inference
    gpu_utilization: float
    memory_used_mb: float
    sm_clock_mhz: int
    power_draw_watts: float

    # Throughput
    samples_per_second: float

    # Queue state (if available)
    queue_depth: Optional[int] = None
    queue_wait_ms: Optional[float] = None


@dataclass
class TensorCoreMetrics:
    """Tensor Core utilization metrics (FP16/mixed precision)."""
    enabled: bool
    utilization: float  # 0-100%
    total_ops: int
    fp16_ops: int
    fp32_ops: int
    tensor_core_speedup: float  # vs FP32


@dataclass
class ProfilingSession:
    """Complete profiling session data."""
    session_id: str
    device: str
    start_time: float
    end_time: float
    duration_seconds: float

    # Hardware info
    gpu_name: str
    compute_capability: Tuple[int, int]
    total_memory_mb: float

    # Aggregate metrics
    total_batches: int
    total_samples: int
    avg_batch_size: float
    avg_throughput: float  # samples/sec

    # Timing breakdown
    avg_h2d_transfer_ms: float
    avg_inference_ms: float
    avg_d2h_transfer_ms: float
    total_time_ms: float

    # GPU utilization
    avg_gpu_utilization: float
    p50_gpu_utilization: float
    p95_gpu_utilization: float

    # Memory
    avg_memory_used_mb: float
    max_memory_used_mb: float
    memory_efficiency: float  # avg_used / total

    # Power
    avg_power_watts: float
    total_energy_joules: float

    # Kernel statistics
    top_kernels: List[KernelMetrics] = field(default_factory=list)

    # Batch metrics
    batch_metrics: List[InferenceBatchMetrics] = field(default_factory=list)


class GPUProfiler:
    """Comprehensive GPU profiling system for neural network inference.

    Provides multi-level profiling:
    1. CUDA hardware metrics via NVML (real-time)
    2. PyTorch kernel profiling via torch.profiler
    3. CUDA event timing for precise measurements
    4. Memory allocation tracking
    5. TensorBoard export for visualization

    Args:
        device: CUDA device ('cuda', 'cuda:0', etc.)
        log_dir: Directory for profiling outputs
        enable_nvml: Enable NVML hardware monitoring
        enable_torch_profiler: Enable torch.profiler (adds overhead)
        sampling_interval_ms: Interval for hardware metrics sampling
        memory_profiling: Enable detailed memory profiling
        tensorboard_export: Export to TensorBoard format
    """

    def __init__(
        self,
        device: str = 'cuda:0',
        log_dir: str = 'runs/gpu_profiling',
        enable_nvml: bool = True,
        enable_torch_profiler: bool = True,
        sampling_interval_ms: float = 100,
        memory_profiling: bool = True,
        tensorboard_export: bool = True
    ):
        self.device = torch.device(device)
        self.device_id = self._parse_device_id(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_nvml = enable_nvml and NVML_AVAILABLE
        self.enable_torch_profiler = enable_torch_profiler and TORCH_PROFILER_AVAILABLE
        self.sampling_interval = sampling_interval_ms / 1000.0
        self.memory_profiling = memory_profiling
        self.tensorboard_export = tensorboard_export

        # State
        self._profiling = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Session tracking
        self.session_id = f"session_{int(time.time())}"
        self.session_start_time: Optional[float] = None
        self.session_end_time: Optional[float] = None

        # Data storage
        self._cuda_metrics: deque = deque(maxlen=10000)  # NVML samples
        self._batch_metrics: List[InferenceBatchMetrics] = []
        self._memory_snapshots: List[MemoryMetrics] = []
        self._kernel_stats: Dict[str, KernelMetrics] = {}

        # CUDA events for timing
        self._cuda_events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._create_cuda_events()

        # Batch tracking
        self._batch_counter = 0
        self._current_batch_start: Optional[float] = None

        # NVML initialization (will be done after logger initialization)
        self._gpu_handle = None

        # Torch profiler
        self._torch_profiler: Optional[profile] = None

        # Logger (initialize early - needed by _init_nvml)
        self.logger = logging.getLogger(__name__)

        # NVML initialization (requires logger to be set up first)
        if self.enable_nvml:
            self._init_nvml()

        self.logger.info(f"GPUProfiler initialized: device={device}, nvml={self.enable_nvml}")

    def _parse_device_id(self, device: str) -> int:
        """Parse device ID from device string."""
        if ':' in device:
            return int(device.split(':')[1])
        return 0

    def _init_nvml(self):
        """Initialize NVML for hardware monitoring."""
        if not NVML_AVAILABLE:
            self.logger.warning("pynvml not available, hardware monitoring disabled")
            self.enable_nvml = False
            return

        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            # Get device info
            name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            self.logger.info(f"NVML initialized for device {self.device_id}: {name}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize NVML: {e}")
            self.enable_nvml = False
            self._gpu_handle = None

    def _create_cuda_events(self):
        """Create reusable CUDA events for timing."""
        event_pairs = [
            'h2d_transfer',
            'inference',
            'd2h_transfer',
            'batch_total'
        ]

        for name in event_pairs:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self._cuda_events[name] = (start, end)

    def start_profiling(
        self,
        trace_memory: bool = True,
        profile_tensorboard: bool = True,
        wait_steps: int = 1,
        warmup_steps: int = 2,
        active_steps: int = 10,
        repeat: int = 1
    ):
        """Start GPU profiling session.

        Args:
            trace_memory: Enable memory profiling (adds overhead)
            profile_tensorboard: Export to TensorBoard
            wait_steps: Steps to skip before profiling
            warmup_steps: Warmup steps (not profiled)
            active_steps: Steps to actively profile
            repeat: Number of profiling cycles
        """
        if self._profiling:
            self.logger.warning("Profiling already active")
            return

        self._profiling = True
        self.session_start_time = time.time()
        self._batch_counter = 0
        self._cuda_metrics.clear()
        self._batch_metrics.clear()
        self._memory_snapshots.clear()

        # Reset memory statistics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.empty_cache()

        # Start monitoring thread for NVML metrics
        if self.enable_nvml:
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name='GPUMonitor',
                daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Hardware monitoring thread started")

        # Initialize torch profiler
        if self.enable_torch_profiler and profile_tensorboard:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

            schedule = torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=repeat
            )

            tensorboard_dir = self.log_dir / 'tensorboard' / self.session_id
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

            self._torch_profiler = profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=tensorboard_trace_handler(str(tensorboard_dir)),
                record_shapes=True,
                profile_memory=trace_memory,
                with_stack=False,  # Disable to reduce overhead
                with_flops=True
            )
            self._torch_profiler.start()
            self.logger.info(f"Torch profiler started, tensorboard dir: {tensorboard_dir}")

        self.logger.info("GPU profiling started")

    def stop_profiling(self):
        """Stop GPU profiling session."""
        if not self._profiling:
            return

        self._profiling = False
        self.session_end_time = time.time()

        # Stop monitoring thread
        if self.enable_nvml and self._monitoring_thread:
            self._stop_event.set()
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None

        # Stop torch profiler
        if self._torch_profiler is not None:
            self._torch_profiler.stop()
            self._torch_profiler = None
            self.logger.info("Torch profiler stopped")

        self.logger.info("GPU profiling stopped")

    def _monitoring_loop(self):
        """Background thread for continuous hardware metrics collection."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_cuda_metrics()
                if metrics:
                    self._cuda_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error collecting CUDA metrics: {e}")

            time.sleep(self.sampling_interval)

    def _collect_cuda_metrics(self) -> Optional[CUDAMetrics]:
        """Collect current CUDA metrics via NVML."""
        if not self._gpu_handle:
            return None

        try:
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            gpu_util = util.gpu
            mem_util = util.memory

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)

            # Clocks
            sm_clock = pynvml.nvmlDeviceGetClockInfo(
                self._gpu_handle, pynvml.NVML_CLOCK_SM
            )
            mem_clock = pynvml.nvmlDeviceGetClockInfo(
                self._gpu_handle, pynvml.NVML_CLOCK_MEM
            )

            # Power
            power_draw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._gpu_handle) / 1000.0

            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(
                self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(self._gpu_handle)
            except:
                fan_speed = 0

            # PCIe throughput
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    self._gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    self._gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                pcie_throughput = (pcie_tx + pcie_rx) / 1024.0  # KB/s to MB/s
            except:
                pcie_throughput = 0.0

            # Compute mode
            try:
                compute_mode = pynvml.nvmlDeviceGetComputeMode(self._gpu_handle)
                compute_mode_str = {
                    0: 'DEFAULT',
                    1: 'EXCLUSIVE_THREAD',
                    2: 'PROHIBITED',
                    3: 'EXCLUSIVE_PROCESS'
                }.get(compute_mode, 'UNKNOWN')
            except:
                compute_mode_str = 'UNKNOWN'

            # Persistence mode
            try:
                persistence = pynvml.nvmlDeviceGetPersistenceMode(self._gpu_handle)
                persistence_enabled = persistence == 1
            except:
                persistence_enabled = False

            return CUDAMetrics(
                timestamp=time.time(),
                gpu_utilization=gpu_util,
                memory_used_mb=mem_used_mb,
                memory_total_mb=mem_total_mb,
                memory_utilization=mem_util,
                sm_clock_mhz=sm_clock,
                memory_clock_mhz=mem_clock,
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                temperature_c=temperature,
                fan_speed_percent=fan_speed,
                pcie_throughput_mbps=pcie_throughput,
                compute_mode=compute_mode_str,
                persistence_mode=persistence_enabled
            )

        except Exception as e:
            self.logger.debug(f"Error in _collect_cuda_metrics: {e}")
            return None

    @contextmanager
    def profile_batch(self, batch_size: int, queue_depth: Optional[int] = None):
        """Context manager for profiling a single inference batch.

        Usage:
            with profiler.profile_batch(batch_size=64):
                output = model(input_tensor)

        Args:
            batch_size: Number of samples in batch
            queue_depth: Optional queue depth for queue metrics
        """
        if not self._profiling:
            yield
            return

        batch_id = self._batch_counter
        self._batch_counter += 1

        # Start timing
        batch_start = time.time()

        # Record start event
        start_event, end_event = self._cuda_events['batch_total']
        start_event.record()

        # Capture GPU state at batch start
        start_metrics = self._collect_cuda_metrics()

        try:
            yield
        finally:
            # Record end event
            end_event.record()
            torch.cuda.synchronize()

            # Calculate timing
            total_time_ms = start_event.elapsed_time(end_event)

            # Capture GPU state at batch end
            end_metrics = self._collect_cuda_metrics()

            # Create batch metrics
            if start_metrics and end_metrics:
                # Average GPU state during batch
                avg_gpu_util = (start_metrics.gpu_utilization + end_metrics.gpu_utilization) / 2
                avg_mem_used = (start_metrics.memory_used_mb + end_metrics.memory_used_mb) / 2
                avg_clock = (start_metrics.sm_clock_mhz + end_metrics.sm_clock_mhz) / 2
                avg_power = (start_metrics.power_draw_watts + end_metrics.power_draw_watts) / 2
            else:
                avg_gpu_util = 0.0
                avg_mem_used = 0.0
                avg_clock = 0
                avg_power = 0.0

            # Calculate throughput
            samples_per_sec = (batch_size * 1000.0) / total_time_ms if total_time_ms > 0 else 0.0

            batch_metrics = InferenceBatchMetrics(
                batch_id=batch_id,
                batch_size=batch_size,
                timestamp=batch_start,
                total_time_ms=total_time_ms,
                h2d_transfer_ms=0.0,  # Will be filled by manual timing
                inference_ms=total_time_ms,  # Approximate if no breakdown
                d2h_transfer_ms=0.0,
                gpu_utilization=avg_gpu_util,
                memory_used_mb=avg_mem_used,
                sm_clock_mhz=avg_clock,
                power_draw_watts=avg_power,
                samples_per_second=samples_per_sec,
                queue_depth=queue_depth
            )

            self._batch_metrics.append(batch_metrics)

            # Capture memory snapshot
            if self.memory_profiling and batch_id % 10 == 0:
                self._capture_memory_snapshot()

            # Step torch profiler
            if self._torch_profiler is not None:
                self._torch_profiler.step()

    @contextmanager
    def profile_transfer(self, direction: str):
        """Profile H2D or D2H transfer timing.

        Args:
            direction: 'h2d' or 'd2h'
        """
        if not self._profiling:
            yield
            return

        event_name = f"{direction}_transfer"
        if event_name not in self._cuda_events:
            yield
            return

        start_event, end_event = self._cuda_events[event_name]
        start_event.record()

        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)

            # Store in most recent batch metrics
            if self._batch_metrics:
                if direction == 'h2d':
                    self._batch_metrics[-1].h2d_transfer_ms = elapsed_ms
                elif direction == 'd2h':
                    self._batch_metrics[-1].d2h_transfer_ms = elapsed_ms

    def _capture_memory_snapshot(self):
        """Capture current GPU memory state."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(self.device_id) / (1024 ** 2)

        # Memory fragmentation: ratio of reserved but not allocated
        fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0.0

        snapshot = MemoryMetrics(
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
            num_allocations=0,  # Would require CUDA memory profiler
            num_deallocations=0,
            allocation_rate=0.0,
            deallocation_rate=0.0,
            fragmentation_ratio=fragmentation
        )

        self._memory_snapshots.append(snapshot)

    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics.

        Returns:
            Dictionary with current GPU state
        """
        metrics = {}

        # CUDA metrics
        cuda_metrics = self._collect_cuda_metrics()
        if cuda_metrics:
            metrics['cuda'] = asdict(cuda_metrics)

        # PyTorch memory
        if torch.cuda.is_available():
            metrics['memory'] = {
                'allocated_mb': torch.cuda.memory_allocated(self.device_id) / (1024 ** 2),
                'reserved_mb': torch.cuda.memory_reserved(self.device_id) / (1024 ** 2),
                'max_allocated_mb': torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
            }

        # Recent batch metrics
        if self._batch_metrics:
            recent = self._batch_metrics[-10:]
            metrics['recent_batches'] = {
                'count': len(recent),
                'avg_time_ms': statistics.mean(b.total_time_ms for b in recent),
                'avg_throughput': statistics.mean(b.samples_per_second for b in recent)
            }

        return metrics

    def generate_report(self) -> ProfilingSession:
        """Generate comprehensive profiling report.

        Returns:
            ProfilingSession with complete statistics
        """
        if self.session_start_time is None:
            raise RuntimeError("No profiling session to report")

        end_time = self.session_end_time or time.time()
        duration = end_time - self.session_start_time

        # Device info
        gpu_name = torch.cuda.get_device_name(self.device_id)
        compute_capability = torch.cuda.get_device_capability(self.device_id)
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory / (1024 ** 2)

        # Batch statistics
        total_batches = len(self._batch_metrics)
        total_samples = sum(b.batch_size for b in self._batch_metrics)
        avg_batch_size = total_samples / total_batches if total_batches > 0 else 0.0

        # Timing breakdown
        if self._batch_metrics:
            avg_h2d = statistics.mean(b.h2d_transfer_ms for b in self._batch_metrics)
            avg_inference = statistics.mean(b.inference_ms for b in self._batch_metrics)
            avg_d2h = statistics.mean(b.d2h_transfer_ms for b in self._batch_metrics)
            total_time = sum(b.total_time_ms for b in self._batch_metrics)
        else:
            avg_h2d = avg_inference = avg_d2h = total_time = 0.0

        # Throughput
        avg_throughput = total_samples / duration if duration > 0 else 0.0

        # GPU utilization from CUDA metrics
        if self._cuda_metrics:
            gpu_utils = [m.gpu_utilization for m in self._cuda_metrics]
            avg_gpu_util = statistics.mean(gpu_utils)
            p50_gpu_util = statistics.median(gpu_utils)
            p95_gpu_util = np.percentile(gpu_utils, 95) if len(gpu_utils) > 1 else avg_gpu_util
        else:
            avg_gpu_util = p50_gpu_util = p95_gpu_util = 0.0

        # Memory
        if self._cuda_metrics:
            mem_used = [m.memory_used_mb for m in self._cuda_metrics]
            avg_memory = statistics.mean(mem_used)
            max_memory = max(mem_used)
        else:
            avg_memory = max_memory = 0.0

        memory_efficiency = avg_memory / total_memory if total_memory > 0 else 0.0

        # Power
        if self._cuda_metrics:
            power_samples = [m.power_draw_watts for m in self._cuda_metrics]
            avg_power = statistics.mean(power_samples)
            # Energy = Power * Time (Joules = Watts * seconds)
            total_energy = avg_power * duration
        else:
            avg_power = total_energy = 0.0

        session = ProfilingSession(
            session_id=self.session_id,
            device=str(self.device),
            start_time=self.session_start_time,
            end_time=end_time,
            duration_seconds=duration,
            gpu_name=gpu_name,
            compute_capability=compute_capability,
            total_memory_mb=total_memory,
            total_batches=total_batches,
            total_samples=total_samples,
            avg_batch_size=avg_batch_size,
            avg_throughput=avg_throughput,
            avg_h2d_transfer_ms=avg_h2d,
            avg_inference_ms=avg_inference,
            avg_d2h_transfer_ms=avg_d2h,
            total_time_ms=total_time,
            avg_gpu_utilization=avg_gpu_util,
            p50_gpu_utilization=p50_gpu_util,
            p95_gpu_utilization=p95_gpu_util,
            avg_memory_used_mb=avg_memory,
            max_memory_used_mb=max_memory,
            memory_efficiency=memory_efficiency,
            avg_power_watts=avg_power,
            total_energy_joules=total_energy,
            batch_metrics=self._batch_metrics
        )

        return session

    def export_report(self, filename: Optional[str] = None) -> Path:
        """Export profiling report to JSON file.

        Args:
            filename: Optional filename, defaults to session_id.json

        Returns:
            Path to exported file
        """
        report = self.generate_report()

        if filename is None:
            filename = f"{self.session_id}_report.json"

        # Ensure log_dir is a Path object
        log_dir = Path(self.log_dir) if not isinstance(self.log_dir, Path) else self.log_dir
        filepath = log_dir / filename

        # Convert to JSON-serializable format
        report_dict = asdict(report)

        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)

        self.logger.info(f"Report exported to {filepath}")
        return filepath

    def print_summary(self):
        """Print human-readable profiling summary."""
        report = self.generate_report()

        print("\n" + "="*80)
        print(f"GPU Profiling Summary - {report.session_id}")
        print("="*80)
        print(f"\nDevice: {report.gpu_name} (compute {report.compute_capability[0]}.{report.compute_capability[1]})")
        print(f"Memory: {report.total_memory_mb:.0f} MB")
        print(f"Duration: {report.duration_seconds:.2f}s")

        print(f"\n{'Workload Statistics':-^80}")
        print(f"  Total batches:    {report.total_batches}")
        print(f"  Total samples:    {report.total_samples}")
        print(f"  Avg batch size:   {report.avg_batch_size:.1f}")
        print(f"  Avg throughput:   {report.avg_throughput:.1f} samples/sec")

        print(f"\n{'Timing Breakdown':-^80}")
        print(f"  H2D transfer:     {report.avg_h2d_transfer_ms:.3f} ms")
        print(f"  Inference:        {report.avg_inference_ms:.3f} ms")
        print(f"  D2H transfer:     {report.avg_d2h_transfer_ms:.3f} ms")
        print(f"  Total per batch:  {report.avg_h2d_transfer_ms + report.avg_inference_ms + report.avg_d2h_transfer_ms:.3f} ms")

        print(f"\n{'GPU Utilization':-^80}")
        print(f"  Average:          {report.avg_gpu_utilization:.1f}%")
        print(f"  Median (P50):     {report.p50_gpu_utilization:.1f}%")
        print(f"  P95:              {report.p95_gpu_utilization:.1f}%")

        # Status vs targets
        target_util = 80.0
        util_status = "✓" if report.avg_gpu_utilization >= target_util else "✗"
        print(f"  Target (80%):     {util_status} {'PASS' if report.avg_gpu_utilization >= target_util else 'FAIL'}")

        print(f"\n{'Memory Usage':-^80}")
        print(f"  Average:          {report.avg_memory_used_mb:.0f} MB ({report.memory_efficiency*100:.1f}%)")
        print(f"  Peak:             {report.max_memory_used_mb:.0f} MB")
        print(f"  Total available:  {report.total_memory_mb:.0f} MB")

        print(f"\n{'Power & Energy':-^80}")
        print(f"  Average power:    {report.avg_power_watts:.1f} W")
        print(f"  Total energy:     {report.total_energy_joules:.1f} J ({report.total_energy_joules/3600:.4f} Wh)")

        print("="*80 + "\n")

    def __enter__(self):
        """Context manager entry."""
        self.start_profiling()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_profiling()
        if exc_type is None:
            self.print_summary()


def profile_inference_pipeline(
    model: torch.nn.Module,
    dataloader: Any,
    num_batches: int = 100,
    device: str = 'cuda:0',
    log_dir: str = 'runs/gpu_profiling',
    export_tensorboard: bool = True
) -> ProfilingSession:
    """Profile a complete inference pipeline.

    Convenience function for profiling model inference over a dataloader.

    Args:
        model: PyTorch model to profile
        dataloader: Data loader providing input batches
        num_batches: Number of batches to profile
        device: CUDA device
        log_dir: Output directory for profiling data
        export_tensorboard: Export to TensorBoard format

    Returns:
        ProfilingSession with complete results

    Example:
        >>> model = load_model()
        >>> dataloader = create_dataloader()
        >>> session = profile_inference_pipeline(model, dataloader, num_batches=100)
        >>> print(f"Avg throughput: {session.avg_throughput:.1f} samples/sec")
    """
    profiler = GPUProfiler(
        device=device,
        log_dir=log_dir,
        tensorboard_export=export_tensorboard
    )

    model = model.to(device)
    model.eval()

    profiler.start_profiling(profile_tensorboard=export_tensorboard)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Move batch to device
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            batch_size = inputs.size(0)

            with profiler.profile_transfer('h2d'):
                inputs_gpu = inputs.to(device, non_blocking=True)

            with profiler.profile_batch(batch_size=batch_size):
                with profiler.profile_transfer('inference'):
                    outputs = model(inputs_gpu)

                with profiler.profile_transfer('d2h'):
                    _ = outputs.cpu()

    profiler.stop_profiling()
    profiler.print_summary()
    profiler.export_report()

    return profiler.generate_report()
