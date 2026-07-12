"""
Profiling Session Manager
==========================

Unified profiling session that coordinates all profilers:
- GILProfiler
- InferencePipelineProfiler
- ThreadCoordinatorProfiler
- MemoryProfiler

Provides single interface for comprehensive system profiling
with integration to C++ instrumentation.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from .gil_profiler import GILProfiler
from .inference_profiler import InferencePipelineProfiler
from .thread_profiler import ThreadCoordinatorProfiler
from .memory_profiler import MemoryProfiler

try:
    import mcts_py
    HAS_MCTS_PY = True
except ImportError:
    HAS_MCTS_PY = False


@dataclass
class ProfilerConfig:
    """Configuration for profiling session."""

    # GIL profiler config
    enable_gil_profiling: bool = True
    gil_sample_rate: float = 0.001  # 1ms
    gil_track_hotspots: bool = True

    # Inference profiler config
    enable_inference_profiling: bool = True
    inference_detailed_tracking: bool = True

    # Thread profiler config
    enable_thread_profiling: bool = True
    thread_track_lifecycle: bool = True

    # Memory profiler config
    enable_memory_profiling: bool = True
    memory_snapshot_interval: float = 1.0  # 1 second
    memory_enable_tracemalloc: bool = True
    memory_track_gc: bool = True

    # C++ instrumentation config
    enable_cpp_instrumentation: bool = True

    # Output config
    auto_save_reports: bool = True
    report_directory: str = "profiling_reports"


class ProfilingSession:
    """
    Unified profiling session manager.

    Coordinates all profilers and provides single interface for:
    - Starting/stopping all profilers
    - Collecting metrics from all sources
    - Integrating C++ instrumentation
    - Generating reports

    Usage:
        config = ProfilerConfig()
        session = ProfilingSession(config)

        with session:
            # ... run workload ...

        metrics = session.get_all_metrics()
        session.save_reports()

    Or explicit start/stop:
        session = ProfilingSession(config)
        session.start()
        # ... run workload ...
        session.stop()
        metrics = session.get_all_metrics()
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize profiling session.

        Args:
            config: Profiler configuration (uses defaults if None)
        """
        self.config = config or ProfilerConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize profilers based on config
        self.gil_profiler: Optional[GILProfiler] = None
        self.inference_profiler: Optional[InferencePipelineProfiler] = None
        self.thread_profiler: Optional[ThreadCoordinatorProfiler] = None
        self.memory_profiler: Optional[MemoryProfiler] = None

        if self.config.enable_gil_profiling:
            self.gil_profiler = GILProfiler(
                sample_rate=self.config.gil_sample_rate,
                track_hotspots=self.config.gil_track_hotspots
            )

        if self.config.enable_inference_profiling:
            self.inference_profiler = InferencePipelineProfiler(
                enable_detailed_tracking=self.config.inference_detailed_tracking
            )

        if self.config.enable_thread_profiling:
            self.thread_profiler = ThreadCoordinatorProfiler(
                track_thread_lifecycle=self.config.thread_track_lifecycle
            )

        if self.config.enable_memory_profiling:
            self.memory_profiler = MemoryProfiler(
                snapshot_interval=self.config.memory_snapshot_interval,
                enable_tracemalloc=self.config.memory_enable_tracemalloc,
                track_gc_events=self.config.memory_track_gc
            )

        # Session state
        self._running = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._session_id = f"profile_{int(time.time())}"

        # C++ instrumentation state
        self._cpp_instrumentation_enabled = False

    def start(self):
        """Start all enabled profilers."""
        if self._running:
            self.logger.warning("Profiling session already running")
            return

        self._running = True
        self._start_time = time.perf_counter()

        self.logger.info(f"Starting profiling session: {self._session_id}")

        # Enable C++ instrumentation
        if self.config.enable_cpp_instrumentation and HAS_MCTS_PY:
            if hasattr(mcts_py, 'set_instrumentation_enabled'):
                mcts_py.set_instrumentation_enabled(True)
                if hasattr(mcts_py, 'reset_instrumentation_metrics'):
                    mcts_py.reset_instrumentation_metrics()
                self._cpp_instrumentation_enabled = True
                self.logger.info("C++ instrumentation enabled")

        # Start profilers
        if self.gil_profiler:
            self.gil_profiler.start()
            self.logger.info("GIL profiler started")

        if self.inference_profiler:
            self.inference_profiler.start()
            self.logger.info("Inference profiler started")

        if self.thread_profiler:
            self.thread_profiler.start()
            self.logger.info("Thread profiler started")

        if self.memory_profiler:
            self.memory_profiler.start()
            self.logger.info("Memory profiler started")

        self.logger.info("All profilers started successfully")

    def stop(self):
        """Stop all profilers and collect final metrics."""
        if not self._running:
            return

        self.logger.info("Stopping profiling session")

        # Stop profilers in reverse order
        if self.memory_profiler:
            self.memory_profiler.stop()

        if self.thread_profiler:
            self.thread_profiler.stop()

        if self.inference_profiler:
            self.inference_profiler.stop()

        if self.gil_profiler:
            self.gil_profiler.stop()

        # Disable C++ instrumentation
        if self._cpp_instrumentation_enabled and HAS_MCTS_PY:
            if hasattr(mcts_py, 'set_instrumentation_enabled'):
                mcts_py.set_instrumentation_enabled(False)
            self._cpp_instrumentation_enabled = False

        self._running = False
        self._stop_time = time.perf_counter()

        self.logger.info("All profilers stopped")

        # Auto-save reports if configured
        if self.config.auto_save_reports:
            try:
                self.save_reports()
            except Exception as e:
                self.logger.error(f"Failed to auto-save reports: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from all profilers.

        Returns:
            Dictionary with metrics from all sources:
                - gil_metrics: GIL profiling data
                - inference_metrics: Inference pipeline data
                - thread_metrics: Thread coordination data
                - memory_metrics: Memory and GC data
                - cpp_instrumentation: C++ metrics
                - summary: Overall session summary
        """
        metrics = {
            'session_id': self._session_id,
            'config': {
                'gil_profiling': self.config.enable_gil_profiling,
                'inference_profiling': self.config.enable_inference_profiling,
                'thread_profiling': self.config.enable_thread_profiling,
                'memory_profiling': self.config.enable_memory_profiling,
                'cpp_instrumentation': self.config.enable_cpp_instrumentation,
            }
        }

        # Collect from each profiler
        if self.gil_profiler:
            metrics['gil_metrics'] = self.gil_profiler.get_metrics()

        if self.inference_profiler:
            metrics['inference_metrics'] = self.inference_profiler.get_metrics()

        if self.thread_profiler:
            metrics['thread_metrics'] = self.thread_profiler.get_metrics()

        if self.memory_profiler:
            metrics['memory_metrics'] = self.memory_profiler.get_metrics()

        # Collect C++ instrumentation
        if self._cpp_instrumentation_enabled and HAS_MCTS_PY:
            if hasattr(mcts_py, 'get_instrumentation_snapshot'):
                metrics['cpp_instrumentation'] = mcts_py.get_instrumentation_snapshot()

        # Add session summary
        if self._start_time and self._stop_time:
            metrics['summary'] = {
                'session_duration_seconds': self._stop_time - self._start_time,
                'start_time': self._start_time,
                'stop_time': self._stop_time,
            }

        return metrics

    def save_reports(self, directory: Optional[str] = None):
        """
        Save profiling reports to disk.

        Args:
            directory: Output directory (uses config default if None)
        """
        output_dir = Path(directory or self.config.report_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all metrics
        metrics = self.get_all_metrics()

        # Save JSON report
        json_path = output_dir / f"{self._session_id}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        self.logger.info(f"Saved JSON report: {json_path}")

        # Generate and save HTML report
        try:
            from .report_generator import generate_html_report
            html_path = output_dir / f"{self._session_id}_report.html"
            generate_html_report(metrics, html_path)
            self.logger.info(f"Saved HTML report: {html_path}")
        except Exception as e:
            self.logger.warning(f"Failed to generate HTML report: {e}")

        # Generate flamegraph if we have profile data
        try:
            from .report_generator import generate_flamegraph
            flamegraph_path = output_dir / f"{self._session_id}_flamegraph.svg"
            generate_flamegraph(metrics, flamegraph_path)
            self.logger.info(f"Saved flamegraph: {flamegraph_path}")
        except Exception as e:
            self.logger.warning(f"Failed to generate flamegraph: {e}")

        return {
            'json_report': str(json_path),
            'html_report': str(html_path) if 'html_path' in locals() else None,
            'flamegraph': str(flamegraph_path) if 'flamegraph_path' in locals() else None,
        }

    def reset(self):
        """Reset all profilers."""
        if self.gil_profiler:
            self.gil_profiler.reset()

        if self.inference_profiler:
            self.inference_profiler.reset()

        if self.thread_profiler:
            self.thread_profiler.reset()

        if self.memory_profiler:
            self.memory_profiler.reset()

        # Reset C++ instrumentation
        if HAS_MCTS_PY and hasattr(mcts_py, 'reset_instrumentation_metrics'):
            mcts_py.reset_instrumentation_metrics()

        self.logger.info("All profilers reset")

    # Convenience methods for accessing individual profilers

    @property
    def gil(self) -> Optional[GILProfiler]:
        """Access GIL profiler."""
        return self.gil_profiler

    @property
    def inference(self) -> Optional[InferencePipelineProfiler]:
        """Access inference profiler."""
        return self.inference_profiler

    @property
    def threads(self) -> Optional[ThreadCoordinatorProfiler]:
        """Access thread profiler."""
        return self.thread_profiler

    @property
    def memory(self) -> Optional[MemoryProfiler]:
        """Access memory profiler."""
        return self.memory_profiler
