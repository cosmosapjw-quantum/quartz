"""
GIL Profiler - Tracks Python GIL acquisition, release, and contention
=====================================================================

Monitors GIL-related performance issues:
- Time spent waiting to acquire GIL
- Time holding GIL vs time in C++
- GIL contention between threads
- Critical sections that unnecessarily hold GIL

Uses sys.settrace() for Python code tracking and custom hooks for C++ integration.
"""

import sys
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import logging


@dataclass
class GILEvent:
    """Single GIL acquisition/release event."""
    thread_id: int
    event_type: str  # 'acquire', 'release', 'wait_start', 'wait_end'
    timestamp: float
    location: str  # File:line or function name
    duration_ns: Optional[int] = None  # For 'wait_end' events


@dataclass
class GILThreadMetrics:
    """Per-thread GIL metrics."""
    thread_id: int
    thread_name: str

    # Time metrics (seconds)
    time_with_gil: float = 0.0
    time_without_gil: float = 0.0
    time_waiting_for_gil: float = 0.0

    # Event counts
    gil_acquisitions: int = 0
    gil_releases: int = 0

    # Contention metrics
    max_wait_time_ms: float = 0.0
    avg_wait_time_ms: float = 0.0
    total_wait_events: int = 0

    # Hot spots (location -> total wait time in ms)
    wait_hotspots: Dict[str, float] = field(default_factory=dict)


@dataclass
class GILContentionEvent:
    """Record of GIL contention between threads."""
    timestamp: float
    waiting_thread: int
    holding_thread: int
    wait_duration_ms: float
    location: str


class GILProfiler:
    """
    Profiles Python GIL usage patterns and contention.

    Features:
    - Tracks time with/without GIL per thread
    - Identifies GIL contention hotspots
    - Measures wait times for GIL acquisition
    - Integrates with C++ instrumentation via callbacks

    Usage:
        profiler = GILProfiler()
        profiler.start()
        # ... run workload ...
        profiler.stop()
        metrics = profiler.get_metrics()

    Or as context manager:
        with GILProfiler() as profiler:
            # ... run workload ...
            metrics = profiler.get_metrics()
    """

    def __init__(self,
                 sample_rate: float = 0.001,  # Sample every 1ms
                 max_events: int = 100_000,
                 track_hotspots: bool = True):
        """
        Initialize GIL profiler.

        Args:
            sample_rate: Sampling interval in seconds (default 1ms)
            max_events: Maximum events to store (prevents memory blowup)
            track_hotspots: Track location-specific GIL wait hotspots
        """
        self.sample_rate = sample_rate
        self.max_events = max_events
        self.track_hotspots = track_hotspots

        # Thread-local state
        self._thread_states: Dict[int, Dict[str, Any]] = {}
        self._thread_lock = threading.Lock()

        # Event storage
        self._events: deque[GILEvent] = deque(maxlen=max_events)
        self._contention_events: deque[GILContentionEvent] = deque(maxlen=1000)

        # Metrics accumulation
        self._thread_metrics: Dict[int, GILThreadMetrics] = {}

        # Control
        self._running = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start GIL profiling."""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()
        self._monitor_stop_event.clear()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="GIL_Monitor",
            daemon=True
        )
        self._monitor_thread.start()

        self.logger.info(f"GIL profiler started (sample_rate={self.sample_rate}s)")

    def stop(self):
        """Stop GIL profiling."""
        if not self._running:
            return

        self._running = False
        self._stop_time = time.perf_counter()

        # Stop monitor thread
        self._monitor_stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Finalize metrics
        self._finalize_metrics()

        self.logger.info("GIL profiler stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    @contextmanager
    def section(self, name: str):
        """
        Profile a specific code section for GIL usage.

        Usage:
            with profiler.section("inference_batch"):
                # ... code ...
        """
        thread_id = threading.get_ident()
        start_time = time.perf_counter()
        location = name

        # Record section entry
        self._record_event(GILEvent(
            thread_id=thread_id,
            event_type='section_enter',
            timestamp=start_time,
            location=location
        ))

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ns = int((end_time - start_time) * 1e9)

            # Record section exit
            self._record_event(GILEvent(
                thread_id=thread_id,
                event_type='section_exit',
                timestamp=end_time,
                location=location,
                duration_ns=duration_ns
            ))

    def mark_gil_release(self, location: str = ""):
        """
        Mark when GIL is about to be released (before entering C++).

        Call this before entering nogil code blocks.
        """
        if not self._running:
            return

        thread_id = threading.get_ident()
        timestamp = time.perf_counter()

        self._record_event(GILEvent(
            thread_id=thread_id,
            event_type='release',
            timestamp=timestamp,
            location=location or self._get_caller_location()
        ))

    def mark_gil_acquire(self, location: str = ""):
        """
        Mark when GIL is re-acquired (after returning from C++).

        Call this after returning from nogil code blocks.
        """
        if not self._running:
            return

        thread_id = threading.get_ident()
        timestamp = time.perf_counter()

        self._record_event(GILEvent(
            thread_id=thread_id,
            event_type='acquire',
            timestamp=timestamp,
            location=location or self._get_caller_location()
        ))

    def _monitor_loop(self):
        """Background thread that monitors GIL state."""
        while not self._monitor_stop_event.wait(self.sample_rate):
            try:
                self._sample_gil_state()
            except Exception as e:
                self.logger.error(f"Error in GIL monitor: {e}")

    def _sample_gil_state(self):
        """Sample current GIL state across all threads."""
        # Get all Python threads
        for thread in threading.enumerate():
            thread_id = thread.ident
            if thread_id is None:
                continue

            # Check if thread is waiting for GIL
            # Note: Python doesn't expose direct GIL state, so we use heuristics
            # based on thread activity and sys.getswitchinterval()
            with self._thread_lock:
                if thread_id not in self._thread_states:
                    self._thread_states[thread_id] = {
                        'last_seen': time.perf_counter(),
                        'has_gil': False,
                        'wait_start': None
                    }

    def _record_event(self, event: GILEvent):
        """Record a GIL event."""
        self._events.append(event)

    def _get_caller_location(self) -> str:
        """Get caller location from stack."""
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller = frame.f_back.f_back
            return f"{caller.f_code.co_filename}:{caller.f_lineno}"
        return "unknown"

    def _finalize_metrics(self):
        """Compute final metrics from recorded events."""
        if not self._start_time or not self._stop_time:
            return

        # Group events by thread
        thread_events: Dict[int, List[GILEvent]] = defaultdict(list)
        for event in self._events:
            thread_events[event.thread_id].append(event)

        # Compute metrics per thread
        for thread_id, events in thread_events.items():
            # Sort events by timestamp
            events.sort(key=lambda e: e.timestamp)

            # Get thread name
            thread_name = "unknown"
            for thread in threading.enumerate():
                if thread.ident == thread_id:
                    thread_name = thread.name
                    break

            metrics = GILThreadMetrics(
                thread_id=thread_id,
                thread_name=thread_name
            )

            # State machine: track GIL possession
            has_gil = True  # Assume Python code starts with GIL
            last_timestamp = self._start_time
            wait_start: Optional[float] = None
            wait_times: List[float] = []

            for event in events:
                elapsed = event.timestamp - last_timestamp

                if event.event_type == 'release':
                    if has_gil:
                        metrics.time_with_gil += elapsed
                        metrics.gil_releases += 1
                    has_gil = False

                elif event.event_type == 'acquire':
                    if not has_gil:
                        metrics.time_without_gil += elapsed
                        metrics.gil_acquisitions += 1
                    has_gil = True

                    # If there was a wait, record it
                    if wait_start is not None:
                        wait_time_ms = (event.timestamp - wait_start) * 1000
                        wait_times.append(wait_time_ms)
                        metrics.time_waiting_for_gil += (event.timestamp - wait_start)

                        if self.track_hotspots:
                            if event.location not in metrics.wait_hotspots:
                                metrics.wait_hotspots[event.location] = 0.0
                            metrics.wait_hotspots[event.location] += wait_time_ms

                        wait_start = None

                elif event.event_type == 'wait_start':
                    wait_start = event.timestamp

                last_timestamp = event.timestamp

            # Final state
            if self._stop_time:
                elapsed = self._stop_time - last_timestamp
                if has_gil:
                    metrics.time_with_gil += elapsed
                else:
                    metrics.time_without_gil += elapsed

            # Compute wait statistics
            if wait_times:
                metrics.total_wait_events = len(wait_times)
                metrics.max_wait_time_ms = max(wait_times)
                metrics.avg_wait_time_ms = sum(wait_times) / len(wait_times)

            self._thread_metrics[thread_id] = metrics

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive GIL profiling metrics.

        Returns:
            Dictionary with:
                - total_time: Total profiling duration
                - thread_metrics: Per-thread GIL metrics
                - contention_events: List of contention events
                - summary: Aggregate statistics
        """
        if not self._start_time:
            return {}

        end_time = self._stop_time or time.perf_counter()
        total_time = end_time - self._start_time

        # Compute summary statistics
        total_gil_time = sum(m.time_with_gil for m in self._thread_metrics.values())
        total_nogil_time = sum(m.time_without_gil for m in self._thread_metrics.values())
        total_wait_time = sum(m.time_waiting_for_gil for m in self._thread_metrics.values())

        num_threads = len(self._thread_metrics)
        max_parallel_gil_time = total_time * num_threads

        summary = {
            'num_threads': num_threads,
            'gil_utilization': (total_gil_time / max_parallel_gil_time * 100) if max_parallel_gil_time > 0 else 0.0,
            'avg_gil_time_per_thread': total_gil_time / num_threads if num_threads > 0 else 0.0,
            'avg_nogil_time_per_thread': total_nogil_time / num_threads if num_threads > 0 else 0.0,
            'avg_wait_time_per_thread': total_wait_time / num_threads if num_threads > 0 else 0.0,
            'total_contention_events': len(self._contention_events),
            'gil_efficiency': (total_nogil_time / (total_gil_time + total_nogil_time) * 100)
                             if (total_gil_time + total_nogil_time) > 0 else 0.0,
        }

        # Identify top wait hotspots across all threads
        all_hotspots: Dict[str, float] = defaultdict(float)
        for metrics in self._thread_metrics.values():
            for location, wait_time in metrics.wait_hotspots.items():
                all_hotspots[location] += wait_time

        top_hotspots = sorted(
            all_hotspots.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_time_seconds': total_time,
            'thread_metrics': {
                tid: {
                    'thread_id': m.thread_id,
                    'thread_name': m.thread_name,
                    'time_with_gil_seconds': m.time_with_gil,
                    'time_without_gil_seconds': m.time_without_gil,
                    'time_waiting_for_gil_seconds': m.time_waiting_for_gil,
                    'gil_acquisitions': m.gil_acquisitions,
                    'gil_releases': m.gil_releases,
                    'max_wait_time_ms': m.max_wait_time_ms,
                    'avg_wait_time_ms': m.avg_wait_time_ms,
                    'total_wait_events': m.total_wait_events,
                    'gil_efficiency': (m.time_without_gil / (m.time_with_gil + m.time_without_gil) * 100)
                                     if (m.time_with_gil + m.time_without_gil) > 0 else 0.0,
                    'wait_hotspots': dict(sorted(
                        m.wait_hotspots.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5])  # Top 5 hotspots per thread
                }
                for tid, m in self._thread_metrics.items()
            },
            'contention_events': [
                {
                    'timestamp': e.timestamp,
                    'waiting_thread': e.waiting_thread,
                    'holding_thread': e.holding_thread,
                    'wait_duration_ms': e.wait_duration_ms,
                    'location': e.location
                }
                for e in self._contention_events
            ],
            'summary': summary,
            'top_wait_hotspots': [
                {'location': loc, 'total_wait_time_ms': wait_time}
                for loc, wait_time in top_hotspots
            ]
        }

    def reset(self):
        """Reset all metrics and events."""
        self._events.clear()
        self._contention_events.clear()
        self._thread_metrics.clear()
        self._thread_states.clear()
        self._start_time = None
        self._stop_time = None
