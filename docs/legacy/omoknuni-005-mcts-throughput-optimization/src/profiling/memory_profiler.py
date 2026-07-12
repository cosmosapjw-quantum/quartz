"""
Memory Profiler
===============

Profiles memory allocation patterns and garbage collection impact:
- Object allocation tracking
- Reference counting overhead
- Garbage collection pauses
- Memory leak detection
- Peak memory usage

Integrates with tracemalloc and gc modules.
"""

import time
import gc
import sys
import threading
import tracemalloc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import logging


@dataclass
class AllocationSnapshot:
    """Snapshot of memory allocations at a point in time."""
    timestamp: float
    total_allocated_mb: float
    peak_allocated_mb: float
    num_allocations: int
    top_allocations: List[Tuple[str, float]]  # (location, size_mb)


@dataclass
class GCEvent:
    """Garbage collection event."""
    timestamp: float
    generation: int
    collected_objects: int
    uncollectable_objects: int
    duration_ms: float


@dataclass
class ObjectStats:
    """Statistics for a specific object type."""
    type_name: str
    count: int
    total_size_mb: float
    avg_size_bytes: float


class MemoryProfiler:
    """
    Profiles memory usage and garbage collection patterns.

    Tracks:
    - Memory allocation hotspots
    - Garbage collection frequency and impact
    - Reference counting overhead
    - Memory leaks (growing allocations)
    - Peak memory usage

    Usage:
        profiler = MemoryProfiler()
        profiler.start()

        with profiler.track_section("inference"):
            # ... code ...

        metrics = profiler.get_metrics()
    """

    def __init__(self,
                 snapshot_interval: float = 1.0,
                 enable_tracemalloc: bool = True,
                 track_gc_events: bool = True):
        """
        Initialize memory profiler.

        Args:
            snapshot_interval: How often to snapshot memory (seconds)
            enable_tracemalloc: Enable detailed allocation tracking
            track_gc_events: Monitor garbage collection events
        """
        self.snapshot_interval = snapshot_interval
        self.enable_tracemalloc = enable_tracemalloc
        self.track_gc_events = track_gc_events

        # Allocation tracking
        self._snapshots: deque[AllocationSnapshot] = deque(maxlen=10_000)
        self._snapshot_lock = threading.Lock()

        # GC event tracking
        self._gc_events: deque[GCEvent] = deque(maxlen=1000)
        self._gc_lock = threading.Lock()

        # Section tracking
        self._section_snapshots: Dict[str, List[AllocationSnapshot]] = defaultdict(list)
        self._active_sections: Dict[str, float] = {}

        # Baseline measurements
        self._baseline_memory: Optional[float] = None
        self._peak_memory: float = 0.0

        # Control
        self._running = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()

        # GC callbacks
        self._gc_callback_installed = False

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start memory profiling."""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()

        # Enable tracemalloc
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.logger.info("tracemalloc started")

        # Install GC callbacks
        if self.track_gc_events:
            gc.callbacks.append(self._gc_callback)
            self._gc_callback_installed = True

        # Record baseline
        self._baseline_memory = self._get_current_memory_mb()

        # Start monitoring thread
        self._monitor_stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="Memory_Monitor",
            daemon=True
        )
        self._monitor_thread.start()

        self.logger.info(f"Memory profiler started (baseline: {self._baseline_memory:.2f} MB)")

    def stop(self):
        """Stop memory profiling."""
        if not self._running:
            return

        self._running = False
        self._stop_time = time.perf_counter()

        # Stop monitor thread
        self._monitor_stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Remove GC callbacks
        if self._gc_callback_installed:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback_installed = False

        # Stop tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

        self.logger.info("Memory profiler stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    @contextmanager
    def track_section(self, section_name: str):
        """
        Track memory usage for a specific code section.

        Usage:
            with profiler.track_section("inference_batch"):
                # ... code ...
        """
        if not self._running:
            yield
            return

        # Take snapshot before
        start_time = time.perf_counter()
        start_memory = self._get_current_memory_mb()
        self._active_sections[section_name] = start_time

        try:
            yield
        finally:
            # Take snapshot after
            end_time = time.perf_counter()
            end_memory = self._get_current_memory_mb()
            delta_memory = end_memory - start_memory

            snapshot = AllocationSnapshot(
                timestamp=end_time,
                total_allocated_mb=end_memory,
                peak_allocated_mb=self._peak_memory,
                num_allocations=0,  # Will be populated if tracemalloc is enabled
                top_allocations=self._get_top_allocations(5) if self.enable_tracemalloc else []
            )

            self._section_snapshots[section_name].append(snapshot)
            self._active_sections.pop(section_name, None)

    def force_gc(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.

        Returns:
            Dictionary with GC statistics
        """
        start_time = time.perf_counter()

        # Get counts before GC
        before_counts = gc.get_count()

        # Run GC on all generations
        collected = gc.collect()

        # Get counts after GC
        after_counts = gc.get_count()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {
            'collected_objects': collected,
            'duration_ms': duration_ms,
            'before_counts': before_counts,
            'after_counts': after_counts
        }

    def _monitor_loop(self):
        """Background thread for memory monitoring."""
        while not self._monitor_stop_event.wait(self.snapshot_interval):
            try:
                self._take_snapshot()
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {e}")

    def _take_snapshot(self):
        """Take a memory snapshot."""
        current_memory = self._get_current_memory_mb()

        # Update peak
        if current_memory > self._peak_memory:
            self._peak_memory = current_memory

        # Get top allocations if tracemalloc enabled
        top_allocations = []
        num_allocations = 0
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            top_allocations = self._get_top_allocations(10)
            stats = tracemalloc.get_traced_memory()
            num_allocations = stats[0] // 1024  # Rough estimate

        snapshot = AllocationSnapshot(
            timestamp=time.perf_counter(),
            total_allocated_mb=current_memory,
            peak_allocated_mb=self._peak_memory,
            num_allocations=num_allocations,
            top_allocations=top_allocations
        )

        with self._snapshot_lock:
            self._snapshots.append(snapshot)

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        else:
            # Fallback to RSS via /proc/self/status on Linux
            try:
                with open('/proc/self/status') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            return int(line.split()[1]) / 1024  # Convert KB to MB
            except:
                pass
            return 0.0

    def _get_top_allocations(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top memory allocations by location."""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        result = []
        for stat in top_stats[:limit]:
            size_mb = stat.size / 1024 / 1024
            location = f"{stat.traceback.format()[0]}" if stat.traceback else "unknown"
            result.append((location, size_mb))

        return result

    def _gc_callback(self, phase: str, info: Dict[str, Any]):
        """Callback for garbage collection events."""
        if phase == 'stop':
            # GC just finished
            event = GCEvent(
                timestamp=time.perf_counter(),
                generation=info.get('generation', 0),
                collected_objects=info.get('collected', 0),
                uncollectable_objects=info.get('uncollectable', 0),
                duration_ms=0.0  # Duration not directly available in callback
            )

            with self._gc_lock:
                self._gc_events.append(event)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory profiling metrics.

        Returns:
            Dictionary with:
                - memory_timeline: Memory usage over time
                - gc_events: Garbage collection statistics
                - section_analysis: Per-section memory usage
                - leak_candidates: Potential memory leaks
                - summary: Aggregate statistics
        """
        if not self._start_time:
            return {}

        end_time = self._stop_time or time.perf_counter()
        total_time = end_time - self._start_time

        import numpy as np

        # Analyze memory growth
        memory_values = [s.total_allocated_mb for s in self._snapshots]
        memory_growth = (memory_values[-1] - memory_values[0]) if len(memory_values) >= 2 else 0.0

        # GC statistics
        gc_count = len(self._gc_events)
        gc_durations = [e.duration_ms for e in self._gc_events if e.duration_ms > 0]
        total_collected = sum(e.collected_objects for e in self._gc_events)

        summary = {
            'total_time_seconds': total_time,
            'baseline_memory_mb': self._baseline_memory or 0.0,
            'current_memory_mb': memory_values[-1] if memory_values else 0.0,
            'peak_memory_mb': self._peak_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_sec': memory_growth / total_time if total_time > 0 else 0.0,

            # GC statistics
            'total_gc_events': gc_count,
            'gc_events_per_second': gc_count / total_time if total_time > 0 else 0.0,
            'total_objects_collected': total_collected,
            'avg_gc_duration_ms': np.mean(gc_durations) if gc_durations else 0.0,
            'max_gc_duration_ms': np.max(gc_durations) if gc_durations else 0.0,
        }

        # Section analysis
        section_analysis = {}
        for section_name, snapshots in self._section_snapshots.items():
            if not snapshots:
                continue

            memory_deltas = []
            for i in range(1, len(snapshots)):
                delta = snapshots[i].total_allocated_mb - snapshots[i-1].total_allocated_mb
                memory_deltas.append(delta)

            section_analysis[section_name] = {
                'num_invocations': len(snapshots),
                'avg_memory_delta_mb': np.mean(memory_deltas) if memory_deltas else 0.0,
                'max_memory_delta_mb': np.max(memory_deltas) if memory_deltas else 0.0,
                'total_memory_delta_mb': np.sum(memory_deltas) if memory_deltas else 0.0,
            }

        # Identify potential leaks (consistent memory growth)
        leak_candidates = []
        if len(memory_values) >= 10:
            # Look for monotonic increasing trend
            window_size = 10
            for i in range(len(memory_values) - window_size):
                window = memory_values[i:i+window_size]
                if all(window[j] <= window[j+1] for j in range(window_size-1)):
                    growth = window[-1] - window[0]
                    if growth > 10.0:  # More than 10MB growth
                        leak_candidates.append({
                            'timestamp': self._snapshots[i].timestamp,
                            'growth_mb': growth,
                            'window_duration_sec': (self._snapshots[i+window_size].timestamp -
                                                   self._snapshots[i].timestamp)
                        })

        return {
            'summary': summary,
            'section_analysis': section_analysis,
            'memory_timeline': [
                {
                    'timestamp': s.timestamp,
                    'memory_mb': s.total_allocated_mb,
                    'peak_mb': s.peak_allocated_mb
                }
                for s in list(self._snapshots)[::max(1, len(self._snapshots) // 100)]  # Sample 100 points
            ],
            'gc_events': [
                {
                    'timestamp': e.timestamp,
                    'generation': e.generation,
                    'collected': e.collected_objects,
                    'duration_ms': e.duration_ms
                }
                for e in list(self._gc_events)[-50:]  # Last 50 GC events
            ],
            'leak_candidates': leak_candidates[:10],  # Top 10 leak candidates
            'top_allocations': self._get_top_allocations(20) if self.enable_tracemalloc else []
        }

    def reset(self):
        """Reset all metrics."""
        self._snapshots.clear()
        self._gc_events.clear()
        self._section_snapshots.clear()
        self._active_sections.clear()
        self._baseline_memory = None
        self._peak_memory = 0.0
        self._start_time = None
        self._stop_time = None
