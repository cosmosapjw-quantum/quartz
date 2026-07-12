"""
Thread Coordinator Profiler
============================

Profiles thread coordination overhead in the MCTS search system:
- ThreadPoolExecutor overhead
- Future creation and result collection
- Inter-thread communication costs
- Thread startup/shutdown overhead
- Thread affinity and NUMA effects

Integrates with SearchCoordinator and ThreadPoolExecutor.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import logging


@dataclass
class ThreadEventMetrics:
    """Metrics for a thread lifecycle event."""
    thread_id: int
    thread_name: str
    event_type: str  # 'created', 'started', 'completed', 'destroyed'
    timestamp: float
    duration_us: Optional[float] = None
    result: Optional[str] = None  # 'success', 'error', 'cancelled'


@dataclass
class FutureMetrics:
    """Metrics for a Future object."""
    future_id: str
    thread_id: int

    # Timing (microseconds)
    creation_time_us: float = 0.0
    submission_time_us: float = 0.0
    execution_time_us: float = 0.0
    result_collection_time_us: float = 0.0
    total_latency_us: float = 0.0

    # State
    completed_successfully: bool = False
    cancelled: bool = False


@dataclass
class ThreadPoolMetrics:
    """Metrics for ThreadPoolExecutor."""
    pool_size: int
    active_threads: int
    queued_tasks: int

    # Utilization
    thread_utilization: float = 0.0  # % of threads actively working
    queue_utilization: float = 0.0   # % of queue capacity used

    # Timing statistics (microseconds)
    avg_task_latency_us: float = 0.0
    p99_task_latency_us: float = 0.0


class ThreadCoordinatorProfiler:
    """
    Profiles thread coordination and communication overhead.

    Tracks:
    - Thread lifecycle events
    - Future creation and collection overhead
    - Thread pool utilization
    - Inter-thread communication latency
    - Thread startup/teardown costs

    Usage:
        profiler = ThreadCoordinatorProfiler()
        profiler.start()

        with profiler.track_future("task_123", thread_id=1):
            future = executor.submit(...)
            result = future.result()

        metrics = profiler.get_metrics()
    """

    def __init__(self,
                 max_samples: int = 10_000,
                 track_thread_lifecycle: bool = True):
        """
        Initialize thread profiler.

        Args:
            max_samples: Maximum event samples to store
            track_thread_lifecycle: Track thread creation/destruction events
        """
        self.max_samples = max_samples
        self.track_thread_lifecycle = track_thread_lifecycle

        # Event tracking
        self._thread_events: deque[ThreadEventMetrics] = deque(maxlen=max_samples)
        self._futures: deque[FutureMetrics] = deque(maxlen=max_samples)
        self._active_futures: Dict[str, Dict[str, Any]] = {}

        # Thread state tracking
        self._thread_states: Dict[int, Dict[str, Any]] = {}
        self._thread_lock = threading.Lock()

        # Pool monitoring
        self._pool_snapshots: deque[Tuple[float, ThreadPoolMetrics]] = deque(maxlen=1000)
        self._pool_lock = threading.Lock()

        # Control
        self._running = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start thread profiling."""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()
        self.logger.info("Thread coordinator profiler started")

    def stop(self):
        """Stop thread profiling."""
        if not self._running:
            return

        self._running = False
        self._stop_time = time.perf_counter()
        self.logger.info("Thread coordinator profiler stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    @contextmanager
    def track_future(self, future_id: str, thread_id: int):
        """
        Track a Future lifecycle from creation to result collection.

        Usage:
            with profiler.track_future("task_123", thread_id=1):
                future = executor.submit(work_fn)
                result = future.result()
        """
        if not self._running:
            yield
            return

        start_time = time.perf_counter()
        creation_time = start_time

        # Initialize future tracking
        with self._thread_lock:
            self._active_futures[future_id] = {
                'start_time': start_time,
                'thread_id': thread_id,
                'stages': {'creation': 0.0}
            }

        try:
            yield
        finally:
            end_time = time.perf_counter()
            total_latency_us = (end_time - start_time) * 1e6

            with self._thread_lock:
                if future_id in self._active_futures:
                    future_data = self._active_futures.pop(future_id)
                    stages = future_data['stages']

                    metrics = FutureMetrics(
                        future_id=future_id,
                        thread_id=thread_id,
                        creation_time_us=stages.get('creation', 0.0),
                        submission_time_us=stages.get('submission', 0.0),
                        execution_time_us=stages.get('execution', 0.0),
                        result_collection_time_us=stages.get('result_collection', 0.0),
                        total_latency_us=total_latency_us,
                        completed_successfully=stages.get('success', False),
                        cancelled=stages.get('cancelled', False)
                    )

                    self._futures.append(metrics)

    @contextmanager
    def track_future_stage(self, future_id: str, stage_name: str):
        """
        Track a specific stage of future processing.

        Usage:
            with profiler.track_future_stage("task_123", "submission"):
                future = executor.submit(work_fn)
        """
        if not self._running:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_us = (end_time - start_time) * 1e6

            with self._thread_lock:
                if future_id in self._active_futures:
                    self._active_futures[future_id]['stages'][stage_name] = duration_us

    def record_thread_event(self,
                           thread_id: int,
                           thread_name: str,
                           event_type: str,
                           duration_us: Optional[float] = None,
                           result: Optional[str] = None):
        """
        Record a thread lifecycle event.

        Args:
            thread_id: Thread identifier
            thread_name: Thread name
            event_type: Event type ('created', 'started', 'completed', 'destroyed')
            duration_us: Event duration in microseconds
            result: Event result ('success', 'error', 'cancelled')
        """
        if not self._running or not self.track_thread_lifecycle:
            return

        event = ThreadEventMetrics(
            thread_id=thread_id,
            thread_name=thread_name,
            event_type=event_type,
            timestamp=time.perf_counter(),
            duration_us=duration_us,
            result=result
        )

        self._thread_events.append(event)

    def record_pool_state(self,
                         pool_size: int,
                         active_threads: int,
                         queued_tasks: int,
                         max_queue_size: Optional[int] = None):
        """
        Record current thread pool state.

        Args:
            pool_size: Total number of threads in pool
            active_threads: Number of threads currently executing tasks
            queued_tasks: Number of tasks in queue
            max_queue_size: Maximum queue capacity
        """
        if not self._running:
            return

        thread_utilization = (active_threads / pool_size * 100) if pool_size > 0 else 0.0
        queue_utilization = (queued_tasks / max_queue_size * 100) if max_queue_size else 0.0

        # Compute recent task latencies
        recent_futures = list(self._futures)[-100:]  # Last 100 futures
        latencies = [f.total_latency_us for f in recent_futures if f.completed_successfully]

        import numpy as np
        avg_latency = np.mean(latencies) if latencies else 0.0
        p99_latency = np.percentile(latencies, 99) if latencies else 0.0

        metrics = ThreadPoolMetrics(
            pool_size=pool_size,
            active_threads=active_threads,
            queued_tasks=queued_tasks,
            thread_utilization=thread_utilization,
            queue_utilization=queue_utilization,
            avg_task_latency_us=avg_latency,
            p99_task_latency_us=p99_latency
        )

        timestamp = time.perf_counter()
        with self._pool_lock:
            self._pool_snapshots.append((timestamp, metrics))

    def mark_future_success(self, future_id: str):
        """Mark a future as successfully completed."""
        with self._thread_lock:
            if future_id in self._active_futures:
                self._active_futures[future_id]['stages']['success'] = True

    def mark_future_cancelled(self, future_id: str):
        """Mark a future as cancelled."""
        with self._thread_lock:
            if future_id in self._active_futures:
                self._active_futures[future_id]['stages']['cancelled'] = True

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive thread coordination metrics.

        Returns:
            Dictionary with:
                - thread_events: Thread lifecycle events
                - future_metrics: Future processing statistics
                - pool_metrics: Thread pool utilization
                - summary: Aggregate statistics
        """
        if not self._start_time:
            return {}

        end_time = self._stop_time or time.perf_counter()
        total_time = end_time - self._start_time

        # Compute future statistics
        future_latencies = [f.total_latency_us for f in self._futures]
        submission_times = [f.submission_time_us for f in self._futures]
        execution_times = [f.execution_time_us for f in self._futures]
        collection_times = [f.result_collection_time_us for f in self._futures]

        successful_futures = sum(1 for f in self._futures if f.completed_successfully)
        cancelled_futures = sum(1 for f in self._futures if f.cancelled)

        # Compute thread event statistics
        created_threads = sum(1 for e in self._thread_events if e.event_type == 'created')
        destroyed_threads = sum(1 for e in self._thread_events if e.event_type == 'destroyed')

        import numpy as np

        summary = {
            'total_time_seconds': total_time,
            'total_futures': len(self._futures),
            'successful_futures': successful_futures,
            'cancelled_futures': cancelled_futures,
            'success_rate': (successful_futures / len(self._futures) * 100) if self._futures else 0.0,

            # Latency statistics
            'avg_future_latency_us': np.mean(future_latencies) if future_latencies else 0.0,
            'p50_future_latency_us': np.percentile(future_latencies, 50) if future_latencies else 0.0,
            'p90_future_latency_us': np.percentile(future_latencies, 90) if future_latencies else 0.0,
            'p99_future_latency_us': np.percentile(future_latencies, 99) if future_latencies else 0.0,

            # Stage breakdown
            'avg_submission_overhead_us': np.mean(submission_times) if submission_times else 0.0,
            'avg_execution_time_us': np.mean(execution_times) if execution_times else 0.0,
            'avg_collection_overhead_us': np.mean(collection_times) if collection_times else 0.0,

            # Thread lifecycle
            'threads_created': created_threads,
            'threads_destroyed': destroyed_threads,
            'thread_churn_rate': (created_threads + destroyed_threads) / total_time if total_time > 0 else 0.0,

            # Throughput
            'futures_per_second': len(self._futures) / total_time if total_time > 0 else 0.0,
        }

        # Pool utilization over time
        pool_utilizations = [snapshot.thread_utilization for _, snapshot in self._pool_snapshots]
        queue_utilizations = [snapshot.queue_utilization for _, snapshot in self._pool_snapshots]

        pool_summary = {
            'avg_thread_utilization': np.mean(pool_utilizations) if pool_utilizations else 0.0,
            'max_thread_utilization': np.max(pool_utilizations) if pool_utilizations else 0.0,
            'avg_queue_utilization': np.mean(queue_utilizations) if queue_utilizations else 0.0,
            'max_queue_utilization': np.max(queue_utilizations) if queue_utilizations else 0.0,
        }

        return {
            'summary': summary,
            'pool_summary': pool_summary,
            'recent_futures': [
                {
                    'future_id': f.future_id,
                    'total_latency_us': f.total_latency_us,
                    'execution_time_us': f.execution_time_us,
                    'success': f.completed_successfully,
                    'cancelled': f.cancelled
                }
                for f in list(self._futures)[-100:]  # Last 100 futures
            ],
            'thread_events': [
                {
                    'thread_id': e.thread_id,
                    'thread_name': e.thread_name,
                    'event_type': e.event_type,
                    'timestamp': e.timestamp,
                    'duration_us': e.duration_us,
                    'result': e.result
                }
                for e in list(self._thread_events)[-100:]  # Last 100 events
            ]
        }

    def reset(self):
        """Reset all metrics."""
        self._thread_events.clear()
        self._futures.clear()
        self._active_futures.clear()
        self._thread_states.clear()
        self._pool_snapshots.clear()
        self._start_time = None
        self._stop_time = None
