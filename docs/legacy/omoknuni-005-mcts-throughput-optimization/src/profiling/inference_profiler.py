"""
Inference Pipeline Profiler
============================

Profiles the neural network inference pipeline end-to-end:
- Batch collection timing and efficiency
- Queue wait times and throughput
- GPU dispatch overhead
- Result distribution latency
- DLPack tensor creation/conversion costs

Integrates with GPUInferenceWorker and DLPackInferenceBridge.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np
import logging


@dataclass
class InferenceRequestMetrics:
    """Metrics for a single inference request."""
    request_id: str
    thread_id: int
    batch_size: int

    # Timing breakdown (microseconds)
    queue_wait_us: float = 0.0
    batch_collection_us: float = 0.0
    dlpack_creation_us: float = 0.0
    h2d_transfer_us: float = 0.0
    inference_us: float = 0.0
    d2h_transfer_us: float = 0.0
    result_distribution_us: float = 0.0
    total_latency_us: float = 0.0

    # Metadata
    timestamp: float = field(default_factory=time.perf_counter)
    used_dlpack: bool = False
    used_fallback: bool = False


@dataclass
class BatchMetrics:
    """Metrics for a batched inference operation."""
    batch_id: str
    batch_size: int
    timestamp: float

    # Timing breakdown (microseconds)
    collection_time_us: float = 0.0
    processing_time_us: float = 0.0
    distribution_time_us: float = 0.0

    # Efficiency metrics
    batch_efficiency: float = 0.0  # actual_size / target_size
    timeout_triggered: bool = False


@dataclass
class QueueMetrics:
    """Queue performance metrics."""

    # Depth statistics
    current_depth: int = 0
    max_depth: int = 0
    avg_depth: float = 0.0

    # Wait time statistics (microseconds)
    min_wait_us: float = 0.0
    max_wait_us: float = 0.0
    avg_wait_us: float = 0.0
    p50_wait_us: float = 0.0
    p90_wait_us: float = 0.0
    p99_wait_us: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    batches_per_second: float = 0.0


class InferencePipelineProfiler:
    """
    Profiles neural network inference pipeline performance.

    Tracks:
    - Request queuing and wait times
    - Batch collection efficiency
    - GPU transfer and computation times
    - Result distribution latency
    - DLPack zero-copy effectiveness

    Usage:
        profiler = InferencePipelineProfiler()
        profiler.start()

        with profiler.track_request("req_123", thread_id=1):
            # ... inference request ...

        metrics = profiler.get_metrics()
    """

    def __init__(self,
                 max_samples: int = 10_000,
                 enable_detailed_tracking: bool = True):
        """
        Initialize inference profiler.

        Args:
            max_samples: Maximum request samples to store
            enable_detailed_tracking: Track per-request detailed timing
        """
        self.max_samples = max_samples
        self.enable_detailed_tracking = enable_detailed_tracking

        # Request tracking
        self._requests: deque[InferenceRequestMetrics] = deque(maxlen=max_samples)
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._request_lock = threading.Lock()

        # Batch tracking
        self._batches: deque[BatchMetrics] = deque(maxlen=1000)
        self._batch_lock = threading.Lock()

        # Queue depth monitoring
        self._queue_depths: deque[Tuple[float, int]] = deque(maxlen=10_000)
        self._queue_lock = threading.Lock()

        # Control
        self._running = False
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start inference profiling."""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()
        self.logger.info("Inference pipeline profiler started")

    def stop(self):
        """Stop inference profiling."""
        if not self._running:
            return

        self._running = False
        self._stop_time = time.perf_counter()
        self.logger.info("Inference pipeline profiler stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    @contextmanager
    def track_request(self, request_id: str, thread_id: int, batch_size: int = 1):
        """
        Track a single inference request end-to-end.

        Usage:
            with profiler.track_request("req_123", thread_id=1, batch_size=32):
                # ... inference request processing ...
        """
        if not self._running:
            yield
            return

        start_time = time.perf_counter()

        # Initialize request tracking
        with self._request_lock:
            self._active_requests[request_id] = {
                'start_time': start_time,
                'thread_id': thread_id,
                'batch_size': batch_size,
                'stages': {}
            }

        try:
            yield
        finally:
            end_time = time.perf_counter()
            total_latency_us = (end_time - start_time) * 1e6

            with self._request_lock:
                if request_id in self._active_requests:
                    req_data = self._active_requests.pop(request_id)
                    stages = req_data['stages']

                    metrics = InferenceRequestMetrics(
                        request_id=request_id,
                        thread_id=thread_id,
                        batch_size=batch_size,
                        total_latency_us=total_latency_us,
                        queue_wait_us=stages.get('queue_wait', 0.0),
                        batch_collection_us=stages.get('batch_collection', 0.0),
                        dlpack_creation_us=stages.get('dlpack_creation', 0.0),
                        h2d_transfer_us=stages.get('h2d_transfer', 0.0),
                        inference_us=stages.get('inference', 0.0),
                        d2h_transfer_us=stages.get('d2h_transfer', 0.0),
                        result_distribution_us=stages.get('result_distribution', 0.0),
                        used_dlpack=stages.get('used_dlpack', False),
                        used_fallback=stages.get('used_fallback', False)
                    )

                    self._requests.append(metrics)

    @contextmanager
    def track_stage(self, request_id: str, stage_name: str):
        """
        Track a specific stage within a request.

        Usage:
            with profiler.track_stage("req_123", "queue_wait"):
                # ... waiting in queue ...
        """
        if not self._running or not self.enable_detailed_tracking:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_us = (end_time - start_time) * 1e6

            with self._request_lock:
                if request_id in self._active_requests:
                    self._active_requests[request_id]['stages'][stage_name] = duration_us

    def record_batch(self,
                    batch_id: str,
                    batch_size: int,
                    collection_time_us: float,
                    processing_time_us: float,
                    distribution_time_us: float,
                    target_batch_size: int,
                    timeout_triggered: bool = False):
        """
        Record metrics for a completed batch.

        Args:
            batch_id: Unique batch identifier
            batch_size: Actual batch size
            collection_time_us: Time to collect batch (microseconds)
            processing_time_us: GPU processing time (microseconds)
            distribution_time_us: Result distribution time (microseconds)
            target_batch_size: Target/maximum batch size
            timeout_triggered: Whether batch was triggered by timeout
        """
        if not self._running:
            return

        batch_efficiency = batch_size / target_batch_size if target_batch_size > 0 else 0.0

        metrics = BatchMetrics(
            batch_id=batch_id,
            batch_size=batch_size,
            timestamp=time.perf_counter(),
            collection_time_us=collection_time_us,
            processing_time_us=processing_time_us,
            distribution_time_us=distribution_time_us,
            batch_efficiency=batch_efficiency,
            timeout_triggered=timeout_triggered
        )

        with self._batch_lock:
            self._batches.append(metrics)

    def record_queue_depth(self, depth: int):
        """
        Record current queue depth for monitoring.

        Args:
            depth: Current number of items in queue
        """
        if not self._running:
            return

        timestamp = time.perf_counter()

        with self._queue_lock:
            self._queue_depths.append((timestamp, depth))

    def mark_dlpack_used(self, request_id: str):
        """Mark that a request used DLPack zero-copy path."""
        with self._request_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]['stages']['used_dlpack'] = True

    def mark_fallback_used(self, request_id: str):
        """Mark that a request used fallback (numpy copy) path."""
        with self._request_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]['stages']['used_fallback'] = True

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive inference pipeline metrics.

        Returns:
            Dictionary with:
                - request_metrics: Per-request timing breakdown
                - batch_metrics: Batch efficiency statistics
                - queue_metrics: Queue performance statistics
                - summary: Aggregate statistics
        """
        if not self._start_time:
            return {}

        end_time = self._stop_time or time.perf_counter()
        total_time = end_time - self._start_time

        # Compute request statistics
        request_latencies = [r.total_latency_us for r in self._requests]
        queue_waits = [r.queue_wait_us for r in self._requests]
        inference_times = [r.inference_us for r in self._requests]

        dlpack_count = sum(1 for r in self._requests if r.used_dlpack)
        fallback_count = sum(1 for r in self._requests if r.used_fallback)

        # Compute batch statistics
        batch_sizes = [b.batch_size for b in self._batches]
        batch_efficiencies = [b.batch_efficiency for b in self._batches]
        timeout_batches = sum(1 for b in self._batches if b.timeout_triggered)

        # Compute queue statistics
        queue_depths_values = [d for _, d in self._queue_depths]

        summary = {
            'total_time_seconds': total_time,
            'total_requests': len(self._requests),
            'total_batches': len(self._batches),
            'requests_per_second': len(self._requests) / total_time if total_time > 0 else 0.0,
            'batches_per_second': len(self._batches) / total_time if total_time > 0 else 0.0,

            # Latency statistics (microseconds)
            'avg_latency_us': np.mean(request_latencies) if request_latencies else 0.0,
            'min_latency_us': np.min(request_latencies) if request_latencies else 0.0,
            'max_latency_us': np.max(request_latencies) if request_latencies else 0.0,
            'p50_latency_us': np.percentile(request_latencies, 50) if request_latencies else 0.0,
            'p90_latency_us': np.percentile(request_latencies, 90) if request_latencies else 0.0,
            'p99_latency_us': np.percentile(request_latencies, 99) if request_latencies else 0.0,

            # Queue wait statistics
            'avg_queue_wait_us': np.mean(queue_waits) if queue_waits else 0.0,
            'p99_queue_wait_us': np.percentile(queue_waits, 99) if queue_waits else 0.0,

            # Inference time statistics
            'avg_inference_us': np.mean(inference_times) if inference_times else 0.0,
            'p99_inference_us': np.percentile(inference_times, 99) if inference_times else 0.0,

            # Batch statistics
            'avg_batch_size': np.mean(batch_sizes) if batch_sizes else 0.0,
            'avg_batch_efficiency': np.mean(batch_efficiencies) if batch_efficiencies else 0.0,
            'timeout_batch_rate': timeout_batches / len(self._batches) if self._batches else 0.0,

            # DLPack effectiveness
            'dlpack_usage_rate': dlpack_count / len(self._requests) if self._requests else 0.0,
            'fallback_usage_rate': fallback_count / len(self._requests) if self._requests else 0.0,

            # Queue depth statistics
            'avg_queue_depth': np.mean(queue_depths_values) if queue_depths_values else 0.0,
            'max_queue_depth': np.max(queue_depths_values) if queue_depths_values else 0,
        }

        # Stage timing breakdown (averaged across all requests)
        stage_times = defaultdict(list)
        for req in self._requests:
            stage_times['queue_wait'].append(req.queue_wait_us)
            stage_times['batch_collection'].append(req.batch_collection_us)
            stage_times['dlpack_creation'].append(req.dlpack_creation_us)
            stage_times['h2d_transfer'].append(req.h2d_transfer_us)
            stage_times['inference'].append(req.inference_us)
            stage_times['d2h_transfer'].append(req.d2h_transfer_us)
            stage_times['result_distribution'].append(req.result_distribution_us)

        stage_breakdown = {
            stage: {
                'avg_us': np.mean(times) if times else 0.0,
                'p50_us': np.percentile(times, 50) if times else 0.0,
                'p90_us': np.percentile(times, 90) if times else 0.0,
                'p99_us': np.percentile(times, 99) if times else 0.0,
                'percentage': (np.mean(times) / np.mean(request_latencies) * 100) if times and request_latencies else 0.0
            }
            for stage, times in stage_times.items()
        }

        return {
            'summary': summary,
            'stage_breakdown': stage_breakdown,
            'recent_requests': [
                {
                    'request_id': r.request_id,
                    'batch_size': r.batch_size,
                    'total_latency_us': r.total_latency_us,
                    'used_dlpack': r.used_dlpack,
                    'used_fallback': r.used_fallback
                }
                for r in list(self._requests)[-100:]  # Last 100 requests
            ],
            'recent_batches': [
                {
                    'batch_id': b.batch_id,
                    'batch_size': b.batch_size,
                    'efficiency': b.batch_efficiency,
                    'timeout_triggered': b.timeout_triggered
                }
                for b in list(self._batches)[-100:]  # Last 100 batches
            ]
        }

    def reset(self):
        """Reset all metrics."""
        self._requests.clear()
        self._batches.clear()
        self._queue_depths.clear()
        self._active_requests.clear()
        self._start_time = None
        self._stop_time = None
