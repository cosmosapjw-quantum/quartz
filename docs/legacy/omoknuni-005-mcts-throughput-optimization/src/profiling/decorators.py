"""
Enhanced Profiling Decorators
==============================

Comprehensive function-level profiling for Python MCTS coordination.

Features:
- Zero-overhead when disabled
- Thread-local metrics storage
- GIL tracking
- State cloning tracking
- Feature extraction tracking
- Automatic aggregation
"""

import time
import functools
import threading
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

# Thread-local storage for profiling data
_thread_local = threading.local()
_global_enabled = threading.Event()  # Global enable/disable flag

def set_profiling_enabled(enabled: bool):
    """Enable or disable profiling globally"""
    if enabled:
        _global_enabled.set()
    else:
        _global_enabled.clear()

def is_profiling_enabled() -> bool:
    """Check if profiling is enabled"""
    return _global_enabled.is_set()

@dataclass
class FunctionStats:
    """Statistics for a single function"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    sum_squares: float = 0.0  # For variance calculation

    def update(self, elapsed: float):
        """Update statistics with new timing"""
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.sum_squares += elapsed * elapsed

    @property
    def mean(self) -> float:
        """Calculate mean time"""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Calculate variance"""
        if self.count == 0:
            return 0.0
        mean = self.mean
        return (self.sum_squares / self.count) - (mean * mean)

    @property
    def std_dev(self) -> float:
        """Calculate standard deviation"""
        return self.variance ** 0.5

@dataclass
class ThreadMetrics:
    """Per-thread profiling metrics"""
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    function_timings: Dict[str, FunctionStats] = field(default_factory=dict)
    gil_timings: Dict[str, FunctionStats] = field(default_factory=dict)
    state_clone_count: int = 0
    state_clone_times: list = field(default_factory=list)
    feature_extraction_count: int = 0
    feature_extraction_times: list = field(default_factory=list)
    python_list_builds: int = 0
    dlpack_conversions: int = 0
    numpy_array_creations: int = 0

    def get_function_stats(self, func_name: str) -> FunctionStats:
        """Get or create function stats"""
        if func_name not in self.function_timings:
            self.function_timings[func_name] = FunctionStats()
        return self.function_timings[func_name]

    def get_gil_stats(self, func_name: str) -> FunctionStats:
        """Get or create GIL stats"""
        if func_name not in self.gil_timings:
            self.gil_timings[func_name] = FunctionStats()
        return self.gil_timings[func_name]

# Global registry of all thread metrics
_all_thread_metrics: Dict[int, ThreadMetrics] = {}
_metrics_lock = threading.Lock()

def get_thread_metrics() -> ThreadMetrics:
    """Get or create thread-local metrics"""
    if not hasattr(_thread_local, 'metrics'):
        thread_id = threading.get_ident()
        _thread_local.metrics = ThreadMetrics(thread_id=thread_id)

        # Register globally
        with _metrics_lock:
            _all_thread_metrics[thread_id] = _thread_local.metrics

    return _thread_local.metrics

def get_all_thread_metrics() -> Dict[int, ThreadMetrics]:
    """Get metrics from all threads"""
    with _metrics_lock:
        return dict(_all_thread_metrics)

def reset_all_metrics():
    """Reset all profiling metrics"""
    with _metrics_lock:
        _all_thread_metrics.clear()
    if hasattr(_thread_local, 'metrics'):
        delattr(_thread_local, 'metrics')

def profile_function(category: str, track_gil: bool = True):
    """
    Decorator for comprehensive function profiling.

    Args:
        category: Metric category ('coordination', 'inference', 'state', etc.)
        track_gil: Whether to track GIL acquisition/release

    Example:
        @profile_function("coordination", track_gil=True)
        def search(self, root_state, simulations):
            # ... implementation ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Early exit if profiling disabled
            if not is_profiling_enabled():
                return func(*args, **kwargs)

            metrics = get_thread_metrics()
            func_name = f"{category}.{func.__name__}"

            # GIL tracking
            gil_start = time.perf_counter() if track_gil else None

            # Function timing
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start

                # Update function stats
                stats = metrics.get_function_stats(func_name)
                stats.update(elapsed)

                # Update GIL stats
                if track_gil and gil_start is not None:
                    gil_elapsed = time.perf_counter() - gil_start
                    gil_stats = metrics.get_gil_stats(func_name)
                    gil_stats.update(gil_elapsed)

        return wrapper
    return decorator

@contextmanager
def profile_state_clone():
    """
    Context manager for tracking state cloning.

    Example:
        with profile_state_clone():
            state_copy = state.clone()
    """
    if not is_profiling_enabled():
        yield
        return

    metrics = get_thread_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics.state_clone_count += 1
        metrics.state_clone_times.append(elapsed)

@contextmanager
def profile_feature_extraction():
    """
    Context manager for tracking feature extraction.

    Example:
        with profile_feature_extraction():
            features = state.get_enhanced_tensor_representation()
    """
    if not is_profiling_enabled():
        yield
        return

    metrics = get_thread_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics.feature_extraction_count += 1
        metrics.feature_extraction_times.append(elapsed)

@contextmanager
def profile_python_list_build():
    """Track Python list building overhead"""
    if not is_profiling_enabled():
        yield
        return

    metrics = get_thread_metrics()
    metrics.python_list_builds += 1
    yield

@contextmanager
def profile_dlpack_conversion():
    """Track DLPack conversion overhead"""
    if not is_profiling_enabled():
        yield
        return

    metrics = get_thread_metrics()
    metrics.dlpack_conversions += 1
    yield

@contextmanager
def profile_numpy_array_creation():
    """Track NumPy array creation overhead"""
    if not is_profiling_enabled():
        yield
        return

    metrics = get_thread_metrics()
    metrics.numpy_array_creations += 1
    yield

def get_profiling_summary() -> Dict[str, Any]:
    """
    Get comprehensive profiling summary.

    Returns:
        Dictionary with aggregated metrics from all threads
    """
    all_metrics = get_all_thread_metrics()

    summary = {
        'total_threads': len(all_metrics),
        'function_stats': defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
        }),
        'gil_stats': defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
        }),
        'state_cloning': {
            'total_clones': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
        },
        'feature_extraction': {
            'total_extractions': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
        },
        'python_overhead': {
            'list_builds': 0,
            'dlpack_conversions': 0,
            'numpy_creations': 0,
        }
    }

    # Aggregate across all threads
    for thread_metrics in all_metrics.values():
        # Function stats
        for func_name, stats in thread_metrics.function_timings.items():
            agg = summary['function_stats'][func_name]
            agg['count'] += stats.count
            agg['total_time'] += stats.total_time
            agg['min_time'] = min(agg['min_time'], stats.min_time)
            agg['max_time'] = max(agg['max_time'], stats.max_time)

        # GIL stats
        for func_name, stats in thread_metrics.gil_timings.items():
            agg = summary['gil_stats'][func_name]
            agg['count'] += stats.count
            agg['total_time'] += stats.total_time

        # State cloning
        if thread_metrics.state_clone_times:
            summary['state_cloning']['total_clones'] += thread_metrics.state_clone_count
            summary['state_cloning']['total_time'] += sum(thread_metrics.state_clone_times)
            summary['state_cloning']['min_time'] = min(
                summary['state_cloning']['min_time'],
                min(thread_metrics.state_clone_times)
            )
            summary['state_cloning']['max_time'] = max(
                summary['state_cloning']['max_time'],
                max(thread_metrics.state_clone_times)
            )

        # Feature extraction
        if thread_metrics.feature_extraction_times:
            summary['feature_extraction']['total_extractions'] += thread_metrics.feature_extraction_count
            summary['feature_extraction']['total_time'] += sum(thread_metrics.feature_extraction_times)
            summary['feature_extraction']['min_time'] = min(
                summary['feature_extraction']['min_time'],
                min(thread_metrics.feature_extraction_times)
            )
            summary['feature_extraction']['max_time'] = max(
                summary['feature_extraction']['max_time'],
                max(thread_metrics.feature_extraction_times)
            )

        # Python overhead
        summary['python_overhead']['list_builds'] += thread_metrics.python_list_builds
        summary['python_overhead']['dlpack_conversions'] += thread_metrics.dlpack_conversions
        summary['python_overhead']['numpy_creations'] += thread_metrics.numpy_array_creations

    # Calculate averages
    if summary['state_cloning']['total_clones'] > 0:
        summary['state_cloning']['avg_time'] = (
            summary['state_cloning']['total_time'] / summary['state_cloning']['total_clones']
        )

    if summary['feature_extraction']['total_extractions'] > 0:
        summary['feature_extraction']['avg_time'] = (
            summary['feature_extraction']['total_time'] / summary['feature_extraction']['total_extractions']
        )

    # Convert defaultdicts to regular dicts
    summary['function_stats'] = dict(summary['function_stats'])
    summary['gil_stats'] = dict(summary['gil_stats'])

    # Calculate mean times for functions
    for func_name, stats in summary['function_stats'].items():
        if stats['count'] > 0:
            stats['mean_time'] = stats['total_time'] / stats['count']

    return summary

def print_profiling_summary():
    """Print human-readable profiling summary"""
    summary = get_profiling_summary()

    print("\n" + "="*80)
    print("ENHANCED PROFILING SUMMARY")
    print("="*80)

    print(f"\nTotal Threads: {summary['total_threads']}")

    # Function stats
    print("\n--- Function Timings ---")
    if summary['function_stats']:
        for func_name, stats in sorted(
            summary['function_stats'].items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:20]:  # Top 20
            print(f"  {func_name}:")
            print(f"    Calls: {stats['count']}")
            print(f"    Total: {stats['total_time']*1000:.2f}ms")
            print(f"    Mean: {stats.get('mean_time', 0)*1000:.2f}ms")
            print(f"    Min/Max: {stats['min_time']*1000:.2f}ms / {stats['max_time']*1000:.2f}ms")

    # State cloning
    print("\n--- State Cloning ---")
    sc = summary['state_cloning']
    if sc['total_clones'] > 0:
        print(f"  Total Clones: {sc['total_clones']}")
        print(f"  Total Time: {sc['total_time']*1000:.2f}ms")
        print(f"  Avg Time: {sc['avg_time']*1000:.4f}ms")
        print(f"  Min/Max: {sc['min_time']*1000:.4f}ms / {sc['max_time']*1000:.4f}ms")
    else:
        print("  (No state clones tracked)")

    # Feature extraction
    print("\n--- Feature Extraction ---")
    fe = summary['feature_extraction']
    if fe['total_extractions'] > 0:
        print(f"  Total Extractions: {fe['total_extractions']}")
        print(f"  Total Time: {fe['total_time']*1000:.2f}ms")
        print(f"  Avg Time: {fe['avg_time']*1000:.4f}ms")
        print(f"  Min/Max: {fe['min_time']*1000:.4f}ms / {fe['max_time']*1000:.4f}ms")
    else:
        print("  (No feature extractions tracked)")

    # Python overhead
    print("\n--- Python Overhead ---")
    po = summary['python_overhead']
    print(f"  List Builds: {po['list_builds']}")
    print(f"  DLPack Conversions: {po['dlpack_conversions']}")
    print(f"  NumPy Creations: {po['numpy_creations']}")

    # GIL stats
    print("\n--- GIL Timings ---")
    if summary['gil_stats']:
        for func_name, stats in sorted(
            summary['gil_stats'].items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]:  # Top 10
            print(f"  {func_name}:")
            print(f"    GIL Acquisitions: {stats['count']}")
            print(f"    Total Time: {stats['total_time']*1000:.2f}ms")

    print("\n" + "="*80)
