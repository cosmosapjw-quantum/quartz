"""
Python Profiling Framework for MCTS Coordination
================================================

Comprehensive profiling layer for analyzing Python coordination overhead,
GIL contention, thread communication, and integration with C++ instrumentation.

Exports:
    - GILProfiler: GIL acquisition/release tracking
    - InferencePipelineProfiler: Inference pipeline profiling
    - ThreadCoordinatorProfiler: Thread coordination overhead
    - MemoryProfiler: Memory allocation and GC tracking
    - ProfilingSession: Unified profiling session manager
"""

from .gil_profiler import GILProfiler
from .inference_profiler import InferencePipelineProfiler
from .thread_profiler import ThreadCoordinatorProfiler
from .memory_profiler import MemoryProfiler
from .profiling_session import ProfilingSession, ProfilerConfig
from .report_generator import (
    generate_html_report,
    generate_json_report,
    generate_flamegraph
)

__all__ = [
    'GILProfiler',
    'InferencePipelineProfiler',
    'ThreadCoordinatorProfiler',
    'MemoryProfiler',
    'ProfilingSession',
    'ProfilerConfig',
    'generate_html_report',
    'generate_json_report',
    'generate_flamegraph',
]
