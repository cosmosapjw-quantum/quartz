"""
Telemetry and monitoring package for the AlphaZero engine.

Provides comprehensive metrics collection, GPU monitoring, and structured logging
for performance analysis and system observability.
"""

from .metrics import (
    MetricsCollector,
    PerformanceMetrics,
    get_metrics_collector,
    cleanup_metrics,
)

from .logger import (
    AlphaZeroLogger,
    LogLevel,
    LogContext,
    get_logger,
    configure_logging,
    disable_logging,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "PerformanceMetrics",
    "get_metrics_collector",
    "cleanup_metrics",
    # Logging
    "AlphaZeroLogger",
    "LogLevel",
    "LogContext",
    "get_logger",
    "configure_logging",
    "disable_logging",
]
