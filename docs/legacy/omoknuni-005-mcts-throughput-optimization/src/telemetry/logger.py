"""
Structured logging framework for the AlphaZero engine.

Provides consistent, structured logging with performance context
and configurable output formats.
"""

import logging
import json
import time
import threading
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Structured logging context."""

    component: str
    operation: Optional[str] = None
    game_type: Optional[str] = None
    thread_id: Optional[int] = None
    session_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "thread": threading.current_thread().name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add structured context if available
        if hasattr(record, "context") and record.context:
            context_dict = (
                asdict(record.context)
                if hasattr(record.context, "__dict__")
                else record.context
            )
            log_entry["context"] = context_dict

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith("_"):
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "context",
                ]:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


class AlphaZeroLogger:
    """
    Structured logger for AlphaZero engine components.

    Provides context-aware logging with performance metrics integration
    and configurable output formats.
    """

    def __init__(
        self,
        name: str,
        level: Union[LogLevel, str] = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = False,
        log_file: Optional[str] = None,
        structured_format: bool = True,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically component name)
            level: Logging level
            enable_console: Enable console output
            enable_file: Enable file output
            log_file: Log file path (required if enable_file=True)
            structured_format: Use structured JSON format
        """
        self.name = name
        self.context = LogContext(component=name)

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            getattr(logging, level.value if isinstance(level, LogLevel) else level)
        )

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            if structured_format:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            self.logger.addHandler(console_handler)

        # File handler
        if enable_file:
            if not log_file:
                raise ValueError("log_file must be specified when enable_file=True")

            file_handler = logging.FileHandler(log_file)
            if structured_format:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def with_context(self, **kwargs) -> "AlphaZeroLogger":
        """Create a new logger instance with additional context."""
        new_logger = AlphaZeroLogger(self.name, enable_console=False, enable_file=False)
        new_logger.logger = self.logger  # Share the same underlying logger

        # Update context
        new_context = LogContext(
            component=self.context.component,
            operation=kwargs.get("operation", self.context.operation),
            game_type=kwargs.get("game_type", self.context.game_type),
            thread_id=kwargs.get("thread_id", self.context.thread_id),
            session_id=kwargs.get("session_id", self.context.session_id),
            performance_metrics=kwargs.get(
                "performance_metrics", self.context.performance_metrics
            ),
        )

        # Add any additional context fields
        for key, value in kwargs.items():
            if not hasattr(new_context, key):
                setattr(new_context, key, value)

        new_logger.context = new_context
        return new_logger

    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method with context."""
        extra = {"context": self.context}
        extra.update(kwargs)

        getattr(self.logger, level.value.lower())(message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def log_performance(
        self,
        operation: str,
        duration: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log performance metrics for an operation.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            additional_metrics: Additional performance metrics
        """
        metrics = {"duration_seconds": duration}
        if additional_metrics:
            metrics.update(additional_metrics)

        performance_logger = self.with_context(
            operation=operation, performance_metrics=metrics
        )
        performance_logger.info(
            f"Performance: {operation} completed in {duration:.4f}s"
        )

    def log_simulation_batch(
        self,
        game_type: str,
        batch_size: int,
        simulations_per_second: float,
        gpu_utilization: float,
        duration: float,
    ) -> None:
        """
        Log MCTS simulation batch performance.

        Args:
            game_type: Type of game
            batch_size: Size of the simulation batch
            simulations_per_second: Current simulation rate
            gpu_utilization: GPU utilization percentage
            duration: Batch duration in seconds
        """
        metrics = {
            "batch_size": batch_size,
            "simulations_per_second": simulations_per_second,
            "gpu_utilization_percent": gpu_utilization,
            "duration_seconds": duration,
        }

        batch_logger = self.with_context(
            operation="simulation_batch",
            game_type=game_type,
            performance_metrics=metrics,
        )
        batch_logger.info(
            f"Simulation batch: {batch_size} sims, "
            f"{simulations_per_second:.1f} sims/s, "
            f"{gpu_utilization:.1f}% GPU"
        )

    def log_inference_batch(
        self,
        batch_size: int,
        inference_time: float,
        queue_wait_time: float,
        throughput: float,
    ) -> None:
        """
        Log neural network inference batch performance.

        Args:
            batch_size: Size of the inference batch
            inference_time: Pure inference time in seconds
            queue_wait_time: Time spent waiting in queue
            throughput: Inferences per second
        """
        metrics = {
            "batch_size": batch_size,
            "inference_time_seconds": inference_time,
            "queue_wait_seconds": queue_wait_time,
            "throughput_per_second": throughput,
            "total_time_seconds": inference_time + queue_wait_time,
        }

        inference_logger = self.with_context(
            operation="inference_batch", performance_metrics=metrics
        )
        inference_logger.info(
            f"Inference batch: {batch_size} positions, "
            f"{inference_time*1000:.1f}ms inference, "
            f"{queue_wait_time*1000:.1f}ms queue, "
            f"{throughput:.1f} pos/s"
        )


# Global logger instances
_loggers: Dict[str, AlphaZeroLogger] = {}
_default_config = {
    "level": LogLevel.INFO,
    "enable_console": True,
    "enable_file": False,
    "structured_format": True,
}


def get_logger(name: str, **config_overrides) -> AlphaZeroLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (component name)
        **config_overrides: Override default configuration

    Returns:
        AlphaZeroLogger instance
    """
    if name not in _loggers:
        config = {**_default_config, **config_overrides}
        _loggers[name] = AlphaZeroLogger(name, **config)

    return _loggers[name]


def configure_logging(
    level: Union[LogLevel, str] = LogLevel.INFO,
    enable_console: bool = True,
    enable_file: bool = False,
    log_file: Optional[str] = None,
    structured_format: bool = True,
) -> None:
    """
    Configure global logging defaults.

    Args:
        level: Default logging level
        enable_console: Enable console output
        enable_file: Enable file output
        log_file: Default log file path
        structured_format: Use structured JSON format
    """
    global _default_config
    _default_config.update(
        {
            "level": level,
            "enable_console": enable_console,
            "enable_file": enable_file,
            "log_file": log_file,
            "structured_format": structured_format,
        }
    )

    # Clear existing loggers to pick up new config
    _loggers.clear()


def disable_logging() -> None:
    """Disable all logging output."""
    for logger in _loggers.values():
        logger.logger.setLevel(logging.CRITICAL + 1)
