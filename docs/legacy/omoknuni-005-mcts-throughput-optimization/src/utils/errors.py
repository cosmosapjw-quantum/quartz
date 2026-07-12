"""
Comprehensive error handling framework for the AlphaZero engine.

This module defines custom exception classes and error handling utilities
to provide informative error messages and graceful degradation across
all system components.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from functools import wraps
from enum import Enum
import torch


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlphaZeroError(Exception):
    """Base exception class for all AlphaZero engine errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (Context: {context_str})"
        return base_msg


class ConfigurationError(AlphaZeroError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, **kwargs):
        context = {"field": field, "value": value} if field else {}
        super().__init__(message, ErrorSeverity.ERROR, context, **kwargs)


class ModelError(AlphaZeroError):
    """Raised when neural network model operations fail."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    def __init__(self, message: str, model_path: str, **kwargs):
        context = {"model_path": model_path}
        super().__init__(message, ErrorSeverity.ERROR, context, **kwargs)


class ModelValidationError(ModelError):
    """Raised when model validation fails."""

    def __init__(self, message: str, expected: Any = None, actual: Any = None, **kwargs):
        context = {"expected": expected, "actual": actual}
        super().__init__(message, ErrorSeverity.ERROR, context, **kwargs)


class InferenceError(AlphaZeroError):
    """Raised when neural network inference fails."""
    pass


class CriticalInferenceError(InferenceError):
    """Raised for critical inference errors that require immediate attention."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.CRITICAL, **kwargs)


class MCTSError(AlphaZeroError):
    """Raised when MCTS operations fail."""
    pass


class TreeCorruptionError(MCTSError):
    """Raised when MCTS tree structure is corrupted."""

    def __init__(self, message: str, node_id: Optional[int] = None, **kwargs):
        context = {"node_id": node_id} if node_id is not None else {}
        super().__init__(message, ErrorSeverity.CRITICAL, context, **kwargs)


class MemoryAllocationError(AlphaZeroError):
    """Raised when memory allocation fails."""

    def __init__(self, message: str, requested_bytes: Optional[int] = None,
                 available_bytes: Optional[int] = None, **kwargs):
        context = {
            "requested_bytes": requested_bytes,
            "available_bytes": available_bytes
        }
        super().__init__(message, ErrorSeverity.CRITICAL, context, **kwargs)


class ThreadCoordinationError(AlphaZeroError):
    """Raised when thread coordination fails."""

    def __init__(self, message: str, thread_name: Optional[str] = None, **kwargs):
        context = {"thread_name": thread_name} if thread_name else {}
        super().__init__(message, ErrorSeverity.ERROR, context, **kwargs)


class GameStateError(AlphaZeroError):
    """Raised when game state operations fail."""
    pass


class TrainingError(AlphaZeroError):
    """Raised when training operations fail."""
    pass


class TrainingStabilityError(TrainingError):
    """Raised when training becomes unstable (NaN, divergence)."""

    def __init__(self, message: str, loss_value: Optional[float] = None,
                 gradient_norm: Optional[float] = None, **kwargs):
        context = {"loss_value": loss_value, "gradient_norm": gradient_norm}
        super().__init__(message, ErrorSeverity.ERROR, context, **kwargs)


class ThreadHealthMonitor:
    """Monitor and manage thread health with failure tracking."""

    def __init__(self, max_consecutive_failures: int = 10,
                 failure_backoff: float = 1.0, max_backoff: float = 10.0):
        self.failure_counts = defaultdict(int)
        self.max_failures = max_consecutive_failures
        self.base_backoff = failure_backoff
        self.max_backoff = max_backoff
        self.logger = logging.getLogger(__name__)

    def record_failure(self, thread_name: str, exception: Exception) -> bool:
        """
        Record a thread failure and determine if thread should continue.

        Args:
            thread_name: Name of the thread that failed
            exception: The exception that caused the failure

        Returns:
            True if thread should continue, False if it should terminate
        """
        self.failure_counts[thread_name] += 1
        failure_count = self.failure_counts[thread_name]

        self.logger.error(
            f"Thread '{thread_name}' failure #{failure_count}: {exception}",
            extra={"thread_name": thread_name, "failure_count": failure_count}
        )

        if failure_count >= self.max_failures:
            self.logger.critical(
                f"Thread '{thread_name}' exceeded max failures ({self.max_failures}), terminating",
                extra={"thread_name": thread_name, "max_failures": self.max_failures}
            )
            return False

        # Apply exponential backoff
        backoff_time = min(failure_count * self.base_backoff, self.max_backoff)
        self.logger.warning(
            f"Thread '{thread_name}' backing off for {backoff_time:.1f}s",
            extra={"thread_name": thread_name, "backoff_time": backoff_time}
        )
        time.sleep(backoff_time)
        return True

    def record_success(self, thread_name: str):
        """Record a successful operation, resetting failure count."""
        if thread_name in self.failure_counts:
            previous_failures = self.failure_counts[thread_name]
            if previous_failures > 0:
                self.logger.info(
                    f"Thread '{thread_name}' recovered after {previous_failures} failures",
                    extra={"thread_name": thread_name, "previous_failures": previous_failures}
                )
            del self.failure_counts[thread_name]


class GPUOperationManager:
    """Manage GPU operations with timeout and error handling."""

    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(__name__)

    def execute_with_timeout(self, operation: Callable, timeout: Optional[float] = None,
                           operation_name: str = "GPU operation") -> Any:
        """
        Execute a GPU operation with timeout protection.

        Args:
            operation: The operation to execute
            timeout: Timeout in seconds (uses default if None)
            operation_name: Name for logging

        Returns:
            Result of the operation

        Raises:
            InferenceError: If operation fails or times out
        """
        timeout = timeout or self.default_timeout

        try:
            # Note: For actual timeout implementation, we'd need to use
            # threading or signal-based approaches depending on the operation
            start_time = time.time()
            result = operation()
            elapsed = time.time() - start_time

            if elapsed > timeout:
                self.logger.warning(
                    f"{operation_name} took {elapsed:.2f}s (timeout: {timeout}s)",
                    extra={"operation": operation_name, "elapsed": elapsed, "timeout": timeout}
                )

            return result

        except torch.cuda.OutOfMemoryError as e:
            raise InferenceError(
                f"GPU out of memory during {operation_name}",
                context={"operation": operation_name}
            ) from e
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise CriticalInferenceError(
                    f"Critical GPU error during {operation_name}: {e}",
                    context={"operation": operation_name}
                ) from e
            raise InferenceError(f"{operation_name} failed: {e}") from e
        except Exception as e:
            raise InferenceError(
                f"Unexpected error during {operation_name}: {e}",
                context={"operation": operation_name}
            ) from e


def with_error_handling(logger: Optional[logging.Logger] = None,
                       reraise: bool = True,
                       default_return: Any = None):
    """
    Decorator for adding standardized error handling to functions.

    Args:
        logger: Logger to use (creates one if None)
        reraise: Whether to reraise exceptions after logging
        default_return: Value to return if exception and reraise=False
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            actual_logger = logger or logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except AlphaZeroError:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                actual_logger.error(
                    f"Unexpected error in {func.__name__}: {e}",
                    extra={
                        "function_name": func.__name__,
                        "function_args": str(args)[:100] if args else "",
                        "function_kwargs": str(kwargs)[:100] if kwargs else "",
                        "error_traceback": traceback.format_exc()
                    }
                )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def validate_gpu_state(device: torch.device) -> None:
    """
    Validate GPU state and availability.

    Args:
        device: PyTorch device to validate

    Raises:
        CriticalInferenceError: If GPU is in invalid state
    """
    try:
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise CriticalInferenceError("CUDA not available")

            device_idx = device.index or 0
            if device_idx >= torch.cuda.device_count():
                raise CriticalInferenceError(
                    f"Invalid CUDA device index: {device_idx}",
                    context={"device_index": device_idx, "available_devices": torch.cuda.device_count()}
                )

            # Check memory availability
            try:
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_cached = torch.cuda.memory_reserved(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory

                if memory_allocated > total_memory * 0.95:
                    raise InferenceError(
                        "GPU memory critically low",
                        context={
                            "allocated": memory_allocated,
                            "total": total_memory,
                            "utilization": memory_allocated / total_memory
                        }
                    )
            except torch.cuda.CudaError as e:
                raise CriticalInferenceError(f"CUDA error during memory check: {e}") from e

    except Exception as e:
        if isinstance(e, (InferenceError, CriticalInferenceError)):
            raise
        raise CriticalInferenceError(f"GPU validation failed: {e}") from e


class ErrorReporter:
    """Centralized error reporting and metrics collection."""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.last_errors = {}
        self.logger = logging.getLogger(__name__)

    def report_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Report an error for monitoring and metrics."""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        self.last_errors[error_type] = {
            "timestamp": time.time(),
            "message": str(error),
            "context": context or {}
        }

        severity = getattr(error, 'severity', ErrorSeverity.ERROR)

        log_method = {
            ErrorSeverity.INFO: self.logger.info,
            ErrorSeverity.WARNING: self.logger.warning,
            ErrorSeverity.ERROR: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }[severity]

        log_method(
            f"{error_type}: {error}",
            extra={
                "error_type": error_type,
                "severity": severity.value,
                "context": context or {},
                "count": self.error_counts[error_type]
            }
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all reported errors."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": dict(self.error_counts),
            "recent_errors": self.last_errors
        }


# Global error reporter instance
error_reporter = ErrorReporter()