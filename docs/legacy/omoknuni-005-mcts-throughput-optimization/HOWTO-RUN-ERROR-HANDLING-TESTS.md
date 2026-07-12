# Error Handling Framework Testing Guide

## Overview

The AlphaZero engine includes a comprehensive error handling framework designed to provide robust operation under adverse conditions. This guide explains how to test and validate the error handling capabilities.

## Error Handling Components

### Custom Exception Hierarchy

The framework provides specialized exceptions for different failure modes:

- `AlphaZeroError` - Base exception class with context and severity
- `ModelError`, `ModelLoadError`, `ModelValidationError` - Neural network model errors
- `InferenceError`, `CriticalInferenceError` - Inference pipeline errors
- `MCTSError`, `TreeCorruptionError` - MCTS tree operation errors
- `ThreadCoordinationError` - Multi-threading coordination errors
- `TrainingError`, `TrainingStabilityError` - Training process errors
- `MemoryAllocationError` - Memory allocation failures

### Thread Health Monitoring

The `ThreadHealthMonitor` tracks thread failures with:
- Consecutive failure counting with configurable thresholds
- Exponential backoff with maximum limits
- Automatic thread termination for repeated failures
- Success tracking to reset failure counts

### GPU Operation Management

The `GPUOperationManager` provides:
- Operation timeout protection
- CUDA error categorization and handling
- Graceful degradation for GPU failures

### Model Validation

The `ModelValidator` ensures model integrity with:
- File validation (existence, size, checksum)
- Architecture validation (parameter counts, structure)
- Forward pass validation with test inputs
- Output validation (shapes, value ranges)
- Memory usage validation

## Running Error Handling Tests

### Unit Tests

Run the comprehensive error handling test suite:

```bash
# Run all error handling tests
python -m pytest tests/unit/test_error_handling.py -v

# Run specific test categories
python -m pytest tests/unit/test_error_handling.py -v -k "TestCustomExceptions"
python -m pytest tests/unit/test_error_handling.py -v -k "TestThreadHealthMonitor"
python -m pytest tests/unit/test_error_handling.py -v -k "TestGPUOperationManager"
python -m pytest tests/unit/test_error_handling.py -v -k "TestModelValidator"

# Run integration tests
python -m pytest tests/unit/test_error_handling.py -v -k "TestIntegrationErrorHandling"
```

### Manual Testing

Test the error handling framework components manually:

```python
# Test basic error handling
from src.utils.errors import AlphaZeroError, ErrorSeverity

error = AlphaZeroError("Test error", ErrorSeverity.ERROR, {"key": "value"})
print(f"Error: {error}")
print(f"Severity: {error.severity}")
```

```python
# Test thread health monitoring
from src.utils.errors import ThreadHealthMonitor

monitor = ThreadHealthMonitor(max_consecutive_failures=3, failure_backoff=0.1)

# Simulate failures
exception = Exception("Test failure")
for i in range(5):
    should_continue = monitor.record_failure("test_thread", exception)
    print(f"Failure {i+1}: should_continue={should_continue}")
    if not should_continue:
        break
```

```python
# Test GPU operation manager
from src.utils.errors import GPUOperationManager
import torch

manager = GPUOperationManager(default_timeout=5.0)

def safe_gpu_operation():
    return torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')

try:
    result = manager.execute_with_timeout(safe_gpu_operation, operation_name="test_operation")
    print(f"Operation successful: {result.shape}")
except Exception as e:
    print(f"Operation failed: {e}")
```

```python
# Test model validation
from src.neural.model_validator import create_gomoku_validator
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
validator = create_gomoku_validator(device)

# Test with a model file (if available)
try:
    result = validator.validate_model_from_path("path/to/model.pth")
    print("Model validation passed!")
except Exception as e:
    print(f"Model validation failed: {e}")
```

### Error Simulation Tests

Test error handling under simulated failure conditions:

```python
# Test search coordinator with simulated failures
from src.core.search_coordinator import SearchCoordinator
from src.utils.errors import InferenceError, ThreadHealthMonitor

# Create coordinator with enhanced error handling
# (Would need mock inference worker for complete test)

# Simulate inference failures
try:
    raise InferenceError("Simulated GPU failure", context={"batch_size": 32})
except InferenceError as e:
    print(f"Caught inference error: {e}")
    print(f"Context: {e.context}")
```

### Performance Impact Testing

Measure the performance impact of error handling:

```python
import time
from src.utils.errors import with_error_handling

@with_error_handling()
def test_function():
    return sum(range(1000000))

# Measure overhead
start_time = time.time()
result = test_function()
end_time = time.time()

print(f"Function result: {result}")
print(f"Execution time: {(end_time - start_time) * 1000:.2f}ms")
```

## Error Handling Integration Points

### Search Coordinator Integration

The search coordinator uses error handling for:
- Thread coordination failures
- Inference request processing errors
- Emergency shutdown procedures
- Thread health monitoring

### Training Pipeline Integration

The training pipeline integrates error handling for:
- Model loading and validation errors
- Training stability monitoring
- Checkpoint save/load failures
- GPU memory management

### MCTS Engine Integration

The MCTS engine uses error handling for:
- Tree corruption detection
- Memory allocation failures
- Node operation errors

## Configuration

Configure error handling behavior through environment variables or configuration files:

```python
# Example configuration
ERROR_HANDLING_CONFIG = {
    "thread_health": {
        "max_consecutive_failures": 5,
        "failure_backoff": 0.5,
        "max_backoff": 10.0
    },
    "gpu_operations": {
        "default_timeout": 30.0
    },
    "error_reporting": {
        "enable_metrics": True,
        "log_level": "INFO"
    }
}
```

## Troubleshooting

### Common Issues

**High error rates during normal operation:**
- Check error thresholds are appropriate for workload
- Verify system resources (memory, GPU availability)
- Review error logs for patterns

**Thread termination due to repeated failures:**
- Increase `max_consecutive_failures` threshold
- Check underlying causes of failures
- Verify backoff timing is appropriate

**GPU operation timeouts:**
- Increase timeout values for long-running operations
- Check GPU memory availability
- Verify CUDA driver stability

### Error Analysis

Use the centralized error reporter to analyze error patterns:

```python
from src.utils.errors import error_reporter

# Get error summary
summary = error_reporter.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Error types: {summary['error_counts']}")
print(f"Recent errors: {summary['recent_errors']}")
```

### Monitoring Integration

The error handling framework integrates with monitoring systems:
- Error counts and rates available as metrics
- Severity-based alerting
- Thread health status monitoring
- GPU operation failure tracking

---

*The error handling framework provides comprehensive protection for the AlphaZero engine, ensuring graceful degradation and recovery under adverse conditions while maintaining system stability and performance.*