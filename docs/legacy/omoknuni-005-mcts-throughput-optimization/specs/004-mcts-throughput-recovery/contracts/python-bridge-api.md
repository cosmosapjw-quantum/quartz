# DLPackInferenceBridge API Design

**Version**: 1.0
**Status**: Design Phase (T008a)
**Author**: Claude Code
**Date**: 2025-10-09

## Overview

The `DLPackInferenceBridge` class provides a zero-copy inference bridge between C++ MCTS simulation runner and Python GPU inference worker using DLPack tensors. This eliminates the numpy copy overhead in the critical inference path.

## Design Principles

1. **Zero-Copy**: Use DLPack tensors to avoid numpy array copies
2. **Backward Compatible**: Implement `BatchInferenceCallback` interface
3. **GPU-Aware**: Handle CPU↔GPU transfers efficiently
4. **Buffer Reuse**: Pre-allocate and reuse buffers for efficiency
5. **Error Resilient**: Graceful fallback on DLPack failures

## Class Interface

### DLPackInferenceBridge

```python
class DLPackInferenceBridge:
    """Zero-copy inference bridge using DLPack tensors.

    Implements BatchInferenceCallback interface for C++ MCTS integration.
    Uses DLPack protocol to eliminate numpy copy overhead.

    Architecture:
    1. C++ provides vector<IGameState*> to batch_inference()
    2. Create DLPack tensor via mcts_py.create_batch_tensor_from_states()
    3. Convert to PyTorch via torch.from_dlpack() (zero-copy)
    4. Run neural network inference on GPU
    5. Extract policy/value and return to C++

    Buffer Management:
    - DLPack tensors created on-demand (managed by C++ buffer pool)
    - PyTorch tensors reference DLPack memory (zero-copy)
    - Results returned as Python lists (minimal overhead)

    Args:
        model: PyTorch neural network model (nn.Module)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        enable_fallback: Enable numpy fallback if DLPack fails
        warmup_iterations: Number of warmup batches for GPU
    """

    def __init__(self,
                 model: torch.nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 enable_fallback: bool = True,
                 warmup_iterations: int = 5):
        ...

    def batch_inference(self,
                       states: List[IGameState]) -> List[Tuple[List[float], float]]:
        """Execute neural network inference for a batch of game states.

        This is the main entry point called by C++ BatchInferenceCoordinator.

        Flow:
        1. Validate inputs (non-empty, same game type)
        2. Create DLPack tensor from states
        3. Convert to PyTorch tensor (zero-copy)
        4. Transfer to GPU if needed (async copy)
        5. Run model forward pass
        6. Transfer results back to CPU
        7. Extract policy/value pairs
        8. Return as list of tuples

        Args:
            states: List of IGameState pointers from C++

        Returns:
            List[(policy, value)] where:
                policy: List[float] - action probabilities
                value: float - position evaluation

        Raises:
            ValueError: If states is empty or contains mixed game types
            RuntimeError: If DLPack conversion fails and fallback disabled
        """
        ...

    def warmup(self, batch_size: int = 64):
        """Warm up GPU with dummy batches.

        Runs several dummy inference batches to:
        - Initialize CUDA kernels
        - Allocate GPU memory
        - Prime memory pools
        - Measure baseline latency

        Args:
            batch_size: Size of warmup batches
        """
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with:
                - total_batches: Total batch_inference calls
                - total_states: Total states evaluated
                - avg_batch_size: Average batch size
                - dlpack_successes: DLPack path used
                - fallback_uses: Numpy fallback used
                - avg_latency_ms: Average inference latency
                - gpu_utilization: GPU utilization percentage (if available)
        """
        ...

    def reset_metrics(self):
        """Reset all performance metrics."""
        ...
```

## Buffer Management Strategy

### DLPack Buffer Creation

```python
# Created on-demand in batch_inference()
capsule = mcts_py.create_batch_tensor_from_states(states, use_cuda=False)
features = torch.from_dlpack(capsule)
```

**Benefits**:
- Zero-copy: PyTorch tensor shares memory with C++ buffer
- Automatic cleanup: DLPack handles memory lifecycle
- No pre-allocation needed: C++ buffer pool manages memory

**Lifetime**:
1. C++ creates buffer in `create_batch_tensor_from_states()`
2. DLPack capsule wraps buffer with deleter callback
3. PyTorch tensor references buffer (increments refcount)
4. When tensor is freed, DLPack deleter runs
5. C++ buffer pool reclaims memory

### GPU Transfer Strategy

```python
# Transfer to GPU (async if pinned memory)
if self.device.type == 'cuda':
    features_gpu = features.to(self.device, non_blocking=True)
else:
    features_gpu = features

# Run inference
with torch.no_grad():
    policy_logits, value = self.model(features_gpu)

# Transfer results back to CPU (async)
policy_cpu = policy_logits.cpu()
value_cpu = value.cpu()
```

**Optimizations**:
- `non_blocking=True`: Async GPU transfer if pinned memory
- `torch.no_grad()`: Disable gradient computation
- Batch CPU transfer: Single call for all results
- Stream synchronization: Implicit via CPU access

## Integration with BatchInferenceCoordinator

### C++ Side

```cpp
// In coordinator_loop()
auto batch = queue.collect_batch(batch_size_, timeout_ms_);
if (batch.empty()) continue;

// Extract state pointers
std::vector<const IGameState*> states;
for (const auto& req : batch) {
    states.push_back(req.state);
}

// Call Python inference bridge (acquires GIL once)
auto results = callback_.batch_inference(states);

// Submit results back to queue
queue.submit_results(batch, results);
```

### Python Side

```python
# DLPackInferenceBridge.batch_inference()
def batch_inference(self, states):
    # Create DLPack tensor (zero-copy)
    capsule = mcts_py.create_batch_tensor_from_states(states)
    features = torch.from_dlpack(capsule)

    # GPU inference
    features_gpu = features.to(self.device)
    with torch.no_grad():
        policy, value = self.model(features_gpu)

    # Extract results
    policy_cpu = policy.cpu().numpy()
    value_cpu = value.cpu().numpy()

    # Convert to Python list
    results = []
    for i in range(len(states)):
        policy_list = policy_cpu[i].tolist()
        value_scalar = float(value_cpu[i])
        results.append((policy_list, value_scalar))

    return results
```

## Error Handling and Fallback

### DLPack Failure Fallback

```python
def batch_inference(self, states):
    try:
        # Try DLPack path (zero-copy)
        return self._dlpack_inference(states)
    except Exception as e:
        if self.enable_fallback:
            self.logger.warning(f"DLPack failed: {e}, using numpy fallback")
            return self._numpy_fallback_inference(states)
        else:
            raise RuntimeError(f"DLPack inference failed: {e}")

def _numpy_fallback_inference(self, states):
    """Fallback to numpy array extraction."""
    # Extract features to numpy
    features_np = np.zeros(...)
    for i, state in enumerate(states):
        buffer = np.zeros(...)
        state.extract_features_to_buffer(buffer)
        features_np[i] = buffer.reshape(...)

    # Convert to torch (with copy)
    features = torch.from_numpy(features_np)

    # Continue with normal inference
    ...
```

## Performance Characteristics

### Expected Latency Breakdown (Batch Size 64, Gomoku)

| Operation | Time | Percentage |
|-----------|------|------------|
| DLPack tensor creation | 6.8 ms | 45% |
| PyTorch conversion (zero-copy) | 0.01 ms | <1% |
| GPU transfer (H→D) | 0.5 ms | 3% |
| Neural network inference | 7.0 ms | 46% |
| GPU transfer (D→H) | 0.3 ms | 2% |
| Result extraction | 0.5 ms | 3% |
| **Total** | **15.1 ms** | **100%** |

**Notes**:
- DLPack tensor creation includes feature extraction (~95% of time)
- Zero-copy eliminates ~0.5ms numpy copy overhead
- GPU transfers can overlap with computation (async)

### Comparison vs Numpy Baseline

| Metric | Numpy Baseline | DLPack | Improvement |
|--------|---------------|---------|-------------|
| Batch 64 latency | 15.5 ms | 15.1 ms | 1.03× faster |
| Memory allocations | 2 (numpy + torch) | 1 (DLPack) | 50% reduction |
| Memory copies | 1 (numpy→torch) | 0 | Eliminated |
| Code complexity | Medium | Low | Simpler |

## Usage Example

```python
import torch
from src.neural.gomoku_net import GomokuNet
from src.core.dlpack_inference_bridge import DLPackInferenceBridge

# Create model
model = GomokuNet(num_blocks=20, num_channels=256)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model = model.cuda()

# Create bridge
bridge = DLPackInferenceBridge(
    model=model,
    device='cuda',
    enable_fallback=True,
    warmup_iterations=10
)

# Warmup
bridge.warmup(batch_size=64)

# Use with C++ coordinator
# (C++ code calls bridge.batch_inference() automatically)

# Check metrics
metrics = bridge.get_metrics()
print(f"Avg batch size: {metrics['avg_batch_size']}")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f} ms")
print(f"DLPack success rate: {metrics['dlpack_successes'] / metrics['total_batches']:.1%}")
```

## Testing Strategy

### Unit Tests

1. **Initialization**: Test device selection, warmup
2. **DLPack Conversion**: Verify zero-copy, correct shapes
3. **Batch Inference**: Test various batch sizes (1, 16, 32, 64, 128)
4. **Error Handling**: Test fallback, invalid inputs
5. **Metrics**: Verify tracking accuracy

### Integration Tests

1. **C++ Integration**: Test with BatchInferenceCoordinator
2. **Training Loop**: Verify gradient-free inference
3. **Mixed Precision**: Test fp16 on GPU
4. **Multi-Game**: Test Gomoku, Chess, Go

### Performance Tests

1. **Latency Benchmarks**: Compare vs numpy baseline
2. **Memory Profiling**: Verify no extra allocations
3. **GPU Utilization**: Measure utilization during inference
4. **Scalability**: Test batch sizes 1-256

## Implementation Notes

### Thread Safety

- `batch_inference()` is called from C++ background thread
- Must be thread-safe with proper GIL handling
- PyTorch operations are thread-safe (GIL released in C++ operations)

### Memory Management

- DLPack handles tensor lifecycle automatically
- No manual cleanup needed
- Buffer pool in C++ manages underlying memory

### Device Placement

- Input tensors always on CPU (DLPack from C++ pinned memory)
- Transfer to GPU if `device='cuda'`
- Results transferred back to CPU before return

## Open Questions

None - design is complete and ready for implementation.

## Acceptance Criteria

- [✅] Class interface defined with clear method signatures
- [✅] Buffer management strategy documented
- [✅] GPU transfer pipeline specified
- [✅] Integration with BatchInferenceCoordinator explained
- [✅] Error handling and fallback strategy defined
- [✅] Performance characteristics estimated
- [✅] Testing strategy outlined
- [✅] Usage examples provided

## Next Steps

Proceed to T008b: Implement `DLPackInferenceBridge` class based on this design.
