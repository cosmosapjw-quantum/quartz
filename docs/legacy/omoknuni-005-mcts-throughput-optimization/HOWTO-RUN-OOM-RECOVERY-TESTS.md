# How to Run OOM Recovery Tests

## OOM Recovery Mechanisms (T050)

The OOM recovery system provides automatic handling of CUDA out-of-memory errors through batch size reduction, chunk processing, and graceful degradation to CPU fallback.

### Quick Commands

```bash
# Run OOM recovery unit tests
python -m pytest tests/unit/test_oom_recovery.py -v

# Run specific OOM test categories
python -m pytest tests/unit/test_oom_recovery.py -v -k "oom_detection"
python -m pytest tests/unit/test_oom_recovery.py -v -k "batch_size_reduction"
python -m pytest tests/unit/test_oom_recovery.py -v -k "integration"

# Run with GPU markers (requires CUDA)
python -m pytest tests/unit/test_oom_recovery.py -v -m "gpu"
```

### OOM Recovery Features

1. **Automatic Detection**: Identifies CUDA OOM errors from various error message patterns
2. **Batch Size Reduction**: Reduces batch size by 50% with minimum limits (1/16 original size)
3. **Chunk Processing**: Splits large batches into smaller chunks when OOM occurs
4. **Gradual Recovery**: Increases batch size gradually when conditions improve
5. **CPU Fallback**: Falls back to CPU inference when GPU recovery fails
6. **Memory Monitoring**: Tracks GPU memory usage and identifies high-risk conditions

### Configuration Parameters

**Default Settings:**
```python
batch_size_reduction_factor = 0.5      # Reduce by 50% on OOM
min_batch_size = original_size // 16    # Minimum batch size limit
oom_recovery_cooldown = 60.0           # Seconds before attempting increase
oom_memory_threshold = 0.9             # Memory usage >90% = high risk
max_consecutive_ooms = 3               # Fallback to CPU after 3 consecutive OOMs
```

### Testing OOM Scenarios

**Manual OOM Testing:**
```python
from src.neural.inference_worker import GPUInferenceWorker

# Create worker with small memory limit
worker = GPUInferenceWorker(
    model_path="models/gomoku.pth",
    batch_size=1024,  # Large batch to trigger OOM
    device="cuda:0"
)

# Generate large batch to trigger OOM
import numpy as np
positions = [np.random.randn(36, 15, 15) for _ in range(2048)]

# This should trigger OOM recovery
try:
    policies, values = worker.batch_inference(positions)
    print("OOM recovery successful!")
except Exception as e:
    print(f"OOM recovery failed: {e}")
```

### Monitoring OOM Recovery

**Check OOM Metrics:**
```python
metrics = worker.get_mixed_precision_metrics()
print(f"Total OOM events: {metrics['oom_total_count']}")
print(f"Consecutive OOMs: {metrics['oom_consecutive_count']}")
print(f"Current batch size: {worker.batch_size}")
print(f"Original batch size: {metrics['original_batch_size']}")
print(f"Memory usage: {metrics['memory_usage_fraction']:.2%}")
print(f"High risk memory: {metrics['memory_usage_high_risk']}")
```

**Log Output Example:**
```
WARNING:InferenceWorker[cuda:0]:CUDA OOM detected (#1, consecutive: 1)
INFO:InferenceWorker[cuda:0]:Cleared CUDA cache
WARNING:InferenceWorker[cuda:0]:Reduced batch size from 64 to 32 due to OOM
INFO:InferenceWorker[cuda:0]:Retrying inference with reduced batch size: 32
INFO:InferenceWorker[cuda:0]:Processing batch of 128 in chunks of 32
INFO:InferenceWorker[cuda:0]:Successfully processed 128 positions in 4 chunks
```

### Expected Behavior

**Successful OOM Recovery:**
1. OOM error detected during inference
2. CUDA cache cleared automatically
3. Batch size reduced (e.g., 64 → 32 → 16)
4. Pinned memory buffers recreated with smaller size
5. Large batches processed in chunks if needed
6. Gradual batch size recovery when memory allows

**CPU Fallback Scenario:**
1. Multiple consecutive OOM errors (3+)
2. Batch size reaches minimum limit
3. OOM recovery attempts fail
4. System falls back to CPU inference
5. Training/inference continues on CPU

### Performance Impact

**OOM Recovery Overhead:**
- Memory clearing: ~10-50ms per OOM event
- Batch size reduction: Immediate (parameter update)
- Buffer recreation: ~50-200ms depending on size
- Chunk processing: Linear scaling with number of chunks

**Memory Usage Reduction:**
- 50% reduction per OOM event
- Minimum 6.25% of original batch size (1/16)
- Gradual recovery over 60+ seconds
- Memory monitoring prevents premature increases

### Integration with Training

The OOM recovery system integrates seamlessly with:
- **MCTS Search**: Handles batch inference during tree search
- **Self-Play Training**: Manages memory during game generation
- **Model Evaluation**: Ensures stable performance during evaluation
- **CPU Fallback**: Works with existing CPU fallback mechanisms

### Troubleshooting

**Common Issues:**
```bash
# OOM recovery not working
# Check if recovery is enabled
python -c "
from src.neural.inference_worker import GPUInferenceWorker
worker = GPUInferenceWorker('model.pth')
print(f'OOM recovery enabled: {worker._oom_recovery_enabled}')
"

# Persistent OOM errors
# Check minimum batch size and memory usage
# Reduce initial batch size or increase GPU memory

# CPU fallback not triggering
# Check consecutive OOM limit and fallback configuration
```

**Debug Logging:**
```python
import logging
logging.getLogger('InferenceWorker').setLevel(logging.DEBUG)
```

---

*The OOM recovery system provides robust handling of memory constraints while maintaining training performance and system stability.*