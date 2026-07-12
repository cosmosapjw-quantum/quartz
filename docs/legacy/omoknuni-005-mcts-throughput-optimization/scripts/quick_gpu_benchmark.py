#!/usr/bin/env python3
"""Quick GPU benchmark with async inference."""

import sys
import os
sys.path.insert(0, '/home/cosmosapjw/omoknuni')
sys.path.insert(0, '/home/cosmosapjw/omoknuni/src')

import torch
import time
import tempfile
import alphazero_py
from core.mcts import AlphaZeroMCTS
from neural.inference_worker import GPUInferenceWorker
from neural.model import AlphaZeroNet

# Create small test model
model = AlphaZeroNet(
    input_channels=36,
    num_actions=225,
    num_blocks=4,
    hidden_channels=128,
    use_se=False
)

with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
    model_path = f.name
torch.save(model, model_path)

print("=" * 80)
print("GPU ASYNC INFERENCE BENCHMARK")
print("=" * 80)

# Test configurations
configs = [
    ("1 thread", 1),
    ("4 threads", 4),
    ("8 threads", 8),
]

state = alphazero_py.GomokuState()

# Create GPU worker
gpu_worker = GPUInferenceWorker(
    model_path=model_path,
    device='cuda',
    batch_size=64,
    timeout_ms=1.0,
    use_mixed_precision=True
)
gpu_worker.warmup(input_shape=(36, 15, 15))

print(f"\nConfiguration: batch_size=64, timeout=1.0ms, async=True")
print(f"{'Threads':>10s} {'Mode':>20s} {'Throughput':>15s} {'Time':>10s}")
print("-" * 60)

for label, num_threads in configs:
    for mode in ["shared", "virtual_loss_free"]:
        try:
            mcts = AlphaZeroMCTS(
                inference_fn=gpu_worker,
                use_async_inference=True,
                async_batch_size=64,
                async_timeout_ms=1.0,
                num_threads=num_threads,
                parallel_mode=mode,
                enable_instrumentation=True
            )

            # Reset instrumentation
            mcts.reset_instrumentation_metrics()

            # Run benchmark
            start = time.perf_counter()
            mcts.search(state, simulations=1000)
            elapsed = time.perf_counter() - start

            throughput = 1000 / elapsed

            print(f"{num_threads:10d} {mode:>20s} {throughput:12.1f}/s {elapsed:10.3f}s")

            # Get instrumentation data
            stats = mcts.get_statistics()
            if 'instrumentation' in stats and stats['instrumentation']:
                instr = stats['instrumentation']
                print(f"           Instrumentation:")
                for metric_name, data in sorted(instr.items()):
                    if data['calls'] > 0:
                        print(f"             {metric_name:30s}: {data['calls']:8d} calls, {data['avg_ns']/1000:8.2f} μs/call")

            mcts.close()

        except Exception as e:
            print(f"{num_threads:10d} {mode:>20s} ERROR: {e}")
            import traceback
            traceback.print_exc()

gpu_worker.stop_worker()
os.unlink(model_path)

print("\n" + "=" * 80)
