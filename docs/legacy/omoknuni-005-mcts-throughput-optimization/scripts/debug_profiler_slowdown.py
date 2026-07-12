#!/usr/bin/env python3
"""
Debug script to isolate the 7× slowdown in unified profiler vs wall-clock.

This script will incrementally add components from unified profiler to see
which one causes the slowdown.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import mcts_py
import alphazero_py
from src.neural.model import create_random_model
from src.core.dlpack_inference_bridge import DLPackInferenceBridge

def run_test(name: str, setup_fn, simulations: int = 1000):
    """Run a test with given setup and measure throughput."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    # Common setup (EXACTLY like wall-clock validation)
    state = alphazero_py.GomokuState(board_size=15)
    action_space_size = state.get_action_space_size()
    tree = mcts_py.create_test_tree(100000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    vl_manager = mcts_py.create_test_virtual_loss_manager(tree)
    runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, vl_manager)
    root = tree.get_root_index()

    # Create AsyncInferenceQueue
    queue = mcts_py.AsyncInferenceQueue()

    # Custom setup from test
    callback, coordinator = setup_fn()

    # Start coordinator
    coordinator.start(queue, callback, 64, 5.0)  # batch_size=64, timeout=5ms

    # Run and measure
    start = time.perf_counter()
    completed = runner.run_continuous(state, root, queue, simulations)
    elapsed = time.perf_counter() - start

    coordinator.stop()

    throughput = completed / elapsed
    print(f"✅ Throughput: {throughput:.1f} sims/sec ({elapsed:.3f}s for {completed} sims)")

    return throughput

# Test 1: Dummy callback (baseline)
def test_dummy():
    """Baseline: dummy callback with uniform policy."""
    def dummy_fn(features, board_sizes, planes):
        return [([0.01] * 225, 0.0) for _ in features]

    callback = mcts_py.PyBatchInferenceCallback(dummy_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()
    return callback, coordinator

# Test 2: Real GPU model (like wall-clock)
def test_real_gpu_simple():
    """Real GPU with minimal setup."""
    model = create_random_model('gomoku', seed=42)
    device = 'cuda'
    model = model.to(device)
    model.eval()

    # Simple inference function
    def inference_fn(features, board_sizes, planes):
        # Convert to tensor
        import numpy as np
        batch_size = len(features)
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        tensor = tensor.reshape(batch_size, planes[0], board_sizes[0], board_sizes[0])

        with torch.no_grad():
            policy_logits, value = model(tensor)
            policy = torch.softmax(policy_logits, dim=1)

        # Convert back to lists
        results = []
        for i in range(batch_size):
            policy_list = policy[i].cpu().numpy().tolist()
            value_scalar = float(value[i].item())
            results.append((policy_list, value_scalar))

        return results

    callback = mcts_py.PyBatchInferenceCallback(inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()
    return callback, coordinator

# Test 3: DLPackInferenceBridge (like unified profiler)
def test_dlpack_bridge():
    """DLPackInferenceBridge (unified profiler setup)."""
    model = create_random_model('gomoku', seed=42)
    device = 'cuda'

    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    inference_bridge.warmup(batch_size=64, game_type='gomoku')

    def batch_inference_fn(features_batch, board_sizes, num_planes_list):
        return inference_bridge.batch_inference_features(
            features_batch, board_sizes, num_planes_list
        )

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()
    return callback, coordinator

# Test 4: DLPack with ProfilingSession (unified profiler full setup)
def test_dlpack_with_profiling_session():
    """DLPackInferenceBridge + ProfilingSession (all disabled)."""
    from src.profiling import ProfilingSession, ProfilerConfig

    # Start profiling session (all disabled)
    config = ProfilerConfig(
        enable_gil_profiling=False,
        enable_inference_profiling=False,
        enable_cpp_instrumentation=False,
        enable_thread_profiling=False,
        enable_memory_profiling=False
    )

    session = ProfilingSession(config)
    session.__enter__()

    model = create_random_model('gomoku', seed=42)
    device = 'cuda'

    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    inference_bridge.warmup(batch_size=64, game_type='gomoku')

    def batch_inference_fn(features_batch, board_sizes, num_planes_list):
        return inference_bridge.batch_inference_features(
            features_batch, board_sizes, num_planes_list
        )

    callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
    coordinator = mcts_py.BatchInferenceCoordinator()
    return callback, coordinator

if __name__ == '__main__':
    print("="*60)
    print("PROFILER SLOWDOWN DEBUG")
    print("="*60)
    print("\nGoal: Find why unified profiler is 7× slower than wall-clock")
    print("Expected: ~7,000 sims/sec")

    results = {}

    # Run tests
    results['dummy'] = run_test("Dummy callback (baseline)", test_dummy, simulations=1000)
    results['simple_gpu'] = run_test("Simple GPU inference", test_real_gpu_simple, simulations=1000)
    results['dlpack'] = run_test("DLPackInferenceBridge", test_dlpack_bridge, simulations=1000)
    results['dlpack_session'] = run_test("DLPack + ProfilingSession", test_dlpack_with_profiling_session, simulations=1000)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, throughput in results.items():
        status = "✅" if throughput > 5000 else "❌"
        print(f"{status} {name:20s}: {throughput:7.1f} sims/sec")

    # Find the culprit
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if results['simple_gpu'] < 5000:
        print("🔴 Simple GPU inference is slow - problem is in basic model setup")
    elif results['dlpack'] < 5000:
        print("🔴 DLPackInferenceBridge is slow - problem is in DLPack bridge")
    elif results['dlpack_session'] < 5000:
        print("🔴 ProfilingSession is slow - problem is in profiling session setup")
    else:
        print("✅ All tests fast - problem must be elsewhere in unified profiler")
