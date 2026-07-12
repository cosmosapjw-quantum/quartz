#!/usr/bin/env python3
"""
Phase 6 Integration Test: Multi-Process Inference (GIL Bypass)
==============================================================

Tests the complete multi-process architecture that bypasses Python GIL
by running inference in separate processes with shared memory tensors.

Success Criteria (SC-013 to SC-015):
- Throughput: Significant improvement over single process (≥1.5× for 2 workers)
- Python callback overhead minimized via shared memory
- GIL completely bypassed (each process has its own GIL)

Architecture Validated:
- Shared memory tensor handoff (zero-copy)
- Semaphore synchronization (no polling waste)
- Process pool management and clean shutdown
- Integration with full MCTS pipeline
"""

import pytest
import torch
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.multiprocess_inference import (
    MultiProcessInferenceManager,
    create_multiprocess_inference_callback
)
from src.core.search_coordinator import MultiCoordinatorManager
from src.neural.model import create_ghost_resnet_eca_model
import alphazero_py
import mcts_py


@pytest.fixture
def model_factory():
    """Factory function to create model (needed for multiprocessing)."""
    def create_model():
        model = create_ghost_resnet_eca_model('gomoku')
        return model
    return create_model


@pytest.mark.integration
def test_multiprocess_manager_lifecycle(model_factory):
    """Test multi-process manager initialization and shutdown."""
    manager = MultiProcessInferenceManager(
        model_factory=model_factory,
        num_workers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Start workers
    manager.start()
    assert manager.running
    assert len(manager.workers) == 2
    assert len(manager.slots) == 2

    # Give processes time to fully initialize
    time.sleep(3.0)

    # All processes should be alive
    for i, process in enumerate(manager.workers):
        assert process.is_alive(), f"Worker {i} (PID={process.pid}) died with exitcode={process.exitcode}"

    # Stop workers
    manager.stop()
    assert not manager.running
    assert len(manager.workers) == 0

    print("\n✅ Multi-process manager lifecycle test passed")


@pytest.mark.integration
def test_multiprocess_inference_correctness(model_factory):
    """Test that multi-process inference produces correct results."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create manager
    manager = MultiProcessInferenceManager(
        model_factory=model_factory,
        num_workers=1,  # Single worker for deterministic test
        device=device
    )
    manager.start()

    try:
        # Create test batch
        batch_size = 4
        features_batch = []
        for _ in range(batch_size):
            # Random Gomoku features (36 planes × 15×15)
            features = np.random.randn(36 * 15 * 15).astype(np.float32).tolist()
            features_batch.append(features)

        board_sizes = [15] * batch_size
        num_planes_list = [36] * batch_size

        # Run inference
        results = manager.batch_inference_features(features_batch, board_sizes, num_planes_list)

        # Validate results
        assert len(results) == batch_size
        for policy, value in results:
            assert len(policy) == 225  # Gomoku action space
            assert abs(sum(policy) - 1.0) < 1e-5  # Policy sums to 1
            assert -1.0 <= value <= 1.0  # Value in valid range

        print(f"\n✅ Multi-process inference correctness test passed")
        print(f"   Batch size: {batch_size}")
        print(f"   Policy sum (first): {sum(results[0][0]):.6f}")
        print(f"   Value range: [{min(v for _, v in results):.3f}, {max(v for _, v in results):.3f}]")

    finally:
        manager.stop()


@pytest.mark.integration
def test_full_mcts_with_multiprocess(model_factory):
    """Test full MCTS simulation with multi-process inference."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create game state and MCTS components
    state = alphazero_py.GomokuState()
    tree = mcts_py.MCTSTree(10000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
    runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)
    queue = mcts_py.AsyncInferenceQueue()

    # Create multi-process callback
    callback = create_multiprocess_inference_callback(
        model_factory=model_factory,
        num_workers=2,
        device=device
    )

    # Single coordinator (multi-process handles parallelism)
    coordinator = mcts_py.BatchInferenceCoordinator()
    coordinator.start(queue, callback, batch_size=16, timeout_ms=5.0)

    try:
        # Add root node
        root_idx = tree.add_root_node(0.0, 0)

        # Run simulations
        num_sims = 100
        start_time = time.perf_counter()
        completed = runner.run_continuous(state, root_idx, queue, num_sims)
        elapsed = time.perf_counter() - start_time

        # Validate
        assert completed == num_sims, f"Expected {num_sims} simulations, got {completed}"
        assert tree.get_node_count() > 1, "Tree should have grown"

        root_visits = tree.get_visit_count(root_idx)
        assert root_visits >= num_sims - 5, f"Root should have ~{num_sims} visits, got {root_visits}"

        throughput = num_sims / elapsed
        print(f"\n✅ Full MCTS with multi-process passed")
        print(f"   Throughput: {throughput:.1f} sims/sec ({elapsed:.3f}s for {num_sims} sims)")
        print(f"   Tree nodes: {tree.get_node_count()}")
        print(f"   Root visits: {root_visits}")

    finally:
        coordinator.stop()
        callback._manager.stop()


@pytest.mark.integration
def test_throughput_single_vs_multiprocess(model_factory):
    """Compare throughput: single process vs multi-process.

    This is the key validation that Phase 6 provides benefit by bypassing GIL.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_sims = 500

    # ===== BASELINE: SINGLE PROCESS (GIL-bound) =====
    print("\n" + "=" * 70)
    print("PHASE 6 VALIDATION: Single Process vs Multi-Process")
    print("=" * 70)

    state_single = alphazero_py.GomokuState()
    tree_single = mcts_py.MCTSTree(20000)
    selector_single = mcts_py.create_puct_selector()
    backup_single = mcts_py.create_backup_manager(tree_single)
    virtual_loss_single = mcts_py.create_test_virtual_loss_manager(tree_single)
    runner_single = mcts_py.ContinuousSimulationRunner(tree_single, selector_single, backup_single, virtual_loss_single)
    queue_single = mcts_py.AsyncInferenceQueue()

    # Single-process callback (baseline)
    model_single = model_factory()
    model_single = model_single.to(device)
    model_single.eval()

    def batch_callback_single(features_batch, board_sizes, num_planes_list):
        batch_size = len(features_batch)
        if batch_size == 0:
            return []

        tensors = []
        for features, board_size, num_planes in zip(features_batch, board_sizes, num_planes_list):
            features_np = np.array(features, dtype=np.float32).reshape(num_planes, board_size, board_size)
            tensors.append(features_np)

        features_tensor = torch.from_numpy(np.stack(tensors, axis=0)).to(device)

        with torch.inference_mode():
            policy_logits, values = model_single(features_tensor)
            policies = torch.softmax(policy_logits, dim=-1)

        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        return [(p.tolist(), float(v)) for p, v in zip(policies_np, values_np)]

    callback_single = mcts_py.PyBatchInferenceCallback(batch_callback_single)
    coordinator_single = mcts_py.BatchInferenceCoordinator()
    coordinator_single.start(queue_single, callback_single, 16, 5.0)

    root_idx_single = tree_single.add_root_node(0.0, 0)

    # Warmup
    runner_single.run_continuous(state_single, root_idx_single, queue_single, 50)
    time.sleep(0.1)

    # Measure
    start_time = time.perf_counter()
    completed_single = runner_single.run_continuous(state_single, root_idx_single, queue_single, num_sims)
    single_elapsed = time.perf_counter() - start_time

    coordinator_single.stop()

    assert completed_single == num_sims
    single_throughput = num_sims / single_elapsed

    # ===== MULTI-PROCESS (GIL-FREE) =====
    state_multi = alphazero_py.GomokuState()
    tree_multi = mcts_py.MCTSTree(20000)
    selector_multi = mcts_py.create_puct_selector()
    backup_multi = mcts_py.create_backup_manager(tree_multi)
    virtual_loss_multi = mcts_py.create_test_virtual_loss_manager(tree_multi)
    runner_multi = mcts_py.ContinuousSimulationRunner(tree_multi, selector_multi, backup_multi, virtual_loss_multi)
    queue_multi = mcts_py.AsyncInferenceQueue()

    # Multi-process callback (2 workers)
    callback_multi = create_multiprocess_inference_callback(
        model_factory=model_factory,
        num_workers=2,
        device=device
    )

    coordinator_multi = mcts_py.BatchInferenceCoordinator()
    coordinator_multi.start(queue_multi, callback_multi, 16, 5.0)

    root_idx_multi = tree_multi.add_root_node(0.0, 0)

    # Warmup
    runner_multi.run_continuous(state_multi, root_idx_multi, queue_multi, 50)
    time.sleep(0.1)

    # Measure
    start_time = time.perf_counter()
    completed_multi = runner_multi.run_continuous(state_multi, root_idx_multi, queue_multi, num_sims)
    multi_elapsed = time.perf_counter() - start_time

    coordinator_multi.stop()
    callback_multi._manager.stop()

    assert completed_multi == num_sims
    multi_throughput = num_sims / multi_elapsed

    # ===== ANALYSIS =====
    speedup = multi_throughput / single_throughput
    efficiency = (speedup / 2.0) * 100  # 2 workers

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({num_sims} simulations)")
    print(f"{'=' * 70}")
    print(f"Single Process:      {single_throughput:7.1f} sims/sec ({single_elapsed:.3f}s)")
    print(f"Multi-Process (N=2): {multi_throughput:7.1f} sims/sec ({multi_elapsed:.3f}s)")
    print(f"Speedup:             {speedup:.2f}×")
    print(f"Efficiency:          {efficiency:.1f}% (N=2 workers)")
    print(f"{'=' * 70}")

    # Success criteria: ≥1.5× speedup (GIL bypassed, true parallelism)
    assert speedup >= 1.5, f"Expected speedup ≥1.5× with multi-process, got {speedup:.2f}×"

    print(f"✅ SUCCESS: Multi-process achieved {speedup:.2f}× speedup (target: ≥1.5×)")
    print(f"   GIL bottleneck eliminated via separate processes")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
