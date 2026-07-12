#!/usr/bin/env python3
"""
Adaptive Batching API Test
==========================

Quick test to validate adaptive batching API changes:
- BatchInferenceCoordinator.set_timeout()
- BatchInferenceCoordinator.get_timeout()
- GPUMonitor and AdaptiveBatchController

Usage:
    python scripts/test_adaptive_api.py

Expected: All tests pass
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    from src.utils.gpu_monitor import GPUMonitor, AdaptiveBatchController
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_gpu_monitor():
    """Test GPU monitor functionality"""
    print("\n[1/3] Testing GPUMonitor...")

    monitor = GPUMonitor()

    # Test get_utilization()
    util = monitor.get_utilization()
    print(f"   GPU utilization: {util*100:.1f}%")

    # Verify range
    assert 0.0 <= util <= 1.0, f"Invalid utilization: {util}"

    # Test get_memory_info()
    mem_info = monitor.get_memory_info()
    if mem_info:
        print(f"   GPU memory: {mem_info['used']/(1024**3):.2f}GB / {mem_info['total']/(1024**3):.2f}GB")
    else:
        print(f"   Memory info not available (may be on CPU)")

    monitor.shutdown()
    print("   ✅ GPUMonitor passed")


def test_adaptive_controller():
    """Test adaptive batch controller"""
    print("\n[2/3] Testing AdaptiveBatchController...")

    controller = AdaptiveBatchController(
        min_timeout_ms=2.0,
        max_timeout_ms=10.0,
        smoothing_factor=0.7
    )

    # Get several timeouts
    timeouts = []
    for i in range(5):
        timeout = controller.get_timeout()
        timeouts.append(timeout)
        print(f"   Iteration {i+1}: timeout = {timeout:.2f} ms")

    # Verify range
    for t in timeouts:
        assert 2.0 <= t <= 10.0, f"Timeout {t:.2f} out of range [2.0, 10.0]"

    # Get stats
    stats = controller.get_stats()
    print(f"   Stats: timeout={stats['current_timeout_ms']:.2f}ms, GPU util={stats['gpu_utilization']*100:.1f}%")

    controller.shutdown()
    print("   ✅ AdaptiveBatchController passed")


def test_coordinator_dynamic_timeout():
    """Test coordinator dynamic timeout updates"""
    print("\n[3/3] Testing BatchInferenceCoordinator dynamic timeout...")

    queue = mcts_py.AsyncInferenceQueue()

    def dummy_batch_inference(features_batch, board_sizes, num_planes_list):
        # Return dummy results
        results = []
        for _ in range(len(features_batch)):
            results.append(([0.0] * 225, 0.0))  # Gomoku: 225 actions
        return results

    callback = mcts_py.PyBatchInferenceCallback(dummy_batch_inference)
    coordinator = mcts_py.BatchInferenceCoordinator()

    # Start with initial values
    initial_timeout = 5.0
    initial_batch_size = 32
    coordinator.start(queue, callback, batch_size=initial_batch_size, timeout_ms=initial_timeout)

    try:
        # Test get_timeout()
        current_timeout = coordinator.get_timeout()
        print(f"   Initial timeout: {current_timeout:.2f} ms")
        assert abs(current_timeout - initial_timeout) < 0.01, f"Timeout mismatch: {current_timeout} vs {initial_timeout}"

        # Test set_timeout()
        new_timeout = 8.0
        coordinator.set_timeout(new_timeout)
        updated_timeout = coordinator.get_timeout()
        print(f"   After set_timeout({new_timeout}): {updated_timeout:.2f} ms")
        assert abs(updated_timeout - new_timeout) < 0.01, f"Timeout not updated: {updated_timeout} vs {new_timeout}"

        # Test multiple updates
        test_timeouts = [2.0, 10.0, 5.0]
        for t in test_timeouts:
            coordinator.set_timeout(t)
            current = coordinator.get_timeout()
            print(f"   Set to {t:.2f} ms → got {current:.2f} ms")
            assert abs(current - t) < 0.01, f"Timeout mismatch"

        # Test batch size updates
        current_batch = coordinator.get_batch_size()
        print(f"   Initial batch size: {current_batch}")
        assert current_batch == initial_batch_size

        coordinator.set_batch_size(64)
        new_batch = coordinator.get_batch_size()
        print(f"   After set_batch_size(64): {new_batch}")
        assert new_batch == 64

        print("   ✅ Dynamic timeout/batch_size updates working")

    finally:
        coordinator.stop()


def main():
    print("=" * 80)
    print("ADAPTIVE BATCHING API TEST")
    print("=" * 80)

    try:
        test_gpu_monitor()
        test_adaptive_controller()
        test_coordinator_dynamic_timeout()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
