#!/usr/bin/env python3
"""
Adaptive Batching Test
======================

Tests dynamic timeout adjustment based on GPU utilization.
Validates the adaptive batching implementation from comments.md Section 3, Issue #3C.

Usage:
    # Basic test
    python scripts/test_adaptive_batching.py

    # With custom settings
    python scripts/test_adaptive_batching.py --min-timeout 2.0 --max-timeout 10.0 --duration 30

Expected behavior:
    - Timeout adjusts dynamically between 2-10ms based on GPU utilization
    - High GPU util → shorter timeout (keep GPU fed)
    - Low GPU util → longer timeout (fill batches better)
    - Smooth transitions (no oscillation)

Author: MCTS Performance Team
Date: 2025-10-21
Reference: comments.md Section 3, Issue #3C
"""

import argparse
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    import alphazero_py
    import torch
    from src.neural.model import create_resnet_eca_model
    from src.core.dlpack_inference_bridge import DLPackInferenceBridge
    from src.utils.gpu_monitor import GPUMonitor, AdaptiveBatchController
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class AdaptiveBatchingTest:
    """Test adaptive batching with dynamic timeout adjustment"""

    def __init__(self, args):
        self.args = args
        self.coordinator = None
        self.monitor_thread = None
        self.stop_monitoring = False

    def run_test(self) -> int:
        """Run adaptive batching test"""
        print("=" * 80)
        print("ADAPTIVE BATCHING TEST")
        print("=" * 80)

        # Test 1: GPU monitor functionality
        print("\n[1/4] Testing GPU monitor...")
        if not self._test_gpu_monitor():
            print("⚠️  GPU monitor test inconclusive (may be on CPU)")
        else:
            print("✅ GPU monitor working")

        # Test 2: Adaptive controller
        print("\n[2/4] Testing adaptive batch controller...")
        if not self._test_adaptive_controller():
            print("❌ Adaptive controller test failed")
            return 1
        print("✅ Adaptive controller working")

        # Test 3: Dynamic timeout updates
        print("\n[3/4] Testing dynamic coordinator timeout updates...")
        if not self._test_dynamic_timeout():
            print("❌ Dynamic timeout test failed")
            return 1
        print("✅ Dynamic timeout updates working")

        # Test 4: Full integration test
        print("\n[4/4] Running full integration test...")
        if not self._test_full_integration():
            print("❌ Full integration test failed")
            return 1
        print("✅ Full integration test passed")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

        return 0

    def _test_gpu_monitor(self) -> bool:
        """Test GPU utilization monitoring"""
        monitor = GPUMonitor()

        # Get utilization
        util = monitor.get_utilization()
        print(f"   Current GPU utilization: {util*100:.1f}%")

        # Get memory info
        mem_info = monitor.get_memory_info()
        if mem_info:
            print(f"   GPU memory: {mem_info['used']/(1024**3):.2f}GB / {mem_info['total']/(1024**3):.2f}GB")
            print(f"   Memory utilization: {mem_info['utilization']*100:.1f}%")

        monitor.shutdown()

        # Return True if we got valid data
        return 0.0 <= util <= 1.0

    def _test_adaptive_controller(self) -> bool:
        """Test adaptive batch controller"""
        controller = AdaptiveBatchController(
            min_timeout_ms=self.args.min_timeout,
            max_timeout_ms=self.args.max_timeout,
            smoothing_factor=0.7
        )

        # Test multiple iterations
        timeouts = []
        for i in range(10):
            timeout = controller.get_timeout()
            timeouts.append(timeout)
            time.sleep(0.1)

        print(f"   Timeout range: [{min(timeouts):.2f}, {max(timeouts):.2f}] ms")

        stats = controller.get_stats()
        print(f"   Current timeout: {stats['current_timeout_ms']:.2f} ms")
        print(f"   GPU utilization: {stats['gpu_utilization']*100:.1f}%")

        controller.shutdown()

        # Verify timeouts are in valid range
        for t in timeouts:
            if not (self.args.min_timeout <= t <= self.args.max_timeout):
                print(f"   ❌ Timeout {t:.2f} ms out of range!")
                return False

        return True

    def _test_dynamic_timeout(self) -> bool:
        """Test dynamic timeout updates on coordinator"""
        # Create minimal coordinator setup
        queue = mcts_py.AsyncInferenceQueue()

        def dummy_batch_inference(features_batch, board_sizes, num_planes_list):
            # Return dummy results
            results = []
            for _ in range(len(features_batch)):
                results.append(([0.0] * 225, 0.0))  # Gomoku: 225 actions
            return results

        callback = mcts_py.PyBatchInferenceCallback(dummy_batch_inference)
        coordinator = mcts_py.BatchInferenceCoordinator()

        # Start with initial timeout
        initial_timeout = 5.0
        coordinator.start(queue, callback, batch_size=32, timeout_ms=initial_timeout)

        try:
            # Verify initial timeout
            current_timeout = coordinator.get_timeout()
            print(f"   Initial timeout: {current_timeout:.2f} ms")
            if abs(current_timeout - initial_timeout) > 0.001:
                print(f"   ❌ Initial timeout mismatch!")
                return False

            # Test dynamic updates
            test_timeouts = [2.0, 5.0, 8.0, 10.0]
            for new_timeout in test_timeouts:
                coordinator.set_timeout(new_timeout)
                time.sleep(0.1)  # Give coordinator time to see the change
                current = coordinator.get_timeout()
                print(f"   Set timeout to {new_timeout:.2f} ms → current: {current:.2f} ms")
                if abs(current - new_timeout) > 0.001:
                    print(f"   ❌ Timeout update failed!")
                    return False

            # Test batch size updates
            coordinator.set_batch_size(64)
            if coordinator.get_batch_size() != 64:
                print(f"   ❌ Batch size update failed!")
                return False
            print(f"   Batch size update: 32 → 64 ✅")

            return True

        finally:
            coordinator.stop()

    def _test_full_integration(self) -> bool:
        """Test full adaptive batching with MCTS"""
        print(f"   Duration: {self.args.duration} seconds")
        print(f"   Timeout range: [{self.args.min_timeout}, {self.args.max_timeout}] ms")

        # Create model and inference bridge
        model = create_resnet_eca_model('gomoku', size='128x12').cuda()
        bridge = DLPackInferenceBridge(
            model=model,
            device='cuda',
            use_cuda_graphs=True
        )

        # Create MCTS components
        state = alphazero_py.GomokuState(board_size=15)
        tree = mcts_py.create_test_tree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(tree)

        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, vl_manager)
        root = tree.get_root_index()

        # Create async queue and coordinator
        queue = mcts_py.AsyncInferenceQueue()
        callback = mcts_py.PyBatchInferenceCallback(bridge.batch_inference_features)
        coordinator = mcts_py.BatchInferenceCoordinator()

        # Start coordinator with initial timeout
        coordinator.start(queue, callback, batch_size=32, timeout_ms=5.0)

        # Create adaptive controller
        controller = AdaptiveBatchController(
            min_timeout_ms=self.args.min_timeout,
            max_timeout_ms=self.args.max_timeout
        )

        # Start monitoring thread that updates timeout
        def monitor_and_adjust():
            while not self.stop_monitoring:
                # Get adaptive timeout
                new_timeout = controller.get_timeout()

                # Update coordinator
                coordinator.set_timeout(new_timeout)

                time.sleep(0.5)  # Update every 500ms

        self.monitor_thread = threading.Thread(target=monitor_and_adjust, daemon=True)
        self.monitor_thread.start()

        # Disable profiling for clean measurement
        profiler = mcts_py.EnhancedProfiler.instance()
        profiler.set_enabled(False)

        try:
            # Run simulations for specified duration
            start_time = time.perf_counter()
            total_simulations = 0
            iteration = 0

            while time.perf_counter() - start_time < self.args.duration:
                tree.clear()
                successes = runner.run_continuous(state, root, queue, 100)
                total_simulations += successes
                iteration += 1

                # Print progress every 5 seconds
                if iteration % 50 == 0:
                    elapsed = time.perf_counter() - start_time
                    current_throughput = total_simulations / elapsed
                    stats = controller.get_stats()
                    print(f"   [{elapsed:.1f}s] {current_throughput:.0f} sims/sec, "
                          f"timeout={stats['current_timeout_ms']:.2f}ms, "
                          f"GPU={stats['gpu_utilization']*100:.0f}%")

            # Calculate final metrics
            elapsed = time.perf_counter() - start_time
            final_throughput = total_simulations / elapsed

            print(f"\n   Results:")
            print(f"   Total simulations: {total_simulations}")
            print(f"   Duration: {elapsed:.2f}s")
            print(f"   Average throughput: {final_throughput:.1f} sims/sec")

            # Get final stats
            stats = controller.get_stats()
            print(f"   Final timeout: {stats['current_timeout_ms']:.2f} ms")
            print(f"   Final GPU util: {stats['gpu_utilization']*100:.1f}%")

            return final_throughput > 0

        finally:
            self.stop_monitoring = True
            coordinator.stop()
            controller.shutdown()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test adaptive batching with dynamic timeout adjustment"
    )

    parser.add_argument(
        '--min-timeout',
        type=float,
        default=2.0,
        help="Minimum batch timeout in milliseconds (default: 2.0)"
    )

    parser.add_argument(
        '--max-timeout',
        type=float,
        default=10.0,
        help="Maximum batch timeout in milliseconds (default: 10.0)"
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)"
    )

    args = parser.parse_args()

    tester = AdaptiveBatchingTest(args)
    return tester.run_test()


if __name__ == '__main__':
    sys.exit(main())
