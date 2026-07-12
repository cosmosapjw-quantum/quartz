"""
CRITICAL TEST: Validate ContinuousSimulationRunner make/unmake pattern (T024f-6).

This test ACTUALLY exercises the make/unmake code path, unlike the previous tests
which incorrectly tested SimulationRunner (the old clone-based implementation).

This is a BLOCKING test - must pass before T024f-6 can be considered complete.
"""

import pytest
import time
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import alphazero_py
import mcts_py


class TestContinuousRunnerMakeUnmake:
    """CRITICAL: Test the ACTUAL make/unmake implementation in ContinuousSimulationRunner."""

    def test_basic_continuous_simulation(self):
        """Test that ContinuousSimulationRunner completes successfully with make/unmake."""
        # Create game state
        state = alphazero_py.GomokuState()

        # Create MCTS components
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

        # Create CONTINUOUS simulation runner (NOT base SimulationRunner!)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)

        # Create async inference queue
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch inference callback
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            results = []
            for _ in features_batch:
                policy = np.ones(225, dtype=np.float32) / 225.0
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()

        # Start coordinator
        coordinator.start(queue, callback, 8, 5.0)

        try:
            # Add root node
            root_idx = tree.add_root_node(0.5, 0)

            # Run continuous simulations (THIS uses make/unmake!)
            num_sims = 100
            completed = runner.run_continuous(state, root_idx, queue, num_sims)

            # Validate
            assert completed == num_sims, f"Expected {num_sims} simulations, got {completed}"
            assert tree.get_node_count() > 1, "Tree should have grown"

            # CRITICAL: Verify root visit count
            root_visits = tree.get_visit_count(root_idx)
            assert root_visits >= num_sims - 5, \
                f"Root should have ~{num_sims} visits, got {root_visits}"

        finally:
            coordinator.stop()

    def test_state_restoration_with_make_unmake(self):
        """CRITICAL: Test that thread-local state is correctly restored to root."""
        state = alphazero_py.GomokuState()
        initial_hash = state.zobrist_hash()

        # Create components
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch inference callback
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            results = []
            for _ in features_batch:
                policy = np.ones(225, dtype=np.float32) / 225.0
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, 8, 5.0)

        try:
            # Add root node
            root_idx = tree.add_root_node(0.5, 0)

            # Run multiple batches of simulations
            for batch in range(5):
                # State hash should always match initial (thread-local state restored)
                current_hash = state.zobrist_hash()
                assert current_hash == initial_hash, \
                    f"Batch {batch}: State hash changed! {current_hash} != {initial_hash}"

                # Run simulations
                completed = runner.run_continuous(state, root_idx, queue, 20)
                assert completed == 20, f"Batch {batch}: Expected 20 simulations, got {completed}"

            # Final check
            assert state.zobrist_hash() == initial_hash, \
                "State should be restored to root after all simulations"

        finally:
            coordinator.stop()

    def test_make_unmake_correctness_under_load(self):
        """CRITICAL: Stress test make/unmake under high simulation load."""
        state = alphazero_py.GomokuState()
        initial_hash = state.zobrist_hash()

        tree = mcts_py.MCTSTree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch inference callback
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            results = []
            for _ in features_batch:
                policy = np.ones(225, dtype=np.float32) / 225.0
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, 8, 5.0)

        try:
            # Add root node
            root_idx = tree.add_root_node(0.5, 0)

            # Run large number of simulations
            num_sims = 1000
            completed = runner.run_continuous(state, root_idx, queue, num_sims)

            # Validate
            assert completed == num_sims, f"Expected {num_sims}, completed {completed}"
            assert state.zobrist_hash() == initial_hash, \
                "State should be restored to root after 1000 simulations"

            # Verify tree statistics
            root_visits = tree.get_visit_count(root_idx)
            assert root_visits >= num_sims - 10, \
                f"Root visits ({root_visits}) should be ~{num_sims}"

            # Verify tree has reasonable structure
            node_count = tree.get_node_count()
            assert node_count > 100, f"Tree should have >100 nodes, got {node_count}"
            assert node_count < 100000, f"Tree shouldn't be excessive, got {node_count}"

        finally:
            coordinator.stop()

    def test_make_unmake_debug_assertions(self):
        """
        CRITICAL: Verify debug assertions catch illegal moves (if any).

        This test passes if no assertions fire. If debug assertions are triggered,
        it means make/unmake has a bug and state/tree are out of sync.
        """
        state = alphazero_py.GomokuState()

        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch inference callback
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            results = []
            for _ in features_batch:
                policy = np.ones(225, dtype=np.float32) / 225.0
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, 8, 5.0)

        try:
            # Add root node
            root_idx = tree.add_root_node(0.5, 0)

            # Run simulations - if debug assertions trigger, this will raise
            try:
                completed = runner.run_continuous(state, root_idx, queue, 500)
                assert completed == 500, "All simulations should complete"
            except RuntimeError as e:
                if "CRITICAL BUG" in str(e):
                    pytest.fail(f"Debug assertion caught bug in make/unmake: {e}")
                raise  # Re-raise if different error

        finally:
            coordinator.stop()

    def test_performance_with_make_unmake(self):
        """Validate performance improvement with make/unmake pattern."""
        state = alphazero_py.GomokuState()

        tree = mcts_py.MCTSTree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.ContinuousSimulationRunner(tree, selector, backup, virtual_loss)
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch inference callback
        def batch_inference_fn(features_batch, board_sizes, num_planes_list):
            results = []
            for _ in features_batch:
                policy = np.ones(225, dtype=np.float32) / 225.0
                value = 0.0
                results.append((policy.tolist(), value))
            return results

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()
        coordinator.start(queue, callback, 8, 5.0)

        try:
            # Add root node
            root_idx = tree.add_root_node(0.5, 0)

            # Warmup
            runner.run_continuous(state, root_idx, queue, 50)

            # Benchmark
            num_sims = 1000
            start = time.perf_counter()
            completed = runner.run_continuous(state, root_idx, queue, num_sims)
            elapsed = time.perf_counter() - start

            sims_per_sec = completed / elapsed
            time_per_sim_us = (elapsed / completed) * 1e6

            print(f"\n  ContinuousSimulationRunner Performance (with make/unmake):")
            print(f"    {sims_per_sec:.0f} sims/sec")
            print(f"    {time_per_sim_us:.1f} μs per simulation")
            print(f"    Total: {elapsed*1000:.1f} ms for {completed} simulations")

            # With make/unmake, we expect better performance than old approach
            # This is synchronous (no GPU batching), so expectations are modest
            assert sims_per_sec > 300, \
                f"Performance regression: {sims_per_sec:.0f} sims/sec (expected >300)"

        finally:
            coordinator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
