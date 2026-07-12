"""
Contract API: C++ MCTS Simulation Runner

This module defines the contract interface for the SimulationRunner class,
which must be implemented in C++ and exposed via pybind11.

Status: DRAFT
Created: 2025-10-02
"""

from typing import Callable, Tuple
from abc import ABC, abstractmethod
import numpy as np


# ==============================================================================
# C++ Class Interfaces (exposed via pybind11)
# ==============================================================================

class SimulationRunnerContract(ABC):
    """
    Contract for the C++ SimulationRunner class.

    The SimulationRunner executes complete MCTS simulations in C++ with
    minimal Python interaction, releasing the GIL for true parallel execution.

    Implementation Location: cpp_extensions/mcts/simulation_runner.cpp
    Python Binding: cpp_extensions/mcts/python_bindings.cpp
    """

    @abstractmethod
    def __init__(self,
                 tree,  # MCTSTree instance
                 selector,  # PUCTSelector instance
                 backup,  # BackupManager instance
                 virtual_loss):  # VirtualLossManager instance
        """
        Initialize SimulationRunner with references to shared components.

        Parameters:
        -----------
        tree : MCTSTree
            Shared MCTS tree (modified in-place during simulations)
        selector : PUCTSelector
            PUCT selector for child selection (stateless)
        backup : BackupManager
            Backup manager for value propagation
        virtual_loss : VirtualLossManager
            Virtual loss manager for thread coordination

        Notes:
        ------
        - All parameters are REFERENCES to existing objects, not owned by runner
        - Runner is thread-safe (each thread has its own instance)
        - Path buffer pre-allocated with 256 capacity for efficiency
        """
        pass

    @abstractmethod
    def run_simulation(self,
                      state,  # IGameState instance
                      root_index: int,
                      inference_callback: Callable) -> bool:
        """
        Execute a complete MCTS simulation (selection → expansion → backup).

        This method releases the GIL for the ENTIRE simulation, only reacquiring
        it when calling the inference callback for neural network evaluation.

        Parameters:
        -----------
        state : IGameState
            Root game state (will be cloned for traversal)
        root_index : int (NodeIndex)
            Index of root node in the tree
        inference_callback : Callable[[IGameState], Tuple[np.ndarray, float]]
            Python callback for neural network inference.
            Signature: (state) -> (policy, value)
            - policy: np.ndarray of shape (action_space_size,)
            - value: float in range [-1, 1]

        Returns:
        --------
        success : bool
            True if simulation completed successfully, False on error

        GIL Management:
        ---------------
        - GIL released at start of method
        - GIL automatically reacquired when calling inference_callback
        - GIL remains released during selection and backup
        - Total GIL cycles: 1-2 per simulation (vs 100-200 in Python loop)

        Thread Safety:
        --------------
        - Safe to call concurrently from multiple threads
        - Each thread operates on independent game state clone
        - Tree updates use atomic operations
        - Virtual loss prevents duplicate node expansion

        Performance:
        ------------
        - Target: ~4,000 sims/sec per thread
        - With 8 threads: ~30,000 sims/sec total (75-85% efficiency)
        - Memory: One game state clone per simulation (~32-64 bytes)

        Error Handling:
        ---------------
        - Returns False on any C++ exception
        - Does not propagate exceptions to Python
        - Logs errors internally (TODO: integrate with Python logger)

        Example:
        --------
        >>> def inference_fn(state):
        ...     policy = np.ones(362) / 362  # Uniform policy
        ...     value = 0.0
        ...     return (policy, value)
        ...
        >>> runner = mcts_py.SimulationRunner(tree, selector, backup, vl)
        >>> success = runner.run_simulation(game_state, root_idx, inference_fn)
        >>> assert success
        """
        pass


class InferenceCallbackContract(ABC):
    """
    Contract for the C++ InferenceCallback abstract interface.

    Python callbacks are wrapped in PyInferenceCallback which implements
    this interface. The callback is invoked during the expansion phase.

    Implementation Location: cpp_extensions/mcts/simulation_runner.hpp
    Python Wrapper: PyInferenceCallback in python_bindings.cpp
    """

    @abstractmethod
    def request_inference(self, state) -> Tuple[np.ndarray, float]:
        """
        Request neural network inference for a game state.

        This method is called from C++ during the expansion phase.
        pybind11 automatically acquires the GIL before invoking the Python
        callback, so no manual GIL management is required.

        Parameters:
        -----------
        state : IGameState
            Game state to evaluate (read-only)

        Returns:
        --------
        policy : np.ndarray
            Policy vector of shape (action_space_size,)
            Values should sum to ~1.0 (will be masked and renormalized)
        value : float
            Value estimate in range [-1, 1] from current player's perspective

        Performance:
        ------------
        - Called once per simulation (during expansion)
        - Should complete in <5ms for real-time search
        - Batching handled by GPUInferenceWorker (outside this interface)

        Thread Safety:
        --------------
        - GIL automatically acquired by pybind11
        - Safe to call from multiple threads sequentially
        - Python callback must be thread-safe if shared

        Example:
        --------
        >>> class MyInference(InferenceCallback):
        ...     def request_inference(self, state):
        ...         features = state.get_tensor_representation()
        ...         policy, value = model(features)
        ...         return (policy.numpy(), float(value))
        """
        pass


# ==============================================================================
# Extended Tree API (Move Storage)
# ==============================================================================

class MCTSTreeMovesExtension(ABC):
    """
    Extended MCTSTree API for move storage.

    These methods must be added to the existing MCTSTree class to eliminate
    the Python _move_mapping dictionary.

    Implementation Location: cpp_extensions/mcts/tree.cpp
    """

    @abstractmethod
    def get_move(self, index: int) -> int:
        """
        Get the move that led to a specific node.

        Parameters:
        -----------
        index : int (NodeIndex)
            Index of the node

        Returns:
        --------
        move : int
            Move index in range [0, action_space_size)
            Returns 0 for root node (no move)

        Performance:
        ------------
        - O(1) array access
        - No GIL acquisition (NoGil() guard)

        Thread Safety:
        --------------
        - Read-only operation (safe from any thread)
        - Move set during expansion, read during selection
        """
        pass

    @abstractmethod
    def set_move(self, index: int, move: int) -> None:
        """
        Set the move that led to a specific node.

        Parameters:
        -----------
        index : int (NodeIndex)
            Index of the node
        move : int
            Move index in range [0, action_space_size)

        Performance:
        ------------
        - O(1) array write
        - No GIL acquisition (NoGil() guard)

        Thread Safety:
        --------------
        - Write-once operation (set during expansion)
        - Safe because each node expanded by only one thread
        """
        pass


# ==============================================================================
# Integration API (Python MCTS Class)
# ==============================================================================

class AlphaZeroMCTSIntegration(ABC):
    """
    Contract for integrating C++ SimulationRunner into AlphaZeroMCTS.

    Implementation Location: src/core/mcts.py
    """

    @abstractmethod
    def __init__(self, ..., use_cpp_runner: bool = True):
        """
        Initialize MCTS with optional C++ simulation runner.

        Parameters:
        -----------
        use_cpp_runner : bool, default=True
            If True, use C++ SimulationRunner for parallel execution
            If False, fallback to Python orchestration (for debugging)

        Implementation:
        ---------------
        if use_cpp_runner:
            self.simulation_runner = mcts_py.SimulationRunner(
                self.tree,
                self.selector,
                self.backup_manager,
                self.virtual_loss_manager
            )
        """
        pass

    @abstractmethod
    def search(self, root_state, simulations: int, add_noise: bool = False):
        """
        Execute MCTS search using C++ runner or Python loop.

        Parameters:
        -----------
        root_state : IGameState
            Starting game state
        simulations : int
            Number of simulations to run
        add_noise : bool
            Whether to add Dirichlet noise to root (for training)

        Implementation:
        ---------------
        if self.use_cpp_runner:
            return self._search_with_cpp_runner(root_state, simulations)
        else:
            return self._search_with_python_loop(root_state, simulations)
        """
        pass

    @abstractmethod
    def _search_with_cpp_runner(self, root_state, simulations: int) -> int:
        """
        Run simulations using C++ runner with parallel threads.

        Returns:
        --------
        successful_simulations : int
            Number of simulations that completed successfully

        Implementation Pattern:
        -----------------------
        def run_sim_batch(batch_size):
            completed = 0
            for _ in range(batch_size):
                if self.simulation_runner.run_simulation(
                    root_state,
                    self.root_index,
                    self._inference_callback
                ):
                    completed += 1
            return completed

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(self.num_threads):
                batch_size = simulations // self.num_threads
                futures.append(executor.submit(run_sim_batch, batch_size))

            return sum(f.result() for f in futures)
        """
        pass

    @abstractmethod
    def _create_inference_callback(self) -> Callable:
        """
        Create Python callback compatible with C++ InferenceCallback.

        Returns:
        --------
        callback : Callable[[IGameState], Tuple[np.ndarray, float]]
            Function that accepts game state and returns (policy, value)

        Implementation:
        ---------------
        def inference_callback(cpp_game_state):
            future = self.inference_fn(cpp_game_state)
            policy, value = future.result(timeout=1.0)
            return (policy, float(value))

        return inference_callback
        """
        pass


# ==============================================================================
# Performance Requirements
# ==============================================================================

PERFORMANCE_REQUIREMENTS = {
    # Throughput targets
    "single_thread_sims_per_sec": 1400,  # Baseline (no parallelism)
    "eight_thread_sims_per_sec": 30000,  # Parallel execution
    "thread_efficiency_min": 0.75,  # 75% efficiency (6x speedup with 8 threads)

    # GIL contention
    "gil_contention_max_percent": 10.0,  # <10% time in GIL acquire/release
    "gil_cycles_per_simulation": 2,  # 1-2 cycles vs 100-200 in Python

    # GPU batching
    "inference_batch_size_min": 32,  # Average batch size
    "gpu_utilization_min_percent": 80.0,  # Sustained utilization

    # Memory
    "memory_max_mb": 1000,  # <1GB for typical searches
    "memory_leak_max_mb_per_hour": 10,  # <10MB growth over 1 hour

    # Correctness tolerances
    "policy_rtol": 1e-3,  # Relative tolerance for policy comparison
    "value_atol": 1e-3,  # Absolute tolerance for value comparison
}


# ==============================================================================
# Test Contracts (TDD)
# ==============================================================================

def test_simulation_runner_class_exists():
    """
    CONTRACT: SimulationRunner class must exist in mcts_py module.

    This test MUST FAIL initially (before implementation).
    """
    import mcts_py
    assert hasattr(mcts_py, 'SimulationRunner'), \
        "SimulationRunner class not found in mcts_py module"


def test_simulation_runner_constructor():
    """
    CONTRACT: SimulationRunner constructor accepts (tree, selector, backup, virtual_loss).

    This test MUST FAIL initially with "SimulationRunner not found".
    After stub implementation, should construct successfully.
    """
    import mcts_py

    tree = mcts_py.MCTSTree(1000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

    runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
    assert runner is not None, "SimulationRunner failed to construct"


def test_run_simulation_signature():
    """
    CONTRACT: run_simulation accepts (state, root_index, callback) and returns bool.

    This test MUST FAIL with "Not implemented yet" until full implementation.
    """
    import mcts_py
    import alphazero_py
    import numpy as np

    # Setup
    tree = mcts_py.MCTSTree(1000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
    runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)

    game_state = alphazero_py.GomokuState(15)
    root_index = 0

    def mock_inference(state):
        policy = np.ones(362) / 362
        value = 0.0
        return (policy, value)

    # This will raise "Not implemented yet" until Phase 2 complete
    success = runner.run_simulation(game_state, root_index, mock_inference)
    assert isinstance(success, bool), "run_simulation must return bool"


def test_move_storage_accessors():
    """
    CONTRACT: MCTSTree must have get_move() and set_move() methods.

    This test MUST FAIL until move storage is added to tree.
    """
    import mcts_py

    tree = mcts_py.MCTSTree(1000)

    # These will fail until move storage implemented
    assert hasattr(tree, 'get_move'), "MCTSTree missing get_move() method"
    assert hasattr(tree, 'set_move'), "MCTSTree missing set_move() method"

    # Test functionality
    tree.set_move(0, 42)
    assert tree.get_move(0) == 42, "Move storage not working correctly"


def test_gil_released_during_simulation():
    """
    CONTRACT: GIL must be released during run_simulation().

    This test verifies true parallelism is possible.
    """
    import mcts_py
    import alphazero_py
    import numpy as np
    import threading
    import time

    # Setup
    tree = mcts_py.MCTSTree(10000)
    selector = mcts_py.create_puct_selector()
    backup = mcts_py.create_backup_manager(tree)
    virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

    game_state = alphazero_py.GomokuState(15)
    root_index = tree.add_root_node(0.5, 0)

    # Expand root to create children
    legal_moves = game_state.get_legal_moves_as_indices()
    first_child = tree.allocate_nodes(len(legal_moves))
    for i, move in enumerate(legal_moves):
        child_index = first_child + i
        tree.set_prior_prob(child_index, 1.0 / len(legal_moves))
        tree.set_parent_index(child_index, root_index)
        tree.set_move(child_index, move)
    tree.set_first_child_index(root_index, first_child)
    tree.set_num_children(root_index, len(legal_moves))

    def mock_inference(state):
        time.sleep(0.001)  # Simulate 1ms inference
        policy = np.ones(362) / 362
        value = 0.0
        return (policy, value)

    # Run 8 threads concurrently
    def run_sims(runner, n):
        for _ in range(n):
            runner.run_simulation(game_state, root_index, mock_inference)

    runners = [mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
               for _ in range(8)]

    start = time.perf_counter()
    threads = []
    for runner in runners:
        t = threading.Thread(target=run_sims, args=(runner, 10))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    total_sims = 8 * 10
    sims_per_sec = total_sims / elapsed

    # If GIL is released, should achieve >1000 sims/sec
    # If GIL is NOT released, will be <200 sims/sec
    assert sims_per_sec > 500, \
        f"GIL not released: only {sims_per_sec:.0f} sims/sec (expected >500)"


# ==============================================================================
# Contract Validation Summary
# ==============================================================================

CONTRACT_TEST_COUNT = 6

CONTRACT_TESTS = [
    test_simulation_runner_class_exists,
    test_simulation_runner_constructor,
    test_run_simulation_signature,
    test_move_storage_accessors,
    test_gil_released_during_simulation,
]

def validate_all_contracts():
    """
    Run all contract tests and report results.

    Expected behavior:
    ------------------
    - Phase 0 (no implementation): All tests FAIL
    - Phase 1 (stub created): First 2 tests PASS, rest FAIL with "not implemented"
    - Phase 2 (implementation complete): All tests PASS
    """
    results = []
    for test in CONTRACT_TESTS:
        try:
            test()
            results.append((test.__name__, "PASS"))
        except Exception as e:
            results.append((test.__name__, f"FAIL: {str(e)}"))

    print(f"\nContract Validation Results ({len(results)} tests):")
    print("=" * 70)
    for name, status in results:
        print(f"{name:50s} {status}")

    passed = sum(1 for _, status in results if status == "PASS")
    print("=" * 70)
    print(f"Summary: {passed}/{len(results)} tests passing")

    return passed == len(results)


if __name__ == "__main__":
    validate_all_contracts()
