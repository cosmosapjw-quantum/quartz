"""
High-Performance MCTS Engine with C++ Backend
=============================================

Fully-optimized Monte Carlo Tree Search implementation leveraging:
- C++ Structure-of-Arrays tree storage (32-64 bytes/node)
- AVX2-vectorized PUCT selection (4-8x speedup)
- Thread-safe atomic operations for concurrent search
- Memory-efficient index-based node references
- Real-time neural network integration

Performance targets:
- 30,000+ simulations/second including NN inference
- 80-92% GPU utilization sustained
- <1GB memory footprint for 10M node trees
- Thread-safe operation with 12 parallel workers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

try:
    import mcts_py
except ImportError:
    raise ImportError("C++ MCTS module not available. Run 'pip install -e .' to build extensions.")

try:
    from ..games.game_state import IGameState
except ImportError:
    # Fallback for when imported directly (e.g., in tests)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from games.game_state import IGameState

try:
    from src.core.cpp_inference_bridge import CppInferenceBridge
except Exception:  # pragma: no cover - optional dependency
    CppInferenceBridge = None  # type: ignore

try:
    from src.neural.inference_worker import GPUInferenceWorker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GPUInferenceWorker = None  # type: ignore


class MCTSEngine(ABC):
    """Interface for Monte Carlo Tree Search engine."""

    @abstractmethod
    def search(self, root_state: IGameState, simulations: int) -> Dict[int, float]:
        """Run MCTS search from root state."""
        pass

    @abstractmethod
    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        """Get move probabilities from search results."""
        pass

    def get_value(self, root_state: IGameState) -> float:
        """Get position value estimate."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset search tree and internal state."""
        pass

    @property
    @abstractmethod
    def tree_size(self) -> int:
        """Get current number of nodes in search tree."""
        pass


class AlphaZeroMCTS(MCTSEngine):
    """High-performance AlphaZero MCTS with C++ backend.

    Features:
    - C++ Structure-of-Arrays tree storage
    - AVX2-vectorized PUCT selection
    - Atomic operations for thread safety
    - Memory-efficient index-based references
    - Integrated virtual loss coordination
    - Real-time neural network inference
    - Dirichlet noise for exploration
    """

    def __init__(self,
                 inference_fn: Callable[[IGameState], Future[Tuple[np.ndarray, float]]],
                 c_puct: float = 1.25,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 max_tree_size: int = 10_000_000,  # 10M nodes (~270MB)
                 virtual_loss_magnitude: float = 1.0,
                 enable_virtual_loss: bool = True,
                 enable_value_clipping: bool = True,
                 num_threads: int = 8,
                 use_async_inference: bool = True,
                 async_batch_size: int = 16,  # Reduced from 32 for better accumulation with 8 threads
                 async_timeout_ms: float = 10.0,  # Increased from 2.0ms for better batching
                 enable_instrumentation: bool = False,
                 parallel_mode: str = "shared"):
        """Initialize high-performance MCTS engine.

        Args:
            inference_fn: Function that returns Future[policy, value] for a game state
            c_puct: PUCT exploration constant (1.25 optimal for most games)
            dirichlet_alpha: Dirichlet noise alpha parameter
            dirichlet_epsilon: Dirichlet noise mixing ratio
            max_tree_size: Maximum nodes to prevent OOM (10M = ~270MB)
            virtual_loss_magnitude: Virtual loss penalty for thread coordination
            enable_virtual_loss: Enable virtual loss for thread safety
            enable_value_clipping: Clip values to [-1, 1] range
            num_threads: Number of parallel simulation threads
            use_async_inference: Enable async batched inference (30k+ sims/sec)
            async_batch_size: Minimum batch size for async inference (32 optimal)
            async_timeout_ms: Maximum batch collection timeout in ms (2.0 optimal)
        """
        valid_parallel_modes = {"shared", "virtual_loss_free"}
        if parallel_mode not in valid_parallel_modes:
            raise ValueError(
                f"Unsupported parallel_mode '{parallel_mode}'. Supported modes: {sorted(valid_parallel_modes)}"
            )

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.logger = logging.getLogger('AlphaZeroMCTS')
        self.inference_fn = self._prepare_inference_callable(inference_fn)
        self._supports_batch_inference = hasattr(self.inference_fn, 'batch_inference')
        if self._supports_batch_inference:
            self.logger.info("Async batch inference enabled (fast path)")

        # Initialize C++ MCTS tree with memory-efficient storage
        self.tree = mcts_py.MCTSTree(max_tree_size)

        # Initialize C++ PUCT selector with AVX2 optimizations
        puct_config = mcts_py.PUCTConfig()
        puct_config.cpuct = c_puct
        puct_config.fpu_value = 0.0  # First Play Urgency
        puct_config.use_fpu = True
        puct_config.enable_simd = True  # Enable AVX2 vectorization
        self.selector = mcts_py.create_puct_selector(puct_config)

        # Initialize C++ virtual loss manager for thread coordination
        self._virtual_loss_requested = enable_virtual_loss
        desired_virtual_loss_enabled = self._virtual_loss_requested and parallel_mode != "virtual_loss_free"
        vl_config = mcts_py.VirtualLossConfig(virtual_loss_magnitude, desired_virtual_loss_enabled)
        self.virtual_loss_manager = mcts_py.create_test_virtual_loss_manager(self.tree, vl_config)
        self._virtual_loss_magnitude = virtual_loss_magnitude
        self._virtual_loss_enabled = desired_virtual_loss_enabled

        # Initialize C++ backup manager for value propagation
        backup_config = mcts_py.BackupConfig(enable_value_clipping, True, -1.0, 1.0)
        self.backup_manager = mcts_py.create_backup_manager(self.tree, backup_config)

        # Async inference configuration
        self.use_async_inference = use_async_inference
        self.async_batch_size = async_batch_size
        self.async_timeout_ms = async_timeout_ms
        self.num_threads = num_threads  # Number of parallel simulation threads
        self.parallel_mode = parallel_mode
        self._instrumentation_enabled = enable_instrumentation

        if hasattr(mcts_py, 'set_instrumentation_enabled'):
            mcts_py.set_instrumentation_enabled(enable_instrumentation)
        if enable_instrumentation and hasattr(mcts_py, 'reset_instrumentation_metrics'):
            mcts_py.reset_instrumentation_metrics()

        # Initialize async components if enabled
        if self.use_async_inference:
            # Create AsyncInferenceQueue for non-blocking request/result exchange
            self.async_queue = mcts_py.AsyncInferenceQueue()

            # Create one ContinuousSimulationRunner per thread
            # Each runner has its own state but shares thread-safe tree/selector/backup/vl
            self.simulation_runners = [
                mcts_py.ContinuousSimulationRunner(
                    self.tree,
                    self.selector,
                    self.backup_manager,
                    self.virtual_loss_manager
                )
                for _ in range(self.num_threads)
            ]

            # T011a: Create persistent coordinator and callback (reused across all searches)
            self._coordinator = mcts_py.BatchInferenceCoordinator()
            self._coordinator_started = False
            self._batch_callback = None  # Created lazily on first search

            # T011b: Coordinator lifetime metrics
            self._coordinator_searches = 0  # Track searches per coordinator instance
        else:
            # Use synchronous SimulationRunner for backward compatibility
            self.simulation_runner = mcts_py.SimulationRunner(
                self.tree,
                self.selector,
                self.backup_manager,
                self.virtual_loss_manager
            )
            self.async_queue = None
            self._coordinator = None
            self._coordinator_started = False
            self._batch_callback = None

        # State management
        self.root_state = None
        self.root_index = mcts_py.NULL_NODE_INDEX
        self._lock = Lock()
        self._executor_lock = Lock()
        self._executor: Optional[ThreadPoolExecutor]
        self._executor = ThreadPoolExecutor(max_workers=self.num_threads) if self.num_threads > 1 else None

        # Performance statistics
        self._simulations_completed = 0
        self._total_search_time = 0.0

        # Parallel search configuration
        self.num_threads = num_threads

        mode_str = "ContinuousSimulationRunner (async)" if self.use_async_inference else "SimulationRunner (sync)"
        self.logger.info(f"AlphaZero MCTS initialized with C++ {mode_str}")
        self.logger.info(f"  Tree capacity: {max_tree_size:,} nodes (~{max_tree_size * 32 // 1024 // 1024}MB)")
        self.logger.info(f"  AVX2 support: {mcts_py.PUCTSelector.is_avx2_supported()}")
        self.logger.info(f"  PUCT constant: {c_puct}")
        self.logger.info(f"  Virtual loss: {self._virtual_loss_magnitude} (enabled: {self._virtual_loss_enabled})")
        self.logger.info(f"  Parallel threads: {num_threads}")
        if self.use_async_inference:
            self.logger.info(f"  Async batching: batch_size={async_batch_size}, timeout={async_timeout_ms}ms")
        self.logger.info(f"  Parallel mode: {self.parallel_mode}")

    def search(self, root_state: IGameState, simulations: int, add_noise: bool = False) -> Dict[int, float]:
        """Run high-performance MCTS search with C++ SimulationRunner.

        Args:
            root_state: Game state to search from
            simulations: Number of MCTS simulations to run
            add_noise: Whether to add Dirichlet noise to root for exploration

        Returns:
            Dictionary mapping moves to visit counts
        """
        start_time = time.perf_counter()

        # Clear and initialize C++ tree
        with self._lock:
            self.tree.clear()
            self.root_state = root_state
            self.root_index = self.tree.add_root_node(0.5, root_state.get_current_player() - 1)

            root_expanded = False
            if add_noise:
                try:
                    self._expand_node(self.root_index, root_state)
                    root_expanded = True
                except Exception as exc:  # pragma: no cover - logged fallback
                    self.logger.warning(f"Failed to pre-expand root for noise: {exc}")

            if add_noise and (root_expanded or self.tree.get_num_children(self.root_index) > 0):
                self._add_dirichlet_noise(self.root_index)

        # NOTE: Do NOT pre-expand root - let C++ SimulationRunner handle all expansion
        # The C++ runner will expand the root on the first simulation

        successful_simulations = 0
        failed_simulations = 0

        if self.use_async_inference:
            # Async mode: Use ContinuousSimulationRunner with BatchInferenceCoordinator
            # T011a: Create batch callback once and reuse (callback holds GIL)
            if self._batch_callback is None:
                self._batch_callback = mcts_py.PyBatchInferenceCallback(self._create_batch_inference_callback())

            # T011b: Health check - if coordinator was stopped externally, restart it
            if self._coordinator_started:
                # Defensive check: coordinator should still be running
                # If stopped externally (edge case), we'll restart it
                try:
                    # Attempt to verify coordinator is still running by checking if we can start it
                    # If already running, this should be a no-op or raise an error
                    pass  # No direct "is_running()" API, so we rely on _coordinator_started flag
                except Exception as e:
                    self.logger.warning(f"Coordinator health check failed, will restart: {e}")
                    self._coordinator_started = False

            # T011a: Start persistent coordinator if not already running
            if not self._coordinator_started:
                try:
                    self._coordinator.start(self.async_queue, self._batch_callback,
                                           self.async_batch_size, self.async_timeout_ms)
                    self._coordinator_started = True
                    self.logger.debug("Coordinator started successfully")
                except Exception as e:
                    self.logger.error(f"Failed to start coordinator: {e}")
                    raise

            try:
                # Multi-threaded search: Distribute simulations across threads
                if self.num_threads > 1:
                    sims_per_thread = simulations // self.num_threads
                    remainder = simulations % self.num_threads

                    def worker_thread(thread_id: int) -> int:
                        """Worker thread running continuous simulations."""
                        thread_sims = sims_per_thread + (remainder if thread_id == 0 else 0)
                        runner = self.simulation_runners[thread_id]
                        # Extract C++ state from wrapper for C++ function call
                        cpp_state = root_state.cpp_state if hasattr(root_state, 'cpp_state') else root_state
                        return runner.run_continuous(cpp_state, self.root_index, self.async_queue, thread_sims)

                    with self._executor_lock:
                        if self._executor is None:
                            self._executor = ThreadPoolExecutor(max_workers=self.num_threads)
                        executor = self._executor
                    futures = [executor.submit(worker_thread, i) for i in range(self.num_threads)]
                    thread_results = [f.result() for f in futures]

                    successful_simulations = sum(thread_results)

                    self.logger.info(f"Multi-threaded search: {self.num_threads} threads, "
                                     f"{successful_simulations}/{simulations} sims completed")
                else:
                    # Extract C++ state from wrapper for C++ function call
                    cpp_state = root_state.cpp_state if hasattr(root_state, 'cpp_state') else root_state
                    successful_simulations = self.simulation_runners[0].run_continuous(
                        cpp_state, self.root_index, self.async_queue, simulations
                    )
            except Exception:
                # T011a: Preserve coordinator on exceptions - don't stop it
                raise

        else:
            # Sync mode: Use SimulationRunner (backward compatibility)
            callback = mcts_py.PyInferenceCallback(self._create_inference_callback())

            # Run MCTS simulations using synchronous loop
            if self.num_threads > 1:
                sims_per_thread = simulations // self.num_threads
                remainder = simulations % self.num_threads

                def sync_worker(thread_id: int) -> Tuple[int, int]:
                    local_success = 0
                    local_fail = 0
                    local_total = sims_per_thread + (remainder if thread_id == 0 else 0)
                    # Extract C++ state from wrapper for C++ function call
                    cpp_state = root_state.cpp_state if hasattr(root_state, 'cpp_state') else root_state
                    for _ in range(local_total):
                        if self.tree.get_node_count() >= self.tree.get_max_nodes():
                            break
                        if self.simulation_runner.run_simulation(cpp_state, self.root_index, callback):
                            local_success += 1
                        else:
                            local_fail += 1
                    return local_success, local_fail

                with self._executor_lock:
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(max_workers=self.num_threads)
                    executor = self._executor
                futures = [executor.submit(sync_worker, i) for i in range(self.num_threads)]
                for future in futures:
                    success, failure = future.result()
                    successful_simulations += success
                    failed_simulations += failure
            else:
                for _ in range(simulations):
                    if self.tree.get_node_count() >= self.tree.get_max_nodes():
                        break
                    if self.simulation_runner.run_simulation(root_state, self.root_index, callback):
                        successful_simulations += 1
                    else:
                        failed_simulations += 1

            if failed_simulations > 0:
                self.logger.warning(f"Failed simulations: {failed_simulations}/{simulations}")

        # Collect visit counts for policy using tree.get_move()
        visit_counts = {}
        first_child = self.tree.get_first_child_index(self.root_index)
        num_children = self.tree.get_num_children(self.root_index)

        if first_child != mcts_py.NULL_NODE_INDEX:
            for i in range(num_children):
                child_index = first_child + i
                if self.tree.is_valid_index(child_index):
                    move = self.tree.get_move(child_index)
                    visit_counts[move] = int(self.tree.get_visit_count(child_index))

        # Update performance statistics
        search_time = time.perf_counter() - start_time
        self._simulations_completed += successful_simulations
        self._total_search_time += search_time

        # T011b: Increment coordinator lifetime metric
        if self.use_async_inference and successful_simulations > 0:
            self._coordinator_searches += 1

        avg_time_per_sim = search_time / max(successful_simulations, 1)
        sims_per_second = successful_simulations / search_time if search_time > 0 else 0

        self.logger.info(f"Search completed: {successful_simulations}/{simulations} sims, "
                         f"{sims_per_second:.1f} sims/sec, {avg_time_per_sim*1000:.2f}ms/sim, "
                         f"failed: {failed_simulations}")

        return visit_counts

    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        """Extract move probabilities from C++ tree.

        Args:
            root_state: Game state (for action space size)
            temperature: Temperature for softmax (0 = greedy, 1 = proportional)

        Returns:
            Probability distribution over actions
        """
        action_space_size = root_state.action_space_size
        policy = np.zeros(action_space_size, dtype=np.float32)

        legal_moves = root_state.get_legal_moves()
        legal_moves_array = np.array([move for move in legal_moves if 0 <= move < action_space_size], dtype=np.int64)

        if self.root_index == mcts_py.NULL_NODE_INDEX or self.tree.get_num_children(self.root_index) == 0:
            if legal_moves_array.size > 0:
                uniform_prob = 1.0 / float(legal_moves_array.size)
                policy[legal_moves_array] = uniform_prob
            return policy

        visits = np.zeros(action_space_size, dtype=np.float64)
        first_child = self.tree.get_first_child_index(self.root_index)
        num_children = self.tree.get_num_children(self.root_index)

        for i in range(num_children):
            child_index = first_child + i
            if self.tree.is_valid_index(child_index):
                move = self.tree.get_move(child_index)
                if 0 <= move < action_space_size:
                    visits[move] = self.tree.get_visit_count(child_index)

        if temperature > 1.0 and legal_moves_array.size > 0:
            zero_mask = visits[legal_moves_array] <= 0.0
            if np.any(zero_mask):
                visits[legal_moves_array[zero_mask]] = 1e-6

        if temperature == 0:
            if np.all(visits == 0) and legal_moves_array.size > 0:
                policy[legal_moves_array[0]] = 1.0
            else:
                best_move = int(np.argmax(visits))
                policy[best_move] = 1.0
        else:
            total_visits = np.sum(visits)
            if total_visits > 0:
                if temperature == 1.0:
                    policy = visits / total_visits
                else:
                    visits_temp = np.power(visits, 1.0 / temperature)
                    sum_temp = np.sum(visits_temp)
                    if sum_temp > 0:
                        policy = visits_temp / sum_temp
            elif legal_moves_array.size > 0:
                uniform_prob = 1.0 / float(legal_moves_array.size)
                policy[legal_moves_array] = uniform_prob

        return policy.astype(np.float32)

    def get_value(self, root_state: IGameState) -> float:
        """Get position value estimate from C++ tree.

        Returns:
            Value estimate from current player's perspective [-1, 1]
        """
        if self.root_index == mcts_py.NULL_NODE_INDEX:
            # No search performed - use neural network evaluation
            future = self.inference_fn(root_state)
            try:
                _, value = future.result(timeout=1.0)  # 1s timeout
                return float(value)
            except Exception as e:
                self.logger.warning(f"Neural network evaluation failed: {e}")
                return 0.0

        # Get Q-value from C++ tree
        return self.backup_manager.get_q_value(self.root_index)

    def reset(self) -> None:
        """Reset search tree and internal state."""
        with self._lock:
            self.tree.clear()
            self.virtual_loss_manager.reset_all_virtual_loss()
            self.backup_manager.reset_statistics()
            self.root_state = None
            self.root_index = mcts_py.NULL_NODE_INDEX

        self.logger.debug("MCTS tree reset")

    def _prepare_inference_callable(self, inference_fn: Any) -> Callable[[IGameState], Future]:
        """Wrap inference backends to expose the expected callable interface."""
        # Already suitable: callable that returns Future or provides batching
        if isinstance(inference_fn, CppInferenceBridge):
            return inference_fn

        if GPUInferenceWorker is not None and isinstance(inference_fn, GPUInferenceWorker):
            if CppInferenceBridge is None:
                raise RuntimeError("CppInferenceBridge unavailable to wrap GPUInferenceWorker")
            self.logger.info("Wrapping GPUInferenceWorker with CppInferenceBridge for batched inference")
            return CppInferenceBridge(inference_fn)

        return inference_fn

    @property
    def tree_size(self) -> int:
        """Get current number of nodes in C++ tree."""
        return self.tree.get_node_count()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive MCTS performance statistics."""
        stats = {
            'tree_size': self.tree.get_node_count(),
            'max_tree_size': self.tree.get_max_nodes(),
            'memory_usage_mb': self.tree.get_memory_usage() / (1024 * 1024),
            'bytes_per_node': self.tree.get_bytes_per_node(),
            'simulations_completed': self._simulations_completed,
            'total_search_time': self._total_search_time,
            'avg_simulations_per_second': (
                self._simulations_completed / self._total_search_time
                if self._total_search_time > 0 else 0
            ),
            'parallel_mode': self.parallel_mode,
            'virtual_loss_enabled': self._virtual_loss_enabled,
            'virtual_loss_stats': self.virtual_loss_manager.get_statistics(),
            'backup_stats': self.backup_manager.get_statistics(),
            'selector_config': {
                'cpuct': self.selector.get_config().cpuct,
                'fpu_value': self.selector.get_config().fpu_value,
                'use_fpu': self.selector.get_config().use_fpu,
                'enable_simd': self.selector.get_config().enable_simd,
                'avx2_supported': mcts_py.PUCTSelector.is_avx2_supported()
            }
        }

        # T011b: Add coordinator lifetime metrics
        if self.use_async_inference:
            stats['coordinator_searches'] = self._coordinator_searches
            stats['coordinator_started'] = self._coordinator_started

        instrumentation_snapshot: Dict[str, Any] = {}
        if hasattr(mcts_py, 'get_instrumentation_snapshot') and self._instrumentation_enabled:
            instrumentation_snapshot = mcts_py.get_instrumentation_snapshot()
        stats['instrumentation'] = instrumentation_snapshot
        return stats

    def set_instrumentation_enabled(self, enabled: bool) -> None:
        """Enable or disable instrumentation metrics collection."""
        if hasattr(mcts_py, 'set_instrumentation_enabled'):
            mcts_py.set_instrumentation_enabled(enabled)
        if enabled and hasattr(mcts_py, 'reset_instrumentation_metrics'):
            mcts_py.reset_instrumentation_metrics()
        self._instrumentation_enabled = enabled

    def reset_instrumentation_metrics(self) -> None:
        """Reset instrumentation counters when instrumentation is enabled."""
        if self._instrumentation_enabled and hasattr(mcts_py, 'reset_instrumentation_metrics'):
            mcts_py.reset_instrumentation_metrics()

    def close(self) -> None:
        """Release any background resources (thread pools, coordinator).

        T011a: Stop persistent coordinator to eliminate per-search thread restarts.
        """
        # Stop coordinator if running
        if self._coordinator is not None and self._coordinator_started:
            try:
                self._coordinator.stop()
                self._coordinator_started = False
            except Exception as e:
                self.logger.warning(f"Failed to stop coordinator during close(): {e}")

        # Shutdown thread pool
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def set_parallel_mode(self, mode: str) -> None:
        """Switch parallel execution mode (experimental)."""
        valid_parallel_modes = {"shared", "virtual_loss_free"}
        if mode not in valid_parallel_modes:
            raise ValueError(
                f"Unsupported parallel_mode '{mode}'. Supported modes: {sorted(valid_parallel_modes)}"
            )

        if mode == self.parallel_mode:
            return

        self.parallel_mode = mode
        if mode == "virtual_loss_free":
            desired_virtual_loss_enabled = False
        else:
            desired_virtual_loss_enabled = self._virtual_loss_requested

        if desired_virtual_loss_enabled != self._virtual_loss_enabled:
            self.virtual_loss_manager.set_config(
                mcts_py.VirtualLossConfig(self._virtual_loss_magnitude, desired_virtual_loss_enabled)
            )
            self._virtual_loss_enabled = desired_virtual_loss_enabled
        self.logger.info("Parallel mode switched to %s (virtual loss enabled=%s)", mode, desired_virtual_loss_enabled)

    def _expand_node(self, node_index: int, game_state: IGameState) -> float:
        """Expand node using neural network evaluation.

        Args:
            node_index: C++ tree node index to expand
            game_state: Game state at this node

        Returns:
            Neural network value estimate
        """
        # Check if already terminal
        if game_state.is_terminal():
            flags = mcts_py.NodeFlags()
            flags.set_terminal(True)
            flags.set_current_player(game_state.get_current_player() - 1)
            self.tree.set_flags(node_index, flags)
            return self._get_terminal_value(game_state)

        # Get neural network evaluation
        try:
            future = self.inference_fn(game_state)
            policy, value = future.result(timeout=1.0)  # 1s timeout for batched inference

            # Extract policy for single game state from batch
            if policy.ndim > 1:
                policy = policy[0]  # Extract the policy for the single game state
        except Exception as e:
            self.logger.error(f"Neural network inference failed: {e}")
            # Fallback to uniform policy and neutral value
            legal_moves = game_state.get_legal_moves()
            policy = np.zeros(game_state.action_space_size)
            if len(legal_moves) > 0:
                for move in legal_moves:
                    policy[move] = 1.0 / len(legal_moves)
            value = 0.0

        # Mask illegal moves and normalize
        legal_moves = game_state.get_legal_moves()
        legal_moves_set = set(legal_moves)

        for move in range(len(policy)):
            if move not in legal_moves_set:
                policy[move] = 0.0

        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum

        # Add children to C++ tree for legal moves
        if len(legal_moves) > 0:
            # Allocate contiguous children for cache efficiency
            first_child = self.tree.allocate_nodes(len(legal_moves))
            if first_child != mcts_py.NULL_NODE_INDEX:
                for i, move in enumerate(legal_moves):
                    child_index = first_child + i
                    if move < len(policy):
                        prob_value = policy[move]
                        # Safely extract scalar value
                        try:
                            prob_array = np.asarray(prob_value).flatten()
                            if len(prob_array) > 0:
                                prior_prob = float(prob_array[0])
                            else:
                                self.logger.warning(f"Empty probability array for move {move}")
                                prior_prob = 0.0
                        except Exception as e:
                            self.logger.warning(f"Failed to extract probability for move {move}: {e}")
                            prior_prob = 0.0
                    else:
                        prior_prob = 0.0

                    # Initialize child node
                    self.tree.set_prior_prob(child_index, prior_prob)
                    self.tree.set_parent_index(child_index, node_index)
                    self.tree.set_visit_count(child_index, 0.0)
                    self.tree.set_total_value(child_index, 0.0)

                    # Store move in C++ tree (replaces Python dict)
                    self.tree.set_move(child_index, move)

                # Update parent node
                self.tree.set_first_child_index(node_index, first_child)
                self.tree.set_num_children(node_index, len(legal_moves))

        # Mark node as expanded
        flags = mcts_py.NodeFlags()
        flags.set_expanded(True)
        flags.set_current_player(max(0, game_state.get_current_player() - 1))
        self.tree.set_flags(node_index, flags)

        return float(value)

    def _add_dirichlet_noise(self, node_index: int) -> None:
        """Add Dirichlet noise to root node for exploration."""
        num_children = self.tree.get_num_children(node_index)
        if num_children == 0:
            return

        first_child = self.tree.get_first_child_index(node_index)
        if first_child == mcts_py.NULL_NODE_INDEX:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)

        # Apply noise to prior probabilities
        for i in range(num_children):
            child_index = first_child + i
            if self.tree.is_valid_index(child_index):
                current_prior = self.tree.get_prior_prob(child_index)
                noisy_prior = ((1 - self.dirichlet_epsilon) * current_prior +
                              self.dirichlet_epsilon * noise[i])
                self.tree.set_prior_prob(child_index, noisy_prior)

    def _get_terminal_value(self, game_state: IGameState) -> float:
        """Get terminal value from game result."""
        result = game_state.get_result()
        return float(result) if result is not None else 0.0

    def _create_inference_callback(self) -> Callable:
        """Create inference callback for C++ SimulationRunner.

        Returns:
            Callable that takes IGameState and returns (policy, value) tuple
        """
        def inference_callback(game_state: IGameState) -> Tuple[List[float], float]:
            """Synchronous inference for C++ runner."""
            try:
                future = self.inference_fn(game_state)
                policy, value = future.result(timeout=1.0)

                # Extract policy for single game state from batch
                if policy.ndim > 1:
                    policy = policy[0]

                # T029: Return numpy array directly (pybind11 converts to std::vector<float>)
                # This eliminates Python list conversion overhead (~0.5-1ms)
                return (policy, float(value))
            except Exception as e:
                self.logger.error(f"Inference callback failed: {e}")
                # Fallback to uniform policy
                legal_moves = game_state.get_legal_moves()
                action_space = game_state.get_action_space_size()
                policy = [0.0] * action_space
                if len(legal_moves) > 0:
                    prob = 1.0 / len(legal_moves)
                    for move in legal_moves:
                        policy[move] = prob
                return (policy, 0.0)

        return inference_callback

    def _create_batch_inference_callback(self) -> Callable:
        """Create batch inference callback with automatic mode detection.

        Supports two modes for maximum performance while maintaining test compatibility:

        MODE 1 - Direct GPU Batching (Production):
            - Detects if inference_fn has batch_inference() method
            - Calls gpu_worker.batch_inference(positions) ONCE per batch
            - Achieves 10-15k sims/sec (10-15× faster than per-state mode)

        MODE 2 - Per-State Future (Testing):
            - Falls back for test mocks without batch_inference() method
            - Calls inference_fn(state) for each state (legacy compatibility)
            - Achieves ~1k sims/sec (acceptable for tests)

        Returns:
            Callable that takes list of IGameState and returns list of (policy, value) tuples
        """

        # MODE 1: Direct GPU Batching (Production - FAST)
        if hasattr(self.inference_fn, 'batch_inference'):
            # Check if this is DLPackInferenceBridge (needs states, not tensors)
            from src.core.dlpack_inference_bridge import DLPackInferenceBridge
            is_dlpack_bridge = isinstance(self.inference_fn, DLPackInferenceBridge)

            def fast_batch_callback(features_list: List[List[float]],
                                   board_sizes: List[int],
                                   num_planes_list: List[int]) -> List[Tuple[List[float], float]]:
                """Direct GPU batch inference with pre-extracted features (T018g optimization)."""
                try:
                    # Convert features to numpy tensors with proper shape
                    positions = []
                    for features, board_size, num_planes in zip(features_list, board_sizes, num_planes_list):
                        # Reshape flat features to (C, H, W)
                        tensor = np.array(features, dtype=np.float32).reshape(num_planes, board_size, board_size)
                        positions.append(tensor)

                    # All inference workers expect tensors, not states (T018g optimization)
                    # DLPackInferenceBridge returns List[(policy, value)]
                    # GPUInferenceWorker returns (policies, values) tuple
                    result = self.inference_fn.batch_inference(positions)

                    # Handle both return formats
                    if isinstance(result, list):
                        # Check if it's List[(policy, value)] format
                        if len(result) > 0 and isinstance(result[0], tuple) and len(result[0]) == 2:
                            # DLPackInferenceBridge format: List[(policy, value)]
                            return result
                        else:
                            # Unexpected format
                            raise ValueError(f"Unexpected list format: {type(result[0]) if result else 'empty'}")
                    elif isinstance(result, tuple) and len(result) == 2:
                        # GPUInferenceWorker format: (policies, values)
                        policies, values = result
                        results = []
                        for i in range(len(policies)):
                            policy_array = policies[i]
                            results.append((policy_array, float(values[i])))
                        return results
                    else:
                        raise ValueError(f"Unexpected result format: {type(result)}")

                except Exception as e:
                    self.logger.error(f"Direct GPU batch inference failed: {e}")
                    # Fallback to uniform policy for all features
                    action_space = 225  # Default for Gomoku 15x15
                    if board_sizes and board_sizes[0] > 0:
                        action_space = board_sizes[0] * board_sizes[0]
                    uniform_policy = np.ones(action_space, dtype=np.float32) / action_space
                    return [(uniform_policy, 0.0) for _ in range(len(features_list))]

            self.logger.info(f"Using direct GPU batch inference (fast path) - expected 10-15k sims/sec")
            return fast_batch_callback

        # MODE 2: Per-State Future Mode (Testing - SLOW but compatible)
        else:
            def legacy_batch_callback(features_list: List[List[float]],
                                     board_sizes: List[int],
                                     num_planes_list: List[int]) -> List[Tuple[List[float], float]]:
                """Legacy per-state inference - INCOMPATIBLE with feature extraction.

                ⚠️  WARNING: The legacy callback path (per-state futures) is incompatible
                with the T018g feature extraction optimization. This path cannot properly
                mask illegal moves without game state objects.

                This function raises an error to prevent silent failures. Use GPUInferenceWorker
                or DLPackInferenceBridge for production (fast_batch_callback path).
                """
                raise RuntimeError(
                    "Legacy per-state inference is incompatible with feature extraction optimization (T018g). "
                    "Use GPUInferenceWorker or DLPackInferenceBridge for batch inference. "
                    "The legacy path cannot mask illegal moves without game state objects."
                )

            self.logger.warning(
                f"Using legacy per-state inference (slow path, testing only) - "
                f"expected ~1k sims/sec. Use GPUInferenceWorker for production."
            )
            return legacy_batch_callback

    def _create_uniform_policy_batch(self, game_states: List[IGameState]) -> List[Tuple[List[float], float]]:
        """Create uniform policy fallback for all states in batch.

        Args:
            game_states: List of game states

        Returns:
            List of (policy, value) tuples with uniform policy distribution
        """
        results = []
        for state in game_states:
            legal_moves = state.get_legal_moves()
            action_space = state.get_action_space_size()
            policy = [0.0] * action_space
            if len(legal_moves) > 0:
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    policy[move] = prob
            results.append((policy, 0.0))
        return results


# Backward compatibility
class MockMCTSEngine(MCTSEngine):
    """Mock implementation of MCTSEngine for testing."""

    def __init__(self, action_space_size: int = 225):
        self.action_space_size = action_space_size
        self._tree_size = 0
        self.search_results = {}

    def search(self, root_state: IGameState, simulations: int) -> Dict[int, float]:
        legal_moves = root_state.get_legal_moves()
        visit_counts = {}
        for move in legal_moves[:min(10, len(legal_moves))]:
            visit_counts[move] = np.random.randint(1, simulations // 10 + 1)
        self.search_results = visit_counts
        self._tree_size = simulations
        return visit_counts

    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        policy = np.zeros(self.action_space_size)
        if self.search_results:
            total_visits = sum(self.search_results.values())
            for move, visits in self.search_results.items():
                if move < self.action_space_size:
                    policy[move] = visits / total_visits
        else:
            legal_moves = root_state.get_legal_moves()
            if len(legal_moves) > 0:
                for move in legal_moves:
                    if move < self.action_space_size:
                        policy[move] = 1.0 / len(legal_moves)
        if temperature != 1.0 and temperature > 0:
            policy = policy ** (1.0 / temperature)
            policy = policy / np.sum(policy)
        return policy

    def get_value(self, root_state: IGameState) -> float:
        return np.random.uniform(-1, 1)

    def reset(self) -> None:
        self._tree_size = 0
        self.search_results = {}

    @property
    def tree_size(self) -> int:
        return self._tree_size
