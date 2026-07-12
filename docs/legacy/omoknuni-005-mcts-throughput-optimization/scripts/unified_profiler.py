#!/usr/bin/env python3
"""
Unified Profiling Orchestrator - Phase 5

Provides a single-command interface to run comprehensive profiling
of both C++ MCTS implementation and Python coordination layer.

Usage:
    # Basic profiling with default settings
    python scripts/unified_profiler.py

    # Custom configuration
    python scripts/unified_profiler.py --simulations 1600 --threads 8 --batch-size 64

    # With validation first
    python scripts/unified_profiler.py --validate --simulations 800

    # Export to specific files
    python scripts/unified_profiler.py --output results/my_analysis

Author: MCTS Performance Team
Date: 2025-10-15
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    import alphazero_py
    from src.profiling import ProfilingSession, ProfilerConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you've built the C++ extensions and installed dependencies:")
    print("   pip install -e .")
    sys.exit(1)


class UnifiedProfiler:
    """Orchestrates profiling of both C++ and Python components"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cpp_profiler = None
        self.python_session = None
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Run validation before profiling"""
        print("\n" + "="*60)
        print("VALIDATION PHASE")
        print("="*60)

        if not mcts_py.run_profiling_validation():
            print("\n❌ Validation failed! Fix issues before profiling.")
            return False

        print("\n✅ Validation passed! Proceeding to profiling...")
        return True

    def setup_cpp_profiling(self):
        """Setup C++ profiler"""
        print("\n" + "="*60)
        print("C++ PROFILING SETUP")
        print("="*60)

        self.cpp_profiler = mcts_py.EnhancedProfiler.instance()
        # Enable C++ profiling with FULL level for comprehensive metrics
        self.cpp_profiler.set_enabled(True)
        self.cpp_profiler.set_level(mcts_py.ProfileLevel.FULL)
        self.cpp_profiler.start_session(f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        print("✅ C++ profiler enabled (FULL level)")
        print(f"   - Timer metrics + hardware counters + memory tracking")
        print(f"   - Tracking: core MCTS operations (selection, expansion, backup, coordinator)")

    def setup_python_profiling(self) -> ProfilingSession:
        """Setup Python profiler"""
        print("\n" + "="*60)
        print("PYTHON PROFILING SETUP")
        print("="*60)

        config = ProfilerConfig(
            # Enable ALL profiling features for comprehensive metrics
            enable_gil_profiling=True,
            enable_inference_profiling=True,
            enable_cpp_instrumentation=True,
            enable_thread_profiling=True,
            enable_memory_profiling=True
        )

        session = ProfilingSession(config)
        session.__enter__()

        print("✅ Python profiler enabled (ALL features)")
        print(f"   - GIL tracking: ENABLED")
        print(f"   - Inference profiling: ENABLED")
        print(f"   - C++ instrumentation: ENABLED")
        print(f"   - Thread profiling: ENABLED")
        print(f"   - Memory profiling: ENABLED")

        return session

    def run_mcts_workload(self):
        """Run MCTS workload for profiling"""
        print("\n" + "="*60)
        print(f"MCTS WORKLOAD ({self.args.simulations} simulations)")
        print("="*60)

        # Create game state using C++ binding
        state = alphazero_py.GomokuState(board_size=15)
        action_space_size = state.get_action_space_size()
        print(f"✅ Created Gomoku game state (15×15)")

        # Create MCTS components
        self.tree = mcts_py.create_test_tree(100000)  # 100k nodes capacity
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(self.tree)
        vl_manager = mcts_py.create_test_virtual_loss_manager(self.tree)
        print(f"✅ Created MCTS components (100k node capacity)")

        if self.args.runner_type == "simulation":
            self._run_simulation_runner(state, action_space_size, selector, backup, vl_manager)
        elif self.args.runner_type == "continuous":
            self._run_continuous_runner(state, action_space_size, selector, backup, vl_manager)
        else:
            raise ValueError(f"Unknown runner_type: {self.args.runner_type}")

    def _run_simulation_runner(self, state, action_space_size, selector, backup, vl_manager):
        """Run profiling with SimulationRunner (baseline, clone-based)"""
        # Create simulation runner
        runner = mcts_py.SimulationRunner(self.tree, selector, backup, vl_manager)
        print(f"✅ Created SimulationRunner (baseline, clone-based)")

        # Create inference callback
        # NOTE: Using uniform random policy for profiling is CORRECT because:
        #   1. ALL MCTS operations are REAL: state cloning, tree traversal, PUCT, virtual loss, backup
        #   2. Game state operations are REAL: GomokuState.clone(), apply_move(), getLegalMoves()
        #   3. Only NN inference is mocked (which is tracked separately in expansion_nn_wait metric)
        #   4. This eliminates GPU variance and focuses on MCTS infrastructure bottlenecks
        #   5. For NN profiling specifically, use separate GPU benchmarking tools
        def dummy_inference(game_state):
            """Uniform random policy for MCTS infrastructure profiling"""
            action_space = game_state.get_action_space_size()
            # Return uniform policy and neutral value
            policy = [1.0 / action_space] * action_space
            value = 0.0
            return (policy, value)

        callback = mcts_py.PyInferenceCallback(dummy_inference)
        print(f"✅ Created inference callback (uniform policy for MCTS profiling)")

        # Run simulations
        print(f"\n🔥 Running {self.args.simulations} simulations...")
        print(f"   Note: Single-threaded runner (use --runner-type continuous for parallel)")

        # Get root node index
        root = self.tree.get_root_index()

        # Run simulations sequentially (profiling will capture all metrics)
        try:
            successes = 0
            for i in range(self.args.simulations):
                success = runner.run_simulation(state, root, callback)
                if success:
                    successes += 1
                if (i + 1) % 20 == 0:  # Progress every 20 sims
                    print(f"   Progress: {i + 1}/{self.args.simulations} simulations")

            # Store tree node count for later export
            self.tree_node_count = self.tree.get_node_count()

            print(f"✅ Completed {successes}/{self.args.simulations} successful simulations")
            print(f"   Tree nodes: {self.tree_node_count}")
        except Exception as e:
            print(f"⚠️  Error during simulation: {e}")
            print("    Profiling data still captured up to this point")
            import traceback
            traceback.print_exc()

    def _run_continuous_runner(self, state, action_space_size, selector, backup, vl_manager):
        """Run profiling with ContinuousSimulationRunner (T024f-6, make/unmake)"""
        import time
        import numpy as np
        import torch

        # Create CONTINUOUS simulation runner (THIS IS THE KEY!)
        runner = mcts_py.ContinuousSimulationRunner(self.tree, selector, backup, vl_manager)
        print(f"✅ Created ContinuousSimulationRunner (T024f-6, make/unmake pattern)")
        print(f"   Expected improvements:")
        print(f"   - State cloning: 2× per sim → 1× per sim (50% reduction)")
        print(f"   - Time per sim: 836μs → 418μs (1.77× speedup)")
        print(f"   - Throughput: 1,650 → 4,700 sims/sec")

        # Create async inference queue
        queue = mcts_py.AsyncInferenceQueue()
        print(f"✅ Created AsyncInferenceQueue (ring buffer, lock-free)")

        # ============================================================================
        # REAL GPU MODEL INFERENCE (instead of dummy callback)
        # ============================================================================
        print(f"\n🔥 Creating REAL GPU model for accurate profiling...")

        try:
            from src.neural.model import create_random_model
            from src.core.dlpack_inference_bridge import DLPackInferenceBridge

            # Create random model (same as benchmark)
            model = create_random_model('gomoku', seed=42)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()

            # Create DLPack inference bridge with FP16
            # CRITICAL: Disable CUDA graphs to avoid 13× fallback slowdown from partial batches
            inference_bridge = DLPackInferenceBridge(
                model=model,
                device=device,
                use_mixed_precision=True,  # T008f optimization
                use_cuda_graphs=False  # Disable to avoid fallback on non-standard batch sizes
            )

            # Warmup GPU
            inference_bridge.warmup(batch_size=64, game_type='gomoku')

            print(f"✅ Created real GPU model on {device}")
            print(f"   Model: FastMCTSNet (random weights)")
            print(f"   FP16 mixed precision: enabled")
            print(f"   Warmup: complete")

            # Create batch inference callback using REAL model
            def batch_inference_fn(features_batch, board_sizes, num_planes_list):
                """REAL GPU batch inference with pre-extracted features (Phase 1 optimization)"""
                # Call new batch_inference_features API (zero-copy optimization)
                return inference_bridge.batch_inference_features(
                    features_batch, board_sizes, num_planes_list
                )

            use_real_gpu = True

        except Exception as e:
            print(f"⚠️  Failed to create GPU model: {e}")
            print(f"   Falling back to dummy callback")

            # Fallback to dummy callback
            def batch_inference_fn(features_batch, board_sizes, num_planes_list):
                """Dummy callback (fallback)"""
                results = []
                for _ in features_batch:
                    policy = np.ones(action_space_size, dtype=np.float32) / action_space_size
                    value = 0.0
                    results.append((policy.tolist(), value))
                return results

            use_real_gpu = False

        callback = mcts_py.PyBatchInferenceCallback(batch_inference_fn)
        coordinator = mcts_py.BatchInferenceCoordinator()
        print(f"✅ Created BatchInferenceCoordinator")
        print(f"   Using: {'REAL GPU inference' if use_real_gpu else 'dummy callback'}")

        # Start coordinator with longer timeout to avoid partial batches
        coordinator.start(queue, callback, self.args.batch_size, 100.0)  # timeout=100ms (avoid partial batches!)
        print(f"✅ Coordinator started (batch_size={self.args.batch_size}, timeout=100ms)")

        try:
            # Get root node (tree already has root from create_test_tree)
            root_idx = self.tree.get_root_index()
            print(f"✅ Using root node (index={root_idx})")

            # Run continuous simulations (THIS USES MAKE/UNMAKE!)
            print(f"\n🔥 Running {self.args.simulations} simulations with make/unmake pattern...")
            print(f"   Thread count: {self.args.threads} (OpenMP)")
            print(f"   Batch size: {self.args.batch_size}")

            # CRITICAL FIX: Measure ONLY simulation loop (same as wall-clock validation)
            # This excludes setup/teardown overhead for accurate comparison
            start = time.perf_counter()
            completed = runner.run_continuous(state, root_idx, queue, self.args.simulations)
            elapsed = time.perf_counter() - start

            # Store metrics
            self.tree_node_count = self.tree.get_node_count()
            self.root_visits = self.tree.get_visit_count(root_idx)
            self.wall_clock_time = elapsed  # Pure simulation time (no setup/export)
            self.throughput = completed / elapsed if elapsed > 0 else 0
            self.time_per_sim_us = (elapsed / completed) * 1e6 if completed > 0 else 0

            print(f"✅ Completed {completed}/{self.args.simulations} simulations")
            print(f"   Wall-clock: {elapsed:.3f}s (simulation loop only)")
            print(f"   Throughput: {self.throughput:.1f} sims/sec")
            print(f"   Time per sim: {self.time_per_sim_us:.1f} μs")
            print(f"   Tree nodes: {self.tree_node_count}")
            print(f"   Root visits: {self.root_visits}")
            print(f"   Note: Profiler overhead NOT included in this timing")

        except Exception as e:
            print(f"⚠️  Error during simulation: {e}")
            print("    Profiling data still captured up to this point")
            import traceback
            traceback.print_exc()
        finally:
            coordinator.stop()
            print(f"✅ Coordinator stopped")

    def export_cpp_results(self):
        """Export C++ profiling results"""
        print("\n" + "="*60)
        print("C++ RESULTS EXPORT")
        print("="*60)

        self.cpp_profiler.stop_session()

        json_path = self.output_dir / "cpp_profiling.json"
        chrome_path = self.output_dir / "cpp_trace.json"
        markdown_path = self.output_dir / "cpp_report.md"

        self.cpp_profiler.export_json(str(json_path))
        self.cpp_profiler.export_chrome_trace(str(chrome_path))
        self.cpp_profiler.export_markdown(str(markdown_path))

        # Add tree node count and ContinuousRunner metrics to JSON (Python-side metrics)
        if hasattr(self, 'tree_node_count'):
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Add gauges
            if 'gauges' not in data:
                data['gauges'] = {}
            data['gauges']['tree_node_count'] = self.tree_node_count

            # Add ContinuousRunner-specific metrics
            if hasattr(self, 'throughput'):
                data['gauges']['throughput_sims_per_sec'] = self.throughput
                data['gauges']['time_per_sim_us'] = self.time_per_sim_us
                data['gauges']['wall_clock_time_sec'] = self.wall_clock_time
                data['gauges']['root_visits'] = self.root_visits

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"✅ JSON report:        {json_path}")
        print(f"✅ Chrome trace:       {chrome_path}")
        print(f"✅ Markdown report:    {markdown_path}")

    def export_python_results(self):
        """Export Python profiling results"""
        print("\n" + "="*60)
        print("PYTHON RESULTS EXPORT")
        print("="*60)

        if self.python_session:
            python_path = self.output_dir / "python_profiling.json"

            # Export Python metrics
            metrics = self.python_session.get_all_metrics()
            with open(python_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"✅ Python metrics:     {python_path}")

    def analyze_bottlenecks(self):
        """Perform unified bottleneck analysis"""
        print("\n" + "="*60)
        print("BOTTLENECK ANALYSIS")
        print("="*60)

        # Print C++ summary to console
        print("\n--- C++ Metrics Summary ---")
        self.cpp_profiler.print_summary()

        # Load and analyze JSON results
        json_path = self.output_dir / "cpp_profiling.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                cpp_metrics = json.load(f)

            print("\n--- Key Bottleneck Indicators ---")

            # State cloning
            state_clone_count = cpp_metrics.get('state_clone_count', {}).get('total', 0)
            if state_clone_count > 0:
                print(f"🔴 State Cloning: {state_clone_count} clones")
                clones_per_sim = state_clone_count / max(1, self.args.simulations)
                if self.args.runner_type == "continuous":
                    print(f"   Expected: 1× per simulation, Actual: {clones_per_sim:.1f}×")
                    if clones_per_sim <= 1.1:  # Within 10% of target
                        print(f"   ✅ Target achieved (≤1.1× per sim)")
                    else:
                        print(f"   ⚠️  Higher than expected - investigate")
                else:
                    print(f"   Baseline (SimulationRunner): {clones_per_sim:.1f}× per simulation")

            # OpenMP
            omp_success = cpp_metrics.get('feature_extraction_omp', {}).get('total', 0)
            if omp_success == 0:
                print(f"🔴 OpenMP: NOT parallelizing (serial execution detected)")
            else:
                print(f"✅ OpenMP: Parallelizing successfully")

            # Thread idle
            thread_idle_ns = cpp_metrics.get('thread_idle_total', {}).get('total', 0)
            if thread_idle_ns > 0:
                idle_ms = thread_idle_ns / 1e6
                print(f"⚠️  Thread Idle: {idle_ms:.1f}ms total")

            # CAS retries
            cas_retries = cpp_metrics.get('cas_retry_count', {}).get('total', 0)
            if cas_retries > 100:
                print(f"⚠️  CAS Contention: {cas_retries} retries")

            # Performance comparison (ContinuousRunner only)
            if self.args.runner_type == "continuous" and hasattr(self, 'throughput'):
                print("\n--- Performance vs Baseline ---")
                baseline_throughput = 1650  # sims/sec (SimulationRunner)
                improvement = self.throughput / baseline_throughput
                target_throughput = 4700  # Expected from T024f-6
                target_progress = (self.throughput / target_throughput) * 100

                print(f"Baseline (SimulationRunner):  {baseline_throughput} sims/sec")
                print(f"Current (ContinuousRunner):   {self.throughput:.1f} sims/sec")
                print(f"Improvement:                  {improvement:.2f}× speedup")
                print(f"Target (T024f-6):             {target_throughput} sims/sec")
                print(f"Progress to target:           {target_progress:.1f}%")

                if improvement >= 1.77:
                    print(f"Status: ✅ Target achieved ({improvement:.2f}× ≥ 1.77×)")
                else:
                    gap = 1.77 - improvement
                    print(f"Status: ⚠️  Below target ({improvement:.2f}× < 1.77×)")
                    print(f"Gap:    {gap:.2f}× additional improvement needed")

        print("\n" + "="*60)

    def run(self) -> int:
        """Execute unified profiling workflow"""
        try:
            # Step 1: Validation (optional)
            if self.args.validate:
                if not self.validate():
                    return 1

            # Step 2: Setup profiling
            self.setup_cpp_profiling()
            self.python_session = self.setup_python_profiling()

            # Step 3: Run workload
            self.run_mcts_workload()

            # Step 4: Export results
            self.export_cpp_results()
            self.export_python_results()

            # Step 5: Analyze
            self.analyze_bottlenecks()

            print("\n" + "="*60)
            print("✅ PROFILING COMPLETE")
            print("="*60)
            print(f"\nResults saved to: {self.output_dir.absolute()}")
            print("\nNext steps:")
            print("1. Review cpp_report.md for comprehensive analysis")
            print("2. Open cpp_trace.json in chrome://tracing for timeline view")
            print("3. Check cpp_profiling.json for raw metrics")
            print()

            return 0

        except Exception as e:
            print(f"\n❌ Error during profiling: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            # Cleanup
            if self.python_session:
                self.python_session.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Profiling Orchestrator for MCTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations to run"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel MCTS threads"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Neural network batch size"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation before profiling"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="profiling_results",
        help="Output directory for profiling results"
    )

    parser.add_argument(
        "--runner-type",
        type=str,
        choices=["simulation", "continuous"],
        default="continuous",
        help="Runner type: 'simulation' (baseline, clone-based DEPRECATED) or 'continuous' (current, make/unmake with zero-copy)"
    )

    args = parser.parse_args()

    profiler = UnifiedProfiler(args)
    sys.exit(profiler.run())


if __name__ == "__main__":
    main()
