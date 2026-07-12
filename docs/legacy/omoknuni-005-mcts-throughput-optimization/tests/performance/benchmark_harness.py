"""
Unified benchmark harness with CSV telemetry for MCTS performance validation.

This module implements the benchmark harness infrastructure defined in
spec.md v2.0 Task T001. It provides reproducible performance measurement
with comprehensive telemetry collection.

Usage:
    harness = BenchmarkHarness(output_dir="results/benchmarks")
    config = BenchmarkConfig(game="gomoku", num_simulations=10000, num_threads=4)
    result = harness.run_benchmark(config, iterations=10)
    print(f"Throughput: {result.mean_throughput:.0f} ± {result.std_throughput:.0f} sims/sec")
"""

import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

from tests.performance.telemetry import Telemetry, BenchmarkStatistics
from tests.performance.fixtures import BenchmarkConfig


class BenchmarkHarness:
    """Unified benchmark harness with telemetry and CSV output."""

    def __init__(self, output_dir: str = "results/benchmarks"):
        """
        Initialize benchmark harness.

        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_csv = self.output_dir / "benchmark_history.csv"

        # Initialize CSV file with headers if it doesn't exist
        if not self.history_csv.exists():
            self._initialize_csv()

        # GPU handle for measurements
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    def _initialize_csv(self):
        """Initialize CSV file with header row."""
        # Get all field names from Telemetry dataclass
        from dataclasses import fields
        telemetry_fields = [f.name for f in fields(Telemetry)]

        with open(self.history_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=telemetry_fields)
            writer.writeheader()

    def run_benchmark(
        self,
        config: BenchmarkConfig,
        iterations: int = 10,
        warmup_runs: int = 2
    ) -> BenchmarkStatistics:
        """
        Run benchmark with specified configuration.

        Args:
            config: Benchmark configuration
            iterations: Number of runs for statistical validation
            warmup_runs: Number of warmup runs (not counted in stats)

        Returns:
            BenchmarkStatistics with mean, std, CV, and target validation
        """
        print(f"="*70)
        print(f"Benchmark: {config.game} @ {config.num_threads} threads")
        print(f"Simulations: {config.num_simulations}, Batch: {config.batch_size}")
        print(f"Iterations: {iterations} (+ {warmup_runs} warmup)")
        print(f"="*70)

        # Set environment variables
        self._setup_environment(config)

        # Warmup runs
        print(f"\nWarmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            self._run_single_benchmark(config, is_warmup=True)
            print(f"  Warmup {i+1}/{warmup_runs} complete")

        # Measurement runs
        print(f"\nMeasurement ({iterations} runs)...")
        measurements: List[Telemetry] = []

        for i in range(iterations):
            # Set seed for reproducibility
            np.random.seed(config.seed + i)
            if TORCH_AVAILABLE:
                torch.manual_seed(config.seed + i)

            # Run single benchmark
            telemetry = self._run_single_benchmark(config, is_warmup=False)
            measurements.append(telemetry)

            # Save to CSV immediately (incremental)
            self._append_to_csv(telemetry)

            # Progress update
            print(f"  Run {i+1}/{iterations}: {telemetry.throughput:.0f} sims/sec, "
                  f"GPU {telemetry.gpu_util_percent:.1f}%, "
                  f"Batch {telemetry.avg_batch_size:.1f}")

        # Compute statistics
        stats = BenchmarkStatistics.from_telemetry_list(measurements)

        # Print summary
        self._print_summary(stats)

        return stats

    def _setup_environment(self, config: BenchmarkConfig):
        """Set up environment variables for benchmark."""
        # OpenMP configuration
        if config.omp_num_threads is not None:
            os.environ['OMP_NUM_THREADS'] = str(config.omp_num_threads)
        elif config.openmp_enabled:
            # Default to 12 threads for Ryzen 5900X (per CLARIFICATIONS.md)
            os.environ['OMP_NUM_THREADS'] = '12'

        # Disable nested parallelism (conflicts with MCTS threads)
        os.environ['OMP_NESTED'] = 'FALSE'

        # CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_device)

    def _run_single_benchmark(
        self,
        config: BenchmarkConfig,
        is_warmup: bool = False
    ) -> Telemetry:
        """
        Run single benchmark iteration with full telemetry.

        Args:
            config: Benchmark configuration
            is_warmup: Whether this is a warmup run

        Returns:
            Telemetry with all collected metrics
        """
        # Import here to avoid circular dependencies
        from src.games import create_game
        from src.core.mcts import MCTS
        from src.neural.network import create_model

        # Create telemetry object
        telemetry = Telemetry()
        telemetry.config = config.to_dict()

        # Set feature flags from config
        telemetry.openmp_enabled = config.openmp_enabled
        telemetry.state_pooling_enabled = config.state_pooling_enabled
        telemetry.condition_vars_enabled = config.condition_vars_enabled
        telemetry.node_allocator_optimized = config.node_allocator_optimized
        telemetry.nn_cache_enabled = config.nn_cache_enabled
        telemetry.fp16_enabled = config.fp16_enabled
        telemetry.root_preexpansion_enabled = config.root_preexpansion_enabled

        # Set thread count
        telemetry.num_threads = config.num_threads

        # Create game
        game = create_game(config.game)

        # Create or load neural network
        if config.model_path:
            model = torch.load(config.model_path)
        else:
            # Use random initialization for testing
            model = create_model(config.game, board_size=config.board_size)

        if TORCH_AVAILABLE and torch.cuda.is_available():
            model = model.cuda()
            if config.fp16_enabled:
                model = model.half()
            telemetry.pytorch_version = torch.__version__
            telemetry.cuda_version = torch.version.cuda or "unknown"

        # Get GPU model name
        if self.gpu_handle:
            try:
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                telemetry.gpu_model = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
            except Exception:
                pass

        # Create MCTS instance
        mcts = MCTS(
            game=game,
            model=model,
            num_simulations=config.num_simulations,
            num_threads=config.num_threads,
            c_puct=config.c_puct,
            temperature=config.temperature,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            virtual_loss=config.virtual_loss,
            batch_size=config.batch_size,
            batch_timeout_ms=config.batch_timeout_ms,
        )

        # Get initial memory baseline
        mem_before = 0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Run MCTS search
        root_state = game.get_initial_state()
        start_time = time.perf_counter()

        # Measure GPU utilization during search
        gpu_samples = []
        if self.gpu_handle:
            gpu_samples = self._measure_gpu_during_search(mcts, root_state)
        else:
            # No GPU monitoring, just run search
            _ = mcts.search(root_state)

        elapsed = time.perf_counter() - start_time

        # Get final memory
        mem_after = 0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB

        # Compute throughput
        telemetry.throughput = config.num_simulations / elapsed
        telemetry.total_time_sec = elapsed

        # Memory metrics
        telemetry.memory_rss_mb = mem_after
        telemetry.memory_peak_mb = mem_after  # Approximate (actual peak requires tracemalloc)

        # GPU metrics
        if gpu_samples:
            telemetry.gpu_util_percent = np.mean([s['util'] for s in gpu_samples])
            telemetry.gpu_memory_mb = np.mean([s['memory'] for s in gpu_samples])

        # MCTS metrics from tree statistics
        if hasattr(mcts, 'get_stats'):
            stats = mcts.get_stats()
            telemetry.tree_size_nodes = stats.get('total_nodes', 0)
            telemetry.avg_batch_size = stats.get('avg_batch_size', 0)
            telemetry.batches_submitted = stats.get('total_batches', 0)
            telemetry.thread_idle_percent = stats.get('thread_idle_percent', 0)
            telemetry.virtual_loss_collisions = stats.get('vl_collisions', 0)
            telemetry.expansion_races = stats.get('expansion_races', 0)

        # Compute derived metrics
        telemetry.compute_derived_metrics()

        return telemetry

    def _measure_gpu_during_search(self, mcts, root_state) -> List[dict]:
        """
        Measure GPU utilization during MCTS search.

        Uses threading to sample GPU metrics while search runs.
        """
        import threading

        gpu_samples = []
        stop_sampling = threading.Event()

        def sample_gpu():
            """Sample GPU utilization in background thread."""
            while not stop_sampling.is_set():
                try:
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_samples.append({
                        'util': util_rates.gpu,
                        'memory': mem_info.used / (1024 * 1024),  # MB
                    })
                except Exception:
                    pass
                time.sleep(0.1)  # Sample every 100ms

        # Start sampling thread
        sample_thread = threading.Thread(target=sample_gpu, daemon=True)
        sample_thread.start()

        # Run search
        try:
            _ = mcts.search(root_state)
        finally:
            # Stop sampling
            stop_sampling.set()
            sample_thread.join(timeout=1.0)

        return gpu_samples

    def _append_to_csv(self, telemetry: Telemetry):
        """Append telemetry to CSV history file."""
        row = telemetry.to_csv_row()

        # Get field names from existing CSV
        with open(self.history_csv, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        # Append row
        with open(self.history_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(row)

    def _print_summary(self, stats: BenchmarkStatistics):
        """Print benchmark summary statistics."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Throughput:     {stats.mean_throughput:7.0f} ± {stats.std_throughput:5.0f} sims/sec "
              f"(CV: {stats.cv_throughput*100:4.1f}%)")
        print(f"GPU Util:       {stats.mean_gpu_util:7.1f} ± {stats.std_gpu_util:5.1f} % "
              f"(CV: {stats.cv_gpu_util*100:4.1f}%)")
        print(f"Batch Size:     {stats.mean_batch_size:7.1f} ± {stats.std_batch_size:5.1f}")
        print(f"Thread Idle:    {stats.mean_thread_idle:7.1f}%")
        print(f"Runs:           {stats.num_runs}")
        print(f"\nTarget Validation:")
        print(f"  Throughput ≥8k:  {'✅ PASS' if stats.meets_throughput_target else '❌ FAIL'}")
        print(f"  GPU 80-95%:      {'✅ PASS' if stats.meets_gpu_target else '❌ FAIL'}")
        print(f"  Memory <1GB:     {'✅ PASS' if stats.meets_memory_target else '❌ FAIL'}")
        print(f"{'='*70}\n")

    def load_history(self) -> List[Telemetry]:
        """Load all telemetry from CSV history file."""
        telemetry_list = []

        if not self.history_csv.exists():
            return telemetry_list

        import pandas as pd
        df = pd.read_csv(self.history_csv)

        for _, row in df.iterrows():
            t = Telemetry()
            # Populate fields from CSV row
            for key, value in row.items():
                if hasattr(t, key) and pd.notna(value):
                    setattr(t, key, value)
            telemetry_list.append(t)

        return telemetry_list


def main():
    """Command-line interface for benchmark harness."""
    import argparse

    parser = argparse.ArgumentParser(description="MCTS Benchmark Harness")
    parser.add_argument("--game", default="gomoku", choices=["gomoku", "chess", "go"])
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        game=args.game,
        num_simulations=args.simulations,
        num_threads=args.threads,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Run benchmark
    harness = BenchmarkHarness(output_dir=args.output_dir)
    stats = harness.run_benchmark(config, iterations=args.iterations)

    # Exit with error code if targets not met
    if not (stats.meets_throughput_target and stats.meets_gpu_target):
        sys.exit(1)


if __name__ == "__main__":
    main()
