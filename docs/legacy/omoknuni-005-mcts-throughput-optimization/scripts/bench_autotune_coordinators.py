#!/usr/bin/env python3
"""
Coordinator Count Auto-Tuner
============================

Automatically determines optimal number of parallel coordinators for multi-stream
GPU inference by running micro-benchmarks with K∈{1,2,3,4} coordinators.

Purpose:
- Find optimal K that maximizes throughput for given hardware (RTX 3060 Ti)
- Account for GPU memory, stream concurrency, and GIL contention
- Persist result to ~/.mcts_autotune.json for runtime loading
- Validate stability (run twice, ensure K matches or differs by ≤1)

Algorithm:
1. For each K in {1, 2, 3, 4}:
   - Create K coordinators with dedicated CUDA streams
   - Run 100 simulations × K coordinators (3-5s micro-benchmark)
   - Measure p95 throughput (sims/sec)
2. Select K with highest p95 throughput
3. Persist to ~/.mcts_autotune.json
4. Validate stability (run tuner twice, check consistency)

Expected Results (RTX 3060 Ti, Ghost-ECA 96×12):
- K=1: ~7,000 sims/sec (baseline)
- K=2: ~11,000 sims/sec (1.57× scaling, 78% efficiency)
- K=3: ~14,000 sims/sec (2.0× scaling, 67% efficiency)  ← EXPECTED OPTIMUM
- K=4: ~15,000 sims/sec (2.14× scaling, 54% efficiency, diminishing returns)

Usage:
    # Auto-tune and save result
    python scripts/bench_autotune_coordinators.py

    # Force re-tune (ignore existing config)
    python scripts/bench_autotune_coordinators.py --force

    # Validate stability (run twice, check consistency)
    python scripts/bench_autotune_coordinators.py --validate

    # Dry-run (don't save result)
    python scripts/bench_autotune_coordinators.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.search_coordinator import MultiCoordinatorManager
from src.neural.model import create_ghost_resnet_eca_model
from src.core.dlpack_inference_bridge import DLPackInferenceBridge
import alphazero_py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_gpu_model() -> str:
    """Detect GPU model name for config persistence.

    Returns:
        GPU model string (e.g., "GA104" for RTX 3060 Ti)
    """
    if not torch.cuda.is_available():
        return "CPU"

    try:
        props = torch.cuda.get_device_properties(0)
        # Extract model identifier (e.g., "NVIDIA GeForce RTX 3060 Ti" → "GA104")
        model_name = props.name
        if "3060 Ti" in model_name or "3060Ti" in model_name:
            return "GA104"  # Ampere GA104 architecture
        elif "3090" in model_name or "3080" in model_name:
            return "GA102"  # Ampere GA102 architecture
        elif "4090" in model_name or "4080" in model_name:
            return "AD102"  # Ada Lovelace architecture
        else:
            return model_name.replace("NVIDIA GeForce ", "").replace(" ", "_")
    except Exception as e:
        logger.warning(f"Failed to detect GPU model: {e}")
        return "Unknown"


def create_dummy_inference_callback(model, device='cuda'):
    """Create inference callback for benchmarking.

    Args:
        model: PyTorch model for inference
        device: Device to run inference on

    Returns:
        DLPackInferenceBridge instance
    """
    bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        enable_buffer_pool=True
    )

    # Warmup
    logger.info("Warming up model...")
    bridge.warmup(batch_size=64, game_type='gomoku')
    logger.info("Warmup complete")

    return bridge


def run_benchmark_trial(num_coordinators: int,
                       simulations: int = 100,
                       timeout_sec: float = 5.0) -> float:
    """Run single benchmark trial with K coordinators.

    Args:
        num_coordinators: Number of parallel coordinators
        simulations: Number of MCTS simulations per trial
        timeout_sec: Maximum time for trial (seconds)

    Returns:
        Measured throughput (sims/sec)
    """
    try:
        import mcts_py

        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_ghost_resnet_eca_model('gomoku')  # Default 96×12
        model = model.to(device)
        model.eval()

        # Note: For benchmarking, we'll just measure coordinator throughput with direct queue submission
        # (not full MCTS simulation which requires more complex setup)

        # Create shared inference queue
        queue = mcts_py.AsyncInferenceQueue()

        # Create batch callback
        def batch_callback_fn(requests):
            """Batch inference callback."""
            batch_size = len(requests)
            if batch_size == 0:
                return np.zeros((0, 225)), np.zeros(0)

            # Extract features from requests
            features_list = []
            for req in requests:
                features = np.array(req.features).reshape(req.planes, req.board_size, req.board_size)
                features_list.append(features)

            # Stack into batch
            features_batch = np.stack(features_list, axis=0)

            # Convert to tensor
            features_tensor = torch.from_numpy(features_batch).to(device, dtype=torch.float32)

            # Run inference
            with torch.no_grad():
                policy_logits, values = model(features_tensor)

            # Convert to numpy
            policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            values = values.cpu().numpy().flatten()

            return policies, values

        callback = mcts_py.PyBatchInferenceCallback(batch_callback_fn)

        # Create multi-coordinator manager (or single coordinator if K=1)
        if num_coordinators == 1:
            # Single coordinator (baseline)
            coordinator = mcts_py.BatchInferenceCoordinator()
            coordinator.start(queue, callback, batch_size=64, timeout_ms=5.0)
            coordinator_manager = None
        else:
            # Multi-coordinator
            manager = MultiCoordinatorManager(
                queue=queue,
                callback=callback,
                batch_size=64,
                timeout_ms=5.0,
                num_coordinators=num_coordinators
            )
            manager.start()
            coordinator_manager = manager

        # Create initial game state
        game = alphazero_py.GomokuState()

        # Run simulations and measure throughput
        start_time = time.perf_counter()

        try:
            successes = runner.run_continuous(game, root=0, queue=queue, simulations=simulations)

            # Wait for all results (with timeout)
            wait_start = time.perf_counter()
            while queue.pending_count() > 0 and (time.perf_counter() - wait_start) < timeout_sec:
                time.sleep(0.01)

        finally:
            # Stop coordinator(s)
            if coordinator_manager:
                coordinator_manager.stop()
            else:
                coordinator.stop()

        elapsed = time.perf_counter() - start_time

        # Calculate throughput
        throughput = simulations / elapsed if elapsed > 0 else 0.0

        return throughput

    except Exception as e:
        logger.error(f"Benchmark trial failed with K={num_coordinators}: {e}")
        return 0.0


def benchmark_coordinator_count(num_coordinators: int,
                                iterations: int = 5) -> Tuple[float, float, List[float]]:
    """Benchmark throughput with K coordinators.

    Args:
        num_coordinators: Number of parallel coordinators
        iterations: Number of trials to run

    Returns:
        Tuple of (mean_throughput, p95_throughput, all_samples)
    """
    logger.info(f"Benchmarking K={num_coordinators} coordinators ({iterations} trials)...")

    samples = []
    for i in range(iterations):
        throughput = run_benchmark_trial(num_coordinators, simulations=100, timeout_sec=10.0)
        samples.append(throughput)
        logger.info(f"  Trial {i+1}/{iterations}: {throughput:.1f} sims/sec")

    mean_throughput = np.mean(samples)
    p95_throughput = np.percentile(samples, 95)

    logger.info(f"  Results: mean={mean_throughput:.1f} sims/sec, p95={p95_throughput:.1f} sims/sec")

    return mean_throughput, p95_throughput, samples


def run_autotune_campaign() -> Dict[str, any]:
    """Run full auto-tuning campaign across K∈{1,2,3,4}.

    Returns:
        Dict with optimal config and benchmark results
    """
    logger.info("=" * 80)
    logger.info("COORDINATOR AUTO-TUNER")
    logger.info("=" * 80)

    # Detect GPU
    gpu_model = detect_gpu_model()
    logger.info(f"Detected GPU: {gpu_model}")

    # Benchmark each coordinator count
    results = {}
    for K in [1, 2, 3, 4]:
        mean_tput, p95_tput, samples = benchmark_coordinator_count(K, iterations=5)
        results[K] = {
            'mean_throughput': mean_tput,
            'p95_throughput': p95_tput,
            'samples': samples
        }

    # Select optimal K (highest p95 throughput)
    optimal_K = max(results.keys(), key=lambda k: results[k]['p95_throughput'])
    optimal_throughput = results[optimal_K]['p95_throughput']

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    for K in sorted(results.keys()):
        mean_tput = results[K]['mean_throughput']
        p95_tput = results[K]['p95_throughput']
        baseline_tput = results[1]['mean_throughput']
        speedup = mean_tput / baseline_tput if baseline_tput > 0 else 0.0
        efficiency = (speedup / K) * 100 if K > 0 else 0.0

        marker = " ← OPTIMAL" if K == optimal_K else ""
        logger.info(f"K={K}: mean={mean_tput:7.1f} sims/sec, p95={p95_tput:7.1f} sims/sec, "
                   f"speedup={speedup:.2f}×, efficiency={efficiency:.1f}%{marker}")

    logger.info("=" * 80)
    logger.info(f"SELECTED: K={optimal_K} coordinators ({optimal_throughput:.1f} sims/sec)")
    logger.info("=" * 80)

    # Build config
    config = {
        'gpu_model': gpu_model,
        'optimal_coordinators': optimal_K,
        'measured_throughput': optimal_throughput,
        'timestamp': datetime.now().isoformat(),
        'benchmark_results': {
            str(k): {
                'mean_throughput': v['mean_throughput'],
                'p95_throughput': v['p95_throughput']
            }
            for k, v in results.items()
        }
    }

    return config


def save_config(config: Dict, config_path: str = "~/.mcts_autotune.json") -> None:
    """Save auto-tuned config to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config (default: ~/.mcts_autotune.json)
    """
    config_path = os.path.expanduser(config_path)

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved auto-tune config to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def load_config(config_path: str = "~/.mcts_autotune.json") -> Dict:
    """Load existing auto-tuned config.

    Args:
        config_path: Path to load config from

    Returns:
        Configuration dictionary
    """
    config_path = os.path.expanduser(config_path)

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return None


def validate_stability() -> bool:
    """Validate auto-tuner stability by running twice and checking consistency.

    Returns:
        True if results are consistent (K matches or differs by ≤1)
    """
    logger.info("Validating auto-tuner stability...")
    logger.info("")

    # Run first tuning
    logger.info("=" * 80)
    logger.info("RUN 1")
    logger.info("=" * 80)
    config1 = run_autotune_campaign()
    K1 = config1['optimal_coordinators']

    logger.info("")
    logger.info("Waiting 2 seconds before run 2...")
    time.sleep(2)
    logger.info("")

    # Run second tuning
    logger.info("=" * 80)
    logger.info("RUN 2")
    logger.info("=" * 80)
    config2 = run_autotune_campaign()
    K2 = config2['optimal_coordinators']

    # Check consistency
    consistent = abs(K1 - K2) <= 1

    logger.info("")
    logger.info("=" * 80)
    logger.info("STABILITY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Run 1: K={K1}")
    logger.info(f"Run 2: K={K2}")
    logger.info(f"Difference: {abs(K1 - K2)}")
    logger.info(f"Consistent: {'✅ YES' if consistent else '❌ NO'}")
    logger.info("=" * 80)

    return consistent


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune optimal coordinator count for multi-stream GPU inference"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force re-tuning even if config exists"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Run benchmarks but don't save result"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help="Validate stability by running twice and checking consistency"
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default="~/.mcts_autotune.json",
        help="Path to save/load config (default: ~/.mcts_autotune.json)"
    )

    args = parser.parse_args()

    # Check for existing config
    if not args.force and not args.validate:
        existing_config = load_config(args.config_path)
        if existing_config:
            logger.info(f"Found existing config at {os.path.expanduser(args.config_path)}")
            logger.info(f"  Optimal coordinators: {existing_config.get('optimal_coordinators')}")
            logger.info(f"  Measured throughput: {existing_config.get('measured_throughput'):.1f} sims/sec")
            logger.info(f"  Tuned: {existing_config.get('timestamp')}")
            logger.info("")
            logger.info("Use --force to re-tune")
            return 0

    # Validate stability if requested
    if args.validate:
        consistent = validate_stability()
        return 0 if consistent else 1

    # Run auto-tuning campaign
    config = run_autotune_campaign()

    # Save config (unless dry-run)
    if not args.dry_run:
        save_config(config, args.config_path)
    else:
        logger.info("DRY-RUN: Config not saved")

    logger.info("")
    logger.info("Auto-tuning complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
