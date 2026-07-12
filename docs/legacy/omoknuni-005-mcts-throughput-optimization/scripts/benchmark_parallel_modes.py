#!/usr/bin/env python
"""Benchmark parallel modes for AlphaZeroMCTS.

This script runs a short benchmark for each supported parallel mode, including a
prototype thread-local experiment implemented at the Python orchestration layer.
Results are printed to stdout as a simple table.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import alphazero_py
from src.core.mcts import AlphaZeroMCTS
from src.neural.inference_worker import GPUInferenceWorker
from src.core.cpp_inference_bridge import CppInferenceBridge


@dataclass
class BenchmarkResult:
    mode: str
    sims_per_second: List[float]
    avg_sps: float
    std_sps: float


def noop_inference(state) -> Future:
    policy = np.ones(state.get_action_space_size(), dtype=np.float32)
    policy /= policy.sum()
    future: Future = Future()
    future.set_result((policy, 0.0))
    return future


def run_engine(
    mode: str,
    simulations: int,
    runs: int,
    num_threads: int,
    use_async: bool,
    async_batch_size: int,
    inference_fn_factory,
) -> BenchmarkResult:
    state = alphazero_py.GomokuState(board_size=15)
    sims_per_second: List[float] = []

    for _ in range(runs):
        engine = AlphaZeroMCTS(
            inference_fn=inference_fn_factory(),
            num_threads=num_threads,
            use_async_inference=use_async,
            async_batch_size=async_batch_size,
            async_timeout_ms=1.0,
            enable_instrumentation=True,
            parallel_mode=mode,
        )
        try:
            engine.reset_instrumentation_metrics()
            start = time.perf_counter()
            engine.search(state, simulations=simulations)
            elapsed = time.perf_counter() - start
            sims_per_second.append(simulations / elapsed if elapsed > 0 else 0.0)
        finally:
            engine.close()

    return BenchmarkResult(
        mode=mode,
        sims_per_second=sims_per_second,
        avg_sps=statistics.mean(sims_per_second),
        std_sps=statistics.pstdev(sims_per_second) if len(sims_per_second) > 1 else 0.0,
    )


def run_thread_local_prototype(
    simulations: int,
    runs: int,
    num_threads: int,
    use_async: bool,
    async_batch_size: int,
    inference_fn_factory,
) -> BenchmarkResult:
    state = alphazero_py.GomokuState(board_size=15)
    sims_per_second: List[float] = []

    for _ in range(runs):
        per_thread = simulations // num_threads
        remainder = simulations % num_threads

        start = time.perf_counter()
        for thread_id in range(num_threads):
            local_sims = per_thread + (1 if thread_id < remainder else 0)
            engine = AlphaZeroMCTS(
                inference_fn=inference_fn_factory(),
                num_threads=1,
                use_async_inference=use_async,
                async_batch_size=async_batch_size,
                async_timeout_ms=1.0,
                enable_instrumentation=False,
                parallel_mode="shared",
            )
            try:
                engine.search(state, simulations=local_sims)
            finally:
                engine.close()
        elapsed = time.perf_counter() - start
        sims_per_second.append(simulations / elapsed if elapsed > 0 else 0.0)

    return BenchmarkResult(
        mode="thread_local_prototype",
        sims_per_second=sims_per_second,
        avg_sps=statistics.mean(sims_per_second),
        std_sps=statistics.pstdev(sims_per_second) if len(sims_per_second) > 1 else 0.0,
    )


def benchmark_all(
    simulations: int,
    runs: int,
    num_threads: int,
    use_async: bool,
    async_batch_size: int,
    thread_counts: List[int],
    inference_fn_factory,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for threads in thread_counts:
        for mode in ("shared", "virtual_loss_free"):
            label = f"{mode}-{'async' if use_async else 'sync'}-{threads}t"
            result = run_engine(
                mode,
                simulations,
                runs,
                threads,
                use_async,
                async_batch_size,
                inference_fn_factory,
            )
            results.append(
                BenchmarkResult(
                    mode=label,
                    sims_per_second=result.sims_per_second,
                    avg_sps=result.avg_sps,
                    std_sps=result.std_sps,
                )
            )

        proto_result = run_thread_local_prototype(
            simulations,
            runs,
            threads,
            use_async,
            async_batch_size,
            inference_fn_factory,
        )
        results.append(
            BenchmarkResult(
                mode=f"thread_local_prototype-{'async' if use_async else 'sync'}-{threads}t",
                sims_per_second=proto_result.sims_per_second,
                avg_sps=proto_result.avg_sps,
                std_sps=proto_result.std_sps,
            )
        )
    return results


def format_results(results: List[BenchmarkResult]) -> str:
    header = f"{'Mode':<24} {'Avg sims/sec':>14} {'Std dev':>10}"
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(f"{result.mode:<24} {result.avg_sps:>14.1f} {result.std_sps:>10.1f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MCTS parallel modes")
    parser.add_argument("--simulations", type=int, default=256, help="Simulations per benchmark run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per mode")
    parser.add_argument("--threads", type=int, nargs="+", default=[4], help="Thread counts to benchmark")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Enable async inference")
    parser.add_argument("--batch", type=int, default=32, help="Async/GPU batch size")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPUInferenceWorker via CppInferenceBridge")
    parser.add_argument("--model-path", default=None, help="Path to trained model (optional)")
    parser.add_argument("--device", default="cuda:0", help="Torch device to use (default cuda:0)")
    parser.set_defaults(use_async=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_state = alphazero_py.GomokuState(board_size=15)
    sample_features = np.asarray(sample_state.get_tensor_representation(), dtype=np.float32)
    input_shape = tuple(sample_features.shape)

    gpu_worker: Optional[GPUInferenceWorker] = None
    bridge: Optional[CppInferenceBridge] = None
    gpu_enabled = False

    if args.use_gpu:
        try:
            gpu_worker = GPUInferenceWorker(
                model_path=args.model_path,
                device=args.device,
                batch_size=args.batch,
                timeout_ms=3.0,
            )
            gpu_worker.warmup(input_shape)
            bridge = CppInferenceBridge(gpu_worker)
            gpu_enabled = True

            def inference_factory():
                return bridge

        except Exception as exc:  # pragma: no cover - env-specific
            print(f"[warn] GPU initialization failed ({exc}); falling back to CPU inference.")
            gpu_worker = None
            bridge = None

    if not gpu_enabled:

        def inference_factory():
            return noop_inference

    results = benchmark_all(
        simulations=args.simulations,
        runs=args.runs,
        num_threads=max(args.threads),
        use_async=args.use_async,
        async_batch_size=args.batch,
        thread_counts=args.threads,
        inference_fn_factory=inference_factory,
    )
    print(format_results(results))

    if gpu_worker is not None:
        gpu_worker.stop_worker()


if __name__ == "__main__":
    main()
