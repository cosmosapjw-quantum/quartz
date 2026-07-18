#!/usr/bin/env python3
"""service_curve_lab — GPU evaluator service curve vs batch x inflight credit.

Part of the metacognitive experiment family (see
``docs/METACOGNITIVE_EXPERIMENTS.md``). A *separate model family* from the
others: this one takes **measured** GPU timings, not a synthetic screen. It
characterizes the neural evaluator's service curve — throughput (items/s) and
latency (ms/batch) — as a function of two knobs a parallel-MCTS scheduler
controls:

* **batch size** — how many leaf evaluations are coalesced per forward pass;
* **global inflight credit** — how many batches may be outstanding to the GPU at
  once (measured here with one CUDA stream per inflight slot).

## Why it matters

It is the throughput measurement that ``pending_flow_lab`` (Stage 5) explicitly
deferred, and the design input for any H4 scheduler ``W(t)``. THESIS.md P4 (the
CPU-friendliness gate) needs the evaluator's real service curve to reason about
operating points; this lab measures that curve. It makes **no** play-strength or
CPU-superiority claim — only quality-free throughput/latency, exactly the
re-scoped H4 mandate.

## Kill check (H4 scheduler lane)

The curve answers: does allowing more than one inflight batch give a *material*
throughput gain over simply picking the single best fixed batch at inflight 1?
If not, an inflight-credit scheduler buys nothing and the ``W(t)`` lane is
demoted — use the best fixed batch. (Inflight credit typically helps at small
batch, where a single forward underutilizes the GPU, and saturates at large
batch; the measured knee is the scheduler's efficient operating point.)

Prohibited (claim firewall): reading throughput as play strength; reading a GPU
service curve as a CPU-superiority claim; treating a representative conv body as
the exact shipped network.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Dict, List, Sequence

SERVICE_CURVE_SCHEMA_VERSION = 1
EXPERIMENT_ID = "service_curve_lab_v1"
EXECUTION_MODE = "measured_gpu_service_curve"


def build_eval_model(in_ch: int, channels: int, blocks: int, board: int, actions: int):
    """A representative AlphaZero-style evaluator body (conv stem + residual
    blocks + policy/value heads). Not the exact shipped network — a workload
    with a comparable compute profile, built with no checkpoint dependency."""
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(c)
            self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(c)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            y = self.act(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return self.act(x + y)

    class EvalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
            self.p_head = nn.Sequential(
                nn.Conv2d(channels, 2, 1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )
            self.p_fc = nn.Linear(2 * board * board, actions)
            self.v_head = nn.Sequential(
                nn.Conv2d(channels, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
            )
            self.v_fc1 = nn.Linear(board * board, channels)
            self.v_fc2 = nn.Linear(channels, 1)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            h = self.body(self.stem(x))
            p = self.p_fc(self.p_head(h).flatten(1))
            v = self.v_head(h).flatten(1)
            v = torch.tanh(self.v_fc2(self.act(self.v_fc1(v))))
            return p, v

    return EvalNet()


def _read_gpu_power_watts() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def measure_point(
    model,
    device,
    *,
    batch_size: int,
    inflight: int,
    in_ch: int,
    board: int,
    n_waves: int = 30,
    warmup: int = 8,
    time_fn: Callable[[], float] = time.perf_counter,
) -> Dict[str, Any]:
    """Measure throughput/latency at one (batch_size, inflight) point.

    ``inflight`` concurrent batches are submitted (one CUDA stream each on GPU),
    then synchronized once per wave, so the measurement captures how much the GPU
    benefits from overlapping in-flight work at that batch size."""
    import torch

    is_cuda = getattr(device, "type", str(device)) == "cuda"
    streams = (
        [torch.cuda.Stream() for _ in range(inflight)] if is_cuda else [None] * inflight
    )
    inputs = [
        torch.randn(batch_size, in_ch, board, board, device=device)
        for _ in range(inflight)
    ]

    def run_wave() -> None:
        with torch.no_grad():
            if is_cuda:
                for s, x in zip(streams, inputs):
                    with torch.cuda.stream(s):
                        model(x)
                torch.cuda.synchronize()
            else:
                for x in inputs:
                    model(x)

    for _ in range(warmup):
        run_wave()
    if is_cuda:
        torch.cuda.synchronize()

    power_before = _read_gpu_power_watts() if is_cuda else None
    start = time_fn()
    for _ in range(n_waves):
        run_wave()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = max(1e-9, time_fn() - start)
    power_after = _read_gpu_power_watts() if is_cuda else None

    items = batch_size * inflight * n_waves
    items_per_s = items / elapsed
    ms_per_batch = elapsed / (inflight * n_waves) * 1000.0
    power = None
    if power_before is not None and power_after is not None:
        power = 0.5 * (power_before + power_after)
    return {
        "batch_size": batch_size,
        "inflight": inflight,
        "items_per_s": items_per_s,
        "ms_per_batch": ms_per_batch,
        "n_waves": n_waves,
        "power_watts": power,
        "items_per_joule": (items_per_s / power) if (power and power > 0) else None,
    }


def service_curve(
    model,
    device,
    *,
    batch_sizes: Sequence[int],
    inflight_grid: Sequence[int],
    in_ch: int,
    board: int,
    n_waves: int = 30,
    warmup: int = 8,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for b in batch_sizes:
        for k in inflight_grid:
            rows.append(
                measure_point(
                    model,
                    device,
                    batch_size=b,
                    inflight=k,
                    in_ch=in_ch,
                    board=board,
                    n_waves=n_waves,
                    warmup=warmup,
                )
            )
    return rows


def scheduler_verdict(
    rows: Sequence[Dict[str, Any]], *, min_gain: float = 0.05
) -> Dict[str, Any]:
    """H4 scheduler-lane check: does inflight credit give a material throughput
    gain over the best fixed batch at inflight 1?

    ``min_gain`` is the fractional throughput improvement required to keep the
    inflight-credit / adaptive-``W(t)`` lane alive."""
    if not rows:
        return {
            "service_curve_schema_version": SERVICE_CURVE_SCHEMA_VERSION,
            "empty": True,
            "h4_inflight_scheduler_lane_alive": False,
        }
    fixed_rows = [r for r in rows if r["inflight"] == 1]
    best_fixed = max(fixed_rows, key=lambda r: r["items_per_s"]) if fixed_rows else None
    best_overall = max(rows, key=lambda r: r["items_per_s"])
    peak = best_overall["items_per_s"]

    gain = None
    if best_fixed and best_fixed["items_per_s"] > 0:
        gain = best_overall["items_per_s"] / best_fixed["items_per_s"] - 1.0

    # knee: smallest batch (at any inflight) reaching 90% of peak throughput
    knee = None
    for b in sorted({r["batch_size"] for r in rows}):
        best_at_b = max(
            (r for r in rows if r["batch_size"] == b), key=lambda r: r["items_per_s"]
        )
        if best_at_b["items_per_s"] >= 0.9 * peak:
            knee = {
                "batch_size": b,
                "inflight": best_at_b["inflight"],
                "items_per_s": best_at_b["items_per_s"],
            }
            break

    # where does inflight help most? largest per-batch gain from inflight>1
    per_batch_inflight_gain: List[Dict[str, Any]] = []
    for b in sorted({r["batch_size"] for r in rows}):
        at_b = [r for r in rows if r["batch_size"] == b]
        base = next((r for r in at_b if r["inflight"] == 1), None)
        best = max(at_b, key=lambda r: r["items_per_s"])
        if base and base["items_per_s"] > 0:
            per_batch_inflight_gain.append(
                {
                    "batch_size": b,
                    "best_inflight": best["inflight"],
                    "inflight_gain": best["items_per_s"] / base["items_per_s"] - 1.0,
                }
            )

    return {
        "service_curve_schema_version": SERVICE_CURVE_SCHEMA_VERSION,
        "min_gain": min_gain,
        "best_fixed_batch": (None if best_fixed is None else best_fixed["batch_size"]),
        "best_fixed_items_per_s": (
            None if best_fixed is None else best_fixed["items_per_s"]
        ),
        "best_overall_batch": best_overall["batch_size"],
        "best_overall_inflight": best_overall["inflight"],
        "best_overall_items_per_s": peak,
        "inflight_throughput_gain": gain,
        "knee": knee,
        "per_batch_inflight_gain": per_batch_inflight_gain,
        "h4_inflight_scheduler_lane_alive": bool(gain is not None and gain > min_gain),
    }
