#!/usr/bin/env python3
"""pending_flow_lab — count-only WU-UCT pending-flow, fixed vs adaptive VL.

Part of the metacognitive experiment family (see
``docs/METACOGNITIVE_EXPERIMENTS.md``). A *separate model family*: an abstract
discrete-event simulation of the parallel-MCTS pending-flow mechanism, stripped
to the collision/throughput dynamics that virtual loss (VL) is supposed to
manage. It is the synthetic screen that gates the real Rust VL ablation
(``src/ablation_vl.rs``); wall-clock efficacy is measured there, not here.

## Model

One root with ``K`` arms carrying fixed UCT value proxies ``Q_i``. ``W`` workers
act in waves. Within a wave each worker is assigned an arm one at a time and
immediately marks it **pending** (in-flight), so later workers in the same wave
see the updated pending state — this is exactly how VL de-collides parallel
selection. Selection is count-only WU-UCT:

    eff_i   = n_i + vl_weight * p_i               # count-only virtual loss
    score_i = Q_i + c * sqrt(ln(N + P) / (eff_i + 1))

with ``n_i`` completed visits, ``p_i`` pending, ``N``/``P`` their totals. The
three VL policies differ only in ``vl_weight``:

* ``disabled`` — ``vl_weight = 0``: pending is invisible; every worker piles onto
  the current argmax (maximum duplication);
* ``fixed`` — ``vl_weight = 1``: each pending counts as one virtual visit
  (standard WU-UCT);
* ``adaptive`` — ``vl_weight = amplifier`` where
  ``amplifier = 1 + dup_rate_ema * (1 + max_pending / W)`` (the
  ``src/ablation_vl.rs`` feedback controller): pending is penalized harder under
  measured contention.

A pending started at wave ``t`` completes at wave ``t + latency`` (elastic
micro-wave), converting to a real visit. Common random numbers (per-wave,
per-worker jitter) are shared across policies so the comparison is paired.

## Metrics & kill checks (CCoT H4 / H5)

* ``dup_rate`` — mean fraction of workers per wave that collided onto an
  already-selected arm;
* ``throughput`` — mean unique arms selected per wave / W (parallel utilization);
* ``best_arm_visit_share`` — quality guard (VL must not over-spread and abandon
  the best arm).

Kill (H5): adaptive VL must reduce ``dup_rate`` vs fixed **and** the improvement
must be *thread-count dependent* (larger at high W than low W — "high-thread
only"). Kill (H4): adaptive must raise ``throughput`` at high W. No improvement
=> the adaptive-VL lane dies before any wall-clock claim.

Prohibited (claim firewall): reading these synthetic collision counts as a
wall-clock speedup, a play-strength change, or a CPU/energy claim — the real
timing lives in the Rust bridge, and even that is throughput, not strength.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Sequence, Tuple

from quartz.experiments.bernoulli_root import stable_seed

EXPERIMENT_ID = "pending_flow_lab_v1"
EXECUTION_MODE = "synthetic_screening"
PENDING_FLOW_SCHEMA_VERSION = 1

VL_POLICIES: Tuple[str, ...] = ("disabled", "fixed", "adaptive")
_DUP_EMA_DECAY = 0.5  # feedback smoothing for the adaptive amplifier


def _selection_scores(
    q: Sequence[float],
    n: Sequence[int],
    p: Sequence[int],
    vl_weight: float,
    c_puct: float,
) -> List[float]:
    total = sum(n) + sum(p)
    ln_total = math.log(max(2.0, float(total)))
    scores = []
    for i in range(len(q)):
        eff = n[i] + vl_weight * p[i]
        scores.append(q[i] + c_puct * math.sqrt(ln_total / (eff + 1.0)))
    return scores


def simulate(
    arm_values: Sequence[float],
    n_workers: int,
    vl_policy: str,
    *,
    waves: int = 200,
    latency: int = 2,
    c_puct: float = 1.4,
    warmup_waves: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run one pending-flow configuration and return collision/throughput
    metrics (means over the post-warmup waves)."""
    if vl_policy not in VL_POLICIES:
        raise ValueError(f"unknown vl_policy {vl_policy!r}")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    k = len(arm_values)
    if k < 2:
        raise ValueError("at least two arms are required")
    q = [float(v) for v in arm_values]
    n = [0] * k
    p = [0] * k
    completion_queue: List[Tuple[int, int]] = []  # (completion_wave, arm)
    best_arm = max(range(k), key=lambda i: q[i])

    dup_ema = 0.0
    max_pending_seen = 0
    dup_rates: List[float] = []
    throughputs: List[float] = []
    best_share_samples: List[float] = []

    for wave in range(waves):
        # complete any pending evaluations scheduled for this wave
        still: List[Tuple[int, int]] = []
        for done_wave, arm in completion_queue:
            if done_wave <= wave:
                n[arm] += 1
                p[arm] = max(0, p[arm] - 1)
            else:
                still.append((done_wave, arm))
        completion_queue = still

        max_pending = max(p) if p else 0
        if vl_policy == "disabled":
            vl_weight = 0.0
        elif vl_policy == "fixed":
            vl_weight = 1.0
        else:  # adaptive feedback controller
            vl_weight = 1.0 + dup_ema * (1.0 + max_pending / float(n_workers))

        selected_this_wave: List[int] = []
        seen: Dict[int, int] = {}
        collisions = 0
        for worker in range(n_workers):
            scores = _selection_scores(q, n, p, vl_weight, c_puct)
            top = max(scores)
            # CRN tie-break: seeded jitter independent of the VL policy
            jr = random.Random(stable_seed(seed, "jitter", wave, worker))
            candidates = [i for i, s in enumerate(scores) if s >= top - 1e-12]
            arm = min(candidates, key=lambda i: (jr.random(), i)) if len(candidates) > 1 else candidates[0]
            if arm in seen:
                collisions += 1
            seen[arm] = seen.get(arm, 0) + 1
            selected_this_wave.append(arm)
            p[arm] += 1
            completion_queue.append((wave + latency, arm))

        unique = len(set(selected_this_wave))
        dup_rate = collisions / float(n_workers)
        throughput = unique / float(n_workers)
        dup_ema = (1.0 - _DUP_EMA_DECAY) * dup_ema + _DUP_EMA_DECAY * dup_rate
        max_pending_seen = max(max_pending_seen, max(p) if p else 0)

        if wave >= warmup_waves:
            dup_rates.append(dup_rate)
            throughputs.append(throughput)
            total_visits = sum(n)
            best_share_samples.append(n[best_arm] / total_visits if total_visits > 0 else 0.0)

    return {
        "vl_policy": vl_policy,
        "n_workers": n_workers,
        "n_arms": k,
        "waves": waves,
        "latency": latency,
        "mean_dup_rate": statistics.fmean(dup_rates) if dup_rates else 0.0,
        "mean_throughput": statistics.fmean(throughputs) if throughputs else 0.0,
        "max_pending": int(max_pending_seen),
        "best_arm_visit_share": statistics.fmean(best_share_samples) if best_share_samples else 0.0,
        "final_total_visits": int(sum(n)),
    }


def screen(
    arm_values: Sequence[float],
    worker_grid: Sequence[int],
    *,
    policies: Sequence[str] = VL_POLICIES,
    waves: int = 200,
    latency: int = 2,
    c_puct: float = 1.4,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for w in worker_grid:
        for policy in policies:
            rows.append(
                simulate(
                    arm_values, w, policy, waves=waves, latency=latency, c_puct=c_puct, seed=seed
                )
            )
    return rows


def _row(rows: Sequence[Dict[str, Any]], policy: str, w: int) -> Dict[str, Any] | None:
    for r in rows:
        if r["vl_policy"] == policy and r["n_workers"] == w:
            return r
    return None


def kill_verdict(
    rows: Sequence[Dict[str, Any]], worker_grid: Sequence[int], *, min_effect: float = 0.02
) -> Dict[str, Any]:
    """H5 (dup_rate, thread-dependent) and H4 (throughput) kill checks for the
    adaptive-VL lane vs fixed VL.

    H5 alive iff adaptive lowers dup_rate at the highest W by a *material*
    margin (``min_effect``, default 2 percentage points) AND that improvement
    is materially larger than at the lowest multi-worker W (a thread-count
    interaction, not a flat offset). H4 alive iff adaptive raises throughput at
    the highest W by ``min_effect``. ``min_effect`` guards against a sign-only
    test firing on noise-level differences (adaptive and fixed are near-identical
    in this abstract collision model)."""
    grid = sorted(set(int(w) for w in worker_grid))
    multi = [w for w in grid if w >= 2]
    if not multi:
        return {
            "pending_flow_schema_version": PENDING_FLOW_SCHEMA_VERSION,
            "insufficient_worker_grid": True,
            "h5_adaptive_dup_lane_alive": False,
            "h4_adaptive_throughput_lane_alive": False,
        }
    low_w, high_w = multi[0], multi[-1]

    def dup_gain(w: int) -> float | None:  # fixed - adaptive (positive = adaptive better)
        fx, ad = _row(rows, "fixed", w), _row(rows, "adaptive", w)
        if fx is None or ad is None:
            return None
        return fx["mean_dup_rate"] - ad["mean_dup_rate"]

    def thr_gain(w: int) -> float | None:  # adaptive - fixed (positive = adaptive better)
        fx, ad = _row(rows, "fixed", w), _row(rows, "adaptive", w)
        if fx is None or ad is None:
            return None
        return ad["mean_throughput"] - fx["mean_throughput"]

    dup_high, dup_low = dup_gain(high_w), dup_gain(low_w)
    thr_high = thr_gain(high_w)
    # quality guard: adaptive must not collapse concentration on the best arm
    ad_high = _row(rows, "adaptive", high_w)
    fx_high = _row(rows, "fixed", high_w)
    quality_ok = bool(
        ad_high is not None
        and fx_high is not None
        and ad_high["best_arm_visit_share"] >= 0.8 * fx_high["best_arm_visit_share"]
    )

    h5_improves = dup_high is not None and dup_high > min_effect
    h5_thread_dependent = (
        dup_high is not None and dup_low is not None and (dup_high - dup_low) > min_effect
    )
    h5_alive = bool(h5_improves and h5_thread_dependent)
    h4_alive = bool(thr_high is not None and thr_high > min_effect)

    return {
        "pending_flow_schema_version": PENDING_FLOW_SCHEMA_VERSION,
        "min_effect": min_effect,
        "low_worker": low_w,
        "high_worker": high_w,
        "dup_gain_high_w_fixed_minus_adaptive": dup_high,
        "dup_gain_low_w_fixed_minus_adaptive": dup_low,
        "dup_gain_thread_interaction_high_minus_low": (
            None if (dup_high is None or dup_low is None) else dup_high - dup_low
        ),
        "throughput_gain_high_w_adaptive_minus_fixed": thr_high,
        "quality_guard_ok": quality_ok,
        "h5_improves_dup_at_high_w": bool(h5_improves),
        "h5_thread_dependent": bool(h5_thread_dependent),
        "h5_adaptive_dup_lane_alive": h5_alive,
        "h4_adaptive_throughput_lane_alive": h4_alive,
    }
