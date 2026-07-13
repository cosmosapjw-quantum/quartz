"""forked_voc_lab — label each next computation by its realized root-decision change.

Part of the metacognitive experiment family (see
`docs/METACOGNITIVE_EXPERIMENTS.md`). This lab is a *separate model family*
from the Bernoulli root lab: it operates on **frozen search traces** (a budget
ladder of root policies, the phase15 trace-bundle format), not on IID arms.

## What it produces

Given a frozen trace `([b_0<b_1<...<b_n], [π_0,...,π_n])`, the "value of
computation" of the step `b_i → b_{i+1}` is *realized* as the change it caused
in the committed root decision. Two per-step labels:

- `argmax_flipped`: did the committed move change? (a discrete decision change)
- `decision_movement`: total-variation distance `TV(π_i, π_{i+1})` (continuous)

The per-position **VOC proxy** (the offline, oracle-style label that
`quartz.phase15_signatures.voc_tightness` consumes — THESIS.md P3
non-circularity guard: the engine never computes this at runtime) is the
shallow-vs-deep disagreement weighted by how much the decision margin moved:

    voc_proxy = 1[argmax(shallow) != argmax(deep)] * |margin(deep) - margin(shallow)|
                + (1 - flip) * total_decision_movement_over_ladder

so a position whose committed move is settled cheaply scores ~0, and a position
where deep search overturns the shallow pick (or churns the margin) scores high.

## Why it matters

It fills the exact gap flagged in `docs/RESEARCH_PLAN_PARTB.md` B1: O3/VOC
were "needs an offline oracle". This lab IS that oracle for frozen traces.
Its kill-criterion: if VOC labels are **degenerate** (all ~0 or all equal)
on real trained traces, the P3 discriminating signature design must be
reworked before any tightness claim.

Prohibited (claim firewall): reading a frozen-trace label as a runtime VOC
signal, or as proof that any allocator improves play — it is an offline
measurement substrate only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quartz.phase15_signatures import _as_prob, argmax_path, voc_tightness

EXPERIMENT_ID = "forked_voc_lab"
EXECUTION_MODE = "frozen_trace_labeling"
FORKED_VOC_SCHEMA_VERSION = 1

__all__ = [
    "EXPERIMENT_ID",
    "EXECUTION_MODE",
    "FORKED_VOC_SCHEMA_VERSION",
    "top2_margin",
    "total_variation",
    "computation_step_labels",
    "decision_movement_curve",
    "voc_proxy",
    "shallow_deep_margin_swing",
    "label_trace_bundle",
    "screen_bundles",
    "measure_tightness",
    "discrimination",
]


def top2_margin(policy: Any) -> float:
    """Top-1 minus top-2 probability mass (0 when fewer than 2 live arms)."""
    p = np.sort(_as_prob(policy))[::-1]
    if p.size == 0 or p.sum() <= 0.0:
        return 0.0
    if p.size == 1:
        return float(p[0])
    return float(p[0] - p[1])


def total_variation(p: Any, q: Any) -> float:
    """TV distance between two policies over a common index space."""
    pp = _as_prob(p)
    qq = _as_prob(q)
    n = max(pp.size, qq.size)
    if n == 0:
        return 0.0
    pp = np.pad(pp, (0, n - pp.size))
    qq = np.pad(qq, (0, n - qq.size))
    return float(0.5 * np.sum(np.abs(pp - qq)))


def computation_step_labels(
    trace_budgets: Sequence[int], trace_policies: Sequence[Any]
) -> list[dict[str, Any]]:
    """Per-step realized-VOC labels for each `b_i -> b_{i+1}` computation."""
    path = argmax_path(trace_policies)
    out: list[dict[str, Any]] = []
    n = min(len(trace_budgets), len(trace_policies))
    for i in range(1, n):
        out.append(
            {
                "from_budget": int(trace_budgets[i - 1]),
                "to_budget": int(trace_budgets[i]),
                "argmax_flipped": bool(path[i] >= 0 and path[i - 1] >= 0 and path[i] != path[i - 1]),
                "decision_movement": total_variation(trace_policies[i - 1], trace_policies[i]),
                "margin_delta": top2_margin(trace_policies[i]) - top2_margin(trace_policies[i - 1]),
            }
        )
    return out


def decision_movement_curve(trace_budgets: Sequence[int], trace_policies: Sequence[Any]) -> list[float]:
    """Per-step TV movement caused by each computation (realized VOC curve)."""
    return [row["decision_movement"] for row in computation_step_labels(trace_budgets, trace_policies)]


def voc_proxy(trace_budgets: Sequence[int], trace_policies: Sequence[Any]) -> float:
    """Primary per-position VOC proxy fed to `voc_tightness` — the total
    realized decision movement the budget ladder caused (sum of per-step TV).

    Robust by design: 0 iff the committed policy never moved, and monotone in
    how much computation churned the decision. Real-trace finding (Stage 2):
    the earlier shallow-vs-deep *margin-swing* form scored 0 on a position
    with 2 argmax flips whose end margin happened to equal the start margin —
    a high-VOC position mislabeled as zero. Total movement fixes that; the
    margin swing is kept as a separate diagnostic (`shallow_deep_margin_swing`).
    """
    return float(sum(decision_movement_curve(trace_budgets, trace_policies)))


def shallow_deep_margin_swing(
    trace_budgets: Sequence[int],
    trace_policies: Sequence[Any],
    *,
    shallow_index: int = 0,
    deep_index: int = -1,
) -> float:
    """Diagnostic: |margin(deep) - margin(shallow)| when the deep search
    overturns the shallow argmax, else 0. NOT the primary proxy — it
    under-labels churning positions whose end margin matches the start."""
    n = min(len(trace_budgets), len(trace_policies))
    if n < 2:
        return 0.0
    path = argmax_path(trace_policies)
    s = shallow_index % n
    d = deep_index % n
    flipped = path[s] >= 0 and path[d] >= 0 and path[s] != path[d]
    if not flipped:
        return 0.0
    return abs(top2_margin(trace_policies[d]) - top2_margin(trace_policies[s]))


def label_trace_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Label one frozen trace bundle (phase15 trace-bundle schema)."""
    budgets = bundle.get("trace_budgets", []) or []
    policies = bundle.get("trace_policies", []) or []
    steps = computation_step_labels(budgets, policies)
    return {
        "forked_voc_schema_version": FORKED_VOC_SCHEMA_VERSION,
        "n_steps": len(steps),
        "voc_proxy": voc_proxy(budgets, policies),
        "shallow_deep_margin_swing": shallow_deep_margin_swing(budgets, policies),
        "n_argmax_flips": int(sum(1 for s in steps if s["argmax_flipped"])),
        "total_decision_movement": float(sum(s["decision_movement"] for s in steps)),
        "final_overturns_shallow": bool(
            len(policies) >= 2
            and argmax_path(policies)[0] >= 0
            and argmax_path(policies)[-1] >= 0
            and argmax_path(policies)[0] != argmax_path(policies)[-1]
        ),
        "steps": steps,
    }


def screen_bundles(bundles: Sequence[dict[str, Any]], *, degenerate_eps: float = 1e-9) -> dict[str, Any]:
    """Aggregate VOC labels across positions and run the degeneracy kill-check.

    Degenerate = every position's VOC proxy is ~equal (std ≈ 0), i.e. the
    label carries no positional information; that fails the kill-criterion.
    """
    labels = [label_trace_bundle(b) for b in bundles]
    proxies = np.asarray([lab["voc_proxy"] for lab in labels], dtype=np.float64)
    if proxies.size == 0:
        return {
            "forked_voc_schema_version": FORKED_VOC_SCHEMA_VERSION,
            "n_positions": 0,
            "voc_proxy_mean": None,
            "voc_proxy_std": None,
            "degenerate": True,
            "overturn_rate": None,
        }
    std = float(np.std(proxies))
    overturn_rate = float(np.mean([1.0 if lab["final_overturns_shallow"] else 0.0 for lab in labels]))
    return {
        "forked_voc_schema_version": FORKED_VOC_SCHEMA_VERSION,
        "n_positions": int(proxies.size),
        "voc_proxy_mean": float(np.mean(proxies)),
        "voc_proxy_std": std,
        "voc_proxy_min": float(np.min(proxies)),
        "voc_proxy_max": float(np.max(proxies)),
        "overturn_rate": overturn_rate,
        "degenerate": bool(std <= degenerate_eps),
    }


def measure_tightness(
    bundles: Sequence[dict[str, Any]],
    per_move_budgets: Sequence[float],
    *,
    method: str = "spearman",
) -> float | None:
    """VOC-tightness = corr(per-move realized budget, per-position VOC proxy).

    Requires a run where positions received DIFFERENT realized budgets (an
    online/adaptive controller); on a fixed shared budget ladder the budget is
    constant and this returns None. This is THE P3 discriminating measurement,
    computed offline by the analyst (never a runtime input)."""
    proxies = [label_trace_bundle(b)["voc_proxy"] for b in bundles]
    return voc_tightness(per_move_budgets, proxies, method=method)


def discrimination(
    weak_bundles: Sequence[dict[str, Any]],
    strong_bundles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """P3 preview: do VOC proxies differ between a weak and a strong
    checkpoint? (A full claim needs the tightness dissociation; this is the
    cheap directional check.)"""
    w = screen_bundles(weak_bundles)
    s = screen_bundles(strong_bundles)
    wm = w["voc_proxy_mean"]
    sm = s["voc_proxy_mean"]
    return {
        "weak_voc_proxy_mean": wm,
        "strong_voc_proxy_mean": sm,
        "delta_strong_minus_weak": (None if (wm is None or sm is None) else float(sm - wm)),
        "weak_degenerate": w["degenerate"],
        "strong_degenerate": s["degenerate"],
    }
