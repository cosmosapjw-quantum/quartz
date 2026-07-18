#!/usr/bin/env python3
"""H1 discrimination-gate synthetic pre-validation (Stage 3 companion).

This is *not* a new model family: it drives the already-implemented,
already-tested H1 primitive ``stability_discrimination_gate`` from
``quartz.phase15_argmax_stability`` on a synthetic ground-truth bank of root
visit-count vectors, to answer the CCoT pre-experiment question **before** any
engine wiring:

> Does the argmax-stability signal *discriminate* across positions, or is it
> stuck saturated at 1.0 (in which case it is no better than the trivial
> point-argmax and H1 dies before the expensive online lane)?

The bank sweeps a controlled top-arm ``peak`` (0 = uniform visits, →1 =
concentrated on one arm) crossed with a total-visit ``budget`` grid. Ground
truth is known by construction: near-uniform low-budget positions must be
*unstable* (stability well below 1), peaked high-budget positions must be
*stable* (→1). A signal that reports ~1 everywhere fails the gate.

Verdict fields:

* ``discriminates_at_each_budget`` — at every total budget, the peak sweep's
  stability std exceeds the gate's ``trivial_std_eps`` (the direct kill test);
* ``not_saturated`` — the minimum stability over the whole bank is below a
  ceiling, i.e. the signal is not pinned at 1.0;
* ``monotone_in_budget`` — diagnostic: at a fixed mid peak, stability does not
  decrease as total visits grow (more search → not less stable);
* ``gate_pass`` = ``discriminates_at_each_budget and not_saturated``.

``gate_pass`` is the green-light for H1 online halt wiring (RESEARCH_PLAN_PARTB
Stage 7); ``not gate_pass`` kills H1 before that ~engine work.

Prohibited (claim firewall): reading ``gate_pass`` as evidence that H1 improves
play strength, stops at the right time online, or transfers to the live engine.
It only certifies that the offline signal is non-degenerate on synthetic ground
truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from quartz.phase15_argmax_stability import (
    ARGMAX_STABILITY_SCHEMA_VERSION,
    argmax_stability,
    counts_from_policy,
    stability_discrimination_gate,
)

H1_SYNTHETIC_GATE_SCHEMA_VERSION = 1

DEFAULT_PEAKS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7)
DEFAULT_BUDGETS = (8, 16, 32, 64)
DEFAULT_N_ARMS = 6
_SATURATION_CEILING = 0.99


def synthetic_counts(peak: float, total: int, n_arms: int) -> List[int]:
    """Integer visit counts for a controlled top-arm ``peak`` and total budget.

    ``peak`` is extra mass placed on arm 0 above the uniform share; the rest is
    spread uniformly. ``peak = 0`` is a uniform allocation; larger ``peak`` is a
    sharper, more decisive allocation."""
    peak = float(min(1.0, max(0.0, peak)))
    base = (1.0 - peak) / n_arms
    shares = [base] * n_arms
    shares[0] += peak
    return [int(c) for c in counts_from_policy(shares, total)]


def build_bank(
    peaks: Sequence[float] = DEFAULT_PEAKS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    n_arms: int = DEFAULT_N_ARMS,
) -> Dict[int, List[List[int]]]:
    """Bank of visit-count vectors grouped by total budget (peak sweep each)."""
    return {
        int(total): [synthetic_counts(peak, int(total), n_arms) for peak in peaks]
        for total in budgets
    }


def run_gate(
    peaks: Sequence[float] = DEFAULT_PEAKS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    n_arms: int = DEFAULT_N_ARMS,
    *,
    seed: int = 0,
    n_boot: int = 4000,
    monotone_peak: float = 0.2,
) -> Dict[str, Any]:
    """Run the H1 discrimination gate on the synthetic bank and return a verdict."""
    bank = build_bank(peaks, budgets, n_arms)
    per_budget: List[Dict[str, Any]] = []
    discriminates_flags: List[bool] = []
    global_min = 1.0
    global_max = 0.0
    for total in sorted(bank):
        gate = stability_discrimination_gate(bank[total], seed=seed, n_boot=n_boot)
        per_budget.append(
            {
                "total_budget": total,
                "n_positions": gate["n_positions"],
                "stability_mean": gate["stability_mean"],
                "stability_std": gate["stability_std"],
                "stability_min": gate["stability_min"],
                "stability_max": gate["stability_max"],
                "discriminates": gate["discriminates"],
            }
        )
        discriminates_flags.append(bool(gate["discriminates"]))
        if gate["stability_min"] is not None:
            global_min = min(global_min, gate["stability_min"])
        if gate["stability_max"] is not None:
            global_max = max(global_max, gate["stability_max"])

    # Monotone-in-budget diagnostic at a fixed mid peak.
    mono_curve = [
        argmax_stability(
            synthetic_counts(monotone_peak, int(total), n_arms),
            seed=seed,
            n_boot=n_boot,
        )
        for total in sorted(bank)
    ]
    monotone = all(b >= a - 1e-9 for a, b in zip(mono_curve, mono_curve[1:]))

    discriminates_at_each_budget = bool(discriminates_flags) and all(
        discriminates_flags
    )
    not_saturated = bool(global_min < _SATURATION_CEILING)
    return {
        "h1_synthetic_gate_schema_version": H1_SYNTHETIC_GATE_SCHEMA_VERSION,
        "argmax_stability_schema_version": ARGMAX_STABILITY_SCHEMA_VERSION,
        "peaks": list(peaks),
        "budgets": list(sorted(bank)),
        "n_arms": int(n_arms),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "per_budget": per_budget,
        "stability_global_min": global_min,
        "stability_global_max": global_max,
        "monotone_peak": monotone_peak,
        "monotone_stability_curve": mono_curve,
        "monotone_in_budget": monotone,
        "discriminates_at_each_budget": discriminates_at_each_budget,
        "not_saturated": not_saturated,
        "gate_pass": bool(discriminates_at_each_budget and not_saturated),
    }
