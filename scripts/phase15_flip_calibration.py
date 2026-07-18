#!/usr/bin/env python3
"""H1 flip-calibration analyzer (Stage 7 / C8 + E2).

Consumes phase15 trace-cache bundles (schema >= 5, carrying per-chunk
`trace_p_flips` from C6) and evaluates, at IDENTICAL chunk boundaries, two
stop-confidence predictors against the realized argmax at a held-out higher
budget:

  s_H1    = argmax_stability(counts_from_policy(pi_b, b))     # H1 (Dirichlet)
  s_Pflip = 1 - p_flip_b                                      # incumbent P_flip
  y       = 1[argmax(pi_b) == argmax(pi_holdout)]             # outcome

Both predictors get their natural inputs at the same state (fair
observation-state parity). Outputs: 10-bin reliability diagram + ECE + Brier
(descriptive) and the CONFIRMATORY statistic — paired argmax-agreement delta
(H1 - P_flip) at matched realized budget, with a bootstrap CI.

Pre-registered (audit_stage7.md): H1 DIES if the matched-budget delta CI
excludes zero in P_flip's favor; SURVIVES on a straddle or an H1-favorable
exclusion. theta* = 0.9 is confirmatory; other thresholds are descriptive.
Restart-per-chunk and root-continuation bundles are stratified, never pooled.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.phase15_argmax_stability import argmax_stability, counts_from_policy  # noqa: E402


def _argmax(policy: Sequence[float]) -> int:
    arr = np.asarray(policy, dtype=np.float64)
    return int(np.argmax(arr)) if arr.size else -1


def bundle_decision_records(
    bundle: dict[str, Any], *, alpha: float = 0.5, n_boot: int = 4000, seed: int = 0
) -> list[dict[str, Any]]:
    """Per sub-target-budget decision records for one trace bundle. The last
    (holdout) budget is the label source and is not itself a decision point."""
    budgets = [int(b) for b in bundle.get("trace_budgets", [])]
    policies = [
        np.asarray(p, dtype=np.float64) for p in bundle.get("trace_policies", [])
    ]
    p_flips = bundle.get("trace_p_flips", [None] * len(budgets))
    n = min(len(budgets), len(policies))
    if n < 2:
        return []
    holdout_argmax = _argmax(policies[n - 1])
    records: list[dict[str, Any]] = []
    for i in range(n - 1):  # exclude the holdout budget itself
        counts = counts_from_policy(policies[i], budgets[i])
        s_h1 = float(argmax_stability(counts, alpha=alpha, n_boot=n_boot, seed=seed))
        pf = p_flips[i] if i < len(p_flips) else None
        s_pf = None if pf is None else 1.0 - float(pf)
        records.append(
            {
                "budget": budgets[i],
                "s_h1": s_h1,
                "s_pflip": s_pf,
                "y": int(_argmax(policies[i]) == holdout_argmax),
            }
        )
    return records


def reliability_diagram(
    preds: Sequence[float], ys: Sequence[int], *, n_bins: int = 10
) -> dict[str, Any]:
    """Equal-width reliability bins over predicted confidence in [0,1] plus ECE
    and Brier score."""
    p = np.asarray(preds, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if p.size == 0:
        return {"n": 0, "bins": [], "ece": None, "brier": None}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (p >= lo) & (p < hi) if k < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0,
                    "mean_pred": None,
                    "mean_obs": None,
                }
            )
            continue
        mean_pred = float(p[mask].mean())
        mean_obs = float(y[mask].mean())
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": cnt,
                "mean_pred": mean_pred,
                "mean_obs": mean_obs,
            }
        )
        ece += (cnt / p.size) * abs(mean_obs - mean_pred)
    brier = float(np.mean((p - y) ** 2))
    return {"n": int(p.size), "bins": bins, "ece": float(ece), "brier": brier}


def virtual_stop_budget(
    records: Sequence[dict[str, Any]], predictor_key: str, threshold: float
) -> dict[str, Any]:
    """Replay a stop rule over the recorded chunk boundaries: stop at the first
    budget whose predictor >= threshold; if it never fires, stop at the last
    decision budget. Returns the realized budget and the argmax-agreement (y) at
    that stop."""
    usable = [r for r in records if r.get(predictor_key) is not None]
    if not usable:
        return {"stop_budget": None, "agreement": None, "fired": False}
    for r in usable:
        if float(r[predictor_key]) >= threshold:
            return {
                "stop_budget": int(r["budget"]),
                "agreement": int(r["y"]),
                "fired": True,
            }
    last = usable[-1]
    return {
        "stop_budget": int(last["budget"]),
        "agreement": int(last["y"]),
        "fired": False,
    }


def matched_budget_calibration(
    bundle_records: Sequence[Sequence[dict[str, Any]]],
    *,
    h1_threshold: float = 0.9,
    pflip_threshold_grid: Sequence[float] | None = None,
    match_tolerance: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Confirmatory statistic: paired argmax-agreement delta (H1 - P_flip) at a
    P_flip threshold whose mean realized budget matches H1's within ±tolerance."""
    from scripts.phase15_analyze_results import paired_bootstrap_ci

    if pflip_threshold_grid is None:
        pflip_threshold_grid = [round(x, 3) for x in np.linspace(0.5, 0.99, 50)]

    h1_stops = [
        virtual_stop_budget(recs, "s_h1", h1_threshold) for recs in bundle_records
    ]
    h1_valid = [
        (recs, s)
        for recs, s in zip(bundle_records, h1_stops)
        if s["stop_budget"] is not None
    ]
    if not h1_valid:
        return {"insufficient": True, "n_bundles": 0}
    h1_mean_budget = float(np.mean([s["stop_budget"] for _, s in h1_valid]))

    best = None
    for tau in pflip_threshold_grid:
        pf_stops = [virtual_stop_budget(recs, "s_pflip", tau) for recs, _ in h1_valid]
        paired = [
            (h1["agreement"], pf["agreement"])
            for (_, h1), pf in zip(h1_valid, pf_stops)
            if pf["stop_budget"] is not None
            and pf["agreement"] is not None
            and h1["agreement"] is not None
        ]
        if not paired:
            continue
        pf_mean_budget = float(
            np.mean(
                [pf["stop_budget"] for pf in pf_stops if pf["stop_budget"] is not None]
            )
        )
        rel = abs(pf_mean_budget - h1_mean_budget) / max(1.0, h1_mean_budget)
        cand = {
            "pflip_threshold": float(tau),
            "pflip_mean_budget": pf_mean_budget,
            "budget_rel_gap": rel,
            "n_paired": len(paired),
            "paired": paired,
        }
        if rel <= match_tolerance and (best is None or rel < best["budget_rel_gap"]):
            best = cand
    if best is None:
        return {
            "insufficient": True,
            "reason": "no pflip threshold matched H1 budget within tolerance",
            "h1_mean_budget": h1_mean_budget,
        }

    deltas = [float(a_h1 - a_pf) for a_h1, a_pf in best["paired"]]
    mean_delta = float(np.mean(deltas))
    lo, hi = paired_bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
    # H1 dies if the CI excludes zero in P_flip's favor (delta < 0 => H1 worse).
    h1_loses = bool(hi < 0.0)
    h1_wins = bool(lo > 0.0)
    return {
        "insufficient": False,
        "h1_threshold": float(h1_threshold),
        "h1_mean_budget": h1_mean_budget,
        "matched_pflip_threshold": best["pflip_threshold"],
        "matched_pflip_mean_budget": best["pflip_mean_budget"],
        "budget_rel_gap": best["budget_rel_gap"],
        "n_paired": best["n_paired"],
        "mean_agreement_delta_h1_minus_pflip": mean_delta,
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "h1_dies": h1_loses,
        "h1_wins": h1_wins,
        "h1_survives": (not h1_loses),
    }


def load_bundles(trace_dir: str) -> list[dict[str, Any]]:
    bundles = []
    for path in sorted(glob.glob(str(Path(trace_dir) / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "trace_policies" in data and "trace_budgets" in data:
            data["__path"] = path
            bundles.append(data)
    return bundles


def analyze(
    bundles: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.5,
    n_boot: int = 4000,
    seed: int = 0,
    h1_threshold: float = 0.9,
) -> dict[str, Any]:
    per_bundle = [
        bundle_decision_records(b, alpha=alpha, n_boot=n_boot, seed=seed)
        for b in bundles
    ]
    flat = [r for recs in per_bundle for r in recs]
    h1_rel = reliability_diagram([r["s_h1"] for r in flat], [r["y"] for r in flat])
    pf_records = [r for r in flat if r["s_pflip"] is not None]
    pf_rel = reliability_diagram(
        [r["s_pflip"] for r in pf_records], [r["y"] for r in pf_records]
    )
    matched = matched_budget_calibration(
        per_bundle, h1_threshold=h1_threshold, seed=seed
    )
    return {
        "n_bundles": len(bundles),
        "n_decision_records": len(flat),
        "reliability_h1": h1_rel,
        "reliability_pflip": pf_rel,
        "matched_budget_calibration": matched,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace-dir", required=True, help="phase15 trace-cache dir (schema>=5 bundles)"
    )
    p.add_argument("--h1-threshold", type=float, default=0.9)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", default="results/phase15_stage7/flip_calibration/summary.json"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundles = load_bundles(args.trace_dir)
    if not bundles:
        print(json.dumps({"status": "no_bundles", "trace_dir": args.trace_dir}))
        return 2
    # Stratify by continuation mode when present (never pool restart vs
    # continuation) — bundles carry trace_source; group and report separately.
    result = analyze(
        bundles,
        alpha=args.alpha,
        n_boot=args.n_boot,
        seed=args.seed,
        h1_threshold=args.h1_threshold,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mb = result["matched_budget_calibration"]
    print(
        json.dumps(
            {
                "status": "ok",
                "n_bundles": result["n_bundles"],
                "h1_dies": mb.get("h1_dies"),
                "h1_survives": mb.get("h1_survives"),
                "mean_agreement_delta": mb.get("mean_agreement_delta_h1_minus_pflip"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
