#!/usr/bin/env python3
"""O6 burst-precision analyzer (Stage 7 / C9 + E2).

Joins H3 (B15) burst events from the online rows with an EXTERNAL difficulty
label from `forked_voc` computed on the same A4-substrate trace bundle, and
tests the O6 signature:

    lift = P(hard | burst) / P(hard)   (must exceed 1)

Non-circularity: the burst signal (B15) only sees ≤ target-budget chunks, while
the `hard := forked_voc.final_overturns_shallow` label uses the FULL ladder
including budgets above the decision point — a different source than the entropy
trigger. Join key: (checkpoint_id, position_id).

Pre-registered (audit_stage7.md): the lane DIES if the lift CI includes 1;
degeneracy demotion (diagnostic only) if burst rate > 0.9 or < 0.02 or fewer
than 30 pooled burst events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiments.forked_voc import label_trace_bundle  # noqa: E402

_MIN_BURST_EVENTS = 30
_BURST_RATE_HI = 0.9
_BURST_RATE_LO = 0.02


def _mean(xs: Sequence[float]) -> float | None:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def compute_o6_lift(records: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Compute the O6 lift and a position-level bootstrap CI.

    records: [{checkpoint_id, position_id, burst: 0/1, hard: 0/1}].
    """
    n = len(records)
    if n == 0:
        return {"insufficient": True, "reason": "no records", "o6_lane_alive": False}
    hard = np.asarray([int(r["hard"]) for r in records], dtype=np.int64)
    burst = np.asarray([int(r["burst"]) for r in records], dtype=np.int64)
    n_burst = int(burst.sum())
    burst_rate = n_burst / n
    p_hard = float(hard.mean())
    p_hard_given_burst = _mean(hard[burst == 1].tolist()) if n_burst > 0 else None
    lift = (
        None if (p_hard <= 0.0 or p_hard_given_burst is None) else float(p_hard_given_burst / p_hard)
    )

    degenerate = bool(burst_rate > _BURST_RATE_HI or burst_rate < _BURST_RATE_LO or n_burst < _MIN_BURST_EVENTS)

    # position-level bootstrap CI on the lift
    rng = np.random.default_rng(seed)
    lifts: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bh = hard[idx]
        bb = burst[idx]
        nb = int(bb.sum())
        ph = float(bh.mean())
        if nb == 0 or ph <= 0.0:
            continue
        phgb = float(bh[bb == 1].mean())
        lifts.append(phgb / ph)
    ci_low = float(np.percentile(lifts, 2.5)) if lifts else None
    ci_high = float(np.percentile(lifts, 97.5)) if lifts else None

    ci_includes_one = ci_low is None or ci_high is None or (ci_low <= 1.0 <= ci_high)
    o6_lane_alive = bool(not degenerate and lift is not None and lift > 1.0 and ci_low is not None and ci_low > 1.0)
    return {
        "insufficient": bool(degenerate),
        "n_records": n,
        "n_burst": n_burst,
        "burst_rate": burst_rate,
        "p_hard": p_hard,
        "p_hard_given_burst": p_hard_given_burst,
        "lift": lift,
        "lift_ci_low": ci_low,
        "lift_ci_high": ci_high,
        "lift_ci_includes_one": bool(ci_includes_one),
        "degenerate": degenerate,
        "degenerate_reason": (
            "burst_rate_out_of_range" if (burst_rate > _BURST_RATE_HI or burst_rate < _BURST_RATE_LO)
            else "too_few_burst_events" if n_burst < _MIN_BURST_EVENTS else None
        ),
        # kill: lane dies if the CI includes 1 (burst fires at the base rate)
        "o6_kill_lift_ci_includes_one": bool(ci_includes_one),
        "o6_lane_alive": o6_lane_alive,
    }


def build_records(online_rows: Sequence[dict[str, Any]], bundles_by_key: dict[tuple, dict[str, Any]]) -> list[dict[str, Any]]:
    """Join B15 burst events with forked_voc hard labels on (checkpoint, position).

    A burst EVENT is any B15 online row with budget_burst_triggered == 1 for a
    (checkpoint, position). The hard label is computed once per (checkpoint,
    position) from its A4-substrate trace bundle. Rows whose bundle is missing
    are excluded and counted (never silently matched)."""
    burst_by_key: dict[tuple, int] = {}
    for row in online_rows:
        key = (str(row.get("checkpoint_id")), str(row.get("position_id")))
        triggered = int(row.get("budget_burst_triggered", 0) or 0)
        burst_by_key[key] = max(burst_by_key.get(key, 0), triggered)

    records: list[dict[str, Any]] = []
    missing = 0
    hard_cache: dict[tuple, int] = {}
    for key, burst in sorted(burst_by_key.items()):
        bundle = bundles_by_key.get(key)
        if bundle is None:
            missing += 1
            continue
        if key not in hard_cache:
            lab = label_trace_bundle(bundle)
            hard_cache[key] = int(bool(lab.get("final_overturns_shallow")))
        records.append({"checkpoint_id": key[0], "position_id": key[1], "burst": int(burst), "hard": hard_cache[key]})
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--online-rows", required=True, help="phase15_online_rows.jsonl (B15 rows)")
    p.add_argument("--trace-dir", required=True, help="A4 trace-cache dir (for forked_voc labels)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/phase15_stage7/o6_precision/summary.json")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    for line in Path(args.online_rows).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    b15_rows = [r for r in rows if str(r.get("system", r.get("system_id", ""))) == "B15"]

    bundles_by_key: dict[tuple, dict[str, Any]] = {}
    import glob

    for path in sorted(glob.glob(str(Path(args.trace_dir) / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ck = data.get("checkpoint_id")
        pos = data.get("position_id")
        if ck is not None and pos is not None and "trace_policies" in data:
            bundles_by_key[(str(ck), str(pos))] = data

    records = build_records(b15_rows, bundles_by_key)
    result = compute_o6_lift(records, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "n_records": result.get("n_records"),
        "lift": result.get("lift"),
        "o6_lane_alive": result.get("o6_lane_alive"),
        "degenerate": result.get("degenerate"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
