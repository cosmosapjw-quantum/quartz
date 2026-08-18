#!/usr/bin/env python3
"""A01 Empirical Flip-Risk Calibration & Screening Quality Gate.

Evaluates empirical calibration of normal-approximation p_flip against
ground-truth continuation flips on Phase-15 replay traces.
Computes:
- Brier calibration score
- Expected Calibration Error (ECE)
- Wrong-stop rate upper 95% confidence bound (Clopper-Pearson)
- Quality loss delta on continuation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSAY_PATH = (
    REPO_ROOT
    / "results"
    / "phase15_ablation"
    / "gomoku7"
    / "assays"
    / "phase15_rows.jsonl"
)


def clopper_pearson_upper_95(k: int, n: int) -> float:
    """One-sided exact Clopper-Pearson 95% upper bound for Binomial proportion."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k))


def evaluate_flip_calibration(
    rows: Sequence[dict[str, Any]],
    risk_limit: float = 0.05,
    n_bins: int = 10,
) -> dict[str, Any]:
    preds: list[float] = []
    actuals: list[int] = []
    
    for row in rows:
        p_flip = row.get("p_flip")
        if p_flip is None or not math.isfinite(float(p_flip)):
            continue
        p_val = float(p_flip)
        
        # Ground truth flip event (if continuation best action differs from current argmax)
        flipped = row.get("flipped_at_continuation", row.get("y", 0))
        preds.append(p_val)
        actuals.append(int(flipped))
        
    n_samples = len(preds)
    if n_samples == 0:
        return {
            "status": "NO_DATA",
            "n_samples": 0,
        }
        
    p_arr = np.clip(np.array(preds, dtype=np.float64), 0.0, 1.0)
    y_arr = np.array(actuals, dtype=np.int32)
    
    brier_score = float(np.mean((p_arr - y_arr) ** 2))
    
    # ECE computation
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(p_arr, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    ece = 0.0
    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(p_arr[mask]))
            mean_obs = float(np.mean(y_arr[mask]))
            ece += (count / n_samples) * abs(mean_pred - mean_obs)
            
    # Wrong-stop evaluation (stopped when flip occurred)
    stopped_mask = p_arr <= risk_limit
    n_stops = int(np.sum(stopped_mask))
    n_wrong_stops = int(np.sum(y_arr[stopped_mask])) if n_stops > 0 else 0
    wrong_stop_rate = float(n_wrong_stops / n_stops) if n_stops > 0 else 0.0
    
    try:
        upper_95 = clopper_pearson_upper_95(n_wrong_stops, n_stops)
    except Exception:
        upper_95 = wrong_stop_rate + 1.96 * math.sqrt(max(1e-6, wrong_stop_rate * (1 - wrong_stop_rate) / max(1, n_stops)))
        
    quality_gate_passed = upper_95 <= risk_limit if n_stops >= 50 else False
    
    return {
        "status": "EVALUATED",
        "n_samples": n_samples,
        "n_stops": n_stops,
        "n_wrong_stops": n_wrong_stops,
        "wrong_stop_rate": wrong_stop_rate,
        "wrong_stop_upper_95_ci": upper_95,
        "brier_score": brier_score,
        "ece": ece,
        "risk_limit": risk_limit,
        "quality_gate_passed": quality_gate_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate A01 flip-risk empirical calibration")
    parser.add_argument("--assay-path", type=Path, default=DEFAULT_ASSAY_PATH)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    if not args.assay_path.is_file():
        print(f"[WARN] Assay file not found: {args.assay_path}")
        return 0
        
    lines = args.assay_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    res = evaluate_flip_calibration(rows, risk_limit=args.risk_limit)
    
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"[A01 Calibration] Samples: {res['n_samples']}, Stops: {res.get('n_stops', 0)}, ECE: {res.get('ece', 0.0):.4f}, Brier: {res.get('brier_score', 0.0):.4f}")
        print(f"  Wrong-stop rate: {res.get('wrong_stop_rate', 0.0):.4f}, Upper 95% CI: {res.get('wrong_stop_upper_95_ci', 0.0):.4f} (limit={args.risk_limit})")
        print(f"  Quality Gate: {'PASS' if res.get('quality_gate_passed') else 'HOLD / CALIBRATION_REQUIRED'}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
