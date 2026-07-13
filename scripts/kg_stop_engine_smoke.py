#!/usr/bin/env python3
"""KG-stop engine smoke (Stage 7 / C4 + E3).

Runs the same positions through the real Rust MCTS engine twice — once with
``QUARTZ_SEARCH_POLICY=kg_stop`` attached, once with no policy (fixed halt at the
budget) — and measures, per (budget, kg_threshold), whether KG-stop halts early
and whether its committed move still agrees with the fixed-budget decision.

Pre-registered criteria (see ``audit_stage7.md``):
- Success (SMOKE ceiling): some grid cell with mean budget saved >= 20% at
  top-1 agreement >= 0.95.
- Kill: zero halts anywhere (KG scale does not transfer to real backups).
- Demote: halts fire but top-1 agreement < 0.80 at every saving level.

The pure ``summarize_kg_smoke`` is unit-tested; the live grid run is exercised
at E3 (needs a trained checkpoint + GPU + the Rust binary).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def summarize_kg_smoke(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired KG-stop-vs-fixed rows into per-cell and overall verdicts.

    Each row: ``{position_id, budget, threshold, kg_iterations, kg_best_move,
    kg_halted, fixed_iterations, fixed_best_move}``.
    """
    cells: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((int(r["budget"]), float(r["threshold"])), []).append(r)

    per_cell: list[dict[str, Any]] = []
    for (budget, threshold), group in sorted(cells.items()):
        n = len(group)
        halted = [g for g in group if g["kg_halted"]]
        halt_rate = len(halted) / n if n else 0.0
        # budget saved: non-halted rows save 0 (ran to full budget).
        saved = [max(0.0, 1.0 - g["kg_iterations"] / max(1, budget)) for g in group]
        mean_saved = sum(saved) / n if n else 0.0
        agreement = sum(1 for g in group if g["kg_best_move"] == g["fixed_best_move"]) / n if n else 0.0
        per_cell.append(
            {
                "budget": budget,
                "threshold": threshold,
                "n": n,
                "halt_rate": halt_rate,
                "mean_budget_saved_pct": mean_saved,
                "top1_agreement": agreement,
            }
        )

    any_halt = any(c["halt_rate"] > 0.0 for c in per_cell)
    success_cells = [
        c for c in per_cell
        if c["mean_budget_saved_pct"] >= 0.20 and c["top1_agreement"] >= 0.95
    ]
    saving_cells = [c for c in per_cell if c["mean_budget_saved_pct"] > 0.0]
    demote = bool(any_halt and saving_cells and all(c["top1_agreement"] < 0.80 for c in saving_cells))
    best_cell = max(per_cell, key=lambda c: (c["mean_budget_saved_pct"], c["top1_agreement"]), default=None)
    return {
        "per_cell": per_cell,
        "kill_no_halts": not any_halt,
        "success": len(success_cells) > 0,
        "success_cells": success_cells,
        "demote_anti_conservative": demote,
        "best_cell": best_cell,
    }


def _minimal_system(check_interval: int):
    from quartz.phase15_ablation import Phase15System

    return Phase15System(
        id="KGSMOKE",
        label="kg-stop engine smoke (A4-like search)",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="identity",
        search_overrides={"check_interval": int(check_interval)},
    )


def run_kg_smoke(
    *,
    checkpoint: str,
    positions: list[dict[str, Any]],
    budgets: Sequence[int],
    thresholds: Sequence[float],
    device: str,
    rust_binary: str,
    base_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Live grid run. For each threshold, attach kg_stop (fresh harness so the
    Rust server inherits the env) and capture per-position iterations/best_move;
    then a fixed baseline with no policy. Pairs on (position, budget)."""
    from scripts.phase15_ablation_study import CheckpointRef, FrozenCheckpointHarness

    ckpt = CheckpointRef(id=f"KG_{Path(checkpoint).stem}", path=checkpoint)

    def _grid(env_policy: str | None, threshold: float | None) -> dict[tuple[str, int], dict[str, Any]]:
        prev = os.environ.get("QUARTZ_SEARCH_POLICY")
        prev_thr = os.environ.get("QUARTZ_KG_THRESHOLD")
        if env_policy:
            os.environ["QUARTZ_SEARCH_POLICY"] = env_policy
            if threshold is not None:
                os.environ["QUARTZ_KG_THRESHOLD"] = repr(float(threshold))
        else:
            os.environ.pop("QUARTZ_SEARCH_POLICY", None)
            os.environ.pop("QUARTZ_KG_THRESHOLD", None)
        out: dict[tuple[str, int], dict[str, Any]] = {}
        harness = FrozenCheckpointHarness(ckpt, base_cfg, device, rust_binary)
        try:
            for budget in budgets:
                system = _minimal_system(check_interval=max(4, int(budget) // 8))
                client = harness._get_client(system, int(budget))
                for pos in positions:
                    import numpy as np

                    board = np.asarray(pos.get("board", []), dtype=np.int8) if "board" in pos else None
                    t0 = time.perf_counter()
                    payload = client.search_move(
                        board, int(pos.get("player", 1)),
                        penalty_mode=client.cfg.get("penalty_mode", "None"),
                        fen=pos.get("fen"), state_meta=dict(pos.get("state_meta") or {}),
                    )
                    out[(harness._position_key(pos), int(budget))] = {
                        "iterations": int(payload.get("iterations", budget)),
                        "best_move": int(payload.get("best_move", -1)),
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    }
        finally:
            harness.close()
            if prev is None:
                os.environ.pop("QUARTZ_SEARCH_POLICY", None)
            else:
                os.environ["QUARTZ_SEARCH_POLICY"] = prev
            if prev_thr is None:
                os.environ.pop("QUARTZ_KG_THRESHOLD", None)
            else:
                os.environ["QUARTZ_KG_THRESHOLD"] = prev_thr
        return out

    fixed = _grid(None, None)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        kg = _grid("kg_stop", threshold)
        for key, kg_row in kg.items():
            fx_row = fixed.get(key)
            if fx_row is None:
                continue
            pos_key, budget = key
            rows.append(
                {
                    "position_id": pos_key,
                    "budget": budget,
                    "threshold": float(threshold),
                    "kg_iterations": kg_row["iterations"],
                    "kg_best_move": kg_row["best_move"],
                    "kg_halted": kg_row["iterations"] < budget,
                    "fixed_iterations": fx_row["iterations"],
                    "fixed_best_move": fx_row["best_move"],
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game", default="gomoku7")
    p.add_argument("--positions-file", default=None)
    p.add_argument("--n-positions", type=int, default=32)
    p.add_argument("--budgets", default="64,128,256")
    p.add_argument("--thresholds", default="1e-4,1e-3,1e-2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--position-min-moves", type=int, default=None)
    p.add_argument("--position-max-moves", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rust-binary", default="./target/release/mcts_demo")
    p.add_argument("--output-dir", default="results/phase15_stage7/kg_smoke")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scripts import controller_sweep as sweep
    from scripts.phase15_ablation_study import load_or_generate_positions

    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
    base_cfg, device = sweep.build_base_cfg(args.game, args.device)
    # load_or_generate_positions needs these Namespace attrs.
    args.suite_size = int(args.n_positions)
    args.position_min_moves = getattr(args, "position_min_moves", None)
    args.position_max_moves = getattr(args, "position_max_moves", None)
    positions = load_or_generate_positions(args, base_cfg, count=args.n_positions)[: args.n_positions]

    rows = run_kg_smoke(
        checkpoint=args.checkpoint, positions=positions, budgets=budgets,
        thresholds=thresholds, device=args.device, rust_binary=args.rust_binary,
        base_cfg=base_cfg,
    )
    summary = summarize_kg_smoke(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "n_rows": len(rows),
        "kill_no_halts": summary["kill_no_halts"],
        "success": summary["success"],
        "demote_anti_conservative": summary["demote_anti_conservative"],
        "best_cell": summary["best_cell"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
