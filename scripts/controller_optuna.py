#!/usr/bin/env python3
"""
Optuna-based low-cost controller sweep for QUARTZ.

This complements ``controller_sweep.py``. Instead of random sampling a fixed
candidate pool, it uses Optuna to search the controller family / refresh /
search-constant space against a frozen-checkpoint surrogate objective.

The default objective is still cheap:
1. Evaluate one trial on a fixed random position suite.
2. Score the trial by agreement/value-gap/latency against a deeper-search
   reference on the same frozen checkpoints.
3. Optionally run a stage2 same-checkpoint arena only for the top trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import optuna

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import controller_sweep as sweep


def load_positions(path_str: str | None, game_name: str, base_cfg: dict, count: int, seed: int,
                   min_moves: int | None, max_moves: int | None) -> list[dict]:
    if not path_str:
        return sweep.generate_random_positions(
            game_name,
            base_cfg,
            count=count,
            seed=seed,
            min_moves=min_moves,
            max_moves=max_moves,
        )
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        positions = payload.get("positions")
    else:
        positions = payload
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"positions file is empty or invalid: {path_str}")
    return list(positions)


def build_param_bounds(base_cfg: dict) -> dict[str, dict[str, float | int]]:
    base_hbar = float(base_cfg.get("hbar_penalty_cap", 0.3) or 0.3)
    base_sigma = float(base_cfg.get("sigma_0", 0.3) or 0.3)
    base_min_visits = int(base_cfg.get("min_visits", 15) or 15)
    base_check_interval = int(base_cfg.get("check_interval", 20) or 20)
    base_cpuct = float(base_cfg.get("c_puct", 2.0) or 2.0)

    def round_step(value: float, step: float) -> float:
        return round(round(value / step) * step, 4)

    min_visit_step = max(1, base_min_visits // 10)
    check_interval_step = max(5, base_check_interval // 10)
    return {
        "hbar_penalty_cap": {
            "low": round_step(max(0.05, base_hbar * 0.5), 0.05),
            "high": round_step(min(1.0, max(base_hbar * 2.0, base_hbar + 0.2)), 0.05),
            "step": 0.05,
        },
        "sigma_0": {
            "low": round_step(max(0.05, base_sigma * 0.5), 0.05),
            "high": round_step(min(1.0, max(base_sigma * 2.0, base_sigma + 0.15)), 0.05),
            "step": 0.05,
        },
        "min_visits": {
            "low": max(4, int(math.floor(base_min_visits * 0.5))),
            "high": max(8, int(math.ceil(base_min_visits * 2.0))),
            "step": int(min_visit_step),
        },
        "check_interval": {
            "low": max(5, int(math.floor(base_check_interval * 0.5))),
            "high": max(10, int(math.ceil(base_check_interval * 2.0))),
            "step": int(check_interval_step),
        },
        "c_puct": {
            "low": round_step(max(0.5, base_cpuct - 1.0), 0.25),
            "high": round_step(min(6.0, max(base_cpuct + 1.5, 1.5)), 0.25),
            "step": 0.25,
        },
        "prior_refresh_rate": {
            "low": 0.10,
            "high": 0.90,
            "step": 0.05,
        },
        "prior_refresh_temp": {
            "low": 0.00,
            "high": 1.00,
            "step": 0.25,
        },
    }


def sample_trial_params(trial: optuna.Trial, base_cfg: dict, allowed_families: list[str]) -> dict[str, Any]:
    bounds = build_param_bounds(base_cfg)
    families = list(allowed_families) or ["legacy", "theory"]
    controller_family = trial.suggest_categorical("controller_family", families)
    refresh_enabled = trial.suggest_categorical("refresh_enabled", [False, True])

    params: dict[str, Any] = {
        "controller_family": controller_family,
        "root_only_shaping": trial.suggest_categorical("root_only_shaping", [False, True]),
        "refresh_enabled": refresh_enabled,
        "hbar_penalty_cap": trial.suggest_float(
            "hbar_penalty_cap",
            float(bounds["hbar_penalty_cap"]["low"]),
            float(bounds["hbar_penalty_cap"]["high"]),
            step=float(bounds["hbar_penalty_cap"]["step"]),
        ),
        "sigma_0": trial.suggest_float(
            "sigma_0",
            float(bounds["sigma_0"]["low"]),
            float(bounds["sigma_0"]["high"]),
            step=float(bounds["sigma_0"]["step"]),
        ),
        "min_visits": trial.suggest_int(
            "min_visits",
            int(bounds["min_visits"]["low"]),
            int(bounds["min_visits"]["high"]),
            step=int(bounds["min_visits"]["step"]),
        ),
        "check_interval": trial.suggest_int(
            "check_interval",
            int(bounds["check_interval"]["low"]),
            int(bounds["check_interval"]["high"]),
            step=int(bounds["check_interval"]["step"]),
        ),
        "c_puct": trial.suggest_float(
            "c_puct",
            float(bounds["c_puct"]["low"]),
            float(bounds["c_puct"]["high"]),
            step=float(bounds["c_puct"]["step"]),
        ),
    }
    if refresh_enabled:
        params["prior_refresh_rate"] = trial.suggest_float(
            "prior_refresh_rate",
            float(bounds["prior_refresh_rate"]["low"]),
            float(bounds["prior_refresh_rate"]["high"]),
            step=float(bounds["prior_refresh_rate"]["step"]),
        )
        params["prior_refresh_temp"] = trial.suggest_float(
            "prior_refresh_temp",
            float(bounds["prior_refresh_temp"]["low"]),
            float(bounds["prior_refresh_temp"]["high"]),
            step=float(bounds["prior_refresh_temp"]["step"]),
        )
    else:
        params["prior_refresh_rate"] = 0.0
        params["prior_refresh_temp"] = float(base_cfg.get("prior_refresh_temp", 1.0) or 1.0)
    return params


def params_to_candidate(params: dict[str, Any], base_cfg: dict, trial_number: int | None = None) -> dict:
    family = str(params.get("controller_family", "legacy"))
    overrides = sweep.canonicalize_candidate(
        {
            "penalty_mode": "GatedRefreshLegacy" if family == "legacy" else "GatedRefresh",
            "root_only_shaping": bool(params.get("root_only_shaping", family == "theory")),
            "prior_refresh_rate": float(params.get("prior_refresh_rate", 0.0) or 0.0),
            "prior_refresh_temp": float(params.get("prior_refresh_temp", base_cfg.get("prior_refresh_temp", 1.0) or 1.0)),
            "hbar_penalty_cap": float(params.get("hbar_penalty_cap", base_cfg.get("hbar_penalty_cap", 0.3) or 0.3)),
            "sigma_0": float(params.get("sigma_0", base_cfg.get("sigma_0", 0.3) or 0.3)),
            "min_visits": int(params.get("min_visits", base_cfg.get("min_visits", 15) or 15)),
            "check_interval": int(params.get("check_interval", base_cfg.get("check_interval", 20) or 20)),
            "c_puct": float(params.get("c_puct", base_cfg.get("c_puct", 2.0) or 2.0)),
        },
        base_cfg,
    )
    key = sweep.candidate_key(overrides)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    number = 0 if trial_number is None else int(trial_number)
    return {
        "id": f"T{number:04d}_{digest}",
        "source": "optuna",
        "label": sweep.candidate_label(overrides),
        "overrides": overrides,
    }


def anchor_candidate_to_params(candidate: dict, base_cfg: dict) -> dict[str, Any]:
    overrides = sweep.canonicalize_candidate(candidate["overrides"], base_cfg)
    return {
        "controller_family": "legacy" if overrides["penalty_mode"] == "GatedRefreshLegacy" else "theory",
        "root_only_shaping": bool(overrides["root_only_shaping"]),
        "refresh_enabled": bool(overrides["prior_refresh_rate"] > 0.0),
        "prior_refresh_rate": float(overrides["prior_refresh_rate"]),
        "prior_refresh_temp": float(overrides["prior_refresh_temp"]),
        "hbar_penalty_cap": float(overrides["hbar_penalty_cap"]),
        "sigma_0": float(overrides["sigma_0"]),
        "min_visits": int(overrides["min_visits"]),
        "check_interval": int(overrides["check_interval"]),
        "c_puct": float(overrides["c_puct"]),
    }


def serialize_trial(trial: optuna.Trial | optuna.trial.FrozenTrial) -> dict[str, Any]:
    state = getattr(trial, "state", None)
    if state is None:
        state_name = "RUNNING"
    else:
        state_name = state.name
    value = getattr(trial, "value", None)
    return {
        "number": int(trial.number),
        "state": state_name,
        "value": None if value is None else float(value),
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def summarize_trials_by_segment(trial_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in trial_rows if row.get("state") == "COMPLETE" and row.get("value") is not None]

    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["value"]) for row in rows]
        if not values:
            return {"count": 0, "best_value": None, "mean_value": None}
        return {
            "count": len(values),
            "best_value": max(values),
            "mean_value": sum(values) / len(values),
        }

    refresh_on = [row for row in completed if bool(row["params"].get("refresh_enabled"))]
    refresh_off = [row for row in completed if not bool(row["params"].get("refresh_enabled"))]
    family_legacy = [row for row in completed if row["params"].get("controller_family") == "legacy"]
    family_theory = [row for row in completed if row["params"].get("controller_family") == "theory"]

    ranked = sorted(completed, key=lambda row: (-float(row["value"]), row["number"]))
    top_cut = max(1, len(ranked) // 4) if ranked else 0
    top_quartile = ranked[:top_cut]
    return {
        "refresh": {
            "off": bucket(refresh_off),
            "on": bucket(refresh_on),
            "top_quartile": {
                "count": len(top_quartile),
                "refresh_on": sum(1 for row in top_quartile if bool(row["params"].get("refresh_enabled"))),
                "refresh_off": sum(1 for row in top_quartile if not bool(row["params"].get("refresh_enabled"))),
            },
        },
        "family": {
            "legacy": bucket(family_legacy),
            "theory": bucket(family_theory),
        },
    }


def select_top_trial_candidates(trial_rows: list[dict[str, Any]], base_cfg: dict, topk: int) -> list[dict]:
    ranked = sorted(
        [row for row in trial_rows if row.get("state") == "COMPLETE" and row.get("value") is not None],
        key=lambda row: (-float(row["value"]), int(row["number"])),
    )
    selected: list[dict] = []
    seen = set()
    for row in ranked:
        candidate = params_to_candidate(row["params"], base_cfg, row["number"])
        key = sweep.candidate_key(candidate["overrides"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= int(topk):
            break
    return selected


def save_trial_payload(path: Path, trial: optuna.Trial | optuna.trial.FrozenTrial, candidate: dict,
                       checkpoint_rows: list[dict], summary: dict | None, error: str | None = None) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial": serialize_trial(trial),
        "candidate": candidate,
        "checkpoint_rows": checkpoint_rows,
        "summary": summary,
        "error": error,
    }
    sweep.json_dump(path, payload)


def build_report(base_dir: Path, manifest: dict, study: optuna.Study,
                 arena_payload: dict | None = None) -> dict[str, Any]:
    trial_rows = [serialize_trial(trial) for trial in study.trials]
    completed = [row for row in trial_rows if row["state"] == "COMPLETE" and row["value"] is not None]
    ranked = sorted(completed, key=lambda row: (-float(row["value"]), row["number"]))
    best_trial = ranked[0] if ranked else None
    try:
        param_importances = optuna.importance.get_param_importances(study)
    except Exception:
        param_importances = None
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest,
        "best_trial": best_trial,
        "trial_count": len(trial_rows),
        "completed_trial_count": len(completed),
        "summaries": summarize_trials_by_segment(trial_rows),
        "param_importances": param_importances,
        "top_trials": ranked[: min(10, len(ranked))],
        "arena": arena_payload,
        "recommended": None,
    }
    if arena_payload and arena_payload.get("overall"):
        report["recommended"] = arena_payload["overall"][0]
    elif best_trial is not None:
        report["recommended"] = best_trial
    sweep.json_dump(base_dir / "optuna_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-based QUARTZ controller sweep")
    parser.add_argument("--game", default="gomoku7", choices=[
        "gomoku7",
        "gomoku15",
        "gomoku15_free",
        "gomoku15_std",
        "gomoku15_omok",
        "gomoku15_renju",
        "gomoku15_caro",
        "tictactoe",
    ])
    parser.add_argument("--output", default="results/controller_optuna", help="Output root directory")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint paths")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to scan recursively for best.pt/latest.pt")
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--bootstrap-if-empty", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2)
    parser.add_argument("--bootstrap-games", type=int, default=8)
    parser.add_argument("--bootstrap-eval-games", type=int, default=4)
    parser.add_argument("--bootstrap-seeds", default="41,42")
    parser.add_argument("--force-bootstrap", action="store_true")
    parser.add_argument("--backend", default="torch", choices=["auto", "torch", "jax"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions-file", default=None, help="Existing stage1_positions.json (or raw list) to reuse")
    parser.add_argument("--stage1-positions", type=int, default=12)
    parser.add_argument("--position-min-moves", type=int, default=None)
    parser.add_argument("--position-max-moves", type=int, default=None)
    parser.add_argument("--probe-iters", type=int, default=96)
    parser.add_argument("--reference-multiplier", type=float, default=4.0)
    parser.add_argument("--search-stall-timeout-s", type=float, default=45.0)
    parser.add_argument("--families", default="legacy,theory", help="Comma-separated controller families to search")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=None, help="Optuna wall-clock timeout in seconds")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=8)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="Optuna storage URL, e.g. sqlite:///results/study.db")
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument("--enqueue-anchors", action="store_true")
    parser.add_argument("--arena-topk", type=int, default=0, help="Run stage2 arena on top-k optuna trials")
    parser.add_argument("--arena-include-anchors", action="store_true")
    parser.add_argument("--arena-iters", type=int, default=96)
    parser.add_argument("--stage2-games", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(float(args.search_stall_timeout_s))

    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, device = sweep.build_base_cfg(args.game, args.device)
    checkpoints = sweep.resolve_checkpoint_paths(args, base_dir)
    positions = load_positions(
        args.positions_file,
        args.game,
        base_cfg,
        count=args.stage1_positions,
        seed=args.seed,
        min_moves=args.position_min_moves,
        max_moves=args.position_max_moves,
    )
    sweep.json_dump(base_dir / "stage1_positions.json", {"positions": positions})

    families = sweep.parse_csv_items(args.families)
    if not families:
        families = ["legacy", "theory"]
    for family in families:
        if family not in {"legacy", "theory"}:
            raise ValueError(f"unsupported family: {family}")

    manifest = {
        "format_version": 1,
        "game": args.game,
        "seed": int(args.seed),
        "sampler_seed": int(args.sampler_seed),
        "trials": int(args.trials),
        "timeout": None if args.timeout is None else float(args.timeout),
        "families": families,
        "probe_iters": int(args.probe_iters),
        "reference_multiplier": float(args.reference_multiplier),
        "search_stall_timeout_s": float(args.search_stall_timeout_s),
        "stage1_positions": len(positions),
        "positions_file": args.positions_file,
        "arena_topk": int(args.arena_topk),
        "arena_iters": int(args.arena_iters),
        "stage2_games": int(args.stage2_games),
        "checkpoints": checkpoints,
        "param_bounds": build_param_bounds(base_cfg),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sweep.json_dump(base_dir / "optuna_manifest.json", manifest)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(0, int(args.startup_trials)))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.load_if_exists),
        sampler=sampler,
        pruner=pruner,
    )

    if args.enqueue_anchors:
        for anchor in sweep.build_anchor_candidates(base_cfg):
            study.enqueue_trial(anchor_candidate_to_params(anchor, base_cfg))

    trial_dir = base_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial, base_cfg, families)
        candidate = params_to_candidate(params, base_cfg, trial.number)
        checkpoint_rows = []
        try:
            for step, checkpoint_path in enumerate(checkpoints, start=1):
                row = sweep.probe_candidate_on_positions(
                    checkpoint_path,
                    base_cfg,
                    device,
                    positions,
                    candidate,
                    args.rust_binary,
                    args.probe_iters,
                    args.reference_multiplier,
                )
                checkpoint_rows.append(row)
                avg_score = sum(float(item["stage1_score"]) for item in checkpoint_rows) / len(checkpoint_rows)
                trial.report(avg_score, step)
                if trial.should_prune():
                    trial.set_user_attr("candidate_id", candidate["id"])
                    trial.set_user_attr("candidate_label", candidate["label"])
                    trial.set_user_attr("candidate_overrides", candidate["overrides"])
                    summary = sweep.summarize_stage1_results([candidate], checkpoint_rows)[0]
                    save_trial_payload(
                        trial_dir / f"trial_{trial.number:04d}.json",
                        trial,
                        candidate,
                        checkpoint_rows,
                        summary,
                        error="pruned",
                    )
                    raise optuna.TrialPruned()

            summary = sweep.summarize_stage1_results([candidate], checkpoint_rows)[0]
            value = float(summary["stage1_score"])
            trial.set_user_attr("candidate_id", candidate["id"])
            trial.set_user_attr("candidate_label", candidate["label"])
            trial.set_user_attr("candidate_overrides", candidate["overrides"])
            trial.set_user_attr("mean_agreement_rate", float(summary["agreement_rate"]))
            trial.set_user_attr("mean_policy_mass", float(summary["reference_policy_mass"]))
            trial.set_user_attr("mean_value_gap", float(summary["mean_value_gap"]))
            trial.set_user_attr("mean_latency_ms", summary["mean_latency_ms"])
            save_trial_payload(
                trial_dir / f"trial_{trial.number:04d}.json",
                trial,
                candidate,
                checkpoint_rows,
                summary,
            )
            return value
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("candidate_id", candidate["id"])
            trial.set_user_attr("candidate_label", candidate["label"])
            trial.set_user_attr("candidate_overrides", candidate["overrides"])
            save_trial_payload(
                trial_dir / f"trial_{trial.number:04d}.json",
                trial,
                candidate,
                checkpoint_rows,
                None,
                error=repr(exc),
            )
            raise

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True)

    arena_payload = None
    if args.arena_topk > 0:
        arena_candidates = select_top_trial_candidates(
            [serialize_trial(trial) for trial in study.trials],
            base_cfg,
            args.arena_topk,
        )
        if args.arena_include_anchors:
            extras = []
            seen = {sweep.candidate_key(row["overrides"]) for row in arena_candidates}
            for anchor in sweep.build_anchor_candidates(base_cfg):
                key = sweep.candidate_key(anchor["overrides"])
                if key in seen:
                    continue
                seen.add(key)
                extras.append(anchor)
            arena_candidates = extras + arena_candidates
        arena_payload = sweep.run_stage2_round_robin(arena_candidates, checkpoints, base_cfg, device, args, base_dir)

    report = build_report(base_dir, manifest, study, arena_payload=arena_payload)
    recommended = report.get("recommended")
    if recommended:
        if recommended.get("candidate_id"):
            print(f"Recommended: {recommended['candidate_id']}")
        elif recommended.get("number") is not None:
            print(f"Recommended trial: {recommended['number']}")
        if recommended.get("candidate_label"):
            print(f"  {recommended['candidate_label']}")
        elif recommended.get("user_attrs", {}).get("candidate_label"):
            print(f"  {recommended['user_attrs']['candidate_label']}")
        if recommended.get("score_rate") is not None:
            print(f"  score={recommended['score_rate']}")
        elif recommended.get("value") is not None:
            print(f"  score={recommended['value']}")
    print(f"Report saved: {base_dir / 'optuna_report.json'}")


if __name__ == "__main__":
    main()
