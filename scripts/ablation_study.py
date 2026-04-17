#!/usr/bin/env python3
"""
QUARTZ ablation runner with reproducible manifests, post-train round-robin
evaluation, and Gomocup champion export.

Usage:
  venv/bin/python scripts/ablation_study.py --game gomoku15 --iterations 30 --eval-games 80
  venv/bin/python scripts/ablation_study.py --game gomoku15_renju --seeds 41,42 --quick
  venv/bin/python scripts/ablation_study.py --report results/ablation/gomoku15
  venv/bin/python scripts/ablation_study.py --report results/ablation/gomoku15 --prepare-gomocup
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Persist torch.compile kernel cache across training subprocesses
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

CLI_CONDITION_KEYS = {"search_profile", "vl_mode"}

SEARCH_VL_TRAIN_CONDITIONS = {
    "T1_noS_noVL": {"search_profile": "baseline", "vl_mode": "disabled"},
    "T2_S_noVL": {"search_profile": "quartz", "vl_mode": "disabled"},
    "T3_noS_VL": {"search_profile": "baseline", "vl_mode": "adaptive"},
    "T4_S_VL": {"search_profile": "quartz", "vl_mode": "adaptive"},
}

SEARCH_VL_EVAL_CONDITIONS = {
    "E1_noS_noVL": {"search_profile": "baseline", "vl_mode": "disabled"},
    "E2_S_noVL": {"search_profile": "quartz", "vl_mode": "disabled"},
    "E3_noS_VL": {"search_profile": "baseline", "vl_mode": "adaptive"},
    "E4_S_VL": {"search_profile": "quartz", "vl_mode": "adaptive"},
}

CONTROLLER_TRAIN_CONDITIONS = {
    "C1_impl_legacy": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
    },
    "C2_theory_doc": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
    },
}

CONTROLLER_EVAL_CONDITIONS = {
    "E1_impl_legacy": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
    },
    "E2_theory_doc": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
    },
}

CONTROLLER_FACTORIAL_TRAIN_CONDITIONS = {
    "F1_legacy_base": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
        "prior_refresh_rate": 0.0,
    },
    "F2_legacy_krefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
        "prior_refresh_rate": 0.5,
        "prior_refresh_temp": 0.0,
    },
    "F3_theory_base": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
    },
    "F4_theory_krefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.5,
        "prior_refresh_temp": 0.0,
    },
}

CONTROLLER_FACTORIAL_EVAL_CONDITIONS = {
    "E1_legacy_base": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
        "prior_refresh_rate": 0.0,
    },
    "E2_legacy_krefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
        "prior_refresh_rate": 0.5,
        "prior_refresh_temp": 0.0,
    },
    "E3_theory_base": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
    },
    "E4_theory_krefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.5,
        "prior_refresh_temp": 0.0,
    },
}

STRICT_REFERENCE_CONDITION = {
    "E0_baseline_strict": {"search_profile": "baseline_strict", "vl_mode": "disabled"},
}

STUDY_PRESETS = {
    "search_vl": {
        "train_conditions": SEARCH_VL_TRAIN_CONDITIONS,
        "eval_conditions": SEARCH_VL_EVAL_CONDITIONS,
    },
    "controller": {
        "train_conditions": CONTROLLER_TRAIN_CONDITIONS,
        "eval_conditions": CONTROLLER_EVAL_CONDITIONS,
    },
    "controller_factorial": {
        "train_conditions": CONTROLLER_FACTORIAL_TRAIN_CONDITIONS,
        "eval_conditions": CONTROLLER_FACTORIAL_EVAL_CONDITIONS,
    },
}


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    head = proc.stdout.strip()
    return head or None


def parse_csv_items(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: str | None) -> list[int]:
    if not raw:
        return [42]
    seeds = []
    for item in parse_csv_items(raw):
        seeds.append(int(item))
    return seeds or [42]


def parse_selected_conditions(raw: str | None, known: dict[str, dict]) -> list[str]:
    if not raw:
        return sorted(known)
    selected = parse_csv_items(raw)
    unknown = sorted(set(selected) - set(known))
    if unknown:
        raise ValueError(f"unknown conditions: {', '.join(unknown)}")
    return selected


def resolve_study_preset(study_name: str) -> dict:
    preset = STUDY_PRESETS.get(study_name)
    if preset is None:
        raise ValueError(f"unknown study preset: {study_name}")
    return preset


def condition_runtime_overrides(condition_cfg: dict) -> dict:
    return {
        key: value
        for key, value in condition_cfg.items()
        if key not in CLI_CONDITION_KEYS
    }


def condition_run_dir(base_dir: Path, condition_name: str, seed: int, multi_seed: bool) -> Path:
    root = base_dir / "models" / condition_name
    return root / f"seed_{seed}" if multi_seed else root


def resolve_model_path(run_dir: Path) -> Path | None:
    for name in ("best.pt", "latest.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def collect_training_metrics(run_dir: Path) -> dict:
    metrics = {
        "published_elo": None,
        "loss": None,
        "games_done": 0,
        "eval_verdict": None,
        "score_rate": None,
        "champion_elo": None,
        "elo_gap": None,
    }
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return metrics
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("published_elo") is not None:
                metrics["published_elo"] = row.get("published_elo")
            if row.get("loss") is not None:
                metrics["loss"] = row.get("loss")
            if row.get("games_done") is not None:
                metrics["games_done"] += int(row.get("games_done") or 0)
            if row.get("eval_verdict") is not None:
                metrics["eval_verdict"] = row.get("eval_verdict")
            if row.get("score_rate") is not None:
                metrics["score_rate"] = row.get("score_rate")
            if row.get("champion_elo") is not None:
                metrics["champion_elo"] = row.get("champion_elo")
            if row.get("elo_gap") is not None:
                metrics["elo_gap"] = row.get("elo_gap")
    return metrics


def discover_model_runs(base_dir: Path) -> list[dict]:
    models_dir = base_dir / "models"
    if not models_dir.is_dir():
        return []

    runs = []
    for condition_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        direct_meta = condition_dir / "condition.json"
        seed_dirs = sorted(path for path in condition_dir.iterdir() if path.is_dir())
        candidate_dirs = []
        if direct_meta.exists():
            candidate_dirs.append(condition_dir)
        for seed_dir in seed_dirs:
            if (seed_dir / "condition.json").exists():
                candidate_dirs.append(seed_dir)

        for run_dir in candidate_dirs:
            meta_path = run_dir / "condition.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            model_path = resolve_model_path(run_dir)
            metrics = collect_training_metrics(run_dir)
            seed = meta.get("seed")
            run_id = meta.get("run_id")
            if not run_id:
                if seed is None:
                    run_id = meta.get("condition", run_dir.name)
                else:
                    run_id = f"{meta.get('condition', condition_dir.name)}_s{seed}"
            runs.append({
                "id": run_id,
                "condition": meta.get("condition", condition_dir.name),
                "seed": seed,
                "game": meta.get("game"),
                "train_cfg": meta.get("train_cfg", {}),
                "run_dir": str(run_dir),
                "elapsed_s": meta.get("elapsed_s", 0),
                "returncode": meta.get("returncode"),
                "success": meta.get("returncode") == 0,
                "model_path": str(model_path) if model_path is not None else None,
                "metrics": metrics,
            })
    return runs


def build_study_manifest(args: argparse.Namespace) -> dict:
    preset = resolve_study_preset(args.study)
    train_conditions = preset["train_conditions"]
    eval_conditions = preset["eval_conditions"]
    selected_train_conditions = parse_selected_conditions(args.conditions, train_conditions)
    selected_eval_conditions = parse_selected_conditions(args.eval_conditions, eval_conditions)
    return {
        "format_version": 2,
        "study": args.study,
        "game": args.game,
        "iterations": args.iterations,
        "eval_games": args.eval_games,
        "quick": bool(args.quick),
        "rust_binary": args.rust_binary,
        "backend": args.backend,
        "device": args.device,
        "seeds": parse_seed_list(args.seeds),
        "conditions": selected_train_conditions,
        "eval_conditions_selected": selected_eval_conditions,
        "train_conditions": {
            name: copy.deepcopy(train_conditions[name]) for name in selected_train_conditions
        },
        "eval_conditions": {
            name: copy.deepcopy(eval_conditions[name]) for name in selected_eval_conditions
        },
        "strict_reference": bool(args.include_strict_reference),
        "git_head": git_head(),
        "python": sys.executable,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_training_command(
    args: argparse.Namespace,
    condition_cfg: dict,
    seed: int,
    output_dir: Path,
) -> list[str]:
    overrides = condition_runtime_overrides(condition_cfg)
    override_path = None
    if overrides:
        override_path = output_dir / "condition_overrides.json"
        json_dump(override_path, overrides)
    cmd = [
        sys.executable,
        "-m",
        "quartz.train",
        "--game",
        args.game,
        "--iterations",
        str(args.iterations),
        "--retune",
        "--rust-binary",
        args.rust_binary,
        "--search-profile",
        condition_cfg["search_profile"],
        "--vl-mode",
        condition_cfg["vl_mode"],
        "--output",
        str(output_dir),
        "--seed",
        str(seed),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-games",
        str(args.eval_games),
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]
    if override_path is not None:
        cmd.extend(["--config", str(override_path)])
    if args.games_per_iter is not None:
        cmd.extend(["--games", str(args.games_per_iter)])
    elif args.quick:
        cmd.extend(["--games", "50"])
    if args.no_autotune:
        cmd.append("--no-autotune")
    if args.resident_session:
        cmd.append("--resident-session")
    if args.runtime_autotune:
        cmd.append("--runtime-autotune")
    return cmd


def run_training(
    args: argparse.Namespace,
    base_dir: Path,
    condition_name: str,
    condition_cfg: dict,
    seed: int,
    multi_seed: bool,
) -> dict:
    output_dir = condition_run_dir(base_dir, condition_name, seed, multi_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_model = resolve_model_path(output_dir)
    condition_path = output_dir / "condition.json"

    if not args.force_train and condition_path.exists():
        try:
            previous = json.loads(condition_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None
        if previous and previous.get("returncode") == 0 and existing_model is not None:
            return {
                "condition": condition_name,
                "seed": seed,
                "run_dir": str(output_dir),
                "elapsed_s": previous.get("elapsed_s", 0),
                "success": True,
                "skipped": True,
            }

    meta = {
        "condition": condition_name,
        "run_id": f"{condition_name}_s{seed}" if multi_seed else condition_name,
        "game": args.game,
        "iterations": args.iterations,
        "seed": seed,
        "train_cfg": condition_cfg,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmd": build_training_command(args, condition_cfg, seed, output_dir),
    }
    json_dump(condition_path, meta)

    print(f"\n{'=' * 76}")
    print(f"TRAIN {meta['run_id']}  game={args.game}  iter={args.iterations}  seed={seed}")
    print(f"search={condition_cfg['search_profile']}  vl={condition_cfg['vl_mode']}  out={output_dir}")
    overrides = condition_runtime_overrides(condition_cfg)
    if overrides:
        print(f"overrides={json.dumps(overrides, sort_keys=True)}")
    print(f"{'=' * 76}")

    t0 = time.time()
    try:
        proc = subprocess.run(meta["cmd"], check=False, timeout=args.timeout_hours * 3600)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {meta['run_id']} exceeded {args.timeout_hours}h")
        returncode = -1
    elapsed = time.time() - t0

    meta["elapsed_s"] = round(elapsed, 1)
    meta["returncode"] = returncode
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(condition_path, meta)

    return {
        "condition": condition_name,
        "seed": seed,
        "run_dir": str(output_dir),
        "elapsed_s": elapsed,
        "success": returncode == 0,
        "skipped": False,
    }


def build_eval_cfg(game_name: str, eval_cfg: dict, device_name: str, model_path: str | None = None) -> tuple[dict, object]:
    import torch
    from quartz.alphazero_train import (
        GAME_CONFIGS,
        apply_config_overrides,
        auto_device_name,
        get_encoder,
    )

    cfg = dict(GAME_CONFIGS[game_name])
    cfg["_name"] = game_name
    cfg["search_profile"] = eval_cfg["search_profile"]
    cfg["vl_mode"] = eval_cfg["vl_mode"]
    cfg = apply_config_overrides(cfg, condition_runtime_overrides(eval_cfg))

    # Read model config from checkpoint metadata (or infer from state_dict keys)
    if model_path is not None:
        try:
            from quartz.backend import load_checkpoint_with_metadata
            sd, ckpt_cfg = load_checkpoint_with_metadata(model_path, torch, map_location="cpu")
            if ckpt_cfg:
                for k in ("blocks", "filters", "vh"):
                    if k in ckpt_cfg:
                        cfg[k] = ckpt_cfg[k]
            else:
                # Legacy checkpoint: infer block count from state_dict keys
                tower_indices = {int(k.split(".")[1]) for k in sd if k.startswith("tower.")}
                if tower_indices:
                    actual_blocks = max(tower_indices) + 1
                    if actual_blocks != cfg.get("blocks"):
                        cfg["blocks"] = actual_blocks
        except Exception:
            pass

    try:
        cfg["_encoder"] = get_encoder(game_name)
    except Exception:
        cfg["_encoder"] = None
    resolved_device = auto_device_name() if device_name == "auto" else device_name
    return cfg, torch.device(resolved_device)


def aggregate_matches(model_runs: list[dict], matches: list[dict]) -> dict:
    by_id = {run["id"]: run for run in model_runs}
    totals = {}
    overall = {run["id"]: {"id": run["id"], "points": 0.0, "games": 0, "wins": 0, "losses": 0, "draws": 0} for run in model_runs}

    for match in matches:
        if match["a_id"] not in by_id or match["b_id"] not in by_id:
            continue
        eval_name = match["eval_condition"]
        totals.setdefault(eval_name, {})
        for run in model_runs:
            totals[eval_name].setdefault(
                run["id"],
                {"id": run["id"], "points": 0.0, "games": 0, "wins": 0, "losses": 0, "draws": 0},
            )

        points_a = float(match["wins_a"]) + 0.5 * float(match["draws"])
        points_b = float(match["wins_b"]) + 0.5 * float(match["draws"])
        games = int(match["games"])

        for slot, points, wins, losses in (
            (match["a_id"], points_a, int(match["wins_a"]), int(match["wins_b"])),
            (match["b_id"], points_b, int(match["wins_b"]), int(match["wins_a"])),
        ):
            row = totals[eval_name][slot]
            row["points"] += points
            row["games"] += games
            row["wins"] += wins
            row["losses"] += losses
            row["draws"] += int(match["draws"])

            agg = overall[slot]
            agg["points"] += points
            agg["games"] += games
            agg["wins"] += wins
            agg["losses"] += losses
            agg["draws"] += int(match["draws"])

    leaderboards = {}
    for eval_name, rows in totals.items():
        ordered = []
        for row in rows.values():
            games = row["games"] or 1
            entry = dict(row)
            entry["score_rate"] = row["points"] / games
            entry["win_rate"] = row["wins"] / games
            entry["condition"] = by_id.get(row["id"], {}).get("condition")
            entry["seed"] = by_id.get(row["id"], {}).get("seed")
            ordered.append(entry)
        ordered.sort(key=lambda item: (-item["score_rate"], -item["win_rate"], item["id"]))
        leaderboards[eval_name] = ordered

    overall_rows = []
    for row in overall.values():
        games = row["games"] or 1
        entry = dict(row)
        entry["score_rate"] = row["points"] / games
        entry["win_rate"] = row["wins"] / games
        entry["condition"] = by_id.get(row["id"], {}).get("condition")
        entry["seed"] = by_id.get(row["id"], {}).get("seed")
        overall_rows.append(entry)
    overall_rows.sort(key=lambda item: (-item["score_rate"], -item["win_rate"], item["id"]))

    return {
        "matches": matches,
        "leaderboards": leaderboards,
        "overall": overall_rows,
    }


def summarize_conditions(runs: list[dict], eval_payload: dict | None = None) -> dict:
    training = {}
    for run in runs:
        cond = run.get("condition")
        if not cond:
            continue
        metrics = run.get("metrics") or {}
        row = training.setdefault(
            cond,
            {
                "condition": cond,
                "runs": 0,
                "sum_elo": 0.0,
                "n_elo": 0,
                "sum_score_rate": 0.0,
                "n_score_rate": 0,
                "sum_loss": 0.0,
                "n_loss": 0,
            },
        )
        row["runs"] += 1
        elo = metrics.get("published_elo")
        if elo is not None:
            row["sum_elo"] += float(elo)
            row["n_elo"] += 1
        score_rate = metrics.get("score_rate")
        if score_rate is not None:
            row["sum_score_rate"] += float(score_rate)
            row["n_score_rate"] += 1
        loss = metrics.get("loss")
        if loss is not None:
            row["sum_loss"] += float(loss)
            row["n_loss"] += 1

    training_rows = []
    for row in training.values():
        training_rows.append(
            {
                "condition": row["condition"],
                "runs": row["runs"],
                "mean_elo": (row["sum_elo"] / row["n_elo"]) if row["n_elo"] else None,
                "mean_score_rate": (row["sum_score_rate"] / row["n_score_rate"]) if row["n_score_rate"] else None,
                "mean_loss": (row["sum_loss"] / row["n_loss"]) if row["n_loss"] else None,
            }
        )
    training_rows.sort(
        key=lambda item: (
            -(item["mean_elo"] if item["mean_elo"] is not None else float("-inf")),
            -(item["mean_score_rate"] if item["mean_score_rate"] is not None else float("-inf")),
            item["condition"],
        )
    )

    evaluation_rows = []
    if eval_payload and eval_payload.get("overall"):
        grouped = {}
        for row in eval_payload["overall"]:
            cond = row.get("condition")
            if not cond:
                continue
            acc = grouped.setdefault(
                cond,
                {"condition": cond, "points": 0.0, "games": 0, "wins": 0, "losses": 0, "draws": 0, "entries": 0},
            )
            acc["points"] += float(row.get("points", 0.0))
            acc["games"] += int(row.get("games", 0))
            acc["wins"] += int(row.get("wins", 0))
            acc["losses"] += int(row.get("losses", 0))
            acc["draws"] += int(row.get("draws", 0))
            acc["entries"] += 1
        for row in grouped.values():
            games = row["games"] or 1
            evaluation_rows.append(
                {
                    "condition": row["condition"],
                    "entries": row["entries"],
                    "points": row["points"],
                    "games": row["games"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "draws": row["draws"],
                    "score_rate": row["points"] / games,
                    "win_rate": row["wins"] / games,
                }
            )
        evaluation_rows.sort(key=lambda item: (-item["score_rate"], -item["win_rate"], item["condition"]))

    return {
        "training": training_rows,
        "evaluation": evaluation_rows,
    }


def run_evaluation_matrix(
    args: argparse.Namespace,
    base_dir: Path,
    model_runs: list[dict],
    eval_conditions: dict[str, dict],
) -> dict | None:
    # Eligible if model_path exists (even if training crashed during eval phase,
    # the model was already saved and is valid for post-hoc evaluation)
    eligible = [run for run in model_runs if run["model_path"]]
    if len(eligible) < 2:
        return None
    eligible_ids = {run["id"] for run in eligible}

    from quartz.alphazero_train import arena_rust_nn

    eval_conditions = dict(eval_conditions)
    if args.include_strict_reference:
        eval_conditions = {**STRICT_REFERENCE_CONDITION, **eval_conditions}

    existing_path = base_dir / "evaluation_matrix.json"
    existing_matches = []
    if existing_path.exists() and not args.force_eval:
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            existing_matches = [
                row
                for row in existing.get("matches", [])
                if row.get("a_id") in eligible_ids and row.get("b_id") in eligible_ids
            ]
        except Exception:
            existing_matches = []

    match_index = {
        (row["eval_condition"], row["a_id"], row["b_id"]): row
        for row in existing_matches
    }

    def should_compare(model_a: dict, model_b: dict) -> bool:
        if not args.paired_seed_eval:
            return True
        seed_a = model_a.get("seed")
        seed_b = model_b.get("seed")
        if seed_a is None or seed_b is None or seed_a != seed_b:
            return False
        return model_a.get("condition") != model_b.get("condition")

    all_matches = list(existing_matches)
    for eval_name, eval_cfg in eval_conditions.items():
        for idx, model_a in enumerate(eligible):
            for model_b in eligible[idx + 1:]:
                if not should_compare(model_a, model_b):
                    continue
                key = (eval_name, model_a["id"], model_b["id"])
                if key in match_index and not args.force_eval:
                    continue

                print(f"  EVAL {eval_name}: {model_a['id']} vs {model_b['id']} ({args.eval_games} games)")
                cfg, device = build_eval_cfg(args.game, eval_cfg, args.device, model_path=model_a["model_path"])
                wa, wb, draws, wr, ci, sprt = arena_rust_nn(
                    model_a["model_path"],
                    model_b["model_path"],
                    cfg,
                    device,
                    n_games=args.eval_games,
                    rust_binary=args.rust_binary,
                    strict=True,
                )
                row = {
                    "eval_condition": eval_name,
                    "search_profile": eval_cfg["search_profile"],
                    "vl_mode": eval_cfg["vl_mode"],
                    "a_id": model_a["id"],
                    "b_id": model_b["id"],
                    "games": int(args.eval_games),
                    "wins_a": int(wa),
                    "wins_b": int(wb),
                    "draws": int(draws),
                    "win_rate_a": float(wr),
                    "ci": [float(ci[0]), float(ci[1])],
                    "sprt": sprt,
                }
                match_index[key] = row
                all_matches = list(match_index.values())
                payload = aggregate_matches(eligible, all_matches)
                payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                json_dump(existing_path, payload)

    payload = aggregate_matches(eligible, list(match_index.values()))
    payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(existing_path, payload)
    return payload


def select_champion(
    base_dir: Path,
    model_runs: list[dict],
    eval_payload: dict | None,
    eval_conditions: dict[str, dict] | None = None,
) -> dict | None:
    if not model_runs:
        return None
    by_id = {run["id"]: run for run in model_runs}
    eval_conditions = eval_conditions or SEARCH_VL_EVAL_CONDITIONS

    if eval_payload and eval_payload.get("overall"):
        top = eval_payload["overall"][0]
        champion_run = by_id[top["id"]]
        deployment_condition = None
        deployment_score = None
        for eval_name, leaderboard in eval_payload.get("leaderboards", {}).items():
            for row in leaderboard:
                if row["id"] != top["id"]:
                    continue
                if deployment_score is None or row["score_rate"] > deployment_score:
                    deployment_condition = eval_name
                    deployment_score = row["score_rate"]
                break
        deployment_cfg = copy.deepcopy(eval_conditions.get(deployment_condition, {}))
        if not deployment_cfg and deployment_condition in STRICT_REFERENCE_CONDITION:
            deployment_cfg = copy.deepcopy(STRICT_REFERENCE_CONDITION[deployment_condition])
        selection_metrics = {
            "overall_score_rate": top["score_rate"],
            "overall_points": top["points"],
            "overall_games": top["games"],
            "overall_win_rate": top["win_rate"],
        }
    else:
        ordered = sorted(
            model_runs,
            key=lambda run: (
                -(run["metrics"].get("published_elo") or float("-inf")),
                -(run["metrics"].get("score_rate") or float("-inf")),
                run["id"],
            ),
        )
        champion_run = ordered[0]
        deployment_condition = None
        deployment_cfg = copy.deepcopy(champion_run.get("train_cfg") or {})
        selection_metrics = {
            "published_elo": champion_run["metrics"].get("published_elo"),
            "score_rate": champion_run["metrics"].get("score_rate"),
        }

    if not champion_run.get("model_path"):
        return None

    payload = {
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": champion_run["id"],
        "model_path": champion_run["model_path"],
        "run_dir": champion_run["run_dir"],
        "condition": champion_run["condition"],
        "seed": champion_run["seed"],
        "game": champion_run["game"],
        "train_cfg": champion_run["train_cfg"],
        "training_metrics": champion_run["metrics"],
        "deployment_eval_condition": deployment_condition,
        "deployment_search_cfg": deployment_cfg,
        "selection_metrics": selection_metrics,
    }
    json_dump(base_dir / "champion.json", payload)
    return payload


def prepare_gomocup_bundle(args: argparse.Namespace, base_dir: Path, champion: dict | None) -> dict | None:
    if not champion:
        return None
    from quartz.gomocup_export import export_gomocup_bundle

    bundle_dir = Path(args.gomocup_dir or (base_dir / "gomocup_bundle"))
    metadata = {
        "condition": champion["condition"],
        "seed": champion["seed"],
        "training_metrics": champion.get("training_metrics", {}),
        "selection_metrics": champion.get("selection_metrics", {}),
    }
    bundle = export_gomocup_bundle(
        champion["model_path"],
        champion["game"],
        bundle_dir,
        metadata=metadata,
        search_cfg=champion.get("deployment_search_cfg", {}),
        include_checkpoint=True,
        verbose=args.verbose_export,
    )
    print(f"  Gomocup bundle: {bundle['bundle_dir']}")
    print(f"  Manifest: {bundle['manifest_path']}")
    print(f"  ONNX: {bundle['onnx_path']}")
    print(
        "  Build brain with: "
        f"scripts/build_gomocup_brain.sh --bundle-dir {bundle['bundle_dir']} --target-name {args.target_name}"
    )
    return bundle


def generate_report(base_dir: Path, selected_conditions: set[str] | None = None) -> dict:
    runs = discover_model_runs(base_dir)
    if selected_conditions:
        runs = [run for run in runs if run.get("condition") in selected_conditions]
    eval_payload = None
    champion = None
    eval_path = base_dir / "evaluation_matrix.json"
    champion_path = base_dir / "champion.json"
    if eval_path.exists():
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if champion_path.exists():
        champion = json.loads(champion_path.read_text(encoding="utf-8"))
    condition_summary = summarize_conditions(runs, eval_payload)

    print(f"\n{'=' * 132}")
    print(f"ABLATION REPORT  {base_dir}")
    print(f"{'=' * 132}")
    if not runs:
        print("  No completed training runs found.")
        return {"runs": []}

    print(
        f"\n{'Run':<20} {'Search':<12} {'VL':<10} {'Penalty':<20} "
        f"{'RootOnly':<8} {'Elo':>8} {'Loss':>9} {'Games':>8} {'Time':>8}"
    )
    print("-" * 132)
    for run in sorted(runs, key=lambda item: item["id"]):
        metrics = run["metrics"]
        elo = metrics.get("published_elo")
        loss = metrics.get("loss")
        elapsed_s = run.get("elapsed_s") or 0
        train_cfg = run["train_cfg"]
        print(
            f"{run['id']:<20} "
            f"{str(train_cfg.get('search_profile', '?')):<12} "
            f"{str(train_cfg.get('vl_mode', '?')):<10} "
            f"{str(train_cfg.get('penalty_mode', 'default')):<20} "
            f"{str(train_cfg.get('root_only_shaping', 'default')):<8} "
            f"{(f'{elo:.0f}' if elo is not None else '—'):>8} "
            f"{(f'{loss:.4f}' if loss is not None else '—'):>9} "
            f"{int(metrics.get('games_done') or 0):>8} "
            f"{(f'{elapsed_s / 60:.1f}m' if elapsed_s else '—'):>8}"
        )

    if eval_payload and eval_payload.get("overall"):
        print(f"\n{'Overall Eval Leaderboard':<40}")
        print("-" * 132)
        for row in eval_payload["overall"][: min(6, len(eval_payload["overall"]))]:
            print(
                f"{row['id']:<20} score={row['score_rate']:.3f} "
                f"win={row['win_rate']:.3f} points={row['points']:.1f}/{row['games']}"
            )

    if condition_summary["training"]:
        print(f"\n{'Condition Training Means':<40}")
        print("-" * 132)
        for row in condition_summary["training"]:
            mean_elo = row.get("mean_elo")
            mean_score = row.get("mean_score_rate")
            mean_loss = row.get("mean_loss")
            print(
                f"{row['condition']:<20} runs={row['runs']:<2} "
                f"mean_elo={(f'{mean_elo:.1f}' if mean_elo is not None else '—'):>8} "
                f"mean_score={(f'{mean_score:.3f}' if mean_score is not None else '—'):>7} "
                f"mean_loss={(f'{mean_loss:.4f}' if mean_loss is not None else '—'):>8}"
            )

    if condition_summary["evaluation"]:
        print(f"\n{'Condition Eval Means':<40}")
        print("-" * 132)
        for row in condition_summary["evaluation"]:
            print(
                f"{row['condition']:<20} entries={row['entries']:<2} "
                f"score={row['score_rate']:.3f} win={row['win_rate']:.3f} "
                f"points={row['points']:.1f}/{row['games']}"
            )

    if champion:
        print(f"\nChampion: {champion['model_id']}")
        dep = champion.get("deployment_search_cfg", {})
        print(
            f"  deploy search={dep.get('search_profile', 'quartz')} "
            f"vl={dep.get('vl_mode', 'adaptive')} "
            f"penalty={dep.get('penalty_mode', 'default')} "
            f"root_only={dep.get('root_only_shaping', 'default')} "
            f"model={champion.get('model_path')}"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base_dir),
        "runs": runs,
        "evaluation": eval_payload,
        "condition_summary": condition_summary,
        "champion": champion,
    }
    json_dump(base_dir / "ablation_report.json", payload)
    print(f"\nReport saved: {base_dir / 'ablation_report.json'}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QUARTZ Ablation Study")
    parser.add_argument(
        "--study",
        default="search_vl",
        choices=sorted(STUDY_PRESETS),
        help="Study preset: search_vl, controller, or controller_factorial",
    )
    parser.add_argument(
        "--game",
        default="gomoku15",
        choices=[
            "gomoku7",
            "gomoku15",
            "gomoku15_free",
            "gomoku15_std",
            "gomoku15_omok",
            "gomoku15_renju",
            "gomoku15_caro",
            "tictactoe",
        ],
    )
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations per condition")
    parser.add_argument("--eval-games", type=int, default=40, help="Games per pairwise post-train evaluation match")
    parser.add_argument("--eval-interval", type=int, default=5, help="Training-time checkpoint tournament cadence")
    parser.add_argument("--output", default="results/ablation", help="Output root directory")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--backend", default="torch", choices=["auto", "torch", "jax"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games-per-iter", type=int, default=None)
    parser.add_argument("--timeout-hours", type=int, default=24)
    parser.add_argument("--quick", action="store_true", help="Reduce self-play games per iteration")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--resident-session", action="store_true")
    parser.add_argument("--runtime-autotune", action="store_true")
    parser.add_argument("--seeds", default="42", help="Comma-separated training seeds")
    parser.add_argument("--conditions", help="Comma-separated condition names to run (default: all)")
    parser.add_argument("--eval-conditions", help="Comma-separated eval condition names to run (default: all)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--paired-seed-eval", action="store_true",
                        help="Only evaluate runs that share the same seed across different conditions")
    parser.add_argument("--include-strict-reference", action="store_true",
                        help="Also evaluate under baseline_strict search settings")
    parser.add_argument("--prepare-gomocup", action="store_true",
                        help="Export the selected champion as a Gomocup bundle")
    parser.add_argument("--gomocup-dir", default=None, help="Output directory for the Gomocup bundle")
    parser.add_argument("--target-name", default="pbrain-quartz", help="Suggested Gomocup binary name")
    parser.add_argument("--verbose-export", action="store_true")
    parser.add_argument("--report", metavar="DIR", help="Report on an existing ablation directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.report:
        base_dir = Path(args.report)
        manifest_path = base_dir / "study_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"eval_conditions": copy.deepcopy(SEARCH_VL_EVAL_CONDITIONS)}
        )
        runs = discover_model_runs(base_dir)
        selected_conditions = None
        if args.conditions:
            known = manifest.get("train_conditions", SEARCH_VL_TRAIN_CONDITIONS)
            selected_conditions = set(parse_selected_conditions(args.conditions, known))
            runs = [run for run in runs if run.get("condition") in selected_conditions]
        eval_payload = None if args.skip_eval else (
            json.loads((base_dir / "evaluation_matrix.json").read_text(encoding="utf-8"))
            if (base_dir / "evaluation_matrix.json").exists()
            else None
        )
        champion = select_champion(
            base_dir,
            runs,
            eval_payload,
            manifest.get("eval_conditions", SEARCH_VL_EVAL_CONDITIONS),
        )
        if args.prepare_gomocup:
            prepare_gomocup_bundle(args, base_dir, champion)
        generate_report(base_dir, selected_conditions=selected_conditions)
        return

    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_study_manifest(args)
    json_dump(base_dir / "study_manifest.json", manifest)
    train_conditions = manifest["train_conditions"]
    eval_conditions = manifest["eval_conditions"]

    seeds = manifest["seeds"]
    selected_conditions = manifest["conditions"]
    selected_condition_set = set(selected_conditions)
    multi_seed = len(seeds) > 1
    training_results = []

    if not args.skip_train:
        for condition_name in selected_conditions:
            condition_cfg = train_conditions[condition_name]
            for seed in seeds:
                result = run_training(args, base_dir, condition_name, condition_cfg, seed, multi_seed)
                training_results.append(result)

        print(f"\n{'=' * 76}")
        print("TRAINING PHASE COMPLETE")
        print(f"{'=' * 76}")
        for row in training_results:
            status = "SKIP" if row.get("skipped") else ("OK" if row["success"] else "FAIL")
            print(
                f"  [{status}] {row['condition']:<16} seed={row['seed']:<6} "
                f"{row['elapsed_s'] / 60:.1f} min"
            )

    model_runs = [
        run for run in discover_model_runs(base_dir)
        if run.get("condition") in selected_condition_set
    ]
    eval_payload = None
    if not args.skip_eval:
        eval_payload = run_evaluation_matrix(args, base_dir, model_runs, eval_conditions)
    champion = select_champion(base_dir, model_runs, eval_payload, eval_conditions)
    if args.prepare_gomocup:
        prepare_gomocup_bundle(args, base_dir, champion)
    generate_report(base_dir, selected_conditions=selected_condition_set)


if __name__ == "__main__":
    main()
