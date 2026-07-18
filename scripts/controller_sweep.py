#!/usr/bin/env python3
"""
Low-cost controller sweep for QUARTZ.

This runner is designed for the "hyperparameters are confounded with controller
family" problem. It avoids full training per trial by:

1. Probing candidate search configs on fixed random positions with frozen
   checkpoints and a deeper-search reference.
2. Running same-checkpoint, dual-config arena matches only for shortlisted
   candidates.
3. Bootstrapping a few weak checkpoints only when no frozen checkpoints are
   provided.

Example:
  venv/bin/python scripts/controller_sweep.py \
    --game gomoku7 \
    --checkpoint-dir results/ablation_controller_factorial_short/gomoku7/models \
    --samples 12 \
    --stage1-positions 12 \
    --stage2-topk 6 \
    --stage2-games 8
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quartz.contract_summary import stable_json_hash, summarize_contract_collection
from quartz.evaluation import score_rate_ci
from quartz.eval_runtime_profile import load_eval_runtime_overrides_from_model
from quartz.eval_timing_summary import summarize_controller_stage2_timings


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_runtime_overrides(base_cfg: dict, overrides: dict) -> dict:
    from quartz.alphazero_train import apply_runtime_overrides as apply_impl

    return apply_impl(base_cfg, overrides)


def parse_csv_items(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(item) for item in parse_csv_items(raw)]


def parse_selected_candidate_ids(raw: str | None, known_ids: list[str]) -> list[str]:
    if not raw:
        return list(known_ids)
    selected = parse_csv_items(raw)
    unknown = sorted(set(selected) - set(known_ids))
    if unknown:
        raise ValueError(f"unknown candidate ids: {', '.join(unknown)}")
    return selected


def resolve_resume_report(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        path = path / "sweep_report.json"
    return path


def load_resume_state(
    path_str: str,
) -> tuple[Path, dict, list[dict], list[str], list[dict] | None]:
    report_path = resolve_resume_report(path_str)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = data.get("manifest") or {}
    base_dir = report_path.parent
    candidates = list(manifest.get("candidates") or [])
    checkpoints = list(manifest.get("checkpoints") or [])
    shortlist = list(((data.get("stage1") or {}).get("shortlist")) or [])
    return base_dir, manifest, candidates, checkpoints, shortlist or None


def build_base_cfg(game_name: str, device_name: str, iters_override: int | None = None):
    import torch
    from quartz.alphazero_train import (
        GAME_CONFIGS,
        apply_config_overrides,
        auto_device_name,
        get_encoder,
    )

    cfg = dict(GAME_CONFIGS[game_name])
    cfg["_name"] = game_name
    cfg["search_profile"] = "quartz"
    cfg["vl_mode"] = "adaptive"
    if iters_override is not None:
        cfg["iters"] = max(1, int(iters_override))
    cfg = apply_config_overrides(cfg, {})
    try:
        cfg["_encoder"] = get_encoder(game_name)
    except Exception:
        cfg["_encoder"] = None
    resolved_device = auto_device_name() if device_name == "auto" else device_name
    return cfg, torch.device(resolved_device)


def build_default_search_space(base_cfg: dict) -> dict[str, list]:
    base_hbar = float(base_cfg.get("hbar_penalty_cap", 0.3) or 0.3)
    base_sigma = float(base_cfg.get("sigma_0", 0.3) or 0.3)
    base_min_visits = int(base_cfg.get("min_visits", 50) or 50)
    base_check_interval = int(base_cfg.get("check_interval", 100) or 100)
    base_cpuct = float(base_cfg.get("c_puct", 2.0) or 2.0)

    def around_int(value: int) -> list[int]:
        return sorted(
            set(
                [
                    max(4, int(round(value * 0.5))),
                    max(4, int(value)),
                    max(4, int(round(value * 1.5))),
                ]
            )
        )

    def around_float(
        value: float, delta: float, lo: float = 0.05, hi: float = 8.0
    ) -> list[float]:
        return sorted(
            set(
                round(max(lo, min(hi, x)), 3)
                for x in (value - delta, value, value + delta)
            )
        )

    return {
        "penalty_mode": ["GatedRefreshLegacy", "GatedRefresh"],
        "root_only_shaping": [False, True],
        "prior_refresh_rate": [0.0, 0.25, 0.5, 0.75],
        "prior_refresh_temp": [0.0, 0.5, 1.0],
        "hbar_penalty_cap": around_float(
            base_hbar, max(0.1, base_hbar * 0.5), lo=0.05, hi=1.0
        ),
        "sigma_0": around_float(base_sigma, 0.1, lo=0.05, hi=1.0),
        "min_visits": around_int(base_min_visits),
        "check_interval": around_int(base_check_interval),
        "c_puct": around_float(base_cpuct, 0.5, lo=0.5, hi=6.0),
    }


def canonicalize_candidate(overrides: dict, base_cfg: dict) -> dict:
    out = {
        "penalty_mode": str(
            overrides.get("penalty_mode", base_cfg.get("penalty_mode", "GatedRefresh"))
        ),
        "root_only_shaping": bool(overrides.get("root_only_shaping", False)),
        "prior_refresh_rate": float(
            overrides.get(
                "prior_refresh_rate", base_cfg.get("prior_refresh_rate", 0.0) or 0.0
            )
        ),
        "prior_refresh_temp": float(
            overrides.get(
                "prior_refresh_temp", base_cfg.get("prior_refresh_temp", 1.0) or 1.0
            )
        ),
        "hbar_penalty_cap": float(
            overrides.get(
                "hbar_penalty_cap", base_cfg.get("hbar_penalty_cap", 0.3) or 0.3
            )
        ),
        "sigma_0": float(overrides.get("sigma_0", base_cfg.get("sigma_0", 0.3) or 0.3)),
        "min_visits": int(
            overrides.get("min_visits", base_cfg.get("min_visits", 50) or 50)
        ),
        "check_interval": int(
            overrides.get("check_interval", base_cfg.get("check_interval", 100) or 100)
        ),
        "c_puct": float(overrides.get("c_puct", base_cfg.get("c_puct", 2.0) or 2.0)),
    }
    if out["prior_refresh_rate"] <= 0.0:
        out["prior_refresh_temp"] = float(
            base_cfg.get("prior_refresh_temp", 1.0) or 1.0
        )
    out["hbar_penalty_cap"] = round(out["hbar_penalty_cap"], 4)
    out["sigma_0"] = round(out["sigma_0"], 4)
    out["c_puct"] = round(out["c_puct"], 4)
    return out


def candidate_key(overrides: dict) -> str:
    return json.dumps(overrides, sort_keys=True, separators=(",", ":"))


def candidate_label(overrides: dict) -> str:
    return (
        f"{overrides['penalty_mode']}"
        f"/root={int(bool(overrides['root_only_shaping']))}"
        f"/pr={overrides['prior_refresh_rate']:.2f}"
        f"/tau={overrides['prior_refresh_temp']:.2f}"
        f"/h={overrides['hbar_penalty_cap']:.2f}"
        f"/s={overrides['sigma_0']:.2f}"
        f"/mv={overrides['min_visits']}"
        f"/ci={overrides['check_interval']}"
        f"/cp={overrides['c_puct']:.2f}"
    )


def build_stage1_probe_contract(
    checkpoint_path: str,
    candidate: dict,
    positions: list[dict],
    probe_iters: int,
    reference_multiplier: float,
) -> dict:
    positions_hash = stable_json_hash(positions)
    contract = {
        "checkpoint_path": str(checkpoint_path),
        "candidate_id": str(candidate["id"]),
        "candidate_key": candidate_key(candidate["overrides"]),
        "positions_hash": positions_hash,
        "positions": int(len(positions)),
        "probe_iters": int(probe_iters),
        "reference_multiplier": float(reference_multiplier),
    }
    contract["probe_contract_hash"] = stable_json_hash(contract)
    return contract


def build_stage1_expected_contracts(
    checkpoints: list[str],
    candidates: list[dict],
    positions: list[dict],
    probe_iters: int,
    reference_multiplier: float,
) -> dict[tuple[str, str], dict]:
    contracts = {}
    for checkpoint_path in checkpoints:
        for candidate in candidates:
            contract = build_stage1_probe_contract(
                checkpoint_path,
                candidate,
                positions,
                probe_iters,
                reference_multiplier,
            )
            contracts[(str(checkpoint_path), str(candidate["id"]))] = contract
    return contracts


def build_anchor_candidates(base_cfg: dict) -> list[dict]:
    anchors = [
        (
            "A1_legacy_base",
            {
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.0,
            },
        ),
        (
            "A2_legacy_krefresh",
            {
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
        ),
        (
            "A3_theory_base",
            {
                "penalty_mode": "GatedRefresh",
                "root_only_shaping": True,
                "prior_refresh_rate": 0.0,
            },
        ),
        (
            "A4_theory_krefresh",
            {
                "penalty_mode": "GatedRefresh",
                "root_only_shaping": True,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
        ),
    ]
    rows = []
    for candidate_id, overrides in anchors:
        merged = canonicalize_candidate(overrides, base_cfg)
        rows.append(
            {
                "id": candidate_id,
                "source": "anchor",
                "label": candidate_label(merged),
                "overrides": merged,
            }
        )
    return rows


def sample_candidate_pool(base_cfg: dict, n_random: int, seed: int) -> list[dict]:
    space = build_default_search_space(base_cfg)
    rng = random.Random(int(seed))
    candidates = build_anchor_candidates(base_cfg)
    seen = {candidate_key(row["overrides"]) for row in candidates}
    random_rows = []

    while len(random_rows) < max(0, int(n_random)):
        sampled = {key: rng.choice(values) for key, values in space.items()}
        overrides = canonicalize_candidate(sampled, base_cfg)
        key = candidate_key(overrides)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        random_rows.append(
            {
                "id": f"R{len(random_rows) + 1:02d}_{digest}",
                "source": "random",
                "label": candidate_label(overrides),
                "overrides": overrides,
            }
        )

    return candidates + random_rows


def load_checkpoint_status(run_dir: Path) -> dict | None:
    status_path = run_dir / "checkpoint_status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def has_promoted_checkpoint(run_dir: Path) -> bool:
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return False
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("_type") != "eval":
                continue
            verdict = row.get("verdict")
            if verdict is None:
                verdict = row.get("eval_verdict")
            if verdict == "promote":
                return True
    return False


def resolve_training_run_checkpoint(run_dir: Path) -> Path | None:
    status = load_checkpoint_status(run_dir)
    if status:
        preferred_name = status.get("preferred_posttrain_checkpoint")
        if preferred_name:
            candidate = run_dir / preferred_name
            if candidate.exists():
                return candidate
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    if best_path.exists() and latest_path.exists():
        return best_path if has_promoted_checkpoint(run_dir) else latest_path
    for name in ("best.pt", "latest.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def discover_checkpoint_paths(root: Path, limit: int | None = None) -> list[str]:
    if not root.exists():
        return []
    run_dirs = sorted(
        {path.parent for name in ("best.pt", "latest.pt") for path in root.rglob(name)}
    )
    found = []
    for run_dir in run_dirs:
        candidate = resolve_training_run_checkpoint(run_dir)
        if candidate is not None:
            found.append(candidate)
    rows = [str(path) for path in found]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def move_creates_line(
    board: np.ndarray, board_size: int, win_len: int, move: int, player: int
) -> bool:
    row0, col0 = divmod(int(move), board_size)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            row, col = row0 + sign * dr, col0 + sign * dc
            while (
                0 <= row < board_size
                and 0 <= col < board_size
                and board[row * board_size + col] == player
            ):
                count += 1
                row += sign * dr
                col += sign * dc
        if count >= win_len:
            return True
    return False


def generate_random_positions(
    game_name: str,
    cfg: dict,
    count: int,
    seed: int,
    min_moves: int | None = None,
    max_moves: int | None = None,
) -> list[dict]:
    if game_name not in {
        "gomoku7",
        "gomoku15",
        "gomoku15_free",
        "gomoku15_std",
        "gomoku15_omok",
        "gomoku15_renju",
        "gomoku15_caro",
        "tictactoe",
    }:
        raise ValueError(
            f"stage1 random position generation is not supported for {game_name}"
        )

    board_size = int(cfg["board"])
    board_cells = board_size * board_size
    win_len = int(cfg["win"])
    min_moves = max(
        2, int(min_moves if min_moves is not None else max(2, board_size // 2))
    )
    default_max = max(min_moves + 1, min(board_cells - 2, board_cells // 3))
    max_moves = int(max_moves if max_moves is not None else default_max)
    max_moves = max(min_moves + 1, min(max_moves, board_cells - 2))

    rng = random.Random(int(seed))
    positions = []
    attempts = 0
    attempt_cap = max(200, count * 100)
    while len(positions) < count and attempts < attempt_cap:
        attempts += 1
        moves_to_play = rng.randint(min_moves, max_moves)
        order = list(range(board_cells))
        rng.shuffle(order)
        board = np.zeros(board_cells, dtype=np.int8)
        player = 1
        legal = True
        for move in order[:moves_to_play]:
            board[move] = player
            if move_creates_line(board, board_size, win_len, move, player):
                legal = False
                break
            player = -player
        if not legal:
            continue
        if int(np.count_nonzero(board == 0)) < 2:
            continue
        positions.append(
            {
                "board": board.astype(int).tolist(),
                "player": int(player),
                "moves_played": int(moves_to_play),
            }
        )

    if len(positions) < count:
        raise RuntimeError(
            f"failed to generate {count} non-terminal positions for {game_name}; got {len(positions)}"
        )
    return positions


def build_bootstrap_command(
    args: argparse.Namespace, seed: int, output_dir: Path
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "quartz.train",
        "--game",
        args.game,
        "--iterations",
        str(args.bootstrap_iterations),
        "--games",
        str(args.bootstrap_games),
        "--output",
        str(output_dir),
        "--seed",
        str(seed),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--rust-binary",
        args.rust_binary,
        "--search-profile",
        "quartz",
        "--vl-mode",
        "adaptive",
        "--eval-interval",
        str(max(1, args.bootstrap_iterations)),
        "--eval-games",
        str(max(2, args.bootstrap_eval_games)),
        "--no-autotune",
    ]
    return cmd


def bootstrap_checkpoints(args: argparse.Namespace, base_dir: Path) -> list[str]:
    seeds = parse_seed_list(args.bootstrap_seeds)
    if not seeds:
        raise ValueError("bootstrap requested but no bootstrap seeds were provided")
    bootstrap_root = base_dir / "bootstrap"
    rows = []
    for seed in seeds:
        run_dir = bootstrap_root / f"seed_{seed}"
        model_path = resolve_training_run_checkpoint(run_dir)
        if model_path is not None and not args.force_bootstrap:
            rows.append(str(model_path))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_bootstrap_command(args, seed, run_dir)
        meta = {
            "seed": seed,
            "cmd": cmd,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_dump(run_dir / "bootstrap.json", meta)
        proc = subprocess.run(cmd, check=False)
        meta["returncode"] = proc.returncode
        meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json_dump(run_dir / "bootstrap.json", meta)
        if proc.returncode != 0:
            raise RuntimeError(
                f"bootstrap training failed for seed {seed} ({proc.returncode})"
            )
        model_path = resolve_training_run_checkpoint(run_dir)
        if model_path is not None:
            rows.append(str(model_path))
    return rows


def resolve_explicit_checkpoint_paths(raw_values: str | None) -> list[str]:
    rows: list[str] = []
    missing: list[str] = []
    directories: list[str] = []
    for raw in parse_csv_items(raw_values):
        path = Path(raw)
        if not path.exists():
            missing.append(raw)
            continue
        if path.is_dir():
            directories.append(raw)
            continue
        rows.append(str(path))
    problems = []
    if directories:
        joined = ", ".join(directories)
        problems.append(
            f"--checkpoints expects checkpoint files, not directories: {joined} (use --checkpoint-dir for directories)"
        )
    if missing:
        joined = ", ".join(missing)
        problems.append(f"checkpoint paths do not exist: {joined}")
    if problems:
        raise ValueError("; ".join(problems))
    return rows


def resolve_checkpoint_paths(args: argparse.Namespace, base_dir: Path) -> list[str]:
    rows: list[str] = []
    rows.extend(resolve_explicit_checkpoint_paths(args.checkpoints))
    if args.checkpoint_dir:
        rows.extend(
            discover_checkpoint_paths(
                Path(args.checkpoint_dir), limit=args.max_checkpoints
            )
        )
    deduped = []
    seen = set()
    for row in rows:
        if row not in seen:
            seen.add(row)
            deduped.append(row)
    if deduped:
        return deduped
    if not args.bootstrap_if_empty:
        raise RuntimeError(
            "no checkpoints found; pass --checkpoints/--checkpoint-dir or enable --bootstrap-if-empty"
        )
    return bootstrap_checkpoints(args, base_dir)


def _new_search_client(model_path: str, cfg: dict, device, rust_binary: str):
    import torch
    from quartz.alphazero_train import (
        AlphaZeroNet,
        NNSearchClient,
        load_torch_state_dict,
    )
    from quartz.backend import load_checkpoint_with_metadata

    model_cfg = dict(cfg)
    state_dict = None
    try:
        state_dict, ckpt_cfg = load_checkpoint_with_metadata(
            model_path, torch, map_location=device
        )
        if ckpt_cfg:
            for key in ("blocks", "filters", "vh"):
                if key in ckpt_cfg:
                    model_cfg[key] = ckpt_cfg[key]
        else:
            tower_indices = {
                int(k.split(".")[1])
                for k in state_dict
                if isinstance(k, str)
                and k.startswith("tower.")
                and k.split(".")[1].isdigit()
            }
            if tower_indices:
                model_cfg["blocks"] = max(tower_indices) + 1
    except Exception:
        state_dict = load_torch_state_dict(model_path, torch, map_location=device)

    model = AlphaZeroNet(model_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    client = NNSearchClient(model, model_cfg, device, rust_binary)
    client.start()
    return client


def _arena_dual_cfg_with_clients(
    client_a,
    cfg_a: dict,
    client_b,
    cfg_b: dict,
    *,
    n_games: int,
    strict: bool = True,
) -> tuple[int, int, int, float, list[float], str | None, dict]:
    board_size = int(cfg_a["board"])
    n2 = board_size**2
    win_len = int(cfg_a["win"])
    max_moves = n2
    wins_a = wins_b = draws = 0

    p0, p1 = 0.5, 0.55
    alpha, beta = 0.05, 0.05
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    sprt_decided = False
    sprt_result = None
    telemetry = {
        "candidate_a": {
            "search_count": 0,
            "benchmark_safe_count": 0,
            "root_visits": [],
            "halt_reason_hist": {},
            "selection_root_selects": 0,
        },
        "candidate_b": {
            "search_count": 0,
            "benchmark_safe_count": 0,
            "root_visits": [],
            "halt_reason_hist": {},
            "selection_root_selects": 0,
        },
    }

    penalty_mode_a = cfg_a.get("penalty_mode", "GatedRefresh")
    penalty_mode_b = cfg_b.get("penalty_mode", "GatedRefresh")

    for game_idx in range(int(n_games)):
        if game_idx % 2 == 0:
            first_client, second_client = client_a, client_b
            first_is_a = True
        else:
            first_client, second_client = client_b, client_a
            first_is_a = False

        board = np.zeros(n2, dtype=np.int8)
        player = 1
        winner = 0

        for _move_n in range(max_moves):
            client = first_client if player == 1 else second_client
            penalty_mode = penalty_mode_a if client is client_a else penalty_mode_b
            result = client.search_move(board, player, penalty_mode)
            if not result or "error" in result:
                if strict:
                    raise RuntimeError(
                        f"stage2 strict arena search failed: {result.get('error') if isinstance(result, dict) else 'empty response'}"
                    )
                break
            slot = "candidate_a" if client is client_a else "candidate_b"
            _record_stage2_search_telemetry(telemetry[slot], result)

            pol_entries = result.get("policy", [])
            if not pol_entries:
                break

            best = result.get("best_move", -1)
            legal = [i for i in range(n2) if board[i] == 0]
            if best < 0 or best >= n2 or board[best] != 0:
                if strict:
                    raise RuntimeError(
                        f"stage2 strict arena produced illegal move: {best}"
                    )
                if legal:
                    best = random.choice(legal)
                else:
                    break

            board[best] = player

            if win_len > 0:
                r0, c0 = best // board_size, best % board_size
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cnt = 1
                    for sign in [1, -1]:
                        nr, nc = r0 + sign * dr, c0 + sign * dc
                        while (
                            0 <= nr < board_size
                            and 0 <= nc < board_size
                            and board[nr * board_size + nc] == player
                        ):
                            cnt += 1
                            nr += sign * dr
                            nc += sign * dc
                    if cnt >= win_len:
                        winner = player
                        break
                if winner:
                    break

            if not [i for i in range(n2) if board[i] == 0]:
                break
            player = -player

        if winner == 1:
            if first_is_a:
                wins_a += 1
            else:
                wins_b += 1
        elif winner == -1:
            if first_is_a:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1

        decisive = wins_a + wins_b
        if decisive > 0 and not sprt_decided:
            llr = wins_a * math.log(p1 / p0) + (decisive - wins_a) * math.log(
                (1 - p1) / (1 - p0)
            )
            if llr >= upper_bound:
                sprt_decided = True
                sprt_result = "H1_accept"
            elif llr <= lower_bound:
                sprt_decided = True
                sprt_result = "H0_accept"

    wr = (wins_a + 0.5 * draws) / max(1, int(n_games))
    _score_rate, ci = score_rate_ci(wins_a, draws, int(n_games))
    return (
        wins_a,
        wins_b,
        draws,
        wr,
        [float(ci[0]), float(ci[1])],
        sprt_result,
        {
            key: _finalize_stage2_search_telemetry(value)
            for key, value in telemetry.items()
        },
    )


def _record_stage2_search_telemetry(bucket: dict, result: dict) -> None:
    bucket["search_count"] = int(bucket.get("search_count", 0) or 0) + 1
    manifest = result.get("search_manifest") or {}
    if manifest.get("benchmark_safe") is not False:
        bucket["benchmark_safe_count"] = (
            int(bucket.get("benchmark_safe_count", 0) or 0) + 1
        )
    realized = result.get("realized_budget") or {}
    controller = result.get("controller_summary") or {}
    root_visits = realized.get("root_visits")
    if root_visits is None:
        root_visits = realized.get("realized_iterations")
    if root_visits is None:
        root_visits = result.get("iters") or result.get("iterations")
    if root_visits is not None:
        try:
            bucket.setdefault("root_visits", []).append(float(root_visits))
        except Exception:
            pass
    reason = (
        realized.get("stop_reason")
        or controller.get("stop_reason")
        or result.get("stop_reason")
    )
    if reason:
        hist = bucket.setdefault("halt_reason_hist", {})
        key = str(reason)
        hist[key] = int(hist.get(key, 0) or 0) + 1
    trace = controller.get("selection_trace") or {}
    if trace.get("root_selects") is not None:
        try:
            bucket["selection_root_selects"] = int(
                bucket.get("selection_root_selects", 0) or 0
            ) + int(trace["root_selects"])
        except Exception:
            pass


def _finalize_stage2_search_telemetry(bucket: dict) -> dict:
    visits = [float(v) for v in bucket.get("root_visits") or []]
    search_count = int(bucket.get("search_count", 0) or 0)
    return {
        "search_count": search_count,
        "benchmark_safe_frac": (
            float(bucket.get("benchmark_safe_count", 0) / search_count)
            if search_count
            else None
        ),
        "root_visits": {
            "count": int(len(visits)),
            "mean": float(sum(visits) / len(visits)) if visits else None,
            "min": float(min(visits)) if visits else None,
            "max": float(max(visits)) if visits else None,
        },
        "halt_reason_hist": dict(bucket.get("halt_reason_hist") or {}),
        "selection_root_selects": int(bucket.get("selection_root_selects", 0) or 0),
    }


def _build_stage2_client_pool(
    checkpoint_path: str,
    candidates: list[dict],
    base_cfg: dict,
    device,
    rust_binary: str,
    arena_iters: int,
) -> tuple[dict[str, dict], dict[str, float]]:
    entries = {}
    t0 = time.perf_counter()
    runtime_overrides = load_eval_runtime_overrides_from_model(
        checkpoint_path, str(device)
    )
    for candidate in candidates:
        cfg = apply_runtime_overrides(base_cfg, candidate["overrides"])
        if runtime_overrides:
            cfg.update(runtime_overrides)
        cfg["iters"] = int(arena_iters)
        client = _new_search_client(checkpoint_path, cfg, device, rust_binary)
        entries[candidate["id"]] = {
            "cfg": cfg,
            "client": client,
        }
    return entries, {
        "client_bootstrap_s": time.perf_counter() - t0,
        "client_count": len(entries),
    }


def probe_candidate_on_positions(
    model_path: str,
    base_cfg: dict,
    device,
    positions: list[dict],
    candidate: dict,
    rust_binary: str,
    probe_iters: int,
    reference_multiplier: float,
) -> dict:
    from quartz.alphazero_train import dense_policy_from_sparse

    cfg_probe = apply_runtime_overrides(base_cfg, candidate["overrides"])
    cfg_probe["iters"] = max(1, int(probe_iters))
    cfg_ref = copy.deepcopy(cfg_probe)
    cfg_ref["iters"] = max(
        cfg_probe["iters"] + 1, int(round(cfg_probe["iters"] * reference_multiplier))
    )

    client_probe = _new_search_client(model_path, cfg_probe, device, rust_binary)
    client_ref = _new_search_client(model_path, cfg_ref, device, rust_binary)
    matched = 0
    valid = 0
    timeout_count = 0
    policy_mass_sum = 0.0
    value_gap_sum = 0.0
    latency_ms_sum = 0.0
    try:
        for position in positions:
            board = np.asarray(position["board"], dtype=np.int8)
            player = int(position["player"])
            try:
                t0 = time.perf_counter()
                probe = client_probe.search_move(
                    board, player, penalty_mode=cfg_probe["penalty_mode"]
                )
                latency_ms_sum += (time.perf_counter() - t0) * 1000.0
                ref = client_ref.search_move(
                    board, player, penalty_mode=cfg_ref["penalty_mode"]
                )
            except TimeoutError:
                timeout_count += 1
                continue
            best_probe = int(probe.get("best_move", -1))
            best_ref = int(ref.get("best_move", -1))
            if best_probe < 0 or best_ref < 0:
                continue
            valid += 1
            if best_probe == best_ref:
                matched += 1
            policy = dense_policy_from_sparse(
                probe.get("policy", []), cfg_probe["actions"]
            )
            if 0 <= best_ref < len(policy):
                policy_mass_sum += float(policy[best_ref])
            value_gap_sum += abs(
                float(probe.get("value", 0.0)) - float(ref.get("value", 0.0))
            )
    finally:
        client_probe.stop()
        client_ref.stop()

    agreement = matched / valid if valid else 0.0
    policy_mass = policy_mass_sum / valid if valid else 0.0
    value_gap = value_gap_sum / valid if valid else 0.0
    avg_latency_ms = latency_ms_sum / valid if valid else None
    score = agreement + 0.25 * policy_mass - 0.10 * value_gap - 0.05 * timeout_count
    if avg_latency_ms is not None:
        score -= 0.0005 * avg_latency_ms
    return {
        "candidate_id": candidate["id"],
        "candidate_label": candidate["label"],
        "candidate_source": candidate["source"],
        "checkpoint_path": model_path,
        "positions": len(positions),
        "valid_positions": valid,
        "agreement_rate": agreement,
        "reference_policy_mass": policy_mass,
        "mean_value_gap": value_gap,
        "mean_latency_ms": avg_latency_ms,
        "timeout_count": timeout_count,
        "stage1_score": score,
    }


def summarize_stage1_results(candidates: list[dict], rows: list[dict]) -> list[dict]:
    grouped = {
        row["id"]: {
            "candidate_id": row["id"],
            "candidate_label": row["label"],
            "candidate_source": row["source"],
            "score_sum": 0.0,
            "weight_sum": 0,
            "latency_sum": 0.0,
            "latency_n": 0,
            "agreement_sum": 0.0,
            "policy_mass_sum": 0.0,
            "value_gap_sum": 0.0,
            "timeout_count": 0,
        }
        for row in candidates
    }
    for row in rows:
        acc = grouped[row["candidate_id"]]
        weight = int(row.get("valid_positions") or 0)
        acc["score_sum"] += float(row.get("stage1_score", 0.0)) * max(1, weight)
        acc["weight_sum"] += max(1, weight)
        acc["agreement_sum"] += float(row.get("agreement_rate", 0.0)) * max(1, weight)
        acc["policy_mass_sum"] += float(row.get("reference_policy_mass", 0.0)) * max(
            1, weight
        )
        acc["value_gap_sum"] += float(row.get("mean_value_gap", 0.0)) * max(1, weight)
        if row.get("mean_latency_ms") is not None:
            acc["latency_sum"] += float(row["mean_latency_ms"]) * max(1, weight)
            acc["latency_n"] += max(1, weight)
        acc["timeout_count"] += int(row.get("timeout_count") or 0)

    summary = []
    for acc in grouped.values():
        weight = max(1, acc["weight_sum"])
        summary.append(
            {
                "candidate_id": acc["candidate_id"],
                "candidate_label": acc["candidate_label"],
                "candidate_source": acc["candidate_source"],
                "stage1_score": acc["score_sum"] / weight,
                "agreement_rate": acc["agreement_sum"] / weight,
                "reference_policy_mass": acc["policy_mass_sum"] / weight,
                "mean_value_gap": acc["value_gap_sum"] / weight,
                "mean_latency_ms": (acc["latency_sum"] / acc["latency_n"])
                if acc["latency_n"]
                else None,
                "timeout_count": acc["timeout_count"],
            }
        )
    summary.sort(
        key=lambda item: (
            -item["stage1_score"],
            -item["agreement_rate"],
            -(item["reference_policy_mass"]),
            item["mean_latency_ms"]
            if item["mean_latency_ms"] is not None
            else float("inf"),
            item["candidate_id"],
        )
    )
    return summary


def normalize_stage1_payload(
    stage1_payload: dict | None,
    candidates: list[dict],
    checkpoints: list[str],
    positions: list[dict],
    probe_iters: int,
    reference_multiplier: float,
    shortlist_topk: int,
) -> tuple[dict, list[dict]]:
    expected_contracts = build_stage1_expected_contracts(
        checkpoints, candidates, positions, probe_iters, reference_multiplier
    )
    rows = []
    discarded_rows = []
    seen_keys = set()
    for row in list((stage1_payload or {}).get("rows") or []):
        key = (str(row.get("checkpoint_path")), str(row.get("candidate_id")))
        expected = expected_contracts.get(key)
        if expected is None:
            discarded_rows.append(
                {
                    "checkpoint_path": key[0],
                    "candidate_id": key[1],
                    "reason": "stage1_contract_missing",
                }
            )
            continue
        found_hash = row.get("probe_contract_hash")
        if found_hash != expected["probe_contract_hash"]:
            discarded_rows.append(
                {
                    "checkpoint_path": key[0],
                    "candidate_id": key[1],
                    "reason": "stage1_probe_contract_changed",
                    "expected_probe_contract_hash": expected["probe_contract_hash"],
                    "found_probe_contract_hash": found_hash,
                }
            )
            continue
        rows.append(row)
        seen_keys.add(key)

    summary = summarize_stage1_results(candidates, rows)
    shortlist = (
        select_stage2_candidates(candidates, summary, shortlist_topk) if summary else []
    )
    payload = {
        "rows": rows,
        "summary": summary,
        "shortlist": shortlist,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "discarded_rows": discarded_rows,
        "expected_probe_contracts": list(expected_contracts.values()),
    }
    missing = [
        contract for key, contract in expected_contracts.items() if key not in seen_keys
    ]
    return attach_stage1_contract_summary(payload), missing


def _legacy_stage1_contracts(payload: dict) -> list[dict]:
    contracts = []
    seen = set()
    for row in list(payload.get("rows") or []):
        contract = {
            "checkpoint_path": str(row.get("checkpoint_path", "")),
            "candidate_id": str(row.get("candidate_id", "")),
            "candidate_label": row.get("candidate_label"),
            "legacy_partial": True,
        }
        contract["probe_contract_hash"] = stable_json_hash(contract)
        key = contract["probe_contract_hash"]
        if key in seen:
            continue
        seen.add(key)
        contracts.append(contract)
    return contracts


def _legacy_stage2_contracts(payload: dict) -> list[dict]:
    contracts = []
    seen = set()
    for row in list(payload.get("matches") or []):
        contract = {
            "checkpoint_path": str(row.get("checkpoint_path", "")),
            "candidate_a": str(row.get("candidate_a", "")),
            "candidate_b": str(row.get("candidate_b", "")),
            "games": int(row.get("games", 0) or 0),
            "manifest_hash_a": str(row.get("manifest_hash_a", "")),
            "manifest_hash_b": str(row.get("manifest_hash_b", "")),
            "legacy_partial": True,
        }
        if not contract["manifest_hash_a"] and not contract["manifest_hash_b"]:
            contract["manifest_hash_a"] = stable_json_hash(
                {
                    "checkpoint_path": contract["checkpoint_path"],
                    "candidate_a": contract["candidate_a"],
                    "candidate_b": contract["candidate_b"],
                    "games": contract["games"],
                }
            )
        contract_key = stable_json_hash(contract)
        if contract_key in seen:
            continue
        seen.add(contract_key)
        contracts.append(contract)
    return contracts


def stage1_contracts_for_summary(payload: dict) -> list[dict]:
    contracts = list(payload.get("expected_probe_contracts") or [])
    if contracts:
        return contracts
    return _legacy_stage1_contracts(payload)


def stage2_contracts_for_summary(payload: dict) -> list[dict]:
    contracts = list(payload.get("expected_match_contracts") or [])
    if contracts:
        return contracts
    return _legacy_stage2_contracts(payload)


def attach_stage1_contract_summary(payload: dict) -> dict:
    payload["contract_summary"] = summarize_contract_collection(
        stage1_contracts_for_summary(payload),
        payload.get("discarded_rows"),
        "probe_contract_hash",
    )
    return payload


def attach_stage2_contract_summary(payload: dict) -> dict:
    payload["contract_summary"] = summarize_contract_collection(
        stage2_contracts_for_summary(payload),
        payload.get("discarded_matches"),
        "manifest_hash_a",
    )
    return payload


def build_controller_sweep_contract_summary(
    stage1_payload: dict, stage2_payload: dict | None
) -> dict:
    return {
        "stage1": summarize_contract_collection(
            stage1_contracts_for_summary(stage1_payload),
            stage1_payload.get("discarded_rows"),
            "probe_contract_hash",
        ),
        "stage2": summarize_contract_collection(
            stage2_contracts_for_summary(stage2_payload or {}),
            (stage2_payload or {}).get("discarded_matches"),
            "manifest_hash_a",
        ),
    }


def select_stage2_candidates(
    candidates: list[dict], stage1_summary: list[dict], topk: int
) -> list[dict]:
    by_id = {row["id"]: row for row in candidates}
    anchors = [row["id"] for row in candidates if row["source"] == "anchor"]
    target = max(int(topk), len(anchors))
    selected = []
    for candidate_id in anchors:
        if candidate_id in by_id:
            selected.append(candidate_id)
    if len(selected) >= target:
        return [by_id[candidate_id] for candidate_id in selected[:target]]
    for row in stage1_summary:
        candidate_id = row["candidate_id"]
        if candidate_id in selected or candidate_id not in by_id:
            continue
        selected.append(candidate_id)
        if len(selected) >= target:
            break
    return [by_id[candidate_id] for candidate_id in selected]


def aggregate_stage2_matches(candidates: list[dict], matches: list[dict]) -> dict:
    by_id = {row["id"]: row for row in candidates}
    overall = {
        row["id"]: {
            "candidate_id": row["id"],
            "candidate_label": row["label"],
            "candidate_source": row["source"],
            "points": 0.0,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
        }
        for row in candidates
    }
    by_checkpoint: dict[str, dict[str, dict]] = {}
    for match in matches:
        checkpoint = str(match["checkpoint_path"])
        rows = by_checkpoint.setdefault(checkpoint, copy.deepcopy(overall))
        points_a = float(match["wins_a"]) + 0.5 * float(match["draws"])
        points_b = float(match["wins_b"]) + 0.5 * float(match["draws"])
        for slot, points, wins, losses in (
            (
                match["candidate_a"],
                points_a,
                int(match["wins_a"]),
                int(match["wins_b"]),
            ),
            (
                match["candidate_b"],
                points_b,
                int(match["wins_b"]),
                int(match["wins_a"]),
            ),
        ):
            entry = overall[slot]
            entry["points"] += points
            entry["games"] += int(match["games"])
            entry["wins"] += wins
            entry["losses"] += losses
            entry["draws"] += int(match["draws"])
            local = rows[slot]
            local["points"] += points
            local["games"] += int(match["games"])
            local["wins"] += wins
            local["losses"] += losses
            local["draws"] += int(match["draws"])

    def finalize(rows: dict[str, dict]) -> list[dict]:
        ordered = []
        for row in rows.values():
            games = row["games"] or 1
            entry = dict(row)
            entry["score_rate"] = row["points"] / games
            entry["win_rate"] = row["wins"] / games
            ordered.append(entry)
        ordered.sort(
            key=lambda item: (
                -item["score_rate"],
                -item["win_rate"],
                item["candidate_id"],
            )
        )
        return ordered

    return {
        "matches": matches,
        "overall": finalize(overall),
        "by_checkpoint": {
            checkpoint: finalize(rows) for checkpoint, rows in by_checkpoint.items()
        },
    }


def run_stage2_round_robin(
    candidates: list[dict],
    checkpoints: list[str],
    base_cfg: dict,
    device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict | None:
    if len(candidates) < 2:
        return None
    from quartz import runtime_support as support_mod

    stage2_path = output_dir / "stage2_round_robin.json"
    matches = []
    discarded_matches = []
    existing_exact = {}
    existing_coarse = {}
    checkpoint_timings = {}
    if stage2_path.exists():
        try:
            existing = json.loads(stage2_path.read_text(encoding="utf-8"))
            for row in existing.get("matches", []):
                coarse_key = (
                    str(row.get("checkpoint_path")),
                    str(row.get("candidate_a")),
                    str(row.get("candidate_b")),
                )
                exact_key = coarse_key + (
                    int(row.get("games", 0) or 0),
                    str(row.get("manifest_hash_a")),
                    str(row.get("manifest_hash_b")),
                )
                existing_exact[exact_key] = row
                existing_coarse.setdefault(coarse_key, []).append(row)
        except Exception:
            existing_exact = {}
            existing_coarse = {}

    expected_contracts = []
    arena_iters = (
        args.arena_iters if args.arena_iters is not None else int(base_cfg["iters"])
    )
    for checkpoint_path in checkpoints:
        pending_pairs = []
        runtime_overrides = load_eval_runtime_overrides_from_model(
            checkpoint_path, str(device)
        )
        for idx, candidate_a in enumerate(candidates):
            for candidate_b in candidates[idx + 1 :]:
                cfg_a = apply_runtime_overrides(base_cfg, candidate_a["overrides"])
                cfg_b = apply_runtime_overrides(base_cfg, candidate_b["overrides"])
                if runtime_overrides:
                    cfg_a.update(runtime_overrides)
                    cfg_b.update(runtime_overrides)
                cfg_a["iters"] = int(arena_iters)
                cfg_b["iters"] = int(arena_iters)
                manifest_a = support_mod.build_search_manifest(cfg_a)
                manifest_b = support_mod.build_search_manifest(cfg_b)
                manifest_hash_a = support_mod.search_manifest_hash(cfg_a)
                manifest_hash_b = support_mod.search_manifest_hash(cfg_b)
                coarse_key = (
                    str(checkpoint_path),
                    candidate_a["id"],
                    candidate_b["id"],
                )
                exact_key = coarse_key + (
                    int(args.stage2_games),
                    manifest_hash_a,
                    manifest_hash_b,
                )
                expected_contracts.append(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "candidate_a": candidate_a["id"],
                        "candidate_b": candidate_b["id"],
                        "manifest_a": manifest_a,
                        "manifest_b": manifest_b,
                        "manifest_hash_a": manifest_hash_a,
                        "manifest_hash_b": manifest_hash_b,
                        "games": int(args.stage2_games),
                    }
                )
                if exact_key in existing_exact:
                    matches.append(existing_exact[exact_key])
                    continue
                for stale in existing_coarse.get(coarse_key, []):
                    stale_games = int(stale.get("games", 0) or 0)
                    stale_hash_a = str(stale.get("manifest_hash_a"))
                    stale_hash_b = str(stale.get("manifest_hash_b"))
                    if (
                        stale_games == int(args.stage2_games)
                        and stale_hash_a == manifest_hash_a
                        and stale_hash_b == manifest_hash_b
                    ):
                        continue
                    reason = "stage2_contract_changed"
                    if stale_games != int(args.stage2_games):
                        reason = "stage2_games_changed"
                    elif (
                        stale_hash_a != manifest_hash_a
                        or stale_hash_b != manifest_hash_b
                    ):
                        reason = "stage2_manifest_hash_changed"
                    discarded_matches.append(
                        {
                            "checkpoint_path": str(checkpoint_path),
                            "candidate_a": candidate_a["id"],
                            "candidate_b": candidate_b["id"],
                            "reason": reason,
                            "expected_games": int(args.stage2_games),
                            "found_games": stale_games,
                            "expected_manifest_hash_a": manifest_hash_a,
                            "expected_manifest_hash_b": manifest_hash_b,
                            "found_manifest_hash_a": stale_hash_a,
                            "found_manifest_hash_b": stale_hash_b,
                        }
                    )
                pending_pairs.append(
                    {
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "manifest_a": manifest_a,
                        "manifest_b": manifest_b,
                        "manifest_hash_a": manifest_hash_a,
                        "manifest_hash_b": manifest_hash_b,
                    }
                )
        if not pending_pairs:
            continue
        client_pool = {}
        checkpoint_timing = {"pairs": len(pending_pairs)}
        try:
            client_pool, pool_meta = _build_stage2_client_pool(
                checkpoint_path,
                candidates,
                base_cfg,
                device,
                args.rust_binary,
                arena_iters,
            )
            checkpoint_timing.update(pool_meta)
            checkpoint_timings[str(checkpoint_path)] = {
                "pairs": int(checkpoint_timing["pairs"]),
                "client_bootstrap_s": round(
                    float(checkpoint_timing["client_bootstrap_s"]), 6
                ),
                "client_count": int(checkpoint_timing["client_count"]),
            }
            for pair in pending_pairs:
                candidate_a = pair["candidate_a"]
                candidate_b = pair["candidate_b"]
                print(
                    f"  STAGE2 {Path(checkpoint_path).name}: {candidate_a['id']} vs {candidate_b['id']} "
                    f"({args.stage2_games} games)"
                )
                match_t0 = time.perf_counter()
                arena_result = _arena_dual_cfg_with_clients(
                    client_pool[candidate_a["id"]]["client"],
                    client_pool[candidate_a["id"]]["cfg"],
                    client_pool[candidate_b["id"]]["client"],
                    client_pool[candidate_b["id"]]["cfg"],
                    n_games=args.stage2_games,
                    strict=True,
                )
                if len(arena_result) == 6:
                    wa, wb, draws, wr, ci, sprt = arena_result
                    budget_trace = {}
                else:
                    wa, wb, draws, wr, ci, sprt, budget_trace = arena_result
                matches.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "candidate_a": candidate_a["id"],
                        "candidate_b": candidate_b["id"],
                        "manifest_a": pair["manifest_a"],
                        "manifest_b": pair["manifest_b"],
                        "manifest_hash_a": pair["manifest_hash_a"],
                        "manifest_hash_b": pair["manifest_hash_b"],
                        "games": int(args.stage2_games),
                        "wins_a": int(wa),
                        "wins_b": int(wb),
                        "draws": int(draws),
                        "win_rate_a": float(wr),
                        "ci": [float(ci[0]), float(ci[1])],
                        "sprt": sprt,
                        "realized_budget_trace": budget_trace,
                        "timing_s": {
                            "client_bootstrap_s": round(
                                float(checkpoint_timing["client_bootstrap_s"]), 6
                            ),
                            "match_elapsed_s": round(time.perf_counter() - match_t0, 6),
                        },
                    }
                )
                payload = aggregate_stage2_matches(candidates, matches)
                payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                payload["discarded_matches"] = discarded_matches
                payload["expected_match_contracts"] = expected_contracts
                payload["checkpoint_timings"] = checkpoint_timings
                payload["stage2_timing_summary"] = summarize_controller_stage2_timings(
                    payload
                )
                attach_stage2_contract_summary(payload)
                json_dump(stage2_path, payload)
        finally:
            for entry in client_pool.values():
                try:
                    entry["client"].stop()
                except Exception:
                    pass
    payload = aggregate_stage2_matches(candidates, matches)
    payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload["discarded_matches"] = discarded_matches
    payload["expected_match_contracts"] = expected_contracts
    payload["checkpoint_timings"] = checkpoint_timings
    payload["stage2_timing_summary"] = summarize_controller_stage2_timings(payload)
    attach_stage2_contract_summary(payload)
    json_dump(stage2_path, payload)
    return payload


def build_report(
    base_dir: Path, manifest: dict, stage1_payload: dict, stage2_payload: dict | None
) -> dict:
    stage2_timing_summary = summarize_controller_stage2_timings(stage2_payload)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest,
        "stage1": stage1_payload,
        "stage2": stage2_payload,
        "contract_summary": build_controller_sweep_contract_summary(
            stage1_payload, stage2_payload
        ),
        "stage2_timing_summary": stage2_timing_summary,
        "recommended": None,
    }
    if stage2_payload and stage2_payload.get("overall"):
        report["recommended"] = stage2_payload["overall"][0]
    elif stage1_payload.get("summary"):
        report["recommended"] = stage1_payload["summary"][0]
    json_dump(base_dir / "sweep_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-cost QUARTZ controller sweep")
    parser.add_argument(
        "--game",
        default="gomoku7",
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
    parser.add_argument(
        "--output", default="results/controller_sweep", help="Output root directory"
    )
    parser.add_argument(
        "--resume-report",
        default=None,
        help="Existing sweep_report.json or its directory; reuse manifest/stage1 shortlist and run stage2 only unless checkpoints/candidates are overridden",
    )
    parser.add_argument(
        "--candidate-ids",
        default=None,
        help="Comma-separated candidate ids to keep for stage2 when resuming from an existing sweep",
    )
    parser.add_argument(
        "--checkpoints", default=None, help="Comma-separated checkpoint paths"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to scan recursively for best.pt/latest.pt",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for candidate sampling and position suite",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of random candidates in addition to anchor configs",
    )
    parser.add_argument("--probe-iters", type=int, default=96)
    parser.add_argument("--reference-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--search-stall-timeout-s",
        type=float,
        default=45.0,
        help="Per-search read timeout for Rust search callbacks during sweep probes and arena",
    )
    parser.add_argument("--stage1-positions", type=int, default=12)
    parser.add_argument(
        "--shortlist-topk",
        "--stage1-topk",
        dest="shortlist_topk",
        type=int,
        default=6,
        help="How many candidates survive into stage2; anchor configs are always preserved",
    )
    parser.add_argument("--position-min-moves", type=int, default=None)
    parser.add_argument("--position-max-moves", type=int, default=None)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--arena-iters", type=int, default=None)
    parser.add_argument("--stage2-games", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(
        float(args.search_stall_timeout_s)
    )

    resume_stage1 = None
    resume_shortlist = None
    if args.resume_report:
        base_dir, manifest, candidates, checkpoints, resume_shortlist = (
            load_resume_state(args.resume_report)
        )
        manifest = dict(manifest)
        game_name = manifest.get("game", args.game)
        base_cfg, device = build_base_cfg(game_name, args.device)
        args.game = game_name
        positions_path = base_dir / "stage1_positions.json"
        positions_payload = (
            json.loads(positions_path.read_text(encoding="utf-8"))
            if positions_path.exists()
            else {"positions": []}
        )
        positions = list(positions_payload.get("positions") or [])
    else:
        base_dir = Path(args.output) / args.game
        base_dir.mkdir(parents=True, exist_ok=True)
        base_cfg, device = build_base_cfg(args.game, args.device)
        checkpoints = resolve_checkpoint_paths(args, base_dir)
        candidates = sample_candidate_pool(base_cfg, args.samples, args.seed)
        positions = generate_random_positions(
            args.game,
            base_cfg,
            args.stage1_positions,
            args.seed,
            min_moves=args.position_min_moves,
            max_moves=args.position_max_moves,
        )

        manifest = {
            "format_version": 1,
            "game": args.game,
            "seed": int(args.seed),
            "rust_binary": args.rust_binary,
            "device": str(device),
            "probe_iters": int(args.probe_iters),
            "reference_multiplier": float(args.reference_multiplier),
            "search_stall_timeout_s": float(args.search_stall_timeout_s),
            "stage1_positions": int(args.stage1_positions),
            "shortlist_topk": int(args.shortlist_topk),
            "stage2_games": int(args.stage2_games),
            "arena_iters": int(args.arena_iters)
            if args.arena_iters is not None
            else None,
            "checkpoints": checkpoints,
            "candidates": candidates,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_dump(base_dir / "sweep_manifest.json", manifest)
        json_dump(base_dir / "stage1_positions.json", {"positions": positions})

    if args.resume_report:
        stage1_path = base_dir / "stage1_surrogate.json"
        if stage1_path.exists():
            resume_stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        else:
            resume_stage1 = {}
        stage1_payload, missing_stage1 = normalize_stage1_payload(
            resume_stage1,
            candidates,
            checkpoints,
            positions,
            args.probe_iters,
            args.reference_multiplier,
            args.shortlist_topk,
        )
        candidate_by_id = {row["id"]: row for row in candidates}
        for contract in missing_stage1:
            candidate = candidate_by_id.get(contract["candidate_id"])
            checkpoint_path = contract["checkpoint_path"]
            if candidate is None:
                continue
            print(f"  RESUME-STAGE1 {Path(checkpoint_path).name}: {candidate['id']}")
            row = probe_candidate_on_positions(
                checkpoint_path,
                base_cfg,
                device,
                positions,
                candidate,
                args.rust_binary,
                args.probe_iters,
                args.reference_multiplier,
            )
            row.update(contract)
            stage1_payload["rows"].append(row)
            stage1_payload["summary"] = summarize_stage1_results(
                candidates, stage1_payload["rows"]
            )
            stage1_payload["shortlist"] = select_stage2_candidates(
                candidates, stage1_payload["summary"], args.shortlist_topk
            )
            stage1_payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            attach_stage1_contract_summary(stage1_payload)
            json_dump(base_dir / "stage1_surrogate.json", stage1_payload)
        shortlisted = stage1_payload.get("shortlist") or resume_shortlist or []
        selected_ids = parse_selected_candidate_ids(
            args.candidate_ids,
            [row["id"] for row in shortlisted],
        )
        shortlisted = [row for row in shortlisted if row["id"] in selected_ids]
    else:
        stage1_rows = []
        expected_stage1_contracts = build_stage1_expected_contracts(
            checkpoints,
            candidates,
            positions,
            args.probe_iters,
            args.reference_multiplier,
        )
        for checkpoint_path in checkpoints:
            print(f"\n{'=' * 76}")
            print(f"STAGE1 checkpoint={checkpoint_path}")
            print(f"{'=' * 76}")
            for candidate in candidates:
                print(f"  PROBE {candidate['id']}: {candidate['label']}")
                row = probe_candidate_on_positions(
                    checkpoint_path,
                    base_cfg,
                    device,
                    positions,
                    candidate,
                    args.rust_binary,
                    args.probe_iters,
                    args.reference_multiplier,
                )
                row.update(
                    expected_stage1_contracts[
                        (str(checkpoint_path), str(candidate["id"]))
                    ]
                )
                stage1_rows.append(row)
                payload = {
                    "rows": stage1_rows,
                    "summary": summarize_stage1_results(candidates, stage1_rows),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "discarded_rows": [],
                    "expected_probe_contracts": list(
                        expected_stage1_contracts.values()
                    ),
                }
                attach_stage1_contract_summary(payload)
                json_dump(base_dir / "stage1_surrogate.json", payload)

        stage1_payload = {
            "rows": stage1_rows,
            "summary": summarize_stage1_results(candidates, stage1_rows),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "discarded_rows": [],
            "expected_probe_contracts": list(expected_stage1_contracts.values()),
        }
        attach_stage1_contract_summary(stage1_payload)
        json_dump(base_dir / "stage1_surrogate.json", stage1_payload)
        shortlisted = select_stage2_candidates(
            candidates, stage1_payload["summary"], args.shortlist_topk
        )
        stage1_payload["shortlist"] = shortlisted
        attach_stage1_contract_summary(stage1_payload)
        json_dump(base_dir / "stage1_surrogate.json", stage1_payload)

        print(f"\n{'=' * 76}")
        print("STAGE1 SHORTLIST")
        print(f"{'=' * 76}")
        for row in shortlisted:
            print(f"  {row['id']:<16} {row['label']}")

    stage2_payload = None
    if not args.skip_stage2:
        stage2_payload = run_stage2_round_robin(
            shortlisted, checkpoints, base_cfg, device, args, base_dir
        )

    manifest["contract_summary"] = build_controller_sweep_contract_summary(
        stage1_payload, stage2_payload
    )
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(base_dir / "sweep_manifest.json", manifest)
    report = build_report(base_dir, manifest, stage1_payload, stage2_payload)
    recommended = report.get("recommended")
    if recommended:
        print(f"\nRecommended: {recommended.get('candidate_id')}")
        print(
            f"  score={recommended.get('score_rate', recommended.get('stage1_score'))}"
        )
        if recommended.get("candidate_label"):
            print(f"  {recommended['candidate_label']}")
    print(f"\nReport saved: {base_dir / 'sweep_report.json'}")


if __name__ == "__main__":
    main()
