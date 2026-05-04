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
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quartz.contract_summary import (
    stable_json_hash,
    summarize_named_contract_map,
    summarize_plain_contracts,
)
from quartz.evaluation import score_rate_ci
from quartz.eval_timing_summary import summarize_ablation_eval_timings
from quartz.eval_runtime_profile import load_eval_runtime_overrides_from_model

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

CONTROLLER_AXES_TRAIN_CONDITIONS = {
    "A1_legacy_tree_norefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
        "prior_refresh_rate": 0.0,
    },
    "A2_legacy_root_norefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
    },
    "A3_theory_root_norefresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
    },
    "A4_theory_root_refresh": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.5,
        "prior_refresh_temp": 0.0,
    },
}

CONTROLLER_AXES_EVAL_CONDITIONS = {
    "EA1_legacy_tree_norefresh": copy.deepcopy(CONTROLLER_AXES_TRAIN_CONDITIONS["A1_legacy_tree_norefresh"]),
    "EA2_legacy_root_norefresh": copy.deepcopy(CONTROLLER_AXES_TRAIN_CONDITIONS["A2_legacy_root_norefresh"]),
    "EA3_theory_root_norefresh": copy.deepcopy(CONTROLLER_AXES_TRAIN_CONDITIONS["A3_theory_root_norefresh"]),
    "EA4_theory_root_refresh": copy.deepcopy(CONTROLLER_AXES_TRAIN_CONDITIONS["A4_theory_root_refresh"]),
}

# Q4 (audit_codex_20260428.md W'4): the existing controller_axes /
# controller_factorial presets pin `halt_mode = "fixed"` to make
# penalty/refresh attribution clean. That solution disables every adaptive
# halt branch, which means VOC and SimpleThreshold cannot themselves be
# studied at fixed NN/eval/visit-cap. The HALT_ATTRIBUTION_* presets fill
# that gap: every row holds penalty_mode, root_only_shaping, and
# prior_refresh constant — only `halt_mode` varies across adjacent rows.
# Compare per-row `mean_root_visits_at_halt` (from replay halt_trace) to
# see how much compute each halt mode saves at equal arena strength.
HALT_ATTRIBUTION_TRAIN_CONDITIONS = {
    "H1_voc_default": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
        "halt_mode": "voc",
    },
    "H2_simple_threshold": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
        "halt_mode": "simple_threshold",
    },
    "H3_fixed_full_budget": {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": True,
        "prior_refresh_rate": 0.0,
        "halt_mode": "fixed",
    },
}

HALT_ATTRIBUTION_EVAL_CONDITIONS = {
    "EH1_voc_default": copy.deepcopy(HALT_ATTRIBUTION_TRAIN_CONDITIONS["H1_voc_default"]),
    "EH2_simple_threshold": copy.deepcopy(HALT_ATTRIBUTION_TRAIN_CONDITIONS["H2_simple_threshold"]),
    "EH3_fixed_full_budget": copy.deepcopy(HALT_ATTRIBUTION_TRAIN_CONDITIONS["H3_fixed_full_budget"]),
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
    "controller_axes": {
        "train_conditions": CONTROLLER_AXES_TRAIN_CONDITIONS,
        "eval_conditions": CONTROLLER_AXES_EVAL_CONDITIONS,
    },
    # Q4: see HALT_ATTRIBUTION_TRAIN_CONDITIONS for design rationale.
    "halt_attribution": {
        "train_conditions": HALT_ATTRIBUTION_TRAIN_CONDITIONS,
        "eval_conditions": HALT_ATTRIBUTION_EVAL_CONDITIONS,
    },
}

# Presets whose design goal is per-factor controller attribution.
# For these presets the arena comparison is only credible when the effective
# per-move search budget is uniform across modes. Because the default halt
# policy on the Rust side is p_flip-mediated (VOC), penalty-mode choice can
# silently change the halt step, which leaks budget into the mode-level
# comparison. `emit_attribution_halt_guard()` below warns at study launch
# and tags each attribution eval manifest so downstream analysis can verify
# budget-fairness via the `halt_trace` block now emitted in replay
# search_summary (see quartz/replay.py _finalize_halt_trace).
CONTROLLER_ATTRIBUTION_PRESETS = frozenset({"controller_axes", "controller_factorial"})

# Q4: presets that hold controller penalty/refresh fixed and vary
# `halt_mode` itself across rows. They want the same frozen-eval-condition
# discipline as the controller_attribution presets, but they MUST NOT have
# `halt_mode` silently overridden to "fixed" — that is the variable being
# studied. `pin_halt_mode_for_attribution` already uses `setdefault`, so an
# explicit `halt_mode` in each condition cfg survives the pin step.
HALT_ATTRIBUTION_PRESETS = frozenset({"halt_attribution"})


def emit_attribution_halt_guard(preset_name: str, stream=None) -> None:
    """Warn at study launch when a controller-attribution preset is used.

    The warning is intentionally loud and points the user at the `halt_trace`
    diagnostic in each evaluation_matrix row so they can audit same-budget
    fairness empirically, not assume it.
    """
    if preset_name not in CONTROLLER_ATTRIBUTION_PRESETS:
        return
    if stream is None:
        stream = sys.stderr
    print(
        f"[ablation_study] attribution preset `{preset_name}` selected.\n"
        f"                 Controller attribution requires same-budget fairness across\n"
        f"                 penalty modes. The default Rust halt policy is p_flip-mediated\n"
        f"                 and can leak budget into the mode comparison (see audit W1).\n"
        f"                 The evaluation_matrix will now include a `halt_trace` block\n"
        f"                 per eval_condition (replay search_summary). After the run,\n"
        f"                 inspect halt_trace.root_visits.mean across modes:\n"
        f"                   * >5%% relative spread -> budget leakage; re-run with an\n"
        f"                     explicit fixed `--iters N` for all conditions, or pin\n"
        f"                     halt_mode=fixed on the Rust side.\n"
        f"                   * <5%% relative spread -> attribution is budget-fair.\n",
        file=stream,
        flush=True,
    )


def resolve_frozen_eval_condition(
    args: argparse.Namespace, eval_conditions: dict
) -> str | None:
    """Resolve the `--frozen-eval-condition` choice into a concrete name or None.

    P8 (audit_codex_20260425.md W7): for attribution presets the
    eval-engine drift across rows confounds (model quality) ×
    (eval search profile). When this returns a non-None name, the
    eval loop pins every pair to that single eval condition's cfg.

    Resolution rules:
      * `--no-frozen-eval` set                → None (explicit opt-out).
      * `--frozen-eval-condition NAME` set    → NAME (must exist in eval_conditions).
      * attribution preset and neither flag set → first eval condition (sorted).
      * non-attribution preset and neither flag set → None (legacy per-row matrix).
    """
    if getattr(args, "no_frozen_eval", False):
        return None
    explicit = getattr(args, "frozen_eval_condition", None)
    if explicit:
        if explicit not in eval_conditions:
            raise SystemExit(
                f"--frozen-eval-condition '{explicit}' not in eval_conditions "
                f"({sorted(eval_conditions)})"
            )
        return explicit
    study = getattr(args, "study", None)
    # Q4: halt_attribution presets also need a frozen eval condition so the
    # comparison varies only `halt_mode`, not the eval engine itself.
    if (
        study in CONTROLLER_ATTRIBUTION_PRESETS
        or study in HALT_ATTRIBUTION_PRESETS
    ) and eval_conditions:
        return sorted(eval_conditions)[0]
    return None


def attribution_preset_tag(preset_name: str) -> dict:
    """Return a small metadata blob identifying attribution guard status.

    Stamped into study_manifest.json so readers and CI gates can tell whether
    the run intended per-factor attribution and therefore should be audited
    against the halt_trace fairness check.

    Q4: extended to surface `halt_axis_preset` for halt_attribution
    presets so downstream tooling can distinguish "controller-axis study"
    (halt pinned to Fixed) from "halt-axis study" (halt is the variable).
    """
    is_controller_attribution = preset_name in CONTROLLER_ATTRIBUTION_PRESETS
    is_halt_attribution = preset_name in HALT_ATTRIBUTION_PRESETS
    if is_controller_attribution:
        check = (
            "inspect evaluation_matrix[*].replay_search_summary.halt_trace "
            "for equal root_visits.mean across penalty modes"
        )
    elif is_halt_attribution:
        check = (
            "inspect evaluation_matrix[*].replay_search_summary.halt_trace "
            "for divergent root_visits.mean across halt modes; verify "
            "score-rate at equal NN/eval/visit-cap to attribute compute "
            "savings of each halt mode (audit_codex_20260428.md W'4)"
        )
    else:
        check = "n/a (not an attribution preset)"
    return {
        "attribution_preset": bool(is_controller_attribution),
        "halt_axis_preset": bool(is_halt_attribution),
        "preset": preset_name,
        "halt_fairness_check": check,
    }


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_ablation_contracts(expected_manifests: dict | None, discarded_matches: list[dict] | None) -> dict:
    return summarize_named_contract_map(
        expected_manifests,
        discarded_matches,
        name_key="eval_condition",
    )


def attach_ablation_contract_summary(payload: dict) -> dict:
    payload["contract_summary"] = summarize_ablation_contracts(
        payload.get("expected_search_manifests"),
        payload.get("discarded_matches"),
    )
    return payload


def build_runtime_contract(args: argparse.Namespace) -> dict:
    rust_binary = getattr(args, "rust_binary", "./target/release/mcts_demo")
    rust_binary_path = Path(rust_binary).expanduser()
    rust_binary_abs = str(rust_binary_path.resolve()) if rust_binary_path.exists() else str(rust_binary_path)
    return {
        "backend": getattr(args, "backend", "auto"),
        "device": getattr(args, "device", "auto"),
        "rust_binary": rust_binary,
        "rust_binary_abs": rust_binary_abs,
        "rust_binary_exists": bool(rust_binary_path.exists()),
        "rust_binary_sha256": sha256_file_prefix(rust_binary_path),
        "quick": bool(getattr(args, "quick", False)),
        "no_autotune": bool(getattr(args, "no_autotune", False)),
        "resident_session": bool(getattr(args, "resident_session", False)),
        "runtime_autotune": bool(getattr(args, "runtime_autotune", False)),
        "no_pipeline": bool(getattr(args, "no_pipeline", False)),
        "allow_unsafe_benchmark": bool(getattr(args, "allow_unsafe_benchmark", False)),
        "eval_seed": getattr(args, "eval_seed", None),
        "paired_seed_eval": bool(getattr(args, "paired_seed_eval", False)),
        "include_strict_reference": bool(getattr(args, "include_strict_reference", False)),
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "config_layout": "repo_top_level_configs",
    }


def build_training_contract(
    args: argparse.Namespace,
    condition_name: str,
    condition_cfg: dict,
    seed: int,
) -> dict:
    resolved_games = args.games_per_iter
    if resolved_games is None and args.quick:
        resolved_games = 50
    runtime_contract = build_runtime_contract(args)
    return {
        "condition": condition_name,
        "game": args.game,
        "iterations": int(args.iterations),
        "seed": int(seed),
        "train_cfg": copy.deepcopy(condition_cfg),
        "effective_search_config": effective_search_config_view(condition_cfg),
        "requested_effective_delta": config_delta(
            controller_surface(condition_cfg),
            effective_search_config_view(condition_cfg),
        ),
        "games_per_iter": int(resolved_games) if resolved_games is not None else None,
        "eval_interval": int(args.eval_interval),
        "eval_games": int(args.eval_games),
        "backend": args.backend,
        "device": args.device,
        "rust_binary": args.rust_binary,
        "no_autotune": bool(args.no_autotune),
        "resident_session": bool(args.resident_session),
        "runtime_autotune": bool(args.runtime_autotune),
        "concurrent": not bool(getattr(args, "no_pipeline", False)),
        "runtime_contract": runtime_contract,
        "runtime_contract_hash": stable_json_hash(runtime_contract),
    }


def summarize_training_contracts(contracts: list[dict] | None) -> dict:
    return summarize_plain_contracts(contracts)


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


def sha256_file_prefix(path: str | Path | None, prefix_len: int = 16) -> str | None:
    if not path:
        return None
    try:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:prefix_len]
    except Exception:
        return None


def sha256_checkpoint(path: str | Path | None) -> str | None:
    """P02: full 64-char SHA256 of a model checkpoint file.

    Distinct from `sha256_file_prefix` (which truncates to 16 chars for log
    readability). Pre-flight uses the full digest so cross-checkpoint
    fingerprinting and corruption detection are robust against birthday
    collisions across the ~10^5 checkpoints a long campaign may produce.
    Returns `None` on missing/unreadable files; the caller decides whether
    to fail or skip.
    """
    if not path:
        return None
    try:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


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


def pin_halt_mode_for_attribution(preset: dict, study_name: str) -> dict:
    """P7 (audit_codex_20260425.md W2): for attribution presets, stamp
    `halt_mode = "fixed"` into every train and eval condition cfg.

    `HaltMode::Fixed { budget = u32::MAX }` (Rust side, parsed in
    `mcts_server.parse_halt_mode_override`) disables every adaptive
    halt branch — P_flip / VOC / ConfAdaptive — so the only halt
    signal is the controller's own `max_visits` ceiling. This makes
    same-budget fairness trivially observable across rows that vary
    penalty mode, prior refresh, etc.

    The original cfg objects are deep-copied so module-level constants
    are not mutated. Returns a new preset dict with the same keys.
    """
    if study_name not in CONTROLLER_ATTRIBUTION_PRESETS:
        return preset
    pinned = {}
    for bucket_name in ("train_conditions", "eval_conditions"):
        bucket = preset.get(bucket_name)
        if bucket is None:
            continue
        new_bucket = {}
        for cond_name, cond_cfg in bucket.items():
            cfg = copy.deepcopy(cond_cfg)
            cfg.setdefault("halt_mode", "fixed")
            new_bucket[cond_name] = cfg
        pinned[bucket_name] = new_bucket
    # Preserve any non-condition keys the preset may carry.
    for key, value in preset.items():
        pinned.setdefault(key, value)
    return pinned


def condition_runtime_overrides(condition_cfg: dict) -> dict:
    return {
        key: value
        for key, value in condition_cfg.items()
        if key not in CLI_CONDITION_KEYS
    }


def condition_run_dir(base_dir: Path, condition_name: str, seed: int, multi_seed: bool) -> Path:
    root = base_dir / "models" / condition_name
    return root / f"seed_{seed}" if multi_seed else root


def controller_surface(condition_cfg: dict | None) -> dict:
    cfg = condition_cfg or {}
    return {
        "search_profile": cfg.get("search_profile"),
        "vl_mode": cfg.get("vl_mode"),
        "penalty_mode": cfg.get("penalty_mode"),
        "root_only_shaping": cfg.get("root_only_shaping"),
        "prior_refresh_rate": cfg.get("prior_refresh_rate"),
        "prior_refresh_temp": cfg.get("prior_refresh_temp"),
    }


def effective_search_config_view(condition_cfg: dict | None) -> dict:
    """Normalize the requested controller/search surface into wire-visible values.

    This intentionally preserves explicit zeroes such as
    `prior_refresh_temp: 0.0`; those are meaningful ablation settings and
    must not be collapsed by Python truthiness before they reach Rust.
    """
    cfg = condition_cfg or {}
    return {
        "search_profile": cfg.get("search_profile", "quartz"),
        "vl_mode": cfg.get("vl_mode", "disabled"),
        "penalty_mode": cfg.get("penalty_mode", "GatedRefresh"),
        "halt_mode": cfg.get("halt_mode"),
        "root_only_shaping": cfg.get("root_only_shaping"),
        "prior_refresh_rate": 0.0 if cfg.get("prior_refresh_rate") is None else cfg.get("prior_refresh_rate"),
        "prior_refresh_temp": 1.0 if cfg.get("prior_refresh_temp") is None else cfg.get("prior_refresh_temp"),
        "c_puct": cfg.get("c_puct"),
        "sigma_0": cfg.get("sigma_0"),
        "min_visits": cfg.get("min_visits"),
        "check_interval": cfg.get("check_interval"),
        "hbar_penalty_cap": cfg.get("hbar_penalty_cap"),
        "n_threads": cfg.get("n_threads"),
        "batch_size": cfg.get("batch_size"),
        "batch_timeout_us": cfg.get("batch_timeout_us"),
        "eval_seed": cfg.get("eval_seed"),
    }


def resolve_eval_seed(
    args: argparse.Namespace,
    eval_name: str | None = None,
    eval_cfg: dict | None = None,
) -> int:
    cfg = eval_cfg or {}
    if cfg.get("eval_seed") is not None:
        return int(cfg["eval_seed"])
    if getattr(args, "eval_seed", None) is not None:
        return int(args.eval_seed)
    # Legacy compatibility: older eval configs occasionally used `seed`.
    # Keep reading it, but publish the normalized value as `eval_seed`.
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    return 0


def apply_eval_seed_contract(
    cfg: dict,
    args: argparse.Namespace,
    eval_name: str | None = None,
    eval_cfg: dict | None = None,
) -> int:
    eval_seed = resolve_eval_seed(args, eval_name, eval_cfg)
    cfg["eval_seed"] = int(eval_seed)
    return int(eval_seed)


def config_delta(requested: dict | None, effective: dict | None) -> dict:
    requested = requested or {}
    effective = effective or {}
    keys = sorted(set(requested) | set(effective))
    return {
        key: {"requested": requested.get(key), "effective": effective.get(key)}
        for key in keys
        if requested.get(key) != effective.get(key)
    }


def expected_benchmark_safe_from_cfg(cfg: dict | None) -> bool:
    cfg = cfg or {}
    profile = str(cfg.get("search_profile", "quartz"))
    runner_mode = str(cfg.get("_eval_runner_mode", ""))
    raw_threads = cfg.get("n_threads", 1)
    try:
        n_threads = int(raw_threads or 1)
    except (TypeError, ValueError):
        n_threads = (
            2
            if str(raw_threads).lower()
            in {"auto", "throughput", "auto-throughput", "quality", "auto-quality"}
            else 1
        )
    # Rust eval state-machine always uses BatchStdioEval/GlobalBroker. Baseline
    # strict and multi-threaded search also force the batch evaluator in Rust.
    return (
        runner_mode == "rust_eval_state_machine"
        or profile == "baseline_strict"
        or n_threads > 1
    )


# P5 (audit_codex_20260425.md W1): cross-paper portability of "the
# QUARTZ controller" requires that every artifact carries a fingerprint
# of *which* of the six dispatch surfaces was actually configured. The
# tuple below is the controller's identity surface — every key whose
# value materially changes the behaviour of select.rs's mode dispatch
# (`src/mcts/select.rs:175-431`) and quartz.rs's halt dispatch
# (`src/mcts/quartz.rs:1862-1959`).
def controller_identity_keys() -> tuple[str, ...]:
    return (
        "search_profile",
        "vl_mode",
        "penalty_mode",
        "halt_mode",
        "root_only_shaping",
        "prior_refresh_rate",
        "prior_refresh_temp",
        "sigma_0",
        "hbar_penalty_cap",
        "ctm_budget_ms",
        "min_visits",
        "check_interval",
        "enable_fisher_puct",
        "enable_one_loop",
    )


def controller_identity_hash(condition_cfg: dict | None) -> str:
    """Stable 16-hex SHA-256 over the controller-identity surface.

    The hash distinguishes any two controller variants that differ in
    *any* identity-defining field; reorderings or unrelated keys do
    not perturb it (handled via `stable_json_hash`'s `sort_keys=True`).
    """
    cfg = condition_cfg or {}
    payload = {key: cfg.get(key) for key in controller_identity_keys()}
    return stable_json_hash(payload)


def controller_identity_hash_for_axes(
    condition_cfg: dict | None, axis_keys: tuple[str, ...] | list[str]
) -> str:
    """Hash all controller-identity fields *except* the named axes.

    Use this to verify single-axis isolation: every row of a
    single-axis preset should share the same value here, even though
    full `controller_identity_hash` values differ. Unknown axis names
    are tolerated.
    """
    cfg = condition_cfg or {}
    excluded = set(axis_keys)
    payload = {
        key: cfg.get(key)
        for key in controller_identity_keys()
        if key not in excluded
    }
    return stable_json_hash(payload)


def assert_single_axis_isolation(
    surfaces: dict, axis_keys: tuple[str, ...] | list[str]
) -> tuple[bool, dict[str, str]]:
    """Return (all_equal, hashes) for the given condition map.

    `surfaces` maps condition name → condition cfg. `axis_keys` are
    the fields the preset declares it varies. Result `all_equal=True`
    iff every condition produces the same `controller_identity_hash_for_axes`,
    i.e., the rows differ only in the declared axes.
    """
    hashes = {
        name: controller_identity_hash_for_axes(cfg, axis_keys)
        for name, cfg in surfaces.items()
    }
    return len(set(hashes.values())) <= 1, hashes


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


def resolve_model_path(run_dir: Path) -> Path | None:
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


def collect_training_metrics(run_dir: Path) -> dict:
    metrics = {
        "published_elo": None,
        "loss": None,
        "p_loss": None,
        "v_loss": None,
        "loss_ema": None,
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
            if row.get("p_loss") is not None:
                metrics["p_loss"] = row.get("p_loss")
            if row.get("v_loss") is not None:
                metrics["v_loss"] = row.get("v_loss")
            if row.get("loss_ema") is not None:
                metrics["loss_ema"] = row.get("loss_ema")
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


def _iter_train_log_rows(run_dir: Path):
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


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
            train_contract = meta.get("train_contract")
            if train_contract is None:
                train_contract = {
                    "condition": meta.get("condition", condition_dir.name),
                    "game": meta.get("game"),
                    "seed": seed,
                    "iterations": meta.get("iterations"),
                    "train_cfg": meta.get("train_cfg", {}),
                    "legacy_partial": True,
                }
            runs.append({
                "id": run_id,
                "condition": meta.get("condition", condition_dir.name),
                "seed": seed,
                "game": meta.get("game"),
                "train_cfg": meta.get("train_cfg", {}),
                "controller_surface": controller_surface(meta.get("train_cfg", {})),
                "train_contract": train_contract,
                "train_contract_hash": meta.get("train_contract_hash") or stable_json_hash(train_contract),
                "run_dir": str(run_dir),
                "elapsed_s": meta.get("elapsed_s", 0),
                "returncode": meta.get("returncode"),
                "success": meta.get("returncode") == 0,
                "model_path": str(model_path) if model_path is not None else None,
                "metrics": metrics,
            })
    return runs


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def summarize_pipeline_telemetry(runs: list[dict]) -> dict:
    """Summarize train_log pipeline health fields across discovered runs."""
    run_summaries = []
    all_freshness = []
    all_pos_per_s = []
    all_worker_queue_latency = []
    all_worker_rolling_pps = []
    row_count = 0
    freshness_rows = 0
    throughput_rows = 0
    worker_rows = 0
    concurrent_runs = 0
    total_new_pos = 0
    total_train_steps = 0
    max_backpressure_waits = 0
    total_inference_items = 0
    total_inference_messages = 0
    total_inference_model_calls = 0
    for run in runs:
        train_contract = run.get("train_contract") or {}
        concurrent = bool(train_contract.get("concurrent", False))
        if concurrent:
            concurrent_runs += 1
        run_dir = Path(run.get("run_dir") or "")
        rows = [row for row in _iter_train_log_rows(run_dir) if row.get("_type") != "eval"]
        run_freshness = []
        run_pos_per_s = []
        run_worker_queue_latency = []
        run_worker_rolling_pps = []
        run_worker_rows = 0
        run_new_pos = 0
        run_train_steps = 0
        run_backpressure_waits = 0
        run_inference_items = 0
        run_inference_messages = 0
        run_inference_model_calls = 0
        for row in rows:
            row_count += 1
            if row.get("new_pos") is not None:
                run_new_pos += int(row.get("new_pos") or 0)
            if row.get("train_steps") is not None:
                run_train_steps += int(row.get("train_steps") or 0)
            if row.get("replay_freshness") is not None:
                freshness = float(row["replay_freshness"])
                run_freshness.append(freshness)
                all_freshness.append(freshness)
                freshness_rows += 1
            if row.get("pos_per_s") is not None:
                pps = float(row["pos_per_s"])
                run_pos_per_s.append(pps)
                all_pos_per_s.append(pps)
                throughput_rows += 1
            telemetry = row.get("selfplay_telemetry") or {}
            if isinstance(telemetry, dict) and telemetry:
                worker_rows += 1
                run_worker_rows += 1
                if telemetry.get("last_progress_age_s") is not None:
                    queue_latency = float(telemetry.get("last_progress_age_s") or 0.0)
                    run_worker_queue_latency.append(queue_latency)
                    all_worker_queue_latency.append(queue_latency)
                if telemetry.get("rolling_positions_per_s") is not None:
                    rolling_pps = float(telemetry.get("rolling_positions_per_s") or 0.0)
                    run_worker_rolling_pps.append(rolling_pps)
                    all_worker_rolling_pps.append(rolling_pps)
                run_backpressure_waits = max(
                    run_backpressure_waits,
                    int(telemetry.get("backpressure_waits") or 0),
                )
                inference = telemetry.get("inference") or {}
                if isinstance(inference, dict):
                    run_inference_items += int(inference.get("eval_items") or 0)
                    run_inference_messages += int(inference.get("eval_messages") or 0)
                    run_inference_model_calls += int(inference.get("model_calls") or 0)
        total_new_pos += run_new_pos
        total_train_steps += run_train_steps
        max_backpressure_waits = max(max_backpressure_waits, run_backpressure_waits)
        total_inference_items += run_inference_items
        total_inference_messages += run_inference_messages
        total_inference_model_calls += run_inference_model_calls
        run_summaries.append(
            {
                "id": run.get("id"),
                "condition": run.get("condition"),
                "seed": run.get("seed"),
                "concurrent": concurrent,
                "rows": len(rows),
                "freshness_rows": len(run_freshness),
                "throughput_rows": len(run_pos_per_s),
                "worker_telemetry_rows": run_worker_rows,
                "freshness": _numeric_summary(run_freshness),
                "pos_per_s": _numeric_summary(run_pos_per_s),
                "selfplay_queue_latency_s": _numeric_summary(run_worker_queue_latency),
                "worker_rolling_positions_per_s": _numeric_summary(run_worker_rolling_pps),
                "new_pos_sum": int(run_new_pos),
                "train_steps_sum": int(run_train_steps),
                "backpressure_waits_max": int(run_backpressure_waits),
                "inference_eval_items": int(run_inference_items),
                "inference_eval_messages": int(run_inference_messages),
                "inference_model_calls": int(run_inference_model_calls),
            }
        )
    aggregate = {
        "run_count": len(runs),
        "concurrent_run_count": int(concurrent_runs),
        "row_count": int(row_count),
        "freshness_coverage_frac": float(freshness_rows / row_count) if row_count else None,
        "throughput_coverage_frac": float(throughput_rows / row_count) if row_count else None,
        "worker_telemetry_coverage_frac": float(worker_rows / row_count) if row_count else None,
        "freshness": _numeric_summary(all_freshness),
        "pos_per_s": _numeric_summary(all_pos_per_s),
        "selfplay_queue_latency_s": _numeric_summary(all_worker_queue_latency),
        "worker_rolling_positions_per_s": _numeric_summary(all_worker_rolling_pps),
        "new_pos_sum": int(total_new_pos),
        "train_steps_sum": int(total_train_steps),
        "backpressure_waits_max": int(max_backpressure_waits),
        "inference_eval_items": int(total_inference_items),
        "inference_eval_messages": int(total_inference_messages),
        "inference_model_calls": int(total_inference_model_calls),
    }
    return {
        "schema_version": 1,
        "runs": run_summaries,
        "aggregate": aggregate,
    }


def summarize_seed_protocol(runs: list[dict], eval_payload: dict | None = None) -> dict:
    seeds_by_condition: dict[str, set] = {}
    id_to_seed = {}
    id_to_condition = {}
    for run in runs:
        condition = str(run.get("condition") or run.get("id") or "unknown")
        seed = run.get("seed")
        seeds_by_condition.setdefault(condition, set())
        if seed is not None:
            seeds_by_condition[condition].add(int(seed))
        if run.get("id") is not None:
            id_to_seed[str(run["id"])] = int(seed) if seed is not None else None
            id_to_condition[str(run["id"])] = condition
    condition_seed_lists = {
        condition: sorted(seeds)
        for condition, seeds in sorted(seeds_by_condition.items())
    }
    seed_sets = [set(seeds) for seeds in seeds_by_condition.values()]
    common_seeds = set.intersection(*seed_sets) if seed_sets else set()
    union_seeds = set.union(*seed_sets) if seed_sets else set()
    min_seed_count = min((len(seeds) for seeds in seed_sets), default=0)
    seed_sets_aligned = bool(seed_sets) and all(seeds == seed_sets[0] for seeds in seed_sets)

    eval_matches = list((eval_payload or {}).get("matches") or [])
    known_eval_pairs = 0
    same_seed_eval_pairs = 0
    cross_condition_eval_pairs = 0
    for match in eval_matches:
        a_id = str(match.get("a_id"))
        b_id = str(match.get("b_id"))
        if a_id not in id_to_seed or b_id not in id_to_seed:
            continue
        seed_a = id_to_seed.get(a_id)
        seed_b = id_to_seed.get(b_id)
        if seed_a is None or seed_b is None:
            continue
        known_eval_pairs += 1
        if id_to_condition.get(a_id) != id_to_condition.get(b_id):
            cross_condition_eval_pairs += 1
        if seed_a == seed_b:
            same_seed_eval_pairs += 1
    paired_seed_eval_requested = bool(((eval_payload or {}).get("runtime_contract") or {}).get("paired_seed_eval"))
    same_seed_pair_frac = float(same_seed_eval_pairs / known_eval_pairs) if known_eval_pairs else None
    paired_seed_claim_ready = (
        len(seeds_by_condition) <= 1
        or (
            len(common_seeds) >= 3
            and bool(seed_sets_aligned)
            and bool(paired_seed_eval_requested)
            and cross_condition_eval_pairs > 0
            and same_seed_pair_frac is not None
            and same_seed_pair_frac >= 1.0
        )
    )
    return {
        "schema_version": 1,
        "condition_count": len(seeds_by_condition),
        "conditions": condition_seed_lists,
        "min_seed_count": int(min_seed_count),
        "common_seeds": sorted(common_seeds),
        "common_seed_count": int(len(common_seeds)),
        "union_seed_count": int(len(union_seeds)),
        "seed_sets_aligned": bool(seed_sets_aligned),
        "common_seed_coverage_frac": (
            float(len(common_seeds) / len(union_seeds)) if union_seeds else None
        ),
        "paired_seed_eval_requested": paired_seed_eval_requested,
        "eval_pairs": {
            "known_seed_pairs": int(known_eval_pairs),
            "same_seed_pairs": int(same_seed_eval_pairs),
            "cross_condition_pairs": int(cross_condition_eval_pairs),
            "same_seed_pair_frac": same_seed_pair_frac,
        },
        "paired_seed_claim_ready": paired_seed_claim_ready,
    }


def summarize_evaluation_protocol(runs: list[dict], eval_payload: dict | None = None) -> dict:
    """Summarize whether eval rows support same-protocol comparison claims.

    This is passive report metadata. It does not decide who won; it records
    whether the artifact exposes enough protocol information to interpret eval
    conditions as same-model-pair, same-runtime, same-game-distribution rows.
    """
    payload = eval_payload or {}
    matches = list(payload.get("matches") or [])
    eval_conditions = sorted(
        {str(row.get("eval_condition") or "unknown") for row in matches}
    )
    eval_condition_set = set(eval_conditions)
    expected_eval_seeds = payload.get("expected_eval_seeds") or {}
    expected_benchmark = payload.get("expected_benchmark_safe") or {}
    runtime_contract = payload.get("runtime_contract") or {}
    runtime_contract_hash = payload.get("runtime_contract_hash")

    def _sorted_values(values) -> list:
        return sorted(values, key=lambda item: (str(type(item)), repr(item)))

    runner_mode_counts: dict[str, int] = {}
    search_hashes_by_condition: dict[str, set[str]] = {
        condition: set() for condition in eval_conditions
    }
    pairs_by_condition: dict[str, set[tuple[str, str]]] = {
        condition: set() for condition in eval_conditions
    }
    pair_condition_coverage: dict[tuple[str, str], set[str]] = {}
    model_ids = set()
    games_per_row = set()
    scored_games_per_row = set()
    rows_with_search_manifest_hash = 0
    rows_with_pair_ids = 0
    rows_with_runner_mode = 0

    for row in matches:
        condition = str(row.get("eval_condition") or "unknown")
        runner_mode = row.get("runner_mode")
        if runner_mode is not None:
            rows_with_runner_mode += 1
            key = str(runner_mode)
            runner_mode_counts[key] = int(runner_mode_counts.get(key, 0)) + 1

        manifest_hash = row.get("search_manifest_hash")
        if manifest_hash:
            rows_with_search_manifest_hash += 1
            search_hashes_by_condition.setdefault(condition, set()).add(str(manifest_hash))

        if row.get("games") is not None:
            games_per_row.add(int(row.get("games") or 0))
        if row.get("scored_games") is not None:
            scored_games_per_row.add(int(row.get("scored_games") or 0))

        a_id = row.get("a_id")
        b_id = row.get("b_id")
        if a_id is None or b_id is None:
            continue
        rows_with_pair_ids += 1
        pair = tuple(sorted((str(a_id), str(b_id))))
        model_ids.update(pair)
        pairs_by_condition.setdefault(condition, set()).add(pair)
        pair_condition_coverage.setdefault(pair, set()).add(condition)

    expected_seed_conditions = {str(key) for key in expected_eval_seeds.keys()}
    expected_benchmark_conditions = {str(key) for key in expected_benchmark.keys()}
    expected_seed_values = _sorted_values(set(expected_eval_seeds.values()))
    one_manifest_per_eval_condition = bool(matches) and all(
        len(search_hashes_by_condition.get(condition, set())) == 1
        for condition in eval_conditions
    )
    expected_eval_seed_coverage = bool(eval_condition_set) and eval_condition_set.issubset(
        expected_seed_conditions
    )
    expected_benchmark_coverage = bool(eval_condition_set) and eval_condition_set.issubset(
        expected_benchmark_conditions
    )
    complete_pair_eval_matrix = bool(matches) and bool(pair_condition_coverage) and all(
        coverage == eval_condition_set for coverage in pair_condition_coverage.values()
    )
    game_count_consistent = bool(matches) and len(games_per_row) <= 1
    eval_seed_consistent = bool(expected_eval_seeds) and len(set(expected_eval_seeds.values())) <= 1
    benchmark_safe_all_expected = (
        bool(expected_benchmark)
        and expected_benchmark_coverage
        and all(bool(value) for value in expected_benchmark.values())
    )
    protocol_ready = (
        bool(matches)
        and bool(runtime_contract_hash)
        and bool(runtime_contract)
        and rows_with_pair_ids == len(matches)
        and rows_with_search_manifest_hash == len(matches)
        and rows_with_runner_mode == len(matches)
        and expected_eval_seed_coverage
        and eval_seed_consistent
        and benchmark_safe_all_expected
        and game_count_consistent
        and one_manifest_per_eval_condition
        and complete_pair_eval_matrix
    )

    return {
        "schema_version": 1,
        "match_count": int(len(matches)),
        "eval_condition_count": int(len(eval_conditions)),
        "eval_conditions": eval_conditions,
        "model_count": int(len(model_ids)),
        "pair_count": int(len(pair_condition_coverage)),
        "runtime_contract_hash": runtime_contract_hash,
        "runtime_contract_present": bool(runtime_contract),
        "paired_seed_eval_requested": bool(runtime_contract.get("paired_seed_eval")),
        "expected_eval_seed_values": expected_seed_values,
        "expected_eval_seed_coverage": bool(expected_eval_seed_coverage),
        "eval_seed_consistent": bool(eval_seed_consistent),
        "runner_mode_counts": dict(sorted(runner_mode_counts.items())),
        "runner_mode_coverage_frac": (
            float(rows_with_runner_mode / len(matches)) if matches else None
        ),
        "expected_benchmark_safe": expected_benchmark,
        "expected_benchmark_coverage": bool(expected_benchmark_coverage),
        "benchmark_safe_all_expected": bool(benchmark_safe_all_expected),
        "games_per_row": sorted(games_per_row),
        "scored_games_per_row": sorted(scored_games_per_row),
        "game_count_consistent": bool(game_count_consistent),
        "search_manifest_hashes_by_eval_condition": {
            condition: sorted(search_hashes_by_condition.get(condition, set()))
            for condition in eval_conditions
        },
        "search_manifest_hash_coverage_frac": (
            float(rows_with_search_manifest_hash / len(matches)) if matches else None
        ),
        "one_manifest_per_eval_condition": bool(one_manifest_per_eval_condition),
        "pairs_by_eval_condition": {
            condition: int(len(pairs_by_condition.get(condition, set())))
            for condition in eval_conditions
        },
        "pair_eval_condition_coverage": {
            f"{pair[0]}||{pair[1]}": sorted(coverage)
            for pair, coverage in sorted(pair_condition_coverage.items())
        },
        "pair_id_coverage_frac": (
            float(rows_with_pair_ids / len(matches)) if matches else None
        ),
        "complete_pair_eval_matrix": bool(complete_pair_eval_matrix),
        "protocol_ready": bool(protocol_ready),
    }


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _loss_quality_bucket(loss: float | None) -> str:
    if loss is None:
        return "loss_unknown"
    if loss < 1.0:
        return "loss_lt_1_0"
    if loss < 1.5:
        return "loss_1_0_to_1_5"
    return "loss_ge_1_5"


def _loss_pair_stratum(bucket_a: str, bucket_b: str) -> str:
    if bucket_a == "loss_unknown" or bucket_b == "loss_unknown":
        return "loss_unknown"
    if bucket_a == bucket_b:
        return f"both_{bucket_a}"
    return "__".join(sorted((bucket_a, bucket_b)))


def summarize_evaluator_quality_strata(runs: list[dict], eval_payload: dict | None = None) -> dict:
    """Group evaluation rows by available NN/evaluator quality proxies.

    The runner does not currently run a held-out value/policy calibration set.
    This summary therefore uses only artifact-native proxies: latest training
    loss components and ladder/arena signals already written in train logs. If
    those proxies are absent, the report says so explicitly instead of implying
    evaluator-quality robustness.
    """
    payload = eval_payload or {}
    matches = list(payload.get("matches") or [])
    model_quality = {}
    loss_bucket_counts: dict[str, int] = {}
    proxy_models = 0
    loss_models = 0
    elo_models = 0
    score_models = 0

    for run in runs:
        model_id = run.get("id")
        if model_id is None:
            continue
        metrics = run.get("metrics") or {}
        loss = _float_or_none(metrics.get("loss"))
        p_loss = _float_or_none(metrics.get("p_loss"))
        v_loss = _float_or_none(metrics.get("v_loss"))
        loss_ema = _float_or_none(metrics.get("loss_ema"))
        published_elo = _float_or_none(metrics.get("published_elo"))
        score_rate = _float_or_none(metrics.get("score_rate"))
        bucket = _loss_quality_bucket(loss)
        loss_bucket_counts[bucket] = int(loss_bucket_counts.get(bucket, 0)) + 1
        has_loss = loss is not None
        has_elo = published_elo is not None
        has_score = score_rate is not None
        has_proxy = bool(has_loss or has_elo or has_score)
        proxy_models += int(has_proxy)
        loss_models += int(has_loss)
        elo_models += int(has_elo)
        score_models += int(has_score)
        model_quality[str(model_id)] = {
            "id": str(model_id),
            "condition": run.get("condition"),
            "seed": run.get("seed"),
            "loss": loss,
            "p_loss": p_loss,
            "v_loss": v_loss,
            "loss_ema": loss_ema,
            "loss_bucket": bucket,
            "published_elo": published_elo,
            "score_rate": score_rate,
            "games_done": int((metrics.get("games_done") or 0)),
            "has_quality_proxy": has_proxy,
            "quality_proxy_sources": [
                name
                for name, present in (
                    ("loss", has_loss),
                    ("published_elo", has_elo),
                    ("score_rate", has_score),
                )
                if present
            ],
        }

    strata: dict[str, dict] = {}
    match_count = len(matches)
    rows_with_known_models = 0
    rows_with_quality_proxy_pair = 0
    rows_with_loss_pair = 0
    rows_with_elo_pair = 0
    rows_with_score_pair = 0
    rows_with_missing_quality = 0
    missing_model_ids = set()

    for match in matches:
        a_id = str(match.get("a_id"))
        b_id = str(match.get("b_id"))
        qa = model_quality.get(a_id)
        qb = model_quality.get(b_id)
        if qa is None:
            missing_model_ids.add(a_id)
        if qb is None:
            missing_model_ids.add(b_id)
        if qa is None or qb is None:
            rows_with_missing_quality += 1
            stratum = "quality_unknown"
        else:
            rows_with_known_models += 1
            has_proxy_pair = bool(qa["has_quality_proxy"] and qb["has_quality_proxy"])
            has_loss_pair = qa["loss"] is not None and qb["loss"] is not None
            has_elo_pair = qa["published_elo"] is not None and qb["published_elo"] is not None
            has_score_pair = qa["score_rate"] is not None and qb["score_rate"] is not None
            rows_with_quality_proxy_pair += int(has_proxy_pair)
            rows_with_loss_pair += int(has_loss_pair)
            rows_with_elo_pair += int(has_elo_pair)
            rows_with_score_pair += int(has_score_pair)
            if has_loss_pair:
                stratum = _loss_pair_stratum(qa["loss_bucket"], qb["loss_bucket"])
            elif has_elo_pair:
                delta = abs(float(qa["published_elo"]) - float(qb["published_elo"]))
                if delta <= 50.0:
                    stratum = "elo_delta_le_50"
                elif delta <= 150.0:
                    stratum = "elo_delta_50_to_150"
                else:
                    stratum = "elo_delta_gt_150"
            elif has_score_pair:
                stratum = "score_rate_proxy_only"
            elif has_proxy_pair:
                stratum = "mixed_quality_proxy"
            else:
                rows_with_missing_quality += 1
                stratum = "quality_unknown"

        eval_name = str(match.get("eval_condition") or "unknown")
        games = int(match.get("scored_games") or match.get("games") or 0)
        score_rate_a = _float_or_none(match.get("score_rate_a"))
        row = strata.setdefault(
            stratum,
            {
                "stratum": stratum,
                "matches": 0,
                "games": 0,
                "eval_conditions": {},
                "_score_rate_a_values": [],
            },
        )
        row["matches"] += 1
        row["games"] += games
        if score_rate_a is not None:
            row["_score_rate_a_values"].append(score_rate_a)
        cond = row["eval_conditions"].setdefault(
            eval_name,
            {
                "matches": 0,
                "games": 0,
                "_score_rate_a_values": [],
            },
        )
        cond["matches"] += 1
        cond["games"] += games
        if score_rate_a is not None:
            cond["_score_rate_a_values"].append(score_rate_a)

    stratum_rows = []
    for row in strata.values():
        score_values = row.pop("_score_rate_a_values")
        eval_conditions = {}
        for name, cond in sorted(row["eval_conditions"].items()):
            cond_values = cond.pop("_score_rate_a_values")
            eval_conditions[name] = {
                **cond,
                "score_rate_a": _numeric_summary(cond_values),
            }
        stratum_rows.append(
            {
                **row,
                "score_rate_a": _numeric_summary(score_values),
                "eval_conditions": eval_conditions,
            }
        )
    stratum_rows.sort(key=lambda item: item["stratum"])
    quality_proxy_pair_coverage = (
        float(rows_with_quality_proxy_pair / match_count) if match_count else None
    )
    loss_pair_coverage = float(rows_with_loss_pair / match_count) if match_count else None
    stratification_ready = (
        bool(matches)
        and rows_with_known_models == match_count
        and rows_with_quality_proxy_pair == match_count
        and bool(stratum_rows)
    )
    return {
        "schema_version": 1,
        "model_count": int(len(model_quality)),
        "models_with_quality_proxy": int(proxy_models),
        "models_with_loss": int(loss_models),
        "models_with_published_elo": int(elo_models),
        "models_with_score_rate": int(score_models),
        "loss_bucket_counts": dict(sorted(loss_bucket_counts.items())),
        "model_quality": {
            key: model_quality[key] for key in sorted(model_quality)
        },
        "match_count": int(match_count),
        "rows_with_known_models": int(rows_with_known_models),
        "rows_with_quality_proxy_pair": int(rows_with_quality_proxy_pair),
        "rows_with_loss_pair": int(rows_with_loss_pair),
        "rows_with_published_elo_pair": int(rows_with_elo_pair),
        "rows_with_score_rate_pair": int(rows_with_score_pair),
        "rows_with_missing_quality": int(rows_with_missing_quality),
        "missing_model_ids": sorted(missing_model_ids),
        "quality_proxy_pair_coverage_frac": quality_proxy_pair_coverage,
        "loss_pair_coverage_frac": loss_pair_coverage,
        "published_elo_pair_coverage_frac": (
            float(rows_with_elo_pair / match_count) if match_count else None
        ),
        "score_rate_pair_coverage_frac": (
            float(rows_with_score_pair / match_count) if match_count else None
        ),
        "strata": stratum_rows,
        "strata_count": int(len(stratum_rows)),
        "stratification_ready": bool(stratification_ready),
        "claim_guidance": (
            "Evaluator-quality robustness claims require quality_proxy_pair_coverage_frac=1.0 "
            "and should prefer loss_pair_coverage_frac=1.0 or a held-out calibration artifact."
        ),
    }


def summarize_heldout_calibration(base_dir: Path, runs: list[dict]) -> dict:
    """Validate optional held-out evaluator calibration artifact coverage.

    Expected schema:
      {
        "models": {
          "cond_s42": {
            "n_positions": 128,
            "policy_nll": 2.1,
            "value_mse": 0.18,
            "top1_acc": 0.42,
            "brier": 0.21
          }
        }
      }
    """
    path = base_dir / "evaluator_calibration.json"
    required_ids = {
        str(run.get("id"))
        for run in runs
        if run.get("id") is not None and run.get("model_path")
    }
    if not path.exists():
        return {
            "schema_version": 1,
            "artifact_present": False,
            "artifact_path": str(path),
            "model_count": int(len(required_ids)),
            "covered_model_count": 0,
            "coverage_frac": None if not required_ids else 0.0,
            "calibration_ready": False,
            "missing_model_ids": sorted(required_ids),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": 1,
            "artifact_present": True,
            "artifact_path": str(path),
            "calibration_ready": False,
            "error": str(exc),
            "missing_model_ids": sorted(required_ids),
        }
    raw_models = payload.get("models") or {}
    if isinstance(raw_models, list):
        models = {str(row.get("id")): row for row in raw_models if row.get("id") is not None}
    else:
        models = {str(key): value for key, value in dict(raw_models).items()}
    metric_keys = ("n_positions", "policy_nll", "value_mse", "top1_acc", "brier")
    covered = []
    incomplete = {}
    for model_id in sorted(required_ids):
        row = models.get(model_id)
        if not isinstance(row, dict):
            continue
        missing = [key for key in metric_keys if row.get(key) is None]
        if missing or int(row.get("n_positions") or 0) <= 0:
            incomplete[model_id] = missing or ["n_positions"]
            continue
        covered.append(model_id)
    missing_ids = sorted(required_ids - set(covered))
    return {
        "schema_version": 1,
        "artifact_present": True,
        "artifact_path": str(path),
        "model_count": int(len(required_ids)),
        "covered_model_count": int(len(covered)),
        "coverage_frac": (
            float(len(covered) / len(required_ids))
            if required_ids
            else None
        ),
        "calibration_ready": bool(required_ids and not missing_ids),
        "missing_model_ids": missing_ids,
        "incomplete_model_metrics": incomplete,
        "metrics": {
            model_id: {key: models[model_id].get(key) for key in metric_keys}
            for model_id in covered
        },
    }


def find_profiler_artifacts(base_dir: Path, limit: int = 20) -> list[str]:
    """Find explicit profiler/throughput artifacts without assuming CUDA/ROCm."""
    if not base_dir.exists():
        return []
    names = []
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "autotune_profile.json":
            continue
        is_profile = (
            "throughput_profile" in name
            or name.startswith("rocprof")
            or name.startswith("nsys")
            or name.startswith("ncu")
            or name.startswith("perf")
            or name.endswith(".pstats")
            or ("profile" in name and name.endswith((".json", ".txt", ".html", ".svg")))
        )
        if is_profile:
            names.append(str(path.relative_to(base_dir)))
            if len(names) >= int(limit):
                break
    return sorted(names)


def find_runtime_tuning_artifacts(base_dir: Path, limit: int = 20) -> list[str]:
    if not base_dir.exists():
        return []
    names = []
    for path in base_dir.rglob("autotune_profile.json"):
        if path.is_file():
            names.append(str(path.relative_to(base_dir)))
            if len(names) >= int(limit):
                break
    return sorted(names)


def summarize_hardware_runtime(
    base_dir: Path,
    runs: list[dict],
    eval_payload: dict | None,
    pipeline_telemetry_summary: dict | None,
) -> dict:
    """Backend-neutral runtime/hardware claim scope summary.

    This intentionally does not require ROCm/CUDA profiler integration. It
    records what was requested, what throughput was observed, and whether an
    explicit profiler artifact exists. Without such an artifact the report can
    still discuss observed runtime telemetry, but should not make hardware
    optimization claims.
    """
    requested_backends = set()
    requested_devices = set()
    runtime_contracts = []
    eval_runtime = (eval_payload or {}).get("runtime_contract") or {}
    if eval_runtime:
        runtime_contracts.append({"source": "evaluation_matrix", "runtime_contract": eval_runtime})
    for run in runs:
        train_contract = run.get("train_contract") or {}
        runtime_contract = train_contract.get("runtime_contract") or {}
        if runtime_contract:
            runtime_contracts.append({"source": run.get("id"), "runtime_contract": runtime_contract})
        else:
            runtime_contract = {
                "backend": train_contract.get("backend"),
                "device": train_contract.get("device"),
            }
        if runtime_contract.get("backend") is not None:
            requested_backends.add(str(runtime_contract.get("backend")))
        if runtime_contract.get("device") is not None:
            requested_devices.add(str(runtime_contract.get("device")))
    if eval_runtime.get("backend") is not None:
        requested_backends.add(str(eval_runtime.get("backend")))
    if eval_runtime.get("device") is not None:
        requested_devices.add(str(eval_runtime.get("device")))

    profiler_artifacts = find_profiler_artifacts(base_dir)
    tuning_artifacts = find_runtime_tuning_artifacts(base_dir)
    pipeline_aggregate = (pipeline_telemetry_summary or {}).get("aggregate") or {}
    observed_runtime_telemetry = {
        "pipeline_rows": int(pipeline_aggregate.get("row_count") or 0),
        "pipeline_pos_per_s_mean": (pipeline_aggregate.get("pos_per_s") or {}).get("mean"),
        "pipeline_freshness_mean": (pipeline_aggregate.get("freshness") or {}).get("mean"),
        "worker_rolling_positions_per_s_mean": (
            pipeline_aggregate.get("worker_rolling_positions_per_s") or {}
        ).get("mean"),
        "inference_eval_items": int(pipeline_aggregate.get("inference_eval_items") or 0),
        "inference_model_calls": int(pipeline_aggregate.get("inference_model_calls") or 0),
    }
    has_profiler = bool(profiler_artifacts)
    has_runtime_telemetry = any(
        value not in (None, 0, 0.0)
        for value in observed_runtime_telemetry.values()
    )
    claim_scope = (
        "hardware_profiled"
        if has_profiler
        else ("runtime_telemetry_only" if has_runtime_telemetry else "requested_runtime_only")
    )
    return {
        "schema_version": 1,
        "requested_backends": sorted(requested_backends),
        "requested_devices": sorted(requested_devices),
        "runtime_contracts": runtime_contracts,
        "observed_runtime_telemetry": observed_runtime_telemetry,
        "profiler_artifact_present": has_profiler,
        "profiler_artifacts": profiler_artifacts,
        "runtime_tuning_artifact_present": bool(tuning_artifacts),
        "runtime_tuning_artifacts": tuning_artifacts,
        "claim_scope": claim_scope,
        "hardware_performance_claims_allowed": bool(has_profiler),
        "claim_guidance": (
            "Hardware optimization claims require profiler_artifact_present=true. "
            "Without it, report only observed throughput/latency telemetry."
        ),
    }


def build_study_manifest(args: argparse.Namespace) -> dict:
    preset = pin_halt_mode_for_attribution(resolve_study_preset(args.study), args.study)
    train_conditions = preset["train_conditions"]
    eval_conditions = preset["eval_conditions"]
    selected_train_conditions = parse_selected_conditions(args.conditions, train_conditions)
    selected_eval_conditions = parse_selected_conditions(args.eval_conditions, eval_conditions)
    from quartz import runtime_support as support_mod

    unknown_keys = {}
    for bucket_name, bucket, selected in (
        ("train", train_conditions, selected_train_conditions),
        ("eval", eval_conditions, selected_eval_conditions),
    ):
        for name in selected:
            unknown = support_mod.validate_search_option_keys(
                bucket[name],
                context=f"{bucket_name} condition {name}",
            )
            if unknown:
                unknown_keys[f"{bucket_name}:{name}"] = unknown
    if unknown_keys:
        raise SystemExit(
            "unknown search-option keys in study conditions: "
            + json.dumps(unknown_keys, sort_keys=True)
        )
    runtime_contract = build_runtime_contract(args)
    manifest = {
        "format_version": 2,
        "study": args.study,
        "game": args.game,
        "iterations": args.iterations,
        "eval_games": args.eval_games,
        "eval_seed": getattr(args, "eval_seed", None),
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
        "train_condition_surfaces": {
            name: controller_surface(train_conditions[name]) for name in selected_train_conditions
        },
        "eval_condition_surfaces": {
            name: controller_surface(eval_conditions[name]) for name in selected_eval_conditions
        },
        "train_condition_effective_search_config": {
            name: effective_search_config_view(train_conditions[name])
            for name in selected_train_conditions
        },
        "eval_condition_effective_search_config": {
            name: effective_search_config_view(eval_conditions[name])
            for name in selected_eval_conditions
        },
        # P5 (audit W1): per-row controller identity fingerprint so
        # cross-paper readers can tell which of the six dispatch
        # surfaces produced an Elo curve. See `controller_identity_keys`.
        "train_condition_identity_hashes": {
            name: controller_identity_hash(train_conditions[name])
            for name in selected_train_conditions
        },
        "eval_condition_identity_hashes": {
            name: controller_identity_hash(eval_conditions[name])
            for name in selected_eval_conditions
        },
        "strict_reference": bool(args.include_strict_reference),
        "git_head": git_head(),
        "python": sys.executable,
        "runtime_contract": runtime_contract,
        "runtime_contract_hash": stable_json_hash(runtime_contract),
        "search_options_schema_version": support_mod.SEARCH_OPTIONS_SCHEMA_VERSION,
        "search_options_keys": list(support_mod.SEARCH_RUNTIME_KEYS),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attribution_guard": attribution_preset_tag(args.study),
        # P8 (audit W7): record the frozen-eval resolution so post-hoc
        # readers can confirm which single eval engine produced the
        # arena results when the matrix was collapsed.
        "frozen_eval_condition": resolve_frozen_eval_condition(
            args,
            {name: eval_conditions[name] for name in selected_eval_conditions},
        ),
    }
    return manifest


def compute_match_statistics(wins_a: int, wins_b: int, draws: int) -> tuple[float, list[float], str, dict]:
    total = int(wins_a) + int(wins_b) + int(draws)
    score_rate, ci = score_rate_ci(int(wins_a), int(draws), total)
    decisive = int(wins_a) + int(wins_b)
    p0 = 0.50
    p1 = 0.55
    alpha = 0.05
    beta = 0.05
    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)
    llr = 0.0
    sprt_result = "inconclusive"
    if decisive > 0:
        llr = float(
            wins_a * math.log(p1 / p0)
            + (decisive - wins_a) * math.log((1.0 - p1) / (1.0 - p0))
        )
        if llr >= upper_bound:
            sprt_result = "H1_accept"
        elif llr <= lower_bound:
            sprt_result = "H0_accept"
    sprt_meta = {
        "model": "wald_binomial_v1",
        "p0": p0,
        "p1": p1,
        "alpha": alpha,
        "beta": beta,
        "decisive_games": decisive,
        "llr": round(llr, 6),
        "lower_bound": round(lower_bound, 6),
        "upper_bound": round(upper_bound, 6),
    }
    return float(score_rate), [float(ci[0]), float(ci[1])], sprt_result, sprt_meta


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
    if getattr(args, "no_pipeline", False):
        cmd.append("--no-pipeline")
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
            expected_train_contract = build_training_contract(args, condition_name, condition_cfg, seed)
            expected_train_contract_hash = stable_json_hash(expected_train_contract)
            if previous.get("train_contract_hash") == expected_train_contract_hash:
                return {
                    "condition": condition_name,
                    "seed": seed,
                    "run_dir": str(output_dir),
                    "elapsed_s": previous.get("elapsed_s", 0),
                    "success": True,
                    "skipped": True,
                }

    train_contract = build_training_contract(args, condition_name, condition_cfg, seed)
    meta = {
        "condition": condition_name,
        "run_id": f"{condition_name}_s{seed}" if multi_seed else condition_name,
        "game": args.game,
        "iterations": args.iterations,
        "seed": seed,
        "train_cfg": condition_cfg,
        "train_contract": train_contract,
        "train_contract_hash": stable_json_hash(train_contract),
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
    runtime_overrides = load_eval_runtime_overrides_from_model(model_path, resolved_device)
    if runtime_overrides:
        cfg = apply_config_overrides(cfg, runtime_overrides)
    try:
        from quartz import runtime_support as support_mod

        if support_mod.supports_rust_eval_state_machine(game_name):
            cfg.setdefault("_eval_runner_mode", "rust_eval_state_machine")
    except Exception:
        pass
    return cfg, torch.device(resolved_device)


def build_eval_engine(model_run: dict, args: argparse.Namespace, eval_cfg: dict, device) -> tuple[object, dict]:
    from quartz import runtime_support as support_mod
    from quartz.evaluator_runtime import RustNNEvaluatorEngine

    engine_cfg, _ = build_eval_cfg(args.game, eval_cfg, args.device, model_path=model_run["model_path"])
    apply_eval_seed_contract(engine_cfg, args, None, eval_cfg)
    actor = support_mod.load_actor_source_from_checkpoint(
        model_run["model_path"],
        engine_cfg,
        device,
        backend_preference=getattr(args, "backend", "auto"),
    )
    engine = RustNNEvaluatorEngine(
        model_run["id"],
        engine_cfg,
        actor,
        device,
        args.rust_binary,
    )
    return engine, engine_cfg


def build_opening_book(game_name: str) -> list[list[int]]:
    if game_name.startswith("go") or game_name.startswith("chess"):
        return []
    return [[]]


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


def _across_seed_summary(values: list[float]) -> dict:
    """Mean + std + 95% CI (normal approximation) for across-seed aggregates.

    Uses the normal approximation for simplicity; the CI is only meaningful
    when n >= 2, and wide for small n. Readers should inspect `n` before
    drawing conclusions.
    """
    if not values:
        return {"n": 0, "mean": None, "std": None, "sem": None, "ci95": None}
    n = len(values)
    mean = float(sum(values) / n)
    if n < 2:
        return {"n": n, "mean": mean, "std": None, "sem": None, "ci95": None}
    # Sample standard deviation (Bessel's correction).
    var = sum((float(v) - mean) ** 2 for v in values) / (n - 1)
    std = var ** 0.5
    sem = std / (n ** 0.5)
    # 1.96 × SEM is the 95% normal-approx CI half-width; cf. score_rate_ci.
    half = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "std": float(std),
        "sem": float(sem),
        "ci95": [float(mean - half), float(mean + half)],
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
                "values_elo": [],
                "values_score_rate": [],
                "values_loss": [],
            },
        )
        row["runs"] += 1
        elo = metrics.get("published_elo")
        if elo is not None:
            row["values_elo"].append(float(elo))
        score_rate = metrics.get("score_rate")
        if score_rate is not None:
            row["values_score_rate"].append(float(score_rate))
        loss = metrics.get("loss")
        if loss is not None:
            row["values_loss"].append(float(loss))

    training_rows = []
    for row in training.values():
        elo_summary = _across_seed_summary(row["values_elo"])
        score_summary = _across_seed_summary(row["values_score_rate"])
        loss_summary = _across_seed_summary(row["values_loss"])
        training_rows.append(
            {
                "condition": row["condition"],
                "runs": row["runs"],
                # Backwards-compatible mean-only fields kept so existing readers
                # (tests, reports) continue to parse. The new `*_ci` block
                # carries the across-seed std / SEM / 95% CI.
                "mean_elo": elo_summary["mean"],
                "mean_score_rate": score_summary["mean"],
                "mean_loss": loss_summary["mean"],
                "elo_ci": elo_summary,
                "score_rate_ci": score_summary,
                "loss_ci": loss_summary,
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


def summarize_selection_trace_contract(eval_payload: dict | None) -> dict:
    """Aggregate exact root-selection telemetry from evaluation_matrix rows."""
    rows = {}
    for match in (eval_payload or {}).get("matches", []):
        eval_name = match.get("eval_condition") or "unknown"
        realized_budget = match.get("realized_budget_trace") or {}
        trace = (realized_budget.get("selection_trace") or {})
        row = rows.setdefault(
            eval_name,
            {
                "eval_condition": eval_name,
                "matches": 0,
                "games": 0,
                "root_selects": 0,
                "refresh_selected_count": 0,
                "selected_penalty_abs_sum": 0.0,
                "selected_effective_prior_l1_sum": 0.0,
                "_coverage_games": 0,
                "_coverage_weighted_sum": 0.0,
            },
        )
        row["matches"] += 1
        games = int(realized_budget.get("games") or match.get("games") or 0)
        row["games"] += games
        coverage = realized_budget.get("selection_trace_coverage_frac")
        if coverage is not None and games > 0:
            row["_coverage_games"] += games
            row["_coverage_weighted_sum"] += float(coverage) * games
        row["root_selects"] += int(trace.get("root_selects") or 0)
        row["refresh_selected_count"] += int(trace.get("refresh_selected_count") or 0)
        row["selected_penalty_abs_sum"] += float(trace.get("selected_penalty_abs_sum") or 0.0)
        row["selected_effective_prior_l1_sum"] += float(
            trace.get("selected_effective_prior_l1_sum") or 0.0
        )
    out = []
    for row in rows.values():
        root_selects = row["root_selects"]
        coverage_games = int(row.pop("_coverage_games", 0))
        coverage_weighted_sum = float(row.pop("_coverage_weighted_sum", 0.0))
        item = dict(row)
        item["selection_trace_coverage_frac"] = (
            float(coverage_weighted_sum / coverage_games) if coverage_games else None
        )
        item["refresh_selected_frac"] = (
            float(row["refresh_selected_count"] / root_selects) if root_selects else None
        )
        item["mean_penalty_abs_per_root_select"] = (
            float(row["selected_penalty_abs_sum"] / root_selects) if root_selects else None
        )
        item["mean_prior_l1_per_root_select"] = (
            float(row["selected_effective_prior_l1_sum"] / root_selects) if root_selects else None
        )
        out.append(item)
    out.sort(key=lambda item: item["eval_condition"])
    return {"conditions": out}


def summarize_budget_fairness(eval_payload: dict | None) -> dict:
    """Aggregate realized eval search budgets by eval condition.

    This does not force a fairness verdict. Adaptive halt studies may
    intentionally change realized root visits. The summary makes that drift
    explicit so controller comparisons can separate strength from compute.
    """
    rows = {}
    match_count = 0
    rows_with_trace = 0
    for match in (eval_payload or {}).get("matches", []):
        match_count += 1
        eval_name = match.get("eval_condition") or "unknown"
        budget = match.get("realized_budget_trace") or {}
        root_visits = budget.get("root_visits") or {}
        row = rows.setdefault(
            eval_name,
            {
                "eval_condition": eval_name,
                "matches": 0,
                "games": 0,
                "moves": 0,
                "halt_reason_hist": {},
                "_root_visit_samples": [],
                "_root_visit_weighted_sum": 0.0,
                "_root_visit_weight": 0,
                "_max_root_visit": None,
                "_benchmark_safe_weighted_sum": 0.0,
                "_benchmark_safe_weight": 0,
                "_telemetry_partial_weighted_sum": 0.0,
                "_telemetry_partial_weight": 0,
            },
        )
        row["matches"] += 1
        games = int(budget.get("games") or match.get("games") or 0)
        moves = int(budget.get("moves") or 0)
        row["games"] += games
        row["moves"] += moves
        samples = []
        for value in root_visits.get("samples") or []:
            try:
                samples.append(float(value))
            except Exception:
                pass
        if samples:
            row["_root_visit_samples"].extend(samples)
            rows_with_trace += 1
        elif root_visits.get("mean") is not None and moves > 0:
            row["_root_visit_weighted_sum"] += float(root_visits.get("mean") or 0.0) * moves
            row["_root_visit_weight"] += moves
            rows_with_trace += 1
        if root_visits.get("max") is not None:
            rv_max = float(root_visits.get("max") or 0.0)
            row["_max_root_visit"] = rv_max if row["_max_root_visit"] is None else max(row["_max_root_visit"], rv_max)
        for reason, count in (budget.get("halt_reason_hist") or {}).items():
            key = str(reason)
            row["halt_reason_hist"][key] = int(row["halt_reason_hist"].get(key, 0)) + int(count or 0)
        if budget.get("benchmark_safe_frac") is not None and games > 0:
            row["_benchmark_safe_weighted_sum"] += float(budget["benchmark_safe_frac"]) * games
            row["_benchmark_safe_weight"] += games
        if budget.get("telemetry_partial_frac") is not None and games > 0:
            row["_telemetry_partial_weighted_sum"] += float(budget["telemetry_partial_frac"]) * games
            row["_telemetry_partial_weight"] += games

    out = []
    condition_means = []
    for row in rows.values():
        samples = row.pop("_root_visit_samples")
        weighted_sum = float(row.pop("_root_visit_weighted_sum"))
        weighted_n = int(row.pop("_root_visit_weight"))
        max_root_visit = row.pop("_max_root_visit")
        benchmark_sum = float(row.pop("_benchmark_safe_weighted_sum"))
        benchmark_n = int(row.pop("_benchmark_safe_weight"))
        telemetry_sum = float(row.pop("_telemetry_partial_weighted_sum"))
        telemetry_n = int(row.pop("_telemetry_partial_weight"))
        if samples:
            mean_root_visits = float(sum(samples) / len(samples))
            min_root_visits = float(min(samples))
            max_root_visits = float(max(samples))
            sample_count = len(samples)
        elif weighted_n > 0:
            mean_root_visits = float(weighted_sum / weighted_n)
            min_root_visits = None
            max_root_visits = float(max_root_visit) if max_root_visit is not None else None
            sample_count = 0
        else:
            mean_root_visits = None
            min_root_visits = None
            max_root_visits = float(max_root_visit) if max_root_visit is not None else None
            sample_count = 0
        if mean_root_visits is not None:
            condition_means.append(mean_root_visits)
        item = dict(row)
        item["root_visits"] = {
            "mean": mean_root_visits,
            "min": min_root_visits,
            "max": max_root_visits,
            "sample_count": int(sample_count),
        }
        item["benchmark_safe_frac"] = (
            float(benchmark_sum / benchmark_n) if benchmark_n else None
        )
        item["telemetry_partial_frac"] = (
            float(telemetry_sum / telemetry_n) if telemetry_n else None
        )
        out.append(item)
    out.sort(key=lambda item: item["eval_condition"])

    relative_spread = None
    fairness_flag = "unknown"
    if len(condition_means) >= 2:
        lo = min(condition_means)
        hi = max(condition_means)
        relative_spread = float((hi - lo) / max(hi, 1.0))
        fairness_flag = "ok" if relative_spread <= 0.05 else "drift"
    return {
        "schema_version": 1,
        "match_count": int(match_count),
        "rows_with_budget_trace": int(rows_with_trace),
        "budget_trace_coverage_frac": (
            float(rows_with_trace / match_count) if match_count else None
        ),
        "root_visit_mean_relative_spread": relative_spread,
        "budget_fairness_flag": fairness_flag,
        "conditions": out,
    }


def pre_flight_check(
    args: argparse.Namespace,
    eligible: list[dict],
    eval_conditions: dict[str, dict],
    expected_manifest_hashes: dict,
) -> dict:
    """P02: pre-flight gate — runs once before the eval matrix executes.

    Checks performed:

    1. **Checkpoint reachability + fingerprinting**: SHA256 of every
       eligible run's `model_path`. Missing or unreadable files become
       errors. Side-effect: each run dict gains a `candidate_hash` field
       so downstream eval rows can record exactly which model bytes they
       evaluated.
    2. **Manifest-hash presence**: every eval condition must have a
       non-null `search_manifest_hash`. A null/empty value means the
       eval runner config didn't produce a stable hash, which silently
       breaks cross-condition comparison.
    3. **Paired-seed cross-condition consistency** (only when
       `--paired-seed-eval` is set): for each seed shared across two or
       more conditions, the runs' `train_contract_hash` values are
       compared *grouped by condition*. Within a single condition,
       different seeds naturally produce different contract hashes (the
       seed is part of the contract). Across conditions but with the
       same seed, the contracts disagree by design (the condition
       differs). What this check catches is *unintentional* drift: e.g.
       two runs labeled `condition=A, seed=42` with different contract
       hashes, indicating the run was re-run with a different config
       under the same label.

    Returns a JSON-shaped result dict with `schema_version: 1`,
    `ok: bool`, `errors: list`, `skipped_pairs: list[run_id]`.
    On `--research-grade`, raises `SystemExit` if any error fires.
    """
    errors: list = []
    skipped: list = []

    # 1. checkpoint reachability + fingerprint
    for run in eligible:
        h = sha256_checkpoint(run.get("model_path"))
        if h is None:
            errors.append(
                {
                    "run_id": run.get("id"),
                    "reason": "checkpoint_missing_or_unreadable",
                    "model_path": run.get("model_path"),
                }
            )
            skipped.append(run.get("id"))
            run["candidate_hash"] = None
        else:
            run["candidate_hash"] = h

    # 2. manifest-hash presence
    for name, h in (expected_manifest_hashes or {}).items():
        if not h:
            errors.append(
                {
                    "eval_condition": name,
                    "reason": "search_manifest_hash_missing",
                }
            )

    # 3. paired-seed cross-condition consistency
    if getattr(args, "paired_seed_eval", False):
        # Index runs by (condition, seed). Within a label-pair, multiple
        # entries indicate a re-run; if their train_contract_hash values
        # disagree, the label is silently overloaded — we flag this as
        # an error so the user knows their `--paired-seed-eval` claim
        # would otherwise pair non-homogeneous models.
        per_label: dict = {}
        for run in eligible:
            key = (run.get("condition"), run.get("seed"))
            per_label.setdefault(key, []).append(run)
        for (cond, seed), runs in per_label.items():
            if len(runs) <= 1:
                continue
            hashes = {r.get("train_contract_hash") for r in runs}
            if len(hashes) > 1:
                errors.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "reason": "duplicate_label_with_drifted_contract",
                        "train_contract_hashes": sorted(h for h in hashes if h),
                    }
                )

    ok = not errors
    if not ok and getattr(args, "research_grade", False):
        msg = "; ".join(f"{e.get('reason')}={e.get('run_id') or e.get('eval_condition') or e.get('condition')}" for e in errors)
        raise SystemExit(f"pre_flight_check failed under --research-grade: {msg}")
    return {
        "schema_version": 1,
        "ok": ok,
        "errors": errors,
        "skipped_pairs": sorted({sid for sid in skipped if sid is not None}),
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

    eval_conditions = dict(eval_conditions)
    if args.include_strict_reference:
        eval_conditions = {**STRICT_REFERENCE_CONDITION, **eval_conditions}

    from quartz import runtime_support as support_mod

    # P8 (audit W7): when frozen-eval is active, collapse the eval matrix
    # to the single named condition so every pair runs through identical
    # search settings.
    frozen_eval_name = resolve_frozen_eval_condition(args, eval_conditions)
    if frozen_eval_name is not None:
        eval_conditions = {frozen_eval_name: eval_conditions[frozen_eval_name]}

    expected_manifest_hashes = {}
    expected_manifests = {}
    expected_effective_configs = {}
    expected_benchmark_safe = {}
    expected_eval_seeds = {}
    reference_model_path = eligible[0]["model_path"]
    eval_condition_timings = {}
    for eval_name, eval_cfg in eval_conditions.items():
        cfg, _ = build_eval_cfg(args.game, eval_cfg, args.device, model_path=reference_model_path)
        expected_eval_seeds[eval_name] = apply_eval_seed_contract(cfg, args, eval_name, eval_cfg)
        expected_manifest_hashes[eval_name] = support_mod.search_manifest_hash(cfg)
        expected_manifests[eval_name] = support_mod.build_search_manifest(cfg)
        expected_effective_configs[eval_name] = effective_search_config_view(cfg)
        expected_benchmark_safe[eval_name] = expected_benchmark_safe_from_cfg(cfg)
        allow_unsafe = getattr(args, "allow_unsafe_benchmark", None)
        if allow_unsafe is False and not expected_benchmark_safe[eval_name]:
            raise SystemExit(
                f"eval condition {eval_name!r} is not benchmark-safe under the effective runtime "
                f"contract ({expected_manifests[eval_name]}). Reconfigure the eval runner/search "
                f"settings or pass --allow-unsafe-benchmark for exploratory runs."
            )

    # P02: pre-flight gate. Runs once before any eval pair launches; under
    # `--research-grade` raises on any error so an entire campaign isn't
    # spent on missing checkpoints or drifted contracts.
    pre_flight_summary = pre_flight_check(args, eligible, eval_conditions, expected_manifest_hashes)
    pre_flight_skip = set(pre_flight_summary.get("skipped_pairs") or [])

    existing_path = base_dir / "evaluation_matrix.json"
    existing_matches = []
    discarded_matches = []
    if existing_path.exists() and not args.force_eval:
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            for row in existing.get("matches", []):
                if row.get("a_id") not in eligible_ids or row.get("b_id") not in eligible_ids:
                    continue
                eval_name = row.get("eval_condition")
                expected_hash = expected_manifest_hashes.get(eval_name)
                found_hash = row.get("search_manifest_hash")
                if expected_hash and found_hash != expected_hash:
                    discarded_matches.append(
                        {
                            "eval_condition": eval_name,
                            "a_id": row.get("a_id"),
                            "b_id": row.get("b_id"),
                            "reason": "search_manifest_hash_changed",
                            "expected_hash": expected_hash,
                            "found_hash": found_hash,
                        }
                    )
                    continue
                existing_matches.append(row)
        except Exception:
            existing_matches = []
            discarded_matches = []

    match_index = {
        (row["eval_condition"], row["a_id"], row["b_id"]): row
        for row in existing_matches
    }

    def should_compare(model_a: dict, model_b: dict) -> bool:
        # P02: skip pairs whose checkpoint failed pre-flight (missing /
        # unreadable). Without this guard the eval would crash mid-pair
        # and pollute evaluation_matrix.json with half-finished rows.
        if model_a.get("id") in pre_flight_skip or model_b.get("id") in pre_flight_skip:
            return False
        if not args.paired_seed_eval:
            return True
        seed_a = model_a.get("seed")
        seed_b = model_b.get("seed")
        if seed_a is None or seed_b is None or seed_a != seed_b:
            return False
        return model_a.get("condition") != model_b.get("condition")

    all_matches = list(existing_matches)
    for eval_name, eval_cfg in eval_conditions.items():
        pending_pairs = []
        needed_ids = set()
        for idx, model_a in enumerate(eligible):
            for model_b in eligible[idx + 1:]:
                if not should_compare(model_a, model_b):
                    continue
                key = (eval_name, model_a["id"], model_b["id"])
                if key in match_index and not args.force_eval:
                    continue
                pending_pairs.append((key, model_a, model_b))
                needed_ids.add(model_a["id"])
                needed_ids.add(model_b["id"])
        if not pending_pairs:
            continue

        cfg_t0 = time.perf_counter()
        cfg, device = build_eval_cfg(args.game, eval_cfg, args.device, model_path=reference_model_path)
        eval_seed = apply_eval_seed_contract(cfg, args, eval_name, eval_cfg)
        cfg_build_s = time.perf_counter() - cfg_t0

        engine_load_t0 = time.perf_counter()
        engines = {}
        for run in eligible:
            if run["id"] not in needed_ids:
                continue
            engine, _engine_cfg = build_eval_engine(run, args, eval_cfg, device)
            engines[run["id"]] = engine
        engine_load_s = time.perf_counter() - engine_load_t0

        opening_book = build_opening_book(args.game)
        board_size = int(cfg.get("board", 0) or 0)
        max_moves = 500 if support_mod.is_chess_game(args.game) else max(1, board_size) ** 2

        from quartz.evaluator_runtime import PersistentRustNNEvalCampaign

        campaign_t0 = time.perf_counter()
        try:
            with PersistentRustNNEvalCampaign(engines.values(), args.eval_games) as campaign:
                campaign_bootstrap_s = float(campaign.timings.get("client_start_s", 0.0) or 0.0)
                eval_condition_timings[eval_name] = {
                    "cfg_build_s": round(cfg_build_s, 6),
                    "engine_load_s": round(engine_load_s, 6),
                    "campaign_bootstrap_s": round(campaign_bootstrap_s, 6),
                    "pairs": len(pending_pairs),
                    "engine_count": len(engines),
                    "eval_seed": eval_seed,
                }
                search_manifest = support_mod.build_search_manifest(cfg)
                search_manifest_hash = support_mod.search_manifest_hash(cfg)

                def _build_row(
                    key,
                    model_a,
                    model_b,
                    tally,
                    timing_meta,
                    *,
                    _eval_name=eval_name,
                    _eval_cfg=eval_cfg,
                    _cfg_build_s=cfg_build_s,
                    _engine_load_s=engine_load_s,
                    _campaign_bootstrap_s=campaign_bootstrap_s,
                    _search_manifest=search_manifest,
                    _search_manifest_hash=search_manifest_hash,
                ):
                    scored_games = int(getattr(tally, "scored", 0))
                    if scored_games <= 0:
                        raise RuntimeError(
                            f"evaluation produced zero scored games for {_eval_name}: "
                            f"errors={getattr(tally, 'errors', 0)} voids={getattr(tally, 'voids', 0)} "
                            f"total={getattr(tally, 'total', 0)}"
                        )
                    wa = int(tally.wins)
                    wb = int(tally.losses)
                    draws = int(tally.draws)
                    score_rate, ci, sprt, sprt_meta = compute_match_statistics(wa, wb, draws)
                    timing_s = {
                        "cfg_build_s": round(_cfg_build_s, 6),
                        "engine_load_s": round(_engine_load_s, 6),
                        "campaign_bootstrap_s": round(_campaign_bootstrap_s, 6),
                        "match_elapsed_s": round(float(timing_meta.get("match_elapsed_s", 0.0) or 0.0), 6),
                    }
                    if timing_meta.get("batch_id"):
                        timing_s["batch_id"] = str(timing_meta.get("batch_id"))
                        timing_s["batch_elapsed_s"] = round(float(timing_meta.get("batch_elapsed_s", 0.0) or 0.0), 6)
                        timing_s["batch_total_games"] = int(timing_meta.get("batch_total_games", 0) or 0)
                    return {
                        "eval_condition": _eval_name,
                        "search_profile": _eval_cfg["search_profile"],
                        "vl_mode": _eval_cfg["vl_mode"],
                        "search_manifest": _search_manifest,
                        "search_manifest_hash": _search_manifest_hash,
                        "requested_eval_cfg": copy.deepcopy(_eval_cfg),
                        "effective_search_config": expected_effective_configs.get(_eval_name, {}),
                        "requested_effective_delta": config_delta(
                            effective_search_config_view(_eval_cfg),
                            expected_effective_configs.get(_eval_name, {}),
                        ),
                        "benchmark_safe_expected": bool(expected_benchmark_safe.get(_eval_name, False)),
                        "realized_budget_trace": timing_meta.get("realized_budget_trace", {}),
                        "a_id": model_a["id"],
                        "b_id": model_b["id"],
                        "games": int(args.eval_games),
                        "wins_a": wa,
                        "wins_b": wb,
                        "draws": draws,
                        "win_rate_a": float(getattr(tally, "score_rate", score_rate)),
                        "score_rate_a": score_rate,
                        "ci": ci,
                        "ci_kind": "score_rate_normal_approx_v1",
                        "sprt": sprt,
                        "sprt_meta": sprt_meta,
                        "errors": int(getattr(tally, "errors", 0)),
                        "voids": int(getattr(tally, "voids", 0)),
                        "scored_games": scored_games,
                        "timing_s": timing_s,
                        "runner_mode": timing_meta.get("runner_mode"),
                    }

                used_batched_compare = bool(getattr(campaign, "compare_many", None)) and len(pending_pairs) > 1
                if used_batched_compare:
                    print(
                        f"  EVAL {eval_name}: batching {len(pending_pairs)} pairings "
                        f"({len(pending_pairs) * int(args.eval_games)} games)"
                    )
                    match_id_map = {}
                    comparisons = []
                    for pair_idx, (key, model_a, model_b) in enumerate(pending_pairs):
                        match_id = f"pair{pair_idx:03d}"
                        match_id_map[match_id] = (key, model_a, model_b)
                        comparisons.append(
                            {
                                "match_id": match_id,
                                "engine_a": engines[model_a["id"]],
                                "engine_b": engines[model_b["id"]],
                                "game_factory": lambda cfg_ref=cfg: support_mod.build_training_game_adapter(dict(cfg_ref)),
                                "opening_book": opening_book,
                                "num_games": args.eval_games,
                                "color_swap": True,
                                "logger": None,
                                "max_moves": max_moves,
                                "seed": eval_seed,
                            }
                        )
                    for match_id, tally, timing_meta in campaign.compare_many(comparisons):
                        key, model_a, model_b = match_id_map[match_id]
                        row = _build_row(key, model_a, model_b, tally, timing_meta)
                        match_index[key] = row
                else:
                    for key, model_a, model_b in pending_pairs:
                        print(f"  EVAL {eval_name}: {model_a['id']} vs {model_b['id']} ({args.eval_games} games)")
                        tally, timing_meta = campaign.compare(
                            engines[model_a["id"]],
                            engines[model_b["id"]],
                            lambda cfg_ref=cfg: support_mod.build_training_game_adapter(dict(cfg_ref)),
                            opening_book,
                            args.eval_games,
                            color_swap=True,
                            logger=None,
                            max_moves=max_moves,
                            seed=eval_seed,
                        )
                        row = _build_row(key, model_a, model_b, tally, timing_meta)
                        match_index[key] = row

                all_matches = list(match_index.values())
                payload = aggregate_matches(eligible, all_matches)
                payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                payload["discarded_matches"] = discarded_matches
                payload["expected_search_manifests"] = expected_manifests
                payload["expected_effective_search_config"] = expected_effective_configs
                payload["expected_benchmark_safe"] = expected_benchmark_safe
                payload["expected_eval_seeds"] = expected_eval_seeds
                payload["runtime_contract"] = build_runtime_contract(args)
                payload["runtime_contract_hash"] = stable_json_hash(payload["runtime_contract"])
                payload["eval_condition_timings"] = eval_condition_timings
                payload["eval_timing_summary"] = summarize_ablation_eval_timings(payload)
                payload["pre_flight"] = pre_flight_summary
                attach_ablation_contract_summary(payload)
                json_dump(existing_path, payload)
        finally:
            for engine in engines.values():
                try:
                    engine.reset()
                except Exception:
                    pass

    payload = aggregate_matches(eligible, list(match_index.values()))
    payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload["discarded_matches"] = discarded_matches
    payload["expected_search_manifests"] = expected_manifests
    payload["expected_effective_search_config"] = expected_effective_configs
    payload["expected_benchmark_safe"] = expected_benchmark_safe
    payload["expected_eval_seeds"] = expected_eval_seeds
    payload["runtime_contract"] = build_runtime_contract(args)
    payload["runtime_contract_hash"] = stable_json_hash(payload["runtime_contract"])
    payload["eval_condition_timings"] = eval_condition_timings
    payload["eval_timing_summary"] = summarize_ablation_eval_timings(payload)
    payload["pre_flight"] = pre_flight_summary
    attach_ablation_contract_summary(payload)
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

    def best_eval_condition_for_model(model_id: str) -> tuple[str | None, dict | None, dict]:
        if not eval_payload:
            return None, None, {}
        candidates = []
        for eval_name, rows in (eval_payload.get("leaderboards") or {}).items():
            for rank, row in enumerate(rows):
                if row.get("id") != model_id:
                    continue
                candidates.append(
                    (
                        float(row.get("score_rate", 0.0) or 0.0),
                        float(row.get("win_rate", 0.0) or 0.0),
                        float(row.get("points", 0.0) or 0.0),
                        int(row.get("games", 0) or 0),
                        -int(rank),
                        str(eval_name),
                        row,
                    )
                )
        if not candidates:
            return None, None, {}
        candidates.sort(reverse=True)
        score_rate, win_rate, points, games, neg_rank, eval_name, row = candidates[0]
        cfg = eval_conditions.get(eval_name)
        if cfg is None:
            return None, None, {
                "deployment_eval_condition_missing": eval_name,
                "deployment_eval_condition_score_rate": score_rate,
                "deployment_eval_condition_win_rate": win_rate,
                "deployment_eval_condition_points": points,
                "deployment_eval_condition_games": games,
                "deployment_eval_condition_rank": -int(neg_rank),
            }
        return eval_name, copy.deepcopy(cfg), {
            "deployment_eval_condition": eval_name,
            "deployment_eval_condition_score_rate": score_rate,
            "deployment_eval_condition_win_rate": win_rate,
            "deployment_eval_condition_points": points,
            "deployment_eval_condition_games": games,
            "deployment_eval_condition_rank": -int(neg_rank),
            "deployment_eval_condition_row": copy.deepcopy(row),
        }

    if eval_payload and eval_payload.get("overall"):
        top = eval_payload["overall"][0]
        champion_run = by_id[top["id"]]
        deployment_condition, deployment_cfg, deployment_metrics = best_eval_condition_for_model(
            champion_run["id"]
        )
        if deployment_cfg is None:
            deployment_condition = None
            deployment_cfg = copy.deepcopy(champion_run.get("train_cfg") or {})
            deployment_cfg_source = "train_cfg"
        else:
            deployment_cfg_source = f"eval_condition:{deployment_condition}"
        selection_metrics = {
            "overall_score_rate": top["score_rate"],
            "overall_points": top["points"],
            "overall_games": top["games"],
            "overall_win_rate": top["win_rate"],
            **deployment_metrics,
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
        deployment_cfg_source = "train_cfg"
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
        "controller_surface": champion_run.get("controller_surface") or controller_surface(champion_run.get("train_cfg")),
        "training_metrics": champion_run["metrics"],
        "deployment_eval_condition": deployment_condition,
        "deployment_search_cfg": deployment_cfg,
        "deployment_cfg_source": deployment_cfg_source,
        "deployment_controller_surface": controller_surface(deployment_cfg),
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


def research_readiness_summary(
    runs: list[dict],
    eval_payload: dict | None,
    selection_trace_summary: dict | None,
    budget_fairness_summary: dict | None = None,
    seed_protocol_summary: dict | None = None,
    pipeline_telemetry_summary: dict | None = None,
    hardware_runtime_summary: dict | None = None,
    champion: dict | None = None,
    evaluation_protocol_summary: dict | None = None,
    evaluator_quality_summary: dict | None = None,
    heldout_calibration_summary: dict | None = None,
) -> dict:
    """Passive research-readiness checklist.

    This report object is passive by default. The CLI-level `--research-grade`
    gate can make missing criteria block execution after the report is written.
    """
    eval_payload = eval_payload or {}
    matches = list(eval_payload.get("matches") or [])
    discarded = list(eval_payload.get("discarded_matches") or [])
    expected_benchmark = eval_payload.get("expected_benchmark_safe") or {}
    expected_eval_seeds = eval_payload.get("expected_eval_seeds") or {}
    expected_manifests = eval_payload.get("expected_search_manifests") or {}
    selection_conditions = list((selection_trace_summary or {}).get("conditions") or [])
    budget_summary = budget_fairness_summary or {}
    seed_summary = seed_protocol_summary or {}
    evaluation_protocol = evaluation_protocol_summary or {}
    pipeline_aggregate = (pipeline_telemetry_summary or {}).get("aggregate") or {}
    hardware_summary = hardware_runtime_summary or {}
    evaluator_quality = evaluator_quality_summary or {}
    heldout_calibration = heldout_calibration_summary or {}
    selection_trace_ready_rows = [
        row
        for row in selection_conditions
        if int(row.get("root_selects") or 0) > 0
        and row.get("selection_trace_coverage_frac") is not None
        and float(row.get("selection_trace_coverage_frac") or 0.0) > 0.0
    ]
    concurrent_runs = int(pipeline_aggregate.get("concurrent_run_count") or 0)
    row_count = int(pipeline_aggregate.get("row_count") or 0)
    freshness_coverage = pipeline_aggregate.get("freshness_coverage_frac")
    throughput_coverage = pipeline_aggregate.get("throughput_coverage_frac")
    worker_coverage = pipeline_aggregate.get("worker_telemetry_coverage_frac")
    pipeline_telemetry_ready = (
        row_count > 0
        and freshness_coverage is not None
        and float(freshness_coverage) > 0.0
        and throughput_coverage is not None
        and float(throughput_coverage) > 0.0
        and (
            concurrent_runs == 0
            or (
                worker_coverage is not None
                and float(worker_coverage) > 0.0
            )
        )
    )
    budget_trace_ready = (
        bool(matches)
        and budget_summary.get("budget_trace_coverage_frac") is not None
        and float(budget_summary.get("budget_trace_coverage_frac") or 0.0) > 0.0
    )

    seeds_by_condition: dict[str, set] = {}
    for run in runs:
        condition = str(run.get("condition") or run.get("id") or "unknown")
        seeds_by_condition.setdefault(condition, set()).add(run.get("seed"))
    finite_seed_counts = [
        len({seed for seed in seeds if seed is not None})
        for seeds in seeds_by_condition.values()
    ]
    min_seed_count = min(finite_seed_counts) if finite_seed_counts else 0

    def criterion(cid, ok, observed, required, rationale, upgrade_path):
        return {
            "id": cid,
            "status": "pass" if ok else "missing",
            "observed": observed,
            "required": required,
            "rationale": rationale,
            "upgrade_path": upgrade_path,
        }

    criteria = [
        criterion(
            "multi_seed_per_condition",
            bool(runs) and min_seed_count >= 3,
            {"min_seed_count": min_seed_count, "conditions": {k: sorted(v) for k, v in seeds_by_condition.items()}},
            ">=3 explicit seeds per condition",
            "Single-seed results are engineering signals, not publication-grade evidence.",
            "Run with at least three seeds per condition and paired-seed eval when comparing controller conditions.",
        ),
        criterion(
            "paired_seed_protocol_ready",
            bool(seed_summary.get("paired_seed_claim_ready")),
            {
                "condition_count": seed_summary.get("condition_count"),
                "min_seed_count": seed_summary.get("min_seed_count"),
                "common_seed_count": seed_summary.get("common_seed_count"),
                "seed_sets_aligned": seed_summary.get("seed_sets_aligned"),
                "eval_pairs": seed_summary.get("eval_pairs"),
            },
            "multi-condition comparisons share at least three common training seeds",
            "Unpaired condition seed sets inflate variance and weaken controller attribution.",
            "Run every compared condition on the same seed list and prefer --paired-seed-eval for post-train matrices.",
        ),
        criterion(
            "evaluation_matrix_present",
            bool(matches),
            {"match_count": len(matches)},
            "post-train evaluation_matrix.json with scored matches",
            "Training loss alone cannot select a research claim.",
            "Run post-train evaluation for every report intended for claims.",
        ),
        criterion(
            "score_ci_present",
            bool(matches) and all(isinstance(row.get("ci"), list) and len(row.get("ci")) == 2 for row in matches),
            {"rows_with_ci": sum(1 for row in matches if isinstance(row.get("ci"), list) and len(row.get("ci")) == 2), "match_count": len(matches)},
            "every eval row has score-rate confidence interval",
            "Win-rate point estimates are not enough for interpretation.",
            "Use the current evaluation matrix path; do not reuse legacy rows without CI.",
        ),
        criterion(
            "no_stale_eval_cache_rows",
            len(discarded) == 0,
            {"discarded_count": len(discarded)},
            "no discarded stale evaluation rows",
            "Stale search manifests mean cached rows came from a different runtime contract.",
            "Re-run evaluation with --force-eval or remove stale evaluation_matrix.json rows.",
        ),
        criterion(
            "benchmark_safe_recorded",
            bool(expected_benchmark) and all(bool(v) for v in expected_benchmark.values()),
            {"expected_benchmark_safe": expected_benchmark},
            "all eval conditions record benchmark-safe expected path",
            "Fallback/serial paths should not be mixed into claims.",
            "Use Rust eval state-machine, baseline_strict, or multi-threaded batched eval; avoid --allow-unsafe-benchmark for claim runs.",
        ),
        criterion(
            "eval_seed_recorded",
            bool(expected_eval_seeds)
            and all("eval_seed" in (expected_manifests.get(name) or {}) for name in expected_eval_seeds),
            {"expected_eval_seeds": expected_eval_seeds},
            "eval_seed recorded in expected eval manifests",
            "Evaluation RNG must be part of cache invalidation and reproduction.",
            "Use current ablation_study.py or set --eval-seed explicitly.",
        ),
        criterion(
            "evaluation_protocol_recorded",
            bool(evaluation_protocol.get("protocol_ready")),
            {
                "match_count": evaluation_protocol.get("match_count"),
                "runtime_contract_hash": evaluation_protocol.get("runtime_contract_hash"),
                "expected_eval_seed_coverage": evaluation_protocol.get("expected_eval_seed_coverage"),
                "eval_seed_consistent": evaluation_protocol.get("eval_seed_consistent"),
                "benchmark_safe_all_expected": evaluation_protocol.get("benchmark_safe_all_expected"),
                "game_count_consistent": evaluation_protocol.get("game_count_consistent"),
                "one_manifest_per_eval_condition": evaluation_protocol.get("one_manifest_per_eval_condition"),
                "complete_pair_eval_matrix": evaluation_protocol.get("complete_pair_eval_matrix"),
            },
            "evaluation protocol summary records same-runtime, same-seed, same-game, same-pair metadata",
            "Same-evaluator and same-game-distribution claims need explicit protocol evidence, not just scored rows.",
            "Regenerate evaluation_matrix.json with current ablation_study.py so eval rows include pair IDs, runner mode, search manifest hash, runtime contract hash, eval seeds, and benchmark-safe metadata.",
        ),
        criterion(
            "evaluator_quality_strata_recorded",
            bool(evaluator_quality.get("stratification_ready")),
            {
                "match_count": evaluator_quality.get("match_count"),
                "quality_proxy_pair_coverage_frac": evaluator_quality.get("quality_proxy_pair_coverage_frac"),
                "loss_pair_coverage_frac": evaluator_quality.get("loss_pair_coverage_frac"),
                "models_with_quality_proxy": evaluator_quality.get("models_with_quality_proxy"),
                "strata_count": evaluator_quality.get("strata_count"),
                "missing_model_ids": evaluator_quality.get("missing_model_ids"),
            },
            "eval rows can be stratified by model/evaluator quality proxies",
            "Controller robustness claims across evaluator quality are uninterpretable if evaluated models lack quality metadata.",
            "Regenerate train logs and ablation_report.json with current scripts so each evaluated model carries loss, published Elo, or score-rate proxies; add held-out calibration artifacts for stronger claims.",
        ),
        criterion(
            "heldout_evaluator_calibration_recorded",
            bool(heldout_calibration.get("calibration_ready")),
            {
                "artifact_present": heldout_calibration.get("artifact_present"),
                "coverage_frac": heldout_calibration.get("coverage_frac"),
                "missing_model_ids": heldout_calibration.get("missing_model_ids"),
            },
            "held-out evaluator calibration artifact covers every evaluated model",
            "Evaluator-quality robustness claims need policy/value calibration, not only train-log proxies.",
            "Write evaluator_calibration.json with n_positions, policy_nll, value_mse, top1_acc, and brier for every evaluated model id.",
        ),
        criterion(
            "selection_trace_recorded",
            bool(selection_trace_ready_rows),
            {
                "conditions": selection_conditions,
                "rows_with_nonzero_trace_and_coverage": len(selection_trace_ready_rows),
            },
            "root-selection trace and coverage present for eval conditions",
            "Controller claims need evidence from the path that actually selected root actions.",
            "Use a Rust binary emitting controller_summary.selection_trace and regenerate evaluation rows with realized_budget_trace.selection_trace_coverage_frac.",
        ),
        criterion(
            "budget_trace_recorded",
            budget_trace_ready,
            {
                "budget_trace_coverage_frac": budget_summary.get("budget_trace_coverage_frac"),
                "root_visit_mean_relative_spread": budget_summary.get("root_visit_mean_relative_spread"),
                "budget_fairness_flag": budget_summary.get("budget_fairness_flag"),
            },
            "eval rows record realized root-visit and halt-reason budget traces",
            "Controller comparisons are uninterpretable when modes silently receive different realized search budgets.",
            "Regenerate evaluation rows with current Rust/Python eval runtime so realized_budget_trace.root_visits and halt_reason_hist are present.",
        ),
        criterion(
            "pipeline_telemetry_recorded",
            pipeline_telemetry_ready,
            {
                "aggregate": pipeline_aggregate,
                "requires_worker_telemetry": bool(concurrent_runs > 0),
            },
            "train rows record replay freshness, throughput, and concurrent-worker telemetry when applicable",
            "Pipeline claims require evidence that replay and self-play throughput were observed in the artifact.",
            "Regenerate train logs with current quartz.train so replay_freshness, pos_per_s, and selfplay_telemetry are present.",
        ),
        criterion(
            "hardware_claim_scope_recorded",
            bool(hardware_summary.get("claim_scope"))
            and isinstance(hardware_summary.get("profiler_artifact_present"), bool),
            {
                "claim_scope": hardware_summary.get("claim_scope"),
                "profiler_artifact_present": hardware_summary.get("profiler_artifact_present"),
                "hardware_performance_claims_allowed": hardware_summary.get("hardware_performance_claims_allowed"),
            },
            "hardware runtime summary records claim scope and profiler-artifact presence",
            "Backend/device names and throughput telemetry are not the same as hardware profiling evidence.",
            "Regenerate ablation_report.json with current ablation_study.py; add profiler artifacts only for hardware-performance claims.",
        ),
        criterion(
            "deployment_source_explicit",
            champion is None
            or bool(champion.get("deployment_cfg_source")),
            {"deployment_cfg_source": None if champion is None else champion.get("deployment_cfg_source")},
            "champion deployment config source is explicit",
            "Deployment export must state whether train_cfg or a specific eval condition supplied the search config.",
            "Regenerate champion.json with the current ablation runner.",
        ),
    ]
    unmet = [item["id"] for item in criteria if item["status"] != "pass"]
    return {
        "schema_version": 1,
        "research_grade_ready": not unmet,
        "unmet_count": len(unmet),
        "unmet_criteria": unmet,
        "criteria": criteria,
        "policy_doc": "docs/RESEARCH_READINESS.md",
        "blocking": False,
    }


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

    eval_timing_summary = summarize_ablation_eval_timings(eval_payload)
    selection_trace_summary = summarize_selection_trace_contract(eval_payload)
    budget_fairness_summary = summarize_budget_fairness(eval_payload)
    seed_protocol_summary = summarize_seed_protocol(runs, eval_payload)
    evaluation_protocol_summary = summarize_evaluation_protocol(runs, eval_payload)
    evaluator_quality_summary = summarize_evaluator_quality_strata(runs, eval_payload)
    heldout_calibration_summary = summarize_heldout_calibration(base_dir, runs)
    pipeline_telemetry_summary = summarize_pipeline_telemetry(runs)
    hardware_runtime_summary = summarize_hardware_runtime(
        base_dir,
        runs,
        eval_payload,
        pipeline_telemetry_summary,
    )
    if eval_timing_summary.get("total_games", 0) > 0:
        print(f"\n{'Eval Timing Summary':<40}")
        print("-" * 132)
        print(
            f"total_games={eval_timing_summary['total_games']} "
            f"startup={eval_timing_summary['total_startup_s']:.3f}s "
            f"search={eval_timing_summary['total_match_elapsed_s']:.3f}s "
            f"end_to_end_gps={eval_timing_summary['games_per_s_end_to_end']:.3f} "
            f"search_only_gps={eval_timing_summary['games_per_s_search_only']:.3f} "
            f"startup_share={eval_timing_summary['startup_share']:.3f}"
        )
        for row in eval_timing_summary.get("conditions", []):
            print(
                f"{row['eval_condition']:<20} games={row['games']:<4} "
                f"startup={row['startup_s']:.3f}s search={row['match_elapsed_s']:.3f}s "
                f"gps={row['games_per_s_end_to_end']:.3f} startup_share={row['startup_share']:.3f}"
            )

    if selection_trace_summary.get("conditions"):
        print(f"\n{'Selection Trace Summary':<40}")
        print("-" * 132)
        for row in selection_trace_summary["conditions"]:
            refresh_frac = row.get("refresh_selected_frac")
            mean_penalty = row.get("mean_penalty_abs_per_root_select")
            mean_prior_l1 = row.get("mean_prior_l1_per_root_select")
            print(
                f"{row['eval_condition']:<20} root_selects={row['root_selects']:<8} "
                f"refresh_frac={(f'{refresh_frac:.4f}' if refresh_frac is not None else '—'):>8} "
                f"penalty/root={(f'{mean_penalty:.6f}' if mean_penalty is not None else '—'):>10} "
                f"prior_l1/root={(f'{mean_prior_l1:.6f}' if mean_prior_l1 is not None else '—'):>10}"
            )

    if budget_fairness_summary.get("conditions"):
        spread = budget_fairness_summary.get("root_visit_mean_relative_spread")
        print(f"\n{'Budget Fairness Summary':<40}")
        print("-" * 132)
        print(
            f"trace_cov={budget_fairness_summary.get('budget_trace_coverage_frac'):.3f} "
            f"root_visit_spread={(f'{spread:.4f}' if spread is not None else '—')} "
            f"flag={budget_fairness_summary.get('budget_fairness_flag')}"
        )
        for row in budget_fairness_summary["conditions"]:
            rv = row.get("root_visits") or {}
            rv_mean = rv.get("mean")
            rv_max = rv.get("max")
            print(
                f"{row['eval_condition']:<20} games={row['games']:<4} moves={row['moves']:<5} "
                f"root_mean={(f'{rv_mean:.2f}' if rv_mean is not None else '—'):>8} "
                f"root_max={(f'{rv_max:.0f}' if rv_max is not None else '—'):>8} "
                f"halts={json.dumps(row.get('halt_reason_hist') or {}, sort_keys=True)}"
            )

    if seed_protocol_summary.get("condition_count", 0) > 0:
        eval_pairs = seed_protocol_summary.get("eval_pairs") or {}
        same_seed_pair_frac = eval_pairs.get("same_seed_pair_frac")
        print(f"\n{'Seed Protocol Summary':<40}")
        print("-" * 132)
        print(
            f"conditions={seed_protocol_summary['condition_count']} "
            f"min_seed_count={seed_protocol_summary['min_seed_count']} "
            f"common_seed_count={seed_protocol_summary['common_seed_count']} "
            f"aligned={seed_protocol_summary['seed_sets_aligned']} "
            f"paired_eval={seed_protocol_summary['paired_seed_eval_requested']} "
            f"same_seed_eval_frac={(f'{same_seed_pair_frac:.3f}' if same_seed_pair_frac is not None else '—')}"
        )

    if evaluation_protocol_summary.get("match_count", 0) > 0:
        print(f"\n{'Evaluation Protocol Summary':<40}")
        print("-" * 132)
        print(
            f"matches={evaluation_protocol_summary['match_count']} "
            f"conditions={evaluation_protocol_summary['eval_condition_count']} "
            f"runtime_hash={evaluation_protocol_summary.get('runtime_contract_hash') or '—'} "
            f"manifest_cov={evaluation_protocol_summary.get('search_manifest_hash_coverage_frac'):.3f} "
            f"pair_cov={evaluation_protocol_summary.get('pair_id_coverage_frac'):.3f} "
            f"ready={evaluation_protocol_summary.get('protocol_ready')}"
        )

    if evaluator_quality_summary.get("match_count", 0) > 0:
        quality_cov = evaluator_quality_summary.get("quality_proxy_pair_coverage_frac")
        loss_cov = evaluator_quality_summary.get("loss_pair_coverage_frac")
        print(f"\n{'Evaluator Quality Stratification':<40}")
        print("-" * 132)
        print(
            f"matches={evaluator_quality_summary['match_count']} "
            f"models={evaluator_quality_summary['model_count']} "
            f"quality_cov={(f'{quality_cov:.3f}' if quality_cov is not None else '—')} "
            f"loss_cov={(f'{loss_cov:.3f}' if loss_cov is not None else '—')} "
            f"strata={evaluator_quality_summary['strata_count']} "
            f"ready={evaluator_quality_summary.get('stratification_ready')}"
        )

    pipeline_agg = pipeline_telemetry_summary.get("aggregate") or {}
    if pipeline_agg.get("row_count", 0) > 0:
        freshness_mean = (pipeline_agg.get("freshness") or {}).get("mean")
        pos_mean = (pipeline_agg.get("pos_per_s") or {}).get("mean")
        worker_cov = pipeline_agg.get("worker_telemetry_coverage_frac")
        print(f"\n{'Pipeline Telemetry Summary':<40}")
        print("-" * 132)
        print(
            f"rows={pipeline_agg['row_count']} "
            f"fresh_cov={pipeline_agg.get('freshness_coverage_frac'):.3f} "
            f"throughput_cov={pipeline_agg.get('throughput_coverage_frac'):.3f} "
            f"worker_cov={(f'{worker_cov:.3f}' if worker_cov is not None else '—')} "
            f"fresh_mean={(f'{freshness_mean:.4f}' if freshness_mean is not None else '—')} "
            f"pos_per_s_mean={(f'{pos_mean:.3f}' if pos_mean is not None else '—')}"
        )

    print(f"\n{'Hardware Claim Scope':<40}")
    print("-" * 132)
    print(
        f"scope={hardware_runtime_summary['claim_scope']} "
        f"profiler={hardware_runtime_summary['profiler_artifact_present']} "
        f"backends={','.join(hardware_runtime_summary['requested_backends']) or '—'} "
        f"devices={','.join(hardware_runtime_summary['requested_devices']) or '—'}"
    )

    if champion:
        print(f"\nChampion: {champion['model_id']}")
        dep = champion.get("deployment_search_cfg", {})
        print(
            f"  deploy search={dep.get('search_profile', 'quartz')} "
            f"vl={dep.get('vl_mode', 'adaptive')} "
            f"penalty={dep.get('penalty_mode', 'default')} "
            f"root_only={dep.get('root_only_shaping', 'default')} "
            f"source={champion.get('deployment_cfg_source', 'train_cfg')} "
            f"model={champion.get('model_path')}"
        )

    readiness = research_readiness_summary(
        runs,
        eval_payload,
        selection_trace_summary,
        budget_fairness_summary,
        seed_protocol_summary,
        pipeline_telemetry_summary,
        hardware_runtime_summary,
        champion,
        evaluation_protocol_summary=evaluation_protocol_summary,
        evaluator_quality_summary=evaluator_quality_summary,
        heldout_calibration_summary=heldout_calibration_summary,
    )
    print(
        f"\nResearch readiness: "
        f"{'ready' if readiness['research_grade_ready'] else 'incomplete'} "
        f"(unmet={readiness['unmet_count']}, doc={readiness['policy_doc']})"
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base_dir),
        "runs": runs,
        "evaluation": eval_payload,
        "train_contract_summary": summarize_training_contracts(
            [run.get("train_contract") for run in runs if run.get("train_contract") is not None]
        ),
        "contract_summary": summarize_ablation_contracts(
            (eval_payload or {}).get("expected_search_manifests"),
            (eval_payload or {}).get("discarded_matches"),
        ),
        "runtime_contract": (eval_payload or {}).get("runtime_contract"),
        "runtime_contract_hash": (eval_payload or {}).get("runtime_contract_hash"),
        "eval_timing_summary": eval_timing_summary,
        "selection_trace_summary": selection_trace_summary,
        "budget_fairness_summary": budget_fairness_summary,
        "seed_protocol_summary": seed_protocol_summary,
        "evaluation_protocol_summary": evaluation_protocol_summary,
        "evaluator_quality_summary": evaluator_quality_summary,
        "heldout_calibration_summary": heldout_calibration_summary,
        "pipeline_telemetry_summary": pipeline_telemetry_summary,
        "hardware_runtime_summary": hardware_runtime_summary,
        "research_readiness": readiness,
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
        help="Study preset: search_vl, controller, controller_factorial, or controller_axes",
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
    # Smoke runs need inline self-play so a single short iteration fills
    # replay above the SGD batch threshold; concurrent mode only waits
    # for `min_new=1` per iter and won't block until batch is reached.
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Pass --no-pipeline to quartz.train (inline self-play; needed for short smokes).")
    parser.add_argument(
        "--allow-unsafe-benchmark",
        action="store_true",
        help=(
            "Allow evaluation rows whose effective search path is not known to be "
            "benchmark-safe. Default is to fail instead of mixing serial/fallback paths."
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Deterministic seed for post-train evaluation games; included in eval manifest hashes.",
    )
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
    # P8 (audit_codex_20260425.md W7): for attribution presets, every
    # row should be evaluated under a single fixed eval engine so that
    # cross-row deltas reflect model quality, not (model × eval search
    # profile). Default: auto-resolve to the first eval condition for
    # attribution presets (controller_axes / controller_factorial),
    # legacy per-row matrix for everything else.
    parser.add_argument(
        "--frozen-eval-condition",
        default=None,
        help="Pin every pair's eval to this single named condition. "
             "Defaults to the first eval condition for attribution presets, "
             "no-op otherwise.",
    )
    parser.add_argument(
        "--no-frozen-eval",
        action="store_true",
        help="Opt out of P8 frozen-eval pinning even for attribution presets.",
    )
    parser.add_argument("--prepare-gomocup", action="store_true",
                        help="Export the selected champion as a Gomocup bundle")
    parser.add_argument("--gomocup-dir", default=None, help="Output directory for the Gomocup bundle")
    parser.add_argument("--target-name", default="pbrain-quartz", help="Suggested Gomocup binary name")
    parser.add_argument("--verbose-export", action="store_true")
    parser.add_argument("--report", metavar="DIR", help="Report on an existing ablation directory")
    parser.add_argument(
        "--research-grade",
        action="store_true",
        help="Fail if the generated report does not pass every research-readiness criterion.",
    )
    parser.add_argument(
        "--min-seeds-for-research-grade",
        type=int,
        default=3,
        help=(
            "Minimum distinct seeds required when --research-grade is set. "
            "Default 3 matches RESEARCH_READINESS.md. (P04)"
        ),
    )
    return parser.parse_args()


def enforce_research_grade(args: argparse.Namespace, report: dict | None) -> None:
    """P04: enforce multi-seed and paired-seed protocols at the CLI level.

    Soft warnings fire whenever `len(seeds) < 3` regardless of
    `--research-grade` so a user running an exploratory single-seed
    sweep sees the message without it blocking iteration.

    Hard gates fire only under `--research-grade`:
    - `len(seeds) < min_seeds_for_research_grade` ⇒ SystemExit before
      training starts (no point spending compute on a claim that will
      be rejected post-hoc).
    - `--paired-seed-eval` set but conditions disagree on which seeds
      were actually realized ⇒ SystemExit. The conditions must share
      an identical seed set; otherwise pairing mixes apples and oranges.
    - `report['research_readiness']['research_grade_ready'] is False`
      ⇒ SystemExit with the unmet criteria list (existing behavior).
    """
    seeds = parse_seed_list(getattr(args, "seeds", "42"))
    research_grade = bool(getattr(args, "research_grade", False))
    min_seeds = int(getattr(args, "min_seeds_for_research_grade", 3))

    # Soft warning: always fires for sub-3-seed runs so users learn the
    # protocol convention even when they're iterating without
    # --research-grade.
    if len(seeds) < 3 and not research_grade:
        print(
            f"[ablation_study] WARN: running with {len(seeds)} seed(s); "
            f"RESEARCH_READINESS requires >=3 seeds for non-debug claims. "
            f"Pass --seeds=A,B,C and --research-grade to enforce.",
            file=sys.stderr,
            flush=True,
        )

    if not research_grade:
        return

    # Hard gate 1: minimum distinct seeds.
    if len(seeds) < min_seeds:
        raise SystemExit(
            f"--research-grade requires at least {min_seeds} seeds; got {seeds!r}. "
            f"Override with --min-seeds-for-research-grade=N (not recommended)."
        )

    # Hard gate 2: paired-seed protocol consistency across conditions. If
    # the report doesn't expose `runs` (e.g. enforce called pre-train),
    # skip this check and rely on the readiness gate below.
    if getattr(args, "paired_seed_eval", False) and report is not None:
        by_condition: dict = {}
        for run_summary in (report or {}).get("runs", []):
            cond = run_summary.get("condition")
            seed = run_summary.get("seed")
            if cond is None or seed is None:
                continue
            by_condition.setdefault(cond, set()).add(int(seed))
        if len(by_condition) >= 2:
            seed_sets = list(by_condition.values())
            if not all(s == seed_sets[0] for s in seed_sets[1:]):
                seed_map = {k: sorted(v) for k, v in by_condition.items()}
                raise SystemExit(
                    "--paired-seed-eval requested but conditions disagree on seeds: "
                    f"{seed_map!r}"
                )

    # Hard gate 3 (existing): report-level readiness.
    readiness = (report or {}).get("research_readiness") or {}
    if readiness.get("research_grade_ready"):
        return
    unmet = ", ".join(readiness.get("unmet_criteria") or [])
    raise SystemExit(
        "research-grade gate failed: "
        f"{unmet or 'research_readiness missing'}"
    )


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
        report = generate_report(base_dir, selected_conditions=selected_conditions)
        enforce_research_grade(args, report)
        return

    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)
    emit_attribution_halt_guard(args.study)
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
    manifest["train_contract_summary"] = summarize_training_contracts(
        [
            build_training_contract(args, condition_name, train_conditions[condition_name], seed)
            for condition_name in selected_conditions
            for seed in seeds
        ]
    )
    manifest["contract_summary"] = summarize_ablation_contracts(
        (eval_payload or {}).get("expected_search_manifests"),
        (eval_payload or {}).get("discarded_matches"),
    )
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(base_dir / "study_manifest.json", manifest)
    champion = select_champion(base_dir, model_runs, eval_payload, eval_conditions)
    if args.prepare_gomocup:
        prepare_gomocup_bundle(args, base_dir, champion)
    report = generate_report(base_dir, selected_conditions=selected_condition_set)
    enforce_research_grade(args, report)


if __name__ == "__main__":
    main()
