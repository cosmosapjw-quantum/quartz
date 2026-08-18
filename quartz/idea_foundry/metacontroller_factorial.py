"""Fail-closed contracts for the metacontroller training/runtime 2x2 study.

The companion execution module launches self-play and arena processes through
an ``idea-foundry`` feature build.  This module remains the pure design,
validation, and analysis layer.  It provides:

* the exact four-cell training x runtime design;
* deterministic fixed-anchor and supplementary round-robin match plans;
* checkpoint, training-treatment, opening-bank, and runtime preflight gates;
* strict raw paired-game validation; and
* seed-level factorial contrasts that never auto-promote a claim.

The repository base configuration intentionally remains BLOCKED; only a
hash-bound run-local configuration may authorize the feature-gated adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from quartz.experiment_manifest import (
    atomic_json_dump,
    canonical_sha256,
    file_sha256,
    utc_now,
)
from quartz.idea_foundry.axis_workflow import atomic_jsonl_dump
from quartz.runtime_support import validate_search_option_keys


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA_VERSION = 1
OPENING_SCHEMA_VERSION = 1
TRAINING_MANIFEST_SCHEMA_VERSION = 1
RAW_ROW_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
PREFLIGHT_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1

CELL_FACTORS: dict[str, tuple[int, int]] = {
    "M00": (0, 0),
    "M01": (0, 1),
    "M10": (1, 0),
    "M11": (1, 1),
}
TRAINING_ARM_BY_FACTOR = {0: "off", 1: "on"}
RUNTIME_ARM_BY_FACTOR = {0: "off", 1: "on"}
PRIMARY_CONTRASTS = (
    "training_main",
    "runtime_main",
    "interaction",
    "joint_stack",
    "training_at_runtime_off",
    "training_at_runtime_on",
    "runtime_at_training_off",
    "runtime_at_training_on",
)
ALLOWED_ADAPTER_STATUSES = {"not_implemented", "implemented"}
ALLOWED_BUDGET_LANES = {
    "fixed_nn_evals",
    "fixed_wallclock",
    "fixed_cap_resource_frontier",
}
HEX_DIGITS = frozenset("0123456789abcdef")


class FactorialHarnessError(RuntimeError):
    """Raised when a factorial design or artifact violates its contract."""


def _reject_nonfinite_constant(value: str) -> None:
    raise FactorialHarnessError(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: str | Path) -> Any:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FactorialHarnessError(
            f"required regular JSON file is missing: {candidate}"
        )
    try:
        return json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorialHarnessError(f"invalid JSON file {candidate}: {exc}") from exc


def load_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FactorialHarnessError(
            f"required regular JSONL file is missing: {candidate}"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        candidate.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite_constant)
        except json.JSONDecodeError as exc:
            raise FactorialHarnessError(
                f"invalid JSONL row {candidate}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FactorialHarnessError(
                f"JSONL row must be an object: {candidate}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise FactorialHarnessError(f"JSONL artifact has no rows: {candidate}")
    return rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorialHarnessError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    keys = set(map(str, value))
    missing = sorted(required_set - keys)
    unknown = sorted(keys - required_set - optional_set)
    if missing or unknown:
        raise FactorialHarnessError(
            f"{label} keys are invalid: missing={missing}, unknown={unknown}"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FactorialHarnessError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise FactorialHarnessError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorialHarnessError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise FactorialHarnessError(f"{label} must be >= {minimum}")
    return value


def _require_finite_number(
    value: Any, label: str, *, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorialHarnessError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FactorialHarnessError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise FactorialHarnessError(f"{label} must be >= {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise FactorialHarnessError(f"{label} must be a lowercase SHA-256 hex string")
    return value


def _safe_repo_path(repo_root: Path, raw_path: str, *, label: str) -> Path:
    text = _require_nonempty_string(raw_path, label)
    candidate = Path(text)
    if candidate.is_absolute():
        raise FactorialHarnessError(f"{label} must be repository-relative: {text}")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise FactorialHarnessError(f"{label} escapes repository root: {text}") from exc
    return resolved


def _validate_seed_template(template: Any, label: str) -> str:
    text = _require_nonempty_string(template, label)
    fields = [field for _, field, _, _ in string.Formatter().parse(text) if field]
    if fields != ["seed"]:
        raise FactorialHarnessError(
            f"{label} must contain exactly one {{seed}} placeholder"
        )
    if Path(text).is_absolute():
        raise FactorialHarnessError(f"{label} must be repository-relative")
    return text


def _render_seed_template(
    repo_root: Path, template: str, seed: int, *, label: str
) -> Path:
    return _safe_repo_path(repo_root, template.format(seed=seed), label=label)


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _runtime_contract_hash(runtime_arm: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "controller_mode": runtime_arm["controller_mode"],
            "live_action_authorized": runtime_arm["live_action_authorized"],
            "authorized_axis_ids": list(runtime_arm["authorized_axis_ids"]),
            "search_config": dict(runtime_arm["search_config"]),
        }
    )


def _training_controller_hash(
    config: Mapping[str, Any], training_arm: Mapping[str, Any]
) -> str:
    controller = config["controller_contract"]
    return canonical_sha256(
        {
            "controller_mode": training_arm["controller_mode"],
            "implementation": controller["implementation"],
            "active_axis_ids": list(controller["active_axis_ids"]),
        }
    )


def _validate_runtime_arm(name: str, raw: Any) -> dict[str, Any]:
    arm = dict(_require_mapping(raw, f"runtime_arms.{name}"))
    _require_exact_keys(
        arm,
        required=(
            "controller_mode",
            "live_action_authorized",
            "authorized_axis_ids",
            "search_config",
        ),
        label=f"runtime_arms.{name}",
    )
    expected_mode = "shadow_noop" if name == "off" else "active"
    if arm["controller_mode"] != expected_mode:
        raise FactorialHarnessError(
            f"runtime_arms.{name}.controller_mode must be {expected_mode!r}"
        )
    expected_authority = name == "on"
    if (
        _require_bool(
            arm["live_action_authorized"],
            f"runtime_arms.{name}.live_action_authorized",
        )
        is not expected_authority
    ):
        raise FactorialHarnessError(
            f"runtime_arms.{name}.live_action_authorized must be {expected_authority}"
        )
    axes = arm["authorized_axis_ids"]
    if not isinstance(axes, list) or not all(
        isinstance(axis, str) and axis for axis in axes
    ):
        raise FactorialHarnessError(
            f"runtime_arms.{name}.authorized_axis_ids must be a string list"
        )
    if len(axes) != len(set(axes)):
        raise FactorialHarnessError(
            f"runtime_arms.{name}.authorized_axis_ids contains duplicates"
        )
    if name == "off" and axes:
        raise FactorialHarnessError("runtime OFF may not authorize live axes")
    allowed_axis_ids = {f"A{index:02d}" for index in range(1, 27)} | {
        "A01.live_pflip_v1",
        "A01.stop_council",
    }
    if not set(axes).issubset(allowed_axis_ids):
        raise FactorialHarnessError(
            f"runtime_arms.{name}.authorized_axis_ids contains an unknown axis"
        )
    search_config = dict(
        _require_mapping(arm["search_config"], f"runtime_arms.{name}.search_config")
    )
    unknown = validate_search_option_keys(
        search_config, context=f"runtime_arms.{name}.search_config"
    )
    if unknown:
        raise FactorialHarnessError(
            f"runtime_arms.{name}.search_config has unknown keys: {unknown}"
        )
    for key in ("search_profile", "vl_mode", "halt_mode", "iters", "n_threads"):
        if key not in search_config:
            raise FactorialHarnessError(
                f"runtime_arms.{name}.search_config is missing {key!r}"
            )
    arm["authorized_axis_ids"] = list(axes)
    arm["search_config"] = search_config
    return arm


def validate_config(payload: Any) -> dict[str, Any]:
    config = dict(_require_mapping(payload, "factorial config"))
    _require_exact_keys(
        config,
        required=(
            "schema_version",
            "experiment_id",
            "claim_scope",
            "game",
            "board_size",
            "controller_contract",
            "training_arms",
            "runtime_arms",
            "anchor",
            "match_design",
            "analysis_contract",
            "profiles",
            "promotion",
            "prohibited_inferences",
        ),
        label="factorial config",
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise FactorialHarnessError("factorial config schema mismatch")
    _require_nonempty_string(config["experiment_id"], "experiment_id")
    _require_nonempty_string(config["claim_scope"], "claim_scope")
    _require_nonempty_string(config["game"], "game")
    _require_int(config["board_size"], "board_size", minimum=1)

    controller = dict(
        _require_mapping(config["controller_contract"], "controller_contract")
    )
    _require_exact_keys(
        controller,
        required=(
            "implementation",
            "training_adapter_status",
            "runtime_adapter_status",
            "production_loop_wired",
            "execution_ready",
            "active_axis_ids",
        ),
        label="controller_contract",
    )
    _require_nonempty_string(controller["implementation"], "controller implementation")
    for key in ("training_adapter_status", "runtime_adapter_status"):
        if controller[key] not in ALLOWED_ADAPTER_STATUSES:
            raise FactorialHarnessError(
                f"controller_contract.{key} must be one of {sorted(ALLOWED_ADAPTER_STATUSES)}"
            )
    _require_bool(controller["production_loop_wired"], "production_loop_wired")
    _require_bool(controller["execution_ready"], "execution_ready")
    active_axes = controller["active_axis_ids"]
    if not isinstance(active_axes, list) or not all(
        isinstance(axis, str) and axis for axis in active_axes
    ):
        raise FactorialHarnessError("controller active_axis_ids must be a string list")
    if len(active_axes) != len(set(active_axes)):
        raise FactorialHarnessError("controller active_axis_ids contains duplicates")
    controller["active_axis_ids"] = list(active_axes)
    config["controller_contract"] = controller

    training_arms = dict(_require_mapping(config["training_arms"], "training_arms"))
    if set(training_arms) != {"off", "on"}:
        raise FactorialHarnessError("training_arms must contain exactly off and on")
    normalized_training: dict[str, dict[str, Any]] = {}
    for name in ("off", "on"):
        arm = dict(_require_mapping(training_arms[name], f"training_arms.{name}"))
        _require_exact_keys(
            arm,
            required=("controller_mode", "checkpoint_template", "manifest_template"),
            label=f"training_arms.{name}",
        )
        expected_mode = "shadow_noop" if name == "off" else "active"
        if arm["controller_mode"] != expected_mode:
            raise FactorialHarnessError(
                f"training_arms.{name}.controller_mode must be {expected_mode!r}"
            )
        arm["checkpoint_template"] = _validate_seed_template(
            arm["checkpoint_template"], f"training_arms.{name}.checkpoint_template"
        )
        arm["manifest_template"] = _validate_seed_template(
            arm["manifest_template"], f"training_arms.{name}.manifest_template"
        )
        normalized_training[name] = arm
    config["training_arms"] = normalized_training

    runtime_arms = dict(_require_mapping(config["runtime_arms"], "runtime_arms"))
    if set(runtime_arms) != {"off", "on"}:
        raise FactorialHarnessError("runtime_arms must contain exactly off and on")
    normalized_runtime = {
        name: _validate_runtime_arm(name, runtime_arms[name]) for name in ("off", "on")
    }
    if (
        normalized_runtime["off"]["search_config"]
        != normalized_runtime["on"]["search_config"]
    ):
        raise FactorialHarnessError(
            "runtime ON/OFF search_config must be identical; the metacontroller "
            "intervention is the only allowed runtime factor"
        )
    if normalized_runtime["on"]["authorized_axis_ids"] != active_axes:
        raise FactorialHarnessError(
            "runtime ON authorized axes must equal controller_contract.active_axis_ids"
        )
    config["runtime_arms"] = normalized_runtime

    anchor = dict(_require_mapping(config["anchor"], "anchor"))
    _require_exact_keys(
        anchor,
        required=(
            "agent_id",
            "checkpoint",
            "checkpoint_sha256",
            "controller_mode",
            "search_config",
        ),
        label="anchor",
    )
    if anchor["agent_id"] != "ANCHOR":
        raise FactorialHarnessError("anchor.agent_id must be 'ANCHOR'")
    _require_nonempty_string(anchor["checkpoint"], "anchor.checkpoint")
    if anchor["checkpoint_sha256"] is not None:
        _require_sha256(anchor["checkpoint_sha256"], "anchor.checkpoint_sha256")
    if anchor["controller_mode"] != "shadow_noop":
        raise FactorialHarnessError("anchor controller_mode must be shadow_noop")
    anchor_search = dict(
        _require_mapping(anchor["search_config"], "anchor.search_config")
    )
    unknown_anchor = validate_search_option_keys(
        anchor_search, context="anchor.search_config"
    )
    if unknown_anchor:
        raise FactorialHarnessError(
            f"anchor.search_config has unknown keys: {unknown_anchor}"
        )
    anchor["search_config"] = anchor_search
    config["anchor"] = anchor

    match_design = dict(_require_mapping(config["match_design"], "match_design"))
    _require_exact_keys(
        match_design,
        required=("primary", "include_round_robin", "color_swap"),
        label="match_design",
    )
    if match_design["primary"] != "fixed_anchor":
        raise FactorialHarnessError("primary match design must be fixed_anchor")
    _require_bool(match_design["include_round_robin"], "include_round_robin")
    if _require_bool(match_design["color_swap"], "color_swap") is not True:
        raise FactorialHarnessError("color_swap must be true")
    config["match_design"] = match_design

    analysis = dict(_require_mapping(config["analysis_contract"], "analysis_contract"))
    required_keys = {
        "primary_endpoint",
        "primary_contrast",
        "effect_scale",
        "max_realized_budget_relative_spread",
        "required_selection_trace_coverage",
    }
    missing = required_keys - set(analysis)
    if missing:
        raise FactorialHarnessError(
            f"analysis_contract missing required keys: {sorted(missing)}"
        )
    allowed_keys = required_keys | {
        "quality_noninferiority_margin",
        "interaction_equivalence_margin",
    }
    unknown = set(analysis) - allowed_keys
    if unknown:
        raise FactorialHarnessError(
            f"analysis_contract has unknown keys: {sorted(unknown)}"
        )
    if analysis["primary_endpoint"] != "paired_score_rate_vs_frozen_anchor":
        raise FactorialHarnessError("unsupported primary endpoint")
    if analysis["primary_contrast"] not in PRIMARY_CONTRASTS:
        raise FactorialHarnessError("unsupported primary contrast")
    if analysis["effect_scale"] != "score_rate_difference":
        raise FactorialHarnessError("effect_scale must be score_rate_difference")
    spread = _require_finite_number(
        analysis["max_realized_budget_relative_spread"],
        "max_realized_budget_relative_spread",
        minimum=0.0,
    )
    if spread > 1.0:
        raise FactorialHarnessError(
            "max_realized_budget_relative_spread may not exceed 1"
        )
    coverage = _require_finite_number(
        analysis["required_selection_trace_coverage"],
        "required_selection_trace_coverage",
        minimum=0.0,
    )
    if coverage > 1.0:
        raise FactorialHarnessError(
            "required_selection_trace_coverage may not exceed 1"
        )
    analysis["max_realized_budget_relative_spread"] = spread
    analysis["required_selection_trace_coverage"] = coverage
    if "quality_noninferiority_margin" in analysis:
        analysis["quality_noninferiority_margin"] = _require_finite_number(
            analysis["quality_noninferiority_margin"],
            "quality_noninferiority_margin",
            minimum=0.0,
        )
    if "interaction_equivalence_margin" in analysis:
        analysis["interaction_equivalence_margin"] = _require_finite_number(
            analysis["interaction_equivalence_margin"],
            "interaction_equivalence_margin",
            minimum=0.0,
        )
    config["analysis_contract"] = analysis

    profiles = dict(_require_mapping(config["profiles"], "profiles"))
    if not profiles:
        raise FactorialHarnessError("profiles must not be empty")
    normalized_profiles: dict[str, dict[str, Any]] = {}
    for name, raw_profile in profiles.items():
        profile_name = _require_nonempty_string(name, "profile name")
        profile = dict(_require_mapping(raw_profile, f"profiles.{profile_name}"))
        _require_exact_keys(
            profile,
            required=(
                "seeds",
                "opening_bank",
                "budget_lane",
                "training_compute_contract",
                "eval_seed_base",
                "min_paired_seeds",
                "require_heldout_opening_bank",
            ),
            label=f"profiles.{profile_name}",
        )
        seeds = profile["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise FactorialHarnessError(
                f"profiles.{profile_name}.seeds must be non-empty"
            )
        normalized_seeds = [
            _require_int(seed, f"profiles.{profile_name}.seed", minimum=0)
            for seed in seeds
        ]
        if len(normalized_seeds) != len(set(normalized_seeds)):
            raise FactorialHarnessError(f"profiles.{profile_name}.seeds has duplicates")
        profile["seeds"] = sorted(normalized_seeds)
        _require_nonempty_string(
            profile["opening_bank"], f"profiles.{profile_name}.opening_bank"
        )
        if profile["budget_lane"] not in ALLOWED_BUDGET_LANES:
            raise FactorialHarnessError(
                f"profiles.{profile_name}.budget_lane must be one of {sorted(ALLOWED_BUDGET_LANES)}"
            )
        if profile["training_compute_contract"] not in {
            "fixed_nn_evals_and_learner_updates",
            "fixed_wallclock_and_learner_updates",
            "fixed_games_and_learner_updates_resource_frontier",
        }:
            raise FactorialHarnessError(
                f"profiles.{profile_name}.training_compute_contract is unsupported"
            )
        profile["eval_seed_base"] = _require_int(
            profile["eval_seed_base"],
            f"profiles.{profile_name}.eval_seed_base",
            minimum=0,
        )
        profile["min_paired_seeds"] = _require_int(
            profile["min_paired_seeds"],
            f"profiles.{profile_name}.min_paired_seeds",
            minimum=1,
        )
        _require_bool(
            profile["require_heldout_opening_bank"],
            f"profiles.{profile_name}.require_heldout_opening_bank",
        )
        normalized_profiles[profile_name] = profile
    config["profiles"] = normalized_profiles

    promotion = dict(_require_mapping(config["promotion"], "promotion"))
    _require_exact_keys(promotion, required=("auto", "eligible"), label="promotion")
    if promotion != {"auto": False, "eligible": False}:
        raise FactorialHarnessError("factorial harness promotion must be disabled")
    prohibited = config["prohibited_inferences"]
    if (
        not isinstance(prohibited, list)
        or not prohibited
        or not all(isinstance(item, str) and item for item in prohibited)
    ):
        raise FactorialHarnessError(
            "prohibited_inferences must be a non-empty string list"
        )
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(load_json_strict(path))


def validate_opening_bank(
    payload: Any, *, game: str, board_size: int
) -> dict[str, Any]:
    bank = dict(_require_mapping(payload, "opening bank"))
    _require_exact_keys(
        bank,
        required=(
            "schema_version",
            "bank_id",
            "game",
            "board_size",
            "bank_kind",
            "claim_scope",
            "independent_of_training",
            "source_manifest",
            "openings",
        ),
        label="opening bank",
    )
    if bank["schema_version"] != OPENING_SCHEMA_VERSION:
        raise FactorialHarnessError("opening bank schema mismatch")
    if bank["game"] != game or bank["board_size"] != board_size:
        raise FactorialHarnessError("opening bank game/board contract mismatch")
    _require_nonempty_string(bank["bank_id"], "opening bank id")
    _require_nonempty_string(bank["bank_kind"], "opening bank kind")
    _require_nonempty_string(bank["claim_scope"], "opening bank claim_scope")
    _require_bool(bank["independent_of_training"], "independent_of_training")
    if bank["source_manifest"] is not None:
        source_manifest = dict(
            _require_mapping(bank["source_manifest"], "source_manifest")
        )
        _require_exact_keys(
            source_manifest,
            required=("path", "sha256"),
            label="source_manifest",
        )
        _require_nonempty_string(source_manifest["path"], "source_manifest.path")
        _require_sha256(source_manifest["sha256"], "source_manifest.sha256")
        bank["source_manifest"] = source_manifest
    openings = bank["openings"]
    if not isinstance(openings, list) or not openings:
        raise FactorialHarnessError("opening bank must contain at least one opening")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_opening in enumerate(openings):
        opening = dict(_require_mapping(raw_opening, f"openings[{index}]"))
        _require_exact_keys(
            opening,
            required=("opening_id", "group_id", "moves"),
            optional=("family_id", "replicate_within_family"),
            label=f"openings[{index}]",
        )
        opening_id = _require_nonempty_string(
            opening["opening_id"], f"openings[{index}].opening_id"
        )
        group_id = _require_nonempty_string(
            opening["group_id"], f"openings[{index}].group_id"
        )
        if opening_id in seen_ids:
            raise FactorialHarnessError(f"duplicate opening id: {opening_id}")
        seen_ids.add(opening_id)
        moves = opening["moves"]
        if not isinstance(moves, list):
            raise FactorialHarnessError(f"opening {opening_id} moves must be a list")
        normalized_moves = [
            _require_int(move, f"opening {opening_id} move", minimum=0)
            for move in moves
        ]
        if any(move >= board_size**2 for move in normalized_moves):
            raise FactorialHarnessError(
                f"opening {opening_id} has an out-of-range move"
            )
        if len(normalized_moves) != len(set(normalized_moves)):
            raise FactorialHarnessError(
                f"opening {opening_id} reuses an occupied point"
            )
        normalized.append(
            {"opening_id": opening_id, "group_id": group_id, "moves": normalized_moves}
        )
    bank["openings"] = sorted(normalized, key=lambda row: row["opening_id"])
    return bank


def _opening_bank_for_profile(
    config: Mapping[str, Any], profile_name: str, repo_root: Path
) -> tuple[Path, dict[str, Any]]:
    profile = config["profiles"][profile_name]
    path = _safe_repo_path(
        repo_root, profile["opening_bank"], label=f"profile {profile_name} opening bank"
    )
    return path, validate_opening_bank(
        load_json_strict(path), game=config["game"], board_size=config["board_size"]
    )


def factors_for_cell(cell_id: str) -> tuple[int, int]:
    try:
        return CELL_FACTORS[cell_id]
    except KeyError as exc:
        raise FactorialHarnessError(f"unknown factorial cell: {cell_id}") from exc


def cell_for_factors(training: int, runtime: int) -> str:
    for cell_id, factors in CELL_FACTORS.items():
        if factors == (training, runtime):
            return cell_id
    raise FactorialHarnessError(
        f"invalid factorial factors: training={training}, runtime={runtime}"
    )


def _deterministic_eval_seed(
    base_seed: int, training_seed: int, opening_id: str, pair_id: str
) -> int:
    payload = f"{base_seed}:{training_seed}:{opening_id}:{pair_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _display_path(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def build_plan(
    config_path: str | Path,
    profile_name: str,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    if profile_name not in config["profiles"]:
        raise FactorialHarnessError(f"unknown factorial profile: {profile_name}")
    profile = config["profiles"][profile_name]
    opening_path, opening_bank = _opening_bank_for_profile(config, profile_name, root)

    agent_contracts: dict[str, dict[str, Any]] = {}
    for seed in profile["seeds"]:
        for cell_id, (training_factor, runtime_factor) in CELL_FACTORS.items():
            training_name = TRAINING_ARM_BY_FACTOR[training_factor]
            runtime_name = RUNTIME_ARM_BY_FACTOR[runtime_factor]
            training_arm = config["training_arms"][training_name]
            runtime_arm = config["runtime_arms"][runtime_name]
            checkpoint_path = _render_seed_template(
                root,
                training_arm["checkpoint_template"],
                seed,
                label=f"{cell_id} checkpoint",
            )
            manifest_path = _render_seed_template(
                root,
                training_arm["manifest_template"],
                seed,
                label=f"{cell_id} training manifest",
            )
            agent_id = f"{cell_id}_s{seed}"
            agent_contracts[agent_id] = {
                "agent_id": agent_id,
                "cell_id": cell_id,
                "training_seed": seed,
                "training_factor": training_factor,
                "runtime_factor": runtime_factor,
                "training_arm": training_name,
                "runtime_arm": runtime_name,
                "checkpoint_path": _display_path(checkpoint_path, root),
                "training_manifest_path": _display_path(manifest_path, root),
                "runtime_contract_hash": _runtime_contract_hash(runtime_arm),
                "runtime_contract": runtime_arm,
            }

    anchor_path = _safe_repo_path(
        root, config["anchor"]["checkpoint"], label="anchor checkpoint"
    )
    anchor_runtime = {
        "controller_mode": config["anchor"]["controller_mode"],
        "live_action_authorized": False,
        "authorized_axis_ids": [],
        "search_config": config["anchor"]["search_config"],
    }
    agent_contracts["ANCHOR"] = {
        "agent_id": "ANCHOR",
        "cell_id": None,
        "training_seed": None,
        "training_factor": None,
        "runtime_factor": 0,
        "training_arm": None,
        "runtime_arm": "anchor",
        "checkpoint_path": _display_path(anchor_path, root),
        "training_manifest_path": None,
        "runtime_contract_hash": _runtime_contract_hash(anchor_runtime),
        "runtime_contract": anchor_runtime,
    }

    pair_specs: list[tuple[str, str, str]] = [
        (f"PRIMARY_{cell_id}_ANCHOR", cell_id, "ANCHOR") for cell_id in CELL_FACTORS
    ]
    if config["match_design"]["include_round_robin"]:
        pair_specs.extend(
            (f"ROUND_ROBIN_{left}_{right}", left, right)
            for left, right in combinations(CELL_FACTORS, 2)
        )

    blocks: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for seed in profile["seeds"]:
        for opening in opening_bank["openings"]:
            for pair_id, left_cell, right_cell in pair_specs:
                left_agent = (
                    "ANCHOR" if left_cell == "ANCHOR" else f"{left_cell}_s{seed}"
                )
                right_agent = (
                    "ANCHOR" if right_cell == "ANCHOR" else f"{right_cell}_s{seed}"
                )
                role = "primary" if pair_id.startswith("PRIMARY_") else "round_robin"
                block_id = f"s{seed}__{opening['opening_id']}__{pair_id}"
                eval_seed = _deterministic_eval_seed(
                    profile["eval_seed_base"], seed, opening["opening_id"], pair_id
                )
                game_specs = []
                for color_index, (black, white) in enumerate(
                    ((left_agent, right_agent), (right_agent, left_agent))
                ):
                    game = {
                        "game_id": f"{block_id}__g{color_index}",
                        "block_id": block_id,
                        "design_role": role,
                        "training_seed": seed,
                        "opening_id": opening["opening_id"],
                        "opening_group_id": opening["group_id"],
                        "opening_moves": list(opening["moves"]),
                        "eval_seed": eval_seed,
                        "black_agent_id": black,
                        "white_agent_id": white,
                    }
                    game_specs.append(game)
                    games.append(game)
                block = {
                    "block_id": block_id,
                    "pair_id": pair_id,
                    "design_role": role,
                    "training_seed": seed,
                    "opening_id": opening["opening_id"],
                    "opening_group_id": opening["group_id"],
                    "eval_seed": eval_seed,
                    "left_agent_id": left_agent,
                    "right_agent_id": right_agent,
                    "game_ids": [row["game_id"] for row in game_specs],
                }
                blocks.append(block)

    source_hashes = []
    for relative_path in (
        "quartz/idea_foundry/metacontroller_factorial.py",
        "scripts/metacontroller_factorial_study.py",
    ):
        source_path = root / relative_path
        if source_path.is_file() and not source_path.is_symlink():
            source_hashes.append(
                {"path": relative_path, "sha256": file_sha256(source_path)}
            )

    plan_contract = {
        "experiment_id": config["experiment_id"],
        "profile": profile_name,
        "config_sha256": file_sha256(config_file),
        "opening_bank_sha256": file_sha256(opening_path),
        "agent_contracts": agent_contracts,
        "blocks": blocks,
        "games": games,
        "analysis_contract": config["analysis_contract"],
        "source_hashes": source_hashes,
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "PLANNED_NOT_EXECUTED",
        "experiment_id": config["experiment_id"],
        "profile": profile_name,
        "claim_scope": config["claim_scope"],
        "created_at": utc_now(),
        "git_head": _git_head(root),
        "python_executable": str(Path(sys.executable).absolute()),
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "config": {
            "path": _display_path(config_file, root),
            "sha256": file_sha256(config_file),
        },
        "opening_bank": {
            "path": _display_path(opening_path, root),
            "sha256": file_sha256(opening_path),
            "bank_id": opening_bank["bank_id"],
            "bank_kind": opening_bank["bank_kind"],
            "claim_scope": opening_bank["claim_scope"],
            "independent_of_training": opening_bank["independent_of_training"],
            "opening_count": len(opening_bank["openings"]),
            "group_count": len(
                {opening["group_id"] for opening in opening_bank["openings"]}
            ),
        },
        "profile_contract": profile,
        "cells": [
            {
                "cell_id": cell_id,
                "training_factor": factors[0],
                "runtime_factor": factors[1],
                "training_arm": TRAINING_ARM_BY_FACTOR[factors[0]],
                "runtime_arm": RUNTIME_ARM_BY_FACTOR[factors[1]],
            }
            for cell_id, factors in CELL_FACTORS.items()
        ],
        "agent_contracts": agent_contracts,
        "blocks": blocks,
        "games": games,
        "counts": {
            "training_seed_count": len(profile["seeds"]),
            "cell_count": len(CELL_FACTORS),
            "agent_count": len(agent_contracts),
            "primary_block_count": sum(
                block["design_role"] == "primary" for block in blocks
            ),
            "round_robin_block_count": sum(
                block["design_role"] == "round_robin" for block in blocks
            ),
            "game_count": len(games),
        },
        "analysis_contract": config["analysis_contract"],
        "source_hashes": source_hashes,
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": list(config["prohibited_inferences"]),
        "plan_contract_hash": canonical_sha256(plan_contract),
    }


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _validate_training_manifest(
    *,
    path: Path,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    training_arm_name: str,
    seed: int,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if not path.is_file() or path.is_symlink():
        return None, [
            _blocker(
                "TRAINING_MANIFEST_MISSING",
                "training treatment manifest is missing",
                path=_display_path(path, repo_root),
            )
        ]
    try:
        manifest = dict(_require_mapping(load_json_strict(path), "training manifest"))
        _require_exact_keys(
            manifest,
            required=(
                "schema_version",
                "experiment_id",
                "training_seed",
                "training_controller_mode",
                "checkpoint_path",
                "checkpoint_sha256",
                "initial_checkpoint_sha256",
                "controller_contract_hash",
                "source_fingerprint",
                "training_trajectory_sha256",
                "learner_updates",
                "sgd_examples",
                "selfplay_nn_evals",
                "selfplay_wallclock_s",
                "metacontroller_observations",
                "metacontroller_actions",
                "status",
            ),
            optional=("telemetry_source",),
            label="training manifest",
        )
        if manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA_VERSION:
            raise FactorialHarnessError("training manifest schema mismatch")
        if manifest["experiment_id"] != config["experiment_id"]:
            raise FactorialHarnessError("training manifest experiment mismatch")
        if manifest["training_seed"] != seed:
            raise FactorialHarnessError("training manifest seed mismatch")
        arm = config["training_arms"][training_arm_name]
        if manifest["training_controller_mode"] != arm["controller_mode"]:
            raise FactorialHarnessError("training controller mode mismatch")
        if manifest["checkpoint_path"] != _display_path(checkpoint_path, repo_root):
            raise FactorialHarnessError("training checkpoint path mismatch")
        if manifest["status"] not in {"succeeded", "completed_no_promotion"}:
            raise FactorialHarnessError("training manifest status is invalid")
        if "telemetry_source" in manifest and manifest["telemetry_source"] != (
            "iteration_trace_totals"
        ):
            raise FactorialHarnessError(
                "training manifest does not use exact iteration trace telemetry"
            )
        for key in (
            "checkpoint_sha256",
            "initial_checkpoint_sha256",
            "controller_contract_hash",
            "source_fingerprint",
            "training_trajectory_sha256",
        ):
            _require_sha256(manifest[key], f"training manifest {key}")
        if manifest["controller_contract_hash"] != _training_controller_hash(
            config, arm
        ):
            raise FactorialHarnessError("training controller contract hash mismatch")
        for key in (
            "learner_updates",
            "sgd_examples",
            "selfplay_nn_evals",
            "metacontroller_observations",
            "metacontroller_actions",
        ):
            _require_int(manifest[key], f"training manifest {key}", minimum=0)
        _require_finite_number(
            manifest["selfplay_wallclock_s"],
            "training manifest selfplay_wallclock_s",
            minimum=0.0,
        )
        if manifest["metacontroller_observations"] <= 0:
            raise FactorialHarnessError(
                "training treatment has no controller observations"
            )
        if training_arm_name == "off" and manifest["metacontroller_actions"] != 0:
            raise FactorialHarnessError("training OFF treatment executed a live action")
        if training_arm_name == "off" and manifest["status"] != "succeeded":
            raise FactorialHarnessError(
                "training OFF treatment cannot be completed_no_promotion"
            )
        if training_arm_name == "on":
            actions = manifest["metacontroller_actions"]
            if actions <= 0 and manifest["status"] != "completed_no_promotion":
                raise FactorialHarnessError(
                    "training ON treatment with no live action must be "
                    "completed_no_promotion"
                )
            if actions > 0 and manifest["status"] != "succeeded":
                raise FactorialHarnessError(
                    "training ON treatment with live actions must be succeeded"
                )
    except FactorialHarnessError as exc:
        blockers.append(
            _blocker(
                "TRAINING_MANIFEST_INVALID",
                str(exc),
                path=_display_path(path, repo_root),
            )
        )
        return None, blockers
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        blockers.append(
            _blocker(
                "CHECKPOINT_MISSING",
                "training checkpoint is missing",
                path=_display_path(checkpoint_path, repo_root),
            )
        )
    elif file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        blockers.append(
            _blocker(
                "CHECKPOINT_HASH_DRIFT",
                "training checkpoint hash does not match its treatment manifest",
                path=_display_path(checkpoint_path, repo_root),
            )
        )
    if manifest["status"] == "completed_no_promotion":
        blockers.append(
            _blocker(
                "TRAINING_TREATMENT_NOT_DELIVERED",
                "active training completed technically but executed no live action",
                path=_display_path(path, repo_root),
                seed=seed,
            )
        )
    return manifest, blockers


def run_preflight(
    config_path: str | Path,
    profile_name: str,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    plan = build_plan(config_file, profile_name, repo_root=root)
    profile = config["profiles"][profile_name]
    blockers: list[dict[str, Any]] = []
    controller = config["controller_contract"]
    for key in ("training_adapter_status", "runtime_adapter_status"):
        if controller[key] != "implemented":
            blockers.append(
                _blocker(
                    "ADAPTER_NOT_IMPLEMENTED",
                    f"{key} is not implemented",
                    field=key,
                    value=controller[key],
                )
            )
    if not controller["production_loop_wired"]:
        blockers.append(
            _blocker(
                "PRODUCTION_LOOP_NOT_WIRED",
                "FoundryCoordinator is not wired into the live MCTS loop",
            )
        )
    if not controller["execution_ready"]:
        blockers.append(
            _blocker(
                "EXECUTION_NOT_AUTHORIZED",
                "controller contract is not execution-ready",
            )
        )
    if not controller["active_axis_ids"]:
        blockers.append(
            _blocker(
                "NO_ACTIVE_AXES",
                "runtime ON treatment has no preregistered active axes",
            )
        )

    opening_path, opening_bank = _opening_bank_for_profile(config, profile_name, root)
    if profile["require_heldout_opening_bank"]:
        if opening_bank["bank_kind"] != "heldout_trajectory_group":
            blockers.append(
                _blocker(
                    "OPENING_BANK_NOT_HELDOUT",
                    "profile requires a held-out trajectory-group opening bank",
                    bank_kind=opening_bank["bank_kind"],
                )
            )
        if not opening_bank["independent_of_training"]:
            blockers.append(
                _blocker(
                    "OPENING_BANK_TRAINING_LEAKAGE",
                    "opening bank is not declared independent of training",
                )
            )
        if opening_bank["source_manifest"] is None:
            blockers.append(
                _blocker(
                    "OPENING_SOURCE_MANIFEST_MISSING",
                    "held-out opening bank requires a source manifest",
                )
            )
    if opening_bank["source_manifest"] is not None:
        try:
            source_contract = opening_bank["source_manifest"]
            source_manifest = _safe_repo_path(
                root,
                source_contract["path"],
                label="opening source manifest",
            )
            if not source_manifest.is_file() or source_manifest.is_symlink():
                blockers.append(
                    _blocker(
                        "OPENING_SOURCE_MANIFEST_MISSING",
                        "opening source manifest is not a regular file",
                        path=_display_path(source_manifest, root),
                    )
                )
            else:
                observed_hash = file_sha256(source_manifest)
                if observed_hash != source_contract["sha256"]:
                    blockers.append(
                        _blocker(
                            "OPENING_SOURCE_MANIFEST_HASH_DRIFT",
                            "opening source manifest SHA-256 drifted",
                            path=_display_path(source_manifest, root),
                            expected=source_contract["sha256"],
                            observed=observed_hash,
                        )
                    )
        except FactorialHarnessError as exc:
            blockers.append(_blocker("OPENING_SOURCE_MANIFEST_INVALID", str(exc)))

    resolved_agents: dict[str, dict[str, Any]] = {}
    manifests_by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for seed in profile["seeds"]:
        for training_name in ("off", "on"):
            arm = config["training_arms"][training_name]
            checkpoint_path = _render_seed_template(
                root,
                arm["checkpoint_template"],
                seed,
                label=f"training {training_name} checkpoint",
            )
            manifest_path = _render_seed_template(
                root,
                arm["manifest_template"],
                seed,
                label=f"training {training_name} manifest",
            )
            manifest, manifest_blockers = _validate_training_manifest(
                path=manifest_path,
                checkpoint_path=checkpoint_path,
                config=config,
                training_arm_name=training_name,
                seed=seed,
                repo_root=root,
            )
            blockers.extend(manifest_blockers)
            if manifest is not None:
                manifests_by_seed[seed][training_name] = manifest

        pair = manifests_by_seed.get(seed, {})
        if set(pair) == {"off", "on"}:
            off = pair["off"]
            on = pair["on"]
            if off["initial_checkpoint_sha256"] != on["initial_checkpoint_sha256"]:
                blockers.append(
                    _blocker(
                        "INITIAL_CHECKPOINT_MISMATCH",
                        "paired training arms do not share the same initial checkpoint",
                        seed=seed,
                    )
                )
            if off["source_fingerprint"] != on["source_fingerprint"]:
                blockers.append(
                    _blocker(
                        "TRAINING_SOURCE_CONTRACT_MISMATCH",
                        "paired training arms do not share the same source/split/schedule fingerprint",
                        seed=seed,
                        off=off["source_fingerprint"],
                        on=on["source_fingerprint"],
                    )
                )
            for key in ("learner_updates", "sgd_examples"):
                if off[key] != on[key]:
                    blockers.append(
                        _blocker(
                            "LEARNER_COMPUTE_MISMATCH",
                            f"paired training arms disagree on {key}",
                            seed=seed,
                            off=off[key],
                            on=on[key],
                        )
                    )
            if (
                profile["training_compute_contract"]
                == "fixed_nn_evals_and_learner_updates"
                and off["selfplay_nn_evals"] != on["selfplay_nn_evals"]
            ):
                blockers.append(
                    _blocker(
                        "SELFPLAY_COMPUTE_MISMATCH",
                        "fixed-NN training arms disagree on self-play NN evaluations",
                        seed=seed,
                        off=off["selfplay_nn_evals"],
                        on=on["selfplay_nn_evals"],
                    )
                )
            if (
                profile["training_compute_contract"]
                == "fixed_wallclock_and_learner_updates"
                and off["selfplay_wallclock_s"] != on["selfplay_wallclock_s"]
            ):
                blockers.append(
                    _blocker(
                        "SELFPLAY_COMPUTE_MISMATCH",
                        "fixed-wallclock training arms disagree on self-play wall time",
                        seed=seed,
                        off=off["selfplay_wallclock_s"],
                        on=on["selfplay_wallclock_s"],
                    )
                )

    anchor_path = _safe_repo_path(
        root, config["anchor"]["checkpoint"], label="anchor checkpoint"
    )
    anchor_hash = None
    expected_anchor_hash = config["anchor"]["checkpoint_sha256"]
    if not anchor_path.is_file() or anchor_path.is_symlink():
        blockers.append(
            _blocker(
                "ANCHOR_CHECKPOINT_MISSING",
                "frozen anchor checkpoint is missing",
                path=_display_path(anchor_path, root),
            )
        )
    else:
        anchor_hash = file_sha256(anchor_path)
        if expected_anchor_hash is None:
            blockers.append(
                _blocker(
                    "ANCHOR_HASH_UNREGISTERED",
                    "frozen anchor SHA-256 is not preregistered",
                    observed=anchor_hash,
                )
            )
        elif anchor_hash != expected_anchor_hash:
            blockers.append(
                _blocker(
                    "ANCHOR_HASH_DRIFT",
                    "frozen anchor checkpoint hash drifted",
                    expected=expected_anchor_hash,
                    observed=anchor_hash,
                )
            )

    for agent_id, agent in plan["agent_contracts"].items():
        checkpoint_path = _safe_repo_path(
            root, agent["checkpoint_path"], label=f"{agent_id} checkpoint"
        )
        if agent_id == "ANCHOR":
            checkpoint_hash = anchor_hash
        else:
            seed = int(agent["training_seed"])
            training_name = str(agent["training_arm"])
            manifest = manifests_by_seed.get(seed, {}).get(training_name)
            checkpoint_hash = (
                None if manifest is None else manifest["checkpoint_sha256"]
            )
        if checkpoint_hash is None:
            continue
        contract = {
            **agent,
            "checkpoint_sha256": checkpoint_hash,
        }
        contract["agent_contract_hash"] = canonical_sha256(contract)
        resolved_agents[agent_id] = contract

    status = "READY" if not blockers else "BLOCKED"
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "ready": not blockers,
        "experiment_id": config["experiment_id"],
        "profile": profile_name,
        "created_at": utc_now(),
        "plan_contract_hash": plan["plan_contract_hash"],
        "config": plan["config"],
        "opening_bank": {
            **plan["opening_bank"],
            "path": _display_path(opening_path, root),
        },
        "blockers": blockers,
        "resolved_agent_contracts": resolved_agents,
        "counts": {
            "blocker_count": len(blockers),
            "resolved_agent_count": len(resolved_agents),
            **plan["counts"],
        },
        "claim_scope": "factorial_execution_readiness_only",
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": list(config["prohibited_inferences"]),
    }


def _validate_preflight_for_analysis(
    preflight: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise FactorialHarnessError("preflight schema mismatch")
    if preflight.get("status") != "READY" or preflight.get("ready") is not True:
        raise FactorialHarnessError("analysis requires a READY preflight artifact")
    if preflight.get("plan_contract_hash") != plan.get("plan_contract_hash"):
        raise FactorialHarnessError("preflight/plan contract hash mismatch")
    agents = preflight.get("resolved_agent_contracts")
    if not isinstance(agents, Mapping) or set(agents) != set(plan["agent_contracts"]):
        raise FactorialHarnessError("preflight agent inventory is incomplete")


def _validate_telemetry(value: Any, label: str) -> dict[str, Any]:
    telemetry = dict(_require_mapping(value, label))
    _require_exact_keys(
        telemetry,
        required=(
            "nn_evals_per_move",
            "root_visits_per_move",
            "wallclock_ms_per_move",
            "metacontroller_decisions",
            "metacontroller_actions",
            "selection_trace_covered",
        ),
        label=label,
    )
    for key in ("nn_evals_per_move", "root_visits_per_move", "wallclock_ms_per_move"):
        telemetry[key] = _require_finite_number(
            telemetry[key], f"{label}.{key}", minimum=0.0
        )
    for key in ("metacontroller_decisions", "metacontroller_actions"):
        telemetry[key] = _require_int(telemetry[key], f"{label}.{key}", minimum=0)
    _require_bool(
        telemetry["selection_trace_covered"], f"{label}.selection_trace_covered"
    )
    return telemetry


def validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    require_complete: bool = True,
    require_treatment_delivery: bool = True,
) -> dict[str, Any]:
    _validate_preflight_for_analysis(preflight, plan)
    expected_games = {str(game["game_id"]): game for game in plan["games"]}
    observed: dict[str, dict[str, Any]] = {}
    agents = preflight["resolved_agent_contracts"]
    active_actions: dict[str, int] = defaultdict(int)
    trace_covered = 0
    trace_total = 0
    budget_values: dict[str, list[float]] = defaultdict(list)
    normalized_rows: list[dict[str, Any]] = []

    for index, raw_row in enumerate(rows):
        row = dict(_require_mapping(raw_row, f"raw row {index}"))
        _require_exact_keys(
            row,
            required=(
                "schema_version",
                "game_id",
                "block_id",
                "design_role",
                "training_seed",
                "opening_id",
                "opening_group_id",
                "eval_seed",
                "black_agent_id",
                "white_agent_id",
                "black_agent_contract_hash",
                "white_agent_contract_hash",
                "status",
                "score_black",
                "black_telemetry",
                "white_telemetry",
            ),
            optional=("error",),
            label=f"raw row {index}",
        )
        if row["schema_version"] != RAW_ROW_SCHEMA_VERSION:
            raise FactorialHarnessError(f"raw row {index} schema mismatch")
        game_id = _require_nonempty_string(row["game_id"], f"raw row {index} game_id")
        if game_id in observed:
            raise FactorialHarnessError(f"duplicate raw game id: {game_id}")
        expected = expected_games.get(game_id)
        if expected is None:
            raise FactorialHarnessError(f"unexpected raw game id: {game_id}")
        for key in (
            "block_id",
            "design_role",
            "training_seed",
            "opening_id",
            "opening_group_id",
            "eval_seed",
            "black_agent_id",
            "white_agent_id",
        ):
            if row[key] != expected[key]:
                raise FactorialHarnessError(
                    f"raw row {game_id} disagrees with plan field {key}"
                )
        if row["status"] != "scored":
            raise FactorialHarnessError(
                f"raw row {game_id} is not scored: {row['status']!r}"
            )
        score_black = _require_finite_number(
            row["score_black"], f"raw row {game_id} score_black", minimum=0.0
        )
        if score_black > 1.0:
            raise FactorialHarnessError(f"raw row {game_id} score exceeds 1")
        row["score_black"] = score_black
        for color in ("black", "white"):
            agent_id = row[f"{color}_agent_id"]
            agent = agents.get(agent_id)
            if not isinstance(agent, Mapping):
                raise FactorialHarnessError(
                    f"raw row {game_id} references unknown agent {agent_id}"
                )
            if row[f"{color}_agent_contract_hash"] != agent["agent_contract_hash"]:
                raise FactorialHarnessError(
                    f"raw row {game_id} {color} agent contract hash drift"
                )
            telemetry = _validate_telemetry(
                row[f"{color}_telemetry"], f"raw row {game_id} {color}_telemetry"
            )
            row[f"{color}_telemetry"] = telemetry
            trace_total += 1
            trace_covered += int(telemetry["selection_trace_covered"])
            mode = agent["runtime_contract"]["controller_mode"]
            if telemetry["metacontroller_decisions"] <= 0:
                raise FactorialHarnessError(
                    f"raw row {game_id} {color} has no controller observation"
                )
            if mode == "shadow_noop" and telemetry["metacontroller_actions"] != 0:
                raise FactorialHarnessError(
                    f"raw row {game_id} {color} shadow controller executed an action"
                )
            if mode == "active":
                active_actions[agent_id] += telemetry["metacontroller_actions"]
            if row["design_role"] == "primary" and agent.get("cell_id") is not None:
                metric = (
                    "wallclock_ms_per_move"
                    if plan["profile_contract"]["budget_lane"] == "fixed_wallclock"
                    else "nn_evals_per_move"
                )
                budget_values[str(agent["cell_id"])].append(float(telemetry[metric]))
        observed[game_id] = row
        normalized_rows.append(row)

    missing = sorted(set(expected_games) - set(observed))
    if require_complete and missing:
        raise FactorialHarnessError(
            f"raw result matrix is incomplete: {len(missing)} games missing"
        )

    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_block[row["block_id"]].append(row)
    for block in plan["blocks"]:
        block_rows = by_block.get(block["block_id"], [])
        if require_complete and len(block_rows) != 2:
            raise FactorialHarnessError(
                f"paired block must contain exactly two games: {block['block_id']}"
            )
        if len(block_rows) < 2:
            continue
        first, second = sorted(block_rows, key=lambda row: row["game_id"])
        if not (
            first["black_agent_id"] == second["white_agent_id"]
            and first["white_agent_id"] == second["black_agent_id"]
        ):
            raise FactorialHarnessError(
                f"paired block does not swap colors: {block['block_id']}"
            )

    active_agent_ids = {
        agent_id
        for agent_id, agent in agents.items()
        if agent["runtime_contract"]["controller_mode"] == "active"
    }
    treatment_delivered = all(
        active_actions.get(agent_id, 0) > 0 for agent_id in active_agent_ids
    )
    if require_complete and require_treatment_delivery:
        for agent_id in sorted(active_agent_ids):
            if active_actions.get(agent_id, 0) <= 0:
                raise FactorialHarnessError(
                    f"active runtime treatment executed no action: {agent_id}"
                )

    coverage = float(trace_covered / trace_total) if trace_total else 0.0
    required_coverage = float(
        plan["analysis_contract"]["required_selection_trace_coverage"]
    )
    if normalized_rows and coverage < required_coverage:
        raise FactorialHarnessError(
            f"selection trace coverage {coverage:.6f} is below {required_coverage:.6f}"
        )

    budget_means = {
        cell_id: mean(values) for cell_id, values in sorted(budget_values.items())
    }
    budget_spread = None
    allowed_spread = float(
        plan["analysis_contract"]["max_realized_budget_relative_spread"]
    )
    if require_complete:
        if set(budget_means) != set(CELL_FACTORS):
            raise FactorialHarnessError("realized budget coverage is incomplete")
        overall_budget = mean(budget_means.values())
        if overall_budget <= 0.0:
            raise FactorialHarnessError("realized budget mean must be positive")
        budget_spread = (
            max(budget_means.values()) - min(budget_means.values())
        ) / overall_budget
        enforce_budget_equality = (
            plan["profile_contract"]["budget_lane"] != "fixed_cap_resource_frontier"
        )
        if enforce_budget_equality and budget_spread > allowed_spread:
            raise FactorialHarnessError(
                f"realized budget relative spread {budget_spread:.6f} exceeds {allowed_spread:.6f}"
            )

    return {
        "rows": sorted(normalized_rows, key=lambda row: row["game_id"]),
        "selection_trace_coverage": coverage,
        "active_action_counts": dict(sorted(active_actions.items())),
        "treatment_delivered": treatment_delivered,
        "realized_budget_means": budget_means,
        "realized_budget_relative_spread": budget_spread,
        "realized_budget_equality_enforced": (
            plan["profile_contract"]["budget_lane"] != "fixed_cap_resource_frontier"
        ),
    }


def _score_for_agent(row: Mapping[str, Any], agent_id: str) -> float:
    if row["black_agent_id"] == agent_id:
        return float(row["score_black"])
    if row["white_agent_id"] == agent_id:
        return 1.0 - float(row["score_black"])
    raise FactorialHarnessError(
        f"agent {agent_id} is absent from game {row['game_id']}"
    )


def _seed_contrasts(cell_scores: Mapping[str, float]) -> dict[str, float]:
    if set(cell_scores) != set(CELL_FACTORS):
        raise FactorialHarnessError("seed cell-score matrix is incomplete")
    m00 = float(cell_scores["M00"])
    m01 = float(cell_scores["M01"])
    m10 = float(cell_scores["M10"])
    m11 = float(cell_scores["M11"])
    return {
        "training_main": 0.5 * ((m10 - m00) + (m11 - m01)),
        "runtime_main": 0.5 * ((m01 - m00) + (m11 - m10)),
        "interaction": m11 - m10 - m01 + m00,
        "joint_stack": m11 - m00,
        "training_at_runtime_off": m10 - m00,
        "training_at_runtime_on": m11 - m01,
        "runtime_at_training_off": m01 - m00,
        "runtime_at_training_on": m11 - m10,
    }


def _t_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise FactorialHarnessError("cannot summarize an empty contrast")
    estimate = mean(values)
    if len(values) == 1:
        return {
            "n_paired_seeds": 1,
            "estimate": estimate,
            "standard_error": None,
            "ci95": None,
            "ci_kind": "unavailable_single_seed",
        }
    standard_error = stdev(values) / math.sqrt(len(values))
    from scipy.stats import t as student_t

    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    half_width = critical * standard_error
    return {
        "n_paired_seeds": len(values),
        "estimate": estimate,
        "standard_error": standard_error,
        "ci95": [estimate - half_width, estimate + half_width],
        "ci_kind": "paired_seed_t_interval_v1",
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_raw_rows(rows, plan=plan, preflight=preflight)
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated["rows"]:
        if row["design_role"] == "primary":
            by_block[row["block_id"]].append(row)

    group_scores: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for block in plan["blocks"]:
        if block["design_role"] != "primary":
            continue
        cell_agent = block["left_agent_id"]
        cell_id = str(cell_agent).split("_s", 1)[0]
        block_rows = by_block[block["block_id"]]
        paired_score = mean(_score_for_agent(row, cell_agent) for row in block_rows)
        group_scores[
            (int(block["training_seed"]), cell_id, str(block["opening_group_id"]))
        ].append(paired_score)

    cell_scores_by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    grouped_by_seed_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (seed, cell_id, _group_id), values in sorted(group_scores.items()):
        grouped_by_seed_cell[(seed, cell_id)].append(mean(values))
    for (seed, cell_id), values in sorted(grouped_by_seed_cell.items()):
        cell_scores_by_seed[seed][cell_id] = mean(values)

    seed_rows: list[dict[str, Any]] = []
    contrast_values: dict[str, list[float]] = defaultdict(list)
    for seed in sorted(cell_scores_by_seed):
        cell_scores = cell_scores_by_seed[seed]
        contrasts = _seed_contrasts(cell_scores)
        seed_rows.append(
            {
                "training_seed": seed,
                "cell_score_rates": dict(sorted(cell_scores.items())),
                "contrasts": contrasts,
            }
        )
        for name, value in contrasts.items():
            contrast_values[name].append(value)

    summaries = {name: _t_summary(contrast_values[name]) for name in PRIMARY_CONTRASTS}

    # Hierarchical / clustered contrast aggregation across (seed, opening_group) pairs
    distinct_seed_groups: set[tuple[int, str]] = {
        (s, g) for (s, _c, g) in group_scores.keys()
    }
    hierarchical_contrast_values: dict[str, list[float]] = defaultdict(list)
    for s, g in sorted(distinct_seed_groups):
        cell_m: dict[str, float] = {}
        for c in ("M00", "M01", "M10", "M11"):
            if (s, c, g) in group_scores and group_scores[(s, c, g)]:
                cell_m[c] = mean(group_scores[(s, c, g)])
        if len(cell_m) == 4:
            g_contrasts = _seed_contrasts(cell_m)
            for name, value in g_contrasts.items():
                hierarchical_contrast_values[name].append(value)

    hierarchical_summaries = {
        name: _t_summary(hierarchical_contrast_values[name])
        for name in PRIMARY_CONTRASTS
        if hierarchical_contrast_values[name]
    }

    minimum = int(plan["profile_contract"]["min_paired_seeds"])
    enough_seeds = len(seed_rows) >= minimum

    delta_q = float(
        plan["analysis_contract"].get("quality_noninferiority_margin", 0.05)
    )
    delta_i = float(
        plan["analysis_contract"].get("interaction_equivalence_margin", 0.05)
    )

    runtime_summary = summaries.get("runtime_main", {})
    interaction_summary = summaries.get("interaction", {})

    quality_noninferiority_passed = None
    if (
        enough_seeds
        and runtime_summary.get("ci95") is not None
        and runtime_summary.get("standard_error") is not None
    ):
        from scipy.stats import t as student_t

        df = max(1, len(seed_rows) - 1)
        t_crit = float(student_t.ppf(0.95, df=df))
        lower_95 = float(runtime_summary["estimate"]) - t_crit * float(
            runtime_summary["standard_error"]
        )
        quality_noninferiority_passed = bool(lower_95 > -delta_q)

    interaction_equivalence_passed = None
    if enough_seeds and interaction_summary.get("ci95") is not None:
        ci = interaction_summary["ci95"]
        interaction_equivalence_passed = bool(ci[0] >= -delta_i and ci[1] <= delta_i)

    compute_reduction_passed = None
    delta_c = None
    if validated.get("realized_budget_means"):
        means = validated["realized_budget_means"]
        if all(k in means for k in ("M00", "M01", "M10", "M11")):
            delta_c = 0.5 * ((means["M01"] - means["M00"]) + (means["M11"] - means["M10"]))
            compute_reduction_passed = bool(
                means["M01"] < means["M00"]
                and means["M11"] < means["M10"]
                and delta_c < 0.0
            )
        elif "M00" in means and "M01" in means:
            compute_reduction_passed = bool(means["M01"] < means["M00"])

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": (
            "COMPLETED_NO_PROMOTION"
            if enough_seeds
            else "COMPLETED_NO_PROMOTION_INSUFFICIENT_SEEDS"
        ),
        "experiment_id": plan["experiment_id"],
        "profile": plan["profile"],
        "claim_scope": (
            "paired_factorial_resource_frontier_analysis_only"
            if plan["profile_contract"]["budget_lane"] == "fixed_cap_resource_frontier"
            else "paired_factorial_analysis_only"
        ),
        "primary_endpoint": plan["analysis_contract"]["primary_endpoint"],
        "primary_contrast": plan["analysis_contract"]["primary_contrast"],
        "effect_scale": plan["analysis_contract"]["effect_scale"],
        "paired_seed_count": len(seed_rows),
        "minimum_paired_seed_count": minimum,
        "paired_seed_gate_passed": enough_seeds,
        "confirmatory_evaluation": {
            "quality_noninferiority_margin": delta_q,
            "quality_noninferiority_criterion": "one-sided 95% lower bound > -delta_q",
            "quality_noninferiority_passed": quality_noninferiority_passed,
            "interaction_equivalence_margin": delta_i,
            "interaction_equivalence_criterion": "95% CI containment within [-delta_i, delta_i]",
            "interaction_equivalence_passed": interaction_equivalence_passed,
            "compute_reduction_delta_nn_evals": delta_c,
            "compute_reduction_passed": compute_reduction_passed,
        },
        "seed_estimates": seed_rows,
        "contrast_summaries": summaries,
        "hierarchical_contrast_summaries": hierarchical_summaries,
        "contract_diagnostics": {
            "selection_trace_coverage": validated["selection_trace_coverage"],
            "active_action_counts": validated["active_action_counts"],
            "realized_budget_means": validated["realized_budget_means"],
            "realized_budget_relative_spread": validated[
                "realized_budget_relative_spread"
            ],
            "realized_budget_equality_enforced": validated[
                "realized_budget_equality_enforced"
            ],
        },
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": list(plan["prohibited_inferences"]),
    }


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
            raise FactorialHarnessError(
                f"analysis output must be a new empty directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_analysis(
    *,
    config_path: str | Path,
    profile_name: str,
    preflight_path: str | Path,
    raw_rows_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    plan = build_plan(config_path, profile_name, repo_root=root)
    preflight = dict(_require_mapping(load_json_strict(preflight_path), "preflight"))
    rows = load_jsonl_strict(raw_rows_path)
    analysis = analyze_rows(rows, plan=plan, preflight=preflight)
    output = Path(output_dir).resolve()
    _ensure_new_directory(output)
    analysis_path = output / "analysis.json"
    seed_path = output / "seed_contrasts.jsonl"
    atomic_json_dump(analysis_path, analysis)
    atomic_jsonl_dump(seed_path, analysis["seed_estimates"])
    manifest = {
        "schema_version": 1,
        "artifact_kind": "metacontroller_factorial_analysis",
        "status": analysis["status"],
        "claim_scope": analysis["claim_scope"],
        "created_at": utc_now(),
        "inputs": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for path in (config_path, preflight_path, raw_rows_path)
        ],
        "sources": [
            {
                "path": str(Path(__file__).resolve().relative_to(root)),
                "sha256": file_sha256(__file__),
            }
        ],
        "artifacts": [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (analysis_path, seed_path)
        ],
        "promotion": {"auto": False, "eligible": False},
    }
    atomic_json_dump(output / "analysis_manifest.json", manifest)
    return analysis


def raw_row_template() -> dict[str, Any]:
    """Return the exact row shape an eventual execution adapter must emit."""

    telemetry = {
        "nn_evals_per_move": 0.0,
        "root_visits_per_move": 0.0,
        "wallclock_ms_per_move": 0.0,
        "metacontroller_decisions": 0,
        "metacontroller_actions": 0,
        "selection_trace_covered": False,
    }
    return {
        "schema_version": RAW_ROW_SCHEMA_VERSION,
        "game_id": "from-plan",
        "block_id": "from-plan",
        "design_role": "primary-or-round_robin",
        "training_seed": 0,
        "opening_id": "from-plan",
        "opening_group_id": "from-plan",
        "eval_seed": 0,
        "black_agent_id": "from-preflight",
        "white_agent_id": "from-preflight",
        "black_agent_contract_hash": "from-preflight",
        "white_agent_contract_hash": "from-preflight",
        "status": "scored",
        "score_black": 0.5,
        "black_telemetry": dict(telemetry),
        "white_telemetry": dict(telemetry),
    }


def training_manifest_template() -> dict[str, Any]:
    """Return the treatment receipt shape required for each training arm."""

    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "experiment_id": "from-config",
        "training_seed": 0,
        "training_controller_mode": "shadow_noop-or-active",
        "checkpoint_path": "repository-relative-path-from-config",
        "checkpoint_sha256": "lowercase-sha256",
        "initial_checkpoint_sha256": "same-within-paired-seed",
        "controller_contract_hash": "derived-from-config",
        "source_fingerprint": "same-input-split-schedule-within-paired-seed",
        "training_trajectory_sha256": "arm-specific-generated-trajectory-hash",
        "learner_updates": 0,
        "sgd_examples": 0,
        "selfplay_nn_evals": 0,
        "selfplay_wallclock_s": 0.0,
        "metacontroller_observations": 0,
        "metacontroller_actions": 0,
        "telemetry_source": "iteration_trace_totals",
        "status": "succeeded-or-completed_no_promotion",
    }
