from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from quartz.experiment_manifest import file_sha256
from quartz.idea_foundry import metacontroller_factorial as factorial
from quartz.idea_foundry import metacontroller_factorial_execution as execution


def _search_config() -> dict[str, object]:
    return {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "halt_mode": "fixed",
        "iters": 64,
        "n_threads": 1,
        "batch_size": 1,
        "batch_timeout_us": 0,
    }


def _opening_payload(count: int = 2) -> dict[str, object]:
    openings = []
    for index in range(count):
        base = index * 4
        openings.append(
            {
                "opening_id": f"O{index:02d}",
                "group_id": f"G{index:02d}",
                "moves": [base, base + 1],
            }
        )
    return {
        "schema_version": 1,
        "bank_id": "test_openings",
        "game": "gomoku7",
        "board_size": 7,
        "bank_kind": "heldout_trajectory_group",
        "claim_scope": "test_fixture",
        "independent_of_training": True,
        "source_manifest": {
            "path": "inputs/opening_source.json",
            "sha256": hashlib.sha256(b"placeholder").hexdigest(),
        },
        "openings": openings,
    }


def _config_payload(seeds: list[int] | None = None) -> dict[str, object]:
    seeds = seeds or [41, 42, 43]
    return {
        "schema_version": 1,
        "experiment_id": "factorial_test",
        "claim_scope": "factorial_design_and_analysis_only",
        "game": "gomoku7",
        "board_size": 7,
        "controller_contract": {
            "implementation": "idea_foundry_coordinator_v1",
            "training_adapter_status": "implemented",
            "runtime_adapter_status": "implemented",
            "production_loop_wired": True,
            "execution_ready": True,
            "active_axis_ids": ["A01"],
        },
        "training_arms": {
            "off": {
                "controller_mode": "shadow_noop",
                "checkpoint_template": "inputs/off/seed_{seed}/latest.pt",
                "manifest_template": "inputs/off/seed_{seed}/training.json",
            },
            "on": {
                "controller_mode": "active",
                "checkpoint_template": "inputs/on/seed_{seed}/latest.pt",
                "manifest_template": "inputs/on/seed_{seed}/training.json",
            },
        },
        "runtime_arms": {
            "off": {
                "controller_mode": "shadow_noop",
                "live_action_authorized": False,
                "authorized_axis_ids": [],
                "search_config": _search_config(),
            },
            "on": {
                "controller_mode": "active",
                "live_action_authorized": True,
                "authorized_axis_ids": ["A01"],
                "search_config": _search_config(),
            },
        },
        "anchor": {
            "agent_id": "ANCHOR",
            "checkpoint": "inputs/anchor.pt",
            "checkpoint_sha256": None,
            "controller_mode": "shadow_noop",
            "search_config": _search_config(),
        },
        "match_design": {
            "primary": "fixed_anchor",
            "include_round_robin": True,
            "color_swap": True,
        },
        "analysis_contract": {
            "primary_endpoint": "paired_score_rate_vs_frozen_anchor",
            "primary_contrast": "interaction",
            "effect_scale": "score_rate_difference",
            "max_realized_budget_relative_spread": 0.01,
            "required_selection_trace_coverage": 1.0,
        },
        "profiles": {
            "test": {
                "seeds": seeds,
                "opening_bank": "inputs/openings.json",
                "budget_lane": "fixed_nn_evals",
                "training_compute_contract": "fixed_nn_evals_and_learner_updates",
                "eval_seed_base": 20260722,
                "min_paired_seeds": 3,
                "require_heldout_opening_bank": True,
            }
        },
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": ["test fixtures do not establish efficacy"],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_execution_config_materializes_hash_bound_run_paths(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path, seeds=[41], opening_count=1)
    anchor = tmp_path / "inputs" / "anchor.pt"
    resolved_path, resolved = execution.materialize_execution_config(
        base_config_path=config_path,
        profile_name="test",
        run_id="factorial-smoke-1",
        anchor_checkpoint=anchor,
        repo_root=tmp_path,
    )

    assert resolved_path.is_file()
    assert resolved["controller_contract"]["execution_ready"] is True
    assert resolved["controller_contract"]["active_axis_ids"] == ["A01.live_pflip_v1"]
    assert resolved["runtime_arms"]["on"]["authorized_axis_ids"] == ["A01.live_pflip_v1"]
    assert resolved["runtime_arms"]["off"]["search_config"]["check_interval"] == 1
    assert (
        "factorial-smoke-1/training/off/seed_{seed}"
        in resolved["training_arms"]["off"]["checkpoint_template"]
    )
    assert (
        factorial.build_plan(resolved_path, "test", repo_root=tmp_path)["counts"][
            "game_count"
        ]
        == 20
    )


def test_execution_config_rejects_run_id_escape(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path, seeds=[41], opening_count=1)
    with pytest.raises(factorial.FactorialHarnessError, match="run id"):
        execution.materialize_execution_config(
            base_config_path=config_path,
            profile_name="test",
            run_id="../escape",
            anchor_checkpoint=tmp_path / "inputs" / "anchor.pt",
            repo_root=tmp_path,
        )


def test_training_telemetry_counts_exact_evaluator_calls(tmp_path: Path) -> None:
    import numpy as np

    output = tmp_path / "training"
    output.mkdir()
    (output / "train_log.jsonl").write_text(
        json.dumps({"train_steps": 3, "selfplay_wallclock_s": 1.25}) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "realized_budget": {"evaluator_calls": 17},
        "controller_summary": {
            "search_policy": {
                "metacontroller_decisions": 4,
                "metacontroller_actions": 1,
            }
        },
    }
    np.savez_compressed(
        output / "replay.npz",
        metadata_json=np.asarray([json.dumps(metadata), json.dumps(metadata)]),
    )

    observed = execution._training_telemetry(output, batch_size=32)

    assert observed == {
        "learner_updates": 3,
        "sgd_examples": 96,
        "selfplay_nn_evals": 34,
        "selfplay_wallclock_s": 1.25,
        "metacontroller_observations": 8,
        "metacontroller_actions": 2,
        "telemetry_source": "bounded_replay_fallback",
    }


def test_training_telemetry_prefers_untruncated_iteration_totals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "training"
    output.mkdir()
    rows = [
        {
            "train_steps": 2,
            "selfplay_wallclock_s": 1.0,
            "selfplay_nn_evals": 70,
            "metacontroller_observations": 8,
            "metacontroller_actions": 0,
        },
        {
            "train_steps": 3,
            "selfplay_wallclock_s": 2.5,
            "selfplay_nn_evals": 90,
            "metacontroller_observations": 11,
            "metacontroller_actions": 2,
        },
    ]
    (output / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    observed = execution._training_telemetry(output, batch_size=32)

    assert observed == {
        "learner_updates": 5,
        "sgd_examples": 160,
        "selfplay_nn_evals": 160,
        "selfplay_wallclock_s": 3.5,
        "metacontroller_observations": 19,
        "metacontroller_actions": 2,
        "telemetry_source": "iteration_trace_totals",
    }


def _prepare_repo(
    root: Path, *, seeds: list[int] | None = None, opening_count: int = 2
) -> tuple[Path, dict[str, object]]:
    config = _config_payload(seeds)
    source_path = root / "inputs" / "opening_source.json"
    _write_json(source_path, {"source": "heldout-test-groups"})
    opening_payload = _opening_payload(opening_count)
    opening_payload["source_manifest"]["sha256"] = file_sha256(source_path)  # type: ignore[index]
    _write_json(root / "inputs" / "openings.json", opening_payload)

    anchor_path = root / "inputs" / "anchor.pt"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_bytes(b"frozen-anchor")
    config["anchor"]["checkpoint_sha256"] = file_sha256(anchor_path)  # type: ignore[index]

    validated = factorial.validate_config(config)
    for seed in validated["profiles"]["test"]["seeds"]:
        initial_hash = hashlib.sha256(f"initial-{seed}".encode()).hexdigest()
        for arm_name in ("off", "on"):
            arm = validated["training_arms"][arm_name]
            checkpoint_path = root / arm["checkpoint_template"].format(seed=seed)
            manifest_path = root / arm["manifest_template"].format(seed=seed)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"checkpoint-{arm_name}-{seed}".encode())
            manifest = {
                "schema_version": 1,
                "experiment_id": validated["experiment_id"],
                "training_seed": seed,
                "training_controller_mode": arm["controller_mode"],
                "checkpoint_path": str(checkpoint_path.relative_to(root)),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "initial_checkpoint_sha256": initial_hash,
                "controller_contract_hash": factorial._training_controller_hash(
                    validated, arm
                ),
                "source_fingerprint": hashlib.sha256(b"source").hexdigest(),
                "training_trajectory_sha256": hashlib.sha256(
                    f"trajectory-{arm_name}-{seed}".encode()
                ).hexdigest(),
                "learner_updates": 10,
                "sgd_examples": 1280,
                "selfplay_nn_evals": 4096,
                "selfplay_wallclock_s": 60.0,
                "metacontroller_observations": 100,
                "metacontroller_actions": 0 if arm_name == "off" else 10,
                "status": "succeeded",
            }
            _write_json(manifest_path, manifest)

    config_path = root / "configs" / "factorial.json"
    _write_json(config_path, config)
    return config_path, config


def _telemetry(agent: dict[str, object]) -> dict[str, object]:
    active = agent["runtime_contract"]["controller_mode"] == "active"  # type: ignore[index]
    return {
        "nn_evals_per_move": 64.0,
        "root_visits_per_move": 64.0,
        "wallclock_ms_per_move": 10.0,
        "metacontroller_decisions": 8,
        "metacontroller_actions": 1 if active else 0,
        "selection_trace_covered": True,
    }


def _raw_rows(
    plan: dict[str, object], preflight: dict[str, object]
) -> list[dict[str, object]]:
    target_scores = {"M00": 0.50, "M01": 0.60, "M10": 0.70, "M11": 0.90}
    agents = preflight["resolved_agent_contracts"]
    rows = []
    for game in plan["games"]:
        black_id = game["black_agent_id"]
        white_id = game["white_agent_id"]
        black = agents[black_id]
        white = agents[white_id]
        if game["design_role"] == "primary":
            cell_agent = black if black_id != "ANCHOR" else white
            cell_id = cell_agent["cell_id"]
            score_cell = target_scores[cell_id]
            score_black = score_cell if black_id != "ANCHOR" else 1.0 - score_cell
        else:
            score_black = 0.5
        rows.append(
            {
                "schema_version": 1,
                "game_id": game["game_id"],
                "block_id": game["block_id"],
                "design_role": game["design_role"],
                "training_seed": game["training_seed"],
                "opening_id": game["opening_id"],
                "opening_group_id": game["opening_group_id"],
                "eval_seed": game["eval_seed"],
                "black_agent_id": black_id,
                "white_agent_id": white_id,
                "black_agent_contract_hash": black["agent_contract_hash"],
                "white_agent_contract_hash": white["agent_contract_hash"],
                "status": "scored",
                "score_black": score_black,
                "black_telemetry": _telemetry(black),
                "white_telemetry": _telemetry(white),
            }
        )
    return rows


def test_plan_has_exact_factorial_cells_and_color_swapped_blocks(
    tmp_path: Path,
) -> None:
    config_path, _ = _prepare_repo(tmp_path, seeds=[41], opening_count=3)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)

    assert {row["cell_id"] for row in plan["cells"]} == set(factorial.CELL_FACTORS)
    assert plan["counts"] == {
        "training_seed_count": 1,
        "cell_count": 4,
        "agent_count": 5,
        "primary_block_count": 12,
        "round_robin_block_count": 18,
        "game_count": 60,
    }
    assert plan["python_executable"] == str(Path(sys.executable).absolute())
    assert (
        plan["plan_contract_hash"]
        == factorial.build_plan(config_path, "test", repo_root=tmp_path)[
            "plan_contract_hash"
        ]
    )
    for block in plan["blocks"]:
        games = [row for row in plan["games"] if row["block_id"] == block["block_id"]]
        assert len(games) == 2
        assert games[0]["black_agent_id"] == games[1]["white_agent_id"]
        assert games[0]["white_agent_id"] == games[1]["black_agent_id"]
        assert games[0]["eval_seed"] == games[1]["eval_seed"]


def test_cell_factor_mapping_roundtrips_for_all_cells() -> None:
    for cell_id, factors in factorial.CELL_FACTORS.items():
        assert (
            factorial.cell_for_factors(*factorial.factors_for_cell(cell_id)) == cell_id
        )
        assert (
            factorial.factors_for_cell(factorial.cell_for_factors(*factors)) == factors
        )


def test_opening_normalization_is_permutation_invariant() -> None:
    payload = _opening_payload(8)
    expected = factorial.validate_opening_bank(payload, game="gomoku7", board_size=7)
    for seed in range(20):
        candidate = copy.deepcopy(payload)
        random.Random(seed).shuffle(candidate["openings"])
        assert (
            factorial.validate_opening_bank(candidate, game="gomoku7", board_size=7)[
                "openings"
            ]
            == expected["openings"]
        )


def test_plan_cardinality_property_across_small_valid_domains(tmp_path: Path) -> None:
    for seed_count in range(1, 4):
        for opening_count in range(1, 5):
            root = tmp_path / f"s{seed_count}_o{opening_count}"
            seeds = list(range(10, 10 + seed_count))
            config_path, _ = _prepare_repo(
                root, seeds=seeds, opening_count=opening_count
            )
            plan = factorial.build_plan(config_path, "test", repo_root=root)
            expected_blocks = seed_count * opening_count * (4 + 6)
            assert len(plan["blocks"]) == expected_blocks
            assert len(plan["games"]) == 2 * expected_blocks
            assert len({row["game_id"] for row in plan["games"]}) == 2 * expected_blocks


def test_config_rejects_runtime_search_drift() -> None:
    payload = _config_payload()
    payload["runtime_arms"]["on"]["search_config"]["iters"] = 65  # type: ignore[index]
    with pytest.raises(factorial.FactorialHarnessError, match="must be identical"):
        factorial.validate_config(payload)


def test_build_plan_rejects_template_path_escape(tmp_path: Path) -> None:
    payload = _config_payload([41])
    payload["training_arms"]["off"]["checkpoint_template"] = (  # type: ignore[index]
        "../escape/seed_{seed}/latest.pt"
    )
    _write_json(tmp_path / "inputs" / "openings.json", _opening_payload())
    config_path = tmp_path / "config.json"
    _write_json(config_path, payload)
    with pytest.raises(factorial.FactorialHarnessError, match="escapes repository"):
        factorial.build_plan(config_path, "test", repo_root=tmp_path)


def test_preflight_accepts_complete_hash_bound_treatments(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "READY"
    assert preflight["blockers"] == []
    assert len(preflight["resolved_agent_contracts"]) == 13


def test_preflight_preserves_undelivered_active_training_as_scientific_blocker(
    tmp_path: Path,
) -> None:
    config_path, config = _prepare_repo(tmp_path)
    for seed in config["profiles"]["test"]["seeds"]:  # type: ignore[index]
        manifest_path = tmp_path / config["training_arms"]["on"][  # type: ignore[index]
            "manifest_template"
        ].format(seed=seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metacontroller_actions"] = 0
        manifest["status"] = "completed_no_promotion"
        _write_json(manifest_path, manifest)

    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "BLOCKED"
    assert {blocker["code"] for blocker in preflight["blockers"]} == {
        "TRAINING_TREATMENT_NOT_DELIVERED"
    }


def test_partial_arena_rows_are_validated_before_resume(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path, seeds=[41], opening_count=1)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    row = _raw_rows(plan, preflight)[0]

    validated = factorial.validate_raw_rows(
        [row],
        plan=plan,
        preflight=preflight,
        require_complete=False,
        require_treatment_delivery=False,
    )

    assert len(validated["rows"]) == 1
    assert validated["realized_budget_relative_spread"] is None
    drifted = copy.deepcopy(row)
    drifted["eval_seed"] += 1
    with pytest.raises(factorial.FactorialHarnessError, match="eval_seed"):
        factorial.validate_raw_rows(
            [drifted],
            plan=plan,
            preflight=preflight,
            require_complete=False,
            require_treatment_delivery=False,
        )


def test_resource_frontier_lane_reports_instead_of_rejecting_compute_savings(
    tmp_path: Path,
) -> None:
    config_path, config = _prepare_repo(tmp_path)
    profile = config["profiles"]["test"]  # type: ignore[index]
    profile["budget_lane"] = "fixed_cap_resource_frontier"
    profile["training_compute_contract"] = (
        "fixed_games_and_learner_updates_resource_frontier"
    )
    _write_json(config_path, config)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    rows = _raw_rows(plan, preflight)
    agents = preflight["resolved_agent_contracts"]
    for row in rows:
        for color in ("black", "white"):
            agent = agents[row[f"{color}_agent_id"]]
            if agent["runtime_contract"]["controller_mode"] == "active":
                row[f"{color}_telemetry"]["nn_evals_per_move"] = 16.0

    validated = factorial.validate_raw_rows(rows, plan=plan, preflight=preflight)
    analysis = factorial.analyze_rows(rows, plan=plan, preflight=preflight)

    assert validated["realized_budget_relative_spread"] > 0.01
    assert validated["realized_budget_equality_enforced"] is False
    assert analysis["claim_scope"] == (
        "paired_factorial_resource_frontier_analysis_only"
    )


def test_campaign_status_reports_hash_bound_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "metacontroller_factorial_runs" / "run-1"
    run_root.mkdir(parents=True)
    _write_json(run_root / "campaign_state.json", {"status": "running"})

    observed = execution.campaign_status("run-1", repo_root=tmp_path)

    assert observed["status"] == "running"
    assert observed["artifacts"]["state"]["exists"] is True
    assert len(observed["artifacts"]["state"]["sha256"]) == 64


def test_resume_reuses_unchanged_preflight_without_timestamp_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _prepare_repo(tmp_path)
    output = tmp_path / "results" / "preflight.json"
    capability = {
        "idea_foundry_feature": True,
        "runtime_adapter": "foundry_search_policy_v1",
        "supported_axis_ids": ["A01"],
        "supported_modes": ["shadow_noop", "active"],
        "path": "target/fake",
        "sha256": hashlib.sha256(b"binary").hexdigest(),
    }
    monkeypatch.setattr(
        execution, "probe_foundry_binary", lambda *_args, **_kw: capability
    )

    first = execution._load_or_prepare_preflight(
        config_path=config_path,
        profile_name="test",
        output_path=output,
        rust_binary="target/fake",
        root=tmp_path,
    )
    first_hash = file_sha256(output)
    second = execution._load_or_prepare_preflight(
        config_path=config_path,
        profile_name="test",
        output_path=output,
        rust_binary="target/fake",
        root=tmp_path,
    )

    assert second == first
    assert file_sha256(output) == first_hash
    checkpoint = tmp_path / config["training_arms"]["off"][  # type: ignore[index]
        "checkpoint_template"
    ].format(seed=41)
    checkpoint.write_bytes(b"drift")
    with pytest.raises(
        factorial.FactorialHarnessError, match="preflight artifact drifted"
    ):
        execution._load_or_prepare_preflight(
            config_path=config_path,
            profile_name="test",
            output_path=output,
            rust_binary="target/fake",
            root=tmp_path,
        )


def test_preflight_fails_closed_on_checkpoint_hash_drift(tmp_path: Path) -> None:
    config_path, config = _prepare_repo(tmp_path)
    checkpoint = tmp_path / config["training_arms"]["on"]["checkpoint_template"].format(  # type: ignore[index]
        seed=41
    )
    checkpoint.write_bytes(b"drifted")

    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "BLOCKED"
    assert "CHECKPOINT_HASH_DRIFT" in {row["code"] for row in preflight["blockers"]}


def test_preflight_fails_closed_on_opening_source_hash_drift(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    (tmp_path / "inputs" / "opening_source.json").write_bytes(b"drifted")

    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "BLOCKED"
    assert "OPENING_SOURCE_MANIFEST_HASH_DRIFT" in {
        row["code"] for row in preflight["blockers"]
    }


def test_preflight_requires_same_training_source_contract(tmp_path: Path) -> None:
    config_path, config = _prepare_repo(tmp_path)
    manifest_path = tmp_path / config["training_arms"]["on"][  # type: ignore[index]
        "manifest_template"
    ].format(seed=41)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_fingerprint"] = hashlib.sha256(b"different-source").hexdigest()
    _write_json(manifest_path, manifest)

    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "BLOCKED"
    assert "TRAINING_SOURCE_CONTRACT_MISMATCH" in {
        row["code"] for row in preflight["blockers"]
    }


def test_preflight_enforces_fixed_wallclock_training_budget(tmp_path: Path) -> None:
    config_path, config = _prepare_repo(tmp_path)
    config["profiles"]["test"]["training_compute_contract"] = (  # type: ignore[index]
        "fixed_wallclock_and_learner_updates"
    )
    _write_json(config_path, config)
    manifest_path = tmp_path / config["training_arms"]["on"][  # type: ignore[index]
        "manifest_template"
    ].format(seed=41)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selfplay_wallclock_s"] = 61.0
    _write_json(manifest_path, manifest)

    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)

    assert preflight["status"] == "BLOCKED"
    assert "SELFPLAY_COMPUTE_MISMATCH" in {row["code"] for row in preflight["blockers"]}


def test_repository_config_is_intentionally_blocked_until_live_wiring() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "metacontroller_factorial.v1.json"
    preflight = factorial.run_preflight(config_path, "smoke", repo_root=root)

    codes = {row["code"] for row in preflight["blockers"]}
    assert preflight["status"] == "BLOCKED"
    assert {
        "ADAPTER_NOT_IMPLEMENTED",
        "PRODUCTION_LOOP_NOT_WIRED",
        "EXECUTION_NOT_AUTHORIZED",
        "NO_ACTIVE_AXES",
        "ANCHOR_CHECKPOINT_MISSING",
        "TRAINING_MANIFEST_MISSING",
    }.issubset(codes)


def test_analysis_recovers_known_seed_level_factorial_contrasts(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    rows = _raw_rows(plan, preflight)

    analysis = factorial.analyze_rows(rows, plan=plan, preflight=preflight)

    assert analysis["status"] == "COMPLETED_NO_PROMOTION"
    assert analysis["paired_seed_count"] == 3
    summaries = analysis["contrast_summaries"]
    assert summaries["training_main"]["estimate"] == pytest.approx(0.25)
    assert summaries["runtime_main"]["estimate"] == pytest.approx(0.15)
    assert summaries["interaction"]["estimate"] == pytest.approx(0.10)
    assert summaries["joint_stack"]["estimate"] == pytest.approx(0.40)
    assert analysis["promotion"] == {"auto": False, "eligible": False}


def test_analysis_rejects_duplicate_or_missing_game_rows(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    rows = _raw_rows(plan, preflight)

    with pytest.raises(factorial.FactorialHarnessError, match="duplicate raw game"):
        factorial.analyze_rows(
            rows + [copy.deepcopy(rows[0])], plan=plan, preflight=preflight
        )
    with pytest.raises(factorial.FactorialHarnessError, match="games missing"):
        factorial.analyze_rows(rows[:-1], plan=plan, preflight=preflight)


def test_analysis_rejects_nominal_active_treatment_without_actions(
    tmp_path: Path,
) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    rows = _raw_rows(plan, preflight)
    for row in rows:
        for color in ("black", "white"):
            agent = preflight["resolved_agent_contracts"][row[f"{color}_agent_id"]]
            if agent["runtime_contract"]["controller_mode"] == "active":
                row[f"{color}_telemetry"]["metacontroller_actions"] = 0

    with pytest.raises(factorial.FactorialHarnessError, match="executed no action"):
        factorial.analyze_rows(rows, plan=plan, preflight=preflight)


def test_analysis_rejects_realized_budget_drift(tmp_path: Path) -> None:
    config_path, _ = _prepare_repo(tmp_path)
    plan = factorial.build_plan(config_path, "test", repo_root=tmp_path)
    preflight = factorial.run_preflight(config_path, "test", repo_root=tmp_path)
    rows = _raw_rows(plan, preflight)
    for row in rows:
        if row["design_role"] != "primary":
            continue
        for color in ("black", "white"):
            agent = preflight["resolved_agent_contracts"][row[f"{color}_agent_id"]]
            if agent.get("cell_id") == "M11":
                row[f"{color}_telemetry"]["nn_evals_per_move"] = 80.0

    with pytest.raises(factorial.FactorialHarnessError, match="budget relative spread"):
        factorial.analyze_rows(rows, plan=plan, preflight=preflight)


def test_strict_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(factorial.FactorialHarnessError, match="non-finite"):
        factorial.load_json_strict(path)


def test_repository_confirmatory_config_and_heldout_opening_bank_are_schema_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "metacontroller_factorial.v1.json"
    cfg = factorial.load_config(config_path)
    assert "confirmatory" in cfg["profiles"]
    plan = factorial.build_plan(config_path, "confirmatory", repo_root=root)
    assert plan["profile"] == "confirmatory"
    assert len(plan["profile_contract"]["seeds"]) == 3
    preflight = factorial.run_preflight(config_path, "confirmatory", repo_root=root)
    # The preflight must not fail on opening bank validation
    blocker_codes = {row["code"] for row in preflight.get("blockers", [])}
    assert "OPENING_BANK_NOT_HELDOUT" not in blocker_codes
    assert "OPENING_BANK_TRAINING_LEAKAGE" not in blocker_codes
    assert "OPENING_BANK_MANIFEST_MISSING" not in blocker_codes
    assert "OPENING_BANK_MANIFEST_DRIFT" not in blocker_codes
