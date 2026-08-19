from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from quartz.idea_foundry.a19_ablation import generate_topology, load_screen_plan
from quartz.idea_foundry.a19_proxy import _split_contract, build_model, operator_trace
from quartz.idea_foundry.studies import (
    STUDY_REGISTRY,
    execute_inprocess,
    load_study_specs,
    publish_outcome,
    study_plan,
)
from quartz.experiment_manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PHASE15_A4_MINIMAL = FIXTURES / "idea_foundry_phase15_a4_minimal.jsonl"
STAGE7_A4_B13_MINIMAL = FIXTURES / "idea_foundry_stage7_a4_b13_minimal.jsonl"
POSITION_SUITE_MINIMAL = FIXTURES / "idea_foundry_position_suite_minimal.json"
FROZEN_V1 = "legacy scientific study schema v1 is frozen; inspection only"
FROZEN_CAMPAIGN = f"IDEA FOUNDRY CAMPAIGN BLOCKED: {FROZEN_V1}"


def _assert_fixture_provenance(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    rows = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else [json.loads(text)]
    )
    assert rows and all(
        row.get("fixture_provenance") == "synthetic_unit_fixture" for row in rows
    )


def _patch_compact_phase15_a4_grid(monkeypatch) -> None:
    from quartz.idea_foundry import studies

    _assert_fixture_provenance(PHASE15_A4_MINIMAL)
    monkeypatch.setattr(studies, "PHASE15_ROWS", PHASE15_A4_MINIMAL)


def _patch_compact_stage7_a4_b13_grid(monkeypatch) -> None:
    from quartz.idea_foundry import studies

    _assert_fixture_provenance(STAGE7_A4_B13_MINIMAL)
    monkeypatch.setattr(studies, "STAGE7_ROWS", STAGE7_A4_B13_MINIMAL)


def _patch_compact_position_suite(monkeypatch) -> None:
    from quartz.idea_foundry import studies

    _assert_fixture_provenance(POSITION_SUITE_MINIMAL)
    monkeypatch.setattr(studies, "POSITION_SUITE", POSITION_SUITE_MINIMAL)


def _load_campaign_runner():
    script_path = ROOT / "scripts" / "idea_foundry_study_all.py"
    module_spec = importlib.util.spec_from_file_location(
        "idea_foundry_study_all_for_test", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _load_axis_study_runner():
    script_path = ROOT / "scripts" / "idea_foundry_study.py"
    module_spec = importlib.util.spec_from_file_location(
        "idea_foundry_study_for_test", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_study_registry_covers_26_axes_in_order_with_positive_estimates():
    specs = load_study_specs()
    assert [spec.axis_id for spec in specs] == [
        f"A{index:02d}" for index in range(1, 27)
    ]
    assert all(spec.pilot_seconds > 0 and spec.full_seconds > 0 for spec in specs)
    plan = study_plan()
    assert plan["axis_count"] == 26
    assert plan["promotion"]["eligible"] is False
    assert plan["estimated_seconds"]["full"] > plan["estimated_seconds"]["pilot"]


def test_representative_trace_synthetic_conditional_and_exact_recipes_are_distinct(
    monkeypatch,
):
    _patch_compact_phase15_a4_grid(monkeypatch)
    _patch_compact_stage7_a4_b13_grid(monkeypatch)
    _patch_compact_position_suite(monkeypatch)
    trace = execute_inprocess("A01", "pilot", 20260719)
    synthetic = execute_inprocess("A06", "pilot", 20260719)
    conditional = execute_inprocess("A10", "pilot", 20260719)
    analysis = execute_inprocess("A17", "pilot", 20260719)
    parity = execute_inprocess("A23", "pilot", 20260719)
    exact = execute_inprocess("A26", "pilot", 20260719)

    assert trace.status == "completed_no_promotion" and len(trace.grouped_effects) == 3
    assert synthetic.status == "completed_no_promotion" and len(synthetic.rows) == 72
    assert conditional.status == "skipped"
    assert conditional.outcome_detail == "DORMANT_NO_ELIGIBLE_SLICE"
    assert analysis.outcome_detail == "REAL_TRACE_ANALYSIS_ONLY_B13_GATE_COMPLETED"
    assert all(row["make_unmake_exact"] for row in parity.rows)
    assert max(row["paired_effect"] for row in exact.rows) <= 1e-12


def test_legacy_publish_and_inprocess_run_are_frozen_before_work(tmp_path, monkeypatch):
    from quartz.idea_foundry import studies

    output = tmp_path / "A02"
    outcome = studies.StudyOutcome([], {})
    common = dict(profile="pilot", seed=20260719, output_dir=output)
    monkeypatch.setattr(studies, "_ensure_output", pytest.fail)
    with pytest.raises(studies.StudyError, match=FROZEN_V1):
        publish_outcome(axis_id="A02", outcome=outcome, **common)
    monkeypatch.setattr(studies, "execute_inprocess", pytest.fail)
    with pytest.raises(studies.StudyError, match=FROZEN_V1):
        studies.run_inprocess_study("A02", entrypoint=Path(__file__), **common)
    assert not output.exists()


def test_a19_split_and_parameter_contracts_are_seed_deterministic():
    hashes = [f"{index:064x}" for index in range(80)]
    first, first_schedule = _split_contract(
        hashes,
        replicate_seed=41,
        train_positions=32,
        validation_positions=16,
        optimizer_steps=2,
        batch_size=8,
    )
    second, second_schedule = _split_contract(
        list(reversed(hashes)),
        replicate_seed=41,
        train_positions=32,
        validation_positions=16,
        optimizer_steps=2,
        batch_size=8,
    )
    assert first == second
    assert first_schedule == second_schedule
    assert set(first["train_state_group_hashes"]).isdisjoint(
        first["validation_state_group_hashes"]
    )

    plan = load_screen_plan(ROOT / "configs" / "idea_foundry.a19.screen.v1.json")
    models = [
        build_model(generate_topology(seed, plan.architecture), channels=8)
        for seed in plan.graph_seeds[:2]
    ]
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters()) for model in models
    }
    assert len(parameter_counts) == 1
    states = np.zeros((1, 17, 7, 7), dtype=np.float32)
    import torch

    with torch.no_grad():
        policy, value = models[0](torch.from_numpy(states))
    assert policy.shape == (1, 49)
    assert value.shape == (1,)


def test_a19_full_model_matches_registered_parameter_and_flop_estimator():
    plan = load_screen_plan(ROOT / "configs" / "idea_foundry.a19.screen.v1.json")
    topology = generate_topology(plan.graph_seeds[0], plan.architecture)
    model = build_model(topology, channels=int(plan.architecture["channels"]))
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    trace = operator_trace(topology, int(plan.architecture["channels"]), 1901, 41)

    script_path = ROOT / "scripts" / "a19_prepare_ablation.py"
    module_spec = importlib.util.spec_from_file_location(
        "a19_prepare_for_test", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    expected = module._resource_estimate(plan, topology)

    assert actual_parameters == expected["parameters"]
    assert trace["total_flops"] == expected["flops"]


def test_study_registry_json_is_strictly_versioned():
    payload = json.loads(STUDY_REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["suite"] == "first-scientific-gate-all"
    assert payload["prohibited_inferences"]


def test_legacy_historical_artifact_validator_is_read_only(
    tmp_path,
):
    runner = _load_campaign_runner()
    runner.REPO_ROOT = tmp_path
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    nested_input = tmp_path / "input.json"
    nested_input.write_text("{}\n", encoding="utf-8")
    nested_dir = tmp_path / "results" / "native"
    nested_dir.mkdir(parents=True)
    nested_artifact = nested_dir / "rows.jsonl"
    nested_artifact.write_text("{}\n", encoding="utf-8")
    nested_manifest = nested_dir / "run_manifest.json"
    nested_manifest.write_text(
        json.dumps(
            {
                "source_hashes": [{"path": "source.py", "sha256": file_sha256(source)}],
                "input_hashes": [
                    {"path": "input.json", "sha256": file_sha256(nested_input)}
                ],
                "artifacts": [
                    {
                        "path": "rows.jsonl",
                        "sha256": file_sha256(nested_artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    axis_dir = tmp_path / "results" / "campaign" / "axes" / "A19"
    axis_dir.mkdir(parents=True)
    summary = axis_dir / "summary.json"
    summary.write_text(
        json.dumps({"execution_status": "completed_no_promotion"}),
        encoding="utf-8",
    )
    outer_artifact = axis_dir / "rows.jsonl"
    outer_artifact.write_text("{}\n", encoding="utf-8")
    (axis_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_hashes": [],
                "input_hashes": [
                    {
                        "path": "results/native/run_manifest.json",
                        "sha256": file_sha256(nested_manifest),
                    }
                ],
                "artifacts": [
                    {"path": "rows.jsonl", "sha256": file_sha256(outer_artifact)}
                ],
            }
        ),
        encoding="utf-8",
    )

    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert runner._validate_artifact_set(axis_dir) == "completed_no_promotion"
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before
    source.write_text("drift\n", encoding="utf-8")
    assert runner._validate_artifact_set(axis_dir) is None


def test_recovered_axis_clears_stale_blocker():
    runner = _load_campaign_runner()
    axis = {"status": "failed", "blocker": "returncode=2"}

    assert runner._recover_axis(axis, "completed_no_promotion") is True
    assert axis["status"] == "completed_no_promotion"
    assert "blocker" not in axis


def test_legacy_direct_execution_helpers_are_frozen_before_any_work(
    tmp_path, monkeypatch
):
    runner, campaign = _load_axis_study_runner(), _load_campaign_runner()
    native_dir = tmp_path / "A19.native"
    native_dir.mkdir()
    marker = native_dir / "marker"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(runner, "_validate_artifact_set", pytest.fail)
    with pytest.raises(runner.StudyError, match=FROZEN_V1):
        runner._run_or_reuse_native(["executor"], native_dir)
    monkeypatch.setattr(runner, "_validate_artifact_set", lambda _path: None)
    monkeypatch.setattr(runner, "_archive_incomplete_output", pytest.fail)
    monkeypatch.setattr(runner, "_run_command", pytest.fail)
    with pytest.raises(runner.StudyError, match=FROZEN_V1):
        runner._run_or_reuse_native(["executor"], native_dir)
    fresh = tmp_path / "fresh.native"
    with pytest.raises(runner.StudyError, match=FROZEN_V1):
        runner._run_or_reuse_native(["executor"], fresh)
    monkeypatch.setattr(runner, "_run_or_reuse_native", pytest.fail)
    native_options = dict(profile="pilot", seed=1, output_dir=tmp_path / "A15")
    with pytest.raises(runner.StudyError, match=FROZEN_V1):
        runner._run_native("A15", **native_options)
    monkeypatch.setattr(campaign.subprocess, "Popen", pytest.fail)
    logs = tmp_path / "logs"
    paths = tmp_path / "A01", logs / "stdout", logs / "stderr"
    axis_options = dict(axis_id="A01", profile="pilot", seed=1, timeout_seconds=1)
    axis_options.update(zip(("output_dir", "stdout_path", "stderr_path"), paths))
    with pytest.raises(campaign.StudyError, match=FROZEN_V1):
        campaign._run_axis(**axis_options)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "A19.native.incomplete-attempt-1").exists()
    assert not fresh.exists() and not (tmp_path / "A15.native").exists()
    assert not logs.exists()


def test_legacy_routes_are_frozen_or_read_only(tmp_path, monkeypatch, capsys):
    from scripts import idea_foundry_study_analyze as analyzer

    axis_runner, campaign_runner = _load_axis_study_runner(), _load_campaign_runner()
    monkeypatch.setattr(axis_runner, "study_spec", pytest.fail)
    output = tmp_path / "output"
    assert axis_runner.main(["run", "--axis", "A01", "--output-dir", str(output)]) == 2
    assert FROZEN_V1 in capsys.readouterr().err
    assert axis_runner.main(["plan", "--json"]) == 0 and not output.exists()

    campaign_runner.CAMPAIGN_ROOT = root = tmp_path / "results" / "idea_foundry_studies"
    with monkeypatch.context() as patch:
        patch.setattr(campaign_runner, "_safe_run_root", pytest.fail)
        for command in ("run", "resume"):
            for timeout in ("0", "-1"):
                args = [command, "--run-id", "x", "--timeout-multiplier", timeout]
                assert campaign_runner.main(args) == 2
                assert capsys.readouterr().err.strip() == FROZEN_CAMPAIGN
    assert not root.exists()
    campaign_runner.REPO_ROOT = tmp_path
    run_root = root / "x"
    run_root.mkdir(parents=True)
    state = run_root / "campaign_state.json"
    payload = (
        '{"schema_version":1,"run_id":"x","suite":"legacy","profile":"full",'
        '"status":"completed_no_promotion","axes":[]}'
    )
    state.write_text(payload, encoding="utf-8")
    before = state.read_bytes()
    assert campaign_runner.main(["status", "--run-id", "x"]) == 0
    assert state.read_bytes() == before and list(run_root.iterdir()) == [state]
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed_no_promotion",
                "axes": [{"axis_id": f"A{index:02d}"} for index in range(1, 27)],
            }
        ),
        encoding="utf-8",
    )
    summary, target = run_root / "campaign_summary.json", tmp_path / "analysis"
    summary.write_text('{"status":"completed_no_promotion"}', encoding="utf-8")
    before = (state.read_bytes(), summary.read_bytes())
    with pytest.raises(analyzer.StudyError, match=FROZEN_V1):
        analyzer.analyze_campaign(run_root, target)
    assert not target.exists()
    assert (state.read_bytes(), summary.read_bytes()) == before
