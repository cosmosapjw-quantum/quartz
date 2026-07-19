from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np

from quartz.idea_foundry.a19_ablation import generate_topology, load_screen_plan
from quartz.idea_foundry.a19_proxy import _split_contract, build_model, operator_trace
from quartz.idea_foundry.meta_analysis import validate_effect_record
from quartz.idea_foundry.studies import (
    STUDY_REGISTRY,
    execute_inprocess,
    load_study_specs,
    publish_outcome,
    study_plan,
)
from quartz.experiment_manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]


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


def test_representative_trace_synthetic_conditional_and_exact_recipes_are_distinct():
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


def test_published_effect_records_are_meta_schema_valid(tmp_path, monkeypatch):
    from quartz.idea_foundry import studies

    monkeypatch.setattr(
        studies,
        "_ensure_output",
        lambda path: path.mkdir(parents=True, exist_ok=True) or path,
    )
    outcome = execute_inprocess("A02", "pilot", 20260719)
    output = tmp_path / "A02"
    summary = publish_outcome(
        axis_id="A02",
        profile="pilot",
        seed=20260719,
        output_dir=output,
        outcome=outcome,
        extra_sources=(Path(__file__).resolve(),),
    )
    assert summary["execution_status"] == "completed_no_promotion"
    records = [
        json.loads(line)
        for line in (output / "effect_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 3
    assert all(validate_effect_record(record) for record in records)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion"]["eligible"] is False


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


def test_campaign_resume_recursively_validates_nested_source_and_input_hashes(
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

    assert runner._validate_artifact_set(axis_dir) == "completed_no_promotion"
    source.write_text("drift\n", encoding="utf-8")
    assert runner._validate_artifact_set(axis_dir) is None


def test_recovered_axis_clears_stale_blocker():
    runner = _load_campaign_runner()
    axis = {"status": "failed", "blocker": "returncode=2"}

    assert runner._recover_axis(axis, "completed_no_promotion") is True
    assert axis["status"] == "completed_no_promotion"
    assert "blocker" not in axis


def test_native_recovery_reuses_valid_and_archives_incomplete_outputs(
    tmp_path, monkeypatch
):
    runner = _load_axis_study_runner()
    native_dir = tmp_path / "A19.native"
    native_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        runner,
        "_validate_artifact_set",
        lambda path: "completed_no_promotion" if (path / "valid").exists() else None,
    )
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda command, path: (
            calls.append(tuple(command)),
            path.mkdir(),
            (path / "valid").touch(),
        ),
    )

    (native_dir / "valid").touch()
    assert runner._run_or_reuse_native(["executor"], native_dir) == "verified_reuse"
    assert calls == []

    (native_dir / "valid").unlink()
    assert runner._run_or_reuse_native(["executor"], native_dir) == "executed"
    assert calls == [("executor",)]
    assert (tmp_path / "A19.native.incomplete-attempt-1").is_dir()
    assert (native_dir / "valid").is_file()
