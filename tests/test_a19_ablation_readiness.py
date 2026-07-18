from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from quartz.idea_foundry.a19_ablation import (
    A19PreparationError,
    canonical_json_bytes,
    file_sha256,
    generate_topology,
    load_jsonl_strict,
    load_screen_plan,
    rank_graph_seeds,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "idea_foundry.a19.screen.v1.json"
REPLAY_MANIFEST_PATH = ROOT / "configs" / "idea_foundry.a19.replays.v1.json"


def load_script():
    path = ROOT / "scripts" / "a19_prepare_ablation.py"
    spec = importlib.util.spec_from_file_location("a19_prepare_ablation_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controller(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "frozen_controller.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "axis_id": "A19",
                "artifact_kind": "frozen_controller_contract",
                "immutable_during_ablation": True,
                "is_model_checkpoint": False,
                "game": "gomoku7",
                "controller_identity": {
                    "controller": "QuartzVL",
                    "halt_mode": "VOC",
                    "penalty_mode": "GatedRefresh",
                    "prior_refresh_rate": 0.0,
                    "root_only_shaping": True,
                },
                "runtime_contract": {"n_threads": 4, "batch_size": 4},
                "allowed_mutations": [],
                "prohibited_inferences": ["test-only frozen controller contract"],
            }
        )
    )
    return path, file_sha256(path)


def launch_args(tmp_path: Path, controller_path: Path, controller_hash: str):
    return argparse.Namespace(
        screen_plan=PLAN_PATH,
        replay_manifest=REPLAY_MANIFEST_PATH,
        controller_checkpoint=controller_path,
        controller_sha256=controller_hash,
        output_dir=tmp_path / "a19-launch",
        run_id="a19-launch-test",
        proxy_results=None,
        proxy_results_sha256=None,
    )


def test_topology_is_deterministic_unique_and_density_matched():
    plan = load_screen_plan(PLAN_PATH)
    rows = [generate_topology(seed, plan.architecture) for seed in plan.graph_seeds]
    assert generate_topology(plan.graph_seeds[0], plan.architecture) == rows[0]
    assert len({row["topology_sha256"] for row in rows}) == len(rows)
    assert {len(row["edges"]) for row in rows} == {110}
    for row in rows:
        assert (
            len(
                {
                    edge["dst"]
                    for edge in row["edges"]
                    if edge["src_kind"] == "cell_input"
                }
            )
            == 2
        )
        incoming: dict[tuple[int, int], int] = {}
        for edge in row["edges"]:
            key = (edge["cell"], edge["dst"])
            incoming[key] = incoming.get(key, 0) + 1
            if edge["src_kind"] == "node":
                assert edge["src"] < edge["dst"]
        assert max(incoming.values()) == 3
        assert len(incoming) == 40


def test_topology_resource_invariants_hold_across_many_seed_values():
    """The graph seed may alter identity, never shape, density, or resource estimates."""
    script = load_script()
    plan = load_screen_plan(PLAN_PATH)
    resources = set()
    for seed in [0, 1, 2, 3, 17, 255, 65_535, *range(10_000, 10_057)]:
        topology = generate_topology(seed, plan.architecture)
        assert topology == generate_topology(seed, plan.architecture)
        assert len(topology["edges"]) == 110
        resources.add(canonical_json_bytes(script._resource_estimate(plan, topology)))
    assert len(resources) == 1


def test_launch_contract_consumes_real_replays_without_fabricating_shortlist(tmp_path):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)
    args = launch_args(tmp_path, controller_path, controller_hash)
    assert script.run(args) == 0

    output = args.output_dir
    assert (output / "diagnostic.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    interpretation = (output / "diagnostic_interpretation.md").read_text(
        encoding="utf-8"
    )
    assert "DIAGNOSTIC" in interpretation
    assert "does not show" in interpretation.lower()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    shortlist = json.loads(
        (output / "a19_graph_seed_shortlist.v1.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (output / "a19_ablation_contract.v1.json").read_text(encoding="utf-8")
    )
    split_contract = json.loads(
        (output / "a19_split_contract.v1.json").read_text(encoding="utf-8")
    )
    rows = load_jsonl_strict(output / "rows.jsonl")

    assert summary["execution_status"] == "completed_no_promotion"
    assert summary["outcome_detail"] == "READY_FOR_PROXY_LAUNCH_NO_SHORTLIST"
    assert summary["real_replay_sources_validated"] == 3
    assert summary["shortlist_status"] == "not_measured"
    assert summary["shortlisted_graph_seeds"] == []
    assert summary["promotion"]["eligible"] is False
    assert manifest["axis_id"] == "A19"
    assert manifest["role"] == "ablation_readiness"
    assert manifest["evidence_status"] == "skeleton_only"
    assert manifest["claim_scope"] == "ablation_readiness_only"
    assert manifest["source_hashes"]
    assert manifest["input_hashes"]
    assert manifest["seed_contract"]["paired_replicate_seeds"] == [41, 42, 43]
    assert shortlist["status"] == "not_measured"
    assert shortlist["shortlisted_graph_seeds"] == []
    assert contract["controller_contract"]["mutation_allowed"] is False
    assert contract["role"] == "ablation_readiness"
    assert contract["evidence_status"] == "skeleton_only"
    assert contract["proxy_trainer"]["measured_finalize_enabled"] is False
    assert summary["proxy_trainer_status"] == "absent_not_implemented"
    assert summary["split_contracts_persisted"] is True
    assert summary["blockers"]
    assert all(
        source["source_checkpoint"]["status"] == "trained_bootstrap_non_promoted"
        for source in contract["replay_sources"]
    )
    assert len(rows) == 8
    assert {row["resources"]["topology_edges"] for row in rows} == {110}
    assert len({canonical_json_bytes(row["resources"]) for row in rows}) == 1
    assert len(split_contract["contracts"]) == 3
    for row in split_contract["contracts"]:
        assert len(row["train_state_group_hashes"]) == 160
        assert len(row["validation_state_group_hashes"]) == 48
        assert set(row["train_state_group_hashes"]).isdisjoint(
            row["validation_state_group_hashes"]
        )
        assert len(row["train_split_sha256"]) == 64
        assert len(row["validation_split_sha256"]) == 64
        assert len(row["batch_schedule_sha256"]) == 64


def test_exact_retry_is_idempotent_and_controller_hash_drift_fails(tmp_path):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)
    args = launch_args(tmp_path, controller_path, controller_hash)
    assert script.run(args) == 0
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in args.output_dir.iterdir()
    }
    assert script.run(args) == 0
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in args.output_dir.iterdir()
    }
    assert after == before

    drift_args = launch_args(tmp_path / "other", controller_path, "0" * 64)
    with pytest.raises(A19PreparationError, match="hash mismatch"):
        script.run(drift_args)


def test_existing_output_rejects_incomplete_and_artifact_drift(tmp_path):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)

    incomplete_args = launch_args(
        tmp_path / "incomplete", controller_path, controller_hash
    )
    assert script.run(incomplete_args) == 0
    (incomplete_args.output_dir / "diagnostic.png").unlink()
    with pytest.raises(A19PreparationError, match="existing output is incomplete"):
        script.run(incomplete_args)

    drift_args = launch_args(
        tmp_path / "artifact-drift", controller_path, controller_hash
    )
    assert script.run(drift_args) == 0
    summary_path = drift_args.output_dir / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    with pytest.raises(A19PreparationError, match="existing artifact size drift"):
        script.run(drift_args)


def test_existing_output_rejects_run_id_source_and_input_hash_drift(
    tmp_path, monkeypatch
):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)
    args = launch_args(tmp_path, controller_path, controller_hash)
    assert script.run(args) == 0

    wrong_run_id = argparse.Namespace(**vars(args))
    wrong_run_id.run_id = "a19-other-run"
    with pytest.raises(A19PreparationError, match="identity drift for run_id"):
        script.run(wrong_run_id)

    original_source_hashes = script._source_hashes
    monkeypatch.setattr(
        script,
        "_source_hashes",
        lambda: [{"path": "scripts/a19_prepare_ablation.py", "sha256": "0" * 64}],
    )
    with pytest.raises(A19PreparationError, match="source hash drift"):
        script.run(args)
    monkeypatch.setattr(script, "_source_hashes", original_source_hashes)

    controller_payload = json.loads(controller_path.read_text(encoding="utf-8"))
    controller_payload["prohibited_inferences"].append("input drift test")
    controller_path.write_bytes(canonical_json_bytes(controller_payload))
    drifted_input = argparse.Namespace(**vars(args))
    drifted_input.controller_sha256 = file_sha256(controller_path)
    with pytest.raises(A19PreparationError, match="input hash drift"):
        script.run(drifted_input)


def test_existing_output_rejects_mode_and_shortlist_status_drift(tmp_path):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)
    args = launch_args(tmp_path, controller_path, controller_hash)
    assert script.run(args) == 0

    manifest_path = args.output_dir / "run_manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["execution_mode"] = "measured_fixed_replay_proxy"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(A19PreparationError, match="identity drift for execution_mode"):
        script.run(args)
    manifest_path.write_bytes(original_manifest)

    shortlist_path = args.output_dir / "a19_graph_seed_shortlist.v1.json"
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    shortlist["status"] = "measured_proxy_shortlist"
    shortlist_path.write_bytes(canonical_json_bytes(shortlist))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        row
        for row in manifest["artifacts"]
        if row["path"] == "a19_graph_seed_shortlist.v1.json"
    )
    artifact["size_bytes"] = shortlist_path.stat().st_size
    artifact["sha256"] = file_sha256(shortlist_path)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(A19PreparationError, match="identity drift for status"):
        script.run(args)


def test_measured_finalize_is_fail_closed_until_proxy_executor_exists(tmp_path):
    script = load_script()
    controller_path, controller_hash = controller(tmp_path)
    proxy = tmp_path / "opaque-proxy.jsonl"
    proxy.write_text("{}\n", encoding="utf-8")
    args = launch_args(tmp_path, controller_path, controller_hash)
    args.output_dir = tmp_path / "a19-measured"
    args.run_id = "a19-measured-test"
    args.proxy_results = proxy
    args.proxy_results_sha256 = file_sha256(proxy)
    with pytest.raises(A19PreparationError, match="PROXY_EXECUTOR_NOT_IMPLEMENTED"):
        script.run(args)
    assert not args.output_dir.exists()


def test_rank_substrate_is_order_independent_but_does_not_emit_artifacts():
    plan = load_screen_plan(PLAN_PATH)
    rows = []
    for graph_index, graph_seed in enumerate(plan.graph_seeds):
        for replicate_index, replicate_seed in enumerate(plan.replicate_seeds):
            rows.append(
                {
                    "graph_seed": graph_seed,
                    "replicate_seed": replicate_seed,
                    "metrics": {
                        "policy_kl": 0.1 + graph_index * 0.01 + replicate_index * 0.001,
                        "value_mse": 0.2 + graph_index * 0.02 + replicate_index * 0.001,
                    },
                    "topology_sha256": "0" * 64,
                    "controller_sha256": "1" * 64,
                    "replay_corpus_sha256": "2" * 64,
                    "replay_source_sha256": "3" * 64,
                    "source_checkpoint_sha256": "4" * 64,
                    "train_split_sha256": "5" * 64,
                    "validation_split_sha256": "6" * 64,
                    "batch_schedule_sha256": "7" * 64,
                    "evaluator_checkpoint": {},
                    "candidate_receipt": {},
                    "budget": {},
                    "resources": {},
                }
            )
    first, _ = rank_graph_seeds(rows, plan)
    second, _ = rank_graph_seeds(list(reversed(rows)), plan)
    assert first == second
    assert [row["graph_seed"] for row in first[:3]] == [1901, 1902, 1903]


def test_proxy_json_rejects_non_finite_and_duplicate_keys(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"x": NaN}\n', encoding="utf-8")
    with pytest.raises(A19PreparationError, match="non-finite"):
        load_jsonl_strict(bad)
    bad.write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
    with pytest.raises(A19PreparationError, match="duplicate JSON key"):
        load_jsonl_strict(bad)
