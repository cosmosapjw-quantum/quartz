from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import quartz.idea_foundry.a18_ablation as a18_module
import scripts.a18_prepare_holdouts as holdout_script
from quartz.idea_foundry.a18_ablation import (
    A18ContractError,
    A18MatchedEvaluator,
    BASELINE_VARIANT,
    DIFFUSION_VARIANT,
    analyze_candidates,
    derive_state_disjoint_evaluation_replay,
    estimate_direct_flops,
    estimate_training_forward_flops,
    inspect_inputs,
    load_spec,
    parameter_breakdown,
    replay_state_hash_contract,
    run_smoke_or_study,
    sha256_file,
    train_candidates,
)
from quartz.models_torch import AlphaZeroNet
from quartz.replay import ReplayBuffer


def tiny_cfg() -> dict:
    return {
        "board": 3,
        "ch": 2,
        "actions": 9,
        "filters": 4,
        "blocks": 1,
        "vh": 4,
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "prior_refresh_rate": 0.0,
        "prior_refresh_temp": 1.0,
        "c_puct": 2.0,
        "iters": 8,
        "n_threads": 1,
        "batch_size": 2,
    }


def write_inputs(tmp_path: Path, *, use_latest: bool = True) -> Path:
    cfg = tiny_cfg()
    seed_dir = tmp_path / "bootstrap" / "seed_7"
    seed_dir.mkdir(parents=True)
    torch.manual_seed(7)
    direct = AlphaZeroNet(cfg)
    checkpoint_name = "latest.pt" if use_latest else "best.pt"
    checkpoint = seed_dir / checkpoint_name
    torch.save({"model_state_dict": direct.state_dict(), "cfg": cfg}, checkpoint)
    status = seed_dir / "checkpoint_status.json"
    status.write_text(
        json.dumps(
            {
                "latest_exists": True,
                "best_checkpoint_bootstrap_seeded": True,
                "preferred_posttrain_checkpoint": "latest.pt",
            }
        ),
        encoding="utf-8",
    )
    replay_path = seed_dir / "replay.npz"
    replay = ReplayBuffer(32)
    for index in range(8):
        state = np.zeros((2, 3, 3), dtype=np.float32)
        state[index % 2, index % 3, (index // 3) % 3] = 1.0
        policy = np.full(9, 1.0 / 9.0, dtype=np.float32)
        replay.add(
            state,
            policy,
            (-1.0) ** index,
            metadata={"controller_summary": {"root_only_shaping": True}},
        )
    replay.save(replay_path)
    spec = {
        "schema_version": 1,
        "axis_id": "A18",
        "study_id": "tiny-a18-smoke",
        "evidence_tier": "smoke_readiness",
        "controller_frozen": True,
        "controller_contract": {
            key: cfg[key]
            for key in (
                "search_profile",
                "vl_mode",
                "penalty_mode",
                "prior_refresh_rate",
                "prior_refresh_temp",
                "c_puct",
                "iters",
                "n_threads",
                "batch_size",
            )
        }
        | {"root_only_shaping": True},
        "compute_contract": {
            "learner_updates": 1,
            "batch_size": 2,
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "grad_clip_norm": 1.0,
            "scheduler": "none",
            "denoise_weight": 0.1,
            "corruption_sigma": 0.1,
            "evaluation_positions": 4,
            "latency_batch_sizes": [1],
            "latency_warmups": 1,
            "latency_repetitions": 1,
            "flop_convention": "conv_linear_multiply_add_as_two_v1",
        },
        "seeds": [7],
        "inputs": [
            {
                "seed": 7,
                "bootstrap_checkpoint": str(checkpoint),
                "bootstrap_checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_status": str(status),
                "checkpoint_status_sha256": sha256_file(status),
                "training_replay": str(replay_path),
                "training_replay_sha256": sha256_file(replay_path),
            }
        ],
        "automatic_promotion": False,
        "prohibited_inferences": ["play_strength", "efficacy"],
    }
    spec_path = tmp_path / "a18_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def test_direct_inference_does_not_call_training_auxiliary_branch(monkeypatch):
    model = A18MatchedEvaluator(tiny_cfg()).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("training-only auxiliary branch reached inference")

    monkeypatch.setattr(model.aux_noise_head, "forward", forbidden)
    with torch.inference_mode():
        logits, values = model(torch.zeros(2, 2, 3, 3))
    assert logits.shape == (2, 9)
    assert values.shape == (2,)


def test_matched_arms_share_parameters_and_forward_flop_surfaces():
    torch.manual_seed(11)
    baseline = A18MatchedEvaluator(tiny_cfg())
    torch.manual_seed(11)
    diffusion = A18MatchedEvaluator(tiny_cfg())
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            baseline.state_dict().values(), diffusion.state_dict().values()
        )
    )
    breakdown = parameter_breakdown(baseline)
    assert breakdown["deployed_direct"] > 0
    assert breakdown["train_only_auxiliary"] > 0
    assert breakdown["total_training"] == (
        breakdown["deployed_direct"] + breakdown["train_only_auxiliary"]
    )
    assert estimate_direct_flops(baseline) == estimate_direct_flops(diffusion)
    assert estimate_training_forward_flops(baseline) == estimate_training_forward_flops(
        diffusion
    )


def test_input_provenance_rejects_nonpreferred_best_checkpoint(tmp_path):
    spec_path = write_inputs(tmp_path, use_latest=False)
    spec = load_spec(spec_path)
    with pytest.raises(
        A18ContractError, match="not the preferred post-train checkpoint"
    ):
        run_smoke_or_study(spec, tmp_path / "out", device="cpu")


def test_analyze_fails_closed_when_real_candidates_are_absent(tmp_path):
    spec = load_spec(write_inputs(tmp_path))
    with pytest.raises(A18ContractError, match="training evidence is absent"):
        analyze_candidates(spec, tmp_path / "out", device="cpu")


def test_study_replay_contract_rejects_overlapping_state_groups(tmp_path):
    spec = load_spec(write_inputs(tmp_path))
    replay_path = spec["inputs"][0]["training_replay"]
    with pytest.raises(A18ContractError, match="state-hash groups overlap"):
        replay_state_hash_contract(replay_path, replay_path)


def test_state_disjoint_holdout_derivation_is_reproducible(tmp_path):
    spec = load_spec(write_inputs(tmp_path))
    training_path = Path(spec["inputs"][0]["training_replay"])
    training = ReplayBuffer(32)
    training.load(str(training_path))
    first = training.examples_at_indices([0])[0]
    source = ReplayBuffer(8)
    source.add(first.state, first.policy.dense(), first.value, metadata=first.metadata)
    for value in (2.0, 3.0):
        source.add(
            np.full((2, 3, 3), value, dtype=np.float32),
            np.full(9, 1.0 / 9.0, dtype=np.float32),
            0.0,
            metadata={"source": "heldout"},
        )
    source_path = tmp_path / "source.npz"
    source.save(source_path)
    output_a = tmp_path / "derived_a.npz"
    output_b = tmp_path / "derived_b.npz"
    receipt_a = derive_state_disjoint_evaluation_replay(
        training_path,
        source_path,
        output_a,
        training_seed=7,
        evaluation_source_seed=8,
    )
    receipt_b = derive_state_disjoint_evaluation_replay(
        training_path,
        source_path,
        output_b,
        training_seed=7,
        evaluation_source_seed=8,
    )
    assert receipt_a["excluded_positions"] == 1
    assert receipt_a["excluded_state_groups"] == 1
    assert receipt_a["retained_positions"] == 2
    assert receipt_a["state_disjoint_contract"]["verified_disjoint"] is True
    assert sha256_file(output_a) == sha256_file(output_b)
    assert receipt_a["output_replay_sha256"] == receipt_b["output_replay_sha256"]


def test_holdout_preparation_reuses_verified_outputs_and_rejects_stale_receipt(
    tmp_path,
):
    spec = load_spec(write_inputs(tmp_path))
    training_path = Path(spec["inputs"][0]["training_replay"])
    source_dir = training_path.parents[1] / "seed_8"
    source_dir.mkdir()
    source = ReplayBuffer(8)
    for value in (2.0, 3.0):
        source.add(
            np.full((2, 3, 3), value, dtype=np.float32),
            np.full(9, 1.0 / 9.0, dtype=np.float32),
            0.0,
            metadata={"source": "heldout"},
        )
    source.save(source_dir / "replay.npz")
    output = tmp_path / "holdouts"
    args = [
        "--bootstrap-root",
        str(training_path.parents[1]),
        "--output-dir",
        str(output),
        "--mapping",
        "7:8",
    ]

    assert holdout_script.main(args) == 0
    replay_path = output / "train_seed_7.eval_from_seed_8.exact_state_disjoint.npz"
    receipt_path = replay_path.with_suffix(".receipt.v1.json")
    original_hashes = (sha256_file(replay_path), sha256_file(receipt_path))
    assert holdout_script.main(args) == 0
    assert (sha256_file(replay_path), sha256_file(receipt_path)) == original_hashes

    stale = json.loads(receipt_path.read_text(encoding="utf-8"))
    stale["generation_rule"] = "stale-rule"
    receipt_path.write_text(json.dumps(stale), encoding="utf-8")
    assert holdout_script.main(args) == 2


def test_study_inspection_rejects_rehashed_stale_derivation_rule(tmp_path):
    spec_path = write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base_row = spec["inputs"][0]
    rows = []
    for seed, offset in ((7, None), (8, 2.0), (9, 12.0)):
        if offset is None:
            replay_path = Path(base_row["training_replay"])
        else:
            replay_path = tmp_path / "bootstrap" / f"seed_{seed}" / "replay.npz"
            replay_path.parent.mkdir()
            replay = ReplayBuffer(16)
            for index in range(8):
                replay.add(
                    np.full((2, 3, 3), offset + index, dtype=np.float32),
                    np.full(9, 1.0 / 9.0, dtype=np.float32),
                    0.0,
                    metadata={"controller_summary": {"root_only_shaping": True}},
                )
            replay.save(replay_path)
        rows.append(
            {
                **base_row,
                "seed": seed,
                "training_replay": str(replay_path),
                "training_replay_sha256": sha256_file(replay_path),
            }
        )

    by_seed = {row["seed"]: row for row in rows}
    for training_seed, source_seed in ((7, 8), (8, 9), (9, 7)):
        row = by_seed[training_seed]
        source_row = by_seed[source_seed]
        evaluation_path = tmp_path / f"eval_{training_seed}_from_{source_seed}.npz"
        receipt = derive_state_disjoint_evaluation_replay(
            row["training_replay"],
            source_row["training_replay"],
            evaluation_path,
            training_seed=training_seed,
            evaluation_source_seed=source_seed,
        )
        receipt_path = evaluation_path.with_suffix(".receipt.json")
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        row.update(
            {
                "evaluation_source_seed": source_seed,
                "evaluation_source_replay": source_row["training_replay"],
                "evaluation_source_replay_sha256": source_row["training_replay_sha256"],
                "evaluation_replay": str(evaluation_path),
                "evaluation_replay_sha256": sha256_file(evaluation_path),
                "evaluation_derivation_receipt": str(receipt_path),
                "evaluation_derivation_receipt_sha256": sha256_file(receipt_path),
            }
        )

    stale_path = Path(by_seed[7]["evaluation_derivation_receipt"])
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    stale["generation_rule"] = "stale-but-rehashed-rule"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    by_seed[7]["evaluation_derivation_receipt_sha256"] = sha256_file(stale_path)
    spec.update(
        {
            "study_id": "tiny-a18-study",
            "evidence_tier": "study_candidate",
            "seeds": [7, 8, 9],
            "inputs": rows,
        }
    )
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(A18ContractError, match="does not match current inputs, rule"):
        inspect_inputs(load_spec(spec_path))


def test_partial_training_evidence_resumes_verified_arms(tmp_path, monkeypatch):
    spec = load_spec(write_inputs(tmp_path))
    output = tmp_path / "out"
    original = a18_module._train_one_candidate
    failed_once = False

    def interrupt_diffusion(*args, variant, **kwargs):
        nonlocal failed_once
        if variant == DIFFUSION_VARIANT and not failed_once:
            failed_once = True
            raise RuntimeError("injected interruption")
        return original(*args, variant=variant, **kwargs)

    monkeypatch.setattr(a18_module, "_train_one_candidate", interrupt_diffusion)
    with pytest.raises(RuntimeError, match="injected interruption"):
        train_candidates(spec, output, device="cpu")
    first_payload = json.loads((output / "training_rows.v1.json").read_text())
    baseline_row = next(
        row
        for row in first_payload["rows"]
        if row["variant"] == BASELINE_VARIANT and row["execution_status"] == "succeeded"
    )
    baseline_hash = baseline_row["checkpoint_sha256"]

    monkeypatch.setattr(a18_module, "_train_one_candidate", original)
    completed = train_candidates(spec, output, device="cpu")
    final_payload = json.loads((output / "training_rows.v1.json").read_text())
    assert len(final_payload["rows"]) == 3
    assert (
        sum(row["execution_status"] == "failed" for row in final_payload["rows"]) == 1
    )
    assert len(completed) == 2
    assert (
        sha256_file(output / "models" / "seed_7" / f"{BASELINE_VARIANT}.a18ckpt")
        == baseline_hash
    )


def test_three_contract_surfaces_emit_standard_diagnostic_artifacts(tmp_path):
    spec = load_spec(write_inputs(tmp_path))
    output = tmp_path / "out"
    manifest = run_smoke_or_study(spec, output, device="cpu")

    assert manifest["axis_id"] == "A18"
    assert manifest["role"] == "ablation_readiness"
    assert manifest["evidence_status"] == "skeleton_only"
    assert manifest["claim_scope"] == "ablation_readiness_only"
    assert manifest["promotion"] == {"auto": False, "eligible": False}
    assert len(manifest["artifacts"]["candidate_checkpoints"]) == 2
    for name in (
        "run_manifest.json",
        "rows.jsonl",
        "summary.json",
        "diagnostic.png",
        "manifest.v1.json",
        "data.v1.json",
        "DIAGNOSTIC_a18_evaluator_ablation.png",
        "training_rows.v1.json",
    ):
        assert (output / name).is_file()
    assert (output / "diagnostic.png").read_bytes().startswith(b"\x89PNG")
    rows = [
        json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()
    ]
    assert {row["variant"] for row in rows} == {BASELINE_VARIANT, DIFFUSION_VARIANT}
    assert all(row["axis_id"] == "A18" for row in rows)
    assert all(row["promotion"]["eligible"] is False for row in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["execution_status"] == "completed_no_promotion"
    assert summary["parameter_match"] is True
    assert summary["direct_inference_flop_match"] is True
    assert summary["training_forward_flop_match"] is True

    second_output = tmp_path / "out_repeat"
    run_smoke_or_study(spec, second_output, device="cpu")
    first_training = json.loads((output / "training_rows.v1.json").read_text())["rows"]
    second_training = json.loads((second_output / "training_rows.v1.json").read_text())[
        "rows"
    ]
    assert [row["checkpoint_sha256"] for row in first_training] == [
        row["checkpoint_sha256"] for row in second_training
    ]


def test_tracked_study_input_receipt_matches_sources_and_available_local_inputs():
    repo_root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (repo_root / "docs/idea_foundry/A18_STUDY_INPUT_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["axis_id"] == "A18"
    assert receipt["claim_scope"] == "input_provenance_only"
    assert receipt["portability_status"] == "repo_local_regenerable_not_embedded"
    assert receipt["scientific_status"] == "DERIVED_INPUT_NOT_EFFICACY"
    assert receipt["automatic_promotion"] is False

    tracked_records = [
        receipt["study_config"],
        receipt["derivation"]["script"],
        receipt["derivation"]["module"],
    ]
    for record in tracked_records:
        path = repo_root / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]

    local_records = [receipt["derivation"]["bundle"]]
    for artifact in receipt["artifacts"]:
        local_records.extend(
            [
                {"path": artifact["path"], "sha256": artifact["sha256"]},
                {
                    "path": artifact["receipt_path"],
                    "sha256": artifact["receipt_sha256"],
                },
            ]
        )
    available = [(repo_root / record["path"]).is_file() for record in local_records]
    assert all(available) or not any(available)
    for record in local_records:
        path = repo_root / record["path"]
        if path.is_file():
            assert sha256_file(path) == record["sha256"]
