from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SOURCES = (
    "scripts/idea_foundry/a03_uncertainty_decomposition.py",
    "scripts/idea_foundry/a09_h3_change_point_router.py",
    "scripts/idea_foundry/a01_calibrated_stop_council.py",
    "scripts/idea_foundry/a02_static_anchor_rpo.py",
    "scripts/idea_foundry/a17_b13_curvature_readout.py",
    "scripts/idea_foundry/a21_coherence_signed_path_shadow.py",
    "scripts/idea_foundry/a22_physics_falsification_dashboard.py",
    "scripts/idea_foundry/a06_gumbel_sequential_halving.py",
    "scripts/idea_foundry/a07_residual_evidence_widening.py",
    "scripts/idea_foundry/a12_jsd_locally_balanced_sampler.py",
    "scripts/idea_foundry/a25_ments_soft_backup.py",
    "scripts/idea_foundry/a26_nested_contour_exact_lab.py",
    "scripts/idea_foundry/a05_counterfactual_meta_teacher.py",
    "scripts/idea_foundry/a04_kg_voc_allocator.py",
    "scripts/idea_foundry/a08_tactical_proof_backend.py",
    "scripts/idea_foundry/a13_pending_flow_wu_uct.py",
    "scripts/idea_foundry/a15_service_curve_scheduler.py",
    "scripts/idea_foundry/a14_semantic_path_lsh.py",
    "scripts/idea_foundry/a16_monte_carlo_graph_sharing.py",
    "scripts/idea_foundry/a11_dynamic_live_set_particles.py",
    "scripts/idea_foundry/a18_diffusion_regularized_evaluator.py",
    "scripts/idea_foundry/a19_rw_rest_lite_evaluator.py",
    "scripts/idea_foundry/a23_cpu_incremental_pattern_student.py",
    "scripts/idea_foundry/a20_regret_state_archive.py",
    "scripts/idea_foundry/a24_learned_budget_gate.py",
    "scripts/idea_foundry/a10_prior_refresh_specialist.py",
    "quartz/experiment_manifest.py",
    "quartz/idea_foundry/__init__.py",
    "quartz/idea_foundry/axis_workflow.py",
    "quartz/idea_foundry/contracts.py",
    "quartz/idea_foundry/control.py",
    "quartz/idea_foundry/execution_seal.py",
    "quartz/idea_foundry/gates.py",
    "quartz/idea_foundry/learning.py",
    "quartz/idea_foundry/search.py",
    "quartz/idea_foundry/sequential.py",
    "quartz/idea_foundry/serialization.py",
    "quartz/idea_foundry/status_schema.py",
    "scripts/idea_foundry_axis_gate.py",
    "scripts/idea_foundry_run_all.py",
)
INPUTS = ("configs/idea_foundry.axes.v1.json", "configs/idea_lab.local.v2.json")
AXES = (
    "A03",
    "A09",
    "A01",
    "A02",
    "A17",
    "A21",
    "A22",
    "A06",
    "A07",
    "A12",
    "A25",
    "A26",
    "A05",
    "A04",
    "A08",
    "A13",
    "A15",
    "A14",
    "A16",
    "A11",
    "A18",
    "A19",
    "A23",
    "A20",
    "A24",
    "A10",
)
CAMPAIGN_ARTIFACTS = (
    "campaign_execution_seal.json",
    "campaign_state.json",
    "campaign_summary.json",
)
ATTEMPT_ARTIFACTS = (
    "run_manifest.json",
    "rows.jsonl",
    "summary.json",
    "analysis/analysis_manifest.json",
    "analysis/analysis.json",
    "analysis/analysis_rows.jsonl",
    "analysis/diagnostic.png",
)
ROOT = Path(__file__).resolve().parents[1]


def _module():
    return importlib.import_module("quartz.idea_foundry.execution_seal")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    )


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in (*SOURCES, *INPUTS):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Seal Test")
    _git(repo, "config", "user.email", "seal@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _capture(repo: Path, *, run_id: str = "run-1", seed: int = 17):
    return _module().capture_execution_seal(
        repo, run_id=run_id, seed=seed, executable=Path(sys.executable)
    )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
        if not any(part == ".git" for part in path.relative_to(root).parts)
    }


def test_execution_seal_public_boundary_exists() -> None:
    # Catches an implementation that never introduces the required owner module.
    spec = importlib.util.find_spec("quartz.idea_foundry.execution_seal")
    assert spec is not None


def test_capture_binds_exact_literal_inventory_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    # Catches runtime discovery, reordered membership, or an incomplete descriptor.
    repo = _repository(tmp_path)
    seal = _capture(repo)
    assert list(seal) == [
        "schema_version",
        "kind",
        "suite",
        "claim_scope",
        "run_id",
        "seed",
        "git",
        "axis_order",
        "sources",
        "inputs",
        "binaries",
        "campaign_artifacts",
        "terminal_attempt_artifacts",
    ]
    assert seal["schema_version"] == 2 and type(seal["schema_version"]) is int
    assert seal["kind"] == "idea_foundry_campaign_execution_seal"
    assert seal["suite"] == "first-gate-all-sequential"
    assert seal["claim_scope"] == "synthetic_contract_execution_only"
    assert seal["axis_order"] == list(AXES)
    assert [row["path"] for row in seal["sources"]] == list(SOURCES)
    assert [row["path"] for row in seal["inputs"]] == list(INPUTS)
    assert seal["campaign_artifacts"] == list(CAMPAIGN_ARTIFACTS)
    assert seal["terminal_attempt_artifacts"] == list(ATTEMPT_ARTIFACTS)
    assert list(seal["binaries"][0]) == [
        "invocation_path",
        "resolved_target_path",
        "size_bytes",
        "sha256",
    ]
    assert seal["binaries"][0]["invocation_path"] == str(Path(sys.executable))
    assert seal["binaries"][0]["resolved_target_path"] == str(
        Path(sys.executable).resolve()
    )
    encoded = _module().canonical_json_bytes(seal)
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert (
        encoded
        == json.dumps(
            seal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )


@pytest.mark.parametrize(
    "document",
    [
        b'{"a":1,"a":1}\n',
        b'{"value":NaN}\n',
        b'{"value":1e400}\n',
        b'{"value":"\xff"}\n',
        b'{ "value":1}\n',
        b'{"value":1}',
    ],
)
def test_canonical_loader_rejects_ambiguous_or_noncanonical_bytes(
    tmp_path: Path, document: bytes
) -> None:
    # Catches permissive parsing before digest/state acceptance.
    path = tmp_path / "seal.json"
    path.write_bytes(document)
    with pytest.raises(_module().ExecutionSealError):
        _module().load_canonical_json(path)


@pytest.mark.parametrize("mutation", ["tracked", "untracked", "assume", "skip"])
def test_capture_rejects_every_git_dirty_hiding_mechanism(
    tmp_path: Path, mutation: str
) -> None:
    # Catches relying on ordinary status while index flags conceal live drift.
    repo = _repository(tmp_path)
    target = repo / SOURCES[0]
    if mutation == "tracked":
        target.write_bytes(target.read_bytes() + b"# drift\n")
    elif mutation == "untracked":
        (repo / "hostile.txt").write_text("drift", encoding="utf-8")
    elif mutation == "assume":
        _git(repo, "update-index", "--assume-unchanged", SOURCES[0])
    else:
        _git(repo, "update-index", "--skip-worktree", SOURCES[0])
    with pytest.raises(_module().ExecutionSealError):
        _capture(repo)


@pytest.mark.parametrize("mutation", ["absent", "symlink", "hardlink", "fifo"])
def test_capture_rejects_missing_nonblob_and_alias_members(
    tmp_path: Path, mutation: str
) -> None:
    # Catches following aliases or accepting a member absent from the captured tree.
    repo = _repository(tmp_path)
    target = repo / SOURCES[0]
    if mutation == "absent":
        _git(repo, "rm", "-q", SOURCES[0])
        _git(repo, "commit", "-qm", "remove source")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(repo / SOURCES[1])
    elif mutation == "hardlink":
        data = target.read_bytes()
        target.unlink()
        alias = repo / "alias"
        alias.write_bytes(data)
        os.link(alias, target)
    else:
        target.unlink()
        os.mkfifo(target)
    with pytest.raises(_module().ExecutionSealError):
        _capture(repo)


def test_capture_ignores_hostile_git_environment_and_rejects_replace_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches ambient Git redirects and replacement-object identity.
    repo = _repository(tmp_path)
    monkeypatch.setenv("GIT_DIR", "/does/not/exist")
    monkeypatch.setenv("GIT_WORK_TREE", "/does/not/exist")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/does/not/exist")
    assert (
        _capture(repo)["git"]["commit"]
        == _git(repo, "rev-parse", "HEAD").stdout.strip()
    )
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "replace", head, head)
    with pytest.raises(_module().ExecutionSealError):
        _capture(repo)


def test_capture_rejects_moving_identity_and_mid_read_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches a capture assembled across two repository identities or file versions.
    repo = _repository(tmp_path)
    module = _module()
    real_git = module._git
    heads = 0

    def moving_git(root: Path, *args: str) -> bytes:
        nonlocal heads
        value = real_git(root, *args)
        if args[-2:] == ("--verify", "HEAD^{commit}"):
            heads += 1
            if heads == 1:
                target = repo / SOURCES[0]
                target.write_bytes(target.read_bytes() + b"# next\n")
                _git(repo, "add", SOURCES[0])
                _git(repo, "commit", "-qm", "move")
        return value

    monkeypatch.setattr(module, "_git", moving_git)
    with pytest.raises(module.ExecutionSealError):
        _capture(repo)

    monkeypatch.setattr(module, "_git", real_git)
    real_read = module._stable_read
    changed = False

    def drifting_read(path: Path, **kwargs: object) -> bytes:
        nonlocal changed
        value = real_read(path, **kwargs)
        if path == repo / SOURCES[0] and not changed:
            changed = True
            path.write_bytes(value + b"# after read\n")
        return value

    monkeypatch.setattr(module, "_stable_read", drifting_read)
    with pytest.raises(module.ExecutionSealError):
        _capture(repo)


def test_claim_and_publish_are_exclusive_and_durable(tmp_path: Path) -> None:
    # Catches reuse/overwrite and publication that omits canonical durable bytes.
    repo = _repository(tmp_path)
    run_root = repo / "results" / "run"
    module = _module()
    module.claim_run_root(run_root)
    with pytest.raises(module.ExecutionSealError):
        module.claim_run_root(run_root)
    seal = _capture(repo, run_id="run")
    persisted = module.publish_execution_seal(run_root, seal)
    final = run_root / "campaign_execution_seal.json"
    assert persisted == module.canonical_json_bytes(seal) == final.read_bytes()
    assert final.stat().st_nlink == 1
    with pytest.raises(module.ExecutionSealError):
        module.publish_execution_seal(run_root, seal)


@pytest.mark.parametrize("failure", ["file", "directory"])
def test_publication_failure_preserves_crash_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    # Catches cleanup or reuse after a durability-stage failure.
    repo = _repository(tmp_path)
    run_root = repo / "results" / "run"
    module = _module()
    module.claim_run_root(run_root)
    seal = _capture(repo, run_id="run")
    real_fsync = module.os.fsync
    calls = 0

    def failed_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if (failure == "file" and calls == 1) or (
            failure == "directory" and calls == 2
        ):
            raise OSError("durability fault")
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", failed_fsync)
    with pytest.raises(OSError, match="durability fault"):
        module.publish_execution_seal(run_root, seal)
    assert any(run_root.iterdir())
    with pytest.raises(module.ExecutionSealError):
        module.claim_run_root(run_root)


def test_persisted_seal_requires_canonical_bytes_digest_and_single_link(
    tmp_path: Path,
) -> None:
    # Catches digesting a permissively decoded or aliased seal.
    repo = _repository(tmp_path)
    run_root = repo / "results" / "run"
    module = _module()
    module.claim_run_root(run_root)
    seal = _capture(repo, run_id="run")
    raw = module.publish_execution_seal(run_root, seal)
    loaded, digest = module.read_execution_seal(run_root)
    assert loaded == seal
    assert digest == hashlib.sha256(raw).hexdigest()
    alias = run_root / "alias"
    os.link(run_root / "campaign_execution_seal.json", alias)
    with pytest.raises(module.ExecutionSealError):
        module.read_execution_seal(run_root)


def _bind_sequential_repo(monkeypatch: pytest.MonkeyPatch, repo: Path):
    axis = importlib.import_module("quartz.idea_foundry.axis_workflow")
    sequential = importlib.import_module("quartz.idea_foundry.sequential")
    monkeypatch.setattr(axis, "REPO_ROOT", repo)
    monkeypatch.setattr(axis, "AXIS_REGISTRY_PATH", repo / INPUTS[0])
    monkeypatch.setattr(axis, "LAB_REGISTRY_PATH", repo / INPUTS[1])
    monkeypatch.setattr(sequential, "REPO_ROOT", repo)
    return sequential, repo / SOURCES[-1]


def _failed_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path, Path]:
    repo = _repository(tmp_path)
    sequential, entrypoint = _bind_sequential_repo(monkeypatch, repo)
    monkeypatch.setattr(sequential, "_run_attempt", lambda **_kwargs: (9, "completed"))
    campaign_root = repo / "results" / "campaigns"
    with pytest.raises(sequential.SequentialCampaignError, match="A03 stopped"):
        sequential.run_campaign(
            campaign_root=campaign_root,
            run_id="failed-run",
            seed=23,
            timeout_seconds=1,
            resume=False,
            entrypoint=entrypoint,
        )
    return sequential, entrypoint, campaign_root / "failed-run"


def test_new_campaign_seals_before_state_and_records_exact_v2_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches state/attempt mutation before coherent durable seal publication.
    sequential, _entrypoint, run_root = _failed_campaign(tmp_path, monkeypatch)
    seal_raw = (run_root / "campaign_execution_seal.json").read_bytes()
    state_raw = (run_root / "campaign_state.json").read_bytes()
    state = json.loads(state_raw)
    assert (
        state_raw
        == json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )
    assert set(state) == {
        "schema_version",
        "run_id",
        "suite",
        "status",
        "seed",
        "created_at",
        "updated_at",
        "execution_seal_sha256",
        "claim_scope",
        "axes",
    }
    assert state["schema_version"] == 2 and type(state["schema_version"]) is int
    assert state["execution_seal_sha256"] == hashlib.sha256(seal_raw).hexdigest()
    assert [row["axis_id"] for row in state["axes"]] == list(AXES)
    assert [row["order_index"] for row in state["axes"]] == list(range(26))
    assert state["status"]["execution"] == "failed"
    assert state["axes"][0]["status"]["execution"] == "failed"
    attempt = state["axes"][0]["attempts"][0]
    assert attempt == {
        "attempt_number": 1,
        "started_at": attempt["started_at"],
        "output_dir": "axes/A03/attempt-001",
        "stdout": "logs/A03.attempt-001.stdout.log",
        "stderr": "logs/A03.attempt-001.stderr.log",
        "process_outcome": "completed",
        "completed_at": attempt["completed_at"],
        "returncode": 9,
    }
    assert not (run_root / "axes" / "A09").exists()
    assert (
        sequential._validate_state(
            state,
            "failed-run",
            23,
            hashlib.sha256(seal_raw).hexdigest(),
        )
        is state
    )


def test_new_campaign_pre_post_drift_leaves_seal_without_state_or_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches state creation when the post-publication capture differs from S_pre.
    repo = _repository(tmp_path)
    sequential, entrypoint = _bind_sequential_repo(monkeypatch, repo)
    real_capture = sequential.capture_execution_seal
    calls = 0

    def drifting_capture(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        seal = real_capture(*args, **kwargs)
        if calls == 2:
            seal = dict(seal)
            seal["seed"] = 99
        return seal

    monkeypatch.setattr(sequential, "capture_execution_seal", drifting_capture)
    campaign_root = repo / "results" / "campaigns"
    with pytest.raises(sequential.SequentialCampaignError, match="changed"):
        sequential.run_campaign(
            campaign_root=campaign_root,
            run_id="drift",
            seed=23,
            timeout_seconds=1,
            resume=False,
            entrypoint=entrypoint,
        )
    run_root = campaign_root / "drift"
    assert (run_root / "campaign_execution_seal.json").is_file()
    assert not (run_root / "campaign_state.json").exists()
    assert not (run_root / "axes").exists()
    assert not (run_root / "logs").exists()


def test_failed_only_resume_and_all_rejections_are_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches terminal/running/legacy rejection after state or output mutation.
    sequential, entrypoint, run_root = _failed_campaign(tmp_path, monkeypatch)
    campaign_root = run_root.parent
    state_path = run_root / "campaign_state.json"
    initial = json.loads(state_path.read_bytes())
    frozen = _snapshot(run_root)

    for campaign_status in ("running", "success", "legacy"):
        state = json.loads(json.dumps(initial))
        if campaign_status == "legacy":
            state["schema_version"] = 1
        elif campaign_status == "running":
            state["status"] = {
                "schema_version": 2,
                "execution": "running",
                "contract": "not_evaluated",
                "effect": "not_evaluated",
                "evidence_maturity": "contract_only",
                "promotion": "ineligible",
            }
            state["axes"][0]["status"] = state["status"]
            state["axes"][0]["attempts"][0] = {
                key: value
                for key, value in state["axes"][0]["attempts"][0].items()
                if key not in {"completed_at", "returncode"}
            }
            state["axes"][0]["attempts"][0]["process_outcome"] = "running"
            state["axes"][0].pop("failure_reason")
        else:
            state["status"] = {
                "schema_version": 2,
                "execution": "success",
                "contract": "passed",
                "effect": "non_estimable",
                "evidence_maturity": "contract_only",
                "promotion": "ineligible",
            }
            state["completed_at"] = "2026-08-22T00:00:00Z"
            for index, row in enumerate(state["axes"]):
                row["status"] = {
                    "schema_version": 2,
                    "execution": "skipped" if index == 25 else "success",
                    "contract": "not_applicable" if index == 25 else "passed",
                    "effect": "non_estimable",
                    "evidence_maturity": "contract_only",
                    "promotion": "ineligible",
                }
                if not row["attempts"]:
                    row["current_attempt"] = f"axes/{row['axis_id']}/attempt-001"
                    row["attempts"] = [
                        {
                            "attempt_number": 1,
                            "started_at": "t",
                            "output_dir": row["current_attempt"],
                            "stdout": f"logs/{row['axis_id']}.attempt-001.stdout.log",
                            "stderr": f"logs/{row['axis_id']}.attempt-001.stderr.log",
                            "process_outcome": "completed",
                            "completed_at": "t",
                            "returncode": 0,
                        }
                    ]
                row.pop("failure_reason", None)
        state_path.write_bytes(_module().canonical_json_bytes(state))
        before = _snapshot(run_root)
        with pytest.raises(sequential.SequentialCampaignError):
            sequential.run_campaign(
                campaign_root=campaign_root,
                run_id="failed-run",
                seed=23,
                timeout_seconds=1,
                resume=True,
                entrypoint=entrypoint,
            )
        assert _snapshot(run_root) == before

    state_path.write_bytes(_module().canonical_json_bytes(initial))
    monkeypatch.setattr(sequential, "_run_attempt", lambda **_kwargs: (8, "completed"))
    with pytest.raises(sequential.SequentialCampaignError, match="A03 stopped"):
        sequential.run_campaign(
            campaign_root=campaign_root,
            run_id="failed-run",
            seed=23,
            timeout_seconds=1,
            resume=True,
            entrypoint=entrypoint,
        )
    resumed = json.loads(state_path.read_bytes())
    assert resumed["resumed_at"]
    assert len(resumed["axes"][0]["attempts"]) == 2
    assert resumed["axes"][0]["attempts"][0]["returncode"] == 9
    assert resumed["axes"][0]["attempts"][1]["returncode"] == 8
    assert frozen != _snapshot(run_root)


@pytest.mark.parametrize(
    "defect",
    [
        "schema_bool",
        "extra",
        "alias",
        "bad_lifecycle",
        "bad_order",
        "bad_axis_metadata",
        "bad_attempt_number",
        "bad_attempt_path",
        "bad_outcome",
        "bool_returncode",
        "returncode_relation",
        "writer_invalid_status",
        "running_after_planned",
    ],
)
def test_state_validator_rejects_exact_type_lifecycle_and_sequence_defects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    # Catches permissive/coerced state shapes and impossible writer histories.
    sequential, _entrypoint, run_root = _failed_campaign(tmp_path, monkeypatch)
    seal_digest = hashlib.sha256(
        (run_root / "campaign_execution_seal.json").read_bytes()
    ).hexdigest()
    state = json.loads((run_root / "campaign_state.json").read_bytes())
    if defect == "schema_bool":
        state["schema_version"] = True
    elif defect == "extra":
        state["extra"] = None
    elif defect == "alias":
        state["status"]["execution_status"] = "failed"
    elif defect == "bad_lifecycle":
        state["completed_at"] = "now"
    elif defect == "bad_order":
        state["axes"][0], state["axes"][1] = state["axes"][1], state["axes"][0]
    elif defect == "bad_axis_metadata":
        state["axes"][0]["role"] += "-drift"
    elif defect == "bad_attempt_number":
        state["axes"][0]["attempts"][0]["attempt_number"] = 2
    elif defect == "bad_attempt_path":
        state["axes"][0]["attempts"][0]["stdout"] = "../escape"
    elif defect == "bad_outcome":
        state["axes"][0]["attempts"][0]["process_outcome"] = "mystery"
    elif defect == "bool_returncode":
        state["axes"][0]["attempts"][0]["returncode"] = True
    elif defect == "returncode_relation":
        state["axes"][0]["attempts"][0]["returncode"] = 0
    elif defect == "writer_invalid_status":
        state["status"]["effect"] = "estimable"
    elif defect == "running_after_planned":
        state["status"] = {
            **state["status"],
            "execution": "running",
            "contract": "not_evaluated",
            "effect": "not_evaluated",
        }
        state["axes"][1]["status"] = state["status"]
        state["axes"][1]["current_attempt"] = "axes/A09/attempt-001"
        state["axes"][1]["attempts"] = [
            {
                "attempt_number": 1,
                "started_at": "t",
                "output_dir": "axes/A09/attempt-001",
                "stdout": "logs/A09.attempt-001.stdout.log",
                "stderr": "logs/A09.attempt-001.stderr.log",
                "process_outcome": "running",
            }
        ]
    with pytest.raises(sequential.SequentialCampaignError):
        sequential._validate_state(state, "failed-run", 23, seal_digest)
