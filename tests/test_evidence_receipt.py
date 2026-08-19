"""Contract tests for Git-tree-bound Idea Foundry evidence receipts."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import verify_evidence_receipt as verifier


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _entry(path: str, payload: bytes) -> dict[str, str]:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}


def _identity(commit: str, regular: dict, raw: dict, *, execution: bool) -> dict:
    return {
        "commit": commit,
        "dirty": False,
        "source_inventory": [regular.copy()],
        "input_inventory": [(regular if execution else raw).copy()],
        "binary_inventory": [regular.copy()],
        "artifact_inventory": [(raw if execution else regular).copy()],
    }


def _save(case) -> None:
    case[1].write_text(json.dumps(case[2]), encoding="utf-8")


def _run(case) -> dict:
    return case[2]["runs"][0]


def _fails(case, message: str) -> None:
    _save(case)
    with pytest.raises(ValueError, match=message):
        verifier.verify_receipt(case[1], repo_root=case[0])


@pytest.fixture
def receipt_case(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Receipt Test")
    for path, payload in {"exec/data": b"exec\n", "raw/manifest": b"raw\n"}.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (repo / "exec/link").symlink_to("data")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "execution")
    execution_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "analysis").write_bytes(b"analysis\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "analysis")
    ordinary, raw, derived = (
        _entry("exec/data", b"exec\n"),
        _entry("raw/manifest", b"raw\n"),
        _entry("analysis", b"analysis\n"),
    )
    execution = _identity(execution_commit, ordinary, raw, execution=True)
    analysis = _identity(_git(repo, "rev-parse", "HEAD"), derived, raw, execution=False)
    data = {
        "schema_version": 2,
        "runs": [
            {
                "run_id": "run-1",
                "execution_identity": execution,
                "analysis_identity": analysis,
                "transformation_link": {"raw_execution_manifest": raw.copy()},
            }
        ],
    }
    case = repo, repo / "valid.receipt.json", data
    _save(case)
    return case


def test_valid_v2_receipt_verifies_two_git_identities(receipt_case) -> None:
    run = _run(receipt_case)
    assert run["execution_identity"]["commit"] != run["analysis_identity"]["commit"]
    assert verifier.verify_receipt(receipt_case[1], repo_root=receipt_case[0]) == {
        "receipt": "valid.receipt.json",
        "status": "VERIFIED",
        "verified_runs": 1,
        "verified_artifacts": 2,
        "verified_entries": 8,
    }


def test_public_structural_seam_has_no_repository_dependency(receipt_case) -> None:
    run = _run(receipt_case)
    execution = verifier.validate_identity_contract(
        run["execution_identity"], "execution_identity"
    )
    analysis = verifier.validate_identity_contract(
        run["analysis_identity"], "analysis_identity"
    )
    verifier.validate_raw_manifest_link(run["transformation_link"], execution, analysis)


def test_git_tree_drift_rejected_even_when_worktree_matches(receipt_case) -> None:
    new = b"worktree-only\n"
    (receipt_case[0] / "raw/manifest").write_bytes(new)
    run = _run(receipt_case)
    digest = hashlib.sha256(new).hexdigest()
    for item in (
        run["execution_identity"]["artifact_inventory"][0],
        run["analysis_identity"]["input_inventory"][0],
        run["transformation_link"]["raw_execution_manifest"],
    ):
        item["sha256"] = digest
    assert (
        hashlib.sha256((receipt_case[0] / "raw/manifest").read_bytes()).hexdigest()
        == digest
    )
    _fails(
        receipt_case, "Git-tree hash drift for execution_identity.artifact_inventory"
    )


def test_schema_v1_is_legacy_unbound(receipt_case) -> None:
    receipt_case[2]["schema_version"] = 1
    _fails(
        receipt_case,
        "legacy receipt schema v1 is invalid/unbound; read-only inspection only",
    )


@pytest.mark.parametrize("version", [True, 2.0, "2", 3])
def test_schema_version_is_exact_integer_two(receipt_case, version) -> None:
    receipt_case[2]["schema_version"] = version
    _fails(receipt_case, "receipt schema_version must be integer 2")


def test_runs_must_be_nonempty(receipt_case) -> None:
    receipt_case[2]["runs"] = []
    _fails(receipt_case, "receipt contains no run entries")


@pytest.mark.parametrize("identity", ["execution_identity", "analysis_identity"])
@pytest.mark.parametrize("inventory", verifier.INVENTORY_NAMES)
def test_identity_inventories_must_be_nonempty(
    receipt_case, identity, inventory
) -> None:
    _run(receipt_case)[identity][inventory] = []
    _fails(receipt_case, rf"{identity}\.{inventory} must be nonempty")


@pytest.mark.parametrize("identity", ["execution_identity", "analysis_identity"])
@pytest.mark.parametrize("oid", ["0" * 40, "A" * 40, "0" * 39])
def test_unresolvable_identity_commit_fails(receipt_case, identity, oid) -> None:
    _run(receipt_case)[identity]["commit"] = oid
    _fails(receipt_case, rf"{identity}\.commit cannot be resolved as commit")


@pytest.mark.parametrize("identity", ["execution_identity", "analysis_identity"])
def test_dirty_identity_fails(receipt_case, identity) -> None:
    _run(receipt_case)[identity]["dirty"] = True
    _fails(receipt_case, rf"{identity}\.dirty must be false")


@pytest.mark.parametrize(
    "path", ["../x", "/x", "a/../x", "./x", "a//x", "a\\x", "a/", "-x", "a\0x"]
)
def test_noncanonical_inventory_path_fails(receipt_case, path) -> None:
    _run(receipt_case)["execution_identity"]["source_inventory"][0]["path"] = path
    _fails(receipt_case, "path must be a normalized in-repository POSIX path")


def test_symlink_tree_entry_fails(receipt_case) -> None:
    _run(receipt_case)["execution_identity"]["source_inventory"] = [
        _entry("exec/link", b"data")
    ]
    _fails(receipt_case, "Git-tree path is not a regular file")


def test_missing_tree_path_fails(receipt_case) -> None:
    _run(receipt_case)["execution_identity"]["source_inventory"] = [
        _entry("exec/missing", b"missing")
    ]
    _fails(receipt_case, "Git-tree path missing")


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "g" * 64])
def test_malformed_sha256_fails(receipt_case, digest) -> None:
    _run(receipt_case)["execution_identity"]["source_inventory"][0]["sha256"] = digest
    _fails(receipt_case, "sha256 must be 64 lowercase hex characters")


def test_duplicate_inventory_path_fails(receipt_case) -> None:
    items = _run(receipt_case)["execution_identity"]["source_inventory"]
    items.append(items[0].copy())
    _fails(receipt_case, "duplicate path in execution_identity.source_inventory")


def test_raw_manifest_must_be_execution_artifact(receipt_case) -> None:
    run = _run(receipt_case)
    run["execution_identity"]["artifact_inventory"] = [
        run["execution_identity"]["source_inventory"][0].copy()
    ]
    _fails(receipt_case, "raw execution manifest is absent from execution_identity")


def test_raw_manifest_must_be_analysis_input(receipt_case) -> None:
    run = _run(receipt_case)
    run["analysis_identity"]["input_inventory"] = [
        run["analysis_identity"]["source_inventory"][0].copy()
    ]
    _fails(receipt_case, "raw execution manifest is absent from analysis_identity")


def test_raw_manifest_digest_must_match_both_identities(receipt_case) -> None:
    _run(receipt_case)["transformation_link"]["raw_execution_manifest"]["sha256"] = (
        "0" * 64
    )
    _fails(receipt_case, "raw execution manifest descriptor mismatch")


def test_receipt_symlink_fails(receipt_case) -> None:
    link = receipt_case[0] / "linked.receipt.json"
    link.symlink_to(receipt_case[1].name)
    with pytest.raises(ValueError, match="receipt is missing or symlink"):
        verifier.verify_receipt(link, repo_root=receipt_case[0])


def _empty_cli(tmp_path, capsys, *, exists: bool, allow: bool):
    receipts = tmp_path / "receipts"
    if exists:
        receipts.mkdir()
    args = ["--receipts-dir", str(receipts)]
    return verifier.main(args + (["--allow-empty-diagnostic"] if allow else [])), capsys


def test_missing_receipts_dir_fails_by_default(tmp_path, capsys) -> None:
    code, captured = _empty_cli(tmp_path, capsys, exists=False, allow=False)
    assert (
        code == 1 and "[FAIL] receipts directory missing:" in captured.readouterr().err
    )


def test_empty_receipts_dir_fails_by_default(tmp_path, capsys) -> None:
    code, captured = _empty_cli(tmp_path, capsys, exists=True, allow=False)
    assert code == 1 and "[FAIL] no receipt files found:" in captured.readouterr().err


@pytest.mark.parametrize("exists", [False, True])
def test_allow_empty_diagnostic_is_explicit(tmp_path, capsys, exists) -> None:
    code, captured = _empty_cli(tmp_path, capsys, exists=exists, allow=True)
    output = captured.readouterr().out
    assert code == 0 and "[DIAGNOSTIC]" in output and "allow-empty accepted" in output
