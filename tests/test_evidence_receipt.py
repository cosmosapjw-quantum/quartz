import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import verify_evidence_receipt as verifier

EXEC, ANALYSIS = "execution_identity", "analysis_identity"
DRIFT = "Git-tree hash drift for execution_identity.artifact_inventory"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _entry(path: str, payload: bytes) -> dict[str, str]:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}


def _id(commit: str, regular: dict, raw: dict, execution: bool) -> dict:
    return {
        "commit": commit,
        "dirty": False,
        "source_inventory": [regular.copy()],
        "input_inventory": [(regular if execution else raw).copy()],
        "binary_inventory": [regular.copy()],
        "artifact_inventory": [(raw if execution else regular).copy()],
    }


def _run(case) -> dict:
    return case[2]["runs"][0]


def _save(case) -> None:
    case[1].write_text(json.dumps(case[2]), encoding="utf-8")


def _fails(case, message: str) -> None:
    _save(case)
    with pytest.raises(ValueError, match=message):
        verifier.verify_receipt(case[1], repo_root=case[0])


def _set_raw_digest(run: dict, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    for item in (
        run[EXEC]["artifact_inventory"][0],
        run[ANALYSIS]["input_inventory"][0],
        run["transformation_link"]["raw_execution_manifest"],
    ):
        item["sha256"] = digest
    return digest


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
    regular = _entry("exec/data", b"exec\n")
    raw = _entry("raw/manifest", b"raw\n")
    derived = _entry("analysis", b"analysis\n")
    run = {
        "run_id": "run-1",
        EXEC: _id(execution_commit, regular, raw, True),
        ANALYSIS: _id(_git(repo, "rev-parse", "HEAD"), derived, raw, False),
        "transformation_link": {"raw_execution_manifest": raw.copy()},
    }
    case = repo, repo / "valid.receipt.json", {"schema_version": 2, "runs": [run]}
    _save(case)
    return case


def test_valid_v2_receipt_verifies_two_git_identities(receipt_case) -> None:
    run = _run(receipt_case)
    assert run[EXEC]["commit"] != run[ANALYSIS]["commit"]
    result = verifier.verify_receipt(receipt_case[1], repo_root=receipt_case[0])
    assert result["status"] == "VERIFIED"
    assert (result["verified_runs"], result["verified_artifacts"]) == (1, 2)
    assert result["verified_entries"] == 8


def test_public_structural_seam_has_no_repository_dependency(receipt_case) -> None:
    run = _run(receipt_case)
    descriptor = run[EXEC]["source_inventory"][0]
    assert verifier.validate_descriptor(descriptor, "descriptor") == descriptor
    for invalid in (
        {**descriptor, "extra": True},
        {**descriptor, "path": "../outside"},
        {**descriptor, "sha256": "A" * 64},
    ):
        with pytest.raises(ValueError):
            verifier.validate_descriptor(invalid, "descriptor")
    execution = verifier.validate_identity_contract(run[EXEC], EXEC)
    analysis = verifier.validate_identity_contract(run[ANALYSIS], ANALYSIS)
    verifier.validate_raw_manifest_link(run["transformation_link"], execution, analysis)


@pytest.mark.parametrize("attack", [None, "blob", "commit", "corrupt"])
def test_literal_tree_binding(receipt_case, attack) -> None:
    repo, run, new = receipt_case[0], _run(receipt_case), b"worktree-only\n"
    declared = run[EXEC]["commit"]
    (repo / "raw/manifest").write_bytes(new)
    if attack in {"blob", "corrupt"}:
        old = _git(repo, "rev-parse", f"{declared}:raw/manifest")
        substitute = _git(repo, "hash-object", "-w", "raw/manifest")
    elif attack == "commit":
        raw = run["transformation_link"]["raw_execution_manifest"]
        run[ANALYSIS] = _id(declared, run[EXEC]["source_inventory"][0], raw, False)
        _git(repo, "add", "raw/manifest")
        _git(repo, "commit", "-qm", "substitute")
        old, substitute = declared, _git(repo, "rev-parse", "HEAD")
    if attack in {"blob", "commit"}:
        _git(repo, "replace", old, substitute)
    elif attack == "corrupt":
        objects = repo / ".git/objects"
        target = objects / old[:2] / old[2:]
        forged = objects / substitute[:2] / substitute[2:]
        target.chmod(0o644)
        target.write_bytes(forged.read_bytes())
    digest = _set_raw_digest(run, new)
    assert hashlib.sha256((repo / "raw/manifest").read_bytes()).hexdigest() == digest
    message = "object graph integrity failed" if attack == "corrupt" else DRIFT
    _fails(receipt_case, message)


def test_poisoned_git_dir_cannot_redirect_repo_root(receipt_case, monkeypatch) -> None:
    poison = receipt_case[0].parent / "poison"
    poison.mkdir()
    _git(poison, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(poison / ".git"))
    result = verifier.verify_receipt(receipt_case[1], repo_root=receipt_case[0])
    assert result["status"] == "VERIFIED"


@pytest.mark.parametrize(
    ("key", "duplicate"),
    [
        ("schema_version", "1"),
        ("schema_version", "2"),
        ("runs", "[]"),
        ("run_id", '"other"'),
        ("source_inventory", "[]"),
        ("sha256", '"' + "0" * 64 + '"'),
        ("raw_execution_manifest", "{}"),
    ],
)
def test_duplicate_json_member_is_rejected(receipt_case, key, duplicate) -> None:
    text = json.dumps(receipt_case[2], separators=(",", ":"))
    needle, injected = f'"{key}":', f'"{key}":{duplicate},"{key}":'
    if key == "schema_version" and duplicate == "2":
        needle, injected = '"schema_version":2', '"schema_version":2,"schema_version":1'
    text = text.replace(needle, injected, 1)
    receipt_case[1].write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"duplicate JSON member: {key}"):
        verifier.verify_receipt(receipt_case[1], repo_root=receipt_case[0])


@pytest.mark.parametrize("version", [1, True, 2.0, "2", 3])
def test_invalid_schema_version_fails(receipt_case, version) -> None:
    receipt_case[2]["schema_version"] = version
    message = (
        "legacy receipt schema v1 is invalid/unbound"
        if type(version) is int and version == 1
        else "receipt schema_version must be integer 2"
    )
    _fails(receipt_case, message)


def test_runs_must_be_nonempty(receipt_case) -> None:
    receipt_case[2]["runs"] = []
    _fails(receipt_case, "receipt contains no run entries")


@pytest.mark.parametrize("identity", [EXEC, ANALYSIS])
@pytest.mark.parametrize("inventory", verifier.INVENTORY_NAMES)
def test_nonempty_identity_inventories(receipt_case, identity, inventory) -> None:
    _run(receipt_case)[identity][inventory] = []
    _fails(receipt_case, rf"{identity}\.{inventory} must be nonempty")


@pytest.mark.parametrize("identity", [EXEC, ANALYSIS])
@pytest.mark.parametrize("oid", ["0" * 40, "A" * 40, "0" * 39])
def test_unresolvable_identity_commit_fails(receipt_case, identity, oid) -> None:
    _run(receipt_case)[identity]["commit"] = oid
    _fails(receipt_case, rf"{identity}\.commit cannot be resolved as commit")


@pytest.mark.parametrize("identity", [EXEC, ANALYSIS])
def test_dirty_identity_fails(receipt_case, identity) -> None:
    _run(receipt_case)[identity]["dirty"] = True
    _fails(receipt_case, rf"{identity}\.dirty must be false")


@pytest.mark.parametrize(
    "path", ["../x", "/x", "a/../x", "./x", "a//x", "a\\x", "a/", "-x", "a\0x"]
)
def test_noncanonical_inventory_path_fails(receipt_case, path) -> None:
    _run(receipt_case)[EXEC]["source_inventory"][0]["path"] = path
    _fails(receipt_case, "path must be a normalized in-repository POSIX path")


@pytest.mark.parametrize(
    ("path", "payload", "message"),
    [
        ("exec/link", b"data", "Git-tree path is not a regular file"),
        ("exec/missing", b"missing", "Git-tree path missing"),
    ],
)
def test_tree_path_rejections(receipt_case, path, payload, message) -> None:
    _run(receipt_case)[EXEC]["source_inventory"] = [_entry(path, payload)]
    _fails(receipt_case, message)


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "g" * 64])
def test_malformed_sha256_fails(receipt_case, digest) -> None:
    _run(receipt_case)[EXEC]["source_inventory"][0]["sha256"] = digest
    _fails(receipt_case, "sha256 must be 64 lowercase hex characters")


def test_duplicate_inventory_path_fails(receipt_case) -> None:
    items = _run(receipt_case)[EXEC]["source_inventory"]
    items.append(items[0].copy())
    _fails(receipt_case, "duplicate path in execution_identity.source_inventory")


@pytest.mark.parametrize(
    ("identity", "inventory", "absent"),
    [
        (EXEC, "artifact_inventory", True),
        (ANALYSIS, "input_inventory", True),
        (EXEC, "artifact_inventory", False),
        (ANALYSIS, "input_inventory", False),
        ("transformation_link", "raw_execution_manifest", False),
    ],
)
def test_raw_manifest_link_failures(receipt_case, identity, inventory, absent) -> None:
    run = _run(receipt_case)
    target = run[identity][inventory]
    if absent:
        run[identity][inventory] = [run[identity]["source_inventory"][0].copy()]
        message = f"raw execution manifest is absent from {identity}"
    else:
        (target[0] if isinstance(target, list) else target)["sha256"] = "0" * 64
        message = "raw execution manifest descriptor mismatch"
    _fails(receipt_case, message)


@pytest.mark.parametrize("ancestor", [False, True])
def test_receipt_symlink_or_symlinked_ancestor_fails(receipt_case, ancestor) -> None:
    if ancestor:
        link = receipt_case[0].parent / "alias" / receipt_case[1].name
        link.parent.symlink_to(receipt_case[0], target_is_directory=True)
    else:
        link = receipt_case[0] / "linked.receipt.json"
        link.symlink_to(receipt_case[1].name)
    with pytest.raises(ValueError, match="receipt is missing or symlink"):
        verifier.verify_receipt(link, repo_root=receipt_case[0])


def _cli(path: Path, allow: bool) -> int:
    args = ["--receipts-dir", str(path)]
    return verifier.main(args + (["--allow-empty-diagnostic"] if allow else []))


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("allow", [False, True])
def test_empty_receipts_policy(tmp_path, capsys, exists, allow) -> None:
    path = tmp_path / "receipts"
    if exists:
        path.mkdir()
    code = _cli(path, allow)
    captured = capsys.readouterr()
    output = captured.out if allow else captured.err
    assert code == (0 if allow else 1)
    markers = ("[DIAGNOSTIC]", "allow-empty accepted") if allow else ("[FAIL]",)
    assert all(marker in output for marker in markers)
    expected = "no receipt files found:" if exists else "receipts directory missing:"
    assert expected in output


@pytest.mark.parametrize("kind", ["file", "final", "ancestor"])
@pytest.mark.parametrize("allow", [False, True])
def test_unsafe_receipts_path_always_fails(tmp_path, capsys, kind, allow) -> None:
    path = tmp_path / "receipts"
    if kind == "file":
        path.touch()
    elif kind == "final":
        path.symlink_to(tmp_path, target_is_directory=True)
    else:
        path = tmp_path / "alias" / "receipts"
        (tmp_path / "real/receipts").mkdir(parents=True)
        path.parent.symlink_to(tmp_path / "real", target_is_directory=True)
    code = _cli(path, allow)
    assert code == 1 and "[FAIL] receipts directory missing:" in capsys.readouterr().err
