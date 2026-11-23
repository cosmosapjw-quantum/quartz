"""Verify retained C27 archives/history and worktree evidence; optionally restore/run."""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REL = Path("docs/research/quartz_general_joint_posterior_20261123")
DOC = ROOT / REL
C26 = Path("docs/research/quartz_general_game_metareasoning_20261123")
BACKUP = Path(
    "/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta27_backups"
)
DEFAULT_BINARY = Path(
    "/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123/target/release/quartz_c27_joint_posterior"
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def consolidation_catalog():
    path = BACKUP / "CONSOLIDATION_CATALOG.json"
    if not path.exists():
        return None
    catalog = json.loads(path.read_text())
    assert catalog["schema"] == "c27-consolidation-v1"
    return catalog


def verify_file(item, checked):
    """Hash each immutable archive/container once per verification invocation."""
    path = Path(item["path"])
    before = path.stat()
    assert path.is_file() and before.st_size == item["bytes"], str(path)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    key = str(path)
    if key in checked:
        prior = checked[key]
        assert prior["sha256"] == item["sha256"] and prior["identity"] == identity
        return prior
    assert digest(path) == item["sha256"], str(path)
    after = path.stat()
    assert identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    checked[key] = {
        "path": key,
        "sha256": item["sha256"],
        "bytes": before.st_size,
        "identity": identity,
    }
    return checked[key]


def verified_history(catalog, checked):
    assert catalog is not None, "missing consolidation catalog for retired evidence"
    history = catalog["history"]
    assert history["verified"] is True, "history bundle lacks verified record"
    verify_file(history, checked)
    heads = {}
    for line in git("bundle", "list-heads", history["path"]).splitlines():
        oid, ref = line.split(" ", 1)
        heads[ref] = oid
    assert history["commit"] in heads.values(), "history commit not advertised"
    return {
        **history,
        "advertised_refs": heads,
        "container_sha256_verified_this_run": True,
        "bundle_restore_performed_this_run": False,
        "basis": "prior verified bundle record, current container hash and advertised refs",
    }


def stage_archives(selected, catalog, checked):
    retired = {}
    if catalog is not None:
        for item in catalog["retired_stage_archives"]:
            assert item["path"] not in retired, "duplicate retired archive path"
            retired[item["path"]] = item
    history = None
    result = []
    for receipt in selected:
        assert receipt["commit"] == receipt["remote"], receipt["stage"]
        record = {
            k: receipt[k]
            for k in ("stage", "commit", "archive", "archive_sha256", "archive_bytes")
        }
        if Path(receipt["archive"]).exists():
            verify_file(
                {
                    "path": receipt["archive"],
                    "sha256": receipt["archive_sha256"],
                    "bytes": receipt["archive_bytes"],
                },
                checked,
            )
            record.update(
                state="present_archive_bytes_verified",
                original_archive_bytes_verified_this_run=True,
            )
        else:
            item = retired.get(receipt["archive"])
            assert item is not None, (
                "missing archive without retirement record",
                record,
            )
            for key, source_key in (
                ("sha256", "archive_sha256"),
                ("bytes", "archive_bytes"),
                ("commit", "commit"),
                ("stage", "stage"),
            ):
                assert item[key] == receipt[source_key], (key, record)
            assert item["recovery"] == "git_bundle"
            if history is None:
                history = verified_history(catalog, checked)
            assert git("cat-file", "-t", item["commit"]) == "commit"
            git("merge-base", "--is-ancestor", item["commit"], history["commit"])
            record.update(
                state="retired_archive_history_preserved",
                original_archive_bytes_verified_this_run=False,
                original_archive_restored=False,
                recovery="git_bundle",
                history_bundle=history["path"],
                history_bundle_sha256=history["sha256"],
                catalog_metadata_matches_receipt=True,
            )
        result.append(record)
    return result, history


def selected_archives(last, catalog):
    if catalog is not None and catalog["full_source"]["commit"] == last["commit"]:
        return [dict(catalog["full_source"])], "consolidated_full_source"
    assert "base_chain" in last, "stage receipt must declare its actual base chain"
    return list(last["base_chain"]) + [
        {
            "path": last["archive"],
            "sha256": last["archive_sha256"],
            "bytes": last["archive_bytes"],
            "kind": last.get("archive_kind", "cumulative C27 overlay"),
        }
    ], "stage_receipt_base_chain_plus_overlay"


def noncache_recovery(receipt, catalog, checked):
    if not receipt.get("noncache_files"):
        return None
    path = Path(
        receipt.get(
            "noncache_preservation_proof", BACKUP / "C24_UNIQUE_PRESERVATION.json"
        )
    )
    proof = json.loads(path.read_text())
    assert proof["verified"] is True
    assert set(receipt["noncache_files"]) <= set(proof["relative_paths"])
    sources = {item["source_path"]: item for item in proof["sources"]}
    for rel in receipt["noncache_files"]:
        item = sources[str(Path(receipt["path"]) / rel)]
        assert item["compressed_payload_readback_sha256_verified"] is True
        assert item["disposition"] == "retained_unique"
    assert catalog["retained_history"]["complete"] is True
    assert (
        proof["retention_summary_sha256"]
        == catalog["retained_history"]["summary_sha256"]
    )
    archive = Path(proof["retained_archive"])
    verify_file(
        {
            "path": str(archive),
            "sha256": proof["retained_archive_sha256"],
            "bytes": archive.stat().st_size,
        },
        checked,
    )
    return {
        "proof_path": str(path),
        "proof_sha256": digest(path),
        "relative_paths": receipt["noncache_files"],
        "retained_archive": str(archive),
        "retained_archive_sha256": proof["retained_archive_sha256"],
        "container_hash_verified_this_run": True,
        "member_readback_basis": "prior verified preservation proof",
        "files_restored_this_run": False,
    }


def old_trees(base, catalog, checked, history=None):
    registered = {}
    for block in git("worktree", "list", "--porcelain").split("\n\n"):
        row = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        if "worktree" in row:
            registered[row["worktree"]] = row
    cleanup_path = BACKUP / "WORKTREE_CLEANUP_RECEIPT.jsonl"
    cleanup = {}
    if cleanup_path.exists():
        for line in cleanup_path.read_text().splitlines():
            receipt = json.loads(line)
            if receipt.get("exit_code") == 0:
                cleanup[receipt["path"]] = receipt
    result = []
    c26_root = DEFAULT_BINARY.parents[2]
    for old in base["old_worktrees"]:
        root = Path(old["path"])
        present = root.is_dir()
        is_registered = str(root) in registered
        record = {
            "path": str(root),
            "baseline_head": old["head"],
            "baseline_status": old["status"],
            "path_exists": present,
            "registered": is_registered,
            "mutated_by_this_verifier": False,
        }
        if root == c26_root:
            assert present and is_registered, "active C26 dependency missing"
        if not present and not is_registered and str(root) in cleanup:
            receipt = cleanup[str(root)]
            assert receipt["head"] == old["head"]
            assert receipt["refs_and_history_deleted"] is False
            assert receipt["live_origin_exact_match"] is True
            assert receipt["bundle_ref_verified"] is True
            assert receipt["command"] == ["git", "worktree", "remove", str(root)]
            branch = receipt["branch"]
            assert branch.startswith("refs/heads/")
            assert git("rev-parse", "--verify", branch) == old["head"]
            assert git("cat-file", "-t", old["head"]) == "commit"
            assert receipt["restore_command"] == [
                "git",
                "worktree",
                "add",
                str(root),
                branch.removeprefix("refs/heads/"),
            ]
            if history is None:
                history = verified_history(catalog, checked)
            assert history["advertised_refs"].get(branch) == old["head"]
            record.update(
                state="intentionally_removed_verified_recovery",
                head=old["head"],
                branch=branch,
                head_status_unchanged=None,
                protected_unchanged=None,
                recovery_commit_and_branch_present=True,
                history_bundle=history["path"],
                cleanup_receipt=receipt,
                noncache_preservation=noncache_recovery(receipt, catalog, checked),
                worktree_restored_this_run=False,
            )
        elif not present or not is_registered:
            record.update(
                state="missing_without_our_receipt"
                if str(root) not in cleanup
                else "inconsistent_presence_after_cleanup",
                cause="external_or_unresolved",
                head_status_unchanged=None,
                protected_unchanged=None,
                cleanup_receipt=None,
            )
        else:
            head = git("rev-parse", "HEAD", cwd=root)
            status = git("status", "--porcelain=v1", "--untracked-files=all", cwd=root)
            same = head == old["head"] and status == old["status"]
            mismatches = [
                rel
                for rel, sha in old["own_protected"].items()
                if not (root / rel).is_file() or digest(root / rel) != sha
            ]
            if root == c26_root:
                assert same and not mismatches, "active C26 baseline changed"
                for rel, sha in base["protected"].items():
                    assert digest(root / rel) == sha, ("C26 dependency", rel)
                assert digest(
                    DEFAULT_BINARY.with_name("quartz_c26_game_metareasoning")
                ) == (
                    "8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c"
                ), "protected C26 executable changed"
            record.update(
                state="unchanged"
                if same and not mismatches
                else "changed_external_or_unresolved",
                head=head,
                observed_status=status,
                branch=registered[str(root)].get("branch"),
                head_status_unchanged=same,
                protected_unchanged=not mismatches,
                protected_mismatches=mismatches,
            )
        result.append(record)
    for rel, sha in base["protected"].items():
        assert digest(ROOT / rel) == sha, ("current protected file", rel)
    return result, history


def tree(commit):
    expected = {}
    for row in subprocess.check_output(
        ["git", "ls-tree", "-rz", commit], cwd=ROOT
    ).split(b"\0"):
        if not row:
            continue
        meta, raw_name = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        assert kind == "blob", meta
        expected[os.fsdecode(raw_name)] = (mode, oid)
    return expected


def stream_tree(archives, algorithm):
    final = {}
    total = 0
    for item in archives:
        with tarfile.open(item["path"], "r|gz") as archive:
            for member in archive:
                rel = member.name.rstrip("/")
                if member.isdir():
                    continue
                assert not Path(rel).is_absolute() and ".." not in Path(rel).parts, rel
                if member.issym():
                    value = os.fsencode(member.linkname)
                    h = hashlib.new(
                        algorithm, b"blob " + str(len(value)).encode() + b"\0"
                    )
                    h.update(value)
                    mode, size = "120000", len(value)
                else:
                    assert member.isfile(), (rel, member.type)
                    h = hashlib.new(
                        algorithm, b"blob " + str(member.size).encode() + b"\0"
                    )
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1 << 20), b""):
                            h.update(block)
                    mode = "100755" if member.mode & 0o111 else "100644"
                    size = member.size
                final[rel] = (mode, h.hexdigest(), size)
    total = sum(x[2] for x in final.values())
    return final, total


def physical_tree(folder, expected, algorithm):
    total = links = 0
    for rel, (mode, oid) in expected.items():
        p = folder / rel
        if mode == "120000":
            assert p.is_symlink(), rel
            value = os.fsencode(os.readlink(p))
            h = hashlib.new(algorithm, b"blob " + str(len(value)).encode() + b"\0")
            h.update(value)
            size = len(value)
            links += 1
        else:
            assert p.is_file() and not p.is_symlink(), rel
            assert bool(p.stat().st_mode & 0o111) == (mode == "100755"), rel
            size = p.stat().st_size
            h = hashlib.new(algorithm, b"blob " + str(size).encode() + b"\0")
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        assert h.hexdigest() == oid, rel
        total += size
    actual = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    assert actual == set(expected), {
        "missing": sorted(set(expected) - actual)[:8],
        "extra": sorted(actual - set(expected))[:8],
    }
    return total, links


def semantic(value):
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if not k.endswith(("_ns", "_us")) and "elapsed" not in k
        }
    if isinstance(value, list):
        return [semantic(x) for x in value]
    return value


def restored_execution(folder, binary):
    sha = digest(binary)
    packed = BACKUP / ("C27_NATIVE_RELEASE_" + sha[:16] + ".gz")
    if not packed.exists():
        with (
            packed.open("xb") as out,
            gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as z,
            binary.open("rb") as src,
        ):
            shutil.copyfileobj(src, z)
    recovered = folder / "c27_restored_native"
    with gzip.open(packed, "rb") as z, recovered.open("xb") as out:
        shutil.copyfileobj(z, out)
    recovered.chmod(0o755)
    assert digest(recovered) == sha
    with gzip.open(folder / C26 / "coding/data/training_raw.jsonl.gz", "rt") as f:
        row = json.loads(next(f))
    request = dict(row["request"])
    request.update(mode="run", additional=2, repeats=1, warmup=0, trace=True, rho=0.5)
    request["model"] = json.loads((folder / C26 / "coding/data/model.json").read_text())
    payload = "{malformed\n" + json.dumps(request) + "\n"
    responses = []
    for exe in (binary, recovered):
        p = subprocess.run(
            [str(exe)],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            cwd=folder,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        assert len(lines) == 2 and "error" in lines[0] and "error" not in lines[1], (
            lines
        )
        assert len(lines[1]["runs"]) == 5
        responses.append(semantic(lines[1]))
    assert responses[0] == responses[1], "restored native semantic mismatch"
    p = subprocess.run(
        [sys.executable, str(folder / REL / "coding/c27_campaign.py"), "--help"],
        cwd=folder,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr
    return {
        "native_sha256": sha,
        "binary_backup": str(packed),
        "compressed_sha256": digest(packed),
        "compressed_bytes": packed.stat().st_size,
        "restored_native_semantics_equal": True,
        "malformed_line_recovery": True,
        "restored_harness_help_exit": p.returncode,
        "scope": "same first C26 TRAIN public request at rho0.5; timing fields excluded from byte-independent semantic comparison",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--keep-restore", action="store_true")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), "preserve prior verification receipt"
    began = time.perf_counter()
    base = json.loads((DOC / "BASELINE.json").read_text())
    receipts = [
        json.loads(x)
        for x in (BACKUP / "STAGE_RECEIPTS.jsonl").read_text().splitlines()
    ]
    order = [f"R{i:02d}" for i in range(1, 11)] + [f"C{i:02d}" for i in range(1, 11)]
    selected = receipts[: order.index(args.stage) + 1]
    assert [x["stage"] for x in selected] == order[: len(selected)]
    assert selected[-1]["stage"] == args.stage
    catalog = consolidation_catalog()
    checked = {}
    stage_checks, history = stage_archives(selected, catalog, checked)
    last = selected[-1]
    archives, archive_selection = selected_archives(last, catalog)
    for item in archives:
        verify_file(item, checked)
    expected = tree(last["commit"])
    algorithm = git("rev-parse", "--show-object-format")
    streamed, byte_count = stream_tree(archives, algorithm)
    assert {k: v[:2] for k, v in streamed.items()} == expected, (
        "archive-composed Git tree mismatch"
    )
    worktrees, history = old_trees(base, catalog, checked, history)
    unresolved = [
        item
        for item in worktrees
        if item["state"] not in ("unchanged", "intentionally_removed_verified_recovery")
    ]
    result = {
        "schema": "c27-delivery-verification-v2",
        "stage": args.stage,
        "commit": last["commit"],
        "archives": archives,
        "archive_selection": archive_selection,
        "archive_chain_length": len(archives),
        "stage_receipts": stage_checks,
        "archive_stage_count": len(stage_checks),
        "full_streamed_tree_verified": True,
        "tracked_files": len(expected),
        "tracked_bytes": byte_count,
        "old_worktrees": worktrees,
        "old_worktree_comparison_complete": not unresolved,
        "old_worktree_comparison_status": "VERIFIED"
        if not unresolved
        else "EXTERNAL_OR_UNRESOLVED",
        "unresolved_worktree_paths": [item["path"] for item in unresolved],
        "history_bundle_verification": history,
        "retired_stage_archive_count": sum(
            item["state"] == "retired_archive_history_preserved"
            for item in stage_checks
        ),
        "c26_dependency_unchanged": True,
        "current_protected_unchanged": True,
        "physical_restore": False,
        "free_bytes_before": shutil.disk_usage(ROOT).free,
        "dropbox_cloud_ack": "NOT_VERIFIED",
    }
    if args.physical:
        assert shutil.disk_usage(ROOT).free > byte_count + 150_000_000, (
            "insufficient physical restore space"
        )
        folder = Path(tempfile.mkdtemp(prefix="quartz_c27_restore_"))
        result["restore_directory"] = str(folder)
        for item in archives:
            with tarfile.open(item["path"], "r:gz") as archive:
                archive.extractall(folder, filter="data")
        count, links = physical_tree(folder, expected, algorithm)
        assert count == byte_count
        result.update(
            physical_restore=True, restored_symlinks=links, restored_file_modes=True
        )
        result["restored_execution"] = restored_execution(folder, args.binary)
        if not args.keep_restore:
            shutil.rmtree(folder)
            result["owned_temporary_restore_removed_after_verification"] = True
    result["utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    result["elapsed_seconds"] = time.perf_counter() - began
    result["status"] = (
        "RESTORE_AND_EXECUTION" if args.physical else "VERIFIED_ARCHIVE_STREAM"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "stage",
                    "commit",
                    "archive_stage_count",
                    "retired_stage_archive_count",
                    "old_worktree_comparison_complete",
                    "tracked_files",
                    "tracked_bytes",
                    "physical_restore",
                    "elapsed_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
