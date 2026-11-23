"""Verify C27 stage archives, preserve old worktrees, and optionally restore/run."""

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


def old_trees(base):
    result = []
    for old in base["old_worktrees"]:
        root = Path(old["path"])
        head = git("rev-parse", "HEAD", cwd=root)
        status = git("status", "--porcelain=v1", "--untracked-files=all", cwd=root)
        same = head == old["head"] and status == old["status"]
        hashes = all(
            digest(root / rel) == sha for rel, sha in old["own_protected"].items()
        )
        result.append(
            {
                "path": str(root),
                "head": head,
                "head_status_unchanged": same,
                "protected_unchanged": hashes,
            }
        )
    assert all(
        x["head_status_unchanged"] and x["protected_unchanged"] for x in result
    ), result
    for rel, sha in base["protected"].items():
        assert digest(ROOT / rel) == sha, rel
    return result


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
    stage_checks = []
    for r in selected:
        p = Path(r["archive"])
        assert (
            p.stat().st_size == r["archive_bytes"] and digest(p) == r["archive_sha256"]
        )
        assert r["commit"] == r["remote"], r["stage"]
        stage_checks.append(
            {
                k: r[k]
                for k in (
                    "stage",
                    "commit",
                    "archive",
                    "archive_sha256",
                    "archive_bytes",
                )
            }
        )
    last = selected[-1]
    archives = base["base_chain"] + [
        {
            "path": last["archive"],
            "sha256": last["archive_sha256"],
            "bytes": last["archive_bytes"],
            "kind": "cumulative C27 overlay",
        }
    ]
    assert len(archives) == 5
    for item in archives:
        assert (
            Path(item["path"]).stat().st_size == item["bytes"]
            and digest(item["path"]) == item["sha256"]
        )
    expected = tree(last["commit"])
    algorithm = git("rev-parse", "--show-object-format")
    streamed, byte_count = stream_tree(archives, algorithm)
    assert {k: v[:2] for k, v in streamed.items()} == expected, (
        "archive-composed Git tree mismatch"
    )
    result = {
        "schema": "c27-delivery-verification-v1",
        "stage": args.stage,
        "commit": last["commit"],
        "archives": archives,
        "stage_receipts": stage_checks,
        "archive_stage_count": len(stage_checks),
        "full_streamed_tree_verified": True,
        "tracked_files": len(expected),
        "tracked_bytes": byte_count,
        "old_worktrees": old_trees(base),
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
        "VERIFIED_FIVE_ARCHIVE_RESTORE_AND_EXECUTION"
        if args.physical
        else "VERIFIED_FIVE_ARCHIVE_STREAM"
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
