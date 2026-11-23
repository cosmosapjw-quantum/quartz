"""Restore the C23+C24+C25+C26 archive chain and verify Git blob identities."""

from pathlib import Path
import argparse, datetime, gzip, hashlib, json, os, shutil, subprocess, tarfile, tempfile, time

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / "docs/research/quartz_general_game_metareasoning_20261123"
BACKUP = Path(
    "/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta26_backups"
)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def verify_old_worktrees(base):
    rows = []
    for old in base["old_worktrees"]:
        root = Path(old["path"])
        assert git("rev-parse", "HEAD", cwd=root) == old["head"], str(root)
        assert (
            git("status", "--porcelain=v1", "--untracked-files=all", cwd=root)
            == old["status"]
        ), str(root)
        for rel, want in old["own_protected"].items():
            assert digest(root / rel) == want, (str(root), rel)
        rows.append(
            {
                "path": str(root),
                "head": old["head"],
                "status_unchanged": True,
                "own_protected_unchanged": True,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="C08")
    parser.add_argument("--output", default=str(DOC / "evidence/C09_RESTORE.json"))
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), "preserve existing restore receipt"
    started = time.perf_counter()
    base = json.loads((DOC / "BASELINE.json").read_text())
    receipts = [
        json.loads(line)
        for line in (BACKUP / "STAGE_RECEIPTS.jsonl").read_text().splitlines()
    ]
    matches = [r for r in receipts if r["stage"] == args.stage]
    assert len(matches) == 1
    last = matches[0]
    archives = base["base_chain"] + [
        {
            "path": last["archive"],
            "sha256": last["archive_sha256"],
            "bytes": last["archive_bytes"],
            "kind": "cumulative C26 overlay",
        }
    ]
    assert len(archives) == 4
    for item in archives:
        assert Path(item["path"]).stat().st_size == item["bytes"], item["path"]
        assert digest(item["path"]) == item["sha256"], item["path"]
    restore = Path(tempfile.mkdtemp(prefix="quartz_c26_restore_"))
    for item in archives:
        with tarfile.open(item["path"], "r:gz") as archive:
            archive.extractall(restore, filter="data")
    fmt = git("rev-parse", "--show-object-format")
    count = total = links = 0
    expected = set()
    for row in subprocess.check_output(
        ["git", "ls-tree", "-rz", last["commit"]], cwd=ROOT
    ).split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        assert kind == "blob", meta
        rel = os.fsdecode(name)
        expected.add(rel)
        path = restore / rel
        if mode == "120000":
            assert path.is_symlink(), rel
            value = os.fsencode(os.readlink(path))
            size = len(value)
            h = hashlib.new(fmt, b"blob " + str(size).encode() + b"\0")
            h.update(value)
            links += 1
        else:
            assert path.is_file() and not path.is_symlink(), rel
            assert bool(path.stat().st_mode & 0o111) == (mode == "100755"), rel
            size = path.stat().st_size
            h = hashlib.new(fmt, b"blob " + str(size).encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
        assert h.hexdigest() == oid, rel
        count += 1
        total += size
    actual = {
        str(p.relative_to(restore))
        for p in restore.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    assert actual == expected, {
        "unexpected": sorted(actual - expected)[:10],
        "missing": sorted(expected - actual)[:10],
    }
    binary = ROOT / "target/release/quartz_c26_game_metareasoning"
    binary_sha = digest(binary)
    binary_backup = BACKUP / f"C26_NATIVE_RELEASE_{binary_sha[:16]}.gz"
    if not binary_backup.exists():
        with binary_backup.open("xb") as stream:
            with gzip.GzipFile(
                filename="", fileobj=stream, mode="wb", mtime=0
            ) as compressed:
                with binary.open("rb") as source:
                    shutil.copyfileobj(source, compressed)
    h = hashlib.sha256()
    with gzip.open(binary_backup, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == binary_sha
    old_worktrees = verify_old_worktrees(base)
    for rel, want in base["protected"].items():
        assert digest(ROOT / rel) == want, rel
    result = {
        "status": "VERIFIED_FOUR_ARCHIVE_RESTORE",
        "restored_stage": args.stage,
        "restored_commit": last["commit"],
        "restore_directory": str(restore),
        "archives": archives,
        "tracked_files_verified": count,
        "tracked_bytes_verified": total,
        "symlinks_verified": links,
        "file_modes_verified": True,
        "unexpected_files": 0,
        "binary_backup": {
            "path": str(binary_backup),
            "compressed_sha256": digest(binary_backup),
            "uncompressed_sha256": binary_sha,
            "bytes": binary_backup.stat().st_size,
        },
        "old_worktrees": old_worktrees,
        "current_protected_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dropbox_cloud_ack": "NOT_VERIFIED",
        "scope": "full restored Git blob/mode identity plus executable backup readback; scientific behavior is verified separately",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "restored_commit",
                    "restore_directory",
                    "tracked_files_verified",
                    "tracked_bytes_verified",
                    "elapsed_seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
