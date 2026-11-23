"""Final C26 stage/archive/source audit; receipt is external to its audited commit."""

from pathlib import Path
import datetime, hashlib, json, os, subprocess, tarfile, time

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / "docs/research/quartz_general_game_metareasoning_20261123"
BACKUP = Path(
    "/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta26_backups"
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    started = time.perf_counter()
    base = json.loads((DOC / "BASELINE.json").read_text())
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    assert branch == base["branch"]
    assert not git("status", "--porcelain=v1", "--untracked-files=all")
    remote = git("ls-remote", "--heads", "origin", branch).split()[0]
    assert remote == head
    receipts = [
        json.loads(line)
        for line in (BACKUP / "STAGE_RECEIPTS.jsonl").read_text().splitlines()
    ]
    expected = [f"R{x:02d}" for x in range(1, 11)] + [f"C{x:02d}" for x in range(1, 11)]
    assert [row["stage"] for row in receipts] == expected, "exact 20-stage order"
    previous = base["parent"]
    archive_bytes = 0
    for row in receipts:
        assert row["commit"] == row["remote"]
        assert git("rev-parse", row["commit"] + "^") == previous
        previous = row["commit"]
        archive = Path(row["archive"])
        assert archive.stat().st_size == row["archive_bytes"]
        assert sha(archive) == row["archive_sha256"]
        archive_bytes += row["archive_bytes"]
    assert previous == head
    for archive in base["base_chain"]:
        assert sha(archive["path"]) == archive["sha256"]
        assert Path(archive["path"]).stat().st_size == archive["bytes"]
    for rel, want in base["protected"].items():
        assert sha(ROOT / rel) == want, rel
    policy = json.loads((DOC / "evidence/POLICY_CORRECTIONS.json").read_text())
    assert sha(ROOT / "docs/THESIS.md") == policy["after_sha256"]
    assert (
        hashlib.sha256(
            subprocess.check_output(
                ["git", "show", base["parent"] + ":docs/THESIS.md"], cwd=ROOT
            )
        ).hexdigest()
        == policy["before_sha256"]
    )
    r09 = next(row["commit"] for row in receipts if row["stage"] == "R09")
    for rel in ["coding/CODE_CONTRACT.md", "coding/SOURCE_SPECS.json"]:
        path = str(DOC.relative_to(ROOT) / rel)
        assert (
            subprocess.check_output(["git", "show", r09 + ":" + path], cwd=ROOT)
            == (ROOT / path).read_bytes()
        ), rel
    old_rows = []
    for old in base["old_worktrees"]:
        root = Path(old["path"])
        assert git("rev-parse", "HEAD", cwd=root) == old["head"]
        assert (
            git("status", "--porcelain=v1", "--untracked-files=all", cwd=root)
            == old["status"]
        )
        for rel, want in old["own_protected"].items():
            assert sha(root / rel) == want, (str(root), rel)
        old_rows.append({"path": str(root), "head": old["head"], "unchanged": True})
    changed = git("diff", "--name-only", base["parent"], head).splitlines()
    assert not git("diff", "--name-only", "--diff-filter=D", base["parent"], head)
    allowed = {
        "src/bin/quartz_c26_game_metareasoning.rs",
        "tests/test_c26_game_metareasoning.py",
        "scripts/quartz_general_c26_publish.py",
        "docs/THESIS.md",
        "docs/CLAIM_LEDGER.md",
    }
    prefix = str(DOC.relative_to(ROOT)) + "/"
    assert all(path.startswith(prefix) or path in allowed for path in changed)
    final_archive = Path(receipts[-1]["archive"])
    archived = set()
    verified_bytes = 0
    with tarfile.open(final_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            assert member.isfile(), "C26 changed surface contains only regular files"
            assert member.name in changed, member.name
            stream = archive.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
                verified_bytes += len(block)
            assert h.hexdigest() == sha(ROOT / member.name), member.name
            archived.add(member.name)
    assert archived == set(changed)
    result_files = {
        str(p.relative_to(ROOT))
        for p in (DOC / "coding/results").rglob("*")
        if p.is_file()
    }
    assert len(result_files) == 16 and result_files <= archived, (
        "all CAL/TEST evidence archived"
    )
    correction = json.loads(
        (DOC / "evidence/C08_PACKAGING_CORRECTION.json").read_text()
    )
    assert sha(correction["supplemental_archive"]) == correction["archive_sha256"]
    result = {
        "status": "VERIFIED_FINAL_C26_DELIVERY",
        "head": head,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": branch,
        "remote": remote,
        "clean_worktree": True,
        "stage_count": len(receipts),
        "stages": expected,
        "stage_archives_sha256_verified": True,
        "stage_archive_bytes": archive_bytes,
        "base_chain_verified": True,
        "final_archive": str(final_archive),
        "final_archive_sha256": sha(final_archive),
        "final_archive_member_count": len(archived),
        "final_archive_uncompressed_bytes_read": verified_bytes,
        "old_worktrees": old_rows,
        "protected_sources_unchanged": True,
        "authorized_thesis_correction_verified": True,
        "R09_contract_and_source_specs_unchanged": True,
        "changed_files": len(changed),
        "CAL_TEST_result_files_archived": len(result_files),
        "C07_evidence_omission_repaired_C08": True,
        "dropbox_cloud_ack": "See separately timestamped cloud metadata receipt; cloud content bytes not independently downloaded",
        "C27_executed": False,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "delivery/storage identity; scientific claims and inherited CI-format failures are decided in the C10 report",
    }
    out = BACKUP / ("FINAL_AUDIT_" + head[:12] + ".json")
    assert not out.exists(), "preserve previous final receipt"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"receipt": str(out), "receipt_sha256": sha(out), **result}))


if __name__ == "__main__":
    main()
