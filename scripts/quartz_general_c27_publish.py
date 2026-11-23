"""Publish one C27 stage on its verified original or consolidated source chain."""

import datetime
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/research/quartz_general_joint_posterior_20261123"
BACKUP = Path(
    "/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta27_backups"
)
stage = sys.argv[1]
paths = sys.argv[2:]
assert paths and stage.replace("_", "").isalnum()
base = json.loads((DOC / "BASELINE.json").read_text())
active_chain_path = BACKUP / "ACTIVE_BASE_CHAIN.json"
active_chain = (
    json.loads(active_chain_path.read_text())
    if active_chain_path.exists()
    else {"archives": base["base_chain"], "anchor_commit": base["parent"]}
)
chain = active_chain["archives"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


assert git("branch", "--show-current") == base["branch"]
for path, sha in base["protected"].items():
    assert digest(ROOT / path) == sha, path
subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
# Research result directories can match inherited ignore rules; never silently omit evidence.
tracked = set(git("ls-files", "--", str(DOC.relative_to(ROOT))).splitlines())
result_files = [p for p in (DOC / "coding/results").rglob("*") if p.is_file()]
assert all(str(p.relative_to(ROOT)) in tracked for p in result_files), (
    "untracked C27 result evidence"
)
# Raw stdout/stderr are byte-preserved evidence; trailing terminal newlines are not source defects.
subprocess.run(
    [
        "git",
        "diff",
        "--cached",
        "--check",
        "--",
        ".",
        ":(exclude,glob)**/*.log",
        ":(exclude,glob)**/*.stdout",
        ":(exclude,glob)**/*.stderr",
    ],
    cwd=ROOT,
    check=True,
)
assert (
    subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
    ).returncode
    == 1
)
subprocess.run(
    [
        "git",
        "commit",
        "-m",
        f"research(c27 {stage}): joint posterior and finite metareasoning",
    ],
    cwd=ROOT,
    check=True,
    stdout=subprocess.DEVNULL,
)
head = git("rev-parse", "HEAD")
assert not git("diff", "--name-only", "--diff-filter=D", base["parent"], head)
changed = git("diff", "--name-only", base["parent"], head).splitlines()
assert changed
for path in changed:
    assert path.startswith(
        "docs/research/quartz_general_joint_posterior_20261123/"
    ) or path in {
        "src/bin/quartz_c27_joint_posterior.rs",
        "tests/test_c27_joint_posterior.py",
        "scripts/quartz_general_c27_publish.py",
    }, path
archive = BACKUP / f"QUARTZ_GENERAL_C27_{stage}_{head[:12]}.tar.gz"
assert not archive.exists()
with archive.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as z:
    p = subprocess.Popen(
        ["git", "archive", "--format=tar", head, "--", *changed],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    for block in iter(lambda: p.stdout.read(1048576), b""):
        z.write(block)
    assert p.wait() == 0
sha = digest(archive)
subprocess.run(
    ["git", "push", "--quiet", "-u", "origin", f"HEAD:refs/heads/{base['branch']}"],
    cwd=ROOT,
    check=True,
)
remote = git("ls-remote", "--heads", "origin", base["branch"]).split()[0]
assert remote == head
receipt = {
    "stage": stage,
    "commit": head,
    "remote": remote,
    "archive": str(archive),
    "archive_sha256": sha,
    "archive_bytes": archive.stat().st_size,
    "archive_kind": "cumulative C27 overlay",
    "base_chain": chain,
    "source_chain_anchor": active_chain["anchor_commit"],
    "changed_files": len(changed),
    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "dropbox_cloud_ack": "NOT_VERIFIED",
}
with (BACKUP / "STAGE_RECEIPTS.jsonl").open("a") as f:
    f.write(json.dumps(receipt) + "\n")
print(json.dumps(receipt))
