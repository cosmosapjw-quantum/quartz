#!/usr/bin/env python3
"""Restore C28 from retained bundles and verify exact bytes and stage ancestry.

This audits the externally recorded snapshot, never the mutable working tree.
It does not contact the origin until all local restoration checks have passed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import time

BACKUP = Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_c28_action_geometry_backups')
RESTORE_PARENT = Path('/home/cosmosapjw/quantum/research')
BASE_REF = 'refs/heads/research/quartz-general-c26-game-metareasoning-20261123'
DOC = 'docs/research/quartz_c28_action_geometry_20261123'
STAGES = [f'R{i:02}' for i in range(1, 11)] + [f'C{i:02}' for i in range(1, 11)]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args: list[str], *, cwd: Path | None = None) -> bytes:
    proc = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f'{args!r} exited {proc.returncode}: {proc.stderr.decode(errors="replace")}')
    return proc.stdout


def git(repo: Path, *args: str) -> bytes:
    return run(['git', '-C', str(repo), *args])


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--expect-stage', required=True, choices=STAGES)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', args.label):
        raise ValueError('label must be an explicit short safe identifier')
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    base = json.loads((BACKUP / 'BASE.json').read_text())
    latest_bytes = (BACKUP / 'LATEST.json').read_bytes()
    latest = json.loads(latest_bytes)
    receipts = [json.loads(line) for line in (BACKUP / 'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    count = STAGES.index(args.expect_stage) + 1
    assert latest['stage'] == args.expect_stage
    assert [r['stage'] for r in receipts] == STAGES[:count]
    assert receipts[-1] == latest
    paths = [Path(base['base_bundle']), Path(latest['bundle']), Path(latest['archive'])]
    expected = [base['base_sha256'], latest['bundle_sha256'], latest['archive_sha256']]
    sizes = [base['base_bytes'], latest['bundle_bytes'], latest['archive_bytes']]
    retained = []
    for path, sha256, size in zip(paths, expected, sizes):
        assert path.parent == BACKUP and path.is_file()
        assert path.stat().st_size == size
        actual = digest(path)
        assert actual == sha256, str(path)
        retained.append({'path': str(path), 'bytes': size, 'sha256': actual})
    restore = RESTORE_PARENT / f'quartz_c28_restore_{args.label}_{latest["commit"][:12]}'
    restore.mkdir(exist_ok=False)
    marker = {'kind': 'quartz-c28-disposable-restore', 'snapshot': latest['commit'], 'label': args.label}
    (restore / 'C28_RESTORE_MARKER.json').write_text(json.dumps(marker) + '\n')
    repo = restore / 'repo'
    run(['git', 'init', '--quiet', str(repo)])
    git(repo, 'fetch', '--quiet', str(paths[0]), BASE_REF + ':refs/heads/baseline')
    git(repo, 'fetch', '--quiet', str(paths[1]), 'refs/heads/' + base['branch'] + ':refs/heads/c28')
    head = git(repo, 'rev-parse', 'refs/heads/c28').decode().strip()
    assert head == latest['commit']
    assert git(repo, 'rev-parse', 'refs/heads/baseline').decode().strip() == base['ssot']
    git(repo, 'checkout', '--quiet', '--detach', head)
    tree = git(repo, 'rev-parse', 'HEAD^{tree}').decode().strip()
    assert tree == latest['tree']
    expected_parent = base['ssot']
    for receipt in receipts:
        commit = receipt['commit']
        assert receipt['remote'] == commit
        assert receipt['parent'] == expected_parent
        parents = git(repo, 'show', '-s', '--format=%P', commit).decode().strip().split()
        assert parents == [expected_parent], (receipt['stage'], parents)
        assert git(repo, 'rev-parse', commit + '^{tree}').decode().strip() == receipt['tree']
        expected_parent = commit
    git(repo, 'fsck', '--connectivity-only', '--no-dangling')
    entries = {}
    payload_bytes = 0
    for record in git(repo, 'ls-tree', '-rz', head).split(b'\0'):
        if not record:
            continue
        metadata, name = record.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        rel = os.fsdecode(name)
        path = repo / rel
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        file_stat = path.lstat()
        if mode == '120000':
            assert stat.S_ISLNK(file_stat.st_mode)
            data = os.fsencode(os.readlink(path))
        else:
            assert kind == 'blob' and mode in ('100644', '100755'), (rel, mode, kind)
            assert stat.S_ISREG(file_stat.st_mode)
            assert bool(file_stat.st_mode & 0o111) == (mode == '100755'), rel
            data = path.read_bytes()
        assert blob_hash(data) == oid, rel
        payload_bytes += len(data)
        entries[rel] = (mode, oid)
    assert not git(repo, 'status', '--porcelain', '--untracked-files=all').strip()
    changed = {os.fsdecode(p) for p in git(repo, 'diff', '--name-only', '-z', '--diff-filter=ACMRT', base['ssot'], head).split(b'\0') if p}
    with tarfile.open(paths[2], 'r:gz') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile() or m.issym()}
        assert set(members) == changed
        for name in changed:
            member = members[name]
            mode, oid = entries[name]
            if member.issym():
                assert mode == '120000'
                data = os.fsencode(member.linkname)
            else:
                assert mode != '120000'
                data = archive.extractfile(member).read()
                assert bool(member.mode & 0o111) == (mode == '100755')
            assert blob_hash(data) == oid, name
    protected = json.loads((repo / DOC / 'evidence/C01_BASELINE.json').read_text())['inputs']
    for name, sha256 in protected.items():
        assert digest(repo / name) == sha256, ('protected input changed', name)
    measured = None
    binary_receipt = BACKUP / 'MEASURED_BINARY.json'
    if count >= 19 or binary_receipt.exists():
        measured = json.loads(binary_receipt.read_text())
        executable = Path(measured['backup'])
        assert executable.parent == BACKUP and executable.name.startswith('MEASURED_NATIVE_')
        assert executable.stat().st_size == measured['bytes']
        assert oct(executable.stat().st_mode & 0o777) == measured['mode']
        assert digest(executable) == measured['sha256']
        benchmark = json.loads((repo / DOC / 'evidence/c08_benchmark/metadata.json').read_text())
        assert measured['sha256'] == benchmark['identities']['native_binary']['sha256']
        assert digest(repo / 'src/bin/quartz_c28_action_geometry.rs') == measured['native_source_sha256']
    origin_head = git(Path(base['worktree']), 'ls-remote', '--heads', 'origin', 'refs/heads/' + base['branch']).decode().split()[0]
    assert origin_head == head
    assert (BACKUP / 'LATEST.json').read_bytes() == latest_bytes, 'snapshot changed during restore audit'
    archive_names = sorted(p.name for p in BACKUP.glob('C28_*.tar.gz'))
    delta_names = sorted(p.name for p in BACKUP.glob('C28_*.bundle'))
    assert archive_names == [paths[2].name]
    assert delta_names == [paths[1].name]
    cleanup = [json.loads(line) for line in (BACKUP / 'CLEANUP_RECEIPTS.jsonl').read_text().splitlines()]
    assert [r['after_stage'] for r in cleanup] == STAGES[:count]
    removal_records = [item for stage in cleanup for item in stage['removed']]
    assert len(removal_records) == 2 * (count - 1)
    assert all(not Path(item['path']).exists() for item in removal_records)
    assert json.loads((restore / 'C28_RESTORE_MARKER.json').read_text()) == marker
    assert restore.parent == RESTORE_PARENT and restore.name.startswith('quartz_c28_restore_')
    shutil.rmtree(restore)
    result = {
        'schema': 'quartz-c28-restoration-audit-v1',
        'status': 'VERIFIED', 'audited_stage': args.expect_stage, 'commit': head,
        'tree': tree, 'ssot': base['ssot'], 'remote_head': origin_head,
        'ordered_stage_count': count, 'ordered_stages': [r['stage'] for r in receipts],
        'retained_backups': retained, 'base_plus_delta_restored': True,
        'exact_measured_executable_verified': measured,
        'all_tracked_blob_bytes_and_modes_verified': len(entries),
        'tracked_payload_bytes': payload_bytes, 'cumulative_archive_verified_files': len(changed),
        'protected_input_hashes_verified': protected,
        'duplicate_archives_removed': len(removal_records),
        'duplicate_archive_bytes_unlinked': sum(item['bytes'] for item in removal_records),
        'temporary_restore_removed_after_success': str(restore),
        'script_sha256': digest(Path(__file__)),
        'cloud_sync': 'NOT_INDEPENDENTLY_VERIFIED',
        'duration_seconds': time.perf_counter() - started,
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
