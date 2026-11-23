#!/usr/bin/env python3
"""C28 scoped stage publication with cumulative, history-preserving backups."""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path('/home/cosmosapjw/quantum/research/quartz_c28_action_geometry_20261123_486e')
BACKUP = Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_c28_action_geometry_backups')
DOC = 'docs/research/quartz_c28_action_geometry_20261123'
BASE = '87c15dc828d6eea519c219fd8a412fa21dfc464d'
BRANCH = 'research/quartz-c28-action-geometry-20261123-486e'
STAGES = [f'R{i:02}' for i in range(1, 11)] + [f'C{i:02}' for i in range(1, 11)]

def git(*args, binary=False):
    p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(f'git {args}: {p.stdout.decode(errors="replace")} {p.stderr.decode(errors="replace")}')
    return p.stdout if binary else p.stdout.decode().strip()

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('--next', required=True)
    a = ap.parse_args()
    receipts = BACKUP / 'STAGE_RECEIPTS.jsonl'
    old = [json.loads(x) for x in receipts.read_text().splitlines()] if receipts.exists() else []
    assert a.stage == STAGES[len(old)], 'stage order/duplicate publication'
    assert git('branch', '--show-current') == BRANCH
    assert (ROOT / DOC / 'stages' / (a.stage + '.md')).is_file(), 'missing actual stage output'
    parent = git('rev-parse', 'HEAD')
    if old:
        assert parent == old[-1]['commit'], 'unexpected HEAD change'
    else:
        assert parent == BASE
    scopes = [ROOT / DOC, ROOT / 'scripts/quartz_c28_publish.py', ROOT / 'src/bin/quartz_c28_action_geometry.rs', ROOT / 'tests/test_c28_action_geometry.py']
    paths = []
    for scope in scopes:
        for path in ([scope] if scope.is_file() else scope.rglob('*') if scope.exists() else []):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.tmp'}:
                paths.append(str(path.relative_to(ROOT)))
    git('add', '-f', '--', *sorted(paths))
    # Preserve raw process log bytes, including the test runner's blank EOF.
    # Whitespace checking still covers every staged source/document/data path.
    checked_paths = [p for p in sorted(paths) if not (p.startswith(DOC + '/evidence/') and p.endswith('.log'))]
    git('diff', '--cached', '--check', '--', *checked_paths)
    assert git('diff', '--cached', '--name-only'), 'empty stage'
    git('commit', '-m', f'research(c28 {a.stage}): signed decision geometry and lifted Bellman')
    head = git('rev-parse', 'HEAD')
    changed = git('diff', '--name-only', '--diff-filter=ACMRT', BASE, head).splitlines()
    deleted = git('diff', '--name-only', '--diff-filter=D', BASE, head).splitlines()
    archive = BACKUP / f'C28_{a.stage}_{head[:12]}.tar.gz'
    raw = git('archive', '--format=tar', head, '--', *changed, binary=True)
    with archive.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as out:
        out.write(raw)
    verified = 0
    with tarfile.open(archive, 'r:gz') as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        assert set(members) == set(changed), 'missing or surplus scientific payload'
        for name in changed:
            data = tf.extractfile(members[name]).read()
            mode, kind, blob = git('ls-tree', head, '--', name).split('\t', 1)[0].split()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert actual == blob, name
            assert bool(members[name].mode & 0o111) == (mode == '100755'), name
            verified += 1
    bundle = BACKUP / f'C28_{a.stage}_{head[:12]}.bundle'
    git('bundle', 'create', str(bundle), BRANCH, '^' + BASE)
    git('bundle', 'verify', str(bundle))
    git('push', 'origin', f'HEAD:refs/heads/{BRANCH}')
    remote = git('ls-remote', '--heads', 'origin', f'refs/heads/{BRANCH}').split()[0]
    assert remote == head, 'remote head mismatch'
    assert not git('status', '--porcelain'), 'unexpected unstaged artifacts'
    record = {'stage': a.stage, 'commit': head, 'parent': parent, 'tree': git('rev-parse', 'HEAD^{tree}'),
              'remote': remote, 'archive': str(archive), 'archive_sha256': sha(archive),
              'archive_bytes': archive.stat().st_size, 'bundle': str(bundle), 'bundle_sha256': sha(bundle),
              'bundle_bytes': bundle.stat().st_size, 'base_receipt': str(BACKUP / 'BASE.json'),
              'verified_current_blobs_and_modes': verified, 'deleted_from_base': deleted,
              'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'cloud_sync': 'NOT_INDEPENDENTLY_VERIFIED'}
    with receipts.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_json(BACKUP / 'LATEST.json', record)
    removed = []
    if old:
        git('merge-base', '--is-ancestor', old[-1]['commit'], head)
        for key in ['archive', 'bundle']:
            p = Path(old[-1][key])
            assert p.parent == BACKUP and p.name.startswith('C28_')
            if p.exists():
                removed.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
                p.unlink()
    cleanup = {'after_stage': a.stage, 'removed': removed, 'retention': 'base bundle plus cumulative latest history bundle retains every stage commit', 'utc': record['utc']}
    with (BACKUP / 'CLEANUP_RECEIPTS.jsonl').open('a') as f:
        f.write(json.dumps(cleanup) + '\n')
    state_path = BACKUP / 'RUN_STATE.json'
    state = json.loads(state_path.read_text())
    state.update(status=a.stage + ' published and backed up', completed=[x['stage'] for x in old] + [a.stage], next=a.next, head=head)
    write_json(state_path, state)
    print(json.dumps(record, ensure_ascii=False))
    print('DUPLICATE_BACKUPS_REMOVED', len(removed), sum(x['bytes'] for x in removed))

if __name__ == '__main__':
    main()
