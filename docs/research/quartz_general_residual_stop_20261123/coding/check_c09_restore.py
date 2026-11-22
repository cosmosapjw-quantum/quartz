"""Restore published C08 files from its actual archive and verify byte identity."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, tarfile, tempfile
ROOT = Path('/home/cosmosapjw/quantum/research/quartz_general_meta23_20261123')
DOC = 'docs/research/quartz_general_residual_stop_20261123'
BACKUP = Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta23_backups')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
receipts = [json.loads(line) for line in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
receipt = next(x for x in receipts if x['stage'] == 'C08')
archive = Path(receipt['archive'])
assert sha(archive) == receipt['archive_sha256']
paths = [
    'src/bin/quartz_c23_residual_gate.rs',
    f'{DOC}/THEORY.md', f'{DOC}/CODING.md', f'{DOC}/SOURCE_SPECS.json',
    *[f'{DOC}/coding/{name}' for name in [
        'c23_sources.py', 'c23_run.py', 'c23_verify.py', 'c23_analyze.py',
        'reassemble_c23_trace.py']],
    *[f'{DOC}/evidence/{name}' for name in [
        'C23_PUBLIC_TEST.jsonl.gz', 'C23_ORACLE_TEST.jsonl.gz',
        'C23_TEST_TIMING.jsonl.gz', 'C23_TEST_TIMING.manifest.json',
        'C23_TEST_TRACE.parts.json', 'C23_TEST_TRACE.jsonl.gz.part00',
        'C23_TEST_TRACE.jsonl.gz.part01', 'C23_TEST_TRACE.jsonl.gz.part02',
        'C23_TEST_SUMMARY.json', 'C23_TEST_PARENT_METRICS.json',
        'C23_VERIFY_TEST_TRACE.json', 'C23_VERIFY_TEST_TIMING.json',
        'C23_COST_QUALITY.png']]
]
restore = Path(tempfile.mkdtemp(prefix='quartz_c23_c08_restore_'))
checks = []
with tarfile.open(archive, 'r:gz') as tar:
    for name in paths:
        member = tar.getmember(name)
        assert member.isfile(), name
        dest = restore/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as inp, dest.open('wb') as out:
            shutil.copyfileobj(inp, out)
        # Every selected live path is unchanged at this point after C08 publication.
        git_hash = subprocess.check_output(
            ['git','rev-parse', f"{receipt['commit']}:{name}"], cwd=ROOT, text=True).strip()
        current_hash = subprocess.check_output(
            ['git','hash-object', name], cwd=ROOT, text=True).strip()
        assert current_hash == git_hash, name
        restored_hash = sha(dest)
        assert restored_hash == sha(ROOT/name), name
        checks.append({'path':name,'git_blob':git_hash,'sha256':restored_hash,'bytes':dest.stat().st_size})
cmd = ['python3', str(restore/DOC/'coding/reassemble_c23_trace.py')]
p = subprocess.run(cmd, text=True, capture_output=True)
assert p.returncode == 0, p.stderr
whole = restore/DOC/'evidence/C23_TEST_TRACE.jsonl.gz'
spec = json.loads((restore/DOC/'evidence/C23_TEST_TRACE.parts.json').read_text())
assert sha(whole) == spec['original_sha256']
assert whole.stat().st_size == 130197446
# Preserve the exact full original outside the Git worktree; all published parts remain.
raw_original = ROOT/DOC/'evidence/C23_TEST_TRACE.jsonl.gz'
raw_backup = BACKUP/'raw/C23_TEST_TRACE.jsonl.gz'
raw_backup.parent.mkdir(exist_ok=True)
assert sha(raw_original) == spec['original_sha256']
assert not raw_backup.exists(), str(raw_backup)
shutil.move(str(raw_original), str(raw_backup))
assert sha(raw_backup) == spec['original_sha256']
review_source = Path('/tmp/quartz_c23_review.py')
review_copy = ROOT/DOC/'coding/c23_independent_review.py'
assert not review_copy.exists()
shutil.copyfile(review_source, review_copy)
assert sha(review_source) == sha(review_copy)
result = {
    'status':'VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'archive_stage':'C08','archive_commit':receipt['commit'],'archive_path':str(archive),
    'archive_sha256':receipt['archive_sha256'],'archive_bytes':archive.stat().st_size,
    'restored_directory':str(restore),'selected_files_verified':len(checks),
    'checks':checks,'reassembly':{
        'argv':cmd,'returncode':p.returncode,'stdout':p.stdout.strip(),
        'whole_sha256':sha(whole),'whole_bytes':whole.stat().st_size},
    'full_original_preserved':{'path':str(raw_backup),'sha256':sha(raw_backup)},
    'independent_review_script':{'path':str(review_copy.relative_to(ROOT)),'sha256':sha(review_copy)},
    'dropbox_cloud_ack':'NOT_VERIFIED',
    'scope':'Actual C08 archive restored selected source, inputs, oracle, raw outputs and figure; no benchmark rerun.'
}
(ROOT/DOC/'evidence/C09_LOCAL_RESTORE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
