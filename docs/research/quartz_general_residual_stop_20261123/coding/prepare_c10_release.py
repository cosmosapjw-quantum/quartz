"""Bind the C23 scientific release payload before its final stage commit."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, subprocess
ROOT=Path('/home/cosmosapjw/quantum/research/quartz_general_meta23_20261123')
DOC=Path('docs/research/quartz_general_residual_stop_20261123')
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta23_backups')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
receipts=[json.loads(s) for s in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
expected=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,7)]+['C07A','C07','C08A','C08','C09']
assert [x['stage'] for x in receipts]==expected
assert receipts[-1]['commit']==git('rev-parse','HEAD')
assert all(x['commit']==x['remote'] for x in receipts)
history={'schema':'quartz.stage-history.v1','through':'C09','verified_receipts':receipts,
         'C10':'Final self-identifying receipt is external: STAGE_RECEIPTS.jsonl after this payload is committed.',
         'expected_final_total':22}
(ROOT/DOC/'STAGE_HISTORY_PRE_C10.json').write_text(json.dumps(history,indent=2)+'\n')
binary=ROOT/'target/release/quartz_c23_residual_gate'
binding=json.loads((ROOT/DOC/'evidence/C06_BINDING.json').read_text())
assert sha(binary)==binding['binary_sha256']
binary_backup=BACKUP/'binaries'/('quartz_c23_residual_gate_'+binding['binary_sha256'][:16])
binary_backup.parent.mkdir(exist_ok=True)
assert not binary_backup.exists()
shutil.copy2(binary,binary_backup)
assert sha(binary_backup)==binding['binary_sha256']
base=json.loads((ROOT/DOC/'BASELINE.json').read_text())
for path,h in base['protected'].items():assert sha(ROOT/path)==h
assert sha(ROOT/'src/bin/quartz_c23_residual_gate.rs')==binding['rust_source_sha256']
nxt=json.loads((ROOT/DOC/'NEXT_STAGE.json').read_text())
assert nxt['status']=='NOT_EXECUTED'
assert len(nxt['theory_loop'])==len(nxt['coding_loop'])==10
# Resolve local report targets before release. No external retrieval is part of this check.
report=ROOT/DOC/'RESULTS_AND_NEXT_KO.md'
local_links=[]
for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if '://' in target or target.startswith('#'):continue
    path=(report.parent/target.split('#')[0]).resolve()
    assert path.is_file(),str(path)
    local_links.append(str(path.relative_to(ROOT)))
paths=set(git('ls-files').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
files=[]
for name in sorted(paths):
    if (name.startswith(str(DOC)+'/') or name in {
        'src/bin/quartz_c23_residual_gate.rs','tests/test_c23_sources.py',
        'scripts/quartz_general_c23_publish.py'
    }) and name!=str(DOC/'RELEASE_MANIFEST.json'):
        p=ROOT/name
        if p.is_file():files.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={
    'schema':'quartz.research-release.v1','client_date':'2026-11-23 Asia/Seoul',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'research_id':base['research_id'],'parent':base['parent'],'branch':base['branch'],
    'pre_release_head':git('rev-parse','HEAD'),
    'scientific_status':'COMPLETED_WITH_NEGATIVE_PRIMARY_RUNTIME_RESULT_AND_EXPLICIT_DOMAIN_LIMIT',
    'publication_status_at_manifest_creation':'C01-C09 and R01-R10 receipts verified; C10 publication follows this payload.',
    'self_reference_policy':'The final commit/archive hash and actual post-release audit are external receipts, not self-hashes inside this manifest.',
    'frozen_rust_source_sha256':binding['rust_source_sha256'],
    'frozen_binary':{'sha256':binding['binary_sha256'],'backup':str(binary_backup),'bytes':binary_backup.stat().st_size},
    'test_execution_head':'c918fbad6255a7e3a1f0a3be3f1e70cbd409326c',
    'protected':base['protected'],'stage_count_before_release':len(receipts),'stage_count_after_success':22,
    'known_open_issue':'C09-F01: norm underflow outside the manufactured cohort; no universal accepted-f64 residual-goal guarantee.',
    'new_cycle':{'research_id':nxt['research_id'],'status':nxt['status']},
    'files':files,'local_report_links_verified':local_links,
    'restore_evidence':str(DOC/'evidence/C09_LOCAL_RESTORE.json'),
    'dropbox_cloud_ack':'NOT_VERIFIED'
}
(ROOT/DOC/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PAYLOAD_BOUND','files':len(files),'prior_stages':len(receipts),
                  'local_report_links':len(local_links),'binary_backup':manifest['frozen_binary'],
                  'manifest_sha256':sha(ROOT/DOC/'RELEASE_MANIFEST.json'),
                  'git_status':git('status','--short')},indent=2))
