"""Verify completed C23 publication and local backups; do not rerun science."""
from pathlib import Path
import datetime, hashlib, json, subprocess, tarfile
ROOT=Path('/home/cosmosapjw/quantum/research/quartz_general_meta23_20261123')
DOC=Path('docs/research/quartz_general_residual_stop_20261123')
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta23_backups')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
base=json.loads((ROOT/DOC/'BASELINE.json').read_text())
manifest=json.loads((ROOT/DOC/'RELEASE_MANIFEST.json').read_text())
head=git('rev-parse','HEAD')
assert git('branch','--show-current')==base['branch']
remote=git('ls-remote','--heads','origin',base['branch']).split()[0]
assert remote==head
assert git('status','--porcelain')==''
receipts=[json.loads(x) for x in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
expected=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,7)]+['C07A','C07','C08A','C08','C09','C10']
assert [x['stage'] for x in receipts]==expected
assert receipts[-1]['commit']==receipts[-1]['remote']==head
archives=[]
for row in receipts:
    p=Path(row['archive'])
    assert p.stat().st_size==row['archive_bytes']
    assert sha(p)==row['archive_sha256'],row['stage']
    archives.append({'stage':row['stage'],'commit':row['commit'],'bytes':p.stat().st_size,'sha256':row['archive_sha256']})
for path,h in base['protected'].items():assert sha(ROOT/path)==h,path
def worktrees(text):
    result={}
    for block in text.strip().split('\n\n'):
        data={}
        for line in block.splitlines():
            if ' ' in line:
                k,v=line.split(' ',1);data[k]=v
        if 'worktree' in data:result[data['worktree']]=data
    return result
old=worktrees(base['prior_worktrees'])
current=worktrees(git('worktree','list','--porcelain'))
unchanged=[]
for path,info in old.items():
    if path==str(ROOT):continue
    assert current[path]['HEAD']==info['HEAD'],path
    unchanged.append({'path':path,'head':info['HEAD']})
original=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz')
original_status=git('status','--porcelain',cwd=original)
assert original_status=='?? QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md',original_status
for path,h in base['protected'].items():assert sha(original/path)==h,path
for row in manifest['files']:
    p=ROOT/row['path']
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
assert sha(Path(manifest['frozen_binary']['backup']))==manifest['frozen_binary']['sha256']
assert sha(ROOT/'target/release/quartz_c23_residual_gate')==manifest['frozen_binary']['sha256']
archive=Path(receipts[-1]['archive'])
final_readback=[]
with tarfile.open(archive,'r:gz') as tar:
    for rel in ['RESULTS_AND_NEXT_KO.md','NEXT_STAGE.json','RELEASE_MANIFEST.json','CODING.md','STAGE_HISTORY_PRE_C10.json','evidence/C09_LOCAL_RESTORE.json']:
        name=str(DOC/rel);data=tar.extractfile(name).read()
        h=hashlib.sha256(data).hexdigest()
        assert h==sha(ROOT/name)
        final_readback.append({'path':name,'sha256':h})
result={
    'status':'VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'branch':base['branch'],'head':head,'remote_head':remote,'git_worktree_clean':True,
    'all_22_stage_archives_hash_verified':True,'stages':archives,
    'all_manifest_file_bindings_verified':len(manifest['files']),
    'original_worktrees_unchanged':unchanged,'original_worktree_status_preserved':original_status,
    'protected_files_unchanged':base['protected'],'final_archive_readback':final_readback,
    'frozen_binary_backup_verified':manifest['frozen_binary'],
    'final_receipt':receipts[-1],'next_cycle':'NOT_EXECUTED',
    'dropbox_cloud_ack':'NOT_VERIFIED',
    'scope':'Publication, artifact identity and archive checks only; no additional benchmark or scientific test rerun.'
}
(BACKUP/'POSTRELEASE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'stages','original_worktrees_unchanged','final_archive_readback'}},indent=2))
