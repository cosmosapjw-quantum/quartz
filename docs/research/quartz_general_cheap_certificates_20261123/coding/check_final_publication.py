"""Final C24 publication audit; writes its self-reference-free receipt outside Git."""
from pathlib import Path
import datetime,hashlib,json,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parents[4];DOC=ROOT/'docs/research/quartz_general_cheap_certificates_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta24_backups')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
started=time.perf_counter();base=json.loads((DOC/'BASELINE.json').read_text());receipt_path=BACKUP/'STAGE_RECEIPTS.jsonl'
receipts=[json.loads(x) for x in receipt_path.read_text().splitlines()]
expected=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,7)]+['C07A','C07','C08A','C08','C09','C10']
assert [x['stage'] for x in receipts]==expected
previous=base['parent'];archives=[]
assert sha(base['base_backup']['archive'])==base['base_backup']['sha256']
for item in receipts:
 assert item['remote']==item['commit'];assert git('rev-parse',item['commit']+'^')==previous
 previous=item['commit'];assert sha(item['archive'])==item['archive_sha256']
 assert Path(item['archive']).stat().st_size==item['archive_bytes'];assert item['base_backup']==base['base_backup']
 archives.append({'stage':item['stage'],'commit':item['commit'],'sha256':item['archive_sha256'],'bytes':item['archive_bytes']})
head=git('rev-parse','HEAD');assert head==receipts[-1]['commit'];assert git('branch','--show-current')==base['branch']
remote=git('ls-remote','--heads','origin',base['branch']).split()[0];assert remote==head
assert git('status','--porcelain=v1')=='','uncommitted/untracked result remains'
snapshot=json.loads((DOC/'STAGE_PUBLICATIONS.json').read_text());assert snapshot['receipts']==receipts[:-1]
manifest_path=DOC/'ARTIFACTS.json';manifest=json.loads(manifest_path.read_text());files=manifest['files']
for path,record in files.items():
 p=ROOT/path;assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],path
manifest_rel=str(manifest_path.relative_to(ROOT))
changed=set(git('diff','--name-only',base['parent'],head).splitlines());assert changed==set(files)|{manifest_rel}
expected_archive=dict(files);expected_archive[manifest_rel]={'sha256':sha(manifest_path),'bytes':manifest_path.stat().st_size};seen=set()
with tarfile.open(receipts[-1]['archive'],'r:gz') as tar:
 for member in tar:
  if not member.isfile():continue
  assert member.name in expected_archive,member.name
  record=expected_archive[member.name];assert member.size==record['bytes']
  h=hashlib.sha256()
  with tar.extractfile(member) as stream:
   for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
  assert h.hexdigest()==record['sha256'],member.name;seen.add(member.name)
assert seen==set(expected_archive)
freeze=json.loads((DOC/'evidence/C07_PRE_TEST_FREEZE.json').read_text())
for path,want in freeze['frozen_inputs'].items():assert sha(ROOT/path)==want,path
for path,want in base['protected'].items():assert sha(ROOT/path)==want,path
oldheads=[]
for block in base['prior_worktrees'].strip().split('\n\n'):
 lines=block.splitlines();path=Path(lines[0].split(' ',1)[1]);old=lines[1].split(' ',1)[1]
 if path==ROOT:continue
 now=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip();assert now==old;oldheads.append({'path':str(path),'head':now})
original=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz');status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=original,text=True).strip();assert status==base['original_worktree_status']
original_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip();original_protected={}
for path in ['docs/THESIS.md','docs/CLAIM_LEDGER.md','src/game.rs']:
 want=hashlib.sha256(subprocess.check_output(['git','show',original_head+':'+path],cwd=original)).hexdigest();assert sha(original/path)==want;original_protected[path]=want
result={'status':'VERIFIED_FINAL_PUBLICATION','client_date':'2026-11-23 Asia/Seoul','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':base['branch'],'head':head,'remote':remote,'clean_worktree':True,'stages':expected,'stage_count':len(receipts),'all_stage_archive_sha256_verified':True,'archives':archives,'base_backup':base['base_backup'],'final_archive':receipts[-1]['archive'],'final_archive_sha256':receipts[-1]['archive_sha256'],'final_archive_readback_files':len(seen),'artifact_manifest_sha256':sha(manifest_path),'receipt_file_sha256':sha(receipt_path),'frozen_scientific_hashes_unchanged':True,'prior_worktree_heads':oldheads,'original_status':status,'original_protected_from_own_head':original_protected,'restore_evidence_sha256':sha(DOC/'evidence/C09_RESTORE.json'),'independent_review_sha256':sha(DOC/'review/C09_REVIEW.md'),'dropbox_cloud_ack':'NOT_VERIFIED','C25_executed':False,'elapsed_seconds':time.perf_counter()-started,'audit_script_sha256':sha(Path(__file__))}
output=BACKUP/'FINAL_PUBLICATION_AUDIT.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'head':head,'remote':remote,'stage_count':len(receipts),'final_archive_readback_files':len(seen),'final_archive':result['final_archive'],'final_archive_sha256':result['final_archive_sha256'],'audit_path':str(output),'audit_sha256':sha(output),'elapsed_seconds':result['elapsed_seconds']}))
