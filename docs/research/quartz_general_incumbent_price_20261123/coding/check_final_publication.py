"""Final C25 publication audit; immutable self-reference receipt outside Git."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess,tarfile,time
ROOT=Path(__file__).resolve().parents[4];DOC=ROOT/'docs/research/quartz_general_incumbent_price_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta25_backups')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def git(*a,cwd=ROOT):return subprocess.check_output(['git',*a],cwd=cwd,text=True).strip()
start=time.perf_counter();base=json.loads((DOC/'BASELINE.json').read_text());rp=BACKUP/'STAGE_RECEIPTS.jsonl'
receipts=[json.loads(x) for x in rp.read_text().splitlines()]
main=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,11)]
assert [x['stage'] for x in receipts if re.fullmatch('[RC][0-9]{2}',x['stage'])]==main
assert len({x['stage'] for x in receipts})==len(receipts)
previous=base['parent'];archives=[]
for b in base['base_chain']:assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes']
for rec in receipts:
 assert rec['remote']==rec['commit'] and git('rev-parse',rec['commit']+'^')==previous
 previous=rec['commit']
 assert sha(rec['archive'])==rec['archive_sha256'] and Path(rec['archive']).stat().st_size==rec['archive_bytes']
 assert rec['base_chain']==base['base_chain']
 archives.append({k:rec[k] for k in ['stage','commit','archive_sha256','archive_bytes']})
head=git('rev-parse','HEAD');assert head==receipts[-1]['commit'];assert receipts[-1]['stage']=='C10'
assert git('branch','--show-current')==base['branch']
remote=git('ls-remote','--heads','origin',base['branch']).split()[0];assert remote==head
assert not git('status','--porcelain=v1','--untracked-files=all'),'uncommitted artifacts remain'
snapshot=json.loads((DOC/'STAGE_PUBLICATIONS.json').read_text());assert snapshot['receipts']==receipts[:-1]
mp=DOC/'ARTIFACTS.json';files=json.loads(mp.read_text())['files']
for rel,rec in files.items():assert sha(ROOT/rel)==rec['sha256'] and (ROOT/rel).stat().st_size==rec['bytes'],rel
mprel=str(mp.relative_to(ROOT));changed=set(git('diff','--name-only',base['parent'],head).splitlines())
assert changed==set(files)|{mprel},'artifact manifest completeness'
expected=dict(files);expected[mprel]={'sha256':sha(mp),'bytes':mp.stat().st_size};seen=set()
with tarfile.open(receipts[-1]['archive'],'r:gz') as tar:
 for member in tar:
  if not member.isfile():continue
  assert member.name in expected
  rec=expected[member.name];assert member.size==rec['bytes'];h=hashlib.sha256()
  with tar.extractfile(member) as s:
   for b in iter(lambda:s.read(1<<20),b''):h.update(b)
  assert h.hexdigest()==rec['sha256'],member.name;seen.add(member.name)
assert seen==set(expected)
freeze=json.loads((DOC/'evidence/C07_PRE_TEST_FREEZE.json').read_text())
for p,want in freeze['frozen_inputs'].items():assert sha(ROOT/p)==want,p
for p,want in base['protected'].items():assert sha(ROOT/p)==want,p
for old in base['old_worktrees']:
 p=Path(old['path']);assert git('rev-parse','HEAD',cwd=p)==old['head'];assert git('status','--porcelain=v1','--untracked-files=all',cwd=p)==old['status']
 for rel,want in old['own_protected'].items():assert sha(p/rel)==want
result={'status':'VERIFIED_FINAL_PUBLICATION','client_date':'2026-11-23 Asia/Seoul','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':base['branch'],'head':head,'remote':remote,'clean_worktree':True,'main_stages':main,'stage_count':len(receipts),'all_stage_archive_sha256_verified':True,'archives':archives,'base_chain':base['base_chain'],'final_archive':receipts[-1]['archive'],'final_archive_sha256':receipts[-1]['archive_sha256'],'final_archive_readback_files':len(seen),'artifact_manifest_sha256':sha(mp),'receipt_file_sha256':sha(rp),'frozen_scientific_inputs_unchanged':True,'old_worktrees_preserved':len(base['old_worktrees']),'restore_evidence_sha256':sha(DOC/'evidence/C09_RESTORE.json'),'review_sha256':sha(DOC/'review/C09_REVIEW.md'),'origin_watchdog_sha256':sha(DOC/'ORIGIN_WATCHDOG.md'),'dropbox_cloud_ack':'NOT_VERIFIED','C26_executed':False,'elapsed_seconds':time.perf_counter()-start,'audit_script_sha256':sha(Path(__file__))}
out=BACKUP/'FINAL_PUBLICATION_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','head','remote','stage_count','final_archive_readback_files','final_archive','final_archive_sha256','elapsed_seconds']}))
