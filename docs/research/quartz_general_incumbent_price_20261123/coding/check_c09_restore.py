"""Restore C23 base + C24 overlay + C25 C08 overlay and verify tracked bytes."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,shutil,subprocess,tarfile,tempfile,time
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/research/quartz_general_incumbent_price_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta25_backups')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def git(*a,cwd=ROOT):return subprocess.check_output(['git',*a],cwd=cwd,text=True).strip()
start=time.perf_counter();base=json.loads((DOC/'BASELINE.json').read_text())
receipts=[json.loads(x) for x in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
last=next(x for x in receipts if x['stage']=='C08')
archives=base['base_chain']+[{'path':last['archive'],'sha256':last['archive_sha256'],'bytes':last['archive_bytes'],'kind':'cumulative C25 overlay'}]
for a in archives:
 assert sha(a['path'])==a['sha256'] and Path(a['path']).stat().st_size==a['bytes']
restore=Path(tempfile.mkdtemp(prefix='quartz_c25_restore_'))
for a in archives:
 with tarfile.open(a['path'],'r:gz') as tar:tar.extractall(restore,filter='data')
fmt=git('rev-parse','--show-object-format');count=total=links=0
for row in subprocess.check_output(['git','ls-tree','-rz',last['commit']],cwd=ROOT).split(b'\0'):
 if not row:continue
 meta,name=row.split(b'\t',1);mode,kind,oid=meta.decode().split();assert kind=='blob'
 p=restore/os.fsdecode(name)
 if mode=='120000':
  assert p.is_symlink();v=os.readlink(p).encode();size=len(v)
  h=hashlib.new(fmt,b'blob '+str(size).encode()+b'\0');h.update(v);links+=1
 else:
  assert p.is_file() and not p.is_symlink(),str(p)
  size=p.stat().st_size;h=hashlib.new(fmt,b'blob '+str(size).encode()+b'\0')
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==oid,str(p)
 count+=1;total+=size
binary=ROOT/'target/release/quartz_c25_incumbent_price';binary_sha=sha(binary)
bb=BACKUP/f'C25_NATIVE_RELEASE_{binary_sha[:16]}.gz'
assert not bb.exists()
with bb.open('xb') as out:
 with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as dest:
  with binary.open('rb') as src:shutil.copyfileobj(src,dest)
h=hashlib.sha256()
with gzip.open(bb,'rb') as src:
 for b in iter(lambda:src.read(1<<20),b''):h.update(b)
assert h.hexdigest()==binary_sha
olds=[]
for old in base['old_worktrees']:
 p=Path(old['path'])
 assert git('rev-parse','HEAD',cwd=p)==old['head']
 assert git('status','--porcelain=v1','--untracked-files=all',cwd=p)==old['status']
 for rel,want in old['own_protected'].items():assert sha(p/rel)==want,(str(p),rel)
 olds.append({'path':str(p),'head':old['head'],'status_unchanged':True,'own_protected_unchanged':True})
for rel,want in base['protected'].items():assert sha(ROOT/rel)==want,rel
result={'status':'VERIFIED_THREE_ARCHIVE_RESTORE','restored_commit':last['commit'],'restore_directory':str(restore),'archives':archives,'tracked_files_verified':count,'tracked_bytes_verified':total,'symlinks_verified':links,'binary_backup':{'path':str(bb),'compressed_sha256':sha(bb),'uncompressed_sha256':binary_sha,'bytes':bb.stat().st_size},'old_worktrees':olds,'current_protected_unchanged':True,'elapsed_seconds':time.perf_counter()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dropbox_cloud_ack':'NOT_VERIFIED','scope':'restored Git byte identity and executable preservation; no duplicate solver experiment'}
out=DOC/'evidence/C09_RESTORE.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','restored_commit','restore_directory','tracked_files_verified','tracked_bytes_verified','elapsed_seconds']}))
