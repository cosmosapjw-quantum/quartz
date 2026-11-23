"""Restore the immutable C08 base+overlay and verify every Git-tracked byte."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,shutil,subprocess,tarfile,tempfile,time
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/research/quartz_general_cheap_certificates_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta24_backups')
PY=ROOT/'venv/bin/python'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
started=time.perf_counter();base=json.loads((DOC/'BASELINE.json').read_text())
receipts=[json.loads(line) for line in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
receipt=next(x for x in receipts if x['stage']=='C08');head=receipt['commit']
archives=[{'archive':base['base_backup']['archive'],'sha256':base['base_backup']['sha256']},{'archive':receipt['archive'],'sha256':receipt['archive_sha256']}]
for rec in archives:assert sha(rec['archive'])==rec['sha256']
restore=Path(tempfile.mkdtemp(prefix='quartz_c24_restore_'))
for rec in archives:
 with tarfile.open(rec['archive'],'r:gz') as tar:tar.extractall(restore,filter='data')
fmt=git('rev-parse','--show-object-format');assert fmt in ('sha1','sha256')
listing=subprocess.check_output(['git','ls-tree','-rz',head],cwd=ROOT)
count=bytecount=symlinks=0
for row in listing.split(b'\0'):
 if not row:continue
 meta,name=row.split(b'\t',1);mode,kind,oid=meta.decode().split();assert kind=='blob'
 p=restore/os.fsdecode(name)
 if mode=='120000':
  assert p.is_symlink();payload=os.readlink(p).encode();size=len(payload);symlinks+=1
  h=hashlib.new(fmt,b'blob '+str(size).encode()+b'\0');h.update(payload)
 else:
  assert p.is_file() and not p.is_symlink(),str(p);size=p.stat().st_size
  h=hashlib.new(fmt,b'blob '+str(size).encode()+b'\0')
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 assert h.hexdigest()==oid,(str(p),'restored Git blob identity mismatch')
 count+=1;bytecount+=size
restored_doc=restore/DOC.relative_to(ROOT);commands=[];traces={}
for split in ['TRAIN','CAL','TEST']:
 cmd=[str(PY),str(restored_doc/'coding/reassemble_c24_trace.py'),'assemble','--split',split]
 proc=subprocess.run(cmd,cwd=restore,text=True,capture_output=True);commands.append({'argv':cmd,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr});proc.check_returncode()
 p=restored_doc/f'evidence/C24_{split}_TRACE.jsonl.gz';native=json.loads((restored_doc/f'evidence/C24_{split}_TRACE.manifest.json').read_text())
 assert sha(p)==native['output_sha256']
 traces[split]={'bytes':p.stat().st_size,'sha256':sha(p),'manifest_sha256':sha(restored_doc/f'evidence/C24_{split}_TRACE.manifest.json')}
# Preserve the exact tested executable as a separate optional resume artifact.
binary=ROOT/'target/release/quartz_c24_cheap_certificate';binary_sha=sha(binary);binary_backup=BACKUP/f'C24_NATIVE_RELEASE_{binary_sha[:16]}.gz'
if not binary_backup.exists():
 with binary_backup.open('xb') as out:
  with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as dest:
   with binary.open('rb') as src:shutil.copyfileobj(src,dest)
h=hashlib.sha256()
with gzip.open(binary_backup,'rb') as src:
 for block in iter(lambda:src.read(1<<20),b''):h.update(block)
assert h.hexdigest()==binary_sha
oldheads=[]
for block in base['prior_worktrees'].strip().split('\n\n'):
 lines=block.splitlines();path=Path(lines[0].split(' ',1)[1]);old=lines[1].split(' ',1)[1]
 if path==ROOT:continue
 actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip();assert actual==old,(str(path),old,actual)
 oldheads.append({'path':str(path),'baseline_head':old,'actual_head':actual})
original=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz')
original_status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=original,text=True).strip();assert original_status==base['original_worktree_status']
original_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()
original_files={}
for rel in ['docs/THESIS.md','docs/CLAIM_LEDGER.md','src/game.rs']:
 expected=hashlib.sha256(subprocess.check_output(['git','show',original_head+':'+rel],cwd=original)).hexdigest();actual=sha(original/rel);assert expected==actual
 original_files[rel]={'expected_from_own_head':expected,'actual':actual}
for rel,want in base['protected'].items():assert sha(ROOT/rel)==want
result={'status':'VERIFIED_BASE_PLUS_OVERLAY_RESTORE','assertion_tested':'Archive hashes, safe base-then-overlay extraction, every tracked file Git blob identity, lossless reconstruction of all C24 raw traces, original worktree HEAD/status and protected own-baseline files, exact tested binary backup.', 'restored_commit':head,'restore_directory':str(restore),'archives':archives,'tracked_files_verified':count,'tracked_bytes_verified':bytecount,'symlinks_verified':symlinks,'traces':traces,'commands':commands,'binary_backup':{'path':str(binary_backup),'compressed_sha256':sha(binary_backup),'uncompressed_sha256':binary_sha,'bytes':binary_backup.stat().st_size},'prior_worktree_heads':oldheads,'original_status':original_status,'original_files':original_files,'current_protected':base['protected'],'elapsed_seconds':time.perf_counter()-started,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dropbox_cloud_ack':'NOT_VERIFIED','scope':'Restore/byte identity only; no duplicate solver campaign or fresh mathematical assertion.'}
output=DOC/'evidence/C09_RESTORE.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'restored_commit':head,'restore_directory':str(restore),'tracked_files_verified':count,'tracked_bytes_verified':bytecount,'traces':traces,'old_worktree_heads_checked':len(oldheads),'elapsed_seconds':result['elapsed_seconds']}))
