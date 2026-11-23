#!/usr/bin/env python3
"""Publish one C24 stage with a cumulative overlay on the verified C23 base."""
from pathlib import Path
import datetime,gzip,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/research/quartz_general_cheap_certificates_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta24_backups')
stage=sys.argv[1];paths=sys.argv[2:]
assert paths and stage.replace('_','').isalnum()
base=json.loads((DOC/'BASELINE.json').read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert git('branch','--show-current')==base['branch']
for path,sha in base['protected'].items():assert digest(ROOT/path)==sha,path
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
# Raw stdout/stderr are byte-preserved evidence; trailing terminal newlines are not source defects.
subprocess.run(['git','diff','--cached','--check','--','.',':(exclude,glob)**/*.log',':(exclude,glob)**/*.stdout',':(exclude,glob)**/*.stderr'],cwd=ROOT,check=True)
assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode==1
subprocess.run(['git','commit','-m',f'research(c24 {stage}): rigorous cheap decision certificates'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
head=git('rev-parse','HEAD')
assert not git('diff','--name-only','--diff-filter=D',base['parent'],head)
changed=git('diff','--name-only',base['parent'],head).splitlines()
assert changed
for path in changed:
 assert path.startswith('docs/research/quartz_general_cheap_certificates_20261123/') or path in {
  'src/bin/quartz_c24_cheap_certificate.rs','tests/test_c24_sources.py',
  'scripts/quartz_general_c24_publish.py'},path
archive=BACKUP/f'QUARTZ_GENERAL_C24_{stage}_{head[:12]}.tar.gz'
assert not archive.exists()
with archive.open('wb') as out:
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:
  p=subprocess.Popen(['git','archive','--format=tar',head,'--',*changed],cwd=ROOT,stdout=subprocess.PIPE)
  for block in iter(lambda:p.stdout.read(1048576),b''):z.write(block)
  assert p.wait()==0
sha=digest(archive)
subprocess.run(['git','push','--quiet','-u','origin',f"HEAD:refs/heads/{base['branch']}"],cwd=ROOT,check=True)
remote=git('ls-remote','--heads','origin',base['branch']).split()[0]
assert remote==head
receipt={'stage':stage,'commit':head,'remote':remote,'archive':str(archive),'archive_sha256':sha,
 'archive_bytes':archive.stat().st_size,'archive_kind':'cumulative C24 overlay',
 'base_backup':base['base_backup'],'changed_files':len(changed),
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dropbox_cloud_ack':'NOT_VERIFIED'}
with (BACKUP/'STAGE_RECEIPTS.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
print(json.dumps(receipt))
