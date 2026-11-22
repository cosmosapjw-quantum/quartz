#!/usr/bin/env python3
"""Publish one authorized C23 stage; preserve distinct scientific and storage status."""
import pathlib,subprocess,hashlib,json,gzip,sys,datetime
root=pathlib.Path(__file__).resolve().parents[1]
doc=root/'docs/research/quartz_general_residual_stop_20261123'
stage=sys.argv[1]; paths=sys.argv[2:]
assert paths and stage.replace('_','').isalnum()
base=json.loads((doc/'BASELINE.json').read_text())
def run(args): return subprocess.check_output(args,cwd=root,text=True).strip()
assert run(['git','branch','--show-current'])==base['branch']
for f,h in base['protected'].items(): assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h,f
subprocess.run(['git','add','--',*paths],cwd=root,check=True)
subprocess.run(['git','diff','--cached','--check','--','.',' :(exclude,glob)**/*.log'.strip()],cwd=root,check=True)
assert subprocess.run(['git','diff','--cached','--quiet'],cwd=root).returncode==1,'no stage delta'
subprocess.run(['git','commit','-m',f'research(c23 {stage}): residual-bounded computation choice'],cwd=root,check=True,stdout=subprocess.DEVNULL)
sha=run(['git','rev-parse','HEAD'])
backup=pathlib.Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta23_backups')
backup.mkdir(exist_ok=True,parents=True)
archive=backup/f'QUARTZ_GENERAL_C23_{stage}_{sha[:12]}.tar.gz'
with archive.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
  p=subprocess.Popen(['git','archive','--format=tar',sha],cwd=root,stdout=subprocess.PIPE)
  while True:
   buf=p.stdout.read(1024*1024)
   if not buf: break
   z.write(buf)
  assert p.wait()==0
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
subprocess.run(['git','push','--quiet','-u','origin',f'HEAD:refs/heads/{base["branch"]}'],cwd=root,check=True)
remote=run(['git','ls-remote','--heads','origin',base['branch']]).split()[0]
assert remote==sha
receipt={'stage':stage,'commit':sha,'remote':remote,'archive':str(archive),'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dropbox_cloud_ack':'NOT_VERIFIED'}
with (backup/'STAGE_RECEIPTS.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
print(json.dumps(receipt))
