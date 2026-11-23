from pathlib import Path
import subprocess,json,sys,os,datetime,time,hashlib
r=Path('/home/cosmosapjw/quantum/research/quartz_general_meta25_20261123');d=r/'docs/research/quartz_general_incumbent_price_20261123';c=d/'coding';e=d/'evidence'
split=sys.argv[1];assert split in ['TRAIN','CAL','TEST']
out=e/f'RUN_{split}_COMMANDS.json';assert not out.exists()
steps=[('c25_sources.py',['--split',split]),('c25_run.py',['--split',split,'--mode','trace']),('c25_run.py',['--split',split,'--mode','timing']),('c25_verify.py',['--split',split,'--mode','trace']),('c25_verify.py',['--split',split,'--mode','timing'])]
if split=='TRAIN':steps.append(('c25_fit_predictor.py',[]))
steps.append(('c25_analyze.py',['--split',split]))
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
rec={'status':'running','split':split,'start_utc':utc(),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'commands':[]}
out.write_text(json.dumps(rec,indent=2)+'\n')
for i,(script,args) in enumerate(steps):
 log=e/f'RUN_{split}_{i:02d}.log';assert not log.exists()
 cmd=[str(r/'venv/bin/python'),str(c/script),*args];start=time.perf_counter()
 print(json.dumps({'start_command':i,'script':script,'args':args}),flush=True)
 with log.open('x') as f:p=subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 item={'argv':cmd,'exit_code':p.returncode,'elapsed_seconds':time.perf_counter()-start,'log':str(log.relative_to(r)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'end_utc':utc()}
 rec['commands'].append(item);out.write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(item),flush=True)
 if p.returncode:
  rec.update(status='failed',end_utc=utc());out.write_text(json.dumps(rec,indent=2)+'\n');sys.exit(p.returncode)
rec.update(status='completed',end_utc=utc());out.write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({'status':rec['status'],'split':split,'commands':len(steps)}),flush=True)
