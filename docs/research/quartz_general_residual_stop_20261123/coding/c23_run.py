"""Run frozen C23 tasks through the actual native policy paths."""
import argparse,gzip,hashlib,json,pathlib,subprocess,sys,datetime
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['TRAIN','CAL','TEST'],required=True);ap.add_argument('--mode',choices=['trace','timing'],required=True);args=ap.parse_args()
 doc=pathlib.Path(__file__).resolve().parents[1];root=doc.parents[2]
 binary=root/'target/release/quartz_c23_residual_gate'
 config=json.loads((doc/'SOURCE_SPECS.json').read_text())
 public=doc/'evidence'/f'C23_PUBLIC_{args.split}.jsonl.gz'
 output=doc/'evidence'/f'C23_{args.split}_{args.mode.upper()}.jsonl.gz'
 manifest=output.with_suffix('').with_suffix('.manifest.json')
 errlog=doc/'evidence'/f'C23_{args.split}_{args.mode.upper()}_STDERR.log'
 if output.exists() or manifest.exists():raise FileExistsError('preserved run already exists')
 subprocess.run(['git','diff','--exit-code','HEAD','--','src/bin/quartz_c23_residual_gate.rs',str(pathlib.Path(__file__).relative_to(root)),str((doc/'coding/c23_sources.py').relative_to(root))],cwd=root,check=True,stdout=subprocess.DEVNULL)
 code_sha=sha(binary);source_sha=sha(public);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 count=0;result_count=0;start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with errlog.open('w') as err,output.open('wb') as raw:
  proc=subprocess.Popen([str(binary)],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
  with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as dest,gzip.open(public,'rt') as inputs:
   for line in inputs:
    task=json.loads(line)
    for budget in config['additional']:
     wire_task={**task,'scale':repr(task['scale']),'rhs':[[repr(v) for v in row] for row in task['rhs']]}
     req={'task':wire_task,'additional':budget,'repeats':config['repeats'] if args.mode=='timing' else 1,'warmup':config['warmup'] if args.mode=='timing' else 0,'trace':args.mode=='trace'}
     proc.stdin.write(json.dumps(req,separators=(',',':'),allow_nan=False)+'\n');proc.stdin.flush()
     answer=proc.stdout.readline()
     if not answer:raise RuntimeError('native process ended without response; inspect stderr')
     out=json.loads(answer)
     if 'error' in out:raise RuntimeError(out['error'])
     assert out['id']==task['id'] and out['additional']==budget
     assert len(out['results'])==6*req['repeats']
     envelope={'source_id':task['id'],'split':args.split,'mode':args.mode,'additional':budget,'task_sha256':hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'native':out}
     dest.write((json.dumps(envelope,separators=(',',':'),allow_nan=False)+'\n').encode())
     count+=1;result_count+=len(out['results'])
    print(json.dumps({'source_completed':task['id'],'requests':count}),flush=True)
  proc.stdin.close();exit_code=proc.wait()
 assert exit_code==0
 assert sha(binary)==code_sha and sha(public)==source_sha
 result={'split':args.split,'mode':args.mode,'requests':count,'native_policy_executions':result_count,'exit_code':exit_code,'source_head':head,'binary_sha256':code_sha,'public_source_sha256':source_sha,'output':output.name,'output_sha256':sha(output),'output_bytes':output.stat().st_size,'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'clock':'native Instant monotonic wall elapsed; JSON and source/oracle generation excluded','repeats_are_independent_parents':False,'wire_encoding':'f64 shortest decimal strings parsed with Rust FromStr; trace asserts exact stored input bits'}
 manifest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
