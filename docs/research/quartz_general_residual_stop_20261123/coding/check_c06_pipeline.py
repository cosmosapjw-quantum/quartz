"""Verify native input identity and full pipeline on non-cohort UNIT fixtures."""
import importlib.util,pathlib,subprocess,json,struct,gzip,tempfile,hashlib
doc=pathlib.Path(__file__).resolve().parents[1]
root=doc.parents[2]
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
source=module('source',doc/'coding/c23_sources.py')
verify=module('verify',doc/'coding/c23_verify.py')
public={};truth={};envelopes=[];probes=[]
for side,rho,seed in [(2,1,23455),(32,16,23456),(96,64,23457)]:
 task,oracle=source.make_source({'id':f'transport_unit_{side}','side':side,'rho':rho,'seed':seed,'target_gap':'medium','stratum':-1,'split':'UNIT'})
 public[task['id']]=task;truth[task['id']]=oracle
 wire={**task,'scale':repr(task['scale']),'rhs':[[repr(v) for v in row] for row in task['rhs']]}
 req={'task':wire,'additional':8,'repeats':1,'warmup':0,'trace':True}
 r=subprocess.run([str(root/'target/release/quartz_c23_residual_gate')],input=json.dumps(req)+'\n',capture_output=True,text=True,check=True)
 out=json.loads(r.stdout);assert 'error' not in out,out
 assert len(out['results'])==6
 mismatch=0
 for a,b in enumerate(task['rhs']):
  for i,v in enumerate(b):
   expected=struct.unpack('<Q',struct.pack('<d',v))[0]
   mismatch+=expected!=out['input_rhs_bits'][a][i]
 assert mismatch==0
 assert out['input_scale_bits']==struct.unpack('<Q',struct.pack('<d',task['scale']))[0]
 probes.append({'side':side,'rho':rho,'components':sum(map(len,task['rhs'])),'mismatches':mismatch,'policies':len(out['results'])})
 envelopes.append({'source_id':task['id'],'split':'UNIT','mode':'trace','additional':8,'native':out})
check_sources=verify.verify_sources(public,truth)
with tempfile.TemporaryDirectory(prefix='quartz-c23-unit-') as tmp:
 path=pathlib.Path(tmp)/'unit.jsonl.gz'
 with gzip.open(path,'wt') as f:
  for e in envelopes:f.write(json.dumps(e)+'\n')
 check_native=verify.verify_native(public,truth,path)
report={'status':'PASS','wire_encoding':'shortest decimal strings with Rust FromStr','probes':probes,'independent_sources':check_sources,'independent_native':check_native,'binary_sha256':hashlib.sha256((root/'target/release/quartz_c23_residual_gate').read_bytes()).hexdigest()}
(doc/'evidence/C06_TRANSPORT_AND_PIPELINE_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
