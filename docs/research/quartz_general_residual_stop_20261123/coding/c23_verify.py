"""Independent exact source, bound and paired-policy validation. No source/solver imports."""
import argparse,gzip,json,pathlib,struct,hashlib
from fractions import Fraction as F
def rows(path):
 with gzip.open(path,'rt') as f:
  for line in f:yield json.loads(line)
def bits(v):return struct.unpack('<Q',struct.pack('<d',v))[0]
def verify_sources(public,oracles):
 count=0;components=0
 for ident,task in public.items():
  oracle=oracles[ident];p=oracle['dyadic_exponent'];n=task['side'];rho=task['rho'];scale=F(task['scale'])
  for x,b,q in zip(oracle['x_num'],task['rhs'],oracle['u_exact']):
   # Independent edge-energy form, not the generator's x dot stencil(x).
   energy=sum(z*z for z in x)
   for i,z in enumerate(x):
    degree=0
    if i%n+1<n:energy+=rho*(z-x[i+1])**2
    if i+n<n*n:energy+=rho*(z-x[i+n])**2
    if i%n:degree+=1
    if i%n+1<n:degree+=1
    if i>=n:degree+=1
    if i+n<n*n:degree+=1
    energy+=rho*(4-degree)*z*z
    adjacent=[j for j in [i-1,i+1,i-n,i+n] if 0<=j<n*n and abs(i//n-j//n)+abs(i%n-j%n)==1]
    numerator=(1+4*rho)*z-rho*sum(x[j] for j in adjacent)
    bn,bd=b[i].as_integer_ratio()
    assert bn*(1<<p)==numerator*bd,(ident,'stored input mismatch',i)
    components+=1
   assert F(energy,1<<(2*p))/scale==F(q),(ident,'energy oracle mismatch')
   assert sum(F(v)**2 for v in b)<=scale
  count+=1
 return {'parents':count,'exact_rhs_components':components}
def verify_native(public,oracles,path):
 intervals=0;certified=0;fallback=0;rows_count=0;transport=0
 for entry in rows(path):
  task=public[entry['source_id']];q=[F(x) for x in oracles[task['id']]['u_exact']]
  native=entry['native'];assert native['id']==task['id']
  if entry['mode']=='trace':
   assert native['input_rhs_bits']==[[bits(v) for v in b] for b in task['rhs']],(task['id'],'transport rhs bits')
   assert native['input_scale_bits']==bits(task['scale']),(task['id'],'transport scale bits')
   transport+=1
  by_rep={}
  for result in native['results']:
   rows_count+=1; by_rep.setdefault(result['rep'],{})[result['policy']]=result
   families=[result['initial_bounds'],result['final_bounds'],result['decision_bounds']]
   for checkpoint in result.get('trace',[]):families.extend([checkpoint['fresh_bounds'],checkpoint['bounds']])
   for bounds in families:
    assert len(bounds)==len(q)
    for u,interval in zip(q,bounds):
     assert F(interval['lo'])<=u<=F(interval['hi']),(task['id'],result['policy'],str(u),interval)
     intervals+=1
   if result['certified']:
    index=result['selected'];b=result['decision_bounds']
    assert F(b[index]['lo'])>max(F(v['hi']) for j,v in enumerate(b) if j!=index)
    assert q[index]==max(q)
    certified+=1
   assert result['total_us']>=0 and result['solver_matvecs']>=0
  for group in by_rep.values():
   assert set(group)=={'NEVER','ALWAYS','CERT_ONCE','CERT_BLOCK','RESIDUAL','PLAIN_ALWAYS'}
   assert group['PLAIN_ALWAYS']['selected']==group['ALWAYS']['selected']
   assert group['PLAIN_ALWAYS']['iterations']==group['ALWAYS']['iterations']
   base=group['ALWAYS']
   for name in ['CERT_ONCE','CERT_BLOCK']:
    result=group[name]
    assert q[result['selected']]>=q[base['selected']],(task['id'],'paired utility failure')
    if not result['certified']:
     assert result['selected']==base['selected'] and result['iterations']==base['iterations']
     if 'final_x' in result:assert result['final_x']==base['final_x']
     fallback+=1
   assert len({r['initial_selected'] for r in group.values()})==1
 return {'policy_executions_checked':rows_count,'interval_scalar_containments':intervals,'certified_optimal':certified,'identical_noncertified_fallbacks':fallback,'byte_exact_transport_requests':transport}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['TRAIN','CAL','TEST'],required=True);ap.add_argument('--mode',choices=['trace','timing'],required=True);args=ap.parse_args()
 doc=pathlib.Path(__file__).resolve().parents[1];ev=doc/'evidence'
 public={r['id']:r for r in rows(ev/f'C23_PUBLIC_{args.split}.jsonl.gz')}
 oracles={r['id']:r for r in rows(ev/f'C23_ORACLE_{args.split}.jsonl.gz')}
 assert public.keys()==oracles.keys()
 source=verify_sources(public,oracles)
 native=verify_native(public,oracles,ev/f'C23_{args.split}_{args.mode.upper()}.jsonl.gz')
 result={'status':'PASS','split':args.split,'mode':args.mode,'independent_of_production_imports':True,'source':source,'native':native,'scope':'finite exact dyadic sources and all returned interval endpoints; runtime performance separate'}
 out=ev/f'C23_VERIFY_{args.split}_{args.mode.upper()}.json'
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
