"""C24 C09 read-only independent raw-timing aggregate and provenance checks."""
from pathlib import Path
from fractions import Fraction as F
import collections,datetime,gzip,hashlib,json,math,platform,statistics,subprocess,sys,time
import numpy as np
ROOT=Path('/home/cosmosapjw/quantum/research/quartz_general_meta24_20261123')
DOC=ROOT/'docs/research/quartz_general_cheap_certificates_20261123'
EV=DOC/'evidence'
OUT=Path('/tmp/quartz_c24_review_checks.json')
assert not OUT.exists()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def j(p):return json.loads(p.read_text())
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
start=time.perf_counter();start_head=head()
cfg=j(DOC/'SOURCE_SPECS.json');freeze=j(EV/'C07_PRE_TEST_FREEZE.json')
inputs=[DOC/'SOURCE_SPECS.json',EV/'C07_PRE_TEST_FREEZE.json',EV/'C24_TEST_SUMMARY.json',EV/'C24_TEST_PARENT_METRICS.json',EV/'C24_TEST_TIMING.jsonl.gz',EV/'C24_ORACLE_TEST.jsonl.gz',EV/'C24_TEST_TRACE.manifest.json',EV/'C24_TEST_TIMING.manifest.json',EV/'C24_VERIFY_TEST_TRACE.json',EV/'C24_VERIFY_TEST_TIMING.json']+[ROOT/p for p in freeze['frozen_inputs']]
before={str(p):digest(p) for p in inputs}
for rel,want in freeze['frozen_inputs'].items():assert digest(ROOT/rel)==want,('frozen source mismatch',rel)
summary=j(EV/'C24_TEST_SUMMARY.json')
assert digest(EV/'C24_TEST_PARENT_METRICS.json')==summary['parent_metrics_sha256']
for mode in ['TRACE','TIMING']:
 manifest=j(EV/f'C24_TEST_{mode}.manifest.json')
 assert manifest['status']=='completed' and manifest['exit_code']==0
 assert manifest['input_hashes_before']==manifest['input_hashes_after']
 for rel,want in manifest['input_hashes_before'].items():
  path=ROOT/rel
  assert digest(path)==want,('run identity',mode,rel)
 assert manifest['binary_sha256']==freeze['frozen_inputs']['target/release/quartz_c24_cheap_certificate']
 if mode=='TIMING':assert digest(EV/'C24_TEST_TIMING.jsonl.gz')==manifest['output_sha256']
oracle={x['id']:list(map(F,x['u_exact'])) for x in gz(EV/'C24_ORACLE_TEST.jsonl.gz')}
expected=[x['id'] for x in cfg['sources'] if x['split']=='TEST']
assert set(oracle)==set(expected) and len(expected)==72
raw=gz(EV/'C24_TEST_TIMING.jsonl.gz');assert len(raw)==216
by_budget={b:[] for b in cfg['additional']}
raw_records=0
for env in raw:
 assert env['mode']=='timing' and env['split']=='TEST'
 ident=env['source_id'];budget=env['additional'];rr=env['native']['results']
 assert len(rr)==54
 assert {(x['policy'],x['rep']) for x in rr}=={(p,i) for p in cfg['policies'] for i in range(9)}
 out={}
 for name in cfg['policies']:
  vals=[x for x in rr if x['policy']==name]
  assert len({(x['selected'],x['certified'],tuple(x['iterations']),x['solver_matvecs']) for x in vals})==1
  v=vals[0];q=oracle[ident]
  assert not any('final_x' in x or 'trace' in x for x in vals)
  out[name]={'time':statistics.median(x['total_us'] for x in vals),'u':q[v['selected']],
     'regret':max(q)-q[v['selected']],'cert':v['certified'],'calls':v['solver_matvecs'],
     'ordinary_stencils':v['solver_matvecs']+v['ordinary_evaluations']+v['fast_evaluations']+v['outward_evaluations'],
     'interval_stencils':v['outward_evaluations']}
 raw_records+=len(rr)
 by_budget[budget].append((ident,out))
rng=np.random.default_rng(cfg['bootstrap_seed']);comparisons=[];primary=None
def mean(x):return math.fsum(x)/len(x)
def percentile(v,p):
 a=sorted(v);i=(len(a)-1)*p;lo=int(i);hi=math.ceil(i)
 return a[lo]+(a[hi]-a[lo])*(i-lo)
for budget in cfg['additional']:
 group=by_budget[budget];assert [x[0] for x in group]==expected
 idx=rng.integers(0,len(group),(2000,len(group)))
 s=next(z for z in summary['summaries'] if z['additional']==budget)
 p={}
 for name in cfg['policies']:
  values=[x[1][name] for x in group];base=[x[1]['PLAIN_ALWAYS'] for x in group]
  times=[b['time']-v['time'] for b,v in zip(base,values)]
  bs=[mean([times[int(i)] for i in row]) for row in idx]
  ci=[percentile(bs,.025),percentile(bs,.975)]
  regret=sum((v['regret'] for v in values),F(0))/72
  target=s['policies'][name]
  found={'mean_us':mean([v['time'] for v in values]),'saved_us':mean(times),'ci95_us':ci,
     'regret_exact':str(regret),'wrong':sum(v['regret']>0 for v in values),
     'certified':sum(v['cert'] for v in values),'early':sum(v['calls']<b['calls'] for v,b in zip(values,base)),
     'total_calls':sum(v['calls'] for v in values),
     'utility_difference_exact':str(sum((v['u']-b['u'] for v,b in zip(values,base)),F(0))/72)}
  assert found['wrong']==target['wrong_decision_parents']
  assert found['certified']==target['certified_parents']
  assert found['early']==target['actual_early_stop_parents']
  assert found['total_calls']==target['total_solver_matvecs']
  assert F(found['regret_exact'])==F(target['mean_regret_exact'])
  assert F(found['utility_difference_exact'])==F(target['mean_utility_difference_vs_PLAIN_ALWAYS_exact'])
  diffs=[abs(found['mean_us']-target['mean_parent_median_us']),abs(found['saved_us']-target['mean_time_saved_vs_PLAIN_ALWAYS_us'])]+[abs(a-b) for a,b in zip(ci,target['paired_parent_bootstrap_time_saved_vs_PLAIN_ALWAYS_95_us'])]
  assert max(diffs)<1e-9,('summary reduction mismatch',budget,name,diffs)
  comparisons.append(max(diffs));p[name]=found
  if name in ['FAST_ONCE','FAST_BLOCK','OUTWARD_BLOCK']:assert all(v['u']>=b['u'] for v,b in zip(values,base))
 if budget==32:
  fast=[x[1]['FAST_BLOCK'] for x in group];stop=[x[1]['PLAIN_STOP'] for x in group]
  du=[a['u']-b['u'] for a,b in zip(fast,stop)]
  dt=[a['time']-b['time'] for a,b in zip(fast,stop)]
  ne=mean([float(u)-cfg['primary_price']*t for u,t in zip(du,dt)])
  cutoff=float(sum(du,F(0))/72)/mean(dt)
  priced=next(x for x in s['policies']['FAST_BLOCK']['priced'] if x['lambda_utility_per_us']==cfg['primary_price'])
  assert abs(ne-priced['mean_net_difference_vs_PLAIN_STOP'])<1e-15
  assert abs(cutoff-s['policies']['FAST_BLOCK']['break_even_vs_PLAIN_STOP']['cohort_mean_break_even_price'])<1e-20
  primary={'policies':p,'fast_vs_stop_net':ne,'fast_vs_stop_break_even':cutoff}
after={str(p):digest(p) for p in inputs};assert before==after
result={'schema':'quartz.c24.independent-review-checks.v1','status':'PASS_FINITE_RAW_AGGREGATE_AND_PROVENANCE',
 'assertion_tested':'Independently reduce all TEST raw timing executions to parent medians, exact rational regrets/quality/work counts and paired percentile intervals; bind frozen code and run inputs. No production analyzer/verifier import.',
 'scope':{'parents':72,'budgets':3,'policies':6,'repetitions':9,'raw_policy_executions':raw_records,'summary_policy_budget_cells':18},
 'primary':primary,'maximum_aggregate_float_difference_us':max(comparisons),
 'identity_before':before,'identity_after':after,'start_head':start_head,'end_head':head(),
 'run':{'argv':sys.argv,'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},
 'non_claims':'No new native benchmark, no rerun of all scientific exact-vector checks, no universal theorem established by these finite statistics.',
 'review_script_sha256':digest(Path(__file__))}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'scope':result['scope'],'primary':primary,'max_float_difference_us':max(comparisons),'result_sha256':digest(OUT)},indent=2))
