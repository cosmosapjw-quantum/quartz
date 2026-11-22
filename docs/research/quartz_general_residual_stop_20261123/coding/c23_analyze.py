"""Prespecified parent-level summaries of C23 held-out elapsed-time economics."""
import argparse,gzip,json,pathlib,statistics,hashlib
from fractions import Fraction as F
import numpy as np
POLICIES=['NEVER','ALWAYS','PLAIN_ALWAYS','CERT_ONCE','CERT_BLOCK','RESIDUAL']
def read(path):
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['TRAIN','CAL','TEST'],default='TEST');args=ap.parse_args()
 doc=pathlib.Path(__file__).resolve().parents[1];ev=doc/'evidence';cfg=json.loads((doc/'SOURCE_SPECS.json').read_text())
 truth={x['id']:x for x in read(ev/f'C23_ORACLE_{args.split}.jsonl.gz')}
 envelopes=read(ev/f'C23_{args.split}_TIMING.jsonl.gz')
 parents=[]
 for env in envelopes:
  oracle=truth[env['source_id']];q=[F(x) for x in oracle['u_exact']]
  grouped={p:[] for p in POLICIES}
  for r in env['native']['results']:grouped[r['policy']].append(r)
  record={'id':env['source_id'],'additional':env['additional'],'side':oracle['spec']['side'],'rho':oracle['spec']['rho'],'gap':oracle['spec']['target_gap'],'stratum':oracle['spec']['stratum'],'policies':{}}
  for policy,rr in grouped.items():
   assert len(rr)==cfg['repeats']
   assert len({(r['selected'],r['certified'],tuple(r['iterations'])) for r in rr})==1
   first=rr[0];u=q[first['selected']];initial=q[first['initial_selected']]
   record['policies'][policy]={'selected':first['selected'],'certified':first['certified'],'utility':float(u),'utility_exact':str(u),'regret':float(max(q)-u),'gain':float(u-initial),'median_total_us':statistics.median(r['total_us'] for r in rr),'median_bounds_us':statistics.median(r['bounds_us'] for r in rr),'median_solve_us':statistics.median(r['solve_us'] for r in rr),'median_initial_us':statistics.median(r['initial_us'] for r in rr),'iterations':sum(first['iterations']),'solver_matvecs':first['solver_matvecs'],'bound_evaluations':first['bound_evaluations'],'ordinary_evaluations':first.get('ordinary_evaluations',0),'total_operator_sweeps':first['solver_matvecs']+first['bound_evaluations']+first.get('ordinary_evaluations',0),'residual_goal_met':first['residual_goal_met'],'all_times_us':[r['total_us'] for r in rr]}
  parents.append(record)
 rng=np.random.default_rng(cfg['bootstrap_seed'])
 summaries=[]
 for budget in cfg['additional']:
  group=[r for r in parents if r['additional']==budget];n=len(group)
  indices=rng.integers(0,n,(cfg['bootstrap_resamples'],n))
  def mean(values):return float(np.mean(values))
  def ci(values):return [float(x) for x in np.percentile(np.asarray(values)[indices].mean(axis=1),[2.5,97.5])]
  policies={}
  for policy in POLICIES:
   vals=[p['policies'][policy] for p in group]
   diff=[p['policies']['ALWAYS']['median_total_us']-p['policies'][policy]['median_total_us'] for p in group]
   delta_u=[p['policies'][policy]['utility']-p['policies']['ALWAYS']['utility'] for p in group]
   policies[policy]={'parents':n,'mean_parent_median_us':mean([v['median_total_us'] for v in vals]),'mean_regret':mean([v['regret'] for v in vals]),'wrong_decision_parents':sum(v['regret']>0 for v in vals),'certified_parents':sum(v['certified'] for v in vals),'actual_early_stop_parents':sum(p['policies'][policy]['iterations']<p['policies']['ALWAYS']['iterations'] for p in group),'mean_solver_iterations':mean([v['iterations'] for v in vals]),'solver_iterations_saved':sum(p['policies']['ALWAYS']['iterations']-p['policies'][policy]['iterations'] for p in group),'mean_time_saved_vs_ALWAYS_us':mean(diff),'paired_parent_bootstrap_time_saved_95_us':ci(diff),'mean_utility_difference_vs_ALWAYS':mean(delta_u),'mean_bounds_us':mean([v['median_bounds_us'] for v in vals]),'priced':[]}
   plain_diff=[p['policies']['PLAIN_ALWAYS']['median_total_us']-p['policies'][policy]['median_total_us'] for p in group]
   policies[policy]['mean_time_saved_vs_PLAIN_ALWAYS_us']=mean(plain_diff)
   policies[policy]['paired_parent_bootstrap_time_saved_vs_PLAIN_ALWAYS_95_us']=ci(plain_diff)
   for price in cfg['prices']:
    deltas=[du+price*dt for du,dt in zip(delta_u,diff)]
    never_deltas=[p['policies'][policy]['utility']-p['policies']['NEVER']['utility']-price*(p['policies'][policy]['median_total_us']-p['policies']['NEVER']['median_total_us']) for p in group]
    policies[policy]['priced'].append({'lambda_utility_per_us':price,'mean_net_gain_from_initial':mean([v['gain']-price*v['median_total_us'] for v in vals]),'mean_net_difference_vs_ALWAYS':mean(deltas),'paired_parent_bootstrap_net_difference_95':ci(deltas),'mean_net_difference_vs_measured_NEVER':mean(never_deltas),'mean_net_difference_vs_PLAIN_ALWAYS':mean([du+price*dt for du,dt in zip(delta_u,plain_diff)]),'paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95':ci([du+price*dt for du,dt in zip(delta_u,plain_diff)])})
  subgroups=[]
  for field,levels in [('side',[8,32,96]),('rho',[1,16,64]),('gap',['wide','medium','tight','tie'])]:
   for level in levels:
    sub=[p for p in group if p[field]==level]
    subgroups.append({'field':field,'value':level,'parents':len(sub),'policies':{name:{'mean_parent_median_us':mean([p['policies'][name]['median_total_us'] for p in sub]),'mean_regret':mean([p['policies'][name]['regret'] for p in sub]),'certified':sum(p['policies'][name]['certified'] for p in sub),'iterations_saved':sum(p['policies']['ALWAYS']['iterations']-p['policies'][name]['iterations'] for p in sub)} for name in POLICIES}})
  summaries.append({'additional':budget,'policies':policies,'subgroups':subgroups})
 result={'split':args.split,'primary_additional':cfg['primary_additional'],'unique_parents':len(truth),'bootstrap_resamples':cfg['bootstrap_resamples'],'bootstrap_seed':cfg['bootstrap_seed'],'timing_statistic':'median of nine physical runs per parent, then arithmetic parent mean','clock':'wall elapsed microseconds','summaries':summaries}
 (ev/f'C23_{args.split}_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
 (ev/f'C23_{args.split}_PARENT_METRICS.json').write_text(json.dumps(parents,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
