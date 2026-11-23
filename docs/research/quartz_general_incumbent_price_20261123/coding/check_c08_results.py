"""Parent consistency/identity audit; no additional native experiment or fitting."""
from pathlib import Path
import json,sys
from fractions import Fraction as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from c25_io import rows,semantic_signature,sha
r=Path(__file__).resolve().parents[4];d=Path(__file__).resolve().parents[1];e=d/'evidence'
freeze=json.loads((e/'C07_PRE_TEST_FREEZE.json').read_text())
for p,h in freeze['frozen_inputs'].items():assert sha(r/p)==h,p
signatures={}
for split in ['TRAIN','CAL','TEST']:
 trace={}
 for env in rows(e/f'C25_{split}_TRACE.manifest.json'):
  for row in env['native']['results']:trace[(env['source_id'],env['additional'],row['policy'])]=semantic_signature(row)
 count=0
 for env in rows(e/f'C25_{split}_TIMING.manifest.json'):
  for row in env['native']['results']:
   assert trace[(env['source_id'],env['additional'],row['policy'])]==semantic_signature(row)
   count+=1
 assert count==len(trace)*9
 signatures[split]={'trace_policy_states':len(trace),'timing_scientific_signatures_matched':count}
s=json.loads((e/'C25_TEST_SUMMARY.json').read_text());p=json.loads((e/'C25_TEST_PARENT_METRICS.json').read_text())
primary=next(x for x in s['summaries'] if x['additional']==32);q=primary['policies']
cases=[]
for row in p:
 for name in ['RETAIN_BLOCK','RETAIN_WORK','RETAIN_TIME']:
  v=row['policies'][name];assert F(v['gain_exact'])>=0
  if F(v['missed_improvement_vs_PLAIN_ALWAYS_exact'])>0 or F(v['protected_utility_vs_own_final_ordinary_exact'])>0:
   cases.append({'id':row['id'],'additional':row['additional'],'side':row['side'],'rho':row['rho'],'gap':row['gap'],'policy':name,'initial':v['initial_selected'],'selected':v['selected'],'final_ordinary':v['final_ordinary_selected'],'missed_vs_ALWAYS_exact':v['missed_improvement_vs_PLAIN_ALWAYS_exact'],'retained_minus_own_ordinary_positive_exact':v['protected_utility_vs_own_final_ordinary_exact']})
result={'status':'PASS','frozen_inputs_unchanged':True,'trace_timing_signatures':signatures,
 'primary_contrast':s['contrasts'][0],
 'TIME_time_saved_percent_vs_ALWAYS':100*(1-q['RETAIN_TIME']['mean_parent_median_us']/q['PLAIN_ALWAYS']['mean_parent_median_us']),
 'TIME_time_saved_percent_vs_FAST':100*(1-q['RETAIN_TIME']['mean_parent_median_us']/q['FAST_BLOCK']['mean_parent_median_us']),
 'TIME_solver_saved_percent_vs_ALWAYS':100*(1-q['RETAIN_TIME']['total_solver_matvecs']/q['PLAIN_ALWAYS']['total_solver_matvecs']),
 'TEST_distinct_retention_source_budget_policy_checks':len(p)*3,'TEST_initial_utility_decreases':0,
 'primary_gain_TIME_vs_STOP':float(F(q['RETAIN_TIME']['mean_utility_difference_vs_PLAIN_STOP_exact'])),
 'primary_extra_time_TIME_vs_STOP_us':q['RETAIN_TIME']['mean_additional_time_vs_PLAIN_STOP_us'],
 'primary_mean_regret_TIME':q['RETAIN_TIME']['mean_regret'],'tradeoff_cases':cases,
 'predictor_diagnostics':[{k:v for k,v in x.items() if k!='observations'} for x in primary['predictor_diagnostics']['by_side']],
 'TEST_verification':{m:json.loads((e/f'C25_VERIFY_TEST_{m}.json').read_text())['native'] for m in ['TRACE','TIMING']},
 'raw_manifests':{f'{split}_{mode}':sha(e/f'C25_{split}_{mode}.manifest.json') for split in ['TRAIN','CAL','TEST'] for mode in ['TRACE','TIMING']},
 'scope':'input/behavior correspondence and published finite outcomes; no fresh timing, no test-driven retuning'}
out=e/'C08_RESULT_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['tradeoff_cases','raw_manifests','predictor_diagnostics']}))
