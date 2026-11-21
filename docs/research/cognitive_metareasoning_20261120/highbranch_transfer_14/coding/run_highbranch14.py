#!/usr/bin/env python3
"""Real 192 QUARTZ 15x15 Freestyle same-root CPU searches vs 48 Rapfi NNUE references.

All action policies, histories, and source hashes were committed before any
Rapfi reference was generated. Executes ALL 192 Rust requests first;
only then reads teacher outputs for posthoc scoring. This is NOT exact minimax.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,math,os,statistics,subprocess,time

ARMS=('closed','evidence','random_1','random_2','random_3',
      'all_after8','full32','closed_no_probe')
BINARY_SHA='435dfe735306f5bf4313f466d828e9569394b93ae04c487bc0c5e25e38682b14'
SOURCE_SHA='a744fa5166cb1cd54ec274128238649e7305a1ae3c4f15b894f35559602f5a9c'

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def choose_immediate_win_utility(board,side,move):
 if len(board)!=225 or side not in (-1,1) or type(move)!=int or not 0<=move<225 or board[move]!=0:
  raise ValueError('source Freestyle state or selected move invalid')
 y,x=divmod(move,15)
 for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
  run=1
  for direction in (-1,1):
   a=x+direction*dx;b=y+direction*dy
   while 0<=a<15 and 0<=b<15 and (board[b*15+a]==side or (b*15+a)==move):
    run+=1;a+=direction*dx;b+=direction*dy
  if run>=5:return 1
 return 0

def replay(history):
 b=[0]*225
 for i,a in enumerate(history):
  if type(a)!=int or not 0<=a<225 or b[a]:raise ValueError('illegal source original history')
  side=1 if i%2==0 else -1
  # Reject any history after an already completed five.
  if choose_immediate_win_utility(b,side,a):
   raise ValueError('training source contains terminal root')
  b[a]=side
 return b,1 if len(history)%2==0 else -1

def validate_real_root(r):
 first=r.get('first_stage',{})
 last=r.get('second_stage',{})
 if (r.get('same_root_identity_verified') is not True or
     r.get('edge_stats_preserved_at_activation') is not True or
     r.get('no_root_evaluator_call_on_mask_change') is not True or
     r.get('root_prior_preserved') is not True or
     r.get('root_hash_before')!=r.get('root_hash_after') or
     first.get('root_visits')!=8 or last.get('root_visits')!=32 or
     last.get('additional_root_visits')!=24 or
     first.get('edge_visit_sum')!=8 or last.get('edge_visit_sum')!=32 or
     first.get('edge_stats')!=r.get('activation_edge_stats')):
  raise ValueError('same root masked activation does not preserve N/W/M2')
 prev=first.get('edge_stats')
 later=last.get('edge_stats')
 if not isinstance(prev,list) or not isinstance(later,list) or len(prev)!=len(later):
  raise ValueError('root edge slab changed after selection-mask activation')
 if any(x[0]!=y[0] or x[2]!=y[2] or x[1]>y[1] for x,y in zip(prev,later)):
  raise ValueError('original root move/prior mutated or edge visits went backwards')
 if sum(int(x[1]) for x in prev)!=8 or sum(int(x[1]) for x in later)!=32:
  raise ValueError('per-action visits do not conserve total root visit count')

def pair_summary(outcomes,a,b):
 if not outcomes:raise ValueError('no root pairs')
 n=len(outcomes)
 xs=[z[a] for z in outcomes]
 ys=[z[b] for z in outcomes]
 ca=sum(z['cost_us'] for z in xs)/n
 cb=sum(z['cost_us'] for z in ys)/n
 if not ca>0 or not cb>0:raise ValueError('positive CPU time required')
 mismatch=abs(ca-cb)/ca
 win=sum(z['is_immediate_win'] for z in xs)-sum(z['is_immediate_win'] for z in ys)
 agree=sum(z['teacher_agree'] for z in xs)-sum(z['teacher_agree'] for z in ys)
 return {
  'n':n,'local_immediate_win_difference':win,
  'teacher_exact_agreement_difference':agree,
  'mean_CPU_wall_us_A':ca,'mean_CPU_wall_us_B':cb,
  'CPU_mean_relative_mismatch':mismatch,
  'cpu_10pct_match':'MATCHED_10_PERCENT' if mismatch<=0.1 else 'NOT_MATCHED_10_PERCENT',
  'A_minus_B_immediate_win_rate':win/n,
  'A_minus_B_teacher_agree_rate':agree/n,
  'finite_positions_only_not_population_inference':True
 }

def make_requests(cohort):
 controls=[]
 for row in cohort['rows']:
  for arm in ARMS:
   controls.append({
    'cmd':'research_gomoku15_highbranch_cpu',
    'rule':'freestyle','history':row['history'],'arm':arm,'seed':17,
   })
 return controls

def run_rust(exe,reqs):
 start=time.perf_counter()
 data='\n'.join(json.dumps(r,separators=(',',':')) for r in (reqs+[{'cmd':'quit'}]))+'\n'
 run=subprocess.run([str(exe),'--server'],input=data.encode(),capture_output=True,
                    timeout=240,check=True,
                    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
 lines=[x for x in run.stdout.decode().splitlines() if x.strip()]
 result=[json.loads(line) for line in lines]
 if len(result)!=len(reqs):
  raise RuntimeError(f'expecting {len(reqs)} real Rust results; got {len(result)}. stdout_last={lines[-2:]}; stderr_last={run.stderr[-1000:]}')
 return result,time.perf_counter()-start,run.stderr.decode(errors='replace')[-1500:]

def summarize_group(positions,scope):
 pairs=[x for x in positions if scope=='pooled' or x['stratum']==scope]
 if not pairs:raise ValueError('no data in stratum')
 output={
  'n':len(pairs),
  'known_immediate_win_state_count':sum(bool(x['exact_legal_immediate_wins']) for x in pairs),
  'stratum_is_constructed_stress':scope=='tactical_stress',
  'arms':{},'paired':{}
 }
 for arm in ARMS:
  records=[x[arm] for x in pairs]
  output['arms'][arm]={
   'n':len(records),
   'teacher_300ms_agree_count':sum(x['teacher_agree'] for x in records),
   'teacher_600ms_agree_count':sum(x['teacher_secondary_agree'] for x in records),
   'immediate_terminal_win_selected_count':sum(x['is_immediate_win'] for x in records),
   'mean_end_to_end_wall_ms':sum(x['cost_us'] for x in records)/len(records)/1000,
   'mean_pure_MCTS_ms':sum(x['pure_search_us'] for x in records)/len(records)/1000,
   'mean_full_root_setup_ms':sum(x['full_root_setup_us'] for x in records)/len(records)/1000,
   'mean_rules_probe_ms':sum(x['tactical_probe_us'] for x in records)/len(records)/1000,
   'mean_final_candidate_count':sum(x['final_active_count'] for x in records)/len(records),
   'actually_reactivated_roots':sum(x['added_count']>0 for x in records),
   'actually_reactivated_legal_immediate_wins':sum(x['reopened_immediate_win'] for x in records),
   'chosen_action_ge64_count':sum(x['action']>=64 for x in records),
  }
 for arm in ARMS:
  if arm=='full32':continue
  output['paired'][arm+'_minus_full32']=pair_summary(pairs,arm,'full32')
  if arm!='evidence':
   output['paired']['evidence_minus_'+arm]=pair_summary(pairs,'evidence',arm)
 return output

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--binary',type=Path,required=True)
 ap.add_argument('--output-dir',type=Path,required=True)
 args=ap.parse_args()
 d=Path(__file__).resolve().parent.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
  raise RuntimeError('run only from clean committed C08 runner')
 model=json.loads((d/'coding/C01_INPUT_LOCK.json').read_text())
 cohortfile=d/'COHORT_24_FROZEN.json'
 cohort=json.loads(cohortfile.read_text())
 if sha(cohortfile)!=SOURCE_SHA or len(cohort['rows'])!=24:
  raise RuntimeError('highbranch C14 test source changed after freeze')
 if model['initial_active_candidates']!=16 or model['MCTS_checkpoint_first']!=8 or model['MCTS_final_target_visits']!=32:
  raise RuntimeError('source compute budget drifted from precommitted')
 if model['random_candidate_seeds']!=[202611210411,202611210412,202611210413]:
  raise RuntimeError('unfrozen source control seeds')
 exe=args.binary.resolve()
 if not exe.is_file() or sha(exe)!=BINARY_SHA:
  raise RuntimeError('wrong compiled real 225-cell Rust CPU release binary')
 reqs=make_requests(cohort)
 if len(reqs)!=24*8 or any('teacher' in str(q) or 'oracle' in str(q) or 'Rapfi' in str(q) for q in reqs):
  raise RuntimeError('teacher leakage or request count changed')
 # THE FIRST AND ONLY ONLINE CODE INVOCATION. Teacher source is not opened.
 results,elapsed,stderr=run_rust(exe,reqs)
 prejoined=[]
 for row in cohort['rows']:
  prejoined.append({
   'id':row['id'],'stratum':row['stratum'],
   'history':row['history'],'side_to_move':row['side_to_move'],
   'calls':{},'metadata':{}
  })
 for j,(q,r) in enumerate(zip(reqs,results)):
  row=cohort['rows'][j//len(ARMS)]
  if 'error' in r:raise RuntimeError(f'actual Rust search returned an error in {row["id"]}/{q["arm"]}: {r["error"]}')
  if r['schema']!='quartz_cog04_highbranch14_real_same_tree_freestyle_v1' or r['history']!=row['history']:
   raise RuntimeError('executed Rust game source identity mismatch')
  if r['rule']!='freestyle' or r['source_prior_type']!='UniformEval_with_separate_center_rank_gate':
   raise RuntimeError('wrong game variant or falsely pretrained QUARTZ root policy')
  validate_real_root(r)
  board,side=replay(row['history'])
  legal={a for a,c in enumerate(board) if c==0}
  if r['side_to_move']!=side or r['root_full_original_legal_count']!=len(legal) or set(r['legal_actions'])!=legal:
   raise RuntimeError('real Rust game source disagrees with original 15x15 legal roots')
  if set(r['initial_candidates'])!=set(r['prior_center_rank'][:16]):
   raise RuntimeError('initial root center attention changed')
  if r['first_stage']['best_action'] not in r['initial_active_actions'] or r['final_legal_action'] not in r['final_active_actions']:
   raise RuntimeError('actual search selected a disabled root edge')
  if set(r['added_candidates'])-legal or len(set(r['added_candidates']))!=len(r['added_candidates']):
   raise RuntimeError('invalid new legal root activation')
  if q['arm']=='evidence' and r['added_candidates']!=r['tactical_probe']['suggested_reopen']:
   raise RuntimeError('online reopening no longer follows oracle-blind one-ply rules')
  if q['arm'] in ('closed','closed_no_probe') and r['added_candidates']:
   raise RuntimeError('closed original mask was secretly opened')
  if q['arm']=='full32' and r['root_mask_initial']!=[0,0,0,0]:
   raise RuntimeError('full-legal baseline not actually full')
  if q['arm']!='full32' and not any(r['root_mask_initial']):
   raise RuntimeError('frozen top16 mask did not activate')
  if r['total_wall_us_without_output_serialization']<=0:
   raise RuntimeError('no real physical wall time')
  prejoined[j//len(ARMS)]['calls'][q['arm']]=r
 # Only NOW load the externally pretrained alpha-beta references, after all
 # 192 Rust MCTS runs have finished; they are never input to the policy.
 teacher_file=d/'evidence/rapfi_teacher_20261121/RAPFI_REFERENCE_24.json'
 teacher=json.loads(teacher_file.read_text())
 if len(teacher['rows'])!=24 or teacher['reference_300ms_status_counts']!={'OK':24}:
  raise RuntimeError('trained independent Rapfi benchmark missing')
 from sys import path as modulepath
 source_dir=str(d/'coding')
 if source_dir not in modulepath:modulepath.insert(0,source_dir)
 from freeze_gomoku14 import immediate_wins as original_wins, replay as original_replay
 paired=[]
 for spec,teacher_row,entry in zip(cohort['rows'],teacher['rows'],prejoined):
  if spec['id']!=teacher_row['id'] or spec['history']!=teacher_row['history']:
   raise RuntimeError('teacher/source game history identity mismatch')
  board,side=replay(spec['history'])
  game=original_replay(spec['history'])
  assert game['side_to_move']==side and game['board']==board
  forced=original_wins(game)
  is_teacher_ok=teacher_row['reference_300ms']['status']=='OK'
  is_teacher2_ok=teacher_row['stability_600ms']['status']=='OK'
  alpha=teacher_row['reference_300ms']['flat_action'] if is_teacher_ok else None
  beta=teacher_row['stability_600ms']['flat_action'] if is_teacher2_ok else None
  if alpha is not None and board[alpha]!=0:raise RuntimeError('Rapfi recommended occupied source square')
  rec={'id':spec['id'],'stratum':spec['stratum'],
       'history':spec['history'],'side_to_move':side,
       'exact_legal_immediate_wins':forced,
       'precommitted_300ms_teacher_best':alpha,
       'secondary_600ms_teacher_best':beta,
       'teacher_300_600_agreement':alpha is not None and alpha==beta}
  for arm in ARMS:
   r=entry['calls'][arm]
   chosen=r['final_legal_action']
   rec[arm]={
    'action':chosen,
    'is_immediate_win':bool(choose_immediate_win_utility(board,side,chosen)),
    'teacher_agree':alpha is not None and chosen==alpha,
    'teacher_secondary_agree':beta is not None and chosen==beta,
    'initial_top16_contains_teacher':alpha in r['initial_candidates'] if alpha is not None else None,
    'initial_top16_contains_winning_action':bool(set(forced)&set(r['initial_candidates'])),
    'reopened_immediate_win':bool(set(forced)&set(r['added_candidates'])),
    'chosen_move_prior_center_rank':r['prior_center_rank'].index(chosen)+1,
    'added_count':len(r['added_candidates']),
    'final_active_count':len(r['final_active_actions']),
    'cost_us':r['total_wall_us_without_output_serialization'],
    'pure_search_us':r['cost']['pure_MCTS_search_wall_us'],
    'full_root_setup_us':r['cost']['full_root_setup_wall_us'],
    'tactical_probe_us':r['cost']['tactical_probe_wall_us'],
   }
  paired.append(rec)
 summary={
  'schema':'quartz_cog04_highbranch14_actual_15x15_realCPU_trained_Rapfi_comparison_v1',
  'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
  'Rust_compiled_source_commit':'7b64a681d7f0be37d10eb4417d079499a3fe1d45',
  'real_Rust_release_binary_SHA256':sha(exe),
  'new_C14_cohort_sha256':sha(cohortfile),
  'real_public_pretrained_Rapfi_reference_SHA256':sha(teacher_file),
  'physical_num_real_Rust_requests':len(results),
  'physical_num_root_search_segments':len(results)*2,
  'MCTS_visit_target':32,'prior_top_k':16,'reopen_max':4,
  'quartz_mcts_root_evaluator':'UniformEval NOT pretrained',
  'pretrained_teacher':'Real bundled Rapfi mix9svq NNUE alpha-beta, finite 300/600ms, NOT exact minimax',
  'actual_server_batch_wall_s':elapsed,
  'not_iid_strata':'12 independent seeded neutral examples, 12 symmetry/translation tactical stress examples from two templates',
  'by_stratum':{
    group:summarize_group(paired,group)
    for group in ('neutral','tactical_stress','pooled')
  },
  'no_Rapfi_teacher_information_was_available_during_online_selection':True,
  'all_root_statistics_preserved_at_reactivation':True,
  'teacher_finite_search_recommendations_not_game_theoretic_WDL':True,
 }
 out=args.output_dir.resolve()
 out.mkdir(parents=True,exist_ok=False)
 for name,obj in (
   ('RAW_192_REAL_RUST_SEARCHES.json',results),
   ('PAIRED_24_COMPARISON.json',paired),
   ('SUMMARY_REAL_15X15.json',summary)
 ):
  (out/name).write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n')
 manifest={
  'schema':'quartz_cog04_highbranch14_actual_CPU_trained_reference_execution_v1',
  'source_commit':summary['source_commit'],
  'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
  'Rust_optimized_source_commit':summary['Rust_compiled_source_commit'],
  'Rust_optimized_binary_sha256':summary['real_Rust_release_binary_SHA256'],
  'C14_source_sha256':summary['new_C14_cohort_sha256'],
  'Rapfi_24_real_pretrained_results_sha256':summary['real_public_pretrained_Rapfi_reference_SHA256'],
  'n_Rust_requests':len(results),'n_same_root_stage_executions':len(results)*2,
  'teacher_read_only_after_all_Rust_search_finished':True,
  'output_sha256':{p.name:sha(p) for p in out.iterdir()},
 }
 (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print('COG04_C08_REAL_PRETRAINED_REFERENCE_AND_CPU_SEARCH_COMPLETED',json.dumps({
  'n_Rust':len(results),'actual_batch_wall_s':elapsed,
  'strata':{
   group:{
    'evidence_win':summary['by_stratum'][group]['arms']['evidence']['immediate_terminal_win_selected_count'],
    'full32_win':summary['by_stratum'][group]['arms']['full32']['immediate_terminal_win_selected_count'],
    'closed_win':summary['by_stratum'][group]['arms']['closed']['immediate_terminal_win_selected_count'],
    'evidence_teacher300':summary['by_stratum'][group]['arms']['evidence']['teacher_300ms_agree_count'],
    'full_teacher300':summary['by_stratum'][group]['arms']['full32']['teacher_300ms_agree_count'],
    'cpu_gate_evidence_vs_full32':summary['by_stratum'][group]['paired']['evidence_minus_full32']['cpu_10pct_match']
   } for group in ('neutral','tactical_stress','pooled')
  }
 },sort_keys=True))
if __name__=='__main__':main()
