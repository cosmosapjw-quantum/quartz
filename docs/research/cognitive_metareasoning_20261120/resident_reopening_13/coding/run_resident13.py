#!/usr/bin/env python3
"""Actual 9216 CPU Rust SAME-root MCTS actions: resident masking vs all-legal24.

New C13 384 source positions and algorithm/thresholds were committed before
running this script. Every root uses ONE engine, real first8+additional16 visits.
Exact Pascal Pons action-score rows loaded only AFTER all Rust searches.
This is finite single-host CPU wall-time evidence, not human Einstellung or Elo.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,math,random,subprocess,sys,time

MODES=('uniform','new_mlp','conv')
ARMS=('resident_closed_sham','resident_evidence','resident_random_1',
      'resident_random_2','resident_random_3','resident_all_after8',
      'resident_full24','resident_closed_no_probe')
SEEDS=(202611210301,202611210302,202611210303)
BINARY_SHA='5dd61a01cc87dd715183f3d4145367b471182e0676a9a6bde7c38c9c225eede0'

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def sign(s):
 if type(s)!=int or s==-1000 or s<-21 or s>21:
  raise ValueError('illegal/sentinel game-score must not enter regret')
 return (s>0)-(s<0)

def exact_regret(scores,legal,action):
 if len(scores)!=7 or not legal or action not in legal:
  raise ValueError('invalid game root or illegal selected action')
 if any((c in legal)!=(scores[c]!=-1000) for c in range(7)):
  raise ValueError('external Pascal Pons illegal column sentinel mismatches legal moves')
 return max(sign(scores[c]) for c in legal)-sign(scores[action])

def decisive_omitted(scores,legal,initial):
 if not initial or len(set(initial))!=len(initial) or not set(initial).issubset(legal):
  raise ValueError('initial candidates are not unique legal root actions')
 best_inside=max(sign(scores[c]) for c in initial)
 return [c for c in legal if c not in initial and sign(scores[c])>best_inside]

def physical_mean_cost_match(policy_ms,baseline_ms):
 if not math.isfinite(policy_ms) or not math.isfinite(baseline_ms) or policy_ms<=0 or baseline_ms<=0:
  raise ValueError('actual physical CPU wall time must be positive')
 return 'MATCHED_10_PERCENT' if abs(policy_ms-baseline_ms)/policy_ms<=.10 else 'NOT_MATCHED_10_PERCENT'

def paired_cluster_ci(left,right,seed=20261121):
 if len(left)!=len(right) or not left:
  raise ValueError('family-level paired nonempty sample required')
 clusters=defaultdict(list)
 for a,b in zip(left,right):
  if a['family']!=b['family']:
   raise ValueError('mismatched opening family')
  clusters[a['family']].append(a['regret']-b['regret'])
 if any(len(z)!=16 for z in clusters.values()):
  raise ValueError('each opening family must have all 16 descendants')
 means=[sum(v)/len(v) for _,v in sorted(clusters.items())]
 rng=random.Random(seed)
 draws=sorted(sum(rng.choice(means) for _ in means)/len(means) for _ in range(3000))
 return {'n_roots':len(left),'n_family_clusters':len(means),
         'mean_regret_delta':sum(means)/len(means),
         'CI95':[draws[74],draws[2924]],
         'not_independent_per_descendant_root':True}

def validate_resident_transition(r):
 a=r.get('first_search',{})
 b=r.get('second_search',{})
 initial=a.get('edge_stats')
 instant=r.get('activation_edge_stats')
 if (a.get('root_visits')!=8 or b.get('root_visits')!=24 or
     b.get('additional_root_visits')!=16 or
     a.get('edge_visit_sum')!=8 or b.get('edge_visit_sum')!=24 or
     r.get('root_visits_immediately_after_activation')!=8 or
     r.get('root_hash_before')!=r.get('root_hash_after') or
     r.get('resident_same_root_node') is not True or
     r.get('root_evaluator_calls_identical_during_activation') is not True or
     r.get('prior_and_edge_stats_identical_immediately_after_activation') is not True or
     not isinstance(initial,list) or initial!=instant or
     sum(x['visits'] for x in initial)!=8 or
     sum(x['visits'] for x in b.get('edge_stats',[]))!=24):
  raise ValueError('actual resident root conservation failure')
 if a.get('root_evaluator_calls')>b.get('root_evaluator_calls',-1):
  raise ValueError('root NN evaluation calls decreased')
 if len(initial)!=r.get('original_full_root_edges'):
  raise ValueError('not the original full root edge slab')
 if any(x['action']!=y['action'] or x['prior']!=y['prior'] or x['visits']>y['visits']
        for x,y in zip(initial,b.get('edge_stats',[]))):
  raise ValueError('resident root edge identity/prior or visit monotonicity failure')

def modules():
 p=Path(__file__).resolve().parent.parent.parent/'cost_matched_voc_08'
 if str(p) not in sys.path:sys.path.insert(0,str(p))
 from train_gain08 import model_identity
 from evaluate_live08 import model_request,run_batch
 return model_identity,model_request,run_batch

def to_request(row,mode,arm,models,shas):
 model_identity,model_request,run_batch=modules()
 q=model_request(row,mode,'full',models,shas,None,None)
 q['cmd']='research_connect4_resident_cpu'
 q['arm']=arm
 q.pop('budgets',None)
 assert all(x not in q for x in ('oracle','exact_action_scores','teacher_WDL','future_root_actions'))
 return q

def metrics(rows,mode,arm):
 a=[r['modes'][mode][arm] for r in rows]
 n=len(a)
 if not n:raise ValueError('empty new cohort')
 return {
 'n':n,'n_families':len({r['family'] for r in rows}),
 'mean_regret':sum(z['regret'] for z in a)/n,
 'mean_WDL':sum(z['WDL'] for z in a)/n,
 'mean_end_to_end_ms_excluding_model_load_and_JSON_serialization':sum(z['decision_wall_ms'] for z in a)/n,
 'mean_pure_search_ms':sum(z['search_ms'] for z in a)/n,
 'mean_root_setup_ms':sum(z['setup_ms'] for z in a)/n,
 'mean_prior_eval_ms':sum(z['prior_eval_ms'] for z in a)/n,
 'mean_tactical_probe_ms':sum(z['probe_ms'] for z in a)/n,
 'mean_completed_evaluator_calls_including_extra_prior':sum(z['evaluator_calls'] for z in a)/n,
 'mean_final_active_candidate_count':sum(len(z['final_candidates']) for z in a)/n,
 'triggered_roots':sum(z['probe_trigger'] for z in a),
 'actually_reactivated_roots':sum(bool(z['actually_added_candidates']) for z in a),
 'reopened_decisive_omission_roots':sum(z['actually_reopened_decisive'] for z in a),
 'final_chosen_decisive_omission_roots':sum(z['chosen_decisive_omission'] for z in a),
 'root_regret_distribution':dict(Counter(str(z['regret']) for z in a)),
 'root_action_WDL_distribution':dict(Counter(str(z['WDL']) for z in a)),
 }

def main():
 cli=argparse.ArgumentParser(description=__doc__)
 cli.add_argument('--binary',type=Path,required=True)
 cli.add_argument('--out-dir',type=Path,required=True)
 args=cli.parse_args()
 d=Path(__file__).resolve().parent.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
  raise RuntimeError('execute only from committed clean frozen search code')
 lock=json.loads((d/'coding/INPUT_LOCK.json').read_text())
 source=d/'COHORT_384_FROZEN.json'
 coh=json.loads(source.read_text())
 if coh['seeds']!=lock['new_source_seeds'] or len(coh['rows'])!=384 or len(coh['families'])!=24:
  raise RuntimeError('new C13 root distribution not frozen correctly')
 if list(SEEDS)!=lock['random_reopen_seeds']:
  raise RuntimeError('online random root activation seeds drifted')
 if lock['budget_first_visits']!=8 or lock['additional_visits']!=16:
  raise RuntimeError('resumable initial/post computational schedule drifted')
 model_identity,model_request,run_batch=modules()
 models=model_identity(repo)
 hashes={name:sha(p) for name,p in models.items()}
 original={r['id']:r for r in coh['rows']}
 wire=[];keys=[]
 for root in coh['rows']:
  for mode in MODES:
   for arm in ARMS:
    wire.append(to_request(root,mode,arm,models,hashes))
    keys.append((root['id'],mode,arm))
 assert len(wire)==384*3*8==9216
 exe=args.binary.resolve()
 if not exe.is_file() or sha(exe)!=BINARY_SHA:
  raise RuntimeError('actual compiled Rust study executable identity changed')
 start=time.perf_counter()
 raw=run_batch(exe,wire)
 elapsed=time.perf_counter()-start
 assert len(raw)==9216
 # First open true game-theoretic external score file after ALL 9216
 # online Rust root searches have finished. It must not alter root gating.
 exactfile=d/'evidence/new384_exact_20261121/EXACT_NEW384_LABELS.json'
 exact=json.loads(exactfile.read_text())
 assert exact['status']['all_384_OK'] and exact['status']['counts']=={'OK':384}
 teacher={r['id']:r for r in exact['rows']}
 assert len(teacher)==384
 paired={root['id']:{
    'id':root['id'],'history':root['history'],'family':root['family'],
    'group':root['group'],'side_to_move':root['current_player'],
    'optimal_exact_root_WDL':teacher[root['id']]['root_wdl'],
    'modes':{},
 } for root in coh['rows']}
 index={}
 for (id,mode,arm),r in zip(keys,raw):
  root=original[id]
  if 'error' in r:
   raise RuntimeError(f'actual Rust root returned error on {id}/{mode}/{arm}: {r["error"]}')
  if (r['schema']!='quartz_cog03_resident13_actual_same_tree_root_activation_v1' or
      r['history']!=root['history'] or r['board_bottom_up']!=root['board'] or
      r['side_to_move']!=root['current_player'] or
      r['legal_columns_0based']!=root['legal_columns'] or r['seed']!=17):
   raise RuntimeError('Rust source game/board/current player/seed mismatch')
  if r['eval_mode']!={'uniform':'uniform_no_neural_network','new_mlp':'trained_tiny_mlp_cpu','conv':'trained_conv3x3_cpu'}[mode]:
   raise RuntimeError('wrong frozen NN eval source selected')
  if mode!='uniform' and not r['cache_model']:
   raise RuntimeError('original trained model weights not reused/resident')
  validate_resident_transition(r)
  if not r['no_external_exact_oracle_in_runtime'] or not r['single_tree_root_activation_not_late_edge_insertion']:
   raise RuntimeError('misrepresented online oracle or root edge approach')
  legal=r['legal_columns_0based']
  initial=r['initial_candidates']
  final=r['final_candidates']
  prior=[a for a,p in r['full_unmasked_prior']]
  if (not initial or len(initial)>3 or prior[:len(initial)]!=initial or
     sorted(prior)!=sorted(legal) or len(set(final))!=len(final) or
     not set(final).issubset(legal)):
   raise RuntimeError('original full evaluator prior or actual active-set invalid')
  if r['first_search']['best_action'] not in r['first_active_candidates'] or r['second_search']['best_action'] not in final:
   raise RuntimeError('Rust root selected an inactive source action')
  if arm=='resident_full24':
   if r['first_root_action_mask']!=0 or r['final_root_action_mask']!=0 or set(final)!=set(legal):
    raise RuntimeError('all-legal reference was secretly masked')
  else:
   bits=lambda moves:sum(1<<m for m in moves)
   if r['first_root_action_mask']!=bits(initial) or r['final_root_action_mask']!=bits(final):
    raise RuntimeError('actual resident root bitset differs from source root actions')
  probe=r.get('tactical_probe')
  if arm=='resident_closed_no_probe':
   if probe is not None:raise RuntimeError('no-probe control actually used oneply check')
  elif not isinstance(probe,dict):
   raise RuntimeError('sham/treatment tactical probe was not executed')
  if arm=='resident_evidence' and r['actually_added_candidates']!=probe['suggested_reopen']:
   raise RuntimeError('online evidence root reactivated without a witnessed move')
  if arm.startswith('resident_random_') and len(r['actually_added_candidates'])!=len(probe['suggested_reopen']):
   raise RuntimeError('random comparator did not activate equal-count root moves')
  if arm=='resident_all_after8' and set(final)!=set(legal):
   raise RuntimeError('all-open-after-eight did not activate every original legal root')
  if arm in ('resident_closed_sham','resident_closed_no_probe') and final!=initial:
   raise RuntimeError('closed sham reactivated a root move')
  if r['root_visits_immediately_after_activation']!=8:
   raise RuntimeError('activation consumed a root MCTS iteration')
  score=teacher[id]['exact_action_scores']
  action=r['second_search']['best_action']
  WDL=sign(score[action])
  regret=exact_regret(score,legal,action)
  omit=decisive_omitted(score,legal,initial)
  receipt={
   'action':action,'regret':regret,'WDL':WDL,
   'first_action':r['first_search']['best_action'],
   'first_action_WDL':sign(score[r['first_search']['best_action']]),
   'prior_rank_of_final_action':prior.index(action)+1,
   'initial_candidates':initial,'final_candidates':final,
   'actually_added_candidates':r['actually_added_candidates'],
   'probe_trigger':bool(probe['trigger']) if probe else False,
   'probe_reason':probe['reason'] if probe else 'no_probe',
   'fixation_share':probe['fixation_share'] if probe else None,
   'strictly_decisive_omitted':omit,
   'actually_reopened_decisive':bool(set(r['actually_added_candidates'])&set(omit)),
   'chosen_decisive_omission':action in omit,
   'decision_wall_ms':r['total_wall_us_without_model_load']/1000,
   'search_ms':r['cost']['two_stage_pure_search_wall_us']/1000,
   'setup_ms':r['cost']['full_root_setup_wall_us']/1000,
   'prior_eval_ms':r['cost']['prior_source_eval_wall_us']/1000,
   'probe_ms':r['cost']['one_ply_probe_wall_us']/1000,
   'activation_ms':r['cost']['activation_mask_wall_us']/1000,
   'model_load_ms':r['cost']['model_load_wall_us']/1000,
   'evaluator_calls':r['total_evaluator_calls_including_prior_source'],
   'first_visit_count':r['first_search']['root_visits'],
   'final_visit_count':r['second_search']['root_visits'],
   'actual_root_identity_preserved':r['resident_same_root_node'],
   'activation_edgestats_preserved':r['prior_and_edge_stats_identical_immediately_after_activation'],
   'root_hash':r['root_hash_before'],
  }
  paired[id]['modes'].setdefault(mode,{})[arm]=receipt
  index[(id,mode,arm)]=r
 for_mode=set(MODES)
 paired_rows=[paired[r['id']] for r in coh['rows']]
 for entry in paired_rows:
  for mode in MODES:
   base=index[(entry['id'],mode,'resident_closed_sham')]
   intervention=index[(entry['id'],mode,'resident_evidence')]
   for field in ('initial_candidates','full_unmasked_prior','initially_omitted'):
    if base[field]!=intervention[field]:raise RuntimeError('root prior/restriction drift among arms')
   for field in ('best_action','root_policy','root_evaluator_calls','edge_stats'):
    if base['first_search'][field]!=intervention['first_search'][field]:
     raise RuntimeError('initial first8 resident MCTS evidence arm differs from closed sham')
   if base['tactical_probe']['suggested_reopen']!=intervention['tactical_probe']['suggested_reopen']:
    raise RuntimeError('sham/treatment not exposed to same oracle-blind rule evidence')
   for arm in ('resident_random_1','resident_random_2','resident_random_3'):
    if index[(entry['id'],mode,arm)]['tactical_probe']['suggested_reopen']!=intervention['tactical_probe']['suggested_reopen']:
     raise RuntimeError('random same-count controlled treatment violated')
   if set(base['final_candidates'])==set(intervention['final_candidates']):
    if base['second_search']['best_action']!=intervention['second_search']['best_action']:
     raise RuntimeError('unchanged resident mask unexpectedly altered deterministic search')
 for name,path in models.items():
  assert sha(path)==hashes[name],'trained model weights changed during online test'
 summary={
   'schema':'quartz_cog03_resident13_finite_CPU_exact_WDL_heldout_v1',
   'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
   'actual_Rust_release_root_requests':len(raw),
   'same_root_first8_plus16_actual_MCTS_searches':2*len(raw),
   'new_roots':384,'new_family_clusters':24,
   'train_model_sha256':hashes,
   'release_binary_sha256':sha(exe),
   'exact_label_sha256':sha(exactfile),
   'cohort_sha256':sha(source),
   'elapsed_all_server_s':elapsed,'results':{},
   'policy_restriction_is_artificial':True,
   'not_truly_new_late_inserted_slab_edges':True,
   'root_full_edges_allocated_upfront':True,
   'live_solver_action_scores_for_online_decision':False,
 }
 for group in ('resident_A','resident_B','pooled'):
  selected=[r for r in paired_rows if group=='pooled' or r['group']==group]
  assert len(selected)==(384 if group=='pooled' else 192)
  summary['results'][group]={}
  for mode in MODES:
   arm_metrics={arm:metrics(selected,mode,arm) for arm in ARMS}
   base=arm_metrics['resident_evidence']
   comparison={}
   for arm in ('resident_closed_sham','resident_random_1','resident_random_2','resident_random_3',
               'resident_all_after8','resident_full24','resident_closed_no_probe'):
    rhs=arm_metrics[arm]
    left=[{'family':r['family'],'regret':r['modes'][mode]['resident_evidence']['regret']} for r in selected]
    right=[{'family':r['family'],'regret':r['modes'][mode][arm]['regret']} for r in selected]
    mismatch=(
      abs(base['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization']-
          rhs['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization'])/
      base['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization']
    )
    comparison['evidence_vs_'+arm]={
     'mean_exact_WDL_regret_difference':base['mean_regret']-rhs['mean_regret'],
     'physical_CPU_mean_time_mismatch_fraction':mismatch,
     'mean_CPU_cost_10pct_admission':physical_mean_cost_match(
         base['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization'],
         rhs['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization']),
     'bootstrap_family_conditional':paired_cluster_ci(left,right)
    }
   evidence=[r['modes'][mode]['resident_evidence'] for r in selected]
   closed=[r['modes'][mode]['resident_closed_sham'] for r in selected]
   meaningful=[r for r in selected if len(
       {sign(teacher[r['id']]['exact_action_scores'][c]) for c in original[r['id']]['legal_columns']})>1]
   response=Counter('better' if a['WDL']>b['WDL'] else 'worse' if a['WDL']<b['WDL'] else 'same'
                    for a,b in zip(evidence,closed))
   summary['results'][group][mode]={
     'arms':arm_metrics,'paired_comparisons':comparison,
     'decision_informative_roots':len(meaningful),
     'root_WDL_class_counts':dict(Counter(str(teacher[r['id']]['root_wdl']) for r in selected)),
     'strictly_decisive_omission_eligible_roots':sum(bool(x['strictly_decisive_omitted']) for x in evidence),
     'eligible_but_never_triggered':sum(bool(x['strictly_decisive_omitted']) and not x['probe_trigger'] for x in evidence),
     'decisive_omission_actually_reopened_count':sum(x['actually_reopened_decisive'] for x in evidence),
     'selected_strictly_better_omission_count':sum(x['chosen_decisive_omission'] for x in evidence),
     'WDL_action_change_vs_closed_sham':dict(response),
     'tactical_probe_reason_counts':dict(Counter(x['probe_reason'] for x in evidence)),
     'informative_only_mean_regret_by_arm':{
         arm:sum(r['modes'][mode][arm]['regret'] for r in meaningful)/len(meaningful)
         if meaningful else None for arm in ARMS},
   }
 out=args.out_dir.resolve()
 out.mkdir(parents=True,exist_ok=False)
 for name,value in [('RAW_9216_RESIDENT_ROOTS.json',raw),
                     ('PAIRED_384_RESIDENT_WDL.json',paired_rows),
                     ('SUMMARY_RESIDENT13.json',summary)]:
  (out/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
 manifest={
  'schema':'quartz_cog03_resident13_real_release_manifest_v1',
  'source_commit':summary['source_commit'],
  'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
  'compiled_Rust_source_commit':'8c63a60973ac11b9e1fa4cdf93d387083c65ee57',
  'actual_Rust_compiled_binary_sha256':sha(exe),
  'new384_sha256':sha(source),
  'new384_exact_oracle_sha256':sha(exactfile),
  'trained_models_sha256':hashes,
  'n_actual_Rust_resident_search_requests':len(raw),
  'n_actual_8_and_24_root_MCTS_invocations':2*len(raw),
  'oracle_source_first_loaded_after_all_Rust_responses':True,
  'all_same_root_stats_preserved_implemented':True,
  'new_game_groups':2,'opening_families':24,
  'output_SHA256':{p.name:sha(p) for p in out.iterdir()},
 }
 (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print('C13_C08_ACTUAL_RESIDENT_CPU_HELDOUT_FINISHED',json.dumps({
   'actual_requests':len(raw),'elapsed_s':elapsed,
   'mode_results':{m:{
     g:{
      'evidence_regret':summary['results'][g][m]['arms']['resident_evidence']['mean_regret'],
      'all_legal24_regret':summary['results'][g][m]['arms']['resident_full24']['mean_regret'],
      'closed_sham_regret':summary['results'][g][m]['arms']['resident_closed_sham']['mean_regret'],
      'evidence_vs_all_legal_CPU_gate':summary['results'][g][m]['paired_comparisons']['evidence_vs_resident_full24']['mean_CPU_cost_10pct_admission'],
      'evidence_activated_count':summary['results'][g][m]['arms']['resident_evidence']['actually_reactivated_roots'],
      'decisive_recovered_count':summary['results'][g][m]['decisive_omission_actually_reopened_count']
     } for g in ('resident_A','resident_B','pooled')
   } for m in MODES},
 },sort_keys=True))

if __name__=='__main__':main()
