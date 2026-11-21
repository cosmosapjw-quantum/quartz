#!/usr/bin/env python3
"""Independent complete 9216 real Rust same-root MCTS transition/action audit.

This file deliberately DOES NOT import run_resident13.py or original one-ply
tactical Rust/Python implementations. All source gravity, winning lines,
FNV randomized reopening, exact WDL scores and family bootstrap are recalced.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math,random,subprocess

MODES=('uniform','new_mlp','conv')
ARMS=('resident_closed_sham','resident_evidence','resident_random_1',
      'resident_random_2','resident_random_3','resident_all_after8',
      'resident_full24','resident_closed_no_probe')
RANDOM_SEEDS=(202611210301,202611210302,202611210303)

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def game(history):
 board=[0]*42
 heights=[0]*7
 for i,c in enumerate(history):
  k=int(c)-1
  if not 0<=k<7 or heights[k]>=6:raise ValueError('invalid gravity')
  board[heights[k]*7+k]=1 if i%2==0 else -1
  heights[k]+=1
 return board,heights

def wins(board,heights,player,col):
 if not 0<=col<7 or heights[col]>=6:return False
 y=heights[col]
 b=board.copy();b[y*7+col]=player
 for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
  n=1
  for side in (-1,1):
   x=col+side*dx;yy=y+side*dy
   while 0<=x<7 and 0<=yy<6 and b[yy*7+x]==player:
    n+=1;x+=side*dx;yy+=side*dy
  if n>=4:return True
 return False

def opponent_immediate(board,heights,player,col):
 if wins(board,heights,player,col):return False
 after=board.copy();next_heights=heights.copy()
 after[next_heights[col]*7+col]=player;next_heights[col]+=1
 if sum(next_heights)==42:return False
 return any(wins(after,next_heights,-player,j) for j in range(7) if next_heights[j]<6)

def tactical(root,root_prior,initial,first_best,root_pi):
 b,ht=game(root['history'])
 player=root['current_player']
 omitted=[c for c in root_prior if c not in initial]
 if first_best not in initial:first_best=initial[0]
 immediate=[c for c in omitted if wins(b,ht,player,c)]
 safe=[c for c in omitted if c not in immediate and not opponent_immediate(b,ht,player,c)]
 unsafe=opponent_immediate(b,ht,player,first_best)
 max_share=max(float(p) for c,p in root_pi)
 fixation=max_share>=.5
 if not fixation:proposal=[];reason='fixation_threshold_not_reached'
 elif immediate:proposal=immediate[:2];reason='excluded_immediate_win'
 elif unsafe and safe:proposal=safe[:2];reason='leader_allows_immediate_opponent_win'
 else:proposal=[];reason='no_tactical_reopening_evidence'
 return {'proposal':proposal,'fixation':fixation,'share':max_share,
         'immediate':immediate,'safe':safe,'unsafe':unsafe,'reason':reason}

def fnv_random_candidates(omitted,n,seed,history):
 def h(c):
  value=0xcbf29ce484222325 ^ seed
  for byte in history.encode()+bytes([c]):
   value=(value^byte)*0x1000000001b3 & 0xffffffffffffffff
  return value
 return sorted(omitted,key=h)[:n]

def sgn(v):
 if type(v)!=int or not -21<=v<=21:raise ValueError('illegal/sentinel external WDL')
 return (v>0)-(v<0)

def pons(exe,book,h):
 r=subprocess.run([str(exe),'-a','-b',str(book)],
    input=(h+'\n').encode(),capture_output=True,check=True,timeout=12)
 fields=r.stdout.decode().split()
 assert len(fields)==8 and fields[0]==h
 return [int(q) for q in fields[1:]]

def bootstrap(rows,mode,arm):
 fam=defaultdict(list)
 for z in rows:
  q=z['modes'][mode]
  fam[z['family']].append(q['resident_evidence']['regret']-q[arm]['regret'])
 assert all(len(v)==16 for v in fam.values())
 m=[sum(v)/len(v) for _,v in sorted(fam.items())]
 rng=random.Random(20261121)
 values=sorted(sum(rng.choice(m) for _ in m)/len(m) for _ in range(3000))
 return sum(m)/len(m),[values[74],values[2924]]

def main():
 d=Path(__file__).resolve().parent.parent
 e=d/'evidence/live_9216_resident_20261121'
 m=json.loads((e/'EXECUTION_MANIFEST.json').read_text())
 data=json.loads((e/'RAW_9216_RESIDENT_ROOTS.json').read_text())
 paired=json.loads((e/'PAIRED_384_RESIDENT_WDL.json').read_text())
 summary=json.loads((e/'SUMMARY_RESIDENT13.json').read_text())
 new=json.loads((d/'COHORT_384_FROZEN.json').read_text())
 exact=json.loads((d/'evidence/new384_exact_20261121/EXACT_NEW384_LABELS.json').read_text())
 lock=json.loads((d/'coding/INPUT_LOCK.json').read_text())
 assert m['source_commit']=='bf5bda76ba817476f635a8d245603d8c95949211'
 assert m['n_actual_Rust_resident_search_requests']==len(data)==9216
 assert m['n_actual_8_and_24_root_MCTS_invocations']==18432
 assert len(paired)==len(new['rows'])==len(exact['rows'])==384
 assert exact['status']['counts']=={'OK':384}
 assert sha(d/'COHORT_384_FROZEN.json')==m['new384_sha256']
 assert sha(d/'evidence/new384_exact_20261121/EXACT_NEW384_LABELS.json')==m['new384_exact_oracle_sha256']
 assert sha(Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo'))==m['actual_Rust_compiled_binary_sha256']
 for filename,digest in m['output_SHA256'].items():
  assert sha(e/filename)==digest,filename
 oracle={x['id']:x for x in exact['rows']}
 choice=Counter()
 residency=Counter()
 mask_visits=Counter()
 root_info=Counter()
 for i,src in enumerate(new['rows']):
  pos=paired[i]
  assert pos['id']==src['id'] and pos['family']==src['family'] and pos['group']==src['group']
  b,heights=game(src['history'])
  legal=[col for col in range(7) if heights[col]<6]
  assert b==src['board'] and legal==src['legal_columns']
  score=oracle[src['id']]['exact_action_scores']
  assert len(score)==7 and all((score[col]==-1000)==(col not in legal) for col in range(7))
  maxu=max(sgn(score[col]) for col in legal)
  if len({sgn(score[col]) for col in legal})>1:
   root_info[src['group']]+=1
  for j,mode in enumerate(MODES):
   plain=data[(i*3+j)*8+0]
   initial=plain['initial_candidates']
   rank=[col for col,prior in plain['full_unmasked_prior']]
   assert rank[:len(initial)]==initial and sorted(rank)==sorted(legal)
   originalp={col:float(p) for col,p in plain['full_unmasked_prior']}
   assert len(initial)==min(3,len(legal))
   order=sorted(plain['full_unmasked_prior'],key=lambda z:(-float(z[1]),abs(z[0]-3),z[0]))
   assert order==plain['full_unmasked_prior']
   omitted=[col for col in rank if col not in initial]
   decisive=[col for col in legal if col not in initial and sgn(score[col])>
          max(sgn(score[a]) for a in initial)]
   first_base=plain['first_search']
   for k,arm in enumerate(ARMS):
    r=data[(i*3+j)*8+k]
    q=pos['modes'][mode][arm]
    assert r['history']==src['history'] and r['seed']==17
    assert r['board_bottom_up']==b and r['side_to_move']==src['current_player']
    assert r['legal_columns_0based']==legal
    assert r['original_full_root_edges']==len(legal)
    assert r['first_search']['root_visits']==8 and r['second_search']['root_visits']==24
    assert r['second_search']['additional_root_visits']==16
    assert r['first_search']['edge_visit_sum']==8 and r['second_search']['edge_visit_sum']==24
    assert r['root_visits_immediately_after_activation']==8
    assert r['root_hash_before']==r['root_hash_after']
    assert r['resident_same_root_node'] and r['root_evaluator_calls_identical_during_activation']
    assert r['prior_and_edge_stats_identical_immediately_after_activation']
    assert r['first_search']['edge_stats']==r['activation_edge_stats']
    assert r['first_search']['root_evaluator_calls']<=r['second_search']['root_evaluator_calls']
    assert r['no_external_exact_oracle_in_runtime'] and r['single_tree_root_activation_not_late_edge_insertion']
    assert r['first_search']['best_action'] in r['first_active_candidates']
    assert r['second_search']['best_action'] in r['final_candidates']
    assert r['initial_candidates']==initial and r['full_unmasked_prior']==plain['full_unmasked_prior']
    if arm!='resident_full24':
     assert r['first_search']['best_action']==first_base['best_action']
     assert r['first_search']['edge_stats']==first_base['edge_stats']
     assert r['first_search']['root_policy']==first_base['root_policy']
    edges=r['first_search']['edge_stats']
    end=r['second_search']['edge_stats']
    assert len(edges)==len(end)==len(legal)
    assert sorted(x['action'] for x in edges)==sorted(legal)
    for x,y in zip(edges,end):
     assert x['action']==y['action'] and x['prior']==y['prior']
     assert abs(x['prior']-originalp[x['action']])<2e-7
     assert 0<=x['visits']<=y['visits']
     if arm!='resident_full24' and x['action'] not in initial:
      assert x['visits']==0
      mask_visits['initial_disabled_has_zero_visits']+=1
     if arm!='resident_full24' and y['action'] not in r['final_candidates']:
      assert y['visits']==0
      mask_visits['final_disabled_has_zero_visits']+=1
    residency[(mode,bool(r['model_cache_hit']))]+=1
    if arm=='resident_full24':
     assert r['first_root_action_mask']==0 and r['final_root_action_mask']==0
     assert r['first_active_candidates']==legal and r['final_candidates']==legal
    else:
     encode=lambda moves:sum(1<<col for col in moves)
     assert r['first_root_action_mask']==encode(initial)
     assert r['final_root_action_mask']==encode(r['final_candidates'])
     assert r['first_root_action_mask']&r['final_root_action_mask']==r['first_root_action_mask']
    if arm=='resident_closed_no_probe':
     assert r['tactical_probe'] is None
     expected_added=[]
    else:
     t=tactical(src,rank,initial,r['first_search']['best_action'],
                r['first_search']['root_policy'])
     probe=r['tactical_probe']
     assert probe['reason']==t['reason'],(src['id'],mode,arm)
     assert probe['fixation_gate']==t['fixation']
     assert abs(probe['fixation_share']-t['share'])<2e-7
     assert probe['original_omitted_immediate_wins']==t['immediate']
     assert probe['original_omitted_safe_moves']==t['safe']
     assert probe['initial_best_reply_vulnerable']==t['unsafe']
     assert probe['suggested_reopen']==t['proposal']
     if arm=='resident_evidence':expected_added=t['proposal']
     elif arm.startswith('resident_random_'):
      n=int(arm.rsplit('_',1)[1])-1
      expected_added=fnv_random_candidates(omitted,len(t['proposal']),
                                               RANDOM_SEEDS[n],src['history'])
     elif arm=='resident_all_after8':expected_added=omitted
     else:expected_added=[]
    assert expected_added==r['actually_added_candidates'],(src['id'],mode,arm)
    wanted=legal if arm=='resident_full24' else initial+expected_added
    assert wanted==r['final_candidates']
    a=r['second_search']['best_action']
    assert q['action']==a and q['WDL']==sgn(score[a]) and q['regret']==maxu-sgn(score[a])
    assert q['strictly_decisive_omitted']==decisive
    assert q['actually_reopened_decisive']==bool(set(expected_added)&set(decisive))
    assert q['chosen_decisive_omission']==(a in decisive)
    assert abs(q['decision_wall_ms']-r['total_wall_us_without_model_load']/1000)<1e-12
    assert abs(q['search_ms']-r['cost']['two_stage_pure_search_wall_us']/1000)<1e-12
    assert r['total_evaluator_calls_including_prior_source']==r['second_search']['root_evaluator_calls']+1
    assert r['total_wall_us_without_model_load']>0 and r['total_wall_us_without_model_load']>=r['cost']['two_stage_pure_search_wall_us']
    choice[(mode,arm)]+=1
 assert len(choice)==3*8
 for mode in MODES:
  for arm in ARMS:assert choice[(mode,arm)]==384
 for mode in ('new_mlp','conv'):
  assert residency[(mode,False)]==1 and residency[(mode,True)]==3071,residency
 groups={}
 for group in ('resident_A','resident_B','pooled'):
  subset=[r for r in paired if group=='pooled' or r['group']==group]
  assert len(subset)==(384 if group=='pooled' else 192)
  groups[group]={}
  for mode in MODES:
   block=summary['results'][group][mode]
   means={}
   for arm in ARMS:
    arr=[r['modes'][mode][arm] for r in subset]
    reg=sum(x['regret'] for x in arr)/len(arr)
    wall=sum(x['decision_wall_ms'] for x in arr)/len(arr)
    assert abs(reg-block['arms'][arm]['mean_regret'])<1e-12
    assert abs(wall-block['arms'][arm]['mean_end_to_end_ms_excluding_model_load_and_JSON_serialization'])<1e-12
    means[arm]={'mean_regret':reg,'mean_CPU_wall_ms':wall}
   contrasts={}
   for arm in ('resident_closed_sham','resident_random_1','resident_random_2',
               'resident_random_3','resident_all_after8','resident_full24',
               'resident_closed_no_probe'):
    datum=block['paired_comparisons']['evidence_vs_'+arm]
    delta=means['resident_evidence']['mean_regret']-means[arm]['mean_regret']
    mismatch=abs(means['resident_evidence']['mean_CPU_wall_ms']-
                 means[arm]['mean_CPU_wall_ms'])/means['resident_evidence']['mean_CPU_wall_ms']
    assert abs(delta-datum['mean_exact_WDL_regret_difference'])<1e-12
    assert abs(mismatch-datum['physical_CPU_mean_time_mismatch_fraction'])<1e-12
    gate='MATCHED_10_PERCENT' if mismatch<=.1 else 'NOT_MATCHED_10_PERCENT'
    assert gate==datum['mean_CPU_cost_10pct_admission']
    _,ci=bootstrap(subset,mode,arm)
    assert all(abs(a-b)<1e-12 for a,b in zip(ci,datum['bootstrap_family_conditional']['CI95']))
    contrasts[arm]={'mean_delta':delta,'bootstrap_CI95':ci,
                    'mean_physical_CPU_time_mismatch':mismatch,'cost_admitted':gate}
   groups[group][mode]={'arms':means,'contrasts':contrasts,
     'n_decision_informative_roots':block['decision_informative_roots'],
     'n_decisive_reactivated':block['decisive_omission_actually_reopened_count']}
 source={q['id']:q for q in lock['src']}
 exe=Path(source['ExternalSolverBinary']['path'])
 book=Path(source['ExternalOpeningBook']['path'])
 assert sha(exe)==source['ExternalSolverBinary']['sha256']
 assert sha(book)==source['ExternalOpeningBook']['sha256']
 checked=0
 for z in exact['rows']:
  if z['id'].endswith('_R00'):
   v=pons(exe,book,z['history'])
   flip=''.join(str(8-int(c)) for c in z['history'])
   reversed_scores=pons(exe,book,flip)
   assert v==z['exact_action_scores'] and reversed_scores==v[::-1]
   checked+=1
 assert checked==24
 report={
  'schema':'quartz_cog03_resident13_C09_independent_Rust_9216_and_external_48_v1',
  'source_commit':m['source_commit'],
  'real_Rust_endpoint_requests_rechecked':9216,
  'same_root_real_MCTS_stage_executions':18432,
  'new_positions':384,'new_opening_family_clusters':24,
  'independent_PascalPons_original_requeries':checked,
  'independent_PascalPons_mirror_requeries':checked,
  'all_original_prior_mask_and_N_W_M2_stat_conservation_verified':True,
  'all_root_tactical_oneply_evidence_independently_rederived_from_board':True,
  'all_oracle_blind_FNV_random_candidate_assignments_rederived':True,
  'all_exact_root_action_WDL_regret_rederived':True,
  'all_24family_bootstrap_conditional_bounds_rederived':True,
  'actual_warm_models_by_mode':{str(k):v for k,v in residency.items()},
  'initial_and_final_disabled_edge_visit_zero_checks':dict(mask_visits),
  'all_original_and_new_data_SHA256':m['output_SHA256'],
  'per_group_per_mode':groups,
  'main_claim':'same-tree activation implemented, does not beat unrestricted root on finite 24family tests',
  'no_human_cognitive_inference':True,
  'no_general_novel_progressive_unpruning_theorem':True
 }
 target=e/'INDEPENDENT_READBACK.json'
 if target.exists():raise FileExistsError('C09 independent evidence already present')
 target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print('C13_C09_INDEPENDENT_9216_REAL_ROOTS_48_SOLVER_PASS',json.dumps({
  'n_Rust':9216,'n_external_requeries':2*checked,
  'informative_root_classes_by_group':dict(root_info),
  'MLP_evidence_vs_full':groups['pooled']['new_mlp']['contrasts']['resident_full24'],
  'Conv_evidence_vs_full':groups['pooled']['conv']['contrasts']['resident_full24']
 },sort_keys=True))

if __name__=='__main__':main()
