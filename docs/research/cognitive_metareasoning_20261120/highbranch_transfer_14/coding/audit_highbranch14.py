#!/usr/bin/env python3
"""Independently rederive 192 actual Rust 225-cell root masks, legal tactics,
pretrained Rapfi finite action references, and stratified physical CPU costs.

No imports from highbranch14 trial runner, source generator, Rapfi adapter,
or production Rust game. A separate finite implementation is used.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib,json,math

ARMS=('closed','evidence','random_1','random_2','random_3',
      'all_after8','full32','closed_no_probe')
SEEDS=(202611210411,202611210412,202611210413)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def makes_five(board,idx,side):
 y,x=divmod(idx,15)
 if not 0<=idx<225 or board[idx]:raise ValueError('illegal new action')
 for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
  k=1
  for t in (-1,1):
   xx=x+t*dx;yy=y+t*dy
   while 0<=xx<15 and 0<=yy<15 and board[15*yy+xx]==side:
    k+=1;xx+=t*dx;yy+=t*dy
  if k>=5:return True
 return False

def from_moves(history):
 b=[0]*225
 for i,a in enumerate(history):
  assert type(a)==int and 0<=a<225 and b[a]==0
  side=1 if i%2==0 else -1
  assert not makes_five(b,a,side)
  b[a]=side
 return b,(1 if len(history)%2==0 else -1)

def opponent_immediately_wins(board,side,loc):
 if makes_five(board,loc,side):return False
 after=board.copy();after[loc]=side
 return any(makes_five(after,x,-side) for x in range(225) if after[x]==0)

def fnv_rank(omitted,count,seed,history):
 def score(a):
  h=0xcbf29ce484222325 ^ seed
  for byte in [v & 255 for v in history]+[a & 255]:
   h^=byte
   h=(h*0x1000000001b3) & ((1<<64)-1)
  return h
 return sorted(omitted,key=score)[:count]

def top_center(legal):
 return sorted(legal,key=lambda a:(
       max(abs(a//15-7),abs(a%15-7)),
       abs(a//15-7)+abs(a%15-7),a))

def encode_moves(items):
 words=[0]*4
 for a in items:
  assert 0<=a<225
  words[a//64]|=1<<(a%64)
 return words

def audit_tactic(board,side,rank,first,pi):
 initial=rank[:16]
 omitted=rank[16:]
 wins=[a for a in omitted if makes_five(board,a,side)]
 if wins:return {'reason':'omitted_legal_immediate_five','proposed':wins[:4],
                  'wins':wins,'unsafe':False}
 leader=first if first in initial else initial[0]
 unsafe=opponent_immediately_wins(board,side,leader)
 share=max(x[1] for x in pi)
 if share>=.5 and unsafe:
  proposed=[]
  for a in omitted:
   if len(proposed)==4:break
   if not opponent_immediately_wins(board,side,a):proposed.append(a)
  reason='oneply_safe_after_fixation' if proposed else 'no_safe_omitted_action'
 else:
  proposed=[];reason='no_tactical_witness'
 return {'reason':reason,'proposed':proposed,'wins':wins,'unsafe':unsafe}

def sample_score(game,side,chosen,teacher_300,teacher_600,winning):
 assert 0<=chosen<225 and game[chosen]==0
 return {'terminal_win':int(chosen in winning),
         'agrees_teacher_300':int(chosen==teacher_300),
         'agrees_teacher_600':int(chosen==teacher_600)}

def main():
 root=Path(__file__).resolve().parent.parent
 ev=root/'evidence/live_192_cpu_20261121'
 manifest=json.loads((ev/'EXECUTION_MANIFEST.json').read_text())
 runs=json.loads((ev/'RAW_192_REAL_RUST_SEARCHES.json').read_text())
 paired=json.loads((ev/'PAIRED_24_COMPARISON.json').read_text())
 summary=json.loads((ev/'SUMMARY_REAL_15X15.json').read_text())
 roots=json.loads((root/'COHORT_24_FROZEN.json').read_text())['rows']
 teachers=json.loads((root/'evidence/rapfi_teacher_20261121/RAPFI_REFERENCE_24.json').read_text())['rows']
 lock=json.loads((root/'coding/C01_INPUT_LOCK.json').read_text())
 assert manifest['source_commit']=='03c85329844adcb953c45ebcfd1f064c0a248cd7'
 assert len(runs)==manifest['n_Rust_requests']==192
 assert manifest['n_same_root_stage_executions']==384
 assert len(roots)==len(paired)==len(teachers)==24
 assert digest(root/'COHORT_24_FROZEN.json')==manifest['C14_source_sha256']
 assert digest(root/'evidence/rapfi_teacher_20261121/RAPFI_REFERENCE_24.json')==manifest['Rapfi_24_real_pretrained_results_sha256']
 assert digest(Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo'))==manifest['Rust_optimized_binary_sha256']
 for name,sha in manifest['output_sha256'].items():
  assert digest(ev/name)==sha,name
 source_assets={x['name']:x for x in lock['sources']}
 for name in ('Rapfi_executable','Rapfi_config','Rapfi_classical_model','Rapfi_NNUE_freestyle'):
  p=Path(source_assets[name]['path'])
  assert digest(p)==source_assets[name]['sha256']
 meta=Counter()
 independent={}
 for i,(spec,teacher,joined) in enumerate(zip(roots,teachers,paired)):
  assert spec['id']==teacher['id']==joined['id'] and joined['history']==spec['history']
  assert spec['stratum']==teacher['stratum']==joined['stratum']
  board,side=from_moves(spec['history'])
  legal=[a for a in range(225) if board[a]==0]
  rank=top_center(legal)
  true_wins=[a for a in legal if makes_five(board,a,side)]
  assert joined['exact_legal_immediate_wins']==true_wins
  assert joined['side_to_move']==side
  for key in ('reference_300ms','stability_600ms'):
   t=teacher[key]
   assert t['status']=='OK' and t['compatible_weight_reported']
   assert t['flat_action'] in legal
   assert t['xy']==[t['flat_action']%15,t['flat_action']//15]
  t300=teacher['reference_300ms']['flat_action']
  t600=teacher['stability_600ms']['flat_action']
  assert joined['precommitted_300ms_teacher_best']==t300
  assert joined['secondary_600ms_teacher_best']==t600
  assert joined['teacher_300_600_agreement']==(t300==t600)
  if spec['stratum']=='tactical_stress':
   assert true_wins and not set(true_wins).intersection(rank[:16])
  for ai,arm in enumerate(ARMS):
   raw=runs[i*len(ARMS)+ai]
   score=joined[arm]
   if 'error' in raw:raise RuntimeError('actual Rust per-position failure')
   assert raw['schema']=='quartz_cog04_highbranch14_real_same_tree_freestyle_v1'
   assert raw['rule']=='freestyle' and raw['history']==spec['history']
   assert raw['side_to_move']==side
   assert raw['root_full_original_legal_count']==len(legal)
   assert raw['legal_actions']==legal
   assert raw['prior_center_rank']==rank
   assert raw['initial_candidates']==rank[:16]
   assert raw['source_prior_type']=='UniformEval_with_separate_center_rank_gate'
   assert raw['no_Rapfi_teacher_data_online'] and raw['original_complete_root_edges_preallocated']
   assert raw['root_full_not_late_edge_insertion']
   assert raw['same_root_identity_verified'] and raw['edge_stats_preserved_at_activation']
   assert raw['root_prior_preserved'] and raw['no_root_evaluator_call_on_mask_change']
   assert raw['root_hash_before']==raw['root_hash_after']
   first=raw['first_stage'];second=raw['second_stage']
   assert first['root_visits']==8 and second['root_visits']==32
   assert second['additional_root_visits']==24
   assert first['edge_visit_sum']==8 and second['edge_visit_sum']==32
   assert first['edge_stats']==raw['activation_edge_stats']
   assert len(first['edge_stats'])==len(second['edge_stats'])==len(legal)
   assert sum(x[1] for x in first['edge_stats'])==8
   assert sum(x[1] for x in second['edge_stats'])==32
   assert set(x[0] for x in first['edge_stats'])==set(legal)
   assert all(x[0]==y[0] and x[2]==y[2] and x[1]<=y[1]
              for x,y in zip(first['edge_stats'],second['edge_stats']))
   assert raw['first_stage']['best_action'] in raw['initial_active_actions']
   assert raw['final_legal_action'] in raw['final_active_actions']
   assert raw['first_stage']['best_action'] in legal and raw['final_legal_action'] in legal
   for e in first['edge_stats']:
    if arm!='full32' and e[0] not in rank[:16]:
     assert e[1]==0
   for e in second['edge_stats']:
    if e[0] not in raw['final_active_actions']:
     assert e[1]==0
   if arm=='full32':
    assert raw['initial_active_actions']==legal
    assert raw['root_mask_initial']==[0,0,0,0]
    assert raw['root_mask_final']==[0,0,0,0]
   else:
    assert raw['initial_active_actions']==rank[:16]
    assert raw['root_mask_initial']==encode_moves(rank[:16])
    assert raw['root_mask_final']==encode_moves(raw['final_active_actions'])
    assert all((x&y)==x for x,y in zip(raw['root_mask_initial'],raw['root_mask_final']))
   probe=raw['tactical_probe']
   if arm=='closed_no_probe':
    assert probe['reason']=='no_probe' and not probe['suggested_reopen']
    additions=[]
   else:
    pred=audit_tactic(board,side,rank,first['best_action'],first['root_policy'])
    assert pred['reason']==probe['reason'],(spec['id'],arm,pred,probe)
    assert pred['wins']==probe['omitted_legal_immediate_wins']
    assert pred['proposed']==probe['suggested_reopen']
    assert pred['unsafe']==probe['initial_best_vulnerable_to_oneply_reply']
    if arm=='evidence':additions=pred['proposed']
    elif arm.startswith('random_'):
     seed=SEEDS[int(arm.rsplit('_',1)[1])-1]
     additions=fnv_rank(rank[16:],len(pred['proposed']),seed,spec['history'])
    elif arm=='all_after8':additions=rank[16:]
    else:additions=[]
   assert raw['added_candidates']==additions,(spec['id'],arm)
   wanted=legal if arm=='full32' else rank[:16]+additions
   assert raw['final_active_actions']==wanted
   calc=sample_score(board,side,raw['final_legal_action'],t300,t600,true_wins)
   assert score['is_immediate_win']==bool(calc['terminal_win'])
   assert score['teacher_agree']==bool(calc['agrees_teacher_300'])
   assert score['teacher_secondary_agree']==bool(calc['agrees_teacher_600'])
   assert score['action']==raw['final_legal_action']
   assert score['added_count']==len(additions)
   assert score['reopened_immediate_win']==bool(set(additions)&set(true_wins))
   assert score['cost_us']==raw['total_wall_us_without_output_serialization']
   assert score['pure_search_us']==raw['cost']['pure_MCTS_search_wall_us']
   assert score['chosen_move_prior_center_rank']==rank.index(raw['final_legal_action'])+1
   assert score['cost_us']>=score['pure_search_us']>0
   meta[(spec['stratum'],arm,'win')]+=calc['terminal_win']
   meta[(spec['stratum'],arm,'teacher')]+=calc['agrees_teacher_300']
   meta[(spec['stratum'],arm,'cost_us')]+=score['cost_us']
 summary_recomputed={}
 for group in ('neutral','tactical_stress','pooled'):
  subset=[j for j in paired if group=='pooled' or j['stratum']==group]
  assert len(subset)==(24 if group=='pooled' else 12)
  group_out=summary['by_stratum'][group]
  new={}
  for arm in ARMS:
   win=sum(j[arm]['is_immediate_win'] for j in subset)
   teacher_agree=sum(j[arm]['teacher_agree'] for j in subset)
   cpu=sum(j[arm]['cost_us'] for j in subset)/len(subset)/1000
   assert win==group_out['arms'][arm]['immediate_terminal_win_selected_count']
   assert teacher_agree==group_out['arms'][arm]['teacher_300ms_agree_count']
   assert abs(cpu-group_out['arms'][arm]['mean_end_to_end_wall_ms'])<1e-12
   new[arm]={'exact_immediate_win':win,'Rapfi_300ms_agreement':teacher_agree,
             'end_to_end_CPU_ms':cpu}
  a=new['evidence']['end_to_end_CPU_ms'];b=new['full32']['end_to_end_CPU_ms']
  mismatch=abs(a-b)/a
  source_gate=group_out['paired']['evidence_minus_full32']['cpu_10pct_match']
  assert source_gate==('MATCHED_10_PERCENT' if mismatch<=.1 else 'NOT_MATCHED_10_PERCENT')
  summary_recomputed[group]={'arms':new,'evidence_to_full_CPU_cost_mismatch':mismatch,
                            'cost_gate':source_gate}
 assert summary_recomputed['neutral']['cost_gate']=='NOT_MATCHED_10_PERCENT'
 assert summary_recomputed['tactical_stress']['cost_gate']=='NOT_MATCHED_10_PERCENT'
 assert summary_recomputed['pooled']['cost_gate']=='MATCHED_10_PERCENT'
 assert summary_recomputed['tactical_stress']['arms']['evidence']['exact_immediate_win']==12
 assert summary_recomputed['tactical_stress']['arms']['full32']['exact_immediate_win']==6
 audit={
  'schema':'quartz_cog04_highbranch14_C09_independent_actual_192_cpu_teacher_tactics_v1',
  'Rust_executable_source_commit':manifest['Rust_optimized_source_commit'],
  'run_source_commit':manifest['source_commit'],
  'real_Rust_calls':len(runs),
  'real_completed_same_root_search_stage_calls':len(runs)*2,
  'real_Rapfi_pretrained_reference_calls':len(teachers)*2,
  'new_freestyle_positions':24,
  'stress_templates_reused_not_IID':True,
  'all_original_225_cell_source_legal_replayed':True,
  'all_immediate_wins_rederived_without_generation_module':True,
  'all_rule_only_tactical_omissions_and_FNV_random_seeds_rederived':True,
  'all_root_mask_four_words_and_edge_visits_rederived':True,
  'all_final_Rapfi_teacher_agreement_independently_rechecked':True,
  'all_measured_CPU_wall_mean_and_cost_gates_independently_rechecked':True,
  'same_root_stats_preserved_verifiable_from_actual_Rust_snapshots':True,
  'cost_pooled_gate_is_misleading_due_to_stratum_cancellation':True,
  'correct_primary_cost_admission':'NOT_MATCHED_IN_EITHER_12_POSITION_STRATUM',
  'Rapfi_is_not_exact_minimax_or_native_QUARTZ_NNUE':True,
  'result_summary':summary_recomputed,
  'input_file_hashes':manifest['output_sha256'],
  'excluded_claims':['human Einstellung','game Elo','strong general algorithm improvement','exact minimax score']
 }
 target=ev/'INDEPENDENT_READBACK.json'
 if target.exists():raise FileExistsError('independent audit already recorded')
 target.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
 print('C14_C09_INDEPENDENT_192_ROOTS_RAPFI_48_TACTIC_MASK_CPU_AUDIT_PASS',
       json.dumps({'stress_evidence_wins':12,'stress_all_legal_wins':6,
         'per_stratum_cost_gate':[summary_recomputed[g]['cost_gate'] for g in ('neutral','tactical_stress')],
         'pooled_gate':summary_recomputed['pooled']['cost_gate']},sort_keys=True))
if __name__=='__main__':main()
