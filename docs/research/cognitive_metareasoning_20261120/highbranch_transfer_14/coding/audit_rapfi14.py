#!/usr/bin/env python3
"""C7 independent Freestyle15 teacher/NNUE validation and tactical rule replay.
No imports from source constructor or model/protocol adaptor.
"""
from collections import Counter
from pathlib import Path
import hashlib,json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def complete_line(board,at,player):
 y,x=divmod(at,15)
 b=board.copy()
 if not 0<=at<225 or b[at]!=0:raise ValueError('invalid target')
 b[at]=player
 for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
  run=1
  for direction in (-1,1):
   xx=x+dx*direction;yy=y+dy*direction
   while 0<=xx<15 and 0<=yy<15 and b[15*yy+xx]==player:
    run+=1;xx+=direction*dx;yy+=direction*dy
  if run>=5:return True
 return False

def board_from(history):
 b=[0]*225
 for i,loc in enumerate(history):
  if not isinstance(loc,int) or not 0<=loc<225 or b[loc]:raise ValueError('bad game history')
  color=1 if i%2==0 else -1
  assert not complete_line(b,loc,color)
  b[loc]=color
 return b,1 if len(history)%2==0 else -1

def main():
 root=Path(__file__).resolve().parent.parent
 target=root/'evidence/rapfi_teacher_20261121'
 source=json.loads((root/'COHORT_24_FROZEN.json').read_text())
 teacher=json.loads((target/'RAPFI_REFERENCE_24.json').read_text())
 manifest=json.loads((target/'EXECUTION_MANIFEST.json').read_text())
 lock=json.loads((root/'coding/C01_INPUT_LOCK.json').read_text())
 assert teacher['source_strata_24_sha256']==manifest['frozen_C14_source_sha256']==sha(root/'COHORT_24_FROZEN.json')
 assert len(source['rows'])==len(teacher['rows'])==24
 assert sha(target/'RAPFI_REFERENCE_24.json')==manifest['result_file_sha256']
 assert teacher['source_commit']=='d449fb9c5a276d0e1318f522abdbf869b1a6ed4b'
 assert teacher['reference_300ms_status_counts']==teacher['stability_600ms_status_counts']=={'OK':24}
 assets={v['name']:v for v in lock['sources']}
 for name in ('Rapfi_executable','Rapfi_config','Rapfi_classical_model','Rapfi_NNUE_freestyle'):
  p=Path(assets[name]['path'])
  assert p.stat().st_size==assets[name]['bytes'] and sha(p)==assets[name]['sha256']
 expected='mix9svqfreestyle_bsmix.bin.lz4'
 answers=Counter();wins=Counter();covered=Counter();stability=Counter()
 for position,res in zip(source['rows'],teacher['rows']):
  assert position['id']==res['id'] and position['stratum']==res['stratum']
  assert position['history']==res['history'] and res['rule']==0
  board,side=board_from(position['history'])
  legal={i for i,x in enumerate(board) if x==0}
  assert len(legal)==225-len(position['history'])
  winning=[a for a in sorted(legal) if complete_line(board,a,side)]
  center=sorted(legal,key=lambda a:(max(abs(a//15-7),abs(a%15-7)),abs(a//15-7)+abs(a%15-7),a))
  if position['stratum']=='tactical_stress':
   assert len(winning)>0 and set(winning).difference(center[:16])
  for budget,key in ((300,'reference_300ms'),(600,'stability_600ms')):
   v=res[key]
   assert v['status']=='OK' and v['compatible_weight_reported'] and v['saw_loaded_nnue']
   assert v['loaded_expected_weight_name']==expected and not v['engine_error_messages']
   assert v['wall_budget_requested_ms']==budget
   a=v['flat_action']
   assert a in legal
   assert v['xy']==[a%15,a//15]
   answers[(position['stratum'],budget)]+=1
   if a in winning:wins[(position['stratum'],budget)]+=1
   if a in center[:16]:covered[(position['stratum'],budget)]+=1
  if res['both_budgets_same_action']:stability[position['stratum']]+=1
 assert all(answers[(s,b)]==12 for s in ('neutral','tactical_stress') for b in (300,600))
 assert sum(stability.values())==23
 record={
  'schema':'quartz_cog04_C07_independently_validated_24_pretrained_Rapfi_v1',
  'new_source_sha256':sha(root/'COHORT_24_FROZEN.json'),
  'teacher_source_commit':teacher['source_commit'],
  'teacher_result_sha256':sha(target/'RAPFI_REFERENCE_24.json'),
  'original_pretrained_binary_SHA256':teacher['Rapfi_binary_sha256'],
  'pretrained_NNUE_SHA256':teacher['Rapfi_trained_NNUE_sha256'],
  'all_48_moves_legal_and_model_loaded':True,
  'valid_reference_calls':{f'{s}/{b}':answers[(s,b)] for s in ('neutral','tactical_stress') for b in (300,600)},
  'moves_that_are_exact_immediate_freestyle_wins':{f'{s}/{b}':wins[(s,b)] for s in ('neutral','tactical_stress') for b in (300,600)},
  'teacher_moves_in_original_center_top16':{f'{s}/{b}':covered[(s,b)] for s in ('neutral','tactical_stress') for b in (300,600)},
  'same_move_across_300_and_600ms_per_stratum':dict(stability),
  'teacher_is_not_exact_full_game_oracle':True,
  'stress_sources_are_constructions_not_IID_random_games':True
 }
 p=target/'INDEPENDENT_READBACK.json'
 if p.exists():raise FileExistsError('teacher audit already frozen')
 p.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
 print('C14_RAPFI_REAL_SCIENCE_TEACHER_AUDIT_PASS',json.dumps(record,sort_keys=True))

if __name__=='__main__': main()
