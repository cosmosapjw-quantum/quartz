#!/usr/bin/env python3
"""Read-only recomputation of all new Connect4 exact columns and 48 mirror solves.

This is separate from the production teacher wrapper. No output fed into the
Rust engine or root-attention controller.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib,json,subprocess

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def replay(history):
 b=[0]*42
 heights=[0]*7
 for turn,c in enumerate(history):
  col=int(c)-1
  if not 0<=col<7 or heights[col]>=6:raise ValueError('invalid Connect4 gravity turn')
  b[heights[col]*7+col]=1 if turn%2==0 else -1
  heights[col]+=1
 return b,[c for c in range(7) if heights[c]<6]

def pons(exe,book,history):
 proc=subprocess.run([str(exe),'-a','-b',str(book)],
    input=(history+'\n').encode(),capture_output=True,timeout=12,check=True)
 fields=proc.stdout.decode().split()
 if len(fields)!=8 or fields[0]!=history:raise RuntimeError('external solver malformed')
 return [int(x) for x in fields[1:]]

def main():
 d=Path(__file__).resolve().parent.parent
 e=d/'evidence/new384_exact_20261121'
 m=json.loads((e/'EXACT_MANIFEST.json').read_text())
 source=json.loads((d/'COHORT_384_FROZEN.json').read_text())
 targets=json.loads((e/'EXACT_NEW384_LABELS.json').read_text())
 lock=json.loads((d/'coding/INPUT_LOCK.json').read_text())
 assert m['source_commit']=='66ab4ef5f527d5ac266f77e8d7078b2199d20680'
 assert m['status']['all_384_OK'] and m['status']['counts']=={'OK':384}
 assert len(source['rows'])==len(targets['rows'])==m['n_source_roots']==384
 assert sha(e/'EXACT_NEW384_LABELS.json')==m['external_full_WDL_label_sha256']
 assert sha(d/'COHORT_384_FROZEN.json')==m['C13_cohort_sha256']
 informative=Counter()
 wdl=Counter()
 families=set()
 for root,result in zip(source['rows'],targets['rows']):
  assert root['id']==result['id'] and root['history']==result['history']
  assert root['family']==result['family'] and root['group']==result['group']
  board,legal=replay(root['history'])
  assert board==root['board'] and legal==root['legal_columns']==result['legal_columns']
  scores=result['exact_action_scores']
  assert len(scores)==7 and all((scores[c]==-1000)==(c not in legal) for c in range(7))
  sign=lambda x:(x>0)-(x<0)
  options={sign(scores[c]) for c in legal}
  optimal=max(options)
  assert optimal==result['root_wdl']
  candidate=[c for c in legal if sign(scores[c])==optimal]
  assert candidate==result['optimal_actions']
  assert abs(sum(result['target_policy'])-1)<1e-9
  for c in range(7):
   expect=1/len(candidate) if c in candidate else 0
   assert abs(result['target_policy'][c]-expect)<1e-9
  if len(options)>1:informative[root['group']]+=1
  wdl[(root['group'],optimal)]+=1
  families.add(root['family'])
 assert len(families)==24
 original={i['id']:i for i in lock['src']}
 solver=Path(original['ExternalSolverBinary']['path'])
 book=Path(original['ExternalOpeningBook']['path'])
 assert sha(solver)==original['ExternalSolverBinary']['sha256']
 assert sha(book)==original['ExternalOpeningBook']['sha256']
 count=0
 for row in targets['rows']:
  if row['id'].endswith('_R00'):
   own=pons(solver,book,row['history'])
   mirror=''.join(str(8-int(c)) for c in row['history'])
   flipped=pons(solver,book,mirror)
   assert own==row['exact_action_scores'] and flipped==own[::-1]
   count+=1
 assert count==24
 r={
  'schema':'quartz_cog03_resident13_C07_independent_oracle_audit_v1',
  'n_frozen_roots':384,'n_new_mirror_disjoint_families':24,
  'n_re_solved_original':count,'n_re_solved_mirror':count,
  'all_384_independent_legal_and_optimal_actions_verified':True,
  'decision_informative_WDL_by_cohort':dict(informative),
  'root_WDL_counts':{f'{g}/{u}':k for (g,u),k in sorted(wdl.items())},
  'labels_sha256':m['external_full_WDL_label_sha256'],
  'cohort_sha256':m['C13_cohort_sha256'],
  'oracle_only_posthoc_no_online_injection':True
 }
 p=e/'INDEPENDENT_READBACK.json'
 if p.exists():raise FileExistsError('independent oracle record already exists')
 p.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
 print('C13_C07_EXACT_384_PLUS_48_MIRROR_AUDIT_PASS',json.dumps(r,sort_keys=True))

if __name__=='__main__':main()
