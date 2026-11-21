#!/usr/bin/env python3
"""Only posthoc complete-game Pascal Pons labels for sealed 384 resident roots.

The source C13 cohort and the selected online root-gate algorithms were
committed before this script runs. Never use these labels for selection or
lambda/threshold fitting, only game-theoretic score readback.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,hashlib,json,subprocess,sys

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def teacher_input(row):
 required=('id','family','group','history','plies','current_player','legal_columns')
 if any(x not in row for x in required):raise ValueError('unfrozen C13 source identity')
 return {'id':row['id'],'family':row['family'],'cohort':row['group'],
         'history':row['history'],'ply':row['plies'],
         'side_to_move':row['current_player'],'legal_columns':row['legal_columns']}

def source_status(rows):
 c=dict(Counter(r['status'] for r in rows))
 return {'counts':c,'all_384_OK':len(rows)==384 and c=={'OK':384},
         'no_missing_label_imputation':True,'n':len(rows)}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out-dir',type=Path,required=True)
 args=ap.parse_args()
 d=Path(__file__).resolve().parent.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
  raise RuntimeError('external solver must use clean committed source')
 cohort=d/'COHORT_384_FROZEN.json'
 lock=d/'coding/INPUT_LOCK.json'
 original=json.loads(lock.read_text())
 sources={q['id']:q for q in original['src']}
 external={
     'ExternalSolverBinary':Path(sources['ExternalSolverBinary']['path']),
     'ExternalOpeningBook':Path(sources['ExternalOpeningBook']['path'])
 }
 for name,path in external.items():
  meta=sources[name]
  if not path.exists() or path.stat().st_size!=meta['bytes'] or sha(path)!=meta['sha256']:
   raise RuntimeError('external complete game source/book identity changed')
 frozen=json.loads(cohort.read_text())
 if frozen['seeds']!=original['new_source_seeds'] or len(frozen['rows'])!=384:
  raise RuntimeError('C13 new families were not frozen as declared')
 prior=d.parent/'voc_calibration_09'
 if str(prior) not in sys.path:sys.path.insert(0,str(prior))
 from oracle_new384 import evaluate_rows
 rows,elapsed=evaluate_rows(
  [teacher_input(row) for row in frozen['rows']],
  external['ExternalSolverBinary'],external['ExternalOpeningBook'],cap_seconds=120)
 for row in rows:row['group']=row.pop('cohort')
 status=source_status(rows)
 result={
  'schema':'quartz_cog03_resident13_exact_posthoc_new384_v1',
  'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
  'frozen_C13_new_cohort_sha256':sha(cohort),
  'input_lock_sha256':sha(lock),
  'external_c4_solver_sha256':sha(external['ExternalSolverBinary']),
  'external_book_sha256':sha(external['ExternalOpeningBook']),
  'status':status,'elapsed_s':elapsed,
  'global_cap_seconds':120,'per_root_cap_seconds':5,
  'exact_labels_not_used_for_online_or_lambda':True,
  'rows':rows,
 }
 out=args.out_dir.resolve()
 out.mkdir(parents=True,exist_ok=False)
 p=out/'EXACT_NEW384_LABELS.json'
 p.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 manifest={
  'schema':'quartz_cog03_resident13_C07_external_teacher_manifest_v1',
  'source_commit':result['source_commit'],
  'C13_cohort_sha256':result['frozen_C13_new_cohort_sha256'],
  'C13_input_lock_sha256':result['input_lock_sha256'],
  'external_solver_sha256':result['external_c4_solver_sha256'],
  'external_book_sha256':result['external_book_sha256'],
  'external_full_WDL_label_sha256':sha(p),
  'n_source_roots':len(rows),'status':status,
  'no_oracle_used_to_fit_or_choose_online_masks':True,
 }
 (out/'EXACT_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 print('C13_C07_EXTERNAL_SOLVER_RESULTS',json.dumps({
    'source_commit':manifest['source_commit'],
    'statuses':status['counts'],
    'elapsed_s':elapsed,
    'exact_output_sha256':sha(p)
 },sort_keys=True))

if __name__=='__main__':main()
