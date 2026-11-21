#!/usr/bin/env python3
"""Exact Connect4 standalone external WDL for 384 *already sealed* new C11 test roots.

No updates to the C10 models or the three precommitted cost-randomization
assignments. New labels are accessible only to POSTHOC scoring, never within
the QUARTZ Rust MCTS controller. Explicitly retain timeout/error statuses.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def map_to_external_teacher(row):
    required=('id','family','group','history','plies','current_player','legal_columns')
    if not all(k in row for k in required):
        raise ValueError('frozen C11 root missing source identity')
    return {
        'id':row['id'],'family':row['family'],'cohort':row['group'],
        'history':row['history'],'ply':row['plies'],
        'side_to_move':row['current_player'],
        'legal_columns':row['legal_columns'],
    }


def classify_teacher_status(rows):
    counts=dict(Counter(r['status'] for r in rows))
    all_ok=len(rows)==384 and counts=={'OK':384}
    return {
        'counts':counts,'all_ok':all_ok,
        'usable_in_full_scientific_test':all_ok,
        'missing_score_rows':sum(r['status']!='OK' for r in rows),
        'no_missing_label_imputation':True,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('C11 external exact teacher only from committed clean source')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    planpath=d/'MIXTURE_PLAN_FROZEN.json'
    cohortpath=d/'COHORT_384_FROZEN.json'
    assignedpath=d/'ASSIGNMENTS_FROZEN.json'
    if sha(planpath)!=lock['precommitted_mode_mix_sha256']:
        raise RuntimeError('frozen pretest validation mixture mutated')
    assigns=json.loads(assignedpath.read_text())
    frozen=json.loads(cohortpath.read_text())
    if assigns['new_cohort_sha256']!=sha(cohortpath) or (
        assigns['model_mixture_input_SHA256']!=sha(planpath)):
        raise RuntimeError('pre-oracle actual assignment map was changed')
    if len(frozen['rows'])!=384 or len(frozen['families'])!=24:
        raise RuntimeError('frozen new cohort invalid')
    old=d.parent/'voc_calibration_09'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from oracle_new384 import evaluate_rows
    solver=next(Path(it['path']) for it in lock['original_source_sha256']
                if it['path'].endswith('/c4solver'))
    book=next(Path(it['path']) for it in lock['original_source_sha256']
              if it['path'].endswith('/7x6_small.book'))
    for p in (solver,book):
        source=next(it for it in lock['original_source_sha256'] if it['path']==str(p))
        if p.stat().st_size!=source['bytes'] or sha(p)!=source['sha256']:
            raise RuntimeError('external solver executable or opening book bytes changed')
    mapped=[map_to_external_teacher(r) for r in frozen['rows']]
    scores,elapsed=evaluate_rows(mapped,solver,book,cap_seconds=120)
    for row in scores:
        row['group']=row.pop('cohort')
    status=classify_teacher_status(scores)
    output={
        'schema':'quartz_cog02_dynamic_costmatch11_new384_exact_pascalpons_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'new_cohort_sha256':sha(cohortpath),
        'frozen_costmix_sha256':sha(planpath),
        'frozen_assignment_sha256':sha(assignedpath),
        'exact_solver_sha256':sha(solver),
        'opening_book_sha256':sha(book),
        'status':status,
        'per_root_timeout_seconds':5,
        'total_global_cap_seconds':120,
        'elapsed_s':elapsed,
        'rows':scores,
        'test_oracle_WDL_never_used_for_fit_or_lambda':True,
    }
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    p=target/'EXACT_NEW384_LABELS.json'
    p.write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    receipt={
        'schema':'quartz_cog02_dynamic_costmatch11_exact_oracle_manifest_v1',
        'source_commit':output['source_commit'],
        'new_cohort_sha256':output['new_cohort_sha256'],
        'frozen_mix_sha256':output['frozen_costmix_sha256'],
        'frozen_assignments_sha256':output['frozen_assignment_sha256'],
        'external_solver_sha256':output['exact_solver_sha256'],
        'external_book_sha256':output['opening_book_sha256'],
        'exact_labels_sha256':sha(p),
        'all_384_status_ok':status['all_ok'],
        'status_count':status['counts'],'n_roots':len(scores),
        'not_train_val_or_lambda_input':True,
    }
    (target/'EXACT_MANIFEST.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('C11_EXTERNAL_384_EXACT_TEACHER',json.dumps({
        'n':len(scores),'statuses':status['counts'],
        'wall_s':elapsed,'full_complete':status['all_ok'],
        'label_bytes_sha256':receipt['exact_labels_sha256']},sort_keys=True))

if __name__=='__main__':
    main()
