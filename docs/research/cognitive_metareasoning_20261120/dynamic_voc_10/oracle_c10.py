#!/usr/bin/env python3
"""Exact external Pascal Pons WDL for 384 frozen new C10 test positions.

Reuse C09's already tested bounded 5s-per-root solver adapter. New cohort
used ONLY for final scientific evaluation, not model training or lambda.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('exact teacher only from committed clean source')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    for r in lock['origin_files']:
        p=Path(r['path'])
        if p.name in ('mcts_server.rs','cog02.rs'):continue
        if not p.exists() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:
            raise RuntimeError('frozen teacher/model/data source changed: '+str(p))
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    if cohort['schema']!='quartz_cog02_dynamic10_new_families_before_oracle_v1' or len(cohort['rows'])!=384:
        raise RuntimeError('wrong frozen dynamic cohort')
    old=d.parent/'voc_calibration_09'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from oracle_new384 import evaluate_rows
    exe=next(Path(r['path']) for r in lock['origin_files'] if r['path'].endswith('/c4solver'))
    book=next(Path(r['path']) for r in lock['origin_files'] if r['path'].endswith('/7x6_small.book'))
    # Existing source adapter's 'cohort' is a group-label alias only. There
    # are no C09 data/test labels imported to this computation.
    mapped=[{
        'id':x['id'],'family':x['family'],'cohort':x['group'],
        'history':x['history'],'ply':x['plies'],
        'side_to_move':x['current_player'],'legal_columns':x['legal_columns']
    } for x in cohort['rows']]
    rows,elapsed=evaluate_rows(mapped,exe,book,cap_seconds=120)
    for row in rows:
        row['group']=row.pop('cohort')
    counts=dict(Counter(x['status'] for x in rows))
    by_group={group:dict(Counter(x['status'] for x in rows if x['group']==group))
              for group in ('fresh_A','fresh_B')}
    result={
        'schema':'quartz_cog02_dynamic10_heldout_exact_oracle_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'frozen_cohort_sha256':sha(d/'COHORT_384_FROZEN.json'),
        'oracle_binary_sha256':sha(exe),'oracle_book_sha256':sha(book),
        'rows':rows,'elapsed_s':elapsed,'statuses':counts,'by_group':by_group,
        'no_missing_exact_game_labels_imputed':True,
        'training_lambda_or_model_fitting_on_new_labels':False,
    }
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    dest=output/'EXACT_384_LABELS.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    manifest={
        'schema':'quartz_cog02_dynamic10_external_teacher_manifest_v1',
        'source_commit':result['source_commit'],
        'frozen_cohort_sha256':result['frozen_cohort_sha256'],
        'external_binary_sha256':result['oracle_binary_sha256'],
        'external_book_sha256':result['oracle_book_sha256'],
        'external_labels_sha256':sha(dest),
        'all_384_exact':counts=={'OK':384},
        'per_root_timeout_s':5,'global_timeout_s':120,
        'actual_statuses':counts,'by_group':by_group,
    }
    (output/'EXACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('C10_NEW_EXACT_TEACHER',json.dumps({
        'status':counts,'by_group':by_group,
        'elapsed_s':elapsed,'labels_sha256':manifest['external_labels_sha256']
    },sort_keys=True))


if __name__=='__main__':main()
