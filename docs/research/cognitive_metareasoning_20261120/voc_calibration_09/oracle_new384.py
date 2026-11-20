#!/usr/bin/env python3
"""Exact Pascal Pons oracle labels for two disjoint frozen new-family cohorts.

Source inputs are locked. The new 384 roots are never used to update
surrogate coefficients, candidate lambda or previous neural weights.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time


def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def teacher_modules():
    directory=Path(__file__).resolve().parent.parent/'connect4_recognizer_06'
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
    from teacher_cpu import query_one, parse_exact_label
    return query_one,parse_exact_label


def verify_inputs(lock):
    for item in lock['input_source_sha256']:
        p=Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            raise RuntimeError('LOCKED_COG02_CAL09_INPUT_CHANGED: '+str(p))


def evaluate_rows(rows,exe,book,cap_seconds=120):
    query_one,parse_exact_label=teacher_modules()
    started=time.perf_counter()
    data=[]
    for index,row in enumerate(rows):
        if time.perf_counter()-started>=cap_seconds:
            query={'status':'GLOBAL_CAP_NOT_ATTEMPTED'}
        else:
            query=query_one(exe,book,row['history'],timeout_s=5.0)
        if query['status']=='OK':
            try:
                target=parse_exact_label(row['history'],query['scores'],row['legal_columns'])
                query.update(target)
            except ValueError as exc:
                query={'status':'BAD_ORACLE_LABEL','error':str(exc)}
        data.append({
            'id':row['id'],'family':row['family'],'cohort':row['cohort'],
            'history':row['history'],'ply':row['ply'],
            'side_to_move':row['side_to_move'],
            'legal_columns':row['legal_columns'],**query})
    return data,time.perf_counter()-started


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('exact teacher requires committed clean source')
    locked=json.loads((d/'INPUT_LOCK.json').read_text())
    verify_inputs(locked)
    root=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    assert root['schema']=='quartz_cog02_voc_calibration09_new_family_cohorts_v1'
    assert len(root['rows'])==384
    pons=next(Path(i['path']) for i in locked['input_source_sha256'] if i['path'].endswith('/c4solver'))
    book=next(Path(i['path']) for i in locked['input_source_sha256'] if i['path'].endswith('/7x6_small.book'))
    rows,elapsed=evaluate_rows(root['rows'],pons,book,cap_seconds=120)
    verify_inputs(locked)
    counts=dict(Counter(x['status'] for x in rows))
    bycohort={name:dict(Counter(x['status'] for x in rows if x['cohort']==name)) for name in ('replica_A','replica_B')}
    result={'schema':'quartz_cog02_voc09_exact_newfamily_teacher_v1',
            'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'cohort_sha256':digest(d/'COHORT_384_FROZEN.json'),
            'teacher_binary_sha256':digest(pons),'teacher_book_sha256':digest(book),
            'rows':rows,'statuses':counts,'cohort_status':bycohort,
            'elapsed_s':elapsed,'no_missing_labels_imputed':True,
            'never_used_for_prior_policy_selection':True}
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    p=output/'labels.json'
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    manifest={'schema':'quartz_cog02_voc09_exact_oracle_manifest_v1',
              'source_commit':result['source_commit'],
              'input_lock_sha256':digest(d/'INPUT_LOCK.json'),
              'cohort_sha256':result['cohort_sha256'],
              'external_c4solver_sha256':digest(pons),'external_book_sha256':digest(book),
              'labels_sha256':digest(p),'count':len(rows),
              'statuses':counts,'bycohort':bycohort,
              'wall_cap_s':120,'per_root_cap_s':5.0}
    (output/'teacher_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('COG02_CAL09_TEACHER_RESULT',json.dumps({
        'cohort_status':bycohort,'statuses':counts,'elapsed_s':elapsed,
        'labels_sha256':manifest['labels_sha256']},sort_keys=True))


if __name__=='__main__':main()
