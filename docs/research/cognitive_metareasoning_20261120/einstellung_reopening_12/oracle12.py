#!/usr/bin/env python3
"""Exact external Pascal Pons WDL for 384 frozen COG03 Connect4 root positions.

Only score results; don't tune the online rule, root priors or trigger threshold.
No oracle result is sent to any actual Rust MCTS request.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,hashlib,json,subprocess,sys


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def teacher_input(r):
    required=('id','family','group','history','plies','current_player','legal_columns')
    if any(k not in r for k in required):raise ValueError('invalid frozen C12 source root')
    return {'id':r['id'],'family':r['family'],'cohort':r['group'],
            'history':r['history'],'ply':r['plies'],
            'side_to_move':r['current_player'],'legal_columns':r['legal_columns']}


def complete_status(rows):
    n=Counter(r['status'] for r in rows)
    return {'all_OK':len(rows)==384 and n=={'OK':384},
            'statuses':dict(n),'total':len(rows),
            'missing_labels_not_imputed':True}


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--out-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('teacher only from committed clean code+data')
    data=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    if len(data['rows'])!=384 or len(data['families'])!=24:
        raise RuntimeError('new fixed root count changed')
    external={x['identity']:x for x in lock['source_entries']}
    binary=Path(external['ExternalSolver']['path'])
    book=Path(external['ExternalBook']['path'])
    for name,path in [('ExternalSolver',binary),('ExternalBook',book)]:
        rec=external[name]
        if not path.is_file() or path.stat().st_size!=rec['bytes'] or sha(path)!=rec['sha256']:
            raise RuntimeError('external exact game solver/book source SHA changed')
    old=d.parent/'voc_calibration_09'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from oracle_new384 import evaluate_rows
    mapping=[teacher_input(r) for r in data['rows']]
    result,elapsed=evaluate_rows(mapping,binary,book,cap_seconds=120)
    for r in result:r['group']=r.pop('cohort')
    status=complete_status(result)
    out={
        'schema':'quartz_cog03_reopening12_exact_new384_pascal_pons_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'new_heldout_cohort_sha256':sha(d/'COHORT_384_FROZEN.json'),
        'original_contract_sha256':sha(d/'INPUT_LOCK.json'),
        'external_solver_sha256':sha(binary),
        'external_book_sha256':sha(book),
        'oracle_results_joined_only_after_live_search':False,
        'oracle_values_excluded_from_search_requests':True,
        'source_used_for_heuristic_threshold_selection':False,
        'per_root_time_limit_s':5,'global_time_limit_s':120,
        'actual_wall_s':elapsed,
        'status':status,'rows':result,
    }
    folder=args.out_dir.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    p=folder/'EXACT_NEW384_LABELS.json'
    p.write_text(json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    (folder/'EXACT_MANIFEST.json').write_text(json.dumps({
        'schema':'quartz_cog03_exact_new384_oracle_receipt_v1',
        'source_commit':out['source_commit'],
        'frozen_cohort_sha256':out['new_heldout_cohort_sha256'],
        'external_solver_sha256':out['external_solver_sha256'],
        'external_book_sha256':out['external_book_sha256'],
        'output_labels_sha256':sha(p),
        'no_missing_labels_imputed':True,
        'statuses':status['statuses'],
        'source_result_count':len(result)
    },sort_keys=True,indent=2)+'\n')
    print('COG03_EXTERNAL_ORACLE_COMPLETED',json.dumps({
        'statuses':status['statuses'],'elapsed_s':elapsed,
        'new_labels_sha256':sha(p)},sort_keys=True))

if __name__=='__main__':main()
