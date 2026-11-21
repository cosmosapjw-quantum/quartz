#!/usr/bin/env python3
"""Independent C11 external exact solver result & mirror readback.

Single bounded verification pass: verifies every frozen gravity source and
its 7-column exact WDL label. Independently re-solves 24 opening-family
representative roots and their horizontal mirror, without any model fitting.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replay(history):
    board=[0]*42
    h=[0]*7
    for i,ch in enumerate(history):
        col=int(ch)-1
        if not 0<=col<7 or h[col]>=6:raise RuntimeError('invalid gravity history')
        board[h[col]*7+col]=1 if i%2==0 else -1
        h[col]+=1
    return board,[c for c in range(7) if h[c]<6]


def solver(exe,book,history):
    p=subprocess.run([str(exe),'-a','-b',str(book)],
        input=(history+'\n').encode(),capture_output=True,timeout=12,check=True)
    fields=p.stdout.decode().split()
    if len(fields)!=8 or fields[0]!=history:
        raise RuntimeError('external exact score response invalid')
    return [int(x) for x in fields[1:]]


def main():
    d=Path(__file__).resolve().parent
    out=d/'evidence/exact_new384_20261121'
    manifest=json.loads((out/'EXACT_MANIFEST.json').read_text())
    labels=json.loads((out/'EXACT_NEW384_LABELS.json').read_text())
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    assignments=json.loads((d/'ASSIGNMENTS_FROZEN.json').read_text())
    assert manifest['source_commit']=='d2f6995c5fa9b87b844d66c46f5e79fb19f30904'
    assert manifest['all_384_status_ok']
    assert manifest['n_roots']==384
    assert labels['status']['all_ok'] and labels['status']['counts']=={'OK':384}
    assert len(labels['rows'])==len(cohort['rows'])==384
    assert manifest['exact_labels_sha256']==sha(out/'EXACT_NEW384_LABELS.json')
    assert manifest['new_cohort_sha256']==sha(d/'COHORT_384_FROZEN.json')
    assert manifest['frozen_assignments_sha256']==sha(d/'ASSIGNMENTS_FROZEN.json')
    assert labels['frozen_assignment_sha256']==manifest['frozen_assignments_sha256']
    wdl=Counter()
    informative=Counter()
    seen_families=set()
    for root,source in zip(labels['rows'],cohort['rows']):
        if (root['id']!=source['id'] or root['history']!=source['history']
            or root['group']!=source['group'] or root['family']!=source['family']):
            raise RuntimeError('frozen solver-to-root identity mismatch')
        board,legal=replay(root['history'])
        assert board==source['board']
        assert legal==source['legal_columns']==root['legal_columns']
        score=root['exact_action_scores']
        assert len(score)==7
        assert all((score[c]==-1000)==(c not in legal) for c in range(7))
        sign=lambda x:(x>0)-(x<0)
        possible={sign(score[c]) for c in legal}
        opt=max(possible)
        choices=[c for c in legal if sign(score[c])==opt]
        assert opt==root['root_wdl'] and choices==root['optimal_actions']
        assert abs(sum(root['target_policy'])-1)<1e-9
        assert all(abs(root['target_policy'][c]-(1/len(choices) if c in choices else 0))<1e-9 for c in range(7))
        wdl[(root['group'],opt)]+=1
        if len(possible)>1:informative[root['group']]+=1
        seen_families.add((root['group'],root['family']))
    assert len(seen_families)==24
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    binary=next(Path(it['path']) for it in lock['original_source_sha256'] if it['path'].endswith('/c4solver'))
    book=next(Path(it['path']) for it in lock['original_source_sha256'] if it['path'].endswith('/7x6_small.book'))
    n=0
    for row in labels['rows']:
        if row['id'].endswith('_R00'):
            actual=solver(binary,book,row['history'])
            reflection=''.join(str(8-int(x)) for x in row['history'])
            reflected=solver(binary,book,reflection)
            assert actual==row['exact_action_scores']
            assert reflected==actual[::-1]
            n+=1
    assert n==24
    summary={
        'schema':'quartz_cog02_costmatch11_new384_exact_independent_readback_v1',
        'source_commit':manifest['source_commit'],
        'n_frozen_new_roots':384,'n_new_families':24,
        'original_solver_new_requeries':n,
        'mirror_solver_new_requeries':n,
        'all_384_exact_WDL_root_legal_and_policy_targets_verified':True,
        'new_test_labels_sha256':manifest['exact_labels_sha256'],
        'mixture_assignment_sha256':manifest['frozen_assignments_sha256'],
        'source_new_cohort_sha256':manifest['new_cohort_sha256'],
        'exact_root_WDL_counts':{f'{g}/{w}':count for(g,w),count in sorted(wdl.items())},
        'decision_informative_WDL_counts':dict(informative),
        'test_results_not_used_for_fit':True,
    }
    p=out/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse duplicate oracle audit')
    p.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('C11_NEW384_EXACT_SOLVER_INDEPENDENT_AUDIT_PASS',json.dumps(summary,sort_keys=True))


if __name__=='__main__':
    main()
