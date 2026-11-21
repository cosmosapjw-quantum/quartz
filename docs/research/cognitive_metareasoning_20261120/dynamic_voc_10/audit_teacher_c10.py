#!/usr/bin/env python3
"""Independent source new-cohort external Pons exact WDL and mirror verification."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replay(history):
    b=[0]*42;h=[0]*7
    for i,s in enumerate(history):
        c=int(s)-1
        if not 0<=c<7 or h[c]>=6:raise RuntimeError('invalid 7x6 gravity history')
        b[h[c]*7+c]=1 if i%2==0 else -1
        h[c]+=1
    return b,[c for c in range(7) if h[c]<6]


def mirrored_board(board):
    return tuple(board[row*7+6-c] for row in range(6) for c in range(7))


def solver(exe,book,h):
    p=subprocess.run([str(exe),'-a','-b',str(book)],
                    input=(h+'\n').encode(),capture_output=True,timeout=12,check=True)
    fields=p.stdout.decode().split()
    if len(fields)!=8 or fields[0]!=h:raise RuntimeError('Pons score vector malformed')
    return [int(x) for x in fields[1:]]


def main():
    d=Path(__file__).resolve().parent
    out=d/'evidence/teacher_new384_20261121'
    m=json.loads((out/'EXACT_MANIFEST.json').read_text())
    labels=json.loads((out/'EXACT_384_LABELS.json').read_text())
    source=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    assert m['all_384_exact'] and m['actual_statuses']=={'OK':384}
    assert m['source_commit']=='3ccb276c4f1c782551094a3e95c1b2d411e4dcd4'
    assert sha(out/'EXACT_384_LABELS.json')==m['external_labels_sha256']
    assert sha(d/'COHORT_384_FROZEN.json')==m['frozen_cohort_sha256']
    assert len(labels['rows'])==len(source['rows'])==384
    used=set();wdl=Counter()
    for case,label in zip(source['rows'],labels['rows']):
        if label['status']!='OK':raise RuntimeError('missing exact solver root')
        assert label['id']==case['id'] and label['group']==case['group']
        assert label['history']==case['history'] and label['family']==case['family']
        board,legal=replay(case['history'])
        assert board==case['board'] and legal==case['legal_columns']==label['legal_columns']
        key=min(tuple(board),mirrored_board(board))
        assert key not in used
        used.add(key)
        S=label['exact_action_scores']
        assert len(S)==7 and all((S[c]==-1000)==(c not in legal) for c in range(7))
        u=lambda x:(x>0)-(x<0)
        optimal_wdl=max(u(S[c]) for c in legal)
        actions=[c for c in legal if u(S[c])==optimal_wdl]
        assert optimal_wdl==label['root_wdl'] and actions==label['optimal_actions']
        assert abs(sum(label['target_policy'])-1)<1e-9
        for c in range(7):
            expect=1/len(actions) if c in actions else 0
            assert abs(label['target_policy'][c]-expect)<1e-9
        wdl[(case['group'],optimal_wdl)]+=1
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    mirror_count=0
    for row in labels['rows']:
        if row['id'].endswith('_R00'):
            h=row['history']
            actual=solver(ext/'c4solver',ext/'7x6_small.book',h)
            flip=''.join(str(8-int(x)) for x in h)
            opposite=solver(ext/'c4solver',ext/'7x6_small.book',flip)
            assert actual==row['exact_action_scores'] and opposite==actual[::-1]
            mirror_count+=1
    assert mirror_count==24
    evidence={
        'schema':'quartz_cog02_dynamic10_new_exact_external_independent_v1',
        'frozen_test_root_count':384,
        'distinct_fresh_test_fiveply_families':24,
        'status_exact':384,
        'full_S7_and_optimal_set_verified':384,
        'external_solver_original_requeries':24,
        'external_solver_mirror_requeries':24,
        'WDL_counts':{f'{g}/{u}':n for (g,u),n in sorted(wdl.items())},
        'external_label_sha256':m['external_labels_sha256'],
        'no_test_ridge_selection':True,
    }
    target=out/'INDEPENDENT_READBACK.json'
    if target.exists():raise FileExistsError('refuse overwrite independent teacher audit')
    target.write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print('C10_NEW384_EXTERNAL_SOLVER_INDEPENDENT_AUDIT_PASS',json.dumps(evidence,sort_keys=True))


if __name__=='__main__':main()
