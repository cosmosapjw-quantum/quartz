#!/usr/bin/env python3
"""Independent new 384-root Pascal Pons full-WDL source and horizontal mirror audit."""
from collections import Counter
from pathlib import Path
import hashlib,json,subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replay(h):
    board=[0]*42;heights=[0]*7
    for i,c in enumerate(h):
        k=int(c)-1
        if not 0<=k<7 or heights[k]>=6:raise ValueError('invalid root gravity')
        board[heights[k]*7+k]=1 if i%2==0 else -1
        heights[k]+=1
    return board,[i for i,n in enumerate(heights) if n<6]


def solve(exe,book,h):
    proc=subprocess.run([str(exe),'-a','-b',str(book)],
        input=(h+'\n').encode(),capture_output=True,timeout=12,check=True)
    pieces=proc.stdout.decode().split()
    assert len(pieces)==8 and pieces[0]==h
    return [int(x) for x in pieces[1:]]


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/new_384_exact_20261121'
    man=json.loads((e/'EXACT_MANIFEST.json').read_text())
    labels=json.loads((e/'EXACT_NEW384_LABELS.json').read_text())
    new=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    assert man['source_commit']=='33b7be4d758a1ced7b6c5733436bd50e5a7d5064'
    assert sha(e/'EXACT_NEW384_LABELS.json')==man['output_labels_sha256']
    assert sha(d/'COHORT_384_FROZEN.json')==man['frozen_cohort_sha256']
    assert man['source_result_count']==len(new['rows'])==len(labels['rows'])==384
    assert labels['status']['all_OK'] and labels['status']['statuses']=={'OK':384}
    informative=Counter();rootWDL=Counter()
    for src,r in zip(new['rows'],labels['rows']):
        assert src['id']==r['id'] and src['history']==r['history']
        assert src['group']==r['group'] and src['family']==r['family']
        board,legal=replay(src['history'])
        assert board==src['board'] and legal==src['legal_columns']==r['legal_columns']
        scores=r['exact_action_scores']
        assert len(scores)==7
        assert all((scores[col]==-1000)==(col not in legal) for col in range(7))
        sign=lambda x:(x>0)-(x<0)
        options={sign(scores[col]) for col in legal}
        ideal=max(options)
        assert ideal==r['root_wdl']
        good=[col for col in legal if sign(scores[col])==ideal]
        assert good==r['optimal_actions']
        assert all(abs(r['target_policy'][col]-
            (1/len(good) if col in good else 0))<1e-9 for col in range(7))
        if len(options)>1:informative[src['group']]+=1
        rootWDL[(src['group'],ideal)]+=1
    sources={p['identity']:p for p in lock['source_entries']}
    binary=Path(sources['ExternalSolver']['path'])
    book=Path(sources['ExternalBook']['path'])
    assert sha(binary)==sources['ExternalSolver']['sha256']
    assert sha(book)==sources['ExternalBook']['sha256']
    checked=0
    for r in labels['rows']:
        if r['id'].endswith('_R00'):
            original=solve(binary,book,r['history'])
            reflection=''.join(str(8-int(ch)) for ch in r['history'])
            mirror=solve(binary,book,reflection)
            assert original==r['exact_action_scores'] and mirror==original[::-1]
            checked+=1
    assert checked==24
    result={
        'schema':'quartz_cog03_reopening12_external_oracle_independent_v1',
        'all_384_full_legal_score_vectors_verified':True,
        'new_opening_families':24,
        'external_exact_solver_additional_original_queries':checked,
        'external_exact_solver_additional_mirror_queries':checked,
        'decision_informative_root_count_by_group':dict(informative),
        'root_WDL_class_counts':{f'{g}/{w}':n for (g,w),n in sorted(rootWDL.items())},
        'input_new_cohort_sha256':man['frozen_cohort_sha256'],
        'external_exact_source_sha256':man['output_labels_sha256'],
        'complete_score_only_not_mcts_policy':True
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('cannot overwrite independent oracle audit')
    p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('COG03_384_ORACLE_AND_48_MIRROR_CROSSCHECK_PASS',json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
