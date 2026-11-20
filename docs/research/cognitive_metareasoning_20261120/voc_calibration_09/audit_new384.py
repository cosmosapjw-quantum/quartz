#!/usr/bin/env python3
"""Independent raw new-family exact-score readback and 24 mirror re-solves.

A separate implementation of root gravity replay and action-WDL checking;
never treats missing teacher results as a draw.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def board_and_legal(history):
    board=[0]*42
    heights=[0]*7
    for i,char in enumerate(history):
        col=int(char)-1
        if not 0<=col<7 or heights[col]>=6:
            raise RuntimeError('incorrect source gravity history')
        board[heights[col]*7+col]=1 if i%2==0 else -1
        heights[col]+=1
    return board,[c for c in range(7) if heights[c]<6]


def mirror(board):
    return tuple(board[row*7+6-col] for row in range(6) for col in range(7))


def oracle(binary,book,history):
    result=subprocess.run([str(binary),'-a','-b',str(book)],
            input=(history+'\n').encode(),capture_output=True,
            timeout=12,check=True)
    split=result.stdout.decode().split()
    if len(split)!=8 or split[0]!=history:raise RuntimeError('exact Pons query malformed')
    return [int(x) for x in split[1:]]


def main():
    root=Path(__file__).resolve().parent
    e=root/'evidence/new384_exact_teacher_20261121'
    m=json.loads((e/'teacher_manifest.json').read_text())
    original=json.loads((root/'COHORT_384_FROZEN.json').read_text())
    labels=json.loads((e/'labels.json').read_text())
    lock=json.loads((root/'INPUT_LOCK.json').read_text())
    assert sha(e/'labels.json')==m['labels_sha256']
    assert sha(root/'COHORT_384_FROZEN.json')==m['cohort_sha256']
    assert labels['source_commit']=='47c7335f8e891e9fe8a2d79c57cc0c31fbabbf11'
    assert len(labels['rows'])==len(original['rows'])==384
    assert labels['statuses']=={'OK':384}
    assert m['count']==384
    old={
        "COG07":Path(lock['input_source_sha256'][2]['path']),
        "COG06":root.parent/'connect4_recognizer_06/COHORT_FROZEN.json',
        "HIST12":root.parent/'standard_external_04/POSITIONS_FIXED.json',
    }
    prior_states=set(); prior_prefixes=set()
    for p in old.values():
        j=json.loads(p.read_text())
        rows=j['rows'] if 'rows' in j else j['connect4']
        for row in rows:
            seq=row['history'] if 'history' in row else row['sequence']
            b=row['board']
            prior_states.add(min(tuple(b),mirror(b)))
            pri=seq[:5]
            reflected=''.join(str(8-int(ch)) for ch in pri)
            prior_prefixes.add(min(pri,reflected))
    seen_states=set();families={};counts=Counter();wdl=Counter()
    for frozen,row in zip(original['rows'],labels['rows']):
        if row['status']!='OK' or row['history']!=frozen['history'] or row['id']!=frozen['id']:
            raise RuntimeError('root/source mismatch')
        if row['cohort']!=frozen['cohort'] or row['family']!=frozen['family']:
            raise RuntimeError('family identity mismatch')
        b,legal=board_and_legal(row['history'])
        assert b==frozen['board'] and legal==row['legal_columns']
        s=min(tuple(b),mirror(b))
        assert s not in prior_states and s not in seen_states
        seen_states.add(s)
        canon=min(row['history'][:5],''.join(str(8-int(ch)) for ch in row['history'][:5]))
        assert canon not in prior_prefixes
        assert row['exact_action_scores'] and len(row['exact_action_scores'])==7
        scores=row['exact_action_scores']
        assert all((scores[col]==-1000)==(col not in legal) for col in range(7))
        u=lambda x:(x>0)-(x<0)
        star=max(u(scores[col]) for col in legal)
        optimal=[col for col in legal if u(scores[col])==star]
        assert optimal==row['optimal_actions'] and star==row['root_wdl']
        assert abs(sum(row['target_policy'])-1)<1e-9
        assert all(abs(row['target_policy'][col]-(1/len(optimal) if col in optimal else 0))<1e-9 for col in range(7))
        families.setdefault(row['family'],set()).add(row['cohort'])
        counts[row['cohort']]+=1
        wdl[(row['cohort'],star)]+=1
    assert counts=={'replica_A':192,'replica_B':192}
    assert len(families)==24 and all(len(s)==1 for s in families.values())
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    checked=0
    for row in labels['rows']:
        if row['id'].endswith('_R00'):
            original_score=oracle(ext/'c4solver',ext/'7x6_small.book',row['history'])
            reflected=''.join(str(8-int(ch)) for ch in row['history'])
            mirror_score=oracle(ext/'c4solver',ext/'7x6_small.book',reflected)
            assert original_score==row['exact_action_scores']
            assert mirror_score==list(reversed(original_score))
            checked+=1
    assert checked==24
    receipt={'schema':'quartz_cog02_calibration09_exact_teacher_readback_v1',
      'full_source_rows':384,
      'two_disjoint_cohorts':{'replica_A':192,'replica_B':192},
      'new_5ply_families':24,
      'prior_state_or_mirror_collisions':0,
      'all_scores_and_targets_verified':384,
      'new_solver_origin_calls':checked,
      'new_solver_mirror_calls':checked,
      'labels_sha256':m['labels_sha256'],
      'WDL_class_counts':{f'{cohort}/{z}':n for (cohort,z),n in sorted(wdl.items())},
      'not_claimed':['iid root sampling','new independent stochastic search seed']
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse duplicate')
    p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('CAL09_NEWFAMILY_EXACT_ORACLE_INDEPENDENT_PASS',json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
