#!/usr/bin/env python3
"""Sealed COG03 fixation root dataset, disjoint from all prior Connect4 studies.

Two fresh 12-family opening groups, each 16 descendant board states.
No exact Pascal Pons score or model choice enters this generator.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import sys


FROZEN_SEEDS={'reopen_A':202611210221,'reopen_B':202611210222}
FROZEN_DEPTHS=[9,10,13,14,17,18,21,22]*2


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def game_modules():
    p=Path(__file__).resolve().parent.parent/'connect4_generalization_07'
    if str(p) not in sys.path:sys.path.insert(0,str(p))
    from generate_cohort import (
        canonical_history,canonical_state,roll_to,board_from_history,mirror
    )
    return canonical_history,canonical_state,roll_to,board_from_history,mirror


def prior_root_sources():
    p=Path(__file__).resolve().parent.parent
    return {
        'G07':(p/'connect4_generalization_07/COHORT_1024_FROZEN.json',1024),
        'C06':(p/'connect4_recognizer_06/COHORT_FROZEN.json',192),
        'HIST12':(p/'standard_external_04/POSITIONS_FIXED.json',12),
        'C09':(p/'voc_calibration_09/COHORT_384_FROZEN.json',384),
        'C10':(p/'dynamic_voc_10/COHORT_384_FROZEN.json',384),
        'C11':(p/'dynamic_costmatch_11/COHORT_384_FROZEN.json',384)
    }


def generate_frozen_new_roots():
    canon,canon_state,roll_to,board_from_history,mirror=game_modules()
    prefixes=set();states=set()
    previous={}
    for k,(path,n) in prior_root_sources().items():
        data=json.loads(path.read_text())
        rows=data.get('rows') or data.get('connect4')
        assert len(rows)==n,(k,len(rows),n)
        previous[k]=sha(path)
        for row in rows:
            history=row.get('history',row.get('sequence'))
            prefixes.add(canon(history[:5]))
            states.add(canon_state(row['board']))
    selected=set();seen=set()
    families=[];roots=[]
    for group,seed in FROZEN_SEEDS.items():
        rng=random.Random(seed)
        for idx in range(12):
            family=len(families)
            for attempt in range(17000):
                prefix=roll_to('',5,rng)
                if prefix is None:continue
                key=canon(prefix)
                if key in prefixes or key in selected:continue
                selected.add(key);break
            else:raise RuntimeError('no new mirror-disjoint 5-ply opening family')
            families.append({'family':family,'group':group,
                             'prefix':prefix,'canonical_prefix':key})
            for j,depth in enumerate(FROZEN_DEPTHS):
                for attempt in range(2300):
                    sequence=roll_to(prefix,depth,rng)
                    if sequence is None:continue
                    board,heights=board_from_history(sequence)
                    statekey=canon_state(board)
                    if statekey in states or statekey in seen:continue
                    seen.add(statekey)
                    roots.append({
                        'id':f'C12_{group}_F{idx:02d}_R{j:02d}',
                        'family':family,'group':group,
                        'history':sequence,'mirror_history':mirror(sequence),
                        'board':board,'plies':depth,
                        'current_player':1 if depth%2==0 else -1,
                        'legal_columns':[c for c,h in enumerate(heights) if h<6],
                    })
                    break
                else:raise RuntimeError('could not fill distinct descendant')
    assert len(families)==24 and len(roots)==384
    assert Counter(x['group'] for x in roots)=={'reopen_A':192,'reopen_B':192}
    assert Counter(x['current_player'] for x in roots)=={1:192,-1:192}
    assert len(seen)==384 and len(selected)==24
    return {
        'schema':'quartz_cog03_einstellung12_new_family_cohort_v1',
        'generation_seeds':FROZEN_SEEDS,
        'families':families,'rows':roots,
        'original_prior_cohort_SHA256':previous,
        'prior_all_cohorts_excluded_modulo_horizontal_mirror':True,
        'prior_5ply_mirror_classes_excluded':len(prefixes),
        'prior_board_mirror_classes_excluded':len(states),
        'source_generated_without_oracle_or_policy_tuning':True,
        'new_cohorts_not_independent_MCTS_search_seed_replicates':True,
    }


if __name__=='__main__':
    d=Path(__file__).resolve().parent
    f=d/'COHORT_384_FROZEN.json'
    if f.exists():raise FileExistsError('new game source already sealed')
    original=json.loads((d/'INPUT_LOCK.json').read_text())
    assert original['project_baseline_commit']=='6dbdcaeab28b2325fbd7f0a0326c51b214b339f3'
    assert original['freeze_new_cohort_seeds']==FROZEN_SEEDS
    assert original['new_cohort_exists_before_contract'] is False
    result=generate_frozen_new_roots()
    f.write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    print('COG03_FROZEN_NEW_DATASET',json.dumps({
        'families':len(result['families']),
        'roots':len(result['rows']),
        'sha256':sha(f),
        'groups':dict(Counter(r['group'] for r in result['rows'])),
        'old_prefix_classes_excluded':result['prior_5ply_mirror_classes_excluded'],
    },sort_keys=True))
