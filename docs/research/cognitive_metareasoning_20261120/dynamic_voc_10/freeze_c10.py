#!/usr/bin/env python3
"""Freeze two wholly new 12-opening-family sets before observing oracle labels.

They are independent *draws of opening families* (not independent MCTS PRNG
simulations); 384 roots x two 12-family groups, 16 descendants each.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import sys


def helpers():
    g=Path(__file__).resolve().parent.parent/'connect4_generalization_07'
    if str(g) not in sys.path:sys.path.insert(0,str(g))
    from generate_cohort import (
        roll_to, board_from_history, canonical_history,
        canonical_state, mirror,
    )
    return roll_to,board_from_history,canonical_history,canonical_state,mirror


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    d=Path(__file__).resolve().parent
    parent=d.parent
    sources={
        'G07':parent/'connect4_generalization_07/COHORT_1024_FROZEN.json',
        'C09':parent/'voc_calibration_09/COHORT_384_FROZEN.json',
        'C06':parent/'connect4_recognizer_06/COHORT_FROZEN.json',
        'HIST12':parent/'standard_external_04/POSITIONS_FIXED.json'
    }
    expected={
        'G07':1024,'C09':384,'C06':192,'HIST12':12
    }
    roll_to,board_from_history,canon,canon_state,mirror=helpers()
    excluded_prefix=set()
    excluded_states=set()
    for label,p in sources.items():
        blob=json.loads(p.read_text())
        items=blob.get('rows') or blob.get('connect4')
        assert len(items)==expected[label]
        for row in items:
            h=row.get('history',row.get('sequence'))
            b=row['board']
            excluded_prefix.add(canon(h[:5]))
            excluded_states.add(canon_state(b))
    prefixes=set()
    states=set()
    families=[]
    rows=[]
    depths=[9,10,13,14,17,18,21,22]*2
    for group,seed in [('fresh_A',20261127),('fresh_B',20261128)]:
        rng=random.Random(seed)
        for group_idx in range(12):
            family=len(families)
            for attempt in range(15000):
                prefix=roll_to('',5,rng)
                if prefix is None:continue
                key=canon(prefix)
                if key in excluded_prefix or key in prefixes:continue
                prefixes.add(key);break
            else:raise RuntimeError('Could not find new five-move family')
            families.append({'family':family,'group':group,'prefix':prefix,
                             'canonical_prefix':key})
            for idx,depth in enumerate(depths):
                for trial in range(2000):
                    history=roll_to(prefix,depth,rng)
                    if history is None:continue
                    board,heights=board_from_history(history)
                    state=canon_state(board)
                    if state in excluded_states or state in states:continue
                    states.add(state)
                    rows.append({
                        'id':f'C10_{group}_F{group_idx:02d}_R{idx:02d}',
                        'family':family,'group':group,
                        'history':history,'mirror_history':mirror(history),
                        'board':board,'plies':depth,'current_player':1 if depth%2==0 else -1,
                        'legal_columns':[c for c,h in enumerate(heights) if h<6],
                    })
                    break
                else:raise RuntimeError('Failed to generate distinct descendant state')
    assert len(families)==24 and len(rows)==384
    assert Counter(r['group'] for r in rows)=={'fresh_A':192,'fresh_B':192}
    assert Counter(r['current_player'] for r in rows)=={1:192,-1:192}
    return {
        'schema':'quartz_cog02_dynamic10_new_families_before_oracle_v1',
        'source_inputs_sha256':{label:sha(p) for label,p in sources.items()},
        'training_validation_families_not_reused':True,
        'old_C09_new_families_not_reused_for_tuning':True,
        'not_stochastic_search_seed_replication':True,
        'cohort_seeds':{'fresh_A':20261127,'fresh_B':20261128},
        'families':families,'rows':rows,
    }


if __name__=='__main__':
    d=Path(__file__).resolve().parent
    dest=d/'COHORT_384_FROZEN.json'
    if dest.exists():raise RuntimeError('source cohort already locked')
    content=freeze()
    dest.write_text(json.dumps(content,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print('C10_NEW_FAMILY_FREEZE',len(content['families']),len(content['rows']),sha(dest))
