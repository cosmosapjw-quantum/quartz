#!/usr/bin/env python3
"""Freeze two wholly new 12-opening-family Connect4 root cohorts before oracle.

Exclude all preceding G07/C06/C09/C10/C11/C12 and legacy12 family opening
prefixes, canonical gravity boards and their horizontal mirrors.
This file does not open external solver action scores or choose a threshold.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib,json,random,sys

SEEDS={'resident_A':202611210321,'resident_B':202611210322}
DEPTHS=[9,10,13,14,17,18,21,22]*2

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def game_api():
    p=Path(__file__).resolve().parent.parent.parent/'connect4_generalization_07'
    if str(p) not in sys.path:sys.path.insert(0,str(p))
    from generate_cohort import (roll_to,board_from_history,
        canonical_history,canonical_state,mirror)
    return roll_to,board_from_history,canonical_history,canonical_state,mirror

def previous_sources():
    p=Path(__file__).resolve().parent.parent.parent
    return {
        'G07':(p/'connect4_generalization_07/COHORT_1024_FROZEN.json',1024),
        'C06':(p/'connect4_recognizer_06/COHORT_FROZEN.json',192),
        'old12':(p/'standard_external_04/POSITIONS_FIXED.json',12),
        'C09':(p/'voc_calibration_09/COHORT_384_FROZEN.json',384),
        'C10':(p/'dynamic_voc_10/COHORT_384_FROZEN.json',384),
        'C11':(p/'dynamic_costmatch_11/COHORT_384_FROZEN.json',384),
        'C12':(p/'einstellung_reopening_12/COHORT_384_FROZEN.json',384),
    }

def generate_cohort():
    roll_to,board_from,canon,canonical_board,mirror=game_api()
    forbidden_prefix=set()
    forbidden_board=set()
    source_digest={}
    for key,(p,n) in previous_sources().items():
        j=json.loads(p.read_text())
        rows=j.get('rows') or j.get('connect4')
        assert len(rows)==n,(key,n,len(rows))
        source_digest[key]=sha(p)
        for r in rows:
            h=r.get('history',r.get('sequence'))
            forbidden_prefix.add(canon(h[:5]))
            forbidden_board.add(canonical_board(r['board']))
    new_prefix=set();new_board=set()
    families=[];rows=[]
    for group,seed in SEEDS.items():
        rng=random.Random(seed)
        for j in range(12):
            family=len(families)
            for attempt in range(18000):
                p=roll_to('',5,rng)
                if p is None:continue
                key=canon(p)
                if key in forbidden_prefix or key in new_prefix:continue
                new_prefix.add(key)
                break
            else:raise RuntimeError('cannot find disjoint new 5ply opening family')
            families.append({'family':family,'group':group,
                             'prefix':p,'canonical_prefix':key})
            for k,depth in enumerate(DEPTHS):
                for trial in range(2600):
                    h=roll_to(p,depth,rng)
                    if h is None:continue
                    board,heights=board_from(h)
                    val=canonical_board(board)
                    if val in forbidden_board or val in new_board:continue
                    new_board.add(val)
                    rows.append({
                        'id':f'C13_{group}_F{j:02d}_R{k:02d}',
                        'family':family,'group':group,'history':h,
                        'mirror_history':mirror(h),'board':board,
                        'plies':depth,
                        'current_player':1 if depth%2==0 else -1,
                        'legal_columns':[c for c,ht in enumerate(heights) if ht<6]
                    })
                    break
                else:raise RuntimeError('failed 16 distinct nonterminal legal descendants')
    assert len(families)==24 and len(rows)==384
    assert len(new_prefix)==24 and len(new_board)==384
    assert Counter(r['group'] for r in rows)=={'resident_A':192,'resident_B':192}
    assert Counter(r['current_player'] for r in rows)=={1:192,-1:192}
    return {
        'schema':'quartz_cog03_resident13_frozen_24family_new384_v1',
        'seeds':SEEDS,
        'families':families,'rows':rows,
        'previous_C12_and_earlier_mirror_sources_excluded':True,
        'previous_5ply_canonical_prefix_count':len(forbidden_prefix),
        'previous_mirror_canonical_board_count':len(forbidden_board),
        'previous_source_sha256':source_digest,
        'not_independent_MCTS_search_seed_replicates':True,
        'frozen_before_external_exact_oracle':True,
    }

if __name__=='__main__':
    d=Path(__file__).resolve().parent.parent
    locked=json.loads((d/'coding/INPUT_LOCK.json').read_text())
    assert locked['new_source_seeds']==SEEDS and not locked['new_data_exists_at_lock']
    target=d/'COHORT_384_FROZEN.json'
    if target.exists():raise FileExistsError('do not regenerate previously frozen new data')
    c=generate_cohort()
    target.write_text(json.dumps(c,sort_keys=True,indent=2)+'\n')
    print('C13_FROZEN_NEW_24_FAMILIES',
          json.dumps({'cohort_hash':sha(target),'families':len(c['families']),
                     'roots':len(c['rows']),
                     'old_prefix_count':c['previous_5ply_canonical_prefix_count']},sort_keys=True))
