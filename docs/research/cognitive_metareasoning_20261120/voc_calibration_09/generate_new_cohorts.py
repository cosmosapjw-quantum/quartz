#!/usr/bin/env python3
"""Freeze two independent new *cohort draws*, not stochastic-search replicates.

The existing fixed policy and all old train/val/test opening families, plus
older COG06 and historical12, are strictly excluded modulo mirror-prefix.
Exact labels are NOT consulted during generation. 24 families x16 roots;
two 12-family draws use separately seeded RNG streams.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import sys


def root_modules():
    d=Path(__file__).resolve().parent.parent/'connect4_generalization_07'
    sys.path.insert(0,str(d))
    from generate_cohort import (
        canonical_history,canonical_state,
        roll_to,board_from_history,mirror
    )
    return canonical_history,canonical_state,roll_to,board_from_history,mirror


def freeze():
    repo=Path(__file__).resolve().parents[4]
    g07=repo/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07'
    c06=repo/'docs/research/cognitive_metareasoning_20261120/connect4_recognizer_06'
    hist=repo/'docs/research/cognitive_metareasoning_20261120/standard_external_04'
    sources={
        "g07_cohort":g07/'COHORT_1024_FROZEN.json',
        "c06_cohort":c06/'COHORT_FROZEN.json',
        "hist12_cohort":hist/'POSITIONS_FIXED.json'
    }
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    existing_g07=json.loads(sources['g07_cohort'].read_text())['rows']
    existing_c06=json.loads(sources['c06_cohort'].read_text())['rows']
    existing_old=json.loads(sources['hist12_cohort'].read_text())['connect4']
    canon,canon_state,roll_to,board_from_history,mirror=root_modules()
    excluded_prefix=set()
    excluded_state=set()
    for r in existing_g07+existing_c06:
        excluded_prefix.add(canon(r['history'][:5]))
        excluded_state.add(canon_state(r['board']))
    for r in existing_old:
        excluded_prefix.add(canon(r['sequence'][:5]))
        excluded_state.add(canon_state(r['board']))
    assert len(existing_g07)==1024 and len(existing_c06)==192 and len(existing_old)==12
    depths=[9,10,13,14,17,18,21,22]*2
    families=[]; rows=[]
    chosen=set(); states=set()
    for cohort_name,seed in [('replica_A',20261123),('replica_B',20261124)]:
        rng=random.Random(seed)
        for j in range(12):
            family=len(families)
            for attempt in range(15000):
                prefix=roll_to('',5,rng)
                if prefix is None:continue
                key=canon(prefix)
                if key in excluded_prefix or key in chosen:
                    continue
                chosen.add(key);break
            else:raise RuntimeError(f'no new 5ply prefix for {cohort_name}/{j}')
            families.append({'family':family,'cohort':cohort_name,
                             'canonical_five_ply_prefix':key,
                             'history_prefix':prefix})
            for k,depth in enumerate(depths):
                for trial in range(2000):
                    h=roll_to(prefix,depth,rng)
                    if h is None:continue
                    board,heights=board_from_history(h)
                    state=canon_state(board)
                    if state in excluded_state or state in states:continue
                    states.add(state)
                    rows.append({
                        'id':f'C09_{cohort_name}_F{j:02d}_R{k:02d}',
                        'family':family,'cohort':cohort_name,
                        'history':h,'mirror_history':mirror(h),
                        'board':board,'ply':depth,'side_to_move':1 if depth%2==0 else -1,
                        'legal_columns':[i for i,height in enumerate(heights) if height<6]
                    })
                    break
                else:raise RuntimeError(f'cannot fill new family {cohort_name}/{j} depth {depth}')
    assert len(families)==24 and len(rows)==384
    assert len(states)==384
    assert Counter(x['cohort'] for x in rows)=={'replica_A':192,'replica_B':192}
    assert Counter(x['side_to_move'] for x in rows)=={1:192,-1:192}
    return {
        'schema':'quartz_cog02_voc_calibration09_new_family_cohorts_v1',
        'freeze_before_solver_labels':True,'not_search_seed_replication':True,
        'cohort_rng_seeds':{'replica_A':20261123,'replica_B':20261124},
        'source_sha256':{k:sha(p) for k,p in sources.items()},
        'excluded_prior_5ply_mirror_classes':len(excluded_prefix),
        'excluded_prior_state_mirror_classes':len(excluded_state),
        'grouping':'24 mirror-distinct 5-ply families, each 16 descendant roots',
        'study_model_weights_and_lambdas':'frozen in prior COG02 VOC08',
        'families':families,'rows':rows,
    }

if __name__=='__main__':
    d=Path(__file__).resolve().parent
    f=d/'COHORT_384_FROZEN.json'
    if f.exists():raise RuntimeError('refuse overwrite frozen new generalization roots')
    content=freeze()
    f.write_text(json.dumps(content,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    print('VOC09_FROZEN',len(content['families']),'families',len(content['rows']),'roots','sha256',hashlib.sha256(f.read_bytes()).hexdigest())
