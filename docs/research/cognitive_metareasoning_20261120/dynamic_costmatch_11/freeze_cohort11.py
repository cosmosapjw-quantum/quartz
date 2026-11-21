#!/usr/bin/env python3
"""Construct two new disjoint opening-family heldout groups; lock assignment before oracle.

Exactly 24 mirror-canonical new 5-ply Connect4 opening families x16 roots, not
old G07/C06/C09/C10/HIST12 states or prefixes. Freeze roots and assignment
identities BEFORE standalone oracle WDL labels or test MCTS are examined.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import sys

from mix_plan11 import balanced_group_assignments,sha


GENERATION_SEEDS={'new_A':202611210111,'new_B':202611210112}
DEPTHS=[9,10,13,14,17,18,21,22]*2


def source_generator():
    p=Path(__file__).resolve().parent.parent/'connect4_generalization_07'
    if str(p) not in sys.path:sys.path.insert(0,str(p))
    from generate_cohort import roll_to,board_from_history,canonical_history,canonical_state,mirror
    return roll_to,board_from_history,canonical_history,canonical_state,mirror


def original_cohorts():
    d=Path(__file__).resolve().parent.parent
    return {
        'G07':(d/'connect4_generalization_07/COHORT_1024_FROZEN.json',1024),
        'C06':(d/'connect4_recognizer_06/COHORT_FROZEN.json',192),
        'HIST12':(d/'standard_external_04/POSITIONS_FIXED.json',12),
        'C09':(d/'voc_calibration_09/COHORT_384_FROZEN.json',384),
        'C10':(d/'dynamic_voc_10/COHORT_384_FROZEN.json',384),
    }


def build_new_cohort():
    roll_to,board_from_history,canon,canon_state,mirror=source_generator()
    excluded_prefix=set()
    excluded_states=set()
    source_h={}
    for name,(path,count) in original_cohorts().items():
        blob=json.loads(path.read_text())
        rows=blob.get('rows') or blob.get('connect4')
        if len(rows)!=count:
            raise RuntimeError('source external cohort count changed')
        source_h[name]=sha(path)
        for row in rows:
            history=row.get('history',row.get('sequence'))
            board=row['board']
            excluded_prefix.add(canon(history[:5]))
            excluded_states.add(canon_state(board))
    roots=[]
    families=[]
    chosen=set()
    states=set()
    for group,seed in GENERATION_SEEDS.items():
        rng=random.Random(seed)
        for j in range(12):
            family=len(families)
            for attempt in range(16000):
                prefix=roll_to('',5,rng)
                if prefix is None:continue
                key=canon(prefix)
                if key in excluded_prefix or key in chosen:continue
                chosen.add(key);break
            else:raise RuntimeError('new 5-ply nonoverlapping family unavailable')
            families.append({'family':family,'group':group,'prefix':prefix,
                             'canonical_prefix':key})
            for k,depth in enumerate(DEPTHS):
                for _ in range(2200):
                    h=roll_to(prefix,depth,rng)
                    if h is None:continue
                    board,heights=board_from_history(h)
                    canonical_board=canon_state(board)
                    if canonical_board in excluded_states or canonical_board in states:
                        continue
                    states.add(canonical_board)
                    roots.append({
                        'id':f'C11_{group}_F{j:02d}_R{k:02d}',
                        'family':family,'group':group,
                        'history':h,'mirror_history':mirror(h),
                        'board':board,'plies':depth,
                        'current_player':1 if depth%2==0 else -1,
                        'legal_columns':[c for c,x in enumerate(heights) if x<6],
                    })
                    break
                else:raise RuntimeError('cannot generate a nonoverlapping 16 descendant root')
    if len(roots)!=384 or len(families)!=24 or len(states)!=384 or len(chosen)!=24:
        raise RuntimeError('wrong test cohort cardinality')
    if Counter(r['group'] for r in roots)!={'new_A':192,'new_B':192}:
        raise RuntimeError('imbalance of root groups')
    if Counter(r['current_player'] for r in roots)!={1:192,-1:192}:
        raise RuntimeError('imbalance of side to move')
    return {
        'schema':'quartz_cog02_dynamic_costmatch11_sealed_new384_v1',
        'generation_seeds':GENERATION_SEEDS,
        'generation_method':'mirror-canonical five-ply families, 16 nonterminal 7x6 gravity descendants, two 12-family groups',
        'all_prior_cohorts_excluded_modulo_mirror':True,
        'excluded_prior_fiveply_classes':len(excluded_prefix),
        'excluded_prior_board_classes':len(excluded_states),
        'source_original_cohort_sha256':source_h,
        'study_mixture_selected_before_new_cohort':True,
        'not_MCTS_stochastic_PRNG_replication':True,
        'families':families,'rows':roots,
    }


def build_locked_assignments(cohort,mixture):
    cases=cohort['rows']
    if len(cases)!=384 or len(cohort['families'])!=24:
        raise ValueError('need frozen two group new384 source')
    if mixture['schema']!='quartz_cog02_dynamic_option_costmatch11_validation_only_random_mixture_v1':
        raise ValueError('mixture had not been sealed on original validation')
    seeds=mixture['assignment_seeds']
    result={}
    for seed in seeds:
        result[str(seed)]=balanced_group_assignments(
            cases,mixture['mode_mixtures'],seed=seed)
        for group in ('new_A','new_B'):
            sub=[r for r in cases if r['group']==group]
            for mode,config in mixture['mode_mixtures'].items():
                assert len(result[str(seed)][mode])==384
                high=sum(result[str(seed)][mode][r['id']]=='higher' for r in sub)
                # A complete block randomized mix: exactly round(192*p)
                # high-cost assignments per cohort, without oracle info.
                assert high==round(192*config['higher_cpu_probability'])
    return {
        'schema':'quartz_cog02_dynamic_costmatch11_preassigned_blocked_randomizations_v1',
        'assignment_seeds':seeds,
        'new_root_count':len(cases),
        'model_mixture_input_SHA256':sha(Path(__file__).with_name('MIXTURE_PLAN_FROZEN.json')),
        'model_selection':'source-only SHA256 ranking of families and roots, never exact WDL and never search future',
        'balanced_group_12family_16positions_each':True,
        'all_seeds_predeclared_before_test_outcomes':True,
        'seed_assignments':result,
    }


def main():
    d=Path(__file__).resolve().parent
    cohort_path=d/'COHORT_384_FROZEN.json'
    mix_path=d/'MIXTURE_PLAN_FROZEN.json'
    assignment_path=d/'ASSIGNMENTS_FROZEN.json'
    if cohort_path.exists() or assignment_path.exists():
        raise FileExistsError('source or assignment cannot be overwritten')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    if lock['new_C11_test_cohort_exists_at_freeze'] or (
        sha(mix_path)!=lock['precommitted_mode_mix_sha256']):
        raise RuntimeError('root generation started before or outside frozen validation mixture')
    mixture=json.loads(mix_path.read_text())
    new=build_new_cohort()
    # Freeze assignment before exact solver and any CPU test is run.
    assignment=build_locked_assignments(new,mixture)
    cohort_path.write_text(json.dumps(new,indent=2,sort_keys=True)+'\n')
    assignment['new_cohort_sha256']=sha(cohort_path)
    assignment_path.write_text(json.dumps(assignment,indent=2,sort_keys=True)+'\n')
    print('C11_NEW_COHORT_AND_ASSIGNMENTS_FROZEN',json.dumps({
        'new_cohort_sha256':sha(cohort_path),
        'assignment_sha256':sha(assignment_path),
        'families':len(new['families']),'roots':len(new['rows']),
        'new_groups':dict(Counter(r['group'] for r in new['rows'])),
        'seeds':mixture['assignment_seeds'],
        'mode_higher_by_group_and_seed':{
            seed:{
                mode:{group:sum(v[r['id']]=='higher' for r in new['rows'] if r['group']==group)
                for group in ('new_A','new_B')}
                for mode,v in mix.items()}
            for seed,mix in assignment['seed_assignments'].items()
        }
    },sort_keys=True))


if __name__=='__main__':
    main()
