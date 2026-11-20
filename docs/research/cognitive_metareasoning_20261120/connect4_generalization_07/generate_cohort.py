#!/usr/bin/env python3
"""Freeze COG-07 grouped Connect Four dataset before teacher labels.

All new roots remain strictly nonterminal. Families are disjoint five-ply
histories modulo left/right reflection, with no board-state collisions
(including mirrored state) with previous 192 training or twelve benchmark
roots. Splits are fixed before seeing teacher labels. No model tuning.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import sys

SEED=20261121
N_FAMILIES=64
SPLITS=['train']*40+['val']*12+['test']*12
DEPTHS=[9,10,13,14,17,18,21,22]*2


def mirror(history: str)->str:
    return ''.join(str(8-int(x)) for x in history)


def canonical_history(history: str)->str:
    return min(history,mirror(history))


def has_four(board: list[int],sign: int)->bool:
    for row in range(6):
        for col in range(7):
            if board[row*7+col]!=sign:continue
            for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
                if all(0<=col+k*dx<7 and 0<=row+k*dy<6 and
                       board[(row+k*dy)*7+col+k*dx]==sign for k in range(4)):
                    return True
    return False


def board_from_history(history: str)->tuple[list[int],list[int]]:
    board=[0]*42
    heights=[0]*7
    for ply,digit in enumerate(history):
        if has_four(board,1) or has_four(board,-1):
            raise ValueError("prefix ended game")
        col=int(digit)-1
        if not 0<=col<7 or heights[col]>=6:
            raise ValueError("illegal gravity history")
        board[heights[col]*7+col]=1 if ply%2==0 else -1
        heights[col]+=1
    if has_four(board,1) or has_four(board,-1):
        raise ValueError("source root terminal")
    return board,heights


def mirrored_board(board: list[int])->tuple[int,...]:
    return tuple(board[row*7+6-col] for row in range(6) for col in range(7))


def canonical_state(board: list[int])->tuple[int,...]:
    original=tuple(board)
    return min(original,mirrored_board(board))


def roll_to(prefix: str, depth: int, rng: random.Random)->str|None:
    try:
        board,heights=board_from_history(prefix)
    except ValueError:
        return None
    history=prefix
    while len(history)<depth:
        sign=1 if len(history)%2==0 else -1
        options=[]
        for col in range(7):
            if heights[col]>=6:continue
            b=board.copy();b[heights[col]*7+col]=sign
            if not has_four(b,sign):options.append(col)
        if not options:return None
        weights=[1+2*(3-abs(3-col)) for col in options]
        col=rng.choices(options,weights=weights,k=1)[0]
        board[heights[col]*7+col]=sign
        heights[col]+=1
        history+=str(col+1)
    return history


def load_prior(repo: Path)->tuple[set[str],set[tuple[int,...]],dict]:
    previous=repo/'docs/research/cognitive_metareasoning_20261120/connect4_recognizer_06'
    parent=json.loads((previous/'COHORT_FROZEN.json').read_text())
    historic=repo/'docs/research/cognitive_metareasoning_20261120/standard_external_04/POSITIONS_FIXED.json'
    previous_benchmark=json.loads(historic.read_text())['connect4']
    excluded_prefix={canonical_history(row['history'][:5]) for row in parent['rows']}
    excluded_prefix.update(canonical_history(row['sequence'][:5]) for row in previous_benchmark)
    excluded_states=set()
    for row in parent['rows']:
        excluded_states.add(canonical_state(row['board']))
    for row in previous_benchmark:
        excluded_states.add(canonical_state(row['board']))
    hashes={
        'previous_COG06_cohort_sha256':hashlib.sha256((previous/'COHORT_FROZEN.json').read_bytes()).hexdigest(),
        'historic_12_benchmark_sha256':hashlib.sha256(historic.read_bytes()).hexdigest(),
    }
    return excluded_prefix,excluded_states,hashes


def freeze()->dict:
    repo=Path(__file__).resolve().parents[4]
    excluded_prefix,excluded_states,source_hashes=load_prior(repo)
    rng=random.Random(SEED)
    families=[];rows=[];used_prefix=set();used_states=set()
    for family_id,split in enumerate(SPLITS):
        for attempt in range(15000):
            prefix=roll_to('',5,rng)
            if prefix is None:continue
            key=canonical_history(prefix)
            if key in excluded_prefix or key in used_prefix:continue
            used_prefix.add(key)
            break
        else:raise RuntimeError(f'failed to find new 5-ply family {family_id}')
        families.append({'family':family_id,'split':split,'five_ply_prefix':prefix,
                         'canonical_prefix':key})
        for j,depth in enumerate(DEPTHS):
            for trial in range(1500):
                history=roll_to(prefix,depth,rng)
                if history is None:continue
                board,heights=board_from_history(history)
                signature=canonical_state(board)
                if signature in used_states or signature in excluded_states:continue
                used_states.add(signature)
                rows.append({
                    'id':f'G07_F{family_id:02d}_R{j:02d}', 'family':family_id,'split':split,
                    'history':history,'mirror_history':mirror(history),'board':board,
                    'plies':depth,'current_player':1 if depth%2==0 else -1,
                    'legal_columns':[col for col in range(7) if heights[col]<6],
                })
                break
            else:raise RuntimeError(f'cannot generate diverse root family {family_id} depth {depth}')
    assert len(families)==64 and len(rows)==1024
    assert Counter(row['split'] for row in rows)=={'train':640,'val':192,'test':192}
    assert len(used_states)==1024
    assert Counter(row['current_player'] for row in rows)=={1:512,-1:512}
    assert set(key for key in used_prefix).isdisjoint(excluded_prefix)
    return {
        'schema':'quartz_connect4_generalization07_frozen_v1',
        'seed':SEED,'family_count':N_FAMILIES,'positions_per_family':16,
        'holdout_prior_COG06_and_12_positions':True,
        'sampled_before_teacher_labels':True,
        'no_mirror_equivalent_board_states_in_combined_cohort':True,
        'previous_input_sha256':source_hashes,
        'split_policy':'whole 5-ply prefix group, left-right reflection same family',
        'families':families,'rows':rows,
    }


if __name__=='__main__':
    output=Path(__file__).with_name('COHORT_1024_FROZEN.json')
    if output.exists():raise RuntimeError('refuse overwrite frozen cohort')
    result=freeze()
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({
        'status':'FROZEN','path':str(output),'rows':len(result['rows']),
        'splits':dict(Counter(row['split'] for row in result['rows'])),
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest()
    },sort_keys=True))
