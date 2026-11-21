#!/usr/bin/env python3
"""Frozen 15x15 freestyle natural and explicit tactical-stress source positions.

Pure Python rule replay, independently cross-check with actual Rust GameState
in C7. No pretrained Rapfi move or score imported at source creation.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib,json,random

SIZE=15
NEUTRAL_SEED=202611210401
STRESS_SEED=202611210402

def five(board,col,player):
    y,x=divmod(col,SIZE)
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        n=1
        for s in (-1,1):
            yy=y+dy*s;xx=x+dx*s
            while 0<=yy<SIZE and 0<=xx<SIZE and board[yy*SIZE+xx]==player:
                n+=1;yy+=dy*s;xx+=dx*s
        if n>=5:return True
    return False

def replay(history):
    board=[0]*(SIZE*SIZE)
    terminal=False
    for i,idx in enumerate(history):
        if not type(idx)==int or not 0<=idx<225 or board[idx]!=0 or terminal:
            raise ValueError('invalid sequential freestyle15 source')
        player=1 if i%2==0 else -1
        board[idx]=player
        terminal=five(board,idx,player)
    legal=[j for j,v in enumerate(board) if v==0] if not terminal else []
    return {'board':board,'terminal':terminal,
            'side_to_move':1 if len(history)%2==0 else -1,
            'legal':legal}

def ordered_center_moves(legal):
    return sorted(legal,key=lambda p:(max(abs(p//15-7),abs(p%15-7)),
                abs(p//15-7)+abs(p%15-7),p))

def immediate_wins(s):
    b=s['board']
    player=s['side_to_move']
    out=[]
    for j in s['legal']:
        b[j]=player
        if five(b,j,player):out.append(j)
        b[j]=0
    return out

def gen_neutral(i):
    rng=random.Random(NEUTRAL_SEED+i*911)
    x=4+i%7
    y=4+(i*3)%7
    history=[y*15+x]
    for _ in range(10+i%4-1):
        state=replay(history)
        centers=[j for j in range(225) if j in state['legal'] and
            any(max(abs(j//15-q//15),abs(j%15-q%15))<=2 for q in history)]
        rng.shuffle(centers)
        next_move=None
        for c in centers:
            test=history+[c]
            if not replay(test)['terminal']:
                next_move=c;break
        if next_move is None:raise RuntimeError('cannot generate next neutral nonterminal move')
        history.append(next_move)
    return history

def transform(pos,rotation,reflect=False,shift=0):
    x,y=pos
    if reflect:x=14-x
    for _ in range(rotation):x,y=14-y,x
    x+=shift
    if not 0<=x<15 or not 0<=y<15:raise ValueError('stress transform leaves board')
    return y*15+x

def gen_stress(i):
    # The two base scenarios have intentionally forced one-ply terminal wins.
    if i%2==0:
        black=[(1,2),(2,2),(3,2),(4,2)]
        white=[(10,10),(12,11),(10,12),(12,13)]
    else:
        white=[(1,12),(2,12),(3,12),(4,12)]
        black=[(7,4),(9,6),(12,8),(8,11),(11,2)]
    rotation=(i//2)%4
    reflect=i%2==1
    shift=1 if i>=8 else 0
    b=[transform(x,rotation,reflect,shift) for x in black]
    w=[transform(x,rotation,reflect,shift) for x in white]
    history=[]
    for t in range(4):
        history.extend((b[t],w[t]))
    if len(b)==5:history.append(b[4])
    state=replay(history)
    if state['terminal']:raise RuntimeError('stress source already terminal')
    wins=immediate_wins(state)
    if not wins:raise RuntimeError('stress source has no exact immediate win')
    included=set(ordered_center_moves(state['legal'])[:16])
    if not set(wins)-included:
        raise RuntimeError('stress win unexpectedly already inside initial top16')
    return history

def build_cohort():
    rows=[]
    history_set=set()
    for group,seed in (('neutral',NEUTRAL_SEED),('tactical_stress',STRESS_SEED)):
        for i in range(12):
            hist=gen_neutral(i) if group=='neutral' else gen_stress(i)
            key=tuple(hist)
            if key in history_set:raise RuntimeError('duplicate full source history')
            history_set.add(key)
            state=replay(hist)
            assert not state['terminal']
            wins=immediate_wins(state)
            unseen=set(wins)-set(ordered_center_moves(state['legal'])[:16])
            rows.append({
                'id':f'C14_{group}_{i:02d}',
                'stratum':group,'family_index':i,
                'history':hist,'plies':len(hist),
                'side_to_move':state['side_to_move'],
                'legal_moves':len(state['legal']),
                'construction_has_omitted_immediate_win':bool(unseen),
                'constructed_immediate_win_count':len(wins),
                'center_rank_k16_definition':'max_chebyshev_radius,manhattan_radius,flat_index',
            })
    assert len(rows)==24 and len(history_set)==24
    assert Counter(r['stratum'] for r in rows)=={'neutral':12,'tactical_stress':12}
    return {
        'schema':'quartz_cog04_highbranch14_new_freestyle15_stratified_24_v1',
        'board_size':15,'variant':'freestyle','rule_exact_five_or_more':True,
        'source_seed_neutral':NEUTRAL_SEED,'source_seed_stress':STRESS_SEED,
        'stratum_note':'neutral seeded random-neighbor plies vs intentionally preconstructed off-center one-move win; NOT IID populations',
        'source_generated_before_Rapfi_labels':True,
        'nonstrict_independent_opening_families_note':'stress source uses symmetry and shifts of two templates; no independent-family CI claim',
        'rows':rows
    }

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    lock=json.loads((root/'coding/C01_INPUT_LOCK.json').read_text())
    assert lock['source_seed_neutral']==NEUTRAL_SEED
    assert lock['source_seed_stress']==STRESS_SEED
    assert lock['no_new_test_positions_at_lock']
    p=root/'COHORT_24_FROZEN.json'
    if p.exists():raise FileExistsError('do not overwrite precommitted source')
    obj=build_cohort()
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    print('COG04_C04_FROZEN_FREESTYLE_24',digest,
          Counter(r['stratum'] for r in obj['rows']))
