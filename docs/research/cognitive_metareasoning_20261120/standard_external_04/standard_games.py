"""Independent standard Connect4 and Gomoku15 synthetic-position contracts.

Connect4 uses a 7x6 bottom-up row-major board, 1-based column-history
protocol and gravity; Gomoku15 uses 15x15 top-down row-major board with
no gravity. Do not mix these games with prior 7x7 non-gravity four-in-row.
"""
from __future__ import annotations
import random
import re

C4_W,C4_H=7,6
GOMOKU_N=15


def c4_has_four(board, player):
    if len(board)!=42 or player not in (-1,1):
        raise ValueError("invalid Connect4 board")
    for y in range(6):
        for x in range(7):
            if board[y*7+x]!=player:continue
            for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
                if all(0<=x+i*dx<7 and 0<=y+i*dy<6 and
                       board[(y+i*dy)*7+x+i*dx]==player for i in range(4)):
                    return True
    return False


def c4_from_sequence(sequence):
    if not isinstance(sequence,str):
        raise ValueError("Connect4 sequence must be text")
    board=[0]*42
    height=[0]*7
    for i,char in enumerate(sequence):
        if char not in '1234567':
            raise ValueError("invalid Connect4 column")
        col=int(char)-1
        if height[col]>=6:
            raise ValueError("column overflow")
        if c4_has_four(board,1) or c4_has_four(board,-1):
            raise ValueError("terminal move sequence")
        row=height[col]
        board[row*7+col]=1 if i%2==0 else -1
        height[col]+=1
        if c4_has_four(board,1) or c4_has_four(board,-1):
            raise ValueError("terminal move sequence")
    return board


def c4_action_scores(sequence,line,board):
    items=line.strip().split()
    if len(items)!=8 or items[0]!=sequence:
        raise ValueError("Connect4 solver sequence/count mismatch")
    try:
        scores=[int(k) for k in items[1:]]
    except ValueError as exc:
        raise ValueError("non-integer Connect4 solver score") from exc
    if len(board)!=42 or c4_has_four(board,1) or c4_has_four(board,-1):
        raise ValueError("terminal or invalid board")
    for col,v in enumerate(scores):
        full=board[(6-1)*7+col]!=0
        if (v==-1000)!=full:
            raise ValueError("Connect4 invalid-column score mismatch")
        if v!=-1000 and not (-21<=v<=21):
            raise ValueError("Connect4 solver score outside exact theoretical range")
    return scores


def gomoku_has_five(board, player):
    if len(board)!=225 or player not in (-1,1):
        raise ValueError("invalid 15x15 Gomoku board")
    for y in range(15):
        for x in range(15):
            if board[y*15+x]!=player:continue
            for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
                if all(0<=x+i*dx<15 and 0<=y+i*dy<15 and
                       board[(y+i*dy)*15+x+i*dx]==player for i in range(5)):
                    return True
    return False


def rapfi_parse_move(stdout,board):
    if not isinstance(stdout,str) or "OK" not in stdout:
        raise ValueError("Rapfi start acknowledgement missing")
    if not ("nnue" in stdout.lower() and "weight" in stdout.lower()):
        raise ValueError("Rapfi pretrained NNUE weight load unverified")
    coords=re.findall(r"(?m)^(\d{1,2}),(\d{1,2})\s*$",stdout)
    if not coords:
        raise ValueError("Rapfi did not return best move coordinate")
    x,y=map(int,coords[-1])
    if not (0<=x<15 and 0<=y<15):
        raise ValueError("Rapfi coordinate out of range")
    action=y*15+x
    if board[action]!=0:
        raise ValueError("Rapfi selected illegal occupied square")
    return action


def generate_c4_positions(seed=20261120):
    rng=random.Random(seed)
    targets=[10]*3+[14]*3+[18]*3+[22]*3
    records=[]
    seen=set()
    for i,plies in enumerate(targets):
        for attempt in range(400):
            b=[0]*42
            heights=[0]*7
            seq=[]
            for step in range(plies):
                sign=1 if step%2==0 else -1
                candidates=[]
                for col in range(7):
                    if heights[col]>=6:continue
                    nb=b.copy()
                    nb[heights[col]*7+col]=sign
                    if not c4_has_four(nb,sign):
                        candidates.append(col)
                if not candidates:
                    break
                # Center-weighted but stochastic legal nonterminal trajectories.
                weights=[1+3*(3-abs(3-col)) for col in candidates]
                col=rng.choices(candidates,weights=weights,k=1)[0]
                b[heights[col]*7+col]=sign
                heights[col]+=1
                seq.append(str(col+1))
            if len(seq)==plies:
                code=''.join(seq)
                if code not in seen:
                    seen.add(code)
                    assert c4_from_sequence(code)==b
                    records.append({
                        'id':f'C4_{i:03d}',
                        'sequence':code,'board':b,'plies':plies,
                        'side_to_move':1 if plies%2==0 else -1,
                        'gravity':True,'connect':4
                    })
                    break
        else:
            raise RuntimeError(f"could not construct legal nonterminal Connect4 at {plies} ply")
    return records


def generate_gomoku_positions(seed=20261120):
    rng=random.Random(seed+1000003)
    records=[]
    seen=set()
    for i,plies in enumerate([4]*3+[6]*3+[8]*3+[10]*3):
        for attempt in range(100):
            board=[0]*225
            history=[]
            for step in range(plies):
                choices=[(x,y) for y in range(4,11) for x in range(4,11)
                         if board[y*15+x]==0]
                x,y=rng.choice(choices)
                board[y*15+x]=1 if step%2==0 else -1
                history.append([x,y,1 if step%2==0 else 2])
            signature=tuple(board)
            if signature in seen or gomoku_has_five(board,1) or gomoku_has_five(board,-1):
                continue
            seen.add(signature)
            records.append({
                'id':f'G15_{i:03d}','board':board,'history':history,
                'plies':plies,'player':1 if plies%2==0 else -1,
                'size':15,'connect':5,'rules':'freestyle'
            })
            break
        else:raise RuntimeError(f"could not generate 15x15 at {plies} plies")
    return records
