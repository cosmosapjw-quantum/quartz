#!/usr/bin/env python3
"""Freeze a 192-root Connect Four teacher-label cohort without oracle outcomes.

Twelve distinct, mirror-canonical 5-ply *history families*; each has 16
nonterminal descendants. Family split 8 train / 2 validation / 2 test.
The original 12 frozen root histories are permanently excluded, including
their 5-ply prefixes and mirror partners.
"""
import hashlib
import json
import random
from pathlib import Path
import sys

SEED=20261120
LEVELS=[9,10,13,14,17,18,21,22]*2
SPLITS=["train"]*8+["val"]*2+["test"]*2


def mirror(history: str)->str:
    return "".join(str(8-int(x)) for x in history)


def canon(history: str)->str:
    return min(history,mirror(history))


def step(board, heights, sequence, rng):
    sign=1 if len(sequence)%2==0 else -1
    candidates=[]
    for c in range(7):
        if heights[c]>=6:continue
        after=board.copy()
        after[heights[c]*7+c]=sign
        if not has_four(after,sign):
            candidates.append(c)
    if not candidates:return None
    weights=[1+2*(3-abs(3-c)) for c in candidates]
    c=rng.choices(candidates,weights=weights,k=1)[0]
    b=board.copy();b[heights[c]*7+c]=sign
    h=heights.copy();h[c]+=1
    return b,h,sequence+str(c+1)


def has_four(board,sign):
    for y in range(6):
        for x in range(7):
            if board[y*7+x]!=sign:continue
            for dx,dy in ((0,1),(1,0),(1,1),(-1,1)):
                if all(0<=x+i*dx<7 and 0<=y+i*dy<6 and
                       board[(y+i*dy)*7+x+i*dx]==sign for i in range(4)):
                    return True
    return False


def roll_to(history,target,rng):
    board=[0]*42
    height=[0]*7
    sequence=""
    for letter in history:
        c=int(letter)-1
        if height[c]>=6:return None
        sign=1 if len(sequence)%2==0 else -1
        board[height[c]*7+c]=sign
        height[c]+=1
        sequence+=letter
        if has_four(board,sign):return None
    while len(sequence)<target:
        ans=step(board,height,sequence,rng)
        if ans is None:return None
        board,height,sequence=ans
    return sequence


def freeze():
    repo=Path(__file__).resolve().parents[4]
    original=repo/"docs/research/cognitive_metareasoning_20261120/standard_external_04/POSITIONS_FIXED.json"
    frozen=json.loads(original.read_text())["connect4"]
    heldout={row["sequence"] for row in frozen}
    forbidden={canon(s[:5]) for s in heldout}
    families=[]
    entries=[]
    rng=random.Random(SEED)
    seen=set()
    used=set()
    for family in range(12):
        for attempt in range(900):
            prefix=roll_to("",5,rng)
            if prefix is None:continue
            key=canon(prefix)
            if key in forbidden or key in used:
                continue
            used.add(key)
            break
        else:raise RuntimeError("cannot construct distinct mirrored opening family")
        families.append({"family":family,"five_ply_prefix":prefix,"canonical_prefix":key,"split":SPLITS[family]})
        for j,n in enumerate(LEVELS):
            for attempt in range(350):
                hist=roll_to(prefix,n,rng)
                if hist is None or hist in seen or mirror(hist) in seen or hist in heldout or mirror(hist) in heldout:
                    continue
                seen.add(hist)
                board=[0]*42;heights=[0]*7
                for ply,ch in enumerate(hist):
                    c=int(ch)-1;board[heights[c]*7+c]=1 if ply%2==0 else -1;heights[c]+=1
                if has_four(board,1) or has_four(board,-1):continue
                entries.append({
                    "id":f"F{family:02d}_R{j:02d}","family":family,"split":SPLITS[family],
                    "history":hist,"mirror_history":mirror(hist),
                    "plies":n,
                    "current_player":1 if n%2==0 else -1,
                    "board":board,
                    "legal_columns":[c for c in range(7) if heights[c]<6],
                })
                break
            else: raise RuntimeError(f"family {family} could not generate depth {n}")
    assert len(entries)==192 and len({x["history"] for x in entries})==192
    assert len({x["id"] for x in entries})==192
    assert all(canon(x["history"][:5]) not in forbidden for x in entries)
    assert all(x["history"].startswith(families[x["family"]]["five_ply_prefix"]) for x in entries)
    assert set(x["split"] for x in entries)=={"train","val","test"}
    return {"schema":"quartz_connect4_recognizer06_frozen_cohort_v1",
            "seed":SEED,"mirror_augmentation":"train_only_and_same_group",
            "split_by":"5-ply canonical opening family",
            "heldout_12_fixed_root_histories_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
            "families":families,"rows":entries}


if __name__=="__main__":
    output=Path(__file__).with_name("COHORT_FROZEN.json")
    if output.exists():raise RuntimeError("refusing to overwrite frozen cohort")
    data=freeze()
    output.write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print("FROZEN_COHORT",output,"rows",len(data["rows"]),
          "splits",{s:sum(x["split"]==s for x in data["rows"]) for s in ["train","val","test"]},
          "sha256",hashlib.sha256(output.read_bytes()).hexdigest())
