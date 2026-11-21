#!/usr/bin/env python3
"""Freeze NEW grouped 15x15 Freestyle source positions before any Rapfi labels."""
from pathlib import Path
import hashlib,json,random,sys

N=15
DIRS=((1,0),(0,1),(1,1),(1,-1))
BASE=202611210515
SIZES=(("train",128),("validation",32),("test_natural",48),("test_tactical",16))

def win(board,side,a):
    x,y=a%N,a//N
    if board[a]:raise ValueError("candidate occupied")
    for dx,dy in DIRS:
        count=1
        for sign in (-1,1):
            xx,yy=x+sign*dx,y+sign*dy
            while 0<=xx<N and 0<=yy<N and board[yy*N+xx]==side:
                count+=1;xx+=sign*dx;yy+=sign*dy
        if count>=5:return True
    return False

def replay(hist):
    b=[0]*(N*N)
    for i,a in enumerate(hist):
        if type(a)!=int or not 0<=a<N*N or b[a] or win(b,1 if i%2==0 else 2,a):
            raise ValueError("terminal, illegal or duplicate source history")
        b[a]=1 if i%2==0 else 2
    return b

def natural(seed):
    rng=random.Random(seed)
    board=[0]*225
    hist=[]
    n=rng.randrange(12,29)
    for ply in range(n):
        side=1 if ply%2==0 else 2
        candidates=[]
        for a in range(225):
            if board[a]:continue
            x,y=a%15,a//15
            if not hist:
                if max(abs(x-7),abs(y-7))>2:continue
            elif not any(max(abs(x-(h%15)),abs(y-(h//15)))<=2 for h in hist[-12:]):
                if rng.random()>0.01:continue
            if not win(board,side,a):candidates.append(a)
        if not candidates:
            break
        a=rng.choice(candidates)
        board[a]=side
        hist.append(a)
    if len(hist)<12:raise RuntimeError("short natural source")
    replay(hist)
    return hist

def diagnostic(seed):
    rng=random.Random(seed)
    y=rng.choice([0,1,2,12,13,14])
    x=rng.choice([0,1,2,8,9,10])
    if x+4>=15: x=9
    blacks=[y*15+x+i for i in range(4)]
    # Filler white stones far from the planted four, and checked for prior terminality.
    candidates=[a for a in range(225)
        if (a//15)>=5 and (a//15)<=9 and (a%15)>=5 and (a%15)<=14
        and a not in blacks]
    rng.shuffle(candidates)
    hist=[]
    used=set()
    for black in blacks:
        hist.append(black);used.add(black)
        selected=None
        for a in candidates:
            if a in used:continue
            b=replay(hist)
            if not win(b,2,a):
                selected=a;break
        if selected is None:raise RuntimeError("filler failure")
        hist.append(selected);used.add(selected)
    board=replay(hist)
    assert win(board,1,y*15+x+4)
    return hist

def sym_coords(x,y,rotation,mirror):
    if mirror:x=14-x
    for _ in range(rotation): x,y=14-y,x
    return x,y

def board_signature(h,rotation,mirror):
    b=replay(h)
    return tuple(b[y*15+x] for y in range(15) for x in range(15)) if rotation==0 and not mirror else tuple(
        b[sy*15+sx] for sy,sx in [] ) if False else canonical_mapped(b,rotation,mirror)

def canonical_mapped(b,rotation,mirror):
    out=[0]*225
    for a,v in enumerate(b):
        x,y=sym_coords(a%15,a//15,rotation,mirror)
        out[y*15+x]=v
    return tuple(out)

def canon(h):
    b=replay(h)
    return min(canonical_mapped(b,rot,mirror) for rot in range(4) for mirror in (False,True))

def main():
    home=Path(__file__).resolve().parent.parent
    c14=home.parent/'highbranch_transfer_14/COHORT_24_FROZEN.json'
    original=json.loads(c14.read_text())
    excluded={canon(r['history']) for r in original['rows']}
    seen=set(excluded)
    splits={}
    group_index=0
    for split,n in SIZES:
        rows=[]
        attempts=0
        while len(rows)<n and attempts<n*100:
            attempts+=1
            seed=BASE+100000*group_index+attempts*9973
            h=diagnostic(seed) if split=="test_tactical" else natural(seed)
            c=canon(h)
            if c in seen:continue
            seen.add(c)
            rows.append({"id":f"{split}-{len(rows):03d}","source_seed":seed,
                         "source_group_id":f"one_source_trajectory_{seed}",
                         "history":h,"n_ply":len(h),
                         "stratum":"constructed_terminal1" if split=="test_tactical" else "seeded_synthetic_neighbor_growth",
                         "has_legal_immediate_win":any(win(replay(h),1 if len(h)%2==0 else 2,a)
                               for a in range(225) if a not in h)})
        assert len(rows)==n,(split,len(rows))
        splits[split]=rows
        group_index+=1
    result={'schema':'quartz_cog05_frozen_separate_source_groups_v1',
            'generation_code':'coding/freeze_cohorts.py',
            'base_seed':BASE,
            'rule':'15x15 Freestyle',
            'not_random_population_sample':True,
            'C14_previous_heldout_sha256':hashlib.sha256(c14.read_bytes()).hexdigest(),
            'D4_canonical_board_group_disjoint':True,
            'no_teacher_labels_or_test_fit':True,
            'split_counts':{k:len(v) for k,v in splits.items()},
            'splits':splits}
    out=home/'COHORT_224_FROZEN.json'
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    manifest={'source_file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'C14_excluded_board_count':len(excluded),
              'total_distinct_frozen_groups':len(seen)-len(excluded),
              'source_file':out.name,'train_labels_collected':False,
              'test_labels_collected':False}
    (home/'C04_FREEZE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(manifest,sort_keys=True))

if __name__=='__main__':main()
