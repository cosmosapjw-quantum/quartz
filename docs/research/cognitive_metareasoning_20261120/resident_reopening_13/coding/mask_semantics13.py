#!/usr/bin/env python3
"""Finite exact seven-column bitmask representation check; no Rust runtime claims."""
from itertools import combinations
import json
from pathlib import Path
import hashlib

def mask(cols):
    if not cols or len(cols)!=len(set(cols)) or any(type(c)!=int or c<0 or c>=7 for c in cols):
        raise ValueError("invalid nonempty seven-column action subset")
    return sum(1<<c for c in cols)

def derive():
    allmoves=set(range(7))
    masks={}
    for count in range(1,8):
        for xs in combinations(range(7),count):
            bits=mask(xs)
            assert bits not in masks
            assert {c for c in range(7) if bits & (1<<c)}==set(xs)
            masks[bits]=xs
    assert len(masks)==127 and sorted(masks)==list(range(1,128))
    initial=mask([2,3,4])
    activated=mask([0,2,3,4])
    assert initial==28 and activated==29 and (initial&activated)==initial
    assert (initial & (1<<0))==0 and (activated & (1<<0))!=0
    return {
        'schema':'quartz_cog03_resident13_C03_finite_masks_v1',
        'number_nonempty_unique_valid_seven_column_masks':len(masks),
        'unrestricted_sentinel':0,
        'empty_API_input_raises':True,
        'all_masks_unique':True,
        'initial_three_example_bitmask':initial,
        'activation_example_bitmask':activated,
        'old_mask_subset_new':(initial&activated)==initial,
        'no_claim_about_actual_MCTS_runtime':True}

if __name__=='__main__':
    r=derive()
    p=Path(__file__).with_name('MASK_FINITE_CHECK13.json')
    if p.exists():raise FileExistsError('finite design result already frozen')
    p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('COG03_C03_FINITE_U64_MASK_CHECK_PASS',hashlib.sha256(p.read_bytes()).hexdigest(),r['number_nonempty_unique_valid_seven_column_masks'])
