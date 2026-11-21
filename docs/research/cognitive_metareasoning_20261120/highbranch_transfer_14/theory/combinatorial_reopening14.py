#!/usr/bin/env python3
"""Exact combinatorial candidate coverage & 225-cell four-word mask audit.

No MCTS game outcomes or pretrained weights are used. Real executable
assertions only concern exact finite combinatorics and representation.
"""
from fractions import Fraction
from itertools import combinations
from math import comb
from decimal import Decimal, getcontext
from pathlib import Path
import hashlib
import json
import random

def p_hit(B,k,r,m):
    assert isinstance(B,int) and 0<=k<B and 0<=m<=B-k and 0<=r<=B-k
    n=B-k
    miss=comb(n-m,r) if n-m>=r else 0
    return Fraction(comb(n,r)-miss,comb(n,r))

def encode(actions):
    assert actions and len(actions)==len(set(actions))
    assert all(type(a)==int and 0<=a<225 for a in actions)
    words=[0]*4
    for a in actions: words[a//64] |= (1 << (a%64))
    return tuple(words)

def decode(words):
    assert len(words)==4
    return [a for a in range(225) if words[a//64] & (1 << (a%64))]

def main():
    p7=p_hit(7,3,2,1)
    p225=p_hit(225,16,4,1)
    p225m5=p_hit(225,16,4,5)
    assert (p7,p225,p225m5)==(Fraction(1,2),Fraction(4,209),
                              Fraction(2393375,25746292))
    missed=[3,4,5,6]; target={4}
    total=list(combinations(missed,2))
    assert Fraction(sum(bool(set(pair)&target) for pair in total),len(total))==p7
    singles=set(encode([a]) for a in range(225))
    assert len(singles)==225
    for a in range(225):
        assert decode(encode([a]))==[a]
    # 25,200 different 2-action combinations, including every word boundary
    pairs=set()
    for a,b in combinations(range(225),2):
        key=encode([a,b])
        assert key not in pairs
        pairs.add(key)
    assert len(pairs)==25200
    rng=random.Random(20261121)
    for _ in range(1000):
        n=rng.randint(1,225)
        A=rng.sample(range(225),n)
        bits=encode(A)
        assert decode(bits)==sorted(A)
        B=A+rng.sample([x for x in range(225) if x not in A],min(225-n,4))
        new=encode(B)
        assert all(x & y == x for x,y in zip(bits,new))
    boundary=[0,63,64,127,128,191,192,224]
    assert decode(encode(boundary))==boundary
    getcontext().prec=50
    root=Decimal(32).sqrt()
    max_unvisited_bonus=Decimal(2)*Decimal('0.01')*root
    assert max_unvisited_bonus<Decimal('0.2')
    result={
        'schema':'quartz_cog04_highbranch14_exact_combinatorial_r7_v1',
        'claim_status':'DERIVED_AND_EXACT_FINITE_CHECKED',
        'p_hit_B7_k3_r2_m1':str(p7),
        'p_hit_B225_k16_r4_m1':str(p225),
        'p_hit_B225_k16_r4_m5':str(p225m5),
        'm1_probability_decimal':str(Decimal(4)/Decimal(209)),
        'm5_probability_decimal':str(Decimal(p225m5.numerator)/Decimal(p225m5.denominator)),
        'exact_small_B7_subsets_enumerated':len(total),
        'all_225_singletons_tested':225,
        'all_225_action_pairs_tested':len(pairs),
        'random_monotone_four_word_subset_checks':1000,
        'four_word_boundary_actions':boundary,
        'last_valid_index':224,
        'unvisited_bonus_at_Nmax32_prior001_c2':str(max_unvisited_bonus),
        'confirmed_below_stable_prior_action_score_point2':True,
        'exclusion':'NOT real Rust MCTS implementation or game playing strength',
    }
    out=Path(__file__).with_name('R07_EXACT_RESULTS.json')
    if out.exists():raise FileExistsError('source-locked exact result already exists')
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('COG04_R07_EXACT_FINITE_CALCULATIONS_PASS',
          hashlib.sha256(out.read_bytes()).hexdigest(),
          result['p_hit_B225_k16_r4_m5'])

if __name__=='__main__': main()
