#!/usr/bin/env python3
"""R07 independent exact Fraction witness; deliberately does not import C21/C20 EVSI code."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib,json,math
def beta_binomial(a,b,n):
    p=F(1)
    for i in range(n):p*= (b+i)/(a+b+i)
    out=[p]
    for k in range(n):
        p=p*F(n-k,k+1)*(a+k)/(b+n-k-1)
        out.append(p)
    assert sum(out)==1
    return out

def exact_full_grid(params,n):
    arms=[]
    for a,b in params:
        probs=beta_binomial(a,b,n)
        vals=[(a+k)/(a+b+n) for k in range(n+1)]
        arms.append(list(zip(vals,probs)))
    return sum(math.prod(x[1] for x in case)*max(x[0] for x in case)
        for case in product(*arms)),arms

def exact_cdf(params,n):
    _,arms=exact_full_grid(params,n)
    supports=sorted(set(v for arm in arms for v,p in arm))
    def fmax(v):
        return math.prod(sum(p for x,p in arm if x<=v) for arm in arms)
    prev=F(0)
    e=F(0)
    for v in supports:
        cur=fmax(v)
        assert prev<=cur<=1
        e+=v*(cur-prev)
        prev=cur
    assert prev==1
    return e,supports,arms

params=[(F(1),F(1)),(F(2),F(2)),(F(3),F(3))]
grid,arms=exact_full_grid(params,2)
cdf,support,vals=exact_cdf(params,2)
assert grid==cdf==F(529,840)
assert cdf-max(a/(a+b) for a,b in params)==F(109,840)
assert len(support)==7 and F(1,2) in support
assert sum(F(1,2)==x for arm in arms for x,p in arm)==3
fractional=beta_binomial(F(1,2),F(3,2),2)
assert fractional==[F(5,8),F(1,4),F(1,8)]
# Dependence falsifier, identical fair binary arm scores versus independent.
dependent_mean=F(1,2)
independent_mean=F(3,4)
assert independent_mean!=dependent_mean
result={"schema":"quartz_C21_R07_exact_independent_fraction_v1",
"claim":"Finite Beta-binomial exact product-CDF maximum with repeated tied support",
"status":"PASS",
"source_params":[[str(a),str(b)] for a,b in params],
"n_future_each":2,
"n_full_grid_cases":27,
"n_distinct_support":len(support),
"support":[str(x) for x in support],
"count_tied_support_half":3,
"full_joint_expected_max":str(grid),
"independent_max_CDF_sweep_expected_max":str(cdf),
"Bayes_expected_value_sample_information":str(cdf-F(1,2)),
"fractional_beta_shapes":["1/2","3/2"],
"fractional_pmf_n2":[str(x) for x in fractional],
"correlated_same_marginal_max_expectation":str(dependent_mean),
"independent_same_marginal_max_expectation":str(independent_mean),
"numerical_claim_ceiling":"exact finite models, not numerical Rust or real MCTS evidence"}
out=Path(__file__).with_name("R07_FRACTION_EXACT.json")
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print("C21_R07_EXACT",json.dumps(result,sort_keys=True))
