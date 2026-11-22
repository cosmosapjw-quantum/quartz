#!/usr/bin/env python3
"""Independent exact Fraction check, no C20 product-code import."""
from pathlib import Path
from fractions import Fraction as Q
from math import comb
from itertools import product
import json
# One accurate binary sensor with prior fair and identity utility.
prior=[Q(1,2),Q(1,2)]
L=((Q(3,4),Q(1,4)),(Q(1,4),Q(3,4)))
base=max(prior)
later=Q(0)
for y in range(2):
 weights=[prior[t]*L[t][y] for t in range(2)]
 later+=max(weights)   # p(y) * posterior-optimal utility
assert later==Q(3,4)
sensor_evsi=later-base
assert sensor_evsi==Q(1,4)
# One draw per each of two independent Beta(1,1) arms.
result=Q(0)
for x,y in product((0,1),repeat=2):
 p=Q(1,4)
 after=max(Q(1+x,3),Q(1+y,3))
 result+=p*after
assert result==Q(7,12)
beta_evsi=result-Q(1,2)
assert beta_evsi==Q(1,12)
# One Beta(1,1) arm with n=2 predictive counts p(k)=1/3 each.
# Gamma/beta ratio reduces to factorials for integer shapes.
factorial=__import__("math").factorial
def B(a,b):
 return Q(factorial(a-1)*factorial(b-1),factorial(a+b-1))
pmf=[Q(comb(2,k))*B(k+1,3-k)/B(1,1) for k in range(3)]
assert pmf==[Q(1,3)]*3 and sum(pmf)==1
# Three independent posterior updates of same hidden state are correlated.
# With deterministic signal reveal, 00 and 11 have mass 1/2, mixed paths zero.
actual={(0,0):Q(1,2),(0,1):Q(0),(1,0):Q(0),(1,1):Q(1,2)}
assert sum(actual.values())==1 and actual[(0,1)]!=Q(1,4)
res={"schema":"quartz_c20_R07_exact_fraction","status":"PASS",
"binary_noisy_sensor_EVSI":str(sensor_evsi),
"independent_two_arms_one_draw_EVSI":str(beta_evsi),
"beta_binomial_n2_distribution":[str(x) for x in pmf],
"beta_binomial_normalization":"1",
"shared_latent_independence_counterexample":"PASS",
"claims":"finite exact only, production model not yet verified"}
p=Path(__file__).with_name("R07_EXACT_RESULTS.json")
p.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
print("R07_EXACT",json.dumps(res,sort_keys=True))
