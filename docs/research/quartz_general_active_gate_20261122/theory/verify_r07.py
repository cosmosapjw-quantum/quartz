#!/usr/bin/env python3
"""Exact Fraction R07 support dominance/EVSI/variance vs an O(K) upper bound."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from math import sqrt
import hashlib,json,math
def betaP(a,b,n):
  p=F(1)
  for i in range(n):p*= (b+i)/(a+b+i)
  probabilities=[p]
  for k in range(n):
    p*=F(n-k,k+1)*(a+k)/(b+n-k-1)
    probabilities.append(p)
  assert sum(probabilities)==1
  return probabilities
def EVSI(arms,budget):
  probs=[betaP(a,b,n) for (a,b),n in zip(arms,budget)]
  current=max(a/(a+b) for a,b in arms)
  gross=F(0)
  for counts in product(*(range(n+1) for n in budget)):
    p=math.prod(probs[j][k] for j,k in enumerate(counts))
    score=max((a+counts[j])/(a+b+budget[j]) for j,(a,b) in enumerate(arms))
    gross+=p*score
  return gross-current
dom=[(F(12),F(2)),(F(2),F(12))]
budget=[1,1]
winner=0
low=dom[0][0]/(sum(dom[0])+1)
other_high=(dom[1][0]+1)/(sum(dom[1])+1)
assert low==F(4,5) and other_high==F(1,5) and low>other_high
assert EVSI(dom,budget)==0
nonzero=[(F(1),F(1)),(F(1),F(1))]
g=EVSI(nonzero,[1,1])
assert g==F(1,12)
variances=[]
for a,b in nonzero:
 s=a+b
 v=F(1)*a*b/(s*s*(s+1)*(s+1))
 variances.append(v)
assert variances==[F(1,36),F(1,36)]
bound=sqrt(float(sum(variances)))/2
assert bound>float(g) and abs(bound-sqrt(2)/12)<1e-15
a,b,n=F(5),F(3),2
pr=betaP(a,b,n)
score=[(a+k)/(a+b+n) for k in range(n+1)]
avg=sum(x*p for x,p in zip(score,pr))
var=sum(p*(x-avg)**2 for x,p in zip(score,pr))
formula=F(n)*a*b/((a+b)**2*(a+b+1)*(a+b+n))
assert pr==[F(1,6),F(5,12),F(5,12)]
assert avg==F(5,8) and var==formula==F(1,192)
receipt={"schema":"quartz_C22_R07_Fraction_exact_v1","status":"PASS",
 "nonnumeric_posteriors":"Beta rational shapes with exact Fraction arithmetic",
 "support_dominance":{"winner":str(low),"competitor_max":str(other_high),
   "EVSI":str(EVSI(dom,budget)),"all_future_actions_unchanged":True},
 "independent_two_uniform_arms_one_draw":{"EVSI":str(g),
   "future_score_variances":[str(x) for x in variances],
   "Cauchy_upper_bound_float":bound,"upper_holds":True},
 "beta_5_3_two_future":{"pmf":[str(x) for x in pr],
   "future_score_mean":str(avg),"future_score_variance":str(var),
   "formula_variance":str(formula)},
 "claims":"finite exact math, not implementation/real-world game calibration"}
out=Path(__file__).with_name("R07_EXACT.json")
out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print("C22_R07",json.dumps(receipt,sort_keys=True))
