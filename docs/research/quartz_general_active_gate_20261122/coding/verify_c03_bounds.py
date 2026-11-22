#!/usr/bin/env python3
"""C03 independent cheap-support/variance bound versus existing full-grid exact EVSI."""
from pathlib import Path
import json,hashlib,math,random
from quartz.experiments.c20_predictive_value import beta_evsi,sensor_evsi
def upper(arms,budgets):
 mu=[a/(a+b) for a,b in arms]
 win=max(range(len(mu)),key=lambda j:(mu[j],-j))
 var=[n*a*b/((a+b)**2*(a+b+1)*(a+b+n)) for (a,b),n in zip(arms,budgets)]
 gross=0.
 for j in range(len(mu)):
  if j==win:continue
  delta=mu[j]-mu[win]
  gross+=0.5*(math.sqrt(delta*delta+var[win]+var[j])+delta)
 return min(max(0.,gross),1-mu[win])
def cert(arms,budgets):
 mu=[a/(a+b) for a,b in arms]
 win=max(range(len(mu)),key=lambda j:(mu[j],-j))
 low=arms[win][0]/(sum(arms[win])+budgets[win])
 high=max((arms[j][0]+budgets[j])/(sum(arms[j])+budgets[j])
          for j in range(len(arms)) if j!=win)
 return low>high
dom=[(12.,2.),(2.,12.)]
assert cert(dom,[1,1])
assert abs(beta_evsi(dom,1)["evsi"])<1e-12
assert not cert([(1.,1.),(1.,1.)],[1,1])
assert abs(beta_evsi([(1.,1.),(1.,1.)],1)["evsi"]-1/12)<1e-12
rng=random.Random(202611222300)
count=0;certified=0;max_slack=0.
for i in range(125):
 k=2+i%3
 arms=[(float(rng.choice([2,3,5,8,12,25,35])),
        float(rng.choice([2,3,4,9,20,35]))) for _ in range(k)]
 n=rng.choice([0,1,2,5,10])
 b=[n]*k
 true=beta_evsi(arms,n)["evsi"]
 cap=upper(arms,b)
 assert true<=cap+1e-10,(i,k,n,arms,true,cap)
 if cert(arms,b):
  assert true<1e-10,(i,true)
  certified+=1
 max_slack=max(max_slack,cap-true)
 count+=1
posterior=[.25,.25,.25,.25]
L=[[.75,.2,.05],[.1,.8,.1],[.05,.15,.8],[.5,.25,.25]]
U=[[.95,.1,.12],[.9,.15,.2],[.85,.1,.3],[.98,.14,.2]]
s=sensor_evsi(posterior,L,U,3)
before=max(sum(posterior[t]*U[t][a] for t in range(4)) for a in range(3))
evpi=sum(posterior[t]*max(U[t]) for t in range(4))-before
assert -1e-12<=s["evsi"]<=evpi+1e-12
res={"schema":"quartz_C22_C03_cheap_bounds_vs_C20_exact_v1","status":"PASS",
"real_beta_full_grid_reference_cases":count,
"support_dominant_cases":certified,
"max_upper_slack":max_slack,
"zero_evsi_dominance_witness":True,
"nonzero_one_drawing_EVSI":"1/12",
"analytic_Cauchy_Beta_bound":"PASS for independently randomized bounded sample",
"sensor_evpi_check":True,
"claims":"mathematical bounded model; not C22 native active policy"}
p=Path(__file__).with_name("C03_BOUND_CHECK.json")
p.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("C22_C03_BOUNDS",json.dumps(res,sort_keys=True))
