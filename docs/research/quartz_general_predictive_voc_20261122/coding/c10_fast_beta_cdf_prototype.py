#!/usr/bin/env python3
"""C10 constructive CDF-of-maximum Beta EVSI alternative, TRAIN-only prototype.

Independent Polya-urn beta-binomial recursion, exact rational support ordering,
no full (n+1)^K count-vector enumeration; keeps correct full joint predictive law.
Not used to retune C20 heldout policy. Objective is a concrete C21 cost solution.
"""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import hashlib,itertools,json,math,statistics,time
from quartz.experiments.c20_predictive_value import beta_evsi
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def polya_pmf(a,b,n):
 a,b=int(a),int(b)
 p=math.prod((b+j)/(a+b+j) for j in range(n))
 out=[p]
 for k in range(n):
  p=p*((n-k)/(k+1))*((a+k)/(b+n-k-1))
  out.append(p)
 assert abs(sum(out)-1)<1e-9
 return out
def cdf_max_exact_support_evsi(ab,n):
 start=time.perf_counter_ns()
 if not(2<=len(ab)<=4) or not 0<=n<=10:raise ValueError("bounded source only")
 # Exact rational support under positive INTEGER Beta shapes used in C20 source.
 distributions=[]
 for a,b in ab:
  if int(a)!=a or int(b)!=b:raise ValueError("C10 prototype integer posterior shapes only")
  probs=polya_pmf(a,b,n)
  scores=[Fraction(int(a)+k,int(a)+int(b)+n) for k in range(n+1)]
  distributions.append(list(zip(scores,probs)))
 support=sorted(set(s for arm in distributions for s,_ in arm))
 previous=0.
 mean_best=0.
 for v in support:
  current=math.prod(math.fsum(p for u,p in arm if u<=v) for arm in distributions)
  mean_best+=float(v)*(current-previous)
  previous=current
 assert abs(previous-1)<1e-9
 before=max(a/(a+b) for a,b in ab)
 evsi=mean_best-before
 assert evsi>=-1e-9
 return max(0.,evsi),(time.perf_counter_ns()-start)/1000,len(support)
rows=[json.loads(s) for s in SOURCE.read_text().splitlines() if s]
train=[r for r in rows if r["split"]=="train" and r["family"]=="beta_arms"]
assert len(train)==216
measure=defaultdict(list)
maxerr=0.;checked=0
for r in train:
 ab=[(2+sum(seq),2+len(seq)-sum(seq)) for seq in r["initial_observations"]]
 assert len(ab)==r["n_actions"]
 baseline=beta_evsi(ab,10)
 candidate,fast_us,support=cdf_max_exact_support_evsi(ab,10)
 assert abs(candidate-baseline["evsi"])<1e-10,(ab,candidate,baseline["evsi"])
 maxerr=max(maxerr,abs(candidate-baseline["evsi"]))
 measure[str(len(ab))].append((baseline["evaluation_us"],fast_us,support))
 checked+=1
res={"schema":"quartz_C20_C10_exploratory_fast_beta_max_CDF_prototype_v1",
 "status":"NUMERICALLY_CHECKED_TRAIN_ONLY",
 "n_train_initial_only_reference_cases":checked,
 "original_C20_source_train_SHA256":sha(SOURCE),
 "max_absolute_EVSI_disagreement":maxerr,
 "exact_algebra":"For independent posterior predictive arm scores X_j, P(max_j X_j <= v)=product_j F_j(v). E[max X]=sum over unique discrete v of v*(F_max(v)-F_max(v-)).",
 "no_test_labels_used":True,
 "not_retroactively_promoted_as_C20_TEST_policy":True,
 "not_in_live_SearchPolicy":True,
 "measured_wall_us_by_arm_count":{
    k:{"n":len(v),"full_joint_median_us":statistics.median(x[0] for x in v),
       "CDF_exact_support_median_us":statistics.median(x[1] for x in v),
       "median_speed_ratio_individual":statistics.median(x[0]/max(1e-9,x[1]) for x in v),
       "number_exact_distinct_support_points":sorted(set(x[2] for x in v))}
 for k,v in measure.items()},
 "limitations":"integer Beta shapes; Python prototype; Rust native implementation and hot-path proof absent"}
out=HOME/"evidence/C10_FAST_BETA_MAX_CDF_WITNESS.json"
out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("C20_FAST_CDF_CONSTRUCTION",json.dumps(res,sort_keys=True))
