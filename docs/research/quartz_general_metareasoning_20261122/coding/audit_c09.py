#!/usr/bin/env python3
"""C9 independent exact-rational and paired-MC audit; NO imports from original runner."""
from __future__ import annotations
from fractions import Fraction as F
from functools import lru_cache
from collections import Counter
from pathlib import Path
import hashlib,json,math
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HERE=Path(__file__).resolve().parent.parent
DATA=HERE/"evidence/C08"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def value(b,ut):
 return max(sum(p*row[a] for p,row in zip(b,ut)) for a in range(len(ut[0])))
def risk(b,ut):
 return sum(p*max(row) for p,row in zip(b,ut))-value(b,ut)
def observations(b,channel):
 for y in range(len(channel[0])):
    mass=sum(p*row[y] for p,row in zip(b,channel))
    if mass>0:yield y,mass,tuple(p*row[y]/mass for p,row in zip(b,channel))
def setup(task):
 b=tuple(map(F,task["posterior"]))
 u=tuple(tuple(map(F,row)) for row in task["utilities"])
 experiments=tuple((exp["name"],F(exp["cost"]),tuple(tuple(map(F,row)) for row in exp["likelihood"])) for exp in task["experiments"])
 return b,u,experiments,F(task["lambda"]),task["horizon"]
def choose(b,u,exps,lam,h,policy):
 if h==0 or policy=="stop":return None
 def gain(c):
    return sum(prob*value(after,u) for _,prob,after in observations(b,c[2]))-value(b,u)
 gains=[gain(c) for c in exps]
 if policy in ("ratio","net"):
    scores=[(g/c[1] if policy=="ratio" else g-lam*c[1]) for g,c in zip(gains,exps)]
    idx=max(range(len(scores)),key=lambda i:scores[i])
    return idx if scores[idx]>(lam if policy=="ratio" else 0) else None
 if policy=="bellman":
    return bellman(b,u,exps,lam,h)[1]
 raise AssertionError(policy)
def bellman(b,u,exps,lam,h):
 stop=risk(b,u)
 if h==0:return stop,None
 best=(stop,None)
 for i,c in enumerate(exps):
    v=lam*c[1]+sum(prob*bellman(after,u,exps,lam,h-1)[0]
                  for _,prob,after in observations(b,c[2]))
    if v<best[0]:best=(v,i)
 return best
def terminal_loss(u,theta,b):
 a=max(range(len(u[0])),key=lambda j:sum(p*row[j] for p,row in zip(b,u)))
 return max(u[theta])-u[theta][a],a
def expected_score(b,u,exps,lam,h,policy):
 def rec(theta,post,remaining):
    idx=choose(post,u,exps,lam,remaining,policy)
    if idx is None:return terminal_loss(u,theta,post)[0]
    c=exps[idx]
    return lam*c[1]+sum(p*rec(theta,tuple(x*y/p1 for x,y in zip(post,[r[j] for r in c[2]])),remaining-1)
                       for j,p in enumerate(c[2][theta])
                       if p>0 for p1 in [sum(x*r[j] for x,r in zip(post,c[2]))])
 return sum(p*rec(i,b,h) for i,p in enumerate(b) if p>0)

def uniform(*parts):
 raw=":".join(map(str,parts)).encode()
 return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")/2**64
def discrete(probs,x):
 cum=0.0
 for i,p in enumerate(probs):
    cum+=float(p)
    if x<cum:return i
 return len(probs)-1
def recompute_trial(task,policy,i,seed):
 b,u,exps,lam,h=setup(task)
 theta=discrete(b,uniform("state",seed,task["id"],i))
 counts=Counter()
 cost=F(0)
 post=b
 for rem in range(h,0,-1):
    idx=choose(post,u,exps,lam,rem,policy)
    if idx is None:break
    c=exps[idx]
    obs=discrete(c[2][theta],uniform("obs",seed,task["id"],i,c[0],counts[c[0]]))
    counts[c[0]]+=1
    mass=sum(p*r[obs] for p,r in zip(post,c[2]))
    assert mass>0
    post=tuple(p*r[obs]/mass for p,r in zip(post,c[2]))
    cost+=lam*c[1]
 loss,chosen=terminal_loss(u,theta,post)
 return {"state_index":theta,"chosen_action":chosen,"n_computations":sum(counts.values()),
         "decision_loss":float(loss),"priced_compute_cost":float(cost)}

def main():
 spec=json.loads((ROOT/"configs/metalevel_reference_cases.v1.json").read_text())
 exp=json.loads((DATA/"C08_RESULTS.json").read_text())
 contract=json.loads((DATA/"EXECUTION_CONTRACT.json").read_text())
 raw=DATA/"PAIRED_MC_10240.jsonl"
 assert sha(raw)==exp["raw_SHA256"]=="dc394b70540753ed6fe120c29bbf4215a2a27c33c7a661f5a140631a86a22c9d"
 assert contract["source_code_sha256"]==sha(HERE/"coding/run_c08_value_lab.py")
 taskby={t["id"]:t for t in spec["cases"]}
 rows=[json.loads(s) for s in raw.read_text().splitlines() if s]
 assert len(rows)==10240
 lookup={(r["task"],r["policy"],r["trial"]):r for r in rows}
 assert len(lookup)==len(rows)
 checks=0; recounted=0; exacts={}
 for s in exp["per_arm"]:
    task=taskby[s["task"]]
    b,u,exps,lam,h=setup(task)
    policy=s["policy"]
    exact=expected_score(b,u,exps,lam,h,policy)
    actual=s["exact_expected_total_loss"]
    if abs(float(exact)-actual)>1e-10:
       raise AssertionError(("independent exact mean mismatch",task["id"],policy,exact,actual))
    if policy=="bellman":
       assert exact==bellman(b,u,exps,lam,h)[0]
    sample=[lookup[(task["id"],policy,i)] for i in range(512)]
    mean=sum(z["decision_loss"]+z["priced_compute_cost"] for z in sample)/512
    assert math.isclose(mean,s["MC_mean_total_loss"],rel_tol=0,abs_tol=1e-12)
    for i in range(128):
       original=sample[i]
       check=recompute_trial(task,policy,i,contract["seed"])
       for field in ("state_index","chosen_action","n_computations"):
         assert check[field]==original[field],(task["id"],policy,i,field)
       for field in ("decision_loss","priced_compute_cost"):
         assert abs(check[field]-original[field])<1e-10,(task["id"],policy,i,field)
       checks+=1
    recounted+=512
    exacts[(task["id"],policy)]=float(exact)
 assert exacts[("heterogeneous_complete_vs_partial","ratio")]==0.21
 assert exacts[("heterogeneous_complete_vs_partial","bellman")]==0.2
 assert exacts[("heterogeneous_complete_vs_partial","net")]==0.2
 assert exacts[("heterogeneous_complete_vs_partial","stop")]==0.5
 summary={"schema":"quartz_meta17_C09_independent_Fraction_audit_v1",
   "status":"PASS",
   "source_sha256":sha(ROOT/"configs/metalevel_reference_cases.v1.json"),
   "raw_MC_sha256":sha(raw),"c08_summary_sha256":sha(DATA/"C08_RESULTS.json"),
   "n_independent_exact_task_policy_expectations":len(exacts),
   "n_all_raw_rows_checked_for_uniqueness_and_reported_mean":recounted,
   "n_trial_paths_independently_recomputed_with_exact_Fractions":checks,
   "exact_heterogeneous_total_loss":{p:exacts[("heterogeneous_complete_vs_partial",p)] for p in ("stop","ratio","net","bellman")},
   "no_game_or_live_neural_MCTS_claim":True}
 out=HERE/"evidence/C09_INDEPENDENT_FRACTION_AUDIT.json"
 out.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
 print("C09_MATH_INDEPENDENT",json.dumps(summary,sort_keys=True))
if __name__=="__main__":main()
