#!/usr/bin/env python3
"""C4 source-only exact rational contract check and freeze, before any new learner results."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
OUT=Path(__file__).resolve().parents[1]/"evidence"
P=ROOT/"configs/metalevel_reference_cases.v1.json"
def sha(path):
 with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
obj=json.loads(P.read_text())
assert obj["schema"]=="quartz_general_meta17_frozen_finite_Bayes_tasks_v1"
ids=set()
for row in obj["cases"]:
 assert row["id"] not in ids
 ids.add(row["id"])
 prior=list(map(F,row["posterior"]))
 assert prior and all(p>=0 for p in prior) and sum(prior)==1
 utils=[list(map(F,r)) for r in row["utilities"]]
 assert len(utils)==len(prior)
 assert len(utils[0])>=2 and all(len(x)==len(utils[0]) for x in utils)
 assert F(row["lambda"])>0 and 0<=row["horizon"]<=4
 for c in row["experiments"]:
  assert F(c["cost"])>0
  likelihood=[list(map(F,x)) for x in c["likelihood"]]
  assert len(likelihood)==len(prior)
  assert len(likelihood[0])>=1
  assert all(len(x)==len(likelihood[0]) and sum(x)==1 and all(0<=q<=1 for q in x) for x in likelihood)
assert len(ids)==5
record={
"schema":"quartz_meta17_C04_frozen_case_lock_v1",
"source_file":"configs/metalevel_reference_cases.v1.json",
"case_file_sha256":sha(P),
"n_tasks":len(ids),"task_ids":sorted(ids),
"n_game_specific_tasks":0,
"n_heterogeneous_cost_tasks":sum(len(set(F(c["cost"]) for c in r["experiments"]))>1 for r in obj["cases"]),
"all_probability_rows_exact_Fraction_normalized":True,
"no_stochastic_trials_or_metric_tuning_pre-freeze":True,
"case_canonicalization":"bytes_locked",
"source_scope":"finite exact reference only",
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"C04_FROZEN_CASES.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C04_FROZEN",json.dumps(record,sort_keys=True))
