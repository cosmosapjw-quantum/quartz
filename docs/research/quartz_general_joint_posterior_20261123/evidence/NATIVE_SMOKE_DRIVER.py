#!/usr/bin/env python3
"""Bounded C27 native smoke on one preexisting TRAIN source; no fitting or price."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT=Path("/home/cosmosapjw/quantum/research/quartz_general_meta27_20261123")
OLD=Path("/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123")
E=ROOT/"docs/research/quartz_general_joint_posterior_20261123/evidence"
DATA=OLD/"docs/research/quartz_general_game_metareasoning_20261123/coding/data"
BIN=OLD/"target/release/quartz_c27_joint_posterior"
OLD_BIN=OLD/"target/release/quartz_c26_game_metareasoning"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD_BIN)=="8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c"
assert sha(DATA/"model.json")=="0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126"
source=next(json.loads(line) for line in (DATA/"public_sources.jsonl").read_text().splitlines() if json.loads(line)["split"]=="TRAIN")
model=json.loads((DATA/"model.json").read_text())
request={"id":source["id"]+"-C27-smoke","history":source["history"],"cue_seed":source["cue_seed"],
         "policy_seed":3,"mode":"run","additional":2,"repeats":5,"warmup":1,"trace":True,"model":model,"rho":.8}
survey=dict(request,mode="survey",policy_seed=0,model=None,rho=0,trace=False,warmup=0)
invalid=dict(request,offline_values=[1,0,1])
lines=["not-json",json.dumps(invalid),json.dumps(request),json.dumps(survey)]
payload="\n".join(lines)+"\n"
(E/"NATIVE_SMOKE_REQUESTS.jsonl").write_text(payload)
start=time.monotonic_ns()
run=subprocess.run([str(BIN)],input=payload,text=True,capture_output=True,timeout=30)
roundtrip=time.monotonic_ns()-start
(E/"NATIVE_SMOKE_RESPONSES.jsonl").write_text(run.stdout)
(E/"NATIVE_SMOKE_STDERR.log").write_text(run.stderr)
assert run.returncode==0,(run.returncode,run.stderr)
rows=[json.loads(x) for x in run.stdout.splitlines()]
assert len(rows)==4
assert "error" in rows[0] and "error" in rows[1]
assert "error" not in rows[2] and "error" not in rows[3]
policies=["INITIAL_STOP","RANDOM_TWO","PRODUCT_H2","INITIAL_PROJECTED_H2","JOINT_H2"]
timing={"elapsed_ns","cue_ns","initial_ns","query_ns","controller_ns"}
def semantic(record):
 return {k:v for k,v in record.items() if k not in timing|{"repeat","execution_slot"}}
records=rows[2]["runs"]
assert len(records)==25
for repeat in range(5):
 for pi,policy in enumerate(policies):
  r=records[repeat*5+pi]
  assert r["policy"]==policy and r["repeat"]==repeat
  assert r["execution_slot"]==(pi-repeat-3)%5
  assert semantic(r)==semantic(records[pi])
  assert len(r["query_moves"])==len(set(r["query_moves"]))==r["additional_queries"]<=2
  assert r["charged_nodes"]==r["cue_nodes"]+r["initial_nodes"]+r["additional_nodes"]
  assert all(type(r[k]) is int and r[k]>=0 for k in timing)
  assert r["elapsed_ns"]==sum(r[k] for k in ["cue_ns","initial_ns","query_ns","controller_ns"])
for row in rows:
 assert type(row["serialize_payload_ns"]) is int
 assert type(row["decode_validate_ns"]) is int
price_samples=0
for c in rows[3]["candidates"]:
 if c["terminal_known"]:
  assert c["timings"]==[]
 else:
  assert len(c["timings"])==5
  for repeat,t in enumerate(c["timings"]):
   assert t["repeat"]==repeat and t["solve_nodes"]==c["solve_nodes"]
   assert type(t["elapsed_ns"]) is int and t["elapsed_ns"]>0
   price_samples+=1
summary={"command":[str(BIN)],"returncode":run.returncode,"roundtrip_ns":roundtrip,
         "source":source,"run_records":len(records),"measured_survey_queries":price_samples,
         "policies":policies,"first_repeat_queries":{r["policy"]:r["query_moves"] for r in records[:5]},
         "binary_sha256":sha(BIN),"old_binary_sha256":sha(OLD_BIN),
         "model_sha256":sha(DATA/"model.json"),
         "limits":"One existing TRAIN root only; rho .8 is a fixed software fixture, no selection/fit/time-price or CAL/TEST."}
(E/"NATIVE_SMOKE.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
