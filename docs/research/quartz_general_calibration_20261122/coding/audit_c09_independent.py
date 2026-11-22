#!/usr/bin/env python3
"""C9 independent source/decision/utility proper-score audit, without importing C5/C8."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math,random,statistics
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
bank=json.loads((HOME/"SOURCE_GROUPS_FROZEN.json").read_text())
model=json.loads((HOME/"SHADOW_MODEL_FROZEN.json").read_text())
report=json.loads((HOME/"evidence/C08_TEST/C08_HELDOUT_RESULTS.json").read_text())
trainfile=HOME/"evidence/C07_TRAIN_CAL/SOURCE_RAW.jsonl"
testfile=HOME/"evidence/C08_TEST/SOURCE_RAW.jsonl"
train=[json.loads(s) for s in trainfile.read_text().splitlines() if s]
rows=[json.loads(s) for s in testfile.read_text().splitlines() if s]
sources={r["id"]:r for r in bank["mcts_real_Rust_groups"]+bank["non_game_groups"]}
assert len(rows)==212 and sha(testfile)==report["test_raw_sha256"]
assert model["heldout_test_labels_seen"] is False
assert model["train_cal_raw_sha256"]==sha(trainfile)
assert not {r["group_id"] for r in train}&{r["group_id"] for r in rows}
assert all(r["split"]=="test" for r in rows)
def is_win(b,p,a):
 x=a%15;y=a//15
 for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
  n=1
  for sign in (-1,1):
   xx=x+sign*dx;yy=y+sign*dy
   while 0<=xx<15 and 0<=yy<15 and b[yy*15+xx]==p:
    xx+=sign*dx;yy+=sign*dy;n+=1
  if n>=5:return True
 return False
def replay(h):
 b=[0]*225
 for i,a in enumerate(h):
  assert 0<=a<225 and b[a]==0 and not is_win(b,1+i%2,a)
  b[a]=1+i%2
 return b
count=Counter()
for r in rows:
 g=sources[r["group_id"]]
 assert g["split"]=="test"
 assert r["label"]==int(r["pre_action"]!=r["post_action"])
 assert r["observed_total_native_decision_wall_us"]>0 if r["domain"]=="real_Rust_MCTS" else r["observed_extra_cost_us"]>0
 if r["domain"]=="real_Rust_MCTS":
  assert r["source_history"] in g["checkpoints"]
  b=replay(r["source_history"])
  assert b[r["pre_action"]]==b[r["post_action"]]==0
  edges=r["first8_root_edge_stats"]
  assert len(edges)==r["n_actions"]==225-len(r["source_history"])
  assert sum(int(x[1]) for x in edges)==8
  order=sorted(edges,key=lambda x:(-int(x[1]),-float(x[2]),int(x[0])))
  assert r["pre_action"]==int(order[0][0])
  assert abs(r["gap"]-(order[0][1]-order[1][1])/8)<1e-12
  count["rust_actual"]+=1
 else:
  seed=int.from_bytes(hashlib.sha256(f"nonga:{g['source_seed']}:{r['rep']}".encode()).digest()[:8],"big")
  rng=random.Random(seed)
  k=g["n_arms"];mu=g["true_utility_means_OFFLINE_ONLY"]
  initial=[];final=[]
  for j in range(k):
   initial.append(sum(int(rng.random()<mu[j]) for _ in range(3)))
  p0=[(s+1)/5 for s in initial]
  a0=max(range(k),key=lambda j:(p0[j],-j))
  for j in range(k):
   final.append(initial[j]+sum(int(rng.random()<mu[j]) for _ in range(10)))
  p1=[(s+1)/15 for s in final]
  a1=max(range(k),key=lambda j:(p1[j],-j))
  assert r["pre_action"]==a0 and r["post_action"]==a1
  assert abs(r["signed_true_utility_delta_OFFLINE"]-(mu[a1]-mu[a0]))<1e-15
  assert abs(r["gap"]-(sorted(p0,reverse=True)[0]-sorted(p0,reverse=True)[1]))<1e-12
  count["non_game_actual"]+=1
assert count["rust_actual"]==140 and count["non_game_actual"]==72
w=model["weights"];offset=model["cal_offset"]
def sigmoid(v):
 return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
for domain in ("real_Rust_MCTS","non_game_Bernoulli"):
 subset=[r for r in rows if r["domain"]==domain]
 baseline=model["train_baseline_constant_by_domain"][domain]
 p=[min(1-1e-6,max(1e-6,sigmoid(w[0]+w[1]*r["gap"]+w[2]*math.log1p(r["n_actions"])+offset))) for r in subset]
 brier=sum((v-r["label"])**2 for r,v in zip(subset,p))/len(subset)
 brier0=sum((baseline-r["label"])**2 for r in subset)/len(subset)
 target=report["heldout_per_domain"][domain]
 assert abs(target["calibrated"]["brier"]-brier)<1e-12
 assert abs(target["train_constant"]["brier"]-brier0)<1e-12
 assert abs(target["calibrated"]["event_rate"]-sum(r["label"] for r in subset)/len(subset))<1e-12
 assert len(set(r["group_id"] for r in subset))==24
res={"schema":"quartz_general_c18_C09_independent_source_and_score_audit_v1",
 "status":"PASS","actual_native_root_traces_validated":140,
 "actual_nongame_sequences_independently_replayed":72,
 "exact_parent_GROUP_test_disjoint":True,
 "source_bank_SHA256":sha(HOME/"SOURCE_GROUPS_FROZEN.json"),
 "train_raw_SHA256":sha(trainfile),"test_raw_SHA256":sha(testfile),
 "model_SHA256":sha(HOME/"SHADOW_MODEL_FROZEN.json"),
 "heldout_score_SHA256":sha(HOME/"evidence/C08_TEST/C08_HELDOUT_RESULTS.json"),
 "independent_Brier_and_train_constant_recalculation":"PASS",
 "game_true_decision_value":"NOT_ESTIMABLE",
 "domain_general_efficiency_superiority":"NOT_ESTABLISHED"}
out=HOME/"evidence/C09_INDEPENDENT_REPLAY.json"
out.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
print("C09_INDEPENDENT",json.dumps(res,sort_keys=True))
