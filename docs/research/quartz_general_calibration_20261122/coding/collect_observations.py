#!/usr/bin/env python3
"""C18 actual source-only observation transcripts, prospective split train/cal then TEST.

This runner is code-frozen and GitHub-published before any TEST outcome run.
SHADOW only. No controller, game policy or trained neural model is changed.
"""
from pathlib import Path
import argparse,hashlib,json,random,subprocess,time
from quartz.experiments.c18_observation import initial_rust_snapshot
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BANK=HOME/"SOURCE_GROUPS_FROZEN.json"
BANK_SHA="52e3654febead1a01253174be0d395f8c2ea6e04ada21e5e0bfbf5938d680445"
LOCK=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
BIN=Path(LOCK["native_path"])
MODEL=Path(LOCK["model_path"])
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def current_published():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("outcome collection requires published clean source")
 sha0=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 ref=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",ref],cwd=ROOT,text=True).split()[0]
 if sha0!=remote:raise RuntimeError("Git source not actually at remote provider")
 return sha0
def sample_bernoulli(group,rep):
 k=group["n_arms"]
 true=group["true_utility_means_OFFLINE_ONLY"]
 seed=int.from_bytes(hashlib.sha256(
  f"nonga:{group['source_seed']}:{rep}".encode()).digest()[:8],"big")
 rng=random.Random(seed)
 success=[0]*k;count=[0]*k
 for j in range(k):
  for _ in range(group["initial_draws_each"]):
   success[j]+=int(rng.random()<true[j]);count[j]+=1
 mean=[(s+1)/(n+2) for s,n in zip(success,count)]
 order=sorted(range(k),key=lambda a:(-mean[a],a))
 a0=order[0]
 gap=mean[order[0]]-mean[order[1]]
 start=time.perf_counter_ns()
 for j in range(k):
  for _ in range(group["extra_draws_each"]):
   success[j]+=int(rng.random()<true[j]);count[j]+=1
 elapsed=max(1,(time.perf_counter_ns()-start)//1000)
 after=[(s+1)/(n+2) for s,n in zip(success,count)]
 a1=max(range(k),key=lambda j:(after[j],-j))
 return {"schema":"quartz_c18_observation_v1","domain":"non_game_Bernoulli",
  "group_id":group["id"],"source_id":group["id"],"rep":rep,"split":group["split"],
  "gap":float(gap),"n_actions":k,"pre_action":a0,"post_action":a1,
  "label":int(a0!=a1),"observed_extra_cost_us":int(elapsed),
  "initial_samples_each":group["initial_draws_each"],
  "additional_samples_each":group["extra_draws_each"],
  "signed_true_utility_delta_OFFLINE":float(true[a1]-true[a0]),
  "true_utility_means_OFFLINE":true,
  "n_compute_actions":k*group["extra_draws_each"],
  "decision_value_identity":"true simulator means known only to offline scorer",
  "simulator_actual_Bernoulli_draws":True}
def native_observation(proc,group,hist,snap_index,rep):
 req={"cmd":"research_gomoku15_fork_cpu","rule":"freestyle",
  "history":hist,"arm":"learned_all_legal","seed":17,
  "cpu_budget_us":20000,"model_path":str(MODEL)}
 proc.stdin.write(json.dumps(req,separators=(",",":"))+"\n");proc.stdin.flush()
 line=proc.stdout.readline()
 if not line:raise RuntimeError("actual native Rust process returned no result")
 out=json.loads(line)
 if "error" in out:raise RuntimeError(f"actual real Rust error at {group['id']}: {out['error']}")
 assert out["arm"]=="learned_all_legal" and out["history"]==hist
 assert out["same_root_N_W_M2_P_preserved_on_activation"]
 assert out["root_edge_visit_sum"]==out["final_root_visits"]
 assert out["teacher_oracle_never_online"]
 start=initial_rust_snapshot(out)
 a0=start["initial_action"];a1=int(out["final_mcts_move"])
 assert a0 in out["legal_actions"] and a1 in out["legal_actions"]
 return {"schema":"quartz_c18_observation_v1","domain":"real_Rust_MCTS",
   "group_id":group["id"],"source_id":f"{group['id']}-snap{snap_index}",
   "rep":rep,"split":group["split"],
   "gap":start["gap"],"n_actions":start["n_actions"],
   "pre_action":a0,"post_action":a1,"label":int(a0!=a1),
   "source_history":hist,"first8_root_edge_stats":out["first8_edge_stats"],
   "initial_root_visits":8,"final_root_visits":out["final_root_visits"],
   "observed_total_native_decision_wall_us":int(out["total_actual_cpu_wall_us"]),
   "actual_native_binary_sha256":LOCK["native_sha256"],
   "game_regret_or_signed_utility_change":"NOT_ESTIMABLE",
   "source_rule":"Freestyle15","continuation_is_real_native_Rust":True}
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--phase",choices=["train_cal","test"],required=True)
 args=parser.parse_args()
 head=current_published()
 if sha(BANK)!=BANK_SHA or sha(BIN)!=LOCK["native_sha256"] or sha(MODEL)!=LOCK["model_sha256"]:
  raise RuntimeError("frozen source, actual Rust binary or model identity violation")
 bank=json.loads(BANK.read_text())
 source_splits=("train","cal") if args.phase=="train_cal" else ("test",)
 native=[r for r in bank["mcts_real_Rust_groups"] if r["split"] in source_splits]
 nongame=[r for r in bank["non_game_groups"] if r["split"] in source_splits]
 assert len(native)==len(nongame)==(96 if args.phase=="train_cal" else 24)
 rows=[]
 proc=subprocess.Popen([str(BIN),"--server"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
  stderr=subprocess.DEVNULL,text=True,bufsize=1)
 try:
  for i,g in enumerate(native):
   for si,h in enumerate(g["checkpoints"]):
    for rep in range(2):
     rows.append(native_observation(proc,g,h,si,rep))
 finally:
  try:proc.stdin.write('{"cmd":"quit"}\n');proc.stdin.flush()
  except Exception:pass
  proc.stdin.close();proc.wait(timeout=15)
 for g in nongame:
  for rep in range(g["replicates"]):
   rows.append(sample_bernoulli(g,rep))
 expected_mcts=sum(len(g["checkpoints"])*2 for g in native)
 assert len(rows)==expected_mcts+3*len(nongame)
 assert all(r["split"] in source_splits for r in rows)
 if args.phase=="train_cal" and any(r["split"]=="test" for r in rows):
  raise AssertionError("test labels leaked into training collection")
 outdir=HOME/"evidence/C07_TRAIN_CAL" if args.phase=="train_cal" else HOME/"evidence/C08_TEST"
 outdir.mkdir(parents=True,exist_ok=False)
 out=outdir/"SOURCE_RAW.jsonl"
 out.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in rows))
 report={"schema":"quartz_c18_source_outcomes_v1","phase":args.phase,"code_HEAD_published_before_run":head,
  "frozen_bank_sha256":sha(BANK),"native_executable_sha256":sha(BIN),
  "native_model_sha256":sha(MODEL),"n_native_root_continuations":expected_mcts,
  "n_nongame_samples":len(nongame)*3,
  "n_parent_groups_native":len(native),"n_parent_groups_nongame":len(nongame),
  "native_Y_flip_rate":sum(x["label"] for x in rows if x["domain"]=="real_Rust_MCTS")/expected_mcts,
  "nongame_Y_flip_rate":sum(x["label"] for x in rows if x["domain"]=="non_game_Bernoulli")/(len(nongame)*3),
  "raw_sha256":sha(out),"source_generated_without_test_recalibration":True,
  "native_game_regret_label":"NOT_ESTIMABLE"}
 (outdir/"COLLECTION_MANIFEST.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("C18_SOURCE_COLLECTION",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main()
