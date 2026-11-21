#!/usr/bin/env python3
"""C16 C8 blinded 39-root × 5-arm real native Rust MCTS, immutable source/model."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
ROOT=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
HOME=Path(__file__).resolve().parent.parent
MODEL=ROOT/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json"
ARMS=("learned_all_legal","learned_fork_mask","learned_fork_proof_choice",
      "learned_sham_mask","learned_top16")
COHORT_SHA="8349441f247f720028f8534ac4abf656be56f8ac58778191de900722b4af1049"
MODEL_SHA="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def check_clean_published():
 if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"]).strip():
  raise RuntimeError("C16 blind trial requires clean and published source before any test")
 head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
 branch=subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"],text=True).strip()
 remote=subprocess.check_output(["git","-C",str(ROOT),"ls-remote","--heads","origin",branch],text=True).split()[0]
 if head!=remote:raise RuntimeError("code source is not published")
 return head
def main():
 p=argparse.ArgumentParser();p.add_argument("--binary",type=Path,required=True)
 args=p.parse_args()
 head=check_clean_published()
 assert sha(MODEL)==MODEL_SHA
 cohortfile=HOME/"COHORT_40_FROZEN.json"
 assert sha(cohortfile)==COHORT_SHA
 src=json.loads(cohortfile.read_text())
 # fork-000 was exposed during C05 preflight and is NOT independent heldout.
 constructed=[s for s in src["data_splits"]["constructed_fork_positive"] if s["id"]!="fork-000"]
 played=src["data_splits"]["independent_MCTS_played_roots"]
 assert len(constructed)==23 and len(played)==16
 subjects=constructed+played
 q=[]
 index=[]
 for i,s in enumerate(subjects):
   rotation=ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)]
   for arm in rotation:
     index.append((s,arm))
     q.append({"cmd":"research_gomoku15_fork_cpu","rule":"freestyle",
      "history":s["history"],"arm":arm,"seed":17,"cpu_budget_us":20000,
      "model_path":str(MODEL)})
 binary=args.binary.resolve()
 assert binary.is_file()
 wire="\n".join(json.dumps(x,separators=(",",":")) for x in q+[{"cmd":"quit"}])+"\n"
 started=time.perf_counter()
 p=subprocess.run([str(binary),"--server"],input=wire,stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,text=True,env={**os.environ,"RAYON_NUM_THREADS":"1",
                                        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},
    timeout=240,check=True)
 results=[json.loads(z) for z in p.stdout.splitlines() if z.strip()]
 if len(results)!=len(q):raise RuntimeError(f"native responses {len(results)} != {len(q)}; stderr {p.stderr[-700:]}")
 per_source=[]
 for i,s in enumerate(subjects):
   arms={}
   for j in range(len(ARMS)):
     exp_s,arm=index[len(ARMS)*i+j]
     out=results[len(ARMS)*i+j]
     assert s["id"]==exp_s["id"]
     if "error" in out:raise RuntimeError(f"real Rust {s['id']}/{arm}: {out['error']}")
     if out["history"]!=s["history"] or out["arm"]!=arm:
       raise RuntimeError("native response source/arm misbound")
     assert out["target_cpu_budget_us"]==20000 and out["teacher_oracle_never_online"]
     assert out["same_root_N_W_M2_P_preserved_on_activation"] and out["same_root_ptr_hash"]
     assert out["root_edge_visit_sum"]==out["final_root_visits"]
     assert out["first8_root_visits"]==8
     assert out["original_full_root_legal_count"]==len(out["legal_actions"])==225-len(s["history"])
     assert out["final_legal_move"] in out["legal_actions"]
     assert out["final_legal_move"] in out["after_active_actions"]
     assert out["model_loaded"] and not out["trained_value_head"]
     arms[arm]=out
   per_source.append({"source_id":s["id"],"source_type":s["source_type"],
    "parent_game_id":s["parent_game_id"],"history":s["history"],"arms":arms})
 report={"schema":"quartz_cog06_C08_39_blind_root_real_native_v1",
         "source_git_commit":head,
         "native_release_sha256":sha(binary),
         "model_sha256":sha(MODEL),"frozen_source_cohort_sha256":sha(cohortfile),
         "source_n_constructed_heldout":len(constructed),
         "excluded_positive_C05_preflight":"fork-000",
         "source_n_MCTS_played_heldout":len(played),
         "n_actual_native_root_requests":len(q),
         "physical_decision_target_us":20000,
         "model_loading_and_JSON_output_excluded":True,
         "no_teacher_labels_or_exact_oracle_online":True,
         "run_elapsed_orchestrator_seconds":time.perf_counter()-started,
         "per_source":per_source}
 outdir=HOME/"evidence/real_root16_blind"
 outdir.mkdir(exist_ok=False,parents=True)
 raw=outdir/"RAW_ROOT_195.json"
 raw.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 gate={"schema":"quartz_cog06_C08_blind_compute_and_activation_v1",
       "source_raw_sha256":sha(raw),
       "posthoc_label_unchanged":True,"min_activated_constructed_source_groups":20,
       "strata":{}}
 for stratum,group in [("constructed",per_source[:23]),("MCTS_played",per_source[23:])]:
   n=len(group)
   def mean(arm):
     return sum(r["arms"][arm]["total_actual_cpu_wall_us"] for r in group)/n
   a,b=mean("learned_fork_mask"),mean("learned_all_legal")
   activated=sum(bool(r["arms"]["learned_fork_mask"]["added_actions"]) for r in group)
   changed=sum(r["arms"]["learned_fork_mask"]["final_legal_move"]!=r["arms"]["learned_all_legal"]["final_legal_move"] for r in group)
   gate["strata"][stratum]={"n":n,"activated":activated,"changed_final_actions_mask_vs_alllegal":changed,
     "actual_cpu_mean_mask_us":a,"actual_cpu_mean_alllegal_us":b,
     "actual_cpu_abs_relative_mismatch":abs(a-b)/a,
     "cost_gate_10pct":abs(a-b)/a<=0.1,
     "n_certified_proof_choice_actions":sum(r["arms"]["learned_fork_proof_choice"]["selected_is_certified_fork"] for r in group),
     "mean_cpu_other_arms_us":{arm:mean(arm) for arm in ARMS}}
 (outdir/"C08_BLIND_GATE.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n")
 print("C08_ROOT195_LIVE",json.dumps(gate,sort_keys=True),flush=True)
if __name__=="__main__":main()
