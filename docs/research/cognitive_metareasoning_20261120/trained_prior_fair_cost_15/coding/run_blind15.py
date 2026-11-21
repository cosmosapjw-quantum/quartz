#!/usr/bin/env python3
"""Blinded real Rust QUARTZ 3-arm, 64-source, matched 20ms budget evaluation."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
ARMS=("learned_gated","learned_all_legal","uniform_all_legal")
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 cli=argparse.ArgumentParser()
 cli.add_argument("--binary",type=Path,required=True)
 args=cli.parse_args()
 binary=args.binary.resolve()
 d=Path(__file__).resolve().parent
 home=d.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
   raise RuntimeError("C08 experimental source must have previously published clean Git commit")
 git_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
 remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','--heads','origin',branch],text=True).split()[0]
 if remote!=git_sha:raise RuntimeError("source commit not verified remotely")
 cohort_file=home/"COHORT_224_FROZEN.json"
 cohort=json.loads(cohort_file.read_text())
 frozen=json.loads((home/"C04_FREEZE_MANIFEST.json").read_text())
 assert sha(cohort_file)==frozen["source_file_sha256"]
 model_file=home/"evidence/model/MODEL_TRAINED.json"
 model=json.loads(model_file.read_text())
 assert model["schema"]=="quartz_cog05_loglinear_policy13_v1"
 assert not model["heldout_test_teacher_labels_used"] and model["trained_value_head"]==False
 receipt=json.loads((home/"evidence/model/MODEL_C07_FREEZE.json").read_text())
 assert receipt["model_sha256"]==sha(model_file)
 if not binary.is_file():raise RuntimeError("actual native Rust binary missing")
 labels=home/"evidence/teacher_heldout/TEACHER_HELDOUT.json"
 if labels.exists():raise RuntimeError("test teacher labels already exist: blind Rust must execute first")
 cases=[(split,source) for split in ("test_natural","test_tactical")
                         for source in cohort["splits"][split]]
 requests=[{"cmd":"research_gomoku15_prior_cpu","rule":"freestyle",
            "history":src["history"],"arm":arm,"seed":17,"cpu_budget_us":20000,
            **({"model_path":str(model_file)} if arm!="uniform_all_legal" else {})}
           for split,src in cases for arm in ARMS]
 payload="\n".join(json.dumps(q,separators=(",",":")) for q in requests+[{"cmd":"quit"}])+"\n"
 began=time.perf_counter()
 proc=subprocess.run([str(binary),"--server"],input=payload,text=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     timeout=300,check=True,env={**os.environ,'RAYON_NUM_THREADS':'1',
                            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 outputs=[json.loads(row) for row in proc.stdout.splitlines() if row.strip()]
 if len(outputs)!=len(requests):raise RuntimeError(f"responses={len(outputs)} requests={len(requests)}")
 rows=[]
 for (split,src), k in zip(cases,range(0,len(outputs),3)):
    dct={}
    for i,arm in enumerate(ARMS):
       row=outputs[k+i]
       if "error" in row:raise RuntimeError(f"actual Rust root failed on {src['id']} /{arm}: {row['error']}")
       if row["arm"]!=arm or row["history"]!=src["history"]:
          raise RuntimeError("misbound JSON response")
       assert row["same_root_hash_preserved"] and row["same_root_ptr_preserved"]
       assert row["root_stats_and_priors_unchanged_on_activation"]
       assert row["root_edge_visits_sum"]==row["final_root_visits"]
       assert row["first8_root_visits"]==8 and row["final_root_visits"]>=8
       assert row["full_original_root_edges"]==225-len(src["history"])
       assert row["no_online_teacher_labels"] and row["target_cpu_budget_us"]==20000
       assert row["final_legal_move"] in row["legal_actions"]
       dct[arm]=row
    rows.append({"source_id":src["id"],"split":split,"history":src["history"],
                 "arms":dct})
 report={"schema":"quartz_cog05_C08_blind192_actual_Rust_MCTS_v1",
         "source_commit":git_sha,"source_cohort_sha256":sha(cohort_file),
         "compiled_native_binary_sha256":sha(binary),
         "frozen_model_sha256":sha(model_file),
         "n_source_roots":len(cases),
         "real_Rust_requests":len(requests),"training_teacher_labels_used_online":False,
         "test_teacher_labels_used_online":False,
         "total_orchestrator_seconds":time.perf_counter()-began,
         "per_source":rows}
 outdir=home/"evidence/real_192_blind_20261121"
 outdir.mkdir(parents=True,exist_ok=False)
 raw=outdir/"RAW_192_RUST.json"
 raw.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 def ave(split,arm):
   q=[v["arms"][arm]["total_actual_cpu_wall_us"] for v in rows if v["split"]==split]
   return sum(q)/len(q)
 blind={"schema":"quartz_cog05_C08_blind_summary_v1",
       "raw_file_sha256":sha(raw),"n_roots":len(rows),
       "teacher_test_labels_not_yet_opened":True,
       "cost_gate_mean_10pct_by_stratum":{}}
 for split in ("test_natural","test_tactical"):
   a,b=ave(split,"learned_gated"),ave(split,"learned_all_legal")
   blind["cost_gate_mean_10pct_by_stratum"][split]={
     "gated_mean_wall_us":a,"all_legal_mean_wall_us":b,
     "abs_mismatch_rel_gated":abs(a-b)/a,
     "cost_matched_10percent":abs(a-b)/a<=0.10,
     "uniform_all_legal_mean_wall_us":ave(split,"uniform_all_legal")}
 (outdir/"BLIND_COST_REPORT.json").write_text(json.dumps(blind,sort_keys=True,indent=2)+"\n")
 (outdir/"EXECUTION_MANIFEST.json").write_text(json.dumps({
   "source_commit":git_sha,"native_binary_sha256":sha(binary),
   "model_sha256":sha(model_file),"cohort_sha256":sha(cohort_file),
   "raw_sha256":sha(raw),"blind_cost_report_sha256":sha(outdir/"BLIND_COST_REPORT.json")
  },sort_keys=True,indent=2)+"\n")
 print("C08_RUST_BLIND_COMPLETED",json.dumps(blind,sort_keys=True),flush=True)
if __name__=="__main__":main()
