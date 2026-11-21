#!/usr/bin/env python3
"""Blind-then-unblinded finite-teacher C08 scoring, no retuning or population inference."""
from pathlib import Path
import hashlib,json,statistics
from freeze_cohorts import replay,win
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
 d=Path(__file__).resolve().parent.parent
 c=json.loads((d/"COHORT_224_FROZEN.json").read_text())
 m=json.loads((d/"evidence/model/MODEL_TRAINED.json").read_text())
 f=json.loads((d/"evidence/model/MODEL_C07_FREEZE.json").read_text())
 assert f["model_sha256"]==sha(d/"evidence/model/MODEL_TRAINED.json")
 raw=d/"evidence/real_192_blind_20261121/RAW_192_RUST.json"
 observed=json.loads(raw.read_text())
 label=d/"evidence/teacher_heldout/TEACHER_HELDOUT.json"
 teacher=json.loads(label.read_text())
 assert teacher["cohort_sha256"]==observed["source_cohort_sha256"]
 assert observed["frozen_model_sha256"]==f["model_sha256"]
 sources={src["id"]:src for split in ("test_natural","test_tactical") for src in c["splits"][split]}
 by_id={row["id"]:row for split in ("test_natural","test_tactical")
        for row in teacher["splits"][split]}
 assert set(by_id)==set(sources)
 arms=("learned_gated","learned_all_legal","uniform_all_legal")
 rows=[]
 for record in observed["per_source"]:
  key=record["source_id"]
  src=sources[key]
  label_row=by_id[key]
  assert label_row["history"]==src["history"]==record["history"]
  assert label_row["status"]=="OK" and label_row["engine_verified_pretrained"]
  b=replay(src["history"])
  side=1 if len(src["history"])%2==0 else 2
  exact_wins=[a for a,v in enumerate(b) if v==0 and win(b,side,a)]
  teach=int(label_row["label"])
  assert teach not in src["history"]
  arm_sc={}
  for arm in arms:
   z=record["arms"][arm]
   chosen=int(z["final_legal_move"])
   assert chosen not in src["history"]
   arm_sc[arm]={"selected":chosen,"teacher_agree":int(chosen==teach),
                "immediate_win":int(chosen in exact_wins),
                "wall_us":int(z["total_actual_cpu_wall_us"]),
                "root_visits":int(z["final_root_visits"])}
  order=record["arms"]["learned_gated"]["initial_ranked_moves"]
  rows.append({"id":key,"split":src["stratum"],"test_split":record["split"],
              "teacher":teach,"known_immediate_wins":exact_wins,
              "learned_prior_rank_of_teacher":order.index(teach)+1,
              "teacher_in_learned_top16":teach in order[:16],
              "arms":arm_sc})
 results={"schema":"quartz_cog05_C08_heldout_finite_teacher_cost_comparison_v1",
          "source_cohort_sha256":observed["source_cohort_sha256"],
          "model_sha256":f["model_sha256"],"native_binary_sha256":observed["compiled_native_binary_sha256"],
          "source_blind_Rust_sha256":sha(raw),
          "source_finite_teacher_label_sha256":sha(label),
          "primary_criterion":"learned_gated vs learned_all_legal physical wall cost within 10 percent PER STRATUM",
          "exact_minimax":False,"full_games_tested":False,
          "conclusions_scoped_to_frozen_synthetic_roots":True,"strata":{},
          "per_source_scored":rows}
 for split in ("test_natural","test_tactical"):
  group=[x for x in rows if x["test_split"]==split]
  n=len(group)
  def avg(k):return sum(x["arms"][k]["wall_us"] for x in group)/n
  a,b=avg("learned_gated"),avg("learned_all_legal")
  summary={"n":n,"immediate_win_root_count":sum(bool(x["known_immediate_wins"]) for x in group),
           "learned_teacher_top1_coverage":sum(x["learned_prior_rank_of_teacher"]==1 for x in group),
           "learned_teacher_top16_coverage":sum(x["teacher_in_learned_top16"] for x in group),
           "cost_mean_mismatch_fraction":abs(a-b)/a,
           "cost_matched_10pct":abs(a-b)/a<=0.1,
           "mean_us_learned_gated":a,"mean_us_learned_all_legal":b,
           "mean_us_uniform_all_legal":avg("uniform_all_legal"),
           "arms":{}}
  for arm in arms:
   summary["arms"][arm]={
      "teacher_agreement_count":sum(x["arms"][arm]["teacher_agree"] for x in group),
      "immediate_win_choice_count":sum(x["arms"][arm]["immediate_win"] for x in group),
      "mean_root_visits":sum(x["arms"][arm]["root_visits"] for x in group)/n}
  A,B="learned_gated","learned_all_legal"
  summary["gated_minus_all_teacher_agree_count"]=(
      summary["arms"][A]["teacher_agreement_count"]-summary["arms"][B]["teacher_agreement_count"])
  summary["gated_minus_all_tactical_win_count"]=(
      summary["arms"][A]["immediate_win_choice_count"]-summary["arms"][B]["immediate_win_choice_count"])
  summary["admissible_CPU_matched_claim"]=summary["cost_matched_10pct"]
  results["strata"][split]=summary
 out=d/"evidence/real_192_blind_20261121/C08_SCORED_RESULTS.json"
 out.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n")
 concise={"strata":results["strata"],"source_raw_hash":results["source_blind_Rust_sha256"],
  "test_teacher_hash":results["source_finite_teacher_label_sha256"]}
 print("C08_FINITE_COMPARISON",json.dumps(concise,sort_keys=True),flush=True)
if __name__=="__main__":main()
