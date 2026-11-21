#!/usr/bin/env python3
"""Prespecified rule-certified action scoring of blind C16 source-root traces."""
import hashlib,json,sys
from pathlib import Path
home=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from freeze_sources16 import source_forks
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
 cohort=json.loads((home/"COHORT_40_FROZEN.json").read_text())
 paths={x["id"]:x for key in ("constructed_fork_positive","independent_MCTS_played_roots")
      for x in cohort["data_splits"][key]}
 rawfile=home/"evidence/real_root16_blind/RAW_ROOT_195.json"
 raw=json.loads(rawfile.read_text())
 assert raw["source_n_constructed_heldout"]==23
 assert raw["source_n_MCTS_played_heldout"]==16
 assert raw["n_actual_native_root_requests"]==195
 assert raw["frozen_source_cohort_sha256"]==sha(home/"COHORT_40_FROZEN.json")
 assert raw["model_sha256"]==cohort["C15_policy_model_sha256"]
 summaries={}
 per_by_stratum={}
 for stratum in ("constructed","MCTS_played"):
   records=[r for r in raw["per_source"] if ("engineered" in r["source_type"])==(stratum=="constructed")]
   n=len(records);assert n==(23 if stratum=="constructed" else 16)
   per=[]
   for r in records:
     source=paths[r["source_id"]]
     assert source["history"]==r["history"]
     possibilities={a["action"] for a in source_forks(r["history"])}
     prior_top16=r["arms"]["learned_all_legal"]["initial_ranked_actions"][:16]
     omitted=possibilities-set(prior_top16)
     assert len(possibilities)==len(set(possibilities))
     if stratum=="constructed":
       assert omitted and len(omitted)>=2
       assert set(a["action"] for a in source["omitted_forks_before_experiment"])==omitted
     res={}
     for arm,x in r["arms"].items():
       move=x["final_legal_move"]
       res[arm]={"move":move,
           "rule_certified":int(move in possibilities),
           "in_prior_top16":int(move in prior_top16),
           "measured_wall_us":x["total_actual_cpu_wall_us"],
           "number_visits":x["final_root_visits"],
           "n_added_actions":len(x["added_actions"]),
           "proof_override":int(x["explicit_proof_choice_override"])}
     per.append({"id":r["source_id"],"n_certified":len(possibilities),
          "n_certified_omitted":len(omitted),"arms":res})
   aa=("learned_all_legal","learned_fork_mask","learned_fork_proof_choice",
       "learned_sham_mask","learned_top16")
   def stats(arm,k):return [row["arms"][arm][k] for row in per]
   get=lambda arm,k:sum(stats(arm,k))/n
   a,b=get("learned_fork_mask","measured_wall_us"),get("learned_all_legal","measured_wall_us")
   per_by_stratum[stratum]=per
   summaries[stratum]={"n":n,
      "n_roots_with_omitted_certified":sum(row["n_certified_omitted"]>0 for row in per),
      "mean_n_omitted_certified":sum(row["n_certified_omitted"] for row in per)/n,
      "n_fork_mask_actually_triggered":sum(v>0 for v in stats("learned_fork_mask","n_added_actions")),
      "mask_vs_full_action_difference_count":sum(row["arms"]["learned_fork_mask"]["move"]!=row["arms"]["learned_all_legal"]["move"] for row in per),
      "proof_choice_vs_full_difference_count":sum(row["arms"]["learned_fork_proof_choice"]["move"]!=row["arms"]["learned_all_legal"]["move"] for row in per),
      "mask_vs_full_certified_gain":sum(row["arms"]["learned_fork_mask"]["rule_certified"]>row["arms"]["learned_all_legal"]["rule_certified"] for row in per),
      "mask_vs_full_certified_loss":sum(row["arms"]["learned_fork_mask"]["rule_certified"]<row["arms"]["learned_all_legal"]["rule_certified"] for row in per),
      "proof_choice_vs_full_certified_gain":sum(row["arms"]["learned_fork_proof_choice"]["rule_certified"]>row["arms"]["learned_all_legal"]["rule_certified"] for row in per),
      "proof_choice_vs_full_certified_loss":sum(row["arms"]["learned_fork_proof_choice"]["rule_certified"]<row["arms"]["learned_all_legal"]["rule_certified"] for row in per),
      "mean_CPU_wall_mask_us":a,"mean_CPU_wall_full_us":b,
      "mean_CPU_absolute_relative_difference":abs(a-b)/a,
      "predeclared_10pct_CPU_cost_gate":abs(a-b)/a<=.1,
      "per_arm":{arm:{"n_certified_final_moves":sum(stats(arm,"rule_certified")),
         "mean_wall_us":get(arm,"measured_wall_us"),
         "mean_root_visits":get(arm,"number_visits"),
         "n_proof_override":sum(stats(arm,"proof_override"))} for arm in aa}}
 summary={"schema":"quartz_cog06_C08_finite_root_rule_scored_v1",
  "source_real_Rust_sha256":sha(rawfile),
  "data_40_frozen_sha256":sha(home/"COHORT_40_FROZEN.json"),
  "no_Rapfi_heldout_labels_used":True,
  "treated_source_groups_constructed_are_shared_template_not_random_game_population":True,
  "population_conditional_reopening_effect_admission":"NOT_ESTIMABLE without >=20 separately generated played-game activated sources",
  "strata":summaries,"per_source_by_stratum":per_by_stratum}
 # Separate per-source output for two strata should not be conflated; main summaries are sufficient and verified from raw in C9.
 out=home/"evidence/real_root16_blind/C08_ROOT_RULE_SCORE.json"
 out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 print("C08_ROOT_RULE_SCORE",json.dumps(summaries,sort_keys=True),flush=True)
if __name__=="__main__":main()
