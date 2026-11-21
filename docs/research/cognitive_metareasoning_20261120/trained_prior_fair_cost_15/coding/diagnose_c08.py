#!/usr/bin/env python3
"""Post-outcome exploratory paired diagnostics (NOT a predeclared acceptance gate)."""
import json,collections
from pathlib import Path
d=Path(__file__).resolve().parent.parent
raw=json.loads((d/"evidence/real_192_blind_20261121/RAW_192_RUST.json").read_text())
score=json.loads((d/"evidence/real_192_blind_20261121/C08_SCORED_RESULTS.json").read_text())
S={x["id"]:x for x in score["per_source_scored"]}
O={}
for split in ("test_natural","test_tactical"):
 g=[x for x in raw["per_source"] if x["split"]==split]
 out={"n":len(g),
      "same_final_move_gated_vs_learned_full":0,
      "different_final_move_gated_vs_learned_full":0,
      "gated_reopened_at_least_one":0,
      "total_reopened_actions":0,
      "gated_teacher_gains_vs_learned_full":0,
      "gated_teacher_losses_vs_learned_full":0,
      "learned_full_teacher_gains_vs_uniform":0,
      "learned_full_teacher_losses_vs_uniform":0,
      "learned_model_winning_moves_not_exact_teacher":0,
      "teacher_terminal_win_count":0,
      "immediate_win_counts":[],
      "gated_root_vs_full_root_visit_difference_mean":0}
 visits=0
 for x in g:
  A=x["arms"]["learned_gated"];B=x["arms"]["learned_all_legal"]
  a=A["final_legal_move"];b=B["final_legal_move"]
  out["same_final_move_gated_vs_learned_full"]+=int(a==b)
  out["different_final_move_gated_vs_learned_full"]+=int(a!=b)
  out["gated_reopened_at_least_one"]+=int(len(A["reopened_candidates"])>0)
  out["total_reopened_actions"]+=len(A["reopened_candidates"])
  o=S[x["source_id"]]
  out["gated_teacher_gains_vs_learned_full"]+=int(o["arms"]["learned_gated"]["teacher_agree"]==1 and o["arms"]["learned_all_legal"]["teacher_agree"]==0)
  out["gated_teacher_losses_vs_learned_full"]+=int(o["arms"]["learned_gated"]["teacher_agree"]==0 and o["arms"]["learned_all_legal"]["teacher_agree"]==1)
  out["learned_full_teacher_gains_vs_uniform"]+=int(o["arms"]["learned_all_legal"]["teacher_agree"]==1 and o["arms"]["uniform_all_legal"]["teacher_agree"]==0)
  out["learned_full_teacher_losses_vs_uniform"]+=int(o["arms"]["learned_all_legal"]["teacher_agree"]==0 and o["arms"]["uniform_all_legal"]["teacher_agree"]==1)
  out["learned_model_winning_moves_not_exact_teacher"]+=int(o["arms"]["learned_all_legal"]["immediate_win"]==1 and o["arms"]["learned_all_legal"]["teacher_agree"]==0)
  out["teacher_terminal_win_count"]+=int(o["teacher"] in o["known_immediate_wins"])
  out["immediate_win_counts"].append(len(o["known_immediate_wins"]))
  visits+=A["final_root_visits"]-B["final_root_visits"]
 out["gated_root_vs_full_root_visit_difference_mean"]=visits/len(g)
 out["immediate_win_counts_summary"]=dict(collections.Counter(out.pop("immediate_win_counts")))
 O[split]=out
path=d/"evidence/real_192_blind_20261121/C08_POSTHOC_DESCRIPTIVE.json"
path.write_text(json.dumps({"status":"EXPLORATORY_AFTER_TEST_UNBLINDING","strata":O},sort_keys=True,indent=2)+"\n")
print("C08_POSTHOC",json.dumps(O,sort_keys=True))
