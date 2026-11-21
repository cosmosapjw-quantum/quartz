#!/usr/bin/env python3
"""C09 independent result readback. No import of C08 runner/score or teacher adapter."""
import hashlib,json,math
from pathlib import Path
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def independent_win(board,side,a):
 x,y=a%15,a//15
 if board[a]:return False
 for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
  count=1
  for sign in (-1,1):
   xx,yy=x+sign*dx,y+sign*dy
   while 0<=xx<15 and 0<=yy<15 and board[yy*15+xx]==side:
    count+=1;xx+=sign*dx;yy+=sign*dy
  if count>=5:return True
 return False
def to_board(history):
 b=[0]*225
 for ply,a in enumerate(history):
  if type(a)!=int or not 0<=a<225 or b[a] or independent_win(b,1 if ply%2==0 else 2,a):
   raise ValueError("source illegal or already terminal")
  b[a]=1 if ply%2==0 else 2
 return b
def mask(actions):
 words=[0]*4
 for a in actions:
  assert 0<=a<225
  words[a//64]|=1<<(a%64)
 return words
def main():
 root=Path(__file__).resolve().parent.parent
 model=root/"evidence/model/MODEL_TRAINED.json"
 model_frozen=json.loads((root/"evidence/model/MODEL_C07_FREEZE.json").read_text())
 assert sha(model)==model_frozen["model_sha256"]
 m=json.loads(model.read_text())
 assert len(m["theta"])==13 and not m["trained_value_head"] and not m["heldout_test_teacher_labels_used"]
 assert all(math.isfinite(x) for x in m["theta"])
 trainlabel=root/"evidence/teacher_train_validation/TEACHER_TRAIN_VALIDATION.json"
 tr=json.loads(trainlabel.read_text())
 assert tr["cohort_sha256"]==m["cohort_sha256"]
 assert sha(trainlabel)==m["train_validation_labels_sha256"]
 cohort=root/"COHORT_224_FROZEN.json"
 freeze=json.loads((root/"C04_FREEZE_MANIFEST.json").read_text())
 assert sha(cohort)==freeze["source_file_sha256"]
 src=json.loads(cohort.read_text())
 assert all(len(src["splits"][split])==n for split,n in
    [("train",128),("validation",32),("test_natural",48),("test_tactical",16)])
 rawfile=root/"evidence/real_192_blind_20261121/RAW_192_RUST.json"
 raw=json.loads(rawfile.read_text())
 labelsfile=root/"evidence/teacher_heldout/TEACHER_HELDOUT.json"
 labels=json.loads(labelsfile.read_text())
 assert raw["frozen_model_sha256"]==model_frozen["model_sha256"]
 assert raw["source_cohort_sha256"]==labels["cohort_sha256"]==sha(cohort)
 assert raw["real_Rust_requests"]==192 and raw["n_source_roots"]==64
 assert raw["test_teacher_labels_used_online"]==False
 teacher={row["id"]:row for split in ["test_natural","test_tactical"]
         for row in labels["splits"][split]}
 assert len(teacher)==64
 by_source={row["id"]:row for split in ["test_natural","test_tactical"]
         for row in src["splits"][split]}
 grouped={}
 for split in ["test_natural","test_tactical"]:
  G=[]
  for item in raw["per_source"]:
   if item["split"]!=split:continue
   name=item["source_id"]
   src_row=by_source[name]
   t=teacher[name]
   assert t["history"]==item["history"]==src_row["history"]
   assert t["status"]=="OK" and t["engine_verified_pretrained"] and t["teacher_budget_ms"]==300
   history=item["history"];board=to_board(history)
   legal=[a for a,v in enumerate(board) if v==0]
   side=1 if len(history)%2==0 else 2
   wins={a for a in legal if independent_win(board,side,a)}
   if split=="test_tactical":assert wins
   scores={}
   for arm in ["learned_gated","learned_all_legal","uniform_all_legal"]:
    x=item["arms"][arm]
    assert x["rule"]=="freestyle" and x["arm"]==arm
    assert x["legal_actions"]==legal
    assert x["full_original_root_edges"]==len(legal)
    assert x["same_root_hash_preserved"] and x["same_root_ptr_preserved"]
    assert x["root_stats_and_priors_unchanged_on_activation"]
    assert x["no_online_teacher_labels"]
    assert x["first8_root_visits"]==8
    assert x["final_root_visits"]==x["root_edge_visits_sum"]
    assert x["final_root_visits"]>=8
    assert x["target_cpu_budget_us"]==20000
    assert x["total_actual_cpu_wall_us"]>0
    assert int(x["final_legal_move"]) in legal
    a0=x["initial_candidates"];reopened=x["reopened_candidates"]
    assert len(set(a0))==len(a0)
    assert len(set(reopened))==len(reopened)
    assert not (set(a0)&set(reopened))
    expected=set(a0)|set(reopened)
    if arm=="learned_gated":
      assert len(a0)==16 and len(reopened)<=4
      for a in reopened:
       assert a in wins or independent_win(board,3-side,a)
    else:assert expected==set(legal)
    if arm=="learned_gated":assert x["root_mask_after"]==mask(expected),arm
    else:assert x["root_mask_after"] in ([0]*4,mask(expected)),arm
    priors=x["root_prior_probabilities"]
    assert len(priors)==len(legal)
    assert {int(x[0]) for x in priors}==set(legal)
    assert abs(sum(float(pair[1]) for pair in priors)-1)<0.0002
    first=x["first8_original_edge_stats"]
    assert len(first)==len(legal)
    assert sum(int(e[1]) for e in first)==8
    assert {int(e[0]) for e in first}==set(legal)
    scores[arm]={"agree":int(x["final_legal_move"]==t["label"]),
                 "win":int(x["final_legal_move"] in wins),
                 "us":x["total_actual_cpu_wall_us"],"visits":x["final_root_visits"]}
   sorted_rank=item["arms"]["learned_gated"]["initial_ranked_moves"]
   assert len(sorted_rank)==len(legal) and set(sorted_rank)==set(legal)
   G.append({"id":name,"wins":len(wins),
             "top1":int(sorted_rank[0]==t["label"]),
             "top16":int(t["label"] in sorted_rank[:16]),"arms":scores})
  assert len(G)==(48 if split=="test_natural" else 16)
  n=len(G)
  a=sum(x["arms"]["learned_gated"]["us"] for x in G)/n
  b=sum(x["arms"]["learned_all_legal"]["us"] for x in G)/n
  grouped[split]={"n":n,
    "learned_teacher_top1_coverage":sum(x["top1"] for x in G),
    "learned_teacher_top16_coverage":sum(x["top16"] for x in G),
    "immediate_win_root_count":sum(x["wins"]>0 for x in G),
    "mean_us_learned_gated":a,"mean_us_learned_all_legal":b,
    "mean_us_uniform_all_legal":sum(x["arms"]["uniform_all_legal"]["us"] for x in G)/n,
    "cost_mean_mismatch_fraction":abs(a-b)/a,
    "cost_matched_10pct":abs(a-b)/a<=.1,
    "arms":{arm:{"teacher_agreement_count":sum(x["arms"][arm]["agree"] for x in G),
                 "immediate_win_choice_count":sum(x["arms"][arm]["win"] for x in G),
                 "mean_root_visits":sum(x["arms"][arm]["visits"] for x in G)/n}
            for arm in ("learned_gated","learned_all_legal","uniform_all_legal")}}
 original=json.loads((root/"evidence/real_192_blind_20261121/C08_SCORED_RESULTS.json").read_text())
 for split,x in grouped.items():
  y=original["strata"][split]
  for field in ("n","learned_teacher_top1_coverage","learned_teacher_top16_coverage",
                "immediate_win_root_count","cost_matched_10pct"):
   assert x[field]==y[field],(split,field)
  for field in ("cost_mean_mismatch_fraction","mean_us_learned_gated","mean_us_learned_all_legal",
                "mean_us_uniform_all_legal"):
   assert math.isclose(x[field],y[field],abs_tol=1e-8),(split,field,x[field],y[field])
  for arm in x["arms"]:
   for field,z in x["arms"][arm].items():
     assert math.isclose(z,y["arms"][arm][field],abs_tol=1e-8),(split,arm,field)
 report={"schema":"quartz_cog05_C09_independent_actual_readback_v1",
         "status":"PASS","frozen_model_sha256":sha(model),
         "frozen_source_sha256":sha(cohort),
         "real_Rust_raw_sha256":sha(rawfile),"finite_teacher_test_labels_sha256":sha(labelsfile),
         "n_actual_Rust_executions":192,"n_real_pretrained_Rapfi_test_labels":64,
         "root_replay_game_legality_check":True,
         "225_mask_words_and_all_initial_edges_check":True,
         "MCTS_root_prior_visits_and_same_tree_reactivation_check":True,
         "no_trained_value_head_or_undisclosed_teacher_online":True,
         "separate_per_stratum_CPU_cost_gate":True,
         "no_population_or_Elo_claims":True,"results":grouped}
 out=root/"evidence/real_192_blind_20261121/C09_INDEPENDENT_READBACK.json"
 out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print("C09_REAL_AUDIT",json.dumps({"status":"PASS","groups":grouped},sort_keys=True))
if __name__=="__main__":main()
