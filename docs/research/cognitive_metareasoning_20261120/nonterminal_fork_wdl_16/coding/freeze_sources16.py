#!/usr/bin/env python3
"""Freeze 24 deliberately constructed forks + 16 real Quartz-played source roots, no C16 results."""
from pathlib import Path
import hashlib,json,random,subprocess,sys
root=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
d=Path(__file__).resolve().parent.parent
prev=root/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15"
sys.path.insert(0,str(prev/"coding"))
from policy15 import position_features,predict_logits
from freeze_cohorts import replay,win,canon

N=15
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def rootwin(board,side):
 return {a for a in range(225) if board[a]==0 and win(board,side,a)}
def proof_fork(board,side,a):
 if board[a]!=0 or win(board,side,a):return []
 q=board.copy();q[a]=side
 ends=[]
 for v in range(225):
  if q[v]==0 and win(q,side,v):ends.append(v)
 return ends if len(ends)>=2 else []
def source_forks(h):
 board=replay(h)
 side=1 if len(h)%2==0 else 2
 if rootwin(board,side) or rootwin(board,3-side):return []
 actions,z=predict_logits([0]*13,h) if False else predict_logits(frozen_theta,h)
 ranked=[a for a,_ in sorted(zip(actions.tolist(),z.tolist()),key=lambda x:(-x[1],x[0]))]
 out=[]
 for ix,a in enumerate(ranked):
  ends=proof_fork(board,side,a)
  if ends:out.append({"action":a,"prior_rank":ix+1,"winning_endpoints":ends})
 return out

model=prev/"evidence/model/MODEL_TRAINED.json"
assert sha(model)=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
frozen_theta=json.loads(model.read_text())["theta"]

def build_positive(seed):
 rng=random.Random(seed)
 for _ in range(500):
  # Eight separated horizontal threes (16 fork moves) plus one distant vertical three.
  b=[0]*225
  black=[]
  for yy in range(0,15,2):
    xx=rng.choice([2,3,4,5])
    for i in range(3):
      a=yy*15+xx+i
      assert not b[a]
      b[a]=1;black.append(a)
  for yy in (5,6,7):
    a=yy*15+13
    b[a]=1;black.append(a)
  assert len(black)==27
  white=[]
  pool=[yy*15+xx for yy in range(15) for xx in (10,11,12,14) if b[yy*15+xx]==0]
  rng.shuffle(pool)
  for a in pool:
    if len(white)==len(black):break
    b[a]=2
    if rootwin(b,2) or rootwin(b,1):
      b[a]=0;continue
    white.append(a)
  if len(white)!=27:continue
  if rootwin(b,1) or rootwin(b,2):continue
  h=[]
  rng.shuffle(black)
  rng.shuffle(white)
  for x,y in zip(black,white):h.extend((x,y))
  try: check=replay(h)
  except ValueError:continue
  assert check==b
  forks=source_forks(h)
  missed=[x for x in forks if x["prior_rank"]>16]
  if len(forks)>=18 and len(missed)>=2:
    return h,forks,missed
 raise RuntimeError(f"no valid fork-positive root for seed={seed}")

def run_selfplay_origins(n=16):
    # C15 executable is an immutable local derivative of its published source.
    release=Path("/home/cosmosapjw/quantum/research/quartz_cog05_20261121/target/release/mcts_demo")
    assert release.exists()
    assert sha(release)=="1bf102650e28c9f0bbe9515f2f6e7d1d0c00a15f2646f78498ea79a9710d8527"
    proc=subprocess.Popen([str(release),"--server"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,text=True,bufsize=1)
    rows=[];seen=set()
    try:
      for idx in range(n):
        rng=random.Random(202611210616+idx*1109)
        for attempt in range(15):
          # Deterministic source prefix, followed by actually played QUARTZ-versus-QUARTZ moves.
          h=[]
          for k in range(4):
            region=[y*15+x for y in range(4,11) for x in range(4,11) if y*15+x not in h]
            rng.shuffle(region)
            h.append(next(a for a in region if not win(replay(h),1 if k%2==0 else 2,a)))
          try: replay(h)
          except ValueError:continue
          selected=[]
          for _ in range(8):
            request={"cmd":"research_gomoku15_prior_cpu","rule":"freestyle",
                  "history":h,"arm":"learned_all_legal","seed":17,"cpu_budget_us":20000,
                  "model_path":str(model)}
            proc.stdin.write(json.dumps(request,separators=(",",":"))+"\n");proc.stdin.flush()
            line=proc.stdout.readline()
            if not line:raise RuntimeError("C15 real server closed during source selfplay")
            output=json.loads(line)
            if "error" in output:raise RuntimeError("C15 live source search error:"+str(output["error"]))
            a=int(output["final_legal_move"])
            if win(replay(h),1 if len(h)%2==0 else 2,a):break
            h.append(a);selected.append(a)
          if len(h)!=12:continue
          c=canon(h)
          if c in seen:continue
          seen.add(c)
          rows.append({"id":f"played-{idx:03d}",
           "parent_game_id":f"mcts_selfplay_prefix_{idx:03d}",
           "source_seed":202611210616+idx*1109,
           "initial_seeded_prefix":h[:4],"actual_quartz_MCTS_moves":h[4:],
           "history":h,"rule":"freestyle",
           "source_type":"original_sequential_quartz_MCTS_play_after_frozen_seed_prefix",
           "family":"distinct_seeded_MCTS_played_game"})
          break
        else:raise RuntimeError(f"failed actual played root {idx}")
    finally:
      try:proc.stdin.write(json.dumps({"cmd":"quit"})+"\n");proc.stdin.flush()
      except Exception:pass
      proc.stdin.close();proc.wait(timeout=10)
    assert len(rows)==n
    return rows,str(release)

def main():
    constructed=[];keys=set()
    for i in range(24):
      seed=202611210777+10007*i
      h,forks,missed=build_positive(seed)
      c=canon(h)
      assert c not in keys,"constructed equivalent source"
      keys.add(c)
      constructed.append({"id":f"fork-{i:03d}","history":h,
        "parent_game_id":f"constructed_independent_rng_{seed}",
        "source_seed":seed,"rule":"freestyle","source_type":"engineered_fork_positive_nonIID",
        "forks_before_experiment":forks,"omitted_forks_before_experiment":missed,
        "source_groups_not_independent_motifs":True})
    played,release=run_selfplay_origins(16)
    for x in played:
      c=canon(x["history"])
      assert c not in keys
      keys.add(c)
    result={"schema":"quartz_cog06_source_cohorts_frozen_v1",
        "parent_C15_commit":"a85bc0ded6bf67f81256e93645dc7147838c6f88",
        "C15_real_native_immutable_binary":release,
        "C15_real_native_binary_sha256":sha(release),
        "C15_policy_model_sha256":sha(model),
        "original_user_Git_untouched":True,
        "rule":"freestyle15","construction_is_stratified_not_game_population":True,
        "constructed_motif_shared_between_sources":True,
        "all_D4_canonical_boards_distinct":True,
        "no_external_teacher_oracle_or_test_label_used":True,
        "data_splits":{"constructed_fork_positive":constructed,
                       "independent_MCTS_played_roots":played,
                       "paired_fullgame_openings":[x["id"] for x in played[:8]]},
        "predeclared_min_constructed_activated_source_groups":20,
        "primary_heldout_sanity":"all 24 constructed positive sources are deliberately conditional, not general natural coverage"}
    outfile=d/"COHORT_40_FROZEN.json"
    outfile.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("C04_COHORT_FROZEN",json.dumps({"n_constructed":len(constructed),
        "n_MCTS_played":len(played),
        "n_balanced_game_openings":8,
        "min_omitted_constructed":min(len(x["omitted_forks_before_experiment"]) for x in constructed),
        "n_distinct_source_groups":len(keys),"source_sha256":sha(outfile),
        "MCTS_original_native_binary":result["C15_real_native_binary_sha256"]},sort_keys=True))
if __name__=="__main__":main()
