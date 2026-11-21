#!/usr/bin/env python3
"""Real paired side-swapped 15x15 Gomoku games, strictly actual Quartz Rust actions."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from collections import Counter
ROOT=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
HOME=Path(__file__).resolve().parent.parent
C15=ROOT/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15"
MODEL=C15/"evidence/model/MODEL_TRAINED.json"
sys.path.insert(0,str(C15/"coding"))
from freeze_cohorts import replay,win
BRANCH="research/quartz-cog06-nonterminal-fork-wdl-20261121"
EXPECTED_SOURCE="8349441f247f720028f8534ac4abf656be56f8ac58778191de900722b4af1049"
EXPECTED_MODEL="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
ACTIVE=("learned_fork_mask","learned_fork_proof_choice")
BASELINE="learned_all_legal"
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def published():
 if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"]).strip():
   raise RuntimeError("paired game science requires immutable published clean branch")
 head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
 assert subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"],text=True).strip()==BRANCH
 remote=subprocess.check_output(["git","-C",str(ROOT),"ls-remote","--heads","origin",BRANCH],text=True).split()[0]
 if head!=remote:raise RuntimeError("paired game running from unconfirmed Git head")
 return head
def active_scores(g):
    policy=g["active_policy"]
    won=g["winner"]
    if won is None:return .5
    return float(g["black_policy"]==policy) if won==1 else float(g["white_policy"]==policy)
def play(proc,game_id,opening,arm,color):
  black=arm if color==1 else BASELINE
  white=arm if color==2 else BASELINE
  history=opening["history"].copy()
  choices=[];terminal=None;winner=None
  for ply in range(len(history),225):
    board=replay(history)
    side=1 if ply%2==0 else 2
    active=black if side==1 else white
    if len(history)!=ply:raise RuntimeError("ply not paired to original history")
    query={"cmd":"research_gomoku15_fork_cpu",
      "rule":"freestyle","history":history,"arm":active,
      "seed":17,"cpu_budget_us":20000,"model_path":str(MODEL)}
    proc.stdin.write(json.dumps(query,separators=(",",":"))+"\n");proc.stdin.flush()
    raw=proc.stdout.readline()
    if not raw:raise RuntimeError("native Rust process terminated mid-game")
    output=json.loads(raw)
    if "error" in output:
       raise RuntimeError(f"native Rust game error id={game_id} ply={ply}: {output['error']}")
    action=int(output["final_legal_move"])
    if action<0 or action>=225 or board[action]:
       raise RuntimeError(f"illegal native output {game_id} ply={ply} action={action}")
    if output["final_root_visits"]!=output["root_edge_visit_sum"] or not output["same_root_N_W_M2_P_preserved_on_activation"]:
       raise RuntimeError("failed same-root invariant in actual game")
    if output["total_actual_cpu_wall_us"]<=0:raise RuntimeError("missing actual per-move CPU time")
    is_win=win(board,side,action)
    choices.append({"ply":ply,"side":side,"policy":active,
      "flat_action":action,"source_history_length":len(history),
      "native_active_wall_us":int(output["total_actual_cpu_wall_us"]),
      "native_mcts_visits":int(output["final_root_visits"]),
      "reopened_count":len(output["added_actions"]),
      "proof_count":len(output["certificates_omitted"]),
      "explicit_proof_override":bool(output["explicit_proof_choice_override"]),
      "actual_certificate_selected":bool(output["selected_is_certified_fork"]),
      "source_rule_precondition":bool(output["source_rule_preconditions_verified"]),
      "terminal_win_by_full_rules":is_win})
    history.append(action)
    if is_win:
       winner=side;terminal="FIVE_IN_ROW";break
    if len(history)==225:
       terminal="FULL_BOARD_DRAW";winner=None;break
  if terminal is None:raise RuntimeError("incomplete game replay without terminal")
  return {"schema":"quartz_cog06_completed_real_pair_game_v1",
     "game_id":game_id,"opening_id":opening["id"],
     "original_opening_history":opening["history"],
     "source_parent_game_id":opening["parent_game_id"],
     "active_policy":arm,"active_color":color,
     "black_policy":black,"white_policy":white,
     "winner":winner,"terminal_reason":terminal,
     "final_history":history,"n_moves_after_opening":len(choices),
     "per_move_real_native":choices,"active_score":active_scores({"active_policy":arm,"winner":winner,"black_policy":black,"white_policy":white})}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--binary",type=Path,required=True)
 args=ap.parse_args()
 git=published()
 assert sha(MODEL)==EXPECTED_MODEL
 f=HOME/"COHORT_40_FROZEN.json"
 assert sha(f)==EXPECTED_SOURCE
 src=json.loads(f.read_text())["data_splits"]
 look={o["id"]:o for o in src["independent_MCTS_played_roots"]}
 first=json.loads(f.read_text())["data_splits"]["paired_fullgame_openings"]
 assert len(first)==8 and len(set(first))==8
 schedule=[]
 for k,id in enumerate(first):
   for arm in ACTIVE:
      for color in ([1,2] if k%2==0 else [2,1]):
       schedule.append((f"{id}/{arm}/color{color}",look[id],arm,color))
    # Each opening is a paired cluster; per-source mode order is frozen.
 outdir=HOME/"evidence/real_games16"
 outdir.mkdir(parents=True,exist_ok=True)
 progress=outdir/"COMPLETED_GAMES.jsonl"
 completed={}
 if progress.exists():
    for line in progress.read_text().splitlines():
      if not line.strip():continue
      row=json.loads(line)
      if row["game_id"] in completed:raise RuntimeError("duplicate append-only game row")
      completed[row["game_id"]]=row
    assert set(completed).issubset({s[0] for s in schedule})
 exe=args.binary.resolve()
 assert exe.is_file()
 frozen_binary=sha(exe)
 proc=subprocess.Popen([str(exe),"--server"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
   stderr=subprocess.DEVNULL,text=True,bufsize=1,env={**os.environ,"RAYON_NUM_THREADS":"1",
         "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
 try:
    for game_id,opening,arm,color in schedule:
      if game_id in completed:continue
      result=play(proc,game_id,opening,arm,color)
      with progress.open("a") as f:
        f.write(json.dumps(result,separators=(",",":"),sort_keys=True)+"\n");f.flush()
      completed[game_id]=result
      print("GAME_FINISHED",len(completed),"/",len(schedule),game_id,
          "WDL",result["active_score"],"n_new_moves",result["n_moves_after_opening"],flush=True)
 finally:
    try:proc.stdin.write(json.dumps({"cmd":"quit"})+"\n");proc.stdin.flush()
    except Exception:pass
    proc.stdin.close();proc.wait(timeout=10)
 assert len(completed)==len(schedule)
 games=[completed[ident] for ident,*_ in schedule]
 report={"schema":"quartz_cog06_C08_actual_complete_32_paired_games_v1",
  "source_git_commit":git,"native_binary_sha256":frozen_binary,
  "model_sha256":sha(MODEL),"frozen_source_cohort_sha256":sha(f),
  "n_legal_complete_games":len(games),"n_source_opening_clusters":8,
  "paired_two_colors_per_active_arm":True,
  "finite_game_WDL_not_elo":True,"per_move_native_budget_target_us":20000,
  "native_decision_wall_excludes_model_disk_loading_JSON_output":True,
  "completed_source_progress_sha256":sha(progress),
  "games":games}
 raw=outdir/"RAW_32_ACTUAL_COMPLETE_GAMES.json"
 raw.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 summary={"schema":"quartz_cog06_C08_pair_WDL_summary_v1",
  "source_raw_sha256":sha(raw),"n_games":len(games),
  "n_paired_source_openings":8,"arms":{}}
 for arm in ACTIVE:
   items=[x for x in games if x["active_policy"]==arm]
   assert len(items)==16
   scores=[x["active_score"] for x in items]
   outcome=Counter(scores)
   summary["arms"][arm]={
     "n_games":len(items),"W":outcome[1.],"D":outcome[.5],"L":outcome[0.],
     "mean_game_score":sum(scores)/len(scores),
     "n_fork_reopen_events":sum(sum(c["reopened_count"]>0 for c in item["per_move_real_native"]) for item in items),
     "n_proof_overrides":sum(sum(c["explicit_proof_override"] for c in item["per_move_real_native"]) for item in items),
     "mean_native_cpu_per_move_us":sum(c["native_active_wall_us"] for item in items for c in item["per_move_real_native"])
                            /sum(len(item["per_move_real_native"]) for item in items),
     "n_actual_moves_after_openings":sum(len(item["per_move_real_native"]) for item in items),
     "pair_scores_by_opening":[sum(x["active_score"] for x in items if x["opening_id"]==id)/2 for id in first]
    }
 (outdir/"C08_WDL_SUMMARY.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
 print("C08_ACTUAL_PAIRED_WDL",json.dumps(summary,sort_keys=True),flush=True)
if __name__=="__main__":main()
