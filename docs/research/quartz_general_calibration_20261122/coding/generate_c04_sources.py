#!/usr/bin/env python3
"""C4: independently seeded, actual Rust-played game histories and separate non-game worlds.
Never reads C18 Y labels or test predictions.
"""
from pathlib import Path
import hashlib,json,random,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HERE=Path(__file__).resolve().parent.parent
LOCK=json.loads((HERE/"C01_SOURCE_LOCK.json").read_text())
NATIVE=Path(LOCK["native_path"])
MODEL=Path(LOCK["model_path"])
AXES=((1,0),(0,1),(1,1),(1,-1))
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(NATIVE)==LOCK["native_sha256"] and sha(MODEL)==LOCK["model_sha256"]
def win(board,side,a):
 x=a%15;y=a//15
 for dx,dy in AXES:
  n=1
  for sign in (-1,1):
   xx=x+sign*dx;yy=y+sign*dy
   while 0<=xx<15 and 0<=yy<15 and board[yy*15+xx]==side:
    n+=1;xx+=sign*dx;yy+=sign*dy
  if n>=5:return True
 return False
def replay(history):
 b=[0]*225
 for i,a in enumerate(history):
  assert 0<=a<225 and b[a]==0
  side=1+i%2
  assert not win(b,side,a),"history must have no prior terminal"
  b[a]=side
 return b
def signature(h):
 b=replay(h)
 transformations=[]
 for flip in (False,True):
  for rotate in range(4):
   mapped=[0]*225
   for y in range(15):
    for x in range(15):
     xx,yy=x,y
     if flip:xx=14-xx
     for _ in range(rotate):xx,yy=14-yy,xx
     mapped[yy*15+xx]=b[y*15+x]
   transformations.append(bytes(mapped))
 return hashlib.sha256(min(transformations)).hexdigest()
def request(proc,history):
 q={"cmd":"research_gomoku15_fork_cpu","rule":"freestyle",
    "history":history,"arm":"learned_all_legal","seed":17,
    "cpu_budget_us":20000,"model_path":str(MODEL)}
 proc.stdin.write(json.dumps(q,separators=(",",":"))+"\n")
 proc.stdin.flush()
 line=proc.stdout.readline()
 if not line:raise RuntimeError("REAL_RUST_MCTS_TERMINATED during source generation")
 result=json.loads(line)
 if "error" in result:raise RuntimeError("REAL_NATIVE_SOURCE_ERROR "+result["error"])
 assert result["final_legal_move"] in result["legal_actions"]
 assert result["same_root_N_W_M2_P_preserved_on_activation"]
 return result
def main():
 proc=subprocess.Popen([str(NATIVE),"--server"],text=True,stdin=subprocess.PIPE,
   stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=1)
 groups=[];seen=set()
 try:
  for i in range(120):
   rng=random.Random(18022026+i*1777)
   for attempt in range(30):
    prefix=[]
    for t in range(4):
     choices=[y*15+x for y in range(4,11) for x in range(4,11)
              if y*15+x not in prefix and not win(replay(prefix),1+t%2,y*15+x)]
     if not choices:raise RuntimeError("no legal prefix")
     prefix.append(rng.choice(choices))
    canonical=signature(prefix)
    if canonical not in seen:break
   else:raise RuntimeError("could not obtain unique parent opening")
   seen.add(canonical)
   h=prefix[:]
   checkpoints=[]
   for count in range(6):
    answer=request(proc,h)
    a=int(answer["final_legal_move"])
    board=replay(h)
    if win(board,1+len(h)%2,a):
     break # source ended legitimately; do not silently add terminal board
    h.append(a)
    if len(h) in (6,8,10):
     checkpoints.append(h.copy())
   if len(checkpoints)<2:
    # Explicitly record source lack of sufficiently many nonterminal observations,
    # rather than replacing with favorable outcome-selected game.
    raise RuntimeError(f"source {i} terminated prematurely; preserve blocker not drop")
   groups.append({"id":f"rust-{i:03d}",
     "split":"train" if i<72 else "cal" if i<96 else "test",
     "parent_game_id":f"rustplayed-parent-{i:03d}",
     "source_seed":18022026+i*1777,
     "frozen_seeded_prefix":prefix,
     "canonical_parent_opening_sha256":canonical,
     "actual_native_played_move_count":len(h)-len(prefix),
     "checkpoints":checkpoints,
     "source_kind":"actual_native_Rust_MCTS_after_random_valid_initial_prefix"})
 finally:
  try:proc.stdin.write('{"cmd":"quit"}\n');proc.stdin.flush()
  except Exception:pass
  proc.stdin.close()
  proc.wait(timeout=15)
 bands=[]
 for i in range(120):
  seed=18020226+i*2719
  gen=random.Random(seed)
  k=2+i%3
  mu=[max(0.05,min(0.95,gen.betavariate(2.0,2.0))) for _ in range(k)]
  bands.append({"id":f"nongame-{i:03d}",
    "split":"train" if i<72 else "cal" if i<96 else "test",
    "parent_world_id":f"nongame-source-{i:03d}",
    "source_seed":seed,"n_arms":k,"true_utility_means_OFFLINE_ONLY":mu,
    "replicates":3,"initial_draws_each":3,"extra_draws_each":10,
    "source_kind":"sequential_Bernoulli_with_known_laws"})
 assert len(groups)==120 and len(bands)==120
 result={"schema":"quartz_general_c18_frozen_source_bank_v1",
   "mcts_real_Rust_groups":groups,"non_game_groups":bands,
   "independence":"parent groups split before outcome labels; repeated queries share parent group",
   "mcts_native_sha256":sha(NATIVE),"mcts_model_sha256":sha(MODEL),
   "mcts_source_generation_actual_native_moves":sum(g["actual_native_played_move_count"] for g in groups),
   "source_generation_not_holdout_labels":True,
   "offline_true_utility_means_never_online_features":True,
   "no_external_teacher":True}
 out=HERE/"SOURCE_GROUPS_FROZEN.json"
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print("C04_FROZEN",json.dumps({"native_parent_groups":len(groups),
 "nongame_parent_groups":len(bands),
 "real_native_played_moves":result["mcts_source_generation_actual_native_moves"],
 "snapshots":sum(len(g["checkpoints"]) for g in groups),
 "source_sha256":sha(out),
 "n_splits":{s:[sum(g["split"]==s for g in groups),sum(g["split"]==s for g in bands)] for s in ("train","cal","test")}},sort_keys=True))
if __name__=="__main__":main()
