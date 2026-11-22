#!/usr/bin/env python3
"""C01 immutable original project and C21 source lock before C22 mutation."""
from pathlib import Path
import subprocess,hashlib,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
BASE="351c63ae810b1fa633fe2a75440bf7d0929f92e4"
HOME=Path(__file__).resolve().parent.parent
FILES=["docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs",
"docs/research/quartz_general_cdf_native_20261122/NEXT_STAGE.json",
"docs/research/quartz_general_cdf_native_20261122/RESULTS_AND_NEXT_KO.md",
"docs/research/quartz_general_cdf_native_20261122/evidence/C09/NATIVE_GRID_TRAIN_ONLY_REPORT.json",
"src/bin/quartz_c21_fast_evsi.rs",
"src/bin/quartz_c20_shadow.rs"]
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert subprocess.run(["git","merge-base","--is-ancestor",BASE,head],cwd=ROOT).returncode==0
assert subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="research/quartz-general-c22-inprocess-gate-20261122"
for p in FILES[:3]:
 assert (ROOT/p).read_bytes()==subprocess.check_output(["git","show",f"{BASE}:{p}"],cwd=ROOT)
nextj=json.loads((ROOT/FILES[3]).read_text())
assert nextj["next_not_executed"] and nextj["next_research_id"]=="QUARTZ-GENERAL-22-INPROCESS-VOC-GATE-AND-REAL-COMPUTE-ACTION"
record={"schema":"quartz_C22_C01_original_source_lock_v1",
"parent_C21_exact_HEAD":BASE,"R10_HEAD_at_C01":head,
"protected_originals":[{"path":n,"sha256":sha(ROOT/n),"bytes":(ROOT/n).stat().st_size} for n in FILES],
"original_goal":"generic rational computation allocation/STOP, not game-specific tactics",
"C22_goal":"real inprocess known-utility COMPUTE or STOP with actual zero future RNG at STOP",
"source_neural_MCTS_true_utility":"NOT_ESTIMABLE",
"previous_C16_C21_worktrees_unmodified":True,
"native_subagent":"NOT_RUN"}
out=HOME/"C01_SOURCE_LOCK.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C22_C01_SOURCE",json.dumps({"n_files":len(FILES),"sha256":sha(out),"source_R10":head},sort_keys=True))
