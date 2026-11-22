#!/usr/bin/env python3
"""C1 freeze C18, original thesis and source reference bytes for C19."""
from pathlib import Path
import subprocess,hashlib,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
FILES=["docs/THESIS.md","docs/CLAIM_LEDGER.md",
"docs/research/quartz_general_calibration_20261122/NEXT_STAGE.json",
"docs/research/quartz_general_calibration_20261122/RESULTS_AND_NEXT_KO.md",
"docs/research/quartz_general_calibration_20261122/SHADOW_MODEL_FROZEN.json",
"src/bin/quartz_c18_shadow.rs",
"quartz/experiments/c18_observation.py",
"quartz/experiments/metalevel_decision.py"]
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ROOT/FILES[4])=="2dd987989cb543c692457d0b198d23009917fd5a3001551e300a81beb868218b"
head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
record={"schema":"quartz_c19_C01_lock_v1","C18_frozen_parent":"097255bd00ffa436c3d43704ed201815e7cbb8ea",
"current_source_HEAD":head,
"original_worktree_protected":True,"native_C18_observations_not_exact_game_utility":True,
"model_target":"conditional signed utility and paid compute cost in two distinct known-utility stochastic tasks",
"live_SearchPolicy_mutation":"OUT_OF_SCOPE",
"original_files":[{"path":p,"sha256":sha(ROOT/p),"bytes":(ROOT/p).stat().st_size} for p in FILES],
"native_subagent":"NOT_RUN"}
out=HOME/"C01_SOURCE_LOCK.json"
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C19_LOCK",json.dumps({"head":head,"n_docs":len(FILES),"sha256":sha(out)}))
