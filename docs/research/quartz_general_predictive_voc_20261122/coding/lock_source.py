#!/usr/bin/env python3
"""C20 source byte lock for original theory and C19 known utility tools."""
from pathlib import Path
import subprocess,hashlib,json
R=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
D=Path(__file__).resolve().parent.parent
PARENT="2a359ad049b6a0be4047e52ad281b665cf68e967"
items=["docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs",
"docs/research/quartz_general_signed_voc_20261122/NEXT_STAGE.json",
"docs/research/quartz_general_signed_voc_20261122/RESULTS_AND_NEXT_KO.md",
"docs/research/quartz_general_signed_voc_20261122/SOURCE_WORLDS_FROZEN.json",
"docs/research/quartz_general_signed_voc_20261122/SHADOW_SIGNED_MODEL_FROZEN.json",
"docs/research/quartz_general_signed_voc_20261122/coding/collect_signed_sources.py",
"quartz/experiments/c19_signed_value.py"]
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=R,text=True).strip()
assert subprocess.run(["git","merge-base","--is-ancestor",PARENT,head],cwd=R).returncode==0
j=json.loads((R/items[3]).read_text())
assert j["next_not_executed"] and j["next_research_id"]=="QUARTZ-GENERAL-20-POSTERIOR-PREDICTIVE-VOC-AND-COST"
for name in items[:3]:
 assert subprocess.check_output(["git","show",f"{PARENT}:{name}"],cwd=R)==(R/name).read_bytes()
record={"schema":"quartz_c20_C01_source_lock_v1",
 "C19_previous_head":PARENT,"C20_before_C01_head":head,
 "scoped_task":"two fresh known utility stochastic sources exact EVSI plus measured evaluator overhead",
 "source_files":[{"file":n,"sha256":sha(R/n)} for n in items],
 "protected":"THESIS, CLAIM_LEDGER, src/game.rs, developer and C16/C17/C18/C19 worktrees",
 "historical_C19_TEST_reuse_as_new_C20_TEST":False,
 "true_neural_MCTS_utility":"NOT_ESTIMABLE",
 "live_SearchPolicy_integration":"NOT_ATTEMPTED",
 "native_subagent":"NOT_RUN"}
o=D/"C01_SOURCE_LOCK.json"
o.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C20_C01_LOCK",json.dumps({"head":head,"docs":len(items),"sha256":sha(o)},sort_keys=True))
