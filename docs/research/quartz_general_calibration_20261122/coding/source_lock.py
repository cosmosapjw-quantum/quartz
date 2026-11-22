#!/usr/bin/env python3
"""C1: verify immutable previous source and original project authority."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
C16=Path("/home/cosmosapjw/quantum/research/quartz_cog06_20261121")
NATIVE=C16/"target/release/mcts_demo"
MODEL=C16/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(NATIVE)=="1168f2979936a283a9e7e613b869fd30ec9aba826e15ac228c6ee9d7e7e4c4d3"
assert sha(MODEL)=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
docs=["docs/THESIS.md","docs/CLAIM_LEDGER.md","docs/research/quartz_general_metareasoning_20261122/QUARTZ_GENERAL_MASTER_PLAN_v1.md","docs/research/quartz_general_metareasoning_20261122/NEXT_STAGE.json","src/mcts/policy/trait_def.rs","quartz/experiments/metalevel_decision.py"]
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
data={"schema":"quartz_general_c18_C01_source_lock","parent":"e4dc4d6a0d5db03189d2f8465f6d5f019018f20c","source_HEAD":head,"native_path":str(NATIVE),"native_sha256":sha(NATIVE),"model_path":str(MODEL),"model_sha256":sha(MODEL),"docs":[{"file":x,"sha256":sha(ROOT/x)} for x in docs],"science_limit":"MCTS first8 modal-vs-final revision is NOT exact regret; non-game task has offline true utility","no_teacher_online":True}
p=HOME/"C01_SOURCE_LOCK.json"
p.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
print("C01_SHA_VERIFIED",json.dumps({"head":head,"docs":len(docs),"binary":sha(NATIVE)},sort_keys=True))
