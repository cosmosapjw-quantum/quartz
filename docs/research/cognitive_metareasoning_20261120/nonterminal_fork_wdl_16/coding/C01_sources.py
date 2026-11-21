#!/usr/bin/env python3
"""C16: byte-verified frozen trained policy and C15 native MCTS source baseline."""
from pathlib import Path
import hashlib,json,subprocess,platform
root=next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
parent=Path(__file__).resolve().parent.parent
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
sources=[]
for p in [
"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json",
"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/COHORT_224_FROZEN.json",
"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/C10_FINAL_RELEASE_MANIFEST.json",
"src/games/gomoku15.rs","src/mcts_server/prior15_cpu_server.rs",
"src/mcts/select.rs","src/mcts_server.rs","Cargo.lock","Cargo.toml"]:
 p=root/p
 assert p.is_file(),p
 sources.append({"file":str(p.relative_to(root)),"sha256":sha(p),"bytes":p.stat().st_size})
model_path=root/sources[0]["file"]
assert sha(model_path)=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
m=json.loads(model_path.read_text())
assert m["schema"]=="quartz_cog05_loglinear_policy13_v1"
assert len(m["theta"])==13 and m["value"]==0 and not m["trained_value_head"]
head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
result={"schema":"quartz_cog06_C01_immutable_source_lock_v1",
  "git_head_before_new_code":head,
  "original_C15_release":"a85bc0ded6bf67f81256e93645dc7147838c6f88",
  "model_sha256":sha(model_path),"game":"Gomoku15 Freestyle >=5, no capture",
  "full_original_root_action_count":225,"top_k":16,"initial_visits":8,
  "decision_cpu_budget_us":20000,
  "per_stratum_mean_relative_wall_mismatch_upper":0.10,
  "min_independently_generated_heldout_activations_for_effect":20,
  "no_Rapfi_oracle_in_online_endpoint":True,
  "model_training_or_replacement_allowed":False,
  "software":{"python":platform.python_version()},
  "source_files":sources}
(parent/"C01_SOURCE_LOCK.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print("C01_SOURCE_LOCK",json.dumps({"model_sha":result["model_sha256"],
"source_files":len(sources),"source_commit":head},sort_keys=True))
