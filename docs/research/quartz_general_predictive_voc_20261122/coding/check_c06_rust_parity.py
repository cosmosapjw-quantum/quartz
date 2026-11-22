#!/usr/bin/env python3
"""C06 independent Rust finite hidden-channel model vs Python exact EVSI."""
from pathlib import Path
import subprocess,tempfile,random,hashlib,json
from quartz.experiments.c20_predictive_value import sensor_evsi
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BIN=ROOT/"target/release/quartz_c20_shadow"
BANK=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
source=json.loads(BANK.read_text())
sensor=[w for w in source["worlds"] if w["family"]=="finite_sensor"]
assert len(sensor)==120
rng=random.Random(202611221918)
cases=[]
for w in sensor[:96]:
  cases.append({"prior":w["initial_prior"],"likelihood":w["observation_likelihood_state_outcome_PUBLIC"],
    "utilities":w["utility_state_action_matrix_PUBLIC"],"extra":rng.choice([0,1,2,4,8])})
# All Rust inputs exclude offline true hidden state, future outcomes and decision labels.
assert all(set(c)=={"prior","likelihood","utilities","extra"} for c in cases)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/"c20_model_only.json"
 p.write_text(json.dumps({"cases":cases},sort_keys=True))
 result=json.loads(subprocess.check_output([str(BIN),str(p)],text=True))
native=result["predictions"]
assert result["no_SearchPolicy_mutation"] is True
assert result["beta_binomial_Rust_port"] is False
assert len(native)==96
diff=[]
for args,r in zip(cases,native):
 py=sensor_evsi(args["prior"],args["likelihood"],args["utilities"],args["extra"])
 for key in ("evsi","before_best","after_expected_best","total_predictive_mass"):
  diff.append(abs(float(py[key])-float(r[key])))
 assert py["outcome_count"]==r["outcome_count"]
assert max(diff)<=1e-10,max(diff)
res={"schema":"quartz_general_c20_C06_independent_Rust_python_sensor_PARITY",
 "status":"PASS","C20_SOURCE_WORLD_SHA256":sha(BANK),
 "sensor_cases":len(native),"numeric_equalities":len(diff),"max_abs_error":max(diff),
 "native_Rust_compiled_SHA256":sha(BIN),
 "Rust_source_SHA256":sha(ROOT/"src/bin/quartz_c20_shadow.rs"),
 "Python_EVSI_source_SHA256":sha(ROOT/"quartz/experiments/c20_predictive_value.py"),
 "Rust_Beta_port_claim":"NOT_IMPLEMENTED", "true_hidden_state_was_given_to_forecast":False,
 "original_live_SearchPolicy_changed":False}
out=HOME/"evidence/C06_RUST_PYTHON_PARITY.json"
out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("C06_INDEPENDENT",json.dumps(res,sort_keys=True))
