#!/usr/bin/env python3
"""C6 actual compiled Rust shadow-only predictor compared with NumPy on initial-only rows."""
from pathlib import Path
import hashlib,json,random,subprocess,tempfile
from quartz.experiments.c18_observation import predict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BIN=ROOT/"target/release/quartz_c18_shadow"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
rng=random.Random(202611221806)
rows=[{"gap":rng.random(),"n_actions":rng.choice([2,3,4,16,80,220])} for i in range(96)]
theta=[-0.314,0.726,-0.11];offset=0.238
req={"theta":theta,"offset":offset,"rows":rows}
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp)/"forecast.json"
 p.write_text(json.dumps(req))
 raw=subprocess.check_output([str(BIN),str(p)],text=True)
 observed=json.loads(raw)
assert observed["online_actions_mutated"] is False
assert observed["model_is_not_bayesian_utility_posterior"] is True
native=observed["prediction"]
py=predict(theta,rows,offset)
assert len(native)==len(py)==96
errors=[abs(float(a)-float(b)) for a,b in zip(native,py)]
assert max(errors)<=1e-10,max(errors)
res={"schema":"quartz_c18_C06_Rust_Python_parity","status":"PASS",
 "independent_Rust_native_binary_sha256":sha(BIN),
 "source_sha256":sha(ROOT/"src/bin/quartz_c18_shadow.rs"),
 "C05_Python_sha256":sha(ROOT/"quartz/experiments/c18_observation.py"),
 "test_rows_initial_only":len(rows),"maximum_abs_probability_error":max(errors),
 "runtime_actions_mutated":False,"no_labels_in_rust_requests":True}
out=HOME/"evidence/C06_RUST_PYTHON_PARITY.json"
out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("C06_PARITY",json.dumps(res,sort_keys=True))
