#!/usr/bin/env python3
"""C6 independently invoke compiled Rust shadow CLI and Python on 96 initial-only inputs."""
from pathlib import Path
import hashlib,json,random,subprocess,tempfile
from quartz.experiments.c19_signed_value import shadow_predict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BIN=ROOT/"target/release/quartz_c19_shadow"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
rng=random.Random(202611221907)
kinds={"beta_arms":7.25,"finite_sensor":4.75}
rows=[{"gap":rng.random(),"n_actions":rng.choice([2,3,4,16,80,220]),
       "compute_kind":rng.choice(list(kinds))} for _ in range(96)]
model={"q_coefficients":[0.14,-1.2,0.25],
       "m_coefficients":[0.08,0.2,-0.04]}
cal={"q_offset":-0.05,"m_offset":0.017}
p=shadow_predict(model,rows,resource_price=0.004,cost_by_kind=kinds,calibration=cal)
request={"q_coefficients":model["q_coefficients"],"m_coefficients":model["m_coefficients"],
 "q_offset":cal["q_offset"],"m_offset":cal["m_offset"],"resource_price":0.004,
 "rows":[{"gap":r["gap"],"n_actions":r["n_actions"],"estimated_cost_us":kinds[r["compute_kind"]]} for r in rows]}
assert all(set(x)=={"gap","n_actions","estimated_cost_us"} for x in request["rows"])
with tempfile.TemporaryDirectory() as temp:
 f=Path(temp)/"initial_only.json"
 f.write_text(json.dumps(request))
 native=json.loads(subprocess.check_output([str(BIN),str(f)],text=True))
assert native["no_search_or_decision_mutation"] is True
assert native["true_utility_input_used"] is False
assert len(native["predictions"])==len(rows)==96
keys=["flip_probability","conditional_signed_value","gross_signed_value",
      "predicted_net_value","estimated_incremental_cost_us"]
diff=[]
for a,b in zip(native["predictions"],p):
 assert a["shadow_choice"]==b["shadow_choice"]
 for key in keys:diff.append(abs(float(a[key])-float(b[key])))
assert max(diff)<=1e-10,max(diff)
record={"schema":"quartz_general_c19_C06_Rust_Python_signed_value_parity_v1",
 "status":"PASS","reference_rows":len(rows),
 "float_equalities_within_tolerance":len(diff),
 "maximum_abs_error":max(diff),
 "Rust_release_SHA256":sha(BIN),
 "Rust_source_sha256":sha(ROOT/"src/bin/quartz_c19_shadow.rs"),
 "Python_source_sha256":sha(ROOT/"quartz/experiments/c19_signed_value.py"),
 "no_labels_or_domain_in_Rust_input":True,
 "no_live_controller_actions":True}
out=HOME/"evidence/C06_PARITY.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C06_PARITY",json.dumps(record,sort_keys=True))
