#!/usr/bin/env python3
"""C09 supplementary bounded native Rust finite-channel evaluator latency.

Measures only time spent in native rust function, as returned by Rust Instant.
Does NOT include process launch/JSON/memcpy or claim integrated SearchPolicy timing.
"""
from pathlib import Path
import hashlib,subprocess,json,statistics
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
REQ=HOME/"evidence/C08_TEST/RUST_SHADOW_NO_LABEL_REQUEST.json"
BIN=ROOT/"target/release/quartz_c20_shadow"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
runs=[]
for iteration in range(3):
 output=json.loads(subprocess.check_output([str(BIN),str(REQ)],text=True))
 assert output["no_SearchPolicy_mutation"]
 vals=[float(r["evaluation_us"]) for r in output["predictions"]]
 assert len(vals)==72 and all(x>0 for x in vals)
 runs.append(vals)
flat=[v for rep in runs for v in rep]
stats={"schema":"quartz_C20_C09_native_Rust_sensor_evaluator_bounded_latency_v1",
 "status":"MEASURED",
 "compilation_SHA256":sha(BIN),
 "initial_only_request_SHA256":sha(REQ),
 "n_initial_only_sensor_models":72,
 "repetitions":3,
 "n_native_function_evaluations":len(flat),
 "native_function_eval_median_us":statistics.median(flat),
 "native_function_eval_mean_us":statistics.mean(flat),
 "native_function_eval_max_us":max(flat),
 "native_function_eval_min_us":min(flat),
 "per_run_mean_us":[statistics.mean(x) for x in runs],
 "cost_scope":"Rust Instant around finite source computation; excludes binary startup, JSON I/O, Python adapter and real Rust SearchPolicy integration",
 "transport_claim":"NOT_ESTABLISHED, separate standalone Rust reference only"}
p=HOME/"evidence/C09_NATIVE_RUST_LATENCY.json"
p.write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n")
print("C20_RUST_EVAL_LATENCY",json.dumps(stats,sort_keys=True))
