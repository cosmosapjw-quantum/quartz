#!/usr/bin/env python3
"""C09 same-Rust-compiler native grid-vs-CDF speed, TRAIN ONLY, posttest exploratory."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
TRAIN=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
BIN=ROOT/"target/release/quartz_c21_grid_reference"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(TRAIN)=="db100b0ee9b4c8d8669c2c20d2beebabeb7bf3d74d017790259d667c10ae7bea"
rows=[json.loads(s) for s in TRAIN.read_text().splitlines() if s.strip()]
train=[r for r in rows if r["split"]=="train" and r["family"]=="beta_arms"]
assert len(train)==216 and len({r["group_id"] for r in train})==72
cases=[]
for row in train:
 seq=row["initial_observations"]
 a=[[float(2+sum(x)),float(2+len(x)-sum(x))] for x in seq]
 cases.append({"posterior":a,"extra_per_arm":[10]*len(a)})
assert {len(c["posterior"]):sum(len(x["posterior"])==len(c["posterior"]) for x in cases) for c in cases}=={2:72,3:72,4:72}
req={"cases":cases,"repetitions":20}
outdir=HOME/"evidence/C09"
outdir.mkdir(parents=True,exist_ok=True)
inputfile=outdir/"NATIVE_GRID_TRAIN_ONLY_INPUT.json"
inputfile.write_text(json.dumps(req,sort_keys=True,separators=(",",":"))+"\n")
raw=subprocess.check_output([str(BIN),str(inputfile)],text=True)
native=json.loads(raw)
assert native["same_Rust_compiler_target"] is True
assert native["n_cases"]==216 and native["repetitions"]==20
by=defaultdict(list);maxerror=0.
for row in native["results"]:
 assert row["K"] in (2,3,4) and row["n"]==[10]*row["K"]
 error=abs(row["grid_evsi"]-row["fast_evsi"])
 assert error<=1e-10
 maxerror=max(maxerror,error)
 by[str(row["K"])].append(row)
summary={}
for K,r in sorted(by.items()):
 summary[K]={"n":len(r),
  "full_joint_grid_Rust_median_us":statistics.median(x["grid_native_median_us"] for x in r),
  "fast_CDF_Rust_median_us":statistics.median(x["fast_native_median_us"] for x in r),
  "median_same_compiler_speed_ratio":statistics.median(x["native_grid_over_fast_speedratio"] for x in r),
  "n_future_per_arm":10}
report={"schema":"quartz_C21_C09_same_compiler_Rust_joint_grid_vs_CDF_TRAIN_only_v1",
 "status":"NUMERICALLY_CHECKED_TRAIN_ONLY_NOT_TEST_POLICY",
 "n_posterior_inputs":len(cases),"repetitions_per_case":20,
 "all_Rust_calculations_in_release_mode":True,
 "source_train_raw_SHA256":sha(TRAIN),"benchmark_native_compiled_SHA256":sha(BIN),
 "input_file_SHA256":sha(inputfile),
 "max_abs_gross_EVSI_difference":maxerror,
 "speed_by_arms":summary,
 "no_C21_TEST_examples_in_benchmark":True,
 "actual_live_policy_efficiency":"NOT_TESTED",
 "hotpath_integration":"NOT_IMPLEMENTED"}
p=outdir/"NATIVE_GRID_TRAIN_ONLY_REPORT.json"
p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C21_SAME_RUST_COMPILER_TRAIN_BENCH",json.dumps(report,sort_keys=True))
