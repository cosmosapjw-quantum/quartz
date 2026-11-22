#!/usr/bin/env python3
"""C07A freezes per-declared-workload clock medians using TRAIN ONLY.

Actual pilot physically executes all policy paths in same Rust process;
source theta lives only in OFFLINE sampler. No posttest retuning.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics,subprocess
from quartz.experiments.c22_active_inputs import build_request,workload_bucket,public_task
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C22_FROZEN.json"
TRAIN=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
BIN=ROOT/"target/release/quartz_c22_active_gate"
EXPECTED_SOURCE="f7bffc98710bae151d0ac909ddeb19fc04f74d51a4273f6b16f7e7583f61398d"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def check_remote():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("C07A requires source code and raw TRAIN/CAL data already committed and clean")
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 local=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 assert local==remote
 return local
head=check_remote()
assert sha(SOURCE)==EXPECTED_SOURCE
worlds={w["id"]:w for w in json.loads(SOURCE.read_text())["worlds"]}
raw=[json.loads(x) for x in TRAIN.read_text().splitlines() if x.strip()]
assert len(raw)==576 and all(r["split"] in ("train","cal") for r in raw)
train=[r for r in raw if r["split"]=="train"]
cal=[r for r in raw if r["split"]=="cal"]
assert len(train)==432 and len(cal)==144
assert len({r["group_id"] for r in train})==144
assert len({r["group_id"] for r in cal})==48
assert not {r["group_id"] for r in train}&{r["group_id"] for r in cal}
req=build_request(train,worlds,lam=0.004,prices=None)
assert all(set(case["public"]).isdisjoint({"theta_offline_true_state","theta_offline_arm_means","signed_gain","label","extra_observations"}) for case in req["cases"])
outdir=HOME/"evidence/C07_TRAIN_CAL"
req_path=outdir/"NATIVE_TRAIN_PRICE_REQUEST.json"
req_path.write_text(json.dumps(req,sort_keys=True,separators=(",",":"))+"\n")
payload=json.loads(subprocess.check_output([str(BIN),str(req_path)],text=True))
assert payload["original_SearchPolicy_unchanged"] is True
assert payload["number_cases"]==432
rows_by_id={case["id"]:case for case in req["cases"]}
bucket=defaultdict(lambda:defaultdict(list))
allrecords=[]
for result in payload["rows"]:
 case=rows_by_id[result["id"]]
 public=case["public"]
 name=workload_bucket(public)
 sub=result["results"]
 task=float(sub["ALWAYS"]["actual_future_task_us"])
 exact=float(sub["FULL_EXACT"]["exact_evaluator_us"])
 cheap=float(sub["GATED"]["cheap_gate_us"])
 total_gated=float(sub["GATED"]["active_total_internal_us"])
 assert task>0 and exact>0 and cheap>=0 and total_gated>=0
 bucket[name]["task"].append(task)
 bucket[name]["exact"].append(exact)
 bucket[name]["cheap"].append(cheap)
 bucket[name]["gated_total"].append(total_gated)
 bucket[name]["baseline_total"].append(float(sub["ALWAYS"]["active_total_internal_us"]))
 allrecords.append({"id":result["id"],"group_id":result["group_id"],
    "split":result["split"],"family":public["family"],"resource_bucket":name,
    "policy_records":sub})
assert len(allrecords)==432
assert len(bucket)>=7
costs={}
for name,x in sorted(bucket.items()):
 costs[name]={"n_original_train_trials":len(x["task"]),
     "task_native_median_us":float(statistics.median(x["task"])),
     "exact_native_median_us":float(statistics.median(x["exact"])),
     "cheap_native_median_us":float(statistics.median(x["cheap"])),
     "task_native_mean_us":float(statistics.mean(x["task"])),
     "exact_native_mean_us":float(statistics.mean(x["exact"])),
     "cheap_native_mean_us":float(statistics.mean(x["cheap"])),
     "GATED_total_internal_mean_us":float(statistics.mean(x["gated_total"])),
     "ALWAYS_total_internal_mean_us":float(statistics.mean(x["baseline_total"]))}
 assert costs[name]["task_native_median_us"]>0 and costs[name]["exact_native_median_us"]>0
train_real=outdir/"NATIVE_TRAIN_ACTIVE_PILOT.jsonl"
train_real.write_text("".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in allrecords))
record={"schema":"quartz_C22_C07A_TRAIN_only_frozen_Rust_active_price_v1",
 "source_previously_published_git_sha":head,
 "source_bank_SHA256":sha(SOURCE),
 "TRAIN_CAL_raw_SHA256":sha(TRAIN),
 "TRAIN_pilot_request_SHA256":sha(req_path),
 "TRAIN_pilot_episodes_sha256":sha(train_real),
 "native_Rust_active_compiled_sha256":sha(BIN),
 "TRAIN_original_parent_groups_each_family":72,
 "CAL_original_parent_groups_each_family":24,
 "TRAIN_episodes":432,"CAL_episodes_not_fit":144,
 "price_by_bucket":costs,
 "precommitted_primary_lambda":0.004,
 "precommitted_sensitivity_lambda":[0.001,0.004,0.02,0.1],
 "TEST_outcomes_opened":False,
 "model_gamma_coefficients_fit_to_TEST":False,
 "all_paid_clocks_same_Rust_binary":True,
 "RNG_source_true_theta_only_offline":True,
 "active_and_baseline_BOTH_physically_executed_in_pilot":True,
 "no_real_neural_MCTS_SearchPolicy_mutation":True}
path=HOME/"FROZEN_C22_RESOURCE_PRICE.json"
path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C22_C07A_FROZEN_COST",json.dumps({"model_sha256":sha(path),
 "TRAIN_episodes":432,"CAL_not_fit":144,"buckets":costs,
 "pilot_actual_certified_stop_count":payload["GATED_true_STOP_count"],
 "pilot_active_skipped_draw_count":payload["GATED_actually_unexecuted_future_draws"]},sort_keys=True))
