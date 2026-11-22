#!/usr/bin/env python3
"""C07A TRAIN-only estimator: real native task, exact native value and boundary price.

Must run on clean published C07. Never opens heldout TEST outcomes, fits
EVSI parameters, trains using offline theta or changes original SearchPolicy.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,statistics,subprocess
from quartz.experiments.c21_native_session import (
  native_beta_with_real_task,native_sensor_model_batch,sha,BETA,SENSOR)
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C21_FROZEN.json"
TRAINRAW=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
SHA_SOURCE="036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8"
def published():
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
        raise RuntimeError("source price fitting requires C07 clean published commit")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
    if head!=remote:raise RuntimeError("GitHub source code not published")
    return head
head=published()
assert sha(SOURCE)==SHA_SOURCE
rows=[json.loads(s) for s in TRAINRAW.read_text().splitlines() if s.strip()]
assert len(rows)==576 and all(r["split"] in ("train","cal") for r in rows)
train=[r for r in rows if r["split"]=="train"]
cal=[r for r in rows if r["split"]=="cal"]
assert len(train)==432 and len(cal)==144
assert len({r["group_id"] for r in train})==144
assert len({r["group_id"] for r in cal})==48
assert not ({r["group_id"] for r in train}&{r["group_id"] for r in cal})
beta=[r for r in train if r["family"]=="beta_arms"]
sensor=[r for r in train if r["family"]=="finite_sensor"]
assert len(beta)==len(sensor)==216
native,beta_meta=native_beta_with_real_task(beta)
actual_sensor,sensor_meta=native_sensor_model_batch(sensor)
assert len(native)==216 and len(actual_sensor)==216
bucket=defaultdict(lambda:defaultdict(list))
for raw,measure in zip(beta,native):
    kind="beta_K"+str(raw["n_actions"])
    bucket[kind]["task_us"].append(measure["actual_native_extra_task_internal_us"])
    bucket[kind]["model_internal_us"].append(measure["native_predictor_function_internal_us"])
    bucket[kind]["model_boundary_us"].append(measure["native_predictor_streaming_roundtrip_us"])
    bucket[kind]["task_server_us"].append(measure["native_offline_task_server_roundtrip_us"])
    bucket[kind]["exact_evsi"].append(measure["gross_exact_EVSI"])
    assert abs(measure["exact_predictive_mass"]-1)<1e-9
for raw,measure in zip(sensor,actual_sensor):
    kind="sensor_K3"
    bucket[kind]["task_us"].append(float(raw["actual_extra_cost_us"]))
    bucket[kind]["model_internal_us"].append(measure["native_predictor_function_internal_us"])
    bucket[kind]["model_boundary_us"].append(sensor_meta["amortized_batch_host_us_per_case"])
    bucket[kind]["exact_evsi"].append(measure["gross_exact_EVSI"])
    assert abs(measure["exact_predictive_mass"]-1)<1e-9
out={}
for kind,columns in sorted(bucket.items()):
    out[kind]={"n_train":len(columns["task_us"]),
         "source_clock_scope":"Rust in-process actual Bernoulli" if kind.startswith("beta") else "Python actual categorical posterior update",
         "predictor_boundary_scope":"long-lived Rust streaming per-case JSON roundtrip" if kind.startswith("beta") else "amortized one-off native sensor batch CLI",
         "train_task_median_us":statistics.median(columns["task_us"]),
         "train_predictor_function_median_us":statistics.median(columns["model_internal_us"]),
         "train_predictor_boundary_median_us":statistics.median(columns["model_boundary_us"]),
         "train_task_mean_us":statistics.mean(columns["task_us"]),
         "train_predictor_function_mean_us":statistics.mean(columns["model_internal_us"]),
         "train_predictor_boundary_mean_us":statistics.mean(columns["model_boundary_us"]),
         "train_mean_exact_EVSI":statistics.mean(columns["exact_evsi"])}
    assert out[kind]["train_task_median_us"]>0 and out[kind]["train_predictor_function_median_us"]>0
assert set(out)=={"beta_K2","beta_K3","beta_K4","sensor_K3"}
# Actual source-only native offline true θ is present ONLY in the task-sampler
# input; never in predictor query. Retain full train-only records for audit.
nfile=HOME/"evidence/C07_TRAIN_CAL/NATIVE_TRAIN_ACTUAL.jsonl"
nfile.write_text("".join(json.dumps(v,sort_keys=True)+"\n" for v in native))
record={"schema":"quartz_C21_C07A_resource_model_frozen_v1",
 "source_code_previously_published_git_head":head,
 "source_world_SHA256":sha(SOURCE),
 "TRAIN_CAL_raw_SHA256":sha(TRAINRAW),
 "native_TRIALS_train_sha256":sha(nfile),
 "native_rust_beta_compiled_sha256":sha(BETA),
 "independent_C20_rust_sensor_compiled_sha256":sha(SENSOR),
 "TRAIN_parent_world_groups_per_domain":72,
 "CAL_parent_world_groups_per_domain":24,
 "TRAIN_rows_per_domain":216,"CAL_rows_per_domain":72,
 "beta_server_first_cold_warmup_us":beta_meta["server_first_request_cold_start_us"],
 "finite_sensor_batched_host_roundtrip_mean_us":sensor_meta["amortized_batch_host_us_per_case"],
 "price_by_declared_workload_bucket":out,
 "fixed_primary_lambda_utility_per_us":0.004,
 "prespecified_lambda_sensitivity":[0.001,0.004,0.010],
 "no_TEST_rows_accessed":True,
 "no_hidden_theta_as_model_input":True,
 "no_live_stop_or_search_action_mutated":True,
 "source_C20_parent_sha":"ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4"}
path=HOME/"FROZEN_C21_RESOURCE_PRICE.json"
path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C21_C07A_COST_FROZEN",json.dumps({"source_sha256":sha(path),"TRAIN":len(train),
 "CAL":len(cal),"price_buckets":out,
 "beta_cold_first_request_us":beta_meta["server_first_request_cold_start_us"]},sort_keys=True))
