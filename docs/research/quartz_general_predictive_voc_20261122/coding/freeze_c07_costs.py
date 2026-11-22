#!/usr/bin/env python3
"""C07A: freeze TRAIN-only physical task and predictor CPU overhead before TEST.

All predictor calls consume exact initial-only typed observations, no future
draws or offline theta. The all-source raw file here has TRAIN/CAL only.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,statistics,json,time
from quartz.experiments.c19_signed_value import shadow_predict as coarse_shadow_predict
from quartz.experiments.c20_task_adapter import initial_only,eval_initial,resource_bucket
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
RAW=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
OLD_BASE=ROOT/"docs/research/quartz_general_signed_voc_20261122/SHADOW_SIGNED_MODEL_FROZEN.json"
SHA_SOURCE="d6cf96c26d9ac913963f702d9b0418f0002038979f6a470da3865999368e4ae7"
SHA_C19_MODEL="5fdd172a13d51c4d17dc8e7829fb7c5518421028046ebac6e418a1158adf6735"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(SOURCE)==SHA_SOURCE and sha(OLD_BASE)==SHA_C19_MODEL
historical=json.loads(OLD_BASE.read_text())
rows=[json.loads(s) for s in RAW.read_text().splitlines() if s.strip()]
assert len(rows)==576 and all(r["split"] in ("train","cal") for r in rows)
train=[r for r in rows if r["split"]=="train"]
cal=[r for r in rows if r["split"]=="cal"]
assert len(train)==432 and len(cal)==144
assert not {r["group_id"] for r in train}&{r["group_id"] for r in cal}
assert len({r["group_id"] for r in train})==144
assert len({r["group_id"] for r in cal})==48
task=defaultdict(list)
evalcost=defaultdict(list)
coarse_evalcost=defaultdict(list)
values=defaultdict(list)
for r in train:
    inp=initial_only(r)  # source-only; strips theta, final action, Y, Z and realised C
    pred=eval_initial(inp)
    assert pred["total_predictive_mass"]>=1-1e-9
    assert pred["evsi"]>=0
    bucket=resource_bucket(inp)
    start_coarse=time.perf_counter_ns()
    coarse_shadow_predict(historical["trained_model"],
        [{"gap":r["gap"],"n_actions":r["n_actions"],"compute_kind":r["family"]}],
        resource_price=.004,
        cost_by_kind=historical["predicted_incremental_cost_us_by_compute_kind"],
        calibration=historical["calibration_only_intercepts"])
    coarse_evalcost[bucket].append(float(max(0,time.perf_counter_ns()-start_coarse))/1000)
    task[bucket].append(float(r["actual_extra_cost_us"]))
    evalcost[bucket].append(float(pred["evaluation_us"]))
    values[bucket].append(float(pred["evsi"]))
assert set(task)=={"beta_arms_K2","beta_arms_K3","beta_arms_K4","finite_sensor_K3"}
summary={k:{
    "n_train_rows":len(task[k]),
    "task_incremental_median_us":float(statistics.median(task[k])),
    "predictor_evaluation_median_us":float(statistics.median(evalcost[k])),
    "coarse_predictor_evaluation_median_us":float(statistics.median(coarse_evalcost[k])),
    "predictor_evaluation_max_us":float(max(evalcost[k])),
    "mean_model_EVSI":float(statistics.mean(values[k])),
    "mean_task_actual_us":float(statistics.mean(task[k]))
} for k in sorted(task)}
record={
 "schema":"quartz_general_c20_frozen_train_priced_exact_EVSI_v1",
 "source_bank_SHA256":sha(SOURCE),"TRAIN_CAL_raw_SHA256":sha(RAW),
 "C19_historical_coarse_model_SHA256":sha(OLD_BASE),
 "TRAIN_groups_each_family":72,"CAL_groups_each_family":24,
 "TRAIN_rows":len(train),"CAL_rows":len(cal),
 "source_world_parent_C19":"2a359ad049b6a0be4047e52ad281b665cf68e967",
 "source_bank_metadata_legacy_field_note":"Inherited source_parent_C18 metadata in C04 generator is a historical template field, not the current C20 base; the actual C20 parent is C19 Git SHA above. Latent worlds/splits unchanged and byte frozen.",
 "fixed_primary_price_utility_per_microsecond":0.004,
 "fixed_price_sensitivity":[0.001,0.004,0.01],
 "shared_exact_bayesian_model_has_no_fitted_decision_parameters":True,
 "cost_and_predictor_overhead_medians_TRAIN_only":summary,
 "test_outcomes_or_labels_accessed":False,
 "real_MCTS_true_utility":"NOT_ESTIMABLE",
 "two_domain_source_only_online_inputs":True
}
out=HOME/"FROZEN_C20_RESOURCE_PRICE.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C07A_COSTS_FROZEN",json.dumps({"model_sha256":sha(out),"train":len(train),
  "cal":len(cal),"buckets":summary},sort_keys=True))
