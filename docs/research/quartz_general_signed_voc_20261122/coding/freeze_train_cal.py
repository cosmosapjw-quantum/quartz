#!/usr/bin/env python3
"""C07A TRAIN+CAL-only fit and frozen cost forecast, no TEST file opened."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib,json,statistics
from quartz.experiments.c19_signed_value import train_signed,calibrate_signed,shadow_predict,score_signed
from quartz.experiments.c18_observation import score_predictions
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BANK=HOME/"SOURCE_WORLDS_FROZEN.json"
RAW=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
EXPECTED="8fce353c5efd3a5ac2a643e5a799d46f8d69c6c429cdff0fb5418a43ea43292e"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(BANK)==EXPECTED
rows=[json.loads(s) for s in RAW.read_text().splitlines() if s]
assert len(rows)==576 and all(r["split"]!="test" for r in rows)
train=[r for r in rows if r["split"]=="train"]
cal=[r for r in rows if r["split"]=="cal"]
assert len(train)==432 and len(cal)==144
assert len({r["group_id"] for r in train})==144
assert len({r["group_id"] for r in cal})==48
assert not ({r["group_id"] for r in train}&{r["group_id"] for r in cal})
model=train_signed(train)
calibration=calibrate_signed(model,cal)
kinds=("beta_arms","finite_sensor")
cost_by_kind={k:float(statistics.median(r["actual_extra_cost_us"] for r in train if r["family"]==k)) for k in kinds}
assert all(x>0 for x in cost_by_kind.values())
p_train=sum(r["label"] for r in train)/len(train)
baseline_p=max(1e-6,min(1-1e-6,p_train))
cal_scores={}
for k in kinds:
 sub=[r for r in cal if r["family"]==k]
 p=shadow_predict(model,[{"gap":r["gap"],"n_actions":r["n_actions"],"compute_kind":r["compute_kind"]} for r in sub],
  resource_price=.004,cost_by_kind=cost_by_kind,calibration=calibration)
 cal_scores[k]={"calibration_descriptive_Brier":score_predictions([v["flip_probability"] for v in p],[r["label"] for r in sub])["brier"],
 "calibration_descriptive_signed_mse":score_signed([v["gross_signed_value"] for v in p],
      [r["signed_gain"] for r in sub],[r["label"] for r in sub])["mse"],
 "n_rows":len(sub)}
res={"schema":"quartz_general_c19_frozen_train_cal_two_part_model_v1",
 "source_world_SHA256":sha(BANK),"TRAIN_CAL_raw_SHA256":sha(RAW),
 "source_original_C18":"097255bd00ffa436c3d43704ed201815e7cbb8ea",
 "train_n":len(train),"cal_n":len(cal),
 "train_groups":144,"cal_groups":48,
 "trained_model":model,"calibration_only_intercepts":calibration,
 "predicted_incremental_cost_us_by_compute_kind":cost_by_kind,
 "TRAIN_global_constant_probability":baseline_p,
 "TRAIN_global_constant_signed_gain":model["train_mean_signed_value"],
 "TRAIN_global_conditional_signed_gain_if_flip":model["train_conditional_mean_if_flip"],
 "fixed_primary_price_utility_per_microsecond":0.004,
 "fixed_price_sensitivity":[0.001,0.004,0.010],
 "fixed_flip_only_comparator_threshold":0.20,
 "frozen_cal_descriptive":cal_scores,
 "TEST_labels_seen_or_used":False,
 "no_source_specific_q_or_m_features":True,
 "no_true_theta_online":True,
 "model_scope":"read-only shadow; two known utility task families; no real MCTS claimed"}
out=HOME/"SHADOW_SIGNED_MODEL_FROZEN.json"
out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("C07A_SIGNED_FROZEN",json.dumps({"model_sha256":sha(out),"train":len(train),"cal":len(cal),
"train_flips":model["n_train_flips"],"cal_flips":calibration["n_cal_flips"],
"predicted_cost_us":cost_by_kind,
"theta_q":model["q_coefficients"],"theta_m":model["m_coefficients"],
"qoff":calibration["q_offset"],"moff":calibration["m_offset"]},sort_keys=True))
