#!/usr/bin/env python3
"""C7A freeze TRAIN+CAL fitted shadow model before touching test labels."""
from pathlib import Path
import hashlib,json,math
from quartz.experiments.c18_observation import fit_logistic,calibrate_intercept,predict,score_predictions
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_GROUPS_FROZEN.json"
RAW=HOME/"evidence/C07_TRAIN_CAL/SOURCE_RAW.jsonl"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(SOURCE)=="52e3654febead1a01253174be0d395f8c2ea6e04ada21e5e0bfbf5938d680445"
records=[json.loads(x) for x in RAW.read_text().splitlines() if x]
assert records and all(row["split"]!="test" for row in records)
train=[r for r in records if r["split"]=="train"]
cal=[r for r in records if r["split"]=="cal"]
assert {r["domain"] for r in train}=={"real_Rust_MCTS","non_game_Bernoulli"}
assert {r["domain"] for r in cal}=={"real_Rust_MCTS","non_game_Bernoulli"}
assert not {r["group_id"] for r in train}&{r["group_id"] for r in cal}
w=fit_logistic(train)
offset=calibrate_intercept(w,cal)
summary={}
for split,rows in (("train",train),("cal",cal)):
 for domain in ("real_Rust_MCTS","non_game_Bernoulli"):
  subset=[r for r in rows if r["domain"]==domain]
  yp=[r["label"] for r in subset]
  pr=predict(w,subset,offset if split=="cal" else 0.0)
  summary[f"{split}/{domain}"]=score_predictions(pr,yp)
baseline={domain:max(1e-6,min(1-1e-6,sum(r["label"] for r in train if r["domain"]==domain)/
    sum(1 for r in train if r["domain"]==domain))) for domain in ("real_Rust_MCTS","non_game_Bernoulli")}
info={"schema":"quartz_general_c18_shadow_train_cal_frozen_v1",
 "source_bank_sha256":sha(SOURCE),"train_cal_raw_sha256":sha(RAW),
 "train_groups_by_domain":{domain:len({r["group_id"] for r in train if r["domain"]==domain}) for domain in baseline},
 "cal_groups_by_domain":{domain:len({r["group_id"] for r in cal if r["domain"]==domain}) for domain in baseline},
 "weights":[float(x) for x in w],"cal_offset":float(offset),
 "feature_order":["intercept","normalized_initial_gap","log1p(n_actions)"],
 "model_domain_feature_used":False,
 "train_baseline_constant_by_domain":baseline,
 "split_exposure":"TRAIN and CAL only; TEST raw file neither opened nor needed",
 "heldout_test_labels_seen":False,
 "L2_weights":0.5,"CAL_L2":0.25,"clip":1e-6,
 "train_cal_descriptive_scores":summary,
 "scientific_status":"FROZEN_SHADOW_NOT_ACTIVE_CONTROLLER"}
out=HOME/"SHADOW_MODEL_FROZEN.json"
out.write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
print("C07A_MODEL_FROZEN",json.dumps({"model_SHA256":sha(out),
 "train_n":len(train),"cal_n":len(cal),
 "theta":info["weights"],"cal_offset":info["cal_offset"],
 "domain_rates":baseline},sort_keys=True))
