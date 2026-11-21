#!/usr/bin/env python3
"""After C07 fitting, freeze model byte identity BEFORE C08 test unblinding."""
from pathlib import Path
import hashlib,json,subprocess
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
home=Path(__file__).resolve().parent.parent
repo=next(x for x in home.parents if (x/'.git').exists())
model=home/"evidence/model/MODEL_TRAINED.json"
train=home/"evidence/teacher_train_validation/TEACHER_TRAIN_VALIDATION.json"
m=json.loads(model.read_text())
t=json.loads(train.read_text())
assert m["schema"]=="quartz_cog05_loglinear_policy13_v1"
assert m["cohort_sha256"]==t["cohort_sha256"]
assert m["train_validation_labels_sha256"]==sha(train)
assert not m["trained_value_head"] and not m["heldout_test_teacher_labels_used"]
assert m["training_metrics"]["n"]==128 and m["validation_metrics"]["n"]==32
assert m["final_grad_inf"]<1e-6
assert not (home/"evidence/teacher_heldout/TEACHER_HELDOUT.json").exists()
report={"schema":"quartz_cog05_model_frozen_before_test_v1",
"model_sha256":sha(model),"teacher_train_validation_sha256":sha(train),
"model_training_source_sha256":sha(Path(__file__).parent/"policy15.py"),
"source_frozen_cohort_sha256":m["cohort_sha256"],
"train_and_validation_only":True,"heldout_teacher_labels_used":False,
"training_source_current_git_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
"n_train":m["training_metrics"]["n"],"n_validation":m["validation_metrics"]["n"],
"training_top1":m["training_metrics"]["top1_accuracy"],
"validation_top1":m["validation_metrics"]["top1_accuracy"],
"validation_top16":m["validation_metrics"]["top16_coverage"],
"training_final_gradient_inf":m["final_grad_inf"],
"frozen_prior_not_a_deep_value_network":True}
(home/"evidence/model/MODEL_C07_FREEZE.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print("C07_FROZEN",json.dumps(report,sort_keys=True))
