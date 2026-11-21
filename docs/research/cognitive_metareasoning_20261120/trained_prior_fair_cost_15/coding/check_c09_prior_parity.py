#!/usr/bin/env python3
"""C09 independent numerical readback of the actual model's full MCTS root prior."""
import hashlib,json
from pathlib import Path
import numpy as np
from policy15 import position_features,stable_prob
d=Path(__file__).resolve().parent.parent
model=json.loads((d/"evidence/model/MODEL_TRAINED.json").read_text())
theta=np.asarray(model["theta"],dtype=np.float64)
raw=json.loads((d/"evidence/real_192_blind_20261121/RAW_192_RUST.json").read_text())
maxdiff=0.
maxnorm=0.
n=0
for row in raw["per_source"]:
 h=row["history"]
 legal,feats=position_features(h)
 probs=stable_prob(feats@theta)
 expected={int(k):float(p) for k,p in zip(legal,probs)}
 for arm in ("learned_gated","learned_all_legal"):
  actual=row["arms"][arm]
  if not actual["model_loaded"] or actual["trained_value_head"]:
   raise RuntimeError("surrogate not actually loaded")
  keys={int(m):float(p) for m,p in actual["root_prior_probabilities"]}
  if set(keys)!=set(expected):raise RuntimeError("missing full-legal learned policy")
  err=max(abs(keys[a]-expected[a]) for a in expected)
  maxdiff=max(maxdiff,err)
  maxnorm=max(maxnorm,abs(sum(keys.values())-1))
  n+=1
 assert not row["arms"]["uniform_all_legal"]["model_loaded"]
assert maxdiff<3e-6,maxdiff
assert maxnorm<0.0002,maxnorm
report={"status":"PASS",
        "actual_trained_Rust_root_prior_calls_compared":n,
        "max_abs_probability_difference_vs_independent_numpy_softmax":maxdiff,
        "max_abs_Rust_probability_normalization_error":maxnorm,
        "model_sha256":hashlib.sha256((d/"evidence/model/MODEL_TRAINED.json").read_bytes()).hexdigest(),
        "binary_sha256":raw["compiled_native_binary_sha256"],
        "no_Rapfi_online":True,
        "validation":"Python NumPy double precision vs actual Rust float32 loaded-policy, across all legal actions in 128 trained real MCTS roots"}
(d/"evidence/real_192_blind_20261121/C09_NATIVE_PRIOR_PARITY.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C09_FULL_POLICY_PRIOR_PARITY",json.dumps(report,sort_keys=True))
