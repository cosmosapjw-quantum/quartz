#!/usr/bin/env python3
"""R7: exact bounded calculations, no external task/model data."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
p=F(3,5);q=F(4,5)
brier=p*(1-q)**2+(1-p)*q**2
irreducible=p*(1-p)
excess=(p-q)**2
assert brier==F(7,25)
assert irreducible==F(6,25)
assert excess==F(1,25)
assert brier==irreducible+excess
flip=F(1,4)
gain_plus=flip*F(1)
gain_minus=flip*F(-1)
assert gain_plus==F(1,4) and gain_minus==F(-1,4)
n,m,rho=12,3,F(1,2)
inflation=1+(m-1)*rho
assert inflation==2
ess=F(n*m)/inflation
assert ess==18
res={"schema":"quartz_c18_R07_exact_v1","status":"PASS",
 "brier":{"true_p":str(p),"forecast":str(q),"score":str(brier),
 "irreducible":str(irreducible),"miscalibration":str(excess)},
 "identical_flip_probability":str(flip),
 "positive_signed_change":str(gain_plus),
 "negative_signed_change":str(gain_minus),
 "repeated_cluster_example":{"groups":n,"per_group":m,"rho":str(rho),
   "variance_inflation":str(inflation),"effective_sample_size":str(ess)},
 "proof_scope":"algebraic identities plus finite counterexample, not MCTS inference",
 "not_human_cognition_or_game_strength":True}
out=Path(__file__).with_name("R07_EXACT_RESULTS.json")
out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
print("R07_EXACT",json.dumps(res,sort_keys=True))
