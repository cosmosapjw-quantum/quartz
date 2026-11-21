#!/usr/bin/env python3
"""Exact action and cross-language numerical feature parity, no teacher labels."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from freeze_cohorts import replay
from policy15 import one_feature,FEATURE_ORDER
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--binary",type=Path,required=True)
 args=parser.parse_args()
 binary=args.binary.resolve()
 assert binary.is_file()
 root=Path(__file__).resolve().parent.parent
 cohort=json.loads((root/"COHORT_224_FROZEN.json").read_text())
 cases=[]
 for split in ("train","validation","test_natural","test_tactical"):
    for source in cohort["splits"][split][:4]:
      board=replay(source["history"])
      rng=np.random.default_rng(source["source_seed"])
      actions=rng.choice([a for a,v in enumerate(board) if not v],size=5,replace=False)
      for a in actions:
        cases.append((split,source["history"],board,1 if len(source["history"])%2==0 else 2,int(a)))
 requests=[{"cmd":"research_gomoku15_prior_features","rule":"freestyle","history":h,"action":a}
           for split,h,b,side,a in cases]
 data="\n".join(json.dumps(x,separators=(",",":")) for x in
      requests+[{"cmd":"quit"}])+"\n"
 p=subprocess.run([str(binary),"--server"],input=data,capture_output=True,text=True,check=True,timeout=90)
 output=[json.loads(z) for z in p.stdout.splitlines() if z.strip()]
 assert len(output)==len(cases),(len(output),len(cases),p.stderr[-500:])
 errors=[]
 sides=set()
 for (split,h,b,side,a),r in zip(cases,output):
    if "error" in r:raise RuntimeError(str(r))
    py=one_feature(b,side,a)
    rust=np.asarray(r["features"],dtype=np.float64)
    assert rust.shape==(13,)
    assert a==r["action"]
    assert r["legal_count"]==225-len(h)
    err=np.max(np.abs(py-rust))
    errors.append(float(err))
    sides.add((split,side))
 assert max(errors)<=2e-7,max(errors)
 result={"status":"PASS","case_count":len(cases),
         "max_feature_abs_error":max(errors),
         "source_binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
         "source_cohort_sha256":hashlib.sha256((root/"COHORT_224_FROZEN.json").read_bytes()).hexdigest(),
         "feature_order":list(FEATURE_ORDER),"split_and_side_coverage":sorted([list(x) for x in sides]),
         "teacher_used":False,"validation":"Python floating features vs actual compiled Rust features, 80 legal-action requests"}
 (root/"C06_FEATURE_PARITY.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print("C06_REAL_RUST_PARITY",json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
