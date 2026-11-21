#!/usr/bin/env python3
"""Posttest diagnostic of *effective missed value*, not retuning or prospectively admitted."""
from pathlib import Path
import hashlib,json,sys
home=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from freeze_sources16 import source_forks
src=json.loads((home/"evidence/real_root16_blind/RAW_ROOT_195.json").read_text())
groups={"constructed":[],"MCTS_played":[]}
for item in src["per_source"]:
 cert={int(x["action"]) for x in source_forks(item["history"])}
 top=set(item["arms"]["learned_all_legal"]["initial_ranked_actions"][:16])
 full=set(item["arms"]["learned_all_legal"]["legal_actions"])
 assert cert<=full
 n_in=len(cert&top)
 n_out=len(cert-top)
 kind="constructed" if "engineered" in item["source_type"] else "MCTS_played"
 groups[kind].append({"id":item["source_id"],"cert_active":n_in,
  "cert_excluded":n_out,"all_certified_count":len(cert),
  "strict_effective_miss":(n_in==0 and n_out>0),
  "missing_but_redundant":(n_in>0 and n_out>0)})
out={"schema":"quartz_cog06_C08_POSTTEST_equivalence_class_reachability_v1",
"status":"EXPLORATORY_POSTTEST_NOT_PREDECLARED",
"theorem":"I(exists certified in full)-I(exists certified in top16)=I(no certified in top16 AND some certified excluded)",
"not_claiming_search_win_value_from_binary_certificates":True,
"source_raw_SHA256":hashlib.sha256((home/"evidence/real_root16_blind/RAW_ROOT_195.json").read_bytes()).hexdigest(),
"groups":{k:{"n":len(v),"n_strict_effective_miss":sum(q["strict_effective_miss"] for q in v),
 "n_missing_but_redundant":sum(q["missing_but_redundant"] for q in v),
 "n_with_any_cert_in_top16":sum(q["cert_active"]>0 for q in v),
 "n_with_any_cert_excluded":sum(q["cert_excluded"]>0 for q in v),
 "details":v} for k,v in groups.items()}}
for k,z in out["groups"].items():
 assert all((x["cert_active"]+x["cert_excluded"]>0)-(x["cert_active"]>0)==int(x["strict_effective_miss"]) for x in z["details"])
p=home/"evidence/real_root16_blind/C08_POSTTEST_REACHABILITY.json"
p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("C08_POSTTEST_REACH",json.dumps({k:{a:v for a,v in z.items() if a!="details"} for k,z in out["groups"].items()},sort_keys=True))
