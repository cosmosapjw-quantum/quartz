#!/usr/bin/env python3
"""C6 independently execute compiled Rust CLI and Python on all five frozen game-neutral tasks."""
import hashlib,json,math,subprocess
from pathlib import Path
from quartz.experiments.metalevel_decision import FiniteProblem

ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HERE=Path(__file__).resolve().parent.parent
P=ROOT/"configs/metalevel_reference_cases.v1.json"
BIN=ROOT/"target/release/quartz_metalevel_reference"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def close(actual,expected,scope):
 if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10):
  raise AssertionError(f"cross-language numerical mismatch {scope}: Rust={actual} Python={expected}")
spec=json.loads(P.read_text())
stdout=subprocess.check_output([str(BIN),str(P)],text=True)
rust=json.loads(stdout)
assert rust["schema"]=="quartz_meta17_rust_game_neutral_reference_v1"
assert rust["game_specific_input_used"] is False
rows={r["id"]:r for r in rust["tasks"]}
assert len(rows)==len(spec["cases"])==5
n=0
for task in spec["cases"]:
 m=FiniteProblem.from_mapping(task)
 actual=rows[task["id"]]
 close(actual["initial_risk"],m.risk(),(task["id"],"risk"));n+=1
 close(actual["terminal_value"],m.terminal_value(),(task["id"],"terminal_value"));n+=1
 d=m.one_step()
 assert actual["rate_choice"]==d["ratio_choice"],task["id"]
 assert actual["net_choice"]==d["net_choice"],task["id"]
 for r,e in zip(actual["experiments"],m.experiments):
  assert r["name"]==e.name
  for key,expect in (("gross",d["gross_gain"][e.name]),
                     ("ratio",d["gain_per_cost"][e.name]),
                     ("net",d["net_gains"][e.name])):
   close(r[key],expect,(task["id"],e.name,key));n+=1
 for item in actual["horizons"]:
  p,choose=m.bellman(item["h"])
  close(item["value"],p,(task["id"],"bellman",item["h"]));n+=1
  assert item["choice"]==choose,(task["id"],item["h"],item["choice"],choose)
report={"schema":"quartz_meta17_C06_independent_cross_language_v1",
 "status":"PASS","frozen_case_sha256":sha(P),"rust_release_sha256":sha(BIN),
 "python_module_sha256":sha(ROOT/"quartz/experiments/metalevel_decision.py"),
 "rust_source_sha256":sha(ROOT/"src/mcts/policy/finite_metalevel.rs"),
 "python_source":"quartz.experiments.metalevel_decision",
 "n_task_families":5,"n_numeric_equalities_with_absolute_tolerance_1e-10":n,
 "all_selection_and_stop_decisions_equal":True,
 "all_tasks_game_neutral":True,
 "not_neural_MCTS_online_evidence":True}
out=HERE/"evidence/C06_CROSS_LANGUAGE.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C06_RUST_PYTHON",json.dumps(report,sort_keys=True))
