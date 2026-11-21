#!/usr/bin/env python3
"""One fresh C16 production server smoke: nonterminal proof, real root, equal 20ms target."""
from pathlib import Path
import hashlib,json,subprocess
home=Path(__file__).resolve().parent.parent
repo=next(x for x in home.parents if (x/'.git').exists())
f=home/"COHORT_40_FROZEN.json"
sources=json.loads(f.read_text())["data_splits"]["constructed_fork_positive"]
model=repo/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json"
assert hashlib.sha256(model.read_bytes()).hexdigest()=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
s=sources[0]
arms=("learned_all_legal","learned_fork_mask","learned_fork_proof_choice","learned_sham_mask","learned_top16")
reqs=[{"cmd":"research_gomoku15_fork_cpu","rule":"freestyle",
       "history":s["history"],"arm":arm,"seed":17,
       "cpu_budget_us":20000,"model_path":str(model)} for arm in arms]
reqs.append({"cmd":"research_gomoku15_fork_inspect","rule":"freestyle",
             "history":s["history"]})
data="\n".join(json.dumps(x,separators=(",",":")) for x in reqs+[{"cmd":"quit"}])+"\n"
exe=repo/"target/debug/mcts_demo"
out=subprocess.run([str(exe),"--server"],input=data,capture_output=True,text=True,timeout=40,check=True)
rows=[json.loads(x) for x in out.stdout.splitlines() if x.strip()]
assert len(rows)==6,(len(rows),out.stderr[-250:])
for j,r in enumerate(rows):
 assert "error" not in r,(j,r.get("error"))
 if j<5:
  assert r["first8_root_visits"]==8
  assert r["same_root_N_W_M2_P_preserved_on_activation"] and r["same_root_ptr_hash"]
  assert r["root_edge_visit_sum"]==r["final_root_visits"]
  assert r["final_legal_move"] in r["legal_actions"]
 if j==1:
  assert r["source_rule_preconditions_verified"]
  assert len(r["added_actions"])>=1,("no certificate activation in C4 positive control")
 if j==2:assert r["explicit_proof_choice_override"] and r["selected_is_certified_fork"]
assert len(rows[5]["certificates_all"])>=18
report={"schema":"c16_C05_native_preflight_v1","status":"PASS",
        "source_id":s["id"],"expected_omitted":len(s["omitted_forks_before_experiment"]),
        "real_server_arms":list(arms),"certificates_all":len(rows[5]["certificates_all"]),
        "per_arm":[{"arm":r["arm"],"added_count":len(r["added_actions"]),
                   "final_move":r["final_legal_move"],"mcts_move":r["final_mcts_move"],
                   "proof_override":r["explicit_proof_choice_override"],
                   "cpu_us":r["total_actual_cpu_wall_us"],
                   "root_visits":r["final_root_visits"]} for r in rows[:5]],
        "actual_debug_native_binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest()}
(home/"evidence/C05_NATIVE_PREFLIGHT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
