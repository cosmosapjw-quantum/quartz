#!/usr/bin/env python3
"""C1: lock original project source authority and separate isolated reference scope."""
from pathlib import Path
import hashlib,json,subprocess
REPO=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HERE=Path(__file__).resolve().parent.parent
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
targets=[
"docs/THESIS.md","docs/CLAIM_LEDGER.md","docs/BQ_PLUS_PLUS_DESIGN.md",
"docs/RESEARCH_PLAN_PARTB.md","docs/METACOGNITIVE_EXPERIMENTS.md",
"docs/idea_foundry/01_control_policy.md","docs/LOCAL_EXPERIMENT_LAB.md",
"quartz/experiments/bernoulli_root.py",
"quartz/experiments/candidate_morphology.py","src/mcts/policy/mod.rs",
"src/mcts/policy/trait_def.rs","Cargo.toml","Cargo.lock"]
for t in targets:assert (REPO/t).is_file(),t
survey=HERE/"evidence/REPOSITORY_PROVENANCE.json"
base=json.loads(survey.read_text())
assert base["GitHub_actual_PR_count"]==0
assert base["ref_name_group_counts"]=={"agent":7,"audit":4,"main":1,"research":18}
p={"schema":"quartz_meta17_C01_source_and_scope_lock_v1",
 "source_parent":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
 "original_dev_untracked_file_left_untouched":True,
 "new_branch":"research/quartz-general-metareasoning-identity-20261122",
 "authoritative_history_survey_sha256":sha(survey),
 "game_independent_reference":"finite posterior, utility, observation channels and priced computation",
 "not_game_strategy_or_pattern_injection":True,
 "do_not_modify":["docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs",
                  "existing production SearchPolicy behavior","existing COG15/16 source"],
 "expected_reference_implementation":["quartz/experiments/metalevel_decision.py",
                                      "src/mcts/policy/finite_metalevel.rs"],
 "production_MCTS_integration":"NOT_AUTHORIZED_IN_THIS_STAGE",
 "documents":[{"name":t,"sha256":sha(REPO/t),"bytes":(REPO/t).stat().st_size} for t in targets]}
out=HERE/"C01_LOCK.json"
out.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
print("C01_LOCK",json.dumps({"source_parent":p["source_parent"],"n_inputs":len(targets),
"survey_sha256":p["authoritative_history_survey_sha256"],"sha256":sha(out)}))
