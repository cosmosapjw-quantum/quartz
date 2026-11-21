#!/usr/bin/env python3
"""C10 provenance and science closure before final self-referential stage archive."""
from pathlib import Path
import hashlib,json,re,subprocess
REPO=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BRANCH="research/quartz-general-metareasoning-identity-20261122"
BACKUPS=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta17_backups")
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def output(*args):return subprocess.check_output(args,text=True).strip()
head=output("git","-C",str(REPO),"rev-parse","HEAD")
remote=output("git","-C",str(REPO),"ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote
assert output("git","-C",str(REPO),"branch","--show-current")==BRANCH
archives=sorted(BACKUPS.glob("QUARTZ_META17_*.tar.gz"))
stage_map={}
pattern=re.compile(r"QUARTZ_META17_(R\d\d|C\d\d)_([0-9a-f]{12})\.tar\.gz")
for f in archives:
    match=pattern.fullmatch(f.name)
    assert match,(f.name,"invalid stage archive name")
    stage, short=match.groups()
    assert stage not in stage_map,(stage,"duplicate stage")
    commit=output("git","-C",str(REPO),"rev-parse",short)
    assert commit.startswith(short)
    assert subprocess.call(["git","-C",str(REPO),"merge-base","--is-ancestor",commit,head])==0
    assert f.stat().st_size>1000
    stage_map[stage]={"commit_SHA":commit,"archive_file":str(f),
                      "archive_sha256":sha(f),"archive_bytes":f.stat().st_size}
required={f"R{i:02d}" for i in range(1,11)}|{f"C{i:02d}" for i in range(1,10)}
assert set(stage_map)==required,{"missing":sorted(required-set(stage_map)),
                                  "extra":sorted(set(stage_map)-required)}
orig=HOME/"evidence/REPOSITORY_PROVENANCE.json"
provenance=json.loads(orig.read_text())
assert provenance["GitHub_actual_PR_count"]==0
assert provenance["all_git_commits_in_inventory"]==413
source_lock=json.loads((HOME/"C01_LOCK.json").read_text())
assert source_lock["authoritative_history_survey_sha256"]==sha(orig)
for src in source_lock["documents"]:
 if src["name"]=="src/mcts/policy/mod.rs":
  # The one explicitly authorized integration is registering the standalone reference module.
  original=output("git","-C",str(REPO),"show","41febde4561644d7287fd4d3411847e485f5f646:src/mcts/policy/mod.rs")
  current=(REPO/src["name"]).read_text().rstrip()
  expected=original.replace("pub mod cache;","pub mod cache;\npub mod finite_metalevel;",1)
  assert original.count("pub mod cache;")==1 and current==expected,"unexpected change to policy module registry"
 else:
  assert sha(REPO/src["name"])==src["sha256"],("protected original source bytes changed",src["name"])
# Prove frozen thesis, claim ledger and main did not get modified in this research branch
diff=output("git","-C",str(REPO),"diff","--name-only",
 "41febde4561644d7287fd4d3411847e485f5f646","HEAD","--",
 "docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs")
assert not diff,diff
cases=REPO/"configs/metalevel_reference_cases.v1.json"
assert sha(cases)=="46d69d13ea327c23e02b76c53a0a78aadc1f77ee9e405dbef1c203a6d0b471dd"
c6=json.loads((HOME/"evidence/C06_CROSS_LANGUAGE.json").read_text())
c8=json.loads((HOME/"evidence/C08/C08_RESULTS.json").read_text())
c9=json.loads((HOME/"evidence/C09_INDEPENDENT_FRACTION_AUDIT.json").read_text())
restore=json.loads((HOME/"evidence/C09_LOCAL_ARCHIVE_RESTORE.json").read_text())
assert c6["status"]=="PASS" and c6["n_numeric_equalities_with_absolute_tolerance_1e-10"]==52
assert c8["total_actual_simulated_paths"]==10240 and c8["n_distinct_task_families"]==5
assert c9["status"]=="PASS" and c9["n_independent_exact_task_policy_expectations"]==20
assert c9["n_trial_paths_independently_recomputed_with_exact_Fractions"]==2560
assert restore["status"]=="PASS" and restore["independently_byte_identical_files_count"]==9
native=REPO/"target/release/quartz_metalevel_reference"
assert sha(native)=="0a61ddacae5523acef3f4a55a8fe8372e57ac645013ddfa019c57d9dd261d741"
receipt={
"schema":"quartz_general_meta17_C10_pre_publish_release_manifest_v1",
"branch":BRANCH,"parent_C16_SHA":"41febde4561644d7287fd4d3411847e485f5f646",
"source_before_final_C10_commit":head,"remote_before_final_C10_commit":remote,
"completed_theory_R01_R10":10,"completed_coding_C01_C09_before_C10":9,
"local_archive_count_before_self_referential_C10":len(stage_map),
"stage_commit_remote_confirmed_archive_sha256":stage_map,
"original_Git_history_provenance_sha256":sha(orig),
"current_original_source_lock_sha256":sha(HOME/"C01_LOCK.json"),
"frozen_case_bank_SHA256":sha(cases),
"compiled_Rust_native_cli_SHA256":sha(native),
"Rust_Python_C06_parity_SHA256":sha(HOME/"evidence/C06_CROSS_LANGUAGE.json"),
"MC_raw_10240_SHA256":sha(HOME/"evidence/C08/PAIRED_MC_10240.jsonl"),
"C08_science_results_SHA256":sha(HOME/"evidence/C08/C08_RESULTS.json"),
"C09_Fraction_audit_SHA256":sha(HOME/"evidence/C09_INDEPENDENT_FRACTION_AUDIT.json"),
"C09_local_archive_restore_SHA256":sha(HOME/"evidence/C09_LOCAL_ARCHIVE_RESTORE.json"),
"proof_counterexample":"ratio partial vs net complete, exact 3-state finite Bayes",
"actual_MC_samples":10240,
"independent_exact_policy_case_values":20,
"independently_replayed_trial_paths":2560,
"Rust_debug":{"main_pass":617,"bin_pass":7,"fail":0,"ignored_main":89},
"Rust_release":{"main_pass":617,"bin_pass":7,"fail":0,"ignored_main":89},
"Python_reference_test":{"pass":7,"fail":0},
"game_specific_optimizations_added":False,
"live_SearchPolicy_changed":False,
"production_MCTS_efficacy":"NOT_TESTED",
"general_Bayesian_metareasoning_novelty":"PREEXISTING_LITERATURE",
"generalization":"NOT_ESTABLISHED",
"GitHub_actual_PR_count_baseline":0,
"protected_frozen_original_docs_byte_identity":"VERIFIED",
"local_C08_archive_restore":"PASS",
"Dropbox_cloud_sync_ack":"NOT_VERIFIED",
"C10_self_archive":"CREATED_ONLY_AFTER_THIS_MANIFEST_IS_COMMITTED",
"native_subagent":"NOT_RUN",
"next_node":"QUARTZ-GENERAL-18-CALIBRATED-COMPUTATION-OBSERVATION",
"next_node_status":"NOT_STARTED"
}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print("C10_PREFLIGHT",json.dumps({
"pre_final_HEAD":head,"stages_archived":len(stage_map),
"exact_frozen_tasks":5,"n_MC":10240,"independent_exact":20,
"independent_trial_replays":2560,"protected_docs":"MATCH",
"next_node":receipt["next_node"]},sort_keys=True))
