#!/usr/bin/env python3
"""C22 C10 PREPUBLICATION gate over real stage archives, source, tests and science.

C10 self-commit+archive must be published AFTER this deterministic manifest.
Separate final POSTRELEASE script checks the full 23 including C10.
"""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BACK=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta22_backups")
BRANCH="research/quartz-general-c22-inprocess-gate-20261122"
PARENT="351c63ae810b1fa633fe2a75440bf7d0929f92e4"
def cmd(*args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=cmd("git","rev-parse","HEAD")
remote=cmd("git","ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote and cmd("git","branch","--show-current")==BRANCH
subprocess.run(["git","merge-base","--is-ancestor",PARENT,head],cwd=ROOT,check=True)
pattern=re.compile(r"QUARTZ_GENERAL_C22_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
stage={}
for p in sorted(BACK.glob("QUARTZ_GENERAL_C22_*.tar.gz")):
 m=pattern.fullmatch(p.name)
 assert m,p.name
 name,short=m.groups()
 assert name not in stage
 committed=cmd("git","rev-parse",short)
 assert committed.startswith(short)
 subprocess.run(["git","merge-base","--is-ancestor",committed,head],cwd=ROOT,check=True)
 assert p.stat().st_size>1000
 stage[name]={"commit":committed,"local_git_archive":str(p),
               "archive_sha256":sha(p),"archive_size":p.stat().st_size}
expected={f"R{i:02d}" for i in range(1,11)}|{f"C{i:02d}" for i in range(1,10)}|{"C07D","C07A","C08A"}
assert set(stage)==expected,{"missing":sorted(expected-set(stage)),"unexpected":sorted(set(stage)-expected)}
assert len(stage)==22 and stage["C09"]["commit"]==head
lock=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
assert lock["R10_HEAD_at_C01"]==stage["R10"]["commit"]
assert lock["parent_C21_exact_HEAD"]==PARENT
for f in lock["protected_originals"]:
 assert sha(ROOT/f["path"])==f["sha256"],("immutable original changed",f["path"])
for f in ("docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs"):
 assert not cmd("git","diff","--name-only",PARENT,head,"--",f)
files={
 "C04_new_source":HOME/"SOURCE_WORLDS_C22_FROZEN.json",
 "C07D_train_cal_raw":HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl",
 "C07A_resource_price":HOME/"FROZEN_C22_RESOURCE_PRICE.json",
 "C08A_test_raw":HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl",
 "C08_active_native_samples":HOME/"evidence/C08_TEST/C22_ACTIVE_ALL_POLICIES.jsonl",
 "C08_scientific_score":HOME/"evidence/C08_TEST/C08_ACTUAL_ACTIVE_RESULTS.json",
 "C09_independent_source_audit":HOME/"evidence/C09_INDEPENDENT_AUDIT.json",
 "C09_local_restore":HOME/"evidence/C09_LOCAL_RESTORE.json",
 "C06_source_independent_control":HOME/"evidence/C06_ACTIVE_INDEPENDENT_CHECK.json",
 "R07_exact_Fraction":HOME/"theory/R07_EXACT.json",
 "C03_independent_bound_check":HOME/"coding/C03_BOUND_CHECK.json"
}
hashes={k:sha(v) for k,v in files.items()}
expected_hashes={
 "C04_new_source":"f7bffc98710bae151d0ac909ddeb19fc04f74d51a4273f6b16f7e7583f61398d",
 "C07D_train_cal_raw":"50819842714731c94b161dc188d69f9c473b71e31788cb8136e54b4252263edc",
 "C07A_resource_price":"7814df9043a2814dfc198f0407ca35fc5808236506e2347ea7983e59db0bdab1",
 "C08A_test_raw":"807fff205d6b7c4b71d9cbd9e57e05f472635ee609f547b8ac68f9588d769dd0",
 "C08_active_native_samples":"6f6eb28abd884b9ed4bca1d05933b90d25e293bf9d6f0b5ce05c7526d6d0e0f0",
 "C08_scientific_score":"b91d6fff56466047b56361ce4cc87b8f60dad365ed4137bf4e3baea7bbb29c64"}
for k,v in expected_hashes.items():assert hashes[k]==v,(k,hashes[k],v)
science=json.loads(files["C08_scientific_score"].read_text())
ind=json.loads(files["C09_independent_source_audit"].read_text())
rest=json.loads(files["C09_local_restore"].read_text())
ctrl=json.loads(files["C06_source_independent_control"].read_text())
math=json.loads(files["R07_exact_Fraction"].read_text())
stability=json.loads(files["C03_independent_bound_check"].read_text())
assert science["n_independent_TEST_original_parent_worlds"]==48
assert science["n_TEST_original_examples"]==144
assert science["n_independent_model_value_vs_C20_reference_checked"]==144
assert science["max_abs_model_EVSI_reference_error"]<1e-10
assert science["active_actual_stop_physically_not_sampled"] if False else science["actual_active_STOP_physically_not_sampled"]
assert ind["status"]=="PASS"
assert ind["original_TEST_python_future_RNG_replays"]==144
assert ind["Beta_exact_Fraction_CDF_n20_bounded_parent_checks"]==24
assert ind["sensor_independent_ordered_likelihood_parent_checks"]==24
assert ind["max_abs_independent_gross_EVSI_error"]<1e-10
assert ind["actual_STOP_all_zero_additional_draws_and_zero_signed_change"]=="PASS"
assert rest["status"]=="PASS" and rest["n_key_original_science_and_code_files_identical"]==18
assert ctrl["status"]=="PASS" and ctrl["GATED_n_physical_STOP"]>0
assert math["status"]=="PASS" and stability["status"]=="PASS"
main=science["price_scenarios"]["0.004"]["families"]
assert main["beta_arms"]["GATED_n_true_STOP"]==48
assert main["finite_sensor"]["GATED_n_true_STOP"]==54
assert main["beta_arms"]["GATED_ACTUALLY_SKIPPED_future_draw_count"]==1092
assert main["finite_sensor"]["GATED_ACTUALLY_SKIPPED_future_draw_count"]==199
assert main["beta_arms"]["policy_mean_realized_net_gain_vs_free_STOP"]["GATED"]<main["beta_arms"]["policy_mean_realized_net_gain_vs_free_STOP"]["ALWAYS"]
assert main["finite_sensor"]["policy_mean_realized_net_gain_vs_free_STOP"]["GATED"]<main["finite_sensor"]["policy_mean_realized_net_gain_vs_free_STOP"]["ALWAYS"]
for log in ("CARGO_DEBUG.log","CARGO_RELEASE.log"):
 text=(HOME/"evidence/C09_LOGS"/log).read_text()
 assert "617 passed; 0 failed; 89 ignored" in text
 assert sum(line.startswith("test result: ok. 3 passed; 0 failed") for line in text.splitlines())==3
 assert "13 passed; 0 failed" in text and "7 passed; 0 failed" in text
 assert "FAILED" not in text
py=(HOME/"evidence/C09_LOGS/PYTHON_C22_SOURCE_ONLY.log").read_text()
assert "Ran 2 tests" in py and "\nOK\n" in py
native_sha=sha(ROOT/"target/release/quartz_c22_active_gate")
assert native_sha=="1dfa410c87422c985e21cd8d6a47f60bf9945aff2c4e199636a8a49325859a8e"
prior={
"/home/cosmosapjw/quantum/research/quartz_general_meta21_20261122":PARENT,
"/home/cosmosapjw/quantum/research/quartz_general_meta20_20261122":"ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4",
"/home/cosmosapjw/quantum/research/quartz_general_meta19_20261122":"2a359ad049b6a0be4047e52ad281b665cf68e967",
"/home/cosmosapjw/quantum/research/quartz_general_meta18_20261122":"097255bd00ffa436c3d43704ed201815e7cbb8ea",
"/home/cosmosapjw/quantum/research/quartz_general_meta17_20261122":"e4dc4d6a0d5db03189d2f8465f6d5f019018f20c",
"/home/cosmosapjw/quantum/research/quartz_cog06_20261121":"41febde4561644d7287fd4d3411847e485f5f646"}
for path,expected_git in prior.items():
 actual=subprocess.check_output(["git","-C",path,"rev-parse","HEAD"],text=True).strip()
 assert actual==expected_git,(path,actual,expected_git)
nextj=json.loads((HOME/"NEXT_STAGE.json").read_text())
assert nextj["next_not_executed"] and nextj["next_research_id"]=="QUARTZ-GENERAL-23-CHEAP-GATE-AND-MATCHED-WORKLOAD-ECONOMICS"
record={
 "schema":"quartz_C22_C10_prepublish_manifest_v1",
 "C22_branch":BRANCH,"preC10_git_HEAD":head,"preC10_remote_HEAD":remote,
 "parent_C21_SHA":PARENT,
 "actual_R01_R10_stages":10,"actual_C01_C09_stages":9,
 "extra_provenance_stages":["C07D","C07A","C08A"],
 "n_distinct_preC10_stage_archives":len(stage),
 "per_stage_git_commit_local_archive_SHA256":stage,
 "source_science_file_sha256":hashes,
 "source_world_parent_groups_total":240,"independent_TEST_world_groups":48,
 "Python_source_original_future_trajectories":720,
 "C22_TEST_actual_native_policy_CASES_per_lambda":144,
 "primary_price_utility_per_measured_us":0.004,
 "C06_standalone_synthetic_cases":ctrl["n_synthetic_source_cases"],
 "C06_genuine_STOP_count":ctrl["GATED_n_physical_STOP"],
 "C06_unexecuted_future_draws":ctrl["GATED_n_truly_unexecuted_future_draws"],
 "C08_actual_PRIMARY_Beta_STOPS":48,
 "C08_actual_PRIMARY_sensor_STOPS":54,
 "C08_actual_PRIMARY_Beta_future_draws_not_executed":1092,
 "C08_actual_PRIMARY_sensor_future_draws_not_executed":199,
 "C08_actual_primary_GATED_vs_ALWAYS_Beta_net_mean":main["beta_arms"]["policy_mean_realized_net_gain_vs_free_STOP"]["GATED"]-main["beta_arms"]["policy_mean_realized_net_gain_vs_free_STOP"]["ALWAYS"],
 "C08_actual_primary_GATED_vs_ALWAYS_sensor_net_mean":main["finite_sensor"]["policy_mean_realized_net_gain_vs_free_STOP"]["GATED"]-main["finite_sensor"]["policy_mean_realized_net_gain_vs_free_STOP"]["ALWAYS"],
 "C08_actual_measured_CPU_time_economic_superiority":"NOT_ESTABLISHED; worse than ALWAYS at all four prices in this bounded TEST",
 "C09_independent_exact_Beta_fraction_checks":24,
 "C09_independent_sensor_ordered_checks":24,
 "C09_independent_source_sequence_replays":144,
 "C09_same_Rust_RNG_deterministic_replay":144,
 "C09_Git_archive_local_keyfiles_byte_restored":18,
 "native_Rust_compiled_sha256":native_sha,
 "Rust_DEBUG":{"main_MCTS":617,"main_ignored":89,"C17":7,"C18":3,"C19":3,"C20":3,"C21_fast":5,"C21_grid":7,"C22_active":13,"total_pass":658,"FAIL":0},
 "Rust_RELEASE":{"main_MCTS":617,"main_ignored":89,"C17":7,"C18":3,"C19":3,"C20":3,"C21_fast":5,"C21_grid":7,"C22_active":13,"total_pass":658,"FAIL":0},
 "Python_C22_source_only_focused_tests":{"PASS":2,"FAIL":0},
 "original_THESIS_CLAIM_LEDGER_GameState_and_previous_C16_C21":"UNCHANGED_SHA_VERIFIED",
 "true_neural_MCTS_utility":"NOT_ESTIMABLE",
 "live_real_SearchPolicy_STOP":"NOT_DEPLOYED",
 "C22_active_non_game_RNG_draw_savings":"ACTUALLY_WITHHELD_FROM_ACTIVE_PATH",
 "global_experiment_total_CPU_saving":"NOT_CLAIMED_SEPARATE_ALWAYS_CONTROLS_PHYSICALLY_EXECUTED",
 "NEVER_baseline":"MEASURED_NOOP_WRAPPER; free theoretical immediate STOP=0",
 "C10_self_archive":"CREATE_AFTER_C10_COMMIT",
 "native_subagent":"NOT_RUN","Dropbox_cloud_sync_ACK":"NOT_VERIFIED",
 "next_research_id":nextj["next_research_id"],"next_still_unexecuted":True
}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C22_C10_PREFLIGHT",json.dumps({"status":"PASS","C22_prior_archive_count":len(stage),
 "R_stages":10,"C_stages_before_C10":9,"extra_stages":3,
 "preC10_git_SHA":head,"physical_source_STOPs_primary":[48,54],
 "actual_future_draws_skipped_primary":[1092,199],
 "native_tests_each":658,"next":nextj["next_research_id"]},sort_keys=True))
