#!/usr/bin/env python3
"""C21 C10 finite prepublication verification. C10's own archive cannot be self-referenced.

After commit/push a separate read-only postrelease process verifies C10.
"""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARCHIVES=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta21_backups")
BRANCH="research/quartz-general-c21-fast-cdf-20261122"
PARENT="ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4"
def cmd(*args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

head=cmd("git","rev-parse","HEAD")
remote=cmd("git","ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote and cmd("git","branch","--show-current")==BRANCH
subprocess.run(["git","merge-base","--is-ancestor",PARENT,head],cwd=ROOT,check=True)
pattern=re.compile(r"QUARTZ_GENERAL_C21_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
expected={f"R{i:02d}" for i in range(1,11)}|{f"C{i:02d}" for i in range(1,10)}|{"C07D","C07A","C08A"}
stages={}
for p in sorted(ARCHIVES.glob("QUARTZ_GENERAL_C21_*.tar.gz")):
 m=pattern.fullmatch(p.name)
 assert m,p.name
 stage,short=m.groups()
 assert stage not in stages,stage
 commit=cmd("git","rev-parse",short)
 assert commit.startswith(short)
 subprocess.run(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT,check=True)
 assert p.stat().st_size>1000
 stages[stage]={"commit":commit,"archive_file":str(p),
   "archive_sha256":sha(p),"archive_bytes":p.stat().st_size}
assert set(stages)==expected,{"missing":sorted(expected-set(stages)),
  "unexpected":sorted(set(stages)-expected)}
assert len(stages)==22 and stages["C09"]["commit"]==head
source_lock=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
assert source_lock["R10_completed_current_git_SHA"]==stages["R10"]["commit"]
for file in source_lock["source_files"]:
 assert sha(ROOT/file["path"])==file["sha256"],("previous authority altered",file["path"])
for file in ("docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs"):
 assert not cmd("git","diff","--name-only",PARENT,head,"--",file)

source=HOME/"SOURCE_WORLDS_C21_FROZEN.json"
train=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
price=HOME/"FROZEN_C21_RESOURCE_PRICE.json"
testraw=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
testnative=HOME/"evidence/C08_TEST/NATIVE_TEST_SHADOW_RECORDS.jsonl"
reportfile=HOME/"evidence/C08_TEST/C08_HELDOUT_NATIVE_EVSI_RESULTS.json"
auditfile=HOME/"evidence/C09_INDEPENDENT_AUDIT.json"
restorefile=HOME/"evidence/C09_LOCAL_RESTORE.json"
gridfile=HOME/"evidence/C09/NATIVE_GRID_TRAIN_ONLY_REPORT.json"
assert sha(source)=="036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8"
assert sha(train)=="db100b0ee9b4c8d8669c2c20d2beebabeb7bf3d74d017790259d667c10ae7bea"
assert sha(price)=="d53b6ffdfa422f5b274d0373d9839e958b96528126a7cc4a66c21a73b4401d01"
assert sha(testraw)=="cb834de8c9b312e1ac168c42adeee571c116600a32cadefb604dd257442dbd74"
assert sha(testnative)=="2b83a0f1d8734fa65153d8df325a75ac2ea8907f2983f2126a2b53f377ab4ec6"
assert sha(reportfile)=="5feecc957bf679b75b6feccc64e7fcbdb70d206ceca37ef477b19b74a67cdd16"
audit=json.loads(auditfile.read_text())
restore=json.loads(restorefile.read_text())
grid=json.loads(gridfile.read_text())
report=json.loads(reportfile.read_text())
parity=json.loads((HOME/"evidence/C06_NUMERIC_PARITY.json").read_text())
assert parity["status"]=="PASS" and parity["n_bounded_cases"]==163
assert parity["n_numeric_float_equalities"]==652 and parity["max_abs_discrepancy"]<1e-10
assert parity["full_grid_independent_cases"]>=69
assert report["n_actual_TEST_task_followups"]==144
assert report["n_new_independent_TEST_parent_groups"]==48
assert report["max_abs_EVSI_discrepancy_vs_separate_Python_reference"]<1e-10
assert audit["status"]=="PASS" and audit["actual_Python_TEST_stochastic_sequence_replays"]==144
assert audit["source_beta_exact_Fraction_joint_10_per_arm_checks"]==24
assert audit["sensor_independent_ordered_3_pow_8_checks"]==24
assert audit["native_Rust_StdRng_task_DETERMINISTIC_replays"]==72
assert restore["status"]=="PASS" and restore["n_byte_identical_files"]==16
assert grid["status"]=="NUMERICALLY_CHECKED_TRAIN_ONLY_NOT_TEST_POLICY"
assert grid["n_posterior_inputs"]==216 and grid["repetitions_per_case"]==20
assert grid["max_abs_gross_EVSI_difference"]<1e-10
assert grid["no_C21_TEST_examples_in_benchmark"] is True

native=ROOT/"target/release/quartz_c21_fast_evsi"
native_grid=ROOT/"target/release/quartz_c21_grid_reference"
assert sha(native)=="ad49fb9fbb125cb0193e454ee508bc083737bf446754200663143c5b0fcd4650"
assert sha(native_grid)=="f4786b642ee0cdbfd7c650872016d5bbde84478e911d62bff5128ed006d85f28"
for fname in ("CARGO_DEBUG.log","CARGO_RELEASE.log"):
 s=(HOME/"evidence/C09_LOGS"/fname).read_text()
 assert "617 passed; 0 failed; 89 ignored" in s
 assert s.count("3 passed; 0 failed")==3,(fname,"legacy C18-C20 CLI test count")
 assert s.count("5 passed; 0 failed")>=1
 assert s.count("7 passed; 0 failed")>=2
 assert "FAILED" not in s

previous={
 "/home/cosmosapjw/quantum/research/quartz_general_meta20_20261122":PARENT,
 "/home/cosmosapjw/quantum/research/quartz_general_meta19_20261122":"2a359ad049b6a0be4047e52ad281b665cf68e967",
 "/home/cosmosapjw/quantum/research/quartz_general_meta18_20261122":"097255bd00ffa436c3d43704ed201815e7cbb8ea",
 "/home/cosmosapjw/quantum/research/quartz_general_meta17_20261122":"e4dc4d6a0d5db03189d2f8465f6d5f019018f20c",
 "/home/cosmosapjw/quantum/research/quartz_cog06_20261121":"41febde4561644d7287fd4d3411847e485f5f646"
}
for path,expected_sha in previous.items():
 actual=subprocess.check_output(["git","-C",path,"rev-parse","HEAD"],text=True).strip()
 assert actual==expected_sha,(path,actual,expected_sha)
nextstage=json.loads((HOME/"NEXT_STAGE.json").read_text())
assert nextstage["next_not_executed"] is True
assert nextstage["next_research_id"]=="QUARTZ-GENERAL-22-INPROCESS-VOC-GATE-AND-REAL-COMPUTE-ACTION"

manifest={
"schema":"quartz_general_c21_C10_prepublication_manifest_v1",
"branch":BRANCH,"preC10_HEAD":head,"preC10_REMOTE_HEAD":remote,
"C20_parent_SHA":PARENT,
"R01_R10_actual_stages":10,"C01_C09_actual_stages":9,
"source_extra_stages":["C07D","C07A","C08A"],
"n_preC10_archive_and_remote_verified_stages":len(stages),
"stagewise_commit_archive_sha256":stages,
"protected_original_authority_and_previous_C16_C20_refs":"PASS",
"original_C01_source_lock_sha256":sha(HOME/"C01_SOURCE_LOCK.json"),
"source_worlds_sha256":sha(source),
"train_cal_raw_sha256":sha(train),
"TRAIN_only_cost_model_sha256":sha(price),
"TEST_raw_sha256":sha(testraw),
"native_TEST_records_sha256":sha(testnative),
"heldout_science_sha256":sha(reportfile),
"C09_independent_source_audit_sha256":sha(auditfile),
"C09_local_C08_archive_restore_sha256":sha(restorefile),
"C09_same_compiler_train_only_speed_sha256":sha(gridfile),
"native_Beta_release_binary_sha256":sha(native),
"native_full_grid_release_binary_sha256":sha(native_grid),
"Rust_debug":{"main_MCTS":617,"C17_CLI":7,"C18_CLI":3,"C19_CLI":3,
 "C20_CLI":3,"C21_fast_CLI":5,"C21_full_grid_CLI":7,"total_pass":645,"fail":0,"main_ignored":89},
"Rust_release":{"main_MCTS":617,"C17_CLI":7,"C18_CLI":3,"C19_CLI":3,
 "C20_CLI":3,"C21_fast_CLI":5,"C21_full_grid_CLI":7,"total_pass":645,"fail":0,"main_ignored":89},
"actual_original_parent_world_groups":240,"TEST_parent_groups":48,
"actual_source_stochastic_future_episodes_total":720,
"TEST_stochastic_future_episodes":144,
"independent_C09_Python_RNG_replays":144,
"independent_C09_Fraction_Beta_EVSI_checks":24,
"independent_C09_sensor_ordered_6561_EVSI_checks":24,
"independent_C09_Rust_native_StdRng_deterministic_replays":72,
"C09_archive_key_files_byte_restored":16,
"independent_C06_CDF_numeric_case_count":163,
"independent_C06_numeric_assertions":652,
"independent_C06_max_abs_error":parity["max_abs_discrepancy"],
"same_Rust_compiler_C09_K4_train_speedup":grid["speed_by_arms"]["4"]["median_same_compiler_speed_ratio"],
"primary_price_utility_per_us":0.004,
"Beta_C08_MEAN_net_always":report["domains"]["beta_arms"]["prices"]["0.004"]["mean_net_utility_gain_vs_immediate_STOP"]["ALWAYS"],
"Beta_C08_MEAN_net_internal_charged":report["domains"]["beta_arms"]["prices"]["0.004"]["mean_net_utility_gain_vs_immediate_STOP"]["C21_NATIVE_FUNCTION_CHARGED"],
"Beta_C08_MEAN_net_host_charged":report["domains"]["beta_arms"]["prices"]["0.004"]["mean_net_utility_gain_vs_immediate_STOP"]["C21_NATIVE_HOST_BOUNDARY_CHARGED"],
"sensor_C08_MEAN_net_host_charged":report["domains"]["finite_sensor"]["prices"]["0.004"]["mean_net_utility_gain_vs_immediate_STOP"]["C21_NATIVE_HOST_BOUNDARY_CHARGED"],
"source_model_math_and_limited_native_implementation":"PASS_SCOPED",
"general_cross_domain_model_gain":"NOT_ESTABLISHED",
"real_original_neural_MCTS_utility":"NOT_ESTIMABLE",
"active_search_STOP_or_measured_physical_CPU_saving":"NOT_RUN_SHADOW_ONLY",
"C10_own_archive":"CREATED_AFTER_COMMIT; VERIFY_POST_RELEASE",
"Dropbox_cloud_sync_ACK":"NOT_VERIFIED",
"native_subagent":"NOT_RUN",
"next_research_id":nextstage["next_research_id"],
"next_node_not_executed":True
}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print("C21_C10_PREFLIGHT",json.dumps({
 "status":"PASS",
 "preC10_head":head,"preC10_remotes_equal":True,
 "stages_before_C10":len(stages),
 "R_stages":10,"C_stages_before_C10":9,
 "extra_provenance_stages":3,
 "same_rust_train_K4_ratio":manifest["same_Rust_compiler_C09_K4_train_speedup"],
 "source_scientific_replay":144,"root_original_unchanged":True,
 "next":manifest["next_research_id"]},sort_keys=True))
