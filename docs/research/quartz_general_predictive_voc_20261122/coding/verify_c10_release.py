#!/usr/bin/env python3
"""C20 pre-C10 release gate: source/science/tests/21 prior commits & real archives.

The C10 commit itself cannot be included within its own manifest. Post-release
separate check verifies C10 actual remote SHA/archive and exact source bytes.
"""
from pathlib import Path
import re,hashlib,json,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARCHIVES=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta20_backups")
BRANCH="research/quartz-general-c20-predictive-voc-20261122"
PARENT="2a359ad049b6a0be4047e52ad281b665cf68e967"
def cmd(*a):return subprocess.check_output(a,cwd=ROOT,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=cmd("git","rev-parse","HEAD")
remote=cmd("git","ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote and cmd("git","branch","--show-current")==BRANCH
assert subprocess.run(["git","merge-base","--is-ancestor",PARENT,head],cwd=ROOT).returncode==0
expected={f"R{i:02d}" for i in range(1,11)}|{f"C{i:02d}" for i in range(1,10)}|{"C07A","C08A"}
rx=re.compile(r"QUARTZ_GENERAL_C20_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
stages={}
for file in sorted(ARCHIVES.glob("QUARTZ_GENERAL_C20_*.tar.gz")):
 match=rx.fullmatch(file.name)
 assert match,file.name
 stage,short=match.groups()
 assert stage not in stages
 commit=cmd("git","rev-parse",short)
 assert commit.startswith(short)
 subprocess.run(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT,check=True)
 assert file.stat().st_size>1000
 stages[stage]={"commit":commit,"archive_file":str(file),
     "archive_sha256":sha(file),"archive_size":file.stat().st_size}
assert set(stages)==expected,{"missing":sorted(expected-set(stages)),
 "extra":sorted(set(stages)-expected)}
lock=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
assert lock["C20_before_C01_head"]==stages["R10"]["commit"]
for file in lock["source_files"]:
 assert sha(ROOT/file["file"])==file["sha256"],file["file"]
for p in ["docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs"]:
 assert not cmd("git","diff","--name-only",PARENT,head,"--",p)
source=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
C07=HOME/"FROZEN_C20_RESOURCE_PRICE.json"
C08RAW=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
C07RAW=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
C08R=HOME/"evidence/C08_TEST/C08_HELDOUT_EXACT_EVSI.json"
C09A=HOME/"evidence/C09_INDEPENDENT_AUDIT.json"
C09REST=HOME/"evidence/C09_LOCAL_RESTORE.json"
C09NATIVE=HOME/"evidence/C09_NATIVE_RUST_LATENCY.json"
C06=json.loads((HOME/"evidence/C06_RUST_PYTHON_PARITY.json").read_text())
C08=json.loads(C08R.read_text())
C09=json.loads(C09A.read_text())
restore=json.loads(C09REST.read_text())
native=json.loads(C09NATIVE.read_text())
fast=json.loads((HOME/"evidence/C10_FAST_BETA_MAX_CDF_WITNESS.json").read_text())
assert sha(source)=="d6cf96c26d9ac913963f702d9b0418f0002038979f6a470da3865999368e4ae7"
assert sha(C07)=="530ca204c96b5fda4f3b060e8ebd87b52e4c4ec7e97be42dac56da93ca81cb04"
assert sha(C08RAW)=="5ab524aff074d8f441f09679309807079c694035eded182d86c5517d82cf5c1f"
assert sha(C07RAW)=="009774dd99f1e2034e74f449c74a13f3470731737725c60fd7e7a142cfdf4a9e"
assert C06["status"]=="PASS" and C06["numeric_equalities"]==384
assert C08["n_actual_extra_stochastic_computations"]==144
assert C08["n_distinct_test_parent_worlds"]==48
assert C08["C20_controlled_live_policy_status"]=="SHADOW_ONLY"
assert C09["status"]=="PASS" and C09["all_actual_TEST_RNG_observation_paths_replayed"]==144
assert C09["independent_Beta_Polya_urn_recurrence_exact_EVSI_paths"]==24
assert C09["independent_sensor_ordered_sequence_3_pow_8_EVSI_paths"]==24
assert restore["status"]=="PASS" and restore["n_byte_identical_files"]==15
assert native["status"]=="MEASURED" and native["n_native_function_evaluations"]==216
assert fast["status"]=="NUMERICALLY_CHECKED_TRAIN_ONLY"
assert fast["n_train_initial_only_reference_cases"]==216
assert fast["max_absolute_EVSI_disagreement"]<1e-10
assert sha(ROOT/"target/release/quartz_c20_shadow")=="7943f2719d26793f0d73c0754933aba14693b970cb8f03a0f594acd3ba6e15f5"
future=json.loads((HOME/"NEXT_STAGE.json").read_text())
assert future["next_not_executed"]
assert future["next_research_id"]=="QUARTZ-GENERAL-21-CDF-EXACT-EVSI-NATIVE-RESOURCE-GATE"
# C09 actually executed Rust debug+release unit tests, not just previous C06.
debug=(HOME/"evidence/C09_LOGS/CARGO_DEBUG.log").read_text()
release=(HOME/"evidence/C09_LOGS/CARGO_RELEASE.log").read_text()
for raw in (debug,release):
 assert "617 passed; 0 failed; 89 ignored" in raw
 assert raw.count("3 passed; 0 failed")==3
 assert "7 passed; 0 failed" in raw
for p,n in [("PYTHON_EXACT.log","Ran 6 tests"),("PYTHON_ADAPTER.log","Ran 2 tests")]:
 value=(HOME/"evidence/C09_LOGS"/p).read_text()
 assert n in value and "\nOK\n" in value
C19=cmd("git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta19_20261122","rev-parse","HEAD")
C18=cmd("git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta18_20261122","rev-parse","HEAD")
C17=cmd("git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta17_20261122","rev-parse","HEAD")
C16=cmd("git","-C","/home/cosmosapjw/quantum/research/quartz_cog06_20261121","rev-parse","HEAD")
assert C19==PARENT
assert C18=="097255bd00ffa436c3d43704ed201815e7cbb8ea"
assert C17=="e4dc4d6a0d5db03189d2f8465f6d5f019018f20c"
assert C16=="41febde4561644d7287fd4d3411847e485f5f646"
m={"schema":"quartz_general_c20_C10_prepublication_full_scientific_manifest_v1",
 "C20_PRE_C10_git_head":head,"C20_PRE_C10_remote_SHA":remote,
 "C19_parent":PARENT,
 "C16_C19_prior_refs_unchanged":"PASS",
 "original_THESIS_CLAIM_LEDGER_GameState_byte_protected":"PASS",
 "n_R_stages_verified":10,"n_C_stages_pre_C10_verified":9,
 "n_extra_source_freeze_stages":2,"n_pre_C10_Git_archive_remote_receipts":len(stages),
 "stagewise_commit_and_local_sha256_archive":stages,
 "original_C01_source_lock_SHA256":sha(HOME/"C01_SOURCE_LOCK.json"),
 "C04_source_parent_bank_sha256":sha(source),
 "C07A_frozen_cost_model_sha256":sha(C07),
 "C07A_train_raw_sha256":sha(C07RAW),
 "C08A_test_raw_sha256":sha(C08RAW),
 "C08_heldout_EVSI_results_sha256":sha(C08R),
 "C09_independent_source_math_sha256":sha(C09A),
 "C09_local_restore_sha256":sha(C09REST),
 "C09_native_latency_sha256":sha(C09NATIVE),
 "C10_constructive_CDF_witness_sha256":sha(HOME/"evidence/C10_FAST_BETA_MAX_CDF_WITNESS.json"),
 "C20_compiled_Rust_shadow_sha256":sha(ROOT/"target/release/quartz_c20_shadow"),
 "real_extra_stochastic_sample_paths":720,
 "source_parent_latent_worlds":240,
 "heldout_actual_sample_paths":144,
 "C09_independent_random_source_replay":144,
 "C09_independent_alt_EVSI_checks":{"beta_Polya":24,"sensor_ordered":24},
 "C10_train_only_fast_CDF_evsi_checks":216,
 "heldout_new_two_domain_exact_vs_C19_gain_MSE":{"beta":[.013811469186467985,.014119182812955619],
 "sensor":[.010683879751465786,.011190799301142992]},
 "C20_actual_priced_Python_meta_policy_primary_net":{"beta":-18.48791027777778,
 "sensor":-.8599278333333333},
 "C09_bounded_standalone_Rust_sensor_median_us":native["native_function_eval_median_us"],
 "C10_FAST_CDF_beta_K4_python_median_speed_ratio":fast["measured_wall_us_by_arm_count"]["4"]["median_speed_ratio_individual"],
 "Rust_DEBUG":{"MCTS":617,"C17":7,"C18":3,"C19":3,"C20":3,"failures":0,"MCTS_ignored":89},
 "Rust_RELEASE":{"MCTS":617,"C17":7,"C18":3,"C19":3,"C20":3,"failures":0,"MCTS_ignored":89},
 "Python_C20":{"exact_tests_pass":6,"adapter_tests_pass":2},
 "live_SearchPolicy_changed":False,"real_Game_signed_EVSI":"NOT_ESTIMABLE",
 "any_actual_hardware_CPU_saved":"NOT_MEASURED_SHADOW_ONLY",
 "Beta_fast_Rust_full_port":"NOT_IMPLEMENTED",
 "C20_TEST_general_superiority":"NOT_ESTABLISHED",
 "native_subagent":"NOT_RUN",
 "cloud_Dropbox_sync_ACK":"NOT_VERIFIED",
 "C10_self_archive":"CREATED_AFTER_THIS_COMMIT",
 "next_node":future["next_research_id"],"next_node_started":False}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
print("C20_C10_PREFLIGHT",json.dumps({"status":"PASS","prior_stage_archives":len(stages),
  "preC10_head":head,"all_protected_refs":"PASS",
  "actual_source_paths":720,"heldout":144,"new_CDF_train_check":216,
  "new_CDF_K4_speedup":m["C10_FAST_CDF_beta_K4_python_median_speed_ratio"],
  "next":m["next_node"]},sort_keys=True))
