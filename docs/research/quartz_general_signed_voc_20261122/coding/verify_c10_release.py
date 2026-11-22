#!/usr/bin/env python3
"""C10 prepublish source/science/source-family/archive verification.

The C10 own commit and archive are created AFTER this finite manifest, so no
self-referential SHA is invented. Post-release tests independently verify C10.
"""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
B=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta19_backups")
BRANCH="research/quartz-general-c19-signed-voc-20261122"
C18="097255bd00ffa436c3d43704ed201815e7cbb8ea"
def cmd(*a):return subprocess.check_output(a,cwd=ROOT,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=cmd("git","rev-parse","HEAD")
remote=cmd("git","ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote and cmd("git","branch","--show-current")==BRANCH
assert subprocess.run(["git","merge-base","--is-ancestor",C18,head],cwd=ROOT).returncode==0
rx=re.compile(r"QUARTZ_GENERAL_C19_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
archives=sorted(B.glob("QUARTZ_GENERAL_C19_*.tar.gz"))
stages={}
for archive in archives:
 m=rx.fullmatch(archive.name)
 assert m,archive.name
 stage,short=m.groups()
 assert stage not in stages,stage
 commit=cmd("git","rev-parse",short)
 assert commit.startswith(short)
 assert subprocess.run(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT).returncode==0
 assert archive.stat().st_size>1000
 stages[stage]={"commit":commit,"archive_file":str(archive),
   "archive_sha256":sha(archive),"archive_bytes":archive.stat().st_size}
expected={f"R{i:02}" for i in range(1,11)}|{f"C{i:02}" for i in range(1,10)}|{"C07A","C08A"}
assert set(stages)==expected,{"missing":sorted(expected-set(stages)),
 "extra":sorted(set(stages)-expected)}
lock=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
assert lock["current_source_HEAD"]==stages["R10"]["commit"]
for o in lock["original_files"]:
 assert sha(ROOT/o["path"])==o["sha256"],("original C18 source changed",o["path"])
for p in ["docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs"]:
 assert not cmd("git","diff","--name-only",C18,head,"--",p)
source=HOME/"SOURCE_WORLDS_FROZEN.json"
trained=HOME/"SHADOW_SIGNED_MODEL_FROZEN.json"
t=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
train=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
r6=json.loads((HOME/"evidence/C06_PARITY.json").read_text())
r8=json.loads((HOME/"evidence/C08_TEST/C08_HELDOUT_SIGNED_RESULTS.json").read_text())
r9=json.loads((HOME/"evidence/C09_INDEPENDENT_AUDIT.json").read_text())
restore=json.loads((HOME/"evidence/C09_LOCAL_RESTORE.json").read_text())
assert sha(source)=="8fce353c5efd3a5ac2a643e5a799d46f8d69c6c429cdff0fb5418a43ea43292e"
assert sha(trained)=="5fdd172a13d51c4d17dc8e7829fb7c5518421028046ebac6e418a1158adf6735"
assert sha(t)=="c9a8e3671a7612eac3e98c3b861ea72871bd9d28ca931f87f18d25bc8a624340"
assert sha(train)=="5ecef7ca8a3074965f8b335ff98891c66abdf4d5e489246379980aa5651ed8f7"
assert r6["status"]=="PASS" and r6["float_equalities_within_tolerance"]==480
assert r8["n_actual_utility_scored_extra_computation_paths"]==144
assert r8["n_test_parent_groups_total"]==48
assert r9["status"]=="PASS" and r9["all_test_random_streams_replayed"]==144
assert restore["status"]=="PASS" and restore["n_files_byte_identical"]==13
assert sha(ROOT/"target/release/quartz_c19_shadow")=="6761476e1901926f85ba3c0410f1221d34834d97c219f90a88f5f5f92bd58f68"
nextj=json.loads((HOME/"NEXT_STAGE.json").read_text())
assert nextj["next_not_executed"]
assert nextj["next_research_id"]=="QUARTZ-GENERAL-20-POSTERIOR-PREDICTIVE-VOC-AND-COST"
C17=subprocess.check_output(["git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta17_20261122","rev-parse","HEAD"],text=True).strip()
C18_NOW=subprocess.check_output(["git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta18_20261122","rev-parse","HEAD"],text=True).strip()
C16=subprocess.check_output(["git","-C","/home/cosmosapjw/quantum/research/quartz_cog06_20261121","rev-parse","HEAD"],text=True).strip()
assert C17=="e4dc4d6a0d5db03189d2f8465f6d5f019018f20c"
assert C18_NOW==C18
assert C16=="41febde4561644d7287fd4d3411847e485f5f646"
record={"schema":"quartz_general_c19_C10_prepublication_manifest_v1",
 "source_pre_C10_git_SHA":head,"source_remote_pre_C10_git_SHA":remote,"branch":BRANCH,
 "C18_frozen_parent_SHA":C18,"C17_current_unchanged":C17,"C16_current_unchanged":C16,
 "R01_R10_done":10,"C01_C09_done":9,
 "extra_stage_backups":["C07A","C08A"],
 "n_prepublish_stages_with_actual_archive":len(stages),
 "per_stage_commit_and_SHA_archive":stages,
 "original_thesis_claim_ledger_and_game_state_untouched":"PASS",
 "source_lock_sha256":sha(HOME/"C01_SOURCE_LOCK.json"),
 "source_worlds_sha256":sha(source),
 "C07A_frozen_model_sha256":sha(trained),
 "C07A_train_cal_raw_sha256":sha(train),"C08A_test_raw_sha256":sha(t),
 "C08_heldout_science_sha256":sha(HOME/"evidence/C08_TEST/C08_HELDOUT_SIGNED_RESULTS.json"),
 "C09_independent_audit_sha256":sha(HOME/"evidence/C09_INDEPENDENT_AUDIT.json"),
 "C09_local_recovery_sha256":sha(HOME/"evidence/C09_LOCAL_RESTORE.json"),
 "compiled_C19_Rust_shadow_sha256":sha(ROOT/"target/release/quartz_c19_shadow"),
 "Rust_C19_Python_numeric_96_x_5_parity":"PASS",
 "independent_all_144_test_stochastic_histories_replayed":"PASS",
 "C08_tar_local_restored_exact_source_files":13,
 "real_extra_stochastic_calculations_total":720,
 "independent_parent_groups_total":240,
 "heldout_parent_groups":48,
 "heldout_samples":144,
 "primary_lambda":0.004,
 "heldout_beta_signed_two_part_mse":r8["heldout_domains"]["beta_arms"]["signed_gain_prediction"]["two_part_signed"]["mse"],
 "heldout_sensor_signed_two_part_mse":r8["heldout_domains"]["finite_sensor"]["signed_gain_prediction"]["two_part_signed"]["mse"],
 "heldout_beta_primary_shadow_net":r8["heldout_domains"]["beta_arms"]["prices"]["0.004"]["mean_paid_net_utility_gain_vs_STOP"]["SIGNED_NET_SHADOW"],
 "heldout_sensor_primary_shadow_net":r8["heldout_domains"]["finite_sensor"]["prices"]["0.004"]["mean_paid_net_utility_gain_vs_STOP"]["SIGNED_NET_SHADOW"],
 "Rust_DEBUG":{"MCTS":617,"C17_reference":7,"C18_shadow":3,"C19_shadow":3,"FAIL":0,"IGNORED_MCTS":89},
 "Rust_RELEASE":{"MCTS":617,"C17_reference":7,"C18_shadow":3,"C19_shadow":3,"FAIL":0,"IGNORED_MCTS":89},
 "Python_C19":{"PASS":5,"FAIL":0},
 "source_family_success_or_general_transfer":"NOT_ESTABLISHED",
 "real_neural_MCTS_signed_utility":"NOT_ESTIMABLE",
 "no_main_deployment_or_PR":True,
 "actual_live_CPU_time_saved":"NOT_MEASURED_COUNTERFACTUAL_ONLY",
 "C10_self_archive":"TO_BE_PUBLISHED_AFTER_C10_COMMIT",
 "local_Dropbox_archive_SHA":"VERIFIED_STAGEWISE",
 "Dropbox_cloud_sync_ACK":"NOT_VERIFIED",
 "native_subagent":"NOT_RUN",
 "next_research_id":nextj["next_research_id"],
 "next_status":"NOT_STARTED"}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C19_C10_PREFLIGHT",json.dumps({"status":"PASS","preC10_SHA":head,
 "verified_pre_C10_stage_archives":len(stages),
 "source_families":2,"actual_source_computations":720,
 "heldout":144,"independent_RNG_replay":144,
 "next":record["next_research_id"]},sort_keys=True))
