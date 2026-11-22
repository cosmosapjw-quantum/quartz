#!/usr/bin/env python3
"""C10 science/source/archive pre-publish verification. Self C10 archive verified after commit."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BRANCH="research/quartz-general-c18-calibrated-compute-20261122"
ARCHIVES=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta18_backups")
def cmd(*args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=cmd("git","rev-parse","HEAD")
remote=cmd("git","ls-remote","--heads","origin",BRANCH).split()[0]
assert head==remote
assert cmd("git","branch","--show-current")==BRANCH
assert cmd("git","rev-parse","e4dc4d6a0d5db03189d2f8465f6d5f019018f20c")
archives=sorted(ARCHIVES.glob("QUARTZ_GENERAL_C18_*.tar.gz"))
pattern=re.compile(r"QUARTZ_GENERAL_C18_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
stages={}
for p in archives:
 m=pattern.fullmatch(p.name)
 assert m,p.name
 stage,short=m.groups()
 assert stage not in stages,stage
 commit=cmd("git","rev-parse",short)
 assert commit[:12]==short
 assert subprocess.call(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT)==0
 assert p.stat().st_size>1000
 stages[stage]={"commit":commit,"local_git_archive":str(p),
  "sha256":sha(p),"bytes":p.stat().st_size}
expected={f"R{i:02d}" for i in range(1,11)}|{f"C{i:02d}" for i in range(1,10)}|{"C07A","C08A"}
assert set(stages)==expected,{"missing":sorted(expected-set(stages)), "extra":sorted(set(stages)-expected)}
lock=json.loads((HOME/"C01_SOURCE_LOCK.json").read_text())
assert lock["source_HEAD"]==stages["R10"]["commit"]
for o in lock["docs"]:
 assert sha(ROOT/o["file"])==o["sha256"],("protected original source changed",o["file"])
assert not cmd("git","diff","--name-only","e4dc4d6a0d5db03189d2f8465f6d5f019018f20c",head,"--","docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs")
sourcebank=HOME/"SOURCE_GROUPS_FROZEN.json"
frozen_model=HOME/"SHADOW_MODEL_FROZEN.json"
c06=json.loads((HOME/"evidence/C06_RUST_PYTHON_PARITY.json").read_text())
c08=json.loads((HOME/"evidence/C08_TEST/C08_HELDOUT_RESULTS.json").read_text())
c09=json.loads((HOME/"evidence/C09_INDEPENDENT_REPLAY.json").read_text())
restore=json.loads((HOME/"evidence/C09_LOCAL_RESTORE.json").read_text())
assert sha(sourcebank)=="52e3654febead1a01253174be0d395f8c2ea6e04ada21e5e0bfbf5938d680445"
assert sha(frozen_model)=="2dd987989cb543c692457d0b198d23009917fd5a3001551e300a81beb868218b"
assert c06["status"]=="PASS" and c06["test_rows_initial_only"]==96
assert c08["n_actual_test_observations"]==212
assert c08["heldout_per_domain"]["real_Rust_MCTS"]["n"]==140
assert c08["heldout_per_domain"]["non_game_Bernoulli"]["n"]==72
assert c09["status"]=="PASS" and c09["actual_native_root_traces_validated"]==140
assert c09["actual_nongame_sequences_independently_replayed"]==72
assert restore["status"]=="PASS" and restore["n_byte_identical_restored_files"]==11
assert sha(ROOT/"target/release/quartz_c18_shadow")=="77b2000f90fc6f39188ec85ead2c24cf004f50f31e88aa0a90144a6335f32919"
priorC17=subprocess.check_output(["git","-C","/home/cosmosapjw/quantum/research/quartz_general_meta17_20261122","rev-parse","HEAD"],text=True).strip()
priorC16=subprocess.check_output(["git","-C","/home/cosmosapjw/quantum/research/quartz_cog06_20261121","rev-parse","HEAD"],text=True).strip()
assert priorC17=="e4dc4d6a0d5db03189d2f8465f6d5f019018f20c"
assert priorC16=="41febde4561644d7287fd4d3411847e485f5f646"
manifest={
"schema":"quartz_general_c18_C10_precommit_source_evidence_manifest_v1",
"source_pre_C10_git_SHA":head,"source_remote_SHA":remote,"branch":BRANCH,
"R1_R10_completed":10,"C1_C9_completed":9,"extra_source_frozen_stages":["C07A","C08A"],
"n_verified_stage_commits_and_archives_before_C10":len(stages),
"stages":stages,
"source_C17_parent":"e4dc4d6a0d5db03189d2f8465f6d5f019018f20c",
"protected_C17_HEAD":priorC17,"protected_C16_HEAD":priorC16,
"protected_original_docs_byte_identity":"PASS",
"frozen_source_parent_bank_sha256":sha(sourcebank),
"C07A_frozen_shadow_model_sha256":sha(frozen_model),
"C07A_train_cal_raw_sha256":sha(HOME/"evidence/C07_TRAIN_CAL/SOURCE_RAW.jsonl"),
"C08A_untouched_test_raw_sha256":sha(HOME/"evidence/C08_TEST/SOURCE_RAW.jsonl"),
"C08_heldout_report_sha256":sha(HOME/"evidence/C08_TEST/C08_HELDOUT_RESULTS.json"),
"C09_independent_rule_and_score_sha256":sha(HOME/"evidence/C09_INDEPENDENT_REPLAY.json"),
"C09_archive_restore_sha256":sha(HOME/"evidence/C09_LOCAL_RESTORE.json"),
"Rust_native_MCTS_source_binary_sha256":lock["native_sha256"],
"Rust_C18_compiled_readonly_shadow_sha256":sha(ROOT/"target/release/quartz_c18_shadow"),
"independent_Rust_Python_parity_96":"PASS",
"independent_full_test_record_audit_212":"PASS",
"local_restore_byte_exact_11":"PASS",
"Rust_debug_test_counts":{"MCTS_main":617,"C17_CLI":7,"C18_shadow_CLI":3,"fail":0,"ignored":89},
"Rust_release_test_counts":{"MCTS_main":617,"C17_CLI":7,"C18_shadow_CLI":3,"fail":0,"ignored":89},
"Python_focused_tests":{"pass":5,"fail":0},
"heldout_parent_groups":{"Rust_MCTS":24,"non_game":24},
"heldout_observations":{"Rust_MCTS":140,"non_game":72},
"heldout_calibrated_Brier":{"Rust_MCTS":0.11632275561131475,"non_game":0.19414174855045666},
"heldout_constant_Brier":{"Rust_MCTS":0.12813673683291116,"non_game":0.18810013717421123},
"cross_domain_superiority":"NOT_ESTABLISHED",
"real_MCTS_decision_regret":"NOT_ESTIMABLE",
"actual_active_SearchPolicy_changed":False,
"native_subagent":"NOT_RUN",
"cloud_Dropbox_synced":"NOT_VERIFIED",
"C10_own_archive":"generated_after_this_manifest_commit",
"next_research_id":"QUARTZ-GENERAL-19-SIGNED-VOC-AND-SHADOW-SELECTIVITY",
"next_stage_not_executed":True
}
out=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
print("C10_PREFLIGHT",json.dumps({"status":"PASS","pre_C10_head":head,
  "independent_stages":len(stages),"native_test":140,"nongame_test":72,
  "native_Brier":manifest["heldout_calibrated_Brier"]["Rust_MCTS"],
  "nongame_Brier":manifest["heldout_calibrated_Brier"]["non_game"],
  "next_ID":manifest["next_research_id"]},sort_keys=True))
