#!/usr/bin/env python3
"""C10: verify independently persisted C16 R1-R10, C1-C9 stages, evidence and remote SHA."""
import hashlib,json,re,subprocess,tarfile
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BRANCH="research/quartz-cog06-nonterminal-fork-wdl-20261121"
BACKUPS=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups")
def sha(path):
 with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def cmd(*args):
 return subprocess.check_output(list(args),text=True).strip()
head=cmd("git","-C",str(ROOT),"rev-parse","HEAD")
remote_rows=cmd("git","-C",str(ROOT),"ls-remote","--heads","origin",BRANCH)
remote_head=remote_rows.split()[0]
assert head==remote_head,("Git remote not at C09",head,remote_head)
assert cmd("git","-C",str(ROOT),"branch","--show-current")==BRANCH
stages={}
files=sorted(BACKUPS.glob("QUARTZ_COG06_NONTERMINAL16_*.tar.gz"))
namepattern=re.compile(r"QUARTZ_COG06_NONTERMINAL16_(R\d\d|C\d\d[A-Z]?)_([a-f0-9]{12})\.tar\.gz")
for archive in files:
    m=namepattern.fullmatch(archive.name)
    assert m,archive
    stage,abbrev=m.groups()
    assert stage not in stages,("duplicate stage",stage)
    commit=cmd("git","-C",str(ROOT),"rev-parse",abbrev)
    assert commit[:12]==abbrev
    assert subprocess.call(["git","-C",str(ROOT),"merge-base","--is-ancestor",commit,head])==0
    digest=sha(archive)
    assert archive.stat().st_size>1000
    stages[stage]={"commit":commit,"local_git_archive":str(archive),
                   "archive_sha256":digest,"archive_bytes":archive.stat().st_size}
expected={f"R{i:02d}" for i in range(1,11)} | {f"C{i:02d}" for i in range(1,10)} | {"C08A","C08B"}
assert set(stages)==expected,{"missing":sorted(expected-set(stages)),
                              "extra":sorted(set(stages)-expected)}
rfile=HOME/"evidence/real_root16_blind/RAW_ROOT_195.json"
gfile=HOME/"evidence/real_games16/RAW_32_ACTUAL_COMPLETE_GAMES.json"
ledger=HOME/"evidence/real_games16/COMPLETED_GAMES.jsonl"
cohort=HOME/"COHORT_40_FROZEN.json"
model=ROOT/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15/evidence/model/MODEL_TRAINED.json"
aud=HOME/"evidence/C09_INDEPENDENT_REPLAY.json"
rest=HOME/"evidence/C09_INDEPENDENT_LOCAL_RESTORE.json"
audit=json.loads(aud.read_text())
assert audit["status"]=="PASS" and audit["n_real_Rust_root_requests"]==195
assert audit["n_native_verified_completed_games"]==32
assert audit["n_actual_native_game_moves"]==217
assert audit["n_reopening_or_override_game_events"]==0
assert all(r["cost_matched_10pct"] for r in audit["strata"].values())
assert json.loads(rest.read_text())["restoration"]=="PASS"
assert sha(cohort)=="8349441f247f720028f8534ac4abf656be56f8ac58778191de900722b4af1049"
assert sha(model)=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
assert sha(rfile)=="6ffcf1b110f7bcfc55964b203c58df46f830fe63d1797f13691d5593e7de7786"
assert sha(ledger)=="a118b4848d87b441c0a9d46e87da384ebd485c05d17559714d77f91b57a83d84"
assert sha(gfile)=="f710f1fea59d2c44aa08292b1b8a839cbac7ee6617a36b8033ba54537ab21b84"
result={
  "schema":"quartz_cog06_C10_final_release_manifest_v1",
  "source_before_C10_git_commit":head,
  "remote_before_C10_git_commit":remote_head,
  "branch":BRANCH,"original_C15_base_commit":"a85bc0ded6bf67f81256e93645dc7147838c6f88",
  "completed_theory_stages":10,
  "completed_coding_stages_before_C10":9,
  "extra_preoutcome_stage_snapshots":["C08A","C08B"],
  "archived_stages_before_C10":len(stages),
  "pre_C10_stage_git_and_archive_identities":stages,
  "raw_cohort_SHA256":sha(cohort),"frozen_model_SHA256":sha(model),
  "actual_native_root_195_SHA256":sha(rfile),
  "actual_games_32_SHA256":sha(gfile),
  "append_only_games_32_SHA256":sha(ledger),
  "independent_C09_readback_SHA256":sha(aud),
  "local_C08_archive_restore_verified_9_files":True,
  "actual_native_Rust_binary_SHA256":"1168f2979936a283a9e7e613b869fd30ec9aba826e15ac228c6ee9d7e7e4c4d3",
  "tested_Rust_debug":{"passed":611,"failed":0,"ignored":89},
  "tested_Rust_release":{"passed":611,"failed":0,"ignored":89},
  "actual_native_MCTS_root_requests":195,
  "actual_real_complete_games":32,
  "actual_game_moves_independently_replayed":217,
  "constructed_source_active_fork_positives":23,
  "played_source_omitted_certificate_activations":0,
  "strict_effective_miss_count":0,
  "game_reopening_or_proof_override_event_count":0,
  "game_WDL":{"mask":[8,0,8],"proof_choice":[9,0,7]},
  "physical_CPU_gate_per_source_stratum":"PASS",
  "positive_new_tactical_mechanism_effect":"NOT_IDENTIFIED",
  "played_game_population_conditional_effect":"NOT_ESTIMABLE",
  "implied_elo":"NOT_ESTIMATED",
  "Dropbox_cloud_sync_ack":"NOT_VERIFIED",
  "final_C10_self_archive":"to_be_generated_after_this_manifest_is_committed",
  "next_research_id":"QUARTZ-COG-06-EFFECTIVE-MISS-COUNTERFACTUAL-17",
  "native_subagents":"NOT_RUN",
}
path=HOME/"C10_FINAL_RELEASE_MANIFEST.json"
path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print("C10_PREFLIGHT",json.dumps({
 "stages":len(stages),"pre_C10_commit":head,
 "model":result["frozen_model_SHA256"],"native_root_n":195,
 "actual_games":32,"played_moves":217,
 "C09_independent_rules":"PASS","local_archive_restore":"PASS",
 "next_id":result["next_research_id"]},sort_keys=True))
