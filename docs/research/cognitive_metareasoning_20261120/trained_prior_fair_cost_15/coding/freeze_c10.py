#!/usr/bin/env python3
"""C10 reproducible predecessor/source/outcome/artifact manifest, before final self-referential commit."""
import hashlib,json,re,subprocess,tarfile
from pathlib import Path
repo=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
home=Path(__file__).resolve().parent.parent
def sh(cmd):return subprocess.check_output(cmd,text=True).strip()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
head=sh(["git","-C",str(repo),"rev-parse","HEAD"])
branch=sh(["git","-C",str(repo),"branch","--show-current"])
assert branch=="research/quartz-cog05-trained-prior-fair-cpu-20261121"
remote=sh(["git","-C",str(repo),"ls-remote","--heads","origin",branch]).split()[0]
assert head==remote,"C09 parent not published"
backupdir=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups")
archives=list(backupdir.glob("QUARTZ_COG05_FAIRCOST15_*.tar.gz"))
stagehash={}
for f in archives:
 match=re.fullmatch(r"QUARTZ_COG05_FAIRCOST15_(R\d\d|C\d\d[A-Z]?)_([0-9a-f]{12})\.tar\.gz",f.name)
 assert match, f.name
 stage,commit12=match.groups()
 assert stage not in stagehash,stage
 sha_committed=sh(["git","-C",str(repo),"rev-parse",commit12])
 assert sha_committed.startswith(commit12)
 assert sha_committed in sh(["git","-C",str(repo),"log","--format=%H","-25"]).splitlines()
 with tarfile.open(f,"r:gz") as tar:
    assert tar.getmembers() and all(not member.name.startswith("/") for member in tar.getmembers())
 stagehash[stage]={"git_commit":sha_committed,"archive":str(f),"archive_sha256":sha(f)}
expected={f"R{i:02d}" for i in range(1,11)}
expected|={f"C{i:02d}" for i in range(1,10)}
expected|={"C07A","C08A","C08B"}
assert set(stagehash)==expected,(len(stagehash),sorted(set(stagehash)^expected))
frozen=home/"COHORT_224_FROZEN.json"
model=home/"evidence/model/MODEL_TRAINED.json"
train=home/"evidence/teacher_train_validation/TEACHER_TRAIN_VALIDATION.json"
test=home/"evidence/teacher_heldout/TEACHER_HELDOUT.json"
raw=home/"evidence/real_192_blind_20261121/RAW_192_RUST.json"
audit=home/"evidence/real_192_blind_20261121/C09_INDEPENDENT_READBACK.json"
c9=json.loads(audit.read_text())
assert c9["status"]=="PASS" and c9["n_actual_Rust_executions"]==192
assert c9["n_real_pretrained_Rapfi_test_labels"]==64
model_sha=sha(model)
assert model_sha=="5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83"
readback={"schema":"quartz_cog05_C10_release_manifest_v1",
           "source_before_C10_git_head":head,"source_before_C10_remote_head":remote,
           "branch":branch,"origin_repo":"cosmosapjw-quantum/quartz",
           "frozen_source_cohort_sha256":sha(frozen),
           "model_policy_only_sha256":model_sha,
           "train_validation_Rapfi_label_sha256":sha(train),
           "test_Rapfi_label_sha256":sha(test),
           "blind_192_real_Rust_SHA256":sha(raw),
           "independent_C09_audit_sha256":sha(audit),
           "science_gate":"trained prior finite-proxy benefit; no observed candidate reopening benefit (0/64 activations), no WDL/Elo",
           "native_Rust_binary_sha256":"1bf102650e28c9f0bbe9515f2f6e7d1d0c00a15f2646f78498ea79a9710d8527",
           "historical_local_archives_sha256":stagehash,
           "theory_stages_archived_and_remotely_committed":10,
           "coding_stages_before_final_archived":9,
           "additional_preunblind_backups":3,
           "archive_count_before_C10":len(stagehash),
           "C10_archive_excluded_to_avoid_self_reference":True,
           "publisher_required_for_final_C10":"scripts/cog05_commit_backup_publish.sh C10",
           "Dropbox_cloud_sync_acknowledged":False,
           "next_stage_id":"QUARTZ-COG-05-NONTERMINAL-MISS-AND-GAME-WDL-16"}
out=home/"C10_FINAL_RELEASE_MANIFEST.json"
out.write_text(json.dumps(readback,sort_keys=True,indent=2)+"\n")
print("C10_FINAL_PREFLIGHT",json.dumps({"head":head,"branch":branch,
    "archived_stages":len(stagehash),"file":str(out),"model_sha":model_sha},sort_keys=True))
