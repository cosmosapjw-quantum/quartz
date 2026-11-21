#!/usr/bin/env python3
"""C09 local Git-archive backup restore and Git readback of the C08 blinded proof."""
import hashlib,json,tarfile,tempfile
from pathlib import Path
repo=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
root=repo/"docs/research/cognitive_metareasoning_20261120/trained_prior_fair_cost_15"
backupdir=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups")
archives=sorted(backupdir.glob("QUARTZ_COG05_FAIRCOST15_*.tar.gz"))
assert len(archives)>=21,len(archives)
latest=backupdir/"QUARTZ_COG05_FAIRCOST15_C08_4f98b07bbea9.tar.gz"
assert latest.is_file()
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert digest(latest)=="37d6ae0fb63ab23fda17ad60501f2d1676c5e489c4d4220d9d3f109ed2e6df44"
checks=["COHORT_224_FROZEN.json",
        "evidence/model/MODEL_TRAINED.json",
        "evidence/teacher_train_validation/TEACHER_TRAIN_VALIDATION.json",
        "evidence/teacher_heldout/TEACHER_HELDOUT.json",
        "evidence/real_192_blind_20261121/RAW_192_RUST.json",
        "evidence/real_192_blind_20261121/C08_SCORED_RESULTS.json",
        "coding/C08_REAL_RESULTS.md"]
with tempfile.TemporaryDirectory(prefix="quartz_cog05_archive_restore_") as tmp:
    tmp=Path(tmp)
    with tarfile.open(latest,"r:gz") as f:
        names=f.getnames()
        assert all(not x.startswith("/") and ".." not in Path(x).parts for x in names)
        f.extractall(tmp,filter="data")
    restored=[]
    for relative in checks:
        p=root/relative
        q=tmp/p.relative_to(repo)
        assert p.is_file() and q.is_file(),relative
        assert digest(p)==digest(q),relative
        restored.append({"file":relative,"sha256":digest(p)})
report={"schema":"quartz_cog05_C09_archive_independent_restore_v1",
"verified_stage_local_archives_at_least":len(archives),
"restored_C08_archive_file":str(latest),
"restored_C08_archive_sha256":digest(latest),
"exact_byte_readback":restored,
"restore_status":"PASS",
"external_Rapfi_binary_and_weights_not_in_archive":True,
"Dropbox_sync_acknowledged":False}
p=root/"evidence/real_192_blind_20261121/C09_ARCHIVE_RESTORE.json"
p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C09_RESTORE",json.dumps({"status":"PASS","archive_n":len(archives),"exact_restored_files":len(restored),"archive_sha":digest(latest)},sort_keys=True))
