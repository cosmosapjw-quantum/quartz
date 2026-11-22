#!/usr/bin/env python3
"""C19 C9 independent local archive extraction and exact SHA256 byte comparisons."""
from pathlib import Path
import tarfile,hashlib,tempfile,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
TAR=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta19_backups/QUARTZ_GENERAL_C19_C08_cc54de56c629.tar.gz")
TAR_SHA="125f0c028c7ff458b3911dffd1784aca6836c85ae730447bafd238c365ccc25d"
def sha(path):
 with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(TAR)==TAR_SHA
paths=[
"docs/THESIS.md",
"docs/CLAIM_LEDGER.md",
"docs/research/quartz_general_signed_voc_20261122/C01_SOURCE_LOCK.json",
"docs/research/quartz_general_signed_voc_20261122/SOURCE_WORLDS_FROZEN.json",
"docs/research/quartz_general_signed_voc_20261122/SHADOW_SIGNED_MODEL_FROZEN.json",
"docs/research/quartz_general_signed_voc_20261122/evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_signed_voc_20261122/evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_signed_voc_20261122/evidence/C08_TEST/C08_HELDOUT_SIGNED_RESULTS.json",
"quartz/experiments/c19_signed_value.py",
"src/bin/quartz_c19_shadow.rs",
"docs/research/quartz_general_signed_voc_20261122/coding/collect_signed_sources.py",
"docs/research/quartz_general_signed_voc_20261122/coding/score_c08_signed.py",
"docs/research/quartz_general_signed_voc_20261122/coding/freeze_train_cal.py"]
with tempfile.TemporaryDirectory(prefix="quartz_c19_restore_") as td:
 rest=Path(td)
 with tarfile.open(TAR,"r:gz") as archive:
  members=archive.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  archive.extractall(rest,filter="data")
 match=[]
 for rel in paths:
  a=ROOT/rel;b=rest/rel
  assert a.is_file() and b.is_file(),rel
  assert sha(a)==sha(b),rel
  match.append({"file":rel,"sha256":sha(a),"size":a.stat().st_size})
record={"schema":"quartz_general_c19_C09_local_restore","status":"PASS",
 "source_C08_git_archive":str(TAR),"source_C08_archive_sha256":TAR_SHA,
 "n_files_byte_identical":len(match),"files":match,
 "local_archive_restored":"YES","Dropbox_cloud_sync_ack":"NOT_VERIFIED"}
out=HOME/"evidence/C09_LOCAL_RESTORE.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C19_C09_RESTORE",json.dumps({"status":"PASS","n_files":len(match),"archive_sha256":TAR_SHA},sort_keys=True))
