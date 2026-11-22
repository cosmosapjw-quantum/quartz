#!/usr/bin/env python3
"""Independent temporary restore of C08 source and scientific data from actual tar."""
import hashlib,json,tarfile,tempfile
from pathlib import Path
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARCHIVE=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta18_backups/QUARTZ_GENERAL_C18_C08_d0de51213948.tar.gz")
EXPECTED="5247d36e90473660e32a52a302043ea1caa99dd56d7a8d24f54a7bfa20ef61fe"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ARCHIVE)==EXPECTED
sources=[
"docs/THESIS.md",
"docs/research/quartz_general_calibration_20261122/C01_SOURCE_LOCK.json",
"docs/research/quartz_general_calibration_20261122/SOURCE_GROUPS_FROZEN.json",
"docs/research/quartz_general_calibration_20261122/SHADOW_MODEL_FROZEN.json",
"docs/research/quartz_general_calibration_20261122/evidence/C07_TRAIN_CAL/SOURCE_RAW.jsonl",
"docs/research/quartz_general_calibration_20261122/evidence/C08_TEST/SOURCE_RAW.jsonl",
"docs/research/quartz_general_calibration_20261122/evidence/C08_TEST/C08_HELDOUT_RESULTS.json",
"quartz/experiments/c18_observation.py",
"src/bin/quartz_c18_shadow.rs",
"docs/research/quartz_general_calibration_20261122/coding/collect_observations.py",
"docs/research/quartz_general_calibration_20261122/coding/score_c08_test.py"]
with tempfile.TemporaryDirectory(prefix="quartz_c18_C08_restore_") as td:
 root=Path(td)
 with tarfile.open(ARCHIVE,"r:gz") as tar:
  members=tar.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  tar.extractall(root,filter="data")
 result=[]
 for rel in sources:
  a=ROOT/rel;b=root/rel
  assert a.is_file() and b.is_file()
  assert sha(a)==sha(b),rel
  result.append({"source":rel,"sha256":sha(a),"bytes":a.stat().st_size})
out={"schema":"quartz_general_c18_C09_local_restore_v1","status":"PASS",
     "local_archive":str(ARCHIVE),"archive_sha256":EXPECTED,
     "n_byte_identical_restored_files":len(result),
     "sources":result,
     "Dropbox_cloud_sync_ack":"NOT_VERIFIED"}
p=HOME/"evidence/C09_LOCAL_RESTORE.json"
p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("C09_LOCAL_RESTORED",json.dumps({"status":"PASS","n":len(result),"sha":EXPECTED},sort_keys=True))
