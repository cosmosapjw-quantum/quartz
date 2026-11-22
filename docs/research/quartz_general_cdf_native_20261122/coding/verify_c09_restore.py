#!/usr/bin/env python3
"""C09 exact local Git archive restore and science source byte identity."""
from pathlib import Path
import hashlib,tarfile,tempfile,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARC=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta21_backups/QUARTZ_GENERAL_C21_C08_2c3e73cf16bc.tar.gz")
EXPECTED="3c08c400866c32b99e328bfd5bf24e129908585255a46a9288a7023753b4992d"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ARC)==EXPECTED
paths=[
"docs/THESIS.md",
"docs/CLAIM_LEDGER.md",
"docs/research/quartz_general_cdf_native_20261122/C01_SOURCE_LOCK.json",
"docs/research/quartz_general_cdf_native_20261122/SOURCE_WORLDS_C21_FROZEN.json",
"docs/research/quartz_general_cdf_native_20261122/FROZEN_C21_RESOURCE_PRICE.json",
"docs/research/quartz_general_cdf_native_20261122/evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_cdf_native_20261122/evidence/C07_TRAIN_CAL/NATIVE_TRAIN_ACTUAL.jsonl",
"docs/research/quartz_general_cdf_native_20261122/evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_cdf_native_20261122/evidence/C08_TEST/NATIVE_TEST_SHADOW_RECORDS.jsonl",
"docs/research/quartz_general_cdf_native_20261122/evidence/C08_TEST/C08_HELDOUT_NATIVE_EVSI_RESULTS.json",
"src/bin/quartz_c21_fast_evsi.rs",
"quartz/experiments/c21_native_session.py",
"docs/research/quartz_general_cdf_native_20261122/coding/collect_c21_sources.py",
"docs/research/quartz_general_cdf_native_20261122/coding/score_c08_native.py",
"docs/research/quartz_general_cdf_native_20261122/coding/freeze_c07_costs.py",
"docs/research/quartz_general_cdf_native_20261122/coding/C07_protocol_frozen.md"]
with tempfile.TemporaryDirectory(prefix="quartz_c21_C08_restore_") as folder:
 rest=Path(folder)
 with tarfile.open(ARC,"r:gz") as tar:
  members=tar.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  tar.extractall(rest,filter="data")
 matches=[]
 for name in paths:
  a=ROOT/name;b=rest/name
  assert a.is_file() and b.is_file(),name
  assert sha(a)==sha(b),name
  matches.append({"path":name,"SHA256":sha(a),"bytes":a.stat().st_size})
result={"schema":"quartz_C21_C09_local_archive_restore_v1","status":"PASS",
"archive":str(ARC),"archive_sha256":EXPECTED,
"n_byte_identical_files":len(matches),"files":matches,
"local_archive_restoration":"VERIFIED","Dropbox_cloud_sync_ACK":"NOT_VERIFIED"}
out=HOME/"evidence/C09_LOCAL_RESTORE.json"
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print("C21_C09_RESTORE",json.dumps({"status":"PASS","n_files":len(matches),"archive_sha256":EXPECTED},sort_keys=True))
