#!/usr/bin/env python3
"""C09 actually restore published C08 Git archive into a fresh directory; SHA compare."""
from pathlib import Path
import hashlib,json,tarfile,tempfile
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARC=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta22_backups/QUARTZ_GENERAL_C22_C08_a82c601965b3.tar.gz")
EXPECTED="11b219f4b5910956a160691d4a8d925ab9ce43d66fdfafe60b98880c558df79f"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ARC)==EXPECTED
paths=[
"docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs",
"docs/research/quartz_general_active_gate_20261122/C01_SOURCE_LOCK.json",
"docs/research/quartz_general_active_gate_20261122/SOURCE_WORLDS_C22_FROZEN.json",
"docs/research/quartz_general_active_gate_20261122/FROZEN_C22_RESOURCE_PRICE.json",
"docs/research/quartz_general_active_gate_20261122/evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_active_gate_20261122/evidence/C07_TRAIN_CAL/NATIVE_TRAIN_ACTIVE_PILOT.jsonl",
"docs/research/quartz_general_active_gate_20261122/evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl",
"docs/research/quartz_general_active_gate_20261122/evidence/C08_TEST/C22_ACTIVE_ALL_POLICIES.jsonl",
"docs/research/quartz_general_active_gate_20261122/evidence/C08_TEST/C08_ACTUAL_ACTIVE_RESULTS.json",
"docs/research/quartz_general_active_gate_20261122/coding/C08_scientific_results.md",
"src/bin/quartz_c22_active_gate.rs",
"quartz/experiments/c22_active_inputs.py",
"docs/research/quartz_general_active_gate_20261122/coding/collect_c22_sources.py",
"docs/research/quartz_general_active_gate_20261122/coding/freeze_c07_costs.py",
"docs/research/quartz_general_active_gate_20261122/coding/score_c08_active.py",
"docs/research/quartz_general_active_gate_20261122/coding/C07_frozen_protocol.md",
]
with tempfile.TemporaryDirectory(prefix="quartz_c22_C08_restore_") as t:
 root=Path(t)
 with tarfile.open(ARC,"r:gz") as tar:
  members=tar.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  tar.extractall(root,filter="data")
 records=[]
 for rel in paths:
  a=ROOT/rel;b=root/rel
  assert a.is_file() and b.is_file(),rel
  assert sha(a)==sha(b),rel
  records.append({"path":rel,"sha256":sha(a),"size_bytes":a.stat().st_size})
receipt={"schema":"quartz_C22_C09_fresh_local_Git_tar_restoration_v1",
 "status":"PASS","source_archive":str(ARC),"archive_SHA256":EXPECTED,
 "n_key_original_science_and_code_files_identical":len(records),
 "exact_byte_identity_records":records,
 "local_archive_actual_restoration":"YES",
 "Dropbox_cloud_sync_acknowledgement":"NOT_VERIFIED"}
p=HOME/"evidence/C09_LOCAL_RESTORE.json"
p.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print("C22_C09_RESTORE",json.dumps({"status":"PASS","n_files":len(records),
 "archive_sha256":EXPECTED},sort_keys=True))
