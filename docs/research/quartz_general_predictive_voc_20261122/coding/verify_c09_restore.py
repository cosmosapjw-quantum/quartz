#!/usr/bin/env python3
"""C09 actual local Git archive restore, SHA256 byte identity and source coverage."""
from pathlib import Path
import tarfile,hashlib,json,tempfile
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
ARCHIVE=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta20_backups/QUARTZ_GENERAL_C20_C08_7048a2af49c4.tar.gz")
EXPECTED="e2c8041c978fc5a54cc284d835b9a5f49d2a549b64b866a381889162a69f9114"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ARCHIVE)==EXPECTED
paths=[
 "docs/THESIS.md",
 "docs/CLAIM_LEDGER.md",
 "docs/research/quartz_general_predictive_voc_20261122/C01_SOURCE_LOCK.json",
 "docs/research/quartz_general_predictive_voc_20261122/SOURCE_WORLDS_C20_FROZEN.json",
 "docs/research/quartz_general_predictive_voc_20261122/FROZEN_C20_RESOURCE_PRICE.json",
 "docs/research/quartz_general_predictive_voc_20261122/evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl",
 "docs/research/quartz_general_predictive_voc_20261122/evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl",
 "docs/research/quartz_general_predictive_voc_20261122/evidence/C08_TEST/C08_HELDOUT_EXACT_EVSI.json",
 "quartz/experiments/c20_predictive_value.py",
 "quartz/experiments/c20_task_adapter.py",
 "src/bin/quartz_c20_shadow.rs",
 "docs/research/quartz_general_predictive_voc_20261122/coding/freeze_c04_worlds.py",
 "docs/research/quartz_general_predictive_voc_20261122/coding/collect_c20_sources.py",
 "docs/research/quartz_general_predictive_voc_20261122/coding/score_c08_exact.py",
 "docs/research/quartz_general_predictive_voc_20261122/coding/freeze_c07_costs.py"]
with tempfile.TemporaryDirectory(prefix="quartz_C20_C08_restore_") as folder:
 rest=Path(folder)
 with tarfile.open(ARCHIVE,"r:gz") as tar:
  members=tar.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  tar.extractall(rest,filter="data")
 matches=[]
 for rel in paths:
  a=ROOT/rel;b=rest/rel
  assert a.is_file() and b.is_file(),rel
  assert sha(a)==sha(b),rel
  matches.append({"file":rel,"sha256":sha(a),"size_bytes":a.stat().st_size})
record={"schema":"quartz_C20_C09_fresh_temp_local_restore_v1",
 "status":"PASS","C08_archive":str(ARCHIVE),"C08_archive_SHA256":EXPECTED,
 "n_byte_identical_files":len(matches),"files":matches,
 "local_archive_restore":"VERIFIED","Dropbox_cloud_sync_ACK":"NOT_VERIFIED"}
out=HOME/"evidence/C09_LOCAL_RESTORE.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C20_C09_LOCAL_RESTORED",json.dumps({"status":"PASS","files":len(matches),"archive_sha256":EXPECTED},sort_keys=True))
