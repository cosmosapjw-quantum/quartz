#!/usr/bin/env python3
"""C9 independently verify local C8 Git tar.gz backup can restore exact source/evidence."""
import hashlib,json,tarfile,tempfile
from pathlib import Path
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HERE=Path(__file__).resolve().parent.parent
ARCHIVE=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta17_backups/QUARTZ_META17_C08_b8b0959d7209.tar.gz")
EXPECTED="80cf6891ca2090db8c70281b33a9730c9038bb003856191904b1b66801a7ad12"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ARCHIVE)==EXPECTED
files=[
"configs/metalevel_reference_cases.v1.json",
"quartz/experiments/metalevel_decision.py",
"src/mcts/policy/finite_metalevel.rs",
"src/bin/quartz_metalevel_reference.rs",
"docs/research/quartz_general_metareasoning_20261122/evidence/REPOSITORY_PROVENANCE.json",
"docs/research/quartz_general_metareasoning_20261122/evidence/C06_CROSS_LANGUAGE.json",
"docs/research/quartz_general_metareasoning_20261122/evidence/C08/EXECUTION_CONTRACT.json",
"docs/research/quartz_general_metareasoning_20261122/evidence/C08/C08_RESULTS.json",
"docs/research/quartz_general_metareasoning_20261122/evidence/C08/PAIRED_MC_10240.jsonl"
]
with tempfile.TemporaryDirectory(prefix="quartz_meta17_C08_restore_") as temp:
 t=Path(temp)
 with tarfile.open(ARCHIVE,"r:gz") as archive:
  members=archive.getmembers()
  assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
  archive.extractall(t,filter="data")
 verified=[]
 for rel in files:
  original=ROOT/rel
  restored=t/rel
  assert original.is_file() and restored.is_file(),rel
  assert sha(original)==sha(restored),rel
  verified.append({"relative_path":rel,"sha256":sha(original),"bytes":original.stat().st_size})
record={"schema":"quartz_meta17_C09_independent_local_restore_v1",
"status":"PASS",
"source_C08_local_tar_archive":str(ARCHIVE),
"source_C08_tar_archive_sha256":EXPECTED,
"independently_byte_identical_files_count":len(verified),
"files":verified,
"Dropbox_cloud_sync_ack":"NOT_VERIFIED",
"restore_only_local_directory":"TRUE"}
out=HERE/"evidence/C09_LOCAL_ARCHIVE_RESTORE.json"
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C09_RESTORE",json.dumps({"status":"PASS","n_file_byte_identity":len(verified),"archive_sha":EXPECTED},sort_keys=True))
