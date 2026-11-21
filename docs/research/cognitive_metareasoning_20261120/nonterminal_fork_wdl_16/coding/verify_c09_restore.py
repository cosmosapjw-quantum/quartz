#!/usr/bin/env python3
"""C09 temporary independent local Git archive restore with exact byte comparison."""
from pathlib import Path
import hashlib,json,tarfile,tempfile
repo=next(x for x in Path(__file__).resolve().parents if (x/'.git').exists())
home=Path(__file__).resolve().parent.parent
archive=Path("/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/QUARTZ_COG06_NONTERMINAL16_C08_3db40e654b49.tar.gz")
expected="520451b32456d4f196529fd07b3883e68485eac4435dad26407db16f84d4f234"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert archive.exists()
assert sha(archive)==expected
paths=[
"COHORT_40_FROZEN.json",
"C01_SOURCE_LOCK.json",
"coding/C08_FINAL_SCIENCE.md",
"evidence/real_root16_blind/RAW_ROOT_195.json",
"evidence/real_root16_blind/C08_ROOT_RULE_SCORE.json",
"evidence/real_root16_blind/C08_POSTTEST_REACHABILITY.json",
"evidence/real_games16/COMPLETED_GAMES.jsonl",
"evidence/real_games16/RAW_32_ACTUAL_COMPLETE_GAMES.json",
"evidence/real_games16/C08_WDL_SUMMARY.json"
]
with tempfile.TemporaryDirectory(prefix="quartz_cog06_verified_restore_") as temp:
 t=Path(temp)
 with tarfile.open(archive,"r:gz") as arc:
   files=arc.getmembers()
   assert all(not f.name.startswith("/") and ".." not in Path(f.name).parts for f in files)
   arc.extractall(t,filter="data")
 match=[]
 for relative in paths:
   a=home/relative
   b=t/a.relative_to(repo)
   assert a.is_file() and b.is_file(),relative
   assert sha(a)==sha(b),relative
   match.append({"relative":relative,"sha256":sha(a),"byte_count":a.stat().st_size})
result={"schema":"quartz_cog06_C09_C08_archive_restore_v1",
"archive":str(archive),"archive_SHA256":expected,
"restoration":"PASS","exact_original_vs_independent_restored_content":match,
"n_checked_files":len(match),
"source_RUST_and_model_unchanged_by_restoration":True,
"remote_Dropbox_cloud_sync_ack":"NOT_VERIFIED",
"local_archive_and_restore":"VERIFIED"}
p=home/"evidence/C09_INDEPENDENT_LOCAL_RESTORE.json"
p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print("C09_RESTORE_VERIFIED",json.dumps({"files":len(match),"archive_SHA256":expected,"status":"PASS"},sort_keys=True))
