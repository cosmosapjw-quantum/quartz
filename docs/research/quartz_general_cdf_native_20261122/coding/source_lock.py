#!/usr/bin/env python3
"""C01 canonical source lock: C20 exact byte authority and C21 isolated identity."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
PARENT="ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4"
FILES=[
"docs/THESIS.md","docs/CLAIM_LEDGER.md","src/game.rs",
"docs/research/quartz_general_predictive_voc_20261122/NEXT_STAGE.json",
"docs/research/quartz_general_predictive_voc_20261122/RESULTS_AND_NEXT_KO.md",
"docs/research/quartz_general_predictive_voc_20261122/evidence/C10_FAST_BETA_MAX_CDF_WITNESS.json",
"docs/research/quartz_general_predictive_voc_20261122/coding/c10_fast_beta_cdf_prototype.py",
"quartz/experiments/c20_predictive_value.py",
"src/bin/quartz_c20_shadow.rs"]
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(ROOT/FILES[5])=="c60250de1fe8c89ee035f22eabfd5c95fb15f4fab4ba2825fdc10a4a739e5c68"
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
assert subprocess.run(["git","merge-base","--is-ancestor",PARENT,head],cwd=ROOT).returncode==0
assert subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="research/quartz-general-c21-fast-cdf-20261122"
for f in FILES[:3]:
 assert subprocess.check_output(["git","show",PARENT+":"+f],cwd=ROOT)==(ROOT/f).read_bytes()
nextj=json.loads((ROOT/FILES[3]).read_text())
assert nextj["next_not_executed"] and nextj["next_research_id"]=="QUARTZ-GENERAL-21-CDF-EXACT-EVSI-NATIVE-RESOURCE-GATE"
record={"schema":"quartz_C21_C01_frozen_authority_v1",
 "C20_original_git_SHA":PARENT,"R10_completed_current_git_SHA":head,
 "source_files":[{"path":f,"sha256":sha(ROOT/f),"bytes":(ROOT/f).stat().st_size} for f in FILES],
 "original_THESIS_CLAIM_LEDGER_GameState_protected":True,
 "existing_developer_and_C16_C20_worktrees_protected":True,
 "historical_C20_TEST_not_new_C21_TEST":True,
 "C21_science":"independent arm Beta-binomial exact CDF and matched Rust CPU resource gate",
 "native_subagent":"NOT_RUN"}
out=HOME/"C01_SOURCE_LOCK.json"
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print("C21_C01_SOURCE_LOCK",json.dumps({"n_source_files":len(FILES),"sha256":sha(out),"git":head},sort_keys=True))
