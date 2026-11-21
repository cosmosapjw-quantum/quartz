#!/usr/bin/env python3
"""Read-only provenance inventory of QUARTZ: Git, GitHub PR records and authority docs."""
from collections import Counter
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve()
REPO=next(p for p in HERE.parents if (p/".git").exists())
OUT=HERE.parents[1]/"evidence"
OUT.mkdir(exist_ok=True,parents=True)
def cmd(*args,check=True):
 p=subprocess.run(args,text=True,capture_output=True,check=check)
 return p.stdout.strip()
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
refs=[]
for line in cmd("git","-C",str(REPO),"for-each-ref","--format=%(refname:short)\t%(objectname)\t%(committerdate:short)\t%(subject)","refs/remotes/origin").splitlines():
 name,oid,date,subject=line.split("\t",3)
 refs.append({"name":name,"sha":oid,"date":date,"subject":subject})
pr_response=subprocess.run(["gh","api","--paginate","repos/cosmosapjw-quantum/quartz/pulls?state=all&per_page=100"],capture_output=True,text=True)
if pr_response.returncode==0:
    # gh --paginate prints concatenated arrays if several pages; handle a zero-page case too.
    try:
       prs=json.loads(pr_response.stdout)
       assert isinstance(prs,list)
    except (ValueError,AssertionError):
       raise RuntimeError("GitHub PR pagination output not a single list; don't claim complete PR count")
    prs_json=[{"number":p["number"],"title":p["title"],"state":p["state"],"head":p["head"]["ref"],"base":p["base"]["ref"],"merged_at":p["merged_at"]} for p in prs]
    pr_status="LIVE_GITHUB_API_COMPLETE_UP_TO_FIRST_PAGE" if len(prs)<100 else "FIRST_PAGE_ONLY"
else:
    prs_json=[]
    pr_status="GITHUB_QUERY_FAILED_NOT_ZERO"
anchors=["README.md","docs/THESIS.md","docs/CLAIM_LEDGER.md",
"docs/BQ_PLUS_PLUS_DESIGN.md","docs/RESEARCH_PLAN_PARTB.md",
"docs/RESEARCH_READINESS.md","docs/METACOGNITIVE_EXPERIMENTS.md",
"docs/LOCAL_EXPERIMENT_LAB.md","docs/idea_foundry/01_control_policy.md",
"docs/LEGACY_VS_BAYESIAN_QUARTZ.md",
"docs/legacy/mcts_study/v6.0/quantum_mcts_foundation.md"]
docs=[]
for relative in anchors:
 p=REPO/relative
 if not p.is_file():raise RuntimeError("authority input missing: "+relative)
 docs.append({"path":relative,"bytes":p.stat().st_size,"sha256":sha(p),
 "last_affecting_commit":cmd("git","-C",str(REPO),"log","-1","--format=%H","--",relative)})
initial=cmd("git","-C",str(REPO),"rev-list","--all","--max-parents=0").splitlines()
metadata=json.loads(cmd("gh","api","repos/cosmosapjw-quantum/quartz"))
cog06="41febde4561644d7287fd4d3411847e485f5f646"
developer=cmd("git","-C",str(REPO),"rev-parse","agent/local-experiment-foundry")
current_head=cmd("git","-C",str(REPO),"rev-parse","HEAD")
result={
"schema":"quartz_general_meta17_read_only_repo_provenance_v1",
"repo_name":"cosmosapjw-quantum/quartz",
"repo_visibility":metadata["visibility"],
"repo_created_utc":metadata["created_at"],
"repo_default_branch":metadata["default_branch"],
"PR_query_status":pr_status,
"GitHub_actual_PR_count":len(prs_json) if pr_status!="GITHUB_QUERY_FAILED_NOT_ZERO" else None,
"GitHub_PR_records":prs_json,
"branch_names_containing_pr_not_equal_github_PR":True,
"original_git_root_commits":initial,
"first_commit_date":cmd("git","-C",str(REPO),"show","-s","--format=%ad","--date=short",initial[0]),
"all_git_commits_in_inventory":int(cmd("git","-C",str(REPO),"rev-list","--all","--count")),
"source_C16_parent_sha":cog06,
"original_developer_worktree_sha":developer,
"new_isolated_branch_source_SHA":current_head,
"refs_origin":refs,
"ref_name_group_counts":dict(Counter((r["name"].removeprefix("origin/").split("/")[0]) for r in refs)),
"source_documents":docs,
"proof_source_THESIS_id":next(x for x in docs if x["path"]=="docs/THESIS.md"),
"protected_prior_main_not_modified":True,
"status_distinguishes_text_claims_and_actual_merged_PR":True}
out=OUT/"REPOSITORY_PROVENANCE.json"
out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
print(json.dumps({
"status":"SURVEY_COMPLETE",
"root_commits":initial,"commit_count":result["all_git_commits_in_inventory"],
"branch_ref_count":len(refs),"branch_group_counts":result["ref_name_group_counts"],
"actual_PR_count":result["GitHub_actual_PR_count"],
"PR_query_status":pr_status,"authority_docs":len(docs),"evidence_sha256":sha(out)},sort_keys=True))
