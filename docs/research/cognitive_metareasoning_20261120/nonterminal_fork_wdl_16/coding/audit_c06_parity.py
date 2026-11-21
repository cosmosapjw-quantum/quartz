#!/usr/bin/env python3
"""C16 cross-language witness exhaustive vs Rust, all 40 frozen boards."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
source=Path(__file__).resolve()
repo=next(x for x in source.parents if (x/'.git').exists())
home=source.parent.parent
sys.path.insert(0,str(source.parent))
from freeze_sources16 import source_forks
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--binary",type=Path,required=True);args=ap.parse_args()
    bin=args.binary.resolve();assert bin.is_file()
    cohortfile=home/"COHORT_40_FROZEN.json"
    cohort=json.loads(cohortfile.read_text())
    rows=cohort["data_splits"]["constructed_fork_positive"]+cohort["data_splits"]["independent_MCTS_played_roots"]
    requests=[{"cmd":"research_gomoku15_fork_inspect","rule":"freestyle","history":row["history"]} for row in rows]
    wire="\n".join(json.dumps(x,separators=(",",":")) for x in requests+[{"cmd":"quit"}])+"\n"
    run=subprocess.run([str(bin),"--server"],input=wire,capture_output=True,text=True,check=True,timeout=90)
    responses=[json.loads(x) for x in run.stdout.splitlines() if x.strip()]
    if len(responses)!=len(rows):raise RuntimeError(f"responses={len(responses)} expected={len(rows)}")
    summary={"schema":"quartz_cog06_C06_native_Python_exhaustive_parity_v1",
             "status":"PASS","n_frozen_sources":len(rows),"n_constructed":24,"n_played_game_roots":16,
             "model_sha256":cohort["C15_policy_model_sha256"],
             "frozen_cohort_sha256":digest(cohortfile),
             "actual_compiled_native_binary_sha256":digest(bin),
             "n_equivalent_fork_actions":0,"n_distinct_win_endpoint_pairs":0,
             "n_actual_source_without_immediate_win":0,
             "independent_checker":"Python full 225-cell candidate and 225-cell winning endpoint search, not native Rust local-32 code",
             "sources":[]}
    for original,result in zip(rows,responses):
        if "error" in result:raise RuntimeError(f"Rust inspect error {original['id']}: {result['error']}")
        assert result["history"]==original["history"]
        assert result["legal_count"]==225-len(original["history"])
        expected=source_forks(original["history"])
        want={x["action"]:sorted(x["winning_endpoints"])[:2] for x in expected}
        got={int(x["action"]):sorted(map(int,x["endpoints"])) for x in result["certificates_all"]}
        assert len(got)==len(result["certificates_all"])
        assert want==got,("C06_MAP_MISMATCH",original["id"],list(want.items())[:3],list(got.items())[:3])
        assert result["no_external_oracle"] and result["rule"]=="freestyle"
        expected_precondition=not (original["source_type"]=="engineered_fork_positive_nonIID" and not want)
        # For non-stress positions empty source checker may indicate an immediate preexisting threat.
        if want:assert result["no_existing_immediate_win_either_side"]
        if original["source_type"]=="engineered_fork_positive_nonIID":
            assert result["no_existing_immediate_win_either_side"]
            assert len(want)>=18
        summary["n_equivalent_fork_actions"]+=len(got)
        summary["n_distinct_win_endpoint_pairs"]+=len(got)
        summary["n_actual_source_without_immediate_win"]+=int(result["no_existing_immediate_win_either_side"])
        summary["sources"].append({"id":original["id"],
            "n_proof_certificates":len(got),
            "certificates_source_type":original["source_type"],
            "legal_actions":result["legal_count"],
            "n_probed_candidates":result["inspected_action_count"]})
    out=home/"evidence/C06_INDEPENDENT_RULE_PARITY.json"
    out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("C06_INDEPENDENT_ORACLE_PARITY",json.dumps({k:v for k,v in summary.items() if k!="sources"},sort_keys=True),flush=True)
if __name__=="__main__":main()
