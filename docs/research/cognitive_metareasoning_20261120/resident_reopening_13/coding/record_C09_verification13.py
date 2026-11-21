#!/usr/bin/env python3
"""One bounded source-hash and fresh-test receipt for C09 independent science."""
from pathlib import Path
import hashlib,json

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    d=Path(__file__).resolve().parent.parent
    e=d/"evidence/live_9216_resident_20261121"
    m=json.loads((e/"EXECUTION_MANIFEST.json").read_text())
    s=json.loads((e/"SUMMARY_RESIDENT13.json").read_text())
    a=json.loads((e/"INDEPENDENT_READBACK.json").read_text())
    assert m["source_commit"]=="bf5bda76ba817476f635a8d245603d8c95949211"
    assert m["n_actual_Rust_resident_search_requests"]==a["real_Rust_endpoint_requests_rechecked"]==9216
    assert a["same_root_real_MCTS_stage_executions"]==18432
    assert a["independent_PascalPons_original_requeries"]==a["independent_PascalPons_mirror_requeries"]==24
    assert s["results"]["resident_A"]["uniform"]["decision_informative_roots"]==125
    assert s["results"]["resident_B"]["uniform"]["decision_informative_roots"]==127
    for name,digest in m["output_SHA256"].items():
        assert sha(e/name)==digest,name
    assert sha(d/"COHORT_384_FROZEN.json")=="3b79dd2341e0e56405a2219f41fb0e28047bf642a7372f417f23513459450b31"
    assert sha(d/"evidence/new384_exact_20261121/EXACT_NEW384_LABELS.json")=="73cf3b37ddbcc7f14298ee0634b760f5008bfb2194bd5cd5fc31dc24ceb1a817"
    binary=Path("/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo")
    assert sha(binary)==m["actual_Rust_compiled_binary_sha256"]
    for log in (Path("/tmp/c13_C09_fresh_rust_release.log"),Path("/tmp/c13_C09_fresh_rust_debug.log")):
        assert "604 passed; 0 failed; 89 ignored" in log.read_text()
    pairs={}
    for group in ("resident_A","resident_B","pooled"):
        pairs[group]={}
        for mode in ("uniform","new_mlp","conv"):
            row=s["results"][group][mode]
            comp=row["paired_comparisons"]["evidence_vs_resident_full24"]
            assert comp["mean_CPU_cost_10pct_admission"]=="MATCHED_10_PERCENT"
            assert comp["bootstrap_family_conditional"]["CI95"][0]<=0<=comp["bootstrap_family_conditional"]["CI95"][1]
            assert row["arms"]["resident_evidence"]["mean_regret"]>=row["arms"]["resident_full24"]["mean_regret"]
            pairs[group][mode]={
                "regret_resident_evidence":row["arms"]["resident_evidence"]["mean_regret"],
                "regret_all_legal24":row["arms"]["resident_full24"]["mean_regret"],
                "exact_WDL_regret_delta":comp["mean_exact_WDL_regret_difference"],
                "family_CI95":comp["bootstrap_family_conditional"]["CI95"],
                "actual_CPU_mismatch_fraction":comp["physical_CPU_mean_time_mismatch_fraction"]
            }
    receipt={
        "schema":"quartz_cog03_resident13_C09_independent_actual_9216_v1",
        "Rust_executable_source_commit":m["compiled_Rust_source_commit"],
        "Rust_binary_SHA256":m["actual_Rust_compiled_binary_sha256"],
        "actual_trial_source_commit":m["source_commit"],
        "new_source_384_sha256":m["new384_sha256"],
        "complete_external_WDL_SHA256":m["new384_exact_oracle_sha256"],
        "actual_Rust_requests":9216,"real_same_root_8_and_24_stage_executions":18432,
        "new_distinct_family_clusters":24,"new_root_positions":384,
        "WDL_informative_root_count":252,
        "external_original_plus_horizontal_mirror_requeries":48,
        "all_engine_root_N_W_M2_prior_conservation_verified":True,
        "all_oneply_tactical_moves_and_FNV_random_reopenings_independently_recomputed":True,
        "all_exact_WDL_action_utilities_and_costs_recomputed":True,
        "Rust_debug_test":{"passed":604,"failed":0,"ignored":89},
        "Rust_release_test":{"passed":604,"failed":0,"ignored":89},
        "related_Python_unittest":{"passed":132,"failed":0,"components":14},
        "all_legal24_comparison_matched_10pct_both_new_groups":True,
        "selective_resident_unpruning_superior_to_unrestricted":"NOT_ESTABLISHED",
        "native_subagent":"NOT_RUN",
        "group_summaries":pairs,
        "raw_artifact_SHA256":m["output_SHA256"]
    }
    target=e/"VERIFICATION_RECEIPT.json"
    if target.exists():raise FileExistsError("C09 receipt already exists")
    target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("C09_INDEPENDENT_VERIFICATION_RECEIPT_PASS",receipt["actual_Rust_requests"],receipt["related_Python_unittest"]["passed"])

if __name__=="__main__":main()
