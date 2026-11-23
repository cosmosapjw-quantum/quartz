#!/usr/bin/env python3
"""Independent C09 read-only audit of the pinned evidence, not a native rerun."""
import collections, gzip, hashlib, json, math, pathlib, statistics, subprocess, sys
from fractions import Fraction as F
from datetime import datetime, timezone
BASE=pathlib.Path("/home/cosmosapjw/quantum/research/quartz_c28_action_geometry_20261123_486e")
C="f791e96e651d1849df601dcd7b98c18bcc3fa082"
P="docs/research/quartz_c28_action_geometry_20261123"
SSOT="87c15dc828d6eea519c219fd8a412fa21dfc464d"
BIN=pathlib.Path("/home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry")
checks=collections.Counter()
def require(x,label):
    checks[label]+=1
    if not x: raise AssertionError(label)
def git(path,commit=C):
    return subprocess.check_output(["git","-C",str(BASE),"show",commit+":"+path])
def js(path): return json.loads(git(path))
def sha(b):return hashlib.sha256(b).hexdigest()
def close(a,b,label):
    require(math.isfinite(a) and abs(a-b)<=5e-12,label)
def raw(stage,kind,meta):
    compressed=git(P+"/evidence/"+stage+"/"+kind+".jsonl.gz")
    data=gzip.decompress(compressed)
    rows=[json.loads(l) for l in data.splitlines()]
    expected=meta["raw_artifacts"][kind]
    require(sha(compressed)==expected["gzip_sha256"],"compressed_hash")
    require(sha(data)==expected["uncompressed_sha256"],"uncompressed_hash")
    require(len(rows)==expected["lines"],"raw_line_count")
    return rows
def aggregate(rows):
    d=statistics.fmean(x["direct_median_ns"] for x in rows)
    l=statistics.fmean(x["lifted_median_ns"] for x in rows)
    return {"roots_or_configurations":len(rows),"mean_of_within_root_direct_medians_ns":d,
      "mean_of_within_root_lifted_medians_ns":l,
      "paired_mean_direct_minus_lifted_ns":statistics.fmean(x["direct_minus_lifted_ns"] for x in rows),
      "ratio_of_paired_means_lifted_over_direct":l/d,
      "mean_of_paired_lifted_over_direct_ratios":statistics.fmean(x["lifted_over_direct_ratio"] for x in rows),
      "median_of_paired_lifted_over_direct_ratios":statistics.median(x["lifted_over_direct_ratio"] for x in rows)}
def choose(q):
    return None if not q or max(q.values())<=F(1,10**12) else min(i for i,s in q.items() if s>=max(q.values())-F(1,10**12))
report={"schema":"quartz-c28-c09-independent-raw-audit-v1","review_commit":C,"ssot":SSOT,
 "started_utc":datetime.now(timezone.utc).isoformat(),"argv":sys.argv,
 "python":sys.version,"native_rerun":False,"interpretation":"Raw evidence reaggregation and provenance audit; no new performance experiment."}
sources=[json.loads(l) for l in git("docs/research/quartz_general_game_metareasoning_20261123/coding/data/public_sources.jsonl").splitlines()]
model=js("docs/research/quartz_general_game_metareasoning_20261123/coding/data/model.json")
contract=js(P+"/coding/CONTRACT.json")
source_map={x["id"]:x for x in sources}
report["observed_current_binary_sha256"]=sha(BIN.read_bytes())
report["observed_pinned_native_source_sha256"]=sha(git("src/bin/quartz_c28_action_geometry.rs"))
report["current_source_matches_pinned"]=sha((BASE/"src/bin/quartz_c28_action_geometry.rs").read_bytes())==report["observed_pinned_native_source_sha256"]
require(report["current_source_matches_pinned"],"current_source_identity")
report["unchanged_dependencies_from_ssot"]={}
for path in ["src/game.rs","src/games/tictactoe.rs","Cargo.toml","Cargo.lock"]:
    same=git(path)==git(path,SSOT)
    require(same,"unchanged_substrate_dependency")
    report["unchanged_dependencies_from_ssot"][path]={"unchanged":same,"sha256":sha(git(path))}
report["c26_source_sha256"]=sha(git("src/bin/quartz_c26_game_metareasoning.rs",SSOT))
datasets={}
for stage in ["c06_synthetic","c07_game","c08_benchmark"]:
    meta=js(P+"/evidence/"+stage+"/metadata.json")
    summary=js(P+"/evidence/"+stage+"/summary.json")
    require(meta["command_exit_code"]==0 and meta["native_exit_code"]==0,"recorded_exit_zero")
    require(meta["final_identity_check"]=="BYTE_IDENTICAL","recorded_unchanged")
    for name,item in meta["identities"].items():
        path=pathlib.Path(item["path"])
        observed=sha(BIN.read_bytes()) if name=="native_binary" else sha(git(str(path.relative_to(BASE))))
        require(observed==item["sha256"],"pinned_identity_matches_metadata")
    data={kind:raw(stage,kind,meta) for kind in meta["raw_artifacts"]}
    datasets[stage]=data
    req,res=data["requests"],data["responses"]
    require(len(req)==len(res),"paired_request_response_count")
    require(len({q["id"] for q in req})==len(req),"unique_ids")
    for q,r in zip(req,res):
        require(q["id"]==r["id"] and q["mode"]==r["mode"] and r["schema"]=="c28-geometry-v1","request_response_alignment")
        require("error" not in r,"no_error_response")
        if q["mode"]=="game":
            require(set(q)=={"mode","id","history","cue_seed","model","rho","trace","repeats","warmup"},"game_no_oracle_request_fields")
            s=source_map[q["id"]]
            require(q["history"]==s["history"] and q["cue_seed"]==s["cue_seed"] and q["model"]==model and q["rho"]==.75,"exact_game_source_and_model")
            for key in ["initial_move","final_move","initial_utility","utility","initial_nodes","additional_nodes","cue_nodes","charged_nodes","additional_queries","query_moves","final_m"]:
                require(r["direct"][key]==r["lifted"][key],"game_path_identity")
            for method in ["direct","lifted"]:
                v=r[method]
                require(v["charged_nodes"]==v["cue_nodes"]+v["initial_nodes"]+v["additional_nodes"],"literal_charged_sum")
                require(len(v["query_moves"])==v["additional_queries"]<=2 and len(set(v["query_moves"]))==len(v["query_moves"]),"query_cap_uniqueness")
        else:
            require(r["direct"]["choice"]==r["lifted"]["choice"],"synthetic_method_choice_identity")
            require(len(r["direct"]["scores"])==len(q["probabilities"])==len(r["lifted"]["scores"]),"score_vector_length")
            for x,y in zip(r["direct"]["scores"],r["lifted"]["scores"]):
                close(x,y,"synthetic_method_score_difference")
            for method in ["direct","lifted"]:
                close(r[method]["value"],q["b"]+max([0]+r[method]["scores"]),"raw_value_identity")
    short={"request_count":len(req),"status":meta["status"],"counts":dict(collections.Counter(q["mode"] for q in req)),
           "input_and_raw_hashes_match":True,"recorded_exit_code":meta["command_exit_code"],
           "native_stderr_bytes":len(git(P+"/evidence/"+stage+"/native_stderr.log")),
           "wall_seconds":meta["wall_seconds"]}
    if "verification" in data:
        ver=data["verification"]
        require(len(ver)==len(req),"verification_count")
        require(ver==summary["checks"],"summary_checks_equal_raw")
        for q,r,v in zip(req,res,ver):
            require(q["id"]==v["id"],"verification_alignment")
            if q["mode"]=="synthetic":
                exact={int(i):F(x) for i,x in v["exact_scores"].items()}
                require(choose(exact)==v["epsilon_choice"],"recorded_exact_epsilon_choice")
                for method in ["direct","lifted"]:
                    require(r[method]["choice"]==choose(exact),"native_equals_recorded_exact_choice")
                    for i,x in enumerate(r[method]["scores"]):close(x,float(exact[i]),"native_vs_recorded_exact_score")
                    close(r[method]["value"],float(F(v["exact_value"])),"native_vs_recorded_exact_value")
        short["max_recorded_oracle_abs_residual"]=max(v["max_oracle_abs_residual"] for v in ver)
    report[stage]=short
synth=datasets["c06_synthetic"]
report["c06_synthetic"]["categories"]=dict(collections.Counter(q["id"].split("-")[0] for q in synth["requests"]))
report["c06_synthetic"]["candidate_counts"]=dict(collections.Counter(len(q["probabilities"]) for q in synth["requests"]))
report["c06_synthetic"]["horizon_counts"]=dict(collections.Counter(q["horizon"] for q in synth["requests"]))
game=datasets["c07_game"]
require([q["id"] for q in game["requests"]]==[x["id"] for x in sources],"c07_exhausts_source_order")
report["c07_game"].update(root_player_counts=dict(collections.Counter(1 if len(q["history"])%2==0 else -1 for q in game["requests"])),
 query_count_distribution=dict(collections.Counter(r["direct"]["additional_queries"] for r in game["responses"])),
 decision_count_distribution=dict(collections.Counter(len(r["direct"]["decisions"]) for r in game["responses"])),
 actual_decisions=sum(len(r["direct"]["decisions"]) for r in game["responses"]),
 actual_queries=sum(r["direct"]["additional_queries"] for r in game["responses"]),
 complete_world_status_counts=dict(collections.Counter(v["complete_world_status"] for v in game["verification"])),
 max_legal_candidates=max(9-len(q["history"]) for q in game["requests"]),
 initial_utility_distribution=dict(collections.Counter(r["direct"]["initial_utility"] for r in game["responses"])),
 explicit_stop_decisions=sum(d["choice"] is None for r in game["responses"] for d in r["direct"]["decisions"]),
 examples=[{"request_without_model":{k:v for k,v in q.items() if k!="model"},"response":r,"verification":v}
           for q,r,v in zip(game["requests"],game["responses"],game["verification"]) if len(r["direct"]["decisions"])==2][:1])
bench=datasets["c08_benchmark"]
req,res,median_rows=bench["requests"],bench["responses"],bench["root_medians"]
sreq=[q for q in req if q["mode"]=="synthetic"]; greq=[q for q in req if q["mode"]=="game"]
require([q["id"] for q in greq]==[x["id"] for x in sources],"c08_exhausts_source_order")
keys={(len(q["probabilities"]),q["m"],q["b"],q["id"].rsplit("-",1)[-1]) for q in sreq}
expected={(n,m,b,p.replace(" ","_")) for n in contract["synthetic_campaign"]["candidate_counts"]
 for m in contract["synthetic_campaign"]["latent_means"] for b in contract["synthetic_campaign"]["incumbents"]
 for p in contract["synthetic_campaign"]["loading_patterns"]}
require(keys==expected and len(sreq)==len(expected)==160,"benchmark_full_contract_product")
semantics={json.dumps({k:v for k,v in q.items() if k not in ["id","warmup","repeats","iterations"]},sort_keys=True) for q in sreq}
groups=collections.defaultdict(list); byorder=collections.defaultdict(list)
timing_pairs=collections.Counter(); planner_calls=collections.Counter()
for q,r,stored in zip(req,res,median_rows):
    require(q["id"]==stored["id"],"median_id_alignment")
    expected_repeats=11 if q["mode"]=="synthetic" else 31
    iterations=256 if q["mode"]=="synthetic" and len(q["probabilities"])<16 else (32 if q["mode"]=="synthetic" else 1)
    require(q["repeats"]==expected_repeats and q["warmup"]==3 and q.get("iterations",1)==iterations,"timing_contract")
    require(len(r["timings"])==expected_repeats,"timing_repeat_count")
    if q["mode"]=="game":require(not q["trace"] and not r["direct"]["decisions"] and not r["lifted"]["decisions"],"benchmark_trace_excluded")
    for i,t in enumerate(r["timings"]):
        require(t["repeat"]==i and t["iterations"]==iterations and t["order"]==("direct_first" if i%2==0 else "lifted_first"),"timing_order_units")
        require(type(t["direct_ns"]) is int and t["direct_ns"]>0 and type(t["lifted_ns"]) is int and t["lifted_ns"]>0,"positive_nanoseconds")
    d=statistics.median(t["direct_ns"]/iterations for t in r["timings"])
    l=statistics.median(t["lifted_ns"]/iterations for t in r["timings"])
    row={"direct_median_ns":d,"lifted_median_ns":l,"direct_minus_lifted_ns":d-l,"lifted_over_direct_ratio":l/d}
    for field,value in row.items():require(value==stored[field],"exact_median_reaggregation")
    groups[q["mode"]].append(row)
    if q["mode"]=="synthetic":groups["synthetic_n"+str(len(q["probabilities"]))+"_"+q["id"].rsplit("-",1)[-1]].append(row)
    for order in ["direct_first","lifted_first"]:
        ts=[t for t in r["timings"] if t["order"]==order]
        od=statistics.median(t["direct_ns"]/iterations for t in ts)
        ol=statistics.median(t["lifted_ns"]/iterations for t in ts)
        byorder[q["mode"]+"_"+order].append({"direct_median_ns":od,"lifted_median_ns":ol,"direct_minus_lifted_ns":od-ol,"lifted_over_direct_ratio":ol/od})
    timing_pairs[q["mode"]]+=len(r["timings"]);planner_calls[q["mode"]]+=len(r["timings"])*iterations
summ=js(P+"/evidence/c08_benchmark/summary.json")
require(median_rows==summ["median_rows"],"summary_medians_equal_raw")
aggs={}
for group,rows in groups.items():
    a=aggregate(rows);aggs[group]=a
    for field,value in a.items():require(value==summ["aggregation"][group][field],"exact_aggregate_reaggregation")
report["c08_benchmark"].update(semantic_unique_synthetic_requests=len(semantics),
 small_n_independent_world_oracle_configs=sum(len(q["probabilities"])<=5 for q in sreq),
 large_n_method_agreement_configs=sum(len(q["probabilities"])>5 for q in sreq),
 timing_pairs=dict(timing_pairs),timed_calls_per_method=dict(planner_calls),
 aggregates=aggs,order_stratified_aggregates={k:aggregate(v) for k,v in byorder.items()},
 all_medians_and_aggregates_reproduced_exactly=True)
report["checks"]=dict(checks)
report["status"]="PASS_SCOPED_RAW_AUDIT"
report["ended_utc"]=datetime.now(timezone.utc).isoformat()
report["script_sha256"]=sha(pathlib.Path(__file__).read_bytes())
out=BASE/P/"evidence/C09_REVIEW_RAW_AUDIT.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ["c06_synthetic","c07_game","c08_benchmark"]},sort_keys=True))
for stage in ["c06_synthetic","c07_game","c08_benchmark"]:
    short={k:v for k,v in report[stage].items() if k not in ["examples","aggregates"]}
    if stage=="c08_benchmark":short["main_aggregates"]={k:report[stage]["aggregates"][k] for k in ["synthetic","game"]}
    print(stage,json.dumps(short,sort_keys=True))
