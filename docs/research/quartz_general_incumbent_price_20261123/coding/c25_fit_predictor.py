"""Fit the prespecified C25 future-block predictor from TRAIN timings only.

No oracle file is read. The predictor is an empirical point estimate, not a
pathwise or universal expected lower bound on future hardware time.
"""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
from c25_io import assert_committed, fingerprints, manifest_files, rows, scientific_paths, sha, utc

def fit_predictor(config,envelopes):
    contract=config["predictor"]
    assert contract["fit_split"]=="TRAIN" and contract["fit_policy"]=="RETAIN_BLOCK"
    assert contract["fit_additional"]==32 and contract["group"]=="side"
    specs={s["id"]:s for s in config["sources"] if s["split"]=="TRAIN"}
    seen=set();parents=[]
    for env in envelopes:
        ident=env["source_id"]
        assert env["split"]=="TRAIN" and env["mode"]=="timing","TRAIN timings only"
        assert ident in specs,"predictor source outside frozen TRAIN"
        if env["additional"]!=32:
            continue
        assert ident not in seen,"duplicate TRAIN parent/budget"
        seen.add(ident)
        rr=[r for r in env["native"]["results"] if r["policy"]=="RETAIN_BLOCK"]
        assert len(rr)==config["repeats"] and {r["rep"] for r in rr}==set(range(config["repeats"])),"TRAIN repeat coverage"
        blocks=[r["c25"]["executed_blocks"] for r in rr]
        assert all(type(v) is int and v>=0 for v in blocks) and len(set(blocks))==1,"stable TRAIN block count"
        times=[r["c25"]["continuation_us"] for r in rr]
        assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in times),"TRAIN continuation time"
        ratios=[t/b for t,b in zip(times,blocks) if b>0]
        value=statistics.mean(ratios) if ratios else None
        if value is not None:
            assert math.isfinite(value) and value>0,"positive active-block prediction"
        parents.append({"id":ident,"side":specs[ident]["side"],"executed_blocks":blocks[0],
                        "parent_mean_block_us":value,"repeat_ratios_us":ratios})
    assert seen==specs.keys(),"complete frozen TRAIN parent coverage"
    by_side={}
    for side in sorted({s["side"] for s in specs.values()}):
        active=[p for p in parents if p["side"]==side and p["executed_blocks"]>0]
        assert active,("no active TRAIN parent for side",side)
        by_side[str(side)]={"predicted_block_us":statistics.median(p["parent_mean_block_us"] for p in active),
                           "active_parent_ids":[p["id"] for p in active],
                           "active_parent_count":len(active)}
    return {"by_side":by_side,"parents":parents,"training_parent_ids":[s for s in specs],
            "inactive_parent_ids":[p["id"] for p in parents if p["executed_blocks"]==0],
            "estimator":"per-parent mean over repeats of continuation_us/executed_blocks, then side median",
            "oracle_fields_used":False,"heldout_inputs_used":False}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];root=doc.parents[2]
    ev=(args.evidence_dir or doc/"evidence").resolve()
    cfg=json.loads((doc/"SOURCE_SPECS.json").read_text())
    output=ev/cfg["predictor"]["artifact"]
    if output.exists():
        raise FileExistsError("preserved TRAIN predictor exists; never refit in place")
    scientific=scientific_paths(root,doc);assert_committed(root,scientific)
    run=ev/"C25_TRAIN_TIMING.manifest.json"
    verification=ev/"C25_VERIFY_TRAIN_TIMING.json"
    verified=json.loads(verification.read_text())
    assert verified["status"]=="PASS" and verified["split"]=="TRAIN" and verified["mode"]=="timing"
    assert verified["input_sha256"][run.name]==sha(run),"verified TRAIN timing identity"
    assert verified["input_sha256"]["SOURCE_SPECS.json"]==sha(doc/"SOURCE_SPECS.json")
    inputs=scientific+manifest_files(run)+[verification]
    before=fingerprints(inputs,root)
    result=fit_predictor(cfg,rows(run))
    after=fingerprints(inputs,root);assert before==after,"prediction inputs changed"
    result.update({"schema":"quartz.c25.train-cost-predictor.v1","status":"TRAIN_ONLY_FROZEN_POINT_ESTIMATE",
                   "created_utc":utc(),"argv":sys.argv,"python":sys.version,
                   "source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                   "source_specs_sha256":sha(doc/"SOURCE_SPECS.json"),"fit_policy":"RETAIN_BLOCK","fit_additional":32,
                   "input_hashes_before":before,"input_hashes_after":after,
                   "interpretation":"Empirical predictor only; no theorem about hardware-time lower bounds"})
    with output.open("x") as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"by_side":result["by_side"]}))

if __name__=="__main__":
    main()
