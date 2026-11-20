#!/usr/bin/env python3
"""Compare trained policy/value MLP versus uniform QUARTZ Rust MCTS on old 12 roots.

Exact Connect4 WDL action scores are joined ONLY AFTER engine responses.
The model weights contain no oracle labels for the 12 held-out roots.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


BUDGETS=[8,16,32,64]
MODES={"uniform":"uniform_no_neural_network","trained":"trained_tiny_mlp_cpu"}


def sha(file:Path)->str:
    with file.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(repo:Path,*args)->str:
    return subprocess.check_output(["git","-C",str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()


def wdl(score:int)->int:
    if type(score)!=int or score == -1000 or not -21<=score<=21:
        raise ValueError("illegal Connect4 score or invalid-action sentinel")
    return (score>0)-(score<0)


def per_root_metrics(r:dict,scores:list[int])->dict:
    if len(scores)!=7:raise ValueError("seven action scores required")
    if r.get("fresh16",{}).get("resident_match") is not True:
        raise ValueError("fresh16 replay diverged")
    steps=r.get("steps")
    if not isinstance(steps,list) or [s.get("budget") for s in steps]!=BUDGETS:
        raise ValueError("bad MCTS budgets")
    root=r.get("root_hash")
    if not root or any(s.get("root_hash")!=root for s in steps):
        raise ValueError("root identity changed")
    util=[];calls=[];wall=[];actions=[]
    prev=0
    for s,budget in zip(steps,BUDGETS):
        if s.get("root_visits")!=budget:raise ValueError("root visit target mismatch")
        a=s.get("best_action")
        if type(a)!=int or not 0<=a<7 or scores[a]==-1000:
            raise ValueError("illegal selected Connect4 column")
        curr=s.get("evaluator_calls")
        if type(curr)!=int or curr<prev:raise ValueError("evaluator counts not monotone")
        calls.append(curr)
        actions.append(a)
        util.append(wdl(scores[a]))
        micros=s.get("step_wall_us")
        if type(micros)!=int or micros<0:raise ValueError("invalid wall timing")
        wall.append(micros)
        prev=curr
    optimal=max(wdl(s) for s in scores if s!=-1000)
    return {
        "action":actions,
        "utility":util,
        "regret":[optimal-u for u in util],
        "delta_utility":[b-a for a,b in zip(util,util[1:])],
        "best_root_WDL":optimal,
        "cumulative_eval_calls":calls,
        "stage_wall_us":wall,
        "fresh16_verified":True,
        "eval_mode":r.get("eval_mode"),
        "root_hash":root,
    }


def summarize_modes(paired:list[dict],*,budgets:list[int]=BUDGETS)->dict:
    if not paired:raise ValueError("no paired roots")
    report={
        "schema":"quartz_c4_recognizer06_heldout_evaluation_v1",
        "n_paired_roots":len(paired),
        "by_budget":{},
        "model_improvement_not_assumed":True,
    }
    for i,b in enumerate(budgets):
        rows={}
        for name in ("uniform","trained"):
            xs=[root[name] for root in paired]
            rows[name]={
                "mean_utility":sum(x["utility"][i] for x in xs)/len(xs),
                "mean_regret":sum(x["regret"][i] for x in xs)/len(xs),
                "optimal_WDL_count":sum(x["regret"][i]==0 for x in xs),
                "selected_WDL":dict(Counter(str(x["utility"][i]) for x in xs)),
                "mean_evaluator_calls":sum(x["cumulative_eval_calls"][i] for x in xs)/len(xs),
                "mean_stage_wall_us":sum(x["stage_wall_us"][i] for x in xs)/len(xs),
            }
        rows["trained_minus_uniform_mean_utility"]=rows["trained"]["mean_utility"]-rows["uniform"]["mean_utility"]
        rows["trained_minus_uniform_mean_regret"]=rows["trained"]["mean_regret"]-rows["uniform"]["mean_regret"]
        report["by_budget"][str(b)]=rows
    for i,(a,b) in enumerate(zip(budgets,budgets[1:])):
        report[f"{a}->{b}"]={
            name:{
                "delta_counts":dict(Counter(str(x[name]["delta_utility"][i]) for x in paired)),
                "mean_delta":sum(x[name]["delta_utility"][i] for x in paired)/len(paired),
                "mean_marginal_evaluator_calls":sum(
                    x[name]["cumulative_eval_calls"][i+1]-x[name]["cumulative_eval_calls"][i]
                    for x in paired)/len(paired),
            }
            for name in ("uniform","trained")
        }
    report["metadata"]={
        "oracle":"external exact Connect4 7x6 game-theoretic signed scores",
        "model":"tiny 126-64-8 frozen CPU oracle-set distillation",
        "root_count":len(paired),
        "seed":17,
        "matched":"MCTS cumulative root visit budgets only; NOT evaluator calls or wall-time",
        "nonclaims":["not Bayesian VOC calibration","not generalizable Elo","not independent position sampling"],
    }
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--binary",type=Path,required=True)
    cli.add_argument("--output-dir",type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/".git").exists())
    if git(repo,"status","--porcelain"):
        raise RuntimeError("evaluation must run from committed clean Git source")
    prior=d.parent/"standard_external_04"
    fixed=json.loads((prior/"POSITIONS_FIXED.json").read_text())["connect4"]
    exact=json.loads((prior/"evidence/20261120_standard_cpu/c4_oracle_results.json").read_text())
    if len(fixed)!=12 or len(exact)!=12:
        raise RuntimeError("original frozen 12-root cohort changed")
    lookup={r["position_id"]:r for r in exact}
    if {x["id"] for x in fixed}!={x["position_id"] for x in exact}:
        raise RuntimeError("frozen roots not matched")
    teacher=json.loads((d/"evidence/teacher_20261120/labels.json").read_text())
    frozen_train={x["history"] for x in teacher["full_rows"]}
    for pos in fixed:
        if pos["sequence"] in frozen_train or "".join(str(8-int(x)) for x in pos["sequence"]) in frozen_train:
            raise RuntimeError("exact evaluation root leaked into teacher corpus")
    trained=d/"evidence/nn_train_20261120/tiny_mlp_weights.json"
    trainmanifest=json.loads((d/"evidence/nn_train_20261120/train_manifest.json").read_text())
    if sha(trained)!=trainmanifest["model_json_sha256"]:
        raise RuntimeError("trained frozen model SHA mismatch")
    lock=json.loads((d/"INPUT_LOCK.json").read_text())
    for file in lock["source_files"]:
        if file["path"].endswith(("/COHORT_FROZEN.json","/7x6_small.book","/c4solver",
                                 "/POSITIONS_FIXED.json","/c4_oracle_results.json")):
            obj=Path(file["path"])
            if sha(obj)!=file["sha256"]:
                raise RuntimeError("experiment inputs changed: "+str(obj))
    requests=[]
    for name in ("uniform","trained"):
        for row in fixed:
            req={"cmd":"research_connect4_cpu",
                 "history":row["sequence"],"seed":17,
                 "budgets":BUDGETS}
            if name=="trained":
                req.update({"eval_mode":"tiny_mlp","model_path":str(trained.resolve())})
            requests.append(req)
    assert len(requests)==24
    binary=args.binary.resolve()
    if not binary.is_file():raise FileNotFoundError(binary)
    send="\n".join(json.dumps(q,sort_keys=True) for q in requests)+"\n"+json.dumps({"cmd":"quit"})+"\n"
    start=time.monotonic()
    proc=subprocess.run([str(binary),"--server"],input=send.encode(),capture_output=True,timeout=35,check=False)
    if proc.returncode!=0:raise RuntimeError("Rust server error: "+proc.stderr.decode(errors="replace")[-500:])
    response=[json.loads(line) for line in proc.stdout.decode().splitlines() if line.strip()]
    if len(response)!=len(requests):raise RuntimeError("Rust server output count mismatch")
    for request,r in zip(requests,response):
        if r.get("error"):
            raise RuntimeError("one MCTS root failed: "+str(r))
        if r.get("schema")!="quartz_connect4_resident_mcts_v1" or r.get("history")!=request["history"]:
            raise RuntimeError("Rust source/result identity mismatch")
    samples=[]
    for i,pos in enumerate(fixed):
        ref=lookup[pos["id"]]
        assert ref["status"]=="EXACT_EXTERNAL_SOLVER" and ref["sequence"]==pos["sequence"]
        row={"id":pos["id"],"history":pos["sequence"],"exact_action_scores":ref["scores"],
             "uniform":per_root_metrics(response[i],ref["scores"]),
             "trained":per_root_metrics(response[12+i],ref["scores"])}
        if row["uniform"]["eval_mode"]!=MODES["uniform"] or row["trained"]["eval_mode"]!=MODES["trained"]:
            raise RuntimeError("baseline/learned evaluator identity confused")
        samples.append(row)
    summ=summarize_modes(samples)
    summ["engine_elapsed_s"]=time.monotonic()-start
    summ["training_best_epoch"]=trainmanifest["epoch_selected"]
    summ["training_model_sha256"]=trainmanifest["model_json_sha256"]
    # The previously published uniform run must reproduce exactly in actions/calls.
    original=json.loads((d.parent/"connect4_gamestate_05/evidence/20261120_connect4_exact_12/raw_mcts_responses.json").read_text())
    discrepancies=[]
    for old,now in zip(original,response[:12]):
        for a,b in zip(old["steps"],now["steps"]):
            if a["best_action"]!=b["best_action"] or a["evaluator_calls"]!=b["evaluator_calls"]:
                discrepancies.append((old["history"],a["budget"],a["best_action"],b["best_action"]))
    summ["previous_frozen_uniform_action_call_discrepancies"]=discrepancies
    if discrepancies:
        raise RuntimeError(f"uniform baseline reproduction failed: {discrepancies[:2]}")
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,body in [
        ("rust_raw_responses.json",response),
        ("paired_heldout.json",samples),
        ("summary.json",summ),
    ]:
        (out/name).write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    manifest={
        "schema":"quartz_c4_recognizer06_evaluation_run_manifest_v1",
        "source_head":git(repo,"rev-parse","HEAD"),
        "source_tree":git(repo,"rev-parse","HEAD^{tree}"),
        "rust_binary_sha256":sha(binary),
        "model_weights_sha256":sha(trained),
        "frozen_heldout_position_sha256":sha(prior/"POSITIONS_FIXED.json"),
        "frozen_heldout_oracle_sha256":sha(prior/"evidence/20261120_standard_cpu/c4_oracle_results.json"),
        "teacher_labels_sha256":sha(d/"evidence/teacher_20261120/labels.json"),
        "outputs_sha256":{p.name:sha(p) for p in out.iterdir()},
        "no_gpu":True,
        "infer_forward":"in-Rust f32 MLP (one forward per evaluator_call)",
        "budget_identity":"same visits; evaluator calls/timing can differ",
    }
    (out/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"completed","n_roots":len(samples),"summary":summ},sort_keys=True))


if __name__=="__main__":
    main()
