#!/usr/bin/env python3
"""CPU-only exact teacher labels from official Pascal Pons standalone solver.

Never train on the prior 12 frozen evaluation roots. Failures/timeouts are
explicit original rows, with no imputation. No solver labels enter Rust runtime.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def sha(path:Path)->str:
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def parse_exact_label(history:str,scores:list[int],legal:list[int])->dict:
    if not isinstance(scores,list) or len(scores)!=7:
        raise ValueError("expected seven exact action scores")
    if len(set(legal))!=len(legal) or not legal or any(type(a)!=int or not 0<=a<7 for a in legal):
        raise ValueError("invalid legal action set")
    if any(type(score)!=int or (score!=-1000 and not -21<=score<=21) for score in scores):
        raise ValueError("score outside documented exact domain")
    for c,score in enumerate(scores):
        if (score==-1000)!=(c not in legal):
            raise ValueError("invalid/sentinel score and legal columns mismatch")
    sign=lambda score:(score>0)-(score<0)
    optimal=max(sign(scores[c]) for c in legal)
    opt=[c for c in legal if sign(scores[c])==optimal]
    strict=max(scores[c] for c in legal)
    strict_actions=[c for c in legal if scores[c]==strict]
    policy=[0.0]*7
    for c in opt:policy[c]=1.0/len(opt)
    return {
        "root_wdl":optimal,"optimal_actions":opt,
        "target_policy":policy,
        "strict_fastest_win_actions":strict_actions,
        "exact_action_scores":list(scores),
        "legal_columns":list(legal),
    }


def encode_relative(board:list[int],current_player:int)->list[float]:
    if len(board)!=42 or current_player not in (-1,1) or any(z not in (-1,0,1) for z in board):
        raise ValueError("invalid 7x6 cell-state encoding")
    a=[1.0 if x==current_player else 0.0 for x in board]
    b=[1.0 if x==-current_player else 0.0 for x in board]
    c=[1.0 if current_player==1 else 0.0]*42
    return a+b+c


def mirror_encoding(x:list[float])->list[float]:
    if len(x)!=126:raise ValueError("expected 3*6*7 encoded features")
    out=[]
    for plane in range(3):
        off=plane*42
        for row in range(6):
            out.extend(x[off+row*7+c] for c in range(6,-1,-1))
    return out


def mirror_target(p:list[float])->list[float]:
    if len(p)!=7:raise ValueError("invalid policy target")
    return list(reversed(p))


def verify_frozen(d:Path,lock:dict)->None:
    for file in lock["source_files"]:
        p=Path(file["path"])
        if not p.is_file() or p.stat().st_size!=file["bytes"] or sha(p)!=file["sha256"]:
            raise RuntimeError("TEACHER_INPUT_SHA_MISMATCH: "+str(p))


def query_one(executable:Path,book:Path,history:str,*,timeout_s:float):
    try:
        t=time.perf_counter()
        proc=subprocess.run(
            [str(executable),"-a","-b",str(book)],input=(history+"\n").encode(),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=timeout_s,check=False)
        wall=time.perf_counter()-t
    except subprocess.TimeoutExpired:
        return {"status":"TIMEOUT","timeout_s":timeout_s}
    if proc.returncode!=0:
        return {"status":"ENGINE_ERROR","code":proc.returncode,"stderr_tail":proc.stderr.decode(errors="replace")[-200:]}
    text=proc.stdout.decode("utf-8","replace").split()
    if len(text)!=8 or text[0]!=history:
        return {"status":"INVALID_OUTPUT","stdout_tail":proc.stdout.decode(errors="replace")[-200:]}
    try:
        vals=[int(x) for x in text[1:]]
    except ValueError:
        return {"status":"INVALID_SCORE"}
    return {"status":"OK","scores":vals,"wall_s":wall}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--max-wall-s",type=float,default=110.0)
    args=parser.parse_args()
    if args.max_wall_s!=110.0:raise ValueError("pre-registered global cap 110 seconds")
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/".git").exists())
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"]).strip():
        raise RuntimeError("teacher label generation requires committed clean source")
    lock=json.loads((d/"INPUT_LOCK.json").read_text())
    if lock["schema"]!="quartz_connect4_recognizer06_input_lock_v1":
        raise RuntimeError("wrong scientific input contract")
    verify_frozen(d,lock)
    cohort=json.loads((d/"COHORT_FROZEN.json").read_text())
    assert len(cohort["rows"])==192
    c4bin=Path(next(x["path"] for x in lock["source_files"] if x["path"].endswith("/c4solver")))
    book=Path(next(x["path"] for x in lock["source_files"] if x["path"].endswith("/7x6_small.book")))
    rows=[]
    started=time.monotonic()
    for row in cohort["rows"]:
        now=time.monotonic()
        if now-started>=args.max_wall_s:
            r={"status":"NOT_ATTEMPTED_GLOBAL_TIME_CAP"}
        else:
            r=query_one(c4bin,book,row["history"],timeout_s=5.0)
        if r["status"]=="OK":
            try:
                # Proof contract: board legal moves have zero-based 7-column indices.
                target=parse_exact_label(row["history"],r["scores"],row["legal_columns"])
                r.update(target)
                r["features"]=encode_relative(row["board"],row["current_player"])
            except ValueError as exc:
                r={"status":"INVALID_TEACHER_LABEL","reason":str(exc)}
        rows.append({
            "id":row["id"],"family":row["family"],"split":row["split"],
            "plies":row["plies"],"history":row["history"],
            "current_player":row["current_player"],**r,
        })
    verify_frozen(d,lock)
    counts=Counter(x["status"] for x in rows)
    q=Counter((x["split"],x["status"]) for x in rows)
    result={
        "schema":"quartz_connect4_oracle_labels_v1",
        "source_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "oracle_source_commit":lock["teacher_source_commit"],
        "cohort_sha256":sha(d/"COHORT_FROZEN.json"),
        "full_rows":rows,
        "counts":dict(counts),
        "by_split":{s:dict(Counter(x["status"] for x in rows if x["split"]==s)) for s in ["train","val","test"]},
        "elapsed_s":time.monotonic()-started,
        "no_missing_imputation":True,
        "heldout_evaluation_roots_in_training":False,
        "label_policy":"all WDL-optimal legal moves, uniform on set",
    }
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    p=target/"labels.json"
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    m={"schema":"quartz_c4_labels_manifest_v1","source_head":result["source_commit"],
       "lock_sha256":sha(d/"INPUT_LOCK.json"),
       "cohort_sha256":result["cohort_sha256"],
       "teacher_binary_sha256":sha(c4bin),
       "teacher_book_sha256":sha(book),
       "labels_sha256":sha(p),"counts":dict(counts),"splits":result["by_split"],
       "wall_time_limit_s":110.0,"per_root_timeout_s":5.0}
    (target/"label_manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"counts":dict(counts),"by_split":result["by_split"],
                      "elapsed_s":result["elapsed_s"],"sha":m["labels_sha256"]},sort_keys=True))


if __name__=="__main__":main()
