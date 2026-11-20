#!/usr/bin/env python3
"""Bounded CPU standard-game research: Connect4 exact external scores + Rapfi proxy.

Run on the authorized CPU host with the locked public standalone engines.
No third-party source/weights are written to the QUARTZ research tree.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

from standard_games import (
    c4_action_scores,c4_from_sequence,c4_has_four,
    generate_c4_positions,generate_gomoku_positions,
    rapfi_parse_move
)


def sha(path:Path)->str:
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def git(repo:Path,*args:str)->str:
    return subprocess.check_output(["git","-C",str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()


def dump(path:Path,value)->None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n")


def verify_locked_inputs(lock):
    for item in lock["input_files"]:
        path=Path(item["path"])
        if not path.is_file() or path.stat().st_size!=item["size"] or sha(path)!=item["sha256"]:
            raise RuntimeError("EXTERNAL_IDENTITY_MISMATCH: "+item["role"])


def select_external_input(lock,role):
    return Path(next(item["path"] for item in lock["input_files"] if item["role"]==role))


def run_connect4_solver(position, exe, book, *, timeout_s=8.0):
    seq=position["sequence"]
    expected_board=c4_from_sequence(seq)
    assert expected_board==position["board"]
    start=time.perf_counter()
    try:
        job=subprocess.run(
            [str(exe),"-a","-b",str(book)],
            input=(seq+"\n").encode(),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=timeout_s,check=False,cwd=str(exe.parent))
    except subprocess.TimeoutExpired:
        return {
            "position_id":position["id"],"sequence":seq,"plies":len(seq),
            "status":"TIMEOUT","timeout_s":timeout_s,
            "meaning":"external exact oracle not obtained; no score imputed",
        }
    elapsed=time.perf_counter()-start
    stdout=job.stdout.decode("utf-8","replace").strip()
    stderr=job.stderr.decode("utf-8","replace").strip()
    if job.returncode!=0:
        raise RuntimeError(f"C4 solver exited {job.returncode} {stderr[-600:]}")
    scores=c4_action_scores(seq,stdout,expected_board)
    sign=1 if len(seq)%2==0 else -1
    for c,score in enumerate(scores):
        if score==-1000:continue
        b=expected_board.copy()
        height=sum(b[y*7+c]!=0 for y in range(6))
        assert height<6
        b[height*7+c]=sign
        if c4_has_four(b,sign) and score<=0:
            raise RuntimeError("C4 external exact score contradicts immediate four-in-row")
    v=max(x for x in scores if x!=-1000)
    return {
        "position_id":position["id"],"sequence":seq,"plies":len(seq),
        "status":"EXACT_EXTERNAL_SOLVER","scores":scores,
        "best_score":v,
        "best_columns_1based":[i+1 for i,x in enumerate(scores) if x==v],
        "root_WDL":"win" if v>0 else "draw" if v==0 else "loss",
        "wall_s":elapsed,"stdout":stdout,"stderr":stderr,
        "game":"Connect4_7x6_gravity",
    }


def run_rapfi_best(position, exe, *, timeout_turn_ms=500, total_timeout_s=8.0):
    header=[
        "START 15", "INFO rule 0",
        f"INFO timeout_turn {timeout_turn_ms}",
        "INFO thread_num 1", "BOARD",
    ]
    header.extend(f"{x},{y},{player}" for x,y,player in position["history"])
    header.append("DONE")
    source="\n".join(header)+"\n"
    t0=time.perf_counter()
    proc=subprocess.Popen(
        [str(exe)],cwd=str(exe.parent),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        bufsize=0)
    output=b""
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(source.encode());proc.stdin.flush()
        move=None
        deadline=t0+total_timeout_s
        while time.perf_counter()<deadline:
            remaining=max(0.0,deadline-time.perf_counter())
            readable,_,_=select.select([proc.stdout],[],[],min(remaining,0.6))
            if not readable:
                if proc.poll() is not None:break
                continue
            block=os.read(proc.stdout.fileno(),16384)
            if not block:break
            output+=block
            try:
                move=rapfi_parse_move(output.decode("utf-8","replace"),position["board"])
                break
            except ValueError:
                continue
        if move is None:
            raise TimeoutError("Rapfi did not produce an NNUE-confirmed legal move; raw_tail="+repr(output.decode("utf-8","replace")[-1800:]))
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write(b"END\n")
                proc.stdin.flush()
            except (OSError,BrokenPipeError):
                pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait()
    text=output.decode("utf-8","replace")
    if proc.returncode!=0:
        raise RuntimeError(f"Rapfi exit {proc.returncode}, output={text[-700:]}")
    return {
        "position_id":position["id"],"rules":"freestyle15",
        "status":"EXTERNAL_STRONG_PROXY",
        "nnue_pretrained_loaded":True,
        "best_action":move,
        "best_x":move%15,"best_y":move//15,
        "wall_s":time.perf_counter()-t0,
        "timeout_turn_ms":timeout_turn_ms,
        "lines":text.splitlines(),
        "solver_confirmation":"not-game-theoretically-exact",
    }


def run_quartz_cpu(position, ckpt, binary, *, torch, network_cls, client_cls):
    cfg_raw=torch.load(ckpt,map_location="cpu",weights_only=True)
    cfg=dict(cfg_raw["cfg"])
    if (cfg["board"],cfg["ch"],cfg["actions"],cfg["win"])!=(15,17,225,5):
        raise RuntimeError("incompatible smoke checkpoint for freestyle Gomoku15")
    cfg.update({
        "_name":"gomoku15","search_profile":"baseline_strict",
        "penalty_mode":"None","n_threads":1,"batch_size":1,
        "batch_timeout_us":1500,"iters":8,"seed":17,
        "min_visits":64,"check_interval":64,
    })
    torch.set_num_threads(2)
    class CountingNet(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net;self.eval_items=0
        def forward(self,x):
            assert x.device.type=="cpu"
            self.eval_items+=int(x.shape[0])
            return self.net(x)
    model=CountingNet(network_cls(cfg).eval())
    model.net.load_state_dict(cfg_raw["model_state_dict"],strict=True)
    cli=client_cls(model,cfg,torch.device("cpu"),rust_binary=str(binary))
    job={"board":position["board"],"player":position["player"]}
    options=cli._rt.rust_search_options(cfg,penalty_mode="None")
    try:
        t0=time.perf_counter()
        opened=cli._exchange_search_request({
            "cmd":"search_nn_multi_engine_session_open","game":"gomoku15",
            "iters":8,"diagnostic_json":True,"jobs":[job],**options,
        })
        t1=time.perf_counter()
        if opened.get("response_schema")!="quartz_resident_diag_json_v1":
            raise RuntimeError("old/incomplete resident diagnostics")
        sid=opened["session_id"]
        a=opened["results"][0]
        try:
            stepped=cli._exchange_search_request({
                "cmd":"search_nn_multi_engine_session_step",
                "session_id":sid,"iters":8,"updates":[{}],
                "diagnostic_json":True,
            })
            t2=time.perf_counter()
            if stepped.get("session_id")!=sid:
                raise RuntimeError("Rust resident session identity changed")
            b=stepped["results"][0]
        finally:
            close=cli.close_search_session(sid)
            if not close.get("ok",False):
                raise RuntimeError("rust session close did not acknowledge")
        assert a["root_visits"]==8 and b["root_visits"]==16
        assert a["root_hash"]==b["root_hash"]
        assert b["evaluator_calls"]==model.eval_items
        assert b["additional_evaluator_calls"]==b["evaluator_calls"]-a["evaluator_calls"]
        for row in (a,b):
            action=row["best_move"]
            if not isinstance(action,int) or not (0<=action<225) or position["board"][action]!=0:
                raise RuntimeError("QUARTZ chose illegal Gomoku15 action")
        return {
            "position_id":position["id"],"status":"CPU_RESIDENT",
            "best8":a["best_move"],"best16":b["best_move"],
            "root_hash":a["root_hash"],"visits":[8,16],
            "calls8":a["evaluator_calls"],"calls16":b["evaluator_calls"],
            "physical_nn_items":model.eval_items,
            "additional_physical_nn_items":b["additional_evaluator_calls"],
            "wall_ms8":(t1-t0)*1000,"wall_ms_add8":(t2-t1)*1000,
            "checkpoint_strength":"smoke_unknown_not_strong",
        }
    finally:
        cli.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(parent for parent in d.parents if (parent/".git").exists())
    if not args.smoke and (args.output_dir is None or git(repo,"status","--porcelain")):
        raise RuntimeError("full run requires unique --output-dir and clean source")
    lock=json.loads((d/"EXTERNAL_LOCK.json").read_text())
    verify_locked_inputs(lock)
    frozen=json.loads((d/"POSITIONS_FIXED.json").read_text())
    assert frozen["schema"]=="quartz_standard_games_positions_v1"
    assert len(frozen["connect4"])==12 and len(frozen["gomoku15"])==12
    if frozen["connect4"]!=generate_c4_positions(seed=20261120) or frozen["gomoku15"]!=generate_gomoku_positions(seed=20261120):
        raise RuntimeError("locked position list changed")
    c4_exe=select_external_input(lock,"c4_compiled_binary")
    c4_book=select_external_input(lock,"c4_opening_book")
    rapfi_exe=select_external_input(lock,"rapfi_prebuilt_avx2")
    checkpoint=select_external_input(lock,"quartz_gomoku15_smoke_checkpoint")
    rust_exe=select_external_input(lock,"quartz_rust_binary")
    source_root=Path(select_external_input(lock,"quartz_rust_server_source")).parents[1]
    sys.path.insert(0,str(source_root))
    import torch
    from quartz.models_torch import AlphaZeroNet
    from quartz.runtime_support import NNSearchClient
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("use CPU-only torch")
    if os.getenv("QUARTZ_DISABLE_NN_CACHE")!="1" or os.getenv("QUARTZ_DISABLE_ASYNC_PIPELINE")!="1":
        raise RuntimeError("disable both NN cache and async pipeline")
    if os.getenv("QUARTZ_SEARCH_POLICY") or os.getenv("QUARTZ_CALIBRATION_DIR"):
        raise RuntimeError("unexpected QUARTZ policy/calibration environment")
    c4_rows=[]; gomoku_rows=[]
    begin=time.perf_counter()
    for pos in (frozen["connect4"][:1] if args.smoke else frozen["connect4"]):
        result=run_connect4_solver(pos,c4_exe,c4_book,timeout_s=8)
        c4_rows.append(result)
        if args.smoke:print("C4_SMOKE",json.dumps(result,sort_keys=True)[:1200],flush=True)
    for pos in (frozen["gomoku15"][:1] if args.smoke else frozen["gomoku15"]):
        rapfi=run_rapfi_best(pos,rapfi_exe,timeout_turn_ms=500,total_timeout_s=8)
        quartz=run_quartz_cpu(pos,checkpoint,rust_exe,torch=torch,network_cls=AlphaZeroNet,client_cls=NNSearchClient)
        combined={"position_id":pos["id"],"rapfi":rapfi,"quartz":quartz,
                  "match8":int(rapfi["best_action"]==quartz["best8"]),
                  "match16":int(rapfi["best_action"]==quartz["best16"]),
                  "delta_match":int(rapfi["best_action"]==quartz["best16"])-int(rapfi["best_action"]==quartz["best8"])}
        gomoku_rows.append(combined)
        if args.smoke:
            print("GOMOKU_SMOKE",json.dumps({"position_id":pos["id"],"rapfi_action":rapfi["best_action"],"quartz8":quartz["best8"],"quartz16":quartz["best16"],"NN_calls":quartz["calls16"],"matchdelta":combined["delta_match"]},sort_keys=True),flush=True)
    if args.smoke:
        print("STANDARD_EXTERNAL_TWO_GAMES_SMOKE_PASS")
        return
    summary={
        "schema":"quartz_standard_external_two_games_v1",
        "connect4":{
            "n":len(c4_rows),"status_counts":dict(Counter(x["status"] for x in c4_rows)),
            "wdl_counts":dict(Counter(x.get("root_WDL","unknown") for x in c4_rows)),
            "exact_wdl_scores_from_pascalpons":True,
            "not_QUARTZ_search_comparison":True,
        },
        "gomoku15":{
            "n":len(gomoku_rows),
            "rapfi_nnue_weight_source":"official release",
            "quartz_checkpoint_strength":"smoke_not_elo_validated",
            "match8":sum(x["match8"] for x in gomoku_rows),
            "match16":sum(x["match16"] for x in gomoku_rows),
            "signed_delta_counts":dict(Counter(str(x["delta_match"]) for x in gomoku_rows)),
            "extra_physical_nn_evals":sum(x["quartz"]["additional_physical_nn_items"] for x in gomoku_rows),
            "mean_additional_wall_ms":sum(x["quartz"]["wall_ms_add8"] for x in gomoku_rows)/len(gomoku_rows),
            "rapfi_mean_wall_s":sum(x["rapfi"]["wall_s"] for x in gomoku_rows)/len(gomoku_rows),
            "statistically_generalizable":False,
        },
        "duration_s":time.perf_counter()-begin,
        "nonclaims":["not_same_rules_between_games","not_minimax_gomoku_value","not_trained_strong_quartz_checkpoint","not_Bayesian_VOC","not_Elo"],
    }
    for item in lock["input_files"]:
        p=Path(item["path"])
        if sha(p)!=item["sha256"]:
            raise RuntimeError("INPUT_CHANGED_DURING_EXECUTION: "+item["role"])
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"c4_oracle_results.json",c4_rows)
    dump(out/"gomoku15_external_vs_quartz.json",gomoku_rows)
    dump(out/"summary.json",summary)
    manifest={
        "schema":"quartz_standard_external_execution_v1",
        "source_commit":git(repo,"rev-parse","HEAD"),
        "source_tree":git(repo,"rev-parse","HEAD^{tree}"),
        "external_lock_sha256":sha(d/"EXTERNAL_LOCK.json"),
        "positions_sha256":sha(d/"POSITIONS_FIXED.json"),
        "python":sys.version,
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "output_sha256":{p.name:sha(p) for p in out.iterdir()},
        "external_source_no_vendor":True,
    }
    dump(out/"execution_manifest.json",manifest)
    print(json.dumps({
        "status":"done","source":manifest["source_commit"],
        "c4":summary["connect4"],"gomoku":summary["gomoku15"],
        "output":str(out),
    },sort_keys=True))


if __name__=="__main__":
    main()
