#!/usr/bin/env python3
"""Execute bounded CPU-only same-root MCTS 8+8 paired evidence and fresh16 control.

Do not confuse reference agreement with Bayes VOC or minimax truth. Ordinary
server replies remain compact; full fields require explicit diagnostic_json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from resident_fork_cpu import validate_pair, summarize_rows


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def atomic_new_json(target: Path, payload) -> None:
    if target.exists():
        raise RuntimeError(f"refusing output overwrite: {target}")
    target.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")


def verified_reply(payload: dict, *, stage: str) -> dict:
    if not isinstance(payload,dict) or payload.get("response_schema")!="quartz_resident_diag_json_v1":
        raise RuntimeError(f"{stage}: diagnostic mode returned incomplete payload: {payload}")
    if "session_id" not in payload:
        raise RuntimeError(f"{stage}: missing session id")
    rows=payload.get("results")
    if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],dict):
        raise RuntimeError(f"{stage}: unexpected result cardinality")
    return rows[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--rust-binary",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=False)
    parser.add_argument("--smoke",action="store_true",help="one trial only, no evidence writes")
    args=parser.parse_args()
    if not args.smoke and args.output_dir is None:
        parser.error("--output-dir is required unless --smoke")

    repo=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
    source_root=args.input_root.resolve()
    rust_bin=args.rust_binary.resolve()
    if not rust_bin.is_file():
        raise RuntimeError("missing exact Rust server binary")
    if not args.smoke and git(repo,"status","--porcelain"):
        raise RuntimeError("study worktree must be clean for published scientific evidence")

    study_head=git(repo,"rev-parse","HEAD")
    lock=json.loads((Path(__file__).parent/"INPUT_LOCK.json").read_text())
    if lock["schema"]!="quartz_resident_fork02_inputs_v1":
        raise ValueError("unrecognized input lock")
    if git(source_root,"rev-parse","HEAD")!=lock["input_checkout_head"]:
        raise RuntimeError("source data checkout no longer at input-locked HEAD")
    for source in lock["frozen_files"]:
        path=(source_root/source["identity_path"])
        if not path.is_file() or path.stat().st_size!=source["bytes"] or sha256(path)!=source["sha256"]:
            raise RuntimeError(f"frozen input mismatch: {source['identity_path']}")

    pfile=Path(__file__).parent/"POSITIONS_LOCK.json"
    if sha256(pfile)!=lock["positions_lock_sha256"]:
        raise RuntimeError("frozen position lock has changed")
    positions=json.loads(pfile.read_text())["positions"]
    if len(positions)!=12 or [p["id"] for p in positions]!=sorted(p["id"] for p in positions):
        raise ValueError("position ordering or cardinality changed")
    if args.smoke:
        positions=positions[:1]

    # Import the exact research-tree engine bindings, not the historic input
    # checkout's Python runtime.
    sys.path.insert(0,str(repo))
    import torch
    from quartz.models_torch import AlphaZeroNet
    from quartz.runtime_support import NNSearchClient

    torch.set_num_threads(2)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("this experiment demands CPU-only PyTorch")
    if os.environ.get("QUARTZ_SEARCH_POLICY"):
        raise RuntimeError("opt-in policy environment could alter baseline_strict trial")
    if os.environ.get("QUARTZ_CALIBRATION_DIR"):
        raise RuntimeError("external calibration environment not permitted")
    if os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")!="1":
        raise RuntimeError("must disable async pipeline for unambiguous CPU item counts")
    if os.environ.get("QUARTZ_DISABLE_NN_CACHE")!="1":
        raise RuntimeError("must disable process-global NN cache for physical forward item counting")

    class CountingNet(torch.nn.Module):
        def __init__(self,net):
            super().__init__();self.net=net;self.n_items=0;self.n_forward_calls=0
        def forward(self,x):
            if x.device.type!="cpu":
                raise RuntimeError("GPU inference forbidden")
            self.n_items+=int(x.shape[0])
            self.n_forward_calls+=1
            return self.net(x)

    ckpt_paths=[
        "results/phase15_stage7/gomoku7/training/seed_101/gen_5.pt",
        "results/phase15_stage7/gomoku7/training/seed_102/gen_5.pt",
    ]
    ckpt_ids=["C01_gen_5","C03_gen_5"]
    seeds=[17,29,43]
    if args.smoke:
        ckpt_paths=ckpt_paths[:1];ckpt_ids=ckpt_ids[:1];seeds=seeds[:1]
    rows=[]
    metrics=[]
    run_started=time.perf_counter()
    for ck_id, relative in zip(ckpt_ids,ckpt_paths):
        checkpoint=source_root/relative
        wrapper=torch.load(checkpoint,weights_only=True,map_location="cpu")
        cfg=dict(wrapper["cfg"])
        if (cfg["board"],cfg["ch"],cfg["actions"])!=(7,17,49):
            raise RuntimeError("checkpoint/encoder shape changed")
        cfg.update({
            "_name":"gomoku7","search_profile":"baseline_strict",
            "penalty_mode":"None","n_threads":1,"iters":8,
            "batch_size":1,"batch_timeout_us":1500,"check_interval":64,
            "min_visits":64,
        })
        nn=CountingNet(AlphaZeroNet(cfg).eval())
        nn.net.load_state_dict(wrapper["model_state_dict"],strict=True)
        client=NNSearchClient(nn,cfg,torch.device("cpu"),rust_binary=str(rust_bin))
        try:
            for position in positions:
                for seed in seeds:
                    cfg["seed"]=seed
                    job={"board":position["board"],"player":int(position["player"])}
                    opts=client._rt.rust_search_options(cfg,penalty_mode="None")
                    open_req={
                        "cmd":"search_nn_multi_engine_session_open",
                        "game":"gomoku7","iters":8,
                        "diagnostic_json":True,"jobs":[job],**opts,
                    }
                    pre=nn.n_items
                    t0=time.perf_counter_ns()
                    opened=client._exchange_search_request(open_req)
                    open_ms=(time.perf_counter_ns()-t0)/1e6
                    first=verified_reply(opened,stage="resident8")
                    sid=opened["session_id"]
                    first_calls=nn.n_items-pre
                    try:
                        t1=time.perf_counter_ns()
                        step=client._exchange_search_request({
                            "cmd":"search_nn_multi_engine_session_step",
                            "session_id":sid,"iters":8,
                            "updates":[{}],"diagnostic_json":True,
                        })
                        step_ms=(time.perf_counter_ns()-t1)/1e6
                        second=verified_reply(step,stage="resident16")
                        if step["session_id"]!=sid:
                            raise RuntimeError("session ID changed across additional computation")
                        step_calls=nn.n_items-pre-first_calls
                    finally:
                        closed=client.close_search_session(sid)
                        if not closed.get("ok",False):
                            raise RuntimeError(f"resident session close not acknowledged: {closed}")

                    # A fresh engine inside the SAME server can reuse cached
                    # evaluator output. Isolate the fresh16 control in a new
                    # Rust server so its physical NN forward calls are measured.
                    client.stop()
                    client.start()
                    pre_fresh=nn.n_items
                    t2=time.perf_counter_ns()
                    fresh_payload=client._exchange_search_request({
                        **open_req,"iters":16,
                    })
                    fresh_ms=(time.perf_counter_ns()-t2)/1e6
                    fresh=verified_reply(fresh_payload,stage="fresh16")
                    try:
                        fresh_calls=nn.n_items-pre_fresh
                    finally:
                        fresh_close=client.close_search_session(fresh_payload["session_id"])
                        if not fresh_close.get("ok",False):
                            raise RuntimeError("fresh16 close not acknowledged")
                    row=validate_pair(
                        position,first,second,fresh,
                        python_calls=(first_calls,step_calls,fresh_calls),
                        wall_ms=(open_ms,step_ms,fresh_ms),
                        checkpoint=ck_id,seed=seed,
                    )
                    rows.append(row)
                    if args.smoke:
                        print(json.dumps({
                            "mode":"smoke","checkpoint":ck_id,
                            "position":position["id"],"seed":seed,
                            "root":row["root_hash"],
                            "NN_counts":[row["nn_calls_stop"],row["nn_calls_continue_total"],row["nn_calls_fresh16"]],
                            "delta_calls":row["delta_eval_calls"],
                            "fresh16_match":row["fresh16_match"],
                            "delta_reference":row["delta_ref"],
                            "step_wall_ms":step_ms,
                        },sort_keys=True),flush=True)
            metrics.append({
                "checkpoint":ck_id,"checkpoint_path":relative,
                "checkpoint_sha256":sha256(checkpoint),
                "model_items":nn.n_items,
                "model_forward_batches":nn.n_forward_calls,
            })
        finally:
            client.stop()
    total_time=time.perf_counter()-run_started
    summary=summarize_rows(rows)
    if args.smoke:
        assert len(rows)==1
        print("CPU_RESIDENT_ONE_TRIAL_SMOKE_PASS")
        return

    if not (len(rows)==72 and summary["root_position_checkpoint_pairs"]==24):
        raise RuntimeError("unexpected complete experiment cardinality")
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    data=target/"paired_rows.jsonl"
    if data.exists():
        raise RuntimeError("existing result file")
    with data.open("w") as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    atomic_new_json(target/"summary.json",summary)
    atomic_new_json(target/"model_items.json",metrics)
    source_map={
        "schema":"quartz_resident_fork02_manifest_v1",
        "study_source_commit":study_head,
        "study_source_tree":git(repo,"rev-parse","HEAD^{tree}"),
        "input_checkout_head":lock["input_checkout_head"],
        "input_lock_sha256":sha256(Path(__file__).parent/"INPUT_LOCK.json"),
        "position_lock_sha256":sha256(pfile),
        "model_checkpoints":metrics,
        "rust_binary_sha256":sha256(rust_bin),
        "rust_binary_path":str(rust_bin),
        "rust_version":subprocess.check_output(["rustc","--version"],text=True).strip(),
        "torch_version":torch.__version__,
        "torch_cuda":torch.version.cuda,
        "cpu_threads":torch.get_num_threads(),
        "complete_wall_s":total_time,
        "environment":{"QUARTZ_DISABLE_ASYNC_PIPELINE":"1","QUARTZ_DISABLE_NN_CACHE":"1","n_threads":1},
        "output_sha256":{p.name:sha256(p) for p in sorted(target.iterdir()) if p.is_file()},
        "prohibited_interpretations":[
            "not_true_minimax_regret",
            "not_Bayesian_expected_VOC",
            "not_QUARTZ_controller_gain",
            "not_GPU_throughput",
            "repeated_seed_signature_may_not_be_statistically_independent",
        ],
    }
    atomic_new_json(target/"execution_manifest.json",source_map)
    print(json.dumps({
        "status":"completed","n_trials":len(rows),
        "n_pairs":summary["root_position_checkpoint_pairs"],
        "signed_ref":summary["signed_ref"],
        "extra_evals":summary["delta_eval_calls_total"],
        "fresh16_mismatches":summary["fresh16_mismatches"],
        "all_seeds_same":summary["all_repeated_seed_trajectories_identical"],
        "output":str(target),
    },sort_keys=True))


if __name__=="__main__":
    main()
