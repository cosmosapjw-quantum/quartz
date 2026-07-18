#!/usr/bin/env python3
"""Debug potentially deadlocked Rust shared evaluation runs.

This script runs the same RustNNEvaluatorEngine shared-eval path in a child
process with:

- Python stall trace
- Rust QIPC profiling
- faulthandler traceback dumps
- timeout + SIGTERM/SIGKILL cleanup

Use it to distinguish "very slow but progressing" from "stalled / deadlocked".
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def summarize_python_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "present": bool(rows),
        "rows": len(rows),
        "events": {},
        "eval_loop_rows": 0,
        "eval_game_done_rows": 0,
        "eval_stall_rows": 0,
        "last_eval_loop": None,
    }
    for row in rows:
        event = str(row.get("event", "?"))
        out["events"][event] = out["events"].get(event, 0) + 1
        if event == "eval_loop":
            out["eval_loop_rows"] += 1
            out["last_eval_loop"] = row
        elif event == "eval_game_done":
            out["eval_game_done_rows"] += 1
        elif event == "eval_stall":
            out["eval_stall_rows"] += 1
    return out


def summarize_rust_qipc(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out = {
        "present": bool(rows),
        "rows": len(rows),
        "single_rows": 0,
        "batch_rows": 0,
        "single_calls": 0,
        "batch_requests": 0,
        "io_time_s": 0.0,
        "codec_time_s": 0.0,
        "mean_batch_weighted": 0.0,
        "singleton_batches": 0,
        "low_concurrency_flushes": 0,
        "adaptive_timeout_min_us": None,
        "adaptive_timeout_max_us": None,
        "adaptive_timeout_last_us": None,
    }
    if not rows:
        return out
    batch_reqs = 0
    batch_sum = 0.0
    for row in rows:
        kind = row.get("kind")
        out["io_time_s"] += float(row.get("write_s", 0.0) or 0.0) + float(
            row.get("read_s", 0.0) or 0.0
        )
        out["codec_time_s"] += float(row.get("encode_s", 0.0) or 0.0) + float(
            row.get("decode_s", 0.0) or 0.0
        )
        if kind == "single":
            out["single_rows"] += 1
            out["single_calls"] += int(row.get("calls", 0) or 0)
        elif kind == "batch":
            reqs = int(row.get("requests", 0) or 0)
            mean_batch = float(row.get("mean_batch", 0.0) or 0.0)
            out["batch_rows"] += 1
            out["batch_requests"] += reqs
            out["singleton_batches"] += int(row.get("singleton_batches", 0) or 0)
            out["low_concurrency_flushes"] += int(
                row.get("low_concurrency_flushes", 0) or 0
            )
            batch_reqs += reqs
            batch_sum += reqs * mean_batch
            for key in (
                "adaptive_timeout_min_us",
                "adaptive_timeout_max_us",
                "adaptive_timeout_last_us",
            ):
                value = row.get(key)
                if value is not None:
                    out[key] = float(value)
    if batch_reqs > 0:
        out["mean_batch_weighted"] = batch_sum / batch_reqs
    out["io_time_s"] = round(out["io_time_s"], 6)
    out["codec_time_s"] = round(out["codec_time_s"], 6)
    out["mean_batch_weighted"] = round(out["mean_batch_weighted"], 6)
    return out


def summarize_rust_server_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "present": bool(rows),
        "rows": len(rows),
        "events": {},
        "last_row": rows[-1] if rows else None,
    }
    for row in rows:
        event = str(row.get("event", "?"))
        out["events"][event] = out["events"].get(event, 0) + 1
    return out


def build_child_cmd(args, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--game",
        args.game,
        "--num-games",
        str(args.num_games),
        "--iters",
        str(args.iters),
        "--rust-binary",
        args.rust_binary,
        "--result-json",
        str(output_dir / "child_result.json"),
        "--traceback-log",
        str(output_dir / "traceback.log"),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.search_profile:
        cmd.extend(["--search-profile", args.search_profile])
    if args.config:
        cmd.extend(["--config", args.config])
    if args.extra_args:
        cmd.extend(["--extra-args", args.extra_args])
    return cmd


def run_supervisor(args) -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir or f"artifacts/eval_deadlock_debug/{args.game}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    python_trace_path = output_dir / "python_trace.jsonl"
    rust_qipc_path = output_dir / "rust_qipc.jsonl"
    rust_server_trace_path = output_dir / "rust_server_trace.jsonl"
    summary_path = output_dir / "summary.json"
    child_cmd = build_child_cmd(args, output_dir)

    env = os.environ.copy()
    env["QUARTZ_STALL_TRACE_PATH"] = str(python_trace_path)
    env["QUARTZ_RUST_QIPC_PROFILE"] = str(rust_qipc_path)
    env["QUARTZ_RUST_SERVER_TRACE"] = str(rust_server_trace_path)
    env["QUARTZ_EVAL_STALL_TIMEOUT_S"] = str(args.stall_timeout_s)
    if args.disable_shm:
        env["QUARTZ_DISABLE_QIPC_SHM"] = "1"

    timed_out = False
    t0 = time.time()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_f,
        stderr_path.open("w", encoding="utf-8") as stderr_f,
    ):
        proc = subprocess.Popen(
            child_cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
            cwd=Path(__file__).resolve().parents[1],
            preexec_fn=os.setsid,
        )
        try:
            proc.wait(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=max(1, int(args.kill_grace_s)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.wait()

    duration_s = time.time() - t0
    child_result_path = output_dir / "child_result.json"
    child_result = None
    if child_result_path.exists():
        try:
            child_result = json.loads(child_result_path.read_text(encoding="utf-8"))
        except Exception:
            child_result = None

    summary = {
        "command": child_cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "duration_s": round(duration_s, 3),
        "artifacts": {
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "python_trace": str(python_trace_path),
            "rust_qipc": str(rust_qipc_path),
            "rust_server_trace": str(rust_server_trace_path),
            "traceback_log": str(output_dir / "traceback.log"),
            "child_result": str(child_result_path),
        },
        "python_trace_summary": summarize_python_trace(python_trace_path),
        "rust_qipc_summary": summarize_rust_qipc(rust_qipc_path),
        "rust_server_trace_summary": summarize_rust_server_trace(
            rust_server_trace_path
        ),
        "child_result": child_result,
    }
    json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(proc.returncode or 0)


def run_child(args) -> int:
    traceback_log = Path(args.traceback_log)
    traceback_log.parent.mkdir(parents=True, exist_ok=True)
    with traceback_log.open("w", encoding="utf-8") as tf:
        faulthandler.enable(tf)
        faulthandler.dump_traceback_later(
            max(1, int(args.stall_timeout_s)), repeat=True, file=tf
        )
        import torch
        from quartz.alphazero_train import (
            AlphaZeroNet,
            GAME_CONFIGS,
            RustNNEvaluatorEngine,
            apply_config_overrides,
            build_training_game_adapter,
            clone_actor_model,
            get_encoder,
            is_chess_game,
            load_torch_state_dict_checked,
            supports_rust_eval_state_machine,
        )
        from quartz.system_runtime import (
            auto_device_name,
            configure_torch_rocm_runtime,
            detect_hardware_spec,
        )

        cfg = dict(GAME_CONFIGS[args.game])
        cfg["_name"] = args.game
        try:
            cfg["_encoder"] = get_encoder(args.game)
        except Exception:
            cfg["_encoder"] = None
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = apply_config_overrides(cfg, json.load(f))
        cfg["iters"] = int(args.iters)
        cfg["_shared_eval_session"] = True
        cfg["_resident_session"] = False
        cfg["_broker_enabled"] = False
        cfg["_eval_runner_mode"] = (
            "rust_eval_state_machine"
            if supports_rust_eval_state_machine(args.game)
            else "shared_client_session"
        )
        if args.search_profile:
            cfg["search_profile"] = args.search_profile

        selected_device_name = (
            auto_device_name() if args.device == "auto" else args.device
        )
        device = torch.device(selected_device_name)
        hardware = detect_hardware_spec(device)
        configure_torch_rocm_runtime(hardware)

        model_a = AlphaZeroNet(cfg).to(device)
        model_b = AlphaZeroNet(cfg).to(device)
        if args.checkpoint and os.path.exists(args.checkpoint):
            load_torch_state_dict_checked(
                model_a, args.checkpoint, torch, map_location=device
            )
            load_torch_state_dict_checked(
                model_b, args.checkpoint, torch, map_location=device
            )
        model_a.eval()
        model_b.eval()

        eng_a = RustNNEvaluatorEngine(
            "candidate", cfg, clone_actor_model(model_a), device, args.rust_binary
        )
        eng_b = RustNNEvaluatorEngine(
            "champion", cfg, clone_actor_model(model_b), device, args.rust_binary
        )
        game_factory = lambda: build_training_game_adapter(cfg)

        t0 = time.time()
        tally = eng_a.play_match_tally_against(
            eng_b,
            game_factory,
            opening_book=[],
            num_games=int(args.num_games),
            color_swap=True,
            max_moves=500,
            seed=42,
        )
        duration_s = time.time() - t0
        faulthandler.cancel_dump_traceback_later()
        result = {
            "duration_s": round(duration_s, 3),
            "game": args.game,
            "num_games": int(args.num_games),
            "iters": int(args.iters),
            "device": str(device),
            "selected_device_name": selected_device_name,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_hip_version": getattr(torch.version, "hip", None),
            "gpu_count": int(getattr(hardware, "gpu_count", 0) or 0),
            "gpu_name": getattr(hardware, "gpu_name", ""),
            "search_profile": cfg.get("search_profile", "quartz"),
            "valid_tally": True,
            "wins": int(getattr(tally, "wins", 0)),
            "draws": int(getattr(tally, "draws", 0)),
            "losses": int(getattr(tally, "losses", 0)),
            "errors": int(getattr(tally, "errors", 0)),
            "voids": int(getattr(tally, "voids", 0)),
            "total": int(getattr(tally, "total", 0)),
        }
        json_dump(Path(args.result_json), result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Debug potentially deadlocked QUARTZ evaluation sessions."
    )
    p.add_argument("--game", default="gomoku7")
    p.add_argument("--num-games", type=int, default=8)
    p.add_argument("--iters", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--rust-binary", default="./target/release/mcts_demo")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--config", default="")
    p.add_argument("--search-profile", choices=["quartz", "baseline"], default="quartz")
    p.add_argument("--output-dir", default="")
    p.add_argument("--timeout-s", type=int, default=120)
    p.add_argument("--stall-timeout-s", type=int, default=45)
    p.add_argument("--kill-grace-s", type=int, default=5)
    p.add_argument("--disable-shm", action="store_true")
    p.add_argument("--extra-args", default="")
    p.add_argument("--child", action="store_true")
    p.add_argument("--result-json", default="")
    p.add_argument("--traceback-log", default="")
    args = p.parse_args()

    if args.child:
        return run_child(args)
    return run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
