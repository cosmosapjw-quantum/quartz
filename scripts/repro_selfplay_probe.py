#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _json_dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize_python_trace(path: Path):
    rows = _read_jsonl(path)
    summary = {
        "rows": len(rows),
        "events": {},
        "last_event": None,
        "max_read_wait_s": 0.0,
        "max_exchange_elapsed_s": 0.0,
        "probe_begin": None,
        "probe_end": None,
    }
    for row in rows:
        event = row.get("event", "?")
        summary["events"][event] = summary["events"].get(event, 0) + 1
        summary["last_event"] = event
        if event == "exchange_message":
            summary["max_read_wait_s"] = max(
                summary["max_read_wait_s"], float(row.get("read_wait_s", 0.0) or 0.0)
            )
        if event == "exchange_end":
            summary["max_exchange_elapsed_s"] = max(
                summary["max_exchange_elapsed_s"],
                float(row.get("elapsed_s", 0.0) or 0.0),
            )
        if event == "selfplay_probe_begin":
            summary["probe_begin"] = row
        if event == "selfplay_probe_end":
            summary["probe_end"] = row
    return summary


def summarize_rust_qipc(path: Path):
    rows = _read_jsonl(path)
    summary = {
        "rows": len(rows),
        "kinds": {},
        "transport": {},
        "single_calls": 0,
        "batch_requests": 0,
        "batch_mean_weighted": 0.0,
        "read_s": 0.0,
        "write_s": 0.0,
        "decode_s": 0.0,
        "encode_s": 0.0,
        "idle_wait_s": 0.0,
    }
    weighted_sum = 0.0
    weighted_n = 0
    for row in rows:
        kind = row.get("kind", "?")
        transport = row.get("transport", "?")
        summary["kinds"][kind] = summary["kinds"].get(kind, 0) + 1
        summary["transport"][transport] = summary["transport"].get(transport, 0) + 1
        summary["read_s"] += float(row.get("read_s", 0.0) or 0.0)
        summary["write_s"] += float(row.get("write_s", 0.0) or 0.0)
        summary["decode_s"] += float(row.get("decode_s", 0.0) or 0.0)
        summary["encode_s"] += float(row.get("encode_s", 0.0) or 0.0)
        summary["idle_wait_s"] += float(row.get("idle_wait_s", 0.0) or 0.0)
        if kind == "single":
            summary["single_calls"] += int(row.get("calls", 0) or 0)
        if kind == "batch":
            req = int(row.get("requests", 0) or 0)
            mb = float(row.get("mean_batch", 0.0) or 0.0)
            summary["batch_requests"] += req
            weighted_sum += mb * req
            weighted_n += req
    if weighted_n > 0:
        summary["batch_mean_weighted"] = weighted_sum / weighted_n
    return summary


def summarize_logs(output_dir: Path):
    py_trace = output_dir / "python_trace.jsonl"
    rust_qipc = output_dir / "rust_qipc.jsonl"
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    summary = {
        "python_trace": summarize_python_trace(py_trace),
        "rust_qipc": summarize_rust_qipc(rust_qipc),
        "stdout_tail": stdout_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-20:]
        if stdout_log.exists()
        else [],
        "stderr_tail": stderr_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-20:]
        if stderr_log.exists()
        else [],
    }
    return summary


def run_child(args):
    import torch

    from quartz import runtime_support
    from quartz import torch_training_runtime as torch_runtime
    from quartz.autotune_runtime import AutotuneRuntimeHooks, _run_selfplay_probe
    from quartz.models_torch import AlphaZeroNet
    from quartz.system_runtime import configure_torch_rocm_runtime, detect_hardware_spec
    from quartz.training_catalog import GAME_CONFIGS

    cfg = dict(GAME_CONFIGS[args.game])
    cfg["_name"] = args.game
    cfg["_resident_session"] = bool(args.resident_session)
    cfg["_disable_resident_session"] = bool(args.disable_resident_session)
    cfg["iters"] = int(args.iters)
    cfg["n_threads"] = int(args.n_threads)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.batch_timeout_us is not None:
        cfg["batch_timeout_us"] = int(args.batch_timeout_us)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    hw = detect_hardware_spec(device)
    configure_torch_rocm_runtime(hw)

    model = None
    model_kind = args.model_mode
    if model_kind == "real":
        model = AlphaZeroNet(cfg).to(device)
        ckpt = args.checkpoint
        if ckpt:
            from quartz.backend import load_torch_state_dict_checked

            load_torch_state_dict_checked(model, ckpt, torch, map_location=device)
        model.eval()
    elif model_kind == "dummy":

        class DummyModel:
            def eval(self):
                return self

            def predict(self, batch_np):
                import numpy as np

                b = int(batch_np.shape[0])
                probs = np.full(
                    (b, cfg["actions"]), 1.0 / max(cfg["actions"], 1), dtype=np.float32
                )
                vals = np.zeros((b,), dtype=np.float32)
                return probs, vals

        model = DummyModel()
    elif model_kind == "uniform":
        model = None
    else:
        raise ValueError(f"unknown model_mode: {model_kind}")

    runtime_hooks = AutotuneRuntimeHooks(
        alphazero_net_cls=runtime_support.AlphaZeroNet,
        run_model_batch=runtime_support.run_model_batch,
        selfplay_rust_nn_batched=torch_runtime.selfplay_rust_nn_batched,
        stall_trace=lambda *args, **kwargs: None,
        tqdm_factory=runtime_support.tqdm_factory,
    )

    t0 = time.time()
    probe = _run_selfplay_probe(
        cfg,
        model,
        device,
        args.rust_binary,
        parallel=args.parallel,
        batch_games=args.batch_games,
        n_threads=args.n_threads,
        runtime_hooks=runtime_hooks,
        concurrent=bool(args.concurrent),
        rounds=args.rounds,
        warmup=not args.no_warmup,
    )
    result = {
        "ok": True,
        "elapsed_s": time.time() - t0,
        "probe": probe,
        "config": {
            "game": args.game,
            "parallel": args.parallel,
            "batch_games": args.batch_games,
            "n_threads": args.n_threads,
            "iters": args.iters,
            "rounds": args.rounds,
            "batch_size": cfg.get("batch_size"),
            "batch_timeout_us": cfg.get("batch_timeout_us"),
            "model_mode": args.model_mode,
            "resident_session": bool(args.resident_session),
            "disable_resident_session": bool(args.disable_resident_session),
            "disable_shm": bool(args.disable_shm),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_parent(args):
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir or f"artifacts/repro_selfplay_probe/{args.game}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    py_trace_path = output_dir / "python_trace.jsonl"
    rust_qipc_path = output_dir / "rust_qipc.jsonl"
    child_result_path = output_dir / "child_result.json"
    summary_path = output_dir / "summary.json"

    env = os.environ.copy()
    env["QUARTZ_STALL_TRACE_PATH"] = str(py_trace_path)
    env["QUARTZ_RUST_QIPC_PROFILE"] = str(rust_qipc_path)
    if args.disable_shm:
        env["QUARTZ_DISABLE_QIPC_SHM"] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--game",
        args.game,
        "--parallel",
        str(args.parallel),
        "--batch-games",
        str(args.batch_games),
        "--n-threads",
        str(args.n_threads),
        "--iters",
        str(args.iters),
        "--rounds",
        str(args.rounds),
        "--device",
        args.device,
        "--model-mode",
        args.model_mode,
        "--rust-binary",
        args.rust_binary,
    ]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.batch_timeout_us is not None:
        cmd += ["--batch-timeout-us", str(args.batch_timeout_us)]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.no_warmup:
        cmd += ["--no-warmup"]
    if args.disable_shm:
        cmd += ["--disable-shm"]
    if args.concurrent:
        cmd += ["--concurrent"]
    if args.resident_session:
        cmd += ["--resident-session"]
    if args.disable_resident_session:
        cmd += ["--disable-resident-session"]

    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=args.kill_grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.wait(timeout=5)

    result = {
        "cmd": cmd,
        "timeout_s": args.timeout_s,
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "output_dir": str(output_dir),
    }
    if stdout_path.exists():
        try:
            stdout_text = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if stdout_text:
                try:
                    child_obj = (
                        json.loads(stdout_text.splitlines()[-1])
                        if stdout_text.startswith("{")
                        else None
                    )
                except Exception:
                    child_obj = None
                if child_obj is None:
                    try:
                        child_obj = json.loads(stdout_text)
                    except Exception:
                        child_obj = None
                if child_obj is not None:
                    _json_dump(child_result_path, child_obj)
                    result["child_result_path"] = str(child_result_path)
        except Exception:
            pass

    result["summary"] = summarize_logs(output_dir)
    _json_dump(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not timed_out and proc.returncode == 0 else 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce and localize self-play probe stalls with timeout+kill."
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--game", default="gomoku7")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--batch-games", type=int, default=2)
    p.add_argument("--n-threads", type=int, default=1)
    p.add_argument("--iters", type=int, default=48)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--batch-timeout-us", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-mode", choices=["real", "dummy", "uniform"], default="real")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rust-binary", default="./target/release/mcts_demo")
    p.add_argument("--timeout-s", type=int, default=90)
    p.add_argument("--kill-grace-s", type=int, default=5)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--concurrent", action="store_true", default=True)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--disable-shm", action="store_true")
    p.add_argument("--resident-session", action="store_true")
    p.add_argument("--disable-resident-session", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
