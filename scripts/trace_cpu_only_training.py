#!/usr/bin/env python3
"""Run a CPU-only QUARTZ training trace without live monitoring.

This wrapper is intentionally lightweight:
- launches `python -m cProfile -m quartz.train ... --device cpu`
- enables existing Python stall trace and Rust QIPC profiling
- writes a small summary plus top cumulative Python profile rows

It is meant for "what is still slow on CPU-only?" debugging without the
sampling overhead and phase distortion of the live monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import pstats
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


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
        "max_exchange_elapsed_s": 0.0,
        "max_read_wait_s": 0.0,
        "max_model_eval_s": 0.0,
    }
    for row in rows:
        event = str(row.get("event", "?"))
        out["events"][event] = out["events"].get(event, 0) + 1
        if event == "exchange_end":
            out["max_exchange_elapsed_s"] = max(
                out["max_exchange_elapsed_s"],
                float(row.get("elapsed_s", 0.0) or 0.0),
            )
        elif event == "exchange_message":
            out["max_read_wait_s"] = max(
                out["max_read_wait_s"],
                float(row.get("read_wait_s", 0.0) or 0.0),
            )
        elif event == "exchange_model_eval":
            out["max_model_eval_s"] = max(
                out["max_model_eval_s"],
                float(row.get("elapsed_s", 0.0) or 0.0),
            )
    out["max_exchange_elapsed_s"] = round(out["max_exchange_elapsed_s"], 6)
    out["max_read_wait_s"] = round(out["max_read_wait_s"], 6)
    out["max_model_eval_s"] = round(out["max_model_eval_s"], 6)
    return out


def summarize_rust_qipc(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out = {
        "present": bool(rows),
        "rows": len(rows),
        "single_rows": 0,
        "batch_rows": 0,
        "transport_kinds": sorted(
            {row.get("transport") for row in rows if row.get("transport")}
        ),
        "single_calls": 0,
        "batch_requests": 0,
        "io_time_s": 0.0,
        "codec_time_s": 0.0,
        "mean_batch_weighted": 0.0,
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
            batch_reqs += reqs
            batch_sum += reqs * mean_batch
    if batch_reqs > 0:
        out["mean_batch_weighted"] = batch_sum / batch_reqs
    out["io_time_s"] = round(out["io_time_s"], 6)
    out["codec_time_s"] = round(out["codec_time_s"], 6)
    out["mean_batch_weighted"] = round(out["mean_batch_weighted"], 6)
    return out


def summarize_profile(profile_path: Path, top_n: int = 40) -> list[dict[str, Any]]:
    if not profile_path.exists():
        return []
    stats = pstats.Stats(str(profile_path))
    rows = []
    for func, stat in stats.stats.items():
        cc, nc, tt, ct, callers = stat
        filename, lineno, name = func
        rows.append(
            {
                "file": filename,
                "line": lineno,
                "func": name,
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "tottime_s": float(tt),
                "cumtime_s": float(ct),
            }
        )
    rows.sort(key=lambda row: row["cumtime_s"], reverse=True)
    return rows[:top_n]


def write_profile_text(profile_rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = []
    for row in profile_rows:
        lines.append(
            f"{row['cumtime_s']:8.3f}s cum  {row['tottime_s']:8.3f}s self  "
            f"{row['total_calls']:7d} calls  {row['file']}:{row['line']}::{row['func']}"
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_override_config(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("filters", "blocks", "vh", "batch", "games"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = int(value)
    return overrides


def count_logged_iterations(train_log_path: Path) -> int:
    if not train_log_path.exists():
        return 0
    count = 0
    with train_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("_type") != "eval" and "iter" in obj:
                count += 1
    return count


def main() -> int:
    p = argparse.ArgumentParser(
        description="Trace CPU-only QUARTZ training without live monitoring."
    )
    p.add_argument("--game", default="gomoku7")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--output-dir", default="")
    p.add_argument("--filters", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--vh", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--games", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=9999)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--search-profile", choices=["quartz", "baseline"], default="quartz")
    p.add_argument("--rust-binary", default="./target/release/mcts_demo")
    p.add_argument(
        "--model-dir",
        default="",
        help="Training output dir to watch for progress (defaults to models/alphazero_<game>)",
    )
    p.add_argument("--no-retune", action="store_true")
    p.add_argument("--no-pipeline", action="store_true")
    p.add_argument("--disable-shm", action="store_true")
    p.add_argument(
        "--timeout-s",
        type=int,
        default=0,
        help="Kill the trace run after N seconds (0 disables)",
    )
    p.add_argument("--kill-grace-s", type=int, default=5)
    p.add_argument(
        "--extra-args", default="", help="Extra args appended verbatim to quartz.train"
    )
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"artifacts/cpu_only_trace/{args.game}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir or f"models/alphazero_{args.game}")
    train_log_path = model_dir / "train_log.jsonl"

    config_path = output_dir / "config_override.json"
    profile_path = output_dir / "py_profile.pstats"
    profile_top_path = output_dir / "py_profile_top.txt"
    python_trace_path = output_dir / "python_trace.jsonl"
    rust_qipc_path = output_dir / "rust_qipc.jsonl"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    summary_path = output_dir / "summary.json"

    overrides = build_override_config(args)
    if overrides:
        json_dump(config_path, overrides)

    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(profile_path),
        "-m",
        "quartz.train",
        "--game",
        args.game,
        "--iterations",
        str(args.iterations),
        "--device",
        "cpu",
        "--eval-interval",
        str(args.eval_interval),
        "--eval-games",
        str(args.eval_games),
        "--search-profile",
        args.search_profile,
        "--rust-binary",
        args.rust_binary,
    ]
    if not args.no_retune:
        cmd.append("--retune")
    if args.no_pipeline:
        cmd.append("--no-pipeline")
    if overrides:
        cmd.extend(["--config", str(config_path)])
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    env = os.environ.copy()
    env["QUARTZ_STALL_TRACE_PATH"] = str(python_trace_path)
    env["QUARTZ_RUST_QIPC_PROFILE"] = str(rust_qipc_path)
    if args.disable_shm:
        env["QUARTZ_DISABLE_QIPC_SHM"] = "1"

    timed_out = False
    t0 = time.time()
    pbar = (
        tqdm(total=max(1, int(args.iterations)), desc="cpu-trace", unit="iter")
        if tqdm is not None
        else None
    )
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_f,
        stderr_path.open("w", encoding="utf-8") as stderr_f,
    ):
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
            cwd=Path(__file__).resolve().parents[1],
            preexec_fn=os.setsid,
        )
        try:
            while True:
                rc = proc.poll()
                iter_count = count_logged_iterations(train_log_path)
                if pbar is not None:
                    pbar.n = min(max(0, iter_count), pbar.total)
                    pbar.set_postfix_str(
                        f"elapsed={time.time() - t0:.1f}s trace_rows={len(read_jsonl(python_trace_path))}"
                    )
                    pbar.refresh()
                if rc is not None:
                    break
                if (
                    args.timeout_s
                    and args.timeout_s > 0
                    and (time.time() - t0) >= args.timeout_s
                ):
                    raise subprocess.TimeoutExpired(cmd=cmd, timeout=args.timeout_s)
                time.sleep(0.5)
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
        finally:
            if pbar is not None:
                pbar.n = min(
                    max(0, count_logged_iterations(train_log_path)), pbar.total
                )
                pbar.refresh()
                pbar.close()
    duration_s = time.time() - t0

    profile_rows = summarize_profile(profile_path)
    write_profile_text(profile_rows, profile_top_path)

    summary = {
        "command": cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "duration_s": round(duration_s, 3),
        "artifacts": {
            "config_override": str(config_path) if overrides else None,
            "py_profile": str(profile_path),
            "py_profile_top": str(profile_top_path),
            "python_trace": str(python_trace_path),
            "rust_qipc": str(rust_qipc_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
        "python_trace_summary": summarize_python_trace(python_trace_path),
        "rust_qipc_summary": summarize_rust_qipc(rust_qipc_path),
        "profile_top": profile_rows[:20],
    }
    json_dump(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
