#!/usr/bin/env python3
"""Detailed QUARTZ training monitor and artifact analyzer.

Reads an existing model directory and can optionally launch a training command
while sampling:
- system CPU / memory
- training process tree CPU / RSS / threads
- GPU usage / VRAM / clocks / power (best-effort via rocm-smi or nvidia-smi)
- training stdout events
- Rust QIPC profiling output

Writes timestamped artifacts under the chosen output directory.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import queue
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ITER_RE = re.compile(
    r"^\[\s*(?P<iter>\d+)/(?P<total>\d+)\]\s+"
    r"(?:loss=(?P<loss>[0-9.]+)\s+\(p=(?P<p_loss>[0-9.]+)\s+v=(?P<v_loss>[0-9.]+)\)\s+)?"
    r".*?replay=(?P<replay>\d+)\s+\+(?P<new_pos>\d+)"
    r"(?:\s+steps=(?P<steps_done>\d+)(?:/(?P<steps_planned>\d+))?)?\s+(?P<time_s>[0-9.]+)s$"
)


def now_ts() -> float:
    return time.time()


def parse_expected_iterations(command: list[str]) -> int | None:
    for i, token in enumerate(command):
        if token == "--iterations" and i + 1 < len(command):
            try:
                return int(command[i + 1])
            except ValueError:
                return None
        if token.startswith("--iterations="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def parse_command_settings(command: list[str]) -> dict[str, Any]:
    settings = {
        "runtime_tuner_enabled": False,
        "eval_selfplay_isolated": True,
        "search_profile": "quartz",
    }
    for idx, arg in enumerate(command):
        if arg == "--runtime-autotune":
            settings["runtime_tuner_enabled"] = True
        elif arg == "--no-eval-selfplay-isolation":
            settings["eval_selfplay_isolated"] = False
        elif arg == "--search-profile" and idx + 1 < len(command):
            settings["search_profile"] = command[idx + 1]
        elif arg.startswith("--search-profile="):
            settings["search_profile"] = arg.split("=", 1)[1]
    return settings


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def summarize_existing_artifacts(model_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
    }
    autotune = read_json(model_dir / "autotune_profile.json")
    train_log = read_jsonl(model_dir / "train_log.jsonl")
    summary["autotune_profile_present"] = autotune is not None
    summary["train_log_present"] = bool(train_log)

    if autotune:
        selfplay = autotune.get("benchmarks", {}).get("selfplay", []) or []
        train = autotune.get("benchmarks", {}).get("train", []) or []
        summary["autotune"] = {
            "version": autotune.get("version"),
            "overrides": autotune.get("overrides", {}),
            "selfplay_candidates": len(selfplay),
            "selfplay_error_count": sum(1 for row in selfplay if "error" in row),
            "selfplay_errors": sorted({row.get("error") for row in selfplay if "error" in row}),
            "train_batches": [row.get("batch") for row in train],
            "best_train_examples_per_s": max((row.get("examples_per_s", 0.0) for row in train), default=0.0),
        }

    iter_rows = [row for row in train_log if row.get("_type") != "eval"]
    eval_rows = [row for row in train_log if row.get("_type") == "eval"]
    if iter_rows:
        time_values = [float(row.get("time_s", 0.0) or 0.0) for row in iter_rows]
        pos_values = [float(row.get("pos_per_s", 0.0) or 0.0) for row in iter_rows]
        summary["train_log"] = {
            "iterations": len(iter_rows),
            "eval_rows": len(eval_rows),
            "total_logged_time_s": round(sum(time_values), 3),
            "mean_iter_time_s": round(statistics.fmean(time_values), 3),
            "mean_pos_per_s": round(statistics.fmean(pos_values), 3),
            "last_iter": iter_rows[-1],
            "last_eval": eval_rows[-1] if eval_rows else None,
        }
    return summary


def summarize_rust_qipc(profile_jsonl: Path) -> dict[str, Any]:
    rows = read_jsonl(profile_jsonl)
    out = {
        "present": bool(rows),
        "rows": len(rows),
        "single_rows": 0,
        "batch_rows": 0,
        "transport_kinds": sorted({row.get("transport") for row in rows if row.get("transport")}),
        "single_calls": 0,
        "batch_requests": 0,
        "io_time_s": 0.0,
        "codec_time_s": 0.0,
        "mean_batch_weighted": 0.0,
        "max_queue_depth": 0,
        "max_active_waiters": 0,
        "queue_wait_s": 0.0,
        "result_wait_s": 0.0,
        "flush_reason_counts": {},
        "singleton_batches": 0,
        "low_concurrency_flushes": 0,
        "adaptive_timeout_min_us": None,
        "adaptive_timeout_max_us": None,
        "adaptive_timeout_last_us": None,
    }
    if not rows:
        return out
    batch_requests = 0
    batch_weight_sum = 0.0
    for row in rows:
        kind = row.get("kind")
        if kind == "single":
            out["single_rows"] += 1
            out["single_calls"] += int(row.get("calls", 0) or 0)
            out["io_time_s"] += float(row.get("write_s", 0.0) or 0.0) + float(row.get("read_s", 0.0) or 0.0)
            out["codec_time_s"] += float(row.get("encode_s", 0.0) or 0.0) + float(row.get("decode_s", 0.0) or 0.0)
        elif kind == "batch":
            reqs = int(row.get("requests", 0) or 0)
            mean_batch = float(row.get("mean_batch", 0.0) or 0.0)
            out["batch_rows"] += 1
            out["batch_requests"] += reqs
            out["io_time_s"] += float(row.get("write_s", 0.0) or 0.0) + float(row.get("read_s", 0.0) or 0.0)
            out["codec_time_s"] += float(row.get("encode_s", 0.0) or 0.0) + float(row.get("decode_s", 0.0) or 0.0)
            batch_requests += reqs
            batch_weight_sum += reqs * mean_batch
            out["max_queue_depth"] = max(out["max_queue_depth"], int(row.get("max_queue_depth", 0) or 0))
            out["max_active_waiters"] = max(
                out["max_active_waiters"], int(row.get("max_active_waiters", 0) or 0)
            )
            out["queue_wait_s"] += float(row.get("queue_wait_s", 0.0) or 0.0)
            out["result_wait_s"] += float(row.get("result_wait_s", 0.0) or 0.0)
            out["singleton_batches"] += int(row.get("singleton_batches", 0) or 0)
            out["low_concurrency_flushes"] += int(row.get("low_concurrency_flushes", 0) or 0)
            for key in ("adaptive_timeout_min_us", "adaptive_timeout_max_us", "adaptive_timeout_last_us"):
                value = row.get(key)
                if value is not None:
                    out[key] = float(value)
            flush_counts = row.get("flush_reason_counts") or {}
            if isinstance(flush_counts, dict):
                for key, value in flush_counts.items():
                    out["flush_reason_counts"][str(key)] = out["flush_reason_counts"].get(str(key), 0) + int(value or 0)
    if batch_requests > 0:
        out["mean_batch_weighted"] = batch_weight_sum / batch_requests
    out["io_time_s"] = round(out["io_time_s"], 6)
    out["codec_time_s"] = round(out["codec_time_s"], 6)
    out["mean_batch_weighted"] = round(out["mean_batch_weighted"], 6)
    out["queue_wait_s"] = round(out["queue_wait_s"], 6)
    out["result_wait_s"] = round(out["result_wait_s"], 6)
    return out


def summarize_rust_server_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "present": bool(rows),
        "rows": len(rows),
        "events": {},
        "max_queue_depth": 0,
        "max_active_waiters": 0,
        "broker_queue_wait_s": 0.0,
        "broker_result_wait_s": 0.0,
        "broker_flush_fallback": 0,
        "broker_flush_target_batch": 0,
        "broker_flush_timeout": 0,
        "broker_flush_low_concurrency": 0,
        "broker_snapshots": 0,
        "max_tt_lock_wait_ms": 0.0,
        "tt_lock_wait_ms_sum": 0.0,
        "tt_get_or_create_calls_sum": 0,
        "tt_get_calls_sum": 0,
        "iterate_calls_sum": 0,
        "select_time_ms_sum": 0.0,
        "expand_eval_time_ms_sum": 0.0,
        "backprop_time_ms_sum": 0.0,
        "edges_lock_calls_sum": 0,
        "edges_lock_wait_ms_sum": 0.0,
        "edges_lock_max_wait_ms": 0.0,
        "async_batch_runs": 0,
        "async_batch_jobs_sum": 0,
        "async_batch_max_jobs": 0,
        "async_batch_null_results_sum": 0,
        "async_batch_null_inactive_slot_sum": 0,
        "async_batch_null_result_miss_sum": 0,
        "worker_done_events": [],
        "async_batch_max_inflight_per_job": 0,
        "selfplay_runner_done_count": 0,
        "selfplay_runner_done_duration_ms_sum": 0.0,
        "selfplay_runner_done_duration_ms_max": 0.0,
        "eval_runner_done_count": 0,
        "eval_runner_done_duration_ms_sum": 0.0,
        "eval_runner_done_duration_ms_max": 0.0,
        "eval_runner_wave_count": 0,
        "eval_runner_active_games_sum": 0,
        "eval_runner_active_games_max": 0,
        "eval_runner_share_ms_sum": 0.0,
        "eval_runner_batch_elapsed_ms_sum": 0.0,
        "runner_mode_counts": {},
        "last_row": rows[-1] if rows else None,
    }
    for row in rows:
        event = str(row.get("event", "?"))
        out["events"][event] = out["events"].get(event, 0) + 1
        broker = row.get("broker")
        if isinstance(broker, dict):
            out["max_queue_depth"] = max(out["max_queue_depth"], int(broker.get("max_queue_depth", 0) or 0))
            out["max_active_waiters"] = max(
                out["max_active_waiters"], int(broker.get("max_active_waiters", 0) or 0)
            )
            # GlobalBroker health: accumulate wait times and flush reasons
            out["broker_queue_wait_s"] = max(
                out["broker_queue_wait_s"], float(broker.get("queue_wait_s", 0.0) or 0.0)
            )
            out["broker_result_wait_s"] = max(
                out["broker_result_wait_s"], float(broker.get("result_wait_s", 0.0) or 0.0)
            )
            flush = broker.get("flush_reason_counts") or {}
            if isinstance(flush, dict):
                out["broker_flush_fallback"] = max(
                    out["broker_flush_fallback"], int(flush.get("fallback", 0) or 0)
                )
                out["broker_flush_target_batch"] = max(
                    out["broker_flush_target_batch"], int(flush.get("target_batch_reached", 0) or 0)
                )
                out["broker_flush_timeout"] = max(
                    out["broker_flush_timeout"], int(flush.get("max_wait_reached", 0) or 0)
                )
                out["broker_flush_low_concurrency"] = max(
                    out["broker_flush_low_concurrency"], int(flush.get("low_concurrency", 0) or 0)
                )
        if event == "batch_broker_snapshot":
            out["broker_snapshots"] += 1
        runner_mode = row.get("runner_mode")
        if runner_mode:
            mode_key = str(runner_mode)
            out["runner_mode_counts"][mode_key] = out["runner_mode_counts"].get(mode_key, 0) + 1
        out["max_queue_depth"] = max(out["max_queue_depth"], int(row.get("max_queue_depth", 0) or 0))
        out["max_active_waiters"] = max(
            out["max_active_waiters"], int(row.get("max_active_waiters", 0) or 0)
        )
        try:
            out["max_tt_lock_wait_ms"] = max(
                out["max_tt_lock_wait_ms"], float(row.get("tt_max_lock_wait_ms", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            pass
        try:
            out["tt_lock_wait_ms_sum"] += float(row.get("tt_lock_wait_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        out["tt_get_or_create_calls_sum"] += int(row.get("tt_get_or_create_calls", 0) or 0)
        out["tt_get_calls_sum"] += int(row.get("tt_get_calls", 0) or 0)
        out["iterate_calls_sum"] += int(row.get("iterate_calls", 0) or 0)
        try:
            out["select_time_ms_sum"] += float(row.get("select_time_ms", 0.0) or 0.0)
            out["expand_eval_time_ms_sum"] += float(row.get("expand_eval_time_ms", 0.0) or 0.0)
            out["backprop_time_ms_sum"] += float(row.get("backprop_time_ms", 0.0) or 0.0)
            out["edges_lock_wait_ms_sum"] += float(row.get("edges_lock_wait_ms", 0.0) or 0.0)
            out["edges_lock_max_wait_ms"] = max(
                out["edges_lock_max_wait_ms"], float(row.get("edges_lock_max_wait_ms", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            pass
        out["edges_lock_calls_sum"] += int(row.get("edges_lock_calls", 0) or 0)
        if event == "run_multi_async_batch_start":
            jobs = int(row.get("jobs", 0) or 0)
            out["async_batch_runs"] += 1
            out["async_batch_jobs_sum"] += jobs
            out["async_batch_max_jobs"] = max(out["async_batch_max_jobs"], jobs)
            out["async_batch_max_inflight_per_job"] = max(
                out["async_batch_max_inflight_per_job"], int(row.get("max_inflight_per_job", 0) or 0)
            )
        elif event == "run_multi_async_batch_done":
            # Support both old (null_results) and new (null_inactive_slot + null_result_miss) formats
            out["async_batch_null_results_sum"] += int(row.get("null_results", 0) or 0)
            out["async_batch_null_inactive_slot_sum"] += int(row.get("null_inactive_slot", 0) or 0)
            out["async_batch_null_result_miss_sum"] += int(row.get("null_result_miss", 0) or 0)
            # Credit system metrics (from PR 2/4)
            if row.get("credit_capacity") is not None:
                out["credit_capacity"] = max(
                    int(out.get("credit_capacity") or 0), int(row.get("credit_capacity", 0) or 0)
                )
            if row.get("peak_inflight") is not None:
                out["peak_inflight"] = max(
                    int(out.get("peak_inflight") or 0), int(row.get("peak_inflight", 0) or 0)
                )
            if row.get("worker_threads") is not None:
                out["max_worker_threads"] = max(
                    int(out.get("max_worker_threads") or 0), int(row.get("worker_threads", 0) or 0)
                )
        elif event == "selfplay_runner_done":
            dur = float(row.get("duration_ms", 0.0) or 0.0)
            out["selfplay_runner_done_count"] += 1
            out["selfplay_runner_done_duration_ms_sum"] += dur
            out["selfplay_runner_done_duration_ms_max"] = max(out["selfplay_runner_done_duration_ms_max"], dur)
        elif event == "eval_runner_done":
            dur = float(row.get("duration_ms", 0.0) or 0.0)
            out["eval_runner_done_count"] += 1
            out["eval_runner_done_duration_ms_sum"] += dur
            out["eval_runner_done_duration_ms_max"] = max(out["eval_runner_done_duration_ms_max"], dur)
        elif event == "eval_runner_wave":
            active_games = int(row.get("active_games", 0) or 0)
            out["eval_runner_wave_count"] += 1
            out["eval_runner_active_games_sum"] += active_games
            out["eval_runner_active_games_max"] = max(out["eval_runner_active_games_max"], active_games)
            out["eval_runner_share_ms_sum"] += float(row.get("share_ms", 0.0) or 0.0)
            out["eval_runner_batch_elapsed_ms_sum"] += float(row.get("batch_elapsed_ms", 0.0) or 0.0)
        elif event == "worker_done":
            out["worker_done_events"].append({
                "worker_id": int(row.get("worker_id", -1)),
                "jobs_count": int(row.get("jobs_count", 0) or 0),
                "iterations_completed": int(row.get("iterations_completed", 0) or 0),
                "idle_spins": int(row.get("idle_spins", 0) or 0),
            })
    return out


_EVAL_SR_RE = re.compile(r"sr=([0-9.]+)")
_EVAL_PV_RE = re.compile(r"p=([0-9.]+)")
_EVAL_VERDICT_RE = re.compile(r"verdict=(\w+)")
_EVAL_GAMES_RE = re.compile(r"(\d+)\s*(?:scored|games)")
_EVAL_VOID_RE = re.compile(r"voids?[=:]\s*(\d+)", re.IGNORECASE)
_EVAL_ERROR_RE = re.compile(r"errors?[=:]\s*(\d+)", re.IGNORECASE)


def _parse_eval_result_line(line: str, out: dict[str, Any]) -> None:
    """Extract promotion verdict and score rate from Eval: stdout lines."""
    if not line:
        return
    entry: dict[str, Any] = {"raw": line}
    m = _EVAL_SR_RE.search(line)
    if m:
        entry["score_rate"] = float(m.group(1))
    m = _EVAL_PV_RE.search(line)
    if m:
        entry["p_value"] = float(m.group(1))
    m = _EVAL_VERDICT_RE.search(line)
    if m:
        entry["verdict"] = m.group(1)
    m = _EVAL_GAMES_RE.search(line)
    if m:
        entry["games"] = int(m.group(1))
    m = _EVAL_VOID_RE.search(line)
    if m:
        entry["voids"] = int(m.group(1))
    m = _EVAL_ERROR_RE.search(line)
    if m:
        entry["errors"] = int(m.group(1))
    out["promotion_verdicts"].append(entry)


def summarize_events(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "present": bool(rows),
        "rows": len(rows),
        "types": {},
        "max_iter": 0,
        "max_total": 0,
        "evaluation_progress_count": 0,
        "evaluation_result_count": 0,
        "evaluation_errors": [],
        "promotion_verdicts": [],
        "training_wait_count": 0,
        "training_wait_total_s": 0.0,
        "training_wait_max_s": 0.0,
        "autotune_warmup_start_count": 0,
        "bg_pause_timeout_count": 0,
        "bg_resumed_count": 0,
        "last_event": rows[-1] if rows else None,
    }
    for row in rows:
        evt_type = str(row.get("type", "unknown"))
        out["types"][evt_type] = out["types"].get(evt_type, 0) + 1
        if evt_type == "evaluation_progress":
            out["evaluation_progress_count"] += 1
        elif evt_type == "evaluation_result":
            out["evaluation_result_count"] += 1
            _parse_eval_result_line(row.get("line", ""), out)
        elif evt_type == "evaluation_error":
            out["evaluation_errors"].append(row.get("line", ""))
        elif evt_type == "training_wait":
            out["training_wait_count"] += 1
            try:
                wait_s = float(row.get("wait_s") or 0.0)
                out["training_wait_total_s"] += wait_s
                out["training_wait_max_s"] = max(out["training_wait_max_s"], wait_s)
            except (TypeError, ValueError):
                pass
        elif evt_type == "autotune_warmup_start":
            out["autotune_warmup_start_count"] += 1
        elif evt_type == "bg_pause_timeout":
            out["bg_pause_timeout_count"] += 1
        elif evt_type == "bg_resumed":
            out["bg_resumed_count"] += 1
        try:
            out["max_iter"] = max(out["max_iter"], int(row.get("iter") or 0))
            out["max_total"] = max(out["max_total"], int(row.get("total") or 0))
        except (TypeError, ValueError):
            pass
    out["training_wait_total_s"] = round(out["training_wait_total_s"], 3)
    out["training_wait_max_s"] = round(out["training_wait_max_s"], 3)
    return out


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 3)


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 3)


def summarize_phase_samples(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {"present": bool(rows), "rows": len(rows), "phases": {}}
    phase_buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = str(row.get("phase", "unknown"))
        phase_buckets.setdefault(phase, []).append(row)
    for phase, samples in phase_buckets.items():
        cpu_vals = [float(s.get("cpu_percent_total") or 0.0) for s in samples if s.get("cpu_percent_total") is not None]
        proc_tree_rows = [s.get("proc_tree") or {} for s in samples]
        cpu_thr_vals = [
            float(pt.get("cpu_equiv_threads") or 0.0)
            for pt in proc_tree_rows
            if pt.get("cpu_equiv_threads") is not None
        ]
        native_vals = [
            float(pt.get("native_threads") or pt.get("total_threads") or 0.0)
            for pt in proc_tree_rows
            if (pt.get("native_threads") is not None or pt.get("total_threads") is not None)
        ]
        gpu_vals = []
        for s in samples:
            gpu = s.get("gpu") or {}
            gpus = gpu.get("gpus") if isinstance(gpu, dict) else None
            if isinstance(gpus, list) and gpus:
                raw = gpus[0].get("gpu_util")
                try:
                    gpu_vals.append(float(str(raw).rstrip("% ")))
                except (TypeError, ValueError):
                    pass
        out["phases"][phase] = {
            "samples": len(samples),
            "cpu_mean": _mean_or_none(cpu_vals),
            "cpu_max": _max_or_none(cpu_vals),
            "gpu_mean": _mean_or_none(gpu_vals),
            "gpu_max": _max_or_none(gpu_vals),
            "cpu_thr_mean": _mean_or_none(cpu_thr_vals),
            "cpu_thr_max": _max_or_none(cpu_thr_vals),
            "native_mean": _mean_or_none(native_vals),
            "native_max": _max_or_none(native_vals),
        }
    return out


def summarize_runner_progress(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "present": bool(rows),
        "rows": len(rows),
        "selfplay_wave_count": 0,
        "selfplay_completed_games": 0,
        "selfplay_positions_emitted": 0,
        "selfplay_replenished_slots": 0,
        "eval_wave_count": 0,
        "eval_completed_games": 0,
        "eval_positions_evaluated": 0,
        "selfplay_wave_elapsed_ms_sum": 0.0,
        "selfplay_frontier_slots_sum": 0,
        "selfplay_active_games_sum": 0,
    }
    for row in rows:
        event = str(row.get("event", ""))
        if event == "selfplay_runner_wave":
            out["selfplay_wave_count"] += 1
            out["selfplay_completed_games"] += int(row.get("newly_completed", 0) or 0)
            out["selfplay_positions_emitted"] += int(row.get("wave_positions_emitted", 0) or 0)
            out["selfplay_replenished_slots"] += int(row.get("replenished_slots", 0) or 0)
            out["selfplay_wave_elapsed_ms_sum"] += float(row.get("batch_elapsed_ms", 0.0) or 0.0)
            out["selfplay_frontier_slots_sum"] += int(row.get("frontier_slots", 0) or 0)
            out["selfplay_active_games_sum"] += int(row.get("active_games", 0) or 0)
        elif event == "eval_runner_wave":
            out["eval_wave_count"] += 1
            out["eval_completed_games"] += int(row.get("newly_completed", 0) or 0)
            out["eval_positions_evaluated"] += int(
                row.get("wave_positions_evaluated", row.get("active_games", 0)) or 0
            )
    return out


def build_bottleneck_report(
    rust_qipc_summary: dict[str, Any],
    rust_server_trace_summary: dict[str, Any],
    phase_summary: dict[str, Any],
    event_summary: dict[str, Any],
    runner_progress_summary: dict[str, Any],
) -> dict[str, Any]:
    queue_wait_s = float(rust_qipc_summary.get("queue_wait_s") or 0.0)
    result_wait_s = float(rust_qipc_summary.get("result_wait_s") or 0.0)
    codec_time_s = float(rust_qipc_summary.get("codec_time_s") or 0.0)
    io_time_s = float(rust_qipc_summary.get("io_time_s") or 0.0)
    mean_batch = float(rust_qipc_summary.get("mean_batch_weighted") or 0.0)
    tt_wait_ms = float(rust_server_trace_summary.get("tt_lock_wait_ms_sum") or 0.0)
    edge_wait_ms = float(rust_server_trace_summary.get("edges_lock_wait_ms_sum") or 0.0)
    expand_eval_ms = float(rust_server_trace_summary.get("expand_eval_time_ms_sum") or 0.0)
    select_ms = float(rust_server_trace_summary.get("select_time_ms_sum") or 0.0)
    backprop_ms = float(rust_server_trace_summary.get("backprop_time_ms_sum") or 0.0)
    phases = phase_summary.get("phases") or {}
    eval_phase = phases.get("evaluation") or {}
    training_phase = phases.get("training") or {}
    training_wait_total_s = float(event_summary.get("training_wait_total_s") or 0.0)
    training_wait_count = int(event_summary.get("training_wait_count") or 0)
    selfplay_wave_count = int(runner_progress_summary.get("selfplay_wave_count") or 0)
    selfplay_positions_emitted = int(runner_progress_summary.get("selfplay_positions_emitted") or 0)
    async_batch_runs = int(rust_server_trace_summary.get("async_batch_runs") or 0)
    async_batch_jobs_sum = int(rust_server_trace_summary.get("async_batch_jobs_sum") or 0)
    async_batch_null_results = int(rust_server_trace_summary.get("async_batch_null_results_sum") or 0)
    async_batch_null_result_miss = int(rust_server_trace_summary.get("async_batch_null_result_miss_sum") or 0)
    async_batch_null_inactive = int(rust_server_trace_summary.get("async_batch_null_inactive_slot_sum") or 0)
    max_wait_reached = int((rust_qipc_summary.get("flush_reason_counts") or {}).get("max_wait_reached") or 0)
    target_batch_reached = int((rust_qipc_summary.get("flush_reason_counts") or {}).get("target_batch_reached") or 0)
    low_concurrency_flushes = int((rust_qipc_summary.get("flush_reason_counts") or {}).get("low_concurrency") or 0)
    singleton_batches = int(rust_qipc_summary.get("singleton_batches") or 0)
    eval_wave_count = int(rust_server_trace_summary.get("eval_runner_wave_count") or 0)
    eval_active_games_sum = int(rust_server_trace_summary.get("eval_runner_active_games_sum") or 0)
    eval_active_games_max = int(rust_server_trace_summary.get("eval_runner_active_games_max") or 0)
    eval_share_ms_sum = float(rust_server_trace_summary.get("eval_runner_share_ms_sum") or 0.0)
    eval_batch_elapsed_ms_sum = float(rust_server_trace_summary.get("eval_runner_batch_elapsed_ms_sum") or 0.0)
    mean_eval_active_games = eval_active_games_sum / max(eval_wave_count, 1)
    batch_vs_active_ratio = mean_batch / max(mean_eval_active_games, 1e-9)

    # GlobalBroker health from rust_server_trace
    broker_queue_wait_s = float(rust_server_trace_summary.get("broker_queue_wait_s") or 0.0)
    broker_result_wait_s = float(rust_server_trace_summary.get("broker_result_wait_s") or 0.0)
    broker_fallback = int(rust_server_trace_summary.get("broker_flush_fallback") or 0)
    broker_snapshots = int(rust_server_trace_summary.get("broker_snapshots") or 0)
    broker_low_concurrency = int(rust_server_trace_summary.get("broker_flush_low_concurrency") or 0)

    # Promotion audit from event summary
    promotion_verdicts = event_summary.get("promotion_verdicts") or []
    eval_errors = event_summary.get("evaluation_errors") or []

    findings: list[str] = []

    # --- GlobalBroker findings ---
    if broker_result_wait_s > max(broker_queue_wait_s * 8.0, 1.0):
        findings.append("broker_result_wait_high")
    if broker_fallback > 0:
        findings.append("broker_fallback_detected")
    if broker_snapshots == 0 and async_batch_runs > 0:
        findings.append("broker_snapshots_missing")
    if broker_low_concurrency > 0 or low_concurrency_flushes > 0:
        findings.append("broker_low_concurrency_mode")

    # --- Worker imbalance detection (per large-wave only) ---
    # Filter to large-wave workers (>= 1000 iterations) to avoid mixing
    # selfplay waves (small, 2 jobs) with eval waves (large, 200 jobs).
    worker_events = rust_server_trace_summary.get("worker_done_events") or []
    worker_completions = [w.get("iterations_completed", 0) for w in worker_events if w.get("iterations_completed", 0) > 0]
    large_wave_completions = [c for c in worker_completions if c >= 1000]
    worker_imbalance_ratio = 1.0
    if len(large_wave_completions) >= 2:
        worker_imbalance_ratio = max(large_wave_completions) / max(min(large_wave_completions), 1)
        if worker_imbalance_ratio > 1.20:
            findings.append("worker_imbalance_high")

    # --- Promotion gate audit ---
    for pv in promotion_verdicts:
        voids = int(pv.get("voids") or 0)
        errs = int(pv.get("errors") or 0)
        if voids > 0 or errs > 0:
            findings.append("eval_voids_or_errors_present")
            break
    if eval_errors:
        findings.append("terminal_void_detected")

    if result_wait_s > max(queue_wait_s * 2.0, 1.0):
        findings.append("result_wait_dominant")
    if io_time_s > max(codec_time_s * 10.0, 1.0):
        findings.append("transport_wait_dominant")
    if mean_batch >= 8.0:
        findings.append("batch_fill_not_primary_issue")
    if (eval_phase.get("cpu_thr_mean") or 0.0) <= 2.0 and (eval_phase.get("native_mean") or 0.0) >= 8.0:
        findings.append("many_native_threads_low_effective_cpu")
    if tt_wait_ms < 50.0:
        findings.append("tt_lock_not_primary")
    if edge_wait_ms < 50.0:
        findings.append("edge_lock_not_primary")
    if expand_eval_ms > max(select_ms + backprop_ms, 1.0):
        findings.append("expand_eval_phase_heaviest")
    if int(event_summary.get("evaluation_progress_count") or 0) == 0 and int((event_summary.get("types") or {}).get("evaluation_start", 0) or 0) > 0:
        findings.append("evaluation_progress_missing")
    if training_wait_total_s >= 60.0:
        findings.append("replay_starvation_visible")
    if selfplay_wave_count > 0 and selfplay_positions_emitted == 0:
        findings.append("selfplay_wave_no_positions")
    if async_batch_runs > 0 and async_batch_null_result_miss > 0:
        findings.append("async_batch_null_results_present")
    if async_batch_runs > 0:
        avg_jobs = async_batch_jobs_sum / max(async_batch_runs, 1)
        if avg_jobs < 4.0:
            findings.append("async_batch_underfed_jobs")
    if max_wait_reached > target_batch_reached * 2 and max_wait_reached > 0:
        findings.append("flush_timeout_dominant")
    if eval_wave_count > 0 and mean_eval_active_games <= 3.0 and batch_vs_active_ratio < 0.85:
        findings.append("eval_batch_headroom_underfilled")
    if singleton_batches > 0 and singleton_batches >= max(1, rust_qipc_summary.get("batch_rows", 0)):
        findings.append("singleton_batches_dominant")

    primary = "undetermined"
    if "result_wait_dominant" in findings and "transport_wait_dominant" in findings:
        primary = "sync_eval_handshake_orchestration"
    elif "expand_eval_phase_heaviest" in findings:
        primary = "expand_eval_path"
    elif "tt_lock_not_primary" in findings and "edge_lock_not_primary" in findings:
        primary = "not_tt_or_edge_lock"

    return {
        "primary_bottleneck": primary,
        "findings": findings,
        "ratios": {
            "result_vs_queue_wait": round(result_wait_s / max(queue_wait_s, 1e-9), 3),
            "io_vs_codec": round(io_time_s / max(codec_time_s, 1e-9), 3),
            "batch_vs_active_games": round(batch_vs_active_ratio, 3),
        },
        "training_wait_summary": {
            "count": training_wait_count,
            "total_s": round(training_wait_total_s, 3),
            "max_s": round(float(event_summary.get("training_wait_max_s") or 0.0), 3),
        },
        "runner_progress_summary": {
            "selfplay_wave_count": selfplay_wave_count,
            "selfplay_completed_games": int(runner_progress_summary.get("selfplay_completed_games") or 0),
            "selfplay_positions_emitted": selfplay_positions_emitted,
            "selfplay_replenished_slots": int(runner_progress_summary.get("selfplay_replenished_slots") or 0),
            "eval_wave_count": int(runner_progress_summary.get("eval_wave_count") or 0),
            "eval_completed_games": int(runner_progress_summary.get("eval_completed_games") or 0),
            "eval_positions_evaluated": int(runner_progress_summary.get("eval_positions_evaluated") or 0),
            "selfplay_wave_elapsed_ms_sum": round(float(runner_progress_summary.get("selfplay_wave_elapsed_ms_sum") or 0.0), 3),
            "selfplay_frontier_slots_sum": int(runner_progress_summary.get("selfplay_frontier_slots_sum") or 0),
            "selfplay_active_games_sum": int(runner_progress_summary.get("selfplay_active_games_sum") or 0),
        },
        "async_batch_summary": {
            "runs": async_batch_runs,
            "jobs_sum": async_batch_jobs_sum,
            "null_results_sum": async_batch_null_results,
            "null_inactive_slot_sum": async_batch_null_inactive,
            "null_result_miss_sum": async_batch_null_result_miss,
            "max_inflight_per_job": int(rust_server_trace_summary.get("async_batch_max_inflight_per_job") or 0),
            "max_jobs": int(rust_server_trace_summary.get("async_batch_max_jobs") or 0),
            "flush_max_wait_reached": max_wait_reached,
            "flush_target_batch_reached": target_batch_reached,
        },
        "broker_health": {
            "snapshots": broker_snapshots,
            "queue_wait_s": round(broker_queue_wait_s, 6),
            "result_wait_s": round(broker_result_wait_s, 6),
            "result_vs_queue_ratio": round(broker_result_wait_s / max(broker_queue_wait_s, 1e-9), 3),
            "fallback_count": broker_fallback,
            "flush_target_batch": int(rust_server_trace_summary.get("broker_flush_target_batch") or 0),
            "flush_timeout": int(rust_server_trace_summary.get("broker_flush_timeout") or 0),
            "flush_low_concurrency": broker_low_concurrency,
            "singleton_batches": singleton_batches,
            "adaptive_timeout_min_us": rust_qipc_summary.get("adaptive_timeout_min_us"),
            "adaptive_timeout_max_us": rust_qipc_summary.get("adaptive_timeout_max_us"),
            "adaptive_timeout_last_us": rust_qipc_summary.get("adaptive_timeout_last_us"),
        },
        "eval_headroom": {
            "wave_count": eval_wave_count,
            "active_games_mean": round(mean_eval_active_games, 3),
            "active_games_max": eval_active_games_max,
            "mean_batch_weighted": mean_batch,
            "batch_vs_active_ratio": round(batch_vs_active_ratio, 3),
            "share_ms_mean": round(eval_share_ms_sum / max(eval_wave_count, 1), 3),
            "batch_elapsed_ms_mean": round(eval_batch_elapsed_ms_sum / max(eval_wave_count, 1), 3),
            "low_concurrency_flushes": low_concurrency_flushes,
        },
        "promotion_audit": {
            "verdicts_captured": len(promotion_verdicts),
            "verdicts": promotion_verdicts[-5:] if promotion_verdicts else [],
            "eval_error_lines": eval_errors[-5:] if eval_errors else [],
        },
        "worker_pool": {
            "worker_count": len(worker_completions),
            "per_worker_completed": worker_completions,
            "imbalance_ratio": round(worker_imbalance_ratio, 3),
        },
        "phase_cpu_thr_mean": {
            "training": training_phase.get("cpu_thr_mean"),
            "evaluation": eval_phase.get("cpu_thr_mean"),
        },
    }


def parse_rocm_smi_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    try:
        data = json.loads(raw)
    except Exception:
        return {"raw": raw.strip()}
    gpu_rows = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        row = {"card": key}
        for k, v in value.items():
            lk = k.lower()
            if "average_gfx_activity" in lk:
                row["average_gfx_activity"] = v
                if "gpu_util" not in row:
                    row["gpu_util"] = v
            elif "gpu use" in lk or "gfx activity" in lk:
                row["gpu_util"] = v
            elif "vram" in lk and ("used" in lk or "use" in lk or "allocated" in lk):
                row["vram"] = v
            elif "power" in lk:
                row["power"] = v
            elif "current_gfxclk" in lk:
                row["gfx_clock_mhz"] = v
            elif "current_uclk" in lk or "uclk" in lk:
                row["mem_clock_mhz"] = v
            elif "sclk" in lk or "mclk" in lk:
                row[k] = v
        gpu_rows.append(row)
    return {"gpus": gpu_rows, "raw": data}


def sample_gpu() -> dict[str, Any] | None:
    if shutil.which("rocm-smi"):
        # Query both instant snapshot and averaged metrics, merge results.
        # Instant "GPU use (%)" can read 0 between short inference bursts;
        # "average_gfx_activity (%)" from --showmetrics is a rolling average
        # that better captures bursty GPU workloads.
        commands = [
            ["rocm-smi", "-u", "--showmemuse", "--showpower", "--showclkfrq", "--json"],
            ["rocm-smi", "--showmetrics", "--json"],
        ]
        merged: dict[str, Any] = {"tool": "rocm-smi"}
        merged_gpus: dict[str, dict[str, Any]] = {}
        errors = []
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                payload = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                if "{" in payload:
                    parsed = parse_rocm_smi_json(payload)
                    for gpu_row in (parsed.get("gpus") or []):
                        card = gpu_row.get("card", "?")
                        merged_gpus.setdefault(card, {"card": card}).update(gpu_row)
                elif payload.strip():
                    errors.append({"command": cmd[1:], "output": payload.strip()})
            except Exception as exc:  # pragma: no cover
                errors.append({"command": cmd[1:], "error": str(exc)})
        if merged_gpus:
            gpu_list = list(merged_gpus.values())
            # Prefer average_gfx_activity over instant gpu_util when both exist
            for g in gpu_list:
                avg = g.get("average_gfx_activity")
                instant = g.get("gpu_util")
                if avg is not None and instant is not None:
                    try:
                        avg_f = float(str(avg).rstrip("%"))
                        inst_f = float(str(instant).rstrip("%"))
                        # Use whichever is higher — the average captures bursty load
                        g["gpu_util"] = max(avg_f, inst_f)
                        g["gpu_util_instant"] = inst_f
                        g["gpu_util_average"] = avg_f
                    except (TypeError, ValueError):
                        pass
                elif avg is not None and instant is None:
                    try:
                        g["gpu_util"] = float(str(avg).rstrip("%"))
                    except (TypeError, ValueError):
                        pass
            merged["gpus"] = gpu_list
            return merged
        if errors:
            return {"tool": "rocm-smi", "error": "no_parseable_json", "attempts": errors}
        return {"tool": "rocm-smi", "error": "no_parseable_json", "attempts": errors}
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            rows = []
            for line in proc.stdout.strip().splitlines():
                idx, gpu_u, mem_u, used, total, power, sm = [part.strip() for part in line.split(",")]
                rows.append({
                    "card": idx,
                    "gpu_util": float(gpu_u),
                    "mem_util": float(mem_u),
                    "vram_used_mb": float(used),
                    "vram_total_mb": float(total),
                    "power_w": float(power),
                    "sm_clock_mhz": float(sm),
                })
            return {"tool": "nvidia-smi", "gpus": rows}
        except Exception as exc:  # pragma: no cover
            return {"tool": "nvidia-smi", "error": str(exc)}
    return None


def sample_process_tree(root_pid: int, cpu_time_cache: dict[int, tuple[float, float]] | None = None) -> dict[str, Any]:
    if psutil is None:
        return {"root_pid": root_pid, "psutil": False}
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return {"root_pid": root_pid, "missing": True}
    now = now_ts()
    procs = [root] + root.children(recursive=True)
    rows = []
    total_cpu = 0.0
    total_rss = 0
    total_threads = 0
    cpu_sample_ready = False
    for proc in procs:
        try:
            info = proc.as_dict(
                attrs=["pid", "name", "cmdline", "status", "num_threads", "memory_info", "cpu_times"]
            )
            cpu = None
            cpu_times = info.get("cpu_times")
            cpu_total_time = float(getattr(cpu_times, "user", 0.0) or 0.0) + float(
                getattr(cpu_times, "system", 0.0) or 0.0
            )
            if cpu_time_cache is not None:
                prev = cpu_time_cache.get(info["pid"])
                cpu_time_cache[info["pid"]] = (cpu_total_time, now)
                if prev is not None:
                    prev_total, prev_ts = prev
                    elapsed = max(1e-6, now - prev_ts)
                    cpu = max(0.0, ((cpu_total_time - prev_total) / elapsed) * 100.0)
                    cpu_sample_ready = True
            rss = int(getattr(info.get("memory_info"), "rss", 0) or 0)
            threads = int(info.get("num_threads") or 0)
            row = {
                "pid": info["pid"],
                "name": info.get("name"),
                "cmdline": info.get("cmdline") or [],
                "status": info.get("status"),
                "cpu_percent": cpu,
                "rss_mb": round(rss / (1024 * 1024), 3),
                "threads": threads,
            }
            rows.append(row)
            total_cpu += float(cpu or 0.0)
            total_rss += rss
            total_threads += threads
        except psutil.Error:
            continue
    if cpu_time_cache is not None:
        live_pids = {row["pid"] for row in rows}
        stale = [pid for pid in cpu_time_cache.keys() if pid not in live_pids]
        for pid in stale:
            cpu_time_cache.pop(pid, None)
    return {
        "root_pid": root_pid,
        "processes": rows,
        "total_cpu_percent": round(total_cpu, 3) if cpu_sample_ready else None,
        "total_rss_mb": round(total_rss / (1024 * 1024), 3),
        "total_threads": total_threads,
        "native_threads": total_threads,
        "cpu_sample_ready": cpu_sample_ready,
    }


def sample_system(root_pid: int, cpu_time_cache: dict[int, tuple[float, float]] | None = None) -> dict[str, Any]:
    sample = {"ts": now_ts()}
    if psutil is not None:
        sample["logical_cpus"] = max(1, int(psutil.cpu_count(logical=True) or 1))
        sample["cpu_percent_total"] = psutil.cpu_percent(interval=None)
        sample["cpu_percent_per_cpu"] = psutil.cpu_percent(interval=None, percpu=True)
        vm = psutil.virtual_memory()
        sample["memory"] = {
            "used_mb": round(vm.used / (1024 * 1024), 3),
            "available_mb": round(vm.available / (1024 * 1024), 3),
            "percent": vm.percent,
        }
    else:
        sample["cpu_percent_total"] = None
    sample["proc_tree"] = sample_process_tree(root_pid, cpu_time_cache=cpu_time_cache)
    proc_tree = sample.get("proc_tree") or {}
    if psutil is not None and proc_tree and proc_tree.get("cpu_sample_ready"):
        logical_cpus = int(sample.get("logical_cpus") or 1)
        cpu_equiv = max(0.0, float(proc_tree.get("total_cpu_percent") or 0.0) / 100.0)
        proc_tree["cpu_equiv_threads"] = round(min(cpu_equiv, float(logical_cpus)), 3)
        proc_tree["cpu_equiv_threads_ceil"] = min(logical_cpus, max(0, math.ceil(cpu_equiv)))
        sample["proc_tree"] = proc_tree
    sample["gpu"] = sample_gpu()
    return sample


@dataclass
class LiveRunArtifacts:
    output_dir: Path
    stdout_log: Path
    samples_jsonl: Path
    events_jsonl: Path
    rust_qipc_jsonl: Path
    rust_server_trace_jsonl: Path
    summary_json: Path


def build_artifacts(output_dir: Path) -> LiveRunArtifacts:
    return LiveRunArtifacts(
        output_dir=output_dir,
        stdout_log=output_dir / "stdout.log",
        samples_jsonl=output_dir / "samples.jsonl",
        events_jsonl=output_dir / "events.jsonl",
        rust_qipc_jsonl=output_dir / "rust_qipc.jsonl",
        rust_server_trace_jsonl=output_dir / "rust_server_trace.jsonl",
        summary_json=output_dir / "summary.json",
    )


def parse_stdout_event(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    evt: dict[str, Any] = {"line": stripped}
    if stripped.startswith("Auto-tuned:") or "Auto-tuned:" in stripped:
        evt["type"] = "autotune_selected"
        return evt
    if "Auto-tune profile: running warmup benchmark" in stripped:
        evt["type"] = "autotune_warmup_start"
        return evt
    if stripped.startswith("  [AutoTune]"):
        evt["type"] = "autotune_runtime_adjust"
        return evt
    if stripped.startswith("[BG] WARN: self-play pause timed out"):
        evt["type"] = "bg_pause_timeout"
        return evt
    if stripped.startswith("[BG] Self-play resumed after evaluation"):
        evt["type"] = "bg_resumed"
        return evt
    if "Filling replay:" in stripped:
        evt["type"] = "replay_fill"
        return evt
    if "Evaluating gen_" in stripped:
        evt["type"] = "evaluation_start"
        return evt
    if stripped.startswith("EvalProgress:"):
        evt["type"] = "evaluation_progress"
        return evt
    if stripped.startswith("Eval:"):
        evt["type"] = "evaluation_result"
        return evt
    if "[EvalInvalid]" in stripped or "[EvalError]" in stripped:
        evt["type"] = "evaluation_error"
        return evt
    if "[DutyCycle]" in stripped:
        evt["type"] = "duty_cycle"
        # Parse: cycles=N read=Xs(P%) collect=Xs(P%) model=Xs(P%) write=Xs(P%) total=Xs
        for key in ("cycles", "read", "collect", "model", "write", "total"):
            m = re.search(rf"{key}=([0-9.]+)", stripped)
            if m:
                evt[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        return evt
    wait_match = re.search(r"waiting for self-play: .*? ([0-9]+(?:\\.[0-9]+)?)s$", stripped)
    if wait_match:
        evt["type"] = "training_wait"
        evt["wait_s"] = float(wait_match.group(1))
        return evt
    match = ITER_RE.match(stripped)
    if match:
        evt["type"] = "iteration"
        evt.update({k: match.group(k) for k in match.groupdict()})
        return evt
    return {"type": "stdout", "line": stripped}


def enqueue_stdout(proc: subprocess.Popen[str], out_q: queue.Queue[str], stdout_log: Path) -> None:
    with stdout_log.open("w", encoding="utf-8") as log_f:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            out_q.put(line.rstrip("\n"))


def run_live_monitor(command: list[str], model_dir: Path, output_dir: Path, interval_s: float) -> dict[str, Any]:
    artifacts = build_artifacts(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = parse_command_settings(command)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["QUARTZ_RUST_QIPC_PROFILE"] = str(artifacts.rust_qipc_jsonl)
    env["QUARTZ_RUST_SERVER_TRACE"] = str(artifacts.rust_server_trace_jsonl)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    out_q: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=enqueue_stdout,
        args=(proc, out_q, artifacts.stdout_log),
        daemon=True,
    )
    reader.start()

    if psutil is not None:
        try:
            root = psutil.Process(proc.pid)
            root.cpu_percent(interval=None)
            for child in root.children(recursive=True):
                child.cpu_percent(interval=None)
        except psutil.Error:
            pass

    phase = "startup"
    started_at = now_ts()
    event_count = 0
    last_iter = 0
    last_eval_progress = None
    expected_iters = parse_expected_iterations(command)
    cpu_time_cache: dict[int, tuple[float, float]] = {}
    pbar = None
    if tqdm is not None and sys.stderr.isatty():
        pbar = tqdm(total=expected_iters, desc="train-monitor", unit="iter", dynamic_ncols=True)
        pbar.set_postfix_str("phase=startup")
    while proc.poll() is None:
        while True:
            try:
                line = out_q.get_nowait()
            except queue.Empty:
                break
            evt = parse_stdout_event(line)
            if evt:
                event_count += 1
                evt["ts"] = now_ts()
                evt["phase"] = phase
                if evt["type"] == "replay_fill":
                    phase = "replay_fill"
                elif evt["type"] == "iteration":
                    phase = "training"
                    try:
                        iter_idx = int(evt.get("iter") or 0)
                        total_idx = int(evt.get("total") or 0)
                    except ValueError:
                        iter_idx = 0
                        total_idx = 0
                    if pbar is not None:
                        if (pbar.total is None or pbar.total == 0) and total_idx > 0:
                            pbar.total = total_idx
                        if iter_idx > last_iter:
                            pbar.update(iter_idx - last_iter)
                            last_iter = iter_idx
                        pbar.set_postfix_str(
                            f"phase=training loss={evt.get('loss','?')} "
                            f"replay={evt.get('replay','?')} +{evt.get('new_pos','?')}"
                        )
                elif evt["type"] == "evaluation_start":
                    phase = "evaluation"
                    last_eval_progress = None
                    if pbar is not None:
                        pbar.set_postfix_str("phase=evaluation")
                elif evt["type"] == "evaluation_progress":
                    phase = "evaluation"
                    last_eval_progress = evt.get("line")
                    if pbar is not None:
                        pbar.set_postfix_str(last_eval_progress.replace("  ", "", 1))
                elif evt["type"] == "evaluation_result":
                    phase = "training_wait"
                    last_eval_progress = None
                    if pbar is not None:
                        pbar.set_postfix_str("phase=training_wait")
                elif evt["type"] == "training_wait":
                    phase = "training_wait"
                    if pbar is not None:
                        pbar.set_postfix_str(
                            f"phase=training_wait replay_pause={evt.get('wait_s', '?')}s"
                        )
                jsonl_append(artifacts.events_jsonl, evt)
        sample = sample_system(proc.pid, cpu_time_cache=cpu_time_cache)
        sample["phase"] = phase
        jsonl_append(artifacts.samples_jsonl, sample)
        if pbar is not None:
            gpu = sample.get("gpu") or {}
            gpu_util = "n/a"
            gpus = gpu.get("gpus") if isinstance(gpu, dict) else None
            if gpus and isinstance(gpus, list) and gpus:
                gpu_util = gpus[0].get("gpu_util", "n/a")
            cpu_total = sample.get("cpu_percent_total", "n/a")
            proc_tree = sample.get("proc_tree") or {}
            cpu_threads = proc_tree.get("cpu_equiv_threads_ceil", "n/a")
            native_threads = proc_tree.get("native_threads", proc_tree.get("total_threads", "n/a"))
            status = (
                last_eval_progress.replace("  ", "", 1)
                if phase == "evaluation" and last_eval_progress
                else f"phase={phase} cpu={cpu_total} gpu={gpu_util} "
                     f"cpu_thr={cpu_threads} native={native_threads}"
            )
            pbar.set_postfix_str(status)
            pbar.refresh()
        time.sleep(interval_s)

    reader.join(timeout=2.0)
    while True:
        try:
            line = out_q.get_nowait()
        except queue.Empty:
            break
        evt = parse_stdout_event(line)
        if evt:
            event_count += 1
            evt["ts"] = now_ts()
            evt["phase"] = phase
            if evt["type"] == "iteration":
                phase = "training"
                try:
                    iter_idx = int(evt.get("iter") or 0)
                    total_idx = int(evt.get("total") or 0)
                except ValueError:
                    iter_idx = 0
                    total_idx = 0
                if pbar is not None:
                    if (pbar.total is None or pbar.total == 0) and total_idx > 0:
                        pbar.total = total_idx
                    if iter_idx > last_iter:
                        pbar.update(iter_idx - last_iter)
                        last_iter = iter_idx
            elif evt["type"] == "evaluation_start":
                phase = "evaluation"
                last_eval_progress = None
            elif evt["type"] == "evaluation_progress":
                phase = "evaluation"
                last_eval_progress = evt.get("line")
            elif evt["type"] == "evaluation_result":
                phase = "training_wait"
                last_eval_progress = None
            elif evt["type"] == "training_wait":
                phase = "training_wait"
            jsonl_append(artifacts.events_jsonl, evt)

    result = {
        "command": command,
        "settings": settings,
        "returncode": proc.returncode,
        "duration_s": round(now_ts() - started_at, 3),
        "event_count": event_count,
        "artifacts": {
            "stdout_log": str(artifacts.stdout_log),
            "samples_jsonl": str(artifacts.samples_jsonl),
            "events_jsonl": str(artifacts.events_jsonl),
            "rust_qipc_jsonl": str(artifacts.rust_qipc_jsonl),
            "rust_server_trace_jsonl": str(artifacts.rust_server_trace_jsonl),
        },
        "model_artifacts": summarize_existing_artifacts(model_dir),
        "rust_qipc_summary": summarize_rust_qipc(artifacts.rust_qipc_jsonl),
        "rust_server_trace_summary": summarize_rust_server_trace(artifacts.rust_server_trace_jsonl),
        "event_summary": summarize_events(artifacts.events_jsonl),
        "phase_summary": summarize_phase_samples(artifacts.samples_jsonl),
        "runner_progress_summary": summarize_runner_progress(artifacts.rust_server_trace_jsonl),
    }
    result["bottleneck_report"] = build_bottleneck_report(
        result["rust_qipc_summary"],
        result["rust_server_trace_summary"],
        result["phase_summary"],
        result["event_summary"],
        result["runner_progress_summary"],
    )
    if pbar is not None:
        pbar.set_postfix_str(f"done rc={proc.returncode} phase={phase}")
        pbar.close()
    json_dump(artifacts.summary_json, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/alphazero_gomoku7")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--run", default="")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts/runtime_monitor") / f"{model_dir.name}_{stamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "analysis_only": not bool(args.run.strip()),
        "existing_model_artifacts": summarize_existing_artifacts(model_dir),
    }

    if args.run.strip():
        command = shlex.split(args.run)
        summary["live_run"] = run_live_monitor(command, model_dir, output_dir, args.interval_s)
    else:
        # Analysis-only: read all available artifact files from output_dir
        artifacts = build_artifacts(output_dir)
        rust_qipc_s = summarize_rust_qipc(artifacts.rust_qipc_jsonl)
        rust_trace_s = summarize_rust_server_trace(artifacts.rust_server_trace_jsonl)
        event_s = summarize_events(artifacts.events_jsonl)
        phase_s = summarize_phase_samples(artifacts.samples_jsonl)
        runner_s = summarize_runner_progress(artifacts.rust_server_trace_jsonl)
        any_data = any([
            rust_qipc_s.get("present"),
            rust_trace_s.get("present"),
            event_s.get("present"),
            phase_s.get("present"),
            runner_s.get("present"),
        ])
        if any_data:
            summary["rust_qipc_summary"] = rust_qipc_s
            summary["rust_server_trace_summary"] = rust_trace_s
            summary["event_summary"] = event_s
            summary["phase_summary"] = phase_s
            summary["runner_progress_summary"] = runner_s
            summary["bottleneck_report"] = build_bottleneck_report(
                rust_qipc_s, rust_trace_s, phase_s, event_s, runner_s,
            )
        json_dump(output_dir / "summary.json", summary)

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
