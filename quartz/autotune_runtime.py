"""Autotune and throughput benchmarking helpers for the training runtime."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


def _torch_module():
    import torch

    return torch


def _torch_functional():
    import torch.nn.functional as F

    return F


AUTOTUNE_PROFILE_VERSION = 16


@dataclass(frozen=True)
class AutotuneRuntimeHooks:
    alphazero_net_cls: object
    run_model_batch: object
    selfplay_rust_nn_batched: object
    stall_trace: object
    tqdm_factory: object


def _round_down_to_multiple(value, multiple):
    if multiple <= 1:
        return value
    return max(multiple, (value // multiple) * multiple)


def _round_up_to_multiple(value, multiple):
    if multiple <= 1:
        return value
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _autotune_parallel_limit(hw, concurrent):
    logical = max(1, int(hw.logical_cpus))
    physical = max(1, int(hw.physical_cpus))
    upper = max(1, min(physical, logical))
    if concurrent and hw.device_kind != "cpu" and hw.gpu_vram_mb > 0:
        ipc_cap = max(4, (physical + 1) // 2)
        upper = min(upper, ipc_cap)
    return max(1, upper)


def _autotune_thread_capacity(hw, parallel):
    logical = max(1, int(hw.logical_cpus))
    capacity_basis = logical
    if hw.device_kind != "cpu" and hw.gpu_vram_mb > 0:
        gpu_cap = 6 if parallel <= 2 else 4 if parallel <= 4 else 3
    else:
        gpu_cap = None
        capacity_basis = max(1, int(hw.physical_cpus))
    if capacity_basis <= 8:
        cap = max(1, min(8, capacity_basis // max(1, int(parallel)) or 1))
        cap = min(cap, logical)
        return min(cap, gpu_cap) if gpu_cap is not None else cap
    reserve = 1 if capacity_basis >= 12 else 0
    usable = max(1, capacity_basis - reserve)
    cap = max(1, min(12, usable // max(1, int(parallel))))
    cap = min(cap, logical)
    return min(cap, gpu_cap) if gpu_cap is not None else cap


def _autotune_thread_candidates(hw, parallel, hinted=None):
    cap = _autotune_thread_capacity(hw, parallel)
    if cap <= 1:
        seeds = [1]
    elif hw.device_kind != "cpu" and hw.gpu_vram_mb > 0 and parallel >= 4:
        seeds = [2, 3, min(4, cap), cap]
    else:
        seeds = [1, 2, 3, min(4, cap), cap]
    if hinted is not None:
        seeds.append(min(cap, max(1, int(hinted))))
    return [t for t in sorted(set(int(x) for x in seeds)) if 1 <= t <= cap]


def _autotune_batch_game_limit(hw, parallel, concurrent):
    parallel = max(1, int(parallel))
    if not concurrent:
        return parallel
    physical = max(1, int(hw.physical_cpus))
    return max(parallel, min(physical * 2, parallel * 2))


def estimate_model_params(cfg, alphazero_net_cls):
    model = alphazero_net_cls(cfg)
    return sum(p.numel() for p in model.parameters())


def autoscale_model_cfg(cfg, hw, alphazero_net_cls):
    tuned = dict(cfg)
    current_params = estimate_model_params(tuned, alphazero_net_cls)
    board_area = cfg["board"] * cfg["board"]
    actions = cfg["actions"]
    if hw.device_kind == "cpu" or hw.gpu_vram_mb <= 0:
        return tuned
    if board_area <= 64 and actions <= 128:
        target_floor = 1_000_000 if hw.gpu_vram_mb >= 12_000 else 750_000
    elif board_area <= 100 and actions <= 256:
        target_floor = 2_000_000 if hw.gpu_vram_mb >= 12_000 else 1_500_000
    else:
        target_floor = 3_500_000 if hw.gpu_vram_mb >= 12_000 else 2_500_000
    if hw.gpu_vram_mb >= 20_000:
        max_params_cap = 8_000_000
    elif hw.gpu_vram_mb >= 16_000:
        max_params_cap = 6_000_000
    elif hw.gpu_vram_mb >= 12_000:
        max_params_cap = 4_500_000
    elif hw.gpu_vram_mb >= 8_000:
        max_params_cap = 3_000_000
    else:
        max_params_cap = 1_500_000
    if current_params >= int(target_floor * 0.9):
        return tuned
    base_filters = cfg["filters"]
    base_blocks = cfg["blocks"]
    base_vh = cfg["vh"]
    filter_cap = max(base_filters, 192 if board_area <= 64 else 256)
    block_cap = max(base_blocks, 8 if board_area <= 64 else 12)
    vh_cap = 512 if hw.gpu_vram_mb >= 12_000 else 256
    filter_values = sorted(
        set(
            [
                base_filters,
                _round_up_to_multiple(base_filters + 32, 32),
                _round_up_to_multiple(base_filters + 64, 32),
                filter_cap,
            ]
        )
    )
    filter_values = [f for f in filter_values if base_filters <= f <= filter_cap]
    block_values = list(range(base_blocks, block_cap + 1, 2))
    vh_values = sorted(
        set(
            [
                base_vh,
                _round_up_to_multiple(max(base_vh, base_filters), 64),
                _round_up_to_multiple(max(base_vh, base_filters * 2), 64),
                vh_cap,
            ]
        )
    )
    vh_values = [v for v in vh_values if base_vh <= v <= vh_cap]
    candidates = []
    for filters in filter_values:
        for blocks in block_values:
            for vh in vh_values:
                candidate = dict(tuned)
                candidate["filters"] = filters
                candidate["blocks"] = blocks
                candidate["vh"] = vh
                params = estimate_model_params(candidate, alphazero_net_cls)
                if params < current_params or params > max_params_cap:
                    continue
                candidates.append((params, filters, blocks, vh))
    if not candidates:
        return tuned
    above_floor = [c for c in candidates if c[0] >= target_floor]
    if above_floor:
        _, filters, blocks, vh = min(above_floor, key=lambda c: c[0])
    else:
        _, filters, blocks, vh = max(candidates, key=lambda c: c[0])
    if (filters, blocks, vh) != (base_filters, base_blocks, base_vh):
        tuned["filters"] = filters
        tuned["blocks"] = blocks
        tuned["vh"] = vh
    return tuned


def autotune_training_cfg(cfg, hw, concurrent=True, alphazero_net_cls=None):
    tuned = autoscale_model_cfg(cfg, hw, alphazero_net_cls)
    proc_target = _autotune_parallel_limit(hw, concurrent)
    if hw.gpu_vram_mb >= 20_000:
        train_batch_scale = 2.0
        eval_batch_cap = 256
    elif hw.gpu_vram_mb >= 16_000:
        train_batch_scale = 1.5
        eval_batch_cap = 192
    elif hw.gpu_vram_mb >= 12_000:
        train_batch_scale = 1.5
        eval_batch_cap = 128
    elif hw.gpu_vram_mb >= 8_000:
        train_batch_scale = 1.0
        eval_batch_cap = 64
    elif hw.gpu_vram_mb >= 4_000:
        train_batch_scale = 0.75
        eval_batch_cap = 32
    else:
        train_batch_scale = 0.5 if hw.device_kind == "cpu" else 0.75
        eval_batch_cap = 16
    max_parallel = proc_target
    tuned["selfplay_parallel"] = max(1, min(max_parallel, cfg.get("games", 1)))
    tuned["bg_parallel"] = max(1, min(max_parallel, cfg.get("games", max_parallel)))
    parallel_den = max(tuned["selfplay_parallel"], tuned["bg_parallel"], 1)
    tuned["n_threads"] = _autotune_thread_capacity(hw, parallel_den)
    tuned["batch_size"] = max(
        cfg.get("batch_size", 8),
        min(eval_batch_cap, max(tuned["n_threads"] * tuned["bg_parallel"], 8)),
    )
    base_batch = cfg.get("batch", 256)
    batch_multiple = 32 if base_batch >= 256 else 16
    tuned["batch"] = _round_down_to_multiple(
        int(base_batch * train_batch_scale), batch_multiple
    )
    tuned["batch"] = max(batch_multiple, tuned["batch"])
    tuned["bg_batch_games"] = (
        _autotune_batch_game_limit(hw, tuned["bg_parallel"], concurrent=True)
        if concurrent
        else 0
    )
    if not concurrent:
        if hw.logical_cpus >= 24 and hw.gpu_vram_mb >= 12_000:
            tuned["games"] = max(cfg.get("games", 1), tuned["selfplay_parallel"] * 60)
        elif hw.logical_cpus >= 12:
            tuned["games"] = max(cfg.get("games", 1), tuned["selfplay_parallel"] * 50)
    tuned["hw_logical_cpus"] = hw.logical_cpus
    tuned["hw_memory_mb"] = hw.memory_mb
    tuned["hw_gpu_vram_mb"] = hw.gpu_vram_mb
    return tuned


def print_autotune_summary(original_cfg, tuned_cfg, hw):
    print("  Hardware:")
    print(f"    CPU: {hw.logical_cpus} logical / {hw.physical_cpus} physical cores")
    if hw.memory_mb:
        print(f"    RAM: {hw.memory_mb:,} MB")
    if hw.gpu_name or hw.gpu_vendor != "none":
        gpu_desc = hw.gpu_name or hw.gpu_vendor
        if hw.gpu_vram_mb:
            gpu_desc += f" ({hw.gpu_vram_mb:,} MB)"
        print(f"    GPU: {gpu_desc}")
    changed = []
    for key in (
        "filters",
        "blocks",
        "vh",
        "games",
        "batch",
        "n_threads",
        "batch_size",
        "selfplay_parallel",
        "bg_parallel",
        "bg_batch_games",
    ):
        if original_cfg.get(key) != tuned_cfg.get(key):
            changed.append(f"{key}={tuned_cfg.get(key)}")
    print("  Auto-tuned:", ", ".join(changed) if changed else "no changes")


def autotune_signature(hw, cfg, hardware_signature):
    return {
        "hardware": hardware_signature(hw),
        "game": cfg.get("_name"),
        "iters": int(cfg.get("iters", 0)),
        "search_profile": str(cfg.get("search_profile", "quartz")),
        "penalty_mode": str(cfg.get("penalty_mode", "GatedRefresh")),
        "batch_timeout_us": int(cfg.get("batch_timeout_us", 0) or 0),
        "selfplay_topology_version": int(cfg.get("_selfplay_topology_version", 4)),
        "resident_session": bool(cfg.get("_resident_session", False)),
        "shared_eval_session": bool(cfg.get("_shared_eval_session", False)),
        "selfplay_runner_mode": str(cfg.get("_selfplay_runner_mode", "python_batched")),
        "autotune_topology_version": 5,
    }


def load_autotune_profile(profile_path, hw, cfg, hardware_signature):
    if not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path) as f:
            data = json.load(f)
        if data.get("version") != AUTOTUNE_PROFILE_VERSION:
            return None
        if data.get("signature") != autotune_signature(hw, cfg, hardware_signature):
            return None
        return data
    except Exception:
        return None


def save_autotune_profile(
    profile_path, hw, cfg, overrides, benchmarks, hardware_signature
):
    payload = {
        "version": AUTOTUNE_PROFILE_VERSION,
        "signature": autotune_signature(hw, cfg, hardware_signature),
        "overrides": overrides,
        "benchmarks": benchmarks,
        "saved_at": int(time.time()),
    }
    with open(profile_path, "w") as f:
        json.dump(payload, f, indent=2)


def apply_runtime_overrides(cfg, overrides):
    out = dict(cfg)
    for key, value in (overrides or {}).items():
        out[key] = value
    return out


def _autotune_parallel_candidates(cfg, hw, concurrent):
    upper = _autotune_parallel_limit(hw, concurrent)
    if concurrent:
        seeds = [
            2,
            3,
            min(4, upper),
            max(1, upper // 2),
            max(1, (upper * 2) // 3),
            cfg.get("bg_parallel", 1),
            upper,
        ]
        if hw.device_kind == "cpu" or hw.gpu_vram_mb <= 0:
            seeds.insert(0, 1)
        if upper > 4:
            seeds.extend([max(1, upper - 1), max(1, upper - 2)])
    else:
        hinted = max(1, cfg.get("selfplay_parallel", 1))
        seeds = [1, 2, 3, min(4, upper), hinted, min(upper, hinted + 2), upper]
    return [p for p in sorted(set(int(x) for x in seeds)) if 1 <= p <= upper]


def _autotune_batch_game_candidates(hw, parallel, concurrent):
    if not concurrent:
        return [parallel]
    cap = _autotune_batch_game_limit(hw, parallel, concurrent)
    return sorted(set([max(1, parallel), max(1, cap)]))


def _score_selfplay_probe(
    positions_per_s,
    cycle_s,
    concurrent,
    positions=0,
    eval_messages=0,
    model_batch_mean=0.0,
    parallel=1,
    n_threads=1,
):
    if not concurrent:
        return positions_per_s
    score = positions_per_s / math.sqrt(max(cycle_s, 1e-6))
    if positions > 0 and eval_messages > 0:
        message_efficiency = max(float(positions) / float(eval_messages), 1e-6)
        score *= message_efficiency**0.25
    if model_batch_mean > 0.0:
        score *= max(float(model_batch_mean), 1e-6) ** 0.35
    if parallel > 1 and n_threads <= 1 and model_batch_mean <= 1.25:
        score *= 0.35
    return score


def _score_train_batch_probe(
    examples_per_s, batch_n, concurrent=False, target_positions_per_cycle=None
):
    if not concurrent or not target_positions_per_cycle:
        return examples_per_s
    target_batch = max(32.0, float(target_positions_per_cycle) * 4.0)
    freshness_penalty = min(1.0, target_batch / max(float(batch_n), 1.0))
    return examples_per_s * freshness_penalty


def _sync_device(device):
    torch = _torch_module()
    if getattr(device, "type", "cpu") != "cpu" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass


def _mean_or_zero(values):
    return 0.0 if not values else float(sum(values)) / float(len(values))


def _autotune_progress_bar(total, desc, tqdm_factory=None):
    return (tqdm_factory or (lambda total, desc: None))(total, desc)


def _warmup_selfplay_probe(
    cfg, model, device, rust_binary, parallel, batch_games, n_threads, runtime_hooks
):
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = n_threads
    probe_cfg["batch_size"] = max(
        cfg.get("batch_size", 8), min(64, max(n_threads * parallel, 8))
    )
    probe_cfg["_disable_resident_session"] = True
    if model is not None:
        ch, bs = probe_cfg["ch"], probe_cfg["board"]
        warm_batch = np.zeros((probe_cfg["batch_size"], ch, bs, bs), dtype=np.float32)
        runtime_hooks.run_model_batch(model, device, warm_batch)
        _sync_device(device)
    warm_cfg = dict(probe_cfg)
    warm_cfg["iters"] = min(8, max(4, int(cfg.get("iters", 8))))
    try:
        runtime_hooks.selfplay_rust_nn_batched(
            warm_cfg,
            model,
            device,
            max(1, batch_games),
            rust_binary,
            parallel=parallel,
            show_progress=False,
        )
    except Exception:
        pass
    _sync_device(device)


def _run_selfplay_probe(
    cfg,
    model,
    device,
    rust_binary,
    parallel,
    batch_games,
    n_threads,
    runtime_hooks,
    concurrent=True,
    rounds=1,
    warmup=True,
):
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = n_threads
    probe_cfg["batch_size"] = max(
        cfg.get("batch_size", 8), min(64, max(n_threads * parallel, 8))
    )
    probe_cfg["_disable_resident_session"] = True
    if warmup:
        _warmup_selfplay_probe(
            cfg,
            model,
            device,
            rust_binary,
            parallel,
            batch_games,
            n_threads,
            runtime_hooks,
        )
    n_games = max(1, batch_games * max(1, rounds))
    perf_stats = {}
    runtime_hooks.stall_trace(
        "selfplay_probe_begin",
        game=cfg.get("_name"),
        parallel=int(parallel),
        batch_games=int(batch_games),
        n_threads=int(n_threads),
        rounds=int(rounds),
        n_games=int(n_games),
        iters=int(probe_cfg.get("iters", 0)),
        batch_size=int(probe_cfg.get("batch_size", 0)),
        resident=bool(probe_cfg.get("_resident_session", False)),
    )
    _sync_device(device)
    t0 = time.time()
    states, _, _, _ = runtime_hooks.selfplay_rust_nn_batched(
        probe_cfg,
        model,
        device,
        n_games,
        rust_binary,
        parallel=parallel,
        show_progress=False,
        perf_stats=perf_stats,
    )
    _sync_device(device)
    elapsed = max(time.time() - t0, 1e-6)
    positions = sum(len(gs) for gs in states)
    cycle_s = elapsed / max(rounds, 1)
    positions_per_s = positions / elapsed
    positions_per_cycle = positions / max(rounds, 1)
    eval_messages = int(perf_stats.get("eval_messages", 0) or 0)
    model_batch_mean = _mean_or_zero(perf_stats.get("model_batch_sizes", []))
    score = _score_selfplay_probe(
        positions_per_s,
        cycle_s,
        concurrent,
        positions=positions,
        eval_messages=eval_messages,
        model_batch_mean=model_batch_mean,
        parallel=parallel,
        n_threads=n_threads,
    )
    runtime_hooks.stall_trace(
        "selfplay_probe_end",
        game=cfg.get("_name"),
        parallel=int(parallel),
        batch_games=int(batch_games),
        n_threads=int(n_threads),
        rounds=int(rounds),
        positions=int(positions),
        elapsed_s=float(elapsed),
        cycle_s=float(cycle_s),
        positions_per_s=float(positions_per_s),
        eval_messages=int(eval_messages),
        model_calls=int(perf_stats.get("model_calls", 0) or 0),
    )
    return {
        "parallel": parallel,
        "batch_games": batch_games,
        "n_threads": n_threads,
        "batch_size": probe_cfg["batch_size"],
        "probe_rounds": rounds,
        "positions": positions,
        "elapsed_s": round(elapsed, 3),
        "cycle_s": round(cycle_s, 3),
        "positions_per_cycle": round(positions_per_cycle, 3),
        "positions_per_s": round(positions_per_s, 3),
        "eval_messages": eval_messages,
        "eval_items": int(perf_stats.get("eval_items", 0) or 0),
        "model_calls": int(perf_stats.get("model_calls", 0) or 0),
        "model_batch_mean": round(model_batch_mean, 3),
        "score": round(score, 4),
    }


def benchmark_selfplay_throughput(
    cfg, model, device, rust_binary, hw, runtime_hooks, concurrent=True
):
    candidates = _autotune_parallel_candidates(cfg, hw, concurrent)
    coarse_cfg = dict(cfg)
    coarse_cfg["iters"] = min(cfg["iters"], 48 if cfg["board"] <= 9 else 32)
    coarse_cfg["temp_th"] = min(cfg["temp_th"], 6)
    refine_cfg = dict(cfg)
    refine_cfg["iters"] = min(cfg["iters"], 128 if cfg["board"] <= 9 else 80)
    refine_cfg["temp_th"] = min(cfg["temp_th"], 8)
    coarse_plan = []
    for parallel in candidates:
        for batch_games in _autotune_batch_game_candidates(hw, parallel, concurrent):
            for n_threads in _autotune_thread_candidates(
                hw, parallel, hinted=cfg.get("n_threads", 1)
            ):
                coarse_plan.append((parallel, batch_games, n_threads))
    results = []
    with runtime_hooks.tqdm_factory(
        len(coarse_plan), "autotune-selfplay-coarse"
    ) as pbar:
        for parallel, batch_games, n_threads in coarse_plan:
            pbar.set_postfix_str(f"p={parallel} bg={batch_games} th={n_threads}")
            try:
                probe = _run_selfplay_probe(
                    coarse_cfg,
                    model,
                    device,
                    rust_binary,
                    parallel,
                    batch_games,
                    n_threads,
                    runtime_hooks,
                    concurrent=concurrent,
                    rounds=1 if concurrent else 1,
                )
            except Exception as e:
                results.append(
                    {
                        "parallel": parallel,
                        "batch_games": batch_games,
                        "n_threads": n_threads,
                        "error": str(e),
                    }
                )
                pbar.update(1)
                continue
            probe["stage"] = "coarse"
            results.append(probe)
            pbar.update(1)
    scored = [r for r in results if "positions_per_s" in r]
    if not scored:
        return {}, results, {}
    finalists = []
    seen = set()
    for row in sorted(
        scored,
        key=lambda r: (r["score"], r["positions_per_s"], -r["cycle_s"]),
        reverse=True,
    ):
        key = (row["parallel"], row["batch_games"], row["n_threads"])
        if key in seen:
            continue
        seen.add(key)
        finalists.append(row)
        if len(finalists) >= min(4, len(scored)):
            break
    refined = []
    with runtime_hooks.tqdm_factory(len(finalists), "autotune-selfplay-refine") as pbar:
        for row in finalists:
            pbar.set_postfix_str(
                f"p={row['parallel']} bg={row['batch_games']} th={row['n_threads']}"
            )
            try:
                probe = _run_selfplay_probe(
                    refine_cfg,
                    model,
                    device,
                    rust_binary,
                    row["parallel"],
                    row["batch_games"],
                    row["n_threads"],
                    runtime_hooks,
                    concurrent=concurrent,
                    rounds=3 if concurrent else 1,
                )
                probe["stage"] = "refine"
                refined.append(probe)
                results.append(probe)
            except Exception as e:
                results.append(
                    {
                        "parallel": row["parallel"],
                        "batch_games": row["batch_games"],
                        "n_threads": row["n_threads"],
                        "stage": "refine",
                        "error": str(e),
                    }
                )
            pbar.update(1)
    ranking_pool = refined or scored
    best = max(
        ranking_pool, key=lambda r: (r["score"], r["positions_per_s"], -r["cycle_s"])
    )
    overrides = {
        "selfplay_parallel": best["parallel"],
        "bg_parallel": best["parallel"] if concurrent else min(best["parallel"], 4),
        "bg_batch_games": best["batch_games"]
        if concurrent
        else max(1, best["parallel"]),
        "n_threads": best["n_threads"],
    }
    summary = {
        "parallel": best["parallel"],
        "batch_games": best["batch_games"],
        "n_threads": best["n_threads"],
        "positions_per_cycle": best["positions_per_cycle"],
        "positions_per_s": best["positions_per_s"],
        "cycle_s": best["cycle_s"],
        "score": best["score"],
    }
    return overrides, results, summary


def benchmark_train_batch(
    cfg,
    backend,
    model,
    optimizer,
    device,
    hw,
    concurrent=False,
    target_positions_per_cycle=None,
):
    base_batch = cfg["batch"]
    batch_multiple = 32 if base_batch >= 256 else 16
    batch_candidates = sorted(
        set(
            [
                _round_down_to_multiple(int(base_batch * 0.5), batch_multiple),
                _round_down_to_multiple(int(base_batch * 0.75), batch_multiple),
                base_batch,
                _round_down_to_multiple(int(base_batch * 1.25), batch_multiple),
                _round_down_to_multiple(int(base_batch * 1.5), batch_multiple),
            ]
        )
    )
    batch_candidates = [b for b in batch_candidates if b >= 32]
    if concurrent and target_positions_per_cycle:
        max_useful_batch = _round_down_to_multiple(
            int(max(32, target_positions_per_cycle * 4.0)), batch_multiple
        )
        constrained = [b for b in batch_candidates if b <= max_useful_batch]
        if constrained:
            batch_candidates = constrained
    results = []
    ch, bs, actions = cfg["ch"], cfg["board"], cfg["actions"]
    for batch_n in batch_candidates:
        states = np.zeros((batch_n, ch, bs, bs), dtype=np.float32)
        policies = np.full((batch_n, actions), 1.0 / actions, dtype=np.float32)
        values = np.zeros(batch_n, dtype=np.float32)
        if backend is not None:
            if hasattr(backend, "optimizer"):
                model_ref = backend.get_torch_model()
                model_state = copy.deepcopy(model_ref.state_dict())
                opt_state = copy.deepcopy(backend.optimizer.state_dict())

                def restore(
                    model_ref=model_ref, model_state=model_state, opt_state=opt_state
                ):
                    model_ref.load_state_dict(model_state)
                    backend.optimizer.load_state_dict(opt_state)
            else:
                params_state = copy.deepcopy(getattr(backend, "params", None))
                batch_stats_state = copy.deepcopy(getattr(backend, "batch_stats", None))
                opt_state_state = copy.deepcopy(getattr(backend, "opt_state", None))

                def restore(
                    params_state=params_state,
                    batch_stats_state=batch_stats_state,
                    opt_state_state=opt_state_state,
                ):
                    backend.params = copy.deepcopy(params_state)
                    backend.batch_stats = copy.deepcopy(batch_stats_state)
                    backend.opt_state = copy.deepcopy(opt_state_state)

            train_once = lambda states=states, policies=policies, values=values: (
                backend.train_step(states, policies, values)
            )
        else:
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())

            def train_once(states=states, policies=policies, values=values):
                torch = _torch_module()
                F = _torch_functional()
                model.train()
                states_t = torch.tensor(states, dtype=torch.float32).to(device)
                policies_t = torch.tensor(policies, dtype=torch.float32).to(device)
                values_t = torch.tensor(values, dtype=torch.float32).to(device)
                logits, pred_v = model(states_t)
                pl = -(policies_t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                vl = F.mse_loss(pred_v, values_t)
                loss_t = pl + vl
                optimizer.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                return float(loss_t.item())

            def restore(model_state=model_state, opt_state=opt_state):
                model.load_state_dict(model_state)
                optimizer.load_state_dict(opt_state)

        try:
            train_once()
            t0 = time.time()
            measured = 2
            for _ in range(measured):
                train_once()
            elapsed = max(time.time() - t0, 1e-6)
            examples_per_s = (batch_n * measured) / elapsed
            score = _score_train_batch_probe(
                examples_per_s,
                batch_n,
                concurrent=concurrent,
                target_positions_per_cycle=target_positions_per_cycle,
            )
            results.append(
                {
                    "batch": batch_n,
                    "examples_per_s": round(examples_per_s, 3),
                    "elapsed_s": round(elapsed, 3),
                    "score": round(score, 4),
                }
            )
        except RuntimeError as e:
            results.append({"batch": batch_n, "error": str(e)})
            torch = _torch_module()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            restore()
    scored = [r for r in results if "examples_per_s" in r]
    if not scored:
        return {}, results
    best = max(scored, key=lambda r: (r["score"], r["examples_per_s"], -r["batch"]))
    return {"batch": best["batch"]}, results


def run_autotune_benchmark_fast(
    cfg,
    backend,
    model,
    optimizer,
    device,
    hw,
    rust_binary=None,
    runtime_hooks=None,
    concurrent=True,
):
    """Inference-only autotune: measures NN throughput + heuristic sizing.

    ~10-30 seconds instead of 5-15 minutes. No Rust server needed.
    """
    import torch

    overrides = {}
    benchmark = {}

    # 1. Measure pure NN inference throughput at various batch sizes
    is_cpu = str(device) == "cpu"
    test_bs = [1, 4, 8, 16, 32] if not is_cpu else [1, 4, 8]
    ch = cfg.get("ch", 17)
    board = cfg["board"]
    warmup_iters = 5
    bench_iters = 30

    throughputs = {}
    try:
        torch_model = (
            backend.get_torch_model() if hasattr(backend, "get_torch_model") else model
        )
        torch_model.eval()
        with torch.inference_mode():
            for bs in test_bs:
                x = torch.randn(bs, ch, board, board, device=device)
                for _ in range(warmup_iters):
                    torch_model(x)
                if not is_cpu:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(bench_iters):
                    torch_model(x)
                if not is_cpu:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                throughputs[bs] = bs * bench_iters / (t1 - t0)
        benchmark["inference"] = [
            {"batch_size": bs, "evals_per_s": round(tp, 1)}
            for bs, tp in throughputs.items()
        ]
    except Exception as e:
        benchmark["inference"] = [{"error": str(e)}]

    # 2. Heuristic sizing from hardware + inference throughput
    peak_tp = max(throughputs.values()) if throughputs else 1000
    optimal_bs = max(throughputs, key=throughputs.get) if throughputs else 8

    # Parallel workers: enough to keep GPU fed, not more than CPU cores allow
    max_parallel = _autotune_parallel_limit(hw, concurrent)
    # Each worker needs ~1 thread; more workers = more MCTS parallelism
    # but GPU is the bottleneck, so limit by inference throughput
    iters_per_move = cfg.get("iters", 200)
    # Approximate: each selfplay position needs iters_per_move NN evals
    # With batch_size=optimal_bs, GPU can handle peak_tp evals/s
    # target: keep GPU ~80% utilized
    target_positions_per_s = peak_tp / iters_per_move * 0.8
    parallel = max(1, min(max_parallel, int(math.ceil(target_positions_per_s * 2))))

    n_threads = _autotune_thread_capacity(hw, parallel)
    batch_size = max(cfg.get("batch_size", 8), min(optimal_bs * 2, 64))
    bg_batch_games = (
        _autotune_batch_game_limit(hw, parallel, concurrent=True) if concurrent else 0
    )

    overrides["selfplay_parallel"] = parallel
    overrides["bg_parallel"] = parallel if concurrent else min(parallel, 4)
    overrides["bg_batch_games"] = bg_batch_games
    overrides["n_threads"] = n_threads
    overrides["batch_size"] = batch_size

    # 3. Training batch: simple scaling from VRAM
    if hw.gpu_vram_mb >= 16_000:
        train_batch_scale = 1.5
    elif hw.gpu_vram_mb >= 12_000:
        train_batch_scale = 1.25
    elif hw.gpu_vram_mb >= 8_000:
        train_batch_scale = 1.0
    else:
        train_batch_scale = 0.75
    base_batch = cfg.get("batch", 256)
    batch_multiple = 32 if base_batch >= 256 else 16
    overrides["batch"] = max(
        batch_multiple,
        _round_down_to_multiple(int(base_batch * train_batch_scale), batch_multiple),
    )
    benchmark["heuristic"] = {
        "peak_inference_tp": round(peak_tp, 1),
        "optimal_batch_size": optimal_bs,
        "target_positions_per_s": round(target_positions_per_s, 2),
    }
    return overrides, benchmark


def run_autotune_benchmark(
    cfg,
    backend,
    model,
    optimizer,
    device,
    hw,
    rust_binary,
    runtime_hooks,
    concurrent=True,
):
    """Full autotune with Rust selfplay benchmark (legacy, 5-15 minutes)."""
    overrides = {}
    benchmark = {}
    selfplay_meta = {}
    try:
        sp_overrides, sp_results, selfplay_meta = benchmark_selfplay_throughput(
            cfg, model, device, rust_binary, hw, runtime_hooks, concurrent=concurrent
        )
        overrides.update(sp_overrides)
        benchmark["selfplay"] = sp_results
    except Exception as e:
        benchmark["selfplay"] = [{"error": str(e)}]
    try:
        train_overrides, train_results = benchmark_train_batch(
            apply_runtime_overrides(cfg, overrides),
            backend,
            model,
            optimizer,
            device,
            hw,
            concurrent=concurrent,
            target_positions_per_cycle=selfplay_meta.get("positions_per_cycle"),
        )
        overrides.update(train_overrides)
        benchmark["train"] = train_results
    except Exception as e:
        benchmark["train"] = [{"error": str(e)}]
    if "bg_parallel" in overrides and "batch_size" not in overrides:
        overrides["batch_size"] = max(
            cfg.get("batch_size", 8),
            min(
                64,
                max(
                    overrides["bg_parallel"]
                    * overrides.get("n_threads", cfg.get("n_threads", 1)),
                    4,
                ),
            ),
        )
    return overrides, benchmark


def plan_online_runtime_overrides(cfg, hw, sample):
    overrides = {}
    parallel = max(1, cfg.get("bg_parallel", 1))
    batch_games = max(1, cfg.get("bg_batch_games", parallel))
    n_threads = max(1, cfg.get("n_threads", 1))
    batch = max(32, cfg.get("batch", 256))
    batch_multiple = 32 if batch >= 256 else 16
    last_cycle_s = float(sample.get("last_cycle_s", 0.0) or 0.0)
    last_cycle_positions = int(sample.get("last_cycle_positions", 0) or 0)
    rolling_cycle_s = float(sample.get("rolling_cycle_s", last_cycle_s) or last_cycle_s)
    positions_per_s = float(
        sample.get("rolling_positions_per_s", sample.get("positions_per_s", 0.0)) or 0.0
    )
    best_positions_per_s = float(
        sample.get("best_positions_per_s", positions_per_s) or positions_per_s
    )
    burst_ratio = float(sample.get("burst_ratio", 1.0) or 1.0)
    n_new = int(sample.get("n_new", 0) or 0)
    train_steps = int(sample.get("train_steps", 0) or 0)
    max_parallel = _autotune_parallel_limit(hw, concurrent=True)
    thread_capacity = _autotune_thread_capacity(hw, parallel)
    batch_game_cap = _autotune_batch_game_limit(hw, max_parallel, concurrent=True)
    if parallel > max_parallel:
        overrides["bg_parallel"] = max_parallel
        parallel = max_parallel
        thread_capacity = _autotune_thread_capacity(hw, parallel)
        batch_game_cap = _autotune_batch_game_limit(hw, max_parallel, concurrent=True)
        if batch_games > batch_game_cap:
            overrides["bg_batch_games"] = batch_game_cap
    if burst_ratio > 1.8 and batch_games > parallel:
        overrides["bg_batch_games"] = max(parallel, batch_games - parallel)
    elif rolling_cycle_s > 3.5 and batch_games > parallel:
        overrides["bg_batch_games"] = max(parallel, max(1, batch_games // 2))
    elif (
        rolling_cycle_s < 1.5
        and batch_games < batch_game_cap
        and positions_per_s >= best_positions_per_s * 0.95
    ):
        overrides["bg_batch_games"] = min(batch_game_cap, batch_games + parallel)
    if (
        hw.device_kind != "cpu"
        and hw.gpu_vram_mb > 0
        and n_threads == 1
        and parallel >= max(4, max_parallel)
        and thread_capacity >= 2
    ):
        overrides["n_threads"] = min(thread_capacity, 2 if parallel >= 6 else 3)
    if (
        hw.device_kind != "cpu"
        and hw.gpu_vram_mb > 0
        and parallel <= 2
        and n_threads > min(thread_capacity, 6)
    ):
        overrides["n_threads"] = min(thread_capacity, 6)
    if n_threads > thread_capacity and positions_per_s < best_positions_per_s * 0.95:
        overrides["n_threads"] = thread_capacity
    elif (
        rolling_cycle_s < 2.0
        and n_threads < thread_capacity
        and positions_per_s >= best_positions_per_s * 0.95
    ):
        overrides["n_threads"] = min(thread_capacity, n_threads + 1)
    eff_parallel = overrides.get("bg_parallel", parallel)
    eff_threads = overrides.get("n_threads", n_threads)
    desired_batch_size = max(4, min(64, max(eff_parallel * eff_threads, 4)))
    if desired_batch_size != cfg.get("batch_size", 8):
        overrides["batch_size"] = desired_batch_size
    if max(last_cycle_positions, int(sample.get("rolling_positions", 0) or 0)) > 0:
        effective_positions = max(
            last_cycle_positions, int(sample.get("rolling_positions", 0) or 0)
        )
        target_batch = _round_down_to_multiple(
            int(max(64, effective_positions * 3.5)), batch_multiple
        )
        target_batch = max(batch_multiple, target_batch)
        if n_new < batch * 0.5 and train_steps <= 3 and target_batch < batch:
            overrides["batch"] = target_batch
        elif (
            n_new > batch * 1.5
            and train_steps >= max(1, cfg.get("steps", 100) // 4)
            and target_batch > batch
        ):
            overrides["batch"] = target_batch
    return overrides


class OnlineAutotuneController:
    def __init__(self, cfg, hw, enabled_iters=10, interval=2):
        self.cfg = cfg
        self.hw = hw
        self.enabled_iters = max(1, enabled_iters)
        self.interval = max(1, interval)
        self.best_positions_per_s = 0.0
        self._last_snapshot = None

    def observe(
        self, iteration_idx, n_new, elapsed_s, train_steps, replay_size, worker
    ):
        if worker is None or iteration_idx >= self.enabled_iters:
            return {}
        if (iteration_idx + 1) % self.interval != 0:
            return {}
        snapshot = dict(worker.telemetry())
        self._last_snapshot = snapshot
        positions_per_s = float(n_new) / max(float(elapsed_s), 1e-6)
        self.best_positions_per_s = max(self.best_positions_per_s, positions_per_s)
        sample = {
            "iteration": iteration_idx + 1,
            "n_new": n_new,
            "elapsed_s": elapsed_s,
            "positions_per_s": positions_per_s,
            "best_positions_per_s": self.best_positions_per_s,
            "train_steps": train_steps,
            "replay_size": replay_size,
            "last_cycle_s": snapshot.get("last_cycle_s", 0.0),
            "last_cycle_positions": snapshot.get("last_cycle_positions", 0),
            "last_cycle_games": snapshot.get("last_cycle_games", 0),
            "rolling_cycle_s": snapshot.get(
                "rolling_cycle_s", snapshot.get("last_cycle_s", 0.0)
            ),
            "rolling_positions_per_s": snapshot.get(
                "rolling_positions_per_s", positions_per_s
            ),
            "rolling_positions": snapshot.get(
                "rolling_positions", snapshot.get("last_cycle_positions", 0)
            ),
            "rolling_games": snapshot.get(
                "rolling_games", snapshot.get("last_cycle_games", 0)
            ),
            "burst_ratio": snapshot.get("burst_ratio", 1.0),
        }
        overrides = plan_online_runtime_overrides(self.cfg, self.hw, sample)
        for key, value in list(overrides.items()):
            if self.cfg.get(key) == value:
                overrides.pop(key)
        if not overrides:
            return {}
        self.cfg.update(overrides)
        return overrides
