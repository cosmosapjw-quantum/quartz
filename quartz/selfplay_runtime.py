"""Helpers for self-play runtime orchestration and streamed game decoding."""

from __future__ import annotations

import logging
import hashlib
import json
import math
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from quartz.replay import sparse_policy_from_entries


_SEARCH_MANIFEST_KEYS = (
    "_name",
    "iters",
    "search_profile",
    "vl_mode",
    "penalty_mode",
    "root_only_shaping",
    "prior_refresh_rate",
    "prior_refresh_temp",
    "c_puct",
    "sigma_0",
    "min_visits",
    "check_interval",
    "hbar_penalty_cap",
    "n_threads",
    "thread_policy",
    "auto_thread_policy",
    "thread_cap",
    "max_threads",
    "n_threads_cap",
    "batch_size",
    "batch_timeout_us",
    "_eval_runner_mode",
    "_arena_low_concurrency_profile",
    "eval_seed",
    "halt_mode",
)


def _search_manifest_hash(cfg):
    manifest = {}
    for key in _SEARCH_MANIFEST_KEYS:
        if key in cfg:
            value = cfg.get(key)
            if value is None:
                continue
            manifest[key] = value
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_LOGICAL_CPU_COUNT = max(1, os.cpu_count() or 1)


def estimate_selfplay_positions_per_game(cfg, recent_chunks):
    rolling_games = 0
    rolling_positions = 0
    for chunk in recent_chunks:
        rolling_games += int(chunk.get("games", 0) or 0)
        rolling_positions += int(chunk.get("positions", 0) or 0)
    if rolling_games > 0 and rolling_positions > 0:
        return max(1.0, rolling_positions / rolling_games)
    board = int(cfg.get("board", 7) or 7)
    return max(4.0, float(board * board) * 0.5)


def replay_fill_ceiling(cfg, replay_capacity=None, backpressure_ratio=None):
    capacity = replay_capacity
    if capacity is None:
        capacity = int(cfg.get("buf", 0) or 0)
    if capacity <= 0:
        return None
    ratio = float(backpressure_ratio if backpressure_ratio is not None else SelfPlayWorker.BACKPRESSURE_RATIO)
    return max(1, int(math.floor(float(capacity) * max(0.0, ratio))))


def initial_replay_fill_target(cfg, recent_chunks):
    train_batch = max(1, int(cfg.get("batch", 256) or 256))
    batch_target = max(1, int(cfg.get("batch_size", 8) or 8))
    base_parallel = max(1, int(cfg.get("bg_parallel", 2) or 2))
    positions_per_game = estimate_selfplay_positions_per_game(cfg, recent_chunks)
    warm_games = max(base_parallel, int(math.ceil(batch_target / max(positions_per_game, 1.0))))
    target = int(min(train_batch, max(batch_target, int(math.ceil(warm_games * positions_per_game)))))
    target_cap = int(cfg.get("_bootstrap_replay_target_cap", 0) or 0)
    if target_cap > 0:
        target = min(target, target_cap)
    ceiling = replay_fill_ceiling(cfg)
    if ceiling is not None:
        target = min(target, ceiling)
    return int(max(1, target))


def plan_selfplay_runner_chunk(cfg, replay_size, recent_chunks):
    cfg_get = cfg.get
    base_parallel = int(cfg_get("bg_parallel", 2) or 2)
    if base_parallel < 1:
        base_parallel = 1
    base_batch_games = int(cfg_get("bg_batch_games", base_parallel) or base_parallel)
    if base_batch_games < 1:
        base_batch_games = 1
    parallel_cap_override = int(cfg_get("_selfplay_parallel_cap", 0) or 0)
    batch_games_cap_override = int(cfg_get("_selfplay_batch_games_cap", 0) or 0)
    batch_target = int(cfg_get("batch_size", 8) or 8)
    if batch_target < 1:
        batch_target = 1
    train_batch = int(cfg_get("batch", 256) or 256)
    if train_batch < 1:
        train_batch = 1

    positions_per_game = estimate_selfplay_positions_per_game(cfg, recent_chunks)
    replay_deficit = train_batch - int(replay_size)
    if replay_deficit < 0:
        replay_deficit = 0
    pos_floor = positions_per_game if positions_per_game > 1.0 else 1.0
    games_needed = int(math.ceil(replay_deficit / pos_floor))
    if games_needed < 1:
        games_needed = 1

    parallel = base_parallel
    batch_games = base_batch_games
    if cfg_get("_selfplay_runner_mode") == "rust_selfplay_state_machine":
        parallel_cap = base_parallel if base_parallel > _LOGICAL_CPU_COUNT else _LOGICAL_CPU_COUNT
        if parallel_cap_override > 0 and parallel_cap_override < parallel_cap:
            parallel_cap = parallel_cap_override
        parallel_target = batch_target if batch_target > games_needed else games_needed
        if parallel_target > parallel_cap:
            parallel_target = parallel_cap
        parallel = base_parallel if base_parallel > parallel_target else parallel_target

        quad_parallel = parallel * 4
        quad_batch = batch_target * 4
        quad = quad_parallel if quad_parallel > quad_batch else quad_batch
        if quad > train_batch:
            quad = train_batch
        max_batch_games_cap = base_batch_games if base_batch_games > quad else quad
        if batch_games_cap_override > 0 and batch_games_cap_override < max_batch_games_cap:
            max_batch_games_cap = batch_games_cap_override

        candidate = base_batch_games
        if parallel > candidate:
            candidate = parallel
        if games_needed > candidate:
            candidate = games_needed
        batch_games = candidate if candidate < max_batch_games_cap else max_batch_games_cap

    if parallel_cap_override > 0 and parallel > parallel_cap_override:
        parallel = parallel_cap_override
    if batch_games_cap_override > 0 and batch_games > batch_games_cap_override:
        batch_games = batch_games_cap_override
    if parallel < 1:
        parallel = 1
    if batch_games < 1:
        batch_games = 1

    games_per_call = batch_games
    base_or_target = base_parallel if base_parallel > batch_target else batch_target
    if games_per_call > base_or_target:
        games_per_call = base_or_target
    if parallel > games_per_call:
        games_per_call = parallel

    return {
        "parallel": int(parallel),
        "batch_games": int(batch_games),
        "games_per_call": int(games_per_call),
        "replay_deficit": int(replay_deficit),
        "estimated_positions_per_game": round(float(positions_per_game), 3),
    }


class RustServerPool:
    """Small reusable pool of Rust search servers for batched self-play."""

    def __init__(self, rust_binary, launch_server, stop_server):
        self.rust_binary = rust_binary
        self._launch_server = launch_server
        self._stop_server = stop_server
        self._lock = threading.Lock()
        self._procs = []
        self._restart_count = 0
        self._restart_streak = 0
        self._last_restart_reason = None
        self._last_restart_ts = 0.0

    def _note_restart(self, reason):
        self._restart_count += 1
        self._restart_streak += 1
        self._last_restart_reason = str(reason)
        self._last_restart_ts = time.time()

    def record_failure(self, reason):
        with self._lock:
            self._note_restart(reason)

    def snapshot(self):
        with self._lock:
            return {
                "cached_procs": int(len(self._procs)),
                "restart_count": int(self._restart_count),
                "restart_streak": int(self._restart_streak),
                "last_restart_reason": self._last_restart_reason,
                "last_restart_age_s": (
                    round(max(0.0, time.time() - self._last_restart_ts), 3) if self._last_restart_ts > 0.0 else None
                ),
            }

    def acquire(self, n):
        with self._lock:
            alive = []
            reaped = 0
            for proc in self._procs:
                if proc.poll() is None:
                    alive.append(proc)
                else:
                    self._stop_server(proc, timeout=0.1)
                    reaped += 1
            self._procs = alive
            if reaped:
                self._note_restart(f"reaped_dead_servers:{reaped}")
            while len(self._procs) < n:
                if self._restart_streak > 0 and self._last_restart_ts > 0.0:
                    backoff_s = min(2.0, 0.25 * (2 ** min(self._restart_streak - 1, 3)))
                    wait_s = self._last_restart_ts + backoff_s - time.time()
                    if wait_s > 0.0:
                        time.sleep(wait_s)
                try:
                    proc = self._launch_server(self.rust_binary)
                except Exception as exc:
                    self._note_restart(f"launch_failed:{type(exc).__name__}:{exc}")
                    raise
                self._procs.append(proc)
                self._restart_streak = 0
            return list(self._procs[:n])

    def kill_active(self):
        with self._lock:
            for proc in self._procs:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._procs = []

    def close(self):
        with self._lock:
            procs, self._procs = self._procs, []
        for proc in procs:
            self._stop_server(proc)


def encode_board_with_history(cfg, board_12_sequence, move_idx, player):
    bs = cfg["board"]
    n2 = bs * bs
    history_len = 8
    total_ch = history_len * 2 + 1
    enc = np.zeros((total_ch, bs, bs), dtype=np.float32)

    # Map player convention: board uses 1=black, 2=white; we need +1/-1
    my_val = 1 if player == 1 else 2
    opp_val = 2 if player == 1 else 1

    for t in range(history_len):
        hist_idx = move_idx - t
        if hist_idx < 0:
            break
        board_12 = board_12_sequence[hist_idx]
        # [OPT] Vectorized: convert list → numpy array → boolean mask → reshape
        board_arr = np.asarray(board_12, dtype=np.int8).ravel()[:n2]
        enc[t * 2].ravel()[:len(board_arr)] = (board_arr == my_val).astype(np.float32)
        enc[t * 2 + 1].ravel()[:len(board_arr)] = (board_arr == opp_val).astype(np.float32)

    if player == 1:
        enc[total_ch - 1] = 1.0
    return enc


def decode_streamed_selfplay_game(cfg, game_payload):
    n_actions = int(cfg["actions"])
    board_hist = game_payload.get("states", []) or []
    player_hist = game_payload.get("players", []) or []
    policy_hist = game_payload.get("policies", []) or []
    traces = game_payload.get("trace", []) or []
    lengths = {
        "states": len(board_hist),
        "players": len(player_hist),
        "policies": len(policy_hist),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"streamed self-play payload length mismatch: {lengths}")
    if traces and len(traces) != len(board_hist):
        raise ValueError(
            "streamed self-play trace length mismatch: "
            f"trace={len(traces)} states={len(board_hist)}"
        )
    states = []
    policies = []
    for move_idx, (_board_12, raw_player, sparse_pol) in enumerate(zip(board_hist, player_hist, policy_hist)):
        player = 1 if int(raw_player) > 0 else -1
        states.append(encode_board_with_history(cfg, board_hist, move_idx, player))
        policies.append(sparse_policy_from_entries(sparse_pol, n_actions))
    outcome = float(game_payload.get("outcome", 0.0) or 0.0)
    return states, policies, outcome, traces


def choose_selfplay_move(policy, legal, move_count, temp_threshold, fallback_best=-1):
    if not legal:
        return 0

    if move_count < temp_threshold:
        probs = np.array([policy[a] for a in legal], dtype=np.float64)
        total = probs.sum()
        if total > 1e-8:
            probs /= total
            return int(np.random.choice(legal, p=probs))
        return int(random.choice(legal))

    if fallback_best in legal:
        return int(fallback_best)
    return int(max(legal, key=lambda a: policy[a]))


def wait_for_worker_progress(worker, previous_count, min_new=1, timeout_s=30.0, poll_s=0.25):
    deadline = time.time() + timeout_s
    current = worker.positions_generated
    stall_timeout_s = getattr(worker, "REPLAY_STALL_TIMEOUT_S", 45.0)
    while current - previous_count < min_new and time.time() < deadline:
        if worker._stop.is_set():
            break
        status = worker.status() if hasattr(worker, "status") else None
        if status is not None:
            if not status.get("alive", True):
                raise RuntimeError(
                    f"background self-play worker stopped unexpectedly: {status.get('last_error') or 'thread exited'}"
                )
            if (
                status.get("consecutive_errors", 0) >= 3
                and status.get("last_progress_age_s", 0.0) > min(10.0, stall_timeout_s / 3.0)
            ):
                raise RuntimeError(
                    "background self-play made no progress after repeated errors: "
                    f"{status.get('last_error') or 'no progress'}"
                )
            if (
                status.get("last_progress_age_s", 0.0) > stall_timeout_s
                and status.get("consecutive_errors", 0) > 0
            ):
                raise RuntimeError(
                    f"background self-play stalled: {status.get('last_error') or 'no progress'}"
                )
        time.sleep(poll_s)
        current = worker.positions_generated
    return max(0, current - previous_count), current


def _supervised_pipeline_collect(pipeline, *, proc, timeout_s, label):
    deadline = time.perf_counter() + max(0.0, float(timeout_s))
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Rust server exited (code={proc.returncode}) while waiting for {label}")
        wait_s = min(0.25, max(0.001, deadline - time.perf_counter()))
        if wait_s <= 0.0:
            raise TimeoutError(f"timed out waiting for {label}")
        try:
            return pipeline.collect(timeout=wait_s)
        except queue.Empty:
            continue


def compute_train_steps(base_steps, batch_size, n_new, concurrent=False):
    if not concurrent:
        return base_steps
    if n_new <= 0:
        return 0
    target_reuse = 8.0
    scaled = math.ceil((n_new / max(batch_size, 1)) * target_reuse)
    return max(1, min(base_steps, scaled))


def default_output_dir(game_name):
    return os.path.join("models", f"alphazero_{game_name}")


@dataclass(frozen=True)
class SearchClientRuntimeHooks:
    launch_server: object
    proc_write_json_line: object
    proc_write_qipc_frame: object
    cleanup_qipc_transport: object
    proc_read_json_line: object
    json_loads_fast: object
    rust_game_name: object
    rust_search_options: object
    is_chess_game: object
    normalize_rust_board: object
    proc_read_message: object
    shm_eval_loop: object
    proc_decode_eval_frame: object
    qipc_arena_eval_req: int
    qipc_arena_eval_resp: int
    qipc_batch_eval_req: int
    qipc_eval_req: int
    make_eval_request_group: object
    pack_qipc_arena_eval_req: object
    unpack_qipc_batch_eval_req: object
    unpack_qipc_arena_eval_resp: object
    unpack_qipc_eval_req: object
    compute_eval_collect_policy: object
    wait_readable: object
    inference_pipeline_thread_cls: object
    should_use_async_pipeline: object | None
    run_batched_eval_groups: object
    write_batched_eval_group: object
    run_model_batch: object
    torch_module: object
    pack_qipc_eval_resp: object
    pack_qipc_batch_eval_resp: object
    parse_eval_request: object


@dataclass(frozen=True)
class SelfPlayLoopRuntimeHooks:
    is_chess_game: object
    is_go_game: object
    search_client_cls: object
    initial_chess_fen: object
    build_training_game_adapter: object
    encode_chess_fen: object
    build_rust_state_meta: object
    chess_state_meta_from_hashes: object
    choose_selfplay_move: object


@dataclass(frozen=True)
class ArenaRuntimeHooks:
    is_chess_game: object
    search_client_cls: object
    alphazero_net_cls: object
    load_torch_state_dict: object
    torch_module: object
    initial_chess_fen: object
    chess_state_meta_from_hashes: object
    arena_compare: object
    build_training_game_adapter: object
    rust_nn_evaluator_engine_cls: object
    match_runner_cls: object


@dataclass(frozen=True)
class BatchedSelfPlayRuntimeHooks:
    is_chess_game: object
    is_go_game: object
    should_use_resident_session: object
    supports_rust_selfplay_state_machine: object
    search_client_cls: object
    decode_streamed_selfplay_game: object
    encode_chess_fen: object
    initial_chess_fen: object
    build_training_game_adapter: object
    chess_state_meta_from_hashes: object
    rust_game_name: object
    normalize_rust_board: object
    build_rust_state_meta: object
    choose_selfplay_move: object
    proc_decode_eval_frame: object
    qipc_batch_eval_req: int
    qipc_eval_req: int
    unpack_qipc_batch_eval_req: object
    unpack_qipc_eval_req: object
    make_eval_request_group: object
    stall_trace: object
    proc_write_json_line: object
    proc_read_json_line: object
    proc_read_message: object
    shm_eval_loop: object
    wait_readable: object
    compute_eval_collect_policy: object
    inference_pipeline_thread_cls: object
    should_use_async_pipeline: object | None
    run_batched_eval_groups: object
    write_batched_eval_group: object
    rust_search_options: object
    launch_server: object
    stop_server: object
    emit_duty_cycle: object


class NNSearchClient:
    """Drives bidirectional NN eval protocol with the Rust MCTS server."""

    def __init__(
        self,
        model,
        cfg,
        device,
        rust_binary="./target/release/mcts_demo",
        *,
        runtime_hooks,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.proc = None
        self.rust_binary = rust_binary
        self._rt = runtime_hooks
        self.search_read_timeout_s = float(
            os.environ.get("QUARTZ_SEARCH_STALL_TIMEOUT_S", "120") or 120.0
        )

    def start(self):
        self.proc = self._rt.launch_server(self.rust_binary)

    def stop(self):
        if self.proc:
            try:
                self._rt.proc_write_json_line(self.proc, {"cmd": "quit"})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self._rt.cleanup_qipc_transport(self.proc)
            self.proc = None

    def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
        game_name = self._rt.rust_game_name(self.cfg["_name"])
        last_error = None
        for attempt in range(2):
            if not self.proc:
                self.start()
            req_dict = {
                "cmd": "search_nn",
                "game": game_name,
                "player": int(player),
                "iters": self.cfg["iters"],
            }
            req_dict.update(self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode))
            if self._rt.is_chess_game(game_name) and fen:
                req_dict["fen"] = fen
                if state_meta:
                    req_dict.update(state_meta)
            else:
                req_dict["board"] = self._rt.normalize_rust_board(game_name, board_flat)
                if state_meta:
                    req_dict.update(state_meta)
            try:
                payload = self._exchange_search_request(req_dict)
                if isinstance(payload, dict) and "result" in payload:
                    return payload.get("result", {})
                return payload if isinstance(payload, dict) else {}
            except TimeoutError as exc:
                last_error = exc
                logging.getLogger(__name__).warning(
                    "search_move timed out on attempt %d/2 for %s; restarting Rust server",
                    attempt + 1,
                    game_name,
                )
                self.stop()
        if last_error is not None:
            raise last_error
        return {}

    def search_moves_multi(self, jobs, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        if not jobs:
            return []
        req_dict = {
            "cmd": "search_nn_multi",
            "game": self._rt.rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "jobs": jobs,
        }
        req_dict.update(self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload.get("results", [])
        return []

    def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "search_nn_multi_session_open",
            "game": self._rt.rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "jobs": jobs,
        }
        req_dict.update(self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload
        return {}

    def open_search_engine_session(self, jobs, penalty_mode="GatedRefresh", iters=None):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "search_nn_multi_engine_session_open",
            "game": self._rt.rust_game_name(self.cfg["_name"]),
            "iters": int(self.cfg["iters"] if iters is None else iters),
            "jobs": jobs,
        }
        req_dict.update(self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload
        return {}

    def step_search_session(self, session_id, updates):
        if not self.proc:
            self.start()
        payload = self._exchange_search_request(
            {
                "cmd": "search_nn_multi_session_step",
                "session_id": int(session_id),
                "updates": updates,
            }
        )
        if isinstance(payload, dict):
            return payload
        return {}

    def step_search_engine_session(self, session_id, updates=None, iters=None):
        if not self.proc:
            self.start()
        payload = self._exchange_search_request(
            {
                "cmd": "search_nn_multi_engine_session_step",
                "session_id": int(session_id),
                "iters": int(self.cfg["iters"] if iters is None else iters),
                "updates": list(updates or []),
            }
        )
        if isinstance(payload, dict):
            return payload
        return {}

    def close_search_session(self, session_id):
        if not self.proc:
            return {}
        try:
            self._rt.proc_write_json_line(
                self.proc,
                {"cmd": "search_nn_multi_session_close", "session_id": int(session_id)},
            )
            payload = self._rt.proc_read_json_line(self.proc)
        except Exception:
            return {}
        if not payload:
            return {}
        try:
            return self._rt.json_loads_fast(payload)
        except Exception:
            return {}

    def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        game_name = self._rt.rust_game_name(self.cfg["_name"])
        frame_payload = self._rt.pack_qipc_arena_eval_req(
            game_name,
            self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode),
            sessions,
            iters=self.cfg["iters"],
            max_moves=int(max_moves),
        )
        payload = self._exchange_search_request(
            frame_kind=self._rt.qipc_arena_eval_req,
            frame_payload=frame_payload,
        )
        if isinstance(payload, dict):
            return payload
        return {}

    def selfplay_run(
        self,
        n_games,
        parallel,
        temp_threshold,
        penalty_mode="GatedRefresh",
        seed=0,
        on_chunk=None,
        on_progress=None,
    ):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "selfplay_nn_run",
            "game": self._rt.rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "n_games": int(n_games),
            "parallel": int(parallel),
            "temp_threshold": int(temp_threshold),
            "seed": int(seed),
        }
        req_dict.update(self._rt.rust_search_options(self.cfg, penalty_mode=penalty_mode))
        ring = getattr(self.proc, "_quartz_ring_buffer", None)
        if ring is not None:
            try:
                baseline_epoch = int(ring.epoch())
            except Exception:
                baseline_epoch = None
        else:
            baseline_epoch = None
        self._rt.proc_write_json_line(self.proc, req_dict)

        if ring is not None:
            aggregated_games = []

            def _on_json(obj):
                if not isinstance(obj, dict):
                    return
                if "selfplay_chunk" in obj:
                    games = obj["selfplay_chunk"].get("games", []) or []
                    if callable(on_chunk):
                        on_chunk(games)
                    else:
                        aggregated_games.extend(games)
                elif "selfplay_progress" in obj and callable(on_progress):
                    on_progress(obj["selfplay_progress"])

            ring_payload = self._rt.shm_eval_loop(
                ring, self.model, self.device, self.cfg, self.proc, on_json=_on_json, baseline_epoch=baseline_epoch
            )
            if isinstance(ring_payload, dict):
                if "selfplay_done" in ring_payload:
                    done = dict(ring_payload["selfplay_done"])
                    done["games"] = aggregated_games
                    return done
                if "games" in ring_payload:
                    payload = dict(ring_payload)
                    payload.setdefault("games", aggregated_games)
                    return payload
            kind, payload = self._rt.proc_read_message(self.proc)
            if kind == "json" and isinstance(payload, dict):
                if "selfplay_done" in payload:
                    done = dict(payload["selfplay_done"])
                    done["games"] = aggregated_games
                    return done
                if "games" in payload:
                    return payload
                return payload
            return {"games": aggregated_games}

        base_collect_timeout_s, base_target_eval_items = self._base_eval_collect_config()
        batch_items_ema = float(base_target_eval_items)
        collect_wait_ema_s = 0.0
        deferred = None
        aggregated_games = []

        while True:
            kind, payload = deferred if deferred is not None else self._rt.proc_read_message(self.proc)
            deferred = None
            first_group, terminal = self._parse_eval_group(kind, payload)
            if terminal is not None:
                if isinstance(terminal, dict):
                    if "selfplay_chunk" in terminal:
                        games = terminal["selfplay_chunk"].get("games", []) or []
                        if callable(on_chunk):
                            on_chunk(games)
                        else:
                            aggregated_games.extend(games)
                        continue
                    if "selfplay_progress" in terminal and callable(on_progress):
                        on_progress(terminal["selfplay_progress"])
                        continue
                    if "selfplay_done" in terminal:
                        done = dict(terminal["selfplay_done"])
                        done["games"] = aggregated_games
                        return done
                    if "games" in terminal:
                        return terminal
                return terminal

            eval_groups = [first_group]
            eval_item_count = len(first_group["requests"])
            dynamic_target_eval_items, dynamic_collect_timeout_s = self._rt.compute_eval_collect_policy(
                base_target_eval_items,
                base_collect_timeout_s,
                batch_items_ema=batch_items_ema,
                wait_ema_s=collect_wait_ema_s,
            )
            collect_t0 = time.perf_counter()
            deadline = time.perf_counter() + dynamic_collect_timeout_s
            while eval_item_count < dynamic_target_eval_items:
                timeout_s = max(0.0, deadline - time.perf_counter())
                if timeout_s <= 0.0 or not self._rt.wait_readable(self.proc.stdout, timeout_s):
                    break
                next_kind, next_payload = self._rt.proc_read_message(self.proc)
                next_group, next_terminal = self._parse_eval_group(next_kind, next_payload)
                if next_terminal is not None:
                    deferred = (next_kind, next_payload)
                    break
                eval_groups.append(next_group)
                eval_item_count += len(next_group["requests"])

            merged_items = sum(len(group["requests"]) for group in eval_groups)
            collect_wait_s = time.perf_counter() - collect_t0
            batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
            collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)
            responses = self._rt.run_batched_eval_groups(eval_groups, self.model, self.device, self.cfg)
            for response_group in responses:
                self._rt.write_batched_eval_group(self.proc, response_group)

    @staticmethod
    def _emit_duty_cycle(duty):
        if not os.environ.get("QUARTZ_DUTY_CYCLE"):
            return
        total = duty["read_s"] + duty["collect_s"] + duty["model_s"] + duty["write_s"]
        if total < 1e-9:
            return
        import sys as _sys

        msg = (
            f'  [DutyCycle] cycles={duty["cycles"]}'
            f' read={duty["read_s"]:.3f}s({duty["read_s"]/total*100:.0f}%)'
            f' collect={duty["collect_s"]:.3f}s({duty["collect_s"]/total*100:.0f}%)'
            f' model={duty["model_s"]:.3f}s({duty["model_s"]/total*100:.0f}%)'
            f' write={duty["write_s"]:.3f}s({duty["write_s"]/total*100:.0f}%)'
            f' total={total:.3f}s\n'
        )
        try:
            _sys.stderr.write(msg)
            _sys.stderr.flush()
        except Exception:
            pass

    def _parse_eval_group(self, kind, payload):
        if kind is None:
            return None, None
        if kind == "frame":
            frame_kind, frame_payload = payload
            frame_kind, frame_payload = self._rt.proc_decode_eval_frame(self.proc, frame_kind, frame_payload)
            if frame_kind == self._rt.qipc_arena_eval_resp:
                return None, self._rt.unpack_qipc_arena_eval_resp(frame_payload)
            if frame_kind == self._rt.qipc_batch_eval_req:
                return self._rt.make_eval_request_group(
                    "binary_batch",
                    self._rt.unpack_qipc_batch_eval_req(frame_payload),
                    gi=0,
                ), None
            if frame_kind == self._rt.qipc_eval_req:
                return self._rt.make_eval_request_group(
                    "binary_single",
                    [self._rt.unpack_qipc_eval_req(frame_payload)],
                    gi=0,
                ), None
            return None, {"error": f"unexpected IPC frame kind: {frame_kind}"}
        if kind == "json":
            if not payload:
                return None, None
            if "batch_eval_req" in payload:
                requests = payload["batch_eval_req"].get("requests", [])
                return self._rt.make_eval_request_group(
                    "json_batch",
                    [
                        (int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))
                        for er in requests
                    ],
                    gi=0,
                ), None
            if "eval_req" in payload:
                er = payload["eval_req"]
                return self._rt.make_eval_request_group(
                    "json_single",
                    [(int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))],
                    gi=0,
                ), None
            return None, payload
        return None, {"error": "unexpected message"}

    def _base_eval_collect_config(self):
        search_opts = self._rt.rust_search_options(self.cfg)
        base_collect_timeout_s = min(
            0.006,
            max(0.00075, float(search_opts.get("batch_timeout_us", 1500)) / 1_000_000.0 * 0.9),
        )
        base_target_eval_items = max(1, int(search_opts.get("batch_size", self.cfg.get("batch_size", 8))))
        return base_collect_timeout_s, base_target_eval_items

    def _exchange_search_request(self, req_dict=None, *, frame_kind=None, frame_payload=None):
        if not self.proc:
            self.start()
        read_timeout_s = max(1.0, float(self.search_read_timeout_s))
        base_collect_timeout_s, base_target_eval_items = self._base_eval_collect_config()
        batch_items_ema = float(base_target_eval_items)
        collect_wait_ema_s = 0.0
        ring = getattr(self.proc, "_quartz_ring_buffer", None)
        if ring is not None:
            try:
                baseline_epoch = int(ring.epoch())
            except Exception:
                baseline_epoch = None
        else:
            baseline_epoch = None
        if frame_kind is None:
            self._rt.proc_write_json_line(self.proc, req_dict)
        else:
            self._rt.proc_write_qipc_frame(self.proc, frame_kind, frame_payload or b"")
        if ring is not None:
            ring_payload = self._rt.shm_eval_loop(
                ring, self.model, self.device, self.cfg, self.proc, baseline_epoch=baseline_epoch
            )
            if isinstance(ring_payload, dict):
                return ring_payload
            kind, payload = self._rt.proc_read_message(self.proc, timeout_s=read_timeout_s)
            if kind == "json" and isinstance(payload, dict):
                return payload
            return {}

        deferred = None
        duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
        duty_log_interval = 16
        pipeline_policy = getattr(self._rt, "should_use_async_pipeline", None)
        if callable(pipeline_policy):
            use_pipeline = bool(pipeline_policy(self.model, self.device, self.cfg))
        else:
            use_pipeline = (
                not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
                and self.model is not None
                and not hasattr(self.model, "predict")
            )
        pipeline = None
        inflight = False
        if use_pipeline:
            pipeline = self._rt.inference_pipeline_thread_cls(self.model, self.device, self.cfg, max_pending=1)
            pipeline.start()

        try:
            while True:
                if inflight and pipeline is not None:
                    model_t0 = time.perf_counter()
                    responses = _supervised_pipeline_collect(
                        pipeline,
                        proc=self.proc,
                        timeout_s=30.0,
                        label="Rust search pipeline responses",
                    )
                    duty["model_s"] += time.perf_counter() - model_t0
                    inflight = False
                    write_t0 = time.perf_counter()
                    for rg in responses:
                        self._rt.write_batched_eval_group(self.proc, rg)
                    duty["write_s"] += time.perf_counter() - write_t0

                read_t0 = time.perf_counter()
                kind, payload = (
                    deferred
                    if deferred is not None
                    else self._rt.proc_read_message(self.proc, timeout_s=read_timeout_s)
                )
                deferred = None
                duty["read_s"] += time.perf_counter() - read_t0

                if kind is None:
                    if duty["cycles"] > 0:
                        self._emit_duty_cycle(duty)
                    return {}
                first_group, terminal = self._parse_eval_group(kind, payload)
                if terminal is not None:
                    if duty["cycles"] > 0:
                        self._emit_duty_cycle(duty)
                    return terminal

                eval_groups = [first_group]
                eval_item_count = len(first_group["requests"])
                dynamic_target_eval_items, dynamic_collect_timeout_s = self._rt.compute_eval_collect_policy(
                    base_target_eval_items,
                    base_collect_timeout_s,
                    batch_items_ema=batch_items_ema,
                    wait_ema_s=collect_wait_ema_s,
                )
                collect_t0 = time.perf_counter()
                deadline = time.perf_counter() + dynamic_collect_timeout_s
                while eval_item_count < dynamic_target_eval_items:
                    timeout_s = max(0.0, deadline - time.perf_counter())
                    if timeout_s <= 0.0 or not self._rt.wait_readable(self.proc.stdout, timeout_s):
                        break
                    next_kind, next_payload = self._rt.proc_read_message(self.proc, timeout_s=read_timeout_s)
                    next_group, next_terminal = self._parse_eval_group(next_kind, next_payload)
                    if next_terminal is not None:
                        deferred = (next_kind, next_payload)
                        break
                    eval_groups.append(next_group)
                    eval_item_count += len(next_group["requests"])

                merged_items = sum(len(group["requests"]) for group in eval_groups)
                collect_wait_s = time.perf_counter() - collect_t0
                duty["collect_s"] += collect_wait_s
                batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
                collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)

                if pipeline is not None:
                    pipeline.submit(eval_groups)
                    inflight = True
                else:
                    model_t0 = time.perf_counter()
                    responses = self._rt.run_batched_eval_groups(eval_groups, self.model, self.device, self.cfg)
                    duty["model_s"] += time.perf_counter() - model_t0
                    write_t0 = time.perf_counter()
                    for rg in responses:
                        self._rt.write_batched_eval_group(self.proc, rg)
                    duty["write_s"] += time.perf_counter() - write_t0

                duty["cycles"] += 1
                if duty["cycles"] % duty_log_interval == 0:
                    self._emit_duty_cycle(duty)
        finally:
            if inflight and pipeline is not None:
                try:
                    drain = _supervised_pipeline_collect(
                        pipeline,
                        proc=self.proc,
                        timeout_s=10.0,
                        label="Rust search pipeline drain",
                    )
                    for rg in drain:
                        self._rt.write_batched_eval_group(self.proc, rg)
                except Exception:
                    pass
            if pipeline is not None:
                pipeline.stop()

    def _eval_from_features(self, features, n_act):
        try:
            ch, bs = self.cfg["ch"], self.cfg["board"]
            expected = ch * bs * bs

            if len(features) == expected and self.model is not None:
                x = np.asarray(features, dtype=np.float32).reshape(1, ch, bs, bs)
                with self._rt.torch_module.inference_mode():
                    probs, vals_np = self._rt.run_model_batch(self.model, self.device, x)
                return probs[0][:n_act], float(vals_np[0])
        except Exception as exc:
            import sys

            print(f"[WARN] NN eval failed: {exc}", file=sys.stderr)
        na = max(1, int(n_act))
        return np.full(na, 1.0 / na, dtype=np.float32), 0.0

    def _eval_nn_json(self, eval_req):
        policy, value = self._eval_from_features(
            eval_req.get("features", []),
            eval_req.get("num_actions", self.cfg["actions"]),
        )
        return {"eval_resp": {"policy": policy.tolist(), "value": value}}

    def _eval_nn_binary(self, payload):
        num_actions, features, _model_tag, _fp_lo, _fp_hi, _encoder_rev = self._rt.unpack_qipc_eval_req(payload)
        policy, value = self._eval_from_features(features, num_actions)
        return self._rt.pack_qipc_eval_resp(policy, value)

    def _eval_nn_batch_json(self, batch_req):
        try:
            requests = batch_req["requests"]
            batch_size = len(requests)
            ch, bs = self.cfg["ch"], self.cfg["board"]
            expected = ch * bs * bs

            if self.model is not None and batch_size > 0:
                features_list = []
                for req in requests:
                    feats = req.get("features", [])
                    if len(feats) == expected:
                        features_list.append(feats)
                    else:
                        features_list.append([0.0] * expected)

                probs, vals_np = self._rt.run_model_batch(
                    self.model,
                    self.device,
                    np.asarray(features_list, dtype=np.float32).reshape(batch_size, ch, bs, bs),
                )

                responses = []
                for i, req in enumerate(requests):
                    na = req.get("num_actions", self.cfg["actions"])
                    responses.append({"policy": probs[i][:na].tolist(), "value": float(vals_np[i])})
                return {"batch_eval_resp": {"responses": responses}}
        except Exception as exc:
            import sys

            print(f"[WARN] Batch NN eval failed: {exc}", file=sys.stderr)
        na = self.cfg["actions"]
        uniform = {"policy": [1.0 / max(1, na)] * na, "value": 0.0}
        n = batch_req.get("batch_size", 1) if "batch_req" in dir() else 1
        return {"batch_eval_resp": {"responses": [uniform] * n}}

    def _eval_nn_batch_binary(self, payload):
        requests = self._rt.unpack_qipc_batch_eval_req(payload)
        batch_size = len(requests)
        ch, bs = self.cfg["ch"], self.cfg["board"]
        expected = ch * bs * bs
        if self.model is not None and batch_size > 0:
            features_list = []
            num_actions = []
            for req in requests:
                na, feats, _model_tag, _fp_lo, _fp_hi, _encoder_rev = self._rt.parse_eval_request(req)
                num_actions.append(int(na))
                if feats.size == expected:
                    features_list.append(np.asarray(feats, dtype=np.float32).reshape(ch, bs, bs))
                else:
                    features_list.append(np.zeros((ch, bs, bs), dtype=np.float32))
            probs, vals_np = self._rt.run_model_batch(self.model, self.device, np.stack(features_list, axis=0))
            policies = [probs[i][: num_actions[i]] for i in range(batch_size)]
            values = [float(vals_np[i]) for i in range(batch_size)]
            return self._rt.pack_qipc_batch_eval_resp(policies, values)
        uniform_policies = []
        for req in requests:
            na, _feats, _model_tag, _fp_lo, _fp_hi, _encoder_rev = self._rt.parse_eval_request(req)
            na = max(1, int(na))
            uniform_policies.append(np.full(na, 1.0 / na, dtype=np.float32))
        return self._rt.pack_qipc_batch_eval_resp(uniform_policies, [0.0] * len(uniform_policies))


def selfplay_rust_nn(cfg, model, device, n_games, rust_binary="./target/release/mcts_demo", *, runtime_hooks):
    """Single-game-loop self-play using Rust search + Python NN callbacks."""
    n_actions = cfg["actions"]
    penalty_mode = cfg.get("penalty_mode", "GatedRefresh")
    is_chess = runtime_hooks.is_chess_game(cfg.get("_name"))
    max_moves = 500 if is_chess or runtime_hooks.is_go_game(cfg.get("_name")) else (cfg["board"] ** 2 + 5)

    client = runtime_hooks.search_client_cls(model, cfg, device, rust_binary)
    try:
        client.start()
    except FileNotFoundError:
        print(f"  [WARN] Rust binary not found: {rust_binary}", file=sys.stderr)
        return [], [], [], n_games
    except RuntimeError as exc:
        print(f"  [WARN] {exc}", file=sys.stderr)
        return [], [], [], n_games

    all_states, all_policies, all_outcomes = [], [], []

    with tqdm(total=n_games, desc="Self-play (Rust+NN)", leave=False) as pbar:
        for _game_idx in range(n_games):
            game_states, game_policies = [], []
            move_count = 0
            outcome = 0.0
            void_result = False

            if is_chess:
                current_fen = runtime_hooks.initial_chess_fen(cfg)
                current_chess_meta = {}
                player = 1
                chess_outcome = 0.0
            else:
                game = runtime_hooks.build_training_game_adapter(cfg)

            while move_count < max_moves:
                if is_chess:
                    enc = runtime_hooks.encode_chess_fen(current_fen)
                else:
                    legal = game.legal_moves()
                    if not legal:
                        if hasattr(game, "is_void_result") and game.is_void_result():
                            void_result = True
                            outcome = 0.0
                        else:
                            outcome = float(game.outcome_for_black() or 0.0)
                        break
                    player = 1 if game.current_player() == 0 else -1
                    enc = game._encode()

                if is_chess:
                    result = client.search_move(
                        None,
                        player,
                        penalty_mode,
                        fen=current_fen,
                        state_meta=current_chess_meta,
                    )
                else:
                    result = client.search_move(
                        game._board,
                        player,
                        penalty_mode,
                        state_meta=runtime_hooks.build_rust_state_meta(cfg.get("_name"), game, cfg),
                    )
                if not result or "error" in result:
                    break

                pol_entries = result.get("policy", [])
                if not pol_entries:
                    if is_chess:
                        terminal_value = result.get("value", 0.0)
                        chess_outcome = terminal_value * player
                    break
                policy = sparse_policy_from_entries(pol_entries, n_actions)

                game_states.append(enc.copy())
                game_policies.append(policy.copy())

                if is_chess:
                    new_fen = result.get("result_fen", "")
                    if not new_fen or new_fen == current_fen:
                        break
                    current_fen = new_fen
                    current_chess_meta = runtime_hooks.chess_state_meta_from_hashes(
                        result.get("result_history_hashes", [])
                    )
                    move_count += 1
                    player = -player
                else:
                    chosen = runtime_hooks.choose_selfplay_move(
                        policy,
                        legal,
                        move_count,
                        cfg["temp_th"],
                        fallback_best=result.get("best_move", -1),
                    )
                    game.apply_move(chosen)
                    move_count += 1
                    if game.is_terminal():
                        if hasattr(game, "is_void_result") and game.is_void_result():
                            void_result = True
                            outcome = 0.0
                        else:
                            outcome = float(game.outcome_for_black() or 0.0)
                        break

            if is_chess:
                outcome = chess_outcome
            elif void_result:
                game_states = []
                game_policies = []
            all_states.append(game_states)
            all_policies.append(game_policies)
            all_outcomes.append(outcome)
            pbar.update(1)
            pbar.set_postfix_str(f"moves={move_count}")

    client.stop()
    return all_states, all_policies, all_outcomes, 0


def arena_rust_nn_impl(
    model_a_path,
    cfg_a,
    model_b_path,
    cfg_b,
    device,
    n_games=50,
    rust_binary="./target/release/mcts_demo",
    strict=True,
    *,
    runtime_hooks,
):
    if cfg_a.get("_name") != cfg_b.get("_name"):
        raise ValueError(
            f"arena_rust_nn requires same game on both sides: {cfg_a.get('_name')} vs {cfg_b.get('_name')}"
        )

    cfg_a = dict(cfg_a)
    cfg_b = dict(cfg_b)
    same_search_cfg = _search_manifest_hash(cfg_a) == _search_manifest_hash(cfg_b)
    if same_search_cfg:
        cfg_a.setdefault("_eval_runner_mode", "rust_eval_state_machine")
        cfg_b.setdefault("_eval_runner_mode", "rust_eval_state_machine")
    same_model_source = same_search_cfg and os.path.abspath(model_a_path) == os.path.abspath(model_b_path)
    share_same_model_object = same_model_source and cfg_a.get("_eval_runner_mode") != "rust_eval_state_machine"

    model_a = runtime_hooks.alphazero_net_cls(cfg_a).to(device)
    model_a.load_state_dict(
        runtime_hooks.load_torch_state_dict(model_a_path, runtime_hooks.torch_module, map_location=device)
    )
    model_a.eval()
    if share_same_model_object:
        model_b = model_a
    else:
        model_b = runtime_hooks.alphazero_net_cls(cfg_b).to(device)
        model_b.load_state_dict(
            runtime_hooks.load_torch_state_dict(model_b_path, runtime_hooks.torch_module, map_location=device)
        )
        model_b.eval()

    game_name = cfg_a.get("_name")
    is_chess = runtime_hooks.is_chess_game(game_name)
    max_moves = 500 if is_chess else int(cfg_a["board"]) ** 2
    wins_a = wins_b = draws = 0

    p0, p1 = 0.5, 0.55
    alpha, beta = 0.05, 0.05
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    sprt_decided = False
    sprt_result = None

    try:
        if not same_search_cfg:
            raise RuntimeError("__arena_use_legacy_dual_cfg__")
        engine_a = runtime_hooks.rust_nn_evaluator_engine_cls(
            "arena_a",
            cfg_a,
            model_a,
            device,
            rust_binary,
        )
        engine_b = runtime_hooks.rust_nn_evaluator_engine_cls(
            "arena_b",
            cfg_b,
            model_b,
            device,
            rust_binary,
        )
        runner = runtime_hooks.match_runner_cls(
            lambda: runtime_hooks.build_training_game_adapter(dict(cfg_a)),
            seed=int(cfg_a.get("seed", 0) or 0),
            max_moves=max_moves,
        )
        tally = runner.play_match_tally_batched(engine_a, engine_b, n_games, color_swap=True)
        if strict and (
            getattr(tally, "errors", 0)
            or getattr(tally, "voids", 0)
            or (getattr(tally, "total", 0) and getattr(tally, "scored", 0) == 0)
        ):
            raise RuntimeError(
                "strict arena produced unscored games "
                f"(errors={getattr(tally, 'errors', 0)} voids={getattr(tally, 'voids', 0)} "
                f"scored={getattr(tally, 'scored', 0)} total={getattr(tally, 'total', 0)})"
            )
        wins_a = int(tally.wins)
        wins_b = int(tally.losses)
        draws = int(tally.draws)
        decisive = wins_a + wins_b
        if decisive > 0:
            llr = wins_a * math.log(p1 / p0) + (decisive - wins_a) * math.log((1 - p1) / (1 - p0))
            if llr >= upper_bound:
                sprt_decided = True
                sprt_result = "H1_accept"
            elif llr <= lower_bound:
                sprt_decided = True
                sprt_result = "H0_accept"
    except RuntimeError as exc:
        if str(exc) != "__arena_use_legacy_dual_cfg__":
            raise
        client_a = runtime_hooks.search_client_cls(model_a, cfg_a, device, rust_binary)
        client_b = runtime_hooks.search_client_cls(model_b, cfg_b, device, rust_binary)
        try:
            client_a.start()
            client_b.start()
        except FileNotFoundError:
            if strict:
                raise RuntimeError(
                    f"Arena (strict mode): Rust binary not found at {rust_binary}. "
                    f"Run: cargo build --release. "
                    f"Use strict=False for Python TreeMCTS fallback (NOT benchmark-grade)."
                )
            print("  [WARN] Rust binary not found, falling back to Python arena (NOT benchmark-grade)")
            raise RuntimeError("strict=False fallback does not support asymmetric search configs")

        board_size = cfg_a["board"]
        n2 = board_size ** 2
        win_len = cfg_a["win"]
        penalty_mode_a = cfg_a.get("penalty_mode", "GatedRefresh")
        penalty_mode_b = cfg_b.get("penalty_mode", "GatedRefresh")

        with tqdm(total=n_games, desc="Arena (Rust+NN)", leave=False) as pbar:
            for game_idx in range(n_games):
                if game_idx % 2 == 0:
                    first_client, second_client = client_a, client_b
                    first_is_a = True
                else:
                    first_client, second_client = client_b, client_a
                    first_is_a = False

                board = np.zeros(n2, dtype=np.int8) if not is_chess else None
                player = 1
                winner = 0
                current_fen = runtime_hooks.initial_chess_fen(cfg_a) if is_chess else None
                current_chess_meta = {} if is_chess else None

                for _move_n in range(max_moves):
                    client = first_client if player == 1 else second_client
                    penalty_mode = penalty_mode_a if client is client_a else penalty_mode_b

                    if is_chess:
                        result = client.search_move(
                            None, player, penalty_mode, fen=current_fen, state_meta=current_chess_meta
                        )
                    else:
                        result = client.search_move(board, player, penalty_mode)
                    if not result or "error" in result:
                        break

                    pol_entries = result.get("policy", [])
                    if not pol_entries:
                        if is_chess:
                            terminal_value = float(result.get("value", 0.0))
                            if terminal_value < -0.5:
                                winner = -player
                            elif terminal_value > 0.5:
                                winner = player
                        break

                    if is_chess:
                        new_fen = result.get("result_fen", "")
                        if not new_fen or new_fen == current_fen:
                            break
                        current_fen = new_fen
                        current_chess_meta = runtime_hooks.chess_state_meta_from_hashes(
                            result.get("result_history_hashes", [])
                        )
                        player = -player
                    else:
                        best = result.get("best_move", -1)
                        legal = [i for i in range(n2) if board[i] == 0]
                        if best < 0 or best >= n2 or board[best] != 0:
                            if legal:
                                best = random.choice(legal)
                            else:
                                break

                        board[best] = player

                        if win_len > 0:
                            r0, c0 = best // board_size, best % board_size
                            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                                cnt = 1
                                for sign in [1, -1]:
                                    nr, nc = r0 + sign * dr, c0 + sign * dc
                                    while 0 <= nr < board_size and 0 <= nc < board_size and board[nr * board_size + nc] == player:
                                        cnt += 1
                                        nr += sign * dr
                                        nc += sign * dc
                                if cnt >= win_len:
                                    winner = player
                                    break
                            if winner:
                                break

                        if not [i for i in range(n2) if board[i] == 0]:
                            break
                        player = -player

                if winner == 1:
                    if first_is_a:
                        wins_a += 1
                    else:
                        wins_b += 1
                elif winner == -1:
                    if first_is_a:
                        wins_b += 1
                    else:
                        wins_a += 1
                else:
                    draws += 1

                pbar.update(1)
                pbar.set_postfix_str(f"A:{wins_a} B:{wins_b} D:{draws}")

        client_a.stop()
        client_b.stop()
    except FileNotFoundError:
        if strict:
            raise RuntimeError(
                f"Arena (strict mode): Rust binary not found at {rust_binary}. "
                f"Run: cargo build --release. "
                f"Use strict=False for Python TreeMCTS fallback (NOT benchmark-grade)."
            )
        print("  [WARN] Rust binary not found, falling back to Python arena (NOT benchmark-grade)")
        if cfg_a == cfg_b:
            return runtime_hooks.arena_compare(model_a_path, model_b_path, cfg_a, device, n_games)
        raise RuntimeError("strict=False fallback does not support asymmetric search configs")

    total = wins_a + wins_b + draws
    wr = wins_a / max(total, 1)
    z = 1.96
    n = max(total, 1)
    p_hat = wr
    ci_lo = (p_hat + z * z / (2 * n) - z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
    ci_hi = (p_hat + z * z / (2 * n) + z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
    sprt_str = sprt_result or "inconclusive"
    return wins_a, wins_b, draws, wr, (ci_lo, ci_hi), sprt_str


def selfplay_rust_nn_batched(
    cfg,
    model,
    device,
    n_games,
    rust_binary="./target/release/mcts_demo",
    parallel=4,
    show_progress=True,
    proc_pool=None,
    perf_stats=None,
    on_game=None,
    active_proc_ref=None,
    *,
    runtime_hooks,
):
    """Run N games in parallel via a single Rust server + shared batched NN eval."""
    board_size = cfg["board"]
    n_actions = cfg["actions"]
    penalty_mode = cfg.get("penalty_mode", "GatedRefresh")
    iters = cfg["iters"]
    is_chess = runtime_hooks.is_chess_game(cfg.get("_name"))
    use_resident_session = (
        not bool(cfg.get("_disable_resident_session", False))
        and runtime_hooks.should_use_resident_session(
            cfg.get("_name"),
            parallel,
            n_games,
            enabled=bool(cfg.get("_resident_session", False)),
        )
    )
    rust_game = runtime_hooks.rust_game_name(cfg["_name"])
    max_moves = 500 if is_chess or runtime_hooks.is_go_game(cfg.get("_name")) else (board_size ** 2 + 5)

    all_states, all_policies, all_outcomes, all_traces = [], [], [], []
    games_done = 0
    if model is not None and hasattr(model, "eval"):
        model.eval()
    proc = None

    if (
        cfg.get("_selfplay_runner_mode") == "rust_selfplay_state_machine"
        and runtime_hooks.supports_rust_selfplay_state_machine(cfg.get("_name"))
        and not is_chess
    ):
        client = runtime_hooks.search_client_cls(model, cfg, device, rust_binary)
        try:
            client.start()
            if active_proc_ref is not None:
                active_proc_ref._active_proc = client.proc

            def _handle_stream_chunk(games):
                for game_payload in games:
                    states, policies, outcome, traces = runtime_hooks.decode_streamed_selfplay_game(cfg, game_payload)
                    all_states.append(states)
                    all_policies.append(policies)
                    all_outcomes.append(outcome)
                    all_traces.append(traces)
                    if callable(on_game):
                        on_game(states, policies, outcome, traces)

            try:
                payload = client.selfplay_run(
                    n_games=n_games,
                    parallel=parallel,
                    temp_threshold=cfg["temp_th"],
                    penalty_mode=penalty_mode,
                    seed=random.randint(0, 2**31),
                    on_chunk=_handle_stream_chunk,
                )
            except TypeError:
                payload = client.selfplay_run(
                    n_games=n_games,
                    parallel=parallel,
                    temp_threshold=cfg["temp_th"],
                    penalty_mode=penalty_mode,
                    seed=random.randint(0, 2**31),
                )
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload.get("error")))
            games = payload.get("games", []) if isinstance(payload, dict) else []
            if isinstance(games, list) and games:
                _handle_stream_chunk(games)
            if isinstance(payload, dict):
                completed_games = int(payload.get("completed_games", len(all_states)) or 0)
                if completed_games == len(all_states) == n_games:
                    return all_states, all_policies, all_outcomes, all_traces
        finally:
            client.stop()

    if perf_stats is not None:
        perf_stats.setdefault("eval_messages", 0)
        perf_stats.setdefault("eval_items", 0)
        perf_stats.setdefault("model_calls", 0)
        perf_stats.setdefault("model_batch_sizes", [])
        perf_stats.setdefault("model_time_s", 0.0)
        perf_stats.setdefault("collect_wait_s", 0.0)
        perf_stats.setdefault("collect_loops", 0)
        perf_stats.setdefault("result_messages", 0)

    def build_game_data():
        gd = {
            "player": 1,
            "moves": 0,
            "states": [],
            "policies": [],
            "finished": False,
            "winner": 0.0,
            "void_result": False,
            "trace": [],
        }
        if is_chess:
            gd["fen"] = runtime_hooks.initial_chess_fen(cfg)
            gd["chess_history_hashes"] = []
        else:
            gd["state"] = runtime_hooks.build_training_game_adapter(cfg)
        return gd

    def build_job(gd):
        if is_chess:
            job = {"fen": gd["fen"]}
            job.update(runtime_hooks.chess_state_meta_from_hashes(gd.get("chess_history_hashes", [])))
            return job
        state = gd["state"]
        player = 1 if state.current_player() == 0 else -1
        job = {
            "player": player,
            "board": runtime_hooks.normalize_rust_board(rust_game, state._board),
        }
        job.update(runtime_hooks.build_rust_state_meta(cfg.get("_name"), state, cfg))
        return job

    def apply_result(gd, result):
        pol_entries = result.get("policy", [])
        if not pol_entries:
            if is_chess:
                gd["winner"] = result.get("value", 0.0) * gd["player"]
            else:
                state = gd["state"]
                if hasattr(state, "is_void_result") and state.is_void_result():
                    gd["void_result"] = True
                    gd["winner"] = 0.0
                else:
                    gd["winner"] = float(state.outcome_for_black() or 0.0)
            gd["finished"] = True
            return None

        policy = sparse_policy_from_entries(pol_entries, n_actions)

        enc = runtime_hooks.encode_chess_fen(gd["fen"]) if is_chess else gd["state"]._encode()

        gd["states"].append(enc.copy())
        gd["policies"].append(policy.copy())
        gd["trace"].append(
            {
                "p_flip": result.get("p_flip", 0.0),
                "value": result.get("value", 0.0),
                "sigma_q": result.get("sigma_q", 0.0),
                "stop_reason": result.get("stop_reason", ""),
                "hbar_eff": result.get("hbar_eff", 0.0),
                "iterations": result.get("iterations", 0),
                "dup_rate": result.get("dup_rate", 0.0),
                "max_pending": result.get("max_pending", 0),
                "avg_vvalue": result.get("avg_vvalue", 0.0),
                "search_manifest": dict(result.get("search_manifest") or {}),
                "realized_budget": dict(result.get("realized_budget") or {}),
                "controller_summary": dict(result.get("controller_summary") or {}),
            }
        )

        if is_chess:
            new_fen = result.get("result_fen", "")
            if not new_fen or new_fen == gd["fen"]:
                gd["finished"] = True
                gd["winner"] = result.get("value", 0.0) * gd["player"]
            else:
                gd["fen"] = new_fen
                gd["chess_history_hashes"] = [int(v) for v in result.get("result_history_hashes", [])]
                gd["moves"] += 1
                gd["player"] = -gd["player"]
            return None

        state = gd["state"]
        legal = state.legal_moves()
        chosen = runtime_hooks.choose_selfplay_move(
            policy,
            legal,
            gd["moves"],
            cfg["temp_th"],
            fallback_best=result.get("best_move", -1),
        )
        state.apply_move(chosen)
        gd["moves"] += 1
        if state.is_terminal():
            gd["finished"] = True
            if hasattr(state, "is_void_result") and state.is_void_result():
                gd["void_result"] = True
                gd["winner"] = 0.0
            else:
                gd["winner"] = float(state.outcome_for_black() or 0.0)
            return None
        return chosen

    def parse_eval_group(kind, payload):
        if kind == "frame":
            frame_kind, frame_payload = runtime_hooks.proc_decode_eval_frame(proc, payload[0], payload[1])
            if frame_kind == runtime_hooks.qipc_batch_eval_req:
                requests = runtime_hooks.unpack_qipc_batch_eval_req(frame_payload)
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += len(requests)
                return runtime_hooks.make_eval_request_group(
                    "binary_batch",
                    requests,
                    gi=0,
                    prefer_shm=True,
                ), None
            if frame_kind == runtime_hooks.qipc_eval_req:
                na, feats, model_tag, fp_lo, fp_hi, encoder_rev = runtime_hooks.unpack_qipc_eval_req(frame_payload)
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += 1
                return runtime_hooks.make_eval_request_group(
                    "binary_single",
                    [(na, feats, model_tag, fp_lo, fp_hi, encoder_rev)],
                    gi=0,
                    prefer_shm=True,
                ), None
            return None, {"error": f"unexpected frame kind {frame_kind}"}
        if kind == "json" and isinstance(payload, dict):
            if "batch_eval_req" in payload:
                parsed = payload["batch_eval_req"]
                reqs = [
                    (int(r.get("num_actions", n_actions)), r.get("features", []))
                    for r in parsed.get("requests", [])
                ]
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += len(reqs)
                return runtime_hooks.make_eval_request_group("json_batch", reqs, gi=0), None
            if "eval_req" in payload:
                er = payload["eval_req"]
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += 1
                return runtime_hooks.make_eval_request_group(
                    "json_single",
                    [(int(er.get("num_actions", n_actions)), er.get("features", []))],
                    gi=0,
                ), None
            if "results" in payload:
                return None, payload
            if "error" in payload:
                return None, payload
        return None, {"error": "unexpected message"}

    duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
    duty_log_interval = 16
    pipeline_policy = getattr(runtime_hooks, "should_use_async_pipeline", None)
    if callable(pipeline_policy):
        use_pipeline = bool(pipeline_policy(model, device, cfg))
    else:
        use_pipeline = (
            not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
            and model is not None
            and not hasattr(model, "predict")
        )

    def exchange_search_request(req):
        nonlocal batch_items_ema, collect_wait_ema_s
        req_cmd = req.get("cmd", "?") if isinstance(req, dict) else "?"
        req_jobs = len(req.get("jobs", [])) if isinstance(req, dict) and isinstance(req.get("jobs"), list) else None
        req_updates = len(req.get("updates", [])) if isinstance(req, dict) and isinstance(req.get("updates"), list) else None
        req_t0 = time.perf_counter()
        runtime_hooks.stall_trace(
            "exchange_begin",
            cmd=req_cmd,
            jobs=req_jobs,
            updates=req_updates,
            parallel=int(parallel),
            n_games=int(n_games),
            resident=bool(use_resident_session),
        )
        ring = getattr(proc, "_quartz_ring_buffer", None)
        # Capture baseline_epoch BEFORE writing the JSON command. Rust bumps
        # the ring epoch as soon as it reads stdin, so capturing after the
        # write races against the server: if the server reads + bumps before
        # this thread captures, baseline becomes equal to the new command's
        # epoch, and shm_eval_loop's `req_epoch <= baseline_epoch` discard
        # filter then rejects the very request we are waiting for, deadlocking
        # against `cmd_done && command_started`. Mirrors the order used in
        # SelfplayRunner.selfplay_run at line 692.
        baseline_epoch = None
        if ring is not None:
            try:
                baseline_epoch = int(ring.epoch())
            except Exception:
                baseline_epoch = None
        runtime_hooks.proc_write_json_line(proc, req)

        if ring is not None:
            ring_payload = runtime_hooks.shm_eval_loop(
                ring, model, device, cfg, proc, baseline_epoch=baseline_epoch
            )
            runtime_hooks.stall_trace(
                "exchange_end",
                cmd=req_cmd,
                loops=0,
                elapsed_s=float(time.perf_counter() - req_t0),
            )
            if perf_stats is not None and ring_payload is not None:
                perf_stats["result_messages"] += 1
            if isinstance(ring_payload, dict):
                if ring_payload.get("error"):
                    raise RuntimeError(str(ring_payload.get("error")))
                return ring_payload
            kind, payload = runtime_hooks.proc_read_message(proc)
            if kind == "json" and isinstance(payload, dict):
                if payload.get("error"):
                    raise RuntimeError(str(payload.get("error")))
                return payload
            return None

        deferred = None
        results_payload = None
        loop_count = 0
        pipeline = None
        inflight = False

        if use_pipeline:
            pipeline = runtime_hooks.inference_pipeline_thread_cls(model, device, cfg, max_pending=1)
            pipeline.start()

        try:
            while results_payload is None:
                loop_count += 1
                if perf_stats is not None:
                    perf_stats["collect_loops"] += 1

                if inflight and pipeline is not None:
                    model_t0 = time.perf_counter()
                    flush_responses = _supervised_pipeline_collect(
                        pipeline,
                        proc=proc,
                        timeout_s=30.0,
                        label=f"{req_cmd} pipeline responses",
                    )
                    duty["model_s"] += time.perf_counter() - model_t0
                    inflight = False
                    write_t0 = time.perf_counter()
                    for rg in flush_responses:
                        runtime_hooks.write_batched_eval_group(proc, rg)
                    duty["write_s"] += time.perf_counter() - write_t0

                read_t0 = time.perf_counter()
                kind, payload = deferred if deferred is not None else runtime_hooks.proc_read_message(proc)
                duty["read_s"] += time.perf_counter() - read_t0
                runtime_hooks.stall_trace(
                    "exchange_message",
                    cmd=req_cmd,
                    loop=int(loop_count),
                    kind=kind,
                    read_wait_s=float(time.perf_counter() - read_t0),
                    deferred=bool(deferred is not None),
                )
                deferred = None
                if kind is None:
                    runtime_hooks.stall_trace("exchange_eof", cmd=req_cmd, loop=int(loop_count))
                    return None

                first_group, terminal = parse_eval_group(kind, payload)
                if terminal is not None:
                    results_payload = terminal
                    runtime_hooks.stall_trace(
                        "exchange_terminal",
                        cmd=req_cmd,
                        loop=int(loop_count),
                        elapsed_s=float(time.perf_counter() - req_t0),
                        keys=sorted(list(terminal.keys())) if isinstance(terminal, dict) else None,
                    )
                    break

                eval_groups = [first_group]
                eval_item_count = len(first_group["requests"])
                dynamic_target_eval_items, dynamic_collect_timeout_s = runtime_hooks.compute_eval_collect_policy(
                    base_target_eval_items,
                    base_collect_timeout_s,
                    batch_items_ema=batch_items_ema,
                    wait_ema_s=collect_wait_ema_s,
                )
                collect_t0 = time.perf_counter()
                deadline = time.perf_counter() + dynamic_collect_timeout_s
                while eval_item_count < dynamic_target_eval_items:
                    timeout_s = max(0.0, deadline - time.perf_counter())
                    if timeout_s <= 0.0:
                        break
                    if not runtime_hooks.wait_readable(proc.stdout, timeout_s):
                        break
                    next_kind, next_payload = runtime_hooks.proc_read_message(proc)
                    next_group, next_terminal = parse_eval_group(next_kind, next_payload)
                    if next_terminal is not None:
                        deferred = (next_kind, next_payload)
                        break
                    eval_groups.append(next_group)
                    eval_item_count += len(next_group["requests"])

                merged_items = sum(len(group["requests"]) for group in eval_groups)
                collect_wait_s = time.perf_counter() - collect_t0
                duty["collect_s"] += collect_wait_s
                if perf_stats is not None:
                    perf_stats["collect_wait_s"] += float(collect_wait_s)
                batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
                collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)

                if pipeline is not None:
                    pipeline.submit(eval_groups)
                    inflight = True
                    runtime_hooks.stall_trace(
                        "exchange_eval",
                        cmd=req_cmd,
                        loop=int(loop_count),
                        groups=int(len(eval_groups)),
                        items=int(merged_items),
                        target_items=int(dynamic_target_eval_items),
                        collect_wait_s=float(collect_wait_s),
                        collect_timeout_s=float(dynamic_collect_timeout_s),
                        eval_s=0.0,
                        pipelined=True,
                    )
                else:
                    t_eval = time.perf_counter()
                    responses = runtime_hooks.run_batched_eval_groups(eval_groups, model, device, cfg)
                    eval_elapsed = time.perf_counter() - t_eval
                    duty["model_s"] += eval_elapsed
                    if perf_stats is not None:
                        perf_stats["model_time_s"] += float(eval_elapsed)
                    runtime_hooks.stall_trace(
                        "exchange_eval",
                        cmd=req_cmd,
                        loop=int(loop_count),
                        groups=int(len(eval_groups)),
                        items=int(merged_items),
                        target_items=int(dynamic_target_eval_items),
                        collect_wait_s=float(collect_wait_s),
                        collect_timeout_s=float(dynamic_collect_timeout_s),
                        eval_s=float(eval_elapsed),
                    )
                    write_t0 = time.perf_counter()
                    for response_group in responses:
                        runtime_hooks.write_batched_eval_group(proc, response_group)
                    duty["write_s"] += time.perf_counter() - write_t0

                if perf_stats is not None:
                    perf_stats["model_calls"] += 1
                    perf_stats["model_batch_sizes"].append(merged_items)
                duty["cycles"] += 1
                if duty["cycles"] % duty_log_interval == 0:
                    runtime_hooks.emit_duty_cycle(duty)
        finally:
            if inflight and pipeline is not None:
                try:
                    drain = _supervised_pipeline_collect(
                        pipeline,
                        proc=proc,
                        timeout_s=10.0,
                        label=f"{req_cmd} pipeline drain",
                    )
                    for rg in drain:
                        runtime_hooks.write_batched_eval_group(proc, rg)
                except Exception:
                    pass
            if pipeline is not None:
                pipeline.stop()

        if duty["cycles"] > 0:
            runtime_hooks.emit_duty_cycle(duty)
        if perf_stats is not None and results_payload is not None:
            perf_stats["result_messages"] += 1
        runtime_hooks.stall_trace(
            "exchange_end",
            cmd=req_cmd,
            loops=int(loop_count),
            elapsed_s=float(time.perf_counter() - req_t0),
        )
        return results_payload

    proc = proc_pool.acquire(1)[0] if proc_pool is not None else runtime_hooks.launch_server(rust_binary)
    if active_proc_ref is not None:
        active_proc_ref._active_proc = proc
    try:
        slot_count = min(max(1, parallel), max(1, n_games))
        with tqdm(total=n_games, desc="Self-play (Rust+NN batched)", leave=False, disable=not show_progress) as pbar:
            game_data = [None] * slot_count
            games_started = 0

            def launch_slot(gi):
                nonlocal games_started
                if games_started >= n_games:
                    game_data[gi] = None
                    return False
                game_data[gi] = build_game_data()
                games_started += 1
                return True

            def finalize_slot(gi):
                nonlocal games_done
                gd = game_data[gi]
                if gd is None:
                    return
                if gd.get("void_result", False):
                    all_states.append([])
                    all_policies.append([])
                else:
                    all_states.append(gd["states"])
                    all_policies.append(gd["policies"])
                all_outcomes.append(float(gd.get("winner", 0.0)))
                all_traces.append(gd.get("trace", []))
                pbar.update(1)
                games_done += 1
                if not launch_slot(gi):
                    game_data[gi] = None

            for gi in range(slot_count):
                launch_slot(gi)

            search_opts = runtime_hooks.rust_search_options(cfg, penalty_mode=penalty_mode)
            base_collect_timeout_s = min(
                0.006,
                max(0.00075, float(search_opts.get("batch_timeout_us", 1500)) / 1_000_000.0 * 0.9),
            )
            base_target_eval_items = max(1, int(search_opts.get("batch_size", cfg.get("batch_size", 8))))
            batch_items_ema = float(base_target_eval_items)
            collect_wait_ema_s = 0.0
            if use_resident_session:
                session_req = {
                    "cmd": "search_nn_multi_session_open",
                    "game": rust_game,
                    "iters": iters,
                    "jobs": [build_job(gd) for gd in game_data if gd is not None],
                }
                session_req.update(search_opts)
                results_payload = exchange_search_request(session_req)
                if isinstance(results_payload, dict) and results_payload.get("error"):
                    raise RuntimeError(str(results_payload.get("error")))
                session_id = results_payload.get("session_id") if isinstance(results_payload, dict) else None
                if session_id is None:
                    raise RuntimeError("resident self-play session open returned no session_id")
                results = results_payload.get("results", []) if isinstance(results_payload, dict) else []

                while games_done < n_games:
                    if not isinstance(results, list) or len(results) != slot_count:
                        raise RuntimeError(
                            "resident self-play result length mismatch: "
                            f"expected {slot_count} got {len(results) if isinstance(results, list) else 'non-list'}"
                        )

                    updates = []
                    for gi, result in enumerate(results):
                        gd = game_data[gi]
                        if gd is None:
                            updates.append({"deactivate": True})
                            continue
                        chosen = apply_result(gd, result if isinstance(result, dict) else {})
                        if gd["moves"] >= max_moves and not gd["finished"]:
                            gd["finished"] = True
                            gd["winner"] = 0.0
                        if gd["finished"]:
                            finalize_slot(gi)
                            gd = game_data[gi]
                            if gd is None:
                                updates.append({"deactivate": True})
                            else:
                                updates.append({"replace": build_job(gd)})
                        else:
                            updates.append({"action": int(chosen)} if chosen is not None else {})

                    if games_done >= n_games and all(gd is None for gd in game_data):
                        break
                    if session_id is None:
                        break
                    results_payload = exchange_search_request(
                        {
                            "cmd": "search_nn_multi_session_step",
                            "session_id": int(session_id),
                            "updates": updates,
                        }
                    )
                    if isinstance(results_payload, dict) and results_payload.get("error"):
                        raise RuntimeError(str(results_payload.get("error")))
                    results = results_payload.get("results", []) if isinstance(results_payload, dict) else []

                if session_id is not None:
                    try:
                        runtime_hooks.proc_write_json_line(
                            proc, {"cmd": "search_nn_multi_session_close", "session_id": int(session_id)}
                        )
                        runtime_hooks.proc_read_json_line(proc)
                    except Exception:
                        pass
            else:
                while games_done < n_games:
                    active = []
                    jobs = []
                    for gi, gd in enumerate(game_data):
                        if gd is None:
                            continue
                        if gd["finished"] or gd["moves"] >= max_moves:
                            if gd["moves"] >= max_moves and not gd["finished"]:
                                gd["finished"] = True
                                gd["winner"] = 0.0
                            finalize_slot(gi)
                            gd = game_data[gi]
                        if gd is None:
                            continue
                        active.append(gi)
                        jobs.append(build_job(gd))

                    if not active:
                        continue

                    req = {
                        "cmd": "search_nn_multi",
                        "game": rust_game,
                        "iters": iters,
                        "jobs": jobs,
                    }
                    req.update(search_opts)
                    results_payload = exchange_search_request(req)
                    if isinstance(results_payload, dict) and results_payload.get("error"):
                        raise RuntimeError(str(results_payload.get("error")))
                    results = []
                    if isinstance(results_payload, dict):
                        results = results_payload.get("results", [])
                    if not isinstance(results, list) or len(results) != len(active):
                        raise RuntimeError(
                            "self-play result length mismatch: "
                            f"expected {len(active)} got {len(results) if isinstance(results, list) else 'non-list'}"
                        )
                    for gi, result in zip(active, results):
                        gd = game_data[gi]
                        if gd is not None:
                            apply_result(gd, result if isinstance(result, dict) else {})
    finally:
        if proc_pool is None:
            runtime_hooks.stop_server(proc)

    return all_states, all_policies, all_outcomes, all_traces


def _summarize_selfplay_perf_stats(stats):
    stats = stats or {}
    batch_sizes = [float(v) for v in (stats.get("model_batch_sizes") or [])]
    model_calls = int(stats.get("model_calls") or 0)
    eval_items = int(stats.get("eval_items") or 0)
    eval_messages = int(stats.get("eval_messages") or 0)
    collect_wait_s = float(stats.get("collect_wait_s") or 0.0)
    model_time_s = float(stats.get("model_time_s") or 0.0)
    return {
        "eval_messages": eval_messages,
        "eval_items": eval_items,
        "model_calls": model_calls,
        "mean_model_batch_size": (
            float(sum(batch_sizes) / len(batch_sizes)) if batch_sizes else None
        ),
        "max_model_batch_size": float(max(batch_sizes)) if batch_sizes else None,
        "eval_items_per_message": (
            float(eval_items / eval_messages) if eval_messages else None
        ),
        "collect_wait_s": collect_wait_s,
        "mean_collect_wait_ms": (
            float(1000.0 * collect_wait_s / model_calls) if model_calls else None
        ),
        "model_time_s": model_time_s,
        "mean_model_time_ms": (
            float(1000.0 * model_time_s / model_calls) if model_calls else None
        ),
        "collect_loops": int(stats.get("collect_loops") or 0),
        "result_messages": int(stats.get("result_messages") or 0),
    }


class SelfPlayWorker:
    """Background actor: Rust MCTS + batched NN eval."""

    BACKPRESSURE_RATIO = 0.8
    BACKPRESSURE_SLEEP = 0.5
    REPLAY_STALL_TIMEOUT_S = 45.0
    PAUSE_IDLE_TIMEOUT_S = 2.0
    PAUSE_CANCEL_TIMEOUT_S = 2.0
    PAUSE_KILL_TIMEOUT_S = 5.0

    def __init__(
            self,
            cfg,
            model,
            device,
            replay,
            rust_binary,
            *,
            server_pool_factory,
            clone_actor_model_fn,
            selfplay_runner,
            logger=None):
        self.cfg = cfg
        self.device = device
        self.replay = replay
        self.rust_binary = rust_binary
        self._server_pool_factory = server_pool_factory
        self._clone_actor_model = clone_actor_model_fn
        self._selfplay_runner = selfplay_runner
        self._logger = logger or logging.getLogger(__name__)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread = None
        self.games_generated = 0
        self.positions_generated = 0
        self._prev_count = 0
        self._cycles = 0
        self._total_time = 0.0
        self._backpressure_waits = 0
        self._last_cycle_s = 0.0
        self._last_cycle_positions = 0
        self._last_cycle_games = 0
        self._recent_chunks = deque(maxlen=8)
        self._proc_pool = self._server_pool_factory(self.rust_binary)
        self._model = self._clone_actor_model(model)
        self._actor_lock = threading.Lock()
        self._active_proc = None
        self._last_progress_ts = time.time()
        self._last_error = None
        self._consecutive_errors = 0
        self._last_plan = None
        # Monotonically increasing counter identifying which learner checkpoint
        # the current `self._model` clone was derived from. Iteration 0 is the
        # initial clone; each successful `update_model()` increments. Replay
        # samples produced by this worker inherit this value in their metadata,
        # so downstream analysis can trace an arena outcome back to the actor
        # identity that produced its training samples.
        self._actor_generation = 0
        self._actor_id = "actor_gen_000000"

    def update_model(self, model):
        next_model = self._clone_actor_model(model)
        with self._actor_lock:
            self._model = next_model
            self._actor_generation += 1
            self._actor_id = f"actor_gen_{self._actor_generation:06d}"

    @property
    def actor_generation(self) -> int:
        with self._actor_lock:
            return self._actor_generation

    def actor_snapshot(self):
        with self._actor_lock:
            return self._model, self._actor_generation, self._actor_id

    def _cancel_active_search(self, *, kill_proc=False):
        proc = self._active_proc
        if proc is None:
            return False
        cancelled = False
        ring = getattr(proc, "_quartz_ring_buffer", None)
        if ring is not None:
            try:
                ring.request_cancel()
                cancelled = True
            except Exception:
                pass
        if kill_proc:
            try:
                proc.kill()
                cancelled = True
            except Exception:
                pass
        return cancelled

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._pause.clear()
        self._cancel_active_search()
        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self._cancel_active_search(kill_proc=True)
                if hasattr(self._proc_pool, "kill_active"):
                    self._proc_pool.kill_active()
                self._thread.join(timeout=5)
            if self._thread.is_alive():
                self._logger.warning("SelfPlayWorker did not stop within timeout")
        self._proc_pool.close()

    def pause(self, wait=True):
        self._pause.set()
        if not wait:
            return True
        if self._idle.wait(timeout=self.PAUSE_IDLE_TIMEOUT_S):
            return True
        self._cancel_active_search()
        if self._idle.wait(timeout=self.PAUSE_CANCEL_TIMEOUT_S):
            return True
        self._cancel_active_search(kill_proc=True)
        if hasattr(self._proc_pool, "kill_active"):
            self._proc_pool.kill_active()
        if self._idle.wait(timeout=self.PAUSE_KILL_TIMEOUT_S):
            return True
        self._last_error = "background self-play failed to become idle after pause request"
        self._logger.warning(self._last_error)
        return False

    def resume(self):
        self._last_error = None
        self._consecutive_errors = 0
        self._pause.clear()

    def telemetry(self):
        rolling_positions = sum(chunk["positions"] for chunk in self._recent_chunks)
        rolling_games = sum(chunk["games"] for chunk in self._recent_chunks)
        rolling_time = sum(chunk["elapsed_s"] for chunk in self._recent_chunks)
        rolling_positions_per_s = rolling_positions / max(rolling_time, 1e-6)
        mean_chunk_positions = (
            rolling_positions / max(len(self._recent_chunks), 1) if self._recent_chunks else 0.0
        )
        peak_chunk_positions = max((chunk["positions"] for chunk in self._recent_chunks), default=0)
        burst_ratio = peak_chunk_positions / max(mean_chunk_positions, 1.0) if mean_chunk_positions > 0.0 else 1.0
        perf_chunks = [chunk.get("perf") or {} for chunk in self._recent_chunks]
        perf_model_calls = sum(int(chunk.get("model_calls") or 0) for chunk in perf_chunks)
        perf_eval_messages = sum(int(chunk.get("eval_messages") or 0) for chunk in perf_chunks)
        perf_eval_items = sum(int(chunk.get("eval_items") or 0) for chunk in perf_chunks)
        perf_collect_wait_s = sum(float(chunk.get("collect_wait_s") or 0.0) for chunk in perf_chunks)
        perf_model_time_s = sum(float(chunk.get("model_time_s") or 0.0) for chunk in perf_chunks)
        weighted_batch_sum = sum(
            float(chunk.get("mean_model_batch_size") or 0.0) * int(chunk.get("model_calls") or 0)
            for chunk in perf_chunks
        )
        return {
            "games": self.games_generated,
            "positions": self.positions_generated,
            "cycles": self._cycles,
            "avg_cycle_s": round(self._total_time / max(self._cycles, 1), 3),
            "last_cycle_s": round(self._last_cycle_s, 3),
            "last_cycle_positions": self._last_cycle_positions,
            "last_cycle_games": self._last_cycle_games,
            "rolling_cycle_s": round(rolling_time / max(len(self._recent_chunks), 1), 3),
            "rolling_positions_per_s": round(rolling_positions_per_s, 3),
            "rolling_positions": int(rolling_positions),
            "rolling_games": int(rolling_games),
            "burst_ratio": round(burst_ratio, 3),
            "backpressure_waits": self._backpressure_waits,
            "worker_alive": bool(self._thread.is_alive()) if self._thread is not None else False,
            "paused": bool(self._pause.is_set()),
            "idle": bool(self._idle.is_set()),
            "last_progress_age_s": round(max(0.0, time.time() - self._last_progress_ts), 3),
            "last_error": self._last_error,
            "consecutive_errors": self._consecutive_errors,
            "last_plan": self._last_plan,
            "actor_generation": self.actor_generation,
            "actor_id": self.actor_snapshot()[2],
            "inference": {
                "chunks": int(len(perf_chunks)),
                "eval_messages": int(perf_eval_messages),
                "eval_items": int(perf_eval_items),
                "model_calls": int(perf_model_calls),
                "mean_model_batch_size": (
                    round(weighted_batch_sum / perf_model_calls, 3)
                    if perf_model_calls
                    else None
                ),
                "eval_items_per_message": (
                    round(perf_eval_items / perf_eval_messages, 3)
                    if perf_eval_messages
                    else None
                ),
                "collect_wait_s": round(perf_collect_wait_s, 6),
                "mean_collect_wait_ms": (
                    round(1000.0 * perf_collect_wait_s / perf_model_calls, 3)
                    if perf_model_calls
                    else None
                ),
                "model_time_s": round(perf_model_time_s, 6),
                "mean_model_time_ms": (
                    round(1000.0 * perf_model_time_s / perf_model_calls, 3)
                    if perf_model_calls
                    else None
                ),
            },
            "server_pool": self._proc_pool.snapshot() if hasattr(self._proc_pool, "snapshot") else None,
            "path": "rust+nn",
        }

    def status(self):
        return {
            "alive": bool(self._thread.is_alive()) if self._thread is not None else False,
            "paused": bool(self._pause.is_set()),
            "idle": bool(self._idle.is_set()),
            "last_progress_age_s": max(0.0, time.time() - self._last_progress_ts),
            "last_error": self._last_error,
            "consecutive_errors": self._consecutive_errors,
            "actor_generation": self.actor_generation,
            "actor_id": self.actor_snapshot()[2],
            "server_pool": self._proc_pool.snapshot() if hasattr(self._proc_pool, "snapshot") else None,
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                if self._pause.is_set():
                    self._idle.set()
                    time.sleep(0.1)
                    continue
                if hasattr(self.replay.buf, "maxlen") and self.replay.buf.maxlen:
                    fill = len(self.replay) / self.replay.buf.maxlen
                    if fill > self.BACKPRESSURE_RATIO:
                        self._backpressure_waits += 1
                        self._idle.set()
                        time.sleep(self.BACKPRESSURE_SLEEP)
                        continue

                t0 = time.time()
                n_new = 0
                plan = plan_selfplay_runner_chunk(self.cfg, len(self.replay), self._recent_chunks)
                self._last_plan = dict(plan)
                batch_games = int(plan["batch_games"])
                parallel = int(plan["parallel"])
                remaining = batch_games
                while remaining > 0 and not self._stop.is_set():
                    if self._pause.is_set():
                        break
                    chunk_t0 = time.time()
                    chunk_games = min(remaining, int(plan.get("games_per_call", parallel)))
                    streamed_positions = 0

                    # Capture one immutable actor snapshot for the chunk. The
                    # runner receives the same model object whose id/generation
                    # are stamped into replay metadata, even if update_model()
                    # fires concurrently while the Rust search is in flight.
                    chunk_model, chunk_actor_generation, chunk_actor_id = self.actor_snapshot()

                    def _on_game_stream(
                        gs,
                        gp,
                        out,
                        traces,
                        chunk_actor_generation=chunk_actor_generation,
                        chunk_actor_id=chunk_actor_id,
                    ):
                        nonlocal n_new, streamed_positions
                        self.replay.add_game(
                            gs, gp, out, traces=traces,
                            actor_generation=chunk_actor_generation,
                            actor_id=chunk_actor_id,
                        )
                        n_new += len(gs)
                        streamed_positions += len(gs)
                        self._last_progress_ts = time.time()
                        self._last_error = None
                        self._consecutive_errors = 0

                    self._idle.clear()
                    try:
                        chunk_perf_stats = {}
                        states, policies, outcomes, traces = self._selfplay_runner(
                            self.cfg,
                            chunk_model,
                            self.device,
                            chunk_games,
                            self.rust_binary,
                            parallel=min(parallel, chunk_games),
                            show_progress=False,
                            proc_pool=self._proc_pool,
                            perf_stats=chunk_perf_stats,
                            on_game=_on_game_stream if self.cfg.get("_selfplay_runner_mode") == "rust_selfplay_state_machine" else None,
                            active_proc_ref=self,
                        )
                    finally:
                        self._active_proc = None
                        self._idle.set()
                    chunk_positions = streamed_positions
                    if self.cfg.get("_selfplay_runner_mode") != "rust_selfplay_state_machine":
                        for gs, gp, out, trace in zip(states, policies, outcomes, traces):
                            self.replay.add_game(
                                gs, gp, out, traces=trace,
                                actor_generation=chunk_actor_generation,
                                actor_id=chunk_actor_id,
                            )
                            n_new += len(gs)
                            chunk_positions += len(gs)
                    if chunk_positions > 0:
                        self._last_progress_ts = time.time()
                        self._last_error = None
                        self._consecutive_errors = 0
                    elif chunk_games > 0:
                        raise RuntimeError(
                            f"self-play chunk produced no positions (games={chunk_games}, parallel={min(parallel, chunk_games)})"
                        )
                    chunk_elapsed = max(time.time() - chunk_t0, 1e-6)
                    self.games_generated += len(states)
                    self.positions_generated += chunk_positions
                    self._recent_chunks.append({
                        "games": int(len(states)),
                        "positions": int(chunk_positions),
                        "elapsed_s": float(chunk_elapsed),
                        "perf": _summarize_selfplay_perf_stats(chunk_perf_stats),
                    })
                    remaining -= chunk_games
                self._cycles += 1
                cycle_s = time.time() - t0
                self._total_time += cycle_s
                self._last_cycle_s = cycle_s
                self._last_cycle_positions = n_new
                self._last_cycle_games = batch_games
            except Exception as exc:
                self._idle.set()
                self._active_proc = None
                cancelled = isinstance(exc, InterruptedError) and (self._pause.is_set() or self._stop.is_set())
                if cancelled:
                    self._last_error = None
                    self._consecutive_errors = 0
                    continue
                self._last_error = str(exc)
                self._consecutive_errors += 1
                if hasattr(self._proc_pool, "record_failure"):
                    self._proc_pool.record_failure(str(exc))
                if hasattr(self._proc_pool, "kill_active"):
                    self._proc_pool.kill_active()
                if self._consecutive_errors <= 3:
                    self._logger.exception("SelfPlayWorker error (%s): %r", type(exc).__name__, exc)
                time.sleep(min(self._consecutive_errors, 5))


def rust_game_name(game_name, game_configs, gomoku15_variants):
    if game_name in game_configs:
        return game_name
    if game_name in gomoku15_variants:
        return game_name
    return "gomoku15"


def is_chess_game(game_name):
    return game_name in {"chess", "chess960"}


def is_go_game(game_name):
    return bool(game_name) and game_name.startswith("go") and len(game_name) > 2 and game_name[2].isdigit()


def rust_search_options(cfg, penalty_mode=None):
    raw_n_threads = cfg.get("n_threads", 1)
    thread_request = raw_n_threads
    thread_budget = None
    if isinstance(raw_n_threads, str):
        thread_request = raw_n_threads.strip().lower() or "1"
        if thread_request in {"auto", "throughput", "auto-throughput", "quality", "auto-quality"}:
            cap = (
                cfg.get("thread_cap")
                or cfg.get("max_threads")
                or cfg.get("n_threads_cap")
                or os.cpu_count()
                or 1
            )
            thread_budget = max(1, int(cap))
        else:
            thread_budget = max(1, int(thread_request))
            thread_request = thread_budget
    else:
        thread_budget = max(1, int(raw_n_threads or 1))
        thread_request = thread_budget
    n_threads = thread_budget
    batch_size = int(cfg.get("batch_size", 8) or 8)
    batch_timeout_us = int(
        cfg.get(
            "batch_timeout_us",
            1500 if n_threads <= 1 else min(6000, 1200 + 250 * max(n_threads, batch_size // 2)),
        )
        or 1500
    )
    return {
        "search_profile": cfg.get("search_profile", "quartz"),
        "penalty_mode": penalty_mode or cfg.get("penalty_mode", "GatedRefresh"),
        "hbar_penalty_cap": cfg.get("hbar_penalty_cap", 0.3),
        "sigma_0": cfg.get("sigma_0", 0.3),
        "min_visits": cfg.get("min_visits", 50),
        "check_interval": cfg.get("check_interval", 100),
        "prior_refresh_rate": cfg.get("prior_refresh_rate", 0.0),
        "prior_refresh_temp": cfg.get("prior_refresh_temp", 1.0),
        "c_puct": cfg.get("c_puct", 0.0),
        "n_threads": thread_request,
        "batch_size": batch_size,
        "batch_timeout_us": batch_timeout_us,
        **({"seed": int(cfg["seed"])} if "seed" in cfg and cfg.get("seed") is not None else {}),
        **({"root_only_shaping": bool(cfg["root_only_shaping"])} if "root_only_shaping" in cfg else {}),
        **({"vl_mode": cfg["vl_mode"]} if "vl_mode" in cfg else {}),
        **({"tt_enabled": bool(cfg["tt_enabled"])} if "tt_enabled" in cfg else {}),
        **({"thread_policy": str(cfg["thread_policy"])} if cfg.get("thread_policy") is not None else {}),
        **({"auto_thread_policy": str(cfg["auto_thread_policy"])} if cfg.get("auto_thread_policy") is not None else {}),
        **({"thread_cap": int(cfg["thread_cap"])} if cfg.get("thread_cap") is not None else {}),
        **({"max_threads": int(cfg["max_threads"])} if cfg.get("max_threads") is not None else {}),
        **({"n_threads_cap": int(cfg["n_threads_cap"])} if cfg.get("n_threads_cap") is not None else {}),
        # P7 (audit W2): forward `halt_mode` so mcts_server's
        # `parse_halt_mode_override` can pin HaltMode::Fixed for
        # attribution rows.
        **({"halt_mode": str(cfg["halt_mode"])} if cfg.get("halt_mode") is not None else {}),
    }


def chess_state_meta_from_hashes(history_hashes):
    hashes = []
    for value in history_hashes or []:
        try:
            hashes.append(int(value))
        except (TypeError, ValueError):
            continue
    return {"chess_history_hashes": hashes} if hashes else {}


def build_rust_state_meta(game_name, state, cfg, is_chess_game_fn=None, is_go_game_fn=None):
    is_chess_game_fn = is_chess_game if is_chess_game_fn is None else is_chess_game_fn
    is_go_game_fn = is_go_game if is_go_game_fn is None else is_go_game_fn
    if is_chess_game_fn(game_name) and state is not None:
        return chess_state_meta_from_hashes(getattr(state, "_chess_history_hashes", None))
    if is_go_game_fn(game_name) and state is not None:
        return {
            "go_ruleset": cfg.get("go_ruleset", "chinese"),
            "go_scoring": cfg.get("go_scoring", "area"),
            "go_komi": float(cfg.get("go_komi", 7.5)),
            "go_allow_suicide": bool(cfg.get("go_allow_suicide", False)),
            "passes": int(getattr(state, "_passes", 0)),
            "ko_point": int(getattr(state, "_ko_point", -1) if getattr(state, "_ko_point", None) is not None else -1),
            "black_caps": int(getattr(state, "_black_caps", 0)),
            "white_caps": int(getattr(state, "_white_caps", 0)),
        }
    return {}


def chess960_start_fen(index):
    if index < 0 or index >= 960:
        raise ValueError("Chess960 position index must be in [0, 959]")
    back = [0] * 8
    n = int(index)
    lb = (n % 4) * 2 + 1
    n //= 4
    back[lb] = 3
    db = (n % 4) * 2
    n //= 4
    back[db] = 3
    q_idx = n % 6
    n //= 6
    empty_idx = 0
    for i in range(8):
        if back[i] == 0:
            if empty_idx == q_idx:
                back[i] = 5
                break
            empty_idx += 1
    knight_table = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    kn1, kn2 = knight_table[n]
    empties = [i for i, piece in enumerate(back) if piece == 0]
    back[empties[kn1]] = 2
    back[empties[kn2]] = 2
    remaining = [i for i, piece in enumerate(back) if piece == 0]
    back[remaining[0]] = 4
    back[remaining[1]] = 6
    back[remaining[2]] = 4
    piece_map = {2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
    white_back = "".join(piece_map[piece] for piece in back)
    black_back = white_back.lower()
    rights = (
        chr(ord("A") + remaining[2])
        + chr(ord("A") + remaining[0])
        + chr(ord("a") + remaining[2])
        + chr(ord("a") + remaining[0])
    )
    return f"{black_back}/pppppppp/8/8/8/8/PPPPPPPP/{white_back} w {rights} - 0 1"


def initial_chess_fen(
    cfg,
    rng=None,
    standard_chess_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
):
    if not cfg.get("chess960", False):
        return standard_chess_fen
    index = cfg.get("chess960_index")
    if index is None:
        picker = rng if rng is not None else random
        index = picker.randrange(960)
    return chess960_start_fen(int(index))


def encode_chess_fen(fen):
    enc = np.zeros((36, 8, 8), dtype=np.float32)
    parts = fen.split()
    if len(parts) < 4:
        return enc
    board_part, side_part, castling_part, ep_part = parts[:4]
    is_white = side_part != "b"
    white_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    black_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
    rank = 7
    file = 0
    white_king_file = 4
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        elif ch in white_map:
            plane = white_map[ch] if is_white else (white_map[ch] + 6)
            enc[plane, rank, file] = 1.0
            if ch == "K":
                white_king_file = file
            file += 1
        elif ch in black_map:
            plane = (black_map[ch] + 6) if is_white else black_map[ch]
            enc[plane, rank, file] = 1.0
            file += 1
    if is_white:
        enc[28] = 1.0
    if len(parts) >= 6:
        try:
            fullmove = int(parts[5])
            enc[29] = min(fullmove / 200.0, 1.0)
        except ValueError:
            pass
    for ch in castling_part:
        if ch == "K":
            enc[30] = 1.0
        elif ch == "Q":
            enc[31] = 1.0
        elif ch == "k":
            enc[32] = 1.0
        elif ch == "q":
            enc[33] = 1.0
        elif "A" <= ch <= "H":
            rook_file = ord(ch) - ord("A")
            enc[30 if rook_file > white_king_file else 31] = 1.0
        elif "a" <= ch <= "h":
            rook_file = ord(ch) - ord("a")
            enc[32 if rook_file > white_king_file else 33] = 1.0
    if len(parts) >= 5:
        try:
            half = int(parts[4])
            enc[34] = min(half / 100.0, 1.0)
        except ValueError:
            pass
    if ep_part != "-" and len(ep_part) >= 2:
        ep_file = ord(ep_part[0]) - ord("a")
        ep_rank = ord(ep_part[1]) - ord("1")
        if 0 <= ep_file < 8 and 0 <= ep_rank < 8:
            enc[35, ep_rank, ep_file] = 1.0
    return enc


@dataclass(frozen=True)
class LegacyRustSelfplayHooks:
    launch_rust_server: object
    proc_write_json_line: object
    proc_read_json_line: object
    json_loads_fast: object
    rust_game_name: object
    rust_search_options: object
    is_go_game: object
    tqdm_factory: object


def selfplay_rust(cfg, n_games, rust_binary="./target/release/mcts_demo", runtime_hooks: LegacyRustSelfplayHooks | None = None):
    if runtime_hooks is None:
        raise RuntimeError("runtime_hooks required for selfplay_rust")
    rust_game = runtime_hooks.rust_game_name(cfg["_name"])
    all_trajectories = []
    remaining = n_games
    try:
        proc = runtime_hooks.launch_rust_server(rust_binary)
    except FileNotFoundError:
        print(f"  [WARN] Rust binary not found: {rust_binary}", file=sys.stderr)
        return [], n_games
    except RuntimeError as e:
        print(f"  [WARN] {e}", file=sys.stderr)
        return [], n_games

    with runtime_hooks.tqdm_factory(total=n_games, desc="Self-play (Rust)", leave=False) as pbar:
        while remaining > 0:
            batch = min(remaining, 5)
            req_dict = {
                "cmd": "selfplay",
                "game": rust_game,
                "iters": cfg["iters"],
                "n_games": batch,
                "temp_threshold": cfg["temp_th"],
            }
            if runtime_hooks.is_go_game(cfg.get("_name", "")):
                req_dict.update({
                    "go_ruleset": cfg.get("go_ruleset", "chinese"),
                    "go_scoring": cfg.get("go_scoring", "area"),
                    "go_komi": float(cfg.get("go_komi", 7.5)),
                    "go_allow_suicide": bool(cfg.get("go_allow_suicide", False)),
                })
            elif cfg.get("chess960", False):
                if cfg.get("chess960_index") is not None:
                    req_dict["chess960_index"] = int(cfg["chess960_index"])
                else:
                    req_dict["chess960_random_start"] = True
            req_dict.update(runtime_hooks.rust_search_options(cfg))
            try:
                runtime_hooks.proc_write_json_line(proc, req_dict)
                line = runtime_hooks.proc_read_json_line(proc)
                if line:
                    games = runtime_hooks.json_loads_fast(line)
                    for g in games:
                        all_trajectories.append(g)
                        pbar.update(1)
                    remaining -= batch
                else:
                    print("  [WARN] Rust server returned empty, falling back", file=sys.stderr)
                    break
            except (json.JSONDecodeError, BrokenPipeError, OSError) as e:
                print(f"  [WARN] Rust server error ({e}), falling back", file=sys.stderr)
                break
    try:
        runtime_hooks.proc_write_json_line(proc, {"cmd": "quit"})
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return all_trajectories, remaining


def should_use_resident_session(game_name, parallel, n_games, enabled=False):
    if not enabled:
        return False
    return int(parallel) > 1 and int(n_games) > 1


def supports_rust_eval_state_machine(game_name, rust_game_name, game_configs, gomoku15_variants, is_chess_game, is_go_game):
    rg = rust_game_name(game_name)
    return rg in game_configs or rg in gomoku15_variants or is_chess_game(rg) or is_go_game(rg)


def supports_rust_selfplay_state_machine(game_name, rust_game_name, game_configs, gomoku15_variants, is_chess_game, is_go_game):
    rg = rust_game_name(game_name)
    return rg in game_configs or rg in gomoku15_variants or is_chess_game(rg) or is_go_game(rg)


__all__ = [
    "ArenaRuntimeHooks",
    "BatchedSelfPlayRuntimeHooks",
    "LegacyRustSelfplayHooks",
    "NNSearchClient",
    "RustServerPool",
    "SearchClientRuntimeHooks",
    "SelfPlayLoopRuntimeHooks",
    "SelfPlayWorker",
    "arena_rust_nn_impl",
    "choose_selfplay_move",
    "compute_train_steps",
    "decode_streamed_selfplay_game",
    "default_output_dir",
    "build_rust_state_meta",
    "chess960_start_fen",
    "chess_state_meta_from_hashes",
    "encode_board_with_history",
    "encode_chess_fen",
    "estimate_selfplay_positions_per_game",
    "initial_chess_fen",
    "initial_replay_fill_target",
    "is_chess_game",
    "is_go_game",
    "plan_selfplay_runner_chunk",
    "rust_game_name",
    "rust_search_options",
    "selfplay_rust",
    "selfplay_rust_nn",
    "selfplay_rust_nn_batched",
    "should_use_resident_session",
    "supports_rust_eval_state_machine",
    "supports_rust_selfplay_state_machine",
    "wait_for_worker_progress",
]
