"""Runtime helpers for evaluation engines backed by Rust search."""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass

from quartz import runtime_support


@dataclass(frozen=True)
class EvaluatorRuntimeHooks:
    search_client_cls: object
    is_chess_game: object
    build_rust_state_meta: object
    iter_sparse_policy_entries: object
    supports_rust_eval_state_machine: object
    stall_trace: object
    game_record_cls: object
    tally_match: object


def _default_runtime_hooks():
    return EvaluatorRuntimeHooks(
        search_client_cls=runtime_support.resolve_search_client_cls(),
        is_chess_game=runtime_support.is_chess_game,
        build_rust_state_meta=runtime_support.build_rust_state_meta,
        iter_sparse_policy_entries=__import__("quartz.replay", fromlist=["iter_sparse_policy_entries"]).iter_sparse_policy_entries,
        supports_rust_eval_state_machine=runtime_support.supports_rust_eval_state_machine,
        stall_trace=lambda *args, **kwargs: None,
        game_record_cls=runtime_support.GameRecord,
        tally_match=runtime_support.tally_match,
    )


def arena_eval_runtime_cfg(base_cfg, num_games):
    cfg = dict(base_cfg)
    if not bool(cfg.get("_arena_eval_topology_override", True)):
        return cfg
    total_games = max(1, int(num_games or 0))
    current_batch_size = max(1, int(cfg.get("batch_size", 8) or 8))
    current_timeout_us = int(cfg.get("batch_timeout_us", 1500) or 1500)
    n_threads = max(1, int(cfg.get("n_threads", 1) or 1))
    if total_games <= 2:
        cfg["batch_size"] = min(current_batch_size, 2)
        cfg["batch_timeout_us"] = min(current_timeout_us, 900 if n_threads <= 1 else 700)
        cfg["_arena_low_concurrency_profile"] = "tiny"
    elif total_games <= 4:
        cfg["batch_size"] = min(current_batch_size, 4)
        cfg["batch_timeout_us"] = min(current_timeout_us, 1200 if n_threads <= 1 else 900)
        cfg["_arena_low_concurrency_profile"] = "small"
    return cfg


def arena_search_manifest(cfg, num_games):
    return runtime_support.build_search_manifest(arena_eval_runtime_cfg(cfg, num_games))


def arena_search_manifest_hash(cfg, num_games):
    runtime_cfg = arena_eval_runtime_cfg(cfg, num_games)
    return runtime_support.search_manifest_hash(runtime_cfg)


def _encode_match_game_id(match_id, game_id):
    return f"{match_id}::{game_id}"


def _decode_match_game_id(game_id):
    text = str(game_id or "")
    if "::" not in text:
        return "match0", text
    return tuple(text.split("::", 1))


def _run_shared_eval_matches(
    *,
    shared_client,
    runtime_cfg,
    runtime_hooks,
    match_specs,
):
    is_chess_game = runtime_hooks.is_chess_game
    build_rust_state_meta = runtime_hooks.build_rust_state_meta
    supports_rust_eval_state_machine = runtime_hooks.supports_rust_eval_state_machine
    stall_trace = runtime_hooks.stall_trace
    GameRecord = runtime_hooks.game_record_cls
    tally_match = runtime_hooks.tally_match

    specs = [dict(spec) for spec in match_specs or []]
    if not specs:
        return {}

    game_name = runtime_cfg.get("_name")
    manifest_hash = runtime_support.search_manifest_hash(runtime_cfg)
    total_games = sum(max(0, int(spec.get("num_games", 0) or 0)) for spec in specs)
    progress_every = max(1, min(25, total_games // 10 if total_games > 0 else 1))
    completed_games = 0
    eval_loop_idx = 0
    stall_timeout_s = float(os.environ.get("QUARTZ_EVAL_STALL_TIMEOUT_S", "0") or 0.0)
    last_progress_sig = None
    last_progress_ts = time.time()
    runner_mode_used = None
    batch_id = f"eval_batch_{time.perf_counter_ns()}"

    sessions = []
    records = []
    records_by_match = {}
    spec_by_match_id = {}
    tag_to_engine_name = {}
    logger_by_match = {}

    for idx, spec in enumerate(specs):
        match_id = str(spec.get("match_id") or f"match{idx}")
        spec["match_id"] = match_id
        spec["num_games"] = max(0, int(spec.get("num_games", 0) or 0))
        spec["color_swap"] = bool(spec.get("color_swap", True))
        spec["max_moves"] = int(spec.get("max_moves", 500) or 500)
        spec["opening_book"] = list(spec.get("opening_book") or [])
        spec["rng"] = random.Random(spec.get("seed"))
        spec_by_match_id[match_id] = spec
        records_by_match[match_id] = []
        logger_by_match[match_id] = spec.get("logger")
        tag_to_engine_name[int(spec["tag_a"])] = spec["engine_a"].name()
        tag_to_engine_name[int(spec["tag_b"])] = spec["engine_b"].name()

    def manifest_fields_for_tags(_black_tag, _white_tag):
        return {
            "search_manifest_hash": manifest_hash,
            "search_manifest_hash_black": manifest_hash,
            "search_manifest_hash_white": manifest_hash,
        }

    def report_progress(force=False):
        if force or (completed_games > 0 and completed_games % progress_every == 0):
            print(f"  EvalProgress: {completed_games}/{total_games}", flush=True)

    def build_job(sess):
        game = sess["game"]
        mover_tag = sess["black_tag"] if game.current_player() == 0 else sess["white_tag"]
        player = 1 if game.current_player() == 0 else -1
        if is_chess_game(game_name):
            job = {
                "fen": getattr(game, "_fen", ""),
                "player": int(player),
                "model_tag": int(mover_tag),
            }
        else:
            job = {
                "board": list(getattr(game, "_board", [])),
                "player": int(player),
                "model_tag": int(mover_tag),
            }
        job.update(build_rust_state_meta(game_name, game, runtime_cfg))
        return job

    def apply_result(sess, result, fallback_ms):
        game = sess["game"]
        if not result or "error" in result:
            sess["error"] = result.get("error", "empty response") if isinstance(result, dict) else "empty response"
            sess["done"] = True
            return
        pol_entries = result.get("policy", [])
        if not pol_entries or game.is_terminal():
            sess["done"] = True
            return
        move_time_ms = float(result.get("time_used_ms", 0.0) or 0.0)
        sess["total_time_ms"] += move_time_ms if move_time_ms > 0.0 else fallback_ms
        action = int(result.get("best_move", 0))
        meta = {
            "time_used_ms": move_time_ms if move_time_ms > 0.0 else fallback_ms,
            "simulations": int(result.get("iterations", runtime_cfg.get("iters", 200)) or runtime_cfg.get("iters", 200)),
            "p_flip": result.get("p_flip", 0),
            "engine": "rust_nn_shared_eval",
        }
        if hasattr(game, "apply_engine_meta") and result.get("result_fen"):
            meta["result_fen"] = result.get("result_fen", "")
            meta["result_history_hashes"] = result.get("result_history_hashes", [])
        applied = False
        if hasattr(game, "apply_engine_meta"):
            applied = bool(game.apply_engine_meta(action, meta))
        if not applied:
            try:
                game.apply_move(action)
            except Exception as exc:
                sess["error"] = str(exc)
                sess["done"] = True
                return
        sess["ply"] += 1
        if game.is_terminal() or sess["ply"] >= sess["max_moves"]:
            sess["done"] = True

    def build_record(sess):
        game = sess["game"]
        if game.is_terminal():
            if hasattr(game, "is_void_result") and game.is_void_result():
                return GameRecord(
                    game_id=sess["game_id"],
                    engine_black=sess["eng_black"].name(),
                    engine_white=sess["eng_white"].name(),
                    outcome="void",
                    score_black=None,
                    move_count=sess["ply"],
                    total_time_ms=sess["total_time_ms"],
                    moves=[],
                    opening=sess["opening"],
                    seed=sess["seed"],
                    error=sess["error"],
                    is_void=True,
                    **manifest_fields_for_tags(sess["black_tag"], sess["white_tag"]),
                )
            outcome_for_black = float(game.outcome_for_black() or 0.0)
            if outcome_for_black > 0:
                outcome, score_black = "black_win", 1.0
            elif outcome_for_black < 0:
                outcome, score_black = "white_win", 0.0
            else:
                outcome, score_black = "draw", 0.5
            return GameRecord(
                game_id=sess["game_id"],
                engine_black=sess["eng_black"].name(),
                engine_white=sess["eng_white"].name(),
                outcome=outcome,
                score_black=score_black,
                move_count=sess["ply"],
                total_time_ms=sess["total_time_ms"],
                moves=[],
                opening=sess["opening"],
                seed=sess["seed"],
                error=sess["error"],
                is_void=bool(sess["error"]),
                **manifest_fields_for_tags(sess["black_tag"], sess["white_tag"]),
            )
        return GameRecord(
            game_id=sess["game_id"],
            engine_black=sess["eng_black"].name(),
            engine_white=sess["eng_white"].name(),
            outcome="draw",
            score_black=0.5,
            move_count=sess["ply"],
            total_time_ms=sess["total_time_ms"],
            moves=[],
            opening=sess["opening"],
            seed=sess["seed"],
            error=sess["error"],
            is_void=bool(sess["error"]),
            **manifest_fields_for_tags(sess["black_tag"], sess["white_tag"]),
        )

    def finish_records():
        nonlocal completed_games
        for sess in sessions:
            rec = build_record(sess)
            records.append(rec)
            records_by_match[sess["match_id"]].append(rec)
            pair_logger = logger_by_match.get(sess["match_id"])
            if pair_logger is not None:
                pair_logger.log(rec)
            completed_games += 1
            trace_kwargs = {
                "game": game_name,
                "completed_games": int(completed_games),
                "total_games": int(total_games),
                "move_count": int(sess["ply"]),
                "has_error": bool(sess["error"]),
                "search_manifest_hash": manifest_hash,
                "match_id": sess["match_id"],
            }
            if runner_mode_used:
                trace_kwargs["runner_mode"] = runner_mode_used
            stall_trace("eval_game_done", **trace_kwargs)
            report_progress()

    def run_shared_session(open_fn, step_fn, session_mode):
        nonlocal eval_loop_idx, last_progress_sig, last_progress_ts, runner_mode_used
        runner_mode_used = session_mode
        payload = open_fn(
            [build_job(sess) for sess in sessions],
            penalty_mode=runtime_cfg.get("penalty_mode", "GatedRefresh"),
        )
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        results = payload.get("results", []) if isinstance(payload, dict) else []
        stall_trace(
            "eval_session_open",
            game=game_name,
            num_games=int(total_games),
            session_id=int(session_id) if session_id is not None else None,
            result_count=int(len(results)) if isinstance(results, list) else None,
            runner_mode=session_mode,
        )

        while True:
            active = [
                sess
                for sess in sessions
                if not sess["done"] and not sess["game"].is_terminal() and sess["ply"] < sess["max_moves"]
            ]
            if not active:
                break
            eval_loop_idx += 1
            progress_sig = (
                int(len(active)),
                int(sum(sess["ply"] for sess in sessions)),
                int(sum(1 for sess in sessions if sess["done"])),
                int(sum(1 for sess in sessions if sess["error"])),
            )
            if progress_sig != last_progress_sig:
                last_progress_sig = progress_sig
                last_progress_ts = time.time()
            elif stall_timeout_s > 0.0 and (time.time() - last_progress_ts) > stall_timeout_s:
                stall_trace(
                    "eval_stall",
                    game=game_name,
                    loop=int(eval_loop_idx),
                    active_games=int(len(active)),
                    total_ply=int(sum(sess["ply"] for sess in sessions)),
                    done_games=int(sum(1 for sess in sessions if sess["done"])),
                    error_games=int(sum(1 for sess in sessions if sess["error"])),
                    session_id=int(session_id) if session_id is not None else None,
                    runner_mode=session_mode,
                )
                raise RuntimeError(
                    f"evaluation stalled for {time.time() - last_progress_ts:.1f}s "
                    f"(active={len(active)} ply={sum(sess['ply'] for sess in sessions)})"
                )
            stall_trace(
                "eval_loop",
                game=game_name,
                loop=int(eval_loop_idx),
                active_games=int(len(active)),
                done_games=int(sum(1 for sess in sessions if sess["done"])),
                total_ply=int(sum(sess["ply"] for sess in sessions)),
                session_mode=bool(session_id is not None),
                runner_mode=session_mode,
            )

            if not isinstance(results, list) or len(results) != len(sessions):
                for sess in active:
                    sess["error"] = (
                        f"shared eval session result length mismatch: expected {len(sessions)} got "
                        f"{len(results) if isinstance(results, list) else 'non-list'}"
                    )
                    sess["done"] = True
                break
            share_ms = 0.0
            updates = []
            for idx, sess in enumerate(sessions):
                if sess["done"] or sess["game"].is_terminal() or sess["ply"] >= sess["max_moves"]:
                    sess["done"] = True
                    updates.append({"deactivate": True})
                    continue
                apply_result(sess, results[idx], share_ms)
                if sess["done"]:
                    updates.append({"deactivate": True})
                else:
                    updates.append({"action": int(results[idx].get("best_move", 0))})
            payload = step_fn(session_id, updates)
            results = payload.get("results", []) if isinstance(payload, dict) else []
            stall_trace(
                "eval_session_step",
                game=game_name,
                loop=int(eval_loop_idx),
                session_id=int(session_id) if session_id is not None else None,
                updates=int(len(updates)),
                result_count=int(len(results)) if isinstance(results, list) else None,
                runner_mode=session_mode,
            )
        return session_id

    def append_session(spec, eng_black, eng_white, black_tag, white_tag, game_id, opening_idx=None):
        game = spec["game_factory"]()
        opening_applied = []
        game_seed = spec["rng"].randint(0, 2**31)
        opening_book = spec["opening_book"]
        if opening_idx is not None and opening_idx < len(opening_book):
            for action in opening_book[opening_idx]:
                if game.is_terminal() or action not in game.legal_moves():
                    break
                game.apply_move(action)
                opening_applied.append(action)
        sessions.append(
            {
                "game_id": _encode_match_game_id(spec["match_id"], game_id),
                "match_id": spec["match_id"],
                "game": game,
                "eng_black": eng_black,
                "eng_white": eng_white,
                "black_tag": int(black_tag),
                "white_tag": int(white_tag),
                "opening": opening_applied,
                "seed": game_seed,
                "ply": len(opening_applied),
                "total_time_ms": 0.0,
                "done": bool(game.is_terminal()) or len(opening_applied) >= spec["max_moves"],
                "error": None,
                "max_moves": int(spec["max_moves"]),
            }
        )

    for spec in specs:
        pairs = spec["num_games"] // 2 if spec["color_swap"] else 0
        opening_n = len(spec["opening_book"])
        for i in range(pairs):
            opening_idx = i % opening_n if opening_n else None
            append_session(spec, spec["engine_a"], spec["engine_b"], spec["tag_a"], spec["tag_b"], f"g{2 * i:04d}", opening_idx)
            append_session(spec, spec["engine_b"], spec["engine_a"], spec["tag_b"], spec["tag_a"], f"g{2 * i + 1:04d}", opening_idx)
        for idx in range(2 * pairs, spec["num_games"]):
            opening_idx = idx % opening_n if opening_n else None
            append_session(spec, spec["engine_a"], spec["engine_b"], spec["tag_a"], spec["tag_b"], f"g{idx:04d}", opening_idx)

    session_id = None
    batch_t0 = time.perf_counter()
    use_rust_eval_runner = (
        runtime_cfg.get("_eval_runner_mode") == "rust_eval_state_machine"
        and supports_rust_eval_state_machine(game_name)
    )
    try:
        if use_rust_eval_runner:
            try:
                runner_mode_used = "rust_eval_state_machine"
                runner_sessions = []
                for sess in sessions:
                    payload = build_job(sess)
                    payload.update(
                        {
                            "game_id": sess["game_id"],
                            "black_tag": int(sess["black_tag"]),
                            "white_tag": int(sess["white_tag"]),
                            "opening": list(sess["opening"]),
                            "seed": int(sess["seed"]),
                            "ply": int(sess["ply"]),
                            "done": bool(sess["done"]),
                            "total_time_ms": float(sess["total_time_ms"]),
                        }
                    )
                    runner_sessions.append(payload)
                stall_trace(
                    "eval_runner_start",
                    game=game_name,
                    num_games=int(total_games),
                    runner_mode=runner_mode_used,
                )
                payload = shared_client.eval_match_run(
                    runner_sessions,
                    max_moves=max(int(sess["max_moves"]) for sess in sessions),
                    penalty_mode=runtime_cfg.get("penalty_mode", "GatedRefresh"),
                )
                raw_records = payload.get("records", []) if isinstance(payload, dict) else []
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(str(payload.get("error")))
                if not isinstance(raw_records, list) or len(raw_records) != len(sessions):
                    raise RuntimeError(
                        f"rust eval runner record length mismatch: expected {len(sessions)} got "
                        f"{len(raw_records) if isinstance(raw_records, list) else 'non-list'}"
                    )
                for rec_data in raw_records:
                    match_id, _ = _decode_match_game_id(rec_data.get("game_id", ""))
                    if match_id not in records_by_match:
                        raise RuntimeError(f"unknown eval match id returned by runner: {match_id}")
                    black_tag = int(rec_data.get("black_tag", 0) or 0)
                    white_tag = (
                        int(rec_data["white_tag"])
                        if "white_tag" in rec_data and rec_data.get("white_tag") is not None
                        else 1
                    )
                    rec = GameRecord(
                        game_id=str(rec_data.get("game_id", "")),
                        engine_black=tag_to_engine_name.get(black_tag, str(black_tag)),
                        engine_white=tag_to_engine_name.get(white_tag, str(white_tag)),
                        outcome=str(rec_data.get("outcome", "draw")),
                        score_black=rec_data.get("score_black", 0.5),
                        move_count=int(rec_data.get("move_count", 0) or 0),
                        total_time_ms=float(rec_data.get("total_time_ms", 0.0) or 0.0),
                        moves=[],
                        opening=list(rec_data.get("opening", []) or []),
                        seed=rec_data.get("seed"),
                        error=rec_data.get("error"),
                        is_void=bool(rec_data.get("is_void", False)),
                        **manifest_fields_for_tags(black_tag, white_tag),
                    )
                    records.append(rec)
                    records_by_match[match_id].append(rec)
                    pair_logger = logger_by_match.get(match_id)
                    if pair_logger is not None:
                        pair_logger.log(rec)
                    completed_games += 1
                    stall_trace(
                        "eval_game_done",
                        game=game_name,
                        completed_games=int(completed_games),
                        total_games=int(total_games),
                        move_count=int(rec.move_count),
                        has_error=bool(rec.error),
                        runner_mode=runner_mode_used,
                        match_id=match_id,
                    )
                report_progress(force=True)
                batch_elapsed_s = time.perf_counter() - batch_t0
                results = {}
                for spec in specs:
                    tally = tally_match(records_by_match[spec["match_id"]], spec["engine_a"].name())
                    game_share = (float(spec["num_games"]) / float(total_games)) if total_games > 0 else 0.0
                    results[spec["match_id"]] = (
                        tally,
                        {
                            "runner_mode": runner_mode_used,
                            "match_elapsed_s": batch_elapsed_s * game_share,
                            "batch_id": batch_id,
                            "batch_elapsed_s": batch_elapsed_s,
                            "batch_total_games": int(total_games),
                            "errors": int(tally.errors),
                            "voids": int(tally.voids),
                            "scored_games": int(tally.scored),
                            "total_games": int(tally.total),
                        },
                    )
                return results
            except Exception as exc:
                records.clear()
                completed_games = 0
                for match_id in records_by_match:
                    records_by_match[match_id] = []
                stall_trace(
                    "eval_runner_fallback",
                    game=game_name,
                    num_games=int(total_games),
                    error=str(exc),
                )

        session_id = run_shared_session(
            shared_client.open_search_session,
            shared_client.step_search_session,
            "shared_client_session",
        )
        finish_records()
    finally:
        if session_id is not None:
            try:
                shared_client.close_search_session(session_id)
            except Exception:
                pass
    report_progress(force=True)
    batch_elapsed_s = time.perf_counter() - batch_t0
    results = {}
    for spec in specs:
        tally = tally_match(records_by_match[spec["match_id"]], spec["engine_a"].name())
        game_share = (float(spec["num_games"]) / float(total_games)) if total_games > 0 else 0.0
        results[spec["match_id"]] = (
            tally,
            {
                "runner_mode": runner_mode_used or "shared_client_session",
                "match_elapsed_s": batch_elapsed_s * game_share,
                "batch_id": batch_id,
                "batch_elapsed_s": batch_elapsed_s,
                "batch_total_games": int(total_games),
                "errors": int(tally.errors),
                "voids": int(tally.voids),
                "scored_games": int(tally.scored),
                "total_games": int(tally.total),
            },
        )
    return results


def _run_shared_eval_match(
    *,
    shared_client,
    runtime_cfg,
    runtime_hooks,
    engine_a,
    engine_b,
    tag_a,
    tag_b,
    game_factory,
    opening_book,
    num_games,
    color_swap=True,
    logger=None,
    max_moves=500,
    seed=None,
):
    results = _run_shared_eval_matches(
        shared_client=shared_client,
        runtime_cfg=runtime_cfg,
        runtime_hooks=runtime_hooks,
        match_specs=[
            {
                "match_id": "match0",
                "engine_a": engine_a,
                "engine_b": engine_b,
                "tag_a": int(tag_a),
                "tag_b": int(tag_b),
                "game_factory": game_factory,
                "opening_book": opening_book,
                "num_games": int(num_games),
                "color_swap": bool(color_swap),
                "logger": logger,
                "max_moves": int(max_moves),
                "seed": seed,
            }
        ],
    )
    return results["match0"]


class InferencePipelineThread:
    """Background thread that runs model inference while the caller collects the next batch."""

    def __init__(self, model, device, cfg, max_pending=1, run_batched_eval_groups_fn=None):
        self._model = model
        self._device = device
        self._cfg = cfg
        self._inbound = queue.Queue(maxsize=max_pending)
        self._outbound = queue.Queue(maxsize=max_pending)
        self._shutdown = threading.Event()
        self._thread = None
        self._run_batched_eval_groups_fn = run_batched_eval_groups_fn or (
            lambda groups, model_obj, dev, cfg_obj: __import__("quartz.eval_runtime", fromlist=["run_batched_eval_groups"]).run_batched_eval_groups(
                groups, model_obj, dev, cfg_obj, runtime_support.run_model_batch
            )
        )

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="quartz-inference")
        self._thread.start()

    def stop(self, timeout=5.0):
        self._shutdown.set()
        try:
            self._inbound.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, eval_groups):
        self._inbound.put(eval_groups, timeout=10.0)

    def collect(self, timeout=30.0):
        result = self._outbound.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise result
        return result

    def _loop(self):
        while not self._shutdown.is_set():
            try:
                groups = self._inbound.get(timeout=0.5)
            except queue.Empty:
                continue
            if groups is None:
                break
            try:
                responses = self._run_batched_eval_groups_fn(groups, self._model, self._device, self._cfg)
                self._outbound.put(responses)
            except Exception as exc:
                self._outbound.put(exc)


@dataclass(frozen=True)
class ShmEvalRuntimeHooks:
    run_batched_eval_groups: object
    make_eval_request_group: object
    unpack_qipc_batch_eval_req: object
    unpack_qipc_arena_eval_resp: object
    unpack_shm_search_response: object
    json_loads_fast: object
    emit_duty_cycle: object
    pack_qipc_batch_eval_resp: object
    logger: object
    shm_msg_eval_batch_req: int
    shm_msg_arena_eval_resp: int
    shm_msg_json: int
    shm_msg_search_resp: int
    inference_pipeline_thread_cls: object = InferencePipelineThread
    should_use_async_pipeline: object | None = None


def _default_shm_eval_runtime_hooks():
    try:
        from quartz.qipc import (
            SHM_MSG_EVAL_BATCH_REQ,
            SHM_MSG_ARENA_EVAL_RESP,
            SHM_MSG_JSON,
            SHM_MSG_SEARCH_RESP,
            pack_qipc_batch_eval_resp,
            unpack_qipc_batch_eval_req,
            unpack_qipc_arena_eval_resp,
            unpack_shm_search_response,
        )
    except ImportError:
        from qipc import (
            SHM_MSG_EVAL_BATCH_REQ,
            SHM_MSG_ARENA_EVAL_RESP,
            SHM_MSG_JSON,
            SHM_MSG_SEARCH_RESP,
            pack_qipc_batch_eval_resp,
            unpack_qipc_batch_eval_req,
            unpack_qipc_arena_eval_resp,
            unpack_shm_search_response,
        )
    return ShmEvalRuntimeHooks(
        run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: __import__("quartz.eval_runtime", fromlist=["run_batched_eval_groups"]).run_batched_eval_groups(
            groups, model_obj, dev, cfg_obj, runtime_support.run_model_batch
        ),
        make_eval_request_group=__import__("quartz.eval_runtime", fromlist=["make_eval_request_group"]).make_eval_request_group,
        unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
        unpack_qipc_arena_eval_resp=unpack_qipc_arena_eval_resp,
        unpack_shm_search_response=unpack_shm_search_response,
        json_loads_fast=runtime_support.json_loads_fast,
        emit_duty_cycle=getattr(runtime_support.resolve_search_client_cls(), "_emit_duty_cycle", lambda duty: None),
        pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
        logger=logging.getLogger(__name__),
        shm_msg_eval_batch_req=SHM_MSG_EVAL_BATCH_REQ,
        shm_msg_arena_eval_resp=SHM_MSG_ARENA_EVAL_RESP,
        shm_msg_json=SHM_MSG_JSON,
        shm_msg_search_resp=SHM_MSG_SEARCH_RESP,
        inference_pipeline_thread_cls=runtime_support.InferencePipelineThread,
        should_use_async_pipeline=runtime_support.should_use_async_pipeline,
    )


def shm_write_eval_response(ring, response_group, epoch=0, seq=0, runtime_hooks=None):
    runtime_hooks = runtime_hooks or _default_shm_eval_runtime_hooks()
    payload = runtime_hooks.pack_qipc_batch_eval_resp(response_group["policies"], response_group["values"])
    for attempt in range(100000):
        for slot_idx in range(ring.p2r_slot_count):
            if ring.p2r_try_write(slot_idx, 2, payload, epoch=epoch, seq=seq):
                return
        if attempt < 64:
            pass
        elif attempt < 512:
            time.sleep(0.000001)
        else:
            time.sleep(0.00001)
    runtime_hooks.logger.warning("_shm_write_eval_response: timed out waiting for p2r slot")


def shm_eval_loop(ring, model, device, cfg, proc, on_json=None, runtime_hooks=None, baseline_epoch=None):
    runtime_hooks = runtime_hooks or _default_shm_eval_runtime_hooks()
    pipeline_policy = runtime_hooks.should_use_async_pipeline
    if callable(pipeline_policy):
        use_pipeline = bool(pipeline_policy(model, device, cfg))
    else:
        use_pipeline = (
            not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
            and model is not None
            and not hasattr(model, "predict")
        )
    pipeline = None
    inflight = False
    inflight_epoch = 0
    inflight_seq = 0
    duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
    duty_log_interval = 16

    if use_pipeline:
        pipeline = runtime_hooks.inference_pipeline_thread_cls(model, device, cfg, max_pending=1)
        pipeline.start()

    terminal_payload = None
    command_epoch = None
    if baseline_epoch is None:
        try:
            baseline_epoch = int(ring.epoch())
        except Exception:
            baseline_epoch = 0

    def _cancel_requested():
        cancel_fn = getattr(ring, "cancel_requested", None)
        if not callable(cancel_fn):
            return False
        try:
            return bool(cancel_fn())
        except Exception:
            return False

    def _raise_if_cancelled_or_dead(context):
        if _cancel_requested():
            raise InterruptedError(f"SHM eval loop cancelled while {context}")
        if proc.poll() is not None:
            raise RuntimeError(f"Rust server exited (code={proc.returncode}) during SHM eval loop")

    def _collect_pipeline(timeout_s, context):
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while True:
            _raise_if_cancelled_or_dead(context)
            wait_s = min(0.25, max(0.001, deadline - time.perf_counter()))
            if wait_s <= 0.0:
                raise TimeoutError(f"timed out while {context}")
            try:
                return pipeline.collect(timeout=wait_s)
            except queue.Empty:
                continue
    try:
        spin = 0
        while True:
            _raise_if_cancelled_or_dead("waiting for SHM work")
            if inflight and pipeline is not None:
                model_t0 = time.perf_counter()
                responses = _collect_pipeline(30.0, "waiting for async SHM eval responses")
                duty["model_s"] += time.perf_counter() - model_t0
                inflight = False
                write_t0 = time.perf_counter()
                for rg in responses:
                    shm_write_eval_response(ring, rg, epoch=inflight_epoch, seq=inflight_seq, runtime_hooks=runtime_hooks)
                duty["write_s"] += time.perf_counter() - write_t0

            read_t0 = time.perf_counter()
            found_eval = False
            for slot_idx in range(ring.r2p_slot_count):
                result = ring.r2p_try_read_meta(slot_idx)
                if result is None:
                    continue
                msg_type, req_epoch, req_seq, payload_bytes = result
                if command_epoch is None:
                    if req_epoch <= baseline_epoch:
                        ring.r2p_mark_done(slot_idx)
                        continue
                    command_epoch = req_epoch
                elif req_epoch != command_epoch:
                    ring.r2p_mark_done(slot_idx)
                    continue
                spin = 0

                if msg_type == runtime_hooks.shm_msg_eval_batch_req:
                    ring.r2p_mark_done(slot_idx)
                    duty["read_s"] += time.perf_counter() - read_t0
                    found_eval = True
                    requests = runtime_hooks.unpack_qipc_batch_eval_req(bytes(payload_bytes))
                    eval_groups = [runtime_hooks.make_eval_request_group("binary_batch", requests, gi=0)]

                    if pipeline is not None:
                        inflight_epoch = req_epoch
                        inflight_seq = req_seq
                        pipeline.submit(eval_groups)
                        inflight = True
                    else:
                        model_t0 = time.perf_counter()
                        responses = runtime_hooks.run_batched_eval_groups(eval_groups, model, device, cfg)
                        duty["model_s"] += time.perf_counter() - model_t0
                        write_t0 = time.perf_counter()
                        for rg in responses:
                            shm_write_eval_response(ring, rg, epoch=req_epoch, seq=req_seq, runtime_hooks=runtime_hooks)
                        duty["write_s"] += time.perf_counter() - write_t0

                    duty["cycles"] += 1
                    if duty["cycles"] % duty_log_interval == 0:
                        runtime_hooks.emit_duty_cycle(duty)
                    break

                if msg_type == runtime_hooks.shm_msg_json:
                    ring.r2p_mark_done(slot_idx)
                    try:
                        json_obj = runtime_hooks.json_loads_fast(payload_bytes.decode("utf-8"))
                        if callable(on_json) and json_obj:
                            on_json(json_obj)
                    except Exception:
                        pass
                    continue

                if msg_type == runtime_hooks.shm_msg_search_resp:
                    ring.r2p_mark_done(slot_idx)
                    try:
                        terminal_payload = runtime_hooks.unpack_shm_search_response(payload_bytes)
                    except Exception:
                        terminal_payload = {"error": "invalid shm search response"}
                    continue

                if msg_type == runtime_hooks.shm_msg_arena_eval_resp:
                    ring.r2p_mark_done(slot_idx)
                    try:
                        terminal_payload = runtime_hooks.unpack_qipc_arena_eval_resp(payload_bytes)
                    except Exception:
                        terminal_payload = {"error": "invalid shm arena eval response"}
                    continue

                ring.r2p_mark_done(slot_idx)

            if not found_eval:
                duty["read_s"] += time.perf_counter() - read_t0
                try:
                    current_epoch = int(ring.epoch())
                except Exception:
                    current_epoch = command_epoch or baseline_epoch
                command_started = current_epoch > baseline_epoch
                if command_epoch is None and command_started:
                    command_epoch = current_epoch
                if ring.cmd_done() and command_started:
                    for slot_idx in range(ring.r2p_slot_count):
                        result = ring.r2p_try_read_meta(slot_idx)
                        if result is None:
                            continue
                        msg_type, req_epoch, _req_seq, payload_bytes = result
                        ring.r2p_mark_done(slot_idx)
                        if command_epoch is not None and req_epoch != command_epoch:
                            continue
                        if msg_type == runtime_hooks.shm_msg_json:
                            try:
                                json_obj = runtime_hooks.json_loads_fast(payload_bytes.decode("utf-8"))
                                if callable(on_json) and json_obj:
                                    on_json(json_obj)
                            except Exception:
                                pass
                        elif msg_type == runtime_hooks.shm_msg_search_resp:
                            try:
                                terminal_payload = runtime_hooks.unpack_shm_search_response(payload_bytes)
                            except Exception:
                                terminal_payload = {"error": "invalid shm search response"}
                        elif msg_type == runtime_hooks.shm_msg_arena_eval_resp:
                            try:
                                terminal_payload = runtime_hooks.unpack_qipc_arena_eval_resp(payload_bytes)
                            except Exception:
                                terminal_payload = {"error": "invalid shm arena eval response"}
                    break

                _raise_if_cancelled_or_dead("spinning for SHM results")

                spin += 1
                if spin < 64:
                    pass
                elif spin < 512:
                    time.sleep(0.000001)
                else:
                    time.sleep(0.00001)
    finally:
        if inflight and pipeline is not None and not _cancel_requested() and proc.poll() is None:
            try:
                drain = _collect_pipeline(10.0, "draining async SHM eval responses")
                for rg in drain:
                    shm_write_eval_response(ring, rg, epoch=inflight_epoch, seq=inflight_seq, runtime_hooks=runtime_hooks)
            except Exception:
                pass
        if pipeline is not None:
            pipeline.stop()
        if duty["cycles"] > 0:
            runtime_hooks.emit_duty_cycle(duty)
    return terminal_payload


class RustNNEvaluatorEngine:
    """Evaluator engine using the Rust MCTS + NN stack for promotion evaluation."""

    def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo", runtime_hooks=None):
        self._name = engine_name
        self._cfg = cfg
        self._model = model
        self._device = device
        self._rust_binary = rust_binary
        self._client = None
        self._simulations = self._cfg.get("iters", 200)
        self._runtime_hooks = runtime_hooks or _default_runtime_hooks()

    def _shares_eval_model_with(self, opponent) -> bool:
        if not isinstance(opponent, RustNNEvaluatorEngine):
            return False
        if self._cfg != opponent._cfg:
            return False
        return self._model is opponent._model

    def _ensure_client(self):
        if self._client is None:
            client_cls = self._runtime_hooks.search_client_cls
            self._client = client_cls(self._model, self._cfg, self._device, self._rust_binary)
            self._client.start()

    def select_move(self, state):
        return self.select_moves_batch([state])[0]

    def select_moves_batch(self, states):
        self._ensure_client()
        penalty_mode = self._cfg.get("penalty_mode", "GatedRefresh")
        game_name = self._cfg.get("_name")
        is_chess_game = self._runtime_hooks.is_chess_game
        build_rust_state_meta = self._runtime_hooks.build_rust_state_meta
        iter_sparse_policy_entries = self._runtime_hooks.iter_sparse_policy_entries

        if is_chess_game(game_name):
            jobs = []
            players = []
            for state in states:
                raw_player = state.current_player()
                player = 1 if raw_player == 1 else -1
                players.append(player)
                job = {
                    "fen": getattr(state, "_fen", ""),
                    "player": int(player),
                }
                job.update(build_rust_state_meta(game_name, state, self._cfg))
                jobs.append(job)
            results = self._client.search_moves_multi(jobs, penalty_mode=penalty_mode)
            parsed = []
            for _state, player, result in zip(states, players, results):
                if not result or "error" in result:
                    raise RuntimeError(
                        f"rust_nn chess eval failed: {result.get('error', 'empty response') if isinstance(result, dict) else 'empty response'}"
                    )
                pol_entries = result.get("policy", [])
                if not pol_entries:
                    terminal_value = float(result.get("value", 0.0))
                    outcome_for_black = -terminal_value * player
                    parsed.append(
                        (
                            0,
                            {
                                "time_used_ms": 0,
                                "simulations": self._simulations,
                                "p_flip": result.get("p_flip", 0),
                                "engine": "rust_nn",
                                "terminal": True,
                                "outcome_for_black": float(outcome_for_black),
                            },
                        )
                    )
                    continue
                best_move = int(result.get("best_move", 0))
                parsed.append(
                    (
                        best_move,
                        {
                            "time_used_ms": 0,
                            "simulations": self._simulations,
                            "p_flip": result.get("p_flip", 0),
                            "engine": "rust_nn",
                            "result_fen": result.get("result_fen", ""),
                            "result_history_hashes": result.get("result_history_hashes", []),
                        },
                    )
                )
            return parsed

        jobs = []
        legals = []
        for state in states:
            raw_player = state.current_player()
            player = 1 if raw_player == 0 else -1
            legal = state.legal_moves()
            legals.append(legal)
            jobs.append(
                {
                    "board": list(state._board),
                    "player": int(player),
                    **build_rust_state_meta(game_name, state, self._cfg),
                }
            )
        results = self._client.search_moves_multi(jobs, penalty_mode=penalty_mode)
        parsed = []
        for legal, result in zip(legals, results):
            if not result or "error" in result:
                parsed.append(((legal[0] if legal else 0), {"time_used_ms": 0, "simulations": 0}))
                continue
            pol_entries = result.get("policy", [])
            if not legal:
                parsed.append(
                    (
                        0,
                        {
                            "time_used_ms": 0,
                            "simulations": self._simulations,
                            "p_flip": result.get("p_flip", 0),
                            "engine": "rust_nn",
                        },
                    )
                )
                continue
            legal_set = set(legal)
            policy = {}
            for action, val in iter_sparse_policy_entries(pol_entries):
                if action in legal_set and action < self._cfg["actions"]:
                    policy[action] = val
            chosen = legal[0]
            best_val = policy.get(chosen, 0.0)
            for action in legal[1:]:
                value = policy.get(action, 0.0)
                if value > best_val:
                    chosen = action
                    best_val = value
            parsed.append(
                (
                    chosen,
                    {
                        "time_used_ms": 0,
                        "simulations": self._simulations,
                        "p_flip": result.get("p_flip", 0),
                        "engine": "rust_nn",
                    },
                )
            )
        return parsed

    def play_match_tally_against(
        self,
        opponent,
        game_factory,
        opening_book,
        num_games,
        color_swap=True,
        logger=None,
        max_moves=500,
        seed=None,
    ):
        if not isinstance(opponent, RustNNEvaluatorEngine):
            raise TypeError("shared Rust evaluation requires RustNNEvaluatorEngine opponent")

        runtime_cfg = arena_eval_runtime_cfg(self._cfg, num_games)
        self_manifest = runtime_support.build_search_manifest(runtime_cfg)
        opponent_runtime_cfg = arena_eval_runtime_cfg(opponent._cfg, num_games)
        opponent_manifest = runtime_support.build_search_manifest(opponent_runtime_cfg)
        self_manifest_hash = runtime_support.search_manifest_hash(runtime_cfg)
        opponent_manifest_hash = runtime_support.search_manifest_hash(opponent_runtime_cfg)
        if self_manifest_hash != opponent_manifest_hash:
            raise RuntimeError(
                "shared Rust evaluation requires matching search manifests: "
                f"{self_manifest_hash} != {opponent_manifest_hash} "
                f"({self_manifest} vs {opponent_manifest})"
            )
        same_model_mode = self._shares_eval_model_with(opponent)
        client_cls = self._runtime_hooks.search_client_cls
        shared_client = client_cls(
            self._model if same_model_mode else {0: self._model, 1: opponent._model},
            runtime_cfg,
            self._device,
            self._rust_binary,
        )
        shared_client.start()
        try:
            tally, _meta = _run_shared_eval_match(
                shared_client=shared_client,
                runtime_cfg=runtime_cfg,
                runtime_hooks=self._runtime_hooks,
                engine_a=self,
                engine_b=opponent,
                tag_a=0,
                tag_b=0 if same_model_mode else 1,
                game_factory=game_factory,
                opening_book=opening_book,
                num_games=num_games,
                color_swap=color_swap,
                logger=logger,
                max_moves=max_moves,
                seed=seed,
            )
        finally:
            shared_client.stop()
        return tally

    def reset(self):
        if self._client:
            self._client.stop()
            self._client = None

    def name(self):
        return self._name

    def __del__(self):
        self.reset()


class PersistentRustNNEvalCampaign:
    def __init__(self, engines, num_games, *, runtime_hooks=None):
        engines = list(engines)
        if not engines:
            raise ValueError("PersistentRustNNEvalCampaign requires at least one engine")
        first = engines[0]
        if not isinstance(first, RustNNEvaluatorEngine):
            raise TypeError("persistent eval campaign requires RustNNEvaluatorEngine instances")
        self._runtime_hooks = runtime_hooks or first._runtime_hooks
        self._device = first._device
        self._rust_binary = first._rust_binary
        self._runtime_cfg = arena_eval_runtime_cfg(first._cfg, num_games)
        self._manifest_hash = runtime_support.search_manifest_hash(self._runtime_cfg)
        self._engines = list(engines)
        self._tags = {}
        self._timings = {
            "engine_count": len(self._engines),
            "client_start_s": 0.0,
        }
        model_map = {}
        for idx, engine in enumerate(self._engines):
            if not isinstance(engine, RustNNEvaluatorEngine):
                raise TypeError("persistent eval campaign requires RustNNEvaluatorEngine instances")
            engine_manifest_hash = runtime_support.search_manifest_hash(arena_eval_runtime_cfg(engine._cfg, num_games))
            if engine_manifest_hash != self._manifest_hash:
                raise RuntimeError(
                    "persistent eval campaign requires matching search manifests: "
                    f"{engine_manifest_hash} != {self._manifest_hash}"
                )
            self._tags[id(engine)] = idx
            model_map[idx] = engine._model
        client_cls = self._runtime_hooks.search_client_cls
        t0 = time.perf_counter()
        self._client = client_cls(model_map, self._runtime_cfg, self._device, self._rust_binary)
        self._client.start()
        self._timings["client_start_s"] = time.perf_counter() - t0

    @property
    def timings(self):
        return dict(self._timings)

    def compare(
        self,
        engine_a,
        engine_b,
        game_factory,
        opening_book,
        num_games,
        color_swap=True,
        logger=None,
        max_moves=500,
        seed=None,
    ):
        if id(engine_a) not in self._tags or id(engine_b) not in self._tags:
            raise KeyError("engine not registered in persistent eval campaign")
        tally, match_meta = _run_shared_eval_match(
            shared_client=self._client,
            runtime_cfg=self._runtime_cfg,
            runtime_hooks=self._runtime_hooks,
            engine_a=engine_a,
            engine_b=engine_b,
            tag_a=self._tags[id(engine_a)],
            tag_b=self._tags[id(engine_b)],
            game_factory=game_factory,
            opening_book=opening_book,
            num_games=num_games,
            color_swap=color_swap,
            logger=logger,
            max_moves=max_moves,
            seed=seed,
        )
        meta = dict(self._timings)
        meta.update(match_meta)
        return tally, meta

    def compare_many(self, comparisons):
        specs = []
        for idx, comparison in enumerate(comparisons or []):
            engine_a = comparison["engine_a"]
            engine_b = comparison["engine_b"]
            if id(engine_a) not in self._tags or id(engine_b) not in self._tags:
                raise KeyError("engine not registered in persistent eval campaign")
            specs.append(
                {
                    "match_id": str(comparison.get("match_id") or f"match{idx}"),
                    "engine_a": engine_a,
                    "engine_b": engine_b,
                    "tag_a": self._tags[id(engine_a)],
                    "tag_b": self._tags[id(engine_b)],
                    "game_factory": comparison["game_factory"],
                    "opening_book": comparison.get("opening_book") or [],
                    "num_games": int(comparison["num_games"]),
                    "color_swap": bool(comparison.get("color_swap", True)),
                    "logger": comparison.get("logger"),
                    "max_moves": int(comparison.get("max_moves", 500)),
                    "seed": comparison.get("seed"),
                }
            )
        results = _run_shared_eval_matches(
            shared_client=self._client,
            runtime_cfg=self._runtime_cfg,
            runtime_hooks=self._runtime_hooks,
            match_specs=specs,
        )
        ordered = []
        for spec in specs:
            tally, match_meta = results[spec["match_id"]]
            meta = dict(self._timings)
            meta.update(match_meta)
            ordered.append((spec["match_id"], tally, meta))
        return ordered

    def close(self):
        if getattr(self, "_client", None) is not None:
            self._client.stop()
            self._client = None
        # Phase 7 follow-up (2026-04-27): explicitly drop any stacked-
        # module-state cache entries left over from the (opt-in) fused
        # multi-model eval forward. Belt-and-suspenders alongside the
        # weakref-on-model anchor: ensures GPU tensors are returned to
        # the caching allocator immediately on campaign close, even
        # if other references to a model still happen to be live.
        try:
            from quartz.eval_runtime import clear_fused_cache as _clear_fused_cache
            _clear_fused_cache()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()
