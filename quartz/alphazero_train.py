#!/usr/bin/env python3
"""
QUARTZ AlphaZero Training Pipeline — Compatibility Facade
=========================================================
Primary runtime logic now lives in focused modules (`qipc`, `eval_runtime`,
`selfplay_runtime`, `train_loop`, `cli_main`, ...). This module keeps the
historical import surface stable for scripts, tests, GUI tools, and older
notebooks while routing new code toward the split runtime modules.

Usage:
  python3 -m quartz.train --game gomoku15 --iterations 50
  python3 -m quartz.train --game gomoku7 --iterations 10 --device cpu
  python3 -m quartz.train --serve --game gomoku15 --model alphazero_gomoku15/best.pt

Requirements: torch, numpy, tqdm
GPU:          pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
              export HSA_OVERRIDE_GFX_VERSION=10.3.0
"""
import os, sys, json, select, time, argparse, subprocess, random, math, signal, threading, logging, struct, warnings, atexit, queue
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass

log = logging.getLogger(__name__)
from pathlib import Path

from quartz.backend import (
    load_torch_state_dict,
    load_torch_state_dict_checked,
    validate_torch_state_dict,
)

try:
    import orjson
except ImportError:
    orjson = None

try:
    from quartz.gpu_detect import detect_gpu, GpuInfo
except ImportError:
    try:
        from gpu_detect import detect_gpu, GpuInfo
    except ImportError:
        detect_gpu = None
        GpuInfo = None

# Game-agnostic encoder system
try:
    from quartz.encoders import get_encoder, GameEncoder
except ImportError:
    # Fallback: encoders.py in same directory
    try:
        from encoders import get_encoder, GameEncoder
    except ImportError:
        get_encoder = None


def encode_board(cfg, board_flat, player):
    """Game-agnostic board encoding using registered encoder.
    For 17-channel history encoding, use _encode_board_with_history instead.
    This function creates a single-timestep snapshot (t=0 only, no history)."""
    enc_obj = cfg.get('_encoder')
    if enc_obj is not None:
        return enc_obj.encode(board_flat, player)
    bs = cfg['board']; n2 = bs * bs
    ch = cfg.get('ch', 17)
    enc = np.zeros((ch, bs, bs), dtype=np.float32)
    # [OPT] Vectorized board encoding
    board_arr = np.asarray(board_flat, dtype=np.int8).ravel()[:n2]
    my_val = np.int8(player)
    enc[0].ravel()[:len(board_arr)] = (board_arr == my_val).astype(np.float32)
    opp_mask = (board_arr != 0) & (board_arr != my_val)
    enc[1].ravel()[:len(board_arr)] = opp_mask.astype(np.float32)
    # Color plane (last channel)
    if player == 1: enc[ch - 1] = 1.0
    return enc


def decode_board(cfg, enc, player):
    """Reconstruct flat board from encoded tensor."""
    enc_obj = cfg.get('_encoder')
    if enc_obj is not None:
        return enc_obj.decode(enc, player)
    # Legacy fallback
    bs = cfg['board']; board = np.zeros(bs * bs, dtype=np.int8)
    for r in range(bs):
        for c in range(bs):
            if enc[0, r, c] > 0.5: board[r * bs + c] = player
            elif enc[1, r, c] > 0.5: board[r * bs + c] = -player
    return board


def json_loads_fast(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def json_dumps_compact(payload):
    if orjson is not None:
        out = orjson.dumps(payload)
        return out.decode("utf-8") if isinstance(out, bytes) else out
    return json.dumps(payload, separators=(",", ":"))

try:
    from quartz.replay import (
        ReplayBuffer,
        ReplayExample,
        ReplayMetrics,
        SparsePolicyTarget,
        collate_replay_samples,
        dense_policy_from_sparse,
        iter_sparse_policy_entries,
        normalize_sparse_policy,
        sparse_policy_from_dense,
        sparse_policy_from_entries,
    )

    from quartz.eval_runtime import (
        NNEvalCache,
        clear_nn_eval_cache as _clear_nn_eval_cache_impl,
        eval_request_cache_key as _eval_request_cache_key_impl,
        legacy_eval_cache_key as _legacy_eval_cache_key_impl,
        make_eval_request_group as _make_eval_request_group_impl,
        parse_eval_request as _parse_eval_request_impl,
        run_batched_eval_groups as _run_batched_eval_groups_impl,
    )
    from quartz.qipc import (
        QIPC_BATCH_EVAL_REQ,
        QIPC_BATCH_EVAL_REQ_SHM,
        QIPC_BATCH_EVAL_RESP,
        QIPC_BATCH_EVAL_RESP_SHM,
        QIPC_EVAL_REQ,
        QIPC_EVAL_REQ_SHM,
        QIPC_EVAL_RESP,
        QIPC_EVAL_RESP_SHM,
        QIPC_HEADER,
        QIPC_MAGIC,
        QIPC_SHM_LEN,
        QipcSharedMemoryTransport,
        SHM_MSG_EVAL_BATCH_REQ,
        SHM_MSG_EVAL_BATCH_RESP,
        SHM_MSG_JSON,
        SHM_MSG_SEARCH_RESP,
        ShmRingBuffer,
        _json_line_bytes as _json_line_bytes_impl,
        _read_exact as _read_exact_impl,
        cleanup_all_shm as _cleanup_all_shm_impl,
        pack_qipc_batch_eval_resp,
        pack_qipc_eval_resp,
        cleanup_qipc_transport as _cleanup_qipc_transport_impl,
        get_qipc_transport as _get_qipc_transport_impl,
        launch_rust_server as _launch_rust_server_impl,
        proc_decode_eval_frame as _proc_decode_eval_frame_impl,
        proc_read_json_line as _proc_read_json_line_impl,
        proc_read_message as _proc_read_message_impl,
        proc_write_eval_response as _proc_write_eval_response_impl,
        proc_write_json_line as _proc_write_json_line_impl,
        proc_write_qipc_frame as _proc_write_qipc_frame_impl,
        register_ring_buffer as _register_ring_buffer_impl,
        stall_trace as _stall_trace_impl,
        stall_trace_path as _stall_trace_path_impl,
        stop_rust_server as _stop_rust_server_impl,
        unpack_qipc_batch_eval_req,
        unpack_qipc_eval_req,
        unpack_shm_search_response,
        unregister_ring_buffer as _unregister_ring_buffer_impl,
        wait_readable as _wait_readable_impl,
    )
    from quartz.selfplay_runtime import (
        ArenaRuntimeHooks,
        BatchedSelfPlayRuntimeHooks,
        LegacyRustSelfplayHooks,
        NNSearchClient as _NNSearchClientImpl,
        SearchClientRuntimeHooks,
        SelfPlayLoopRuntimeHooks,
        RustServerPool as _RustServerPoolImpl,
        SelfPlayWorker as _SelfPlayWorkerImpl,
        build_rust_state_meta as _build_rust_state_meta_impl,
        chess960_start_fen as _chess960_start_fen_impl,
        chess_state_meta_from_hashes as _chess_state_meta_from_hashes_impl,
        arena_rust_nn_impl as _arena_rust_nn_impl_runtime,
        choose_selfplay_move as _choose_selfplay_move_impl,
        compute_train_steps as _compute_train_steps_impl,
        decode_streamed_selfplay_game as _decode_streamed_selfplay_game_impl,
        default_output_dir as _default_output_dir_impl,
        encode_board_with_history as _encode_board_with_history_impl,
        encode_chess_fen as _encode_chess_fen_impl,
        estimate_selfplay_positions_per_game as _estimate_selfplay_positions_per_game_impl,
        initial_replay_fill_target as _initial_replay_fill_target_impl,
        initial_chess_fen as _initial_chess_fen_impl,
        is_chess_game as _is_chess_game_impl,
        is_go_game as _is_go_game_impl,
        plan_selfplay_runner_chunk as _plan_selfplay_runner_chunk_impl,
        rust_game_name as _rust_game_name_impl,
        rust_search_options as _rust_search_options_impl,
        selfplay_rust as _selfplay_rust_impl,
        selfplay_rust_nn as _selfplay_rust_nn_impl,
        selfplay_rust_nn_batched as _selfplay_rust_nn_batched_impl,
        should_use_resident_session as _should_use_resident_session_impl,
        supports_rust_eval_state_machine as _supports_rust_eval_state_machine_impl,
        supports_rust_selfplay_state_machine as _supports_rust_selfplay_state_machine_impl,
        wait_for_worker_progress as _wait_for_worker_progress_impl,
    )
    from quartz.game_adapters import (
        ChessEvaluationAdapter,
        GameAdapterRuntimeHooks,
        GoGameAdapter,
        GomokuGameAdapter,
        TicTacToeGameAdapter,
        build_training_game_adapter as _build_training_game_adapter_impl,
    )
    from quartz.arena_runtime import (
        Glicko2Rating,
        Glicko2System,
        MCTSNode,
        RandomRolloutAgent,
        TreeMCTS,
        TreeMCTSEngine,
        arena_3agent,
        arena_compare,
    )
    from quartz.evaluator_runtime import (
        EvaluatorRuntimeHooks,
        InferencePipelineThread as _InferencePipelineThreadImpl,
        RustNNEvaluatorEngine as _RustNNEvaluatorEngineImpl,
        ShmEvalRuntimeHooks,
        shm_eval_loop as _shm_eval_loop_impl,
        shm_write_eval_response as _shm_write_eval_response_impl,
    )
    from quartz.autotune_runtime import (
        AUTOTUNE_PROFILE_VERSION,
        OnlineAutotuneController as _OnlineAutotuneControllerImpl,
        AutotuneRuntimeHooks,
        _autotune_batch_game_candidates as _autotune_batch_game_candidates_impl,
        _autotune_batch_game_limit as _autotune_batch_game_limit_impl,
        _autotune_parallel_candidates as _autotune_parallel_candidates_impl,
        _autotune_parallel_limit as _autotune_parallel_limit_impl,
        _autotune_thread_candidates as _autotune_thread_candidates_impl,
        _autotune_thread_capacity as _autotune_thread_capacity_impl,
        _round_down_to_multiple as _round_down_to_multiple_impl,
        _round_up_to_multiple as _round_up_to_multiple_impl,
        _score_selfplay_probe as _score_selfplay_probe_impl,
        _score_train_batch_probe as _score_train_batch_probe_impl,
        apply_runtime_overrides as _apply_runtime_overrides_impl,
        autoscale_model_cfg as _autoscale_model_cfg_impl,
        autotune_signature as _autotune_signature_impl,
        autotune_training_cfg as _autotune_training_cfg_impl,
        benchmark_selfplay_throughput as _benchmark_selfplay_throughput_impl,
        benchmark_train_batch as _benchmark_train_batch_impl,
        estimate_model_params as _estimate_model_params_impl,
        load_autotune_profile as _load_autotune_profile_impl,
        plan_online_runtime_overrides as _plan_online_runtime_overrides_impl,
        print_autotune_summary as _print_autotune_summary_impl,
        run_autotune_benchmark as _run_autotune_benchmark_impl,
        save_autotune_profile as _save_autotune_profile_impl,
    )
    from quartz.system_runtime import (
        EVAL_AUTOTUNE_PROFILE_VERSION,
        HardwareSpec,
        auto_device_name,
        clamp_runtime_cfg_to_hardware,
        clamp_thread_count,
        compute_eval_collect_policy as _compute_eval_collect_policy_impl,
        configure_torch_rocm_runtime as _configure_torch_rocm_runtime_impl,
        detect_hardware_spec as _detect_hardware_spec_impl,
        eval_autotune_signature as _eval_autotune_signature_impl,
        eval_worker_candidates,
        gpu_host_thread_cap,
        gpu_interop_thread_cap,
        hardware_signature,
        load_eval_autotune_profile as _load_eval_autotune_profile_impl,
        max_supported_threads,
        recommend_eval_parallel_workers,
        save_eval_autotune_profile,
    )
    from quartz.train_loop import (
        EarlyStopping as _EarlyStoppingImpl,
        StepEarlyStopping as _StepEarlyStoppingImpl,
        build_best_elo_series as _build_best_elo_series_impl,
        build_elo_plot_series as _build_elo_plot_series_impl,
        build_metric_plot_series as _build_metric_plot_series_impl,
        early_stopping_enabled as _early_stopping_enabled_impl,
        generate_training_plots as _generate_training_plots_impl,
        load_epoch_history as _load_epoch_history_impl,
        round_or_none as _round_or_none_impl,
        train_epoch as _train_epoch_impl,
    )
    from quartz.cli_runtime import (
        CliRuntimeHooks,
        load_actor_source_from_checkpoint as _load_actor_source_from_checkpoint_impl,
        serve as _serve_impl,
    )
    from quartz.cli_main import build_arg_parser as _build_arg_parser_impl
    from quartz.cli_main import (
        MainRuntimeHooks,
        CliPrepareHooks,
        prepare_training_context as _prepare_training_context_impl,
        run_training_main as _run_training_main_impl,
    )
except ImportError:
    from replay import (
        ReplayBuffer,
        ReplayExample,
        ReplayMetrics,
        SparsePolicyTarget,
        collate_replay_samples,
        dense_policy_from_sparse,
        iter_sparse_policy_entries,
        normalize_sparse_policy,
        sparse_policy_from_dense,
        sparse_policy_from_entries,
    )
    from eval_runtime import (
        NNEvalCache,
        clear_nn_eval_cache as _clear_nn_eval_cache_impl,
        eval_request_cache_key as _eval_request_cache_key_impl,
        legacy_eval_cache_key as _legacy_eval_cache_key_impl,
        make_eval_request_group as _make_eval_request_group_impl,
        parse_eval_request as _parse_eval_request_impl,
        run_batched_eval_groups as _run_batched_eval_groups_impl,
    )
    from qipc import (
        QIPC_BATCH_EVAL_REQ,
        QIPC_BATCH_EVAL_REQ_SHM,
        QIPC_BATCH_EVAL_RESP,
        QIPC_BATCH_EVAL_RESP_SHM,
        QIPC_EVAL_REQ,
        QIPC_EVAL_REQ_SHM,
        QIPC_EVAL_RESP,
        QIPC_EVAL_RESP_SHM,
        QIPC_HEADER,
        QIPC_MAGIC,
        QIPC_SHM_LEN,
        QipcSharedMemoryTransport,
        SHM_MSG_EVAL_BATCH_REQ,
        SHM_MSG_EVAL_BATCH_RESP,
        SHM_MSG_JSON,
        SHM_MSG_SEARCH_RESP,
        ShmRingBuffer,
        _json_line_bytes as _json_line_bytes_impl,
        _read_exact as _read_exact_impl,
        cleanup_all_shm as _cleanup_all_shm_impl,
        pack_qipc_batch_eval_resp,
        pack_qipc_eval_resp,
        cleanup_qipc_transport as _cleanup_qipc_transport_impl,
        get_qipc_transport as _get_qipc_transport_impl,
        launch_rust_server as _launch_rust_server_impl,
        proc_decode_eval_frame as _proc_decode_eval_frame_impl,
        proc_read_json_line as _proc_read_json_line_impl,
        proc_read_message as _proc_read_message_impl,
        proc_write_eval_response as _proc_write_eval_response_impl,
        proc_write_json_line as _proc_write_json_line_impl,
        proc_write_qipc_frame as _proc_write_qipc_frame_impl,
        register_ring_buffer as _register_ring_buffer_impl,
        stall_trace as _stall_trace_impl,
        stall_trace_path as _stall_trace_path_impl,
        stop_rust_server as _stop_rust_server_impl,
        unpack_qipc_batch_eval_req,
        unpack_qipc_eval_req,
        unpack_shm_search_response,
        unregister_ring_buffer as _unregister_ring_buffer_impl,
        wait_readable as _wait_readable_impl,
    )
    from selfplay_runtime import (
        ArenaRuntimeHooks,
        BatchedSelfPlayRuntimeHooks,
        LegacyRustSelfplayHooks,
        NNSearchClient as _NNSearchClientImpl,
        SearchClientRuntimeHooks,
        SelfPlayLoopRuntimeHooks,
        RustServerPool as _RustServerPoolImpl,
        SelfPlayWorker as _SelfPlayWorkerImpl,
        build_rust_state_meta as _build_rust_state_meta_impl,
        chess960_start_fen as _chess960_start_fen_impl,
        chess_state_meta_from_hashes as _chess_state_meta_from_hashes_impl,
        arena_rust_nn_impl as _arena_rust_nn_impl_runtime,
        choose_selfplay_move as _choose_selfplay_move_impl,
        compute_train_steps as _compute_train_steps_impl,
        decode_streamed_selfplay_game as _decode_streamed_selfplay_game_impl,
        default_output_dir as _default_output_dir_impl,
        encode_board_with_history as _encode_board_with_history_impl,
        encode_chess_fen as _encode_chess_fen_impl,
        estimate_selfplay_positions_per_game as _estimate_selfplay_positions_per_game_impl,
        initial_replay_fill_target as _initial_replay_fill_target_impl,
        initial_chess_fen as _initial_chess_fen_impl,
        is_chess_game as _is_chess_game_impl,
        is_go_game as _is_go_game_impl,
        plan_selfplay_runner_chunk as _plan_selfplay_runner_chunk_impl,
        rust_game_name as _rust_game_name_impl,
        rust_search_options as _rust_search_options_impl,
        selfplay_rust as _selfplay_rust_impl,
        selfplay_rust_nn as _selfplay_rust_nn_impl,
        selfplay_rust_nn_batched as _selfplay_rust_nn_batched_impl,
        should_use_resident_session as _should_use_resident_session_impl,
        supports_rust_eval_state_machine as _supports_rust_eval_state_machine_impl,
        supports_rust_selfplay_state_machine as _supports_rust_selfplay_state_machine_impl,
        wait_for_worker_progress as _wait_for_worker_progress_impl,
    )
    from game_adapters import (
        ChessEvaluationAdapter,
        GameAdapterRuntimeHooks,
        GoGameAdapter,
        GomokuGameAdapter,
        TicTacToeGameAdapter,
        build_training_game_adapter as _build_training_game_adapter_impl,
    )
    from arena_runtime import (
        Glicko2Rating,
        Glicko2System,
        MCTSNode,
        RandomRolloutAgent,
        TreeMCTS,
        TreeMCTSEngine,
        arena_3agent,
        arena_compare,
    )
    from evaluator_runtime import (
        EvaluatorRuntimeHooks,
        InferencePipelineThread as _InferencePipelineThreadImpl,
        RustNNEvaluatorEngine as _RustNNEvaluatorEngineImpl,
        ShmEvalRuntimeHooks,
        shm_eval_loop as _shm_eval_loop_impl,
        shm_write_eval_response as _shm_write_eval_response_impl,
    )
    from autotune_runtime import (
        AUTOTUNE_PROFILE_VERSION,
        OnlineAutotuneController as _OnlineAutotuneControllerImpl,
        AutotuneRuntimeHooks,
        _autotune_batch_game_candidates as _autotune_batch_game_candidates_impl,
        _autotune_batch_game_limit as _autotune_batch_game_limit_impl,
        _autotune_parallel_candidates as _autotune_parallel_candidates_impl,
        _autotune_parallel_limit as _autotune_parallel_limit_impl,
        _autotune_thread_candidates as _autotune_thread_candidates_impl,
        _autotune_thread_capacity as _autotune_thread_capacity_impl,
        _round_down_to_multiple as _round_down_to_multiple_impl,
        _round_up_to_multiple as _round_up_to_multiple_impl,
        _score_selfplay_probe as _score_selfplay_probe_impl,
        _score_train_batch_probe as _score_train_batch_probe_impl,
        apply_runtime_overrides as _apply_runtime_overrides_impl,
        autoscale_model_cfg as _autoscale_model_cfg_impl,
        autotune_signature as _autotune_signature_impl,
        autotune_training_cfg as _autotune_training_cfg_impl,
        benchmark_selfplay_throughput as _benchmark_selfplay_throughput_impl,
        benchmark_train_batch as _benchmark_train_batch_impl,
        estimate_model_params as _estimate_model_params_impl,
        load_autotune_profile as _load_autotune_profile_impl,
        plan_online_runtime_overrides as _plan_online_runtime_overrides_impl,
        print_autotune_summary as _print_autotune_summary_impl,
        run_autotune_benchmark as _run_autotune_benchmark_impl,
        save_autotune_profile as _save_autotune_profile_impl,
    )
    from system_runtime import (
        EVAL_AUTOTUNE_PROFILE_VERSION,
        HardwareSpec,
        auto_device_name,
        clamp_runtime_cfg_to_hardware,
        clamp_thread_count,
        compute_eval_collect_policy as _compute_eval_collect_policy_impl,
        configure_torch_rocm_runtime as _configure_torch_rocm_runtime_impl,
        detect_hardware_spec as _detect_hardware_spec_impl,
        eval_autotune_signature as _eval_autotune_signature_impl,
        eval_worker_candidates,
        gpu_host_thread_cap,
        gpu_interop_thread_cap,
        hardware_signature,
        load_eval_autotune_profile as _load_eval_autotune_profile_impl,
        max_supported_threads,
        recommend_eval_parallel_workers,
        save_eval_autotune_profile,
    )
    from train_loop import (
        EarlyStopping as _EarlyStoppingImpl,
        StepEarlyStopping as _StepEarlyStoppingImpl,
        build_best_elo_series as _build_best_elo_series_impl,
        build_elo_plot_series as _build_elo_plot_series_impl,
        build_metric_plot_series as _build_metric_plot_series_impl,
        early_stopping_enabled as _early_stopping_enabled_impl,
        generate_training_plots as _generate_training_plots_impl,
        load_epoch_history as _load_epoch_history_impl,
        round_or_none as _round_or_none_impl,
        train_epoch as _train_epoch_impl,
    )
    from cli_runtime import (
        CliRuntimeHooks,
        load_actor_source_from_checkpoint as _load_actor_source_from_checkpoint_impl,
        serve as _serve_impl,
    )
    from cli_main import build_arg_parser as _build_arg_parser_impl
    from cli_main import (
        MainRuntimeHooks,
        CliPrepareHooks,
        prepare_training_context as _prepare_training_context_impl,
        run_training_main as _run_training_main_impl,
    )


wait_readable = _wait_readable_impl
_register_ring_buffer = _register_ring_buffer_impl
_unregister_ring_buffer = _unregister_ring_buffer_impl


def _cleanup_all_shm():
    return _cleanup_all_shm_impl()


atexit.register(_cleanup_all_shm)


def _get_qipc_transport(proc):
    return _get_qipc_transport_impl(proc)


def _cleanup_qipc_transport(proc):
    return _cleanup_qipc_transport_impl(proc, unregister_ring_buffer_fn=_unregister_ring_buffer)


def _json_line_bytes(payload):
    return _json_line_bytes_impl(payload, json_dumps_compact_fn=json_dumps_compact)


def _read_exact(stream, n_bytes, timeout_s=None):
    return _read_exact_impl(stream, n_bytes, timeout_s=timeout_s, wait_readable_fn=wait_readable)


def _stall_trace_path():
    return _stall_trace_path_impl()


def _stall_trace(event, **fields):
    return _stall_trace_impl(event, path_fn=_stall_trace_path, **fields)


def proc_write_json_line(proc_or_stream, payload):
    return _proc_write_json_line_impl(proc_or_stream, payload, json_dumps_compact_fn=json_dumps_compact)


def proc_write_qipc_frame(proc_or_stream, frame_kind, payload):
    return _proc_write_qipc_frame_impl(proc_or_stream, frame_kind, payload)


def proc_read_json_line(proc_or_stream):
    return _proc_read_json_line_impl(proc_or_stream)


def proc_read_message(proc_or_stream, timeout_s=None):
    return _proc_read_message_impl(proc_or_stream, timeout_s=timeout_s, json_loads_fast_fn=json_loads_fast, logger=log)


def proc_decode_eval_frame(proc, frame_kind, payload):
    return _proc_decode_eval_frame_impl(proc, frame_kind, payload)


def proc_write_eval_response(proc, logical_kind, payload, prefer_shm=False):
    return _proc_write_eval_response_impl(proc, logical_kind, payload, prefer_shm=prefer_shm)

class InferencePipelineThread(_InferencePipelineThreadImpl):
    def __init__(self, model, device, cfg, max_pending=1):
        super().__init__(
            model,
            device,
            cfg,
            max_pending=max_pending,
            run_batched_eval_groups_fn=_run_batched_eval_groups,
        )


def clear_nn_eval_cache():
    """Call after model training to invalidate cached predictions."""
    return _clear_nn_eval_cache_impl(logger=log)


_parse_eval_request = _parse_eval_request_impl


def _legacy_eval_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg):
    return _legacy_eval_cache_key_impl(model_tag, num_actions, feat_array, ch_cfg, bs_cfg)


def _eval_request_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg, fp_lo, fp_hi, encoder_rev):
    if fp_lo is not None and fp_hi is not None:
        return (int(model_tag), int(fp_hi), int(fp_lo), int(encoder_rev or 0))
    return _legacy_eval_cache_key(model_tag, num_actions, feat_array, ch_cfg, bs_cfg)


def _run_batched_eval_groups(eval_groups, model, device, cfg):
    return _run_batched_eval_groups_impl(
        eval_groups,
        model,
        device,
        cfg,
        run_model_batch=_run_model_batch,
        cache_key_fn=_eval_request_cache_key,
    )


_make_eval_request_group = _make_eval_request_group_impl


def _make_shm_eval_runtime_hooks():
    return ShmEvalRuntimeHooks(
        run_batched_eval_groups=_run_batched_eval_groups,
        make_eval_request_group=_make_eval_request_group,
        unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
        unpack_shm_search_response=unpack_shm_search_response,
        json_loads_fast=json_loads_fast,
        emit_duty_cycle=NNSearchClient._emit_duty_cycle,
        pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
        logger=log,
        shm_msg_eval_batch_req=SHM_MSG_EVAL_BATCH_REQ,
        shm_msg_json=SHM_MSG_JSON,
        shm_msg_search_resp=SHM_MSG_SEARCH_RESP,
        inference_pipeline_thread_cls=InferencePipelineThread,
    )


def _write_batched_eval_group(proc, response_group):
    kind = response_group["kind"]
    policies = response_group["policies"]
    values = response_group["values"]
    if kind == "binary_batch":
        proc_write_eval_response(
            proc,
            QIPC_BATCH_EVAL_RESP,
            pack_qipc_batch_eval_resp(policies, values),
            prefer_shm=bool(response_group.get("prefer_shm")),
        )
    elif kind == "binary_single":
        proc_write_eval_response(
            proc,
            QIPC_EVAL_RESP,
            pack_qipc_eval_resp(policies[0], values[0]),
            prefer_shm=bool(response_group.get("prefer_shm")),
        )
    elif kind == "json_batch":
        proc_write_json_line(proc, {
            "batch_eval_resp": {
                "responses": [
                    {"policy": policy.tolist(), "value": float(value)}
                    for policy, value in zip(policies, values)
                ]
            }
        })
    elif kind == "json_single":
        proc_write_json_line(proc, {
            "eval_resp": {
                "policy": policies[0].tolist(),
                "value": float(values[0]),
            }
        })
    else:
        raise ValueError(f"unknown eval response group kind: {kind}")


def _shm_eval_loop(ring, model, device, cfg, proc, on_json=None):
    return _shm_eval_loop_impl(
        ring,
        model,
        device,
        cfg,
        proc,
        on_json=on_json,
        runtime_hooks=_make_shm_eval_runtime_hooks(),
    )


def _shm_write_eval_response(ring, response_group, epoch=0, seq=0):
    return _shm_write_eval_response_impl(
        ring,
        response_group,
        epoch=epoch,
        seq=seq,
        runtime_hooks=_make_shm_eval_runtime_hooks(),
    )


def launch_rust_server(rust_binary):
    return _launch_rust_server_impl(
        rust_binary,
        qipc_transport_cls=QipcSharedMemoryTransport,
        shm_ring_buffer_cls=ShmRingBuffer,
        register_ring_buffer_fn=_register_ring_buffer,
        stall_trace_fn=_stall_trace,
    )


def stop_rust_server(proc, timeout=3.0):
    return _stop_rust_server_impl(
        proc,
        timeout=timeout,
        write_json_line_fn=proc_write_json_line,
        cleanup_qipc_transport_fn=_cleanup_qipc_transport,
        stall_trace_fn=_stall_trace,
    )


class RustServerPool(_RustServerPoolImpl):
    def __init__(self, rust_binary):
        super().__init__(
            rust_binary,
            launch_server=launch_rust_server,
            stop_server=stop_rust_server,
        )

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        """Fallback when tqdm not installed."""
        def __init__(self, iterable=None, total=None, desc="", leave=True, **kw):
            self.iterable = iterable; self.total = total; self.desc = desc; self.n = 0
        def __iter__(self):
            for x in self.iterable: yield x; self.n += 1
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def set_postfix_str(self, s): pass
        def set_postfix(self, **kw): pass
        def close(self): pass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from quartz.models_torch import AlphaZeroNet, ResBlock, SEBlock
except ImportError:
    from models_torch import AlphaZeroNet, ResBlock, SEBlock
try:
    from quartz.training_catalog import (
        CHESS_POLICY_ACTIONS,
        GAME_CONFIGS,
        GOMOKU15_VARIANTS,
        GO_RULESET_PRESETS,
        SEARCH_RUNTIME_KEYS,
        STANDARD_CHESS_FEN,
        apply_config_overrides,
        is_chess_game,
        is_go_game,
        resolve_runtime_paths,
        rust_game_name,
    )
except ImportError:
    from training_catalog import (
        CHESS_POLICY_ACTIONS,
        GAME_CONFIGS,
        GOMOKU15_VARIANTS,
        GO_RULESET_PRESETS,
        SEARCH_RUNTIME_KEYS,
        STANDARD_CHESS_FEN,
        apply_config_overrides,
        is_chess_game,
        is_go_game,
        resolve_runtime_paths,
        rust_game_name,
    )

try:
    from quartz.training_runtime_utils import (
        benchmark_eval_parallel_workers as _benchmark_eval_parallel_workers_impl,
        make_json_safe as _make_json_safe_impl,
    )
except ImportError:
    from training_runtime_utils import (
        benchmark_eval_parallel_workers as _benchmark_eval_parallel_workers_impl,
        make_json_safe as _make_json_safe_impl,
    )

try:
    from quartz import runtime_support as _runtime_support
    from quartz import torch_training_runtime as _torch_training_runtime
except ImportError:
    import runtime_support as _runtime_support
    import torch_training_runtime as _torch_training_runtime


detect_checkpoint_backend_hint = _runtime_support.detect_checkpoint_backend_hint


def ensure_best_checkpoint_compatible(best_model_path, backend, model, device):
    """Keep best.pt aligned with the active training backend.

    Old experiments may leave a JAX checkpoint behind while current training is
    running with PyTorch. That breaks evaluation promotion loading. When the
    format obviously mismatches, reseed best.pt from the current model.
    """
    if not os.path.exists(best_model_path):
        return None
    active_backend = getattr(backend, "name", "torch") if backend is not None else "torch"
    hint = detect_checkpoint_backend_hint(best_model_path)
    mismatch = (
        (active_backend == "torch" and hint == "jax")
        or (active_backend == "jax" and hint == "torch")
    )
    if not mismatch and active_backend == "torch" and model is not None and hint in {"torch", "unknown"}:
        try:
            state_dict = load_torch_state_dict(best_model_path, torch, map_location=device)
            mismatch = validate_torch_state_dict(model, state_dict) is not None
        except Exception:
            mismatch = True
    if not mismatch:
        return hint
    if backend is not None:
        backend.save(best_model_path)
    elif model is not None:
        torch.save(model.state_dict(), best_model_path)
    print(f"  [Eval] Reset incompatible best checkpoint ({hint} -> {active_backend})")
    return active_backend


def rust_search_options(cfg, penalty_mode=None):
    return _rust_search_options_impl(cfg, penalty_mode=penalty_mode)


def normalize_rust_board(game_name, board_flat):
    if board_flat is None:
        return None
    if is_go_game(game_name):
        return [1 if v == 1 else 2 if v in (-1, 2) else 0 for v in board_flat]
    return board_flat.tolist() if hasattr(board_flat, "tolist") else list(board_flat)


chess_state_meta_from_hashes = _chess_state_meta_from_hashes_impl


build_rust_state_meta = _runtime_support.build_rust_state_meta


chess960_start_fen = _chess960_start_fen_impl


initial_chess_fen = _runtime_support.initial_chess_fen


encode_chess_fen = _runtime_support.encode_chess_fen

# ═══════════════════════════════════════════
# § Rust Server Self-Play
# ═══════════════════════════════════════════

def selfplay_rust(cfg, n_games, rust_binary="./target/release/mcts_demo"):
    return _selfplay_rust_impl(
        cfg,
        n_games,
        rust_binary=rust_binary,
        runtime_hooks=LegacyRustSelfplayHooks(
            launch_rust_server=launch_rust_server,
            proc_write_json_line=proc_write_json_line,
            proc_read_json_line=proc_read_json_line,
            json_loads_fast=json_loads_fast,
            rust_game_name=lambda game_name: rust_game_name(game_name),
            rust_search_options=rust_search_options,
            is_go_game=is_go_game,
            tqdm_factory=tqdm,
        ),
    )


# ═══════════════════════════════════════════
# § NN-Backed Rust Search Client (search_nn protocol)
# ═══════════════════════════════════════════

class NNSearchClient(_NNSearchClientImpl):
    def __init__(self, model, cfg, device, rust_binary="./target/release/mcts_demo"):
        super().__init__(
            model,
            cfg,
            device,
            rust_binary,
            runtime_hooks=SearchClientRuntimeHooks(
                launch_server=launch_rust_server,
                proc_write_json_line=proc_write_json_line,
                cleanup_qipc_transport=_cleanup_qipc_transport,
                proc_read_json_line=proc_read_json_line,
                json_loads_fast=json_loads_fast,
                rust_game_name=rust_game_name,
                rust_search_options=rust_search_options,
                is_chess_game=is_chess_game,
                normalize_rust_board=normalize_rust_board,
                proc_read_message=proc_read_message,
                shm_eval_loop=_shm_eval_loop,
                proc_decode_eval_frame=proc_decode_eval_frame,
                qipc_batch_eval_req=QIPC_BATCH_EVAL_REQ,
                qipc_eval_req=QIPC_EVAL_REQ,
                make_eval_request_group=_make_eval_request_group,
                unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
                unpack_qipc_eval_req=unpack_qipc_eval_req,
                compute_eval_collect_policy=compute_eval_collect_policy,
                wait_readable=wait_readable,
                inference_pipeline_thread_cls=InferencePipelineThread,
                run_batched_eval_groups=_run_batched_eval_groups,
                write_batched_eval_group=_write_batched_eval_group,
                run_model_batch=_run_model_batch,
                torch_module=torch,
                pack_qipc_eval_resp=pack_qipc_eval_resp,
                pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
                parse_eval_request=_parse_eval_request,
            ),
        )


# ═══════════════════════════════════════════
# § Actor/Learner Separation: Background Self-Play Worker
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# § Rust NN-Backed Self-Play
# ═══════════════════════════════════════════

def selfplay_rust_nn(cfg, model, device, n_games, rust_binary="./target/release/mcts_demo"):
    return _selfplay_rust_nn_impl(
        cfg,
        model,
        device,
        n_games,
        rust_binary,
        runtime_hooks=SelfPlayLoopRuntimeHooks(
            is_chess_game=is_chess_game,
            is_go_game=is_go_game,
            search_client_cls=NNSearchClient,
            initial_chess_fen=initial_chess_fen,
            build_training_game_adapter=build_training_game_adapter,
            encode_chess_fen=encode_chess_fen,
            build_rust_state_meta=build_rust_state_meta,
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            choose_selfplay_move=choose_selfplay_move,
        ),
    )


_estimate_selfplay_positions_per_game = _estimate_selfplay_positions_per_game_impl


initial_replay_fill_target = _initial_replay_fill_target_impl


plan_selfplay_runner_chunk = _plan_selfplay_runner_chunk_impl


class SelfPlayWorker(_SelfPlayWorkerImpl):
    def __init__(self, cfg, model, device, replay, rust_binary):
        super().__init__(
            cfg,
            model,
            device,
            replay,
            rust_binary,
            server_pool_factory=lambda rb: RustServerPool(rb),
            clone_actor_model_fn=clone_actor_model,
            selfplay_runner=selfplay_rust_nn_batched,
            logger=logging.getLogger(__name__),
        )


_encode_board_with_history = _encode_board_with_history_impl
_decode_streamed_selfplay_game = _decode_streamed_selfplay_game_impl
StepEarlyStopping = _StepEarlyStoppingImpl


def train_epoch(model, optimizer, replay, cfg, device, n_steps, backend=None, inner_stop_cfg=None):
    return _train_epoch_impl(
        model,
        optimizer,
        replay,
        cfg,
        device,
        n_steps,
        backend=backend,
        inner_stop_cfg=inner_stop_cfg,
    )


_COMPILED_MODELS = {}  # id(model) → compiled model cache


def _get_compiled_model(model):
    """Lazily compile model with torch.compile for faster inference."""
    key = id(model)
    compiled = _COMPILED_MODELS.get(key)
    if compiled is not None:
        return compiled
    if os.environ.get("QUARTZ_DISABLE_COMPILE"):
        return model
    try:
        import torch as _torch
        compiled = _torch.compile(model, mode="default", dynamic=True)
        _COMPILED_MODELS[key] = compiled
        return compiled
    except Exception:
        _COMPILED_MODELS[key] = model
        return model


_PINNED_BUFS = {}  # (device_str, C, H, W) → (pinned_tensor, gpu_tensor, max_bs)


def _get_inference_buffers(device, batch_np):
    """Get or create pre-allocated pinned + GPU buffers for inference."""
    if batch_np.ndim != 4:
        return None
    bs, C, H, W = batch_np.shape
    key = (str(device), C, H, W)
    entry = _PINNED_BUFS.get(key)
    if entry is not None:
        pinned, gpu, max_bs = entry
        if bs <= max_bs:
            pinned[:bs].copy_(torch.from_numpy(batch_np))
            gpu[:bs].copy_(pinned[:bs], non_blocking=True)
            return gpu[:bs]
    # Allocate new buffers (2x headroom for batch size growth)
    max_bs = max(bs * 2, 64)
    pinned = torch.zeros(max_bs, C, H, W, dtype=torch.float32).pin_memory()
    gpu = torch.zeros(max_bs, C, H, W, dtype=torch.float32, device=device)
    _PINNED_BUFS[key] = (pinned, gpu, max_bs)
    pinned[:bs].copy_(torch.from_numpy(batch_np))
    gpu[:bs].copy_(pinned[:bs], non_blocking=True)
    return gpu[:bs]


def _run_model_batch(model, device, batch_features):
    batch_np = np.asarray(batch_features, dtype=np.float32)
    if not batch_np.flags.c_contiguous:
        batch_np = np.ascontiguousarray(batch_np)
    if not batch_np.flags.writeable:
        batch_np = batch_np.copy()
    if hasattr(model, "predict"):
        probs_batch, vals_np = model.predict(batch_np)
        return np.asarray(probs_batch, dtype=np.float32), np.asarray(vals_np, dtype=np.float32).reshape(-1)
    if getattr(device, "type", "cpu") != "cpu":
        x_batch = _get_inference_buffers(device, batch_np)
        if x_batch is None:
            x_batch = torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True)
    else:
        x_batch = torch.from_numpy(batch_np).to(device)
    compiled = _get_compiled_model(model)
    with torch.inference_mode():
        logits_batch, vals_batch = compiled(x_batch)
        probs_batch = torch.softmax(logits_batch, dim=-1).cpu().numpy()
        vals_np = vals_batch.cpu().numpy()
    return probs_batch, vals_np


def choose_selfplay_move(policy, legal, move_count, temp_threshold, fallback_best=-1):
    return _choose_selfplay_move_impl(policy, legal, move_count, temp_threshold, fallback_best=fallback_best)


def get_actor_model(training_model, backend):
    """Return the model object that self-play/eval should query for NN eval."""
    if backend is None:
        return training_model
    return backend


def clone_actor_model(actor_source):
    if actor_source is None:
        return None
    if hasattr(actor_source, "create_actor"):
        actor = actor_source.create_actor()
    else:
        import copy
        actor = copy.deepcopy(actor_source)
    if hasattr(actor, "eval"):
        actor.eval()
    return actor


def _make_cli_runtime_hooks():
    return CliRuntimeHooks(
        alphazero_net_cls=AlphaZeroNet,
        load_torch_state_dict=load_torch_state_dict,
        run_model_batch=_run_model_batch,
    )


def _make_autotune_runtime_hooks():
    return AutotuneRuntimeHooks(
        alphazero_net_cls=AlphaZeroNet,
        run_model_batch=_run_model_batch,
        selfplay_rust_nn_batched=selfplay_rust_nn_batched,
        stall_trace=_stall_trace,
        tqdm_factory=_autotune_progress_bar,
    )


def load_actor_source_from_checkpoint(
        checkpoint_path, cfg, device, backend_preference="torch", backend_template=None):
    return _load_actor_source_from_checkpoint_impl(
        checkpoint_path,
        cfg,
        device,
        backend_preference=backend_preference,
        backend_template=backend_template,
        runtime_hooks=_make_cli_runtime_hooks(),
    )


wait_for_worker_progress = _wait_for_worker_progress_impl
compute_train_steps = _compute_train_steps_impl
default_output_dir = _default_output_dir_impl


def detect_hardware_spec(device):
    return _detect_hardware_spec_impl(device, detect_gpu_fn=detect_gpu)


configure_torch_rocm_runtime = _configure_torch_rocm_runtime_impl
load_eval_autotune_profile = _load_eval_autotune_profile_impl
eval_autotune_signature = _eval_autotune_signature_impl


def compute_eval_collect_policy(base_target_items, base_timeout_s, batch_items_ema=None, wait_ema_s=None):
    return _compute_eval_collect_policy_impl(
        base_target_items,
        base_timeout_s,
        batch_items_ema=batch_items_ema,
        wait_ema_s=wait_ema_s,
    )


def benchmark_eval_parallel_workers(
        hw, cfg, eval_games, candidate_factory, champion_factory, game_factory, profile_path):
    return _benchmark_eval_parallel_workers_impl(
        hw,
        cfg,
        eval_games,
        candidate_factory,
        champion_factory,
        game_factory,
        profile_path,
        has_eval_system=HAS_EVAL_SYSTEM,
        eval_worker_candidates_fn=eval_worker_candidates,
        eval_config_cls=EvalConfig,
        training_evaluator_cls=TrainingEvaluator,
        save_eval_autotune_profile_fn=save_eval_autotune_profile,
    )


_round_down_to_multiple = _round_down_to_multiple_impl
_round_up_to_multiple = _round_up_to_multiple_impl
_autotune_parallel_limit = _autotune_parallel_limit_impl
_autotune_thread_capacity = _autotune_thread_capacity_impl
_autotune_thread_candidates = _autotune_thread_candidates_impl
_autotune_batch_game_limit = _autotune_batch_game_limit_impl


def estimate_model_params(cfg):
    return _estimate_model_params_impl(cfg, AlphaZeroNet)


def autoscale_model_cfg(cfg, hw):
    return _autoscale_model_cfg_impl(cfg, hw, AlphaZeroNet)


def _probe_inference_batch_size(model, device, cfg, eval_batch_cap):
    if model is None or hasattr(model, "predict") or getattr(device, "type", "cpu") == "cpu":
        return cfg.get("batch_size", 8)
    ch, bs = cfg.get("ch", 3), cfg.get("board", 7)
    current_bs = cfg.get("batch_size", 8)
    candidates = sorted(set([current_bs] + [c for c in [32, 64, 128, 256] if c <= eval_batch_cap]))
    best_bs, best_ips = current_bs, 0.0
    for cand in candidates:
        try:
            batch = [np.random.randn(ch, bs, bs).astype(np.float32) for _ in range(cand)]
            for _ in range(5):
                _run_model_batch(model, device, batch)
            N = max(20, 200 // cand)
            t0 = time.perf_counter()
            for _ in range(N):
                _run_model_batch(model, device, batch)
            elapsed = time.perf_counter() - t0
            ips = cand * N / max(elapsed, 1e-9)
            if ips > best_ips:
                best_ips = ips
                best_bs = cand
        except Exception:
            break
    return best_bs


autotune_training_cfg = _torch_training_runtime.autotune_training_cfg


print_autotune_summary = _print_autotune_summary_impl


def autotune_signature(hw, cfg):
    return _autotune_signature_impl(hw, cfg, hardware_signature)


load_autotune_profile = _torch_training_runtime.load_autotune_profile


save_autotune_profile = _torch_training_runtime.save_autotune_profile


apply_runtime_overrides = _apply_runtime_overrides_impl
_autotune_parallel_candidates = _autotune_parallel_candidates_impl
_autotune_batch_game_candidates = _autotune_batch_game_candidates_impl


def _score_selfplay_probe(positions_per_s, cycle_s, concurrent, positions=0, eval_messages=0,
                          model_batch_mean=0.0, parallel=1, n_threads=1):
    return _score_selfplay_probe_impl(
        positions_per_s,
        cycle_s,
        concurrent,
        positions=positions,
        eval_messages=eval_messages,
        model_batch_mean=model_batch_mean,
        parallel=parallel,
        n_threads=n_threads,
    )


def _score_train_batch_probe(examples_per_s, batch_n, concurrent=False, target_positions_per_cycle=None):
    return _score_train_batch_probe_impl(
        examples_per_s,
        batch_n,
        concurrent=concurrent,
        target_positions_per_cycle=target_positions_per_cycle,
    )


should_use_resident_session = _should_use_resident_session_impl


supports_rust_eval_state_machine = _runtime_support.supports_rust_eval_state_machine


supports_rust_selfplay_state_machine = _torch_training_runtime.supports_rust_selfplay_state_machine


def _autotune_progress_bar(total, desc):
    use_tqdm = HAS_TQDM and sys.stderr.isatty()
    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True, disable=not use_tqdm)


benchmark_selfplay_throughput = _torch_training_runtime.benchmark_selfplay_throughput


def benchmark_train_batch(cfg, backend, model, optimizer, device, hw,
                          concurrent=False, target_positions_per_cycle=None):
    return _benchmark_train_batch_impl(
        cfg,
        backend,
        model,
        optimizer,
        device,
        hw,
        concurrent=concurrent,
        target_positions_per_cycle=target_positions_per_cycle,
    )


run_autotune_benchmark = _torch_training_runtime.run_autotune_benchmark


def plan_online_runtime_overrides(cfg, hw, sample):
    return _plan_online_runtime_overrides_impl(cfg, hw, sample)


OnlineAutotuneController = _OnlineAutotuneControllerImpl
early_stopping_enabled = _early_stopping_enabled_impl
round_or_none = _round_or_none_impl


def make_json_safe(value):
    return _make_json_safe_impl(value)


load_epoch_history = _load_epoch_history_impl
build_elo_plot_series = _build_elo_plot_series_impl
build_metric_plot_series = _build_metric_plot_series_impl
build_best_elo_series = _build_best_elo_series_impl
generate_training_plots = _generate_training_plots_impl

# ═══════════════════════════════════════════
# § Early Stopping
# ═══════════════════════════════════════════

EarlyStopping = _EarlyStoppingImpl

# ═══════════════════════════════════════════
# § Eval Server (for Rust MCTS PythonIpcEval)
# ═══════════════════════════════════════════

def serve(model, cfg, device):
    return _serve_impl(
        model,
        cfg,
        device,
        runtime_hooks=_make_cli_runtime_hooks(),
    )

# ═══════════════════════════════════════════
# § Main
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# § Arena: Head-to-Head Model Comparison
# ═══════════════════════════════════════════

def _arena_rust_nn_impl(model_a_path, cfg_a, model_b_path, cfg_b, device, n_games=50,
                        rust_binary="./target/release/mcts_demo", strict=True):
    return _arena_rust_nn_impl_runtime(
        model_a_path,
        cfg_a,
        model_b_path,
        cfg_b,
        device,
        n_games=n_games,
        rust_binary=rust_binary,
        strict=strict,
        runtime_hooks=ArenaRuntimeHooks(
            is_chess_game=is_chess_game,
            search_client_cls=NNSearchClient,
            alphazero_net_cls=AlphaZeroNet,
            load_torch_state_dict=load_torch_state_dict,
            torch_module=torch,
            initial_chess_fen=initial_chess_fen,
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            arena_compare=arena_compare,
            build_training_game_adapter=build_training_game_adapter,
            rust_nn_evaluator_engine_cls=RustNNEvaluatorEngine,
            match_runner_cls=MatchRunner,
        ),
    )


arena_rust_nn = _torch_training_runtime.arena_rust_nn


def arena_rust_nn_dual_cfg(model_a_path, cfg_a, model_b_path, cfg_b, device, n_games=50,
                           rust_binary="./target/release/mcts_demo", strict=True):
    return _arena_rust_nn_impl(
        model_a_path,
        cfg_a,
        model_b_path,
        cfg_b,
        device,
        n_games=n_games,
        rust_binary=rust_binary,
        strict=strict,
    )


selfplay_rust_nn_batched = _torch_training_runtime.selfplay_rust_nn_batched


# ═══════════════════════════════════════════
# § Evaluation Integration (Glicko-2 + PromotionGate)
# ═══════════════════════════════════════════

try:
    from quartz.evaluation import (
        TrainingEvaluator, EvalConfig, PromotionVerdict,
        RatingLadder, PromotionGate, ChampionTracker,
        RandomEngine as EvalRandomEngine, MatchRunner, tally_match,
        GameAdapter, Engine as EvalEngine, GameRecord,
    )
    HAS_EVAL_SYSTEM = True
except ImportError:
    HAS_EVAL_SYSTEM = False


class RustNNEvaluatorEngine(_RustNNEvaluatorEngineImpl):
    def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo"):
        super().__init__(
            engine_name,
            cfg,
            model,
            device,
            rust_binary,
            runtime_hooks=EvaluatorRuntimeHooks(
                search_client_cls=NNSearchClient,
                is_chess_game=is_chess_game,
                build_rust_state_meta=build_rust_state_meta,
                iter_sparse_policy_entries=iter_sparse_policy_entries,
                supports_rust_eval_state_machine=supports_rust_eval_state_machine,
                stall_trace=_stall_trace,
                game_record_cls=GameRecord,
                tally_match=tally_match,
            ),
        )


build_training_game_adapter = _runtime_support.build_training_game_adapter


build_arg_parser = _torch_training_runtime.build_arg_parser


def prepare_training_context(args):
    return _prepare_training_context_impl(
        args,
        CliPrepareHooks(
            torch=torch,
            np=np,
            random_mod=random,
            game_configs=GAME_CONFIGS,
            get_encoder=get_encoder,
            apply_config_overrides=apply_config_overrides,
            is_go_game=is_go_game,
            default_output_dir=default_output_dir,
            resolve_runtime_paths=resolve_runtime_paths,
            auto_device_name=auto_device_name,
            detect_hardware_spec=detect_hardware_spec,
            configure_torch_rocm_runtime=configure_torch_rocm_runtime,
            supports_rust_eval_state_machine=supports_rust_eval_state_machine,
            supports_rust_selfplay_state_machine=supports_rust_selfplay_state_machine,
            autotune_training_cfg=autotune_training_cfg,
            clamp_runtime_cfg_to_hardware=clamp_runtime_cfg_to_hardware,
            max_supported_threads=max_supported_threads,
            gpu_host_thread_cap=gpu_host_thread_cap,
            gpu_interop_thread_cap=gpu_interop_thread_cap,
            alphazero_net_cls=AlphaZeroNet,
            load_torch_state_dict_checked=load_torch_state_dict_checked,
            get_actor_model=get_actor_model,
            load_autotune_profile=load_autotune_profile,
            apply_runtime_overrides=apply_runtime_overrides,
            run_autotune_benchmark=run_autotune_benchmark,
            save_autotune_profile=save_autotune_profile,
            probe_inference_batch_size=_probe_inference_batch_size,
            clamp_thread_count=clamp_thread_count,
        ),
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not hasattr(args, "autotune"):
        args.autotune = True
    ctx = prepare_training_context(args)
    return _run_training_main_impl(
        args,
        ctx,
        MainRuntimeHooks(
            torch=torch,
            np=np,
            game_configs=GAME_CONFIGS,
            serve=serve,
            arena_3agent=arena_3agent,
            arena_rust_nn=arena_rust_nn,
            arena_compare=arena_compare,
            print_autotune_summary=print_autotune_summary,
            is_go_game=is_go_game,
            replay_buffer_cls=ReplayBuffer,
            early_stopping_cls=EarlyStopping,
            early_stopping_enabled=early_stopping_enabled,
            load_eval_autotune_profile=load_eval_autotune_profile,
            has_eval_system=HAS_EVAL_SYSTEM,
            recommend_eval_parallel_workers=recommend_eval_parallel_workers,
            max_supported_threads=max_supported_threads,
            eval_config_cls=EvalConfig,
            training_evaluator_cls=TrainingEvaluator,
            build_training_game_adapter=build_training_game_adapter,
            ensure_best_checkpoint_compatible=ensure_best_checkpoint_compatible,
            selfplay_worker_cls=SelfPlayWorker,
            initial_replay_fill_target=initial_replay_fill_target,
            online_autotune_controller_cls=OnlineAutotuneController,
            clear_nn_eval_cache=clear_nn_eval_cache,
            round_or_none=round_or_none,
            wait_for_worker_progress=wait_for_worker_progress,
            selfplay_rust_nn_batched=selfplay_rust_nn_batched,
            compute_train_steps=compute_train_steps,
            train_epoch=train_epoch,
            replay_metrics=ReplayMetrics,
            rust_nn_evaluator_engine_cls=RustNNEvaluatorEngine,
            clone_actor_model=clone_actor_model,
            load_actor_source_from_checkpoint=load_actor_source_from_checkpoint,
            tree_mcts_engine_cls=TreeMCTSEngine,
            benchmark_eval_parallel_workers=benchmark_eval_parallel_workers,
            make_json_safe=make_json_safe,
            generate_training_plots=generate_training_plots,
        ),
    )

if __name__ == "__main__":
    main()
