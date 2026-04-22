"""PyTorch-oriented runtime assembly without the legacy compatibility facade."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch

from quartz import runtime_support
from quartz.autotune_runtime import (
    AutotuneRuntimeHooks,
    OnlineAutotuneController,
    apply_runtime_overrides,
    autotune_signature,
    autotune_training_cfg as _autotune_training_cfg_impl,
    benchmark_selfplay_throughput as _benchmark_selfplay_throughput_impl,
    benchmark_train_batch as _benchmark_train_batch_impl,
    load_autotune_profile as _load_autotune_profile_impl,
    print_autotune_summary,
    run_autotune_benchmark as _run_autotune_benchmark_impl,
    run_autotune_benchmark_fast as _run_autotune_benchmark_fast_impl,
    save_autotune_profile as _save_autotune_profile_impl,
)
from quartz.cli_main import (
    CliPrepareHooks,
    MainRuntimeHooks,
    build_arg_parser as _build_arg_parser_impl,
    prepare_training_context as _prepare_training_context_impl,
    run_training_main as _run_training_main_impl,
)
from quartz.cli_runtime import CliRuntimeHooks, serve as _serve_impl
from quartz.evaluation import EvalConfig, TrainingEvaluator
from quartz.evaluator_runtime import EvaluatorRuntimeHooks, RustNNEvaluatorEngine as _RustNNEvaluatorEngineImpl
from quartz.qipc import (
    QIPC_BATCH_EVAL_REQ,
    QIPC_EVAL_REQ,
    launch_rust_server,
    proc_decode_eval_frame,
    proc_read_json_line,
    proc_read_message,
    proc_write_json_line,
    stop_rust_server,
    wait_readable,
)
from quartz.replay import ReplayBuffer, ReplayMetrics, iter_sparse_policy_entries
from quartz.selfplay_runtime import (
    BatchedSelfPlayRuntimeHooks,
    RustServerPool as _RustServerPoolImpl,
    SelfPlayWorker as _SelfPlayWorkerImpl,
    chess_state_meta_from_hashes,
    choose_selfplay_move,
    compute_train_steps,
    decode_streamed_selfplay_game,
    initial_replay_fill_target,
    should_use_resident_session,
    supports_rust_selfplay_state_machine as _supports_rust_selfplay_state_machine_impl,
    wait_for_worker_progress,
    selfplay_rust_nn_batched as _selfplay_rust_nn_batched_impl,
)
from quartz.system_runtime import (
    clamp_runtime_cfg_to_hardware,
    clamp_thread_count,
    compute_eval_collect_policy,
    configure_torch_rocm_runtime,
    detect_hardware_spec,
    gpu_host_thread_cap,
    gpu_interop_thread_cap,
    hardware_signature,
    load_eval_autotune_profile,
    max_supported_threads,
    recommend_eval_parallel_workers,
    save_eval_autotune_profile,
)
from quartz.train_loop import EarlyStopping, early_stopping_enabled, generate_training_plots, round_or_none, train_epoch
from quartz.training_catalog import GAME_CONFIGS, GOMOKU15_VARIANTS, apply_config_overrides, resolve_runtime_paths
from quartz.training_runtime_utils import benchmark_eval_parallel_workers, make_json_safe

log = logging.getLogger(__name__)


def build_arg_parser():
    return _build_arg_parser_impl(GAME_CONFIGS.keys())


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


def get_actor_model(training_model, backend):
    return backend if backend is not None else training_model


def supports_rust_eval_state_machine(game_name):
    return runtime_support.supports_rust_eval_state_machine(game_name)


def supports_rust_selfplay_state_machine(game_name):
    return _supports_rust_selfplay_state_machine_impl(
        game_name,
        runtime_support.rust_game_name,
        GAME_CONFIGS,
        GOMOKU15_VARIANTS,
        runtime_support.is_chess_game,
        runtime_support.is_go_game,
    )


def arena_rust_nn(
    model_a_path,
    model_b_path,
    cfg,
    device,
    n_games=50,
    rust_binary="./target/release/mcts_demo",
    strict=True,
):
    from quartz.arena_runtime import arena_compare
    from quartz.selfplay_runtime import ArenaRuntimeHooks, arena_rust_nn_impl

    return arena_rust_nn_impl(
        model_a_path,
        cfg,
        model_b_path,
        cfg,
        device,
        n_games=n_games,
        rust_binary=rust_binary,
        strict=strict,
        runtime_hooks=ArenaRuntimeHooks(
            is_chess_game=runtime_support.is_chess_game,
            search_client_cls=runtime_support.resolve_search_client_cls(),
            alphazero_net_cls=runtime_support.AlphaZeroNet,
            load_torch_state_dict=runtime_support.load_torch_state_dict,
            torch_module=torch,
            initial_chess_fen=runtime_support.initial_chess_fen,
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            arena_compare=arena_compare,
            build_training_game_adapter=runtime_support.build_training_game_adapter,
            rust_nn_evaluator_engine_cls=RustNNEvaluatorEngine,
            match_runner_cls=__import__("quartz.evaluation", fromlist=["MatchRunner"]).MatchRunner,
        ),
    )


class RustServerPool(_RustServerPoolImpl):
    def __init__(self, rust_binary):
        super().__init__(rust_binary, launch_server=launch_rust_server, stop_server=stop_rust_server)


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
):
    return _selfplay_rust_nn_batched_impl(
        cfg,
        model,
        device,
        n_games,
        rust_binary,
        parallel=parallel,
        show_progress=show_progress,
        proc_pool=proc_pool,
        perf_stats=perf_stats,
        on_game=on_game,
        active_proc_ref=active_proc_ref,
        runtime_hooks=BatchedSelfPlayRuntimeHooks(
            is_chess_game=runtime_support.is_chess_game,
            is_go_game=runtime_support.is_go_game,
            should_use_resident_session=should_use_resident_session,
            supports_rust_selfplay_state_machine=supports_rust_selfplay_state_machine,
            search_client_cls=runtime_support.resolve_search_client_cls(),
            decode_streamed_selfplay_game=decode_streamed_selfplay_game,
            encode_chess_fen=runtime_support.encode_chess_fen,
            initial_chess_fen=runtime_support.initial_chess_fen,
            build_training_game_adapter=runtime_support.build_training_game_adapter,
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            rust_game_name=runtime_support.rust_game_name,
            normalize_rust_board=runtime_support.normalize_rust_board,
            build_rust_state_meta=runtime_support.build_rust_state_meta,
            choose_selfplay_move=choose_selfplay_move,
            proc_decode_eval_frame=proc_decode_eval_frame,
            qipc_batch_eval_req=QIPC_BATCH_EVAL_REQ,
            qipc_eval_req=QIPC_EVAL_REQ,
            unpack_qipc_batch_eval_req=__import__("quartz.qipc", fromlist=["unpack_qipc_batch_eval_req"]).unpack_qipc_batch_eval_req,
            unpack_qipc_eval_req=__import__("quartz.qipc", fromlist=["unpack_qipc_eval_req"]).unpack_qipc_eval_req,
            make_eval_request_group=__import__("quartz.eval_runtime", fromlist=["make_eval_request_group"]).make_eval_request_group,
            stall_trace=lambda *args, **kwargs: None,
            proc_write_json_line=proc_write_json_line,
            proc_read_json_line=proc_read_json_line,
            proc_read_message=proc_read_message,
            shm_eval_loop=runtime_support._shm_eval_loop,
            wait_readable=wait_readable,
            compute_eval_collect_policy=compute_eval_collect_policy,
            inference_pipeline_thread_cls=runtime_support.InferencePipelineThread,
            should_use_async_pipeline=runtime_support.should_use_async_pipeline,
            run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: __import__("quartz.eval_runtime", fromlist=["run_batched_eval_groups"]).run_batched_eval_groups(groups, model_obj, dev, cfg_obj, runtime_support.run_model_batch),
            write_batched_eval_group=runtime_support._write_batched_eval_group,
            rust_search_options=runtime_support.rust_search_options,
            launch_server=launch_rust_server,
            stop_server=stop_rust_server,
            emit_duty_cycle=getattr(runtime_support.resolve_search_client_cls(), "_emit_duty_cycle", lambda duty: None),
        ),
    )


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
            logger=log,
        )


class RustNNEvaluatorEngine(_RustNNEvaluatorEngineImpl):
    def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo"):
        super().__init__(
            engine_name,
            cfg,
            model,
            device,
            rust_binary,
            runtime_hooks=EvaluatorRuntimeHooks(
                search_client_cls=runtime_support.resolve_search_client_cls(),
                is_chess_game=runtime_support.is_chess_game,
                build_rust_state_meta=runtime_support.build_rust_state_meta,
                iter_sparse_policy_entries=iter_sparse_policy_entries,
                supports_rust_eval_state_machine=supports_rust_eval_state_machine,
                stall_trace=lambda *args, **kwargs: None,
                game_record_cls=runtime_support.GameRecord,
                tally_match=runtime_support.tally_match,
            ),
        )


def benchmark_selfplay_throughput(cfg, model, device, rust_binary, hw, concurrent=True):
    return _benchmark_selfplay_throughput_impl(
        cfg,
        model,
        device,
        rust_binary,
        hw,
        runtime_hooks=AutotuneRuntimeHooks(
            alphazero_net_cls=runtime_support.AlphaZeroNet,
            run_model_batch=runtime_support.run_model_batch,
            selfplay_rust_nn_batched=selfplay_rust_nn_batched,
            stall_trace=lambda *args, **kwargs: None,
            tqdm_factory=runtime_support.tqdm_factory,
        ),
        concurrent=concurrent,
    )


def run_autotune_benchmark(cfg, backend, model, optimizer, device, hw, rust_binary, concurrent=True):
    # Use fast inference-only benchmark by default (~10-30s instead of 5-15min)
    return _run_autotune_benchmark_fast_impl(
        cfg, backend, model, optimizer, device, hw,
        rust_binary=rust_binary, runtime_hooks=None, concurrent=concurrent,
    )


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
                runtime_support.run_model_batch(model, device, batch)
            loops = max(20, 200 // cand)
            t0 = __import__("time").perf_counter()
            for _ in range(loops):
                runtime_support.run_model_batch(model, device, batch)
            elapsed = __import__("time").perf_counter() - t0
            ips = cand * loops / max(elapsed, 1e-9)
            if ips > best_ips:
                best_ips = ips
                best_bs = cand
        except Exception:
            break
    return best_bs


def detect_hw(device):
    return detect_hardware_spec(device, detect_gpu_fn=__import__("quartz.gpu_detect", fromlist=["detect_gpu"]).detect_gpu)


def load_autotune_profile(profile_path, hw, cfg):
    return _load_autotune_profile_impl(profile_path, hw, cfg, hardware_signature)


def save_autotune_profile(profile_path, hw, cfg, overrides, benchmarks):
    return _save_autotune_profile_impl(profile_path, hw, cfg, overrides, benchmarks, hardware_signature)


def autotune_training_cfg(cfg, hw, concurrent=True):
    return _autotune_training_cfg_impl(cfg, hw, concurrent=concurrent, alphazero_net_cls=runtime_support.AlphaZeroNet)


def serve(model, cfg, device):
    return _serve_impl(
        model,
        cfg,
        device,
        runtime_hooks=CliRuntimeHooks(
            alphazero_net_cls=runtime_support.AlphaZeroNet,
            load_torch_state_dict=runtime_support.load_torch_state_dict,
            run_model_batch=runtime_support.run_model_batch,
        ),
    )


def prepare_training_context(args):
    return _prepare_training_context_impl(
        args,
        CliPrepareHooks(
            torch=torch,
            np=np,
            random_mod=random,
            game_configs=GAME_CONFIGS,
            get_encoder=__import__("quartz.encoders", fromlist=["get_encoder"]).get_encoder,
            apply_config_overrides=apply_config_overrides,
            is_go_game=runtime_support.is_go_game,
            default_output_dir=__import__("quartz.selfplay_runtime", fromlist=["default_output_dir"]).default_output_dir,
            resolve_runtime_paths=resolve_runtime_paths,
            auto_device_name=__import__("quartz.system_runtime", fromlist=["auto_device_name"]).auto_device_name,
            detect_hardware_spec=detect_hw,
            configure_torch_rocm_runtime=configure_torch_rocm_runtime,
            supports_rust_eval_state_machine=supports_rust_eval_state_machine,
            supports_rust_selfplay_state_machine=supports_rust_selfplay_state_machine,
            autotune_training_cfg=autotune_training_cfg,
            clamp_runtime_cfg_to_hardware=clamp_runtime_cfg_to_hardware,
            max_supported_threads=max_supported_threads,
            gpu_host_thread_cap=gpu_host_thread_cap,
            gpu_interop_thread_cap=gpu_interop_thread_cap,
            alphazero_net_cls=runtime_support.AlphaZeroNet,
            load_torch_state_dict_checked=__import__("quartz.backend", fromlist=["load_torch_state_dict_checked"]).load_torch_state_dict_checked,
            get_actor_model=get_actor_model,
            load_autotune_profile=load_autotune_profile,
            apply_runtime_overrides=apply_runtime_overrides,
            run_autotune_benchmark=run_autotune_benchmark,
            save_autotune_profile=save_autotune_profile,
            probe_inference_batch_size=_probe_inference_batch_size,
            clamp_thread_count=clamp_thread_count,
        ),
    )


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
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
            arena_3agent=__import__("quartz.arena_runtime", fromlist=["arena_3agent"]).arena_3agent,
            arena_rust_nn=arena_rust_nn,
            arena_compare=__import__("quartz.arena_runtime", fromlist=["arena_compare"]).arena_compare,
            print_autotune_summary=print_autotune_summary,
            is_go_game=runtime_support.is_go_game,
            replay_buffer_cls=ReplayBuffer,
            early_stopping_cls=EarlyStopping,
            early_stopping_enabled=early_stopping_enabled,
            load_eval_autotune_profile=load_eval_autotune_profile,
            has_eval_system=True,
            recommend_eval_parallel_workers=recommend_eval_parallel_workers,
            max_supported_threads=max_supported_threads,
            eval_config_cls=EvalConfig,
            training_evaluator_cls=TrainingEvaluator,
            build_training_game_adapter=runtime_support.build_training_game_adapter,
            ensure_best_checkpoint_compatible=runtime_support.ensure_best_checkpoint_compatible,
            selfplay_worker_cls=SelfPlayWorker,
            initial_replay_fill_target=initial_replay_fill_target,
            online_autotune_controller_cls=OnlineAutotuneController,
            clear_nn_eval_cache=__import__("quartz.eval_runtime", fromlist=["clear_nn_eval_cache"]).clear_nn_eval_cache,
            round_or_none=round_or_none,
            wait_for_worker_progress=wait_for_worker_progress,
            selfplay_rust_nn_batched=selfplay_rust_nn_batched,
            compute_train_steps=compute_train_steps,
            train_epoch=train_epoch,
            replay_metrics=ReplayMetrics,
            rust_nn_evaluator_engine_cls=RustNNEvaluatorEngine,
            clone_actor_model=clone_actor_model,
            load_actor_source_from_checkpoint=runtime_support.load_actor_source_from_checkpoint,
            tree_mcts_engine_cls=__import__("quartz.arena_runtime", fromlist=["TreeMCTSEngine"]).TreeMCTSEngine,
            benchmark_eval_parallel_workers=lambda hw, cfg, eval_games, candidate_factory, champion_factory, game_factory, profile_path: benchmark_eval_parallel_workers(
                hw,
                cfg,
                eval_games,
                candidate_factory,
                champion_factory,
                game_factory,
                profile_path,
                has_eval_system=True,
                eval_worker_candidates_fn=__import__("quartz.system_runtime", fromlist=["eval_worker_candidates"]).eval_worker_candidates,
                eval_config_cls=EvalConfig,
                training_evaluator_cls=TrainingEvaluator,
                save_eval_autotune_profile_fn=save_eval_autotune_profile,
            ),
            make_json_safe=make_json_safe,
            generate_training_plots=generate_training_plots,
        ),
    )


__all__ = ["build_arg_parser", "main", "prepare_training_context"]
