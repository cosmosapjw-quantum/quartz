"""JAX-oriented runtime assembly without the legacy compatibility facade."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

from quartz.backend import load_torch_state_dict_checked
from quartz.cli_main import (
    CliPrepareHooks,
    MainRuntimeHooks,
    build_arg_parser as _build_arg_parser_impl,
    prepare_training_context as _prepare_training_context_impl,
    run_training_main as _run_training_main_impl,
)
from quartz.cli_runtime import CliRuntimeHooks, load_actor_source_from_checkpoint as _load_actor_source_from_checkpoint_impl
from quartz.encoders import get_encoder
from quartz.eval_runtime import clear_nn_eval_cache
from quartz.evaluation import EvalConfig, GameRecord, TrainingEvaluator, tally_match
from quartz.evaluator_runtime import (
    EvaluatorRuntimeHooks,
    InferencePipelineThread,
    RustNNEvaluatorEngine as _RustNNEvaluatorEngineImpl,
    ShmEvalRuntimeHooks,
    shm_eval_loop as _shm_eval_loop_impl,
)
from quartz.game_adapters import (
    ChessEvaluationAdapter,
    GameAdapterRuntimeHooks,
    GoGameAdapter,
    GomokuGameAdapter,
    TicTacToeGameAdapter,
    build_training_game_adapter as _build_training_game_adapter_impl,
)
from quartz.qipc import (
    QIPC_BATCH_EVAL_REQ,
    QIPC_BATCH_EVAL_RESP,
    QIPC_EVAL_REQ,
    QIPC_EVAL_RESP,
    SHM_MSG_EVAL_BATCH_REQ,
    SHM_MSG_JSON,
    SHM_MSG_SEARCH_RESP,
    cleanup_qipc_transport,
    launch_rust_server,
    pack_qipc_batch_eval_resp,
    pack_qipc_eval_resp,
    proc_decode_eval_frame,
    proc_read_json_line,
    proc_read_message,
    proc_write_eval_response,
    proc_write_json_line,
    stop_rust_server,
    unpack_qipc_batch_eval_req,
    unpack_qipc_eval_req,
    unpack_shm_search_response,
    wait_readable,
)
from quartz.replay import ReplayBuffer, ReplayMetrics, iter_sparse_policy_entries
from quartz.selfplay_runtime import (
    ArenaRuntimeHooks,
    LegacyRustSelfplayHooks,
    NNSearchClient as _NNSearchClientImpl,
    RustServerPool as _RustServerPoolImpl,
    SearchClientRuntimeHooks,
    SelfPlayLoopRuntimeHooks,
    SelfPlayWorker as _SelfPlayWorkerImpl,
    build_rust_state_meta as _build_rust_state_meta_impl,
    chess_state_meta_from_hashes,
    choose_selfplay_move,
    compute_train_steps,
    decode_streamed_selfplay_game,
    default_output_dir,
    encode_chess_fen,
    initial_chess_fen,
    initial_replay_fill_target,
    is_chess_game,
    is_go_game,
    rust_game_name as _rust_game_name_impl,
    rust_search_options,
    selfplay_rust as _selfplay_rust_impl,
    selfplay_rust_nn as _selfplay_rust_nn_impl,
    selfplay_rust_nn_batched as _selfplay_rust_nn_batched_impl,
    should_use_resident_session,
    supports_rust_eval_state_machine as _supports_rust_eval_state_machine_impl,
    supports_rust_selfplay_state_machine as _supports_rust_selfplay_state_machine_impl,
    wait_for_worker_progress,
)
from quartz.system_runtime import (
    HardwareSpec,
    clamp_runtime_cfg_to_hardware,
    clamp_thread_count,
    compute_eval_collect_policy,
    gpu_host_thread_cap,
    gpu_interop_thread_cap,
    load_eval_autotune_profile,
    max_supported_threads,
    recommend_eval_parallel_workers,
    save_eval_autotune_profile,
)
from quartz.train_loop import EarlyStopping, early_stopping_enabled, generate_training_plots, round_or_none, train_epoch
from quartz.training_catalog import (
    CHESS_POLICY_ACTIONS,
    GAME_CONFIGS,
    GOMOKU15_VARIANTS,
    STANDARD_CHESS_FEN,
    apply_config_overrides,
    resolve_runtime_paths,
)
from quartz.training_runtime_utils import benchmark_eval_parallel_workers, make_json_safe

try:
    from quartz.gpu_detect import detect_gpu
except Exception:
    detect_gpu = None

try:
    from tqdm import tqdm

    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    tqdm = None


log = logging.getLogger(__name__)


class _TorchShim:
    float32 = "float32"

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def manual_seed_all(_seed):
            return None

    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def save(*_args, **_kwargs):
        raise RuntimeError("torch save path is unavailable in JAX runtime")


class _TorchModelUnavailable:
    def __init__(self, _cfg):
        raise RuntimeError("PyTorch model construction is unavailable in JAX runtime")


def _detect_cpu_counts():
    logical = 0
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1
    logical = max(1, logical)
    physical = max(1, logical // 2)
    if sys.platform.startswith("linux"):
        try:
            pairs = set()
            current_phys = None
            current_core = None
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("physical id"):
                        current_phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        current_core = line.split(":", 1)[1].strip()
                    elif not line.strip():
                        if current_phys is not None and current_core is not None:
                            pairs.add((current_phys, current_core))
                        current_phys = None
                        current_core = None
            if pairs:
                physical = len(pairs)
        except Exception:
            pass
    return logical, physical


def _detect_memory_mb():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
    return 0


def detect_hardware_spec(device):
    logical, physical = _detect_cpu_counts()
    gpu_vendor = "none"
    gpu_name = ""
    gpu_vram_mb = 0
    gpu_count = 0
    if detect_gpu is not None:
        try:
            gpu_info = detect_gpu()
            gpu_vendor = gpu_info.vendor or gpu_vendor
            gpu_name = gpu_info.device_name or gpu_name
            gpu_vram_mb = gpu_info.vram_mb or gpu_vram_mb
            gpu_count = max(gpu_count, 1 if gpu_name or gpu_vendor != "none" else 0)
        except Exception:
            pass
    return HardwareSpec(
        logical_cpus=logical,
        physical_cpus=physical,
        memory_mb=_detect_memory_mb(),
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_count=gpu_count,
        torch_cuda=False,
        device_kind=str(device),
    )


def _autotune_progress_bar(total, desc):
    use_tqdm = HAS_TQDM and sys.stderr.isatty() and tqdm is not None
    if not use_tqdm:
        class _NullBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set_postfix_str(self, *_args, **_kwargs):
                return None

            def update(self, *_args, **_kwargs):
                return None

        return _NullBar()
    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True, disable=False)


def _run_model_batch(model, device, batch_features):
    batch_np = np.asarray(batch_features, dtype=np.float32)
    if not batch_np.flags.c_contiguous:
        batch_np = np.ascontiguousarray(batch_np)
    if hasattr(model, "predict"):
        probs_batch, vals_np = model.predict(batch_np)
        return np.asarray(probs_batch, dtype=np.float32), np.asarray(vals_np, dtype=np.float32).reshape(-1)

    import torch
    import torch.nn.functional as F

    x_batch = torch.from_numpy(batch_np).to(device)
    with torch.inference_mode():
        logits_batch, vals_batch = model(x_batch)
        probs_batch = F.softmax(logits_batch, dim=-1).cpu().numpy()
        vals_np = vals_batch.cpu().numpy()
    return probs_batch, vals_np


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


def ensure_best_checkpoint_compatible(best_model_path, backend, model, device):
    return runtime_support.ensure_best_checkpoint_compatible(best_model_path, backend, model, device)


def rust_game_name(game_name):
    return _rust_game_name_impl(game_name, GAME_CONFIGS, GOMOKU15_VARIANTS)


def build_rust_state_meta(game_name, state, cfg):
    return _build_rust_state_meta_impl(game_name, state, cfg, is_chess_game_fn=is_chess_game, is_go_game_fn=is_go_game)


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
        proc_write_json_line(
            proc,
            {"batch_eval_resp": {"responses": [{"policy": policy.tolist(), "value": float(value)} for policy, value in zip(policies, values)]}},
        )
    elif kind == "json_single":
        proc_write_json_line(proc, {"eval_resp": {"policy": policies[0].tolist(), "value": float(values[0])}})
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
        runtime_hooks=ShmEvalRuntimeHooks(
            run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: __import__("quartz.eval_runtime", fromlist=["run_batched_eval_groups"]).run_batched_eval_groups(groups, model_obj, dev, cfg_obj, _run_model_batch),
            make_eval_request_group=__import__("quartz.eval_runtime", fromlist=["make_eval_request_group"]).make_eval_request_group,
            unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
            unpack_shm_search_response=unpack_shm_search_response,
            json_loads_fast=json.loads,
            emit_duty_cycle=getattr(NNSearchClient, "_emit_duty_cycle", lambda duty: None),
            pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
            logger=log,
            shm_msg_eval_batch_req=SHM_MSG_EVAL_BATCH_REQ,
            shm_msg_json=SHM_MSG_JSON,
            shm_msg_search_resp=SHM_MSG_SEARCH_RESP,
            inference_pipeline_thread_cls=InferencePipelineThread,
        ),
    )


class RustServerPool(_RustServerPoolImpl):
    def __init__(self, rust_binary):
        super().__init__(rust_binary, launch_server=launch_rust_server, stop_server=stop_rust_server)


class NNSearchClient(_NNSearchClientImpl):
    def __init__(self, model, cfg, device, rust_binary="./target/release/mcts_demo"):
        from quartz.eval_runtime import make_eval_request_group, parse_eval_request, run_batched_eval_groups

        super().__init__(
            model,
            cfg,
            device,
            rust_binary,
            runtime_hooks=SearchClientRuntimeHooks(
                launch_server=launch_rust_server,
                proc_write_json_line=proc_write_json_line,
                cleanup_qipc_transport=cleanup_qipc_transport,
                proc_read_json_line=proc_read_json_line,
                json_loads_fast=json.loads,
                rust_game_name=rust_game_name,
                rust_search_options=rust_search_options,
                is_chess_game=is_chess_game,
                normalize_rust_board=lambda game_name, board_flat: [1 if v == 1 else 2 if v in (-1, 2) else 0 for v in board_flat] if is_go_game(game_name) else (board_flat.tolist() if hasattr(board_flat, "tolist") else list(board_flat)),
                proc_read_message=proc_read_message,
                shm_eval_loop=_shm_eval_loop,
                proc_decode_eval_frame=proc_decode_eval_frame,
                qipc_batch_eval_req=QIPC_BATCH_EVAL_REQ,
                qipc_eval_req=QIPC_EVAL_REQ,
                make_eval_request_group=make_eval_request_group,
                unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
                unpack_qipc_eval_req=unpack_qipc_eval_req,
                compute_eval_collect_policy=compute_eval_collect_policy,
                wait_readable=wait_readable,
                inference_pipeline_thread_cls=InferencePipelineThread,
                run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: run_batched_eval_groups(groups, model_obj, dev, cfg_obj, _run_model_batch),
                write_batched_eval_group=_write_batched_eval_group,
                run_model_batch=_run_model_batch,
                torch_module=_TorchShim,
                pack_qipc_eval_resp=pack_qipc_eval_resp,
                pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
                parse_eval_request=parse_eval_request,
            ),
        )


def selfplay_rust(cfg, n_games, rust_binary="./target/release/mcts_demo"):
    return _selfplay_rust_impl(
        cfg,
        n_games,
        rust_binary=rust_binary,
        runtime_hooks=LegacyRustSelfplayHooks(
            launch_rust_server=launch_rust_server,
            proc_write_json_line=proc_write_json_line,
            proc_read_json_line=proc_read_json_line,
            json_loads_fast=json.loads,
            rust_game_name=rust_game_name,
            rust_search_options=rust_search_options,
            is_go_game=is_go_game,
            tqdm_factory=_autotune_progress_bar,
        ),
    )


def build_training_game_adapter(cfg):
    return _build_training_game_adapter_impl(
        cfg,
        runtime_hooks=GameAdapterRuntimeHooks(
            is_chess_game=is_chess_game,
            is_go_game=is_go_game,
            initial_chess_fen=lambda game_cfg, rng=None: initial_chess_fen(game_cfg, rng=rng, standard_chess_fen=STANDARD_CHESS_FEN),
            encode_chess_fen=encode_chess_fen,
            gomoku15_variants=GOMOKU15_VARIANTS,
            chess_policy_actions=CHESS_POLICY_ACTIONS,
            gomoku_adapter_cls=GomokuGameAdapter,
            go_adapter_cls=GoGameAdapter,
            tictactoe_adapter_cls=TicTacToeGameAdapter,
            chess_adapter_cls=ChessEvaluationAdapter,
        ),
    )


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
            initial_chess_fen=lambda game_cfg, rng=None: initial_chess_fen(game_cfg, rng=rng, standard_chess_fen=STANDARD_CHESS_FEN),
            build_training_game_adapter=build_training_game_adapter,
            encode_chess_fen=encode_chess_fen,
            build_rust_state_meta=build_rust_state_meta,
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            choose_selfplay_move=choose_selfplay_move,
        ),
    )


def supports_rust_eval_state_machine(game_name):
    return _supports_rust_eval_state_machine_impl(game_name, rust_game_name, GAME_CONFIGS, GOMOKU15_VARIANTS, is_chess_game, is_go_game)


def supports_rust_selfplay_state_machine(game_name):
    return _supports_rust_selfplay_state_machine_impl(game_name, rust_game_name, GAME_CONFIGS, GOMOKU15_VARIANTS, is_chess_game, is_go_game)


def selfplay_rust_nn_batched(cfg, model, device, n_games, rust_binary="./target/release/mcts_demo", parallel=4, show_progress=True, proc_pool=None, perf_stats=None, on_game=None, active_proc_ref=None):
    return _selfplay_rust_nn_batched_impl(
        cfg,
        model,
        device,
        n_games,
        rust_binary=rust_binary,
        parallel=parallel,
        show_progress=show_progress,
        proc_pool=proc_pool,
        perf_stats=perf_stats,
        on_game=on_game,
        active_proc_ref=active_proc_ref,
        runtime_hooks=ArenaRuntimeHooks(
            is_chess_game=is_chess_game,
            search_client_cls=NNSearchClient,
            alphazero_net_cls=_TorchModelUnavailable,
            load_torch_state_dict=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("torch load unavailable in JAX runtime")),
            torch_module=_TorchShim,
            initial_chess_fen=lambda game_cfg, rng=None: initial_chess_fen(game_cfg, rng=rng, standard_chess_fen=STANDARD_CHESS_FEN),
            chess_state_meta_from_hashes=chess_state_meta_from_hashes,
            arena_compare=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("arena_compare unavailable in JAX runtime")),
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
            logger=logging.getLogger(__name__),
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
                search_client_cls=NNSearchClient,
                is_chess_game=is_chess_game,
                build_rust_state_meta=build_rust_state_meta,
                iter_sparse_policy_entries=iter_sparse_policy_entries,
                supports_rust_eval_state_machine=supports_rust_eval_state_machine,
                stall_trace=lambda *args, **kwargs: None,
                game_record_cls=GameRecord,
                tally_match=tally_match,
            ),
        )


def serve(model, cfg, device):
    from quartz.cli_runtime import serve as _serve_impl

    return _serve_impl(model, cfg, device, runtime_hooks=CliRuntimeHooks(alphazero_net_cls=_TorchModelUnavailable, load_torch_state_dict=lambda *args, **kwargs: None, run_model_batch=_run_model_batch))


def load_actor_source_from_checkpoint(checkpoint_path, cfg, device, backend_preference="torch", backend_template=None):
    return _load_actor_source_from_checkpoint_impl(
        checkpoint_path,
        cfg,
        device,
        backend_preference=backend_preference,
        backend_template=backend_template,
        runtime_hooks=CliRuntimeHooks(alphazero_net_cls=_TorchModelUnavailable, load_torch_state_dict=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("torch actor load unavailable in JAX runtime")), run_model_batch=_run_model_batch),
    )


def autotune_training_cfg(cfg, hw, concurrent=True):
    try:
        from quartz import models_torch as models_mod
        from quartz.autotune_runtime import autotune_training_cfg as _autotune_training_cfg

        return _autotune_training_cfg(cfg, hw, concurrent=concurrent, alphazero_net_cls=models_mod.AlphaZeroNet)
    except Exception:
        return dict(cfg)


def run_autotune_benchmark(cfg, backend, model, optimizer, device, hw, rust_binary, concurrent=True):
    from quartz.autotune_runtime import run_autotune_benchmark as _run_autotune_benchmark_impl
    from quartz.autotune_runtime import AutotuneRuntimeHooks

    return _run_autotune_benchmark_impl(
        cfg,
        backend,
        model,
        optimizer,
        device,
        hw,
        rust_binary,
        runtime_hooks=AutotuneRuntimeHooks(
            alphazero_net_cls=_TorchModelUnavailable,
            run_model_batch=_run_model_batch,
            selfplay_rust_nn_batched=selfplay_rust_nn_batched,
            stall_trace=lambda *args, **kwargs: None,
            tqdm_factory=_autotune_progress_bar,
        ),
        concurrent=concurrent,
    )


def build_arg_parser():
    return _build_arg_parser_impl(GAME_CONFIGS.keys())


def prepare_training_context(args):
    return _prepare_training_context_impl(
        args,
        CliPrepareHooks(
            torch=_TorchShim,
            np=np,
            random_mod=random,
            game_configs=GAME_CONFIGS,
            get_encoder=get_encoder,
            apply_config_overrides=apply_config_overrides,
            is_go_game=is_go_game,
            default_output_dir=default_output_dir,
            resolve_runtime_paths=resolve_runtime_paths,
            auto_device_name=lambda: "cpu",
            detect_hardware_spec=detect_hardware_spec,
            configure_torch_rocm_runtime=lambda _hw: None,
            supports_rust_eval_state_machine=supports_rust_eval_state_machine,
            supports_rust_selfplay_state_machine=supports_rust_selfplay_state_machine,
            autotune_training_cfg=autotune_training_cfg,
            clamp_runtime_cfg_to_hardware=clamp_runtime_cfg_to_hardware,
            max_supported_threads=max_supported_threads,
            gpu_host_thread_cap=gpu_host_thread_cap,
            gpu_interop_thread_cap=gpu_interop_thread_cap,
            alphazero_net_cls=_TorchModelUnavailable,
            load_torch_state_dict_checked=load_torch_state_dict_checked,
            get_actor_model=get_actor_model,
            load_autotune_profile=lambda *args, **kwargs: None,
            apply_runtime_overrides=lambda cfg, overrides: __import__("quartz.autotune_runtime", fromlist=["apply_runtime_overrides"]).apply_runtime_overrides(cfg, overrides),
            run_autotune_benchmark=run_autotune_benchmark,
            save_autotune_profile=lambda *args, **kwargs: None,
            probe_inference_batch_size=lambda model, device, cfg, cap: cfg.get("batch_size", 8),
            clamp_thread_count=clamp_thread_count,
        ),
    )


def _unsupported_arena(*_args, **_kwargs):
    raise RuntimeError("Arena modes are not wired through the direct JAX runtime yet.")


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
            torch=_TorchShim,
            np=np,
            game_configs=GAME_CONFIGS,
            serve=serve,
            arena_3agent=_unsupported_arena,
            arena_rust_nn=_unsupported_arena,
            arena_compare=_unsupported_arena,
            print_autotune_summary=lambda *args, **kwargs: None,
            is_go_game=is_go_game,
            replay_buffer_cls=ReplayBuffer,
            early_stopping_cls=EarlyStopping,
            early_stopping_enabled=early_stopping_enabled,
            load_eval_autotune_profile=lambda *args, **kwargs: None,
            has_eval_system=True,
            recommend_eval_parallel_workers=recommend_eval_parallel_workers,
            max_supported_threads=max_supported_threads,
            eval_config_cls=EvalConfig,
            training_evaluator_cls=TrainingEvaluator,
            build_training_game_adapter=build_training_game_adapter,
            ensure_best_checkpoint_compatible=ensure_best_checkpoint_compatible,
            selfplay_worker_cls=SelfPlayWorker,
            initial_replay_fill_target=initial_replay_fill_target,
            online_autotune_controller_cls=__import__("quartz.autotune_runtime", fromlist=["OnlineAutotuneController"]).OnlineAutotuneController,
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
