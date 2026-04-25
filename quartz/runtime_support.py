"""Shared runtime support without depending on the legacy training facade."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np

from quartz.backend import load_torch_state_dict
from quartz.cli_runtime import CliRuntimeHooks, load_actor_source_from_checkpoint as _load_actor_source_from_checkpoint_impl
from quartz.encoders import get_encoder
from quartz.eval_runtime import make_eval_request_group, parse_eval_request, run_batched_eval_groups
from quartz.evaluation import GameRecord, tally_match
from quartz.game_adapters import (
    ChessEvaluationAdapter,
    GameAdapterRuntimeHooks,
    GoGameAdapter,
    GomokuGameAdapter,
    TicTacToeGameAdapter,
    build_training_game_adapter as _build_training_game_adapter_impl,
)
from quartz.qipc import (
    QIPC_ARENA_EVAL_REQ,
    QIPC_ARENA_EVAL_RESP,
    QIPC_BATCH_EVAL_REQ,
    QIPC_BATCH_EVAL_RESP,
    QIPC_EVAL_REQ,
    QIPC_EVAL_RESP,
    SHM_MSG_EVAL_BATCH_REQ,
    SHM_MSG_ARENA_EVAL_RESP,
    SHM_MSG_JSON,
    SHM_MSG_SEARCH_RESP,
    cleanup_qipc_transport,
    launch_rust_server,
    pack_qipc_batch_eval_resp,
    pack_qipc_arena_eval_req,
    pack_qipc_eval_resp,
    proc_decode_eval_frame,
    proc_read_json_line,
    proc_read_message,
    proc_write_eval_response,
    proc_write_json_line,
    proc_write_qipc_frame,
    unpack_qipc_batch_eval_req,
    unpack_qipc_arena_eval_resp,
    unpack_qipc_eval_req,
    unpack_shm_search_response,
    wait_readable,
)
from quartz.selfplay_runtime import (
    NNSearchClient as _NNSearchClientImpl,
    SearchClientRuntimeHooks,
    build_rust_state_meta as _build_rust_state_meta_impl,
    encode_chess_fen,
    initial_chess_fen as _initial_chess_fen_impl,
    is_chess_game,
    is_go_game,
    rust_game_name as _rust_game_name_impl,
    rust_search_options,
    supports_rust_eval_state_machine as _supports_rust_eval_state_machine_impl,
)
from quartz.training_catalog import CHESS_POLICY_ACTIONS, GAME_CONFIGS, GOMOKU15_VARIANTS, STANDARD_CHESS_FEN

log = logging.getLogger(__name__)

try:
    import orjson
except Exception:
    orjson = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _torch_module():
    import torch

    return torch


def _alphazero_net_cls():
    from quartz.models_torch import AlphaZeroNet

    return AlphaZeroNet


_SEARCH_CLIENT_RESOLVER = None


def register_search_client_resolver(resolver):
    global _SEARCH_CLIENT_RESOLVER
    _SEARCH_CLIENT_RESOLVER = resolver


def resolve_search_client_cls(default_cls=None):
    default_cls = default_cls or NNSearchClient
    if callable(_SEARCH_CLIENT_RESOLVER):
        try:
            candidate = _SEARCH_CLIENT_RESOLVER()
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate
    candidates = []
    for module_name in ("quartz.alphazero_train", "alphazero_train", "quartz_alphazero_train"):
        module = sys.modules.get(module_name)
        if module is not None:
            candidates.append(module)
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and module_file.endswith(os.path.join("quartz", "alphazero_train.py")):
            candidates.append(module)
    seen = set()
    for module in candidates:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        candidate = getattr(module, "NNSearchClient", None)
        if candidate is not None:
            return candidate
    return default_cls


def json_loads_fast(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


SEARCH_MANIFEST_KEYS = (
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
    "batch_size",
    "batch_timeout_us",
    "_eval_runner_mode",
    "_arena_low_concurrency_profile",
    # P7 (audit_codex_20260425.md W2): include halt_mode in the
    # search-manifest hash so two runs with different halt policies
    # produce distinct manifest hashes (the eval_matrix uses the hash
    # to detect engine drift across rows).
    "halt_mode",
)


def build_search_manifest(cfg):
    manifest = {}
    for key in SEARCH_MANIFEST_KEYS:
        if key in cfg:
            value = cfg.get(key)
            if value is None:
                continue
            manifest[key] = value
    return manifest


def search_manifest_hash(cfg):
    manifest = build_search_manifest(cfg)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rust_game_name(game_name):
    return _rust_game_name_impl(game_name, GAME_CONFIGS, GOMOKU15_VARIANTS)


def initial_chess_fen(cfg, rng=None):
    return _initial_chess_fen_impl(cfg, rng=rng, standard_chess_fen=STANDARD_CHESS_FEN)


def build_rust_state_meta(game_name, state, cfg):
    return _build_rust_state_meta_impl(game_name, state, cfg, is_chess_game_fn=is_chess_game, is_go_game_fn=is_go_game)


def supports_rust_eval_state_machine(game_name):
    return _supports_rust_eval_state_machine_impl(
        game_name,
        rust_game_name,
        GAME_CONFIGS,
        GOMOKU15_VARIANTS,
        is_chess_game,
        is_go_game,
    )


def encode_board(cfg, board_flat, player):
    enc_obj = cfg.get("_encoder")
    if enc_obj is not None:
        return enc_obj.encode(board_flat, player)
    bs = cfg["board"]
    n2 = bs * bs
    ch = cfg.get("ch", 17)
    enc = np.zeros((ch, bs, bs), dtype=np.float32)
    # [OPT] Vectorized board encoding — replaces per-cell Python loop
    board_arr = np.asarray(board_flat, dtype=np.int8).ravel()[:n2]
    my_val = np.int8(player)
    enc[0].ravel()[:len(board_arr)] = (board_arr == my_val).astype(np.float32)
    opp_mask = (board_arr != 0) & (board_arr != my_val)
    enc[1].ravel()[:len(board_arr)] = opp_mask.astype(np.float32)
    if player == 1:
        enc[ch - 1] = 1.0
    return enc


def decode_board(cfg, enc, player):
    enc_obj = cfg.get("_encoder")
    if enc_obj is not None:
        return enc_obj.decode(enc, player)
    bs = cfg["board"]
    board = np.zeros(bs * bs, dtype=np.int8)
    for r in range(bs):
        for c in range(bs):
            if enc[0, r, c] > 0.5:
                board[r * bs + c] = player
            elif enc[1, r, c] > 0.5:
                board[r * bs + c] = -player
    return board


def normalize_rust_board(game_name, board_flat):
    if board_flat is None:
        return None
    if is_go_game(game_name):
        # [OPT] Vectorized: convert -1→2, keep 1→1, rest→0
        arr = np.asarray(board_flat, dtype=np.int8)
        out = np.where(arr == 1, np.int8(1), np.where((arr == -1) | (arr == 2), np.int8(2), np.int8(0)))
        return out.tolist()
    return board_flat.tolist() if hasattr(board_flat, "tolist") else list(board_flat)


def _try_compile_model(model):
    """Lazily wrap model with torch.compile for Triton acceleration.

    The compiled model is cached on the original model object so compilation
    happens only once per model instance.  Falls back to eager if compile
    is unavailable or fails (e.g. CPU-only, older PyTorch).

    Set QUARTZ_NO_COMPILE=1 to disable (useful for debugging segfaults).
    """
    if os.environ.get("QUARTZ_NO_COMPILE", "") == "1":
        return model
    compiled = getattr(model, "_quartz_compiled", None)
    if compiled is not None:
        return compiled
    try:
        torch = _torch_module()
        if not hasattr(torch, "compile"):
            return model
        # 'default' mode avoids CUDA graphs that break on ROCm/HIP stream capture
        compiled = torch.compile(model, backend="inductor")
        # Tag so we don't compile again
        object.__setattr__(model, "_quartz_compiled", compiled)
        return compiled
    except Exception:
        object.__setattr__(model, "_quartz_compiled", model)
        return model


def _run_model_batch(model, device, batch_features):
    batch_np = np.asarray(batch_features, dtype=np.float32)
    if not batch_np.flags.c_contiguous:
        batch_np = np.ascontiguousarray(batch_np)
    if hasattr(model, "predict"):
        probs_batch, vals_np = model.predict(batch_np)
        return np.asarray(probs_batch, dtype=np.float32), np.asarray(vals_np, dtype=np.float32).reshape(-1)
    torch = _torch_module()
    # [OPT] Use torch.compile (Triton) for ~2-3x inference speedup on GPU
    run_model = _try_compile_model(model) if str(device) != "cpu" else model
    x_batch = torch.from_numpy(batch_np).to(device)
    with torch.inference_mode():
        logits_batch, vals_batch = run_model(x_batch)
        probs_batch = torch.softmax(logits_batch, dim=-1).cpu().numpy()
        vals_np = vals_batch.cpu().numpy()
    return probs_batch, vals_np


run_model_batch = _run_model_batch


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
            lambda groups, model_obj, dev, cfg_obj: run_batched_eval_groups(groups, model_obj, dev, cfg_obj, _run_model_batch)
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


def should_use_async_pipeline(model, device, cfg):
    if os.environ.get("QUARTZ_FORCE_ASYNC_PIPELINE"):
        return bool(model is not None and not hasattr(model, "predict"))
    if os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE"):
        return False
    if model is None or hasattr(model, "predict"):
        return False
    if str(device) == "cpu":
        return False
    batch_size = int(cfg.get("batch_size", 0) or 0)
    return batch_size > 1


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
        proc_write_json_line(proc, {"batch_eval_resp": {"responses": [{"policy": policy.tolist(), "value": float(value)} for policy, value in zip(policies, values)]}})
    elif kind == "json_single":
        proc_write_json_line(proc, {"eval_resp": {"policy": policies[0].tolist(), "value": float(values[0])}})
    else:
        raise ValueError(f"unknown eval response group kind: {kind}")


def _shm_eval_loop(ring, model, device, cfg, proc, on_json=None, baseline_epoch=None):
    from quartz.evaluator_runtime import ShmEvalRuntimeHooks, shm_eval_loop as _shm_eval_loop_impl

    return _shm_eval_loop_impl(
        ring,
        model,
        device,
        cfg,
        proc,
        on_json=on_json,
        baseline_epoch=baseline_epoch,
        runtime_hooks=ShmEvalRuntimeHooks(
            run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: run_batched_eval_groups(groups, model_obj, dev, cfg_obj, _run_model_batch),
            make_eval_request_group=make_eval_request_group,
            unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
            unpack_shm_search_response=unpack_shm_search_response,
            unpack_qipc_arena_eval_resp=unpack_qipc_arena_eval_resp,
            json_loads_fast=json_loads_fast,
            emit_duty_cycle=getattr(resolve_search_client_cls(), "_emit_duty_cycle", lambda duty: None),
            pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
            logger=log,
            shm_msg_eval_batch_req=SHM_MSG_EVAL_BATCH_REQ,
            shm_msg_arena_eval_resp=SHM_MSG_ARENA_EVAL_RESP,
            shm_msg_json=SHM_MSG_JSON,
            shm_msg_search_resp=SHM_MSG_SEARCH_RESP,
            inference_pipeline_thread_cls=InferencePipelineThread,
            should_use_async_pipeline=should_use_async_pipeline,
        ),
    )


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
                proc_write_qipc_frame=proc_write_qipc_frame,
                cleanup_qipc_transport=cleanup_qipc_transport,
                proc_read_json_line=proc_read_json_line,
                json_loads_fast=json_loads_fast,
                rust_game_name=rust_game_name,
                rust_search_options=rust_search_options,
                is_chess_game=is_chess_game,
                normalize_rust_board=normalize_rust_board,
                proc_read_message=proc_read_message,
                shm_eval_loop=_shm_eval_loop,
                proc_decode_eval_frame=proc_decode_eval_frame,
                qipc_arena_eval_req=QIPC_ARENA_EVAL_REQ,
                qipc_arena_eval_resp=QIPC_ARENA_EVAL_RESP,
                qipc_batch_eval_req=QIPC_BATCH_EVAL_REQ,
                qipc_eval_req=QIPC_EVAL_REQ,
                make_eval_request_group=make_eval_request_group,
                pack_qipc_arena_eval_req=pack_qipc_arena_eval_req,
                unpack_qipc_batch_eval_req=unpack_qipc_batch_eval_req,
                unpack_qipc_arena_eval_resp=unpack_qipc_arena_eval_resp,
                unpack_qipc_eval_req=unpack_qipc_eval_req,
                compute_eval_collect_policy=__import__("quartz.system_runtime", fromlist=["compute_eval_collect_policy"]).compute_eval_collect_policy,
                wait_readable=wait_readable,
                inference_pipeline_thread_cls=InferencePipelineThread,
                should_use_async_pipeline=should_use_async_pipeline,
                run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: run_batched_eval_groups(groups, model_obj, dev, cfg_obj, _run_model_batch),
                write_batched_eval_group=_write_batched_eval_group,
                run_model_batch=_run_model_batch,
                torch_module=_torch_module(),
                pack_qipc_eval_resp=pack_qipc_eval_resp,
                pack_qipc_batch_eval_resp=pack_qipc_batch_eval_resp,
                parse_eval_request=parse_eval_request,
            ),
        )


def build_training_game_adapter(cfg):
    return _build_training_game_adapter_impl(
        cfg,
        runtime_hooks=GameAdapterRuntimeHooks(
            is_chess_game=is_chess_game,
            is_go_game=is_go_game,
            initial_chess_fen=initial_chess_fen,
            encode_chess_fen=encode_chess_fen,
            gomoku15_variants=GOMOKU15_VARIANTS,
            chess_policy_actions=CHESS_POLICY_ACTIONS,
            gomoku_adapter_cls=GomokuGameAdapter,
            go_adapter_cls=GoGameAdapter,
            tictactoe_adapter_cls=TicTacToeGameAdapter,
            chess_adapter_cls=ChessEvaluationAdapter,
        ),
    )


def load_actor_source_from_checkpoint(checkpoint_path, cfg, device, backend_preference="torch", backend_template=None):
    return _load_actor_source_from_checkpoint_impl(
        checkpoint_path,
        cfg,
        device,
        backend_preference=backend_preference,
        backend_template=backend_template,
        runtime_hooks=CliRuntimeHooks(
            alphazero_net_cls=_alphazero_net_cls(),
            load_torch_state_dict=load_torch_state_dict,
            run_model_batch=_run_model_batch,
        ),
    )


def detect_checkpoint_backend_hint(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(512)
    except OSError:
        return "missing"
    if not head:
        return "empty"
    if head.startswith(b"PK\x03\x04"):
        return "torch"
    if b"jax._src" in head or b"flax" in head:
        return "jax"
    if head.startswith(b"\x80\x04") and b"params" in head and b"BatchNorm_" in head:
        return "jax"
    return "unknown"


def ensure_best_checkpoint_compatible(best_model_path, backend, model, device):
    if not os.path.exists(best_model_path):
        return None
    active_backend = getattr(backend, "name", "torch") if backend is not None else "torch"
    hint = detect_checkpoint_backend_hint(best_model_path)
    mismatch = (
        (active_backend == "torch" and hint == "jax")
        or (active_backend == "jax" and hint == "torch")
    )
    if not mismatch:
        return hint
    if backend is not None:
        backend.save(best_model_path)
    raise RuntimeError(f"incompatible best checkpoint for runtime backend: {hint} -> {active_backend}")


def default_encoder_cfg(game_name):
    cfg = dict(GAME_CONFIGS[game_name])
    cfg["_name"] = game_name
    try:
        cfg["_encoder"] = get_encoder(game_name)
    except KeyError:
        cfg["_encoder"] = None
    return cfg


def tqdm_factory(*args, **kwargs):
    if tqdm is None:
        return contextlib.nullcontext(SimpleNamespace(update=lambda *_a, **_k: None, set_postfix_str=lambda *_a, **_k: None))
    return tqdm(*args, **kwargs)


def __getattr__(name):
    if name == "torch":
        return _torch_module()
    if name == "AlphaZeroNet":
        return _alphazero_net_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlphaZeroNet",  # noqa: F822 — lazy attribute via module __getattr__
    "GAME_CONFIGS",
    "GameRecord",
    "InferencePipelineThread",
    "should_use_async_pipeline",
    "NNSearchClient",
    "build_rust_state_meta",
    "build_search_manifest",
    "build_training_game_adapter",
    "decode_board",
    "detect_checkpoint_backend_hint",
    "default_encoder_cfg",
    "encode_board",
    "encode_chess_fen",
    "ensure_best_checkpoint_compatible",
    "initial_chess_fen",
    "is_chess_game",
    "is_go_game",
    "json_loads_fast",
    "launch_rust_server",
    "load_actor_source_from_checkpoint",
    "load_torch_state_dict",
    "normalize_rust_board",
    "proc_read_json_line",
    "proc_read_message",
    "proc_write_json_line",
    "run_model_batch",
    "register_search_client_resolver",
    "resolve_search_client_cls",
    "rust_game_name",
    "search_manifest_hash",
    "supports_rust_eval_state_machine",
    "tally_match",
    "torch",  # noqa: F822 — lazy attribute via module __getattr__
    "tqdm_factory",
]
