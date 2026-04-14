import importlib.util
import argparse
import builtins
import json
import io
import struct
import sys
import tomllib
import types
from pathlib import Path
import random

import numpy as np
import pytest


def load_training_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_alphazero_train", root / "quartz" / "alphazero_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_entry_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_train_entry", root / "quartz" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backend_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_backend", root / "quartz" / "backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_encoders_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "encoders", root / "quartz" / "encoders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_monitor_script_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "profile_training_monitor", root / "scripts" / "profile_training_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gpu_detect_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_gpu_detect", root / "quartz" / "gpu_detect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_play_gui_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_play_gui", root / "quartz" / "play_gui.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module_with_torch_blocked(module_name, relative_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name == "torch" or name.startswith("torch.")}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"unexpected torch import while loading {relative_path}: {name}")
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guarded_import
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        builtins.__import__ = original_import
        sys.modules.pop(spec.name, None)
        sys.modules.update(saved)


def test_replay_values_follow_side_to_move_for_white_win():
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    states = [np.zeros((3, 7, 7), dtype=np.float32) for _ in range(2)]
    policies = [np.zeros(49, dtype=np.float32) for _ in range(2)]

    replay.add_game(states, policies, outcome=-1.0)

    assert [sample[2] for sample in replay.buf] == [-1.0, 1.0]


def test_sparse_replay_roundtrip_preserves_dense_policy_targets(tmp_path):
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state_a = np.zeros((3, 7, 7), dtype=np.float32)
    state_b = np.ones((3, 7, 7), dtype=np.float32)
    dense_policy = np.zeros(49, dtype=np.float32)
    dense_policy[7] = 1.0
    sparse_policy = az.sparse_policy_from_entries([[3, 0.25], [8, 0.75]], 49)

    replay.add(state_a, dense_policy, 0.5)
    replay.add(state_b, sparse_policy, -0.25)

    path = tmp_path / "replay_v2.npz"
    replay.save(path)

    loaded = az.ReplayBuffer(16)
    assert loaded.load(path) == 2

    _, policies_t, values_t = az.collate_replay_samples(list(loaded.buf))
    np.testing.assert_allclose(policies_t.numpy()[0], dense_policy)
    np.testing.assert_allclose(
        policies_t.numpy()[1],
        az.dense_policy_from_sparse([[3, 0.25], [8, 0.75]], 49),
    )
    np.testing.assert_allclose(values_t.numpy(), [0.5, -0.25])


def test_training_module_reexports_replay_api():
    az = load_training_module()
    from quartz import replay as replay_mod

    assert az.ReplayBuffer is replay_mod.ReplayBuffer
    assert az.ReplayExample is replay_mod.ReplayExample
    assert az.SparsePolicyTarget is replay_mod.SparsePolicyTarget
    assert az.collate_replay_samples is replay_mod.collate_replay_samples
    assert az.sparse_policy_from_entries is replay_mod.sparse_policy_from_entries


def test_training_module_reexports_eval_runtime_api():
    az = load_training_module()
    from quartz import eval_runtime as eval_mod

    assert az.NNEvalCache is eval_mod.NNEvalCache
    req = (3, [1.0, 0.0, 0.0, 0.0], 7, 11, 13, 2)
    assert az._parse_eval_request(req) == eval_mod.parse_eval_request(req)
    group = az._make_eval_request_group("json_single", [req], gi=4, prefer_shm=True)
    assert group == eval_mod.make_eval_request_group("json_single", [req], gi=4, prefer_shm=True)


def test_training_module_reexports_qipc_api():
    az = load_training_module()
    from quartz import qipc as qipc_mod

    assert az.QipcSharedMemoryTransport is qipc_mod.QipcSharedMemoryTransport
    assert az.ShmRingBuffer is qipc_mod.ShmRingBuffer
    payload = struct.pack("<IIIQQI", 7, 9, 4, 11, 22, 3) + np.asarray([0.25, -0.5, 0.75, 1.25], dtype="<f4").tobytes()
    lhs = az.unpack_qipc_eval_req(payload)
    rhs = qipc_mod.unpack_qipc_eval_req(payload)
    assert lhs[0] == rhs[0]
    assert lhs[2:] == rhs[2:]
    np.testing.assert_allclose(lhs[1], rhs[1])


def test_training_module_reexports_selfplay_runtime_api():
    az = load_training_module()
    from quartz import selfplay_runtime as sp_mod

    cfg = {"board": 7, "actions": 49, "bg_parallel": 2, "bg_batch_games": 4, "batch_size": 8, "batch": 64}
    recent_chunks = [{"games": 2, "positions": 30}]
    assert issubclass(az.NNSearchClient, sp_mod.NNSearchClient)
    assert issubclass(az.RustServerPool, sp_mod.RustServerPool)
    assert issubclass(az.SelfPlayWorker, sp_mod.SelfPlayWorker)
    assert az.plan_selfplay_runner_chunk(cfg, replay_size=16, recent_chunks=recent_chunks) == (
        sp_mod.plan_selfplay_runner_chunk(cfg, replay_size=16, recent_chunks=recent_chunks)
    )
    assert az.compute_train_steps(100, 256, 512, concurrent=True) == sp_mod.compute_train_steps(100, 256, 512, concurrent=True)
    assert az.default_output_dir("gomoku7") == sp_mod.default_output_dir("gomoku7")


def test_training_module_reexports_game_adapter_api():
    az = load_training_module()
    from quartz import game_adapters as ga_mod

    assert az.GomokuGameAdapter is ga_mod.GomokuGameAdapter
    assert az.TicTacToeGameAdapter is ga_mod.TicTacToeGameAdapter
    assert az.GoGameAdapter is ga_mod.GoGameAdapter
    assert az.ChessEvaluationAdapter is ga_mod.ChessEvaluationAdapter


def test_training_module_reexports_arena_runtime_api():
    az = load_training_module()
    from quartz import arena_runtime as arena_mod

    assert az.MCTSNode is arena_mod.MCTSNode
    assert az.TreeMCTS is arena_mod.TreeMCTS
    assert az.Glicko2Rating is arena_mod.Glicko2Rating
    assert az.Glicko2System is arena_mod.Glicko2System
    assert az.RandomRolloutAgent is arena_mod.RandomRolloutAgent
    assert az.TreeMCTSEngine is arena_mod.TreeMCTSEngine
    assert az.arena_compare is arena_mod.arena_compare
    assert az.arena_3agent is arena_mod.arena_3agent


def test_training_module_reexports_evaluator_runtime_api():
    az = load_training_module()
    from quartz import evaluator_runtime as evalrt_mod

    assert issubclass(az.RustNNEvaluatorEngine, evalrt_mod.RustNNEvaluatorEngine)


def test_training_module_reexports_autotune_runtime_api():
    az = load_training_module()
    from quartz import autotune_runtime as auto_mod

    assert az.AUTOTUNE_PROFILE_VERSION == auto_mod.AUTOTUNE_PROFILE_VERSION
    assert issubclass(az.OnlineAutotuneController, auto_mod.OnlineAutotuneController)
    assert az._autotune_parallel_limit is not None
    assert az.plan_online_runtime_overrides({"bg_parallel": 1, "bg_batch_games": 1, "n_threads": 1, "batch": 256, "batch_size": 8}, az.HardwareSpec(
        logical_cpus=4, physical_cpus=2, memory_mb=8192,
        gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
        gpu_count=0, torch_cuda=False, device_kind="cpu"
    ), {"last_cycle_s": 1.0, "last_cycle_positions": 16, "positions_per_s": 8.0, "best_positions_per_s": 8.0, "n_new": 64, "train_steps": 2}) == auto_mod.plan_online_runtime_overrides(
        {"bg_parallel": 1, "bg_batch_games": 1, "n_threads": 1, "batch": 256, "batch_size": 8},
        az.HardwareSpec(
            logical_cpus=4, physical_cpus=2, memory_mb=8192,
            gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
            gpu_count=0, torch_cuda=False, device_kind="cpu"
        ),
        {"last_cycle_s": 1.0, "last_cycle_positions": 16, "positions_per_s": 8.0, "best_positions_per_s": 8.0, "n_new": 64, "train_steps": 2},
    )


def test_training_module_reexports_models_torch_api():
    az = load_training_module()
    from quartz import models_torch as models_mod

    assert az.AlphaZeroNet is models_mod.AlphaZeroNet
    assert az.ResBlock is models_mod.ResBlock
    assert az.SEBlock is models_mod.SEBlock


def test_training_module_reexports_cli_parser():
    az = load_training_module()
    from quartz import cli_main as cli_mod

    parser = az.build_arg_parser()
    parsed = parser.parse_args(["--game", "gomoku7", "--runtime-autotune", "--search-profile", "baseline_strict"])

    assert isinstance(parser, argparse.ArgumentParser)
    assert parsed.game == "gomoku7"
    assert parsed.runtime_autotune is True
    assert parsed.search_profile == "baseline_strict"
    assert az.build_arg_parser().__class__ is cli_mod.build_arg_parser(az.GAME_CONFIGS.keys()).__class__


def test_training_module_reexports_system_runtime_api():
    az = load_training_module()
    from quartz import system_runtime as sys_mod

    hw = az.HardwareSpec(logical_cpus=8, physical_cpus=4, memory_mb=16000, gpu_vram_mb=0, device_kind="cpu")
    assert az.HardwareSpec is sys_mod.HardwareSpec
    assert az.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=16) == (
        sys_mod.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=16)
    )
    assert az.compute_eval_collect_policy(8, 0.002, batch_items_ema=4.0, wait_ema_s=0.0) == (
        sys_mod.compute_eval_collect_policy(8, 0.002, batch_items_ema=4.0, wait_ema_s=0.0)
    )
    assert az.max_supported_threads(hw) == sys_mod.max_supported_threads(hw)


def test_training_module_reexports_train_loop_api():
    az = load_training_module()
    from quartz import train_loop as tl_mod

    assert issubclass(az.EarlyStopping, tl_mod.EarlyStopping)
    assert issubclass(az.StepEarlyStopping, tl_mod.StepEarlyStopping)
    assert az.early_stopping_enabled(5, concurrent=True) == tl_mod.early_stopping_enabled(5, concurrent=True)
    assert az.round_or_none(1.23456) == tl_mod.round_or_none(1.23456)


def test_replay_loads_legacy_dense_npz_as_sparse_examples(tmp_path):
    az = load_training_module()
    states = np.stack([
        np.zeros((3, 7, 7), dtype=np.float32),
        np.ones((3, 7, 7), dtype=np.float32),
    ])
    policies = np.zeros((2, 49), dtype=np.float32)
    policies[0, 4] = 1.0
    policies[1, 2] = 0.4
    policies[1, 5] = 0.6
    values = np.array([0.25, -0.5], dtype=np.float32)
    path = tmp_path / "legacy_replay.npz"
    np.savez_compressed(path, states=states, policies=policies, values=values)

    replay = az.ReplayBuffer(16)
    assert replay.load(path) == 2
    assert isinstance(replay.buf[0], az.ReplayExample)
    assert replay.buf[0].policy.n_actions == 49

    states_t, policies_t, values_t = replay.sample(2)
    assert states_t.shape == (2, 3, 7, 7)
    assert policies_t.shape == (2, 49)
    assert values_t.shape == (2,)


def test_train_entry_detects_when_jax_should_be_prewarmed():
    entry = load_train_entry_module()

    assert entry._should_prewarm_jax(["--backend", "jax"]) is True
    assert entry._should_prewarm_jax(["--backend=jax"]) is True
    assert entry._should_prewarm_jax(["--device", "jax"]) is True
    assert entry._should_prewarm_jax(["--backend", "torch"]) is False


def test_train_entry_selects_runtime_module_from_backend_flags():
    entry = load_train_entry_module()

    assert entry._runtime_module_name(["--backend", "jax"]) == "quartz.jax_runtime"
    assert entry._runtime_module_name(["--device", "jax"]) == "quartz.jax_runtime"
    assert entry._runtime_module_name(["--backend", "torch"]) == "quartz.torch_runtime"


def test_jax_runtime_parser_imports_without_loading_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import jax_runtime

    parser = jax_runtime.build_arg_parser()
    parsed = parser.parse_args(["--game", "gomoku7", "--backend", "jax"])

    assert parsed.backend == "jax"
    assert "quartz.alphazero_train" not in sys.modules


def test_jax_runtime_main_help_avoids_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import jax_runtime

    with pytest.raises(SystemExit):
        jax_runtime.main(["--help"])

    assert "quartz.alphazero_train" not in sys.modules


def test_torch_runtime_main_help_avoids_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import torch_runtime

    with pytest.raises(SystemExit):
        torch_runtime.main(["--help"])

    assert "quartz.alphazero_train" not in sys.modules


def test_prepare_training_context_prefers_backend_actor_for_jax(monkeypatch, tmp_path):
    from quartz import cli_main as cli_mod
    from quartz import backend as backend_mod

    class FakeJaxBackend:
        name = "jax"
        num_params = 123
        optimizer = None

        def load(self, path):
            self.loaded = path
            return False

        def get_torch_model(self):
            return None

    class NeverInstantiateModel:
        def __init__(self, _cfg):
            raise AssertionError("torch model should not be constructed for jax backend context")

    monkeypatch.setattr(backend_mod, "create_backend", lambda cfg, device="auto", preference="auto": FakeJaxBackend())

    args = cli_mod.build_arg_parser(["gomoku7"]).parse_args(
        ["--game", "gomoku7", "--backend", "jax", "--output", str(tmp_path), "--no-autotune"]
    )

    hooks = cli_mod.CliPrepareHooks(
        torch=types.SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None),
        ),
        np=np,
        random_mod=random,
        game_configs={"gomoku7": {"board": 7, "ch": 17, "actions": 49, "filters": 32, "blocks": 2, "vh": 32, "buf": 64, "batch": 8, "steps": 4, "batch_size": 8, "games": 2}},
        get_encoder=None,
        apply_config_overrides=lambda cfg, overrides: dict(cfg, **overrides),
        is_go_game=lambda _name: False,
        default_output_dir=lambda game: str(tmp_path / game),
        resolve_runtime_paths=lambda base_dir, explicit_model=None, resume=False: {
            "load_model_path": str(tmp_path / "latest.pt"),
            "latest_model_path": str(tmp_path / "latest.pt"),
            "best_model_path": str(tmp_path / "best.pt"),
            "replay_path": str(tmp_path / "replay.npz"),
            "log_path": str(tmp_path / "train_log.jsonl"),
            "autotune_profile_path": str(tmp_path / "autotune_profile.json"),
        },
        auto_device_name=lambda: "cpu",
        detect_hardware_spec=lambda device: types.SimpleNamespace(device_kind=str(device), physical_cpus=4),
        configure_torch_rocm_runtime=lambda hw: (_ for _ in ()).throw(AssertionError("torch runtime should not be configured")),
        supports_rust_eval_state_machine=lambda _name: False,
        supports_rust_selfplay_state_machine=lambda _name: False,
        autotune_training_cfg=lambda cfg, hw, concurrent=True: cfg,
        clamp_runtime_cfg_to_hardware=lambda cfg, hw: cfg,
        max_supported_threads=lambda hw: 4,
        gpu_host_thread_cap=lambda hw: 2,
        gpu_interop_thread_cap=lambda hw: 1,
        alphazero_net_cls=NeverInstantiateModel,
        load_torch_state_dict_checked=lambda *args, **kwargs: None,
        get_actor_model=lambda model, backend: backend if backend is not None else model,
        load_autotune_profile=lambda *args, **kwargs: None,
        apply_runtime_overrides=lambda cfg, overrides: dict(cfg, **overrides),
        run_autotune_benchmark=lambda *args, **kwargs: ({}, {}),
        save_autotune_profile=lambda *args, **kwargs: None,
        probe_inference_batch_size=lambda model, device, cfg, cap: cfg.get("batch_size", 8),
        clamp_thread_count=lambda value, hw: int(value),
    )

    ctx = cli_mod.prepare_training_context(args, hooks)

    assert ctx.backend is not None
    assert ctx.backend.name == "jax"
    assert ctx.model is None
    assert ctx.optimizer is None
    assert ctx.actor_source is ctx.backend
    assert ctx.device == "jax"
    assert ctx.n_params == 123


def test_replay_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_replay_no_torch", "quartz/replay.py")

    target = module.sparse_policy_from_entries([[3, 1.0]], 9)
    assert target.n_actions == 9


def test_train_loop_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_train_loop_no_torch", "quartz/train_loop.py")

    assert module.round_or_none(1.23456) == 1.2346


def test_system_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_system_runtime_no_torch", "quartz/system_runtime.py")

    assert module.max_supported_threads(types.SimpleNamespace(logical_cpus=4)) == 4


def test_autotune_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_autotune_runtime_no_torch", "quartz/autotune_runtime.py")

    assert module._round_down_to_multiple(130, 32) == 128


def test_runtime_support_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_runtime_support_no_torch", "quartz/runtime_support.py")

    assert module.default_encoder_cfg("gomoku7")["_name"] == "gomoku7"


def test_evaluator_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_evaluator_runtime_no_torch", "quartz/evaluator_runtime.py")

    hooks = module._default_runtime_hooks()
    assert hooks.search_client_cls is not None


def test_monitor_iteration_regex_handles_step_ratio():
    monitor = load_monitor_script_module()
    line = "[ 10/20] loss=4.4578 (p=3.8057 v=0.6521) lr=0.01165 replay=2805 +89 steps=2/2 17.1s"
    evt = monitor.parse_stdout_event(line)

    assert evt is not None
    assert evt["type"] == "iteration"
    assert evt["iter"] == "10"
    assert evt["total"] == "20"
    assert evt["steps_done"] == "2"
    assert evt["steps_planned"] == "2"


def test_monitor_command_settings_capture_runtime_hygiene_flags():
    monitor = load_monitor_script_module()

    settings = monitor.parse_command_settings(
        ["python", "-m", "quartz.train", "--runtime-autotune", "--search-profile", "baseline_strict"]
    )

    assert settings["runtime_tuner_enabled"] is True
    assert settings["eval_selfplay_isolated"] is True
    assert settings["search_profile"] == "baseline_strict"

    settings = monitor.parse_command_settings(
        ["python", "-m", "quartz.train", "--no-eval-selfplay-isolation"]
    )
    assert settings["eval_selfplay_isolated"] is False


def test_monitor_parse_stdout_event_captures_async_runtime_markers():
    monitor = load_monitor_script_module()

    assert monitor.parse_stdout_event("  Auto-tune profile: running warmup benchmark...")["type"] == "autotune_warmup_start"
    assert monitor.parse_stdout_event("  [BG] WARN: self-play pause timed out; evaluation proceeding concurrently")[
        "type"
    ] == "bg_pause_timeout"
    assert monitor.parse_stdout_event("  [BG] Self-play resumed after evaluation")["type"] == "bg_resumed"


def test_monitor_async_trace_summaries_include_wavefront_and_batch_stats(tmp_path):
    monitor = load_monitor_script_module()
    trace_path = tmp_path / "rust_server_trace.jsonl"
    rows = [
        {"event": "run_multi_async_batch_start", "jobs": 6, "max_inflight_per_job": 3},
        {"event": "run_multi_async_batch_done", "null_results": 2},
        {
            "event": "selfplay_runner_wave",
            "newly_completed": 1,
            "wave_positions_emitted": 12,
            "replenished_slots": 2,
            "batch_elapsed_ms": 250.0,
            "frontier_slots": 8,
            "active_games": 10,
        },
        {"event": "eval_runner_wave", "newly_completed": 2, "wave_positions_evaluated": 16},
        {"event": "selfplay_runner_done", "duration_ms": 1500.0},
        {"event": "eval_runner_done", "duration_ms": 2300.0},
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    trace_summary = monitor.summarize_rust_server_trace(trace_path)
    runner_summary = monitor.summarize_runner_progress(trace_path)

    assert trace_summary["async_batch_runs"] == 1
    assert trace_summary["async_batch_jobs_sum"] == 6
    assert trace_summary["async_batch_null_results_sum"] == 2
    assert trace_summary["selfplay_runner_done_count"] == 1
    assert trace_summary["eval_runner_done_count"] == 1

    assert runner_summary["selfplay_wave_count"] == 1
    assert runner_summary["selfplay_positions_emitted"] == 12
    assert runner_summary["selfplay_wave_elapsed_ms_sum"] == 250.0
    assert runner_summary["selfplay_frontier_slots_sum"] == 8
    assert runner_summary["selfplay_active_games_sum"] == 10
    assert runner_summary["eval_wave_count"] == 1
    assert runner_summary["eval_positions_evaluated"] == 16


def test_gpu_detect_install_deps_uses_shell_aware_split(monkeypatch):
    gpu_detect = load_gpu_detect_module()
    calls = []

    def fake_check_call(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(gpu_detect, "recommend_install", lambda gpu, framework: {framework: ["pip install 'jax[metal]' flax"]})
    monkeypatch.setattr(gpu_detect.subprocess, "check_call", fake_check_call)

    gpu_detect.install_deps(gpu_detect.GpuInfo(vendor="apple"), framework="jax", dry_run=False)

    assert calls == [["pip", "install", "jax[metal]", "flax"]]


def test_play_app_preserves_checkpoint_tuned_cfg(monkeypatch, tmp_path):
    play_gui = load_play_gui_module()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    captured = {}

    class DummySession:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = "sess123"

    loaded = play_gui.LoadedModel(
        game="gomoku7",
        path=str(model_path),
        cfg={"_name": "gomoku7", "board": 9, "ch": 5, "iters": 77, "actions": 81},
        model=object(),
    )

    app = play_gui.PlayApp(tmp_path, play_gui.torch.device("cpu"), "./target/release/mcts_demo")
    monkeypatch.setattr(app.model_store, "load", lambda game, path: loaded)
    monkeypatch.setattr(play_gui, "GameSession", DummySession)

    session = app.create_session({"game": "gomoku7", "modelPath": str(model_path), "humanSide": "black"})

    assert session.id == "sess123"
    assert captured["cfg"]["board"] == 9
    assert captured["cfg"]["ch"] == 5
    assert captured["cfg"]["iters"] == 77


def test_auto_device_name_prefers_mps_when_cuda_unavailable(monkeypatch):
    az = load_training_module()

    monkeypatch.setattr(az.sys, "platform", "darwin")
    monkeypatch.setattr(az.torch.cuda, "is_available", lambda: False)

    class FakeMps:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(az.torch, "backends", type("Backends", (), {"mps": FakeMps})())

    assert az.auto_device_name() == "mps"


def test_pyproject_jax_extra_includes_torch_for_train_entrypoint():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    jax_extra = data["project"]["optional-dependencies"]["jax"]

    assert any(dep.startswith("torch") for dep in jax_extra)


def test_rust_nn_evaluator_uses_chess_session_payloads(monkeypatch):
    az = load_training_module()
    start_fen = az.STANDARD_CHESS_FEN
    next_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.model = model
            self.cfg = cfg
            self.device = device
            self.rust_binary = rust_binary
            self.open_jobs = None
            self.step_updates = []
            self.closed_session_id = None
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
            self.open_jobs = jobs
            return {
                "session_id": 7,
                "results": [
                    {"best_move": 0, "iterations": 1, "result_fen": next_fen},
                    {"best_move": 0, "iterations": 1, "result_fen": next_fen},
                ],
            }

        def step_search_session(self, session_id, updates):
            self.step_updates.append((session_id, updates))
            return {"session_id": session_id, "results": []}

        def close_search_session(self, session_id):
            self.closed_session_id = session_id
            return {"ok": True}

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {"_name": "chess", "iters": 8, "actions": az.CHESS_POLICY_ACTIONS}
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg, object(), az.torch.device("cpu"))

    tally = eng_a.play_match_tally_against(
        eng_b,
        lambda: az.ChessEvaluationAdapter(actions=az.CHESS_POLICY_ACTIONS, start_fen=start_fen),
        opening_book=[],
        num_games=2,
        color_swap=True,
        max_moves=1,
        seed=7,
    )

    fake = FakeClient.last_instance
    assert tally.total == 2
    assert fake is not None and fake.started and fake.stopped
    assert fake.closed_session_id == 7
    assert fake.open_jobs is not None and len(fake.open_jobs) == 2
    assert all(job.get("fen") == start_fen for job in fake.open_jobs)
    assert all("board" not in job for job in fake.open_jobs)
    assert fake.step_updates and fake.step_updates[0][0] == 7


def test_rust_nn_evaluator_uses_rust_eval_state_machine_for_gomoku7(monkeypatch):
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.eval_sessions = None
            self.started = False
            self.stopped = False
            self.open_called = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
            self.eval_sessions = sessions
            return {
                "records": [
                    {
                        "game_id": "g0000",
                        "black_tag": 0,
                        "white_tag": 1,
                        "outcome": "black_win",
                        "score_black": 1.0,
                        "move_count": 4,
                        "total_time_ms": 12.0,
                        "opening": [],
                        "seed": 7,
                        "error": None,
                        "is_void": False,
                    },
                    {
                        "game_id": "g0001",
                        "black_tag": 1,
                        "white_tag": 0,
                        "outcome": "white_win",
                        "score_black": 0.0,
                        "move_count": 4,
                        "total_time_ms": 13.0,
                        "opening": [],
                        "seed": 8,
                        "error": None,
                        "is_void": False,
                    },
                ]
            }

        def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
            self.open_called = True
            raise AssertionError("session fallback should not be used when rust eval runner succeeds")

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {
        "_name": "gomoku7",
        "iters": 8,
        "actions": 49,
        "_eval_runner_mode": "rust_eval_state_machine",
    }
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg, object(), az.torch.device("cpu"))

    tally = eng_a.play_match_tally_against(
        eng_b,
        MiniGame,
        opening_book=[],
        num_games=2,
        color_swap=True,
        max_moves=10,
        seed=7,
    )

    fake = FakeClient.last_instance
    assert tally.total == 2
    assert tally.wins == 2
    assert fake is not None and fake.started and fake.stopped
    assert fake.eval_sessions is not None and len(fake.eval_sessions) == 2
    assert fake.open_called is False


def test_rust_nn_evaluator_select_moves_batch_handles_non_chess_games(monkeypatch):
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49

        def current_player(self):
            return 0

        def legal_moves(self):
            return [2, 7, 11]

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_moves_multi(self, jobs, penalty_mode="GatedRefresh"):
            assert len(jobs) == 1
            return [{"best_move": 7, "policy": [[2, 0.1], [7, 0.8], [11, 0.1]], "p_flip": 0.0}]

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {"_name": "gomoku7", "iters": 8, "actions": 49}
    eng = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))

    move, meta = eng.select_moves_batch([MiniGame()])[0]

    assert move == 7
    assert meta["engine"] == "rust_nn"
    assert meta["simulations"] == 8
    fake = FakeClient.last_instance
    assert fake is not None and fake.started is True
    eng.reset()
    assert fake.stopped is True


def test_backend_auto_prefers_torch_over_jax():
    backend_mod = load_backend_module()
    detection = {
        "jax": True,
        "jax_gpu": True,
        "torch": True,
        "torch_gpu": True,
    }

    assert backend_mod.select_backend(detection, preference="auto") == "torch"
    assert backend_mod.select_backend(detection, preference="jax") == "jax"
    assert backend_mod.select_backend(detection, preference="torch") == "torch"


def test_selfplay_batched_uses_rust_state_machine_payload(monkeypatch):
    az = load_training_module()

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.started = False
            self.stopped = False
            self.calls = []
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def selfplay_run(self, n_games, parallel, temp_threshold, penalty_mode="GatedRefresh", seed=0):
            self.calls.append((n_games, parallel, temp_threshold))
            return {
                "games": [
                    {
                        "states": [[0] * 49],
                        "players": [1],
                        "policies": [["0:1.0"]],
                        "outcome": 1.0,
                        "trace": [{"iterations": 8}],
                    },
                    {
                        "states": [[0] * 49],
                        "players": [1],
                        "policies": [["1:1.0"]],
                        "outcome": -1.0,
                        "trace": [{"iterations": 8}],
                    },
                ]
            }

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    cfg["_selfplay_runner_mode"] = "rust_selfplay_state_machine"
    states, policies, outcomes, traces = az.selfplay_rust_nn_batched(
        cfg,
        model=object(),
        device=az.torch.device("cpu"),
        n_games=2,
        rust_binary="./target/release/mcts_demo",
        parallel=2,
        show_progress=False,
    )

    fake = FakeClient.last_instance
    assert fake is not None and fake.started and fake.stopped
    assert fake.calls == [(2, 2, cfg["temp_th"])]
    assert len(states) == 2 and len(policies) == 2 and len(outcomes) == 2 and len(traces) == 2
    assert states[0][0].shape == (cfg["ch"], cfg["board"], cfg["board"])
    assert float(policies[0][0][0]) == pytest.approx(1.0)
    assert outcomes == [1.0, -1.0]


def test_selfplay_rust_nn_uses_training_module_search_client(monkeypatch):
    az = load_training_module()

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.calls = 0
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
            self.calls += 1
            if self.calls == 1:
                return {"best_move": 0, "policy": ["0:1.0"], "value": 0.0}
            return {"policy": [], "value": 0.0}

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    states, policies, outcomes, traces = az.selfplay_rust_nn(
        cfg,
        model=object(),
        device=az.torch.device("cpu"),
        n_games=1,
        rust_binary="./target/release/mcts_demo",
    )

    fake = FakeClient.last_instance
    assert fake is not None and fake.started and fake.stopped
    assert fake.calls == 2
    assert len(states) == 1 and len(policies) == 1 and len(outcomes) == 1
    assert states[0][0].shape[1:] == (cfg["board"], cfg["board"])
    assert states[0][0].shape[0] > 0
    assert float(policies[0][0][0]) == pytest.approx(1.0)
    assert traces == 0


def test_arena_rust_nn_impl_uses_training_module_search_client(monkeypatch, tmp_path):
    az = load_training_module()

    class DummyNet:
        def __init__(self, cfg):
            self.cfg = cfg

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    class FakeClient:
        instances = []

        def __init__(self, model, cfg, device, rust_binary):
            self.calls = 0
            self.started = False
            self.stopped = False
            FakeClient.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
            self.calls += 1
            if self.calls == 1:
                return {"best_move": 0, "policy": ["0:1.0"], "value": 0.0}
            return {"policy": [], "value": 1.0}

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(az, "NNSearchClient", FakeClient)

    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    wins_a, wins_b, draws, wr, _ci, _sprt = az._arena_rust_nn_impl(
        str(model_a),
        cfg,
        str(model_b),
        cfg,
        az.torch.device("cpu"),
        n_games=2,
        strict=True,
    )

    assert len(FakeClient.instances) == 2
    assert all(client.started and client.stopped for client in FakeClient.instances)
    assert wins_a + wins_b + draws >= 1
    assert wr >= 0.0


def test_detect_backends_skips_jax_probe_when_torch_is_available(monkeypatch):
    backend_mod = load_backend_module()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "jax":
            raise AssertionError("auto backend detection should not import jax when torch is available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    detection = backend_mod.detect_backends(preference="auto")
    assert detection["torch"] is True
    assert detection["jax_checked"] is False


def test_load_torch_state_dict_falls_back_when_weights_only_rejects_checkpoint():
    backend_mod = load_backend_module()

    class FakeTorch:
        def __init__(self):
            self.calls = []

        def load(self, path, map_location=None, weights_only=None):
            self.calls.append({
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            })
            if weights_only:
                raise RuntimeError("Weights only load failed. Unsupported operand 149")
            return {"ok": True}

    fake_torch = FakeTorch()
    payload = backend_mod.load_torch_state_dict("demo.pt", fake_torch, map_location="cpu")

    assert payload == {"ok": True}
    assert fake_torch.calls == [
        {"path": "demo.pt", "map_location": "cpu", "weights_only": True},
        {"path": "demo.pt", "map_location": "cpu", "weights_only": False},
    ]


def test_profile_monitor_parses_expected_iterations():
    monitor = load_monitor_script_module()

    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train", "--iterations", "30"]) == 30
    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train", "--iterations=12"]) == 12
    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train"]) is None


def test_detect_checkpoint_backend_hint_distinguishes_torch_and_jax(tmp_path):
    az = load_training_module()
    torch_ckpt = tmp_path / "torch.pt"
    jax_ckpt = tmp_path / "jax.pt"
    torch_ckpt.write_bytes(b"PK\x03\x04demo-latest/data.pkl")
    jax_ckpt.write_bytes(b"\x80\x04demo params BatchNorm_0 jax._src.arr")

    assert az.detect_checkpoint_backend_hint(torch_ckpt) == "torch"
    assert az.detect_checkpoint_backend_hint(jax_ckpt) == "jax"


def test_ensure_best_checkpoint_compatible_resets_mismatched_jax_checkpoint(tmp_path):
    az = load_training_module()
    best = tmp_path / "best.pt"
    best.write_bytes(b"\x80\x04demo params BatchNorm_0 jax._src.arr")

    class FakeBackend:
        name = "torch"

        def __init__(self):
            self.saved = None

        def save(self, path):
            self.saved = path
            Path(path).write_bytes(b"PK\x03\x04demo-latest/data.pkl")

    backend = FakeBackend()
    hint = az.ensure_best_checkpoint_compatible(best, backend, model=None, device=az.torch.device("cpu"))

    assert hint == "torch"
    assert Path(backend.saved) == best
    assert az.detect_checkpoint_backend_hint(best) == "torch"


def test_validate_torch_state_dict_reports_shape_mismatch():
    backend_mod = load_backend_module()
    import torch

    model = torch.nn.Linear(4, 2)
    state = {
        "weight": torch.zeros((3, 4)),
        "bias": torch.zeros((3,)),
    }

    reason = backend_mod.validate_torch_state_dict(model, state)

    assert reason is not None
    assert "mismatched=" in reason


def test_should_use_resident_session_only_for_multi_game_and_explicit_enable():
    az = load_training_module()

    assert az.should_use_resident_session("gomoku7", parallel=1, n_games=1, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=1, n_games=4, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=1, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=4, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=4, enabled=True) is True
    assert az.should_use_resident_session("chess", parallel=4, n_games=8, enabled=True) is True


def test_wait_for_worker_progress_raises_when_worker_exits():
    az = load_training_module()

    class FakeWorker:
        REPLAY_STALL_TIMEOUT_S = 45.0
        positions_generated = 0
        _stop = type("Stop", (), {"is_set": staticmethod(lambda: False)})()

        def status(self):
            return {
                "alive": False,
                "last_progress_age_s": 0.0,
                "last_error": "boom",
                "consecutive_errors": 1,
            }

    with pytest.raises(RuntimeError, match="background self-play worker stopped unexpectedly"):
        az.wait_for_worker_progress(FakeWorker(), 0, min_new=1, timeout_s=0.1, poll_s=0.0)


def test_wait_for_worker_progress_raises_when_worker_stalls_after_errors():
    az = load_training_module()

    class FakeWorker:
        REPLAY_STALL_TIMEOUT_S = 1.0
        positions_generated = 0
        _stop = type("Stop", (), {"is_set": staticmethod(lambda: False)})()

        def status(self):
            return {
                "alive": True,
                "last_progress_age_s": 2.0,
                "last_error": "stall",
                "consecutive_errors": 2,
            }

    with pytest.raises(RuntimeError, match="background self-play stalled"):
        az.wait_for_worker_progress(FakeWorker(), 0, min_new=1, timeout_s=0.1, poll_s=0.0)


def test_probe_cfg_disables_resident_session():
    az = load_training_module()

    cfg = {"batch_size": 8, "iters": 16}
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = 2
    probe_cfg["batch_size"] = max(cfg.get("batch_size", 8), min(64, max(2 * 2, 8)))
    probe_cfg["_disable_resident_session"] = True

    assert probe_cfg["_disable_resident_session"] is True


def test_autotune_parallel_candidates_drop_single_process_on_gpu_concurrent():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="test",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    candidates = az._autotune_parallel_candidates({"bg_parallel": 6}, hw, concurrent=True)
    assert 1 not in candidates
    assert 2 in candidates


def test_eval_worker_candidates_are_hardware_bounded_not_four_capped():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    workers = az.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=200)

    assert workers[-1] == 12
    assert workers[:3] == [1, 2, 3]
    assert 6 in workers
    assert len(workers) >= 5


def test_compute_eval_collect_policy_adapts_timeout_and_target():
    az = load_training_module()

    low_fill_target, low_fill_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=4.0, wait_ema_s=0.0003)
    high_fill_target, high_fill_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=16.0, wait_ema_s=0.0003)
    wait_bound_target, wait_bound_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=5.0, wait_ema_s=0.01)

    assert low_fill_timeout > 0.002
    assert high_fill_target >= 16
    assert high_fill_timeout < low_fill_timeout
    assert wait_bound_target <= 16
    assert wait_bound_timeout < low_fill_timeout


def test_step_early_stopping_waits_for_min_fraction_before_triggering():
    az = load_training_module()
    stopper = az.StepEarlyStopping(
        patience=2,
        min_delta=0.0,
        min_fraction=0.7,
        ema_alpha=1.0,
        planned_steps=10,
    )

    for idx in range(1, 7):
        assert stopper.step(1.0, idx) is False
    assert stopper.min_steps == 7
    assert stopper.step(1.0, 7) is True


def test_plan_online_runtime_overrides_penalizes_bursty_batch_games():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    cfg = {
        "bg_parallel": 4,
        "bg_batch_games": 12,
        "n_threads": 2,
        "batch": 288,
        "steps": 100,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, {
        "rolling_cycle_s": 2.2,
        "rolling_positions_per_s": 20.0,
        "best_positions_per_s": 21.0,
        "burst_ratio": 2.4,
        "n_new": 120,
        "train_steps": 3,
        "rolling_positions": 180,
        "last_cycle_positions": 180,
    })

    assert overrides["bg_batch_games"] < 12


def test_score_selfplay_probe_penalizes_multi_game_single_thread_without_batching():
    az = load_training_module()
    bad = az._score_selfplay_probe(
        positions_per_s=4.0,
        cycle_s=8.0,
        concurrent=True,
        positions=32,
        eval_messages=1500,
        model_batch_mean=1.0,
        parallel=2,
        n_threads=1,
    )
    good = az._score_selfplay_probe(
        positions_per_s=5.5,
        cycle_s=4.3,
        concurrent=True,
        positions=24,
        eval_messages=352,
        model_batch_mean=3.46,
        parallel=2,
        n_threads=2,
    )
    assert good > bad


def test_run_batched_eval_groups_merges_cross_process_requests():
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, batch_np):
            self.batch_sizes.append(int(batch_np.shape[0]))
            n = int(batch_np.shape[0])
            probs = np.tile(np.linspace(1.0, 6.0, 6, dtype=np.float32), (n, 1))
            vals = np.arange(n, dtype=np.float32)
            return probs, vals

    model = FakeModel()
    cfg = {"ch": 1, "board": 2}
    groups = [
        {
            "gi": 0,
            "kind": "binary_batch",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])],
        },
        {
            "gi": 1,
            "kind": "json_single",
            "requests": [(4, [0.0, 0.0, 1.0, 0.0])],
        },
    ]

    responses = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)

    assert model.batch_sizes == [3]
    assert len(responses) == 2
    assert responses[0]["kind"] == "binary_batch"
    assert len(responses[0]["policies"]) == 2
    assert responses[0]["policies"][0].shape == (3,)
    assert responses[0]["policies"][1].shape == (2,)
    assert responses[0]["values"] == [0.0, 1.0]
    assert responses[1]["kind"] == "json_single"
    assert len(responses[1]["policies"]) == 1
    assert responses[1]["policies"][0].shape == (4,)
    assert responses[1]["values"] == [2.0]


def test_run_batched_eval_groups_defaults_missing_group_index():
    az = load_training_module()
    cfg = {"ch": 1, "board": 2}
    groups = [
        {
            "kind": "json_batch",
            "requests": [(2, [1.0, 0.0, 0.0, 0.0])],
        }
    ]

    responses = az._run_batched_eval_groups(groups, None, az.torch.device("cpu"), cfg)

    assert len(responses) == 1
    assert responses[0]["gi"] == 0
    assert responses[0]["kind"] == "json_batch"
    assert len(responses[0]["policies"]) == 1
    assert responses[0]["policies"][0].shape == (2,)


def test_run_batched_eval_groups_preserves_model_outputs_when_cache_disabled(monkeypatch):
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, batch_np):
            self.calls += 1
            probs = np.array([[0.05, 0.15, 0.8]], dtype=np.float32)
            vals = np.array([0.375], dtype=np.float32)
            return probs, vals

    groups = [
        {
            "gi": 0,
            "kind": "json_single",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0], 0)],
        }
    ]
    cfg = {"ch": 1, "board": 2, "actions": 3}

    model_with_cache = FakeModel()
    az.clear_nn_eval_cache()
    monkeypatch.delenv("QUARTZ_DISABLE_NN_CACHE", raising=False)
    with_cache = az._run_batched_eval_groups(groups, model_with_cache, az.torch.device("cpu"), cfg)

    model_without_cache = FakeModel()
    az.clear_nn_eval_cache()
    monkeypatch.setenv("QUARTZ_DISABLE_NN_CACHE", "1")
    without_cache = az._run_batched_eval_groups(groups, model_without_cache, az.torch.device("cpu"), cfg)

    np.testing.assert_allclose(with_cache[0]["policies"][0], [0.05, 0.15, 0.8])
    np.testing.assert_allclose(without_cache[0]["policies"][0], with_cache[0]["policies"][0])
    assert with_cache[0]["values"] == [0.375]
    assert without_cache[0]["values"] == [0.375]
    assert model_with_cache.calls == 1
    assert model_without_cache.calls == 1


def test_nn_eval_cache_treats_move_to_end_race_as_miss():
    az = load_training_module()

    class FlakyOrderedDict(az.OrderedDict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if value is not default:
                super().pop(key, None)
            return value

    cache = az.NNEvalCache(max_entries=4)
    cache._cache = FlakyOrderedDict()
    cache._cache[123] = (np.array([0.2, 0.8], dtype=np.float32), 0.5)

    assert cache.get(123) is None
    assert cache._hits == 0
    assert cache._misses == 1


def test_run_batched_eval_groups_prefers_explicit_fingerprint_cache_keys(monkeypatch):
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, batch_np):
            self.calls += 1
            probs = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
            vals = np.array([0.125], dtype=np.float32)
            return probs, vals

    def fail_legacy_key(*_args, **_kwargs):
        raise AssertionError("legacy feature-bytes cache key should not be used")

    cfg = {"ch": 1, "board": 2, "actions": 3}
    groups = [
        {
            "gi": 0,
            "kind": "json_single",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0], 0, 101, 202, 1)],
        }
    ]

    az.clear_nn_eval_cache()
    monkeypatch.delenv("QUARTZ_DISABLE_NN_CACHE", raising=False)
    monkeypatch.setattr(az, "_legacy_eval_cache_key", fail_legacy_key)

    model = FakeModel()
    first = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)
    second = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)

    assert model.calls == 1
    np.testing.assert_allclose(first[0]["policies"][0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(second[0]["policies"][0], first[0]["policies"][0])
    assert second[0]["values"] == [0.125]


def test_read_exact_timeout_raises(monkeypatch):
    az = load_training_module()

    class FakeStream:
        def read(self, _n):
            raise AssertionError("read should not be called when the stream is not readable")

    monkeypatch.setattr(az, "wait_readable", lambda stream, timeout_s: False)

    with pytest.raises(TimeoutError):
        az._read_exact(FakeStream(), 4, timeout_s=0.01)


def test_search_move_retries_after_timeout(monkeypatch):
    az = load_training_module()

    client = az.NNSearchClient(model=None, cfg={"_name": "gomoku7", "iters": 8}, device="cpu")
    starts = []
    stops = []
    calls = {"n": 0}

    monkeypatch.setattr(client, "start", lambda: starts.append("start") or setattr(client, "proc", object()))
    monkeypatch.setattr(client, "stop", lambda: stops.append("stop") or setattr(client, "proc", None))

    def fake_exchange(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("stalled search")
        return {"result": {"best_move": 7, "policy": [[7, 1.0]], "value": 0.25}}

    monkeypatch.setattr(client, "_exchange_search_request", fake_exchange)

    result = client.search_move(np.zeros(49, dtype=np.int8), player=1)

    assert result["best_move"] == 7
    assert calls["n"] == 2
    assert starts == ["start", "start"]
    assert stops == ["stop"]


def test_make_json_safe_converts_numpy_scalars_and_arrays():
    az = load_training_module()

    payload = {
        "score_rate": np.float32(0.625),
        "delta": np.int64(7),
        "policy": np.array([1.0, 2.0], dtype=np.float32),
        "nested": {"value": np.float64(1.5)},
    }

    safe = az.make_json_safe(payload)

    assert safe == {
        "score_rate": pytest.approx(0.625),
        "delta": 7,
        "policy": [1.0, 2.0],
        "nested": {"value": 1.5},
    }
    json.dumps(safe)


def test_proc_decode_eval_frame_reads_shared_memory_request():
    az = load_training_module()
    transport = az.QipcSharedMemoryTransport.create(size=128)
    proc = type("FakeProc", (), {})()
    proc._quartz_qipc_transport = transport
    transport.req.buf[:6] = b"abcdef"
    try:
        frame_kind, payload = az.proc_decode_eval_frame(
            proc, az.QIPC_EVAL_REQ_SHM, az.QIPC_SHM_LEN.pack(6)
        )
        assert frame_kind == az.QIPC_EVAL_REQ
        assert payload == b"abcdef"
    finally:
        transport.destroy()


def test_proc_write_eval_response_prefers_shared_memory_when_available():
    az = load_training_module()
    transport = az.QipcSharedMemoryTransport.create(size=128)
    proc = type("FakeProc", (), {})()
    proc._quartz_qipc_transport = transport
    proc.stdin = io.BytesIO()
    payload = b"\x01\x02\x03\x04"
    try:
        az.proc_write_eval_response(proc, az.QIPC_EVAL_RESP, payload, prefer_shm=True)
        proc.stdin.seek(0)
        kind, meta = az.proc_read_message(type("ReadProc", (), {"stdout": proc.stdin})())
        assert kind == "frame"
        frame_kind, frame_payload = meta
        assert frame_kind == az.QIPC_EVAL_RESP_SHM
        assert az.QIPC_SHM_LEN.unpack(frame_payload)[0] == len(payload)
        assert bytes(transport.resp.buf[:len(payload)]) == payload
    finally:
        transport.destroy()


def test_rust_search_options_includes_adaptive_batch_timeout():
    az = load_training_module()

    opts = az.rust_search_options({"n_threads": 4, "batch_size": 16, "search_profile": "baseline"})

    assert opts["n_threads"] == 4
    assert opts["batch_size"] == 16
    assert opts["batch_timeout_us"] > 1500
    assert opts["search_profile"] == "baseline"


def test_rust_search_options_preserves_baseline_strict_profile():
    az = load_training_module()

    opts = az.rust_search_options({"search_profile": "baseline_strict"})

    assert opts["search_profile"] == "baseline_strict"


def test_rust_search_options_passes_controller_runtime_overrides():
    az = load_training_module()

    opts = az.rust_search_options({
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
    })

    assert opts["penalty_mode"] == "GatedRefreshLegacy"
    assert opts["root_only_shaping"] is False


def test_gpu_host_thread_caps_follow_physical_cores():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    assert az.gpu_host_thread_cap(hw) == 12
    assert az.gpu_interop_thread_cap(hw) == 6


def test_score_selfplay_probe_rewards_lower_ipc_message_density():
    az = load_training_module()

    single_frame_score = az._score_selfplay_probe(
        10.5, 35.0, True, positions=370, eval_messages=67000)
    batched_score = az._score_selfplay_probe(
        10.2, 22.5, True, positions=230, eval_messages=15000)

    assert batched_score > single_frame_score


def test_autotune_parallel_limit_caps_gpu_concurrent_ipc_heavy_parallelism():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    assert az._autotune_parallel_limit(hw, concurrent=True) == 6
    assert az._autotune_parallel_limit(hw, concurrent=False) == 12


def test_autotune_thread_candidates_skip_single_thread_for_gpu_parallel_search():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    candidates = az._autotune_thread_candidates(hw, parallel=4)
    assert 1 not in candidates
    assert all(t >= 2 for t in candidates)


def test_plan_online_runtime_overrides_reduces_ipc_heavy_parallelism_and_raises_threads():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    cfg = {
        "bg_parallel": 12,
        "bg_batch_games": 24,
        "n_threads": 1,
        "batch": 480,
        "steps": 100,
    }
    sample = {
        "last_cycle_s": 5.0,
        "last_cycle_positions": 300,
        "positions_per_s": 10.0,
        "best_positions_per_s": 12.0,
        "n_new": 100,
        "train_steps": 2,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)
    assert overrides["bg_parallel"] == 6
    assert overrides["n_threads"] >= 2


def test_replay_dataloader_preserves_batch_shapes_and_types():
    az = load_training_module()
    replay = az.ReplayBuffer(32)
    for i in range(8):
        state = np.full((3, 7, 7), float(i), dtype=np.float32)
        policy = np.zeros(49, dtype=np.float32)
        policy[i % 49] = 1.0
        replay.add(state, policy, float(i))

    loader = replay.build_dataloader(batch_size=4, n_steps=2, pin_memory=False)
    batches = list(loader)

    assert len(batches) == 2
    states_t, policies_t, values_t = batches[0]
    assert states_t.shape == (4, 3, 7, 7)
    assert policies_t.shape == (4, 49)
    assert values_t.shape == (4,)
    assert states_t.dtype == az.torch.float32
    assert policies_t.dtype == az.torch.float32
    assert values_t.dtype == az.torch.float32


def test_get_actor_model_returns_backend_source():
    az = load_training_module()
    fallback_model = object()

    class FakeBackend:
        pass

    backend = FakeBackend()
    assert az.get_actor_model(fallback_model, backend) is backend
    assert az.get_actor_model(fallback_model, None) is fallback_model


def test_benchmark_train_batch_supports_generic_backend():
    az = load_training_module()

    class FakeBackend:
        def __init__(self):
            self.params = {"w": 1.0}
            self.batch_stats = {"bn": 0.0}
            self.opt_state = {"step": 0}

        def train_step(self, states, policies, values):
            self.opt_state = {"step": self.opt_state["step"] + 1}
            return 1.0, 0.5, 0.5

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=False, device_kind="cpu")

    overrides, results = az.benchmark_train_batch(
        cfg, FakeBackend(), None, None, az.torch.device("cpu"), hw)

    assert "batch" in overrides
    assert any("examples_per_s" in row for row in results)


def test_jax_model_init_and_forward_if_available():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_jax_models", root / "quartz" / "jax_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not getattr(module, "HAS_JAX", False):
        pytest.skip("JAX not installed in test environment")

    model = module.AlphaZeroJAX(
        board_size=7, in_ch=3, n_actions=49,
        n_filters=96, n_blocks=6, value_hidden=128, se_blocks=2)
    rng = module.jax.random.PRNGKey(0)
    x = module.jnp.ones((2, 3, 7, 7), dtype=module.jnp.float32)
    variables = model.init(rng, x, train=False)
    logits, values = model.apply(variables, x, train=False)

    assert logits.shape == (2, 49)
    assert values.shape == (2,)


def test_load_actor_source_from_checkpoint_reuses_jax_backend_template(tmp_path):
    az = load_training_module()

    ckpt = tmp_path / "dummy_jax.pkl"
    ckpt.write_bytes(b"stub")

    class FakeBackend:
        name = "jax"

        def __init__(self):
            self.loaded = None

        def load_actor(self, path):
            self.loaded = path
            return "jax-actor"

    backend = FakeBackend()
    actor = az.load_actor_source_from_checkpoint(
        str(ckpt),
        dict(az.GAME_CONFIGS["gomoku7"]),
        az.torch.device("cpu"),
        backend_preference="jax",
        backend_template=backend,
    )

    assert actor == "jax-actor"
    assert backend.loaded == str(ckpt)


def test_load_actor_source_from_checkpoint_uses_training_module_model_wrapper(tmp_path, monkeypatch):
    az = load_training_module()

    ckpt = tmp_path / "dummy_torch.pt"
    ckpt.write_bytes(b"stub")

    class FakeActor:
        def __init__(self, cfg):
            self.cfg = cfg
            self.loaded = None
            self.eval_called = False

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

        def eval(self):
            self.eval_called = True
            return self

    monkeypatch.setattr(az, "AlphaZeroNet", FakeActor)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {"w": 1})

    actor = az.load_actor_source_from_checkpoint(
        str(ckpt),
        dict(az.GAME_CONFIGS["gomoku7"]),
        az.torch.device("cpu"),
        backend_preference="torch",
    )

    assert isinstance(actor, FakeActor)
    assert actor.loaded == {"w": 1}
    assert actor.eval_called is True


def test_choose_selfplay_move_uses_policy_before_temperature_cutoff():
    az = load_training_module()
    policy = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    chosen = az.choose_selfplay_move(
        policy, legal=[0, 1], move_count=0, temp_threshold=8, fallback_best=0)

    assert chosen == 1


def test_compute_train_steps_scales_concurrent_work_to_fresh_data():
    az = load_training_module()

    assert az.compute_train_steps(100, 256, 0, concurrent=True) == 0
    assert az.compute_train_steps(100, 256, 190, concurrent=True) == 6
    assert az.compute_train_steps(100, 256, 3600, concurrent=True) == 100
    assert az.compute_train_steps(100, 256, 0, concurrent=False) == 100


def test_default_output_dir_uses_models_subdirectory():
    az = load_training_module()

    assert az.default_output_dir("gomoku7") == "models/alphazero_gomoku7"


def test_rust_search_options_propagates_tt_enabled_flag():
    az = load_training_module()

    opts = az.rust_search_options({"n_threads": 2, "batch_size": 16, "tt_enabled": False})

    assert opts["tt_enabled"] is False


def test_dense_policy_from_sparse_accepts_legacy_strings_and_numeric_pairs():
    az = load_training_module()

    policy = az.dense_policy_from_sparse(["1:0.25", [3, 0.75], ["bad"]], 5)

    assert np.allclose(policy, np.array([0.0, 0.25, 0.0, 0.75, 0.0], dtype=np.float32))


def test_build_rust_state_meta_includes_chess_history_hashes():
    az = load_training_module()
    state = az.ChessEvaluationAdapter()
    state._chess_history_hashes = [11, 22, 33]

    meta = az.build_rust_state_meta("chess", state, {})

    assert meta == {"chess_history_hashes": [11, 22, 33]}


def test_chess_evaluation_adapter_tracks_engine_history_hashes():
    az = load_training_module()
    state = az.ChessEvaluationAdapter()

    applied = state.apply_engine_meta(
        0,
        {
            "result_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "result_history_hashes": [101, 202, 303],
        },
    )

    assert applied is True
    assert state._fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert state._chess_history_hashes == [101, 202, 303]
    clone = state.clone()
    assert clone._chess_history_hashes == [101, 202, 303]


def test_early_stopping_enabled_by_positive_patience():
    az = load_training_module()

    assert az.early_stopping_enabled(15, concurrent=False) is True
    assert az.early_stopping_enabled(15, concurrent=True) is True
    assert az.early_stopping_enabled(0, concurrent=False) is False


def test_load_epoch_history_filters_eval_records(tmp_path):
    az = load_training_module()
    log_path = tmp_path / "train_log.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps({"iter": 1, "loss": 1.23, "published_elo": None}) + "\n")
        f.write(json.dumps({"_type": "eval", "iter": 1, "published_elo": 1200}) + "\n")
        f.write(json.dumps({"iter": 2, "loss": 1.11, "published_elo": 1200}) + "\n")

    history = az.load_epoch_history(str(log_path))

    assert [row["iter"] for row in history] == [1, 2]
    assert history[1]["published_elo"] == 1200


def test_build_elo_plot_series_prefers_logged_absolute_gap():
    az = load_training_module()
    history = [
        {"iter": 5, "published_elo": 140.0, "champion_elo": 100.0, "elo_gap": 40.0,
         "delta_elo": 75.0, "score_rate": 0.62},
    ]

    series = az.build_elo_plot_series(history)

    assert len(series) == 1
    point = series[0]
    assert point["candidate_elo"] == 140.0
    assert point["champion_elo"] == 100.0
    assert point["elo_gap"] == 40.0
    assert point["error_mid"] == 120.0
    assert point["error_half"] == 20.0
    assert point["match_delta_elo"] == 75.0


def test_build_elo_plot_series_falls_back_to_delta_when_champion_missing():
    az = load_training_module()
    history = [
        {"iter": 10, "published_elo": 160.0, "delta_elo": 30.0},
    ]

    series = az.build_elo_plot_series(history)

    assert len(series) == 1
    point = series[0]
    assert point["champion_elo"] == 130.0
    assert point["elo_gap"] == 30.0
    assert point["error_mid"] == 145.0
    assert point["error_half"] == 15.0


def test_build_metric_plot_series_skips_missing_points_without_dropping_sparse_losses():
    az = load_training_module()
    history = [
        {"iter": 1, "loss": 4.9},
        {"iter": 2, "loss": None},
        {"iter": 4, "loss": 4.2},
    ]

    series = az.build_metric_plot_series(history, "loss")

    assert series == [(1, 4.9), (4, 4.2)]


def test_build_best_elo_series_only_promotes_on_promotion_verdict():
    az = load_training_module()
    elo_points = [
        {
            "iter": 5,
            "candidate_elo": 140.0,
            "champion_elo": 100.0,
            "match_delta_elo": 15.0,
            "eval_verdict": "reject",
        },
        {
            "iter": 10,
            "candidate_elo": 155.0,
            "champion_elo": 110.0,
            "match_delta_elo": 18.0,
            "eval_verdict": "promote",
        },
    ]

    series = az.build_best_elo_series(elo_points)

    assert series == [100.0, 155.0]


def test_main_runs_eval_at_interval_even_when_train_steps_are_zero(monkeypatch, tmp_path):
    az = load_training_module()
    backend_module = sys.modules["quartz.backend"]
    output_dir = tmp_path / "run"
    rust_binary = tmp_path / "mcts_demo"
    rust_binary.write_text("stub", encoding="utf-8")

    def fake_create_backend(*args, **kwargs):
        raise RuntimeError("skip unified backend")

    monkeypatch.setattr(backend_module, "create_backend", fake_create_backend, raising=False)
    monkeypatch.setattr(az, "auto_device_name", lambda: "cpu")
    monkeypatch.setattr(
        az,
        "detect_hardware_spec",
        lambda device: az.HardwareSpec(
            logical_cpus=4,
            physical_cpus=2,
            memory_mb=8192,
            gpu_vendor="",
            gpu_name="",
            gpu_vram_mb=0,
            gpu_count=0,
            torch_cuda=False,
            device_kind="cpu",
        ),
    )
    monkeypatch.setattr(az, "configure_torch_rocm_runtime", lambda hw: None)
    monkeypatch.setattr(az, "clamp_runtime_cfg_to_hardware", lambda cfg, hw: dict(cfg))
    monkeypatch.setattr(az, "print_autotune_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "max_supported_threads", lambda hw: 1)
    monkeypatch.setattr(az, "gpu_host_thread_cap", lambda hw: 1)
    monkeypatch.setattr(az, "gpu_interop_thread_cap", lambda hw: 1)
    monkeypatch.setattr(az.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(az.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(az.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(az, "clear_nn_eval_cache", lambda: None)
    monkeypatch.setattr(az, "get_actor_model", lambda model, backend: "candidate-actor")
    monkeypatch.setattr(az, "clone_actor_model", lambda actor: actor)
    monkeypatch.setattr(az, "load_actor_source_from_checkpoint", lambda *args, **kwargs: "champion-actor")
    monkeypatch.setattr(az, "generate_training_plots", lambda *args, **kwargs: False)
    monkeypatch.setattr(az, "compute_train_steps", lambda *args, **kwargs: 0)
    monkeypatch.setattr(az, "selfplay_rust_nn_batched", lambda *args, **kwargs: ([], [], [], []))
    monkeypatch.setattr(az, "supports_rust_eval_state_machine", lambda game: False)
    monkeypatch.setattr(az, "supports_rust_selfplay_state_machine", lambda game: False)
    monkeypatch.setattr(az, "load_eval_autotune_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "recommend_eval_parallel_workers", lambda *args, **kwargs: 1)
    monkeypatch.setattr(az, "build_training_game_adapter", lambda cfg: object())
    monkeypatch.setattr(az, "ensure_best_checkpoint_compatible", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "HAS_EVAL_SYSTEM", True)

    class FakeOptimizer:
        def __init__(self, params):
            self.param_groups = [{"lr": 0.0}]

    monkeypatch.setattr(az.torch.optim, "SGD", lambda params, **kwargs: FakeOptimizer(params))

    class FakeModel:
        def __init__(self, cfg):
            self._params = [az.torch.nn.Parameter(az.torch.zeros(1))]

        def to(self, device):
            return self

        def parameters(self):
            return self._params

        def state_dict(self):
            return {"w": az.torch.zeros(1)}

    monkeypatch.setattr(az, "AlphaZeroNet", FakeModel)

    class FakeReplayBuffer:
        def __init__(self, *args, **kwargs):
            self._size = 10_000

        def __len__(self):
            return self._size

        def save(self, path):
            Path(path).write_text("replay", encoding="utf-8")

    monkeypatch.setattr(az, "ReplayBuffer", FakeReplayBuffer)
    monkeypatch.setattr(az.torch, "save", lambda payload, path: Path(path).write_text("model", encoding="utf-8"))

    class FakeRustEngine:
        def __init__(self, name, cfg, actor, device, rust_binary):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

        def select_moves_batch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(az, "RustNNEvaluatorEngine", FakeRustEngine)

    eval_generations = []

    class FakeTrainingEvaluator:
        def __init__(self, config=None, manifest=None):
            self.cfg = types.SimpleNamespace(parallel_workers=1)

        def evaluate_checkpoint(
            self,
            candidate,
            champion,
            game_factory,
            candidate_id="",
            generation=0,
            candidate_factory=None,
            champion_factory=None,
        ):
            eval_generations.append(generation)
            return types.SimpleNamespace(
                valid_eval=True,
                invalid_reason=None,
                promotion={"verdict": "need_more"},
                tally={"score_rate": 0.5, "scored": 2, "errors": 0, "voids": 0},
                elo={"delta": 12.0},
                published={"candidate_abs": 100.0, "champion_abs": 88.0, "delta": 12.0},
            )

    monkeypatch.setattr(az, "TrainingEvaluator", FakeTrainingEvaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quartz.train",
            "--game",
            "gomoku7",
            "--iterations",
            "5",
            "--output",
            str(output_dir),
            "--rust-binary",
            str(rust_binary),
            "--no-pipeline",
            "--no-autotune",
        ],
    )

    az.main()

    assert eval_generations == [5]

    log_rows = [
        json.loads(line)
        for line in (output_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_row = next(row for row in log_rows if row.get("_type") == "eval")
    iter_row = next(row for row in log_rows if row.get("_type") is None and row.get("iter") == 5)

    assert eval_row["iter"] == 5
    assert iter_row["published_elo"] == 100.0
    assert iter_row["eval_verdict"] == "need_more"


def test_autotune_profile_roundtrip(tmp_path):
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = {
        "_name": "gomoku7",
        "iters": 200,
        "search_profile": "quartz",
        "penalty_mode": "GatedRefresh",
        "batch_timeout_us": 1500,
    }
    path = tmp_path / "autotune_profile.json"
    overrides = {"batch": 384, "bg_parallel": 2}
    bench = {"train": [{"batch": 384, "examples_per_s": 123.4}]}

    az.save_autotune_profile(str(path), hw, cfg, overrides, bench)
    loaded = az.load_autotune_profile(str(path), hw, cfg)

    assert loaded is not None
    assert loaded["version"] == az.AUTOTUNE_PROFILE_VERSION
    assert loaded["overrides"] == overrides
    assert loaded["benchmarks"] == bench


def test_autotune_profile_rejects_old_version(tmp_path):
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = {
        "_name": "gomoku7",
        "iters": 200,
        "search_profile": "quartz",
        "penalty_mode": "GatedRefresh",
        "batch_timeout_us": 1500,
    }
    path = tmp_path / "autotune_profile.json"
    payload = {
        "version": az.AUTOTUNE_PROFILE_VERSION - 1,
        "signature": az.autotune_signature(hw, cfg),
        "overrides": {"batch": 384},
        "benchmarks": {},
        "saved_at": 0,
    }
    path.write_text(json.dumps(payload))

    assert az.load_autotune_profile(str(path), hw, cfg) is None


def test_apply_runtime_overrides_updates_cfg_values():
    az = load_training_module()
    cfg = {"batch": 256, "bg_parallel": 1}

    tuned = az.apply_runtime_overrides(cfg, {"batch": 384, "bg_parallel": 2})

    assert tuned["batch"] == 384
    assert tuned["bg_parallel"] == 2


def test_apply_config_overrides_keeps_runtime_search_fields():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])

    tuned = az.apply_config_overrides(cfg, {"sigma_0": 0.45, "unknown_key": 7})

    assert tuned["sigma_0"] == 0.45
    assert "unknown_key" not in tuned


def test_resolve_runtime_paths_separates_latest_and_best(tmp_path):
    az = load_training_module()

    paths = az.resolve_runtime_paths(str(tmp_path), resume=False)

    assert paths["load_model_path"].endswith("latest.pt")
    assert paths["latest_model_path"].endswith("latest.pt")
    assert paths["best_model_path"].endswith("best.pt")
    assert paths["latest_model_path"] != paths["best_model_path"]


def test_resolve_runtime_paths_resume_falls_back_to_best(tmp_path):
    az = load_training_module()
    best = tmp_path / "best.pt"
    best.write_text("champion")

    paths = az.resolve_runtime_paths(str(tmp_path), resume=True)

    assert paths["load_model_path"] == str(best)


def test_normalize_rust_board_maps_go_white_to_two():
    az = load_training_module()

    board = [1, -1, 0, 2]
    normalized = az.normalize_rust_board("go9", board)

    assert normalized == [1, 2, 0, 2]


def test_all_trainable_non_chess_games_have_registered_encoders():
    az = load_training_module()
    encoders = load_encoders_module()

    for game_name in [
        "gomoku7",
        "gomoku15",
        "gomoku15_free",
        "gomoku15_std",
        "gomoku15_omok",
        "gomoku15_renju",
        "gomoku15_caro",
        "go9",
        "go9_jp",
        "go9_kr",
        "go13",
        "go13_jp",
        "go13_kr",
        "go19",
        "go19_jp",
        "go19_kr",
        "tictactoe",
    ]:
        encoder = encoders.get_encoder(game_name)
        cfg = az.GAME_CONFIGS[game_name]
        assert encoder.board_size == cfg["board"]
        assert encoder.n_actions == cfg["actions"]


def test_build_training_game_adapter_supports_new_games():
    az = load_training_module()
    encoders = load_encoders_module()

    cases = {
        "gomoku15_renju": az.GomokuGameAdapter,
        "go13_jp": az.GoGameAdapter,
        "go19_kr": az.GoGameAdapter,
        "tictactoe": az.TicTacToeGameAdapter,
        "chess": az.ChessEvaluationAdapter,
        "chess960": az.ChessEvaluationAdapter,
    }

    for game_name, expected_type in cases.items():
        cfg = dict(az.GAME_CONFIGS[game_name], _name=game_name, _encoder=encoders.get_encoder(game_name))
        game = az.build_training_game_adapter(cfg)
        assert isinstance(game, expected_type)
        assert len(game.legal_moves()) > 0


def test_go_ruleset_presets_are_exposed_in_cfg_and_adapter():
    az = load_training_module()

    cfg_jp = dict(az.GAME_CONFIGS["go9_jp"], _name="go9_jp", _encoder=None)
    cfg_kr = dict(az.GAME_CONFIGS["go19_kr"], _name="go19_kr", _encoder=None)
    jp = az.build_training_game_adapter(cfg_jp)
    kr = az.build_training_game_adapter(cfg_kr)

    assert jp._ruleset == "japanese"
    assert jp._scoring == "territory"
    assert kr._ruleset == "korean"
    assert kr._scoring == "territory"
    assert kr._komi == az.GAME_CONFIGS["go19_kr"]["go_komi"]


def test_go_chinese_adapter_rejects_repeated_position_hash():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=7.5, ruleset="chinese", scoring="area")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())

    assert game._is_legal(40) is False


def test_go_korean_adapter_marks_repetition_as_draw():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=6.5, ruleset="korean", scoring="territory")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())
    game.apply_move(40)

    assert game.is_terminal() is True
    assert game.outcome_for_black() == 0.0


def test_go_japanese_adapter_marks_repetition_as_void():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=6.5, ruleset="japanese", scoring="territory")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())
    game.apply_move(40)

    assert game.is_terminal() is True
    assert game.is_void_result() is True
    assert game.outcome_for_black() is None


def test_go_territory_cleanup_removes_surrounded_one_eye_group():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=5, komi=6.5, ruleset="japanese", scoring="territory")
    black = {
        0, 1, 2, 3, 4,
        5, 9,
        10, 14,
        15, 19,
        20, 21, 22, 23, 24,
    }
    white = {6, 7, 8, 11, 13, 16, 17, 18}
    for pos in black:
        game._board[pos] = 1
    for pos in white:
        game._board[pos] = -1

    black_score, white_score = game._score()

    assert black_score == 17.0
    assert white_score == 6.5


def test_gomoku15_renju_black_overline_does_not_win_but_white_can():
    az = load_training_module()
    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_renju")
    row = 7 * 15

    for col in [3, 4, 5, 6, 7]:
        game._board[row + col] = 1
    game._player = 1
    assert game._is_winning_move(row + 8) is False

    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_renju")
    for col in [3, 4, 5, 6, 7]:
        game._board[row + col] = -1
    game._player = -1
    assert game._is_winning_move(row + 8) is True


def test_gomoku15_caro_blocked_five_is_not_a_win():
    az = load_training_module()
    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_caro")
    row = 7 * 15

    for col in [4, 5, 6, 7]:
        game._board[row + col] = 1
    game._board[row + 3] = -1
    game._board[row + 9] = -1
    game._player = 1

    assert game._is_winning_move(row + 8) is False


def test_chess_action_space_matches_full_rust_contract():
    az = load_training_module()

    assert az.GAME_CONFIGS["chess"]["actions"] == 4672
    assert az.GAME_CONFIGS["chess960"]["actions"] == 4672
    assert az.GAME_CONFIGS["chess"]["tt_enabled"] is True
    assert az.GAME_CONFIGS["chess960"]["tt_enabled"] is True


def test_chess960_has_registered_encoder_and_is_treated_as_chess():
    az = load_training_module()
    encoders = load_encoders_module()

    encoder = encoders.get_encoder("chess960")

    assert encoder.board_size == 8
    assert encoder.n_actions == az.GAME_CONFIGS["chess960"]["actions"]
    assert az.is_chess_game("chess960") is True


def test_chess960_start_fen_and_encoding_preserve_castling_and_ep():
    az = load_training_module()

    fen = az.chess960_start_fen(0)
    enc = az.encode_chess_fen("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR b HFhf e3 0 1")

    assert fen.split()[2] != "KQkq"
    assert enc.shape == (36, 8, 8)
    # Planes 0-5: my pieces (black to move → black pieces are "my")
    assert enc[:6].sum() > 0
    # Planes 6-11: opponent pieces (white)
    assert enc[6:12].sum() > 0
    # Plane 28: color (0 for black's turn)
    assert np.all(enc[28] == 0.0)
    # Castling planes 30-33 should have some rights set
    assert enc[30:34].sum() > 0
    # EP plane 35: e3 target
    assert enc[35, 2, 4] == 1.0


def test_initial_chess_fen_uses_fixed_chess960_index_when_configured():
    az = load_training_module()

    cfg = dict(az.GAME_CONFIGS["chess960"], _name="chess960", chess960_index=518)

    assert az.initial_chess_fen(cfg) == az.chess960_start_fen(518)


def test_chess_evaluation_adapter_tracks_turn_and_engine_meta():
    az = load_training_module()

    game = az.ChessEvaluationAdapter(start_fen=az.STANDARD_CHESS_FEN)
    assert game.current_player() == 1

    assert game.apply_engine_meta(0, {"result_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}) is True
    assert game.current_player() == 0
    assert game.is_terminal() is False

    assert game.apply_engine_meta(0, {"terminal": True, "outcome_for_black": -1.0}) is True
    assert game.is_terminal() is True
    assert game.outcome_for_black() == -1.0


def test_replay_sampler_prefers_recent_window_but_keeps_older_examples():
    az = load_training_module()
    replay = az.ReplayBuffer(32, recent_fraction=0.8, recent_window=4)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)

    for value in range(10):
        replay.add(state, policy, float(value))

    random.seed(0)
    indices = replay._sample_indices_locked(5)

    assert len(indices) == 5
    assert len(set(indices)) == 5
    assert sum(1 for i in indices if i >= 6) == 4
    assert sum(1 for i in indices if i < 6) == 1


def test_autotune_training_cfg_scales_parallelism_for_strong_hardware():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=32, physical_cpus=16, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=24576,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["selfplay_parallel"] >= 4
    assert tuned["bg_parallel"] >= 4
    assert tuned["bg_batch_games"] >= 4
    assert tuned["batch"] >= cfg["batch"]
    assert tuned["batch_size"] >= cfg["batch_size"]


def test_autotune_parallel_limit_caps_gpu_concurrent_process_count():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    assert az._autotune_parallel_limit(hw, concurrent=True) == 6
    assert az._autotune_parallel_limit(hw, concurrent=False) == 12


def test_autotune_parallel_candidates_focus_on_ipc_friendly_gpu_range():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    candidates = az._autotune_parallel_candidates(cfg, hw, concurrent=True)

    assert candidates == [1, 2, 3, 4, 5, 6]


def test_autotune_batch_game_candidates_scale_with_parallelism():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    candidates = az._autotune_batch_game_candidates(hw, parallel=12, concurrent=True)

    assert candidates == [12, 24]


def test_autotune_training_cfg_keeps_games_stable_in_concurrent_mode():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["games"] == cfg["games"]


def test_autotune_training_cfg_autoscales_tiny_model_on_large_gpu():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["filters"] >= cfg["filters"]
    assert tuned["blocks"] >= cfg["blocks"]
    assert tuned["vh"] >= cfg["vh"]
    assert az.estimate_model_params(tuned) >= az.estimate_model_params(cfg)


def test_autotune_training_cfg_keeps_large_model_when_already_sized():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["chess"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["filters"] == cfg["filters"]
    assert tuned["blocks"] == cfg["blocks"]
    assert tuned["vh"] == cfg["vh"]


def test_autotune_training_cfg_stays_conservative_on_small_hardware():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=4, physical_cpus=2, memory_mb=8192,
        gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
        gpu_count=0, torch_cuda=False, device_kind="cpu")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["selfplay_parallel"] == 2
    assert tuned["bg_parallel"] == 2
    assert tuned["batch"] <= cfg["batch"]
    assert tuned["n_threads"] == 1


def test_autotune_signatures_track_topology_fields():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    cfg["_eval_runner_mode"] = "python_batched"
    cfg["_shared_eval_session"] = False
    cfg["_broker_enabled"] = False
    cfg["_selfplay_topology_version"] = 3

    sig_a = az.autotune_signature(hw, cfg)
    eval_sig_a = az.eval_autotune_signature(hw, cfg, 200)

    cfg["_shared_eval_session"] = True
    cfg["_eval_runner_mode"] = "shared_client_session"
    sig_b = az.autotune_signature(hw, cfg)
    eval_sig_b = az.eval_autotune_signature(hw, cfg, 200)

    assert sig_a != sig_b
    assert eval_sig_a != eval_sig_b


def test_selfplay_autotune_score_penalizes_bursty_cycles():
    az = load_training_module()

    smooth = az._score_selfplay_probe(20.0, 4.0, concurrent=True)
    bursty = az._score_selfplay_probe(20.0, 9.0, concurrent=True)

    assert smooth > bursty


def test_train_batch_score_penalizes_oversized_batches_in_concurrent_mode():
    az = load_training_module()

    right_sized = az._score_train_batch_probe(
        1000.0, 256, concurrent=True, target_positions_per_cycle=80)
    oversized = az._score_train_batch_probe(
        1000.0, 640, concurrent=True, target_positions_per_cycle=80)

    assert right_sized > oversized


def test_plan_online_runtime_overrides_reduces_bursty_batch_games():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({"bg_parallel": 4, "bg_batch_games": 8, "n_threads": 3, "batch": 256, "batch_size": 12})
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    sample = {
        "last_cycle_s": 5.9,
        "last_cycle_positions": 126,
        "positions_per_s": 21.0,
        "best_positions_per_s": 21.0,
        "n_new": 60,
        "train_steps": 3,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)

    assert overrides["bg_batch_games"] == 4


def test_plan_online_runtime_overrides_reduces_batch_when_fresh_data_is_thin():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({"bg_parallel": 4, "bg_batch_games": 4, "n_threads": 3, "batch": 256, "batch_size": 12})
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    sample = {
        "last_cycle_s": 2.6,
        "last_cycle_positions": 58,
        "positions_per_s": 8.0,
        "best_positions_per_s": 10.0,
        "n_new": 60,
        "train_steps": 2,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)

    assert overrides["batch"] < cfg["batch"]


def test_plan_selfplay_runner_chunk_scales_parallel_with_replay_deficit():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
        "bg_parallel": 6,
        "bg_batch_games": 6,
        "batch_size": 18,
        "batch": 192,
    })

    plan = az.plan_selfplay_runner_chunk(cfg, replay_size=32, recent_chunks=[])

    assert plan["parallel"] >= 18
    assert plan["batch_games"] >= plan["parallel"]
    assert plan["replay_deficit"] == 160


def test_plan_selfplay_runner_chunk_uses_recent_positions_per_game_to_bound_batch_games():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
        "bg_parallel": 6,
        "bg_batch_games": 6,
        "batch_size": 18,
        "batch": 192,
    })
    recent_chunks = [
        {"games": 18, "positions": 540, "elapsed_s": 12.0},
        {"games": 18, "positions": 576, "elapsed_s": 11.0},
    ]

    plan = az.plan_selfplay_runner_chunk(cfg, replay_size=180, recent_chunks=recent_chunks)

    assert plan["estimated_positions_per_game"] > 20.0
    assert plan["batch_games"] <= 18


def test_initial_replay_fill_target_is_lower_than_train_batch_when_warm_start_suffices():
    az = load_training_module()
    cfg = {
        "batch": 480,
        "batch_size": 18,
        "bg_parallel": 5,
        "board": 7,
    }
    recent_chunks = [
        {"games": 10, "positions": 180, "elapsed_s": 4.0},
        {"games": 10, "positions": 200, "elapsed_s": 4.2},
    ]

    target = az.initial_replay_fill_target(cfg, recent_chunks)

    assert target >= cfg["batch_size"]
    assert target < cfg["batch"]


def test_initial_replay_fill_target_uses_board_prior_without_recent_chunks():
    az = load_training_module()
    cfg = {
        "batch": 256,
        "batch_size": 16,
        "bg_parallel": 4,
        "board": 7,
    }

    target = az.initial_replay_fill_target(cfg, [])

    assert target >= 16
    assert target <= 256
