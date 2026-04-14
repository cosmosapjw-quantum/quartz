#!/usr/bin/env python3
"""CLI parser helpers for QUARTZ training entrypoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def build_arg_parser(game_names):
    parser = argparse.ArgumentParser(description="QUARTZ AlphaZero Training")
    parser.add_argument("--game", choices=list(game_names), default="gomoku15")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument(
        "--arena",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Compare two models head-to-head (e.g. --arena best_a.pt best_b.pt)",
    )
    parser.add_argument("--arena-games", type=int, default=50)
    parser.add_argument(
        "--arena-3agent",
        nargs=2,
        metavar=("CURRENT", "BEST"),
        help="3-agent arena: current model, best model, + random anchor",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        default=True,
        help="Run self-play in background thread while training (default: on)",
    )
    parser.add_argument(
        "--no-pipeline",
        dest="concurrent",
        action="store_false",
        help="Disable pipelined self-play (sequential mode for debugging)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "torch"],
        default="auto",
        help="ML backend: auto (prefer torch), jax (explicit opt-in), torch",
    )
    parser.add_argument(
        "--no-autotune",
        dest="autotune",
        action="store_false",
        help="Disable hardware-based runtime autotuning",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Ignore saved autotune profile and rerun warmup benchmark",
    )
    parser.add_argument("--rust-nn", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--replay-buffer", type=int, default=None, help="Replay buffer capacity override")
    parser.add_argument(
        "--replay-recent-frac",
        type=float,
        default=None,
        help="Fraction of each training batch drawn from recent replay",
    )
    parser.add_argument("--replay-window", type=int, default=None, help="Recent replay sampling window size")
    parser.add_argument(
        "--go-ruleset",
        choices=["chinese", "japanese", "korean"],
        default=None,
        help="Override Go ruleset preset",
    )
    parser.add_argument(
        "--go-scoring",
        choices=["area", "territory"],
        default=None,
        help="Override Go scoring mode",
    )
    parser.add_argument("--go-komi", type=float, default=None, help="Override Go komi")
    parser.add_argument("--go-allow-suicide", action="store_true", help="Allow suicide moves in Go")
    parser.add_argument(
        "--chess960-index",
        type=int,
        default=None,
        help="Use a fixed Chess960 Scharnagl index (0-959)",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument(
        "--inner-patience",
        type=int,
        default=8,
        help="Loose within-iteration early stopping patience (0 disables)",
    )
    parser.add_argument(
        "--inner-min-fraction",
        type=float,
        default=0.7,
        help="Minimum fraction of planned train steps to run before inner stopping can trigger",
    )
    parser.add_argument(
        "--inner-min-delta",
        type=float,
        default=5e-4,
        help="Minimum inner-step loss improvement to reset plateau tracking",
    )
    parser.add_argument(
        "--inner-ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing for inner-step plateau tracking",
    )
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval-interval", type=int, default=5, help="Run Glicko-2 evaluation every N iterations")
    parser.add_argument("--eval-games", type=int, default=200, help="Number of games per evaluation")
    parser.add_argument("--selfplay-parallel", type=int, default=None, help="Override foreground self-play process parallelism")
    parser.add_argument("--bg-parallel", type=int, default=None, help="Override background self-play process parallelism")
    parser.add_argument("--bg-batch-games", type=int, default=None, help="Override background self-play games per refill cycle")
    parser.add_argument("--mcts-threads", type=int, default=None, help="Override Rust MCTS threads per self-play search")
    parser.add_argument("--nn-batch-size", type=int, default=None, help="Override Rust->Python NN batch size")
    parser.add_argument(
        "--search-profile",
        choices=["quartz", "baseline", "baseline_strict"],
        default=None,
        help="Rust MCTS profile: quartz, baseline shared substrate, or baseline_strict",
    )
    parser.add_argument(
        "--vl-mode",
        choices=["disabled", "fixed", "adaptive", "vvisit_only", "vvalue_only"],
        default=None,
        help="Virtual loss mode override for ablation study",
    )
    parser.add_argument("--games", type=int, default=None, help="Self-play games per iteration override")
    parser.add_argument(
        "--resident-session",
        action="store_true",
        help="Experimental: enable Rust resident search sessions for self-play",
    )
    parser.add_argument(
        "--runtime-autotune",
        action="store_true",
        help="Enable experimental online runtime retuning during training",
    )
    parser.add_argument(
        "--no-eval-selfplay-isolation",
        dest="eval_selfplay_isolation",
        action="store_false",
        help="Allow background self-play to continue during evaluation",
    )
    parser.set_defaults(eval_selfplay_isolation=True)
    return parser


@dataclass(frozen=True)
class CliPrepareHooks:
    torch: object
    np: object
    random_mod: object
    game_configs: dict
    get_encoder: object
    apply_config_overrides: object
    is_go_game: object
    default_output_dir: object
    resolve_runtime_paths: object
    auto_device_name: object
    detect_hardware_spec: object
    configure_torch_rocm_runtime: object
    supports_rust_eval_state_machine: object
    supports_rust_selfplay_state_machine: object
    autotune_training_cfg: object
    clamp_runtime_cfg_to_hardware: object
    max_supported_threads: object
    gpu_host_thread_cap: object
    gpu_interop_thread_cap: object
    alphazero_net_cls: type
    load_torch_state_dict_checked: object
    get_actor_model: object
    load_autotune_profile: object
    apply_runtime_overrides: object
    run_autotune_benchmark: object
    save_autotune_profile: object
    probe_inference_batch_size: object
    clamp_thread_count: object


@dataclass(frozen=True)
class PreparedTrainingContext:
    cfg: dict
    base_cfg: dict
    base_dir: str
    device: object
    hw: object
    model: object
    backend: object
    optimizer: object
    actor_source: object
    benchmark_info: object
    model_path: str
    latest_model_path: str
    best_model_path: str
    replay_path: str
    log_path: str
    autotune_profile_path: str
    n_params: int


@dataclass(frozen=True)
class MainRuntimeHooks:
    torch: object
    np: object
    game_configs: dict
    serve: object
    arena_3agent: object
    arena_rust_nn: object
    arena_compare: object
    print_autotune_summary: object
    is_go_game: object
    replay_buffer_cls: type
    early_stopping_cls: type
    early_stopping_enabled: object
    load_eval_autotune_profile: object
    has_eval_system: bool
    recommend_eval_parallel_workers: object
    max_supported_threads: object
    eval_config_cls: type
    training_evaluator_cls: type
    build_training_game_adapter: object
    ensure_best_checkpoint_compatible: object
    selfplay_worker_cls: type
    initial_replay_fill_target: object
    online_autotune_controller_cls: type
    clear_nn_eval_cache: object
    round_or_none: object
    wait_for_worker_progress: object
    selfplay_rust_nn_batched: object
    compute_train_steps: object
    train_epoch: object
    replay_metrics: object
    rust_nn_evaluator_engine_cls: type
    clone_actor_model: object
    load_actor_source_from_checkpoint: object
    tree_mcts_engine_cls: type
    benchmark_eval_parallel_workers: object
    make_json_safe: object
    generate_training_plots: object


def prepare_training_context(args, runtime_hooks: CliPrepareHooks):
    torch = runtime_hooks.torch
    np = runtime_hooks.np
    random_mod = runtime_hooks.random_mod
    requested_backend = str(args.backend or "auto").lower()
    jax_requested = requested_backend == "jax" or str(args.device or "").lower() == "jax"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random_mod.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = dict(runtime_hooks.game_configs[args.game])
    cfg["_name"] = args.game
    cfg["_resident_session"] = bool(args.resident_session)

    if runtime_hooks.get_encoder is not None:
        try:
            cfg["_encoder"] = runtime_hooks.get_encoder(args.game)
            print(
                f"  Encoder: {type(cfg['_encoder']).__name__} "
                f"({cfg['_encoder'].n_channels}ch, {cfg['_encoder'].n_actions} actions)"
            )
        except KeyError:
            cfg["_encoder"] = None
    else:
        cfg["_encoder"] = None

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            overrides = json.load(f)
            cfg = runtime_hooks.apply_config_overrides(cfg, overrides)

    if args.replay_buffer is not None:
        cfg["buf"] = max(1, int(args.replay_buffer))
    if args.replay_recent_frac is not None:
        cfg["recent_frac"] = float(max(0.0, min(1.0, args.replay_recent_frac)))
    if args.replay_window is not None:
        cfg["recent_window"] = max(0, int(args.replay_window))
    if runtime_hooks.is_go_game(args.game):
        if args.go_ruleset is not None:
            cfg["go_ruleset"] = args.go_ruleset
        if args.go_scoring is not None:
            cfg["go_scoring"] = args.go_scoring
        if args.go_komi is not None:
            cfg["go_komi"] = float(args.go_komi)
        if args.go_allow_suicide:
            cfg["go_allow_suicide"] = True
    if cfg.get("chess960", False) and args.chess960_index is not None:
        cfg["chess960_index"] = max(0, min(959, int(args.chess960_index)))
    if args.search_profile is not None:
        cfg["search_profile"] = args.search_profile
    if args.vl_mode is not None:
        cfg["vl_mode"] = args.vl_mode
    if args.games is not None:
        cfg["games"] = max(1, int(args.games))

    base_dir = args.output or runtime_hooks.default_output_dir(args.game)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    paths = runtime_hooks.resolve_runtime_paths(base_dir, explicit_model=args.model, resume=args.resume)
    model_path = paths["load_model_path"]
    latest_model_path = paths["latest_model_path"]
    best_model_path = paths["best_model_path"]
    replay_path = paths["replay_path"]
    log_path = paths["log_path"]
    autotune_profile_path = paths["autotune_profile_path"]

    device = "jax" if jax_requested else (
        torch.device(runtime_hooks.auto_device_name()) if args.device == "auto" else torch.device(args.device)
    )
    hw = runtime_hooks.detect_hardware_spec(device)
    if not jax_requested:
        runtime_hooks.configure_torch_rocm_runtime(hw)
    base_cfg = dict(cfg)
    eval_runner_mode = "python_batched"
    if os.path.exists(args.rust_binary):
        eval_runner_mode = (
            "rust_eval_state_machine"
            if runtime_hooks.supports_rust_eval_state_machine(cfg.get("_name"))
            else "shared_client_session"
        )
    selfplay_runner_mode = (
        "rust_selfplay_state_machine"
        if os.path.exists(args.rust_binary) and runtime_hooks.supports_rust_selfplay_state_machine(cfg.get("_name"))
        else "python_batched"
    )
    cfg["_selfplay_topology_version"] = 6
    cfg["_shared_eval_session"] = True
    cfg["_broker_enabled"] = False
    cfg["_eval_runner_mode"] = eval_runner_mode
    cfg["_selfplay_runner_mode"] = selfplay_runner_mode
    cfg["_runtime_tuner_enabled"] = bool(args.runtime_autotune)
    cfg["_eval_selfplay_isolated"] = bool(args.eval_selfplay_isolation)
    if args.autotune:
        cfg = runtime_hooks.autotune_training_cfg(cfg, hw, concurrent=args.concurrent)
        cfg["_resident_session"] = bool(args.resident_session)
    cfg["_selfplay_topology_version"] = 6
    cfg["_shared_eval_session"] = True
    cfg["_broker_enabled"] = False
    cfg["_eval_runner_mode"] = eval_runner_mode
    cfg["_selfplay_runner_mode"] = selfplay_runner_mode
    cfg = runtime_hooks.clamp_runtime_cfg_to_hardware(cfg, hw)
    if not jax_requested:
        try:
            if getattr(device, "type", str(device)) == "cpu":
                torch.set_num_threads(runtime_hooks.max_supported_threads(hw))
            else:
                torch.set_num_threads(runtime_hooks.gpu_host_thread_cap(hw))
        except Exception:
            pass
        try:
            if getattr(device, "type", str(device)) == "cpu":
                torch.set_num_interop_threads(
                    max(1, min(runtime_hooks.max_supported_threads(hw), getattr(hw, "physical_cpus", 1) or 1))
                )
            else:
                torch.set_num_interop_threads(runtime_hooks.gpu_interop_thread_cap(hw))
        except Exception:
            pass

    backend = None
    model = None
    n_params = 0
    optimizer = None

    if not args.arena:
        try:
            from quartz.backend import create_backend

            backend_device = "jax" if jax_requested else args.device
            backend = create_backend(cfg, device=backend_device, preference=args.backend)
            if os.path.exists(model_path):
                backend.load(model_path)
            model = backend.get_torch_model()
            optimizer = getattr(backend, "optimizer", None)
            n_params = int(getattr(backend, "num_params", 0) or 0)
            print(f"  Using {backend.name.upper()} backend ({n_params:,} params)")
        except Exception as e:
            if jax_requested:
                print(
                    "  JAX backend requested explicitly, but initialization failed.\n"
                    "  On AMD Radeon Linux, current official JAX support is inference-only,\n"
                    "  so QUARTZ training with --backend jax is experimental.\n"
                    "  Use --backend torch for reliable local training."
                )
                raise
            print(f"  Backend init failed ({e}), using direct PyTorch")
            backend = None

    if backend is None:
        model = runtime_hooks.alphazero_net_cls(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if os.path.exists(model_path):
            try:
                runtime_hooks.load_torch_state_dict_checked(model, model_path, torch, map_location=device)
                print(f"Loaded model: {model_path}")
            except Exception as exc:
                explicit_model = bool(args.model)
                if explicit_model:
                    raise
                print(f"  [WARN] Ignoring incompatible checkpoint {model_path} ({exc})")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
    actor_source = runtime_hooks.get_actor_model(model, backend)
    cfg["_backend_name"] = backend.name if backend is not None else "torch"

    benchmark_info = None
    if args.autotune and not args.serve and not args.arena and not args.arena_3agent:
        profile = None if args.retune else runtime_hooks.load_autotune_profile(autotune_profile_path, hw, cfg)
        if profile is not None:
            cfg = runtime_hooks.apply_runtime_overrides(cfg, profile.get("overrides", {}))
            cfg = runtime_hooks.clamp_runtime_cfg_to_hardware(cfg, hw)
            benchmark_info = profile.get("benchmarks", {})
            print("  Auto-tune profile: loaded cached benchmark")
        else:
            print("  Auto-tune profile: running warmup benchmark...")
            overrides, benchmark_info = runtime_hooks.run_autotune_benchmark(
                cfg, backend, actor_source, optimizer, device, hw, args.rust_binary, concurrent=args.concurrent
            )
            if overrides:
                cfg = runtime_hooks.apply_runtime_overrides(cfg, overrides)
                cfg = runtime_hooks.clamp_runtime_cfg_to_hardware(cfg, hw)
                runtime_hooks.save_autotune_profile(autotune_profile_path, hw, cfg, overrides, benchmark_info)
                print(f"  Auto-tune profile: saved to {autotune_profile_path}")
            else:
                print("  Auto-tune profile: benchmark produced no overrides")

    if args.autotune and not args.serve and not args.arena:
        eval_batch_cap = cfg.get("_eval_batch_cap", 192)
        if eval_batch_cap < 32:
            eval_batch_cap = 32
        probe_model = actor_source if not isinstance(actor_source, dict) else None
        if probe_model is not None and not os.environ.get("QUARTZ_DISABLE_BATCH_PROBE"):
            try:
                probed_bs = runtime_hooks.probe_inference_batch_size(probe_model, device, cfg, eval_batch_cap)
                if probed_bs > cfg.get("batch_size", 8):
                    cfg["batch_size"] = probed_bs
                    print(f"  Batch size probe: optimal BS={probed_bs} (cap={eval_batch_cap})")
            except Exception as e:
                print(f"  Batch size probe: skipped ({e})")

    manual_runtime_overrides = {}
    if args.selfplay_parallel is not None:
        manual_runtime_overrides["selfplay_parallel"] = max(1, int(args.selfplay_parallel))
    if args.bg_parallel is not None:
        manual_runtime_overrides["bg_parallel"] = max(1, int(args.bg_parallel))
    if args.bg_batch_games is not None:
        manual_runtime_overrides["bg_batch_games"] = max(1, int(args.bg_batch_games))
    if args.mcts_threads is not None:
        manual_runtime_overrides["n_threads"] = runtime_hooks.clamp_thread_count(args.mcts_threads, hw)
    if args.nn_batch_size is not None:
        manual_runtime_overrides["batch_size"] = max(1, int(args.nn_batch_size))
    if manual_runtime_overrides:
        cfg = runtime_hooks.apply_runtime_overrides(cfg, manual_runtime_overrides)
        cfg = runtime_hooks.clamp_runtime_cfg_to_hardware(cfg, hw)

    return PreparedTrainingContext(
        cfg=cfg,
        base_cfg=base_cfg,
        base_dir=base_dir,
        device=device,
        hw=hw,
        model=model,
        backend=backend,
        optimizer=optimizer,
        actor_source=actor_source,
        benchmark_info=benchmark_info,
        model_path=model_path,
        latest_model_path=latest_model_path,
        best_model_path=best_model_path,
        replay_path=replay_path,
        log_path=log_path,
        autotune_profile_path=autotune_profile_path,
        n_params=n_params,
    )


def run_training_main(args, ctx: PreparedTrainingContext, runtime_hooks: MainRuntimeHooks):
    cfg = ctx.cfg
    base_cfg = ctx.base_cfg
    base_dir = ctx.base_dir
    device = ctx.device
    hw = ctx.hw
    model = ctx.model
    backend = ctx.backend
    optimizer = ctx.optimizer
    actor_source = ctx.actor_source
    latest_model_path = ctx.latest_model_path
    best_model_path = ctx.best_model_path
    replay_path = ctx.replay_path
    log_path = ctx.log_path
    n_params = ctx.n_params
    torch = runtime_hooks.torch

    if args.serve:
        serve_model = model if model is not None else actor_source
        runtime_hooks.serve(serve_model, cfg, device)
        return

    if args.arena_3agent:
        rating_path = os.path.join(base_dir, "glicko2_ratings.json")
        runtime_hooks.arena_3agent(
            args.arena_3agent[0],
            args.arena_3agent[1],
            cfg,
            device,
            games_per_pair=args.arena_games // 3,
            rust_binary=args.rust_binary,
            use_rust_nn=args.rust_nn,
            rating_path=rating_path,
        )
        return

    if args.arena:
        if args.rust_nn:
            print("  Arena mode: Rust MCTS + NN (full search stack)")
            wa, wb, d, wr, ci, sprt = runtime_hooks.arena_rust_nn(
                args.arena[0], args.arena[1], cfg, device, args.arena_games, args.rust_binary
            )
        else:
            print("  Arena mode: Python TreeMCTS")
            wa, wb, d, wr, ci, sprt = runtime_hooks.arena_compare(
                args.arena[0], args.arena[1], cfg, device, args.arena_games
            )
        print(f"Arena: A={wa} B={wb} D={d} | WinRate_A={wr:.3f} 95%CI=[{ci[0]:.3f},{ci[1]:.3f}] SPRT={sprt}")
        if sprt == "H1_accept":
            print("  → SPRT: Model A is significantly stronger (p<0.05)")
        elif sprt == "H0_accept":
            print("  → SPRT: No significant difference (p<0.05)")
        else:
            print(f"  → SPRT: Inconclusive after {args.arena_games} games")
        return

    print(f"Game: {args.game} ({cfg['board']}×{cfg['board']})")
    print(f"Model: {n_params:,} params, {cfg['filters']}f×{cfg['blocks']}b")
    print(f"Backend: {backend.name if backend else 'PyTorch (direct)'}")
    print(f"Device: {device}")
    print(f"Output: {base_dir}")
    if cfg.get("search_profile", "quartz") != "quartz":
        print(f"Search profile: {cfg.get('search_profile')}")
    if cfg.get("_resident_session", False):
        print("  Self-play transport: experimental resident Rust session ENABLED")
    runtime_hooks.print_autotune_summary(base_cfg, cfg, hw)
    print(
        f"Replay sampling: {int(cfg.get('recent_frac', 0.0) * 100)}% recent,"
        f" window={cfg.get('recent_window', 0):,}"
    )
    print(f"Replay buffer: capacity={cfg['buf']:,}, batch={cfg['batch']}")
    if runtime_hooks.is_go_game(args.game):
        print(
            "Go rules: "
            f"ruleset={cfg.get('go_ruleset', 'chinese')} "
            f"scoring={cfg.get('go_scoring', 'area')} "
            f"komi={cfg.get('go_komi', 7.5):.1f} "
            f"allow_suicide={bool(cfg.get('go_allow_suicide', False))}"
        )
    if cfg.get("chess960", False):
        start_desc = cfg.get("chess960_index")
        print(f"Chess960 start: {'randomized' if start_desc is None else f'index={start_desc}'}")

    replay = runtime_hooks.replay_buffer_cls(
        cfg["buf"],
        recent_fraction=cfg.get("recent_frac", 0.0),
        recent_window=cfg.get("recent_window", 0),
    )
    if args.resume:
        n = replay.load(replay_path)
        if n:
            print(f"Loaded {n} positions from replay")
    outer_stopper = (
        runtime_hooks.early_stopping_cls(
            patience=args.patience,
            warmup=20 if args.concurrent else 10,
            ema_alpha=0.25 if args.concurrent else 0.30,
        )
        if runtime_hooks.early_stopping_enabled(args.patience, concurrent=args.concurrent)
        else None
    )
    inner_stop_cfg = {
        "patience": max(0, int(args.inner_patience)),
        "min_fraction": float(max(0.0, min(1.0, args.inner_min_fraction))),
        "min_delta": float(max(0.0, args.inner_min_delta)),
        "ema_alpha": float(max(0.01, min(1.0, args.inner_ema_alpha))),
    }
    log_f = open(log_path, "a")

    training_evaluator = None
    eval_autotune_profile_path = os.path.join(base_dir, "eval_autotune.json")
    cached_eval_parallel_workers = runtime_hooks.load_eval_autotune_profile(
        eval_autotune_profile_path, hw, cfg, args.eval_games
    )
    eval_workers_autotuned = cached_eval_parallel_workers is not None
    rust_ok = os.path.exists(args.rust_binary)
    if runtime_hooks.has_eval_system:
        eval_parallel_workers = cached_eval_parallel_workers or runtime_hooks.recommend_eval_parallel_workers(
            hw, cfg, args.eval_games, rust_ok=rust_ok
        )
        eval_parallel_workers = max(
            1, min(int(eval_parallel_workers), runtime_hooks.max_supported_threads(hw))
        )
        eval_cfg = runtime_hooks.eval_config_cls(
            num_games=args.eval_games,
            promotion_threshold=0.55,
            confidence=0.95,
            sanity_check_interval=5,
            ladder_path=os.path.join(base_dir, "glicko2_ladder.json"),
            log_path=os.path.join(base_dir, "eval_matches.jsonl"),
            champion_path=os.path.join(base_dir, "champion.json"),
            seed=args.seed,
            parallel_workers=eval_parallel_workers,
        )
        game_factories = {
            game_name: (
                lambda game_name=game_name: runtime_hooks.build_training_game_adapter(dict(cfg, _name=game_name))
            )
            for game_name in runtime_hooks.game_configs
        }
        game_factory = game_factories.get(args.game)
        if game_factory:
            training_evaluator = runtime_hooks.training_evaluator_cls(config=eval_cfg)
            if not os.path.exists(best_model_path):
                if backend:
                    backend.save(best_model_path)
                elif model:
                    torch.save(model.state_dict(), best_model_path)
            else:
                runtime_hooks.ensure_best_checkpoint_compatible(best_model_path, backend, model, device)
            eval_worker_msg = str(eval_parallel_workers)
            if not eval_workers_autotuned:
                eval_worker_msg += " (first eval will benchmark)"
            print(
                f"  Eval system: Glicko-2 + PromotionGate "
                f"(every {args.eval_interval} iters, {args.eval_games} games, workers={eval_worker_msg})"
            )
        else:
            print(f"  Eval system: not available for {args.game}")

    if not rust_ok:
        print(f"WARNING: Rust binary not found at {args.rust_binary}")
        print("  Training requires Rust. Run: cargo build --release")
    cfg["_resident_session"] = bool(cfg.get("_resident_session", False))

    print(f"\n{'='*60}")
    print(f"  Training: {args.iterations} iterations, {cfg['games']} games/iter")
    if args.concurrent:
        print("  Mode: CONCURRENT (background self-play)")
    print(f"  Runtime tuner: {'enabled' if args.runtime_autotune else 'disabled'}")
    if args.concurrent:
        print(f"  Eval/self-play isolation: {'enabled' if args.eval_selfplay_isolation else 'disabled'}")
    if outer_stopper is not None:
        print(f"  Outer early stopping: enabled (patience={args.patience}, warmup={outer_stopper.warmup})")
    if inner_stop_cfg["patience"] > 0:
        print(
            "  Inner early stopping: "
            f"enabled (patience={inner_stop_cfg['patience']}, min_fraction={inner_stop_cfg['min_fraction']:.2f})"
        )
    print(f"{'='*60}\n")

    bg_worker = None
    if args.concurrent:
        if not rust_ok:
            print("ERROR: --concurrent requires Rust binary. Run: cargo build --release")
            sys.exit(1)
        bg_worker = runtime_hooks.selfplay_worker_cls(cfg, actor_source, device, replay, args.rust_binary)
        bg_worker.start()
        print("  [BG] Self-play worker started (Rust+NN)")
        while len(replay) < runtime_hooks.initial_replay_fill_target(cfg, bg_worker._recent_chunks) and not (
            outer_stopper and outer_stopper.should_stop
        ):
            time.sleep(0.5)
            bg_status = bg_worker.status()
            if not bg_status.get("alive", True):
                raise RuntimeError(
                    "background self-play worker exited during replay fill: "
                    f"{bg_status.get('last_error') or 'thread exited'}"
                )
            if (
                bg_status.get("last_progress_age_s", 0.0) > bg_worker.REPLAY_STALL_TIMEOUT_S
                and bg_status.get("consecutive_errors", 0) > 0
            ):
                raise RuntimeError(
                    "background self-play worker stalled during replay fill: "
                    f"{bg_status.get('last_error') or 'no progress'}"
                )
            status_suffix = ""
            if bg_status.get("consecutive_errors", 0) > 0:
                status_suffix = f" err={bg_status['consecutive_errors']}"
            fill_target = runtime_hooks.initial_replay_fill_target(cfg, bg_worker._recent_chunks)
            print(
                f"\r  [BG] Filling replay: {len(replay)}/{fill_target}...{status_suffix}",
                end="",
                flush=True,
            )
        print()

    latest_eval = {
        "published_elo": None,
        "champion_elo": None,
        "elo_gap": None,
        "delta_elo": None,
        "score_rate": None,
        "eval_verdict": None,
    }
    online_tuner = (
        runtime_hooks.online_autotune_controller_cls(cfg, hw, enabled_iters=min(10, args.iterations), interval=2)
        if args.concurrent and args.runtime_autotune
        else None
    )

    for iteration in range(args.iterations):
        runtime_hooks.clear_nn_eval_cache()
        t0 = time.time()
        progress = iteration / max(args.iterations, 1)
        lr = 0.0002 + 0.5 * (0.02 - 0.0002) * (1 + math.cos(math.pi * progress))
        if optimizer:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        if backend:
            backend.set_lr(lr)
        avg_pflip = None
        should_early_stop = False
        entry = {
            "iter": iteration + 1,
            "loss": None,
            "p_loss": None,
            "v_loss": None,
            "loss_ema": runtime_hooks.round_or_none(outer_stopper.loss_ema if outer_stopper else None),
            "lr": round(lr, 6),
            "replay": len(replay),
            "new_pos": 0,
            "train_steps": 0,
            "planned_train_steps": 0,
            "time_s": None,
            "pos_per_s": None,
            "games_done": cfg["games"],
            "avg_pflip": None,
            "replay_freshness": None,
            "policy_entropy": None,
            "value_std": None,
            "runtime_tune": None,
        }

        n_new = 0
        if args.concurrent:
            prev_bg = getattr(bg_worker, "_prev_count", 0)
            n_new, bg_now = runtime_hooks.wait_for_worker_progress(
                bg_worker, prev_bg, min_new=1, timeout_s=30.0
            )
            bg_worker._prev_count = bg_now
        elif rust_ok:
            states, policies, outcomes, traces = runtime_hooks.selfplay_rust_nn_batched(
                cfg, actor_source, device, cfg["games"], args.rust_binary, parallel=cfg.get("selfplay_parallel", 4)
            )
            for gs, gp, out in zip(states, policies, outcomes):
                replay.add_game(gs, gp, out)
                n_new += len(gs)
            all_pflips = [t.get("p_flip", 0) for tr in traces for t in tr if t]
            avg_pflip = sum(all_pflips) / max(len(all_pflips), 1) if all_pflips else 0
        else:
            print("ERROR: Rust binary required for training. Run: cargo build --release")
            print(f"  Expected: {args.rust_binary}")
            sys.exit(1)

        if len(replay) >= cfg["batch"]:
            train_steps = runtime_hooks.compute_train_steps(
                cfg["steps"], cfg["batch"], n_new, concurrent=args.concurrent
            )
            if train_steps <= 0:
                elapsed = time.time() - t0
                entry.update(
                    {
                        "replay": len(replay),
                        "new_pos": n_new,
                        "time_s": round(elapsed, 1),
                        "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
                    }
                )
                print(
                    f"[{iteration+1:>3}/{args.iterations}] waiting for self-play: "
                    f"replay={len(replay)} +0 {elapsed:.1f}s"
                )
            else:
                avg_loss, avg_pl, avg_vl, executed_steps, inner_stop = runtime_hooks.train_epoch(
                    model, optimizer, replay, cfg, device, train_steps, backend=backend, inner_stop_cfg=inner_stop_cfg
                )
                elapsed = time.time() - t0
                if outer_stopper:
                    should_early_stop = outer_stopper.step(avg_loss)
                entry.update(
                    {
                        "loss": round(avg_loss, 4),
                        "p_loss": round(avg_pl, 4),
                        "v_loss": round(avg_vl, 4),
                        "loss_ema": runtime_hooks.round_or_none(outer_stopper.loss_ema if outer_stopper else None),
                        "replay": len(replay),
                        "new_pos": n_new,
                        "train_steps": executed_steps,
                        "planned_train_steps": train_steps,
                        "time_s": round(elapsed, 1),
                        "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
                        "avg_pflip": round(avg_pflip, 4) if avg_pflip is not None else None,
                        "replay_freshness": round(runtime_hooks.replay_metrics.freshness(n_new, len(replay)), 4),
                        "policy_entropy": round(runtime_hooks.replay_metrics.policy_entropy(replay), 3),
                        "value_std": round(runtime_hooks.replay_metrics.value_std(replay), 4),
                    }
                )
                if inner_stop is not None:
                    entry["inner_stop"] = inner_stop
                print(
                    f"[{iteration+1:>3}/{args.iterations}] loss={avg_loss:.4f} "
                    f"(p={avg_pl:.4f} v={avg_vl:.4f}) lr={lr:.5f} replay={len(replay)} "
                    f"+{n_new} steps={executed_steps}/{train_steps} {elapsed:.1f}s"
                )
        else:
            elapsed = time.time() - t0
            entry.update(
                {
                    "replay": len(replay),
                    "new_pos": n_new,
                    "time_s": round(elapsed, 1),
                    "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
                }
            )
            print(f"[{iteration+1:>3}/{args.iterations}] filling replay: {len(replay)}/{cfg['batch']} +{n_new} {elapsed:.1f}s")

        if online_tuner is not None:
            runtime_overrides = online_tuner.observe(
                iteration, n_new, time.time() - t0, entry.get("train_steps") or 0, len(replay), bg_worker
            )
            if runtime_overrides:
                entry["runtime_tune"] = dict(runtime_overrides)
                changes = ", ".join(f"{k}={v}" for k, v in runtime_overrides.items())
                print(f"  [AutoTune] iter {iteration+1}: adjusted {changes}")

        if (iteration + 1) % 5 == 0:
            if backend:
                backend.save(latest_model_path)
            else:
                torch.save(model.state_dict(), latest_model_path)
            replay.save(replay_path)
            print(f"  Checkpoint: {latest_model_path} (replay={len(replay)})")
            if bg_worker:
                bg_worker.update_model(actor_source)

        if training_evaluator and (iteration + 1) % args.eval_interval == 0 and game_factory:
            print(f"  Evaluating gen_{iteration+1} vs champion...")
            bg_pause_requested = False
            cand_eng = None
            champ_eng = None
            if bg_worker and args.eval_selfplay_isolation:
                bg_pause_requested = True
                bg_worker.pause(wait=True)
            try:
                candidate_name = f"gen_{iteration+1}"
                candidate_actor_template = runtime_hooks.clone_actor_model(actor_source)
                if rust_ok:
                    cand_factory = lambda: runtime_hooks.rust_nn_evaluator_engine_cls(
                        candidate_name,
                        cfg,
                        runtime_hooks.clone_actor_model(candidate_actor_template),
                        device,
                        args.rust_binary,
                    )
                else:
                    print("  [WARN] Rust binary not found, using TreeMCTS for evaluation (NOT benchmark-grade)")
                    cand_factory = lambda: runtime_hooks.tree_mcts_engine_cls(
                        candidate_name, cfg, runtime_hooks.clone_actor_model(candidate_actor_template), device
                    )
                cand_eng = cand_factory()
                champion_actor = runtime_hooks.clone_actor_model(actor_source)
                if os.path.exists(best_model_path):
                    champion_actor = runtime_hooks.load_actor_source_from_checkpoint(
                        best_model_path,
                        cfg,
                        device,
                        backend_preference=backend.name if backend is not None else "torch",
                        backend_template=backend,
                    )
                champion_actor_template = runtime_hooks.clone_actor_model(champion_actor)
                if rust_ok:
                    champ_factory = lambda: runtime_hooks.rust_nn_evaluator_engine_cls(
                        "champion",
                        cfg,
                        runtime_hooks.clone_actor_model(champion_actor_template),
                        device,
                        args.rust_binary,
                    )
                else:
                    champ_factory = lambda: runtime_hooks.tree_mcts_engine_cls(
                        "champion", cfg, runtime_hooks.clone_actor_model(champion_actor_template), device
                    )
                champ_eng = champ_factory()
                if not eval_workers_autotuned and not (
                    hasattr(cand_eng, "select_moves_batch") and hasattr(champ_eng, "select_moves_batch")
                ):
                    tuned_workers, eval_benchmarks = runtime_hooks.benchmark_eval_parallel_workers(
                        hw, cfg, args.eval_games, cand_factory, champ_factory, game_factory, eval_autotune_profile_path
                    )
                    training_evaluator.cfg.parallel_workers = tuned_workers
                    eval_workers_autotuned = True
                    print(
                        f"  [EvalAutoTune] workers={tuned_workers} "
                        f"({len([b for b in eval_benchmarks if 'games_per_s' in b])} candidates benchmarked)"
                    )
                    entry["eval_worker_tune"] = {"workers": tuned_workers, "benchmarks": eval_benchmarks}
                elif not eval_workers_autotuned:
                    training_evaluator.cfg.parallel_workers = 1
                    eval_workers_autotuned = True
                    entry["eval_worker_tune"] = {"mode": "batched_rust", "workers": 1, "benchmarks": []}
                    print("  [EvalAutoTune] batched Rust evaluation active (worker autotune skipped)")
                eval_result = training_evaluator.evaluate_checkpoint(
                    candidate=cand_eng,
                    champion=champ_eng,
                    game_factory=game_factory,
                    candidate_id=candidate_name,
                    generation=iteration + 1,
                    candidate_factory=cand_factory,
                    champion_factory=champ_factory,
                )
            finally:
                if bg_worker and args.eval_selfplay_isolation and bg_pause_requested:
                    bg_worker.resume()
                    print("  [BG] Self-play resumed after evaluation")
                if cand_eng is not None:
                    try:
                        cand_eng.reset()
                    except Exception:
                        pass
                if champ_eng is not None:
                    try:
                        champ_eng.reset()
                    except Exception:
                        pass
            eval_valid = bool(getattr(eval_result, "valid_eval", True))
            eval_invalid_reason = getattr(eval_result, "invalid_reason", None)
            v = eval_result.promotion.get("verdict", "?")
            sr = eval_result.tally.get("score_rate", 0) if eval_result.tally else 0
            elo_d = eval_result.elo.get("delta", 0) if eval_result.elo else 0
            pub = eval_result.published.get("candidate_abs") if eval_result.published else None
            champ_pub = eval_result.published.get("champion_abs") if eval_result.published else None
            elo_gap = eval_result.published.get("delta") if eval_result.published else None
            if eval_valid:
                latest_eval.update(
                    {
                        "published_elo": pub,
                        "champion_elo": champ_pub,
                        "elo_gap": elo_gap,
                        "delta_elo": elo_d,
                        "score_rate": sr,
                        "eval_verdict": v,
                    }
                )
                print(f"  Eval: {v} | sr={sr:.3f} | ΔElo={elo_d:+.0f} | AbsElo={pub} | ChampElo={champ_pub}")
                if v == "promote":
                    if backend:
                        backend.save(best_model_path)
                    else:
                        torch.save(model.state_dict(), best_model_path)
                    print(f"  ★ PROMOTED: gen_{iteration+1} is new champion!")
            else:
                entry["eval_invalid_reason"] = str(eval_invalid_reason or "invalid evaluation")
                print(f"  [EvalInvalid] {entry['eval_invalid_reason']}")
            log_f.write(
                json.dumps(
                    runtime_hooks.make_json_safe(
                        {
                            "_type": "eval",
                            "iter": iteration + 1,
                            "valid_eval": eval_valid,
                            "invalid_reason": eval_invalid_reason,
                            "verdict": v,
                            "score_rate": sr,
                            "delta_elo": elo_d,
                            "published_elo": pub,
                            "champion_elo": champ_pub,
                            "elo_gap": elo_gap,
                            "games": eval_result.tally.get("scored", 0) if eval_result.tally else 0,
                            "errors": eval_result.tally.get("errors", 0) if eval_result.tally else 0,
                            "voids": eval_result.tally.get("voids", 0) if eval_result.tally else 0,
                        }
                    )
                )
                + "\n"
            )
            log_f.flush()

        entry.update(
            {
                "published_elo": latest_eval["published_elo"],
                "champion_elo": latest_eval["champion_elo"],
                "elo_gap": latest_eval["elo_gap"],
                "delta_elo": latest_eval["delta_elo"],
                "score_rate": latest_eval["score_rate"],
                "eval_verdict": latest_eval["eval_verdict"],
            }
        )
        log_f.write(json.dumps(runtime_hooks.make_json_safe(entry)) + "\n")
        log_f.flush()

        if should_early_stop:
            print(f"\n  Early stopping at iter {iteration+1} (patience={args.patience})")
            break

    log_f.close()
    if bg_worker:
        bg_worker.stop()
    if backend:
        backend.save(latest_model_path)
    else:
        torch.save(model.state_dict(), latest_model_path)
    if runtime_hooks.generate_training_plots(log_path, base_dir):
        print(
            f"  Plots: {os.path.join(base_dir, 'training_loss.png')}"
            f" | {os.path.join(base_dir, 'training_elo.png')}"
        )
    print(f"\nDone. Model: {latest_model_path}")
