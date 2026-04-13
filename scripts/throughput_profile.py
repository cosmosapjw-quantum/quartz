#!/usr/bin/env python3
"""
QUARTZ Throughput Profiler — Comprehensive GPU/CPU/MCTS Benchmarking
====================================================================

Generates a structured dataset of actual throughput measurements across:
  - 3 network sizes: S(64x6), M(128x10), L(256x20)
  - Multiple games: tictactoe, gomoku7, gomoku15, chess, go9, go19
  - Multiple batch sizes: 16, 64, 128, 256
  - Raw GPU inference + full selfplay pipeline

Output: JSON dataset for validating training time estimates.

Usage:
  venv/bin/python scripts/throughput_profile.py [--device cuda] [--games gomoku7,chess] [--quick]
"""
import os, sys, json, time, argparse, subprocess, signal
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Game × Network size matrix ──────────────────────────────────────

NETWORK_SIZES = {
    "S": {"filters": 64, "blocks": 6, "vh": 64},
    "M": {"filters": 128, "blocks": 10, "vh": 256},
    "L": {"filters": 256, "blocks": 20, "vh": 512},
}

GAME_SPECS = {
    "tictactoe": {"board": 3, "ch": 3, "actions": 9, "avg_ply": 6, "iters": 100, "profile_iters": 32},
    "gomoku7":   {"board": 7, "ch": 3, "actions": 49, "avg_ply": 30, "iters": 200, "profile_iters": 32},
    "gomoku15":  {"board": 15, "ch": 3, "actions": 225, "avg_ply": 50, "iters": 400, "profile_iters": 32},
    "chess":     {"board": 8, "ch": 16, "actions": 4096, "avg_ply": 80, "iters": 200, "profile_iters": 16},
    "go9":       {"board": 9, "ch": 17, "actions": 82, "avg_ply": 60, "iters": 200, "profile_iters": 32},
    "go19":      {"board": 19, "ch": 17, "actions": 362, "avg_ply": 200, "iters": 200, "profile_iters": 16},
}

BATCH_SIZES = [16, 64, 128, 256]


def make_cfg(game, net_size):
    """Build a config dict for a given game + network size."""
    gs = GAME_SPECS[game]
    ns = NETWORK_SIZES[net_size]
    # Use reduced iters for profiling (enough to measure throughput, not full strength)
    profile_iters = gs.get("profile_iters", 32)
    return {
        "_name": game,
        "board": gs["board"],
        "ch": gs["ch"],
        "actions": gs["actions"],
        "filters": ns["filters"],
        "blocks": ns["blocks"],
        "vh": ns["vh"],
        "iters": profile_iters,
        "games": 10,
        "temp_th": max(4, gs["avg_ply"] // 4),
        "batch_size": 64,
        "n_threads": 4,
        "penalty_mode": "GatedRefresh",
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
        "batch_timeout_us": 1500,
        "search_profile": "quartz",
        "recent_frac": 0.8,
        "recent_window": 10000,
        "dir_a": 0.3,
        "buf": 10000,
        "steps": 10,
        "batch": 64,
        "win": 4 if "gomoku" in game else 0,
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
    }


# ── Phase 1: Raw GPU inference benchmark ────────────────────────────

def benchmark_raw_inference(model, device, cfg, batch_sizes, warmup=20, n_iters=200):
    """Benchmark raw NN forward pass for each batch size."""
    import torch
    from quartz.alphazero_train import _run_model_batch, _get_compiled_model

    ch, bs = cfg["ch"], cfg["board"]
    results = []

    # Force compile warmup
    dummy = [np.random.randn(ch, bs, bs).astype(np.float32) for _ in range(16)]
    for _ in range(warmup):
        _run_model_batch(model, device, dummy)

    for batch_size in batch_sizes:
        batch = [np.random.randn(ch, bs, bs).astype(np.float32) for _ in range(batch_size)]
        # Warmup this specific size
        for _ in range(10):
            _run_model_batch(model, device, batch)

        N = max(30, n_iters * 16 // batch_size)
        t0 = time.perf_counter()
        for _ in range(N):
            _run_model_batch(model, device, batch)
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_batch = elapsed / N * 1000
        items_per_s = batch_size * N / elapsed
        results.append({
            "batch_size": batch_size,
            "ms_per_batch": round(ms_per_batch, 3),
            "items_per_s": round(items_per_s, 1),
            "leaf_eval_per_s": round(items_per_s, 1),
        })
    return results


# ── Phase 2: Selfplay throughput via Rust server ────────────────────

def benchmark_selfplay(model, device, cfg, rust_binary, n_games=2, parallel=2, timeout_s=90):
    """Run actual selfplay and measure games/s, positions/s.
    Uses a timeout thread to kill the server if games take too long.
    """
    from quartz.alphazero_train import (
        NNSearchClient, rust_game_name, supports_rust_selfplay_state_machine,
        clear_nn_eval_cache,
    )
    import random, threading

    clear_nn_eval_cache()
    game_name = cfg["_name"]
    iters = cfg["iters"]

    if not os.path.exists(rust_binary):
        return {"error": f"rust binary not found: {rust_binary}", "games_per_s": 0}

    if not supports_rust_selfplay_state_machine(game_name):
        return {"error": f"rust selfplay not supported for {game_name}", "games_per_s": 0}

    client = NNSearchClient(model, cfg, device, rust_binary)
    result_box = [None]  # mutable container for thread result
    error_box = [None]
    all_games = []

    def on_chunk(games):
        all_games.extend(games)

    def run_selfplay():
        try:
            result_box[0] = client.selfplay_run(
                n_games=n_games, parallel=parallel,
                temp_threshold=cfg.get("temp_th", 8),
                penalty_mode=cfg.get("penalty_mode", "GatedRefresh"),
                seed=random.randint(0, 2**31),
                on_chunk=on_chunk,
            )
        except Exception as e:
            error_box[0] = e

    try:
        client.start()
        t0 = time.perf_counter()

        worker = threading.Thread(target=run_selfplay, daemon=True)
        worker.start()
        worker.join(timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        timed_out = worker.is_alive()

        if timed_out:
            # Kill server to unblock the worker thread
            try:
                client.proc.kill()
            except Exception:
                pass
            worker.join(timeout=5)

        if error_box[0] is not None:
            raise error_box[0]

        # Count completed games
        completed = 0
        total_positions = 0
        result = result_box[0]
        if isinstance(result, dict):
            completed = result.get("completed_games", 0) or len(all_games)
        if completed == 0:
            completed = len(all_games)
        for g in all_games:
            if isinstance(g, dict):
                total_positions += len(g.get("states", g.get("policies", [])))

        if completed == 0:
            suffix = " (timed out)" if timed_out else ""
            return {"error": f"0 games completed in {elapsed:.0f}s{suffix}", "games_per_s": 0}

        games_per_s = completed / max(elapsed, 0.001)
        avg_ply = total_positions / max(completed, 1)
        leaf_evals = completed * iters * avg_ply
        leaf_eval_per_s = leaf_evals / max(elapsed, 0.001)

        return {
            "completed_games": completed,
            "elapsed_s": round(elapsed, 2),
            "games_per_s": round(games_per_s, 2),
            "positions_per_s": round(total_positions / max(elapsed, 0.001), 1),
            "avg_ply": round(avg_ply, 1),
            "total_positions": total_positions,
            "leaf_eval_per_s_approx": round(leaf_eval_per_s, 0),
            "games_per_day": round(games_per_s * 86400, 0),
            "timed_out": timed_out,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "games_per_s": 0}
    finally:
        try:
            client.stop()
        except Exception:
            pass


# ── Phase 3: Training step benchmark ────────────────────────────────

def benchmark_training_step(model, device, cfg, n_steps=20):
    """Benchmark training throughput (examples/s)."""
    import torch

    if hasattr(model, "predict"):
        return {"error": "non-torch model", "examples_per_s": 0}

    ch, bs = cfg["ch"], cfg["board"]
    actions = cfg["actions"]
    train_batch = cfg.get("batch", 64)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn_p = torch.nn.CrossEntropyLoss()
    loss_fn_v = torch.nn.MSELoss()

    model.train()
    # Generate random training data
    x = torch.randn(train_batch, ch, bs, bs, device=device)
    policy_target = torch.randint(0, actions, (train_batch,), device=device)
    value_target = torch.randn(train_batch, device=device).clamp(-1, 1)

    # Warmup
    for _ in range(5):
        logits, vals = model(x)
        loss = loss_fn_p(logits, policy_target) + loss_fn_v(vals, value_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        logits, vals = model(x)
        loss = loss_fn_p(logits, policy_target) + loss_fn_v(vals, value_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    model.eval()
    examples_per_s = train_batch * n_steps / elapsed
    return {
        "train_batch": train_batch,
        "steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "examples_per_s": round(examples_per_s, 1),
        "ms_per_step": round(elapsed / n_steps * 1000, 2),
    }


# ── Main profiler ───────────────────────────────────────────────────

def run_profile(args):
    import torch
    torch.set_float32_matmul_precision("high")
    import warnings, logging as _logging
    warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
    warnings.filterwarnings("ignore", message=".*TensorFloat32.*")
    warnings.filterwarnings("ignore", message=".*cache_size_limit.*")
    _logging.getLogger("torch._dynamo").setLevel(_logging.ERROR)
    from quartz.alphazero_train import AlphaZeroNet, _COMPILED_MODELS, _PINNED_BUFS

    device = torch.device(args.device)
    games = [g.strip() for g in args.games.split(",")]
    net_sizes = [s.strip() for s in args.sizes.split(",")]
    rust_binary = args.rust_binary

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"Games: {games}")
    print(f"Sizes: {net_sizes}")
    print()

    dataset = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "results": [],
    }

    for game in games:
        if game not in GAME_SPECS:
            print(f"  [SKIP] Unknown game: {game}")
            continue

        for net_size in net_sizes:
            if net_size not in NETWORK_SIZES:
                print(f"  [SKIP] Unknown size: {net_size}")
                continue

            cfg = make_cfg(game, net_size)
            ns = NETWORK_SIZES[net_size]
            label = f"{game}/{net_size}({ns['filters']}x{ns['blocks']})"
            print(f"{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            # Clear caches for fresh measurement
            _COMPILED_MODELS.clear()
            _PINNED_BUFS.clear()

            # Create model
            try:
                model = AlphaZeroNet(cfg).to(device).eval()
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  Params: {n_params:,}")
            except Exception as e:
                print(f"  [FAIL] Model creation: {e}")
                dataset["results"].append({
                    "game": game, "net_size": net_size, "error": str(e),
                })
                continue

            entry = {
                "game": game,
                "net_size": net_size,
                "filters": ns["filters"],
                "blocks": ns["blocks"],
                "board": cfg["board"],
                "ch": cfg["ch"],
                "actions": cfg["actions"],
                "params": n_params,
                "avg_ply_estimate": GAME_SPECS[game]["avg_ply"],
                "iters": cfg["iters"],
            }

            # Phase 1: Raw inference
            print(f"  [1/3] Raw inference benchmark...")
            try:
                bs_list = [bs for bs in BATCH_SIZES if bs * cfg["ch"] * cfg["board"]**2 * 4 < 512 * 1024 * 1024]
                if not bs_list:
                    bs_list = [16, 64]
                inference = benchmark_raw_inference(model, device, cfg, bs_list,
                                                   warmup=10 if args.quick else 20,
                                                   n_iters=50 if args.quick else 200)
                entry["inference"] = inference
                best = max(inference, key=lambda r: r["items_per_s"])
                print(f"       Best: BS={best['batch_size']} → {best['items_per_s']:.0f} items/s "
                      f"({best['ms_per_batch']:.2f} ms/batch)")
            except Exception as e:
                print(f"       [FAIL] {e}")
                entry["inference"] = {"error": str(e)}

            # Phase 2: Selfplay (only if rust binary exists)
            if os.path.exists(rust_binary):
                print(f"  [2/3] Selfplay benchmark...")
                try:
                    n_games = 2
                    sp_timeout = 60 if args.quick else 120
                    selfplay = benchmark_selfplay(
                        model, device, cfg, rust_binary,
                        n_games=n_games, parallel=min(2, n_games),
                        timeout_s=sp_timeout)
                    entry["selfplay"] = selfplay
                    if selfplay.get("games_per_s", 0) > 0:
                        to = " [partial]" if selfplay.get("timed_out") else ""
                        print(f"       {selfplay['completed_games']} games in {selfplay['elapsed_s']:.1f}s "
                              f"→ {selfplay['games_per_s']:.2f} games/s, "
                              f"{selfplay.get('games_per_day', 0):.0f} games/day{to}")
                    else:
                        err = selfplay.get("error", "0 games completed")
                        if "\n" in str(err):
                            err = str(err).split("\n")[0]
                        print(f"       [NO DATA] {err}")
                except Exception as e:
                    print(f"       [FAIL] {e}")
                    entry["selfplay"] = {"error": str(e)}
            else:
                print(f"  [2/3] Selfplay: SKIPPED (no rust binary)")
                entry["selfplay"] = {"skipped": True}

            # Phase 3: Training step
            print(f"  [3/3] Training step benchmark...")
            try:
                train_result = benchmark_training_step(
                    model, device, cfg,
                    n_steps=5 if args.quick else 20)
                entry["training"] = train_result
                print(f"       {train_result['examples_per_s']:.0f} examples/s "
                      f"({train_result['ms_per_step']:.1f} ms/step)")
            except Exception as e:
                print(f"       [FAIL] {e}")
                entry["training"] = {"error": str(e)}

            # Derived estimates
            try:
                best_inf = max(entry.get("inference", [{"items_per_s": 0}]),
                              key=lambda r: r.get("items_per_s", 0))
                raw_leaf_eval_s = best_inf.get("items_per_s", 0)
                avg_ply = GAME_SPECS[game]["avg_ply"]
                iters = cfg["iters"]

                # Selfplay efficiency ratio (actual vs raw)
                sp = entry.get("selfplay", {})
                actual_leaf_s = sp.get("leaf_eval_per_s_approx", 0)
                efficiency = actual_leaf_s / raw_leaf_eval_s if raw_leaf_eval_s > 0 else 0

                entry["estimates"] = {
                    "raw_leaf_eval_per_s": round(raw_leaf_eval_s, 0),
                    "selfplay_efficiency": round(efficiency, 3),
                    "sustained_leaf_eval_per_s": round(actual_leaf_s, 0) if actual_leaf_s > 0
                        else round(raw_leaf_eval_s * 0.15, 0),  # conservative estimate
                    "estimated_games_per_day": sp.get("games_per_day", 0),
                }
            except Exception:
                pass

            dataset["results"].append(entry)

            # Free GPU memory
            del model
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            _COMPILED_MODELS.clear()
            _PINNED_BUFS.clear()
            print()

    # Save results
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Print summary table
    print_summary_table(dataset)
    return dataset


def print_summary_table(dataset):
    """Print a human-readable summary table."""
    results = dataset.get("results", [])
    if not results:
        return

    print(f"\n{'='*90}")
    print(f"  THROUGHPUT SUMMARY")
    print(f"{'='*90}")
    print(f"{'Game':<12} {'Net':<12} {'Params':>10} {'Best leaf/s':>12} {'Selfplay g/s':>13} {'Games/day':>12} {'Train ex/s':>11}")
    print(f"{'-'*90}")

    for r in results:
        if "error" in r and isinstance(r.get("error"), str):
            continue
        game = r.get("game", "?")
        net = f"{r.get('net_size','?')}({r.get('filters','?')}x{r.get('blocks','?')})"
        params = f"{r.get('params', 0):,}"

        inf = r.get("inference", [])
        if isinstance(inf, list) and inf:
            best_leaf = max(inf, key=lambda x: x.get("items_per_s", 0)).get("items_per_s", 0)
        else:
            best_leaf = 0

        sp = r.get("selfplay", {})
        gps = sp.get("games_per_s", 0)
        gpd = sp.get("games_per_day", 0)

        tr = r.get("training", {})
        eps = tr.get("examples_per_s", 0)

        print(f"{game:<12} {net:<12} {params:>10} {best_leaf:>12,.0f} {gps:>13.2f} {gpd:>12,.0f} {eps:>11,.0f}")

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="QUARTZ Throughput Profiler")
    parser.add_argument("--device", default="cuda" if "torch" not in sys.modules or __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--games", default="tictactoe,gomoku7,gomoku15,chess,go9",
                        help="Comma-separated game list")
    parser.add_argument("--sizes", default="S,M,L",
                        help="Comma-separated network sizes (S, M, L)")
    parser.add_argument("--output", default="tmp/throughput_profile.json")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer iterations for faster results (less accurate)")
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
