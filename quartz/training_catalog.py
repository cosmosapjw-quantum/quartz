"""Shared training catalog and config helpers for QUARTZ runtimes."""

from __future__ import annotations

import os

from quartz.selfplay_runtime import (
    is_chess_game,
    is_go_game,
    rust_game_name as _rust_game_name_impl,
)

CHESS_POLICY_ACTIONS = 4672
SEARCH_RUNTIME_KEYS = {
    "sigma_0",
    "min_visits",
    "check_interval",
    "prior_refresh_rate",
    "prior_refresh_temp",
    "hbar_penalty_cap",
    "c_puct",
    "penalty_mode",
    "root_only_shaping",
    "n_threads",
    "batch_size",
    "batch_timeout_us",
    # P7 (audit_codex_20260425.md W2): per-condition halt-mode override
    # plumbed through to mcts_server's `parse_halt_mode_override`.
    # Accepts "fixed" / "voc" / "simple_threshold". "fixed" pins
    # `HaltMode::Fixed { budget = u32::MAX }`, disabling adaptive halts
    # so attribution presets see same-budget arena rows.
    "halt_mode",
}

GOMOKU15_VARIANTS = {
    "gomoku15",
    "gomoku15_free",
    "gomoku15_std",
    "gomoku15_omok",
    "gomoku15_renju",
    "gomoku15_caro",
}

GO_RULESET_PRESETS = {
    "cn": dict(
        go_ruleset="chinese", go_scoring="area", go_komi=7.5, go_allow_suicide=False
    ),
    "jp": dict(
        go_ruleset="japanese",
        go_scoring="territory",
        go_komi=6.5,
        go_allow_suicide=False,
    ),
    "kr": dict(
        go_ruleset="korean", go_scoring="territory", go_komi=6.5, go_allow_suicide=False
    ),
}

STANDARD_CHESS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_go_cfg(
    board,
    filters,
    blocks,
    vh,
    iters,
    games,
    temp_th,
    dir_a,
    steps,
    batch,
    min_visits,
    check_interval,
    recent_window,
    suffix="cn",
):
    cfg = dict(
        board=board,
        ch=17,
        actions=board * board + 1,
        win=0,
        filters=filters,
        blocks=blocks,
        vh=vh,
        iters=iters,
        games=games,
        temp_th=temp_th,
        dir_a=dir_a,
        buf=1_000_000,
        steps=steps,
        batch=batch,
        penalty_mode="GatedRefresh",
        hbar_penalty_cap=0.3,
        sigma_0=0.3,
        min_visits=min_visits,
        check_interval=check_interval,
        prior_refresh_rate=0.0,
        prior_refresh_temp=1.0,
        c_puct=2.5,
        n_threads=4,
        batch_size=8,
        recent_frac=0.8,
        recent_window=recent_window,
        tt_enabled=True,
    )
    cfg.update(GO_RULESET_PRESETS[suffix])
    return cfg


_GOMOKU15_BASE = dict(
    board=15,
    ch=17,
    actions=225,
    win=5,
    filters=128,
    blocks=8,
    vh=256,
    iters=200,
    games=100,
    temp_th=15,
    dir_a=0.15,
    buf=500_000,
    steps=200,
    batch=512,
    penalty_mode="GatedRefresh",
    hbar_penalty_cap=0.3,
    sigma_0=0.3,
    min_visits=50,
    check_interval=100,
    prior_refresh_rate=0.0,
    prior_refresh_temp=1.0,
    c_puct=2.0,
    n_threads=4,
    batch_size=8,
    recent_frac=0.8,
    recent_window=50_000,
)

_CHESS_BASE = dict(
    board=8,
    ch=36,
    actions=CHESS_POLICY_ACTIONS,
    win=0,
    filters=128,
    blocks=10,
    vh=256,
    iters=800,
    games=40,
    temp_th=15,
    dir_a=0.3,
    buf=1_000_000,
    steps=400,
    batch=256,
    penalty_mode="GatedRefresh",
    hbar_penalty_cap=0.3,
    sigma_0=0.3,
    min_visits=50,
    check_interval=100,
    prior_refresh_rate=0.0,
    prior_refresh_temp=1.0,
    c_puct=2.5,
    n_threads=4,
    batch_size=8,
    recent_frac=0.8,
    recent_window=100_000,
    tt_enabled=True,
)

GAME_CONFIGS = {
    "gomoku7": dict(
        board=7,
        ch=17,
        actions=49,
        win=4,
        filters=64,
        blocks=4,
        vh=64,
        iters=200,
        games=200,
        temp_th=8,
        dir_a=0.5,
        buf=200_000,
        steps=100,
        batch=256,
        penalty_mode="GatedRefresh",
        hbar_penalty_cap=0.3,
        sigma_0=0.3,
        min_visits=15,
        check_interval=20,
        prior_refresh_rate=0.0,
        prior_refresh_temp=1.0,
        c_puct=2.0,
        n_threads=4,
        batch_size=8,
        recent_frac=0.8,
        recent_window=20_000,
    ),
    "gomoku15": dict(_GOMOKU15_BASE),
    "gomoku15_free": dict(_GOMOKU15_BASE),
    "gomoku15_std": dict(_GOMOKU15_BASE),
    "gomoku15_omok": dict(_GOMOKU15_BASE),
    "gomoku15_renju": dict(_GOMOKU15_BASE),
    "gomoku15_caro": dict(_GOMOKU15_BASE),
    "go9": _make_go_cfg(
        9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "cn"
    ),
    "go9_cn": _make_go_cfg(
        9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "cn"
    ),
    "go9_jp": _make_go_cfg(
        9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "jp"
    ),
    "go9_kr": _make_go_cfg(
        9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "kr"
    ),
    "go13": _make_go_cfg(
        13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "cn"
    ),
    "go13_cn": _make_go_cfg(
        13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "cn"
    ),
    "go13_jp": _make_go_cfg(
        13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "jp"
    ),
    "go13_kr": _make_go_cfg(
        13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "kr"
    ),
    "go19": _make_go_cfg(
        19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "cn"
    ),
    "go19_cn": _make_go_cfg(
        19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "cn"
    ),
    "go19_jp": _make_go_cfg(
        19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "jp"
    ),
    "go19_kr": _make_go_cfg(
        19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "kr"
    ),
    "chess": dict(_CHESS_BASE, chess960=False, chess960_index=None),
    "chess960": dict(_CHESS_BASE, chess960=True, chess960_index=None),
    "tictactoe": dict(
        board=3,
        ch=17,
        actions=9,
        win=3,
        filters=32,
        blocks=2,
        vh=32,
        iters=100,
        games=400,
        temp_th=4,
        dir_a=0.6,
        buf=50_000,
        steps=64,
        batch=128,
        penalty_mode="GatedRefresh",
        hbar_penalty_cap=0.3,
        sigma_0=0.3,
        min_visits=8,
        check_interval=16,
        prior_refresh_rate=0.0,
        prior_refresh_temp=1.0,
        c_puct=1.5,
        n_threads=2,
        batch_size=8,
        recent_frac=0.8,
        recent_window=10_000,
    ),
}


def rust_game_name(game_name):
    return _rust_game_name_impl(game_name, GAME_CONFIGS, GOMOKU15_VARIANTS)


def apply_config_overrides(cfg, overrides):
    merged = dict(cfg)
    unknown = []
    for key, value in overrides.items():
        if key in merged or key in SEARCH_RUNTIME_KEYS:
            merged[key] = value
        else:
            unknown.append(key)
    if unknown:
        print(
            f"  [WARN] Ignoring unsupported config keys: {', '.join(sorted(unknown))}"
        )
    return merged


def resolve_runtime_paths(base_dir, explicit_model=None, resume=False):
    latest_model_path = os.path.join(base_dir, "latest.pt")
    best_model_path = os.path.join(base_dir, "best.pt")
    if explicit_model:
        load_model_path = explicit_model
    elif resume and os.path.exists(latest_model_path):
        load_model_path = latest_model_path
    elif resume and os.path.exists(best_model_path):
        load_model_path = best_model_path
    else:
        load_model_path = latest_model_path
    return {
        "load_model_path": load_model_path,
        "latest_model_path": latest_model_path,
        "best_model_path": best_model_path,
        "replay_path": os.path.join(base_dir, "replay.npz"),
        "log_path": os.path.join(base_dir, "train_log.jsonl"),
        "autotune_profile_path": os.path.join(base_dir, "autotune_profile.json"),
    }


__all__ = [
    "CHESS_POLICY_ACTIONS",
    "GAME_CONFIGS",
    "GOMOKU15_VARIANTS",
    "GO_RULESET_PRESETS",
    "SEARCH_RUNTIME_KEYS",
    "STANDARD_CHESS_FEN",
    "apply_config_overrides",
    "is_chess_game",
    "is_go_game",
    "resolve_runtime_paths",
    "rust_game_name",
]
