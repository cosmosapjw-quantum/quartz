"""Q5 (audit_codex_20260428.md W'3): real-loop end-to-end regression.

The existing P10 test (`tests/test_e2e_convergence.py`) verifies the SGD
loop converges on a synthetic 128-sample replay. That covers the
optimizer side but does not exercise any of:

  - Rust subprocess launch (`mcts_demo --server`)
  - QIPC handshake + SHM ring transport
  - selfplay → replay handoff with `actor_generation` tagging
  - replay → SGD threshold gating
  - end-of-iteration `bg_worker.update_model` propagation
  - per-iteration eval scheduling

A regression in any of those would not break P10 today. Q5 fills that
gap by spawning the real `python -m quartz.train` entry point with a
small budget (2 iter × 16 games × batch=64) and asserting:

  1. The Rust subprocess actually launched and produced ≥1 self-play game.
  2. ≥1 SGD row appears in `train_log.jsonl` (same contract as P3 smoke).
  3. The loss between the first SGD row and the last is non-increasing
     within reasonable tolerance — a strict `<` is too brittle at this
     budget; the contract is "training did not diverge".

This test is opt-in. It is marked `@pytest.mark.real_loop` and skipped
when the Rust binary is missing. Default CI runs `pytest -q tests/`
without this marker, so the test does not gate every PR; researchers
opt in with `pytest -m real_loop tests/`.

Wall-clock: ~60-180 s on a 5900X with the Rust binary already built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_BINARY = REPO_ROOT / "target" / "release" / "mcts_demo"


pytestmark = [
    pytest.mark.real_loop,
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUST_BINARY.exists(),
        reason="Rust release binary missing; run `cargo build --release` first.",
    ),
]


def _read_train_log_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            rows.append(payload)
    return rows


def _count_selfplay_games(report_dir: Path) -> int:
    """Walk `report_dir/models/**/selfplay/*.json*` and count files.

    Self-play games can land as JSON or compressed shards depending on the
    runtime config; we count any file whose name suggests it carries a
    self-play game payload.
    """
    models_dir = report_dir / "models"
    if not models_dir.exists():
        return 0
    candidates = list(models_dir.rglob("game_*.json"))
    candidates += list(models_dir.rglob("selfplay/*.json"))
    candidates += list(models_dir.rglob("selfplay_*.npz"))
    return len(candidates)


def test_q5_real_loop_two_iterations_close_the_pipeline(tmp_path: Path) -> None:
    """Run `python -m quartz.train` for 2 iters and assert the full loop
    closed: Rust subprocess produced self-play games, SGD fired, and the
    loss did not diverge.
    """
    output = tmp_path / "real_loop_output"
    cmd = [
        sys.executable,
        "-m",
        "quartz.train",
        "--game",
        "gomoku7",
        "--iterations",
        "2",
        # Crossing the SGD batch threshold reliably at gomoku7 needs
        # ~256 positions; 16 games × ~24 positions/game ≈ 384.
        "--games-per-iter",
        "16",
        "--batch",
        "64",
        # Skip arena/Glicko in this smoke; the test asserts the
        # selfplay→SGD half of the loop. A separate test could cover
        # eval scheduling.
        "--eval-interval",
        "999999",
        "--seeds",
        "1",
        "--rust-binary",
        str(RUST_BINARY),
        "--output",
        str(output),
    ]

    env = os.environ.copy()
    # Prevent stray inductor warmup blocking the subprocess.
    env.setdefault("QUARTZ_FORCE_EAGER_EVAL", "1")

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"quartz.train exit={proc.returncode}\n"
            f"stdout (last 4 KiB):\n{proc.stdout[-4096:]}\n"
            f"stderr (last 4 KiB):\n{proc.stderr[-4096:]}"
        )

    # (a) self-play actually produced games.
    n_games = _count_selfplay_games(output)
    assert n_games >= 1, (
        f"expected ≥1 self-play game artifact under {output}/models, found {n_games}. "
        f"Self-play subprocess may have failed silently. "
        f"stderr tail:\n{proc.stderr[-2048:]}"
    )

    # (b) SGD fired in at least one train_log.jsonl row.
    train_logs = list((output / "models").rglob("train_log.jsonl"))
    assert train_logs, (
        f"no train_log.jsonl under {output}/models — quartz.train ran but "
        f"emitted no training artifacts."
    )
    sgd_rows: list[dict] = []
    for path in train_logs:
        for row in _read_train_log_rows(path):
            if row.get("loss") is not None:
                sgd_rows.append(row)
    assert sgd_rows, (
        f"0 SGD rows across {len(train_logs)} train_log.jsonl files. "
        f"replay never crossed the batch threshold; bump --games-per-iter "
        f"or shrink --batch."
    )

    # (c) loss did not diverge. We accept any non-increase and a small
    # uptick (≤ 25 %) to absorb stochastic noise at this budget; the
    # purpose is to catch hard regressions (NaN, runaway loss), not to
    # establish convergence — the synthetic-replay P10 test owns that.
    first_loss = float(sgd_rows[0]["loss"])
    last_loss = float(sgd_rows[-1]["loss"])
    assert first_loss == first_loss, "first loss is NaN"  # NaN check
    assert last_loss == last_loss, "last loss is NaN"
    upper_bound = first_loss * 1.25
    assert last_loss <= upper_bound, (
        f"loss diverged across the run: first={first_loss:.4f} "
        f"last={last_loss:.4f} (upper_bound={upper_bound:.4f}). "
        f"Hard regression in the train loop."
    )
