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
small budget (tictactoe, 1 iter × 1 game × 4 MCTS iters × batch=1)
and asserting:

  1. The Rust subprocess actually launched and persisted replay samples.
  2. ≥1 SGD row appears in `train_log.jsonl` (same contract as P3 smoke).
  3. The loss between the first SGD row and the last is non-increasing
     within reasonable tolerance — a strict `<` is too brittle at this
     budget; the contract is "training did not diverge".

This test is opt-in. It is marked `@pytest.mark.real_loop` and skipped
when the Rust binary is missing. Default CI runs `pytest -q tests/`
without this marker, so the test does not gate every PR; researchers
opt in with `pytest -m real_loop tests/`.

Wall-clock: ~10-45 s on a 5900X with the Rust binary already built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _count_selfplay_artifacts(report_dir: Path) -> int:
    """Count persisted self-play/replay artifacts from the current trainer.

    The active `quartz.train` path stores samples in `replay.npz`; older
    sharded JSON names are kept here only as compatibility fallbacks.
    """
    roots = [report_dir]
    models_dir = report_dir / "models"
    if models_dir.exists():
        roots.append(models_dir)

    candidates: list[Path] = []
    for root in roots:
        candidates += list(root.rglob("game_*.json"))
        candidates += list(root.rglob("selfplay/*.json"))
        candidates += list(root.rglob("selfplay_*.npz"))
        candidates += list(root.rglob("replay.npz"))
    return len(candidates)


def _find_train_logs(report_dir: Path) -> list[Path]:
    """Find train logs in both direct-output and legacy models layouts."""
    logs = []
    direct = report_dir / "train_log.jsonl"
    if direct.exists():
        logs.append(direct)
    models_dir = report_dir / "models"
    if models_dir.exists():
        logs += list(models_dir.rglob("train_log.jsonl"))
    return logs


def test_q5_real_loop_smoke_closes_the_pipeline(tmp_path: Path) -> None:
    """Run `python -m quartz.train` and assert the full loop
    closed: Rust subprocess produced self-play games, SGD fired, and the
    loss did not diverge.
    """
    output = tmp_path / "real_loop_output"
    cmd = [
        sys.executable,
        "-m",
        "quartz.train",
        "--game",
        "tictactoe",
        "--iterations",
        "1",
        # Keep the smoke small and force SGD with batch=1 below. This test
        # validates the handoff contracts, not convergence speed.
        "--games",
        "1",
        "--device",
        "cpu",
        "--no-autotune",
        "--selfplay-parallel",
        "1",
        "--mcts-threads",
        "1",
        "--nn-batch-size",
        "1",
        "--config",
        str(tmp_path / "real_loop_cfg.json"),
        # Skip arena/Glicko in this smoke; the test asserts the
        # selfplay→SGD half of the loop. A separate test could cover
        # eval scheduling.
        "--eval-interval",
        "999999",
        "--seed",
        "1",
        "--no-pipeline",
        "--rust-binary",
        str(RUST_BINARY),
        "--output",
        str(output),
    ]
    (tmp_path / "real_loop_cfg.json").write_text(
        json.dumps(
            {"batch": 1, "steps": 1, "iters": 4, "n_threads": 1, "batch_size": 1}
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    # Prevent stray inductor warmup blocking the subprocess.
    env.setdefault("QUARTZ_FORCE_EAGER_EVAL", "1")
    # Bound worker fan-out in environments where the safe runtime caps are
    # honored; harmless for the sequential `--no-pipeline` path.
    env.setdefault("QUARTZ_SAFE_RUNTIME", "1")
    env.setdefault("QUARTZ_SAFE_BOOTSTRAP_TARGET_CAP", "2")
    env.setdefault("QUARTZ_SAFE_SELFPLAY_PARALLEL_CAP", "1")
    env.setdefault("QUARTZ_SAFE_SELFPLAY_BATCH_GAMES_CAP", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "quartz.train timed out in the real-loop smoke.\n"
            f"stdout (last 4 KiB):\n{(exc.stdout or '')[-4096:]}\n"
            f"stderr (last 4 KiB):\n{(exc.stderr or '')[-4096:]}"
        )

    if proc.returncode != 0:
        pytest.fail(
            f"quartz.train exit={proc.returncode}\n"
            f"stdout (last 4 KiB):\n{proc.stdout[-4096:]}\n"
            f"stderr (last 4 KiB):\n{proc.stderr[-4096:]}"
        )

    # (a) self-play actually produced games.
    n_games = _count_selfplay_artifacts(output)
    assert n_games >= 1, (
        f"expected ≥1 replay/self-play artifact under {output}, found {n_games}. "
        f"Self-play subprocess may have failed silently. "
        f"stderr tail:\n{proc.stderr[-2048:]}"
    )

    # (b) SGD fired in at least one train_log.jsonl row.
    train_logs = _find_train_logs(output)
    assert train_logs, (
        f"no train_log.jsonl under {output} — quartz.train ran but "
        f"emitted no training artifacts."
    )
    sgd_rows: list[dict] = []
    for path in train_logs:
        for row in _read_train_log_rows(path):
            if row.get("loss") is not None:
                sgd_rows.append(row)
    assert sgd_rows, (
        f"0 SGD rows across {len(train_logs)} train_log.jsonl files. "
        f"replay never crossed the batch threshold; bump --games "
        f"or shrink batch in the config override."
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


def test_real_loop_eval_uses_durable_candidate_checkpoint(tmp_path: Path) -> None:
    """Run the live entrypoint through the eval barrier.

    This is intentionally tiny; it verifies artifact identity, not playing
    strength. The candidate evaluated by arena must exist as `gen_1.pt` and the
    eval log row must record its hash.
    """
    output = tmp_path / "real_loop_eval_output"
    cfg_path = tmp_path / "real_loop_eval_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {"batch": 1, "steps": 1, "iters": 4, "n_threads": 1, "batch_size": 1}
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "quartz.train",
        "--game",
        "tictactoe",
        "--iterations",
        "1",
        "--games",
        "1",
        "--device",
        "cpu",
        "--no-autotune",
        "--selfplay-parallel",
        "1",
        "--mcts-threads",
        "1",
        "--nn-batch-size",
        "1",
        "--config",
        str(cfg_path),
        "--eval-interval",
        "1",
        "--eval-games",
        "2",
        "--seed",
        "2",
        "--no-pipeline",
        "--rust-binary",
        str(RUST_BINARY),
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env.setdefault("QUARTZ_FORCE_EAGER_EVAL", "1")
    env.setdefault("QUARTZ_SAFE_RUNTIME", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"quartz.train eval-loop exit={proc.returncode}\n"
            f"stdout (last 4 KiB):\n{proc.stdout[-4096:]}\n"
            f"stderr (last 4 KiB):\n{proc.stderr[-4096:]}"
        )

    candidate = output / "gen_1.pt"
    assert candidate.exists(), "training-time eval did not persist gen_1.pt"
    train_logs = _find_train_logs(output)
    rows = [row for path in train_logs for row in _read_train_log_rows(path)]
    eval_rows = [row for row in rows if row.get("_type") == "eval"]
    assert eval_rows, "expected an eval log row"
    assert eval_rows[-1]["candidate_model_path"] == str(candidate)
    assert eval_rows[-1]["candidate_model_sha256"]
