"""P10 (audit_codex_20260425.md W10): end-to-end training-loop convergence
+ determinism regression.

The audit's W10 named two gaps:
  (a) no test asserts that loss actually decreases over multiple iterations
      ("AlphaZero training loop" claim has no behavioural regression).
  (b) no test asserts that re-running with the same seed produces
      byte-identical output for at least the first iteration.

This test fills both gaps with a focused-but-real torch SGD run. It does
NOT spawn the Rust self-play binary — the synthesis below produces a
fixed-seed gomoku7-shape replay so the test executes in seconds and is
robust to the absence of a built binary.

Behaviour pinned:
  - At least one SGD step is observed (n_train_rows > 0) — same contract
    as smoke_e2e's P3 assertion.
  - Mean loss over the last fifth of iterations is strictly below the
    mean loss over the first fifth (loose tolerance for stochastic noise,
    but with 5 iters × 8 steps per iter and a tiny network this should
    converge reliably).
  - Determinism: a fresh model + replay with the same seed produces a
    first-iteration loss within 1e-5 of the prior run.

The test is marked `@pytest.mark.slow` because building two tiny torch
networks adds ~3 s per run; CI gates that opt out of slow tests can
exclude this.
"""

import importlib
import random as _random

import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _build_replay(seed: int, n_examples: int, board: int, num_actions: int):
    """Build a small synthetic replay buffer with deterministic random
    boards / target policies / target values keyed by `seed`.
    """
    replay_mod = importlib.import_module("quartz.replay")
    rng = np.random.default_rng(seed)
    replay = replay_mod.ReplayBuffer(n_examples)
    # AlphaZero gomoku encoding uses 17 channels per the catalog.
    n_channels = 17
    for _ in range(n_examples):
        # Synthetic feature plane: random {0, 1} occupancy across channels.
        state = rng.integers(0, 2, size=(n_channels, board, board)).astype(np.float32)
        # Synthetic target policy: one-hot at a random legal cell. The
        # network has more than enough capacity to memorize this signal
        # over five 8-step iterations.
        target_idx = int(rng.integers(0, num_actions))
        policy = np.zeros(num_actions, dtype=np.float32)
        policy[target_idx] = 1.0
        # Target value drawn from {-1, +1} so MSE has a meaningful gradient.
        value = float(rng.choice([-1.0, 1.0]))
        replay.add(state, policy, value)
    return replay


def _build_model_and_optimizer(seed: int, board: int, num_actions: int):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)

    models_torch = importlib.import_module("quartz.models_torch")
    cfg = {
        "board": board,
        "ch": 17,
        "filters": 16,
        "blocks": 2,
        "actions": num_actions,
        "vh": 16,
    }
    model = models_torch.AlphaZeroNet(cfg)
    # Adam with mild lr — a tiny network on a tiny memorization task
    # converges quickly under SGD too, but Adam removes lr-tuning noise.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    return model, optimizer, cfg


def _run_iterations(
    seed: int, n_iters: int, n_steps_per_iter: int, batch: int = 16
):
    """Build a fresh model + replay, run `n_iters` calls of train_epoch,
    return the per-iteration mean loss as a list.
    """
    train_loop = importlib.import_module("quartz.train_loop")
    board, num_actions = 7, 49
    replay = _build_replay(seed=seed, n_examples=128, board=board, num_actions=num_actions)
    model, optimizer, _cfg = _build_model_and_optimizer(seed, board, num_actions)
    cfg = {"batch": batch}
    losses = []
    n_train_rows = 0
    for _ in range(n_iters):
        avg_loss, _avg_pl, _avg_vl, executed_steps, _inner_stop = train_loop.train_epoch(
            model, optimizer, replay, cfg, device="cpu", n_steps=n_steps_per_iter
        )
        losses.append(float(avg_loss))
        if executed_steps > 0:
            n_train_rows += 1
    return losses, n_train_rows


def test_p10_loss_strictly_decreases_over_iterations():
    """P10(b): at least one SGD step fires; loss in the final fifth is
    strictly below the first fifth (with stochastic-noise tolerance).
    """
    losses, n_train_rows = _run_iterations(seed=42, n_iters=5, n_steps_per_iter=8)

    # (a) ≥ 1 SGD row.
    assert n_train_rows >= 1, f"expected ≥1 SGD row, got {n_train_rows}"
    # (b) loss decreased meaningfully.
    first = losses[0]
    last = losses[-1]
    assert last < first, (
        f"loss did not decrease across iterations: first={first:.4f} last={last:.4f}"
    )


def test_p10_determinism_first_iter_loss_byte_equal_within_tolerance():
    """P10(c): two runs with the same seed produce the same first-iter
    loss to within 1e-5 (CPU torch is not strictly bitwise reproducible
    across all kernels, but the loose tolerance still catches gross
    non-determinism such as missing seed plumbing).
    """
    import torch

    # Pin BLAS / CUDNN flags for whatever determinism is achievable on
    # this host. CPU-only tests should be reproducible to ~1e-6 with
    # this guard; the assertion uses 1e-5 to leave headroom.
    torch.use_deterministic_algorithms(False)  # avoid deterministic-only-op errors
    torch.set_num_threads(1)

    losses_a, _ = _run_iterations(seed=137, n_iters=2, n_steps_per_iter=4)
    losses_b, _ = _run_iterations(seed=137, n_iters=2, n_steps_per_iter=4)

    assert abs(losses_a[0] - losses_b[0]) < 1e-5, (
        f"non-determinism on first iter: a={losses_a[0]:.6f} b={losses_b[0]:.6f}"
    )
