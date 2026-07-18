"""Controller regression tests against pinned fixture positions.

Consumes `tests/fixtures/regression_positions.json` (previously defined but
unused) to pin controller behavior on two canonical Gomoku-15 positions:

- Position 1: center stone; expected best move within distance 2 of (7,7)
- Position 2: open four-in-a-row; white MUST block at index 108 or 113

The Rust search command used here (`search_gomoku15`) evaluates with a
rollout-only heuristic (`ShortRollout::new(12)`), not a trained NN. At
realistic budgets the rollout evaluator is too weak to reliably discover
the fixture's expected moves on 15×15 — it can return corner-ish noise.

So this file runs two tiers:

- Tier A — always on. Asserts the JSON protocol is alive, every penalty
  mode produces a legal move, and the halt returns a finite p_flip. This
  catches gross Rust-side breakage (protocol, QuartzConfig parsing,
  search loop) without requiring a trained NN.

- Tier B — gated behind `QUARTZ_RUN_CONTROLLER_REGRESSION_STRICT=1`.
  Asserts the fixture-level move expectations. Intended for CI jobs or
  local runs that have a better evaluator wired into the server, or for
  future work that extends `search_gomoku15` to accept an ONNX NN.

The test is skipped entirely when the Rust release binary is absent so
local dev environments without a release build do not see red tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "regression_positions.json"
RUST_BINARY = REPO_ROOT / "target" / "release" / "mcts_demo"

# Gate for the strict fixture-level checks; see module docstring.
STRICT_MODE = os.environ.get("QUARTZ_RUN_CONTROLLER_REGRESSION_STRICT", "").lower() in {
    "1",
    "true",
    "yes",
}

# Penalty modes that gate arena outcomes in the current controller_axes
# preset. A regression in any of them invalidates the attribution
# comparison, so we run the protocol+sanity check across all three.
PENALTY_MODES = ["GatedRefresh", "GatedRefreshLegacy", "Legacy"]


def _load_positions() -> list[dict]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return list(payload.get("positions") or [])


def _server_alive() -> bool:
    if not RUST_BINARY.exists():
        return False
    if shutil.which(str(RUST_BINARY)) is None and not os.access(
        str(RUST_BINARY), os.X_OK
    ):
        return False
    return True


def _ask_server(request: dict, timeout: float = 60.0) -> dict:
    """Spawn the Rust server, send one JSON command, read one JSON line."""
    proc = subprocess.Popen(
        [str(RUST_BINARY), "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, "rust server produced no response"
        return json.loads(line.decode("utf-8"))
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.write(b'{"cmd":"quit"}\n')
                proc.stdin.flush()
            except Exception:
                pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
            proc.wait(timeout=5.0)


def _move_in_center_neighborhood(
    move_idx: int, center=(7, 7), distance: int = 2
) -> bool:
    size = 15
    row, col = divmod(int(move_idx), size)
    return abs(row - center[0]) <= distance and abs(col - center[1]) <= distance


@pytest.fixture(scope="module")
def positions() -> list[dict]:
    return _load_positions()


def test_fixture_has_expected_positions(positions):
    """Fixture sanity — runs even without the Rust binary."""
    assert len(positions) >= 2, "fixture should carry at least 2 canonical positions"
    assert positions[0]["game"] == "gomoku15_std"
    assert positions[1]["game"] == "gomoku15_std"
    assert "p_flip_band" in positions[0]
    assert "expected_moves" in positions[1]


# ─────────────────────────────────────────────
# Tier A — protocol + sanity, always on
# ─────────────────────────────────────────────


@pytest.mark.skipif(not _server_alive(), reason="Rust release binary not built")
@pytest.mark.parametrize("penalty_mode", PENALTY_MODES)
def test_protocol_search_returns_legal_move(positions, penalty_mode):
    """Each penalty mode must return a legal move on both positions.

    Catches gross Rust-side breakage (JSON parsing, QuartzConfig wiring,
    search loop) without depending on evaluator quality.
    """
    for pos in positions:
        resp = _ask_server(
            {
                "cmd": "search",
                "game": "gomoku15_std",
                "board": pos["board"],
                "player": pos["player"],
                "iters": int(pos["budget"]),
                "penalty_mode": penalty_mode,
            }
        )
        assert "error" not in resp, f"mode={penalty_mode} err: {resp}"
        move_idx = int(resp["move"])
        assert 0 <= move_idx < 225, f"mode={penalty_mode} illegal move idx={move_idx}"
        # A legal move on a non-empty board must be on an empty cell.
        assert pos["board"][move_idx] == 0, (
            f"mode={penalty_mode} returned occupied cell idx={move_idx}"
        )
        p_flip = resp.get("p_flip")
        assert p_flip is not None, f"mode={penalty_mode} missing p_flip"
        p_flip = float(p_flip)
        # p_flip is a probability; must be in [0, 1].
        assert 0.0 <= p_flip <= 1.0, (
            f"mode={penalty_mode} p_flip out of [0,1]: {p_flip}"
        )
        # iters reported must be positive and <= requested budget (+ small slack).
        it = int(resp.get("iters") or 0)
        assert 0 < it <= int(pos["budget"]) + 64


# ─────────────────────────────────────────────
# Tier B — fixture-level expectations, env-gated
# ─────────────────────────────────────────────


@pytest.mark.skipif(not _server_alive(), reason="Rust release binary not built")
@pytest.mark.skipif(
    not STRICT_MODE,
    reason="strict fixture expectations require a trained NN evaluator; "
    "set QUARTZ_RUN_CONTROLLER_REGRESSION_STRICT=1 to enable",
)
@pytest.mark.parametrize("penalty_mode", PENALTY_MODES)
def test_center_stone_prefers_neighborhood(positions, penalty_mode):
    pos = positions[0]  # center stone, white to move
    resp = _ask_server(
        {
            "cmd": "search",
            "game": "gomoku15_std",
            "board": pos["board"],
            "player": pos["player"],
            "iters": int(pos["budget"]),
            "penalty_mode": penalty_mode,
        }
    )
    assert "error" not in resp, resp
    move_idx = int(resp["move"])
    assert _move_in_center_neighborhood(move_idx, center=(7, 7), distance=2), (
        f"mode={penalty_mode} picked non-neighborhood move: idx={move_idx} "
        f"({move_idx // 15},{move_idx % 15})"
    )


@pytest.mark.skipif(not _server_alive(), reason="Rust release binary not built")
@pytest.mark.skipif(
    not STRICT_MODE,
    reason="strict fixture expectations require a trained NN evaluator; "
    "set QUARTZ_RUN_CONTROLLER_REGRESSION_STRICT=1 to enable",
)
@pytest.mark.parametrize("penalty_mode", PENALTY_MODES)
def test_open_four_must_block(positions, penalty_mode):
    pos = positions[1]  # open-four; MUST block
    resp = _ask_server(
        {
            "cmd": "search",
            "game": "gomoku15_std",
            "board": pos["board"],
            "player": pos["player"],
            "iters": int(pos["budget"]),
            "penalty_mode": penalty_mode,
        }
    )
    assert "error" not in resp, resp
    move_idx = int(resp["move"])
    expected = set(pos["expected_moves"])
    assert move_idx in expected, (
        f"mode={penalty_mode} failed the forced-block test: picked "
        f"idx={move_idx} ({move_idx // 15},{move_idx % 15}); expected one of {sorted(expected)}"
    )
