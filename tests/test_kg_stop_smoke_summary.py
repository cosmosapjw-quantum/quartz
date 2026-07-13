"""Tests for scripts/kg_stop_engine_smoke.py::summarize_kg_smoke (Stage 7 C4)."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = REPO_ROOT / "scripts" / "kg_stop_engine_smoke.py"
    spec = importlib.util.spec_from_file_location("kg_stop_engine_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(pos, budget, thr, kg_iters, kg_mv, fx_mv):
    return {
        "position_id": pos,
        "budget": budget,
        "threshold": thr,
        "kg_iterations": kg_iters,
        "kg_best_move": kg_mv,
        "kg_halted": kg_iters < budget,
        "fixed_iterations": budget,
        "fixed_best_move": fx_mv,
    }


def test_success_grid_cell_detected():
    m = _load()
    # budget 64, kg halts at 32 (50% saved), same best move as fixed => success
    rows = [_row(f"p{i}", 64, 1e-3, 32, 5, 5) for i in range(10)]
    s = m.summarize_kg_smoke(rows)
    assert s["kill_no_halts"] is False
    assert s["success"] is True
    cell = s["per_cell"][0]
    assert cell["halt_rate"] == 1.0
    assert abs(cell["mean_budget_saved_pct"] - 0.5) < 1e-9
    assert cell["top1_agreement"] == 1.0


def test_kill_when_no_halts():
    m = _load()
    # kg never halts (ran full budget) => kill
    rows = [_row(f"p{i}", 128, 1e-4, 128, 3, 3) for i in range(8)]
    s = m.summarize_kg_smoke(rows)
    assert s["kill_no_halts"] is True
    assert s["success"] is False
    assert s["per_cell"][0]["halt_rate"] == 0.0


def test_demote_anti_conservative():
    m = _load()
    # halts a lot (saves budget) but disagrees with fixed most of the time
    rows = [_row(f"p{i}", 64, 1e-2, 20, 1, 2) for i in range(10)]  # all disagree
    s = m.summarize_kg_smoke(rows)
    assert s["kill_no_halts"] is False
    assert s["success"] is False
    assert s["demote_anti_conservative"] is True
    assert s["per_cell"][0]["top1_agreement"] == 0.0


def test_per_cell_grouping_and_best_cell():
    m = _load()
    rows = (
        [_row(f"a{i}", 64, 1e-3, 48, 4, 4) for i in range(5)]  # 25% saved, agree
        + [_row(f"b{i}", 128, 1e-3, 40, 7, 7) for i in range(5)]  # ~69% saved, agree
    )
    s = m.summarize_kg_smoke(rows)
    assert len(s["per_cell"]) == 2
    # best cell = highest mean saved (the 128-budget cell)
    assert s["best_cell"]["budget"] == 128
    assert s["success"] is True  # 128 cell: ~69% saved, agreement 1.0
