"""Tests for A01 canonical calibration script wrapper."""

from pathlib import Path
import pytest
from scripts.a01_empirical_calibration import main, reliability_diagram


def test_reliability_diagram_empty() -> None:
    res = reliability_diagram([], [])
    assert res["n"] == 0
    assert res["ece"] is None


def test_main_no_bundles_fails_closed(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_traces"
    empty_dir.mkdir()
    exit_code = main(["--trace-dir", str(empty_dir)])
    assert exit_code == 2  # Non-zero fail-closed exit
