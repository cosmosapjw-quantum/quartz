#!/usr/bin/env python3
"""A01 Canonical Empirical Flip-Risk Calibration Wrapper.

Authoritative substrate is scripts/phase15_flip_calibration.py (Stage 7 / C8).
This script provides the unified A01 CLI interface with fail-closed semantics:
- Missing labels or unverified traces fail closed (exit non-zero)
- Paired bootstrap CI and stratified continuation/restart handling
- Evaluates ECE, Brier, and matched realized budget calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase15_flip_calibration import (
    main as phase15_main,
    analyze,
    load_bundles,
    reliability_diagram,
)  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return phase15_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
