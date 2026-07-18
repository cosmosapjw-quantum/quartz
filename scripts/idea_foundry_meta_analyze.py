#!/usr/bin/env python3
"""Pool only compatible, independent Idea Foundry effect records."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.idea_foundry.meta_analysis import meta_analysis_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(meta_analysis_main())
