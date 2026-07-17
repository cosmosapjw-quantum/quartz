"""Experimental contracts and non-production skeletons for the QUARTZ idea foundry.

Nothing in this package is wired into the Rust search loop by import side effect.
Promotion requires an explicit Phase-15 system, an idea-lab lane, tests, and a
claim-ledger entry.
"""

from .contracts import (
    ActionEvidence,
    AnalysisResult,
    AxisSpec,
    AxisState,
    ExecutionPlane,
    MetaActionKind,
    MetaCost,
    MetaProposal,
    RootSnapshot,
    RuntimeEvidence,
)
from .registry import AXIS_SPECS, get_axis_spec

__all__ = [
    "ActionEvidence",
    "AnalysisResult",
    "AxisSpec",
    "AxisState",
    "ExecutionPlane",
    "MetaActionKind",
    "MetaCost",
    "MetaProposal",
    "RootSnapshot",
    "RuntimeEvidence",
    "AXIS_SPECS",
    "get_axis_spec",
]
