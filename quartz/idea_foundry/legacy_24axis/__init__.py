"""Deprecated 24-axis compatibility snapshot for the QUARTZ idea foundry.

The controlling registry is the top-level 26-axis package. Nothing here is
wired into the Rust search loop, and passing its retained tests cannot promote
any current axis or scientific claim.
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
