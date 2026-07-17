"""Common contracts for QUARTZ idea-foundry experiment skeletons.

The production Rust engine owns live tree mutation.  This Python package mirrors
only the immutable root snapshot and proposal surfaces needed by Phase-15
replay, synthetic screens, counterfactual teachers, and offline calibration.

The names intentionally track ``src/mcts/policy/trait_def.rs``:
``RootSnapshot`` extends the existing ``SearchSnapshot`` concept and
``ActionEvidence`` extends ``EdgeView`` with fields that currently live only in
analysis artifacts or planned telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

FOUNDRY_SCHEMA_VERSION = 1


class AxisState(str, Enum):
    """Lifecycle state of one experiment axis."""

    SEED = "seed"
    MECHANISM_VALID = "mechanism_valid"
    SHADOW = "shadow"
    CONDITIONAL = "conditional"
    ACTIVE_EXPERIMENTAL = "active_experimental"
    DEPLOYMENT_CANDIDATE = "deployment_candidate"
    DORMANT = "dormant"
    ANALYSIS_ONLY = "analysis_only"


class ExecutionPlane(str, Enum):
    RUST_ONLINE = "rust_online"
    PYTHON_ONLINE = "python_online"
    POSTHOC = "posthoc"
    TRAINING = "training"
    ANALYSIS = "analysis"


class MetaActionKind(str, Enum):
    SAMPLE = "sample"
    CHALLENGE = "challenge"
    WIDEN = "widen"
    DEEPEN = "deepen"
    PROVE = "prove"
    REWEIGHT_POLICY = "reweight_policy"
    MERGE_OR_SHARE = "merge_or_share"
    SET_BATCH = "set_batch"
    SET_INFLIGHT = "set_inflight"
    SET_THREADS = "set_threads"
    ARCHIVE_STATE = "archive_state"
    STOP = "stop"
    SHADOW_ONLY = "shadow_only"


@dataclass(frozen=True)
class ActionEvidence:
    """Read-only action/edge evidence at a root checkpoint.

    ``edge_pos`` is the dense index into a published root edge slab.
    ``action_id`` is the game action encoding.  They must never be conflated.
    """

    edge_pos: int
    action_id: int
    visible: bool
    prior_anchor: float
    current_prior: float
    visits: int
    virtual_visits: int = 0
    pending: int = 0
    mean_q: float = 0.0
    value_sum: float = 0.0
    m2: float = 0.0
    mc_radius: float = 0.0
    epistemic_radius: float = 0.0
    drift_radius: float = 0.0
    bias_radius: float = 0.0
    upper_hint: float | None = None
    cost_ms: float = 1.0
    tactical_flags: tuple[str, ...] = ()
    policy_signature: tuple[float, ...] = ()
    path_signature: tuple[int, ...] = ()

    @property
    def total_radius(self) -> float:
        """Conservative default: add uncertainty channels without independence."""

        return max(0.0, self.mc_radius) + max(0.0, self.epistemic_radius) + max(
            0.0, self.drift_radius
        ) + max(0.0, self.bias_radius)

    @property
    def lower(self) -> float:
        return self.mean_q - self.total_radius

    @property
    def upper(self) -> float:
        explicit = self.upper_hint
        if explicit is not None and np.isfinite(explicit):
            return float(explicit)
        return self.mean_q + self.total_radius


@dataclass(frozen=True)
class RuntimeEvidence:
    threads: int = 1
    batch_size: int = 1
    inflight: int = 1
    queue_wait_ms: float = 0.0
    eval_latency_ms: float = 0.0
    items_per_s: float = 0.0
    edge_duplicate_rate: float = 0.0
    semantic_path_overlap: float = 0.0
    tt_wait_ms: float = 0.0
    nps: float = 0.0


@dataclass(frozen=True)
class RootSnapshot:
    """Immutable root checkpoint consumed by every foundry operator."""

    root_hash: str
    checkpoint_id: str
    position_id: str
    search_epoch: int
    root_visits: int
    iteration: int
    elapsed_ms: float
    remaining_visits: int | None
    actions: tuple[ActionEvidence, ...]
    runtime: RuntimeEvidence = field(default_factory=RuntimeEvidence)
    policy_entropy: float = 0.0
    effective_branching: float = 1.0
    top2_margin: float = 0.0
    margin_slope: float = 0.0
    entropy_slope: float = 0.0
    h1_stability: float | None = None
    p_flip: float | None = None
    prior_visit_js: float = 0.0
    candidate_omission_risk: float = 0.0
    revision_count: int = 0
    fresh: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        positions = [a.edge_pos for a in self.actions]
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate edge_pos in RootSnapshot")
        if any(a.visits < 0 or a.pending < 0 or a.virtual_visits < 0 for a in self.actions):
            raise ValueError("negative count in RootSnapshot")
        if self.root_visits < 0 or self.iteration < 0:
            raise ValueError("negative root clock")
        if not 0.0 <= self.candidate_omission_risk <= 1.0:
            raise ValueError("candidate_omission_risk must be in [0,1]")

    @property
    def visible_actions(self) -> tuple[ActionEvidence, ...]:
        return tuple(a for a in self.actions if a.visible)

    @property
    def best_action(self) -> ActionEvidence | None:
        visible = self.visible_actions
        return max(visible, key=lambda a: (a.mean_q, a.visits, -a.edge_pos), default=None)

    @property
    def best_lower_action(self) -> ActionEvidence | None:
        visible = self.visible_actions
        return max(visible, key=lambda a: (a.lower, a.visits, -a.edge_pos), default=None)


@dataclass(frozen=True)
class MetaCost:
    nn_evals: float = 0.0
    cpu_ms: float = 0.0
    gpu_ms: float = 0.0
    energy_proxy: float = 0.0

    @property
    def scalar(self) -> float:
        # The arbiter must replace this provisional unit mix with a calibrated
        # shadow-price model before any claim-bearing online use.
        return self.nn_evals + self.cpu_ms + self.gpu_ms + self.energy_proxy


@dataclass(frozen=True)
class MetaProposal:
    axis_id: str
    kind: MetaActionKind
    target_action_ids: tuple[int, ...] = ()
    amount: int = 0
    expected_regret_reduction: float = 0.0
    regret_reduction_lcb: float = 0.0
    cost: MetaCost = field(default_factory=MetaCost)
    confidence: float = 0.0
    activation_guard: str = ""
    explanation: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def conservative_utility(self) -> float:
        return self.regret_reduction_lcb - self.cost.scalar


@dataclass(frozen=True)
class AxisSpec:
    axis_id: str
    title: str
    state: AxisState
    plane: ExecutionPlane
    python_symbol: str | None
    rust_symbol: str | None
    baselines: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    blockers: tuple[str, ...] = ()
    negative_scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisResult:
    axis_id: str
    metrics: Mapping[str, float | int | str | bool | None]
    notes: tuple[str, ...] = ()


class AxisOperator(Protocol):
    axis_id: str

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        """Return zero or more immutable proposals; never mutate the tree."""


def normalize_prob(values: Sequence[float], *, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, floor)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return arr / total


def stable_softmax(logits: Sequence[float]) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.where(np.isfinite(arr), arr, -np.inf)
    max_value = float(np.max(finite))
    if not np.isfinite(max_value):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    weights = np.exp(np.clip(finite - max_value, -80.0, 0.0))
    return normalize_prob(weights)


def shannon_entropy(policy: Sequence[float]) -> float:
    p = np.clip(normalize_prob(policy), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def jensen_shannon(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    p = np.clip(normalize_prob(lhs), 1e-12, 1.0)
    q = np.clip(normalize_prob(rhs), 1e-12, 1.0)
    if p.shape != q.shape:
        raise ValueError("JSD inputs must have identical support")
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(q * (np.log(q) - np.log(m))))
