"""Shared contracts for QUARTZ idea-foundry experiment skeletons.

These types intentionally sit above the live Rust ``SearchPolicy`` surface.  A
module reads a frozen root observation and emits a *proposal*; it does not
silently mutate PUCT.  The Rust arbiter may later translate an accepted
proposal into a score cache, a root-session action, or a training job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class AxisStatus(str, Enum):
    SEED = "seed"
    MECHANISM_VALID = "mechanism_valid"
    SHADOW = "shadow"
    CONDITIONAL = "conditional"
    ACTIVE_EXPERIMENTAL = "active_experimental"
    DEPLOYMENT_CANDIDATE = "deployment_candidate"
    DORMANT = "dormant"
    ANALYSIS_ONLY = "analysis_only"


class MetaActionKind(str, Enum):
    STOP = "stop"
    SAMPLE = "sample"
    CHALLENGE = "challenge"
    WIDEN = "widen"
    DEEPEN = "deepen"
    PROVE = "prove"
    RESAMPLE_MODE = "resample_mode"
    MERGE_OR_SHARE = "merge_or_share"
    SET_BATCH = "set_batch"
    SET_INFLIGHT = "set_inflight"
    SET_THREADS = "set_threads"
    REANALYSE = "reanalyse"
    ARCHIVE_STATE = "archive_state"
    NOOP = "noop"


@dataclass(frozen=True)
class CostVector:
    nn_evals: float = 0.0
    cpu_ms: float = 0.0
    gpu_ms: float = 0.0
    energy_proxy: float = 0.0

    def weighted(self, prices: Mapping[str, float]) -> float:
        return (
            self.nn_evals * float(prices.get("nn_evals", 0.0))
            + self.cpu_ms * float(prices.get("cpu_ms", 0.0))
            + self.gpu_ms * float(prices.get("gpu_ms", 0.0))
            + self.energy_proxy * float(prices.get("energy_proxy", 0.0))
        )


@dataclass(frozen=True)
class MetaAction:
    kind: MetaActionKind
    primary: int | None = None
    secondary: int | None = None
    amount: int = 0
    value: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class ProposalEstimate:
    regret_reduction_mean: float = 0.0
    regret_reduction_lcb: float = 0.0
    confidence: float = 0.0
    cost: CostVector = field(default_factory=CostVector)

    def net_lcb(self, prices: Mapping[str, float]) -> float:
        return self.regret_reduction_lcb - self.cost.weighted(prices)


@dataclass(frozen=True)
class MetaProposal:
    axis_id: str
    action: MetaAction
    estimate: ProposalEstimate
    activation_guard: str
    explanation: str
    evidence_scope: str = "skeleton_only"
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeObservation:
    edge_pos: int
    action_id: int
    visible: bool
    prior_anchor: float
    prior_current: float
    visits: int
    virtual_visits: int
    pending: int
    q_mean: float
    q_sum: float
    m2: float
    last_value: float = 0.0
    mc_radius: float = 0.0
    epistemic_radius: float = 0.0
    drift_radius: float = 0.0
    bias_radius: float = 0.0
    lower: float = -1.0
    upper: float = 1.0
    tactical_flags: tuple[str, ...] = ()

    @property
    def total_radius(self) -> float:
        # Conservative default until covariance calibration is earned.
        return (
            max(0.0, self.mc_radius)
            + max(0.0, self.epistemic_radius)
            + max(0.0, self.drift_radius)
            + max(0.0, self.bias_radius)
        )


@dataclass(frozen=True)
class RuntimeObservation:
    threads: int = 1
    batch_size: int = 1
    inflight: int = 1
    queue_wait_ms: float = 0.0
    eval_latency_ms: float = 0.0
    nps: float = 0.0
    edge_duplicate_rate: float = 0.0
    semantic_path_overlap: float = 0.0
    max_pending: int = 0
    tt_wait_ns: int = 0


@dataclass(frozen=True)
class RootObservation:
    root_hash: int
    checkpoint_id: str
    position_id: str
    game: str
    root_visits: int
    iteration: int
    elapsed_ms: int
    remaining_visits: int
    n_children: int
    n_visible: int
    entropy: float
    effective_branching: float
    top2_margin: float
    margin_slope: float
    entropy_slope: float
    h1_stability: float | None
    p_flip: float | None
    prior_visit_js: float
    candidate_omission_bound: float
    revision_count: int
    edges: tuple[EdgeObservation, ...]
    runtime: RuntimeObservation = field(default_factory=RuntimeObservation)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def best_edge(self) -> EdgeObservation | None:
        return max(self.edges, key=lambda edge: edge.q_mean, default=None)

    def runner_up(self) -> EdgeObservation | None:
        ranked = sorted(self.edges, key=lambda edge: edge.q_mean, reverse=True)
        return ranked[1] if len(ranked) > 1 else None


@dataclass(frozen=True)
class CounterfactualLabel:
    checkpoint_id: str
    position_id: str
    budget: int
    action: MetaAction
    decision_loss_before: float
    decision_loss_after: float
    realized_cost: CostVector
    oracle_action_id: int | None = None

    @property
    def regret_reduction(self) -> float:
        return self.decision_loss_before - self.decision_loss_after


class FoundryAxis(Protocol):
    axis_id: str
    status: AxisStatus

    def propose(self, observation: RootObservation) -> Sequence[MetaProposal]:
        """Return zero or more proposals without mutating the observation."""


class ProposalRanker(Protocol):
    def choose(
        self,
        proposals: Sequence[MetaProposal],
        prices: Mapping[str, float],
    ) -> MetaProposal | None:
        """Choose the highest conservative net-value proposal."""


class ConservativeArbiter:
    """Minimal reference arbiter used by trace-replay experiments.

    Live-engine integration should keep the same ordering but add freshness,
    budget, tactical, and runtime guards on the Rust side.
    """

    def choose(
        self,
        proposals: Sequence[MetaProposal],
        prices: Mapping[str, float],
    ) -> MetaProposal | None:
        eligible = [p for p in proposals if p.estimate.confidence >= 0.0]
        if not eligible:
            return None
        best = max(eligible, key=lambda p: p.estimate.net_lcb(prices))
        if best.action.kind is MetaActionKind.STOP:
            return best
        return best if best.estimate.net_lcb(prices) > 0.0 else None
