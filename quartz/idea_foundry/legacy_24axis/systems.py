"""Regime routing, runtime scheduling, and parallel-search skeletons.

Axes covered:
A11 entropy/margin change-point router
A12 evaluator service-curve scheduler
A13 pending-flow / WU-UCT observer
A14 semantic whole-path LSH diversity controller
A23 graph-state sharing consistency observer
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .contracts import MetaActionKind, MetaCost, MetaProposal, RootSnapshot


@dataclass(frozen=True)
class EntropyMarginRegimeRouterSkeleton:
    """A11: replace the dead binary H3 burst gate with a continuous score."""

    axis_id: str = "A11_entropy_margin_regime_router"
    entropy_growth_scale: float = 0.02
    margin_shrink_scale: float = 0.02
    trigger_score: float = 1.0
    extra_batch: int = 16

    def score(self, snapshot: RootSnapshot) -> float:
        entropy_component = max(0.0, snapshot.entropy_slope) / max(
            self.entropy_growth_scale, 1e-9
        )
        margin_component = max(0.0, -snapshot.margin_slope) / max(
            self.margin_shrink_scale, 1e-9
        )
        revision_component = min(snapshot.revision_count / 4.0, 1.0)
        return entropy_component + margin_component + revision_component

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        score = self.score(snapshot)
        if score < self.trigger_score:
            return []
        best = snapshot.best_action
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.DEEPEN,
                target_action_ids=(() if best is None else (best.action_id,)),
                amount=self.extra_batch,
                cost=MetaCost(nn_evals=float(self.extra_batch)),
                activation_guard="continuous change-point score; threshold calibrated on forked-VOC labels",
                explanation="Allocate an extra continuation chunk when entropy rises while the margin contracts.",
                telemetry={"regime_score": score},
            )
        ]


@dataclass(frozen=True)
class ServiceCurvePoint:
    batch_size: int
    inflight: int
    threads: int
    items_per_s: float
    p95_latency_ms: float
    queue_wait_ms: float = 0.0
    energy_per_item: float | None = None


@dataclass(frozen=True)
class ServiceCurveSchedulerSkeleton:
    """A12: measured service-curve scheduler for batch/inflight/thread settings."""

    axis_id: str = "A12_service_curve_scheduler"
    latency_cap_ms: float = 100.0
    energy_weight: float = 0.0

    def choose(self, points: Sequence[ServiceCurvePoint]) -> ServiceCurvePoint | None:
        feasible = [p for p in points if p.p95_latency_ms <= self.latency_cap_ms]
        if not feasible:
            return min(points, key=lambda p: p.p95_latency_ms, default=None)

        def utility(point: ServiceCurvePoint) -> float:
            energy = 0.0 if point.energy_per_item is None else point.energy_per_item
            return point.items_per_s - self.energy_weight * energy - point.queue_wait_ms

        return max(feasible, key=utility, default=None)

    def propose_from_points(
        self, points: Sequence[ServiceCurvePoint]
    ) -> Sequence[MetaProposal]:
        selected = self.choose(points)
        if selected is None:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SET_BATCH,
                amount=selected.batch_size,
                confidence=1.0,
                activation_guard="hardware-profile-specific measured service curve",
                explanation="Select the highest-throughput configuration under the p95 latency contract.",
                telemetry={
                    "batch_size": selected.batch_size,
                    "inflight": selected.inflight,
                    "threads": selected.threads,
                    "items_per_s": selected.items_per_s,
                    "p95_latency_ms": selected.p95_latency_ms,
                },
            ),
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SET_INFLIGHT,
                amount=selected.inflight,
                confidence=1.0,
                activation_guard="paired with selected batch setting",
            ),
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SET_THREADS,
                amount=selected.threads,
                confidence=1.0,
                activation_guard="single-position path only until multi-session fairness contract exists",
            ),
        ]

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        return []


@dataclass(frozen=True)
class PendingFlowWuUctSkeleton:
    """A13: expose incomplete simulations as evidence-free pending counts."""

    axis_id: str = "A13_pending_flow_wu_uct"
    pending_penalty: float = 1.0
    collision_threshold: float = 0.35

    def pending_scores(self, snapshot: RootSnapshot) -> Mapping[int, float]:
        return {
            action.action_id: self.pending_penalty
            * action.pending
            / max(1, action.visits + action.pending)
            for action in snapshot.visible_actions
        }

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        penalties = self.pending_scores(snapshot)
        if snapshot.runtime.edge_duplicate_rate < self.collision_threshold:
            return [
                MetaProposal(
                    axis_id=self.axis_id,
                    kind=MetaActionKind.SHADOW_ONLY,
                    activation_guard="pending counts never enter certificates",
                    explanation="Record WU-UCT-style unobserved-sample pressure without changing the policy.",
                    telemetry={"pending_penalties": dict(penalties)},
                )
            ]
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SET_THREADS,
                amount=max(1, snapshot.runtime.threads - 1),
                activation_guard="high duplicate rate after pending-aware score shaping",
                explanation="Reduce worker pressure when incomplete simulations dominate root contention.",
                telemetry={
                    "pending_penalties": dict(penalties),
                    "duplicate_rate": snapshot.runtime.edge_duplicate_rate,
                },
            )
        ]


def _hash64(value: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class SemanticPathLshSkeleton:
    """A14: whole-path semantic overlap, not the rejected edge-duplication claim."""

    axis_id: str = "A14_semantic_path_lsh"
    n_hashes: int = 16
    overlap_threshold: float = 0.65
    min_threads: int = 8

    def minhash(self, path: Sequence[int]) -> tuple[int, ...]:
        shingles = [f"{a}:{b}".encode("utf-8") for a, b in zip(path, path[1:])] or [
            b"<empty>"
        ]
        signature = []
        for seed in range(self.n_hashes):
            signature.append(
                min(_hash64(seed.to_bytes(4, "big") + shingle) for shingle in shingles)
            )
        return tuple(signature)

    @staticmethod
    def estimated_similarity(lhs: Sequence[int], rhs: Sequence[int]) -> float:
        if len(lhs) != len(rhs) or not lhs:
            return 0.0
        return float(sum(int(a == b) for a, b in zip(lhs, rhs)) / len(lhs))

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        if snapshot.runtime.threads < self.min_threads:
            return []
        if snapshot.runtime.semantic_path_overlap < self.overlap_threshold:
            return []
        targets = tuple(
            action.action_id
            for action in sorted(
                snapshot.visible_actions,
                key=lambda a: (a.upper, -a.edge_pos),
                reverse=True,
            )
            if action.path_signature
        )
        if not targets:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.CHALLENGE,
                target_action_ids=targets,
                amount=min(len(targets), snapshot.runtime.threads),
                cost=MetaCost(cpu_ms=0.05 * len(targets)),
                activation_guard="high-thread && edge duplication already controlled && semantic overlap remains high",
                explanation="Diversify in-flight root trajectories across distinct path-signature clusters.",
                telemetry={
                    "semantic_path_overlap": snapshot.runtime.semantic_path_overlap
                },
            )
        ]


@dataclass(frozen=True)
class GraphStateSharingConsistencySkeleton:
    """A23: state/evaluation sharing observer before graph-statistic merging."""

    axis_id: str = "A23_graph_state_sharing_consistency"
    max_value_disagreement: float = 0.05

    def analyze_occurrences(
        self, occurrences: Iterable[Mapping[str, float | int | str]]
    ) -> Mapping[str, float | bool]:
        values = [float(row["value"]) for row in occurrences if "value" in row]
        if not values:
            return {"n": 0.0, "spread": 0.0, "merge_safe": False}
        spread = max(values) - min(values)
        return {
            "n": float(len(values)),
            "spread": float(spread),
            "merge_safe": bool(spread <= self.max_value_disagreement),
        }

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                activation_guard="share state/evaluation cache first; parent-edge statistics remain separate",
                explanation="Measure path-dependent disagreement before any MCGS-style statistic merge.",
            )
        ]
