"""Candidate, allocation, and search-backend skeletons (A06-A16, A25-A26)."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from hashlib import blake2b
from math import exp, log, sqrt
from random import Random
from typing import Mapping, Sequence

from .contracts import (
    AxisStatus,
    CostVector,
    EdgeObservation,
    MetaAction,
    MetaActionKind,
    MetaProposal,
    ProposalEstimate,
    RootObservation,
)


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    m = max(logits)
    weights = [exp(x - m) for x in logits]
    total = sum(weights)
    return [x / total for x in weights]


@dataclass
class A06GumbelSequentialHalving:
    axis_id: str = "A06.gumbel_sequential_halving"
    status: AxisStatus = AxisStatus.MECHANISM_VALID
    candidate_count: int = 16
    seed: int = 0

    def initial_candidates(self, obs: RootObservation) -> list[int]:
        rng = Random(self.seed ^ obs.root_hash ^ obs.iteration)
        scored = []
        for edge in obs.edges:
            prior = max(1e-12, edge.prior_anchor)
            u = min(1.0 - 1e-12, max(1e-12, rng.random()))
            gumbel = -log(-log(u))
            scored.append((log(prior) + gumbel, edge.edge_pos))
        scored.sort(reverse=True)
        return [pos for _, pos in scored[: max(1, min(self.candidate_count, len(scored)))]]

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        candidates = self.initial_candidates(obs)
        if not candidates:
            return ()
        per_arm = max(1, obs.remaining_visits // max(1, len(candidates)))
        return tuple(
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.SAMPLE, primary=pos, amount=per_arm, label="gumbel_round"),
                estimate=ProposalEstimate(
                    regret_reduction_mean=0.0,
                    regret_reduction_lcb=0.0,
                    confidence=0.25,
                    cost=CostVector(nn_evals=per_arm),
                ),
                activation_guard="root epoch fixed; without-replacement bracket; preserve tactical candidates",
                explanation=f"initial Gumbel/SH candidate edge={pos}",
                telemetry={"candidate_set": candidates},
            )
            for pos in candidates
        )


@dataclass
class A07ResidualEvidenceWidening:
    axis_id: str = "A07.residual_evidence_widening"
    status: AxisStatus = AxisStatus.SEED
    temperature: float = 0.25
    max_tail_mass: float = 0.05
    widen_count: int = 4

    def bound(self, obs: RootObservation) -> float:
        tau = max(1e-5, self.temperature)
        visible = [e for e in obs.edges if e.visible]
        hidden = [e for e in obs.edges if not e.visible]
        z_live = sum(max(1e-12, e.prior_anchor) * exp(e.q_mean / tau) for e in visible)
        z_out = sum(max(1e-12, e.prior_anchor) * exp(e.upper / tau) for e in hidden)
        denom = z_live + z_out
        return z_out / denom if denom > 0.0 else 1.0

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        residual = self.bound(obs)
        if residual <= self.max_tail_mass or obs.n_visible >= obs.n_children:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.WIDEN, amount=self.widen_count, label="residual_mass"),
                estimate=ProposalEstimate(
                    regret_reduction_mean=residual,
                    regret_reduction_lcb=0.0,
                    confidence=0.25,
                    cost=CostVector(nn_evals=self.widen_count),
                ),
                activation_guard="unmaterialized-action upper scores calibrated; edge set fresh",
                explanation=f"upper bound on outside posterior mass={residual:.4f}",
                telemetry={"residual_mass_bound": residual},
            ),
        )


@dataclass
class A08TacticalProofBackend:
    axis_id: str = "A08.tactical_proof_backend"
    status: AxisStatus = AxisStatus.CONDITIONAL
    proof_budget: int = 64

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        proposals = []
        for edge in obs.edges:
            flags = set(edge.tactical_flags)
            if "forced_win" in flags:
                proposals.append(
                    MetaProposal(
                        axis_id=self.axis_id,
                        action=MetaAction(MetaActionKind.STOP, primary=edge.edge_pos, label="tactical_forced"),
                        estimate=ProposalEstimate(confidence=1.0),
                        activation_guard="proof object verified by game-specific solver",
                        explanation="verified forced move overrides statistical controller",
                    )
                )
            elif flags & {"candidate_win", "forced_block", "high_threat"}:
                proposals.append(
                    MetaProposal(
                        axis_id=self.axis_id,
                        action=MetaAction(MetaActionKind.PROVE, primary=edge.edge_pos, amount=self.proof_budget),
                        estimate=ProposalEstimate(
                            regret_reduction_mean=edge.total_radius,
                            regret_reduction_lcb=0.0,
                            confidence=0.5,
                            cost=CostVector(cpu_ms=float(self.proof_budget)),
                        ),
                        activation_guard="game-specific lane; generic controller records provenance",
                        explanation=f"tactical proof request for edge {edge.edge_pos}",
                    )
                )
        return tuple(proposals)


@dataclass
class A09H3ChangePointRouter:
    axis_id: str = "A09.h3_change_point_router"
    status: AxisStatus = AxisStatus.SHADOW
    entropy_quantile: float = 0.9
    shrink_quantile: float = 0.9
    burst_visits: int = 16

    def score(self, obs: RootObservation) -> float:
        entropy_backflow = max(0.0, obs.entropy_slope)
        margin_shrink = max(0.0, -obs.margin_slope)
        return entropy_backflow * margin_shrink

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        score = self.score(obs)
        # Threshold must be learned from a reference distribution; zero is only
        # a wiring guard and intentionally does not claim useful calibration.
        if score <= 0.0:
            return ()
        best = obs.best_edge()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(
                    MetaActionKind.DEEPEN,
                    primary=None if best is None else best.edge_pos,
                    amount=self.burst_visits,
                    label="change_point_burst",
                ),
                estimate=ProposalEstimate(
                    regret_reduction_mean=score,
                    regret_reduction_lcb=0.0,
                    confidence=0.0,
                    cost=CostVector(nn_evals=self.burst_visits),
                ),
                activation_guard="threshold calibrated on external difficulty labels; not fixed zero floors",
                explanation=f"entropy-margin change-point score={score:.6f}",
            ),
        )


@dataclass
class A10PriorRefreshSpecialist:
    axis_id: str = "A10.prior_refresh_specialist"
    status: AxisStatus = AxisStatus.DORMANT
    divergence_threshold: float = 0.5
    max_blend: float = 0.2

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        if obs.prior_visit_js < self.divergence_threshold:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="conditional_refresh"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="OOD/weak-evaluator specialist only; frozen anchor retained; never recursive",
                explanation=f"prior-search divergence={obs.prior_visit_js:.4f}",
                telemetry={"max_blend": self.max_blend},
            ),
        )


@dataclass
class ParticleGroup:
    edge_pos: int
    active: bool = True
    hibernating: bool = False
    pulls: int = 0
    mean: float = 0.0
    radius: float = 1.0
    probability_best: float = 0.0


@dataclass
class A11DynamicLiveSetParticles:
    axis_id: str = "A11.dynamic_live_set_particles"
    status: AxisStatus = AxisStatus.SEED
    batch: int = 4
    resurrection_fraction: float = 0.05

    def weight(self, edge: EdgeObservation, best_lower: float) -> float:
        competition = max(0.0, edge.upper - best_lower)
        uncertainty = edge.total_radius
        return competition + uncertainty + 0.25 * max(0.0, 1.0 - edge.prior_anchor)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        best = obs.best_edge()
        best_lower = best.lower if best is not None else -1.0
        scored = sorted(
            ((self.weight(edge, best_lower), edge.edge_pos) for edge in obs.edges),
            reverse=True,
        )
        return tuple(
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.RESAMPLE_MODE, primary=pos, amount=self.batch),
                estimate=ProposalEstimate(
                    regret_reduction_mean=score,
                    regret_reduction_lcb=0.0,
                    confidence=0.0,
                    cost=CostVector(nn_evals=self.batch),
                ),
                activation_guard="independent particle groups; hibernation is reversible; mode quota enforced",
                explanation=f"live-set particle weight={score:.4f}",
            )
            for score, pos in scored[: max(1, min(8, len(scored)))]
        )


@dataclass
class A12JsdLocallyBalancedSampler:
    axis_id: str = "A12.jsd_locally_balanced_sampler"
    status: AxisStatus = AxisStatus.SEED
    temperature: float = 0.25
    bandwidth: float = 0.2

    @staticmethod
    def jsd(p: Sequence[float], q: Sequence[float]) -> float:
        if len(p) != len(q) or not p:
            raise ValueError("JSD inputs must have equal non-zero length")
        p_n = _softmax([log(max(1e-12, x)) for x in p])
        q_n = _softmax([log(max(1e-12, x)) for x in q])
        m = [(a + b) * 0.5 for a, b in zip(p_n, q_n)]
        kl_pm = sum(a * log(a / max(1e-12, c)) for a, c in zip(p_n, m))
        kl_qm = sum(b * log(b / max(1e-12, c)) for b, c in zip(q_n, m))
        return 0.5 * (kl_pm + kl_qm)

    def transition_rate(self, kernel: float, mu_from: float, mu_to: float) -> float:
        return max(0.0, kernel) * sqrt(max(1e-12, mu_to) / max(1e-12, mu_from))

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="build_sibling_geometry"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="sibling successor policy/value caches on common legal support",
                explanation="geometry build requested; target density remains P/Q regularized, not JSD penalty",
                telemetry={"temperature": self.temperature, "bandwidth": self.bandwidth},
            ),
        )


@dataclass
class A13PendingFlowWuUct:
    axis_id: str = "A13.pending_flow_wu_uct"
    status: AxisStatus = AxisStatus.CONDITIONAL
    reservation_penalty: float = 1.0

    def effective_visits(self, edge: EdgeObservation) -> float:
        return float(edge.visits + edge.pending + edge.virtual_visits)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        pending = sum(edge.pending for edge in obs.edges)
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="publish_unobserved_counts"),
                estimate=ProposalEstimate(confidence=0.5),
                activation_guard="pending/in-flight counts never enter confidence evidence",
                explanation=f"separate pending-flow correction for {pending} in-flight evaluations",
                telemetry={str(edge.edge_pos): self.effective_visits(edge) for edge in obs.edges},
            ),
        )


@dataclass
class A14SemanticPathLsh:
    axis_id: str = "A14.semantic_path_lsh"
    status: AxisStatus = AxisStatus.SHADOW
    min_threads: int = 8
    overlap_threshold: float = 0.5

    @staticmethod
    def signature(shingles: Sequence[str], seed: int = 0) -> str:
        h = blake2b(digest_size=16, person=f"qz{seed}".encode()[:16])
        for item in sorted(set(shingles)):
            h.update(item.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        rt = obs.runtime
        if rt.threads < self.min_threads or rt.semantic_path_overlap < self.overlap_threshold:
            return ()
        best = obs.best_edge()
        if best is None:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(
                    MetaActionKind.RESAMPLE_MODE,
                    primary=best.edge_pos,
                    amount=max(1, rt.inflight),
                    label="path_diversity",
                ),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="edge duplication already controlled; high-thread semantic overlap remains",
                explanation=f"semantic path overlap={rt.semantic_path_overlap:.3f}",
            ),
        )


@dataclass
class A15ServiceCurveScheduler:
    axis_id: str = "A15.service_curve_scheduler"
    status: AxisStatus = AxisStatus.MECHANISM_VALID
    table: Mapping[tuple[int, int], float] = field(default_factory=dict)

    def best(self) -> tuple[int, int] | None:
        return max(self.table, key=self.table.get, default=None)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        target = self.best()
        if target is None:
            return ()
        batch, inflight = target
        proposals = []
        if batch != obs.runtime.batch_size:
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    action=MetaAction(MetaActionKind.SET_BATCH, amount=batch),
                    estimate=ProposalEstimate(confidence=0.5),
                    activation_guard="hardware/profile-specific service-curve artifact matches runtime contract",
                    explanation=f"service-curve batch target={batch}",
                )
            )
        if inflight != obs.runtime.inflight:
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    action=MetaAction(MetaActionKind.SET_INFLIGHT, amount=inflight),
                    estimate=ProposalEstimate(confidence=0.5),
                    activation_guard="hardware/profile-specific service-curve artifact matches runtime contract",
                    explanation=f"service-curve inflight target={inflight}",
                )
            )
        return tuple(proposals)


@dataclass
class A16MonteCarloGraphSharing:
    axis_id: str = "A16.monte_carlo_graph_sharing"
    status: AxisStatus = AxisStatus.SEED

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        transpositions = obs.extras.get("shareable_transpositions", 0)
        state_key = obs.extras.get("shareable_state_key")
        if (
            not isinstance(transpositions, int)
            or isinstance(transpositions, bool)
            or transpositions <= 0
            or not isinstance(state_key, int)
            or isinstance(state_key, bool)
            or state_key < 0
            or state_key >= 2**64
        ):
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(
                    MetaActionKind.MERGE_OR_SHARE,
                    primary=state_key,
                    label="state_cache_only",
                ),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="state/eval cache sharing first; parent-edge N/W/Q remain parent-specific",
                explanation=f"shareable transposition states={transpositions}",
                telemetry={"shareable_transpositions": transpositions},
            ),
        )


@dataclass
class A25MentsSoftBackup:
    axis_id: str = "A25.ments_soft_backup"
    status: AxisStatus = AxisStatus.DORMANT
    temperature: float = 0.1

    def soft_value(self, q: Sequence[float], prior: Sequence[float]) -> float:
        if len(q) != len(prior) or not q:
            raise ValueError("q/prior shape mismatch")
        tau = max(1e-6, self.temperature)
        logits = [log(max(1e-12, p)) + value / tau for p, value in zip(prior, q)]
        m = max(logits)
        return tau * (m + log(sum(exp(x - m) for x in logits)))

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="soft_backup_ablation"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="opt-in shallow/root ablation; final game objective evaluated separately",
                explanation="maximum-entropy backup candidate; not a default minimax replacement",
            ),
        )


@dataclass
class A26NestedContourExactLab:
    axis_id: str = "A26.nested_contour_exact_lab"
    status: AxisStatus = AxisStatus.ANALYSIS_ONLY
    live_points: int = 32
    depth: int = 6

    @staticmethod
    def _finite_inputs(
        likelihoods: Sequence[float], prior: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        if len(likelihoods) != len(prior) or not likelihoods:
            raise ValueError("likelihood/prior shape mismatch")
        likelihood_rows = [float(value) for value in likelihoods]
        prior_rows = [float(value) for value in prior]
        if any(
            not math.isfinite(value) or value < 0.0 for value in likelihood_rows
        ):
            raise ValueError("likelihoods must be finite and non-negative")
        if any(not math.isfinite(value) or value < 0.0 for value in prior_rows):
            raise ValueError("prior must have non-negative finite mass")
        total = sum(prior_rows)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("prior must have positive finite mass")
        return likelihood_rows, [value / total for value in prior_rows]

    @classmethod
    def enumerated_evidence(
        cls, likelihoods: Sequence[float], prior: Sequence[float]
    ) -> float:
        """Exact finite-state evidence by direct enumeration."""

        likelihood_rows, prior_rows = cls._finite_inputs(likelihoods, prior)
        return sum(value * mass for value, mass in zip(likelihood_rows, prior_rows))

    @classmethod
    def finite_contour_evidence(
        cls, likelihoods: Sequence[float], prior: Sequence[float]
    ) -> float:
        """Exact layer-cake/contour form of the same finite-state evidence."""

        likelihood_rows, prior_rows = cls._finite_inputs(likelihoods, prior)
        levels = sorted(set(likelihood_rows))
        evidence = 0.0
        previous = 0.0
        for level in levels:
            surviving_mass = sum(
                mass
                for likelihood, mass in zip(likelihood_rows, prior_rows)
                if likelihood >= level
            )
            evidence += (level - previous) * surviving_mass
            previous = level
        return evidence

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="offline_exact_nested_contour"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="enumerable/small-tree lab; explicit prior, likelihood, and constrained sampler",
                explanation="exact nested-contour validation remains separate from live-set particle search",
                telemetry={"live_points": self.live_points, "depth": self.depth},
            ),
        )
