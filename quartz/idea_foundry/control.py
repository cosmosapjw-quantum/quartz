"""Control-plane experiment skeletons (A01-A05, A24).

Every class is deliberately usable in trace replay before live-engine wiring.
The default implementations emit conservative proposals or metadata only; they
must not be interpreted as efficacy claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from typing import Callable, Mapping, Sequence

from .contracts import (
    AxisStatus,
    CostVector,
    CounterfactualLabel,
    EdgeObservation,
    MetaAction,
    MetaActionKind,
    MetaProposal,
    ProposalEstimate,
    RootObservation,
)


def _normalize(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(x)) for x in values]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / len(clipped)] * len(clipped) if clipped else []
    return [x / total for x in clipped]


@dataclass
class A01StopCouncil:
    """Calibrated wrong-decision risk council.

    The predictor is injected so the skeleton remains dependency-light.  A
    production candidate should fit a position-grouped calibration model from
    H1, P_flip, interval overlap, slopes, omission risk, and evaluator quality.
    """

    axis_id: str = "A01.stop_council"
    status: AxisStatus = AxisStatus.SHADOW
    risk_limit: float = 0.05
    min_visits: int = 16
    predict_wrong: Callable[[RootObservation], float] | None = None

    def features(self, obs: RootObservation) -> dict[str, float]:
        best = obs.best_edge()
        runner = obs.runner_up()
        interval_overlap = 1.0
        if best is not None and runner is not None:
            interval_overlap = max(0.0, min(best.upper, runner.upper) - max(best.lower, runner.lower))
        return {
            "one_minus_h1": 1.0 - float(obs.h1_stability or 0.0),
            "p_flip": float(obs.p_flip if obs.p_flip is not None else 1.0),
            "top2_margin": obs.top2_margin,
            "margin_slope": obs.margin_slope,
            "entropy": obs.entropy,
            "entropy_slope": obs.entropy_slope,
            "interval_overlap": interval_overlap,
            "omission_bound": obs.candidate_omission_bound,
            "revision_count": float(obs.revision_count),
        }

    def risk(self, obs: RootObservation) -> float:
        if self.predict_wrong is not None:
            return min(1.0, max(0.0, float(self.predict_wrong(obs))))
        # Safe fallback: never claims low risk from a single signal.
        f = self.features(obs)
        return min(1.0, max(f["one_minus_h1"], f["p_flip"], f["omission_bound"]))

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        risk = self.risk(obs)
        if obs.root_visits < self.min_visits or risk > self.risk_limit:
            return ()
        best = obs.best_edge()
        if best is None:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.STOP, primary=best.edge_pos, label="risk_council"),
                estimate=ProposalEstimate(
                    regret_reduction_mean=0.0,
                    regret_reduction_lcb=0.0,
                    confidence=1.0 - risk,
                ),
                activation_guard="fresh snapshot; calibrated risk below limit; non-stop VOC checked by arbiter",
                explanation=f"estimated wrong-decision risk={risk:.4f}",
                telemetry=self.features(obs),
            ),
        )


@dataclass
class A02StaticAnchorRPO:
    """Temporary KL-regularized policy improvement from frozen anchor prior."""

    axis_id: str = "A02.static_anchor_rpo"
    status: AxisStatus = AxisStatus.SHADOW
    temperature: float = 0.25
    prior_floor: float = 1e-8
    use_lower_confidence: bool = True

    def solve(self, edges: Sequence[EdgeObservation]) -> dict[int, float]:
        if not edges:
            return {}
        tau = max(1e-5, float(self.temperature))
        logits: list[float] = []
        for edge in edges:
            anchor = max(self.prior_floor, edge.prior_anchor)
            score = edge.lower if self.use_lower_confidence else edge.q_mean
            logits.append(log(anchor) + score / tau)
        shift = max(logits)
        weights = [exp(x - shift) for x in logits]
        probs = _normalize(weights)
        return {edge.edge_pos: prob for edge, prob in zip(edges, probs)}

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        policy = self.solve(obs.edges)
        if not policy:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="publish_temporary_root_policy"),
                estimate=ProposalEstimate(confidence=0.5),
                activation_guard="root-only; frozen anchor; no cumulative mutation",
                explanation="temporary KL-regularized root policy ready for posthoc/readout comparison",
                telemetry={"policy": policy, "temperature": self.temperature},
            ),
        )


@dataclass
class A03UncertaintyDecomposition:
    """Separates MC, epistemic, drift, and bias radii."""

    axis_id: str = "A03.uncertainty_decomposition"
    status: AxisStatus = AxisStatus.SHADOW
    combine: str = "sum"

    def radius(self, edge: EdgeObservation) -> float:
        terms = [edge.mc_radius, edge.epistemic_radius, edge.drift_radius, edge.bias_radius]
        if self.combine == "rss":
            return sum(max(0.0, x) ** 2 for x in terms) ** 0.5
        return sum(max(0.0, x) for x in terms)

    def bounds(self, edge: EdgeObservation) -> tuple[float, float]:
        r = self.radius(edge)
        return max(-1.0, edge.q_mean - r), min(1.0, edge.q_mean + r)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        rows = {
            edge.edge_pos: {
                "radius": self.radius(edge),
                "lower": self.bounds(edge)[0],
                "upper": self.bounds(edge)[1],
            }
            for edge in obs.edges
        }
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="publish_uncertainty_channels"),
                estimate=ProposalEstimate(confidence=0.5),
                activation_guard="completed backups only; evaluator version frozen per root epoch",
                explanation="uncertainty channels computed without assuming independence",
                telemetry={"combine": self.combine, "edges": rows},
            ),
        )


@dataclass
class A04KgVocAllocator:
    """Knowledge-gradient/VOC allocator; not a default low-budget stop rule."""

    axis_id: str = "A04.kg_voc_allocator"
    status: AxisStatus = AxisStatus.SHADOW
    batch: int = 8
    cost_per_eval_ms: float = 1.0

    def proxy(self, edge: EdgeObservation, best: EdgeObservation) -> float:
        gap = max(0.0, best.q_mean - edge.q_mean)
        uncertainty = max(1e-6, edge.total_radius + best.total_radius)
        # Monotone heuristic placeholder. Replace with quadrature or learned
        # residual; do not call it a calibrated KG estimate.
        return uncertainty * exp(-gap / uncertainty)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        best = obs.best_edge()
        if best is None:
            return ()
        proposals: list[MetaProposal] = []
        for edge in obs.edges:
            gain = self.proxy(edge, best)
            if gain <= 0.0:
                continue
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    action=MetaAction(MetaActionKind.SAMPLE, primary=edge.edge_pos, amount=self.batch),
                    estimate=ProposalEstimate(
                        regret_reduction_mean=gain,
                        regret_reduction_lcb=0.5 * gain,
                        confidence=0.25,
                        cost=CostVector(nn_evals=self.batch, cpu_ms=self.batch * self.cost_per_eval_ms),
                    ),
                    activation_guard="allocation only; challenger visible; measured cost replaces constant",
                    explanation=f"KG proxy for edge {edge.edge_pos}",
                )
            )
        return tuple(proposals)


@dataclass
class A05CounterfactualMetaTeacher:
    """Builds labels from forked STOP/SAMPLE/WIDEN/etc. continuations."""

    axis_id: str = "A05.counterfactual_meta_teacher"
    status: AxisStatus = AxisStatus.SEED

    def build_label(
        self,
        *,
        obs: RootObservation,
        action: MetaAction,
        loss_before: float,
        loss_after: float,
        cost: CostVector,
        oracle_action_id: int | None,
    ) -> CounterfactualLabel:
        return CounterfactualLabel(
            checkpoint_id=obs.checkpoint_id,
            position_id=obs.position_id,
            budget=obs.root_visits,
            action=action,
            decision_loss_before=float(loss_before),
            decision_loss_after=float(loss_after),
            realized_cost=cost,
            oracle_action_id=oracle_action_id,
        )

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="fork_snapshot_offline"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="offline or deterministic resident-session fork only",
                explanation="request counterfactual branch generation; never executes inside selection hot path",
                telemetry={"position_id": obs.position_id, "budget": obs.root_visits},
            ),
        )


@dataclass
class A24LearnedBudgetGate:
    """Selects a discrete planning budget on top of a frozen planner."""

    axis_id: str = "A24.learned_budget_gate"
    status: AxisStatus = AxisStatus.SEED
    budgets: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    predict_gain: Callable[[RootObservation, int], float] | None = None
    cost_per_visit_ms: float = 0.1

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        proposals: list[MetaProposal] = []
        for budget in self.budgets:
            if budget <= 0:
                continue
            gain = float(self.predict_gain(obs, budget)) if self.predict_gain else 0.0
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    action=MetaAction(MetaActionKind.SAMPLE, amount=budget, label="budget_gate"),
                    estimate=ProposalEstimate(
                        regret_reduction_mean=gain,
                        regret_reduction_lcb=min(0.0, gain),
                        confidence=0.0 if self.predict_gain is None else 0.5,
                        cost=CostVector(nn_evals=budget, cpu_ms=budget * self.cost_per_visit_ms),
                    ),
                    activation_guard="frozen planner; grouped holdout calibration; hard deadline respected",
                    explanation=f"candidate extra budget={budget}",
                )
            )
        return tuple(proposals)
