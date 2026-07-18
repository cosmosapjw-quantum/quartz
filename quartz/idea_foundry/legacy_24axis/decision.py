"""Decision, policy-improvement, and computation-value skeletons.

Axes covered:
A01 calibrated stop council
A02 static-anchor regularized policy improvement
A03 uncertainty decomposition and bias-aware bounds
A04 KG/VOI allocation (allocation only, not low-budget stopping)
A10 conditional prior-refresh specialist
A22 decaying-entropy/MENTS branch
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import (
    ActionEvidence,
    MetaActionKind,
    MetaCost,
    MetaProposal,
    RootSnapshot,
    normalize_prob,
    stable_softmax,
)


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class UncertaintyDecompositionSkeleton:
    """A03: conservative uncertainty channel composition.

    The default additive radius deliberately avoids assuming independence among
    search sampling, evaluator epistemic error, search drift, and calibrated
    systematic bias.  A covariance-aware path belongs in a separate ablation.
    """

    axis_id: str = "A03_uncertainty_decomposition"

    @staticmethod
    def radius(action: ActionEvidence) -> float:
        return action.total_radius

    @staticmethod
    def interval(action: ActionEvidence) -> tuple[float, float]:
        return action.lower, action.upper

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        mean_radius = (
            float(np.mean([a.total_radius for a in snapshot.visible_actions]))
            if snapshot.visible_actions
            else 0.0
        )
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                confidence=1.0,
                explanation="Publish decomposed root uncertainty; do not alter selection by itself.",
                telemetry={"mean_total_radius": mean_radius},
            )
        ]


@dataclass(frozen=True)
class StopCouncilSkeleton:
    """A01: combine calibrated signals into one conservative STOP proposal.

    This is intentionally a rule skeleton, not the final learned/calibrated
    council.  The production implementation should fit ``p_wrong`` with
    position-grouped splits and apply an upper confidence bound to that risk.
    """

    axis_id: str = "A01_stop_council"
    min_root_visits: int = 16
    max_wrong_risk: float = 0.05
    require_interval_certificate: bool = False

    def estimated_wrong_risk(self, snapshot: RootSnapshot) -> float:
        signals: list[float] = [
            float(np.clip(snapshot.candidate_omission_risk, 0.0, 1.0))
        ]
        if snapshot.h1_stability is not None:
            signals.append(1.0 - float(np.clip(snapshot.h1_stability, 0.0, 1.0)))
        if snapshot.p_flip is not None:
            signals.append(float(np.clip(snapshot.p_flip, 0.0, 1.0)))
        signals.append(float(math.exp(-10.0 * max(snapshot.top2_margin, 0.0))))
        signals.append(float(np.clip(snapshot.revision_count / 8.0, 0.0, 1.0)))
        return max(signals, default=1.0)

    @staticmethod
    def interval_certificate(snapshot: RootSnapshot) -> tuple[bool, float]:
        best = snapshot.best_lower_action
        if best is None:
            return False, float("-inf")
        runner_upper = max(
            (
                a.upper
                for a in snapshot.visible_actions
                if a.action_id != best.action_id
            ),
            default=float("-inf"),
        )
        gap = float(best.lower - runner_upper)
        return gap > 0.0, gap

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        risk = self.estimated_wrong_risk(snapshot)
        certified, cert_gap = self.interval_certificate(snapshot)
        best = snapshot.best_action
        guard_ok = (
            snapshot.fresh
            and snapshot.root_visits >= self.min_root_visits
            and best is not None
            and risk <= self.max_wrong_risk
            and (certified or not self.require_interval_certificate)
        )
        if not guard_ok:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.STOP,
                target_action_ids=(best.action_id,),
                expected_regret_reduction=0.0,
                regret_reduction_lcb=0.0,
                confidence=1.0 - risk,
                activation_guard="fresh && min_visits && calibrated_risk && optional_certificate",
                explanation="All non-STOP proposals must still have non-positive conservative utility in the arbiter.",
                telemetry={"estimated_wrong_risk": risk, "certificate_gap": cert_gap},
            )
        ]


@dataclass(frozen=True)
class StaticAnchorRpoSkeleton:
    """A02: temporary KL-regularized policy around the frozen NN anchor.

    The operator never recursively uses the previous improved policy as the next
    anchor.  Hidden/unmaterialized anchor mass is preserved exactly.
    """

    axis_id: str = "A02_static_anchor_rpo"
    temperature: float = 0.25
    use_lower_bound: bool = True
    prior_floor: float = 1e-8

    def improved_policy(self, actions: Sequence[ActionEvidence]) -> dict[int, float]:
        if not actions:
            return {}
        anchor = normalize_prob(
            [max(a.prior_anchor, self.prior_floor) for a in actions]
        )
        visible_mask = np.asarray([a.visible for a in actions], dtype=bool)
        hidden_mass = float(anchor[~visible_mask].sum())
        if not np.any(visible_mask):
            return {a.action_id: float(p) for a, p in zip(actions, anchor)}

        visible_anchor = normalize_prob(anchor[visible_mask])
        scores = np.asarray(
            [
                a.lower if self.use_lower_bound else a.mean_q
                for a in actions
                if a.visible
            ],
            dtype=np.float64,
        )
        logits = np.log(np.clip(visible_anchor, self.prior_floor, 1.0)) + scores / max(
            self.temperature, 1e-6
        )
        improved_visible = stable_softmax(logits) * (1.0 - hidden_mass)
        output: dict[int, float] = {}
        v_idx = 0
        for action, base_prob in zip(actions, anchor):
            if action.visible:
                output[action.action_id] = float(improved_visible[v_idx])
                v_idx += 1
            else:
                output[action.action_id] = float(base_prob)
        return output

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        policy = self.improved_policy(snapshot.actions)
        if not policy:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.REWEIGHT_POLICY,
                expected_regret_reduction=0.0,
                regret_reduction_lcb=0.0,
                confidence=0.0,
                activation_guard="shadow until paired against no-refresh and Gumbel/RPO baselines",
                explanation="Temporary root policy only; the network anchor remains immutable.",
                telemetry={
                    "policy": policy,
                    "temperature": self.temperature,
                    "uses_lower_bound": self.use_lower_bound,
                },
            )
        ]


@dataclass(frozen=True)
class KgVoiAllocatorSkeleton:
    """A04: one-step Gaussian KG/EI-inspired allocation proposal.

    Stage-7 evidence closed the low-budget KG *stopping* claim.  This skeleton
    keeps KG only as an allocation feature and counterfactual-teacher action.
    """

    axis_id: str = "A04_kg_voi_allocator"
    batch_amount: int = 8

    @staticmethod
    def expected_improvement(best: ActionEvidence, challenger: ActionEvidence) -> float:
        delta = max(0.0, best.mean_q - challenger.mean_q)
        variance = max(1e-12, best.total_radius**2 + challenger.total_radius**2)
        scale = math.sqrt(variance)
        z = delta / scale
        return max(0.0, scale * _normal_pdf(z) - delta * _normal_cdf(-z))

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        best = snapshot.best_action
        if best is None:
            return []
        candidates = [
            a for a in snapshot.visible_actions if a.action_id != best.action_id
        ]
        if not candidates:
            return []
        scored = []
        for action in candidates:
            gain = self.expected_improvement(best, action)
            cost = max(action.cost_ms, 1e-6)
            scored.append((gain / cost, gain, action))
        _, gain, target = max(
            scored, key=lambda item: (item[0], item[2].upper, -item[2].edge_pos)
        )
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SAMPLE,
                target_action_ids=(target.action_id,),
                amount=self.batch_amount,
                expected_regret_reduction=gain,
                regret_reduction_lcb=0.0,
                cost=MetaCost(
                    nn_evals=float(self.batch_amount),
                    cpu_ms=target.cost_ms * self.batch_amount,
                ),
                confidence=0.0,
                activation_guard="allocation-only; do not interpret as a PAC or low-budget halt certificate",
                explanation="Sample the challenger with the largest approximate improvement per measured cost.",
                telemetry={"incumbent": best.action_id, "ei": gain},
            )
        ]


@dataclass(frozen=True)
class PriorRefreshSpecialistSkeleton:
    """A10: conditional expert preserving the historical prior-refresh idea.

    It is deliberately dormant as a default.  A router may activate it only in
    a separately validated weak-evaluator or distribution-shift regime.
    """

    axis_id: str = "A10_prior_refresh_specialist"
    min_prior_visit_js: float = 0.25
    min_eval_uncertainty: float = 0.15

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        eval_unc = max(
            (a.epistemic_radius + a.bias_radius for a in snapshot.visible_actions),
            default=0.0,
        )
        if (
            snapshot.prior_visit_js < self.min_prior_visit_js
            or eval_unc < self.min_eval_uncertainty
        ):
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.REWEIGHT_POLICY,
                confidence=0.0,
                activation_guard="router-selected specialist; never production default",
                explanation="Preserve prior refresh as a conditional expert under explicit shift/weak-evaluator evidence.",
                telemetry={
                    "prior_visit_js": snapshot.prior_visit_js,
                    "eval_uncertainty": eval_unc,
                },
            )
        ]


@dataclass(frozen=True)
class MentsDecayingEntropySkeleton:
    """A22: decaying-entropy search branch.

    Maximum-entropy planning remains an opt-in comparator.  Temperature decays
    so the final objective returns toward the original game value rather than a
    permanent maximum-entropy objective.
    """

    axis_id: str = "A22_ments_decaying_entropy"
    initial_temperature: float = 1.0
    half_life_visits: float = 64.0

    def temperature(self, root_visits: int) -> float:
        return self.initial_temperature * math.exp(
            -math.log(2.0) * root_visits / max(self.half_life_visits, 1e-6)
        )

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        tau = self.temperature(snapshot.root_visits)
        if tau < 1e-3:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.REWEIGHT_POLICY,
                activation_guard="opt-in comparator; decaying temperature only",
                explanation="Expose a decaying Boltzmann/MENTS policy as a baseline, not a default objective.",
                telemetry={"temperature": tau},
            )
        ]
