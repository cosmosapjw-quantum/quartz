"""Candidate construction, widening, sampling, and proof-search skeletons.

Axes covered:
A05 Gumbel + Sequential Halving
A06 residual-evidence widening
A07 JSD-preconditioned locally balanced root sampling
A08 dynamic live-set particle search
A09 tactical sentinel / proof action
"""

from __future__ import annotations

import hashlib
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
    jensen_shannon,
    normalize_prob,
    stable_softmax,
)


def _stable_unit_interval(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(2**64)


def _gumbel(*parts: object) -> float:
    u = min(max(_stable_unit_interval(*parts), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


@dataclass(frozen=True)
class GumbelSequentialHalvingSkeleton:
    """A05: root candidate coverage and bracket proposal.

    The skeleton emits WIDEN/SAMPLE actions; it does not replace interior PUCT.
    A production bracket must be resumable across resident root-continuation
    checkpoints and use the exact fixed-visit ticket accounting of the engine.
    """

    axis_id: str = "A05_gumbel_sequential_halving"
    candidate_count: int = 8
    round_batch: int = 4
    prior_floor: float = 1e-8

    def root_candidates(self, snapshot: RootSnapshot) -> tuple[int, ...]:
        scored = []
        for action in snapshot.actions:
            score = math.log(max(action.prior_anchor, self.prior_floor)) + _gumbel(
                snapshot.root_hash, snapshot.search_epoch, action.action_id
            )
            scored.append((score, action.action_id))
        scored.sort(reverse=True)
        return tuple(action_id for _, action_id in scored[: self.candidate_count])

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        candidate_ids = self.root_candidates(snapshot)
        visible_ids = {a.action_id for a in snapshot.visible_actions}
        hidden = tuple(a for a in candidate_ids if a not in visible_ids)
        proposals: list[MetaProposal] = []
        if hidden:
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    kind=MetaActionKind.WIDEN,
                    target_action_ids=hidden,
                    amount=len(hidden),
                    cost=MetaCost(cpu_ms=0.01 * len(hidden)),
                    activation_guard="root-only candidate bracket",
                    explanation="Materialize Gumbel-without-replacement candidates before certification.",
                )
            )
        survivors = tuple(a for a in candidate_ids if a in visible_ids)
        if survivors:
            proposals.append(
                MetaProposal(
                    axis_id=self.axis_id,
                    kind=MetaActionKind.CHALLENGE,
                    target_action_ids=survivors,
                    amount=self.round_batch,
                    cost=MetaCost(nn_evals=float(self.round_batch * len(survivors))),
                    activation_guard="resumable sequential-halving round",
                    explanation="Allocate an equal bracket round, then eliminate only at the next checkpoint.",
                    telemetry={"candidate_ids": candidate_ids},
                )
            )
        return proposals


@dataclass(frozen=True)
class ResidualEvidenceWideningSkeleton:
    """A06: bound outside live-set mass under a static anchor posterior.

    This is not a literal nested-sampling evidence estimator.  It uses the
    residual partition mass as a candidate-truncation diagnostic and WIDEN gate.
    """

    axis_id: str = "A06_residual_evidence_widening"
    temperature: float = 0.25
    max_residual_ratio: float = 0.05
    widen_count: int = 4
    prior_floor: float = 1e-12

    def bound(self, snapshot: RootSnapshot) -> tuple[float, float, float]:
        visible_terms = []
        outside_terms = []
        tau = max(self.temperature, 1e-6)
        for action in snapshot.actions:
            prior = max(action.prior_anchor, self.prior_floor)
            if action.visible:
                visible_terms.append(prior * math.exp(np.clip(action.mean_q / tau, -80.0, 80.0)))
            else:
                upper = np.clip(action.upper / tau, -80.0, 80.0)
                outside_terms.append(prior * math.exp(upper))
        z_live = float(sum(visible_terms))
        z_out_upper = float(sum(outside_terms))
        ratio = z_out_upper / max(z_live + z_out_upper, self.prior_floor)
        return z_live, z_out_upper, ratio

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        z_live, z_out_upper, ratio = self.bound(snapshot)
        if ratio <= self.max_residual_ratio:
            return []
        hidden = [a for a in snapshot.actions if not a.visible]
        hidden.sort(key=lambda a: (a.upper, a.prior_anchor, -a.edge_pos), reverse=True)
        targets = tuple(a.action_id for a in hidden[: self.widen_count])
        if not targets:
            return []
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.WIDEN,
                target_action_ids=targets,
                amount=len(targets),
                expected_regret_reduction=ratio,
                regret_reduction_lcb=0.0,
                cost=MetaCost(nn_evals=float(len(targets))),
                confidence=0.0,
                activation_guard="residual posterior-mass bound above threshold",
                explanation="Widen the highest upper-score hidden actions until truncation risk is acceptably small.",
                telemetry={"z_live": z_live, "z_out_upper": z_out_upper, "residual_ratio_upper": ratio},
            )
        ]


@dataclass(frozen=True)
class JlbRootSamplerSkeleton:
    """A07: JSD-preconditioned locally balanced root transition.

    ``policy_signature`` must be a common-support policy/value representation of
    each sibling successor state.  JSD defines geometry; P/Q define the target.
    """

    axis_id: str = "A07_jsd_locally_balanced_root"
    bandwidth: float = 0.25
    target_temperature: float = 0.25
    prior_power: float = 1.0

    def transition_matrix(self, actions: Sequence[ActionEvidence]) -> tuple[np.ndarray, np.ndarray]:
        if not actions:
            return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
        if any(not a.policy_signature for a in actions):
            raise ValueError("JLB requires policy_signature on every candidate")
        signatures = [normalize_prob(a.policy_signature) for a in actions]
        if len({sig.shape for sig in signatures}) != 1:
            raise ValueError("JLB signatures require identical common support")
        n = len(actions)
        kernel = np.eye(n, dtype=np.float64)
        bw2 = max(self.bandwidth, 1e-6) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dist2 = jensen_shannon(signatures[i], signatures[j])
                value = math.exp(-dist2 / (2.0 * bw2))
                kernel[i, j] = value
                kernel[j, i] = value
        logits = [
            self.prior_power * math.log(max(a.prior_anchor, 1e-12))
            + a.mean_q / max(self.target_temperature, 1e-6)
            for a in actions
        ]
        target = stable_softmax(logits)
        rates = np.zeros_like(kernel)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rates[i, j] = kernel[i, j] * math.sqrt(max(target[j], 1e-12) / max(target[i], 1e-12))
        row_max = max(float(rates.sum(axis=1).max()), 1.0)
        transition = rates / row_max
        np.fill_diagonal(transition, 1.0 - transition.sum(axis=1))
        return transition, target

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        candidates = [a for a in snapshot.visible_actions if a.policy_signature]
        if len(candidates) < 2:
            return []
        transition, target = self.transition_matrix(candidates)
        current_idx = int(np.argmax([a.visits for a in candidates]))
        next_idx = int(np.argmax(transition[current_idx]))
        target_action = candidates[next_idx]
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SAMPLE,
                target_action_ids=(target_action.action_id,),
                amount=1,
                cost=MetaCost(nn_evals=1.0, cpu_ms=target_action.cost_ms),
                activation_guard="root-only shadow until common-support signature costs are measured",
                explanation="Move on the sibling geometry toward the regularized target distribution.",
                telemetry={"target_probability": float(target[next_idx]), "transition_probability": float(transition[current_idx, next_idx])},
            )
        ]


@dataclass(frozen=True)
class DynamicLiveSetParticleSkeleton:
    """A08: dynamic active/hibernating particle allocation at the root."""

    axis_id: str = "A08_dynamic_live_set_particle"
    total_batch: int = 32
    min_per_active: int = 1
    uncertainty_weight: float = 1.0
    competition_weight: float = 1.0
    resurrection_fraction: float = 0.05

    def weights(self, snapshot: RootSnapshot) -> dict[int, float]:
        best_lower = max((a.lower for a in snapshot.visible_actions), default=-1.0)
        raw = {}
        for action in snapshot.actions:
            competition = max(0.0, action.upper - best_lower)
            uncertainty = action.total_radius
            raw[action.action_id] = self.competition_weight * competition + self.uncertainty_weight * uncertainty
            if not action.visible:
                raw[action.action_id] += self.resurrection_fraction * max(action.prior_anchor, 0.0)
        total = sum(max(v, 0.0) for v in raw.values())
        if total <= 0.0 and raw:
            return {k: 1.0 / len(raw) for k in raw}
        return {k: max(v, 0.0) / total for k, v in raw.items()}

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        weights = self.weights(snapshot)
        if not weights:
            return []
        allocations = {action_id: max(self.min_per_active, int(round(self.total_batch * weight))) for action_id, weight in weights.items() if weight > 0.0}
        ranked = sorted(allocations, key=lambda action_id: (weights[action_id], action_id), reverse=True)
        targets = tuple(ranked)
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.CHALLENGE,
                target_action_ids=targets,
                amount=sum(allocations.values()),
                cost=MetaCost(nn_evals=float(sum(allocations.values()))),
                activation_guard="independent particle groups required before any confidence interpretation",
                explanation="Allocate particles by competition and uncertainty while preserving a resurrection channel.",
                telemetry={"allocations": allocations, "weights": weights},
            )
        ]


@dataclass(frozen=True)
class TacticalSentinelSkeleton:
    """A09: cheap game-specific proof/sentinel channel with explicit provenance."""

    axis_id: str = "A09_tactical_sentinel_proof"
    proof_budget: int = 32

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        forced = [a for a in snapshot.actions if any(flag in {"forced_win", "forced_block", "mate", "immediate_loss_guard"} for flag in a.tactical_flags)]
        if not forced:
            return []
        forced.sort(key=lambda a: ("forced_win" in a.tactical_flags or "mate" in a.tactical_flags, a.upper), reverse=True)
        target = forced[0]
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.PROVE,
                target_action_ids=(target.action_id,),
                amount=self.proof_budget,
                cost=MetaCost(cpu_ms=float(self.proof_budget)),
                confidence=0.0,
                activation_guard="game-specific sentinel flag; generic and tactical claims reported separately",
                explanation="Run a bounded deterministic proof search before statistical elimination or STOP.",
                telemetry={"flags": target.tactical_flags},
            )
        ]
