"""Readout, signed-path, physics-falsification, and symmetry skeletons.

Axes covered:
A15 B13 finite-N curvature readout / target smoothing
A16 coherence-gated signed path disagreement shadow
A17 physics-analogy falsification dashboard
A24 symmetry-orbit and representation-invariance audit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    AnalysisResult,
    MetaActionKind,
    MetaProposal,
    RootSnapshot,
    jensen_shannon,
    normalize_prob,
    shannon_entropy,
    stable_softmax,
)


@dataclass(frozen=True)
class B13CurvatureReadoutSkeleton:
    """A15: decision-neutral finite-N curvature policy readout.

    Existing Stage-7 evidence supports a KL-to-oracle readout effect, not a
    play-strength or action-selection effect.  The default skeleton therefore
    returns a post-hoc policy and never a live selection proposal.
    """

    axis_id: str = "A15_b13_curvature_readout"
    curvature: float = 1.0
    floor: float = 1e-8

    def readout(self, snapshot: RootSnapshot) -> Mapping[int, float]:
        visible = snapshot.visible_actions
        if not visible:
            return {}
        base = normalize_prob([max(action.visits, 0) for action in visible])
        correction = []
        for action in visible:
            stiffness = max(
                self.floor,
                action.total_radius + 1.0 / max(action.visits + 1, 1),
            )
            correction.append(-0.5 * self.curvature * math.log(stiffness))
        logits = np.log(np.clip(base, self.floor, 1.0)) + np.asarray(correction)
        policy = stable_softmax(logits)
        return {action.action_id: float(prob) for action, prob in zip(visible, policy)}

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                activation_guard="posthoc or training-target ablation only",
                explanation="Generate a curvature-aware policy readout without changing the committed action.",
                telemetry={
                    "readout_policy": self.readout(snapshot),
                    "curvature": self.curvature,
                },
            )
        ]


@dataclass(frozen=True)
class CoherenceSignedPathShadowSkeleton:
    """A16: bounded two-dimensional signed path feature.

    This is an operator-inspired diagnostic.  It is not a quantum amplitude and
    must demonstrate incremental prediction beyond ordinary disagreement before
    it may influence selection.
    """

    axis_id: str = "A16_coherence_signed_path_shadow"
    decay_visits: float = 64.0
    clip: float = 10.0

    def coherence(self, snapshot: RootSnapshot) -> float:
        stability = (
            0.0
            if snapshot.h1_stability is None
            else float(np.clip(snapshot.h1_stability, 0.0, 1.0))
        )
        return float(
            math.exp(-snapshot.root_visits / max(self.decay_visits, 1e-6))
            * (1.0 - stability)
        )

    def feature(self, snapshot: RootSnapshot) -> Mapping[int, float]:
        coh = self.coherence(snapshot)
        output = {}
        for action in snapshot.visible_actions:
            phase = math.pi * math.tanh((action.mean_q - action.prior_anchor) * 2.0)
            magnitude = min(
                self.clip, max(0.0, action.total_radius + abs(action.mean_q))
            )
            vector = np.asarray(
                [magnitude * math.cos(phase), magnitude * math.sin(phase)]
            )
            output[action.action_id] = float(coh * np.dot(vector, vector))
        return output

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                activation_guard="shadow-only until incremental predictive value is demonstrated",
                explanation="Record a coherence-gated signed disagreement feature and its classical decay.",
                telemetry={
                    "coherence": self.coherence(snapshot),
                    "feature": self.feature(snapshot),
                },
            )
        ]


@dataclass(frozen=True)
class PhysicsFalsifierSkeleton:
    """A17: analysis-only tests for temperature, redundancy, and scale flow."""

    axis_id: str = "A17_physics_falsifiers"
    beta_grid: tuple[float, ...] = tuple(np.linspace(0.0, 20.0, 201))

    def fit_effective_beta(
        self, policy: Sequence[float], scores: Sequence[float]
    ) -> Mapping[str, float]:
        p = normalize_prob(policy)
        s = np.asarray(scores, dtype=np.float64)
        if p.shape != s.shape or p.size == 0:
            raise ValueError("policy/scores shape mismatch")
        best_beta = 0.0
        best_kl = float("inf")
        for beta in self.beta_grid:
            model = stable_softmax(beta * s)
            kl = float(
                np.sum(
                    p
                    * (
                        np.log(np.clip(p, 1e-12, 1.0))
                        - np.log(np.clip(model, 1e-12, 1.0))
                    )
                )
            )
            if kl < best_kl:
                best_kl = kl
                best_beta = float(beta)
        return {"beta_eff": best_beta, "residual_kl": best_kl}

    @staticmethod
    def redundancy_curve(
        fragment_decisions: Sequence[int], full_decision: int
    ) -> Mapping[str, float]:
        if not fragment_decisions:
            return {"n_fragments": 0.0, "agreement": 0.0}
        agreement = sum(int(x == full_decision) for x in fragment_decisions) / len(
            fragment_decisions
        )
        return {
            "n_fragments": float(len(fragment_decisions)),
            "agreement": float(agreement),
        }

    def analyze_snapshot(self, snapshot: RootSnapshot) -> AnalysisResult:
        visible = snapshot.visible_actions
        if not visible:
            return AnalysisResult(self.axis_id, {"available": False})
        policy = normalize_prob([max(action.visits, 0) for action in visible])
        scores = [action.mean_q for action in visible]
        beta_fit = self.fit_effective_beta(policy, scores)
        metrics = {
            "available": True,
            "policy_entropy": shannon_entropy(policy),
            "effective_branching": math.exp(shannon_entropy(policy)),
            **beta_fit,
        }
        notes = (
            "A low residual KL is required before interpreting beta_eff as a useful surrogate.",
            "FDT/Jarzynski/Crooks tests remain prohibited without explicit forward/reverse kernels and work functional.",
        )
        return AnalysisResult(self.axis_id, metrics, notes)

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        result = self.analyze_snapshot(snapshot)
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                activation_guard="analysis-only; never controls search",
                explanation="Emit falsifiable thermal/redundancy observables and residual goodness-of-fit.",
                telemetry=dict(result.metrics),
            )
        ]


@dataclass(frozen=True)
class SymmetryOrbitAuditSkeleton:
    """A24: D4/action-permutation invariance audit for foundry operators."""

    axis_id: str = "A24_symmetry_orbit_audit"
    tolerance: float = 1e-8

    def audit_policy(
        self,
        original: Sequence[float],
        transformed: Sequence[float],
        inverse_permutation: Sequence[int],
    ) -> Mapping[str, float | bool]:
        p = normalize_prob(original)
        q = normalize_prob(transformed)
        inv = np.asarray(inverse_permutation, dtype=np.int64)
        if q.size != inv.size or p.size != q.size:
            raise ValueError("invalid permutation shapes")
        restored = q[inv]
        max_error = float(np.max(np.abs(p - restored))) if p.size else 0.0
        return {
            "max_error": max_error,
            "js_error": jensen_shannon(p, restored) if p.size else 0.0,
            "equivariant": bool(max_error <= self.tolerance),
        }

    def propose(self, snapshot: RootSnapshot) -> Sequence[MetaProposal]:
        return [
            MetaProposal(
                axis_id=self.axis_id,
                kind=MetaActionKind.SHADOW_ONLY,
                activation_guard="required diagnostic for every game-agnostic operator",
                explanation="Audit action permutations, D4 transforms, zero-mass clones, and negative controls.",
            )
        ]
