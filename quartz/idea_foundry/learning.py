"""Readout, representation, training-control, analysis, and deployment skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, exp, sin
from typing import Any, Mapping, Sequence

from .contracts import (
    AxisStatus,
    CostVector,
    MetaAction,
    MetaActionKind,
    MetaProposal,
    ProposalEstimate,
    RootObservation,
)


def _normalize(values: Sequence[float]) -> list[float]:
    arr = [max(0.0, float(x)) for x in values]
    total = sum(arr)
    return [x / total for x in arr] if total > 0.0 else ([1.0 / len(arr)] * len(arr) if arr else [])


@dataclass
class A17B13CurvatureReadout:
    """Decision-neutral finite-N readout skeleton.

    This module intentionally returns a policy artifact, not a live score
    mutation. Selection, readout, and training-target roles must be ablated
    separately.
    """

    axis_id: str = "A17.b13_curvature_readout"
    status: AxisStatus = AxisStatus.MECHANISM_VALID
    curvature: float = 1.0
    eps: float = 1e-8

    def transform(self, obs: RootObservation) -> dict[int, float]:
        weights = []
        for edge in obs.edges:
            base = max(self.eps, edge.visits + 1.0)
            # Placeholder shaped by finite-N curvature; replace with the pinned
            # Phase-15 B13 implementation in efficacy runs.
            correction = exp(-self.curvature / base)
            weights.append(max(self.eps, edge.prior_current) * correction)
        probs = _normalize(weights)
        return {edge.edge_pos: p for edge, p in zip(obs.edges, probs)}

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="posthoc_readout"),
                estimate=ProposalEstimate(confidence=0.5),
                activation_guard="readout-only by default; decision-neutral evidence not promoted to play strength",
                explanation="finite-N policy readout ready for oracle-KL replay",
                telemetry={
                    "policy": {
                        str(edge_pos): weight
                        for edge_pos, weight in self.transform(obs).items()
                    },
                    "curvature": self.curvature,
                },
            ),
        )


@dataclass
class A18DiffusionRegularizedEvaluator:
    axis_id: str = "A18.diffusion_regularized_evaluator"
    status: AxisStatus = AxisStatus.SEED
    denoise_weight: float = 0.1
    corruption: str = "latent_gaussian"

    def loss_contract(self) -> Mapping[str, Any]:
        return {
            "policy": "cross_entropy_or_search_target_kl",
            "value": "mse_or_distributional_value_loss",
            "denoise": self.corruption,
            "denoise_weight": self.denoise_weight,
            "inference": "direct_deterministic_policy_value_only",
        }

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="training_ablation"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="parameter/FLOP matched deterministic baseline; no diffusion steps in MCTS inference",
                explanation="register denoising auxiliary evaluator training run",
                telemetry=dict(self.loss_contract()),
            ),
        )


@dataclass
class A19RwRestLiteEvaluator:
    axis_id: str = "A19.rw_rest_lite_evaluator"
    status: AxisStatus = AxisStatus.SEED
    nodes: int = 40
    channels: int = 144
    graph_seed: int = 0
    static_pruning: bool = True

    def architecture_contract(self) -> Mapping[str, Any]:
        return {
            "nodes": self.nodes,
            "channels": self.channels,
            "node_op": "one_conv_pre_activation_residual",
            "topology": "ws_dominant_degree_capped_dag",
            "global_blocks": 2,
            "routing": "soft_train_then_static_prune" if self.static_pruning else "soft_only",
            "graph_seed": self.graph_seed,
        }

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="architecture_ablation"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="controller frozen; same replay, eval count, wall-clock, and graph-seed screen",
                explanation="register RW-ResT Lite evaluator experiment",
                telemetry=dict(self.architecture_contract()),
            ),
        )


@dataclass(frozen=True)
class ArchiveRecord:
    position_id: str
    checkpoint_id: str
    game: str
    priority: float
    reason: str
    board_ref: str
    importance_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class A20RegretStateArchive:
    axis_id: str = "A20.regret_state_archive"
    status: AxisStatus = AxisStatus.SEED
    max_records: int = 100_000

    def priority(self, obs: RootObservation) -> float:
        instability = max(0.0, 1.0 - float(obs.h1_stability or 0.0))
        return (
            instability
            + obs.candidate_omission_bound
            + 0.25 * obs.revision_count
            + max(0.0, obs.prior_visit_js)
        )

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        score = self.priority(obs)
        if score <= 0.0:
            return ()
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.ARCHIVE_STATE, value=score, label="regret_instability"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="training control only; deduplicate by position group; importance correction recorded",
                explanation=f"archive priority={score:.4f}",
                telemetry={"priority": score, "position_id": obs.position_id},
            ),
        )


@dataclass
class A21CoherenceSignedPathShadow:
    axis_id: str = "A21.coherence_signed_path_shadow"
    status: AxisStatus = AxisStatus.ANALYSIS_ONLY
    decay: float = 0.05

    def vector(self, magnitude: float, phase: float) -> tuple[float, float]:
        return magnitude * cos(phase), magnitude * sin(phase)

    def coherence(self, obs: RootObservation) -> float:
        stability = float(obs.h1_stability or 0.0)
        return exp(-self.decay * obs.root_visits) * max(0.0, 1.0 - stability)

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        c = self.coherence(obs)
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="signed_path_shadow"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="shadow only until incremental predictive value beats ordinary disagreement features",
                explanation=f"coherence gate={c:.6f}",
                telemetry={"coherence": c},
            ),
        )


@dataclass
class A22PhysicsFalsificationDashboard:
    axis_id: str = "A22.physics_falsification_dashboard"
    status: AxisStatus = AxisStatus.ANALYSIS_ONLY

    def beta_fit_inputs(self, obs: RootObservation) -> Mapping[str, Any]:
        return {
            "policy": [edge.prior_current for edge in obs.edges],
            "scores": [edge.q_mean for edge in obs.edges],
            "budget": obs.root_visits,
        }

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="physics_dashboard"),
                estimate=ProposalEstimate(confidence=0.0),
                activation_guard="analysis-only; null models mandatory; no Jarzynski/FDT without explicit protocols",
                explanation="record beta residual, redundancy, susceptibility, and scale-flow observables",
                telemetry=dict(self.beta_fit_inputs(obs)),
            ),
        )


@dataclass
class A23CpuIncrementalPatternStudent:
    axis_id: str = "A23.cpu_incremental_pattern_student"
    status: AxisStatus = AxisStatus.SEED
    quantization: str = "int8"
    incremental: bool = True

    def deployment_contract(self) -> Mapping[str, Any]:
        return {
            "representation": "line_pattern_codebook_or_nnue_like",
            "quantization": self.quantization,
            "incremental_update": self.incremental,
            "reference_baselines": ["small_resnet", "onnx_cpu", "rapfi_like_pattern_student"],
        }

    def propose(self, obs: RootObservation) -> Sequence[MetaProposal]:
        return (
            MetaProposal(
                axis_id=self.axis_id,
                action=MetaAction(MetaActionKind.NOOP, label="cpu_student_ablation"),
                estimate=ProposalEstimate(
                    confidence=0.0,
                    cost=CostVector(cpu_ms=obs.runtime.eval_latency_ms),
                ),
                activation_guard="teacher/controller frozen; incremental cache correctness and fixed-time Elo measured",
                explanation="register CPU pattern-student deployment experiment",
                telemetry=dict(self.deployment_contract()),
            ),
        )
