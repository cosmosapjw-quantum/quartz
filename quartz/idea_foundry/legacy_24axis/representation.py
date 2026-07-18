"""Representation, deployment, and training-control experiment skeletons.

Axes covered:
A18 diffusion-regularized deterministic evaluator
A19 RW-ResT Lite evaluator
A20 CPU incremental student
A21 regret/instability state archive

These are configuration and data-contract skeletons.  They deliberately avoid a
PyTorch import so the local CPU mechanism suite remains dependency-light.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import AxisState, ExecutionPlane


@dataclass(frozen=True)
class DiffusionRegularizedEvaluatorSpec:
    """A18: direct deterministic policy/value inference + training-only denoising."""

    axis_id: str = "A18_diffusion_regularized_evaluator"
    input_channels: int = 20
    base_channels: int = 64
    board_size: int = 15
    latent_denoising: bool = True
    masked_board_denoising: bool = False
    diffusion_weight: float = 0.1
    policy_value_finetune_fraction: float = 0.3
    inference_uses_diffusion: bool = False

    def validate(self) -> None:
        if self.inference_uses_diffusion:
            raise ValueError("MCTS inference must remain direct and deterministic")
        if self.latent_denoising and self.masked_board_denoising:
            raise ValueError("run latent and masked denoising as separate ablations")
        if not 0.0 <= self.policy_value_finetune_fraction <= 1.0:
            raise ValueError("invalid finetune fraction")


@dataclass(frozen=True)
class RwRestLiteSpec:
    """A19: static-pruned sparse RandWire/ResT evaluator configuration."""

    axis_id: str = "A19_rw_rest_lite_evaluator"
    channels: int = 144
    random_nodes: int = 40
    cells: int = 2
    average_out_degree: float = 4.0
    max_degree: int = 6
    attention_blocks: int = 2
    node_convs: int = 1
    soft_router_during_training: bool = True
    static_prune_for_inference: bool = True
    graph_seed: int = 0

    def validate(self) -> None:
        if self.random_nodes % self.cells != 0:
            raise ValueError("random_nodes must divide into equal cells")
        if self.node_convs != 1:
            raise ValueError("Lite skeleton pins one convolution per random node")
        if not self.static_prune_for_inference:
            raise ValueError(
                "per-position hard graphs are incompatible with efficient MCTS batching"
            )


@dataclass(frozen=True)
class CpuIncrementalStudentSpec:
    """A20: Rapfi/NNUE-inspired deployment student contract."""

    axis_id: str = "A20_cpu_incremental_student"
    pattern_lengths: tuple[int, ...] = (5, 6, 7, 9)
    codebook_dim: int = 64
    accumulator_dim: int = 256
    quantization: str = "int8"
    simd_target: str = "avx2"
    incremental_updates: bool = True
    teacher_heads: tuple[str, ...] = ("policy", "value", "uncertainty")

    def validate(self) -> None:
        if not self.incremental_updates:
            raise ValueError("CPU student axis requires local incremental updates")
        if self.quantization not in {"int8", "int16", "bf16"}:
            raise ValueError("unsupported quantization skeleton")


@dataclass(frozen=True)
class ArchiveRecord:
    schema_version: int
    game: str
    position_id: str
    board_payload: Mapping[str, Any]
    checkpoint_id: str
    source: str
    oracle_regret: float | None
    h1_instability: float | None
    epistemic_error: float | None
    prior_q_js: float | None
    residual_mass_upper: float | None
    revision_count: int
    tactical_tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def stable_key(self) -> str:
        payload = {
            "game": self.game,
            "position_id": self.position_id,
            "board_payload": self.board_payload,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RegretStateArchiveSkeleton:
    """A21: prioritized training-state reuse under a fixed compute budget."""

    axis_id: str = "A21_regret_state_archive"
    regret_weight: float = 1.0
    instability_weight: float = 0.5
    epistemic_weight: float = 0.5
    disagreement_weight: float = 0.25
    residual_weight: float = 0.5
    uniform_mix: float = 0.25

    def priority(self, record: ArchiveRecord) -> float:
        def nz(value: float | None) -> float:
            return (
                0.0
                if value is None or not np.isfinite(value)
                else max(0.0, float(value))
            )

        return (
            self.regret_weight * nz(record.oracle_regret)
            + self.instability_weight * nz(record.h1_instability)
            + self.epistemic_weight * nz(record.epistemic_error)
            + self.disagreement_weight * nz(record.prior_q_js)
            + self.residual_weight * nz(record.residual_mass_upper)
            + 0.05 * min(record.revision_count, 20)
        )

    def normalized_priorities(self, records: Sequence[ArchiveRecord]) -> np.ndarray:
        if not records:
            return np.zeros(0, dtype=np.float64)
        scores = np.asarray(
            [self.priority(record) for record in records], dtype=np.float64
        )
        if float(scores.sum()) <= 0.0:
            focused = np.full(scores.shape, 1.0 / scores.size)
        else:
            focused = scores / scores.sum()
        uniform = np.full(scores.shape, 1.0 / scores.size)
        return (1.0 - self.uniform_mix) * focused + self.uniform_mix * uniform


REPRESENTATION_AXIS_SUMMARY = {
    "A18_diffusion_regularized_evaluator": {
        "state": AxisState.SEED.value,
        "plane": ExecutionPlane.TRAINING.value,
        "core_rule": "denoising is training-only; MCTS policy/value inference is direct and deterministic",
    },
    "A19_rw_rest_lite_evaluator": {
        "state": AxisState.SEED.value,
        "plane": ExecutionPlane.TRAINING.value,
        "core_rule": "soft routing during training, static graph pruning for batched inference",
    },
    "A20_cpu_incremental_student": {
        "state": AxisState.SEED.value,
        "plane": ExecutionPlane.TRAINING.value,
        "core_rule": "pattern codebook + incremental accumulator + quantized CPU deployment",
    },
    "A21_regret_state_archive": {
        "state": AxisState.SEED.value,
        "plane": ExecutionPlane.TRAINING.value,
        "core_rule": "position-grouped archive with uniform mixture and fixed total NN-eval budget",
    },
}
