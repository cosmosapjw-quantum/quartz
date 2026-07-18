# 07. Code Skeleton Map

> DEPRECATED 24-axis snapshot; not a current implementation or evidence contract.

이 문서는 24개 실험 축의 **실제 파일·클래스·Rust promotion 위치**를 한눈에 보여 준다. 현재 skeleton은 importable reference code와 interface contract를 제공하지만 production engine semantics를 자동으로 변경하지 않는다.

---

## 1. 공통 Python 계약

파일: `quartz/idea_foundry/legacy_24axis/contracts.py`

```python
class AxisState(str, Enum): ...
class ExecutionPlane(str, Enum): ...
class MetaActionKind(str, Enum): ...

@dataclass(frozen=True)
class ActionEvidence:
    edge_pos: int
    action_id: int
    visible: bool
    prior_anchor: float
    current_prior: float
    visits: int
    virtual_visits: int = 0
    pending: int = 0
    mean_q: float = 0.0
    m2: float = 0.0
    mc_radius: float = 0.0
    epistemic_radius: float = 0.0
    drift_radius: float = 0.0
    bias_radius: float = 0.0
    policy_signature: tuple[float, ...] = ()
    ...

@dataclass(frozen=True)
class RootSnapshot:
    root_hash: str
    checkpoint_id: str
    position_id: str
    search_epoch: int
    root_visits: int
    iteration: int
    elapsed_ms: float
    remaining_visits: int | None
    actions: tuple[ActionEvidence, ...]
    runtime: RuntimeEvidence
    h1_stability: float | None = None
    p_flip: float | None = None
    candidate_omission_risk: float = 0.0
    fresh: bool = True
    ...

@dataclass(frozen=True)
class MetaProposal:
    axis_id: str
    kind: MetaActionKind
    target_action_ids: tuple[int, ...] = ()
    amount: int = 0
    expected_regret_reduction: float = 0.0
    regret_reduction_lcb: float = 0.0
    cost: MetaCost = field(default_factory=MetaCost)
    activation_guard: str = ""
    explanation: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)
```

`ActionEvidence.edge_pos`는 Rust dense edge index를 의미하고 `action_id`와 분리한다.

---

## 2. 공통 Rust 계약

파일: `src/mcts/foundry/legacy_24axis/foundry_contracts.rs`

```rust
pub enum AxisMode { Shadow, Online, AnalysisOnly }
pub enum MetaActionKind { Sample, Challenge, Widen, ... }

pub struct MetaProposal {
    pub axis_id: &'static str,
    pub kind: MetaActionKind,
    pub target_edge_pos: Vec<u16>,
    pub amount: u32,
    pub expected_regret_reduction: f32,
    pub regret_reduction_lcb: f32,
    pub cost: MetaCost,
    ...
}

pub struct FoundryRootView<'a> {
    pub snapshot: &'a SearchSnapshot,
    pub edges: &'a [EdgeView<'a>],
    pub runtime: FoundryRuntimeView,
    pub search_epoch: u64,
    pub fresh: bool,
    ...
}

pub trait FoundryAxis: Send + Sync {
    fn axis_id(&self) -> &'static str;
    fn mode(&self) -> AxisMode;
    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal>;
}
```

파일: `src/mcts/foundry/legacy_24axis/foundry_axes.rs`

- `StopCouncilAxis`
- `StaticAnchorRpoAxis`
- shadow marker axes A03–A16/A22/A23
- `ShadowAxisPolicy<A>` adapter

두 파일은 의도적으로 `policy/mod.rs`에서 export되지 않는다. 승격 commit은 focused tests와 함께 export해야 한다.

---

## 3. Axis-by-axis map

| ID | Python skeleton | Rust skeleton/promotion target |
|---|---|---|
| A01 | `decision.StopCouncilSkeleton` | `foundry_axes.StopCouncilAxis`, later named `SearchPolicy` |
| A02 | `decision.StaticAnchorRpoSkeleton` | `foundry_axes.StaticAnchorRpoAxis`, immutable policy cache |
| A03 | `decision.UncertaintyDecompositionSkeleton` | edge uncertainty cache; evaluator ingestion |
| A04 | `decision.KgVoiAllocatorSkeleton` | reuse `policy/kg_stop.rs` primitives; future SAMPLE executor |
| A05 | `candidates.GumbelSequentialHalvingSkeleton` | reuse `policy/gumbel_sh.rs`; root bracket state |
| A06 | `candidates.ResidualEvidenceWideningSkeleton` | future WIDEN executor + hidden action upper hints |
| A07 | `candidates.JlbRootSamplerSkeleton` | root-only transition cache; no hot-path JSD computation |
| A08 | `candidates.DynamicLiveSetParticleSkeleton` | future root particle-group scheduler |
| A09 | `candidates.TacticalSentinelSkeleton` | game-specific `TacticalSentinel<G>` trait |
| A10 | `decision.PriorRefreshSpecialistSkeleton` | separate conditional policy; default remains no-refresh |
| A11 | `systems.EntropyMarginRegimeRouterSkeleton` | checkpoint cache + future DEEPEN executor |
| A12 | `systems.ServiceCurveSchedulerSkeleton` | server/runtime scheduler, not `SearchPolicy` |
| A13 | `systems.PendingFlowWuUctSkeleton` | `EdgeView.n` vs `n_virtual`; selection-only correction |
| A14 | `systems.SemanticPathLshSkeleton` | worker-local `PathSketch` telemetry and guarded repulsion |
| A15 | `analysis.B13CurvatureReadoutSkeleton` | Phase-15 readout first; separate selection experiment |
| A16 | `analysis.CoherenceSignedPathShadowSkeleton` | analysis/shadow only until incremental signal |
| A17 | `analysis.PhysicsFalsifierSkeleton` | Python analysis runner only |
| A18 | `representation.DiffusionRegularizedEvaluatorSpec` | PyTorch model/training path; inference direct |
| A19 | `representation.RwRestLiteSpec` | PyTorch architecture branch; static-pruned deployment graph |
| A20 | `representation.CpuIncrementalStudentSpec` | new Rust incremental evaluator interface/backend |
| A21 | `representation.RegretStateArchiveSkeleton` | Python replay/self-play state controller |
| A22 | `decision.MentsDecayingEntropySkeleton` | reuse `policy/ments.rs`, root/shallow comparator |
| A23 | `systems.GraphStateSharingConsistencySkeleton` | TT/state/eval sharing audit; MCGS comparator separate |
| A24 | `analysis.SymmetryOrbitAuditSkeleton` | Python invariance gate required before promotion |

---

## 4. Minimal usage example

```python
from quartz.idea_foundry import RootSnapshot
from quartz.idea_foundry.legacy_24axis.decision import StopCouncilSkeleton
from quartz.idea_foundry.legacy_24axis.candidates import (
    GumbelSequentialHalvingSkeleton,
    ResidualEvidenceWideningSkeleton,
)

snapshot = RootSnapshot(...)

proposals = []
proposals.extend(StopCouncilSkeleton().propose(snapshot))
proposals.extend(GumbelSequentialHalvingSkeleton().propose(snapshot))
proposals.extend(ResidualEvidenceWideningSkeleton().propose(snapshot))
```

실제 arbiter는 estimated gains가 없는 skeleton proposal을 자동 실행해서는 안 된다. 첫 단계에서는 proposal table을 artifact로 저장한다.

---

## 5. Trace adapter skeleton to implement next

권장 신규 파일: `quartz/idea_foundry/legacy_24axis/trace_adapter.py`

```python
from typing import Any
from .contracts import RootSnapshot


def snapshot_from_phase15_bundle(
    bundle: dict[str, Any],
    budget_index: int,
    *,
    edge_table: list[dict[str, Any]] | None = None,
) -> RootSnapshot:
    """Convert one schema-versioned Phase-15 checkpoint to RootSnapshot.

    Must reject:
    - unsupported trace schema
    - policy/action length mismatch
    - restart/continuation pooling
    - missing checkpoint/position identity
    """
    raise NotImplementedError
```

현재 schema-6 trace에는 policy, latency, p_flip, checkpoint/position identity가 있지만 per-action Q/M2/prior/pending 전체는 없다. A02/A03/A04/A06/A07을 실제 trace에서 돌리려면 trace schema 또는 companion edge sidecar가 필요하다.

---

## 6. Meta-action executor skeleton to implement after counterfactual validation

```rust
pub trait MetaActionExecutor<G: GameState> {
    fn sample(&mut self, edge_pos: u16, amount: u32) -> Result<(), ExecError>;
    fn challenge(&mut self, lhs: u16, rhs: u16, amount: u32) -> Result<(), ExecError>;
    fn widen(&mut self, source: CandidateSource, count: u16) -> Result<(), ExecError>;
    fn prove(&mut self, edge_pos: u16, budget: u32) -> Result<ProofResult, ExecError>;
    fn set_runtime(&mut self, setting: RuntimeSetting) -> Result<(), ExecError>;
}
```

첫 online version은 `sample`과 `stop`만 지원한다. unsupported action은 명시적 error를 반환하고 silently fallback하지 않는다.

---

## 7. Tests

현재 reference tests:

```bash
python -m pytest -q tests/test_idea_foundry_skeleton.py
python -m compileall quartz/idea_foundry
```

Rust skeleton은 미-export 상태라 production crate compile에 포함되지 않는다. promotion 시 최소 tests:

```text
proposal serialization
fresh/stale STOP behavior
edge_pos/action_id separation
ArcSwap cache publication
no-op shadow adapter
hot-path allocation benchmark
```
