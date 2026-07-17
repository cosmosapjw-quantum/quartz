# 00. Architecture and Contracts

## 1. 설계 목표

Foundry의 목적은 모든 아이디어를 동시에 production score에 넣는 것이 아니다. 목적은 각 아이디어가 같은 root checkpoint와 비용계약을 읽고, 비교 가능한 `MetaProposal`을 내게 만드는 것이다.

핵심 분리:

```text
Rust data plane
  tree mutation, exact budget, TT, PW, VL, evaluator requests

Rust SearchPolicy plane
  periodic observe, O(1) root score adjustment, fresh halt

Python Phase-15 plane
  trace replay, posthoc readout, calibration, counterfactual labels

Training plane
  evaluator architecture, distillation, archive/restart control

Analysis plane
  physics/geometry observables and falsification
```

## 2. 기존 Rust 계약을 유지하는 이유

현재 엔진은 `Arc<dyn SearchPolicy>`를 공유하고 다음 세 메서드만 호출한다.

```rust
pub trait SearchPolicy: Send + Sync {
    fn name(&self) -> &'static str;
    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]);
    fn score_adjustment(&self, edge: EdgeView<'_>) -> ScoreAdjustment;
    fn should_halt(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) -> HaltDecision;
    fn telemetry(&self) -> ControllerTelemetry;
}
```

이 경계는 다음 이유로 보존한다.

- hot path와 periodic heavy work가 이미 분리되어 있다.
- `EdgeView.idx`는 edge-local dense index이고, game action encoding과 구분할 수 있다.
- policy 내부 cache를 worker가 공유하는 구조가 이미 존재한다.
- LegacyQuartz와 새 정책의 A/B 비교가 가능하다.

다만 WIDEN, PROVE, SET_BATCH처럼 score/halt를 넘어서는 행동은 새 executor가 필요하다. 이를 위해 Rust skeleton은 별도 `FoundryAxis`를 정의한다.

```rust
pub trait FoundryAxis: Send + Sync {
    fn axis_id(&self) -> &'static str;
    fn mode(&self) -> AxisMode;
    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal>;
}
```

`src/mcts/policy/foundry_axes.rs`의 `ShadowAxisPolicy<A>`는 foundry axis를 기존 `SearchPolicy::observe`에 연결하되 selection을 바꾸지 않는 adapter다.

## 3. Python 계약

`quartz/idea_foundry/contracts.py`의 핵심 타입:

```python
@dataclass(frozen=True)
class ActionEvidence:
    edge_pos: int
    action_id: int
    visible: bool
    prior_anchor: float
    current_prior: float
    visits: int
    virtual_visits: int
    pending: int
    mean_q: float
    m2: float
    mc_radius: float
    epistemic_radius: float
    drift_radius: float
    bias_radius: float
    cost_ms: float
```

```python
@dataclass(frozen=True)
class RootSnapshot:
    root_hash: str
    checkpoint_id: str
    position_id: str
    search_epoch: int
    root_visits: int
    iteration: int
    elapsed_ms: float
    actions: tuple[ActionEvidence, ...]
    h1_stability: float | None
    p_flip: float | None
    candidate_omission_risk: float
    fresh: bool
```

```python
@dataclass(frozen=True)
class MetaProposal:
    axis_id: str
    kind: MetaActionKind
    target_action_ids: tuple[int, ...]
    amount: int
    expected_regret_reduction: float
    regret_reduction_lcb: float
    cost: MetaCost
    confidence: float
    activation_guard: str
    telemetry: Mapping[str, Any]
```

## 4. Phase-15 trace 확장 원칙

기존 Phase-15 bundle은 다음을 보존한다.

```text
trace_budgets
trace_policies
trace_latencies_ms
trace_p_flips
checkpoint_id
position_id
trace_source
trace_code_salt
```

Foundry 확장은 schema를 한 번에 크게 늘리지 않는다. 각 추가 channel은 다음 순서로 들어간다.

1. posthoc 계산으로 재구성 가능한 값은 bundle에 저장하지 않는다.
2. engine에서만 알 수 있는 값만 저장한다.
3. source path를 `TRACE_CACHE_RELEVANT_PATHS`에 추가한다.
4. schema version을 올린다.
5. cache salt를 자동 변경한다.
6. restart trace와 root-continuation trace를 절대 pool하지 않는다.

우선 추가 후보:

```text
trace_root_visits
trace_completed_backups
trace_edge_counts
trace_edge_q
trace_edge_m2
trace_pending_counts
trace_visible_masks
trace_runtime (batch, inflight, threads, queue wait)
trace_path_signatures (high-thread lane only)
```

## 5. Freshness contract

STOP과 certificate는 다음 identity가 모두 맞을 때만 허용한다.

```text
root hash
model/checkpoint version
legal action generation
edge version hash
root visits at observe
completed backup count
search epoch
```

stale cache의 허용 범위:

- selection shaping: safe fallback 또는 약한 stale read 가능
- shadow telemetry: stale flag를 포함하면 가능
- STOP/certificate: 금지
- training label: source epoch를 반드시 저장

## 6. Evidence count와 pending count 분리

```text
completed backups  → mean, Welford M2, confidence, readout
pending simulations → selection pressure, thread scheduling
virtual visits       → reservation / collision avoidance
```

pending/virtual count를 MC sample 수로 사용하면 안 된다. A13의 Rust skeleton도 이 원칙을 고정한다.

## 7. Arbiter contract

Arbiter가 비교하는 값:

```text
regret_reduction_lcb
- calibrated NN-eval shadow price
- CPU/GPU wall-clock shadow price
- energy proxy shadow price
```

Skeleton의 `MetaCost.provisional_scalar`는 단위가 섞인 smoke 전용이다. claim-bearing controller는 hardware profile과 teacher forks에서 cost model을 보정해야 한다.

충돌 해결 예:

```text
STOP vs any positive-utility compute
  → compute wins

WIDEN and CHALLENGE
  → omission risk > ranking risk이면 WIDEN 먼저

PROVE forced tactic
  → bounded proof has priority over statistical STOP

SET_BATCH/SET_INFLIGHT
  → one coupled scheduler decision으로 적용
```

## 8. Promotion checklist

한 축을 Rust online으로 승격하려면:

1. Python module과 focused test
2. synthetic 또는 trace-replay mechanism gate
3. symmetry-orbit audit
4. artifact contract와 failure row
5. Rust module export
6. immutable cache
7. hot-path allocation 0
8. fresh-stop test
9. exact budget test
10. paired live smoke
11. claim ledger row

## 9. 코드 위치

```text
quartz/idea_foundry/
  contracts.py
  decision.py
  candidates.py
  systems.py
  analysis.py
  representation.py
  registry.py

src/mcts/policy/
  foundry_contracts.rs       # not exported yet
  foundry_axes.rs           # not exported yet

tests/
  test_idea_foundry_skeleton.py
```
