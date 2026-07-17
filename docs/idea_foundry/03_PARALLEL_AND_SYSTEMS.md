# 03. Parallel Search, Runtime Scheduling, and Graph Sharing

이 문서는 A11–A14와 A23을 다룬다. 공통 원칙은 다음이다.

> 검색 품질을 설명하는 통계와 시스템이 아직 완료하지 못한 작업량을 같은 증거로 취급하지 않는다.

현재 QUARTZ는 tree-parallel MCTS, adaptive virtual loss, exact fixed-budget ticket, QIPC inference transport, optional automatic thread selection을 이미 갖는다. Foundry는 이 기반을 교체하기보다 다음 빈틈을 채운다.

- entropy/margin change를 이산 burst가 아니라 연속 regime feature로 만들기
- 측정된 evaluator service curve에서 batch/inflight/thread proposal 만들기
- pending simulation을 완료된 backup과 분리하기
- edge-level collision과 whole-path semantic redundancy 구분하기
- transposition의 state identity sharing과 parent-edge statistics sharing 구분하기

---

## A11. Entropy–Margin Change-Point Router

### 기존 증거와 재정의

Stage-7의 H3/O6 2-signal gate는 기본 threshold에서 `0/288`회 발화했다. 이 결과는 다음 주장을 닫는다.

```text
fixed H3 binary floors
× current Gomoku7 Stage-7 traces
× burst trigger role
→ degenerate / unmeasurable
```

그러나 entropy slope와 top-2 margin slope 자체가 무정보라는 결론은 아니다. 새 가설은 binary trigger가 아니라 **추가 계산의 가치가 급변하는 regime change feature**다.

### Feature

checkpoint `t`에서:

\[
\Delta H_t = H(\pi_t)-H(\pi_{t-1}),
\qquad
\Delta m_t = m_t-m_{t-1}.
\]

후보 score:

\[
z_t
=
\frac{\Delta H_t-\operatorname{median}(\Delta H)}{
\operatorname{MAD}(\Delta H)+\epsilon}
-
\lambda_m
\frac{\Delta m_t-\operatorname{median}(\Delta m)}{
\operatorname{MAD}(\Delta m)+\epsilon}.
\]

- entropy가 증가하고 margin이 줄면 `z_t`가 커진다.
- 실제 forked-search에서 추가 계산이 유익했는지를 label로 사용한다.
- threshold는 trace quantile 또는 calibrated probability에서 정한다.

### Python skeleton

```python
from quartz.idea_foundry.systems import EntropyMarginRegimeRouterSkeleton

router = EntropyMarginRegimeRouterSkeleton(
    entropy_growth_scale=0.02,
    margin_shrink_scale=0.02,
    trigger_score=1.0,
    extra_batch=16,
)
proposals = router.propose(snapshot)
```

현재 skeleton은 snapshot의 `entropy_slope`, `margin_slope`로 `DEEPEN` proposal을 만든다.

### Phase-15 integration

승격 전 post-hoc operator를 먼저 추가한다.

```python
POSTHOC_OPERATORS.add("entropy_margin_regime_score")
```

추가 artifact:

```text
checkpoint_id
position_id
budget
entropy_slope
margin_slope
regime_score
forked_voc_hard_label
future_argmax_revision
```

### Rust promotion shape

`SearchSnapshot`에 무조건 slope를 추가하지 않는다. 별도 immutable cache에서 두 checkpoint를 비교한다.

```rust
struct RegimeCache {
    epoch: u64,
    previous_entropy: f32,
    previous_margin: f32,
    entropy_slope: f32,
    margin_slope: f32,
    calibrated_hard_probability: f32,
}
```

`score_adjustment()`는 사용하지 않는다. future meta-action executor가 `DEEPEN`이나 추가 root batch를 실행할 때만 소비한다.

### Baseline과 지표

- binary H3 gate
- entropy only
- margin only
- H1 instability
- small logistic/GBDT regime classifier

지표:

- PR-AUC for high counterfactual gain
- `P(hard|trigger)/P(hard)` lift
- trigger frequency와 budget별 stability
- extra-eval precision
- false burst cost

---

## A12. Hardware Service-Curve Scheduler

### 목적

`nn_evals_per_move` 감소와 accelerator wall-clock 감소는 동일하지 않다. 작은 batch로 너무 빨리 STOP하면 GPU utilization이 나빠질 수 있다. 따라서 scheduler는 다음을 별도 최적화한다.

```text
search controller: 어떤 상태를 더 평가할지
systems controller: 평가 요청을 어떤 batch/inflight/thread 조합으로 실행할지
```

### 기존 재사용 코드

현재 저장소에는:

- `quartz/experiments/service_curve.py`
- `scripts/service_curve_lab.py`
- `configs/service_curve.v1.json`

가 있고 `(batch_size, inflight) -> items/s, ms/batch`를 측정한다. Foundry skeleton은 이 artifact를 읽어 제약 하의 실행 proposal을 만든다.

### Python skeleton

```python
from quartz.idea_foundry.systems import (
    ServiceCurvePoint,
    ServiceCurveSchedulerSkeleton,
)

points = [
    ServiceCurvePoint(batch_size=32, inflight=1, threads=2,
                      items_per_s=800, p95_latency_ms=8.0),
    ServiceCurvePoint(batch_size=64, inflight=2, threads=4,
                      items_per_s=1200, p95_latency_ms=15.0),
]

scheduler = ServiceCurveSchedulerSkeleton(latency_cap_ms=20.0)
proposals = scheduler.propose_from_points(points)
```

### Artifact contract

하드웨어별로 절대 재사용하지 않는다.

```text
hardware_id
OS / driver
PyTorch / CUDA / ROCm build
model architecture hash
precision
batch_size
inflight
threads
items_per_s
mean/p95 latency
power sample provenance
```

CPU, CUDA, ROCm service curve는 서로 다른 artifact다.

### Rust/server promotion

`SearchPolicy`에 scheduler를 넣지 않는다. 서버/session executor가 proposal을 소비한다.

```rust
pub struct RuntimeSetting {
    pub batch_size: usize,
    pub inflight_credit: usize,
    pub search_threads: usize,
}

pub trait RuntimeScheduler: Send + Sync {
    fn propose(&self, runtime: &FoundryRuntimeView,
               deadline_ms: u64) -> RuntimeSetting;
}
```

특히 multi-position/session path는 fairness contract가 없으므로 처음에는 단일 position `search_nn`에서만 실험한다.

### Objective

\[
U_{sys}(b,i,t)
=
\widehat{\Delta R}/\mathrm{ms}
-
\lambda_{p95}\max(0,L_{p95}-D)
-
\lambda_E E.
\]

초기에는 quality-free throughput scheduler와 quality-aware scheduler를 분리한다.

### Baseline과 지표

- current explicit batch/inflight
- current auto-thread throughput/quality
- fixed best service-curve knee
- adaptive scheduler

지표:

- items/s
- end-to-end moves/s
- p50/p95/p99 move latency
- batch occupancy
- queue wait
- GPU seconds/game
- decision regret per wall-clock

---

## A13. Pending Flow / WU-UCT-Style Correction

### 핵심 계약

각 edge에는 최소 세 count가 존재한다.

```text
N_completed  # evidence
N_pending    # work reserved but result not observed
N_outside    # progressive-widening denominator/helper
```

`N_pending`은 selection collision을 줄이는 데 사용하지만 confidence interval, Welford variance, KG posterior, STOP certificate에는 들어가지 않는다.

### Python skeleton

```python
from quartz.idea_foundry.systems import PendingFlowWuUctSkeleton

operator = PendingFlowWuUctSkeleton(
    pending_penalty=1.0,
    collision_threshold=0.35,
)
pending_penalties = operator.pending_scores(snapshot)
proposals = operator.propose(snapshot)
```

### Rust mapping

현재 `EdgeView`에는 이미 다음이 있다.

```rust
pub n: u32,
pub n_virtual: u32,
pub o_a: u32,
```

승격 원칙:

```rust
let evidence_n = edge.n;
let selection_n = edge.n + edge.n_virtual;

// CI / Welford / halt
use evidence_n;

// collision-aware PUCT denominator
use selection_n;
```

virtual value pessimism도 live reservation이 있을 때만 읽는다. fixed VL, adaptive split-VL, vvisit-only, WU-count correction을 별도 조건으로 둔다.

### Pending cancellation

평가 요청 timeout/cancel 시 reservation은 정확히 되돌려야 한다.

```rust
struct PendingGuard<'a> { /* edge refs */ }
impl Drop for PendingGuard<'_> {
    fn drop(&mut self) { self.release_if_uncommitted(); }
}
```

### Baseline과 지표

- disabled
- fixed virtual loss
- adaptive split virtual loss
- vvisit-only
- WU-style pending count

지표:

- edge duplicate rate
- unique leaf ratio
- average virtual pessimism
- selection agreement with serial reference
- NPS
- pending leaks after timeout/panic

---

## A14. Semantic Whole-Path LSH

### 재정의

Stage-5가 지지하지 않은 것은:

```text
adaptive VL보다 edge duplicate를 더 줄인다
```

라는 claim이다. 새 claim은:

```text
서로 다른 첫 edge를 선택했지만
긴 prefix / transposition / motif가 유사한 trajectory의
중복 evaluator work를 줄일 수 있는가
```

이다.

### Path signature

trajectory `γ`의 shingle 예:

```text
(state_hash_t, action_t)
(local_pattern_hash_t, player_to_move)
(depth_bucket, tactical_flag)
```

MinHash signature:

\[
h_j(\gamma)=\min_{x\in shingles(\gamma)} H_j(x).
\]

두 trajectory의 estimated Jaccard overlap이 threshold보다 크면 repulsion proposal을 낸다.

### Python shadow skeleton

```python
from quartz.idea_foundry.systems import SemanticPathLshSkeleton

lsh = SemanticPathLshSkeleton(
    n_hashes=16,
    overlap_threshold=0.8,
    min_threads=8,
)
sig_a = lsh.minhash(path_a)
sig_b = lsh.minhash(path_b)
similarity = lsh.estimated_similarity(sig_a, sig_b)
proposals = lsh.propose(snapshot)
```

### Rust telemetry required

검색 hot path에서 문자열/Vec 할당을 하지 않는다.

```rust
#[derive(Clone, Copy)]
struct PathSketch {
    minhash: [u32; 16],
    prefix_hash: u64,
    depth: u16,
    root_edge_pos: u16,
}
```

worker-local scratch에 sketch를 누적하고 leaf enqueue 시 telemetry에 첨부한다.

### Online activation guard

다음 조건을 모두 만족할 때만 online candidate다.

```text
threads >= 8
edge duplicate already controlled
semantic_path_overlap high
batch diversity low
estimated saved eval cost > sketch overhead
```

repulsion은 path를 삭제하지 않고 alternative root/branch candidate를 한 번 더 sample하게 한다.

### Baseline과 지표

- no semantic control
- adaptive VL only
- transposition dedup only
- MinHash shadow
- online repulsion

지표:

- semantic overlap
- unique state/leaf count
- evaluator duplicate hash rate
- decision gain per additional CPU ns
- high-thread scaling

---

## A23. Graph-State Sharing and Consistency

### 목적

TT/DAG 아이디어는 다음 세 층을 분리해야 한다.

```text
1. state identity / terminal / legal moves sharing
2. evaluator policy/value cache sharing
3. search edge statistics sharing
```

1과 2는 비교적 안전하다. 3은 parent-dependent prior, visit, pending, minimax path context를 섞을 수 있다.

### Recommended data model

```rust
struct StateNode<M> {
    state_hash: u64,
    terminal: Option<f32>,
    eval_cache: EvalCache,
    children_identity: Vec<(M, StateId)>,
}

struct ParentEdge {
    parent: StateId,
    child: StateId,
    action_id: u32,
    visits: AtomicU32,
    value_sum: AtomicF64Like,
    prior: f32,
    pending: AtomicU32,
}
```

현재 engine 구조를 한 번에 DAG로 개조하지 않는다. 첫 실험은 state/eval cache identity sharing만 유지하고, MCGS-style statistic sharing은 별도 comparator다.

### Python skeleton

```python
from quartz.idea_foundry.systems import GraphStateSharingConsistencySkeleton

checker = GraphStateSharingConsistencySkeleton()
result = checker.analyze_occurrences([
    {"value": 0.21, "parent": "p0"},
    {"value": 0.24, "parent": "p1"},
])
proposals = checker.propose(snapshot)
```

### Consistency diagnostics

- 동일 state의 evaluator output mismatch
- parent path별 backed Q discrepancy
- sign/player-to-move mismatch
- repeated-state/repetition rule conflict
- graph merge 전후 root action revision

### Baseline

- TT disabled
- state/eval cache only
- identity-sharing DAG, edge stats separate
- full MCGS comparator

### Promotion gate

- same-state evaluator dedup는 유지 가능
- parent-edge statistics merge는 matched-time simple regret/Elo가 분리 baseline보다 좋아야 함
- repetition/ko/chess history가 hash identity에 포함되지 않으면 merge 금지

---

## 공통 테스트 스켈레톤

```python
def test_pending_never_enters_evidence():
    ...

def test_scheduler_respects_p95_deadline():
    ...

def test_lsh_does_not_activate_at_low_threads():
    ...

def test_graph_sharing_preserves_parent_edge_priors():
    ...
```

Rust promotion commit에서 최소한 다음을 추가한다.

```text
pending reservation RAII test
parallel timeout cleanup test
runtime-setting manifest test
path-sketch no-allocation benchmark
TT state/eval identity consistency test
```
