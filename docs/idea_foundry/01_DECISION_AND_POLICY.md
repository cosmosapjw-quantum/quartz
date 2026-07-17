# 01. Decision, Policy Improvement, and Allocation

이 문서는 A01–A05, A10, A22를 다룬다.

---

## A01. Calibrated Multi-Signal Stop Council

### 가설

H1, `P_flip`, top-2 margin, margin/entropy slope, uncertainty interval, candidate omission risk를 함께 사용하면 단독 signal보다 현재 root 행동이 이후 충분한 예산에서 뒤집힐 위험을 잘 보정할 수 있다.

### 현재 증거

- H1은 trained-net Stage-7 trace에서 `P_flip`보다 훨씬 좋은 calibration을 보였다.
- H1 단독 stop은 저예산에서 너무 보수적이었다.
- `P_flip`은 특정 저예산 checkpoint에서 과도하게 eager했다.
- 따라서 둘 중 하나를 최종 stop 법칙으로 선택할 것이 아니라 `p_wrong` feature로 사용한다.

### Python skeleton

```python
from quartz.idea_foundry.decision import StopCouncilSkeleton

council = StopCouncilSkeleton(
    min_root_visits=16,
    max_wrong_risk=0.05,
    require_interval_certificate=False,
)
proposals = council.propose(snapshot)
```

Production 버전은 skeleton의 max-rule을 제거하고 다음을 학습한다.

```text
label = 1[current argmax != held-out high-budget argmax]
model = calibrated logistic / isotonic / small tree
split = position-grouped + game/checkpoint holdout
output = upper confidence bound of p_wrong
```

### Rust skeleton

```rust
let axis = StopCouncilAxis::default();
let policy = ShadowAxisPolicy::new(axis);
```

초기에는 `ShadowAxisPolicy`로 proposal만 기록한다. 실제 STOP 승격 후에는 dedicated named `SearchPolicy`와 fresh edge-hash 검증을 추가한다.

### Baseline

- fixed budget
- existing `P_flip`
- H1 virtual stop
- margin-only
- V-MCTS-style policy stability

### Metric

- ECE, Brier, log loss
- selective risk / coverage
- realized budget
- high-budget action agreement
- tactical false-stop rate

---

## A02. Static-Anchor Regularized Policy Improvement

### 가설

누적 prior refresh를 제거하고, root 시작 시의 NN prior `π0`를 고정 anchor로 사용해 checkpoint마다 임시 정책을 새로 계산하면 feedback amplification을 줄일 수 있다.

\[
q_\tau(a)
\propto
\pi_0(a)\exp\left(\frac{s_a}{\tau}\right).
\]

`s_a` 후보:

- discovery: optimistic score
- verification/readout: posterior mean
- safety-first: lower confidence value

미공개 action의 anchor mass는 그대로 보존한다.

### Python skeleton

```python
from quartz.idea_foundry.decision import StaticAnchorRpoSkeleton

operator = StaticAnchorRpoSkeleton(
    temperature=0.25,
    use_lower_bound=True,
)
policy = operator.improved_policy(snapshot.actions)
```

Skeleton은 hidden anchor mass를 정확히 보존한다. Production에서는 temperature를 손으로 고정하기보다 robust improvement lower bound를 만족하는 최대 step을 1차원 탐색한다.

### Rust mapping

- `SearchPolicy::observe`: full root vector 계산
- `PolicyCache.effective_prior[edge_pos]`: 임시 정책 저장
- `score_adjustment`: O(1) prior read
- 원 anchor prior는 edge의 raw prior에 남겨둔다.

### Baseline

- current no-refresh legacy
- cumulative visit refresh
- cumulative Q refresh
- exact RPO
- Gumbel root policy

### Kill / promotion

- static prior보다 cross-domain matched-time 성능이 없으면 online update 제거
- oracle KL만 좋아지고 action/learning gain이 없으면 readout 전용으로 제한

---

## A03. Uncertainty Decomposition

### 가설

단일 Welford variance로 search dispersion과 evaluator error를 모두 표현할 수 없다.

\[
r_a^{total}
=
r_a^{MC}+r_a^{epi}+r_a^{drift}+r_a^{bias}.
\]

- MC: completed backup stream
- epistemic: shared trunk + multiple value heads 또는 calibrated proxy
- drift: recent block vs old block
- bias: held-out deep search/exact oracle residual

독립성 검증 전에는 합으로 조합한다.

### Python skeleton

```python
from quartz.idea_foundry.decision import UncertaintyDecompositionSkeleton

lower, upper = UncertaintyDecompositionSkeleton.interval(action)
```

### Rust design changes required

현재 `EdgeView`가 가진 것은 `m2`와 root-level `sigma_eval`이다. 승격 시 다음을 edge-local cache에 추가한다.

```rust
struct EdgeUncertainty {
    mc_radius: f32,
    epistemic_radius: f32,
    drift_radius: f32,
    bias_radius: f32,
}
```

원시 evaluator head output을 hot path에서 계산하지 않는다. `observe()` 또는 inference result ingestion에서 업데이트한다.

### Metric

- nominal interval coverage
- interval width
- high-budget Q error correlation
- OOD/checkpoint-age calibration
- same-action repeat prediction error

---

## A04. KG/VOI Allocation

### 가설

KG의 저예산 stopping claim은 닫혔지만, challenger에 다음 계산을 할당하는 신호로는 여전히 유용할 수 있다.

Skeleton은 incumbent–challenger Gaussian EI를 사용한다.

\[
EI_a=s\phi(\Delta/s)-\Delta\Phi(-\Delta/s).
\]

최종 목표는 forked computation에서 직접 측정한 regret-reduction-per-cost와 KG rank의 상관이다.

### Python skeleton

```python
from quartz.idea_foundry.decision import KgVoiAllocatorSkeleton

proposal = KgVoiAllocatorSkeleton(batch_amount=8).propose(snapshot)
```

### Rust mapping

- 기존 `policy/kg_stop.rs`의 primitive를 재사용할 수 있다.
- `should_halt_by_kg`는 이 lane에서 호출하지 않는다.
- top-m challenger에만 계산한다.
- proposal은 future `MetaActionExecutor::sample(edge_pos, amount)`로 실행한다.

### Baseline

- PUCT next-edge
- top-2 uncertainty
- uniform challenger
- forked VOC oracle rank

### Metric

- Spearman rank with counterfactual gain
- top-1 best computation accuracy
- regret reduction per NN eval/ms

---

## A05. Gumbel + Sequential Halving

### 가설

낮은 budget에서 가장 먼저 해결해야 할 것은 stop보다 candidate coverage와 staged allocation이다.

### Existing code

현재 Rust policy package에는 이미 다음 primitive가 있다.

```text
gumbel_top_m
initial_bracket
SequentialHalvingBracket
```

Foundry work는 이를 다음 contract로 승격한다.

```text
root-only
without replacement
resumable across continuation checkpoints
exact fixed-budget tickets
candidate source union:
  top prior
  gumbel
  top uncertainty/upper
  tactical sentinel
```

### Python skeleton

```python
from quartz.idea_foundry.candidates import GumbelSequentialHalvingSkeleton

operator = GumbelSequentialHalvingSkeleton(candidate_count=8, round_batch=4)
proposals = operator.propose(snapshot)
```

### Rust skeleton

`GumbelSequentialHalvingAxis`는 현재 shadow marker다. 승격 구현은 existing `policy/gumbel_sh.rs`를 직접 사용하고 bracket state를 cache에 저장한다.

### Baseline

- pure PUCT
- prior top-k
- current progressive widening
- Anytime Sequential Halving

### Metric

- low-budget oracle best recall
- hidden low-prior tactical recall
- action coverage
- top1 regret
- eval/time cost

---

## A10. Conditional Prior-Refresh Specialist

### 목적

기존 prior refresh 아이디어를 삭제하지 않되 production default에서 분리한다.

활성 regime 후보:

```text
weak evaluator
large calibrated epistemic/bias radius
large prior/search disagreement
OOD opening or model drift
static-anchor RPO abstains
```

### Skeleton

```python
PriorRefreshSpecialistSkeleton(
    min_prior_visit_js=0.25,
    min_eval_uncertainty=0.15,
)
```

실제 update는 이전 improved prior를 다음 anchor로 쓰지 않는 non-recursive contract를 기본으로 한다.

### Negative scope

현재 Gomoku7 no-refresh 결과는 다음만 말한다.

```text
historical refresh × current Gomoku7 regime × current evaluator/training
  → default로 지지되지 않음
```

다른 regime specialist 가능성까지 폐기하지 않는다.

---

## A22. MENTS / Decaying Entropy Comparator

### 목적

free-energy/maximum-entropy 아이디어를 검증 가능한 comparator로 보존한다.

Permanent MENTS objective는 원 게임 objective와 최적 action이 다를 수 있다. 따라서 temperature를 감소시킨다.

\[
\tau(N)=\tau_0\exp[-(\log 2)N/N_{1/2}].
\]

### Skeleton

```python
MentsDecayingEntropySkeleton(
    initial_temperature=1.0,
    half_life_visits=64.0,
)
```

### Scope

- root 또는 shallow depth만
- final readout은 raw game-value objective
- PUCT, MENTS, BTS, DENTS 비교
- tactical dilution 별도 측정
