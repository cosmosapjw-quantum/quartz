# 02. Candidate Coverage, Widening, and Sampling

이 문서는 A06–A09를 다룬다.

---

## A06. Residual-Partition-Mass Widening

### 핵심 수학

static anchor prior와 root score로 임시 posterior를 정의한다.

\[
q(a)=\frac{\pi_0(a)e^{s_a/\tau}}{Z},
\qquad
Z=\sum_a\pi_0(a)e^{s_a/\tau}.
\]

현재 live/visible set을 `L`이라 하면

\[
Z_L=\sum_{a\in L}\pi_0(a)e^{s_a/\tau}.
\]

미공개 action의 upper score `U_a`로

\[
Z_{out}\le \overline Z_{out}
=
\sum_{a\notin L}\pi_0(a)e^{U_a/\tau}.
\]

그러면 truncation risk upper bound는

\[
R_{omit}^{upper}
=
\frac{\overline Z_{out}}
{Z_L+\overline Z_{out}}.
\]

이 값이 threshold보다 크면 WIDEN을 제안한다.

### Python skeleton

```python
from quartz.idea_foundry.candidates import ResidualEvidenceWideningSkeleton

op = ResidualEvidenceWideningSkeleton(
    temperature=0.25,
    max_residual_ratio=0.05,
    widen_count=4,
)
z_live, z_out_upper, ratio = op.bound(snapshot)
proposal = op.propose(snapshot)
```

### 필요한 engine telemetry

미공개 action마다 다음 중 최소 하나가 필요하다.

- direct child NN value + uncertainty
- tactical upper bound
- calibrated action embedding neighbor bound
- game value range upper bound (fallback; 보통 너무 느슨함)

`upper_hint`가 trivial `+1`뿐이면 초기에는 거의 항상 widening할 수 있으므로 calibration이 핵심이다.

### Baseline

- no widen
- standard progressive widening
- prior top-k
- Gumbel/SH
- priced widening synthetic lab

### Metric

- omission regret
- hidden-best recall
- residual bound calibration
- WIDEN precision
- cost per useful candidate

---

## A07. JSD-Preconditioned Locally Balanced Root Sampling

### 역할 분리

- `P,Q`: 어디를 선호할지 정하는 target density
- JSD: sibling successor state 사이의 local geometry

현재 node의 sibling successor representations를 `ρ_a`라 하면

\[
d_{ab}=\sqrt{JSD(\rho_a,\rho_b)},
\qquad
K_{ab}=\exp[-d_{ab}^2/(2\ell^2)].
\]

Target:

\[
\mu(a)\propto P(a)^{\lambda_P}e^{Q(a)/\tau}.
\]

Locally balanced rate:

\[
R_{ab}=K_{ab}\sqrt{\mu(b)/\mu(a)}.
\]

### Representation contract

`ActionEvidence.policy_signature`는 반드시 sibling 공통 support 위에 있어야 한다.

후보:

1. common legal support에 제한한 child policy
2. child policy + parent-perspective value pseudo-outcomes
3. shared-trunk latent summary

parent-child policy JSD를 직접 사용하지 않는다. player-to-move 관점과 legal mask가 다르기 때문이다.

### Python skeleton

```python
from quartz.idea_foundry.candidates import JlbRootSamplerSkeleton

op = JlbRootSamplerSkeleton(
    bandwidth=0.25,
    target_temperature=0.25,
)
transition, target = op.transition_matrix(snapshot.visible_actions)
```

### 실행 순서

1. frozen trace replay에서 policy signature 계산
2. JSD matrix와 transition만 posthoc 계산
3. prior/Boltzmann/Gumbel과 candidate coverage 비교
4. 신호가 있으면 root 첫 16–32 visits에만 online
5. deeper nodes는 별도 실험 전까지 금지

### Baseline

- prior-proportional sample
- Boltzmann sample
- Gumbel
- PTSA-like aggregation
- board Hamming / latent cosine geometry

### 위험

- 전술 수는 큰 JSD를 가질 수 있다.
- MCTS target은 checkpoint마다 변하므로 exact stationary MCMC가 아니다.
- signature network evaluation cost가 selection gain보다 클 수 있다.

명칭은 `JSD-preconditioned adaptive locally balanced tree policy`가 정확하다.

---

## A08. Dynamic Live-Set Particle Search

### 목적

literal nested sampling의 evidence 계산을 가져오는 대신 variable live population과 mode survival만 사용한다.

Action state:

```text
ACTIVE
HIBERNATING
FROZEN
PROVEN
```

Weight 예:

\[
w_a
=
\lambda_1\max(0,U_a-\max_{b\ne a}L_b)
+
\lambda_2 r_a^{total}
+
\lambda_3 p_a^{best}(1-p_a^{best}).
\]

Batch allocation:

\[
m_a
=
m_{min}
+
\left\lfloor
B\frac{(w_a/c_a)^\gamma}
{\sum_b(w_b/c_b)^\gamma}
\right\rfloor.
\]

### Python skeleton

```python
from quartz.idea_foundry.candidates import DynamicLiveSetParticleSkeleton

op = DynamicLiveSetParticleSkeleton(
    total_batch=32,
    resurrection_fraction=0.05,
)
weights = op.weights(snapshot)
proposal = op.propose(snapshot)
```

### Engine design

첫 실험은 full tree particle MCTS가 아니라 root allocation backend다.

```text
one root action
  └─ multiple independent particle groups / seed families
       └─ group mean and robust uncertainty
```

공유 트리 rollout을 독립 particle이라고 부르지 않는다. 최소한 group identity와 duplicate trajectory를 기록한다.

### Resurrection

휴면 action 전체에 2–5% quota를 분배한다. low-prior hidden tactic을 영구 삭제하지 않는다.

### Baseline

- PUCT
- Gumbel/SH
- uniform root particles
- PMCTS/SMC-inspired implementation

### Metric

- mode survival
- discovery latency
- probability-of-best calibration
- decision regret
- parallel scaling
- duplicate evaluation

---

## A09. Tactical Sentinel and Bounded Proof

### 목적

통계 controller가 강제 전술을 제거하거나 잘못 STOP하는 것을 막는다.

Gomoku 후보:

```text
immediate five
opponent immediate five block
open four
multiple forcing threats
forbidden move guard
shallow threat-space proof
```

Chess 후보:

```text
mate-in-1
king safety legality
forced check/capture extension
repetition guard
```

Go 후보:

```text
ko/suicide legality
immediate capture/atari
bounded ladder probe
```

### Python skeleton

```python
from quartz.idea_foundry.candidates import TacticalSentinelSkeleton

proposal = TacticalSentinelSkeleton(proof_budget=32).propose(snapshot)
```

### Rust design

game-generic controller에 pattern logic을 넣지 않는다.

```rust
pub trait TacticalSentinel<G: GameState> {
    fn scan_root(&self, state: &G) -> Vec<TacticalCandidate<G::Move>>;
    fn prove(&self, state: &G, mv: G::Move, budget: u32) -> ProofResult;
}
```

Production priority:

```text
legality guard
proven forced win/loss
statistical controller
```

### Reporting firewall

- generic controller result
- tactical channel enabled result
- proof false-positive/timeout result

을 분리한다. 게임 특화 이득을 generic metareasoning 이득으로 보고하지 않는다.
