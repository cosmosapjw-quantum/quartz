# 04. Readout, Physics-Inspired Shadows, and Falsification

> DEPRECATED 24-axis snapshot; not a current implementation or evidence contract.

이 문서는 A15–A17과 A24를 다룬다. 이 그룹의 기본 규칙은 명확하다.

> 분석축은 검색을 제어하지 않는다. 독립적인 예측력 또는 decision gain을 보인 뒤에만 online promotion을 논의한다.

특히 `quantum`, `decoherence`, `RG`, `temperature`라는 명칭은 물리적 동일성을 뜻하지 않는다. 해당 observable이 실제 검색 통계에서 계산되고, null model을 이기며, 기존 feature 이상의 설명력을 가질 때만 알고리즘적 의미가 생긴다.

---

## A15. B13 Finite-N Curvature Readout

### 현재 evidence scope

Stage-7에서 B13은 trained checkpoint의 policy distribution을 oracle 쪽으로 이동시켰지만:

- top-1 action은 바뀌지 않았고
- top-k accuracy도 바뀌지 않았으며
- play strength는 측정하지 않았다.

따라서 현재 역할은 **post-hoc policy readout**이다.

### Skeleton definition

visible action policy를 visit distribution `p_a`라 하고, action stiffness proxy를

\[
\kappa_a
=
r_a^{total}+\frac{1}{N_a+1}
\]

로 둔다. skeleton readout은:

\[
\log p'_a
=
\log(p_a+\epsilon)
-
\frac{c}{2}\log(\kappa_a+\epsilon).
\]

이 식은 theorem claim이 아니라 Stage-7 readout family의 인터페이스 예시다.

### Python skeleton

```python
from quartz.idea_foundry.legacy_24axis.analysis import B13CurvatureReadoutSkeleton

readout = B13CurvatureReadoutSkeleton(curvature=1.0)
policy = readout.readout(snapshot)
```

### 세 역할을 분리한 실험

1. **Readout**
   - final policy artifact만 바꿈
   - search trace 공유
2. **Selection shaping**
   - live root score를 바꿈
   - 별도 trace signature 필요
3. **Training target**
   - self-play target만 바꿈
   - frozen evaluator search와 분리

현재 evidence는 1만 지지한다.

### Baseline/metric

- raw visits
- softmax Q/RPO
- current B13 coefficients
- entropy smoothing

지표:

- KL to high-budget oracle
- top-1/top-k
- calibration
- target gradient norm
- downstream learning speed
- same-wall-clock Elo for selection role

---

## A16. Coherence-Gated Signed Path Disagreement

### 의미

복소 amplitude를 직접 도입하지 않고 2D real vector로 signed disagreement를 표현한다.

trajectory 또는 action `p`에:

\[
z_p
=r_p(\cos\theta_p,\sin\theta_p)
\]

를 정의한다.

후보 mapping:

- `r_p`: confidence radius, value magnitude, novelty의 bounded combination
- `θ_p`: prior–Q disagreement sign, temporal revision direction, player parity

aggregate shadow feature:

\[
I(a)
=
\left\|\sum_{p\to a}w_p z_p\right\|^2
-
\sum_{p\to a}w_p^2\|z_p\|^2.
\]

Skeleton은 더 단순한 per-action bounded norm을 제공한다. 실제 pairwise term은 path telemetry가 생긴 뒤 추가한다.

### Classical decay gate

\[
c_t
=
\exp(-N_t/N_{decay})
(1-H1_t).
\]

검색이 안정되고 예산이 늘수록 feature가 0으로 간다.

### Python skeleton

```python
from quartz.idea_foundry.legacy_24axis.analysis import CoherenceSignedPathShadowSkeleton

shadow = CoherenceSignedPathShadowSkeleton(decay_visits=64)
feature = shadow.feature(snapshot)
```

### 비교해야 할 ordinary features

- absolute prior–Q difference
- evaluator disagreement
- revision count
- H1 instability
- entropy/margin slopes

signed representation이 이들을 넘는 incremental AUC/likelihood gain을 보여야 한다.

### Promotion prohibition

다음이 없으면 selection에 넣지 않는다.

```text
held-out incremental prediction
bounded scale and clipping
D4/action permutation audit
low-overhead path telemetry
matched-cost online ablation
```

---

## A17. Physics-Analogy Falsification Dashboard

A17은 하나의 controller가 아니라 여러 null-test를 묶은 analysis package다.

### 1. Effective temperature fit

observed root policy `π`와 score `S`에 대해:

\[
\beta_{eff}
=
\arg\min_\beta
D_{KL}\left[
\pi\Vert\operatorname{softmax}(\beta S)
\right].
\]

반드시 `residual_kl`을 함께 기록한다. residual이 크면 temperature interpretation을 기각한다.

```python
from quartz.idea_foundry.legacy_24axis.analysis import PhysicsFalsifierSkeleton

fit = PhysicsFalsifierSkeleton().fit_effective_beta(policy, scores)
# {"beta_eff": ..., "residual_kl": ...}
```

분석 질문:

- `β_eff`가 budget과 단조 증가하는가?
- evaluator strength가 바뀌면 관계가 유지되는가?
- residual KL이 trivial margin model보다 낮은가?

### 2. Fragment redundancy

independent seed/group 또는 bootstrap fragment가 full-budget decision을 얼마나 재현하는지 측정한다.

```text
fragment_size
fragment_argmax
full_argmax
agreement
mutual-information surrogate
```

이것은 Quantum Darwinism의 literal quantum mutual information이 아니라 **classical redundant decision record**다.

### 3. RG-like scale flow

budget `b`에 따라:

```text
policy entropy
K_eff
Q gap
curvature readout magnitude
candidate residual mass
H1 stability
```

를 기록하고 rescaling 아래 curve collapse가 있는지 본다.

- curve collapse가 없으면 RG-like description을 기각한다.
- power law는 충분한 dynamic range와 alternative fit 비교 없이 주장하지 않는다.

### 4. Susceptibility

near-tie root에서 small score perturbation `h`를 가한다.

\[
\chi
\approx
\frac{m(h)-m(-h)}{2h},
\qquad
m=\pi_1-\pi_2.
\]

finite budget별 peak를 분석하되 phase-transition claim은 하지 않는다.

### 5. FDT / Jarzynski / Crooks firewall

다음 모두를 명시하지 못하면 test를 실행하지 않는다.

```text
state variable
algorithmic energy
forward transition kernel
reverse transition kernel/protocol
path probability ratio
work functional
stationarity/equilibrium condition
```

단순 return이나 search score를 `work`라고 부르는 것은 금지한다.

### Dashboard artifact

```json
{
  "schema_version": 1,
  "position_id": "...",
  "checkpoint_id": "...",
  "budgets": [8, 16, 32, 64],
  "temperature_fit": [],
  "redundancy_curve": [],
  "scale_flow": [],
  "susceptibility": [],
  "prohibited_tests": ["jarzynski:no_reverse_kernel"]
}
```

---

## A24. Symmetry-Orbit and Representation Audit

### 목적

게임 비특화 controller가 action index, board orientation, zero-mass clone에 우연히 의존하는 것을 막는다.

현재 저장소의 `symmetry_orbit_lab.py` 구조를 모든 Foundry operator에 확장한다.

### Audit laws

#### Equivariant policy/action output

D4 변환 `g`에 대해:

\[
\pi(g s)=g\pi(s).
\]

#### Invariant scalar

```text
entropy
H1 stability
residual mass ratio
stop probability
regime score
```

는 transform에 불변이어야 한다.

#### Candidate clone robustness

zero-prior/zero-mass clone을 추가해도 기존 action ranking과 scalar가 바뀌지 않아야 한다. 단 candidate-count-dependent method라면 명시된 law를 따로 둔다.

### Python skeleton

```python
from quartz.idea_foundry.legacy_24axis.analysis import SymmetryOrbitAuditSkeleton

result = SymmetryOrbitAuditSkeleton().audit_policy(
    original,
    transformed,
    inverse_permutation,
)
```

### Negative controls

- raw index bonus
- first-dict-item tie break
- rotation하지 않은 coordinate channels
- policy와 board augmentation 방향 불일치

harness가 negative control을 잡지 못하면 audit 자체가 실패다.

### Required use

다음 축은 online promotion 전 A24를 통과해야 한다.

```text
A01 stop council
A02 static-anchor RPO
A05 Gumbel/SH
A06 residual widening
A07 JLB
A11 regime router
A15 B13 target/readout
A18/A19 evaluator policy outputs
```

---

## Analysis-only code layout

```text
quartz/idea_foundry/legacy_24axis/analysis.py
scripts/idea_foundry_analysis.py             # future runner
configs/idea_foundry_analysis.v1.json         # future config
results/idea_foundry/<run>/analysis/
```

Analysis output은 `SearchPolicy` telemetry와 구분한다. 검색을 전혀 바꾸지 않았다는 `execution_plane=analysis` 필드를 항상 기록한다.
