# QUARTZ Idea Foundry — 전체 실험 후보 카탈로그

이 디렉터리는 QUARTZ에 제안된 모든 주요 실험 아이디어를 **하나의 거대한 score 식으로 합치지 않고**, 서로 다른 실행 plane과 evidence tier에 배치하기 위한 설계 문서다.

현재 카탈로그는 최소 요구치 16개를 넘는 **24개 축**을 등록한다. 각 축은 다음을 가진다.

- 문제 정의와 가설
- 현재 저장소에서 재사용할 코드 경계
- Python/Rust code skeleton
- 최소 baseline과 metric
- 활성화 guard
- 알려진 negative evidence의 정확한 범위
- promotion blocker와 revival condition

Machine-readable source of truth:

- `quartz/idea_foundry/registry.py`
- `configs/idea_foundry.axes.v1.json`

공통 Python 계약:

- `quartz/idea_foundry/contracts.py`

Rust runtime skeleton:

- `src/mcts/policy/foundry_contracts.rs`
- `src/mcts/policy/foundry_axes.rs`

> Rust skeleton 두 파일은 의도적으로 아직 `src/mcts/policy/mod.rs`에서 export하지 않는다. Python/Phase-15 shadow gate를 통과한 축만 별도 promotion commit에서 모듈 export, focused unit test, engine wiring을 추가한다.

---

## 1. 현재 저장소와의 정렬

QUARTZ production search의 기존 경계는 그대로 유지한다.

```text
MctsEngine
  ├─ legacy QuartzController / SearchController
  ├─ Arc<dyn SearchPolicy>
  │    ├─ observe(SearchSnapshot, [EdgeView])      # periodic heavy work
  │    ├─ score_adjustment(EdgeView)               # O(1) hot path
  │    └─ should_halt(SearchSnapshot, [EdgeView])  # fresh checkpoint only
  └─ Phase-15 resident continuation / trace cache
```

Foundry는 그 위에 다음 계층을 추가한다.

```text
SearchSnapshot + EdgeView + runtime telemetry
                 │
                 ▼
         immutable FoundryRootView
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
 FoundryAxis             Analysis axis
 MetaProposal[]          shadow metrics
      │
      ▼
 ConservativeArbiter
      │
      ▼
 MetaAction executor (future engine work)
```

현재 `SearchPolicy`는 score와 halt를 직접 다루지만, Foundry의 최종 계산 행동은 더 넓다.

```text
SAMPLE(action, amount)
CHALLENGE(actions, amount)
WIDEN(actions)
DEEPEN(path, amount)
PROVE(action, budget)
REWEIGHT_POLICY
MERGE_OR_SHARE
SET_BATCH / SET_INFLIGHT / SET_THREADS
ARCHIVE_STATE
STOP
```

따라서 `SearchPolicy`는 shadow/readout/STOP adapter로 재사용하고, WIDEN·PROVE·SCHEDULE 같은 행동은 장차 명시적 executor를 추가한다.

---

## 2. 24개 실험 축

| ID | 축 | 기본 plane | 현재 상태 | 핵심 역할 |
|---|---|---|---|---|
| A01 | Calibrated Stop Council | posthoc → Rust | shadow | H1, P_flip, margin, omission risk를 보정된 `p_wrong`으로 통합 |
| A02 | Static-Anchor RPO | posthoc → Rust | seed | 누적 refresh 없이 고정 NN anchor 주위에서 임시 정책개선 |
| A03 | Uncertainty Decomposition | posthoc | seed | MC, epistemic, drift, bias 반경 분리 |
| A04 | KG/VOI Allocator | posthoc → executor | conditional | KG를 저예산 stop이 아니라 다음 계산 배분에 사용 |
| A05 | Gumbel Sequential Halving | Rust root | mechanism-valid | 낮은 예산의 후보 coverage와 단계적 배분 |
| A06 | Residual-Evidence Widening | posthoc → executor | seed | hidden posterior mass bound를 WIDEN 기준으로 사용 |
| A07 | JSD Locally Balanced Root | posthoc → Rust | seed | sibling JSD는 geometry, P/Q는 target density |
| A08 | Dynamic Live-Set Particle | Python/Rust | seed | active/hibernate/resurrect particle group 배분 |
| A09 | Tactical Sentinel / Proof | Rust | seed | cheap forced-win/block guard와 bounded proof search |
| A10 | Prior-Refresh Specialist | Rust | dormant | 약한 evaluator/OOD regime의 conditional expert |
| A11 | Entropy-Margin Router | posthoc → online | shadow | 0회 발화 H3 gate를 연속 change-point feature로 재구성 |
| A12 | Service-Curve Scheduler | Python systems | mechanism-valid | 측정된 batch/inflight/thread service curve 최적화 |
| A13 | Pending Flow / WU-UCT | Rust parallel | mechanism-valid | 미완료 simulation을 증거와 분리해 selection에만 반영 |
| A14 | Semantic Path LSH | Rust parallel | seed | edge duplicate가 아닌 whole-path semantic overlap 제어 |
| A15 | B13 Curvature Readout | posthoc/training | shadow | decision-neutral oracle-KL readout과 target smoothing |
| A16 | Signed Path Shadow | analysis | analysis-only | bounded 2D disagreement feature와 classical decay |
| A17 | Physics Falsifiers | analysis | analysis-only | beta residual, redundancy, susceptibility, scale collapse 검정 |
| A18 | Diffusion-Regularized Evaluator | training | seed | direct deterministic inference + training-only denoising |
| A19 | RW-ResT Lite | training | seed | sparse random wiring + static pruning + local/global mixing |
| A20 | CPU Incremental Student | training/deploy | seed | pattern codebook, incremental accumulator, quantization |
| A21 | Regret State Archive | training control | seed | high-regret/high-instability state 재사용 |
| A22 | MENTS / Decaying Entropy | Rust comparator | conditional | 원 objective로 돌아오는 decaying Boltzmann baseline |
| A23 | Graph Sharing Consistency | posthoc → Rust | seed | state/eval 공유와 parent-edge 통계 공유를 분리 |
| A24 | Symmetry Orbit Audit | analysis | mechanism-valid | D4/action permutation/zero-mass clone invariance 감사 |

---

## 3. 문서 지도

- [`00_ARCHITECTURE_AND_CONTRACTS.md`](00_ARCHITECTURE_AND_CONTRACTS.md) — 공통 데이터·proposal·promotion 계약
- [`01_DECISION_AND_POLICY.md`](01_DECISION_AND_POLICY.md) — A01–A05, A10, A22
- [`02_CANDIDATES_AND_SAMPLING.md`](02_CANDIDATES_AND_SAMPLING.md) — A06–A09
- [`03_PARALLEL_AND_SYSTEMS.md`](03_PARALLEL_AND_SYSTEMS.md) — A11–A14, A23
- [`04_READOUT_PHYSICS_AND_ANALYSIS.md`](04_READOUT_PHYSICS_AND_ANALYSIS.md) — A15–A17, A24
- [`05_REPRESENTATION_AND_TRAINING.md`](05_REPRESENTATION_AND_TRAINING.md) — A18–A21
- [`06_EXPERIMENT_MATRIX_AND_ROADMAP.md`](06_EXPERIMENT_MATRIX_AND_ROADMAP.md) — baseline, campaign, promotion 순서
- [`07_CODE_SKELETON_MAP.md`](07_CODE_SKELETON_MAP.md) — 축별 Python/Rust 파일·클래스·승격 위치

---

## 4. 공통 원리

Foundry의 계산 선택 원리는 다음이다.

\[
U(c\mid x_t)
=
\operatorname{LCB}\left[\widehat{\Delta\mathcal R}(c\mid x_t)\right]
-
\lambda_{NN}C_{NN}(c)
-
\lambda_{time}C_{ms}(c)
-
\lambda_{energy}C_{energy}(c).
\]

- `ΔR`: root posterior simple regret 감소량
- `C`: 실제 NN eval, CPU/GPU wall-clock, energy proxy
- LCB: 낙관적 feature가 controller 전체를 과잉 활성화하지 않게 하는 보수적 하한

STOP은 별도 magic threshold가 아니다.

1. 현재 결정의 calibrated wrong-risk upper bound가 허용치 이하여야 하고,
2. 모든 비정지 계산 행동의 conservative utility가 0 이하여야 하며,
3. 후보 누락 risk가 허용치 이하여야 하고,
4. fresh checkpoint여야 한다.

---

## 5. 증거 계층

```text
SEED
  ↓
MECHANISM_VALID       # synthetic/reference computation works
  ↓
SHADOW                # real traces, no search effect
  ↓
CONDITIONAL           # a specific regime shows signal
  ↓
ACTIVE_EXPERIMENTAL   # live engine, paired budget/cost
  ↓
DEPLOYMENT_CANDIDATE  # multi-seed, hardware-pinned, non-inferior quality
```

실패는 축 전체가 아니라 다음 tuple에 기록한다.

```text
(module, role, game, budget, evaluator, thread regime, hardware)
```

예:

```text
KG × STOP × Gomoku7 × visits 64–256
  → negative / closed

KG × ALLOCATE × near-tie root × visits 32–256
  → open
```

---

## 6. 주요 1차 문헌

- [MCTS as Regularized Policy Optimization](https://proceedings.mlr.press/v119/grill20a.html)
- [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
- [MCTS by Best Arm Identification](https://proceedings.neurips.cc/paper/2017/hash/a6d259bfbfa2062843ef543e21d7ec8e-Abstract.html)
- [Static and Dynamic Values of Computation in MCTS](https://proceedings.mlr.press/v124/sezener20a.html)
- [Maximum Entropy Monte-Carlo Planning](https://papers.neurips.cc/paper_files/paper/2019/hash/7ffb4e0ece07869880d51662a2234143-Abstract.html)
- [Monte Carlo Tree Search with Boltzmann Exploration](https://papers.neurips.cc/paper_files/paper/2023/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html)
- [WU-UCT: Watch the Unobserved](https://arxiv.org/abs/1810.11755)
- [Monte-Carlo Graph Search for AlphaZero](https://arxiv.org/abs/2012.11045)
- [Probability Tree State Abstraction](https://papers.neurips.cc/paper_files/paper/2023/hash/bf89c9fcd0ef605571a03666f6a6a44d-Abstract-Conference.html)
- [Locally Balanced Discrete MCMC](https://arxiv.org/abs/1711.07424)
- [Discrete Langevin Proposal](https://arxiv.org/abs/2206.09914)
- [Dynamic Nested Sampling](https://arxiv.org/abs/1704.03459)
- [Finding the Time to Think](https://arxiv.org/abs/2606.26463)
- [Go-Exploit](https://arxiv.org/abs/2302.12359)
- [Regret-Guided Search Control](https://arxiv.org/abs/2602.20809)
- [D3PM](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
- [RandWire](https://openaccess.thecvf.com/content_ICCV_2019/html/Xie_Exploring_Randomly_Wired_Neural_Networks_for_Image_Recognition_ICCV_2019_paper.html)
- [Differentiable Dynamic Wirings](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Differentiable_Dynamic_Wirings_for_Neural_Networks_ICCV_2021_paper.html)
- [Rapfi](https://arxiv.org/abs/2503.13178)

이 문헌들은 각 축의 선례와 강한 baseline을 정하는 용도다. 개별 선례가 QUARTZ에서의 효능을 대신 증명하지 않는다.
