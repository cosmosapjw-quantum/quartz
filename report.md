## 짧은 요약

업로드한 문서는 방향은 꽤 좋아. “물리학 수사로 포장된 휴리스틱 묶음”을 “검증 가능한 Bayesian/metareasoning controller”로 낮춰 다시 세우려는 태도는 맞다. LegacyQuartz가 실제 구현되어 있고 BayesianQuartz는 아직 P09 계획 단계라는 상태 구분도 솔직하게 되어 있어.  

하지만 현재 설계는 네 목표인 “GPU 의존적 NN+MCTS의 시뮬레이션 수를 줄이면서 탐색 품질은 유지/개선하는 단일원리 기반 CPU-friendly search controller”에는 아직 부족하다. 핵심 결함은 세 가지야. 첫째, “언제 멈출지”는 다루지만 “어느 계산을 다음에 할지”라는 allocation controller가 약하다. 둘째, 수식 검산상 몇 군데는 실제로 틀렸거나 보증을 과장하고 있다. 셋째, CPU-friendly라고 하면서 hot path에 mutex/cache/인덱스 위험이 남아 있다.

내 결론은 BayesianQuartz를 그대로 구현하지 말고, **Bayesian Free-Energy Metareasoning Controller**, 줄여서 **BQ++**, 로 재정의하는 게 맞다. 단일 원리는 이것 하나로 잡으면 된다.

[
\textbf{다음 계산은 “기대 의사결정 손실 감소량 / 실제 계산비용”이 최대인 계산이고, 그 값이 비용보다 작거나 PAC/신뢰구간 인증이 성립하면 멈춘다.}
]

이 원리 아래에서 KL-LUCB/Empirical Bernstein/VOI/Gumbel Sequential Halving/nested-reservoir/physics-inspired free-energy formalism을 역할별로 배치하면 된다.

---

## 1. 현재 문서의 강점

문서의 가장 큰 장점은 LegacyQuartz를 과장하지 않는다는 점이야. LegacyQuartz는 penalty mode, halt mode, cost mode, boolean flag가 얽힌 거대한 휴리스틱 표면이고, BayesianQuartz는 Welford posterior variance, empirical Bernstein scaling, Pearson χ², VOI, KL-LUCB 같은 더 감사 가능한 통계 primitive로 바꾸려는 계획이라고 정리되어 있어. 

외부 문헌과의 방향도 대체로 맞다. AlphaZero류는 NN-guided tree search를 중심으로 강력한 성능을 냈지만, 고정된 수의 search simulation에 크게 의존한다. AlphaZero 원 논문은 chess/shogi/Go에서 general self-play algorithm으로 성능을 보였다고 보고하고 있고, KataGo는 AlphaZero 프로세스/아키텍처 개선으로 compute를 크게 줄였다고 보고한다. 즉 “시뮬레이션/compute 효율 개선”은 이미 정당한 연구 축이다. ([arXiv][1])

또한 “metareasoning” 방향은 매우 적절하다. Hay, Russell, Tolpin, Shimony의 selecting computations 논문은 어떤 future sequence를 simulate할지 선택하는 문제를 Bayesian selection problem으로 다루며, 계산의 expected improvement를 기준으로 Monte Carlo simulation을 제어하는 이론적 틀을 제공한다. 이후 static/dynamic value of computation in MCTS 논문도 UCT의 목표가 approximate planning의 목표와 다를 수 있고, computation value가 root-level MCTS 개선에 쓰일 수 있음을 논의한다. ([People @ EECS][2])

문서가 “BayesianQuartz는 아직 구현되지 않았다”고 명시한 것도 중요하다. P06/P08 일부 scaffold와 KLLUCBStop은 존재하지만 P09 BayesianQuartz, P10 engine wiring, P11 MENTS, P14 evaluator uncertainty hook 등은 아직 pending이라고 되어 있다. 이건 외부 감사 관점에서 좋은 정직성이다. 

---

## 2. 치명적인 검산 오류와 설계상 위험

첫 번째로, 문서의 LegacyQuartz configuration surface 계산이 틀렸다. 문서에는 `7 × 4 × 3 × 2¹¹ ≈ 229,376`이라고 되어 있는데, 실제 곱은

[
7 \times 4 \times 3 \times 2^{11}
= 172{,}032
]

이야. 229,376은

[
7 \times 4 \times 4 \times 2^{11}
]

일 때 나오는 값이다. 게다가 boolean flag 목록에는 `enable_fisher_puct`가 중복되어 있어 unique flag가 10개라면 표면은

[
7 \times 4 \times 3 \times 2^{10}
= 86{,}016
]

이 된다. 결론은 “표면이 너무 크다”는 정성적 비판은 맞지만, 숫자는 고쳐야 한다. 

두 번째로, Pinsker 관련 기존 비판은 맞다. 문서가 지적하듯이 Pinsker는

[
D_{\mathrm{KL}}(P|Q) \ge 2,\mathrm{TV}(P,Q)^2
]

이므로 (\epsilon_t = 1/\sqrt N) 수준의 TV threshold를 원한다면 KL threshold는 (O(1/N))이어야지 (0.5/\sqrt N)이 아니다. 이 부분은 LegacyQuartz의 “물리학/정보이론 수사”를 걷어내야 하는 좋은 예다. 

세 번째로, BayesianQuartz의 VOI 식은 그대로 쓰면 안 된다. 문서는 challenger arm (a)에 대해 대략

[
\mathrm{VOI}(a)=\phi(-\Delta/s_a)s_a
]

를 쓰는데, (\Delta=\mu_b-\mu_a\ge 0), (X=Q_a-Q_b\sim \mathcal N(-\Delta,s^2))라면 실제 expected improvement는

[
\mathbb E[\max(X,0)]
====================

s\phi(\Delta/s)-\Delta\Phi(-\Delta/s)
]

이다. 즉 (s\phi)만 남기는 것은 “보수적 underestimate”가 아니라 대체로 **overestimate**다. 특히 clear-lead regime에서는 두 항이 점근적으로 상쇄되므로 (s\phi)만 쓰면 VOI를 과대평가해서 불필요하게 search를 오래 끌 수 있다. 문서의 “under-estimating VOI is safer” 논리는 부호와 정지 조건을 동시에 혼동하고 있다. 

네 번째로, empirical Bernstein diagnostic의 `eb_gap = best_mu - 2 EB_b`는 gap confidence interval이 아니다. 올바른 best-vs-runner-up certificate는

[
g_{\mathrm{EB}}
===============

# L_b-\max_{a\ne b}U_a

(\hat\mu_b-w_b)-\max_{a\ne b}(\hat\mu_a+w_a)
]

이다. 즉 best arm의 bound만으로는 부족하고, runner-up 또는 모든 competitor의 upper bound가 들어가야 한다. 또한 (Q\in[-1,1]) scale에서 bound를 계산하는지, ([0,1])로 변환한 Bernoulli scale에서 계산하는지 일관되어야 한다. 현재 pseudocode는 `best_mu`는 Q-scale, KL-LUCB는 mapped Bernoulli scale, EB는 다시 Q-scale처럼 섞여 있다. 

다섯 번째로, KL-LUCB의 δ-PAC claim은 root arms가 독립적이고 동일한 의미의 stochastic samples를 낸다는 이상화 아래에서만 안전하다. AlphaZero MCTS에서는 shared subtree, transposition, virtual loss, NN value bias, adaptive sampling 때문에 그 가정이 깨진다. Kaufmann–Koolen의 BAI-MCTS도 깊은 tree를 depth-one confidence interval로 요약하고 root에 BAI를 적용하는 식으로 조심스럽게 문제를 세운다. 따라서 BayesianQuartz 문서의 PAC 문구는 “idealized root-bandit abstraction under calibrated bounded/iid backup model”로 낮춰 써야 한다. ([arXiv][3])

여섯 번째로, χ² test도 “p-value가 있는 정식 hypothesis test”로 쓰면 위험하다. Pearson χ²는 iid multinomial count라는 null model이 있을 때 해석이 깔끔한데, MCTS visit count는 prior에서 독립 샘플링된 count가 아니라 PUCT/virtual loss/value feedback으로 적응적으로 만든 count다. 따라서 χ²는 “network prior surprise / prior miscalibration diagnostic”으로는 좋지만, 그대로 formal envariance test라고 부르면 다시 과장이다. 문서도 Legacy Pinsker 문제를 고치려 하지만, Pearson χ²를 너무 정식 test처럼 말하는 부분은 낮춰야 한다. 

일곱 번째로, concurrency 설명의 “gap_bits can only increase for a stable best arm”은 믿으면 안 된다. empirical best가 바뀔 수 있고, mean estimate는 내려갈 수 있고, (\beta(t,\delta))는 (t)와 함께 증가한다. 그러므로 stale cache가 항상 under-eager halt만 만든다는 주장은 틀릴 수 있다. 정지 판단은 fresh snapshot epoch에서 재계산하거나, cache에 `root_visits_at_observe`, `edge_version_hash`, `best_idx`, `cert_valid_until_visits`를 넣어 stale certificate를 금지해야 한다. 

여덟 번째로, pseudocode에 실제 Rust 이식 때 터질 가능성이 높은 인덱스 버그가 있다. `sigma_a[best as usize]`는 `best`가 edge-local index가 아니라 action id일 경우 바로 잘못된다. Chess/Gomoku/Go policy index는 sparse 또는 game-specific encoding일 수 있으므로 `edge_pos`, `action_id`, `policy_index`를 분리해야 한다. 같은 코드에서 `best` 초기값이 0이고 `best_mu=-inf`인 상태로 `min_pulls`를 만족하는 edge가 하나도 없으면 invalid best가 생긴다. 

---

## 3. 외부 CRAG 판정: 어떤 알고리즘을 살리고, 뭘 버릴지

문헌 대조 결과, 네 목표에 가장 직접적으로 맞는 축은 네 개다.

첫째, **Gumbel AlphaZero/MuZero + Sequential Halving**은 반드시 넣어야 한다. ICLR 2022 Gumbel planning 논문은 AlphaZero가 root에서 모든 action을 방문하지 않으면 policy improvement가 실패할 수 있다고 지적하고, sampling without replacement 기반 정책 개선을 제안하며, few simulations regime에서 prior보다 크게 개선된다고 보고한다. 이건 “적은 simulation으로도 탐색 품질을 유지/개선”이라는 네 목표와 거의 정면으로 맞물린다. ([OpenReview][4])

둘째, **BAI-MCTS / KL-LUCB / empirical confidence certificate**는 stop rule과 root certification에 적합하다. Kaufmann–Koolen은 arbitrary-depth tree를 depth-one confidence interval로 요약하고 root에서 best-arm identification을 적용하는 구조를 제안한다. 이건 BayesianQuartz의 KL-LUCB 방향을 정당화하지만, 동시에 “root-only abstraction의 한계”도 보여준다. ([arXiv][3])

셋째, **Bayesian Thompson/Mixture MCTS**는 hidden move/local minima escape 쪽에 좋다. Bai–Wu–Chen의 DNG-MCTS는 action reward uncertainty를 mixture of Normal distributions와 conjugate priors로 표현하고 Thompson sampling을 각 decision node에서 쓴다. BayesianQuartz가 root-only confidence controller로 가더라도, low-prior hidden candidate를 살리는 reservoir나 challenger sampler에는 Thompson/Gumbel sampling이 잘 맞는다. ([NeurIPS Proceedings][5])

넷째, **MENTS / entropy-regularized MCTS**는 물리학 free-energy 유비를 실제 알고리즘으로 바꾸는 가장 좋은 다리다. MENTS는 MCTS를 maximum entropy policy optimization과 결합하고 softmax value backup을 사용하며, UCT보다 sample-efficient할 수 있음을 주장한다. 네가 말한 “field/path-integral/superposition”은 실제 양자역학이 아니라 log-sum-exp/free-energy/soft Bellman 쪽으로 정식화하는 게 안전하다. ([NeurIPS Papers][6])

반대로, **Keldysh field theory, Jarzynski equality, IIT**는 hot path algorithm으로 넣으면 안 된다. Jarzynski equality는 finite-time work ensemble로 free-energy difference를 얻는 비평형 통계역학 결과지만, exponential average라 고분산 문제가 심한 쪽이고 MCTS root controller의 직접 구현 primitive가 아니다. Nested sampling도 multi-modal posterior/phase-change 문제를 다루는 강한 아이디어지만, finite discrete action root에서는 그대로 쓰기보다 “candidate reservoir / constrained live set”의 설계 유비로만 쓰는 게 낫다. ([arXiv][7])

---

## 4. 업그레이드된 단일 원리: BQ++의 핵심 정의

완성판은 BayesianQuartz가 아니라 다음처럼 재정의하자.

**BQ++: Bounded-rational Bayesian Free-Energy Search Controller**

현재 root state (s), legal action set (A(s)), NN prior (\pi_0(a)), MCTS posterior belief (B_t)가 있다고 하자. 각 action의 true value를 (\theta_a\in[-1,1])라고 두고, 현재 최종 선택은

[
\hat a_t=\arg\max_a \hat\mu_a
]

다. 의사결정 손실을 posterior simple regret으로 둔다.

[
\mathcal R(B_t)
===============

\mathbb E_{\theta\sim B_t}
\left[
\max_a \theta_a-\theta_{\hat a_t}
\right]
]

가능한 계산 (c)는 “edge (a)를 한 번 더 simulate”, “leaf NN eval”, “tactical solver”, “transposition merge”, “hidden candidate expansion” 같은 meta-action이다. 그러면 controller는

[
c_t
===

\arg\max_c
\frac{
\mathbb E[\mathcal R(B_t)-\mathcal R(B_{t+1})\mid c]
}{
\mathrm{cost}(c)
}
]

를 선택한다. 멈춤 조건은

[
\max_c
\mathbb E[\mathcal R(B_t)-\mathcal R(B_{t+1})\mid c]
\le
\mathrm{cost}(c)
]

또는

[
L_{\hat a_t}>\max_{a\ne \hat a_t}U_a
]

같은 certificate가 성립할 때다.

이게 단일 원리다. KL-LUCB, empirical Bernstein, VOI, Gumbel SH, nested reservoir는 전부 이 원리의 근사 구현일 뿐이다. 인간 그랜드마스터식 “몇 수만 깊게 본다”, “수마다 시간 배분이 다르다”, “직관은 쓰되 그대로 믿지는 않는다”도 이 원리로 설명된다. 직관은 (\pi_0)와 pattern prior이고, 계산은 (\mathrm{VOI}/\mathrm{cost})가 높은 후보에게 집중된다.

---

## 5. 물리학 유비를 제대로 살리는 형식화

양자역학 용어는 버리고, **path integral이 아니라 path measure / free-energy log-sum-exp**로 정식화하면 된다.

MCTS trajectory를

[
\gamma=(s_0,a_0,s_1,a_1,\ldots,s_L)
]

라고 하자. NN prior가 주는 proposal path measure는

[
\Pi_\theta(\gamma)
==================

\prod_{\ell=0}^{L-1}
\pi_\theta(a_\ell\mid s_\ell)
P(s_{\ell+1}\mid s_\ell,a_\ell)
]

이다. Search controller가 실제로 유도하는 trajectory distribution을 (q_t(\gamma))라고 두면, entropy-regularized decision free energy는

[
\mathcal F_t[q]
===============

\mathbb E_{\gamma\sim q}
\left[
-\hat R(\gamma)
+
\lambda,C(\gamma)
\right]
+
\tau D_{\mathrm{KL}}(q|\Pi_\theta)
]

로 쓸 수 있다. 최소화하면

[
q_t^*(\gamma)
\propto
\Pi_\theta(\gamma)
\exp\left(
\frac{\hat R(\gamma)-\lambda C(\gamma)}{\tau}
\right)
]

가 된다. 이것이 “선형대수적/path integral적 유비”의 안전한 버전이다. superposition은 “여러 trajectory에 대한 확률질량/가중치”, entanglement는 “shared subtree/transposition이 만드는 posterior covariance”로만 말해야 한다. 실제 양자역학 claim은 금지다.

Root action의 soft value는

[
F_t(a)
======

\tau\log
\sum_{\gamma:a_0=a}
\Pi_\theta(\gamma)
\exp\left(
\frac{\hat R(\gamma)-\lambda C(\gamma)}{\tau}
\right)
]

로 볼 수 있고, one-loop correction을 억지로 쓰고 싶다면 saddle 근처에서

[
F_t(a)
\approx
\hat R(\gamma_a^*)
-\lambda C(\gamma_a^*)
-\frac{\tau}{2}\log\det H_a
]

같은 꼴이 나온다. 하지만 (H_a)를 실제로 계산하는 건 비싸고 불안정하다. 따라서 one-loop는 hot path 수식으로 쓰지 말고, empirical variance/confidence width/entropy correction으로 대체한다.

[
\text{loop_proxy}_a
===================

w_a
\quad\text{or}\quad
\log(\hat\sigma_a^2+\epsilon)
\quad\text{or}\quad
\mathrm{KG}_a
]

이렇게 하면 “one-loop correction”이 더 이상 수사가 아니라 “uncertainty/certification correction”으로 떨어진다.

---

## 6. BQ++의 모듈 구조

BQ++는 다섯 모듈이면 충분하다.

**1. Calibrated Belief Module**

각 edge (a)에 대해 다음을 유지한다.

[
n_a,\quad
\hat\mu_a,\quad
M2_a,\quad
\hat\sigma_a^2,\quad
\pi_0(a),\quad
o_a,\quad
\sigma_{\mathrm{eval}}^2
]

값은 반드시 하나의 scale로 통일한다. 추천은 ([0,1]) scale이다.

[
x=\frac{Q+1}{2}
]

backup mean과 variance를 ([0,1])에서 계산하고, 최종 score에 넣을 때만 다시 ([-1,1])로 바꾼다.

Welford shrinkage는 “Beta-Binomial posterior”라고 부르지 말고 **empirical-Bayes variance shrinkage**라고 불러야 한다.

[
\hat\sigma_{a,\mathrm{shrunk}}^2
================================

\frac{M2_a+\lambda_0\sigma_{\mathrm{parent}}^2}{\max(n_a-1,1)+\lambda_0}
]

단, (n_a<2)일 때는 parent/root variance와 evaluator uncertainty를 섞는다.

[
\hat\sigma_a^2
==============

\max(
\hat\sigma_{a,\mathrm{shrunk}}^2,
\sigma_{\mathrm{eval}}^2,
\sigma_{\min}^2
)
]

**2. Confidence/Certificate Module**

Empirical Bernstein width를 쓴다.

[
w_a
===

\sqrt{
\frac{2\hat\sigma_a^2\log(3K t^\alpha/\delta)}
{\max(n_a,1)}
}
+
\frac{7R\log(3K t^\alpha/\delta)}
{3\max(n_a-1,1)}
]

여기서 (R=1) if ([0,1]), (R=2) if ([-1,1]). 추천은 ([0,1])라서 (R=1).

[
L_a=\hat\mu_a-w_a,\qquad U_a=\hat\mu_a+w_a
]

Stop certificate는

[
L_b>\max_{a\ne b}U_a
]

이다. KL-LUCB는 terminal win/loss backup 또는 calibrated Bernoulli-like value에서만 strong certificate로 쓰고, 일반 NN value backup에서는 EB certificate를 primary로 둔다.

**3. Computation-Value Module**

현재 문서의 VOI를 다음으로 교체한다. Challenger (a\ne b)에 대해

[
\Delta_a=\hat\mu_b-\hat\mu_a\ge 0
]

[
s_a^2
=====

\mathrm{Var}(\hat\mu_b)+\mathrm{Var}(\hat\mu_a)
\approx
\frac{\hat\sigma_b^2}{n_b+\lambda_0}
+
\frac{\hat\sigma_a^2}{n_a+\lambda_0}
]

이면 expected improvement proxy는

[
\mathrm{EI}_a
=============

## s_a\phi(\Delta_a/s_a)

\Delta_a\Phi(-\Delta_a/s_a)
]

이다. 하지만 실제로는 “action (a)를 선택했을 때의 improvement”가 아니라 “action (a)를 한 번 더 계산했을 때 posterior simple regret이 얼마나 줄어드는가”가 필요하므로, 구현에서는 cheap Knowledge Gradient approximation을 쓴다.

[
\mathrm{KG}_a
\approx
\mathbb E[
\max_j \mu_j^+
]
-

\max_j\mu_j
]

CPU-friendly하게는 top (m) 후보에 대해서만 계산하고, 나머지는 (U_a-L_b) 기반 bound로 근사한다.

**4. Candidate Reservoir / Anti-local-minimum Module**

Root 후보를 네 source에서 만든다.

[
C
=

C_{\mathrm{prior}}
\cup
C_{\mathrm{gumbel}}
\cup
C_{\mathrm{upper}}
\cup
C_{\mathrm{tactical}}
]

여기서

[
C_{\mathrm{prior}}=\operatorname{top}_m \pi_0(a)
]

[
C_{\mathrm{gumbel}}=\operatorname{top}_m {\log\tilde\pi_0(a)+g_a}
]

[
C_{\mathrm{upper}}=\operatorname{top}_m U_a
]

[
C_{\mathrm{tactical}}=\text{forced win/loss/threat detector candidates}
]

이다. (\tilde\pi_0)는 prior floor를 둔 calibrated prior다.

[
\tilde\pi_0(a)
==============

(1-\epsilon)\pi_0(a)+\epsilon/K
]

Nested sampling은 여기서만 살린다. “likelihood threshold”를 action upper score로 바꿔서 live set을 유지한다.

[
\Lambda_a
=========

U_a+\rho,\mathrm{KG}_a+\tau\log\tilde\pi_0(a)
]

낮은 (\Lambda_a) 후보를 live set에서 제거하고, unexplored 또는 low-prior action 중 Gumbel/Thompson으로 새 후보를 넣는다. 이건 실제 nested evidence estimator가 아니라 **nested-reservoir search**다.

**5. CPU-friendly Cache Module**

Hot path에서는 계산하지 않는다. `observe()`에서 모든 배열을 한 번 계산해서 immutable cache로 publish한다.

캐시는 이런 꼴이어야 한다.

```rust
struct PolicyCache {
    epoch: u64,
    root_visits: u32,
    edge_version_hash: u64,

    // indexed by edge_pos, not action_id
    p_eff: Vec<f32>,
    q_ctrl: Vec<f32>,
    penalty: Vec<f32>,
    kg: Vec<f32>,
    lower: Vec<f32>,
    upper: Vec<f32>,

    best_pos: usize,
    cert_gap: f32,
    max_kg_per_ms: f32,
    prior_surprise: f32,
    forced_move_pos: Option<usize>,
}
```

`score_adjustment()`는 `ArcSwap` 또는 RCU-style atomic pointer로 cache를 읽고, edge-local index만 사용한다. mutex는 hot path에서 제거한다.

---

## 7. 최종 selection score

기존 AlphaZero-style PUCT는 유지한다.

[
S_a^{\mathrm{PUCT}}
===================

Q_a
+
c_{\mathrm{puct}}
P_{\mathrm{eff}}(a)
\frac{\sqrt{N}}{1+n_a+o_a}
]

다만 (P_{\mathrm{eff}})는 raw NN prior가 아니다.

[
P_{\mathrm{eff}}(a)
===================

\operatorname{Normalize}
\left[
\tilde\pi_0(a)
\exp\left(
\frac{\mathrm{KG}_a/\bar c_a+s_a^{\mathrm{tactical}}+s_a^{\mathrm{surprise}}}{\tau_t}
\right)
\right]
]

여기서 (\bar c_a)는 action (a)를 더 계산하는 평균 wall-clock cost다. 즉 계산 가치가 높은 후보는 prior가 약해도 다시 살아난다.

Q 쪽은 risk-aware하게 조정한다.

[
Q_a^{\mathrm{ctrl}}
===================

## 2\hat\mu_a-1

\rho_t(2w_a)
+
\eta_t(2,\mathrm{KG}_a)
]

하지만 이 조정은 너무 강하면 hidden win을 죽인다. 그래서 추천은 이렇게 한다.

* selection 중에는 (Q_a^{\mathrm{ctrl}})를 사용하되,
* final move decision에서는 raw posterior mean (\hat\mu_a), confidence certificate, tactical override를 같이 본다.
* forced win/loss solver가 있으면 MCTS score보다 우선한다.

(\rho_t)와 (\eta_t)는 새 hyperparameter로 두지 말고, entropy/certificate 상태에서 유도한다.

[
\rho_t=
\mathbf 1[\text{certification phase}]
]

[
\eta_t=
\mathbf 1[\text{exploration phase}]
]

즉 early phase는 KG/Gumbel 중심, late phase는 confidence/certificate 중심이다.

---

## 8. 최종 pseudocode

```text
BQ++ observe(snapshot, edges):

    # 0. Freshness guard
    if snapshot.root_visits < warmup_min:
        publish_warmup_cache()
        return

    # 1. Build local edge-indexed arrays
    for each edge position i:
        action_id[i]  = edge.action_id
        n[i]          = edge.visits
        o[i]          = edge.virtual_visits
        q_raw[i]      = edge.q              # [-1, 1]
        x_mean[i]     = 0.5 * (q_raw[i] + 1) # [0, 1]
        prior_raw[i]  = edge.prior
        m2[i]         = edge.m2_f64

    # 2. Calibrate and smooth prior
    prior = normalize_legal(prior_raw)
    prior = (1 - eps_prior) * prior + eps_prior / K
    prior = renormalize(prior)

    # 3. Posterior variance, all on [0,1] scale
    parent_var = max(snapshot.root_value_var_on_01, var_floor)

    for i in 0..K:
        denom = max(n[i] - 1, 1) + lambda0_empirical(snapshot)
        var_backup[i] = (m2[i] + lambda0 * parent_var) / denom
        var_eval[i]   = evaluator_uncertainty_if_available(i)
        sigma2[i]     = clamp(max(var_backup[i], var_eval[i], var_floor),
                              var_floor, 0.25)

    # 4. Confidence intervals
    beta = log(3 * K * snapshot.root_visits^alpha / delta_move(snapshot))
    for i in 0..K:
        width[i] = sqrt(2 * sigma2[i] * beta / max(n[i],1))
                 + 7 * beta / (3 * max(n[i]-1,1))  # [0,1] range
        lower[i] = clamp(x_mean[i] - width[i], 0, 1)
        upper[i] = clamp(x_mean[i] + width[i], 0, 1)

    best = argmax_i x_mean[i]
    second_upper = max_i_except(best, upper[i])
    cert_gap = lower[best] - second_upper

    # 5. Prior surprise diagnostic, not a formal iid χ² test
    # Clamp prior, renormalize first. Use as calibration signal.
    surprise = adaptive_prior_residual(counts=n, prior=prior)
    broaden = surprise > calibrated_surprise_threshold

    # 6. Candidate reservoir
    C = empty set
    C.add(top_m_by(prior, m_prior))
    C.add(top_m_by(upper, m_upper))
    C.add(gumbel_top_m(log(prior), m_gumbel))
    C.add(tactical_sentinel_candidates(snapshot, edges))

    if broaden:
        C.add(low_prior_high_uncertainty_candidates(prior, upper, n))

    C = deduplicate_and_cap(C, max_candidates_from_budget(snapshot))

    # 7. Approximate KG / VOI only on C
    kg = zeros(K)
    for i in C:
        kg[i] = approximate_knowledge_gradient(
                    i, best, x_mean, sigma2, n, top_competitors=C)

    # 8. Nested-reservoir refinement
    if local_minimum_risk(prior, surprise, cert_gap, kg):
        threshold = quantile([upper[i] + kg[i] for i in C], q=0.25)
        C.remove_if(lambda i: upper[i] + kg[i] < threshold)
        C.add(replenish_from_unexplored_by_gumbel_or_thompson())

    # 9. Effective prior for PUCT
    tau = entropy_temperature_from_budget_and_uncertainty(snapshot, cert_gap)
    for i in 0..K:
        alloc_score[i] = log(prior[i]) + kg[i] / max(cost_ms[i], eps) / tau
        if i in tactical_forced_win:
            alloc_score[i] = +INF
        if broaden and upper[i] > lower[best]:
            alloc_score[i] += surprise_bonus

    p_eff = softmax_stable(alloc_score)

    # 10. Risk-aware q override, but do not kill hidden candidates
    for i in 0..K:
        if phase == EXPLORATION:
            q_ctrl[i] = 2*x_mean[i] - 1 + 2*kg[i]
        else if phase == CERTIFICATION:
            q_ctrl[i] = 2*x_mean[i] - 1 - 2*width[i]
        else:
            q_ctrl[i] = 2*x_mean[i] - 1

    # 11. Publish immutable cache
    publish_cache(epoch, p_eff, q_ctrl, lower, upper, kg,
                  best, cert_gap, surprise, forced_move)
```

Hot path는 이렇게 단순해야 한다.

```text
BQ++ score_adjustment(edge):

    cache = atomic_load_policy_cache()

    i = edge.edge_pos

    if cache.epoch is stale:
        return no_adjustment_or_safe_default(edge)

    return ScoreAdjustment {
        effective_prior = cache.p_eff[i],
        q_override      = cache.q_ctrl[i],
        penalty         = virtual_loss_penalty(edge.virtual_visits),
        fisher_alpha    = 0.0
    }
```

Stop rule은 이렇게 간다.

```text
BQ++ should_halt(snapshot, edges):

    if snapshot.root_visits >= max_visits:
        return Stop(MaxVisits)

    if snapshot.elapsed_ms >= time_cap_ms:
        return Stop(MaxTime)

    cache = atomic_load_policy_cache()

    if cache.edge_version_hash != current_edge_version_hash(edges):
        return Continue  # never stop from stale certificate

    if cache.forced_move_pos is Some:
        return Stop(TacticalForced)

    if snapshot.root_visits < min_total_from_budget(snapshot):
        return Continue

    if cache.cert_gap > 0:
        return Stop(EmpBernsteinCertified)

    if kl_lucb_enabled_for_calibrated_binary_backup:
        if fresh_kl_lucb_gap(edges, delta_move) > 0:
            return Stop(KLLUCBStop)

    if cache.max_kg_per_ms < compute_value_threshold
       and root_entropy(cache.p_eff) < entropy_threshold
       and posterior_switch_probability(edges) < switch_threshold:
        return Stop(PolicyConverged)

    return Continue
```

---

## 9. Hidden win move / local minima escape 설계

네 목표에서 제일 중요한 부분 중 하나가 “hidden win move detection”인데, 현재 문서의 BayesianQuartz는 이 부분이 약하다. 단순 KL-LUCB stop은 이미 방문한 arms 사이의 best certification에는 강하지만, low-prior action이 아예 후보군에 못 들어오면 아무것도 못 한다.

그래서 BQ++에는 네 가지 escape channel이 있어야 한다.

첫째, **Gumbel without replacement root sampling**. 낮은 prior action도 Gumbel perturbation으로 가끔 후보군에 들어온다. 이건 AlphaZero/MuZero few-simulation 문제에 대해 이미 직접적인 외부 근거가 있다. ([OpenReview][4])

둘째, **upper-confidence challenger injection**. (U_a>L_b)인 action은 prior가 낮아도 제거하면 안 된다.

셋째, **tactical sentinel**. Gomoku라면 open-four, double-three, immediate win/block 같은 pattern detector를 CPU에서 매우 싸게 돌릴 수 있다. Chess라면 legal move generator 위에 checkmate-in-1, hanging king, forced capture/check extension 같은 sentinel을 둘 수 있다. Go는 더 어렵지만 ladder/atari/local capture 정도는 cheap tactical feature로 넣을 수 있다.

넷째, **nested-reservoir**. Nested sampling 자체를 쓰는 게 아니라, “live candidates under score threshold” 구조만 가져온다. Skilling의 nested sampling은 likelihood-prior-mass 관계를 직접 추정하고 phase-change 문제에 강하다는 점이 핵심인데, root action search에서는 이를 evidence estimator가 아니라 multi-modal candidate 유지 장치로 축소해야 한다. ([Semantic Scholar][8])

---

## 10. CPU-friendly 검토

현재 문서의 CPU cost 평가는 너무 낙관적이다. “mutex read 50ns, overhead 0.2%” 같은 수치는 실제 multi-threaded MCTS에서는 믿으면 안 된다. 특히 parallel MCTS는 tree/root/leaf parallelization, mutex, virtual loss가 성능과 search quality에 직접 영향을 준다는 고전적 문제가 있다. Parallel MCTS 문헌도 tree parallelization에서 local mutex와 virtual loss 처리가 중요하다고 본다. ([DKE Maastricht University][9])

BQ++의 CPU 원칙은 다음이다.

Hot path에서는 allocation 없음, mutex 없음, dynamic dispatch 최소화. `score_adjustment()`는 edge-local index로 precomputed cache array를 읽는 O(1) 함수여야 한다. `observe()`는 check interval마다 O(K) 또는 O(K + m²)로 돈다. (K=361), (m\le 32)라면 CPU에서 충분하다.

구현 권장사항은 이렇다.

* `SmallVec`를 매 observe마다 새로 만들지 말고, fixed-capacity `Vec<f32>` buffer를 재사용한다.
* `statrs::ChiSquared::inverse_cdf`는 hot-ish path에서 빼고, threshold table 또는 Wilson–Hilferty approximation을 쓴다.
* KL bisection은 모든 arm이 아니라 certificate 후보 top (m)에만 쓴다.
* Welford `M2`와 누적 mean은 f64, score path는 f32로 간다.
* cache publish는 `ArcSwap<PolicyCache>` 또는 double-buffer RCU로 한다.
* `edge_pos`와 `action_id`를 절대 혼동하지 않는다.
* `observe()` writer는 하나만 허용하고, 나머지 worker는 stale-safe read만 한다.
* stop decision은 반드시 fresh epoch에서만 허용한다.
* GPU 부담 감소는 “simulation 수”가 아니라 `nn_evals_per_move`, `batched_eval_latency`, `gpu_seconds/game`으로 측정한다.

---

## 11. 하이퍼파라미터 제거/축소

현재 BayesianQuartz도 hyperparameter가 많다. δ, λ₀, min_pulls, min_total, max_visits, time_cap, chi2_alpha, voi_cost_floor, eval_uncertainty_kappa가 있다. 문서가 Legacy보다 줄인 건 맞지만 “hyperparameter-free/single-principle”은 아니다. 

BQ++에서는 다음처럼 파생값으로 바꾼다.

[
\delta_{\mathrm{move}}
======================

\frac{\delta_{\mathrm{game}}}{1+T_{\mathrm{remaining}}}
]

또는 더 완만하게

[
\delta_{\mathrm{move}}
======================

\frac{\delta_0}{(1+\mathrm{ply})^\alpha}
]

(\lambda_0)는 고정값이 아니라 calibration에서 온다.

[
\lambda_0
=========

\operatorname{clip}
\left(
\frac{1}{\mathrm{ECE}*{\mathrm{value}}+\epsilon},
\lambda*{\min},
\lambda_{\max}
\right)
]

value network calibration error가 크면 prior pseudo-count를 약하게, 안정적이면 강하게 둔다.

`min_pulls`는 없애고 warmup candidate schedule로 바꾼다.

[
m_{\mathrm{root}}
=================

\min\left(K,\max(4,\lfloor B_{\mathrm{visits}}/B_{\mathrm{per_candidate}}\rfloor)\right)
]

`voi_cost_floor`는 고정 (10^{-3})이 아니라 실제 latency에서 온다.

[
c_{\mathrm{eval}}
=================

\mathrm{median_eval_ms}
\times
\frac{\partial U}{\partial \mathrm{ms}}
]

(\partial U/\partial \mathrm{ms})를 모르면 self-play/arena calibration에서 “extra 100 ms가 Elo 또는 winrate에 주는 marginal gain”으로 추정한다.

(\tau_t)는 entropy target에서 온다.

[
H(P_{\mathrm{eff}})
===================

H_0\exp(-N/B)
]

초기에는 넓게, 후반에는 좁게 본다. 이게 인간식 “처음 후보를 넓게 보고, 나중에 소수 후보를 깊게 계산”하는 구조다.

---

## 12. 단계적 업그레이드 계획

### Phase 0. Claim freeze와 수식 정리

먼저 문서의 claim을 고쳐야 한다.

* `229,376` configuration count 수정.
* Beta-Binomial vs Normal-inverse-Gamma 용어 분리.
* Pearson χ²를 formal p-value test가 아니라 adaptive prior-surprise diagnostic으로 downgrade.
* KL-LUCB guarantee를 idealized calibrated root-bandit assumption 아래로 제한.
* VOI 식을 full expected improvement / KG approximation으로 교체.
* EB gap을 (L_b-\max U_a)로 교체.
* concurrency monotonic claim 삭제.
* “Keldysh/Jarzynski/IIT에서 직접 유도” 같은 표현 삭제. “free-energy/log-sum-exp/regularized planning analogy”로 낮춘다.

Exit criterion: 문서 안의 모든 수식이 scale, sign, stopping condition과 호환되어야 한다.

### Phase 1. Numerical primitive test suite

코드 전에 pure Python 또는 Rust unit test로 primitive를 고정한다.

테스트 목록:

* Welford variance vs two-pass variance.
* shrinkage variance (n=0,1,2,50).
* EB width scale: ([0,1]) range와 ([-1,1]) range 비교.
* EB certificate: synthetic two-arm case에서 (L_b>U_c).
* VOI/EI: (\Delta=0), (\Delta=s), (\Delta=3s) hand value.
* KG approximation monotonicity: 더 불확실한 challenger의 KG가 더 커야 함.
* prior smoothing: sum exactly 1, zero-prior action finite.
* χ²/surprise: adaptive count라 p-value claim 금지, only scalar residual.
* index mapping: action id sparse case.
* stale cache: changed edge hash이면 stop 금지.

Exit criterion: 모든 수식 test가 1e-6 또는 명시 tolerance로 통과.

### Phase 2. CPU cache architecture

`SearchPolicy` hot path에 lock-free immutable cache를 넣는다.

* `PolicyCache`는 edge-local indexed arrays만 가진다.
* `observe()`가 새 cache를 만들고 atomic publish한다.
* `score_adjustment()`는 cache read + array indexing만 한다.
* cache miss/stale이면 pure PUCT fallback.
* Criterion benchmark를 만든다.

Exit criterion:

* `score_adjustment()` overhead가 pure PUCT 대비 1–2% 이하.
* `observe()` (K=361)에서 check당 목표 microsecond budget 이하.
* allocation count가 hot path에서 0.

### Phase 3. Gumbel Sequential Halving root scheduler

이 단계가 simulation reduction에 제일 중요하다.

* Root candidate set을 `top_prior ∪ gumbel_top_m ∪ top_upper ∪ tactical`로 만든다.
* Budget이 고정이면 Sequential Halving bracket을 사용한다.
* Anytime budget이면 resumable bracket을 사용한다.
* 후보 탈락은 raw mean이 아니라 (U/L/KG) 조합으로 한다.
* Gumbel noise는 training/self-play에서는 사용하고, evaluation에서는 deterministic seed 또는 disabled/controlled mode를 둔다.

Exit criterion:

* 낮은 visit budget에서 기존 PUCT보다 top-1 policy degradation이 줄어야 한다.
* low-prior hidden tactical move fixture에서 recall이 올라야 한다.

### Phase 4. VOI/KG stop rule 통합

BayesianQuartz의 `voi_cost_floor`를 버리고 actual compute cost 기반으로 간다.

* median eval latency, CPU selection latency, batch wait latency를 telemetry로 기록.
* KG per ms를 계산한다.
* `max_kg_per_ms < threshold`이고 entropy 낮고 switch probability 낮으면 stop.
* EB/KL certificate가 있으면 즉시 stop.
* KG와 certificate가 충돌하면 telemetry에 conflict reason 기록.

Exit criterion:

* 같은 move quality에서 `nn_evals_per_move` 감소.
* early stop 때문에 tactical blunder가 늘지 않아야 함.
* stop reason distribution이 해석 가능해야 함.

### Phase 5. Hidden win / tactical sentinel

게임별 cheap solver를 넣는다.

Gomoku:

* immediate five
* opponent immediate five block
* open-four
* double-three
* forced threat chain shallow detector

Chess:

* legal checkmate-in-1
* forced check/capture extensions
* illegal king safety hard guard
* transposition/repetition guard

Go:

* legal suicide/ko guard
* atari/capture local features
* ladder-like shallow heuristic은 optional, 과장 금지

Exit criterion:

* tactical suite에서 hidden win recall 측정.
* sentinel false positive가 final move를 망치지 않아야 함.

### Phase 6. Nested-reservoir escape

Nested sampling을 evidence estimator가 아니라 live candidate maintenance로 구현한다.

* live set size (m).
* score (\Lambda_a=U_a+\rho KG_a+\tau\log\pi_a).
* 하위 quantile 제거.
* unexplored/low-prior/high-uncertainty에서 Gumbel/Thompson replenishment.
* search가 narrow collapse하면 broaden phase 재진입.

Exit criterion:

* local-minimum fixture에서 PUCT/BayesianQuartz보다 recovery rate 증가.
* overhead O(K) 유지.

### Phase 7. MENTS/free-energy optional branch

MENTS는 default가 아니라 opt-in으로 둔다.

* soft Bellman backup을 root 또는 shallow depth에만 실험.
* temperature는 entropy target으로 조정.
* 기존 Q backup과 soft backup을 telemetry에서 분리.
* sample efficiency는 synthetic + Gomoku short-budget에서 먼저 검증한다.

MENTS는 maximum entropy policy optimization과 softmax value backup으로 UCT 대비 sample efficiency를 개선하려는 문헌적 근거가 있지만, AlphaZero-style NN value와 섞으면 calibration 문제가 생길 수 있으므로 opt-in이 맞다. ([NeurIPS Papers][6])

### Phase 8. Research-grade experiment protocol

최종적으로 비교는 다음 축으로 해야 한다.

* Pure AlphaZero PUCT fixed budget
* LegacyQuartz
* BayesianQuartz original P09
* BQ++ without Gumbel
* BQ++ with Gumbel SH
* BQ++ with Gumbel SH + tactical sentinel
* BQ++ full nested-reservoir

핵심 지표:

* winrate / Elo with paired seeds
* top-1 agreement with high-budget oracle
* root simple regret on known tactical fixtures
* hidden win recall
* blunder rate
* simulations per move
* NN evals per move
* GPU seconds per game
* CPU overhead per selection
* stop reason distribution
* cache stale-stop attempts
* calibration ECE/Brier score
* prior surprise residual

Exit criterion은 이렇게 잡으면 좋다.

* GPU evals per move 30% 이상 감소.
* 같은 wall-clock에서 winrate non-inferiority.
* tactical hidden-win recall 감소 금지.
* hot-path overhead 2% 이하.
* 모든 stop decision에 fresh certificate 또는 KG/cost rationale 기록.

---

## 13. 최종 문서에 넣어야 할 핵심 문장

업그레이드된 설계문서의 중심 문장은 이렇게 써야 한다.

“BQ++는 물리학적 양자 유비를 실제 양자역학 주장으로 사용하지 않는다. Search ensemble은 trajectory path measure이고, free-energy는 NN prior와 posterior computation value 사이의 KL-regularized decision objective다. Controller는 expected reduction in posterior simple regret per unit compute를 최대화하는 computation을 선택하며, empirical confidence certificate 또는 non-positive computation value가 성립할 때 search를 멈춘다.”

그리고 novelty claim은 이렇게 낮춰야 한다.

“BQ++의 novelty는 Welford, KL-LUCB, empirical Bernstein, Gumbel SH, MENTS, nested sampling 자체가 아니라, AlphaZero-style NN-guided MCTS root control에서 이들을 하나의 bounded-rational metareasoning objective 아래 CPU-friendly하게 결합하고 telemetry로 falsifiable하게 만든 engineering integration이다.”

이 문장은 문서의 기존 novelty discipline과도 잘 맞는다. 문서 자체도 BayesianQuartz의 구성요소가 새 수학은 아니고, novelty는 AlphaZero MCTS root control에 통합하는 데 있다고 말하고 있다. 

---

## 14. 실행 체크리스트

바로 다음 작업은 이 순서로 가면 돼.

1. 설계문서 수식 수정: config count, VOI, EB gap, KL-LUCB guarantee scope, χ² scope, concurrency claim.
2. BQ++ objective를 문서 맨 앞에 추가: “expected decision-loss reduction per compute cost.”
3. BayesianQuartz P09 pseudocode를 BQ++ pseudocode로 교체.
4. CPU cache 설계 확정: edge-local indexing, immutable cache, no mutex hot path.
5. primitive numerical tests 먼저 작성.
6. Python prototype으로 synthetic bandit/root-MCTS fixture 검증.
7. Rust primitive 이식.
8. engine integration 전 Criterion benchmark.
9. Gumbel SH candidate scheduler 이식.
10. EB/KG stop rule 이식.
11. tactical sentinel 이식.
12. nested-reservoir는 마지막에 opt-in으로 추가.
13. 실험은 Gomoku 7×7 short-budget → Gomoku 15×15 → Go 9×9 → Chess 순서로 확장.
14. claim은 winrate 전까지 “simulation/eval reduction with non-inferiority target”으로 제한.

이렇게 가면 “물리학 수사로 멋있는 controller”가 아니라, 실제로 코드 이식 가능한 CPU-friendly search controller가 된다. 핵심은 Keldysh/Jarzynski/IIT를 hot path에서 빼고, 그 자리에 rational metareasoning + Gumbel SH + calibrated confidence + tactical reservoir를 넣는 거야.

[1]: https://arxiv.org/abs/1712.01815 "https://arxiv.org/abs/1712.01815"
[2]: https://people.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf "https://people.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf"
[3]: https://arxiv.org/abs/1706.02986 "https://arxiv.org/abs/1706.02986"
[4]: https://openreview.net/forum?id=bERaNdoegnO "https://openreview.net/forum?id=bERaNdoegnO"
[5]: https://proceedings.neurips.cc/paper/2013/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html "https://proceedings.neurips.cc/paper/2013/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html"
[6]: https://papers.neurips.cc/paper/9148-maximum-entropy-monte-carlo-planning "https://papers.neurips.cc/paper/9148-maximum-entropy-monte-carlo-planning"
[7]: https://arxiv.org/abs/cond-mat/9610209 "https://arxiv.org/abs/cond-mat/9610209"
[8]: https://www.semanticscholar.org/paper/Nested-sampling-for-general-Bayesian-computation-Skilling/7783ccc95726972c5f12cdc610b8f5bf53deb0b2 "https://www.semanticscholar.org/paper/Nested-sampling-for-general-Bayesian-computation-Skilling/7783ccc95726972c5f12cdc610b8f5bf53deb0b2"
[9]: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf "https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf"