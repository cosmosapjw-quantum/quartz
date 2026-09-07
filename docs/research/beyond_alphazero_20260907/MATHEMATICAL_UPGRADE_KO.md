# QUARTZ의 직관을 유지하는 수학적 재정식화

작성일 2026-09-07. 기준 소스 `a74adb4e0176b0e7f0ad6cb7c1870a483b787749`.
이 문서의 등식은 명시된 확률모형에서의 유도다. 실제 신경망 MCTS의 관측이 그 모형을 만족한다는 주장과 구분한다. 실행 검산은 [연구 기록](RESEARCH_RECORD.md), [12개 검사](test_research_math.py), [원본 출력](evidence/math_checks.stderr), [소스 반례](evidence/source_probes.stdout)를 참조한다.

## 1. 먼저 무엇의 효용을 추정하는가

국면 s와 합법적인 루트 행동 a에 대해 θ_a를 최종 결정에 필요한 효용으로 둔다. θ_a를 정확한 minimax 값, 고정 rollout policy의 기대값, 고정된 신경망·검색 예산에서의 참조 추정량 가운데 무엇으로 정의했는지 실험마다 고정한다. 이 셋은 동일하지 않다. 고예산 검색의 출력은 더 많은 계산을 쓴 참조값이며, 자동으로 minimax 진실이 되지 않는다.

관측 이력과 실행 중인 계산 정보를 포함한 정보집합을 F_t, 조건부 belief를 B_t, 평균을 μ_a=E[θ_a|F_t]라 하자. 위험 중립적 최종 행동은 b=argmax_a μ_a다. posterior simple regret는

\[
R(B)=\mathbb E[\max_a\theta_a\mid B]-\max_a\mu_a.
\]

이는 θ와 같은 효용 단위를 가진다. 비용 C(c)는 초, 실제 NN 호출 수, 에너지 중 하나로 분리하여 측정한다. 초를 쓰면 λ는 효용/초이며, 비용을 효용으로 변환한 값은 λC(c)다. 호출 수와 GPU 시간을 하나의 수치처럼 더하지 않는다. λ는 환경의 기회비용 또는 고정 총예산의 Lagrange multiplier이지 단위 없는 만능 파라미터가 아니다.

목표는 현재 수를 정제하는 계산, 다른 후보/반박을 여는 계산, 다른 정밀도의 평가, 중단 가운데 고르는 것이다. 과거의 QFT·다봉형 탐색 직관은 이 계산 공간과 belief의 구조를 제안하는 데 쓴다. 각 개념은 관측 가능한 연산과 연결되어야 한다.

## 2. 계산가치의 정확한 정의와 현재 구현의 간극

계산 c의 결과를 Y_c라고 하자. 올바른 Bayesian conditioning과 동일한 θ에 대해 tower property를 쓰면

\[
\begin{aligned}
\operatorname{VOC}_1(c;B)
&=R(B)-\mathbb E_{Y_c|B}[R(B^{c,Y_c})]\\
&=\mathbb E_{Y_c|B}\max_a\mu_a^{c,Y_c}-\max_a\mu_a.
\end{aligned}
\]

따라서 정보량, 정책 이동량, 현재 불확실성은 이 기대 결정 개선과 같지 않다. 실제 실현값 `θ_{b_after}-θ_{b_before}`는 음수일 수 있다. 모형이 맞을 때 관측 전 평균 VOC가 음수가 아닌 것은 Jensen 부등식의 결과이며, 매번 좋은 수로 바뀐다는 뜻은 아니다.

현재 `prototype/bqpp_prototype/kg.py`와 `src/mcts/policy/kg_stop.rs`는 EI 함수의 모양을 사용하지만, 입력 표준편차를 현재 두 행동의 추정오차 합으로 잡는다. 아래에서 보듯 필요한 양은 **새 관측으로 이동하는 posterior mean의 표준편차**다. 두 양은 서로 다르며, 관측모형 없이 한쪽을 다른 쪽으로 바꿀 수 없다.

또한 현재 wrapper의 `KG[best]=0`은 일반적인 계산가치 정리가 아니다. 선두가 잘못 평가되었는지 확인하는 계산은 다른 행동으로 전환하게 해줄 수 있다. 기존 수치가 heuristic proxy로 쓰일 수는 있지만, cited Gaussian KG의 계산값으로 해석하는 것은 성립하지 않는다.

## 3. 상관 Gaussian belief: 정확히 검산 가능한 출발점

유한한 K개 후보에 대해

\[
\theta\mid B\sim\mathcal N(\mu,\Sigma),\qquad
Y_c=h_c^T\theta+\varepsilon_c,\quad
\varepsilon_c\sim\mathcal N(0,r_c),\quad r_c>0
\]

를 가정한다. 노이즈는 θ와 독립이고, 관측 사이에도 독립이며, h_c와 r_c는 계산을 선택할 때 알려져 있다고 가정한다. h_c=e_a는 특정 행동을 평가하는 경우다. 다른 h_c는 두 후보의 공통 잠재변수를 조사하는 연산일 수 있지만, 실제 검색의 어떤 연산이 이 likelihood를 구현하는지는 별도로 밝혀야 한다.

관측 후 조건부 평균·공분산은

\[
\mu^+=\mu+\frac{\Sigma h_c}{h_c^T\Sigma h_c+r_c}(Y_c-h_c^T\mu),\qquad
\Sigma^+=\Sigma-\frac{\Sigma h_ch_c^T\Sigma}{h_c^T\Sigma h_c+r_c}.
\]

관측 전에는

\[
\mu^+=\mu+u_cZ,\qquad
u_c\equiv u_c=\frac{\Sigma h_c}{\sqrt{h_c^T\Sigma h_c+r_c}},\qquad Z\sim\mathcal N(0,1).
\]

이때 `u_c`의 단위는 효용이다. μ,θ는 효용, Σ와 r은 효용 제곱(h가 무차원일 때)으로 차원이 맞는다. Σ가 양의 준정부호이면 갱신도 양의 준정부호다. r→∞이면 u→0, Σ→0이면 VOC→0이다.

두 후보만 있을 때 d=e_1-e_2, δ=|d^Tμ|, s_c=|d^Tu_c|로 두면

\[
\boxed{\operatorname{KG}(c)=s_c\varphi(\delta/s_c)-\delta\Phi(-\delta/s_c)}
\]

이며 s_c=0이면 0으로 연속 연장한다. 유도는 `max(x,y)=(x+y+|x-y|)/2`와 평균이 0인 이동을 사용하거나

\[
\int_{\delta/s_c}^{\infty}(s_cz-\delta)\varphi(z)\,dz
=s_c\varphi(\delta/s_c)-\delta\Phi(-\delta/s_c)
\]

를 적분하면 된다. δ=0이면 KG=s/√(2π), δ/s→∞이면 0이다. 이 유도는 **후보가 두 개일 때 정확**하다. 다수 후보에서 top-two만 남기는 것은 추가 근사이며, 제3 후보 또는 미관측 후보의 역전 가능성을 통제해야 한다.

독립 후보의 posterior variance를 v_a, 다음 관측 노이즈를 r_a라 하면 후보 a를 관측할 때

\[
s_c^2=\frac{v_a^2}{v_a+r_a}.
\]

관측하지 않은 b의 v_b를 여기에 더하지 않는다. 상관이 있으면 `(Σ_aa-Σ_ba)^2/(Σ_aa+r_a)`가 된다. v_a=σ²/(n_a+λ_0), r_a=σ²인 동일 노이즈 모형에서는 정확한 이동 표준편차가 O(n_a^{-1})다. 현재 gap 표준편차 O(n^{-1/2})를 쓸 경우 큰 예산에서 계산가치를 과대평가하는 스케일 차이가 생긴다. 다만 해당 σ²가 실제 likelihood noise인지부터 확인해야 하므로 이 식을 현재 backup 통계에 곧바로 대입하지 않는다.

상관 Gaussian KG 자체는 신규 이론이 아니다. [Frazier–Powell–Dayanik (2009)](https://pubsonline.informs.org/doi/10.1287/ijoc.1080.0314), 그리고 이를 MCTS에 적용한 [Sezener–Dayan (2020)](https://proceedings.mlr.press/v124/sezener20a.html)이 직접적인 선행연구다.

## 4. 왜 ‘분산이 큰 곳’과 ‘생각할 가치가 큰 곳’은 다른가

`std_a Q_a`는 행동 평균 사이의 분산이다. 두 값이 정확히 0,1로 알려져 있으면 이 값은 크지만 추가 epistemic uncertainty는 0이다. 두 평균이 모두 0.5이고 각 행동의 불확실성이 크면 action dispersion은 0이지만 무엇을 선택할지는 불확실하다.

효용 차이의 분산은

\[
\operatorname{Var}(\theta_a-\theta_b)=\Sigma_{aa}+\Sigma_{bb}-2\Sigma_{ab}.
\]

모든 행동에 동일한 불확실한 offset Z가 붙으면 개별 분산과 latent entropy는 커질 수 있으나 결정은 바뀌지 않는다. `Σ=[[1,.8],[.8,1]], h=(1,1)`인 대칭 공통모드 관측에서는 u_1=u_2이므로 KG=0이다. 모든 정보를 동일하게 유용하다고 보는 정보획득률 최대화가 결정 목적과 충돌하는 정확한 반례다.

반대로 선두의 불확실성을 줄이는 계산은 유용할 수 있다. 검사에서 μ=(.1,0), Σ=diag(1,.01), 선두 관측 r=1을 사용하면 양의 정확한 KG가 나오지만 기존 wrapper는 0을 반환한다. 이는 실행 가능한 구현 의미 반례이며, 기존 프로그램을 고쳤다는 뜻은 아니다.

## 5. 통계물리 직관을 살릴 수 있는 자유에너지와 요동 보정

p_a>0, Σ_a p_a=1인 기준 분포와 효용 단위의 τ>0을 두자. 유한 후보 집합에서

\[
F_\tau(\theta)=\tau\log\sum_a p_a e^{\theta_a/\tau}
=\max_{\pi\in\Delta_K}\{\pi^T\theta-\tau D_{\rm KL}(\pi\Vert p)\}
\]

는 정확한 Gibbs variational identity다. 최대화 분포는 `π_a∝p_a exp(θ_a/τ)`다. 여기의 τ는 알고리즘의 정규화 척도이며 실제 열역학적 `k_B T`나 플랑크 상수로 식별하지 않는다. 유한합을 쓰므로 연속 경로적분의 측도·경계조건을 암묵적으로 빌리지 않는다. [Grill et al. (2020)](https://proceedings.mlr.press/v119/grill20a.html)의 regularized MCTS와 관련되지만, **KL의 방향까지 동일한지는 별도 문제**다. 위 `KL(π||p)`를 모든 AlphaZero 정규화 공식과 동일시하면 안 된다.

평균 μ에서의 정책을 π라 할 때

\[
\nabla F_\tau=\pi,\qquad
\nabla^2F_\tau=\frac{1}{\tau}[\operatorname{diag}(\pi)-\pi\pi^T].
\]

작은 centered fluctuation ξ=θ−μ에 대한 2차 항은

\[
\begin{aligned}
\mathbb E F_\tau(\theta)
&\approx F_\tau(\mu)+\frac{1}{2\tau}\operatorname{tr}[(\operatorname{diag}\pi-\pi\pi^T)\Sigma]\\
&=F_\tau(\mu)+\frac{1}{4\tau}\sum_{a,b}\pi_a\pi_b\operatorname{Var}(\theta_a-\theta_b).
\end{aligned}
\]

이것은 원래 요동·곡률 직관을 유지하면서 어떤 요동이 중요한지를 정확하게 정한다. 공통모드는 소거된다. `Σ=ε²C`, 고정 τ와 유한 K의 Gaussian 모형에서는 홀수 centered moment가 0이므로 다음 비영 항은 O(ε⁴/τ³) 규모다. 일반 비대칭 분포는 3차 항을 따로 통제한다. 실제 사용에서는 `sd(θ_a−θ_b)/τ ≪ 1` 및 고차 항 검사가 필요하다. τ→0과 ε→0을 아무 순서로 취해도 되는 것이 아니며, 역전·다봉형 구간에서는 Gaussian 한 점 전개가 실패할 수 있다.

**중요한 범위:** 이 보정은 posterior-averaged soft value의 근사다. 이를 각 행동의 bonus 또는 정확한 VOC로 쓰는 규칙은 아직 유도되지 않았다. 근사의 오차를 줄이거나 계산 순위를 더 잘 예측하는지를 먼저 확인해야 한다. 정책 π의 반복 업데이트에 같은 evidence를 넣으면 likelihood를 중복 곱할 위험도 있다.

현재 B13의 `log π_a + curvature/max(Nπ_a,N_floor)`는 이 Hessian·공분산 식에서 유도되지 않는다. 그 항은 같은 support 안에서 방문정책을 수정하는 별도 heuristic이다. N_a→∞일 때 항이 사라진다는 사실만으로 올바른 1-loop 보정이 되지는 않는다. 방문하지 않은 후보의 확률이 0이면 B13은 그 후보를 발견할 수도 없다.

## 6. 트리·다봉형·기하를 실제 알고리즘으로 옮기는 조건

트리가 비순환이라는 것은 Hessian이 대각이라는 뜻이 아니다. 두 노드의 작용

\[
S(x,y)=\tfrac12(x^2+y^2)+\tfrac12(x-y)^2
\]

의 Hessian은 `[[2,-1],[-1,2]]`, determinant는 3이며 대각 원소 곱 4와 다르다. 비순환성은 sparse elimination/message passing에 도움이 될 수 있지만 결합을 없애지는 않는다. 경로/노드 사이 관계를 실제 `L_G` 등의 연산자로 정의해야 한다.

연속 latent variable z를 정말 도입했다면 제한된 Laplace 근사를 쓸 수 있다. `Z=∫exp[-S(z)/ε]dz`에서 각 minimum z_k의 Hessian H_k가 양의 정부호이고, 경계와 다른 saddle의 기여를 통제하면

\[
Z\approx\sum_k e^{-S(z_k)/\varepsilon}(2\pi\varepsilon)^{d/2}(\det H_k)^{-1/2}.
\]

음의 eigenvalue에 절댓값을 씌워 `log|det H|`를 만드는 것은 발산하는 실수 Gaussian 적분을 수렴시키지 않는다. 유한 이산 상태에서는 먼저 정확한 합이나 검증 가능한 혼합분포를 기준으로 삼는 편이 명료하다.

Wang–Landau/flattening은 후보 basin을 찾는 proposal 단계의 영감으로 남긴다. 정의할 항목은 탐색 상태공간, reversible proposal 또는 별도로 정당화한 비가역 절차, energy/score bin, density 추정량, 적응 step size, reweighting이다. 가치를 낮게 추정한 거대한 basin을 histogram 때문에 계속 방문하면 regret가 증가할 수 있다. 따라서 **mode discovery objective와 최종 decision objective를 분리하고 연결 비용을 측정**한다. [Wang–Landau 원 논문](https://arxiv.org/abs/cond-mat/0011174)의 density-of-states 목표와 θ_a의 최대화를 동일시하지 않는다.

정보기하는 `p(θ|z)`의 Fisher metric 또는 categorical policy simplex의 metric을 명시할 때 의미를 가진다. metric의 좌표 불변성이 root utility 개선을 보장하지는 않는다. 여기서는 latent basis와 basin 간 evidence 전달을 안정적으로 만드는 후보 기법이며, 독립 축 효과가 관찰된 뒤 사용한다. [IGO](https://jmlr.org/papers/v18/14-467.html)는 필수 비교 대상이다.

## 7. 큰 예산의 배치 계산가치

이미 실행 중인 계산 집합 P를 먼저 조건부로 포함하자. 새 batch C의 joint value는

\[
\operatorname{KG}(C\mid B,P)
=\mathbb E\max_a\mu_a^{P,C}-\mathbb E\max_a\mu_a^{P}.
\]

Gaussian linear model에서 batch observation matrix H와 noise covariance R를 쓰면

\[
\Sigma_C^+=\Sigma-\Sigma H^T(H\Sigma H^T+R)^{-1}H\Sigma.
\]

pending 결과를 아직 모르더라도, 올바른 선형 Gaussian 모형에서는 그 결과를 관측했을 때의 공분산을 미리 계산할 수 있다. 관측값을 얻었다고 꾸미거나 pending을 완료 횟수에 넣는 것과 다르다. correlated observation noise가 있으면 R의 비대각 성분도 필요하다.

두 후보에서 전체 batch가 평균 차이를 이동시키는 분산은

\[
s_C^2=d^T(\Sigma-\Sigma_C^+)d.
\]

P가 비어 있으면 앞의 g(δ,s_C)가 batch 전체의 정확한 KG다. P가 이미 있으면 결과를 보지 않고 고정한 Gaussian 설계 A에 대해 `s_A²=dᵀ(Σ−Σ_A)d`로 두고 추가 가치를 다음처럼 구분한다.

\[
\operatorname{KG}(C\mid B,P)=g(\delta,s_{P\cup C})-g(\delta,s_P).
\]

이는 pending 결과에 대해 적분한 기대 최대값의 차이다. `g(δ,s_C)`만을 기존 pending 위에 더하는 식이 아니다. P의 관측값을 받은 뒤 C를 고르는 적응 설계라면 그 결과에 대해 바깥 기대값을 추가한다. 각 계산의 standalone KG를 더하면 중복된 정보를 여러 번 살 수 있다. 순차적으로 covariance만 갱신하는 greedy 방법은 중복을 줄이는 근사다. **일반 joint VOC가 submodular이거나 greedy가 전역 최적이라는 주장은 하지 않는다.**

현재 pilot은 두 finalist와 좌표 관측만 다룬다. 이 특수한 경우 고정 δ에서 g가 s에 대해 단조 증가하므로 최대 KG 선택과 최대 contrast-variance reduction 선택이 같다. 따라서 allocation이 관측된 평균에 의존하지 않는 것도 특수 모형의 결과다. 이것을 일반 MCTS의 적응적 의사결정 성능으로 해석하면 안 된다.

## 8. pilot의 정확한 위험식과 큰 예산 극한

선형 Gaussian likelihood가 올바르고 배분 설계가 관측값에 의존하지 않아 최종 posterior covariance가 결정론적이면, 초기 gap variance v_0=d^TΣd, 최종 posterior gap variance v_B=d^TΣ_Bd에 대해

\[
\boxed{\mathbb E R_B=g(\delta,\sqrt{v_0})-g(\delta,\sqrt{v_0-v_B})}.
\]

총분산 법칙으로 관측 후 평균의 prior-predictive variance가 v_0−v_B이고, `E maxθ−E maxμ_B`를 취하면 나온다. 따라서 Monte Carlo sampling error 없이 평균 regret를 계산할 수 있다. 잘못 지정된 covariance 시나리오에는 이 Bayesian 공식을 실제 위험으로 적용하지 않고 실현된 regret·coverage를 보고한다. 관측 결과에 따라 배분하거나 중단하는 일반 설계에는 이 단일 v_B 공식이 적용되지 않는다. 해당 posterior와 적응 과정에 대한 기대값을 직접 평가해야 한다.

독립 관측이 충분하고 초기 prior 영향이 줄어들면

\[
v_B\simeq\frac{r_1}{n_1}+\frac{r_2}{n_2},\quad n_1+n_2=B.
\]

Lagrange multiplier로 미분하면 `n_1/n_2=√(r_1/r_2)`이고 최소값은 `(√r_1+√r_2)²/B`다. 동일 배분은 `2(r_1+r_2)/B`다. pilot의 r=(.04,.25)에서 계수는 각각 .49와 .58이다. 알려진 heteroskedastic noise를 배분에 반영하는 것만으로도 큰 B에서 계수를 약 15.52% 줄일 수 있다. 이것은 **Neyman allocation의 알려진 이점**이며 Quartz의 novelty가 아니다. 그 강한 baseline을 포함해야 ‘큰 예산에도 남는 개선’과 ‘새로운 알고리즘’을 구분할 수 있다.

위 Bayes-risk 식을 작은 v_B에 대해 전개하면

\[
\mathbb E R_B=\frac{\varphi(\delta/\sqrt{v_0})}{2\sqrt{v_0}}v_B+O(v_B^2).
\]

따라서 절대 regret는 0으로 가면서도, 미리 정한 두 배분법의 상대 계수 차이는 남을 수 있다. ‘고예산 향상’은 더 좋은 무한예산 한계를 뜻할 필요가 없다. 반대로 동일한 known-noise optimum으로 수렴하는 방법 사이에는 이런 계수 우위도 남지 않는다.

## 9. 반복 검색의 진짜 병목: 독립 표본 수와 bias floor

예를 들어 반복값 `Y_i=θ+b+ε_i`, 공통 오차 Var(b)=τ_b², 독립 오차 Var(ε_i)=σ²이면

\[
\operatorname{Var}(\bar Y-\theta)=\tau_b^2+\sigma^2/n.
\]

n을 늘려도 공통 model error가 사라지지 않는다. 각 관측 분산 τ_b²+σ²로 정규화한 유효 표본 수는

\[
n_{\rm eff}=\frac{n}{1+(n-1)\rho},\qquad
\rho=\frac{\tau_b^2}{\tau_b^2+\sigma^2}.
\]

이것은 exchangeable additive-noise 모형의 예시이지 모든 MCTS에 적용할 보편 ESS 공식이 아니다. 결정에서는 공통 b가 모든 행동에 동일하게 붙는지, 행동 차이를 왜곡하는지도 구분해야 한다. 완전히 결정론적인 동일 NN 입력의 재평가는 새 epistemic evidence가 아니다. 다른 leaf·새 rollout도 공유 model error가 있으면 부분 의존한다. [EMCTS (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbd670951a6dede2123961de19e5ea3-Paper-Conference.pdf)는 이 문제를 직접 다룬다.

제안하는 latent belief는 `θ=μ_net+Lz+η` 형태로 시작할 수 있다. L은 작은 network가 표현한 공유 구조, z는 관측으로 갱신되는 저차 잠재변수, η는 국소 residual이다. 그 외 model discrepancy/drift는 독립적인 uncertainty 채널로 유지한다. checkpoint, root, player perspective, evaluator, depth/target definition, transposition identity가 달라질 때 어떤 posterior를 전달할지 정의한다. 동일 observation ID의 재사용은 likelihood를 다시 곱하지 않는다. 표현 L이 바뀌면 기본 동작은 reset이며, 전달은 별도 검증된 map이 있을 때만 허용한다.

이때 새로운 정보가 decision-relevant한 공통 latent coefficient를 교정하면 큰 예산에서도 효율이 좋아질 수 있다. 잘못된 covariance가 여러 후보로 오류를 전파하면 오히려 악화된다. 이 양면성이 핵심 반증 실험이다.

## 10. 정지 조건의 한계와 연구 순서

`max_c[KG(c)−λC(c)]≤0`는 한 단계 근시안적 정지 규칙이다. 메타수준 Bellman 방정식은 남은 예산을 포함하여

\[
V(B,b)=\max\{\max_a\mu_a,\ \max_{c:C(c)\le b}[\mathbb E V(B^{c,Y},b-C(c))-\lambda C(c)]\}
\]

처럼 미래 계산의 조합을 고려한다. 독립 Rademacher 두 bit Z_1,Z_2, 행동 효용 θ_0=0, θ_1=Z_1Z_2를 생각하자. bit 하나를 관측하면 KG=0이지만 둘을 관측하면 기대 최종 효용이 0에서 .5로 올라간다. bit당 비용 κ<.25면 두 단계 계산은 유익하다. 따라서 1-step KG=0은 일반적인 전역 정지 증명서가 아니다.

confidence certificate 역시 fixed-time CI를 매번 들여다보는 것으로 얻어지지 않는다. 고정 θ에 대해 adaptively selected bounded observations가 `E[X_t|F_{t-1}]=θ` 등의 필요한 조건을 만족할 때 적용 가능한 time-uniform bound를 골라야 한다. 진화하는 rollout·NN backup에 iid arm 보장을 그대로 붙일 수 없다. fresh snapshot, omission risk, pending work, target drift도 별도로 남는다.

실행 순서는 관측 의미와 정확한 작은 모형 → 효용 변화에 대한 held-out calibration → 한 개의 SAMPLE/REOPEN 개입 → pending batch 조건화 → 고예산·반복검색 성능 → 학습 재투자다. 자유에너지·곡률·다봉형 아이디어는 이 순서에서 구체적 실패를 해결하는 수학으로 유지한다.
