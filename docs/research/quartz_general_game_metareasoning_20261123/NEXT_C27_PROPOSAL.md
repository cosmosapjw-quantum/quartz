# C27 제안 — 관측에 반응하는 공동신념과 2단계 계산가치

상태: C26 후속 계획만 작성. **C27 코드·적합·campaign 미실행.** 원 동기와 C26 primary 미통과를 보존한다.

## 한 가지 질문

같은 초기 주변분포·cue·초기 검증·계산 가격·horizon을 가진 두 controller 중, joint posterior를 갱신하는 쪽이 product closure보다 불필요한 확인 비용과 selected-query gain 오차를 줄이는가?

C26의 사후 증거는 선택 뒤 gain calibration의 문제다. 142/150root가 이미 최적이었으며 Pandora 추가 노드의85.91%가 그 집단에 쓰였다. 이것만으로 상관 누락이 원인이라고 확정하지 않는다.

## 수학 계약 후보

공통 cue \(X\), exact observation 이력 \(h=\{(a,u_a)\}\)에서 하나의 정규화된 공동분포를 갱신한다.

\[
P_\theta(\mathbf U\mid X,h)
=\frac{P_\theta(\mathbf U\mid X)
\prod_{(a,u_a)\in h}{\bf1}\{U_a=u_a\}}
{P_\theta(h\mid X)}.
\]

초기 incumbent의 exact reveal과 terminal 후보도 \(h\)에 포함한다. 불가능 관측의 분모0에는 명시적 model-support failure 처리가 필요하다. 관측값을 버리거나 oracle를 runtime input으로 넣지 않는다. positive smoothing·정규화·재현 가능한 update를 설계한다.

\[
V_0(h)=b_h,\quad
V_H(h)=\max\left\{b_h,\max_{a\in S_h}
\left[-d_a(h)+\sum_u P_\theta(U_a=u\mid X,h)
V_{H-1}(h\cup\{(a,u)\})\right]\right\}.
\]

우선 \(H=2\)다. 직접 incumbent를 개선하지 않아도 다른 후보를 구별하는 첫 질의를 표현한다. posterior update·branch evaluation·allocation의 controller 비용도 측정한다. 3점 효용과 \(m\)후보의 explicit depth2 enumeration은 outcome 쌍 기준 \(O(9m^2)\)개 작은 분기를 요구하며, posterior 계산의 추가 비용은 별도다.

행동 선택 Gibbs \(q(m\mid h)\propto\pi_0(m\mid h)e^{Q(m,h)/\tau}\)와 joint utility distribution은 다르다. 전체 regularized optimum이 목표라면 Bellman의 max를 prior-weighted log-sum-exp로 바꾸고 soft continuation을 재귀 사용해야 한다. 마지막 softmax만으로 soft Bellman 최적성을 주장하지 않는다.

## 최소 joint family

후보는 두 성분 latent mixture다.

\[
P_\theta(\mathbf U\mid X)=\sum_{z=1}^2w_z(X)\prod_a p_\theta(U_a\mid x_a,z).
\]

관측 후 latent weight가 바뀌어 미질의 후보도 함께 바뀐다. categorical predictor 전체를 두 벌 자유롭게 적합하지 말고 소수 공유계수와 독립 모형으로의 강한 수축부터 검토한다. 90TRAIN root에 unrestricted pairwise Potts 항을 붙이는 것은 우선 범위 밖이다. hidden mode·TT 기하라는 역사적 모티프만으로 그 모형이 정당화되지는 않는다.

공동모형의 최초 주변분포 \(p_a^\theta\)를 구한 뒤 대조군을 \(\prod_a p_a^\theta\)로 만든다. 따라서 두 controller의 초기 주변 예측은 같다. 기존 C26 model 비교는 별도 secondary다. 공유계수·수축강도·fit은 다음 주기 TRAIN 내부 root CV에서 결정하고 TEST 전에 동결한다. covariance 몇 항만으로 전체 joint law가 식별되는 것은 아니다.

## 필요한 경계 사례

1. 독립 곱에서는 다른 후보 reveal이 미질의 분포를 바꾸지 않아야 한다. H1은 one-step net, full horizon은 기존 independent Bellman/Pandora와 같아야 한다.
2. C26 반례 \(b=1/2\), equiprobable \((A,B,C)=(0,1,0),(1/2,0,1)\), 비용 \((.01,.3,.3)\)에서 H2는 A를 정보 질의로 선택하여 net terminal value0.69를 반환해야 한다.
3. 모든 terminal alternative와 비교 기준을 동일하게 이동시키는 순수 common shift는 argmax를 바꾸지 않는다. covariance가 양수라는 이유로 VOC를 부여하지 않는다. 고정된 별도 outside option이 있으면 가정이 달라진다.
4. exact-known incumbent retention은 joint prior가 틀려도 유지한다. 모델 정확도와 안전성을 분리한다.
5. Bayes normalization, support failure, root-player sign, 실제 solver/feature/planning 비용을 독립 reference로 검산한다.

## 최소 경험 비교·중단 조건

동일 초기 주변분포·H2의 frozen-product controller 대 updating-joint controller를 primary contrast로 삼는다. CAL을 보며 후보를 계속 바꾸지 않는다. 최종 가격과 비용 단위를 TEST 전에 하나로 고정한다. CPU 효율이 목표이면 native elapsed time에 controller 전체 비용을 포함하고 TRAIN에서 utility/time 변환을 먼저 고정한다. C26 node-price는 별도 연결 자료이며 node 감소만으로 CPU 효율을 선언하지 않는다.

Primary는 새 source-root 집합의 paired net-utility 차이와 사전 신뢰구간 기준이다. 보조 진단은 이미 최적인 root의 추가 비용, reveal 뒤 conditional Brier/selected-gain calibration, 실제 총시간이다. 여러 seed는 같은 root의 반복이다.

C26 TEST는 새 확증 표본으로 재사용하지 않는다. canonical symmetry·새 seed로 표본 수를 부풀리지 않는다. C26이 ply4/5/6에서 이미 많은 canonical root를 사용했으므로 같은 크기·층화의 새 집합이 남았다고 가정하지 않는다. 가용 root와 descendant 관계를 먼저 세고 필요하면 다른 source frame·더 큰 게임 이전을 별도 계약으로 정한다. 자료가 부족하면 외삽 성능을 보류한다.

작은 joint family가 TRAIN 내부 predictive score와 새 TEST 비용·gain을 개선하지 못하면 모드 수를 자동으로 늘리지 않는다. 그 제한된 가설을 기각/보류하고 error decomposition을 보존한다. P2·P3나 literal physical equivalence는 이 비교만으로 승격하지 않는다.

## 다음 턴 순서

C26 final receipt·clean HEAD 확인 → 원 repo 역사·원 동기 drift watchdog → R01–R10 전부 → 동결 계약으로 C01–C10 전부. 매 단계 backup·게시·remote identity를 확인하고 실제 산출물의 Git 포함 여부도 확인한다. C27 실행은 이번 주기에 시작하지 않았다.
