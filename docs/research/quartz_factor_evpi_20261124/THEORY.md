# 잠재요인 posterior의 완전정보 STOP

## R01 — SSOT와 질문

직접 부모는 C26 87c15dc828d6eea519c219fd8a412fa21dfc464d, tree 021a61b574557378ed5917322e10c9884d3bdc24다. 직전 45722dcc의 세계열거 oracle 및 C29 576cbe554ea0의 수정된 noisy filter는 비교·이식 자료다. 이번 질문은 고정 효용에 대한 완전정보 상계를 C29의 2성분 posterior에서 선형 비용으로 계산해 H2 STOP을 재현하며 전체 시간을 줄일 수 있는지다. 실제 게임 효용 우위는 새로운 root 및 비용 보정이 필요한 별도 주장이다.

## R02 — 선행 근거

Hay et al., UAI 2012 정의 1–2 및 정리 4–5는 유료 metalevel MDP와 완전정보 비용 상계를 정식화한다. Callaway et al., UAI 2018 §3 식 (2)–(4)는 myopic VOI, VPI, 부분 VPI를 나눈다. 이번 기여는 C29 고정 효용 posterior에서 VPI를 선형 비용으로 계산하고 native 경로에서 검사하는 것이다. C24의 SPD 응답 오차 certificate와 적용 대상이 다르다.
- https://www.auai.org/uai2012/papers/123.pdf
- https://cocosci.princeton.edu/papers/callawayLearningToSelect.pdf

## R03 — 원 동기와 가정

최초 커밋 15f3a198의 README와 QUARTZ_THEORY.md는 root risk, P_flip과 비용 차감 VOC를 명시한다. 3bcaa435의 원 논문 §5.1–5.2는 실제 시간을 EMA로 측정하고 기대 판단 개선과 비교했다. 이번 연구는 같은 후속 결정과 STOP을 보존하며 planner 비용을 직접 잰다. 단순 hard H2이고, Eq. 1 KL/Gibbs 동역학과 실게임의 compute-matched 성능은 별도 과제다.

조건은 고정 효용 벡터, U_i∈{0,1/2,1}, 상수 outside b∈[0,1], 비음수 미래 계산 비용, 현재 history에서 Z를 조건으로 독립인 후보별 효용이다. 편의상 후보 N=0도 허용한다. 유틸리티·비용 단위는 dimensionless WDL 효용이다. 채널에서 관측된 신호를 실제 효용으로 보지 않는다.

## R04 — 구성 선택

World enumeration O(3^N)과 조건부 CDF 두 개 O(N)를 비교한다. 다른 선형 상계인 sum_i E[(U_i−b)_+]는 양의 이득이 동시에 가능하면 느슨하다. 예를 들어 독립 fair Bernoulli 두 후보, b=1/2일 때 실제 perfect-information 개선은 3/8, 이 합 상계는 1/2다. 먼저 전체 STOP만 조기 반환한다. 질의별 생략은 C29의 score 공개와 EPS tie를 바꾸기 쉬우므로 별도 과제로 둔다.

## R05 — 유도

F_t=Σ_z q_z Π_i F_{i,z}(t), F_{i,z}(t)=Σ_{u≤t}π_{i,z}(u). 전체 max를 취하는 사건은 모든 후보가 t 이하인 교집합이고, 조건부 독립에 의해 곱이다. X=max(b,U_1,...,U_N)에 대해 E[X]=b+∫_b^1 P(X>t)dt. c=(1/2−b)_+, e=1−max(b,1/2)이면

M=b+c(1−F_0)+e(1−F_{1/2})=1−c F_0−e F_{1/2}.

임의 정책은 세계별 최대 효용보다 높은 결과를 낼 수 없으며, i로 시작했으면 적어도 c_i를 지불한다. 그러므로 Q_H(i)≤M−c_i. B=max(b,max_i E[U_i]) 혹은 known-only terminal value라 하면 eligible 질의 최소 비용 c_min에서 M−c_min≤B인 경우 STOP이 optimal이다. 빈 eligible 집합도 직접 STOP한다. 전체 likelihood P(Y|U_i,Z,h)는 해당 후보의 local factor와 q_z만 갱신해 같은 계산식을 유지한다.

## R06 — 독립 검토와 실패 범위

Native reviewer cdfaudit가 CDF, 빈 곱, b=1, q_z=0, C29 EPS tie 및 채널 행 오차를 확인했다. 두 fair Bernoulli 후보에서 XOR=1 관측은 후보 주변분포를 그대로 두면서 E[max]=1로 만든다. 이를 independent product로 잘못 압축하면 3/4가 된다. 관측모형이 임의여도 완전정보 부등식은 유지되나 현재 posterior가 factor contract를 위반하면 이 CDF 계산은 상계가 아니다. P(Y|U_i,Z,h)는 factor contract를 유지해도 U_i만 공개하는 H1 상계를 깨뜨릴 수 있다. C29의 coarse channel 자체는 어느 계약의 실제 타당성도 판정하지 못한다.

## R07 — 실행 검산

Fraction으로 192개 혼합 posterior를 세계 열거와 교차검증했다. Native reviewer는 별도 280개 exact rational fixture의 일치를 확인했다. 빈 후보, zero component weight, outside=0/1 및 1/2 양쪽을 포함한다.

## R08 — 근거 판정

실수 모델의 CDF 식과 STOP 지배조건은 유도 완료다. C29의 f64 연산은 채널 행 합 tolerance, posterior update, EPS 선택 규칙을 포함한다. 구현에서는 수치 여유를 둔 엄격 비교를 쓰며, 그 가드는 검산 범위에서의 수치적 방어다. 전 입력에 대한 IEEE interval proof 및 실제 game belief의 calibration은 열린 상태다.
