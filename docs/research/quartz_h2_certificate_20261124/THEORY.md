# 정확한 H2 계산 생략 조건

## R01 — 연구 질문과 기준

지정된 직접 부모는 C26 87c15dc828d6eea519c219fd8a412fa21dfc464d, tree 021a61b574557378ed5917322e10c9884d3bdc24다. 후속 C27/C28/C29는 비교 자료다. C27 C08에서는 120개 root의 joint/product 판단과 solve 노드가 같았고 joint wall 비용이 높았다. 교정된 C29의 60-root 소급 CAL은 평균 utility 차이 0과 4.599배 시간비를 보고했다. 질문은 주어진 고정 utility의 유한 noisy-query 모델에서 H2 최적 STOP/query 결정을 정확히 보존하며 평가 분기를 줄일 수 있는가다. 동일한 belief, 질의, 비용, tie 순서를 쓰는 full enumeration에 대해 정책과 최적값의 exact agreement를 요구한다. 신선한 게임 TEST나 Gomoku 검증으로 해석할 범위는 없다.

## R02 — 문헌과 기존 계산

Hay, Russell, Tolpin & Shimony (UAI 2012), Definitions 1–2와 Theorems 4–5에서 계산의 비용을 차감하는 metalevel MDP, posterior 기대 효용을 최대화하는 STOP, perfect-information 비용 상계를 재사용했다. 원문 https://www.auai.org/uai2012/papers/123.pdf. Blackwell (1953), DOI 10.1214/aoms/1177729032의 실험 순서에 해당하는 한 단계 부등식을 직접 유도한다. C26 NEXT_C27의 exact-known incumbent와 C29의 posterior terminal max는 다른 선택 계약이다. 이번 모형은 C29의 deploy-any terminal utility를 사용하고 C26의 결과를 변경하지 않는다.

## R03 — 가정과 원래 동기

유한 세계 w에 고정된 root-player 효용 U_a(w)와 비음수 posterior 질량이 있다. 대안 집합은 고정이며 상수 outside option을 포함한다. 계산 i는 효용을 변경하지 않으며, 질의별 비음수 비용 c_i, 정규화 채널 P(Y_i|U_i), 잠재 효용 조건부 독립인 반복 관측을 사용한다. 유도는 서로 상관된 후보 효용까지 포함한다. 효용과 비용은 같은 단위다. H=0이면 STOP이고 비용은 아직 발생하지 않은 계산만 차감한다.

초기 commit 15f3a198의 README와 QUARTZ_THEORY는 root uncertainty, P_flip과 계산 비용을 고려한 의사결정을 명시한다. 3bcaa435에 보존된 초기 PDF §5.1은 wall-time EMA, §5.2 Eq. 2는 myopic 가치, Eq. 1은 KL/Gibbs 제어를 쓴다. 이번에는 hard H2의 결정 근거를 보강한다. 실제 시간을 비용으로 보정하는 작업과 KL-regularized Bellman, TT 기하, Gomoku/NN P2 및 인지 P3는 열린 과제다.

## R04 — 상계 후보

m_a(h)=E[U_a|h], B(h)=max_a m_a(h), M(h)=E[max_a U_a|h].
상계 A는 모든 남은 지평에서 현재 질의 i의 가치에 M(h)−c_i를 사용한다. 상계 B는 잔여 지평 1에서 해당 U_i를 알려 주는 가상 질의의 기대 STOP에서 실제 c_i를 뺀다. A는 root에서 공통으로 재사용하고 B는 실제 첫 관측 이후에만 적용한다. 두 상계를 각 위치에서 함께 쓴다. posterior 분산 자체를 상계로 쓰는 후보는 action set과 비용을 반영하지 못한다. 예를 들어 모든 대안에 더해진 동일 변동은 최적 대안의 순서를 바꾸지 않는다.

## R05 — 구성적 유도

어떤 계산 정책도 실현된 최종 효용 U_{a(h_T)}(w)≤max_a U_a(w)를 만족한다. 첫 질의 비용 c_i를 이미 낸 정책의 나머지 비용은 비음수다. 기대값을 취해 Q_H(h,i)≤M(h)−c_i를 얻는다.

H=1일 때 Y_i는 U_i의 garbling이다. h를 고정하고 L_a(u)=E[U_a|h,U_i=u]라 하자. 채널 Markov 조건과 tower property로 E[U_a|h,Y_i]=E[L_a(U_i)|h,Y_i]. 좌표별 최대의 볼록성으로 max_a E[L_a(U_i)|h,Y_i]≤E[max_a L_a(U_i)|h,Y_i]. 관측 평균을 취하면 Q_1(h,i)≤E[max_a L_a(U_i)|h]−c_i. 가상 질의의 값과 비교할 때 비용은 실제 i의 비용이다.

STOP을 먼저 두고 질의를 고정 순서로 검사한다. 현재 최선가치 L은 이미 정확 평가한 실행 가능한 선택의 값이다. 상계 ≤L인 다음 질의는 최적값 및 첫 승자 tie 순서를 바꾸지 않는다. H2에서 살아남은 첫 질의의 실제 관측을 열고 각 child의 H1을 같은 방식으로 풀면 Bellman 값과 두 단계 선택이 보존된다. 신호가 발생할 때 posterior를 계산하며, 질량 0인 child는 합에 기여하지 않는다.

## R06 — 독립 검토와 반례

Native reviewer origin이 tower/Jensen, 비용 차감과 fixed alternative의 필요성을 별도로 확인했다. 첫 noisy 신호를 perfect reveal로 직접 대체하면 Y에 의존하는 후속 계약을 잃을 수 있으므로 H2에서는 실제 Y를 적분한다. 반례로 U_i=0, outside=0, U_j=±1 등확률인 상태에서 Y_i=sign(U_j)라 하자. U_i 정확 공개의 이득은 0이고 Y_i 이득은 1/2이다. 이 신호는 U_i만의 채널이 아니다. 공통 전술 상황을 반영하는 게임 rollout에서는 이 가정을 실증적으로 확인하거나 해당 상계를 비활성화해야 한다. 0 비용, 퇴화 효용 및 동일 최적값에서는 고정 tie 순서로 안전하다. 절대 비용과 효용에 같은 양의 스케일을 쓰면 정책이 보존된다.

## R07 — 정확 유리수 검사

독립적인 단일 후보 scalar recurrence를 실행했다. outside 3/5, Bernoulli(1/2), 정확도 3/4, 비용 1/100일 때 H0/H1/H2는 각각 3/5, 133/200, 543/800이고 H2 개선은 11/800이다. 첫 양성 뒤에는 계산을 계속하고 음성 뒤에는 STOP한다. 이 fixture는 C29의 기존 결과를 수치적으로 재현하며 pruning으로 이 행동을 잃으면 실패로 판정한다.

## R08 — 주장 판정

A/B의 admissibility와 tie 보존은 주어진 가정에서 유도됐다. R07의 지속 계산 사례는 exact rational로 검산됐다. 상계가 실제로 분기를 얼마나 자르는지, 상계 비용을 포함해 더 빠른지, 실제 게임 rollout에서 채널 가정이 맞는지는 별도의 실험 명제다. 유한 모델에서 계산량 감소가 작은 사례도 그대로 기록한다.

## R09 — 계산 계약

현재 비정규화 세계 질량 벡터를 w, 각 채널 분모를 공통 D, outcome별 질량 배율을 정수 k(y,u)라 한다. child 질량은 w_t k(y,U_i(t)). 남은 지평 H에서 점수 단위는 D^H이다. STOP score는 `D^H max_a sum_t w_t U_a(t)`. 첫 비용은 `D^H c_i sum_t w_t`. child들의 H−1 score를 합하면 채널의 1/D를 보정한 동일 단위가 된다. 상계 B의 numerator는 U_i 값별 그룹에 대해 `sum_u max_a sum_{t:U_i(t)=u} w_t U_a(t)`. 입력 검증은 채널 행 합, 효용 support, 비용과 질량, horizon을 검사한다. 정확한 정수 비교로 부동소수점 near-tie를 배제한다.
