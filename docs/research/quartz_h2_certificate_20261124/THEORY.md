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
