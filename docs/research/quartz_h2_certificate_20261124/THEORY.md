# 정확한 H2 계산 생략 조건

## R01 — 연구 질문과 기준

지정된 직접 부모는 C26 87c15dc828d6eea519c219fd8a412fa21dfc464d, tree 021a61b574557378ed5917322e10c9884d3bdc24다. 후속 C27/C28/C29는 비교 자료다. C27 C08에서는 120개 root의 joint/product 판단과 solve 노드가 같았고 joint wall 비용이 높았다. 교정된 C29의 60-root 소급 CAL은 평균 utility 차이 0과 4.599배 시간비를 보고했다. 질문은 주어진 고정 utility의 유한 noisy-query 모델에서 H2 최적 STOP/query 결정을 정확히 보존하며 평가 분기를 줄일 수 있는가다. 동일한 belief, 질의, 비용, tie 순서를 쓰는 full enumeration에 대해 정책과 최적값의 exact agreement를 요구한다. 신선한 게임 TEST나 Gomoku 검증으로 해석할 범위는 없다.

## R02 — 문헌과 기존 계산

Hay, Russell, Tolpin & Shimony (UAI 2012), Definitions 1–2와 Theorems 4–5에서 계산의 비용을 차감하는 metalevel MDP, posterior 기대 효용을 최대화하는 STOP, perfect-information 비용 상계를 재사용했다. 원문 https://www.auai.org/uai2012/papers/123.pdf. Blackwell (1953), DOI 10.1214/aoms/1177729032의 실험 순서에 해당하는 한 단계 부등식을 직접 유도한다. C26 NEXT_C27의 exact-known incumbent와 C29의 posterior terminal max는 다른 선택 계약이다. 이번 모형은 C29의 deploy-any terminal utility를 사용하고 C26의 결과를 변경하지 않는다.
