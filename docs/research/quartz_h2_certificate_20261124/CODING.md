# 유한 세계 exact reference와 상계 적용

## C01 — 비교 기준

C26의 코드와 고정 캠페인에는 변경을 가하지 않는다. C29 H2 사례의 543/800 및 branch별 선택을 외부 회귀 fixture로 쓴다. 동일 입력의 exhaustive 정수 Bellman 및 별도 normalized Fraction oracle을 기준으로 exact value와 전체 선택 정책의 일치를 요구한다. 총 planner 시간을 비롯해 전개한 관측 수와 상계 계산 수를 기록한다.

## C02 — 식과 함수

`Problem`은 세계 행별 root-player utility, prior mass, 고정 outside utility 및 query의 utility-indexed 채널을 받는다. `_solve`는 Bellman score를 저장한다. `_stop`, `_perfect_worlds`, `_perfect_query`는 각각 STOP 및 두 상계의 질량 가중 점수다. 반환 `Plan`은 exact Fraction 값과 outcome별 후속 계획이다. 게임 label이나 oracle을 runtime에 연결하지 않는다.
