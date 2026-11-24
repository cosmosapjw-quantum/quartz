# 유한 세계 exact reference와 상계 적용

## C01 — 비교 기준

C26의 코드와 고정 캠페인에는 변경을 가하지 않는다. C29 H2 사례의 543/800 및 branch별 선택을 외부 회귀 fixture로 쓴다. 동일 입력의 exhaustive 정수 Bellman 및 별도 normalized Fraction oracle을 기준으로 exact value와 전체 선택 정책의 일치를 요구한다. 총 planner 시간을 비롯해 전개한 관측 수와 상계 계산 수를 기록한다.

## C02 — 식과 함수

`Problem`은 세계 행별 root-player utility, prior mass, 고정 outside utility 및 query의 utility-indexed 채널을 받는다. `_solve`는 Bellman score를 저장한다. `_stop`, `_perfect_worlds`, `_perfect_query`는 각각 STOP 및 두 상계의 질량 가중 점수다. 반환 `Plan`은 exact Fraction 값과 outcome별 후속 계획이다. 게임 label이나 oracle을 runtime에 연결하지 않는다.

## C03 — 산술과 검증 설계

입력과 채널의 정수격자는 예제 모형의 명시적 계약이다. 그 위에서는 정수 오버플로를 피하는 Python arbitrary precision을 쓴다. 비용과 utility에 signed integer tick을 사용하되 비용은 비음수다. 현 구현의 horizon은 0..2로 제한한다. exact-world complexity는 W개 세계, A개 대안, K개 질의, 최대 Y개의 outcome에서 대략 O(W A K²Y²)다. 상계도 세계를 순회하므로 pruning 수로 속도를 대신 보고하지 않는다.

## C04 — 실행 순서

Python 3.12의 표준 라이브러리로 JSON CLI, 독립 oracle, 결정 동등성 실험을 실행한다. 모형 적합이나 외부 rollout 수집은 필요하지 않다. exhaustive와 pruned의 비교는 각 fixture에 같은 JSON 계약과 tie 순서를 쓴다. wall-time은 같은 프로세스에서 실행 순서를 교대하고 중앙값을 기록한다. 반복 시계는 CPU 성능의 환경 의존 관측이다.

## C05 — 구현

`certificate.py` CLI가 fixture를 읽고 각 모드의 exact Fraction 값, STOP/deploy 및 query의 조건부 계획, 실행 카운터를 출력한다. 입력 fixture의 단위는 1 utility당 100 tick이다. 동일 후보의 반복 관측은 world row의 고정 latent utility를 계속 참조한다.
