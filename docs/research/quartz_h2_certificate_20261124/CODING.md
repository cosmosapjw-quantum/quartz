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

## C06 — 실행과 계약 검사

`python3 check_certificate.py`의 5개 unittest가 종료 코드 0으로 통과했다. CLI, negative cost, invalid probability mass, malformed support, 0/잘못된 horizon, zero-weight 세계, outside-only와 tie를 포함한다. 독립 normalized Fraction oracle은 integer unnormalized 구현을 참조하지 않는다.

## C07 — 과학 검증

고정 seed로 생성한 240개 유한 문제 × horizon 0/1/2 × 3 모드에서 총 2,160개 값 및 전체 선택 계획이 독립 oracle과 정확히 일치했다. 범위는 signed 효용, 상관 세계, 중복 후보 질의, 0 비용과 impossible outcomes를 포함한다. 밖의 옵션이 평균에서 최고의 후보와 동일하면 outside tie 우선순위를 유지했다. 고정 latent 회귀에서 543/800와 양성 관측에서만 추가 계산하는 정책을 재현했다. 이 증거는 정의한 유한 query 모델에 귀속된다.

## C08 — 수치 비용과 재현성

`benchmark_certificate.py`는 고정 seed의 다섯 synthetic regime마다 full, world-only, both에서 값과 정책을 대조했다. 60회씩 동일 프로세스에서 순서를 교대하며 validation을 포함한 planner 시간을 쟀다. 관측 전개 수는 결정론적이다. 실행 시간은 측정 환경에 의존한다. 정확 산술로 truncation 및 roundoff가 없으며, 세계 상태와 관측 모델 자체의 근사 오차는 이 benchmark 범위 밖이다. 상세 수치는 C08_benchmark.json에 있다.

## C09 — 독립 검토와 수정

Native reviewer codeaudit가 정규화 없는 score scale, zero weights, tie와 pruning을 검토했다. 비재귀적으로 모든 두 번째 행동 매핑을 열거한 별도 검산에서 160개 signed 문제 × 3 horizon × 3 mode의 값 1,440건을 확인했다. 검토에서 JSON 키 "0"과 "00"이 정수 변환 중 충돌함을 재현했다. canonical key와 JSON 중복 키를 검사하도록 수정하고 CLI 회귀 2개 하위 사례를 추가했다. 전체 6개 unittest를 다시 통과했다. 이 수정은 benchmark 경로의 solver나 입력을 변경하지 않는다.

## C10 — 판정과 재개

Exact-world 툴은 소형 모형의 결정 기준으로 사용 가능하다. 두 상계의 일반적인 시간 이득은 입증되지 않았다. 특히 competitive 및 free에서 상계 추가로 elapsed 중앙값이 증가했다. README의 factor CDF 완전정보 상계를 먼저 구현하고 exact-world와 교차 검증한다. 원래 Quartz의 root 선택 개선은 fresh game data의 resource-matched 검증이 남아 있다.
