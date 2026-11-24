# C01–C10 Rust 실행

## C01 — 고정 기준

대조군은 C29 576cbe554ea0의 full horizon 및 EPS 선택. 동일 binary, 같은 입력, 같은 C26 predictor와 rollout 노드를 비교한다. 우선 동일한 관측, 최종 move, 계획 value와 utility를 요구한다. 비용 차이는 paired resident timing으로 해석하고 판정 경계의 동점은 그대로 full path에 맡긴다.

## C02 — 소스와 경로

C29의 Rust 바이너리 소스를 exact blob에서 별도 이름으로 이식한다. C26의 정본 tree와 정상 엔진 경로는 사용자가 준 commit을 기준으로 한다. `Filter::perfect_information`은 local component와 q를 사용한다. `strict_evpi_stop`은 eligible 비용과 terminal value를 취하며 `plan`의 opt-in 분기에서만 호출한다. 게임에서 known terminal 및 초기 exact 계산을 먼저 condition하는 C29 corrected 경로를 보존한다.

## C03 — 수치 설계

Outside가 1에 가깝거나 비용이 충분히 크면 support ceiling 1만으로 STOP한다. 나머지 상태는 CDF O(N)으로 계산한다. 경계에서는 fail-open하고 `1e-8` 여유를 사용한다. 채널 행은 C29의 1e-12 계약이며, 연산 전체의 rigorous enclosure는 이 구현의 검증 범위에서 제외한다. 남은 horizon과 비용의 단위는 C29와 같다.

## C04 — 실행 계획

워크스테이션 Rust offline release에서 test 후 300개 고정 C29 retrospective 요청을 paired 실행한다. 같은 과정에서 planner의 전체 비용과 실제 rollout cost를 별도로 확인한다. 선행 관측을 사용한 retrospective 분석이며 새 TEST로 쓰지 않는다.

## C05 — 이식과 opt-in

`src/bin/quartz_factor_evpi.rs`는 576cbe5의 corrected C29 binary source를 이식한 연구용 진입점이다. `guard=true`인 JSON 요청만 CDF 조기 STOP을 평가한다. 원 경로에는 동일 후보·posterior·비용·EPS를 사용한다. opt-in 게임 응답은 full H2의 resident timing 및 trajectory 비교를 함께 반환한다.

## C06 — Rust build와 기본 테스트

Ryzen 9 5900X x86_64, rustc 1.99.0. `cargo test --locked --offline --release --bin quartz_factor_evpi` 8개 통과. 같은 옵션으로 release build 성공. C05 실행물 SHA256 d4078a721eefda955f6d56d703a64cd2cc94cbf31dbf8e78456cec72110d7751. 이식 소스에서 상수·필드 3개의 기존 dead-code 경고가 발생했다.

## C07 — 유리수 세계 교차검증

220개 무작위 모형 및 4개 비용 경계에서 full/guarded Rust planner를 정규화된 exact-world Fraction Bellman의 값·선택·완전정보값과 비교했다. 448개 응답이 조건을 만족했다. 최대 절대 오차 1.11e-16. ±1e−9 및 동점은 full planner로 넘겼다. `verify_native.py`가 독립 계산과 native 호출을 재현한다.

## C08 — TicTacToe 비용 및 잔차

C29 frozen request 300개, root당 23회 resident 비교에서 300/300 final move, rollout trace, 비용과 계획값이 일치했다. 초기 실행에서 H2 14.768→7.473µs, initial-only 2.851µs였고, C09 계측 수정 후 재측정한다. 두 H2 모두 C29 원래의 root planned_value 계산 및 round 0 재계획을 보존한다.

별도 Python exact minimax 및 균일 random-rollout law는 (ply, focal utility) 6 strata 중 5개에서 서로 다른 law를 확인했다. TRAIN 180개로 적합한 sibling-best 변수를 소급 117개 root에 평가했을 때 root 평균 log-score가 0.00558 bit 개선됐다. 2,000회 root bootstrap의 기술 구간 [0.00086,0.01107]. 과거 source이며 루트 간 공유 후손을 고려한 확증 구간이 아니다. 정확 sibling minimax는 채널 진단에만 사용했고 native 입력에는 주지 않았다.

## C09 — 별도 검토와 계측 수정

Native reviewer factorcodeaudit은 complete worlds에서 비정규화 Fraction Bellman을 따로 계산하여 224개 모델의 값·선택을 교차검증했다. 실제 질의가 없는 node도 초기 counter에 포함되던 점을 발견했다. 빈 menu는 이제 guard count에서 제외하고, root pruning 플래그를 명시한다. 비교하지 않은 응답은 equivalent를 null로 내보낸다. 두 native 회귀 테스트와 frozen C29 응답으로 300개 추적 일치 조건을 추가했다. f64 guard margin은 audited numerical guard로 범위를 유지한다.
