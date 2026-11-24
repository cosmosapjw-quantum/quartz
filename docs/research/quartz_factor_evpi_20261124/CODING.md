# C01–C10 Rust 실행

## C01 — 고정 기준

대조군은 C29 576cbe554ea0의 full horizon 및 EPS 선택. 동일 binary, 같은 입력, 같은 C26 predictor와 rollout 노드를 비교한다. 우선 동일한 관측, 최종 move, 계획 value와 utility를 요구한다. 비용 차이는 paired resident timing으로 해석하고 판정 경계의 동점은 그대로 full path에 맡긴다.

## C02 — 소스와 경로

C29의 Rust 바이너리 소스를 exact blob에서 별도 이름으로 이식한다. C26의 정본 tree와 정상 엔진 경로는 사용자가 준 commit을 기준으로 한다. `Filter::perfect_information`은 local component와 q를 사용한다. `strict_evpi_stop`은 eligible 비용과 terminal value를 취하며 `plan`의 opt-in 분기에서만 호출한다. 게임에서 known terminal 및 초기 exact 계산을 먼저 condition하는 C29 corrected 경로를 보존한다.

## C03 — 수치 설계

Outside가 1에 가깝거나 비용이 충분히 크면 support ceiling 1만으로 STOP한다. 나머지 상태는 CDF O(N)으로 계산한다. 경계에서는 fail-open하고 `1e-8` 여유를 사용한다. 채널 행은 C29의 1e-12 계약이며, 연산 전체의 rigorous enclosure는 이 구현의 검증 범위에서 제외한다. 남은 horizon과 비용의 단위는 C29와 같다.
