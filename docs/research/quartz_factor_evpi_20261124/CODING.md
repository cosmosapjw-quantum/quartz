# C01–C10 Rust 실행

## C01 — 고정 기준

대조군은 C29 576cbe554ea0의 full horizon 및 EPS 선택. 동일 binary, 같은 입력, 같은 C26 predictor와 rollout 노드를 비교한다. 우선 동일한 관측, 최종 move, 계획 value와 utility를 요구한다. 비용 차이는 paired resident timing으로 해석하고 판정 경계의 동점은 그대로 full path에 맡긴다.
