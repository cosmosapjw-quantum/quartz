# QUARTZ-COG-01B-RESIDENT-FORK-02 | CPU 실제 추가계산 검증

Date 2026-11-20. Parent `af5f5a95707a0fc443a6ee42aabefaadf9a8c5d4`. Branch `research/quartz-cog01b-resident-fork-20261120`. **서버의 실제 동일 root 연장** 경로와 신경망 CPU 평가를 검증한다.

## R1/C1: 대상과 증거 수준

- `src/mcts_server.rs::EngineSearchSession::search_with_iters`는 동일 엔진 root에 누적 방문 목표 `current + delta`를 준다. 기존 Rust `test_engine_session_step_iters_are_delta_not_absolute`는 8→16 동일 root 방문을 검사한다. `MctsEngine::advance_root`는 선택 수로 이동하는 **다른 경로**다.
- `MctsEngine::evaluator_call_count()`는 root 초기 확장 포함 평가기 호출 수, 말단/TT 해결 시 zero increment. 연구용 session JSON 응답에는 이 누적값과 root hash를 추가해 실제 측정한다. `n_completed_NN_calls`와 `n_iterations`는 구분한다. 비동기 pending 없는 n_threads=1 경로만 사용.
- `search_profile=baseline_strict`, `n_threads=1`, `seed` 고정, temperature=0, evaluator 고정. QUARTZ physics/controller의 효과를 이 실험에서 주장하지 않는다. CPU-only PyTorch 2.9.1+cpu, fixed 2 inference threads, Rust 서버 `mcts_demo --server`. Dirichlet 주입 없음.
- 12개 국면은 기존 48개 `position_suite.json`을 **ID 정렬하여 첫 12개** 선택. 의도적으로 reference 난도 분류로 채굴된 사례임을 보존. `P0018/P0032`처럼 reference_best와 oracle_best가 불일치하는 경우도 포함하며 **두 레이블을 따로 사용**한다.
- 두 기 학습 checkpoint `C01_gen_5` (seed_101) 및 `C03_gen_5` (seed_102). 실험 seed `17,29,43`. 총 12×2×3=72 resident-root trials; 각 trial은 처음부터 같은 root에서 8회 후 snapshot, 그 엔진에 8회 추가 snapshot을 기록. 별도 fresh16 같은 seed replay로 통제 비교. `control=not run`일 경우 명시; 72 fresh16이 비용상 가능할 때만 모두 실행한다.
- Reference label은 기존 `reference_best`, `oracle_best`: 출처는 C06_latest 고예산 엔진 대리 기준. **실제 game-theoretic truth가 아니다**. 각 행동의 정확한 효용이나 손실 폭을 갖고 있지 않아 0/1 reference-action agreement difference만 정의 가능하다. `signed_change_ref = 1(a16==ref)-1(a8==ref)`. 이것이 full Bayesian VOC나 minimax regret가 아니다.
- True marginal *computation* comparison: 동일 root의 8시뮬레이션 상태에서 중단한 정책과 8시뮬레이션을 더 계산하여 나온 정책의 paired difference. 이미 수행한 8시뮬레이션의 난수/관측을 동일하게 보존하고, 뒤 8시뮬레이션의 **실제 완료 evaluator calls와 wall-clock**을 측정. 각 실험의 재시작 seed는 반복으로 오해하지 않는다: 동일 코드가 seed를 무시하는 결정적 경로이면 seed 3개는 독립 통계 표본이 아니므로 고유 궤적을 하나로 집계한다.
- CPU 실제 NN model과 Rust MCTS 상태는 진짜 객체여야 한다. 동일한 evaluator state hash/epoch가 들어가지 않으면 실험 중단. 예측기 학습, 신경망 업데이트, 게임 기력/논문 주장 없음.

## 수용 게이트

1. 실험 이전 원본 source HEAD, checkpoint 2개, reference position file, `Cargo.lock`, `Cargo.toml` byte identities 고정, 각 archive 원격 SHA 확인.
2. 전용 Rust 단위 테스트는 같은 엔진에서 첫 8과 추가 8의 누적 `root_visits` 8→16, 동일 root identity, evaluator_call_count 단조 증가 및 가능한 upper bound 확인. 먼저 기대 실패(red)를 보고, 최소 수정 후 green.
3. Python 통합 smoke는 두 checkpoint 중 하나와 비전술 판 1–2개로 실행. `8→16`에서 `eval_calls_after>=eval_calls_before`, 추가 calls<=8, root hash 불변, 두 정책 모두 합법수만 포함, `best_move` 가 합법임을 보장. Zero-eval steps는 합법 terminal/TT reuse 가능하므로 무조건 calls=8로 주장 금지.
4. Fresh16과 resident(8+8)의 identity 비교: root, seed, cfg, NN, process condition, policy (finite precision tolerance), calls. 결과가 다르면 자체 알고리즘의 멱등성/난수 재시작과 controller reset부터 조사하고 불일치 그대로 기록; `match`를 강제하기 위해 계약을 바꾸지 않는다.
5. 72 trials small CPU batch; 동일 seed/동일 position 사이 상관과 동일 reference 간 종속성 기록. `sigma`, p-value와 confidence intervals은 **독립 trial 없이는 보고하지 않는다**. 원시 실패 보존.
6. `main`, 기존 `agent/local-experiment-foundry`, `research/quartz-cog01b-cpu-realtrace-20261120`, 타 worktree 및 기존 W1/suite/checkpoint 내용 불변.

## 단계와 백업·게시

- A: 본 계약과 고정 입력 식별·실제 엔진 읽기 → commit → 로컬 Dropbox archive SHA → push → `ls-remote` SHA 대조.
- B: Rust telemetry와 단위 테스트 red/green 및 Python 실험실행기, 별도 CPU smoke → 위와 동일.
- C: 신경망 포함 bounded CPU batch, raw JSONL, 프로세스/입력/output hash manifest, 독립 aggregation → 위와 동일.
- D: 검증 가능한 진술·반례·인계 결과 보고서 → 위와 동일.

백업 폴더 `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups` 는 **로컬 Dropbox 동기화 경로**이며 실제 클라우드 동기화 ACK나 복원까지 검증했다는 뜻이 아니다.

## 직접 참고한 선행연구

Sezener & Dayan (2020), *Static and Dynamic Values of Computation in MCTS*, UAI https://proceedings.mlr.press/v124/sezener20a.html. 완전한 정보모형의 Bayesian VOC는 terminal simple regret의 기대감소로 정의된다. 이번 것은 실제 공통-prefix MCTS의 observational realized **reference agreement**, true VOC는 아니다.
