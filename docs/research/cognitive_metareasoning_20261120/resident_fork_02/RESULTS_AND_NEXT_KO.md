# QUARTZ: 실제 resident-root CPU 8→16 계산 증분 연구

**연구일:** 2026-11-20. **ID:** `QUARTZ-COG-01B-RESIDENT-FORK-02`.
**판정:** 실제 학습 체크포인트+실제 Rust MCTS의 **동일 root 추가 계산 실행·계측** 완료. 인지적 VOC 방법론의 성능 우위 **미확인**. Full Bayesian posterior VOC, minimax regret, 실제 Elo 평가 **NOT_RUN**.

## 1. SSOT와 실행 계약

- 연구 기준: `af5f5a95707a0fc443a6ee42aabefaadf9a8c5d4`; 입력·범위 Lock: `5276e6bd965522ebf9d5654bd07c7fdf1376fe2c`.
- Rust resident JSON telemetry 구현 `82dedd303e7aee3e4fb3250efe4c43f9435eb391`, 실험 runner `8c659eea8d6ca964030e5e98d5908690eec09226`; 이 커밋의 깨끗한 tree에서 72개 실험 실행.
- 연구 브랜치: `research/quartz-cog01b-resident-fork-20261120`; 격리 worktree `/home/cosmosapjw/quantum/research/quartz_cog01_20261120`.
- 원자료: 로컬 QUARTZ checkout `65e7d3e9596ca259ebfe523b0c74058692b6a755`, `INPUT_LOCK.json`에 10개 사용 소스의 길이와 SHA256 기록. suite SHA `2eaa84932d6637a03ccb44b571d2df5cbb832f5ee1bfa5bd52d722603de017fd`.
- 국면: `POSITIONS_LOCK.json`의 ID 오름차순 12개(의도적 고난도·특수 bucket 추출); 두 체크포인트: `seed_101/gen_5.pt` (C01)와 `seed_102/gen_5.pt` (C03). 두 checkpoint는 서로 다른 학습 seed지만 성능이 독립적으로 확립된 강한 모델들은 아니다.
- 고정: Gomoku7 (서버 규칙 win length 4), CPU-only `torch 2.9.1+cpu`, 2 inference threads, MCTS n_threads=1, `search_profile=baseline_strict`, `penalty_mode=None`, 시드 {17,29,43}, 8→16, 외부 calibration/policy opt-in 비활성, Dirichlet root noise 없음. 입력 파일은 수정하지 않음.
- Rust 실행물 `/home/cosmosapjw/quantum/research/quartz_resident_20261120_target/debug/mcts_demo`, SHA256 `3fc0d61b5e878fca404daf0ed809508a35a8fb3aff063e69759f261f33f2bf5b` (정확 해시 실행 manifest와 독립 대조).
- `src/mcts_server.rs` 새 opt-in `diagnostic_json=true` 응답: `root_hash`, `evaluator_calls`, `additional_evaluator_calls`, `additional_root_visits`. 기존 QIPC binary 응답은 같은 API로 유지. Rust `engine_session` 관련 8개 테스트 통과; 실제 CPU 테스트에서 root hash 일치 및 physical NN counts 교차확인.

## 2. 과학적 종속변수의 의미

같은 root에서 실제 8번의 MCTS 시뮬레이션 후, 선택한 행동을 기록하고 **동일 엔진**으로 추가 8번을 수행한다. 그 전후 행동 `a8, a16`을 고정된 **더 강한 탐색 대리 기준** `a_ref`와 대조한다.

`Delta_ref = 1[a16=a_ref] - 1[a8=a_ref] ∈ {-1,0,+1}`.

이는 계산 *이후* 실제 행동 결정에 대한 **realized reference agreement change**다. 상대방이 최적으로 대응한 게임 값과 `a8/a16`의 효용 차이를 직접 계산한 것이 아니므로, 수학적으로 `E[Δ Bayesian regret | evidence]` 또는 minimax simple regret와 등치할 수 없다. 또한 baseline_strict를 시험한 것이며 QUARTZ의 새로운 adaptive VOC actuator를 시험한 것이 아니다.

각 trial마다 새 Rust server가 동일 input/seed로 fresh16을 재현하는 별도 통제를 포함한다. 16회 fresh와 8+8회 resident의 행동·정책·evaluator boundary calls가 같아야 한다는 것은 시험 대상이며 결과를 강제로 일치시키지 않았다.

## 3. 정확 실행 결과

| 항목 | 관측 |
|---|---:|
| 시도·완료한 paired trials | 72 / 72 |
| 독립적인 (position, checkpoint) 조합 | 12 × 2 = **24** |
| 각 조합의 3개 시드 탐색 궤적 | **모두 동일** |
| reference agreement +1 / -1 / 0 (trial 관측) | **3 / 0 / 69** |
| reference agreement +1 / -1 / 0 (고유 조합) | **1 / 0 / 23** |
| 처음 8회 root visits | **72/72 trial에서 8회** |
| 추가 8회 이후 root visits | **72/72 trial에서 16회** |
| root hash 변경 | **0** |
| resident16 대 fresh16 불일치 | **0/72** |
| Rust evaluator 대비 실제 NN forward item 오류 | **0/72** (캐시 비활성 조건) |
| 72회에서 추가된 물리적 NN 평가 횟수 | **567** |
| 고유한 24개 조합 기준 추가 평가 횟수 | **189** |
| 72회 실험 전체 PyTorch forward items (fresh16 대조 포함) | **2430** |
| 추가 8회 탐색의 평균 wall time | **약 17.398 ms** |
| 8회 시점 동률 policy 수 (72 trials) | 42 |
| 16회 시점 동률 policy 수 (72 trials) | 42 |

유일한 개선 고유 조합은 `P0012 / C01_gen_5`: 초기 수 **26 → 15**, 참조수 **15**, 별도 oracle 수 **15**, 추가 NN forward 8회. 반면 `P0019 / C03_gen_5`는 방문 8회를 추가하는 동안 물리적 NN forward가 **5회**만 발생했다. 시뮬레이션 방문과 별도 NN evaluation이 항상 1:1 대응하는 것은 아니다. 5회가 된 구체적인 경로(terminal/TT 등)는 이번 단계에서 별도 사건별 로그로 입증하지 않았으므로 원인을 단정하지 않는다.

**매우 중요한 통계 한계:** 셋의 설정된 seed가 이 제한된 결정적 탐색 체계에서 서로 다른 궤적을 생산하지 않았다. 따라서 72건을 독립 시행으로 사용한 p-value, confidence interval, 기력 상승률은 무효다. 24개 조합도 의도적으로 선택된 국면에 체크포인트 두 개를 반복 사용하므로 iid 모집단 표본이 아니다. 이 표본에서는 불확실성 범위를 승격하지 않는다.

## 4. 실험 중 실제 결함/환경 문제의 분류 및 수정

- TDD Rust red: `test_engine_session_exposes_resident_root_and_evaluator_accounting` 실패(새 필드 없음). 필요한 경계 계측을 구현한 후 green.
- Rust 통신 red: `test_engine_session_diagnostic_json_preserves_fields_without_changing_default` 함수 부재 실패. 기본 compact transport가 새 키를 버리는 것을 발견한 뒤 `diagnostic_json=true` opt-in 경로 추가, 두 테스트 green. 기존 기본 QIPC는 유지.
- Python TDD: 초기 `NotImplementedError` red 후 새 계산계약·요약 모듈과 7개 테스트 green. Fresh16의 추가 방문수가 16이라는 test fixture를 교정했으며 수용 assertion을 삭제하지 않았다.
- Python/Rust NN 계측 차이: Python process-global `quartz/eval_runtime.py::_NN_EVAL_CACHE` 때문에 새로운 Rust server로 fresh16을 다시 실행해도 Rust evaluator 호출 17에 실제 NN forward는 0인 사례 발견. 이는 기존 캐시의 효율화 동작이며, Rust 호출을 물리 NN inference로 오인하면 측정 오류다.
- 실험 중엔 `QUARTZ_DISABLE_NN_CACHE=1`, `QUARTZ_DISABLE_ASYNC_PIPELINE=1`을 강제하고 cache-enabled production 측정과 분리했다. 첫 한 국면 smoke에서 Python/Rust 누적 counts 9→17, fresh16=17 일치; 72건 모두 계산 계약을 만족했다.
- 관련 사건 기록 `CACHE_FINDING_AND_REPAIR.md`에서 저장. cache key의 model epoch 교차 전송의 실제 오류가 발생했다는 주장은 이번 증거만으로 하지 않는다.

## 5. 재현·검증·백업

증거: `evidence/20261120_resident72/{paired_rows.jsonl,summary.json,model_items.json,execution_manifest.json,INDEPENDENT_READBACK.json}`. 실행 source SHA `8c659eea8d6ca964030e5e98d5908690eec09226`. input 10 파일, position, binary SHA256을 실행 전 고정하고 종료 후 `INDEPENDENT_READBACK.json` 작성 시 재조회했다. 네 개 raw output sha와 독립 row 카운트·평가 비용·reference delta를 검사했다.

기록된 검증: Rust 관련 engine_session 테스트 8 PASS, 새 Python resident 계약 7 PASS, 앞선 CPU realtrace 7 PASS, 인지 COG01 9 PASS, W1 참조 27 PASS. Rust 디버그 바이너리 빌드 PASS; 전 저장소의 모든 Rust 테스트나 GPU 실험은 별도로 주장하지 않는다.

단계별 GitHub push + remote SHA 확인, 로컬 Dropbox 동기화 경로 archive 생성·sha256:
- 계약/input A: `5276e6bd965522ebf9d5654bd07c7fdf1376fe2c`, `COG01B_RF_A_5276e6bd9655.tar.gz`.
- Rust 계측 B: `82dedd303e7aee3e4fb3250efe4c43f9435eb391`, `COG01B_RF_B_82dedd303e7a.tar.gz`.
- Python 실행기 C: `8c659eea8d6ca964030e5e98d5908690eec09226`, `COG01B_RF_C_8c659eea8d6c.tar.gz`.
- 원시 결과·독립 검산 D: `e878a1e096fab821e56e11c5d4397e9bcd48f44c`, `COG01B_RF_D_e878a1e096fa.tar.gz`.
- 최종 판정 E: 이 문서 커밋에 종속. 별도 sha/backup/원격 일치 확인 후 handoff에 최종 표기.

Dropbox **로컬 폴더 백업은 확인**했지만 외부 동기화 서비스의 완료 ACK나 복원 검증은 별도 수행하지 않았다. 작업 브랜치 외 main, 이전 QUARTZ scientific branch, 사용자 로컬 개발 브랜치, 다른 worktree는 변경하지 않는다.

## 6. 다음 작업: QUARTZ-COG-01B-DECISION-UTILITY-03

**새 목표:** '추가 계산이 참조 행동을 맞혔다'와 '최종 결정의 효용이 개선됐다'를 분리할 수 있는 독립 **action-utility benchmark**를 우선 획득한다.

1. 국면 대표성: 48개 고정 미니 suite 전체 또는 새로 독립 선택한 held-out 국면으로 확장하고, 학습 체크포인트와 reference 데이터 제작에 사용한 source family를 분리한다. 현재의 고난도 채굴 case 12개는 난도 및 성공률 보편 주장에 사용하지 않는다.
2. 독립 seed 문제를 해결하라. 실제 모델이 결정적이면 seed 반복을 포기하고 unique roots로 표본 단위를 정의한다. 별도 root Dirichlet 또는 stochastic evaluator를 도입할 때에는 noise와 prior/likelihood identity를 명시하고 양 경로의 공통 prefix/random stream을 고정해야 한다.
3. `P0012` 단일 성공과 `P0019` 5-eval 현상을 개별 로그로 재현하고, terminal·TT·NN cache 등 관측 차이가 계산가치에 미치는 영향을 정의한다. 종료 조건·cost 차이를 검증한 뒤에만 더 긴 16→32, 32→64를 평가한다.
4. 가능하면 exact tactical solver 또는 충분히 독립적인 고예산 정책평가로 모든 합법 행동의 비교가능한 효용 `U(a)`를 제공하고 `L(a_8)-L(a_{16})`를 직접 계산한다. 대용 reference를 사용하면 `proxy`라고 표기한다.
5. 그 후에야 recognition-only, channel-aware VOC, fixed-budget, strong Gumbel/PUCT를 동일한 물리 NN 호출 수·시간으로 대조한다. 특히 캐시 비활성 vs 활성 설정은 서로 다른 estimator cost model로 별도 분리한다.
6. 사용자 지시 유지: 각 완료 단계마다 commit → 로컬 Dropbox backup SHA → GitHub push → 원격 SHA readback.

**종결 판정:** 이번 stage는 "진짜 resident CPU same-root incremental experiment"의 기술적 준비와 제한된 관찰을 완료했다. Bayesian expected regret 개선이나 범용 인지과학적 성능 이득이 입증된 것은 아니다. 네이티브 skill-bound subagent 기능은 제공되지 않아 `native_subagent=NOT_RUN`; 직접 실행과 독립 코드/자료 검산으로 진행했다.

문헌: E. Sezener & P. Dayan, *Static and Dynamic Values of Computation in MCTS*, UAI 2020, https://proceedings.mlr.press/v124/sezener20a.html . 이 논문은 Bayesian simple regret 기반 계산가치에 관한 선행연구이며, 본 CPU 무작위 추론보다 강한 이론 계약을 다룬다.
