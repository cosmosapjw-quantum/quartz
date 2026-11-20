# QUARTZ COG-01B Connect4 GameState: 정확 승패효용에 따른 CPU 탐색 연구

**2026-11-20, 연구 ID:** `QUARTZ-COG-01B-CONNECT4-GAMESTATE-05`
**상태:** Rust standard gravity Connect4 GameState `IMPLEMENTATION_VERIFIED`, 사전 고정 12개 원시 국면의 uniform MCTS 의사결정 효용 `OBSERVATIONALLY_CHECKED`, 인지적 계산가치 모델의 이득 `UNRESOLVED / NOT_TESTED`.

## 1. 목표·정의·정확한 참조 범위

이전 7×7 무중력 4목은 CPU에 적합한 축소 연구였고, 원래 연구 목표의 유효범위를 제한하지 않는다. 이번에는 표준 7×6 Connect Four로 옮겨 실제 MCTS를 구현했으며, Pascal Pons 공개 `connect4` solver의 12개 위치 × 7개 열에 대한 행동별 게임이론 점수와 결합했다.

- 중력: 각 행동 (a∈\{0,...,6\})은 열 하나를 선택하고, 가장 낮은 빈 칸에 현재 플레이어의 돌을 놓는다.
- 종결: 연결된 4개 수평/수직/대각선이 승리, 42칸이 꽉 차고 승자가 없으면 무승부.
- Rust `GameState::outcome` 부호는 **다음 차례 플레이어 기준** negamax `(+1,-1,0)`; Python 외부 oracle의 선택 효용은 **현재 root 플레이어 기준**으로 따로 정의.
- 외부 solver action score `S(a)`의 부호만 실제 승패효용으로 사용: (U(a)=\operatorname{sgn}S(a)∈\{-1,0,1\}). `S(a)=-1000`은 꽉 찬 열의 invalid sentinel이며 효용이 아니다. 정수 점수의 **절댓값**에는 승리·패배 시점 우선순위가 있으므로 cardinal utility의 개선량으로 사용하지 않는다.
- root의 최적 승패효용 (U_*=\max_{a\text{ legal}} U(a)),  `regret_b=U_*-U(a_b)`. 동일 실제 root 방문 예산 (b_0,b_1) 사이 `ΔU=U(a_b1)-U(a_b0)`는 기록된 실제 결정 효용 변화. **Bayesian 기대 value of computation**나 잠재적 엘로가 아니다.

## 2. 구현과 독립 검증

새 `src/games/connect4.rs`는 7×6 bottom-up row-major 보드, 열 높이, 턴, 누적 착수, 승자, stable TT 해시, 합법 열, 수평/수직/대각선 승리, strict move-history parser, negamax terminal, make/unmake Undo 및 3-plane 연구용 인코더를 제공한다. 17-plane 타 게임 NN과 호환된다고 주장하지 않는다.

Rust 서버에는 기존 프로토콜을 변경하지 않고 `research_connect4_cpu` 연구용 별도 JSON 명령을 추가했다. 실제 `MctsEngine<Connect4>` 하나를 고정된 8→16→32→64 cumulative root visit budget에서 연속 구동하며 다음을 반환한다: `root_hash, best_action, root_visits, evaluator_calls, additional_evaluator_calls, step_wall_us, visit_policy`. 각 국면에서 독립 fresh16을 같은 seed로 다시 풀어 root prefix의 행동·정책·평가기 호출 identity를 검사한다.

초기 evaluator는 사전학습된 모델이 아니라 `UniformEval`이고, `root_forced_win=false`, 온도 0, 단일 Rust thread, 고정 seed17, Dirichlet noise 없음. **따라서 NN 호출이 0개**이며, 기록되는 비용은 실제 uniform evaluator 경계 호출 수다. 탐색에 외부 oracle값을 전달하지 않는다.

TDD:
- 실제 Connect4 중력/보드 테스트가 기존 stub에서 실패하는 RED가 기록됐고, 최소 구현 후 6개 기존 테스트 GREEN, 이후 외부 solver가 직접 구성한 42수 무승부 history의 terminal/undo 검증 테스트가 추가로 통과했다.
- 연구 endpoint가 존재하지 않아 컴파일 실패한 RED 후, 동일 root 연장 및 불량 history/budget 거부 테스트 2개 GREEN.
- Python utility join이 누락되어 RED, 실제 점수 부호·regret·실측 eval 회수·불법열/원점해시/재시작 identity 교차검사 구현 후 4개 GREEN. 초기 테스트 fixture 두 문제(가득 찬 열 가정과 evaluator 증분 오기)는 수용 검사를 없애지 않고 고쳤다.
- 최종 관련성 높은 검증: `cargo test --offline --locked -- --test-threads=4`에서 Rust 577개 PASS, 89개 ignored, 0 fail. 관련 Python 연구 테스트 61개 PASS (4+7+7+7+9+27).
- 직접 서버 smoke: 8→16→32→64 visits가 단일 실제 root hash를 유지하며 완료. Fresh16은 실제 별도 엔진.

## 3. 사전 고정 12개 국면의 관측 결과

이 연구 대상은 `standard_external_04/POSITIONS_FIXED.json`의 **사전에 선정한 12개 nonterminal root**이며, 게임이론적 참조는 `standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json`의 7-행동 점수다. 첫 12개는 인위적으로 생성한 단일 범주의 역사 위치이며 모두 짝수 ply이므로 **흑 차례만 포함**한다. 모집단 대표성이 없다. 서로 구별되는 위치 12개, 시드 17 하나.

| 누적 MCTS 방문 | 선택 효용 평균 | 평균 regret | 완전 승패효용 최적 행동 |
|---|---:|---:|---:|
| 8 | 0.5000 | 0.2500 | 10/12 |
| 16 | 0.5000 | 0.2500 | 10/12 |
| 32 | 0.6667 | 0.0833 | 11/12 |
| 64 | 0.6667 | 0.0833 | 11/12 |

8→16: 12개 국면 모두 `ΔU=0` (실제 추가 evaluator 총 44회).
16→32: 11개 `ΔU=0`, **1개 `ΔU=+2`** (실제 추가 evaluator 총 69회).
32→64: 모두 `ΔU=0` (추가 evaluator 총 145회).
따라서 8회 이후 64회까지 추가 평가 호출은 총 258회지만 승패효용의 변화는 `C4_003` 하나다. 논문의 계산 선택 정책이 아니라 고정 증가 budget의 관찰이므로 비효율성 수치 자체를 새로운 VOC 정책의 실적이라고 해석하면 안 된다.

**직접 확인된 의미 있는 사례:** `C4_003`, 역사 `64547341472363`, 누적 visit [8,16,32,64]의 action index [0,1,2,2], 외부 원점수 [-14,-14,+13,+13], 따라서 정확 승패효용 [-1,-1,+1,+1]. 즉 **16→32회에 강제패 행동에서 강제승 행동으로 변화**함. 이후 반복 계산은 같은 승패효용을 유지한다.

**중요한 음성 사례:** `C4_002`는 게임이론 최적 결과가 무승부 `U_*=0`이지만 모든 budget에서 QUARTZ UniformEval의 선택 효용이 -1이었어. `regret=1`이 64회까지 유지됐다. 단순한 계산량 증가가 올바른 인식기를 대신한다는 일반 주장은 이번에도 성립하지 않는다.

12개 중 5개에서 선택 열 자체는 바뀌었지만 **효용이 실제로 바뀐 국면은 1개**였다. 관측된 행동 변경과 결정가치 개선을 같은 지표로 취급하면 잘못된 결론이 나온다.

## 4. 입력 권위·재현성·독립 검산

- 부모 SHA `bc44e5c85c1f37201f80880f18d59fdb55a509a8`.
- 계약/고정 입력: `INPUT_LOCK.json`, 기준 커밋 `b7488b2100b2983dde5b438c2294f2844940928a`. 총 10개 기존 Git source 및 2개 외부 oracle 파일에 대한 checksum 기록. source server는 정당하게 수정되었으므로 원래 Git 기준 바이트와 수정 후 바이트를 구분한다.
- Rust GameState/endpoint 구현 `43f26bc76cdd40bd9ef3a7f60952b6d7a37d046b`. Python 실험실행기 `1a8305f8d7a9c9fdd499364f0646ffe0787ff7ec`. **원시 실험은 커밋 `1a8305f...`의 깨끗한 소스**와 독립 Rust 실행 바이너리 SHA `738d7186e883026ad59242cb3007b3ef530ff272f76dd66cc462049a153c955f`에 종속된다.
- 원시 증거: `evidence/20261120_connect4_exact_12/{raw_mcts_responses.json,joined_action_utility.json,summary.json,execution_manifest.json}`.
- 독립 readback은 외부 Pascal Pons solver에 원본 12개와 좌우 반전 12개를 다시 질의해 **24개의 독립 재실행**에서 전체 7열 점수를 검증했다. Rust/Python 중력 규칙과 12개 전체 root 보드, evaluator calls, regret, 행동효용/부호, 16방문 fresh restart 12개를 재검산했다. 결과: `INDEPENDENT_READBACK.json`; 36개의 budget 전환 중 nonzero `ΔU`는 +2 하나뿐이고 음의 사례는 0이다.
- 외부 solver의 본질적 완전해결성은 원 구현의 검증 범위를 사용한 것이지 이 보고서가 solver를 독립 형식증명한 것은 아니다. 접속·라이선스: https://github.com/PascalPons/connect4 (AGPL-3.0). 실행 바이너리나 외부 book을 GitHub/백업 archive에 재배포하지 않았다.
- 최종 42수 무승부 수열 `354544424252253123243353566766667777711111`은 외부 solver의 무승부 보존 열을 순차 선택해 구성했다. GameState는 전체 합법 이력을 승인하고 `terminal=true, outcome=0, legal_moves=[]`을 확인하며 이후 추가 이동을 거부했다. 이 테스트는 *원시 실험 이후 코드의 테스트 전용 변경*으로, 원시 실행의 binary/source identity를 사후 변경하지 않는다.
- 기존 QUARTZ source branches 및 external binary/book/weights는 불변, 모든 연구 변경은 분리된 worktree 연구 브랜치에서 수행. 사용자 원본 `agent/local-experiment-foundry`의 미추적 파일을 그대로 보존한다.

## 5. 과학적 판정과 다음 작업

`established/implementation-verified`: 표준 중력 Connect4 Rule/Move/Terminal, 동일 MCTS root의 실제 탐색 예산 연장, evaluator 호출 계측 및 12개 점수 결합. `numerically checked`: 12개 국면의 정확 oracle 승패효용·regret 집계, 12+12 requery/mirror, full-board draw.

`unresolved`: 학습된 Connect4 인식기가 실제 추가 계산의 평균 가치와 예산 배분을 개선하는가? 적절한 인식기 없이 uniform MCTS만 실행했으므로 현재 데이터는 이 주장에 대한 증거가 아니다. `blocked`가 아니라 실행 가능한 다음 가설과 데이터 생성 경로가 존재한다.

**다음 노드: `QUARTZ-COG-01B-CONNECT4-RECOGNIZER-06`.**
1. 독립 수천 개까지의 다양한 합법 Connect4 opening/midgame history를 생성하되 컴퓨팅 예산에 맞춰 **bounded CPU cohort**로 시작하고, external exact solver 7열 WDL로 라벨을 얻는다. timeout은 unresolved로 보존.
2. 훈련/검증 분리는 동일 game history prefix family와 좌우 반전 국면이 **같은 split**에 들게 한다. 현재 12개 root는 최종 분리 heldout으로 고정해 leakage를 차단한다. root 파생 위치도 같은 그룹으로 취급한다.
3. 3×6×7 encoder 또는 명시적 history encoding을 기반으로 작은 CPU policy/value recognizer를 teacher-distill한다. 최적 승패효용 행동 *집합*과 root WDL을 분리하여 학습하고, 과도한 단일 행동 모방을 피한다. 외부 solver 점수는 training labels로만 사용; MCTS runtime에는 전달하지 않는다.
4. NN evaluator의 실제 forward count, NN cache identity/model epoch, TT/terminal 해결을 구분해 uniform-MCTS 및 trained-recognizer-MCTS를 동일 root/seed/계산 비용에서 비교한다. 정확한 승패효용 `ΔU`, regret, 완료 NN 호출과 wall time을 모두 수집한다. CPU 학습 속도와 목표 정확도의 실제 증거 없이 성능 향상을 주장하지 않는다.
5. 그 후 연구의 원래 인지적 질문인 recognition-only vs channel-aware Bayesian VOC를 비교하되 **동일 정보원·동일 cost model**, 후보 누락, 적대적 변칙·재개방을 분리한다. 미흡하면 선택적 계산 가설의 적용범위를 제한하고 다른 게임 전이에 과장하지 않는다.

중간 예비 탐색 7×7 축소 게임의 결과는 별도 정본으로 유지하며 이 새 Connect4 자료와 혼합하지 않는다.

## 6. 백업 및 게시 이력

사용자 승인에 따라 매 검증단위마다 Git commit → 로컬 Dropbox archive tar.gz → sha256 → GitHub push → 원격 `git ls-remote` identity 확인을 시행.

- A `b7488b2100b2983dde5b438c2294f2844940928a`, `COG01B_C4_A_b7488b2100b2.tar.gz`.
- B `43f26bc76cdd40bd9ef3a7f60952b6d7a37d046b`, `COG01B_C4_B_43f26bc76cdd.tar.gz`.
- C0 `1a8305f8d7a9c9fdd499364f0646ffe0787ff7ec`, `COG01B_C4_C0_1a8305f8d7a9.tar.gz`.
- C `c6e06cb56f6f7ad2089442f8fedec179abd3b400`, `COG01B_C4_C_c6e06cb56f6f.tar.gz`.
- D final `RESULTS_AND_NEXT_KO.md` 및 42수 무승부 unit test + 검산 receipt를 추가 게시한다. 게시 후 exact commit 및 checksum을 인계한다.

Dropbox 동기화 **로컬 폴더**에 생성된 백업 파일 존재와 SHA를 확인했지만 Dropbox 서버 동기화 완료 ACK 및 복원 테스트까지 입증한 것은 아니다. 네이티브 subagent 기능 부재로 `native_subagent: NOT_RUN`; 직접 실행과 독립 oracle/Python 재조회 감사를 사용했다.
