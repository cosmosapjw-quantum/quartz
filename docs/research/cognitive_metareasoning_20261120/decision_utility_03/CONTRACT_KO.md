# QUARTZ-COG-01B-DECISION-UTILITY-03 | 독립 전술 효용 감사 계약

Date: 2026-11-20. Parent committed HEAD `26d8a4d0bc39338c46d08b189483484d295a0133`. Isolated branch `research/quartz-cog01b-decision-utility-20261120`.

## 목표와 범위

직전 72회의 실제 resident MCTS (8→16) 실험은 reference action과 일치하는지 확인했다. 이 실험에서는 실제 Gomoku7(보드 7×7, 행/열 자유 착수, **4개 연속으로 승리**, 무캡처/무중력) 규칙만을 사용하여 **독립적인 3-ply 이내 전술 증명 가능 효용**을 계산하고, 행동의 참조 일치도와 증명된 효용 차이를 분리한다. NN 재훈련·GPU·새 MCTS 강화나 게임의 완전해결을 주장하지 않는다.

## 정확한 부분 판정의 정의

현재 수를 둘 플레이어를 (s∈\{-1,1\})이라 하고, 합법 행동 a마다 후속 게임의 실제 완전게임 효용 (U_s(a)∈\{-1,0,+1\})을 정의한다(완전한 최적상대, 종료 시 현재 root player 기준). 다음 결과만 증명한다.

- `WIN_NOW`: a를 놓아 연속 4가 형성. (U_s(a)=+1).
- `LOSS_NEXT`: a 이후 상대에게 즉시 완성할 승리 수 b가 **하나라도** 존재(현재 a가 종료시키지 않은 경우). 상대가 그 수를 선택하면 (U_s(a)=-1).
- `WIN_BY_THIRD_PLY`: 모든 상대의 합법 응답 b에 대하여 b가 상대 승리가 아니고, 그 후 현재 플레이어에게 즉시 승리하는 c가 존재. 그러면 (U_s(a)=+1). 다수의 응답 b에 대한 보편 전칭 검사 필요.
- `DRAW_TERMINAL`: a로 전판이 찼고 승자가 없음. (U_s(a)=0).
- 그 외에는 `UNRESOLVED` (서로 다른 잠재 완전게임 효용 가능). 절대로 0/draw로 대체하지 않는다.

`certified_lower,certified_upper`: WIN=[1,1], LOSS=[-1,-1], DRAW=[0,0], unresolved=[-1,1]. 실제 완전게임 `U_s(a)`는 항상 닫힌 구간 내부에 있다. 상대의 한 수 승리와 현재의 3-ply 강제승리는 구조적으로 증명되지만, 길고 깊은 전술은 모두 미정으로 남긴다.

실제 추가 계산의 잠재효용 차이는 `ΔU=U_s(a_16)-U_s(a_8)`이다. 지지 가능한 구간은 `[lower16-upper8, upper16-lower8]`이다. 하한이 >0이면 엄밀한 개선, 상한이 <0이면 엄밀한 악화, 하한=상한=0이면 동등, 나머지는 **판정 불가**. `a8 == a16`은 둘이 같은 행동이므로 전술 평가의 미결성과 무관하게 정확히 `ΔU=0`인 점을 별도로 이용한다. 이는 '독립적 정확한 utility 해석'이지 NN 자체의 Bayesian VOC가 아니다.

## 독립 규칙·구현·검산

- 입력은 `src/games/gomoku.rs`의 명시적 규칙(7×7, 4-in-row)과 과거 원시 suite의 48개국면, 이전 `paired_rows.jsonl` 72개. 소스 결함 혼입 방지를 위해 기존 Rust `tactical_sentinel`이나 reference_best/oracle_best로 참값을 만들지 않는다.
- 49칸에 대한 모든 길이 4의 가로·세로·두 방향 대각선 segment mask를 생성. 각 action의 2/3-ply 직접판정은 **모든 적대적 응답을 열거**한다. 새 방어자 행동을 생략하거나 reference 행동을 후보집합으로 사용하지 않는다.
- 새 Python solver + 독립적으로 구현한 scan-based wins 판정 둘 다 단위 테스트와 48개 소스 국면에서 대조. 검증용 Rust 동작은 `Gomoku::wins_if`의 동일 게임 규칙과 일치해야 하며 실제 Rust와 다른 rule/board parity를 발견하면 별도 FAIL.
- 진행 전/후 바이트 해시: 외부 source suite, Rust game code, 연구 직전 72행 데이터 및 checkpoint provenance; 고정 경로 내용이 변경되면 실행 중단. 입력 lock은 시행 전에 게시한다. 결과는 시간/시드 의존하지 않는 전수 판정으로 기록.
- 72 rows의 seed 중복은 24개의 고유 (position,checkpoint) 결과만 독립 단위의 **기술적 중복제거**로 사용. 이 24개를 iid로 간주해 통계적 p값을 만들지 않는다.
- 48개 suite는 고난도 의도 선정이어서 일반 기력의 대표 표본이 아니다. 24개 전에 선택한 12개는 인지학습/validation holdout이 아니다.

## 성공 수용 기준

(1) 49칸 중 WIN masks 수 정확히 (2×7×4+2×4×4=88), 모든 mask는 4비트이며 바깥 인덱스 누출 없음. (2) immediate win, forced-loss-next, forced-win-third, unproven, terminal-draw, invalid board, diagonal edge/corner, wrong board polarity를 검증. (3) 두 독립 계산 알고리즘의 48*2*전체 합법 행동 판정 일치. (4) `P0012/C01_gen_5`의 26→15 사건에서 실제 `certified_delta_interval` 확인. (5) 이전 72개 실측 증거를 byte-locked readback으로 검사하고, seed 중복 24개만 판정. (6) 음성·미결정을 보존하고 독립적 exact tactical utility를 상용 game outcome이나 기력 향상으로 승격하지 않는다.

## 작업과 게시 경계

- A: 위 계약 및 외부 immutable input lock. 테스트 → commit → Dropbox 로컬 archive SHA256 → GitHub push → remote HEAD readback.
- B: 구현과 red/green 테스트, 별도 Python 직접 판정기와 scan 기반 독립 verifier. 같은 방식으로 백업·게시.
- C: 48개 전체 국면 × 전체 합법 행동 인증 결과 + 이전 72 paired utility 구간; 해시·독립 결과로 별도 게시.
- D: 의미론적 결론·검증 현황·다음 단계 handoff, 또 별도 commit/backups/push. main, 기존 연구/개발 브랜치, 다른 worktree는 수정 금지.

선행 참고: Allis, van den Herik, Huntjens, *Go-Moku Solved by New Search Techniques* (1996) 및 *Go-Moku and Threat-Space Search* (1993); Sezener and Dayan (UAI 2020) `https://proceedings.mlr.press/v124/sezener20a.html`. 다른 규칙의 Gomoku 완전해결 문헌을 현재 7×7 connect-four variant의 해법이라고 주장하지 않는다.
