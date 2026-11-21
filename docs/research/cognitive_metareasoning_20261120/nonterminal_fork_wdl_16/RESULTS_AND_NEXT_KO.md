# Quartz C16: Nonterminal Fork Witness & Game WDL, 최종 연구 판정과 인계

**연구일** 2026-11-21 KST. **연구 ID** QUARTZ-COG-05-NONTERMINAL-MISS-AND-GAME-WDL-16.
**기준 C15** a85bc0ded6bf67f81256e93645dc7147838c6f88. **새 브랜치** research/quartz-cog06-nonterminal-fork-wdl-20261121. **작업 폴더** /home/cosmosapjw/quantum/research/quartz_cog06_20261121. 기존 developer worktree, C14/C15 정본은 수정하지 않았으며 main 병합·PR 생성도 수행하지 않았다.

## 1. 이론 R1–R10: 정확히 보존된 과학적 산출물

R1 범위 확정, R2 Allis·Chaslot·PN-MCTS·Rapfi 1차 문헌과 C15 정본 확인, R3 source-only 비종단 fork certificate 정의, R4 원래 MCTS root/동일 학습 prior에 대응한 대안, R5 구성적 증명 및 32-local-cell 보조정리, R6 즉시 승리 상대·중복 endpoint·게임규칙 반례, R7 실제 Python/Wolfram 검산, R8 admissible claim 결정, R9 Rust 수치 실행 계약, R10 코드 인계까지 **10개 각자 독립 Git push/해시 백업 완료**.

명제: terminal-free Freestyle15 원국면에서 현재 선수와 상대 모두 즉시 승리 수가 없고, 합법 착수 a가 비종단이면서 a 직후 현재 선수에게 서로 다른 즉시 승리 지점 c1,c2가 생기면, 상대의 어떤 다음 착수에도 적어도 하나의 승리 지점이 남아 3-ply 이내 강제 승리한다. 기존 직선 open-four와 위협 공간 검색의 충분조건을 온라인 검산 가능한 certificate로 다시 구성한 것이며, **새로운 Gomoku 승리 정리/알고리즘이라는 독창성 주장은 금지**한다.

직접 증명한 국소성: 이전에 즉시 승리가 없었다면 a가 새로 만드는 단일 착수 승리 지점은 a와 같은 네 축, 거리 1..4에 있어 최대 32칸만 확인하면 충분하다. R7 실행: 명시적 예제에서 상대 답변 **218/218** 유효, 전체 행동 전수 탐색과 국소 계산 **90/90 일치**, Wolfram 32칸 정수 집합 확인. 이러한 확인은 이론의 유효범위를 넘는 population MCTS 주장과 별개다.

## 2. 코딩 C1–C10: 실제 구현과 반증 가능한 실험

C1 기존 13-feature policy-only frozen model hash와 정본 lock, C2 방정식·코드 매핑, C3 원본 priors/edge P/N/W/M2 보존 및 20ms 물리 예산 설계, C4 새 소스 freeze, C5 Rust 실제 feature와 원본 root 구현, C6 독립 Python·Rust 룰/회귀 확인, C7 시험 이전 code/arm/게임 순서 봉인, C8 native MCTS와 실제 32판, C9 독립 전체 착수·규칙 재생 및 백업 복원, C10 본 판정·다음 단계 인계의 10개 단계를 진행했다. 개별 C08A(블라인드 195 root)와 C08B(완결된 32판 원 기록과 요약 포장 오류의 제한 수정)도 별도 Git·백업으로 봉인했다. 마지막 C10 SHA의 최종 게시 사실은 실제 publisher 출력·원격 대조 결과를 기준으로 판단한다.

동결된 모델 SHA256: 5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83. **실제로 컴파일·실행된** C16 Rust 릴리스 바이너리 SHA256: 1168f2979936a283a9e7e613b869fd30ec9aba826e15ac228c6ee9d7e7e4c4d3. 신규 40 국면 freeze SHA256: 8349441f247f720028f8534ac4abf656be56f8ac58778191de900722b4af1049. 독립 Python은 실제 40개 국면에서 **548개 개별 certificate action**을 전체 root 합법 행동 × 후속 승리 칸 전수 조사했고 Rust 32-local checker와 완전 일치했다. Rust Debug 전체 테스트 **611 PASS/0 FAIL/89 ignored**, 릴리스도 **611 PASS/0 FAIL/89 ignored**를 C6과 C9에서 실행했다.

C4의 24개 기획된 전술 소스는 **하나의 광범위한 템플릿 가족**의 서로 다른 RNG 구성원으로 일반적인 오목 국면 표본이 아니다. C05 사전 점검에 사용한 fork-000은 C8 시험에서 제외했다. 새 C8 시험: 23개 구성 전술 + 16개 C15에서 실제 생성한 게임 이력 = 39개 시험 국면, 각 5개 정책으로 **195개 실제 MCTS 응답**. 모든 root는 전체 합법 수 원래 prior/edge를 그대로 보존했으며 8회 첫 방문→실제 CPU 시간 예산의 동일 root 추가 방문을 수행했다.

| C8 시험 층 | 구성 비종단 fork 23 | 실제 Quartz 게임 이력 16 |
| --- | ---: | ---: |
| 누락된 fork의 실제 재활성화 | 23/23 | 0/16 |
| 전체 합법 탐색군과 다른 최종 착수 | 1/23 | 1/16 |
| 모든 비교군이 certificate 승리 수 선택 | **23/23** | 2/16 |
| 학습 prior 재개방 vs 전체 root 평균 시간 차 | **0.157%** | **0.028%** |
| 지정한 별도 10% CPU 일치 기준 | PASS | PASS |

추가로 검산한 **사후탐색적 이항 승리 접근성 동등류**는 F(s)=certificate action 집합, K(s)=top16, E(s)=전체 합법 수에 대해 R(S)=1{F(s)∩S≠∅}이고
R(E)−R(K)=1{F∩K=∅, F∩(E\K)≠∅} 이다. **기획된 23/23 국면은 F∩(E\K)가 비어 있지 않지만 동시에 F∩K도 비어 있지 않았다.** 이미 top16에 승리를 보장하는 수가 있으므로, 누락된 추가 certificate를 발견하더라도 이항 승리 접근성은 개선되지 않는다. 실제 게임 이력 16개에는 누락된 certificate가 없었다. 따라서 *엄밀히 필요한 누락된 강제 승리 후보(strict effective miss)*는 39개 중 **0/39**이다. 이 조건은 C8 결과를 본 뒤 개념적으로 도출했으므로 **새 시험의 사전 성공 기준으로 소급 주장하지 않는다**.

## 3. 완결된 실제 오목 게임 WDL과 측정 불가능한 메커니즘

**8개 사전에 동결된 실제 Quartz 게임 기반 opening** × 양 색상 교환 × 두 개 intervention = **32판 실제 Rust MCTS 전체 대국**. C7 발행된 runner가 32판 전체를 실제 수행했고, 결과를 append-only 완료 ledger에 저장했다. 원본 JSONL SHA256 a118b4848d87b441c0a9d46e87da384ebd485c05d17559714d77f91b57a83d84는 **결과 요약 오류의 최소 수정 전후 동일**하다. 32판의 추가 착수 **217수**를 C9 독립 Python Freestyle 룰 구현으로 전수 재생하여 승리자·종료 시점·합법 수 검증 **PASS**했다.

| 학습 full-legal 대비 정책 | 16판 W | D | L | 평균 점수 | 실제 fork 발동/선택 개입 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_fork_mask | 8 | 0 | 8 | 0.500 | **0회** |
| learned_fork_proof_choice | 9 | 0 | 7 | 0.5625 | **0회** |

모든 대국은 선수 교환을 포함하지만 통계 단위는 독립된 8개 original opening block이다. **32판 전체에서 certificate의 재개방 또는 강제 착수 선택은 한 번도 실행되지 않았다.** 따라서 9–7 대 8–8의 차이는 certificate 메커니즘의 양성 증거가 아니다. 검색 시행 중의 CPU wall timing과 제한된 분기 순서 변동만으로도 결과가 달라질 수 있어 해석은 보류한다. 소규모 8-block 게임에서 일반 Elo·성능 우위를 말하지 않는다. 이는 과학적 음성 결과 또는 인과 노출 부재이지, 범용 성능 불가능 정리가 아니다.

## 4. 독립 C9 감사, 소스·실행·보존의 검증

- 실제 보드와 root 195개 전체 rule/mask/prior/방문·가치통계, 39 국면의 certificate **523개**(사전 노출 fork-000 제외), per-stratum 물리 CPU 평균과 비용 일치, 세부 게임 WDL을 독립 구현으로 재계산하여 PASS. 실제 구현 검증 산출물: evidence/C09_INDEPENDENT_REPLAY.json.
- Rust Debug/Release 각각 fresh 611 PASS/0 FAIL, binary SHA 불변. 로그 evidence/C09/DEBUG_FINAL.log와 RELEASE_FINAL.log.
- **C8 로컬 Git tar 아카이브를 실제 새로운 임시 디렉터리에 풀고** 동결 소스/모델 서명, 195개 native root 데이터와 32판 원본·최종 기록 등 핵심 **9개 파일을 바이트 해시로 대조해 PASS**. 산출물 evidence/C09_INDEPENDENT_LOCAL_RESTORE.json. Dropbox 클라우드 업로드의 독립 ACK는 아직 **NOT_VERIFIED**.
- R01–R10 및 C01–C09와 별도 C08A/C08B의 각 커밋과 아카이브는 별도의 GitHub push+remote SHA 일치 결과 및 현재 Git 계보로 재확인. 각 archive SHA/bytes는 C10_FINAL_RELEASE_MANIFEST.json에 기록. C10 최종 출판은 자기 참조 없는 별도 Git commit/hashed archive로 마무리한다.

**최종 claim 판정:** theorem certificate DERIVED, R7 finite check NUMERICALLY CHECKED, Rust 195 MCTS·32 actual full games·independent 217 move replay IMPLEMENTATION_VERIFIED. 실제 fork 재활성화 23/23 engineered sources는 구현 검증이지만 **incremental force-win gain 0**. 16 played source 및 32판의 실제 controller 발동 0, 일반 played-source 조건부 효과 **NOT_ESTIMABLE**. 인간 Einstellung, 새로운 threat-space 알고리즘, 일반적 오목 실력 향상, Elo 우위는 **UNRESOLVED/NOT_ADMITTED**.

## 5. 다음 독립 연구 단계

**QUARTZ-COG-06-EFFECTIVE-MISS-COUNTERFACTUAL-17**. 이제 단순히 빠진 fork 수를 세지 않고 **기존 top16에는 증명된 승리 수가 하나도 없고, 제외 후보에만 해당 수가 존재하는 엄격한 effective miss**를 찾는 것이 연구 질문이다. R(S)의 집합론적 차이가 0이 아닌 국면을 먼저 분리해야 root 재개방으로 승리 가능 접근성이 증가한다.

실험은 (a) 별도의 구조적 effective-miss 양성 대조군, (b) 독립된 실제 플레이 origin과 샘플링 분모를 모두 기록하는 탐색, (c) 같은 고정 learned-policy full-legal·mask·sham·top16·명시적 proof choice, (d) 20ms 실제 CPU 일치 비교와 게임별 양 색상 교환, (e) 동결 모델/소스·실행 로그와 재현가능한 증명/테스트로 분리한다. 실제 플레이에서 독립된 20개 이상 effective-miss/activated parent family를 확보하지 못하면 일반 조건부 효과는 NOT_ESTIMABLE을 정직하게 유지한다. 결과 확인 후 C16 국면으로 기존 모델/threshold를 재튜닝하지 않는다.

**다음 단계의 정확한 계약:** NEXT_STAGE.json. 수행할 때 새 별도 Git worktree에서 다시 이론 R1–R10 전부, 이어서 코딩 C1–C10 전부를 진행하고 각 단계 백업·게시·검증을 수행할 수 있도록 설정했다. 현 대화는 C16의 R1–R10/C1–C10으로 정확히 종료하고 C17을 자동 시작하지 않는다.

**권위 있는 C16 자료 경로:** 본 폴더 아래 theory, coding, evidence, COHORT_40_FROZEN.json, C10_FINAL_RELEASE_MANIFEST.json, NEXT_STAGE.json. 실제 외부 Rapfi 바이너리·NNUE는 본 연구 Git/백업에 재배포하지 않았다.
