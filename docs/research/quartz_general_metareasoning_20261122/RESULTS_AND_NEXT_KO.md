# Quartz 범용 메타추론 정체성 복원 연구 | C17 최종 결과와 인계

연구일: 2026-11-22 KST. 격리 브랜치: research/quartz-general-metareasoning-identity-20261122. 원본 C16 부모 SHA: 41febde4561644d7287fd4d3411847e485f5f646. 원본 개발 워크트리 agent/local-experiment-foundry, THESIS.md, CLAIM_LEDGER.md, main은 수정하지 않았다. PR/병합/배포는 수행하지 않았다.

## 1. 원래 Quartz 연구 정체성, 확인한 근거

2026-04-09 최초 Git root commit부터 413개의 Git commit과 30개 origin refs를 실제 조사했다. GitHub 공개 저장소에 대한 all-state PR API 호출은 PR 객체 0개를 반환했다. 다만 PR00–PR05 등 이름을 가진 개발 브랜치와 Git 커밋은 실제로 존재한다. 이를 곧바로 실제 PR 생성·리뷰·병합이라고 주장하지 않는다. 각 검증된 문서 바이트 해시와 branch/commit identity는 evidence/REPOSITORY_PROVENANCE.json에 저장했다.

원본 README의 QUARTZ: Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable. 핵심 연구는 범용 제한계산 합리적 에이전트의 계산 선택, 계산 자원 배정과 중단 제어다. frozen docs/THESIS.md P1은 기대 손실 감소량/계산비용이라는 한 원리, P2는 구체적 게임에서 별도 검정할 자원·성능 비열등성 목표, P3는 최적화하지 않는 전문가 탐색 통계의 emergent prediction, P4는 CPU hot-path의 엔지니어링 제약이다. THESIS에는 게임별 전략·전술 패턴을 범용 제어기에 주입하지 않는다는 금지가 존재한다. COG15/C16 오목 결과는 범용 목표를 바꾼 것이 아니라 국소적 실험이고, 실제 성능 효과 미입증 부분도 보존했다. BQ++ May 메타추론 모듈, July 베이지안 root/omission 실험, August 26축 foundry도 각각 출처/증거 상태를 구분해 계승한다.

## 2. 이론 연구 R1–R10 실제 수행

R01 역사·권위 잠금, R02 원전과 Git 신뢰도, R03 베이지안 계산-결정 모델, R04 이득률/순이득 반례, R05 직접 유도, R06 적대적 반례·적용 한계, R07 Wolfram·독립 Python Fraction 계산, R08 근거 등급, R09 방정식→코드 계약, R10 범용 연구 DAG 재구성을 각각 개별 커밋·로컬 해시 아카이브·GitHub push 및 원격 SHA 확인으로 완료했다.

유한 세계 상태 θ, 최종 행동 a, 계산 c, 상태후험 b, 효용 U에 대해 M(b)=max_a E_b U(a,θ), K(b)=E_b max_a U(a,θ), R=K−M≥0. 보정된 계산 관측 채널 L_c 아래 gross EVSI G_c=E_y M(b_y)−M(b)≥0. 단일 단계 순이득 N=G−λκ, 비율 I=G/κ는 비용이 다르면 동일 계산을 선택하지 않을 수 있다. 한 단계 STOP의 양/음 조건은 κ>0에서 동일하다는 점을 함께 증명했다. 유한 지평 V_h=min{R, min_c[λκ+E_y V_{h−1}]}를 직접 도출했으나 기존 고전적인 메타추론 이론으로, 새 정리라는 신규성 주장을 하지 않는다.

정확한 3상태 예제: gross 정보가치 1/5 대1/2, 비용1/20 대1/5, 비율4 대5/2, net3/20 대3/10. 비율은 부분 관측을, 순이득은 전체 관측을 고른다. Wolfram 유리수와 Python Fraction에서 독립 검산, 라벨 순열6+6, H0–H3, 불가능한 관측, 정보가치0, 비용 단위 재척도 확인.

## 3. 코딩 C1–C10 및 완성된 실제 계산

C1 원본13개 핵심 문서·소스 바이트 잠금, C2 이론/코드 대응, C3 확률·비용 규약, C4 비게임 유한 과제5개 사전동결, C5 순수 Python 베이지안 계산 기준 구현과 관찰한 RED 7개 실패→GREEN 7 PASS, C6 독립 Rust 참고 구현과 RED→GREEN 6 PASS 및 CLI 실제 테스트, C7 동결 시나리오/실험자원·검정수준·정확평가 스크립트 사전게시, C8 실제 정확모형/짝지은 10,240개 시뮬레이션 경로, C9 독립 유리수·원자료·회귀테스트와 백업 실제 복원, C10 최종 보고 및 다음 DAG 인계를 각각 수행하고 개별 Git/백업을 요구한다. 모든 단계의 각 커밋/원격 상태는 C10 매니페스트에 보존하고 마지막 단계도 같은 게시 스크립트로 종료한다.

Python 구현: quartz/experiments/metalevel_decision.py. Rust 구현: src/mcts/policy/finite_metalevel.rs, 실행 가능 별도 CLI src/bin/quartz_metalevel_reference.rs. GAMESTATE나 실제 PUCT 제어기를 변경하지 않고 Rust 메타추론 참고 모델만 추가했다. Rust CLI는 진짜 릴리스 바이너리 SHA256 0a61ddacae5523acef3f4a55a8fe8372e57ac645013ddfa019c57d9dd261d741. C4 원자료 SHA256 46d69d13ea327c23e02b76c53a0a78aadc1f77ee9e405dbef1c203a6d0b471dd.

C6: Rust 디버그·릴리스 각각 주요 mcts_demo 617 PASS/0 FAIL/89 ignored 및 새 참조 CLI 7 PASS/0 FAIL, Python 7 PASS. 실제 Rust/Python 비교는 동결 5과제에서 52개 수치 (절대허용오차1e−10)와 모든 선택/중단 결정이 일치했다. 처음 실행에서 PYTHONPATH 미설정에 따른 ModuleNotFoundError가 있었으나 실행환경 문제를 분리하고 올바른 모듈 경로를 지정해 재실행했으며 수학 코드를 변경하지 않았다.

C8: 5개 독립 과제 ×4개 정책 (STOP, rate, net, Bellman) ×512개 시뮬레이션 = **10,240개 실제 합성 경로**, 전체 정확 기대손실 및 MC 평균/표준오차 보존. 대표 이질비용 과제 H2의 정확한 총기대손실: 즉시 중단0.50, 비율 기반0.21, 순이득 기반0.20, 유한 Bellman0.20. 임의 표본의 최대 exact-empirical 간격 0.029296875이며 20개 그룹 모두 사전 지정한 수치 적합 검사 통과. 원자료 SHA256 dc394b70540753ed6fe120c29bbf4215a2a27c33c7a661f5a140631a86a22c9d.

C9: 독립 Fraction으로 과제×정책 20개 정확 기대손실, 원자료10,240행의 고유/평균, 2,560개 추출 경로의 잠재상태/관측/사후분포/최종결정/실측 과금 재생까지 PASS. Rust Debug+Release 각각 617+7 PASS, Python 7 PASS로 다시 회귀 확인. C8 로컬 Git tar를 독립 임시 폴더로 풀어 중요한 9개 파일의 바이트 해시 일치 PASS. Dropbox 로컬 아카이브/복원은 검증했으나 Dropbox 클라우드 동기화 독립 ACK는 미검증이다.

**증거 등급:** 원본 목표와 Git 이력 source-verified, 베이지안 수학 derived/literature-supported, 정확 합성검정 numerically checked, Python/Rust 참고 구현 및 10,240 합성 경로 implementation-verified. 실제 MCTS의 관측법칙 보정, 범용 SearchPolicy 통합, 게임 성능/자원개선, P3 인간 유사 탐색 통계의 검정은 모두 미수행 또는 unresolved다. 표본 결과를 인간 인지의 실재 메커니즘이나 보편적 알고리즘 우위로 해석하지 않는다.

## 4. 재구성한 후속 연구 계획과 다음 단계

범용 마스터플랜: QUARTZ_GENERAL_MASTER_PLAN_v1.md. 정식 단계는 T0 finite 베이지안 계산 원리(현재), T1 실제 메타계산 결과의 관측확률/상관 구조 보정(다음), T2 계산 행동 STOP/MORE_SAMPLES/WIDEN/CHALLENGE/DEEPEN/VERIFY/REPLAY의 source-first shadow통합, T3 적어도 2개 서로 다른 과제/효용에서 독립 그룹별 전이 검증, T4 emergent 서명 분석(목적함수 최적화에 사용하지 않음).

**다음 독립 노드:** QUARTZ-GENERAL-18-CALIBRATED-COMPUTATION-OBSERVATION. 실제 Rust MCTS의 계속 계산/검색 관측과 별도 비게임 의사결정 과제를 함께 수집하여, 메타계산이 실제로 얼마나 의사결정 손실을 개선하는지 선험적으로 예측하는 보정된 관측모형을 도출/검증한다. 독립 소스 가족별 데이터 분리, 정확한 계산 비용, 부정적 결과와 시스템 성능의 구분, Shadow 실행 우선, 이후 범용 인터페이스에 제한적으로 통합할 수 있는지 판단한다.

다음 역시 연구 이론 R1–R10과 코딩 C1–C10 전체 및 단계별 SHA 보존·GitHub 게시로 진행하도록 NEXT_STAGE.json에 명시했다. 현재 C17은 이 범위에서 종료하고 C18을 자동 실행하지 않는다. 원래 개발 브랜치, 메인 브랜치와 동결 문서들은 수정되지 않았다.

## 5. 재현과 출처

현재 연구 브랜치: research/quartz-general-metareasoning-identity-20261122.
권위 있는 현 단계 문서: theory/R01..R10, coding/C01..C09, evidence/, QUARTZ_GENERAL_MASTER_PLAN_v1.md, C10_FINAL_RELEASE_MANIFEST.json, NEXT_STAGE.json.
원래 SSOT 문서: docs/THESIS.md, docs/CLAIM_LEDGER.md, docs/BQ_PLUS_PLUS_DESIGN.md, docs/RESEARCH_PLAN_PARTB.md. 기존 연구 브랜치/실험/원자료 전체 보존. Git push/원격 SHA 확인과 로컬 SHA 아카이브는 각 stage publisher 출력이 증거이며, final C10은 자기 자신을 매니페스트에 포함할 수 없으므로 별도 SHA 검증된 archive로 확인한다.
