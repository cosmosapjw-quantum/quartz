# QUARTZ C18 | 범용 계산 관측모형 보정 연구 최종 보고와 다음 단계

연구 노드 QUARTZ-GENERAL-18-CALIBRATED-COMPUTATION-OBSERVATION. 2026-11-22 KST. 기준 정본 C17 e4dc4d6a0d5db03189d2f8465f6d5f019018f20c. 새 독립 연구 브랜치 research/quartz-general-c18-calibrated-compute-20261122. 원래 QUARTZ 연구 목표는 게임 최적화가 아니라 제한된 계산에서 상태 의존적 계산 선택/중단의 메타추론 원리이고, 본 연구는 그 원리의 확률 관측모형을 실험적으로 검증하는 한정 작업이다. docs/THESIS.md, CLAIM_LEDGER 및 기존 개발·C16·C17 정본을 변경하지 않았다.

## 1. 이론 R1–R10 실제 수행

R1 원본 권위와 미완성 근거 확인, R2 Russell–Wefald/Hay–Russell–Tolpin–Shimony/Sezener–Dayan/Malik 등 원전 대비, R3 '추가 계산 뒤 결정 변경'과 '실제 효용 변화'의 구별, R4 상수/범용 로지스틱/선택적 모형 경로, R5 직접 정리와 Brier 분해/cluster variance, R6 비-IID·누락 후보·비용·지평·테스트 누수 반례, R7 Wolfram+Python Fraction 독립 검산, R8 claim 판정, R9 정의→수치→소스 계약, R10 코딩 인계를 단계별 SHA 아카이브와 GitHub 원격 대조로 완료했다.

Y=1[a1≠a0], Z=U(a1,theta)-U(a0,theta), q(X)=P(Y=1|X). 정의만으로 E[Z|X]=q(X)E[Z|Y=1,X] (정의역에서) 이므로 **revision probability는 signed value of computation이 아니다**. U의 범위 D를 알고 있을 때 |E[Z|X]|≤Dq(X)라는 상한만 주며 양의 이득을 보증하지 않는다. 동일 q=1/4에 대해 이득+1/4와 −1/4의 직접 반례가 있다. 확률 예측 Brier의 조건부 분해 E[(Y−qhat)^2|X]=p(1−p)+(qhat−p)^2를 직접 유도했다. 관측자료가 같은 parent/root를 공유하면 N observations를 iid로 셀 수 없으므로 group 단위 분리를 고정했다. R7 실제 유리수 검산은 p=3/5,qhat=4/5에서 Brier7/25=irreducible6/25+calibration1/25, 12×3 source/m=3, rho=1/2 의 design effect2도 확인했다. 이들은 표준 이론의 재정식화/검산이며 새 일반 정리가 아니다.

## 2. 코딩 C1–C10 실제 구현/실험

C1 C17·C16·frozen C15 policy/원본 문서의 SHA lock; C2 초기 source feature↔실제 Rust 상태↔후속 행동 label 대응; C3 수치·보정·분리 계약; C4 소스·split 선동결; C5 Python generic logistic/shadow 실구현과 TDD 5 RED→GREEN; C6 standalone Rust shadow CLI와 TDD 1 RED→3 GREEN, 실제 Rust debug/release 기존 MCTS 617+CLI7+new CLI3 PASS, Rust/Python 초기96점 최대 오차1.11e−16; C7 실험 코드 사전 push; 별도 C07A train+cal raw/모델 고정 push; 별도 C08A test raw 봉인 push; C8 TEST metrics와 native Rust shadow parity, C9 독립 raw 재생/소스 감사 및 C8 archive restore, C10 본 최종 보고/다음 DAG 인계. **각 정식 R/C 단계별 commit/sha256 backup/GitHub push와 원격 HEAD readback**을 수행했고 C07A/C08A도 독립 보존했다. C10 최종 커밋/아카이브의 확정은 실제 publisher + 이후 독립 remote/backup readback 결과를 기준으로 한다.

실제 C4 소스: **120개 Rust-played parent games**, 첫4수는 seeded legal prefix, 이후 **712개 실제 C16 native Rust MCTS 착수**로 354개 nonterminal source root histories를 구성했다. 소스는 72 train /24 cal /24 test 그룹으로 선분리하며 일부 parent의 조기 종료로 snapshot 수가 달라졌다. 별도 **120개 비게임 Bernoulli latent utility parent**는 K=2/3/4 arms에서 샘플링해 동일하게 분리했다. 예측기에는 game ID나 true latent means, final search 결과가 입력되지 않는다. 다만 n_actions 크기 분포가 도메인을 거의 암시할 수 있어, 공유 모델이 엄밀하게 cross-domain transfer를 보여준 것은 아니다.

C07A 실제 원시학습/보정: 568번 genuine Rust search 20ms +288 non-game sample transcripts, train644와 CAL212 observations; 원시 TEST는 일절 보지 않고 logistic weights [0.1534339791613846,-4.825294219353853,-0.10828824023012702], CAL intercept0.03688539789272929를 고정했다. 원시 모델 SHA256 2dd987989cb543c692457d0b198d23009917fd5a3001551e300a81beb868218b. C07A GitHub 게시 후 TEST 실행했다.

C08 실제 TEST: 24+24 independent parent source groups; **140개 genuine Rust MCTS search** (20ms active total wall each) + **72개 실제 non-game stochastic continuation sample paths**. Real native Rust의 observable은 '첫8 root edge 중 visit 최빈 후보'와 최종 full MCTS move가 다른지이고, **실제 엔진 first8 best_move를 직접 조회한 결과가 아니다**. 이것은 search-decision revision proxy이며 true minimax regret/positive VOC가 아니다.

| HELDOUT domain | 원시 Y 빈도 | train-only constant Brier | calibrated model Brier | calibration ECE10 | mean actual computation wall |
|---|---:|---:|---:|---:|---:|
| 140 genuine Rust MCTS roots, 24 parents | 21/140 =0.150 | 0.1281367 | 0.1163228 | 0.07337 | 20,103.5µs / **full active** decision |
| 72 stochastic non-game paths, 24 parents | 17/72 =0.2361 | 0.1881001 | 0.1941417 | 0.12376 | 3.889µs / **extra Python draws only** |

원시 model heldout Brier: Rust0.11581, non-game0.19239로 CAL intercept 보정은 양 도메인에서 Brier를 다소 악화시켰다. Rust의 calibrated model 대 constant 차이는 −0.011814이지만 250번 parent bootstrap descriptive 95% interval [−0.03141,+0.00846]로 0을 포함. Non-game 차이는 +0.006042이며 interval [−0.02395,+0.03470]로 0을 포함. **두 도메인에 공통으로 유효한 확률적 calibration 성공이나 범용 우위는 입증하지 못했다.** Rust mean predicted0.19357 vs actual event0.15, non-game mean predicted0.33516 vs actual0.23611; ECE10도 domain-specific constant baseline보다 악화했다.

사전 등록한 SHADOW 'STOP' 판단 기준 q<0.2를 단지 기록했을 때 Rust에서는 72개의 STOP 후보 중 **4개가 이후 실제 행동을 바꿨고** (missed-revision 5.56%), non-game에서는 12 STOP 후보 중 2개 (16.67%)가 이후 행동을 바꿨다. 실제 검색을 정지시키거나 절감한 계산이 하나도 없으므로 cost saving/decision non-inferiority claim은 **NOT_TESTED**. 실제 non-game 후속 계산의 평균 signed true utility gain은 +0.03033355이고 17 changed paths 중 12 positive/5 negative, parent24 그룹 수준 단순 normal descriptive interval [+0.00611,+0.05456]로 실험적으로 양수이나 생성된 독립 known Bernoulli world에만 적용된다. 원시 Rust Gomoku의 true utility label은 없으므로 E[Z] **NOT_ESTIMABLE**. 외부 Rapfi는 온라인 oracle로 사용하지 않았다.

C08 본 모델 TEST에 대해 compiled Rust shadow CLI와 실제 Python probability 최대 절대차 **2.7756e−17**, 212 rows. C09 별도 구현은 모든 **140 Rust first8 edge/action/history**와 **72 non-game RNG full stochastic paths**를 재생해 label, parent split, signed utility, Brier를 독립 재계산 PASS. C8 archived tar.gz에서 original THESIS, full source/metric/model and executable source 등 **11개 파일을 독립 temp에 복원해 byte SHA 일치 PASS**. Fresh C09 Rust debug+release 기존 MCTS **617 PASS**, C17 CLI7 PASS, C18 shadow3 PASS; Python5 PASS. Native C18 shadow 바이너리 SHA256 77b2000f90fc6f39188ec85ead2c24cf004f50f31e88aa0a90144a6335f32919.

## 3. 최종 claim gate

- **DERIVED / LITERATURE-SUPPORTED:** source-condition revision vs signed value distinction, bounded signed change, Brier decomposition, group dependence.
- **NUMERICALLY CHECKED:** Wolfram/Python exact finite counterexamples.
- **IMPLEMENTATION_VERIFIED:** actual native Rust MCTS source and 212 heldout queries, second non-game stochastic source, Python+Rust shadow forecast equality and independent rule/score/data auditor, real local archive restore.
- **UNRESOLVED / NOT_ADMITTED:** generalizable calibration, utility-positive true-game computation selection, SearchPolicy deployment, P2 physical compute savings, gameplay or other-domain performance transfer, P3 human/GM cognitive emergence.
- **SCIENTIFIC NEGATIVE:** current simple domain-neutral model does not outperform source-specific constant baseline in non-game proper score and is visibly miscalibrated in both sources; no robust advantage supported by current group bootstrap CIs.
- Native subagents NOT_RUN; completed serial without falsely claimed children. Remote Dropbox cloud sync ACK NOT_VERIFIED, local SHA backup and Git remote SHA are separate verified preservation mechanisms.

## 4. 다음 연구 ID와 기술적 의무

**QUARTZ-GENERAL-19-SIGNED-VOC-AND-SHADOW-SELECTIVITY**. 비게임 모형에는 Z label이 존재하나 Rust 게임에는 없다. 다음 노드는 참 효용/의사결정 손실을 관측 가능한 서로 다른 적어도 2가지 과제와 적절한 가격의 계산 행동을 구성하고, P(Y)·E[Z|Y=1,X]·계산비용/불확실성 모형을 별도로 예측한 뒤 유효 순이득과 안전한 선택적 계산 제안을 SHADOW 방식으로 검증해야 한다. 정확한 게임 utility를 획득하려면 독립적인 solved small game/finite MDP 등을 사용하고, 특정 게임 규칙은 제어기가 아닌 task adapter에 둔다. 실제 game-specific 성능 개선이나 휴먼 인지의 재현을 연구 목표로 바꾸지 않는다.

다음 C19 역시 이론 R1–R10 전체 다음 코딩 C1–C10 전체, 단계별 새 Git commit + 해시 Git archive + 원격 Git SHA 확인 및 부정적 결과 보존. 원래 C17, C16, developer worktree와 frozen THESIS/CLAIM_LEDGER/마스터플랜 손대지 않는다. C19은 본 C18 범위에서 자동 시작하지 않는다.

## 5. 검증 가능한 근거 경로

동일 폴더의 theory/, coding/, evidence/, C01_SOURCE_LOCK.json, SOURCE_GROUPS_FROZEN.json, SHADOW_MODEL_FROZEN.json, C10_FINAL_RELEASE_MANIFEST.json, NEXT_STAGE.json. Backend: origin/cosmosapjw-quantum/quartz research/quartz-general-c18-calibrated-compute-20261122.
