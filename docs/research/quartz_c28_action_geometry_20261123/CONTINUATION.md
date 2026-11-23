# C28 이후의 다음 연구 계약

## 현재 checkpoint와 유지할 운영 범위

현재 과학 checkpoint는 C28-DG다. 사용자 SSOT는 C26 `87c15dc828d6eea519c219fd8a412fa21dfc464d`이며, C28 branch는 이 커밋에서 직접 시작했다. C08 `f791e96e651d1849df601dcd7b98c18bcc3fa082`가 동결된 과학 코드·측정 snapshot이고 C09 검토는 `1688ea6164db6e11d8ceca3634d45f269cd66303`에 있다. 최종 C10 HEAD와 복원 결과는 해당 branch, `LATEST.json`, `FINAL_RESTORE_AUDIT.json`을 대조한다. 다른 C27 worktree와 그 history는 별도 소유로 유지한다.

사용자가 갱신하기 전까지 연구 turn의 기본 범위는 다음과 같다. 원동기 drift watchdog을 실행하고 실제 이론 R01–R10을 모두 수행한 다음 실제 코딩 C01–C10을 수행한다. 각 단계는 실질적 산출물을 갖추고 게시·백업·원격 readback을 완료한다. 보존할 최신 누적본이 검증된 뒤 그 단계에서 만든 이전 중복본만 정리한다. 이미 확인한 근거는 identity와 적용 범위를 확인하여 재사용하며, 검토 기록을 늘리는 것이 실제 유도·실험을 대신하지 않는다.

원동기 watchdog의 비교 기준은 `ORIGIN_DRIFT_WATCHDOG.md`의 원 PDF/PRD와 정확한 역사 commit이다. 새 turn에서는 그 핀과 현재 변경 내용을 다시 연결하고, resource-priced decision gain, actual GameState, 실제 CPU 비용, FOCUS/EXPAND/MERGE/STOP 의미 및 P2/P3의 미해결 상태가 유지되는지 확인한다. 모든 옛 로그를 무조건 다시 실행할 필요는 없다. 원문 오류의 조건 교정과 연구 동기의 포기는 구분한다.

## 다음 한 가지 우선 목표: 잡음 있는 질의와 terminal decision

**추천하는 C29 질문:** “짧고 잡음 있는 계산 결과를 관측할 때, 어떤 posterior 변화가 최종 배치와 다음 계산 선택을 바꾸며, 그 이득이 실제 계산 비용을 지불하는가?”

선택 이유는 명확하다. C28은 정확한 공개와 known-only STOP 안에서 이론과 구현을 닫았다. 실제 게임 evaluator는 보통 잡음 있는 정보를 제공하고 반복 평가가 가능하다. 이 차이를 다루는 것이 같은 finite example의 timing을 다시 재는 것보다 원래 계산 배분 목표에 직접 연결된다. 단순한 표기 변경으로 C28의 증명을 옮기지 않는다.

처음 고정할 수학적 계약은 다음과 같다.

- 숨은 효용은 우선 C28과 같은 유한 support를 사용한다. 초기 joint prior는 비교 가능한 product/common/signed 계열이다.
- 계산 행동 `query(i)`는 정확한 $U_i$ 대신 알려진 유한 channel $C_i(y\mid u)$에서 $Y$를 관측한다. 반복 관측의 조건부 독립성 여부를 명시한다.
- STOP에서 후보 전체를 사용할 때의 terminal value는 $\max\{b,\max_i E[U_i\mid h]\}$다. C28의 known-only 배치와 이 계약은 다르다.
- Binary latent mean $m$ 하나만으로 충분한 상태가 유지되는지 다시 유도한다. 필요하면 각 latent component의 후보별 조건부분포까지 상태에 포함한다. 관측 후 local law가 바뀌는 것을 기존 고정-law certificate에 숨기지 않는다.
- 우선 H2, 고정된 비음수 질의 비용, 명시적 tie rule로 제한한다. 실제 ns와 효용 가격은 단위를 맞춰 별도 다룬다.

**수용 조건:** finite-world posterior/tree enumeration과 갱신식·Bellman 값이 일치해야 한다. 정확 관측 channel와 C28 배치 모드를 명시적으로 복원하면 C28 결과를 회복해야 한다. 무정보 channel와 양의 비용에서는 정보 질의로 양의 순이득을 만들면 안 된다. 반복 질의가 이미 확정된 효용에 정보를 더한 것으로 잘못 계산되어서는 안 된다. STOP 기준 변경의 효과와 dependence 변경의 효과를 서로 혼동하지 않는다.

이것은 다음 연구의 제안된 계약이다. 잡음 있는 관측에 대한 새 closure, 선택 경계, 속도 또는 게임 효용은 아직 증명하거나 측정하지 않았다.

## 다음 이론 루프 R01–R10

| 단계 | 실제 작업 | 남길 산출물 / 결정 |
|---|---|---|
| R01 | C28 핀, 원동기, noisy-query/STOP 계약과 목표 고정 | 정의·단위·관측·배치·비용 계약과 drift 차이 |
| R02 | 기존 finite POMDP 및 metareasoning 원전에서 필요한 명제만 전이 | 정확한 정리·가정·C28/C29 terminal 차이 |
| R03 | joint prior와 관측 likelihood로 posterior 상태를 유도 | 정규화·충분상태·반복 관측 갱신식 |
| R04 | H1과 H2 Bellman 식 및 unnormalized computational form 유도 | STOP 포함 전체 행동 점수, 비용 계상 |
| R05 | 관측 noise와 dependence가 행동 경계를 바꾸는 구성 계산 | 실제 선택이 갈리는 예와 닫힌 조건 또는 정확 finite boundary |
| R06 | 새 핵심 주장 하나에 독립 수학 심사 | 반례·필요한 제한·수정된 정확한 주장 |
| R07 | small complete-world tree를 실제 exact arithmetic으로 계산 | perfect/no-information/degenerate/repeated-query 극한과 유리수 결과 |
| R08 | 계산 이득, 모형 표현력, 기존 결과의 조건을 분리해 판정 | 살아남은 주장, 실패한 메커니즘과 재개 조건 |
| R09 | 구현·오차·데이터 분리·실제 비용·실패 기준 동결 | 코드 계약과 실험 계획; TEST 보기 전 결정 |
| R10 | 완성된 유도·계산을 코딩 단계로 인계 | 식→함수 대응, 미해결 사항, 실행 가능한 C01 진입점 |

## 다음 코딩 루프 C01–C10

| 단계 | 실제 작업 | 남길 산출물 / 결정 |
|---|---|---|
| C01 | 현재 branch·source·환경 및 새 data cohort 가용성 확인 | 정확한 핀, 입력 provenance, 이론과 구현의 범위 |
| C02 | posterior state, channel, STOP, query API와 자료구조 설계 | 각 수식에 대응하는 함수·상태·입출력 계약 |
| C03 | 작은 likelihood·zero support·정밀도·tie 처리 설계 | 안정적 계산식과 독립 oracle 허용오차 |
| C04 | 구현/검증 owner와 제한된 계산 자원 고정 | 충돌 없는 작업 순서와 측정 경계 |
| C05 | noisy-query native controller와 별도 world oracle 구현 | 동작하는 전체 경로; source와 binary identity |
| C06 | 의미론·protocol·한계 경우를 실제 실행 | C28 compatibility, 무정보 channel, posterior/배치/반복질의 검사 |
| C07 | 실제 GameState의 짧은 rollout을 noisy observation으로 연결 | 입력에 oracle label이 없고 실제 관측·비용만 쓰는 trace 검증 |
| C08 | 동결된 cohort·budget에서 모형 및 계산 비용 비교 | 원시 측정, action changes, realized regret/utility, 비용 포함 판정 |
| C09 | 한 번의 독립 코드/과학 검토 및 실제 백업 복원 | 검토 범위·실질적 수정·복원 근거 |
| C10 | 결과·실패·다음 판단을 문서화하고 게시 | 20단계 실제 완료 기록과 최종 원격/백업 identity |

실제 rollout channel은 이론의 알려진 channel과 자동으로 동일하지 않다. 실제 게임의 channel을 추정한다면 TRAIN에서만 적합하고 CAL에서 모형·비용 가격·budget을 고정한다. Offline exact utility는 적합과 검증에만 쓰며, runtime controller 입력에 전달하지 않는다. Conditional calibration의 실패를 posterior algebra나 native 구현 오류와 분리한다.

새 policy-generalization 주장이 필요하면 이미 관측한 C26/C27/C28 source를 TEST로 다시 이름 붙이지 않는다. Board identity, transposition 및 symmetry group을 고려하여 새로운 cohort를 확보할 수 있는지 먼저 확인한다. 확보하지 못하면 결과를 retrospective/model-check 수준으로 남기고 새 TEST로 주장하지 않는다. Gomoku 같은 더 큰 도메인으로 넘어갈 때 exact reveal 가정을 유지한 척해서는 안 된다.

모형 비교는 동일한 초기 cue·marginal·관측 budget·STOP 계약·native kernel을 공유하도록 설계한다. Product, common, signed의 차이와 direct/lifted 계산 방식의 차이를 분리해야 한다. 실제 matched-time 효용의 기준은 결과를 보기 전에 고정한다. 작은 kernel 절감이나 posterior 변화만으로 게임 효용을 승인하지 않는다.

## 보존할 열린 경로와 재개 조건

| 열린 경로 | 현재까지의 상태 | 다음 최소 행동 / 재개 조건 |
|---|---|---|
| Noisy-query metareasoning | C28 범위 밖; C29의 우선 경로 | 위 관측/STOP 계약에서 충분한 posterior 상태를 유도 |
| 게임 signed dependence 학습 | 구성 예는 성공; 실제 게임에 맞춘 signed fit은 없음 | TRAIN-only 적합, 표현력 진단, CAL 동결, 새 TEST 확보 |
| 단일 latent factor의 표현력 | 세 covariance의 곱이 비음수라는 제한이 증명됨 | 실제 필요한 dependence가 이 제한을 위반하는지 확인한 후에만 다인자 확장 |
| 전체 게임 비용 절감 | 현재 300루트에서 이득 입증 안 됨 | 더 큰 실제 후보 집합 또는 다른 측정 가능한 비용 지배항이라는 새 조건이 있을 때 재개 |
| Gomoku P2 / 인지 P3 | 이번 범위에서 해결하지 않음 | 실제 evaluator·동일 자원 비교·각 도메인의 검증 계약 필요 |
| TT geometry / RG·Wick·criticality / 물리적 해석 | 원래의 연구 관심으로 보존; 새 동등성 없음 | 정확한 source-to-target 수학 사상과 검증할 관측량이 준비될 때 별도 연구 |

**바로 할 다음 행동은 R01–R03이다.** 새 noisy-observation 계약과 충분한 상태를 먼저 고정하고 실제로 유도한다. C28의 성공한 계산이나 초미세 timing을 다시 실행하는 일로 다음 turn을 소모하지 않는다. C28의 exact runner와 native tests는 변경이 영향을 미치는 범위에 한해 재사용한다.
