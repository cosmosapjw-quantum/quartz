# QUARTZ-COG-04-HIGH-BRANCH-TRANSFER-14: 최종 연구 판정 및 인계

2026-11-21 (Asia/Seoul). 이전 C13 정본 `d6875102ebc07d2eba899d073ac7214a7f7d9dad`. 본 연구 브랜치: `research/quartz-cog04-highbranch-transfer-20261121`.

**판정:** 15×15 Freestyle에서 225개 행동을 표현하는 동일 MCTS 트리의 루트 후보 재활성화, 공개 사전학습 엔진 Rapfi의 실제 탐색, 그리고 전술적 즉시 승리 수 복구가 구현·실험 검증됐다. 그러나 **기준선 대비 CPU 비용 일치 조건을 충족하지 못했고, 일반적인 오목 실력 향상이나 비근시안적 인지 기제의 우위는 미입증**이다.

## 이론 R1–R10: 새로 도출한 내용

`theory/R01_QUESTION.md`부터 `theory/R10_THEORY_HANDOFF.md`까지 열 단계가 각각 별도 Git commit/백업/원격 대조를 통과했다. 핵심 수학 결과는 합법 행동 수 (B), 초기 후보 (k), 제외된 결정적 행동 (m), 무작위 재개방 수 (r)에 대하여
[
P_{\mathrm{hit}}=1-\frac{\binom{B-k-m}{r}}{\binom{B-k}{r}},
]
라는 정확한 **후보 발견 확률**이다. 이는 무작위 제외 후보에서 중요한 후보를 하나 이상 찾을 확률을 계수로 센 것이다. (B=7,k=3,r=2,m=1)에서 (1/2), (B=225,k=16,r=4,m=1)에서 (4/209≈1.91\%\)다. 225개 행동의 4×u64 마스크에 대해 각 index와 포함관계를 수학적으로 유도했다. Wolfram의 정확 계산과 별도 Python Fraction 계산, 225 singleton 및 25,200 action pair 테스트로 검산했다. 이는 후보 발견만을 의미하며 **실제 방문·최종 착수·게임 승리 확률을 보장하지 않는다**.

사용한 기존 연구: Chaslot et al., *Progressive Strategies for Monte-Carlo Tree Search* (2008), https://doi.org/10.1142/S1793005708001094 ; Cao & Lin (2019), *UCT-ADP Progressive Bias Algorithm for Solving Gomoku*, https://scispace.com/papers/uct-adp-progressive-bias-algorithm-for-solving-gomoku-4sy39vnaou . Progressive unpruning 자체는 선행 기법이며 새로움은 **확인 가능한 코드·실험적 이행**의 범위 안에서만 주장한다.

## 코딩 C1–C10: 실제 구현·실험

C1–C4: `coding/C01_INPUT_LOCK.json`에서 Rust source와 실제 Rapfi Linux AVX2 binary, 고전 모델, freestyle·standard·Renju NNUE, config bytes를 각각 SHA256로 고정했다. 24개 Freestyle 국면을 Rapfi 결과 보기 전에 봉인했다. `COHORT_24_FROZEN.json` SHA256 **`a744fa5166cb1cd54ec274128238649e7305a1ae3c4f15b894f35559602f5a9c`**. 자연적 중반/오프닝 계열 12개와 의도적인 즉시 승리 전술 스트레스 12개를 분리했다. 후자는 소수 전술 템플릿의 대칭·이동을 이용했으므로 IID 모집단이 아니다.

C5–C6: 기존 64비트 후보 마스크를 **[AtomicU64;4]**로 확대해 0..224 모든 행동과 64비트 경계의 활성화를 지원했다. 실제 Rust `MctsEngine`의 루트, 모든 원래 edge, prior, 누적 방문·가치·M2 및 TT hash는 그대로 두고 first8→total32 **같은 트리** 탐색을 수행했다. 모든 자식의 원래 합법 수는 유지된다. TDD와 회귀 테스트를 실제 실행했다. 최적화 Rust source commit **`7b64a681d7f0be37d10eb4417d079499a3fe1d45`**, 실행파일 SHA256 **`435dfe735306f5bf4313f466d828e9569394b93ae04c487bc0c5e25e38682b14`**.

C7: 별도 로컬 실제 **Rapfi 2025-06-15** alpha-beta/NNUE engine을 Piskvork 규칙0 Freestyle로 실행했다. Linux AVX2 SHA256 **`4217c3202cdd8dbb58524110d9064299e314a695e58ddbd055d9a4dd99de201c`**, 해당 freestyle NNUE weight **`6bc0d1b0ff8e1d857f7f412923cd458f38f1a435087c91ae71fec3676c23ef62`**. 정확한 출처/규칙/모델을 기록하고 300ms와 600ms 두 조건에서 **24×2=48회** 실제 reference search를 수행했다. Rapfi 출력은 학습된 독립 엔진의 **유한 검색 추천 수**일 뿐 Gomoku exact minimax oracle이 아니다. 이 NNUE가 QUARTZ의 ONNX 모델로 자동 변환됐다는 주장도 하지 않는다.

C8: 봉인된 24개 Freestyle 국면 × QUARTZ 8개 정책 = **192개 실제 CPU Rust 요청**, 총 **384회 동일 트리 8→32단계 호출**. 실제 runner source commit **`03c85329844adcb953c45ebcfd1f064c0a248cd7`**. QUARTZ의 현 평가기는 **UniformEval + 명시된 중심거리 prior 후보 순서**, Rapfi 모델이 QUARTZ 내부 prior는 아니다. 정책: 초기 top16 유지, 한 수 앞 전술 증거로 최대4개 재활성화, 동일 개수 무작위3종, 8회 후 전체 재개방, 처음부터 전체 합법 수 탐색, 비용 대조군. 외부 Rapfi 결과는 온라인 재개방에 사용되지 않았다.

| 구분 | 전술 증거 재개방 | 전체 합법 수 MCTS | 재개방 vs 전체 CPU |
|---|---:|---:|---:|
| 자연 12국면, Rapfi 300ms 착수 일치 | 0/12 | 0/12 | 1.506 vs 1.2077ms |
| 전술 스트레스 12국면, 즉시 승리 수 선택 | **12/12** | 6/12 | 0.72775 vs 1.07083ms |

전술 스트레스에서 재개방군은 Rapfi 300/600ms의 착수 추천과 12/12 일치했다. 무작위 재개방3종은 각각 0/12,1/12,0/12. 스트레스 국면의 즉시 승리 수는 **구성 시 주입된, 룰 검사로 확인할 수 있는 전술 표적**이므로 일반 게임 실력이나 새 인지 이론의 실험 증거가 아니다. 자연 12국면은 어떤 QUARTZ 정책도 Rapfi 추천 수와 일치하지 않았다. 실제 Rapfi 두 예산의 추천은 11/12 동일했다.

**물리적 계산비용 판정:** 사전 규약은 두 집합 각각 **실제 elapsed wall search cost 평균 차이 ≤10%**다. 자연 국면은 **19.81%**, 전술 스트레스는 **47.14%**로 둘 다 FAIL이다. 전체 24개를 합치면 약2.0%이지만 서로 반대 방향의 시간 차이가 상쇄되는 오류이므로 **비용 일치 증거로 인정하지 않는다**. 전술 승리 발견이 탐색시간을 단축하는 종료 조건에 영향을 주는 점 역시 기록한다. Rapfi의 300ms/600ms 시간 예산과 QUARTZ 32회 방문도 동등한 compute가 아니다.

C9: 이전과 독립적으로 작성된 `coding/audit_highbranch14.py`는 192개 실행 기록의 게임규칙·225비트 root mask·루트 통계·전술 증거·FNV 후보 선정·Rapfi 추천 착수 일치·실제 CPU 시간·비용 판정을 각각 다시 계산했다. `evidence/live_192_cpu_20261121/INDEPENDENT_READBACK.json` 및 `VERIFICATION_RECEIPT.json` 보존. **현재 재실행** Rust release 606 PASS, debug 606 PASS (각 FAIL0, ignored89), 관련 Python 15개 테스트 묶음 총 **140 PASS**. 최종 C9 원격 Git SHA **`d99074bf179d5c252cf5cb7226d84e9410bd4fb1`**.

C10: 본 과학적 판정·재현 경로·다음 단계 인계의 게시. 기본 개발 worktree `agent/local-experiment-foundry`와 기존 미추적 파일은 변경하지 않았다. 외부 GPLv3 Rapfi 바이너리·NNUE는 연구 Git/백업 아카이브에 재배포하지 않았다. 로컬 Dropbox 아카이브 생성/해시 확인은 각 단계별 이행했고, **Dropbox 클라우드 동기화 승인·독립 복원 시험은 미실행**.

## Claim gate 및 다음 연구

**DERIVED:** 높은 분기수에서 균등 무작위 누락 후보 복구 확률이 낮아진다는 정확한 유한조합론적 결과. **IMPLEMENTATION-VERIFIED:** 225개 행동, 동일 루트 방문 통계 보존, 실제 QUARTZ CPU 검색, 외부 Rapfi NNUE 실행. **NUMERICALLY CHECKED:** 구성한 12개 전술 판에서의 즉시 승리 수 복구 12/12, 자연 12국면의 Rapfi 불일치 0/12. **UNRESOLVED/NOT-ADMITTED:** 계산비용을 맞춘 전형적인 15×15 오목의 성능 개선, 새로운 비근시안적 metareasoning 우위, 인간 Einstellung 메커니즘.

**다음 독립 연구 ID: `QUARTZ-COG-04-TRAINED-PRIOR-FAIR-COST-15`.** 자연 국면 성능이 현재 UniformEval에 묶여 있는 문제와 두 집합의 CPU 비용 비교가 불일치한 문제를 분리한다. 먼저 진짜 Gomoku15 정책망의 **실제 가중치·입출력·규칙·검증분포**를 고정하거나, Rapfi를 외부 교사로 사용하는 별도 훈련-평가 분리를 설계해야 한다. Rapfi mix9svq NNUE는 QUARTZ ONNX가 아니다. 두 번째로 **검증집합에서만 고정한 CPU 예산/정책**으로 자연 국면과 전술 국면을 각각 독립적으로 비교하고, 생존·종료 상태에 따른 시간 편향과 다중 좋은 착수 문제를 분리한다. C14 공개 시험집합을 재튜닝 데이터로 사용하지 않는다. 새 연구에서 결과가 음성이라면 억지로 threshold를 낮춰 성공으로 만들지 않는다.

**C14 종료 조건 충족:** R1–R10 및 C1–C10의 bounded stage deliverables와 검산된 음성/양성 결과를 모두 보존한다. 새로운 루프를 자동으로 시작하지 않는다.
