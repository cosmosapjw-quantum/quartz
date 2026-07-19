# QUARTZ Idea Foundry — 실험 후보 통합 색인

이 문서 묶음은 현재 QUARTZ의 Rust MCTS 엔진과 Phase-15 Python trace
인프라 위에서 실험할 후보를 **26개 축**으로 정리한다. 각 축은 처음부터
성능 향상 기능으로 켜지지 않는다. 먼저 trace replay 또는 synthetic lab에서
관측되고, 그다음 shadow, 조건부 online, 배포 후보 순으로 승격한다.

## 공통 코드 계약

Rust의 기존 `SearchPolicy`는 루트 통계를 주기적으로 `observe`하고,
`score_adjustment`에서는 캐시된 값을 O(1)로 읽으며, `should_halt`에서 정지
여부를 반환한다. 기존 `PolicyCachePublisher`는 `ArcSwap`을 이용하고,
`edge_pos`와 게임의 `action_id`를 분리하며, stale certificate 정지를 막는다.
새 foundry 코드는 이 구조를 대체하지 않고 상위에 다음 계약을 추가한다.

```text
FoundryAxis.observe(snapshot) → MetaProposal[]
MetaProposal = 계산 행동 + 기대 regret 감소 LCB + 비용 + guard + telemetry
ConservativeArbiter → checkpoint마다 명시적인 계산 행동 하나만 선택
```

온라인 모듈이 각자 PUCT에 bonus를 더하지 않는 것이 핵심이다. 동시에 여러
shadow observer는 실행할 수 있지만, 실제 계산 행동은 한 arbiter가 선택한다.

## 26개 축

| ID | 축 | 초기 상태 | 주요 행동/산출물 |
|---|---|---|---|
| A01 | calibrated stop council | SHADOW | `STOP`, 잘못된 결정 위험 보정 |
| A02 | static-anchor RPO | SHADOW | 누적 refresh 없는 임시 root policy |
| A03 | 불확실성 분해 | SHADOW | MC/epistemic/drift/bias bounds |
| A04 | KG/VOC allocator | SHADOW | `SAMPLE`, `CHALLENGE` |
| A05 | counterfactual meta teacher | SEED | 동일 snapshot의 fork label |
| A06 | Gumbel + Sequential Halving | MECHANISM_VALID | 저예산 후보 coverage와 배분 |
| A07 | residual-evidence widening | SEED | `WIDEN`, live-set 밖 mass bound |
| A08 | tactical proof backend | CONDITIONAL | `PROVE`, 강제 수 보호 |
| A09 | H3 change-point router | SHADOW | entropy/margin 변화 기반 `DEEPEN` 후보 |
| A10 | prior-refresh specialist | DORMANT | 약한 evaluator/OOD 전용 조건부 실험 |
| A11 | dynamic live-set particles | SEED | `RESAMPLE_MODE`, 휴면·부활 후보군 |
| A12 | JSD locally balanced sampler | SEED | sibling geometry 기반 root sampling |
| A13 | pending-flow / WU-UCT | CONDITIONAL | 미완료 작업과 완료 evidence 분리 |
| A14 | semantic path LSH | SHADOW | 고스레드 whole-path 중복 진단 |
| A15 | service-curve scheduler | MECHANISM_VALID | `SET_BATCH`, `SET_INFLIGHT` |
| A16 | graph/state sharing | SEED | state cache와 parent-edge stats 분리 |
| A17 | B13 curvature readout | MECHANISM_VALID | decision-neutral policy readout |
| A18 | diffusion-regularized evaluator | SEED | 학습 보조 denoising, 직접 추론 |
| A19 | RW-ResT Lite evaluator | SEED | sparse local-global 평가기 |
| A20 | regret state archive | SEED | `ARCHIVE_STATE`, 재시작·재분석 데이터 |
| A21 | signed-path coherence shadow | ANALYSIS_ONLY | bounded disagreement feature |
| A22 | physics falsification dashboard | ANALYSIS_ONLY | beta/RG/redundancy/null tests |
| A23 | CPU incremental pattern student | SEED | quantized make/unmake 평가기 |
| A24 | learned budget gate | SEED | 상태별 추가 budget 또는 stop |
| A25 | MENTS soft backup | DORMANT | root/shallow entropy backup ablation |
| A26 | exact nested-contour lab | ANALYSIS_ONLY | 작은 유한 모형의 정확도 검증 |

## 문서 분할

- `01_control_policy.md`: A01–A05, A09, A10, A24
- `02_candidates_allocation.md`: A06–A08, A12, A25, A26
- `03_parallel_backends.md`: A11, A13–A16
- `04_representation_training.md`: A17–A20, A23, A24 학습 데이터
- `05_analysis_physics.md`: A21, A22, A26 및 물리 유비의 승격 조건
- `06_execution_and_meta_analysis.md`: A01--A26 축별 실행, 직렬 재개, 캠페인 분석 및 메타분석

## 코드 위치

```text
quartz/idea_foundry/
  contracts.py    공통 snapshot/proposal/cost/arbiter 계약
  control.py      A01-A05, A24
  search.py       A06-A16, A25-A26
  learning.py     A17-A23

src/mcts/foundry/
  types.rs        공통 Rust 계약
  control.rs      A01-A05, A24
  search.rs       A06-A16, A25-A26
  systems.rs      A17-A23

configs/idea_foundry.axes.v1.json
  26개 축의 문서, skeleton symbol, 의존조건을 기록한 registry

tests/test_idea_foundry_skeletons.py
  registry 완전성, 모든 Python skeleton 호출, index 분리, 정책 정규화,
  conservative cost arbitration을 확인

scripts/idea_foundry/a01_*.py ... a26_*.py
  각 축의 describe/run/analyze/run-and-analyze 고정 진입점

scripts/idea_foundry_run_all.py
  26축을 레지스트리 순서로 실행하고 검증된 축만 재개 skip

scripts/idea_foundry_analyze_campaign.py
scripts/idea_foundry_meta_analyze.py
  단일 캠페인 집계와 동일 estimand의 독립 효과만 허용하는 메타분석

configs/idea_foundry.studies.v1.json
scripts/idea_foundry_study.py
scripts/idea_foundry_study_all.py
scripts/idea_foundry_study_analyze.py
  계약 게이트와 분리된 26축 첫 과학 게이트, 직렬 재개, 축내 메타분석
```

Rust 모듈은 이 단계에서 의도적으로 `src/mcts/mod.rs`에 연결하지 않는다. 따라서
production search semantics는 바뀌지 않는다. Python skeleton은 바로 import와
trace replay 실험에 사용할 수 있다. Rust 연결은 다음 순서로 수행한다.

1. `SearchSnapshot`을 몰래 재해석하지 말고 `FoundryRootExtras`에 새 field를 추가한다.
2. `observe` checkpoint에서 모든 axis proposal을 생성한다.
3. arbiter가 고른 proposal만 안전한 executor에 전달한다.
4. `score_adjustment`에는 `PolicyCache`에 publish된 root-only vector만 노출한다.
5. fresh snapshot/edge hash가 아닌 경우 `STOP`과 proof certificate를 거부한다.
6. Rust module을 feature gate 또는 명시적 experiment branch에서만 연결한다.

## 실행 확인

bundle을 repo root에 덮어쓴 뒤:

```bash
venv/bin/python -m pytest -q tests/test_idea_foundry_skeletons.py
venv/bin/python -m py_compile quartz/idea_foundry/*.py
python -m json.tool configs/idea_foundry.axes.v1.json >/dev/null
```

첫 실제 trace 실험은 새 엔진 기능이 아니라 기존 Phase-15/Stage-7 trace를
이용한 A01/A02/A03/A09/A17 replay로 구현되어 있다. A05의 현재 첫 게이트는
`STOP/SAMPLE/WIDEN`만 사용하는 deterministic synthetic freeze/fork이며,
보존 trace가 independent restart이므로 resident-root 효능으로 승격할 수 없다.

현재 즉시 실행 가능한 26축 first-gate 계약 캠페인과 첫 과학 게이트 명령은
`06_execution_and_meta_analysis.md`를 따른다. 이 캠페인의 contract pass는
실험 효능이나 play strength 검증이 아니다.
