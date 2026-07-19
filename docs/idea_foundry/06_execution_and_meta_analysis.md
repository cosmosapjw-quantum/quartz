# 26축 순차 실행·분석·메타분석 런북

이 런북은 A01--A26을 각각 독립적으로 실행·분석하고, 동일 계약을 사용한
전체 캠페인을 순차 실행한 뒤, 이후 축적될 비교 효과를 보수적으로
메타분석하는 방법을 정의한다. 기존 `scripts/idea_foundry/a01_*.py` 계열은
synthetic/trace/shadow/conditional **계약 게이트**이고,
`scripts/idea_foundry_study.py` 계열은 별도의 **첫 과학 게이트**다. 어느 쪽의
통과 여부와 진단 플롯도 play strength 또는 production readiness 증거가 아니다.

## 0. Ablation 진입 전 release preflight

실제 ablation을 시작하기 직전에 다음 단일 진입점을 실행한다.

```bash
venv/bin/python scripts/idea_foundry_preflight.py \
  --run-id pre-ablation-<run-id> \
  --mode release \
  --python venv/bin/python \
  --timeout-seconds 1800
```

`release` 모드는 다음 유한 상태공간과 실행 경로를 fail-closed로 검사한다.

- 원본 ZIP/PATCH hash, 21개 payload 목록, 적용 commit과 ZIP의 byte parity;
- A01--A26 coverage, lane 순서/DAG, 각 축 CLI와 artifact/schema/hash 계약;
- 안전·위험 run ID, output confinement, missing/duplicate/absolute/symlink artifact;
- timeout, technical failure, scientific negative, source/input/plan drift와 verified resume;
- effect 필수 필드, 유한 수치, representable variance, 여섯 compatibility key,
  독립 group 중복, 모든 입력 순열, source path/hash와 극단 부동소수점;
- default/`idea-foundry` Rust build, 전체 Python 회귀, eager real-loop,
  Phase-15 deterministic CI smoke;
- CUDA doctor, A15/A18/A19 readiness 실행·resume, A18 study input inspect;
- 차단된 live/accelerator promotion lane이 계속 exit code 2로 닫혀 있는지 확인;
- 별도 A01--A26 순차 캠페인, 26축 verified resume, campaign 분석과
  `NO_COMPARABLE_EFFECTS` 메타분석.

결과는 `results/idea_foundry_preflight/<run-id>/`의
`preflight_state.json`, `preflight_report.json`, 단계별 stdout/stderr 및
SHA-256으로 남는다. 실행 중 source worktree가 바뀌면 마지막 gate에서 실패한다.
깨끗한 checkout에서는 등록된 Idea Foundry Python 실행·분석 소스 전체를
lint하고, dirty checkout에서는 그 고정 목록에 모든 변경 Python 파일을 합쳐
검사하므로 커밋 직후에도 preflight를 재현할 수 있다.
`readiness.ablation_execution_preflight=READY`는 기술적 실행 준비만 뜻한다.
과학적 효능은 `NOT_EVALUATED`, 자동 claim 승격은 `FORBIDDEN_AUTOMATICALLY`로
고정된다.

빠른 개발용 부분집합은 `--mode quick`으로 실행할 수 있지만 실제 ablation
진입 승인에는 `release` 결과만 사용한다. "모든 가능한 경우"는 무한한 임의
입력을 뜻하지 않으며, 위 계약이 정의한 유한 분기와 경계값을 완전 열거한다.

## 1. 축별 스크립트

각 축은 `scripts/idea_foundry/a01_*.py`부터
`scripts/idea_foundry/a26_*.py`까지 하나의 고정 진입점을 가진다. 진입점은
레지스트리에 기록된 축 ID와 `first-gate-all` 역할을 강제한다.

```bash
# 등록 계약 확인
venv/bin/python scripts/idea_foundry/a06_gumbel_sequential_halving.py \
  describe --json

# 실행만
venv/bin/python scripts/idea_foundry/a06_gumbel_sequential_halving.py \
  run --output-dir results/idea_foundry_single/a06_<run-id> --seed 20260718

# 기존 실행을 분석
venv/bin/python scripts/idea_foundry/a06_gumbel_sequential_halving.py \
  analyze --input-dir results/idea_foundry_single/a06_<run-id>

# 실행 후 즉시 분석
venv/bin/python scripts/idea_foundry/a06_gumbel_sequential_halving.py \
  run-and-analyze \
  --output-dir results/idea_foundry_single/a06_<run-id> \
  --seed 20260718
```

축별 실행 디렉터리는 다음 파일을 가진다.

```text
run_manifest.json
rows.jsonl
summary.json
analysis/
  analysis_manifest.json
  analysis.json
  analysis_rows.jsonl
  diagnostic.png
```

분석기는 실행 manifest의 artifact SHA-256과 schema를 다시 확인한다. 기존
`analysis/`가 있으면 모든 입력·출력 hash와 aggregate가 유효한 경우에만
재사용한다. 현재 계약 분석의 `effect_records`는 의도적으로 빈 배열이다.

## 2. A01--A26 순차 캠페인

계획 확인과 실행은 다음과 같다. `--campaign-root`를 바꿀 때도 저장소의
`results/` 아래만 허용된다.

```bash
venv/bin/python scripts/idea_foundry_run_all.py plan --json

venv/bin/python scripts/idea_foundry_run_all.py run \
  --run-id first-gate-all-<run-id> \
  --seed 20260718 \
  --timeout-seconds 300

venv/bin/python scripts/idea_foundry_run_all.py status \
  --run-id first-gate-all-<run-id>

venv/bin/python scripts/idea_foundry_run_all.py resume \
  --run-id first-gate-all-<run-id> \
  --seed 20260718 \
  --timeout-seconds 300
```

순서는 `configs/idea_lab.local.v2.json`의 `first-gate-all`을 그대로 따른다.
각 축은 별도 process group, attempt 디렉터리, stdout/stderr 로그를 가진다.
timeout, SIGINT, SIGTERM은 자식 process group을 정리하고 중단 attempt를
보존한다. 기술적 실패는 즉시 전체 캠페인을 멈춘다. `resume`은 Python
interpreter, 두 레지스트리, 공통 모듈, 오케스트레이터, 26개 진입점 hash가
동일하고 축별 실행·분석 산출물이 모두 유효할 때만 완료 축을 건너뛴다.
실패 또는 중단 축을 재개하면 기존 결과를 덮지 않고 다음 `attempt-NNN`에
기록한다.

캠페인 디렉터리는 `campaign_state.json`, `campaign_summary.json`, 축별
attempt, streaming log를 남긴다. 정상 종료 상태는
`completed_no_promotion`이며 자동 승격은 항상 거부된다.

## 3. 단일 캠페인 종합 분석

```bash
venv/bin/python scripts/idea_foundry_analyze_campaign.py \
  --campaign-dir results/idea_foundry_sequential/first-gate-all-<run-id>
```

`campaign_analysis/`에는 26축 행, 상태·evidence 집계, 전체 계약 check 수,
진단 플롯과 입력·출력 hash manifest가 생성된다. 정확히 26축이 완료되지
않았거나 어느 축의 hash/schema가 맞지 않으면 분석은 실패한다. 축마다 다른
계약 check 수나 fixture 수는 coverage 진단으로만 표시하며 효과로 pooling하지
않는다.

## 4. 이후 효과기록과 메타분석

향후 paired ablation이 실제 효과와 표준오차를 산출하면 JSON 또는 JSONL의
`effect_records`에 다음 필드를 기록한다.

| 필드 | 계약 |
|---|---|
| `axis_id` | A01--A26 |
| `estimand_id` | 사전등록된 동일 추정대상 ID |
| `effect_scale` | risk difference, log ratio 등 동일 척도 |
| `reference_id` | 동일 baseline/controller/checkpoint 계약 |
| `unit` | 동일 단위 |
| `higher_is_better` | 방향 계약 |
| `run_id` | 원 실행 ID |
| `independent_group_id` | 독립 seed/game/position group; 중복 금지 |
| `effect`, `standard_error` | 유한값, `standard_error > 0` |
| `claim_scope`, `evidence_status` | 계약 게이트가 아닌 ablation 증거 범위 |
| `source_artifact_path` | effect 입력 파일 기준 상대 경로, 상위 경로 탈출 금지 |
| `source_artifact_sha256` | 위 원시 결과의 SHA-256 |

```bash
venv/bin/python scripts/idea_foundry_meta_analyze.py \
  --input results/study_1/campaign_analysis.json \
  --input results/study_2/effects.jsonl \
  --output-dir results/idea_foundry_meta/<meta-run-id>
```

메타분석기는 `axis_id`, `estimand_id`, `effect_scale`, `reference_id`, `unit`,
`higher_is_better`가 모두 동일한 기록만 하나의 군으로 묶는다. 동일 군 안의
`independent_group_id` 중복은 double counting으로 거부한다. 독립 기록이 두
개 이상일 때 inverse-variance fixed effect와 DerSimonian--Laird random
effects, 95% CI, Q, tau-squared, I-squared를 분석용으로 계산한다. 입력 source
artifact의 실제 hash도 확인한다.

현재 first-gate 캠페인만 입력하면 정상 결과는 `NO_COMPARABLE_EFFECTS`다.
계약 check, proposal 수, fixture 수 또는 diagnostic pass rate를 효과크기로
변환하는 것은 금지된다. 메타분석 결과 또한 claim 자동 승격을 허용하지 않으며,
`docs/CLAIM_LEDGER.md`의 paired seed, 동일 checkpoint/runtime, reference/oracle
분리 및 실패행 보존 게이트를 별도로 통과해야 한다.

## 5. 실제 첫 과학 게이트 실행

26축의 실행 계약과 예상 시간을 먼저 확인한다.

```bash
venv/bin/python scripts/idea_foundry_study.py plan --json
```

단일 축은 공통 진입점으로 실행한다. `pilot`은 파이프라인·수치·artifact
검증용 축소 실행이고, `full`만 등록된 첫 과학 게이트의 전체 표본을 사용한다.

```bash
venv/bin/python scripts/idea_foundry_study.py run \
  --axis A01 --profile full \
  --output-dir results/idea_foundry_studies/manual-full/A01

venv/bin/python scripts/idea_foundry_study.py run \
  --axis A19 --profile full \
  --output-dir results/idea_foundry_studies/manual-full/A19
```

전체 26축은 한 번에 하나씩 순서대로 실행하며, 기술 실패에는 즉시 멈춘다.
검증된 완료 축만 resume에서 건너뛴다.

```bash
venv/bin/python scripts/idea_foundry_study_all.py plan --json

venv/bin/python scripts/idea_foundry_study_all.py run \
  --run-id first-scientific-gates-<run-id> \
  --profile full --seed 20260719

venv/bin/python scripts/idea_foundry_study_all.py status \
  --run-id first-scientific-gates-<run-id>

venv/bin/python scripts/idea_foundry_study_all.py resume \
  --run-id first-scientific-gates-<run-id> \
  --profile full --seed 20260719
```

입력과 범위는 다음처럼 분리된다.

- A01--A04, A09, A20--A24: Phase-15의 실제 3-checkpoint/position trace 또는
  실제 Gomoku7 position suite를 사용한다.
- A17: Stage-7 A4/B13 paired trace를 사용하고 decision-neutral analysis-only로
  유지한다.
- A05--A07, A11--A13, A16, A25, A26: seed가 고정된 hidden-best/near-tie/
  multimodal synthetic bank를 사용한다. A05는 보존 trace가 independent restart라
  resident-root 효능을 주장하지 않는다.
- A08: 실제 position suite에서 즉시 win/block이 존재하는 position만 조건부로
  분석한다.
- A10: 보존 trace에 bounded-refresh 비교 arm이 없으므로
  `DORMANT_NO_ELIGIBLE_SLICE`로 종료하고 no-refresh 기본값을 유지한다.
- A15: 동일 representative workload의 CPU/CUDA service curve다. 전력 또는
  shipped-network 효율 증거가 아니다.
- A18: matched deterministic evaluator의 paired replay 학습이다. `full`은
  exact-state-disjoint cyclic holdout을 쓰지만 game/trajectory-group 일반화는
  주장하지 않는다.
- A19: 실제 replay의 고정 split과 batch schedule로 graph seed별 evaluator를
  학습하고 checkpoint, receipt, operator trace, parameter/FLOP 재계산을 남긴다.
  proxy shortlist는 play-strength 증거가 아니다.

각 축은 `run_manifest.json`, `rows.jsonl`, `effect_records.jsonl`, `summary.json`,
`diagnostic.png`, `interpretation.md`를 남긴다. 표준오차를 추정할 수 없는 exact
deterministic 결과는 원시 row에는 보존하되 inverse-variance 메타분석 기록으로
위조하지 않는다.

현재 RTX 3080 Ti 호스트의 실측/계획 시간은 다음과 같다. A18/A19 budget은
첫 full 실행의 실측값에 재시도 여유를 더한 값이다.

| 범위 | 시간 | 근거 |
|---|---:|---|
| 26축 `pilot` | 72.2초 | `implementation-pilot-20260719-v4` step 합계 |
| A15/A18/A19 제외 23축 `full` | 39초 | `full-lightweight-timing-20260719-v1` 실측 |
| A15 `full` | 실측 3시간 13분 36초; budget 4시간 | 24-cell CPU/CUDA 반복 service curve |
| A18 `full` | 실측 16.7초; budget 60초 | 3 seed × 2 arm × 200 update 및 holdout 평가 |
| A19 `full` | 성공 재실행 113.8초; budget 180초 | 8 graph × 3 replay seed × 128 update |
| 26축 전체 `full` | 실측 wall 3시간 41분; budget 합계 15,095초 | `first-scientific-gates-20260719-v1`; A19 validator 복구 대기 포함 |

GPU 경쟁, thermal throttling, CUDA deterministic kernel, Dropbox I/O에 따라 각
native lane은 더 길어질 수 있다. 장기 실행은 새 `run-id`로 시작하고 중단 시 같은
source/Git/input hash에서만 `resume`한다.

## 6. 실제 캠페인 분석과 메타분석

전체 캠페인이 완료되면 다음 분석기를 실행한다.

```bash
venv/bin/python scripts/idea_foundry_study_analyze.py \
  --campaign-dir results/idea_foundry_studies/first-scientific-gates-<run-id>
```

분석기는 모든 per-axis artifact hash를 다시 확인하고, 동일 축의 동일 estimand
안에서만 독립 group 효과를 요약한다. 축끼리는 단위와 estimand가 다르므로 한
효과로 pooling하지 않는다. 결과는 `campaign_analysis/`의
`campaign_analysis.json`, `effect_records.jsonl`,
`within_axis_meta_rows.jsonl`, `diagnostic.png`, `interpretation.md`,
`analysis_manifest.json`에 저장된다.
