# Phase 1.5 재평가 및 후속 설계 문서

## 목적

이 문서는 현재 저장소에 구현된 phase 1.5 clean-split ablation 경로를 다시 평가하고, 다음 질문에 답하기 위해 작성한다.

1. 현재 구현이 원래 문서 의도에 얼마나 부합하는가
2. 어떤 구조적 결함과 해석 위험이 남아 있는가
3. 무엇을 먼저 고쳐야 하는가
4. 정확도를 유지하면서 어떤 식으로 속도를 높일 수 있는가

대상 구현은 다음이다.

- [phase15_strategy_revision_v2.md](../phase15_strategy_revision_v2.md)
- [quartz/phase15_ablation.py](../quartz/phase15_ablation.py)
- [scripts/phase15_ablation_study.py](../scripts/phase15_ablation_study.py)
- [configs/phase15_systems.default.json](../configs/phase15_systems.default.json)

이 문서는 구현 비판용 메모가 아니라, 다음 구현 단계의 기준 문서다.

## 구현 상태 메모

현재 저장소에는 이 문서의 우선순위 중 다음이 반영되어 있다.

- post-hoc runner의 `reference_policy` / `oracle_policy` 분리
- `trace_acquire_ms` / `readout_ms` / `effective_runtime_ms` 분리
- Group A/B/C invariant validator
- `B0`의 report-alias화
- bucket-balanced suite mining 스크립트
- online chunked runner scaffold
- resident root-continuation 우선 online runner + restart fallback reason 계측
- post-hoc / online / benchmark의 amortized trace-bundle 재사용
- continuation benchmark의 `bundle_summary` / `gate` / headwind 진단
- post-hoc / online summary의 `headwind_summary`
- self-contained GitHub Actions benchmark gate workflow

아직 남은 핵심은 resident root-continuation 경로의 semantic drift를 더 줄이고,
그 결과를 gate와 반복 benchmark로 고정하는 일이다.

추가 메모:

- 현재 online runner는 resident root-continuation을 우선 사용한다.
- post-hoc와 benchmark도 이제 같은 `checkpoint x position x system`에서
  full trace를 한 번만 만들고 budget prefix를 재사용한다.
- phase15 benchmark는 smoke/CI gate를 코드로 내장한다.

### 현재 benchmark 상태

빠른 gate smoke artifact:

- `results/phase15_benchmarks_gate_smoke/gomoku7/phase15_continuation_benchmark_summary.json`

이 smoke run의 핵심 수치:

- `bundle_summary.wallclock_speedup_mean = 1.979`
- `summary.tie_aware_match_rate = 0.9375`
- `summary.policy_kl_restart_vs_continuation.mean = 0.1546`
- gate: `pass`

기본 gate는 현재 다음이다.

- bundle speedup `>= 1.80`
- tie-aware match `>= 0.65`
- mean KL `<= 0.25`

이 threshold는 “연구 종결선”이 아니라 CI smoke gate다. tie-aware 하한을
`0.65`로 둔 이유는, 현재 stable 3-checkpoint artifact에서 전체 tie-aware
match가 약 `0.677` 수준이기 때문이다. 즉 flat-root ambiguity를 감안한
현실적 하한이다.

### low-speedup 분해 관찰

stable artifact 기준:

- `results/phase15_benchmarks_amortized_stable/gomoku7/phase15_continuation_benchmark_summary.json`

`A4` 저속 사례는 주로 readout 불안정이 아니라 cost 구조 문제다.

- 예: `C01_best / A4`
  - bundle speedup `1.45x`
  - continuation trace acquire `161.44ms`
  - continuation overhead `58.50ms`
  - tie-aware match `0.875`
  - mean KL `0.111`
  - 해석: continuation도 여전히 baseline search 자체 비용이 크고, 그 위에
    resident-session 고정 overhead가 얹힌다. 즉 “semantic drift 때문에
    느리다”기보다 “절대 검색비용 + 고정 orchestration 비용”의 문제다.

일부 `B3` 사례는 mixed headwind로 보는 편이 맞다.

- 예: `C03_best / B3`
  - bundle speedup `2.39x`
  - continuation overhead ratio `0.398`
  - tie-aware match `1.0`
  - mean KL `0.303`
  - 해석: session overhead도 높고, readout 민감도도 남아 있다. strict
    argmax보다 KL 쪽에서 민감도가 먼저 드러난다.

### summary payload 동기화

현재 phase15 계열 summary는 다음처럼 맞춰져 있다.

- post-hoc `phase15_summary.json`
  - `raw_summary`
  - `semantic_summary`
  - `headwind_summary`
  - `trace_cache_stats`
- online `phase15_online_summary.json`
  - `raw_summary`
  - `semantic_summary`
  - `headwind_summary`
  - `trace_cache_stats`
- benchmark `phase15_continuation_benchmark_summary.json`
  - `summary`
  - `bundle_summary`
  - `gate`

즉 post-hoc / online / benchmark 모두 “고수준 요약 + headwind 진단” 구조를
갖는다. 다만 benchmark의 `speedup_headwind`는 continuation-vs-restart
비교이고, post-hoc / online의 `speedup_headwind`는 assay 내부에서
`trace_acquire` 대 `readout` 대 `reference divergence` 중 어디가 병목인지
정리하는 payload라는 점은 구분해야 한다.

### CI 정착 상태

현재 저장소에는 phase15 benchmark gate를 위한 GitHub Actions workflow가 있다.

- [`.github/workflows/phase15-benchmark-gate.yml`](../.github/workflows/phase15-benchmark-gate.yml)
- smoke entrypoint:
  [scripts/phase15_benchmark_ci_smoke.py](../scripts/phase15_benchmark_ci_smoke.py)

이 workflow는 외부 `results/` artifact에 의존하지 않는다. 실행 시점에:

1. deterministic random checkpoint를 생성하고
2. 작은 fixed position suite를 생성한 뒤
3. 실제 Rust-backed continuation benchmark gate를 실행하고
4. benchmark artifact를 업로드한다

즉 “CI에서는 unit test만 돌고 benchmark는 문서상으로만 존재”하는 상태가
아니라, 최소 smoke 수준의 real benchmark gate가 저장소에 고정된 상태다.

---

## 1. 배경 요약

이전 prior-revision 실험은 다음 이유로 실패했다.

1. baseline contract가 깨져 있었다.
2. refresh 의미론이 search substrate 내부와 섞여 있었다.
3. `B0/B1/N1/N2` 비교가 clean split이 아니라 hybrid bundle 비교에 가까웠다.
4. belief revision을 보려던 실험이 실제로는 root readout heuristic 비교로 수렴했다.

`phase15_strategy_revision_v2.md`는 이를 바로잡기 위해 다음 원칙을 제시했다.

- `QuartzController`를 메타인지 control plane으로 정화한다.
- refresh를 search substrate 밖으로 완전히 외부화한다.
- `substrate / meta-control / refresh / legacy anchor`를 분리한다.
- Group A/B/C matrix로 실험한다.

현재 구현은 이 방향으로 상당히 전진했지만, 아직 완전한 구현은 아니다.

---

## 2. 현재 구현의 장점

## 2.1 이전 prior-revision runner보다 훨씬 정직하다

현재 구현은 최소한 다음을 명시적으로 분리했다.

- Group A: substrate/controller sanity
- Group B: refresh isolation
- Group C: legacy anchor

이는 이전처럼 `GatedRefreshLegacy` substrate 위에 모든 실험을 얹고도 clean comparison처럼 보이던 상태보다 훨씬 낫다.

## 2.2 system definition이 코드와 config에 고정돼 있다

[configs/phase15_systems.default.json](../configs/phase15_systems.default.json)에 시스템 정의가 고정돼 있고, [quartz/phase15_ablation.py](../quartz/phase15_ablation.py)에도 동일 구조가 존재한다.

이건 중요하다. 실험 이름과 실제 override가 문서-코드 사이에서 쉽게 어긋나지 않기 때문이다.

## 2.3 legacy를 mainline 후보가 아니라 anchor로 내렸다

`C0/C1/C2`를 별도 comparison anchor로 둔 것은 설계상 매우 올바르다.

- `C0 = GatedRefreshLegacy`
- `C1 = PFlipMixture`
- `C2 = SelfAdaptive`

이들은 이제 비교 기준이지, phase 1.5 mainline 후보가 아니다.

## 2.4 root-only 기본값은 올바른 방향이다

Group A/B 기본에서 `root_only_shaping=true`를 둔 것은 구조 식별 관점에서 옳다.

문제는 아직 구현이 root-only online control 실험까지는 못 갔다는 것이지, 방향 자체는 맞다.

---

## 3. 핵심 결함

이 절은 현재 구현의 가장 중요한 문제를 우선순위대로 정리한다.

## 3.1 `oracle`가 실제 oracle이 아니다

현재 runner는 bucketization과 최종 평가의 기준 policy를 모두 `reference_system=A0`의 고예산 search 결과로 둔다.

즉 현재 `oracle_policy`는 사실상 다음이다.

- 동일 frozen checkpoint
- 동일 evaluator
- `A0` profile
- 더 큰 budget

이것은 강한 reference는 될 수 있지만, 엄밀한 의미의 oracle은 아니다.

### 왜 문제인가

현재 지표:

- `wrong_prior_correction`
- `easy_case_regret`
- `kl_to_oracle`
- `accuracy`

는 이름상 oracle 기준처럼 보이지만 실제로는 “A0 high-budget reference와의 일치도”에 가깝다.

### 결과적 해석 위험

- Group B가 Group A보다 좋아 보여도, 그건 “oracle에 더 가까워졌다”가 아니라 “A0 high-budget readout에 더 가까워졌다”일 수 있다.
- legacy anchor가 약해 보이는 것도 실제 truth가 아니라 reference profile과의 차이일 수 있다.

### 수정 원칙

다음 둘을 분리해야 한다.

- `reference_policy`
- `oracle_policy`

최소 수정안:

1. 현재 `oracle_policy`를 `reference_policy`로 rename
2. 별도 `--oracle-system` 또는 `--oracle-profile baseline_strict` 추가
3. 가능하면 `oracle_checkpoint`도 따로 받기

---

## 3.2 Group B는 online control 실험이 아니라 post-hoc readout 실험이다

현재 Group B 구현은 search trace를 먼저 만든 뒤, 그 trace 위에서 후처리 연산을 적용한다.

예:

- `dual_channel_commit`
- `root_challenger`
- `budget_routing`

은 모두 이미 끝난 budget trace를 보고 최종 policy를 재구성한다.

### 왜 문제인가

문서가 요구한 phase 1.5는 다음이었다.

- chunked search
- root snapshot
- controller signal
- stop / commit / challenger / budget burst
- 다음 chunk에 실제 개입

즉 online control이어야 한다.

현재 구현은 그보다 한 단계 약한 다음 실험이다.

> 동일 search trace를 놓고 root readout operator를 비교하는 post-hoc assay

이건 의미는 있지만, phase 1.5의 본체는 아니다.

### 특히 `budget_routing`이 가장 크게 어긋난다

현재 구현의 `budget_routing`은 사실상

- instability면
- 더 큰 budget의 미리 계산된 결과를 선택

이다.

이건 adaptive compute allocation이 아니라 budget escalation selector다.

### 수정 원칙

Group B를 둘로 나눠야 한다.

1. `B*-posthoc`
2. `B*-online`

현재 구현은 `B*-posthoc`로 정직하게 rename하는 것이 맞다.

---

## 3.3 `compute_time_ms`가 왜곡되어 있다

현재 `compute_time_ms`는 trace를 구성하는 여러 budget search의 시간을 모두 더한 값이다.

예를 들어 `8,16,32` trace를 쓰는 경우:

- 실제 보고되는 시간은 독립 search 3회의 합이다.

하지만 phase 1.5 문서가 원하는 건 대체로 다음 둘 중 하나다.

- chunked one-run wallclock
- incremental control 추가비용

현재 수치는 둘 다 아니다.

### 왜 문제인가

다음 지표 해석이 오염된다.

- `same-wallclock gain`
- `compute-normalized gain`
- `extra-budget efficiency`
- `budget burst efficiency`

### 수정 원칙

시간 지표를 최소 세 개로 나눠야 한다.

1. `trace_acquire_ms`
2. `readout_ms`
3. `effective_runtime_ms`

post-hoc 그룹은 특히 `readout_ms`가 거의 0에 가까울 수 있으므로, 이를 분리해주지 않으면 성능 해석이 틀어진다.

---

## 3.4 Group A/B matrix의 일부는 의미론적으로 중복된다

현재 기본 matrix에서 `A4`와 `B0`는 사실상 같은 search semantics를 갖는다.

이건 문서상으로는 이해할 수 있다.

- `A4`는 sanity track의 끝
- `B0`는 refresh track의 baseline

하지만 실제 측정 행으로 둘 다 생성되면 다음 문제가 생긴다.

- report가 중복된다
- 독자가 별도 실험이라고 오해할 수 있다
- compare logic이 번잡해진다

### 수정 원칙

`B0`는 report alias로 두는 편이 낫다.

즉:

- raw rows는 `A4`
- summary/report에서는 `B0 := A4`

이 방식이 더 정직하다.

---

## 3.5 `controller`와 `substrate`는 메타데이터일 뿐, 실행 invariant가 아니다

현재 `Phase15System`의

- `substrate`
- `controller`
- `refresh_operator`

중 실제 실행에 직접 영향을 주는 것은 거의 `search_overrides`와 `refresh_operator`뿐이다.

그 결과 다음 같은 일이 가능하다.

- Group A라고 해도 legacy penalty를 넣을 수 있음
- Group B라고 해도 clean substrate invariant가 깨질 수 있음
- `controller="QuartzStopOnly"`라고 적혀 있어도 실제론 stop-only 검증이 아님

### 수정 원칙

system validator를 추가해야 한다.

예:

#### Group A/B invariant

- `root_only_shaping == true`
- `penalty_mode not in {GatedRefreshLegacy, GatedRefresh, SelfAdaptive, PFlipMixture}`
- `search_profile in {baseline, quartz}`

#### Group C invariant

- legacy mode만 허용

이 validator는 runner 시작 시 실패해야 한다.

---

## 3.6 bucketization은 아직 deep-search mining 수준이 아니다

현재 bucketization은 다음 입력만 본다.

- prior
- low-budget reference
- high-budget reference

그리고 top-k / confidence 기반 휴리스틱으로 bucket을 붙인다.

이건 간단하고 빠르지만, 문서가 요구한 수준은 아니다.

문서가 기대한 건:

- mined suite
- disagreement-driven selection
- bucket별 샘플 목표 수
- root conflict / deep conflict의 좀 더 직접적인 판정

### 현재 상태의 한계

- `deep_conflict`는 실제 depth conflict라기보다 “prior top-k 밖의 oracle move”에 가까움
- `late_evidence`도 search-depth 구조보다는 low/high disagreement 라벨에 가까움
- bucket imbalance 관리가 없다

### 수정 원칙

`--suite-source mined|random`를 추가하고, mined suite를 별도 artifact로 저장해야 한다.

---

## 4. 현재 설계의 정확도 평가

현재 구현은 “정답을 맞히는 controller”보다 “reference readout에 가까운 readout operator”를 찾는 데는 꽤 유용할 수 있다.

즉 다음 목적에는 적합하다.

- root posterior 재해석 아이디어를 빠르게 비교
- commit gate / challenger set / budget burst heuristic의 초벌 비교
- legacy 대비 행태 차이 탐색

반면 다음 목적에는 아직 부족하다.

- 실제 online controller quality 판정
- stop/continue 제어의 가치 판정
- same-wallclock gain 판정
- compute allocation 전략의 진짜 효율 판정
- “QuartzController가 메타인지 control plane인가”의 직접 검증

결론적으로:

- 현재 버전은 exploratory post-hoc assay로는 유용하다.
- 하지만 phase 1.5 최종판정 실험으로 쓰면 과장이다.

---

## 5. 수정 우선순위

## P0. naming / contract honesty 복구

가장 먼저 해야 한다.

### 작업

1. `oracle_policy`를 `reference_policy`로 rename
2. `Group B`를 `posthoc`와 `online`으로 분리
3. `compute_time_ms`를 `trace_acquire_ms`와 `readout_ms`로 분해
4. system invariant validator 추가

### 이유

이 단계는 성능보다 해석 정직성을 위해 필요하다.

---

## P1. oracle/reference 분리

### 작업

1. `--oracle-profile baseline_strict`
2. `--oracle-checkpoint`
3. manifest에 `reference_system`과 `oracle_system` 동시 저장
4. summary에 `accuracy_to_reference`, `accuracy_to_oracle` 분리

### 이유

현재 가장 큰 해석 오류를 줄인다.

---

## P2. mined suite 도입

### 작업

1. random suite와 mined suite 분리
2. low/high budget disagreement mining
3. bucket quota 도입
4. imbalance warning 도입

### 이유

현재 bucket quality가 낮아, system 차이를 뚜렷하게 드러내는 포지션 밀도가 부족하다.

---

## P3. true online runner 추가

### 작업

새 runner를 분리한다.

예:

- `scripts/phase15_online_ablation.py`

최소 요구:

1. search chunk execution
2. root summary extraction
3. commit/challenger/budget-burst signal
4. stop/continue control
5. chunk별 telemetry 저장

### 이유

이게 있어야 phase 1.5 문서의 핵심 주장과 실제 코드가 맞물린다.

---

## 6. 정확도 유지형 최적화 전략

이 절의 기준은 단순하다.

> search semantics와 측정 의미를 바꾸지 않으면서 속도만 올린다.

## 6.1 가장 큰 최적화: incremental budget trace

현재는 budget별 search를 독립 호출한다.

예:

- 8
- 16
- 32
- 64

이걸 전부 새 search로 돌린다.

### 문제

- 느리다
- `compute_time_ms` 해석도 불분명하다
- trace coherence도 약하다

### 해결

resident session 기반 incremental checkpoint를 추가한다.

예:

1. 같은 root/session 시작
2. 8 visits 후 snapshot
3. 이어서 16
4. 이어서 32
5. 이어서 64

### 장점

- 정확도/의미론 유지
- wallclock 크게 절약
- chunked telemetry와 자연스럽게 연결

이건 phase 1.5의 최우선 최적화다.

---

## 6.2 trace artifact 디스크 캐시

현재도 in-memory cache는 있다.

하지만 rerun 관점에선 부족하다.

### 해결

다음을 artifact로 저장한다.

- `checkpoint x position x search_overrides` 단위 trace

예:

- `trace_store.jsonl`
- 또는 budget별 shard

### 장점

- post-hoc readout 비교를 탐색 비용 없이 반복 가능
- parameter sweep를 매우 빠르게 돌릴 수 있음
- 동일 trace 위에서 B1/B2/B3 변형을 마음껏 비교 가능

정확도 손실은 없다.

---

## 6.3 prior inference batch화

현재 prior는 position마다 단건 forward다.

### 해결

checkpoint별 suite 전체를 batch로 묶는다.

예:

- `run_model_batch`
- encoder output precompute

### 장점

- GPU 사용 효율 증가
- CPU overhead 감소
- semantics 불변

---

## 6.4 metric 계산 벡터화

현재 `entropy`, `KL`, `top-k`, summary 계산은 Python loop 비중이 높다.

### 해결

- `numpy` 기반 batch metric 함수 도입
- summary reduce 벡터화
- bucket label 계산도 가능한 범위에서 batch화

### 장점

- suite가 커질수록 효과가 커진다
- 의미론 변화 없음

---

## 6.5 suite mining 오프라인화

현재 position suite를 매번 생성/라벨링한다.

### 해결

- mined suite 생성과 assay 실행을 분리
- 예:
  - `phase15_mine_suite.py`
  - `phase15_ablation_study.py --positions-file mined_suite.json`

### 장점

- expensive mining을 반복하지 않아도 된다
- bucketized benchmark를 안정적으로 재사용할 수 있다

---

## 6.6 post-hoc와 online cost 분리

현재 설계에선 post-hoc readout 비용이 trace acquisition 비용에 묻힌다.

### 해결

측정치를 분리한다.

- `trace_acquire_ms`
- `readout_ms`
- `online_control_ms`

### 장점

- 후처리 readout 설계 탐색을 더 공정하게 비교 가능
- online controller 추가비용을 명확히 볼 수 있다

---

## 7. 권장 구조 개편안

다음 구조를 권장한다.

## 7.1 모듈 분리

- `quartz/phase15_ablation.py`
  - 현재처럼 공통 metric, system config, post-hoc readout 보관

- `quartz/phase15_trace.py`
  - trace schema
  - trace serialization
  - trace cache load/save

- `quartz/phase15_suite.py`
  - suite mining
  - bucket labeling
  - suite balancing

- `quartz/phase15_online.py`
  - online controller experiment contract

## 7.2 runner 분리

- `scripts/phase15_ablation_study.py`
  - post-hoc trace assay

- `scripts/phase15_online_ablation.py`
  - true online control assay

- `scripts/phase15_mine_suite.py`
  - suite generation

이렇게 나누면 설계 의미가 선명해진다.

---

## 8. 권장 metric 재정의

현재 지표 중 일부는 이름과 의미가 어긋난다.

권장 재정의는 아래와 같다.

## 8.1 reference/oracle 분리

- `accuracy_to_reference`
- `accuracy_to_oracle`
- `kl_to_reference`
- `kl_to_oracle`

## 8.2 commit 계열

- `commit_rate`
- `commit_confidence_mean`
- `commit_latency_budget`
- `commit_false_positive_rate`

## 8.3 challenger 계열

- `challenger_recall@k`
- `challenger_undercoverage`
- `challenger_overlap_stability`

## 8.4 budget routing 계열

- `burst_rate`
- `burst_gain`
- `burst_gain_per_extra_visit`
- `same_wallclock_accuracy`

---

## 9. 현재 구현에 대한 최종 판정

현재 버전은 이전 prior-revision runner보다 명백히 낫다.

특히:

- legacy anchor 격리
- clean system matrix 명시
- root-only default
- 문서-코드 alignment 개선

은 모두 실제 진전이다.

그러나 냉정하게 말하면, 현재 구현은 아직 다음 단계다.

> phase 1.5의 완전한 online control 실험이 아니라, clean substrate 위의 post-hoc root readout assay

이 평가는 부정적 의미만은 아니다.

이 post-hoc assay는 여전히 가치가 있다.

- 빠르다
- 구조 가설을 초기 필터링하기 좋다
- legacy anchor와의 대략적 행태 비교가 가능하다

다만 이걸 final phase 1.5 evidence처럼 해석하면 안 된다.

---

## 10. 바로 다음 액션

가장 권장하는 순서는 아래와 같다.

1. naming/contract honesty 수정
   - oracle/reference 분리
   - Group B를 post-hoc로 명시
   - 시간 지표 분리

2. system invariant validator 추가
   - Group A/B clean substrate 강제
   - Group C legacy-only 강제

3. trace artifact cache 추가
   - rerun 속도 향상
   - post-hoc readout 탐색 비용 절감

4. mined suite 도입
   - bucket quality 향상
   - signal density 향상

5. true online runner 구현
   - phase 1.5 문서의 핵심을 실제 코드로 옮김

---

## 한 줄 결론

현재 phase 1.5 구현은 방향은 맞지만 아직 완성형은 아니다.  
지금 있는 코드는 “clean-split online control 증명”이 아니라 “clean-split post-hoc readout screening”으로 해석해야 하며, 다음 단계는 oracle 계약 복구와 true online runner 구현이다.
