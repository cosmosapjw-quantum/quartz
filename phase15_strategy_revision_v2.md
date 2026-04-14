# Phase 1.5 전략 수정안 v2  
## QuartzController 정화 + Refresh 외부화 + Adaptive Thinking 실험 재설계

## 0. 문서 목적

이 문서는 phase 1 결과와 최신 코드 구조를 반영하여, phase 1.5를 **QuartzController의 메타인지 역할 정화(clean split)** 와 **refresh의 독립 실험축 분리**라는 원칙 위에서 다시 설계하기 위한 self-contained 계획서다.

핵심 목표는 세 가지다.

1. `QuartzController`와 `VL(parallelism) controller`를 가능한 한 **진짜 메타인지적 제어층**으로 재해석하고, action-wise preference rewrite와 분리한다.
2. `select.rs`에 암묵적으로 섞여 있는 refresh를 메인 실험축에서 제거하고, refresh를 **완전히 별도 트랙**으로 옮긴다.
3. 네가 원래 의도한 **adaptive thinking**을 “naive prior refresh”가 아니라,  
   - belief revision의 commit,
   - challenger set 재구성,
   - compute allocation,
   - stop/continue control  
   의 문제로 다시 정식화한다.

---

## 1. 현재 구현에 대한 비판적 진단

## 1.1 `SearchController` 인터페이스는 본래 메타인지적이다

현재 `SearchController` trait는 본질적으로 다음만 다룬다.

- `should_stop`
- `reset`
- `visit_limit_hint`
- `stop_reason`

즉 인터페이스 수준에서는 “무슨 수가 좋은가”가 아니라 “언제 멈출까 / 더 볼까”를 다루는 층이다. 이 점에서 `SearchController`라는 추상은 메타인지 해석이 가능하다.

## 1.2 `QuartzController` 자체도 꽤 메타인지적이다

최신 구현에서 `QuartzController`는 root-level search statistics를 바탕으로 대체로 아래와 같은 신호를 계산한다.

- `p_flip`
- `surprise_kl`
- `p_hidden`
- `conf_t`
- `prior_q_divergence`
- `sigma_response`
- `defect_value`

이들은 모두 “지금 이 탐색 결과가 얼마나 믿을 만한가”, “기존 prior와 현재 evidence가 얼마나 충돌하는가”, “hidden mode가 남아 있는가”, “계속 더 봐야 하는가” 같은 질문에 대응하는 **메타인지적 신뢰도/불확실성/결함 신호**로 읽을 수 있다.

즉 **controller 객체만 놓고 보면**, 네가 말한 “메타인지”라는 자기 이해는 상당히 plausibility가 있다.

## 1.3 `ParallelismController`는 더 순수한 메타인지적 보조층이다

`VL(parallelism) controller`는 QUARTZ로부터 `sigma_q`, `root_entropy`를 one-way로 읽고, 자기 telemetry인 `dup_rate`, `max_pending`과 결합해 `vvisit/vvalue`만 조절한다.

즉 이 controller는
- 어떤 action이 더 좋아 보이는지 말하지 않고,
- 지금 탐색이 얼마나 noisy/duplicated/overcontended한지를 보고,
- 병렬 탐색 완충 정도만 조절한다.

이는 phase 1.5에서 유지할 가치가 높다.

## 1.4 실제 문제는 `select.rs`에 있다

현재 `select.rs`의 `PenaltyMode`들은 이름과 달리 penalty만 수행하지 않는다.  
일부 모드는 controller-derived statistics를 사용해 `effective_prior`를 직접 다시 만든다.

대표적으로 다음 모드들이 그렇다.

- `SelfAdaptive`
- `GatedRefresh`
- `GatedRefreshLegacy`
- `PFlipMixture`

즉 현재 번들 전체는
- 위에서는 메타인지처럼 감시하면서
- 아래에서는 그 통계를 이용해 object-level prior를 rewrite하는
**hybrid 구조**다.

이 때문에 현재 구현을 “순수한 메타인지 controller”라고 부르기는 어렵다.

---

## 2. 가장 먼저 고쳐야 할 계약 오류

## 2.1 최신 코드 기준으로 `B0`는 진짜 no-refresh baseline이 아니다

현재 기본 config에서는 `B0`와 `B1`이 모두 `GatedRefreshLegacy` substrate를 공유하고 있다.

그런데 최신 `select.rs`에서 `GatedRefreshLegacy`는 내부적으로 이미 `p_flip`를 사용해 `effective_prior`를 다시 만들며, 이 분기에서 early return이 발생하기 때문에, config 상의 `prior_refresh_rate` 차이가 baseline semantics를 분리하지 못할 수 있다.

즉 최신 코드 기준으로는:

- `B0`도 사실상 refresh가 들어갈 수 있고
- `B1`과의 차이가 기대만큼 clean하지 않다

이건 phase 1과 이후 phase 1.5 해석을 오염시키는 구조적 문제다.

## 2.2 phase 1.5의 0번 작업: baseline substrate를 복구한다

phase 1.5를 시작하기 전에 반드시 다음을 수행해야 한다.

1. implicit refresh가 없는 baseline을 다시 정의한다.
2. `PenaltyMode`와 refresh semantics를 분리한다.
3. `B0/B1`를 더 이상 `GatedRefreshLegacy` 위에서 비교하지 않는다.

---

## 3. phase 1.5의 새 원칙

## 원칙 A. QuartzController는 “진짜 메타인지”만 한다

QuartzController는 다음만 담당한다.

- stop / continue
- commit / hold
- challenger set proposal
- budget burst / routing
- trust discount
- visit limit hint

그리고 절대 하지 않는 것:

- action-wise pseudo-prior 출력
- `Q`, `visit_share`, `PUCT score`를 다시 조합해 preference 생성
- same-pass 내에서 posterior를 곧바로 selection에 재주입

## 원칙 B. refresh는 완전히 다른 트랙으로 뺀다

refresh는 더 이상 controller 내부나 `PenaltyMode`의 side effect로 존재하면 안 된다.  
반드시 **독립된 refresh operator**로만 존재해야 한다.

즉 다음과 같이 역할을 분리한다.

- `QuartzController` = metacognitive control plane
- `RefreshOperator` = optional object-level intervention module

## 원칙 C. two-timescale update만 허용한다

한 simulation 내부에서 다음과 같은 고주파 피드백 루프는 금지한다.

`posterior -> controller -> selection -> posterior`

대신 반드시 다음처럼 chunked / checkpointed update를 쓴다.

1. 일정 visit chunk 수행
2. root stats snapshot 생성
3. controller가 snapshot을 보고 control signal 생성
4. 다음 chunk의 stop/commit/challenger/budget만 변경

## 원칙 D. root-first

phase 1.5에서는 online 전략 조정을 일단 root-level로 제한한다.

- `root_only_shaping = true`
- non-root refresh 금지
- depth > 0에서는 direct policy rewrite 금지

이렇게 해야 구조 식별력이 생긴다.

---

## 4. phase 1.5의 새 구조

phase 1.5는 세 층으로 재구성한다.

### Layer 0. Search substrate
refresh가 전혀 없는 순수 탐색 substrate

### Layer 1. Meta-control
Quartz + VL이 stop / trust / budget / challenger proposal만 수행

### Layer 2. Refresh track
refresh operator를 완전히 외부화하여 독립 ablation 수행

---

## 5. Layer 0 — 순수 substrate 재정의

## S0. Plain PUCT baseline

- `PenaltyMode = None`
- fixed budget 또는 fixed Quartz stop
- no refresh
- root outside shaping 없음

이것이 진짜 무기준 baseline이다.

## S1. Penalty-only baseline

- `PenaltyMode = Legacy` 또는 `EffectiveV2`
- no refresh
- Quartz stop 허용
- root prior rewrite 금지

이는 “탐색 penalty는 허용하지만 object-level policy rewrite는 없는 substrate”다.

### 권고
phase 1.5에서는 반드시 `S0`와 `S1`을 모두 둔다.  
그렇지 않으면 penalty 효과와 metacognitive 효과가 다시 섞인다.

---

## 6. Layer 1 — Meta-control only track

이 그룹은 “controller만으로도 online 전략 조정이 가능한가?”를 보는 그룹이다.

## M0. Substrate only

- substrate = `S0` 또는 `S1`
- Quartz off
- VL off

## M1. Quartz stop-only

- substrate = `S0` 또는 `S1`
- Quartz on
- output used:
  - `should_stop`
  - `visit_limit_hint`
  - `stop_reason`
- selection preference unchanged

질문:
- Quartz는 stop-controller만으로도 가치가 있는가?

## M2. Quartz + VL only

- substrate = `S0` 또는 `S1`
- Quartz on
- VL on
- Quartz -> VL one-way coupling 유지
- selection preference unchanged

질문:
- 메타인지적 탐색 제어만으로 실제 search quality나 efficiency가 좋아지는가?

### 이 그룹의 핵심 원칙
**여기에는 refresh가 절대 들어가면 안 된다.**

---

## 7. Layer 2 — Refresh track (완전 분리)

이 그룹은 refresh를 독립 operator로 평가하는 그룹이다.

## R0. No refresh

- substrate = `S1`
- controller = `M2`까지 허용
- refresh operator 없음

이게 refresh track의 진짜 baseline이다.

## R1. External dual-channel refresh

- `prior_base`는 immutable
- `posterior_search`는 raw search evidence로 계산
- controller는 `commit_gate`만 제공
- `effective_policy = (1-g) p0 + g p_post`

중요:
- `g`는 scalar이어야 한다
- action-wise gate 금지
- `g`의 입력은 신뢰도/안정성 정보뿐이어야 한다

허용 입력 예:
- `conf_t`
- `p_flip`
- `p_hidden`
- `prior_q_divergence`
- `sigma_response`
- `defect_value`
- `dup_rate`
- `max_pending`
- argmax persistence
- top-2 margin stability
- posterior entropy slope

금지 입력 예:
- action-wise `Q(a)`
- action-wise `visit_share(a)`
- action-wise controller logits

## R2. Root challenger refresh

- controller는 challenger set만 제안
- posterior는 challenger set 안에서만 계산
- final root readout에만 사용
- non-root selection에 재주입 금지

이 구조는 “어떤 후보를 다시 볼 것인가”를 메타인지적으로 결정하는 방식이다.

## R3. Budget-only adaptation

- posterior를 거의 사용하지 않거나 매우 약하게만 사용
- controller는
  - `budget_burst`
  - `branch focus`
  - `stop suppression`
  만 수행

즉 online 전략 조정을 **계산 자원 재배분**으로 구현한다.

이 트랙은 오히려 인간 유비에 가장 가까운 후보일 수 있다.

---

## 8. QuartzController의 새 출력 계약

phase 1.5에서 `QuartzController`는 아래와 같은 **메타 신호 묶음**만 외부로 내야 한다.

```rust
pub struct MetaControlSignal {
    pub stop_continue: bool,
    pub commit_confidence: f32,   // scalar in [0,1]
    pub challenger_k: usize,      // root only
    pub budget_burst: u32,        // extra visits
    pub trust_discount: f32,      // scalar in [0,1]
}
```

핵심은 다음이다.

- action index 없음
- action-wise preference 없음
- controller는 content가 아니라 control만 담당

### 기존 코드에서 재활용 가능한 입력 신호
- `conf_t`
- `p_flip`
- `p_hidden`
- `surprise_kl`
- `prior_q_divergence`
- `sigma_response`
- `defect_value`

### phase 1.5용으로 추가해야 할 root-level telemetry
- argmax persistence over checks
- top-2 margin stability
- challenger overlap across checks
- revision flip-flop count
- posterior entropy slope

---

## 9. select.rs 재설계 원칙

## 9.1 현재 문제
`PenaltyMode` 하나에
- penalty kernel
- refresh kernel
- sometimes gating logic
이 다 들어가 있다.

이건 phase 1.5에서 더 이상 허용되면 안 된다.

## 9.2 개념적 분리
phase 1.5에선 conceptually라도 아래처럼 분리해서 생각해야 한다.

```rust
enum PenaltyKernel {
    None,
    Legacy,
    EffectiveV2,
}

enum RefreshKernel {
    None,
    ExternalDualChannel,
    RootChallenger,
}
```

### 설계 원칙
- `select.rs`는 penalty와 exploration bonus만 담당
- refresh는 root-level external operator로만 존재
- `visit_share`, `q_eff/tau`, `p_flip`를 prior mixing weight로 쓰는 path는 phase 1.5 main matrix에서 제거

## 9.3 최소 수정안
대공사 이전이라면 적어도:

- phase 1.5 config에서는 `PenaltyMode = None` 또는 `EffectiveV2`
- `GatedRefresh*`, `PFlipMixture`, `SelfAdaptive`는 main matrix에서 제외
- `root_only_shaping = true`

---

## 10. phase 1.5 실험 매트릭스 v2

## Group A — Substrate / controller sanity

- `A0 = S0`
- `A1 = S0 + M1`
- `A2 = S0 + M2`
- `A3 = S1 + M1`
- `A4 = S1 + M2`

질문:
- controller만으로 online 전략 조정이 가능한가?
- penalty-only substrate 위에서도 Quartz/VL의 순수 기여가 존재하는가?

## Group B — Refresh isolation

- `B0 = A4 + R0`
- `B1 = A4 + R1`
- `B2 = A4 + R2`
- `B3 = A4 + R3`

질문:
- refresh나 adaptive strategy adjustment가 실제로 baseline 대비 이득을 주는가?
- dual-channel / challenger / budget routing 중 무엇이 더 유효한가?

## Group C — Legacy anchor

- `C0 = current GatedRefreshLegacy`
- `C1 = current PFlipMixture`
- `C2 = current SelfAdaptive`

질문:
- 새 clean-split 구조가 기존 legacy 방식보다 실제로 나은가?

주의:
- Group C는 mainline 후보가 아니라 comparison anchor다.

---

## 11. 트랙별 핵심 지표

## Track R1 — Dual-channel + commit gate
1차 지표:
- root best-move accuracy
- wrong-prior correction
- easy-case regret
- commit rate
- commit latency

2차 지표:
- `commit_confidence`
- argmax persistence
- top-2 margin stability
- posterior entropy drop
- defect trend

핵심 질문:
- phase 1에서 N1이 보인 “KL은 개선되나 accuracy는 약함” 문제를 commit contract가 해결하는가?

## Track R2 — Root challenger refresh
1차 지표:
- root best-move accuracy
- challenger recall
- candidate undercoverage
- low-budget gain

2차 지표:
- challenger set size
- challenger persistence
- root margin ambiguity
- `conf_t` vs `challenger_k`

핵심 질문:
- root-only 구조는 root 후보 재구성 문제로 다시 살릴 수 있는가?

## Track R3 — Adaptive budget routing
1차 지표:
- same-wallclock accuracy
- same-sim accuracy
- compute-normalized gain
- stop suppression rate
- extra-budget efficiency

2차 지표:
- budget burst frequency
- branch focus concentration
- `dup_rate`, `max_pending`, `sigma_response`와 성능 관계

핵심 질문:
- online 전략 조정을 직접적인 prior rewrite 없이 compute allocation만으로 구현할 수 있는가?

---

## 12. posterior의 올바른 해석

phase 1.5에서 posterior를 “selection에 다시 넣는 새 prior”로 보면 안 된다.

posterior의 올바른 역할은:

**working hypothesis buffer**

즉 root에서 각 체크포인트마다
- `p_post^(t)`를 만든다.

그리고 controller는 이걸 보고 다음만 판단한다.

- `KL(p0 || p_post^(t))`
- `KL(p_post^(t-1) || p_post^(t))`
- argmax persistence
- margin stability
- challenger overlap
- revision persistence

즉 posterior는
- 전략 조정의 대상이 아니라
- 전략 조정 여부를 판단하는 증거물이다

인간 유비로 치면:
- `prior_base` = 미리 학습된 패턴 인식
- `posterior_search` = 지금 읽어본 뒤 형성된 잠정 가설
- `controller` = 이 가설을 믿을지, 더 볼지, 버릴지 정하는 메타인지

---

## 13. Track별 pseudocode

## 13.1 M1 — Quartz stop-only

```python
def run_quartz_stop_only(root_state, evaluator, substrate, quartz):
    quartz.reset()
    search = init_search(root_state, evaluator, substrate)
    while True:
        run_visit_chunk(search)
        root_stats = summarize_root(search)
        quartz.update(root_stats)
        if quartz.should_stop(root_stats):
            break
    return finalize_root_policy(search)
```

## 13.2 R1 — External dual-channel refresh with commit gate

```python
def run_dual_channel_refresh(root_state, evaluator, substrate, quartz):
    quartz.reset()
    search = init_search(root_state, evaluator, substrate)
    prior_base = evaluator.prior(root_state)
    posterior_prev = None

    while True:
        run_visit_chunk(search)
        root_stats = summarize_root(search)
        posterior_now = estimate_search_posterior(root_stats)

        meta = quartz.meta_signal(
            root_stats=root_stats,
            prior_base=prior_base,
            posterior_prev=posterior_prev,
            posterior_now=posterior_now,
        )

        if meta.commit_confidence >= COMMIT_THRESHOLD:
            effective_policy = mix(prior_base, posterior_now, meta.commit_confidence)
        else:
            effective_policy = prior_base

        if meta.stop_continue is False:
            break

        posterior_prev = posterior_now

    return effective_policy
```

## 13.3 R2 — Controller-proposed challenger refresh

```python
def run_root_challenger_refresh(root_state, evaluator, substrate, quartz):
    quartz.reset()
    search = init_search(root_state, evaluator, substrate)
    prior_base = evaluator.prior(root_state)

    while True:
        run_visit_chunk(search)
        root_stats = summarize_root(search)

        meta = quartz.meta_signal(
            root_stats=root_stats,
            prior_base=prior_base,
            posterior_prev=None,
            posterior_now=None,
        )

        if meta.stop_continue is False:
            break

    challenger_set = propose_challengers(root_stats, k=meta.challenger_k)
    posterior = restricted_root_posterior(root_stats, challenger_set)
    final_policy = restricted_commit(prior_base, posterior, challenger_set)

    return final_policy
```

## 13.4 R3 — Adaptive budget routing

```python
def run_budget_routing(root_state, evaluator, substrate, quartz, vl):
    quartz.reset()
    search = init_search(root_state, evaluator, substrate)

    while True:
        root_stats = summarize_root(search)
        quartz.update(root_stats)
        vl.update_from_quartz(root_stats)

        if should_burst_budget(root_stats, quartz, vl):
            extra = compute_budget_burst(root_stats, quartz, vl)
            run_targeted_visits(search, extra)
        else:
            run_visit_chunk(search)

        if quartz.should_stop(root_stats):
            break

    return finalize_root_policy(search)
```

---

## 14. phase 1.5의 실행 순서

### Step 0
baseline contract repair
- `B0/B1` implicit refresh 제거
- `root_only_shaping = true`
- phase 1.5 전용 config 분리

### Step 1
Group A 실행
- controller만으로 online 전략 조정이 가능한지 확인

### Step 2
Group B 실행
- refresh를 완전히 외부화한 상태에서 N1/N2/C 후보 평가

### Step 3
Group C 실행
- legacy 방식과 새 clean-split 구조 비교

### Step 4
survivor selection
- accuracy
- wrong-prior correction
- easy-case regret
- same-wallclock gain
- compute-normalized gain
- candidate undercoverage
를 기준으로 survivor를 정한다.

---

## 15. kill criteria

### 공통 kill criteria
- baseline `S0/S1`보다 accuracy가 일관되게 낮다
- easy-case regret가 의미 있게 높다
- same-wallclock에서도 손해다
- telemetry가 해석 불가능할 정도로 unstable하다

### R1 전용
- commit confidence가 높은데도 accuracy improvement가 없다
- posterior entropy만 줄고 argmax quality는 개선되지 않는다

### R2 전용
- challenger undercoverage가 높다
- root-only candidate restriction이 low-budget에서도 이득을 못 낸다

### R3 전용
- extra budget을 써도 compute-normalized gain이 없다
- contention telemetry만 흔들고 실제 품질은 안 좋아진다

---

## 16. 최종 권고

이 phase 1.5 수정안의 핵심은 아주 단순하다.

1. **QuartzController를 정화한다.**
   - 메타인지 역할만 남긴다.
   - preference rewrite는 빼낸다.

2. **Refresh를 완전히 외부화한다.**
   - selection substrate에 숨어 있지 않게 한다.
   - separate ablation이 가능하게 만든다.

3. **Adaptive thinking을 prior rewrite가 아니라 control problem으로 재해석한다.**
   - commit
   - challenger proposal
   - budget routing
   - stop/continue

내 최종 판정은 이렇다.

- `QuartzController`를 “메타인지”라고 부르는 건 충분히 가능하다.
- 다만 현재 구현 전체는 아직 hybrid다.
- phase 1.5의 목표는 controller-derived statistics가 `effective_prior` rewrite로 들어가는 경로를 잘라내고,
  controller를 **진짜 metacognitive control plane**으로 올리는 것이다.

---

## 17. 바로 다음 액션

1. phase 1.5 전용 config 생성
2. implicit-refresh 없는 baseline substrate 복구
3. `select.rs` main matrix에서 `GatedRefresh*`, `PFlipMixture`, `SelfAdaptive` 제거
4. `MetaControlSignal` 개념 정리
5. Group A부터 실행
6. 이후 Group B / Group C 순서로 확장

---

## 18. 한 문장 결론

**phase 1.5는 “Quartz가 더 좋은 prior를 만든다”가 아니라, “Quartz가 belief revision의 commit, challenger proposal, budget routing을 더 잘 통제하는가”를 보는 실험이어야 한다.**
