# Quartz C21 최종 연구결과 및 다음 단계 C22 인계

연구 ID: QUARTZ-GENERAL-21-CDF-EXACT-EVSI-NATIVE-RESOURCE-GATE.
기준 원본 C20 Git SHA ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4, 브랜치 research/quartz-general-c21-fast-cdf-20261122, 연구소스 디렉터리 docs/research/quartz_general_cdf_native_20261122. 날짜 2026-11-22 KST. 연구 목표는 original frozen docs/THESIS.md P1–P4에 명시된 **도메인 중립 계산 선택, 배분 및 STOP**이며 오목 한 게임의 성능 최적화나 특정 게임 지식 추가가 아니다. 기존 C16–C20 연구 브랜치, 원래 개발 브랜치, THESIS/CLAIM_LEDGER/GameState 원본은 변경하지 않았다.

## 1. 이론 R01–R10: Bayesian value of computation의 빠른 정확 적분

R1 원본 권위/가정과 과학 질문, R2 직접 선행연구, R3 최대 사후평균의 CDF 직접 유도, R4 수치/관측채널의 대안 구성, R5 mass/EVSI/cost 정리, R6 상관 모수/중복 지원점/비용 반례, R7 Wolfram+독립 Python Fraction 검산, R8 근거 상태 판정, R9 코드/측정 단위 계약, R10 범위종료와 구현 인계. R1..R10 각 단계 별도 Git commit, local Git tar.gz SHA256 백업, GitHub origin SHA 일치 확인 후 다음 단계 수행.

서로 독립인 posterior arm θ_j∼Beta(a_j,b_j)에서 future Bernoulli count S_j∼BetaBinomial(n_j,a_j,b_j), 사후 기대 terminal action utility X_j=(a_j+S_j)/(a_j+b_j+n_j), F_j(v)=P(X_j≤v). Then F_max(v)=prod_j F_j(v). 서로 다른 score 값의 정렬집합 V에 대해 E(max_j X_j)=sum_{v∈V}v[F_max(v)-F_max(v^-)] 또는 bounded [0,1]에서 survival sum. Gross EVSI G=E(max_j X_j)-max_j(a_j/(a_j+b_j)). 전체 count grid ∏(n_j+1) 대신 E=sum_j(n_j+1) score events를 정렬하고 동일 지원점 그룹을 동시에 update하면 **O(E log E)** 시간, O(E+K) 공간 (K arm, distinct scores). Beta shape a,b>0, independent arm posterior, known complete terminal candidates and correct observation channel이 필요하다. Repeated Bernoulli within arm은 동일 θ를 공유해 unconditional IID가 아니므로 Beta-binomial을 사용한다. Correlated arm posterior/neural tree backups에는 product-CDF identity를 직접 적용할 수 없다. Bayes model correct일 때 0≤G≤EVPI; realized task Z는 음수 가능. Operational net=G−λ(C_task+C_meta), λ utility/μs.

Wolfram과 별도 Python 정확 유리수 검산에서 (a,b)=(1,1),(2,2),(3,3), n2 each, 27 joint outcomes와 7 unique support 점을 비교, E(max posterior mean)=529/840, G=109/840, 유리수 차0; 공통 score 1/2 동률 3개를 검증. Fractional Beta(1/2,3/2),n2 predictive masses (5/8,1/4,1/8), total1. 이는 기존의 확률론·Bayesian metareasoning 항등식을 계산에 효율적으로 적용한 것이지 새로운 일반 정리나 독립 posterior 모형의 신경망 MCTS에 대한 정당화가 아니다.

문헌: Hay et al UAI2012 selecting computations, Callaway et al UAI2018 learning to select computations, Hironaka et al 2019 MLMC EVSI, Fleischhacker et al 2023 multinomial EVSI, Efrem 2024 joint maximum distributions. 각 출처는 theoretical prior art이며 C21의 실제 구현 PASS 근거는 실행 로그/코드/자료다.

## 2. 코딩 C01–C10: native Rust CDF implementation

C1 C20 original Git/THESIS/CDF witness source lock, C2 typed source-only posterior + resource API, C3 수치·비용·TEST 계약, C4 새로운 independent C21 cohort source freeze, C5 Rust exact Beta PMF + CDF sweep TDD RED1→GREEN5, C6 Rust/Python exact reference parity, original code full Rust regression, C7 source/price/score 실행코드 **데이터 수집 이전** 게시, extra C07D source TRAIN+CAL raw separately sealed, extra C07A TRAIN-only native price fit, extra C08A untouched TEST source raw separately sealed, C8 genuine matched Rust extra task + standalone native sensor control/scoped shadow result, C9 independent source/math replay, archive recovery, same-compiler full-grid vs CDF native bench, C10 report and handoff. 각 R1–R10/C1–C10 단계 개별 Git+sha archive+push/remote ACK, extra C07D/C07A/C08A까지 총 **23** 별도 stage archives는 C10 최종 게시 후 다시 검증한다.

C21 release Rust source src/bin/quartz_c21_fast_evsi.rs, compiled release binary SHA256 **ad49fb9fbb125cb0193e454ee508bc083737bf446754200663143c5b0fcd4650**. 두 형태의 실행: source-only beta_eval는 posterior Beta shapes 및 sample budgets만 받으며 latent actual arm means/후속 observation/final action/actual source cost 금지. Separate bench_task OFFLINE-only permits θ to perform actual native Rust Bernoulli future draws and choose outcome; no θ is routed to beta_eval. Long-lived --server JSONL and CLI support same numeric evaluator. Validated numeric scope K2..32, n_j0..20, positive proper shapes [0.1,100], distinct/tied postdecision means, PMF normalization tolerance1e-10 and EVSI CDF check1e-9. Rust TDD RED missing method failed with exit101, implementation GREEN5/0, real full old code regression.

Independent Python lgamma count PMF and CDF reference on **163 bounded cases ×4 values=652 numeric equalities**, max abs error **5.60618e-12**, 69 separate Python full-grid cases and fractional/heterogeneous/K32/zero-budget stress. Later independent same-Rust-compiler benchmark source src/bin/quartz_c21_grid_reference.rs provides a full count enumeration separate implementation and 2 new focused tests; this does not change original QUARTZ SearchPolicy.

## 3. Prospective source and scientific outcomes

New source bank frozen before observations, SHA256 **036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8**: independent Beta(2,2) arm parents 120 and four-hidden-state noisy-sensor parents 120, each x3 future outcomes. 72 TRAIN/24 CAL/24 TEST original parents per family. Source RNG names new C21 namespace, independent C20 prior TEST, model-law true θ only offline.

C07 frozen source + score code Git9b6cf2407558d5a2f022196df95f155138cb91f9 before raw generation; C07D actual TRAIN/CAL raw 576 genuine Python stochastic future paths SHA256 **db100b0ee9b4c8d8669c2c20d2beebabeb7bf3d74d017790259d667c10ae7bea** separately published Git8f8244ade4ac099499248837a5a63146c6cd1c23. C07A TRAIN-only price resource model SHA256 **d53b6ffdfa422f5b274d0373d9839e958b96528126a7cc4a66c21a73b4401d01** published Git6c9ce294e25b6c3dca7cbd6f96c7ea53f9ca3dbc. C08A untouched TEST raw separately published Git5e41c2e4f5a0ea8a5ede4ec1fbae458e3be38f96, SHA256 **cb834de8c9b312e1ac168c42adeee571c116600a32cadefb604dd257442dbd74** before scores.

Heldout TEST 48 independent parent groups, 24 Beta and 24 sensor original worlds, 3 actual future stochastic continuations per original parent =144. Beta true native Rust extra Bernoulli future draws were independently generated by matched Rust executable bench_task, from the SAME original latent world/initial observations (distinct future RNG from Python source). Sensor actual source was Python conditional categorical sample/decision update with C20 frozen compiled Rust sensor exact predictor in one batch. This deliberate execution-boundary asymmetry is reported and precludes asserting uniformly matched two-domain host speed.

Bayes gross G Rust native vs independent bounded C20 full grid Python Beta or independent sensor Python on all 144 initial-only cases, maximum absolute model value residual **2.77556e-15**. Actual task true terminal utility label is latent expected mean U(a,θ), not one sampled terminal reward. Hidden θ never present in model source-only request. Beta n72: mean predicted G0.04617149, mean observed native sampled decision signed gain Z0.01966566, C21 exact G-v-Z MSE0.00935592 vs frozen C19 coarse q×m MSE0.00857196. Sensor n72: mean G0.03584255, mean observed signed Z0.04667378, exact MSE0.04318472 vs coarse0.04568795. Parent bootstrap500 descriptive 95% MSE exact-minus-coarse beta[-0.0005653,+0.0020301], sensor[-0.0071045,+0.0010804], both include0; **no broad superiority or calibrated postmean gain claim**.

## 4. Measured CPU resource economics and decision gate

TRAIN-only **median** measured task/native evaluation function/native long-lived JSON host roundtrip in microseconds (sensor is Python task/Rust model native/AMORTIZED one-off batch host, not per-case streaming):
- Beta K2: **0.120 /0.551 /22.405**
- Beta K3: **0.131 /0.9165 /23.182**
- Beta K4: **0.191 /1.1575 /24.094**
- Sensor K3: **8.0 /2.696 /12.7183** (mixed Python/Rust and batch averaged).

TEST actual mean Rust Beta task0.26656 μs, native exact evaluator1.73781 μs, long-lived JSON/pipe roundtrip28.28306 μs. Sensor actual Python source task8.90278 μs, native Rust exact finite model function3.42321 μs, amortized single-batch native host24.16022 μs. First cold Beta server request with process startup separate ~651.531 μs on TRAIN.

Primary preregistered price lambda=.004 terminal utility per microsecond; paired counterfactual mean net improvement vs immediate STOP:

| Shadow proposal | Beta native | Finite sensor |
|---|---:|---:|
| NEVER |0|0|
| ALWAYS execute task without meta-evaluator |+0.01859944|+0.01106267|
| EXACT expected gain, evaluator cost IGNORED |+0.01859944|+0.03146992|
| EXACT, charge native FUNCTION CPU on every decision |+0.01173172|+0.00564950|
| EXACT, charge JSON/host BOUNDARY on every decision |**-0.11347520**|**-0.07346363**|

Beta native internal policy computes71/72 under fixed TRAIN medians, host-boundary policy computes8/72. Sensor native internal computes25/72, batch-host policy6/72. Beta internal charged minus ALWAYS group bootstrap500 95% [-0.007909,-0.005990], negative in this calibrated PRICE experiment; sensor [-0.041332,+0.026180] inconclusive; full host boundary charged minus ALWAYS bootstrap for both domains negative. All three prespecified prices0.001/.004/.010 reported in raw full JSON. Each extra computation actually executed regardless of SHADOW recommendation; avoided source tasks and CPU savings are RETROSPECTIVE COUNTERFACTUAL ACCOUNTING only, NOT measured active STOP improvement, nor real adaptive neural-MCTS posterior/utility. Correct mathematical EVSI is not automatically operationally valuable if getting that answer costs too much CPU.

## 5. Independent C09 science audit and new same-compiler efficiency measurement

A separate auditor, not importing production generator, Beta EVSI model or score driver, **replayed all144** Python source stochastic observations and Bayes actions, then24 Beta test parent examples via exact Fraction **full joint 11^K grid** and 24 sensor parent examples via **all ordered 3^8 sequences**, max numeric EVSI error **4.85167e-14**. The Beta Rust native task RNG was deterministically REPLAYED 72 tasks using the same native StdRng code and exact source seeds, matching signed Z and action-change labels; this verifies source identity, not an independent RNG implementation. The original naive source-row vs forecast pairing was corrected by key (original parent id, replicate, family) after discovering interleaving vs grouped presentation; no TEST data/source edited and independent audit then PASS. Signed MSE and priced means at all three predeclared lambdas independently recomputed to <=1e-12.

Actual local Git C08 archive **QUARTZ_GENERAL_C21_C08_2c3e73cf16bc.tar.gz**, sha256 **3c08c400866c32b99e328bfd5bf24e129908585255a46a9288a7023753b4992d**, safely extracted to fresh temporary directory and **16 original/source/model/RAW/report/code artifacts** byte SHA matched. Local archive RESTORED; independent Dropbox cloud sync ACK not verified.

New SOURCE-TRAIN-only, POST-C08 exploratory same-compiler Rust full grid vs CDF fast benchmark: src/bin/quartz_c21_grid_reference.rs independent joint count-vector Rust version. Same Rust release compiler target and process, 216 original C21 TRAIN-only initial posterior cases (72 per K2/3/4), each variant x20 alternating-order repeated timings. Max absolute model EVSI difference **1.96509e-14**. Rust median full-grid versus fast-CDF internal compute us, median paired ratios:
- K2 **0.78125 vs0.4860us**, **1.61x** speedup.
- K3 **5.94425 vs0.7865us**, **7.72x** speedup.
- K4 **60.733 vs1.022us**, **60.72x** speedup.

This demonstrates a real algorithmic speedup in matched native code on these bounded model instances, not generic runtime/GPU/CPU hotpath effectiveness. Compiled grid reference SHA256 **f4786b642ee0cdbfd7c650872016d5bbde84478e911d62bff5128ed006d85f28**. Native C21 predictor SHA256 **ad49fb9fbb125cb0193e454ee508bc083737bf446754200663143c5b0fcd4650**.

Full C09 post-C08 Rust debug+release test commands both PASS: original main MCTS617 PASS/89 ignored, C17 CLI7, C18 CLI3, C19 CLI3, C20 CLI3, C21 fast CLI5, C21 grid-reference CLI7 (includes nested five tests and two extra grid tests), **645 PASS/0 FAIL** each configuration. Logs evidence/C09_LOGS. No original module/Git main changed.

## 6. Science verdict and concrete C22 target

**Derived/established** exact product-CDF posterior max, event sweep O(ElogE) under independence, Beta predictive count masses and EVSI bound. **Numerically checked/implementation verified** actual stable bounded C21 Rust K2..32 n0..20 shape[.1,100], 163-case Python reference numeric parity 652 values, 216 TRAIN same compiler full-grid benchmark, 144 prospective two-family heldout native-source scientific outcomes and independent auditor. **Unresolved/not admitted** across-domain model superiority/calibration, paid full protocol runtime viability (negative net here), true original neural MCTS task utility and likelihood (NOT_ESTIMABLE), actual physical task CPU saving from active STOP (NOT_TESTED), user original P2/P3/P4 production/scientific signatures (NOT_TESTED). No general mathematical no-go is claimed; the host protocol bottleneck is a contingent measured implementation issue.

Next bounded stage **QUARTZ-GENERAL-22-INPROCESS-VOC-GATE-AND-REAL-COMPUTE-ACTION**. Instead of adding a new heuristic or more process-only gates, actually integrate the source-only fast CDF EVSI into the **same Rust process** as a controlled *known-utility non-game task executor*. Design one cheaper pregate or analytical cost bound to avoid expensive meta-evaluation when it cannot be economical, with correct prospective independent source observations and TRUE active STOP execution on a bounded subset, so that actual wall CPU saved, utility noninferiority and wrong-stop loss can finally be estimated. Keep real MCTS policy read-only until true utility/likelihood identified; not optimize Gomoku itself. Require next full R01-R10+C01-C10 and per-stage Git push/hash backup, separate parent Git worktree and frozen original. C22 is NOT started by C21.

Provenance: full docs at theory/, coding/, SOURCE_WORLDS_C21_FROZEN.json, FROZEN_C21_RESOURCE_PRICE.json, evidence/C07_TRAIN_CAL, evidence/C08_TEST/C08_HELDOUT_NATIVE_EVSI_RESULTS.json, evidence/C09/NATIVE_GRID_TRAIN_ONLY_REPORT.json, C10_FINAL_RELEASE_MANIFEST.json and NEXT_STAGE.json. No main merge or PR was created. Native subagents unavailable NOT_RUN; serial implementation. Final C10 archive and provider ACK require actual postrelease verification to be claimed, never pre-asserted.
