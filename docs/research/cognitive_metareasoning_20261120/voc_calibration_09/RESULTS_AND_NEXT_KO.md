# QUARTZ COG-02 VOC-CALIBRATION-09: 신규 두 계열에서의 실제 CPU 재현성과 보정

Date: 2026-11-21 (Asia/Seoul). Branch: `research/quartz-cog02-voc-calibration-20261121`. Parent: `569a3ff106a1f0b84a3cd94c189a48a8b05bcdb1`. **Status: scoped verified computation / limited empirical generalization, not Bayes KG.**

## 1. 핵심 질문과 재현 단위

원래 `QUARTZ-COG-02-COST-MATCHED-VOC-08`의 정규화된 current-root 9-feature Ridge signed-gain 예측기, 단계별 CPU cost forecast, 검증 선택 λ를 **어떠한 재학습·변경도 없이** 다른 24개 오프닝 계열에서 적용하면 실제 CPU release 환경의 의사결정 효용이 유지되는가? 이를 별도 `live_fixed16`, validation-only time-cap과 실제 동일 root engine의 full64 궤적에 대조했다.

현재 게임은 standard 7×6 gravity Connect Four. Pascal Pons external exact solver의 합법 행동 점수 `S(s,a)`로 `u(s,a)=sign S(s,a)∈{-1,0,+1}`, `regret=max_legal u−u(s,chosen)`, `G_b=u(s,a_2b)−u(s,a_b)`를 정의한다. `-1000`은 불법 행동 sentinel이므로 효용이 아니다. `E[G_b|X_b]`를 회귀한 예측기는 **realized signed continuation-gain surrogate**, 베이지안 사후 분포나 진정한 preposterior knowledge-gradient를 구현하지 않는다.

## 2. Source·cohort·정확한 oracle authority

- Repo parent `569a3ff106a1f0b84a3cd94c189a48a8b05bcdb1`, new branch `research/quartz-cog02-voc-calibration-20261121`; `INPUT_LOCK.json`에 15개 원본 input/model/runtime/external solver-byte SHA256. New immutable `COHORT_384_FROZEN.json` SHA256 `30101889eb4c17ff11ace381e6366a53b2af5130341a61e6969ac9ea091fe54e`.
- 24 **new mirror-disjoint five-ply families**, 16 descendant roots each: `replica_A` RNG20261123: 192 roots; `replica_B` RNG20261124: 192 roots. Each subgroup black/white next-to-move 96/96. Every prefix and board state modulo mirror excludes previous 64-family G07 cohort, COG06 source and earlier historical 12. Two new *cohort draws* are not independent MCTS stochastic search replicas, nor 384 IID roots.
- Exact oracle Pascal Pons: 384/384 complete 7-column score vectors, **0 timeout**, about 35.30s standalone CPU; original/mirror 24+24 further re-solve (teacher audit) and another 24+24 after live experiment. `new384_exact_teacher_20261121/labels.json` SHA256 `2220c6959934b26425594a8151948a8c7140300f429303a34be733aff56ed511`. Wins/draws/losses: A 140/6/46; B 127/11/54. 100/384 root forced-loss, and only **229/384** new roots have more than one possible legal action WDL class, so report both full and decision-informative subsets.
- Old 24 sampled G07 test positions: changing MCTS seed17→231 with Dirichlet off changed **0 root best action/visit policies**. A mere integer seed change was explicitly *not* promoted to independent stochastic replication.

## 3. 실제 16회 중단, release 실행 및 CPU 비용 이동

- Rust `src/mcts_server.rs` research-only `force_stop_visits=16` pathway implemented **TDD RED→GREEN**, does not execute 32 or 64, actual resident MCTS checked `fresh16` at stop. Existing adaptive, clock and full64 paths unchanged. `force_stop_visits` malformed/budget-conflict requests rejected.
- Rust `cargo build --offline --locked --release --bin mcts_demo`: optimized binary SHA256 `e3c44f8630c67d0c00929f10b429f35d62805b779e025d73ce16f142100c7f78`, 3,646,296 bytes. Compiled from Rust source commit `1c2b00c1bad2585380fee97bde65e3862dc1f25d`. Debug and release source semantics checked by test suites.
- Previous **debug-trained** stage next-cost forecasts averaged milliseconds; moving to the same CPU host's release binary reduced actual per-stage timings by about **20–30×** on the new test cohorts. Old λ, old gain coefficients, old next-cost forecasts and prior trained MLP/ConvNet weights remained absolutely unchanged. This provides a concrete **CPU-cost calibration failure** under build-optimization domain shift, not physical novelty or a solver error.
- Frozen validation-only release clock caps were determined with **1,152 actual Rust requests** on the OLD 192 validation roots. First derive release adaptive mean physical search milliseconds with old policy, then grid-select fixed time cap from `0.01..10.00 ms` by minimum average CPU-time mismatch, without observing any new test oracle or changing the signed-gain policy. Selected cap: Uniform `0.12ms`, MLP `0.01ms`, Conv `0.18ms`. Released source `release_validate09.py`; `evidence/release_validation_1152_20261121` includes raw and independent readback.
- Clock cap is checked after completing checkpoint 8/16/32; it can overshoot. All wall times are per-root release Rust **wall-clock**, not processor CPU time; model weights resident within one process after first load. Model cache keyed by caller-provided SHA, not per-request recomputed file-content digest.

## 4. 완전히 고정된 두 새로운 평가 코호트, RELEASE 실제 4,608 MCTS

384 new roots × 3 evaluator modes (Uniform, new MLP, Conv) × 4 actual Rust arms (`full64`, online `adaptive`, independent `live_fixed16`, `validation_clock`) = **4,608 actual Rust root searches**, 3.54s experiment process time (excluding wrapper and separate teacher). Policy and comparison budget were committed **before** opening the new-family exact scores. External label file was accessed only after 4,608 Rust searches completed; early stop decisions saw no external oracle or future completed search result.

### Mean exact WDL simple regret, two separately frozen 12-family cohorts

| Cohort | Evaluator | Adaptive regret | Independently live16 regret | Validation-clock regret | Adaptive search ms | Clock search ms | clock cost match |
|---|---|---:|---:|---:|---:|---:|---|
| Replica A | Uniform | **0.14583** | 0.20313 | 0.21354 | 0.13532 | 0.13679 | PASS 10% |
| Replica B | Uniform | **0.19271** | 0.30729 | 0.30208 | 0.13277 | 0.13342 | PASS 10% |
| Pooled | Uniform | **0.16927** | 0.25521 | 0.25781 | 0.13404 | 0.13510 | PASS 10% |
| Pooled | New MLP | 0.23698 | **0.18750** | 0.23698 | 0.09060 | 0.08782 | PASS 10% |
| Pooled | Conv | 0.24219 | **0.16927** | 0.24219 | 0.22697 | 0.26590 | **FAIL 10%** |

- Uniform, adaptive−**genuine live16** pooled mean regret difference `−0.0859375`, fixed-policy **24-family cluster bootstrap** 95% conditional interval `[−0.138021, −0.03125]`. Separately A difference `−0.057292`, CI `[−0.130208,+0.015625]` crosses zero; B `−0.114583`, CI `[−0.192708,−0.041667]` excludes zero. Same sign in both, statistical strength not equal.
- Uniform, adaptive−validation-clock pooled mean regret `−0.088542`, 24-family conditional CI `[−0.135417,−0.039063]`; average measured CPU-time mismatch **0.79%**. Both A and B individually met pre-fixed 10% mean-time admission. This is now stronger than previous VOC08 fixed16 **posthoc prefix-only** comparator because fixed16 was itself run as an independent live termination arm.
- MLP surrogate chooses visit8 for **all 384 roots**: the model has not learned a discriminative stop gate in this regime. Its pooled adaptive regret0.23698 exceeds genuine live16 regret0.18750; CPU is cheaper, so one should not call the difference pure algorithmic harm under equal cost.
- Conv surrogate adaptive pooled regret0.24219 vs live16 regret0.16927, but it is substantially cheaper; its validation-clock comparison is **not admitted as cost-matched** because pooled mean time mismatch is **17.15%**, replica A 22.51% and replica B 12.33%, all exceeding pre-specified 10%. Its adaptive–clock equal regret is descriptive only.
- Decision-informative new roots: 229/384. Uniform adaptive regret on these=**0.28384** versus genuine live16 **0.42795**, while aggregate regret is lowered by 155 roots in which all legal actions have same game WDL. This gate prevents trivial forced-loss or equivalent-action positions from overstating effective decision improvement.
- All 4,608 actual root searches had independent action/eval/visit-policy prefix identity and external-oracle WDL audit. This is **conditional finite-sample experimental replication**, not universal human cognition nor literature-theorem novelty.

## 5. Gain and cost forecast calibration

From each full64 actual release trace, evaluate all possible next-step gains at 8→16/16→32/32→64 with the **unchanged** original model, **without sending future outputs into online adaptive control**. The six gain forecast bins were registered before measuring the test: `(-inf,-.2]`, `(-.2,-.05]`, `(-.05,0]`, `(0,.05]`, `(.05,.2]`, `(.2,inf)`. Full raw bin counts, MSE/MAE, false-stop and needless-continue counts are in `SUMMARY_CALIBRATION.json`.

### Pooled Uniform, same two new cohorts

| stage | Predicted mean signed gain | Exact realized mean gain | Debug-trained forecast cost / release actual next-step time |
|---|---:|---:|---:|
| 8→16 | +0.08349 | +0.04427 | **29.3×** |
| 16→32 | +0.08933 | +0.10417 | **28.8×** |
| 32→64 | −0.04785 | −0.04427 | **27.4×** |

Uniform expected gain means have the correct **sign** on these three pooled stages, but this is not a high-resolution calibration guarantee. Offline future-step diagnostic: false-stops despite next-step beneficial gain count 6,0,16 per stage; predicts continue despite nonpositive gain 147,155,5. These are **counterfactual full64 observations**, not data seen by the online policy. The 20–30× time forecast discrepancy is a material blocker to claiming release-time-optimal VOC.

## 6. Verification and code identity

- Final Rust debug `cargo test --offline --locked -- --test-threads=4`: **593 PASS / 0 FAIL / 89 ignored**.
- Final Rust release `cargo test --offline --locked --release -- --test-threads=4`: **593 PASS / 0 FAIL / 89 ignored** after preserving and fixing one unrelated release NPS timer-quantization test-contract failure. An intermittent clean release-suite rerun had returned **592 PASS / 1 FAIL** at `test_run_par_quartz_four_threads`: on a sub-1ms optimized run the implementation rounds `elapsed_ms` to 0 and deliberately reports `nps=0`, but the test had assumed `nps>0` unconditionally. **The core search algorithm and experimental release binary were not changed**; only the post-experiment test assertion now checks positive NPS when elapsed_ms>0, and zero NPS when elapsed_ms==0. Full forensic record: `RELEASE_TEST_TIMING_REGRESSION.md`. Focused release test and both full debug/release suites were actually rerun and passed.
- Related Python research suites: **93 PASS / 0 FAIL**, including 6 new calibration09 tests.
- Input/teacher: `audit_new384.py` externally re-solved 24 new original and 24 mirror positions and found no collisions.
- Release validation: `audit_release_validation09.py` independently checked all 1,152 val Rust responses and WDL-unaware cap selection.
- Live test: `audit_release_new38409.py` independently verified all 4,608 raw responses, true 16-visit stops, stage root prefix, exact WDL regret and costs, policy forecast root features and signed-gain summaries, then re-solved 24 new originals and mirrors (48 additional external oracle checks).
- Exact frozen release binary SHA, source commit, model weights, old policy and validation cap SHA, new root teacher SHA, all result bytes SHA are stored in `EXECUTION_MANIFEST.json` and `VERIFICATION_RECEIPT.json`.
- Native subagent: **NOT_RUN** (host has no native agent dispatch). Work was actually performed with serial Rust/Python CPU, external Pascal Pons solver and independent readback code. GPU not used.

## 7. Stage-by-stage source publication and preservation

Each unit: Git commit → local Dropbox archive `tar.gz` → SHA256 → GitHub push → remote `git ls-remote` exact SHA equality.

- A: `302f9f13a746a9e3fccfe8d55b69dd7d979b0bbd` (new 384 root /2 cohorts and input lock).
- B0: `47c7335f8e891e9fe8a2d79c57cc0c31fbabbf11` (strict external teacher wrapper).
- B1: `6a0596cf47cc5867ceb6b94e75ff810fcab6b39c` (384 exact labels and independent solver mirror audit).
- B2: `1c2b00c1bad2585380fee97bde65e3862dc1f25d` (actual Rust `force_stop_visits=16` and test).
- C0: `0009e5d2da46fa063f8da6714f31380c8835217c` (release validation calibration runner).
- C1: `63f87a0c23553cf092d0bb79340a322b946ba328` (1,152 release validation roots, clock caps, independent cost readback).
- D0: `7b854f1b1c05e87ca7fceb6c153d630fafb30469` (sealed actual 384×3×4 release runner).
- D1: `fc13e38b5ee099e051182e5042ddba0051cef4c3` (raw 4,608 release trials, calibration, independent audit).
- D2: `2cda651ba0fdb736378514c780d852c92edab91a` (test-only release NPS timer rounding contract correction and failure provenance; core search runtime and original frozen executable untouched).
- E: this report + next handoff, exact final commit/tree/backup SHA verified after publication.

Backups at `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`. **Dropbox cloud sync ACK and archive restore test NOT_RUN**. Original user working tree `/home/cosmosapjw/Dropbox/personal_projects/quartz` on `agent/local-experiment-foundry` at `65e7d3e9596ca259ebfe523b0c74058692b6a755`, including its pre-existing untracked context file, preserved unchanged. No external AGPL solver source/book/binary redistributed.

## 8. Next research ID: QUARTZ-COG-02-DYNAMIC-VOC-10

**Scientific admission:** A fixed, empirical signed-gain stop policy gave reproducible **finite two-new-cohort Uniform CPU benefits** under genuine live16 and time-matched clock baselines, but not a Bayesian optimality theorem and **not** a generally calibrated evaluator-specific VOC. The MLP gate collapsed to 8 visits and Conv failed matched-cost gate. This is the terminal verdict for node09, not a reason to erase these negative results.

1. Build a release-calibrated **joint model of signed next-gain and actual CPU expense** using only original train/validation family data; keep current C09 new families and all previous test families fully held out from refitting. Compare current 9-feature Ridge to uncertainty-calibrated alternatives and include asymmetric false-stop/risk costs, not only mean signed gain.
2. Construct and rigorously test one **explicit two-step/nonmyopic computation option**. A pair of individually valueless correlated computations may jointly be valuable; demonstrate the finite conditional-expectation counterexample first, then implement bounded two-check stop/continue policy and ablate on distinct new root families.
3. Ensure actual MCTS search PRNG branch genuinely differs across runs before counting multiple search-seed replicas. If seed changes are inert with Dirichlet off, report them as **same deterministic trajectory**, not new sample size. Freeze any noise model or policy modification separately.
4. Validate at least one fresh non-overlapping opening cohort and separate stronger board-game transfer (15×15 Gomoku/Rapfi) before proposing broad cognitive mechanism conclusions or attention-reopening performance.
5. Once utility improvement and real cost savings survive out-of-family tests, open `COG-02 Einstellung / candidate attention reopening` as its own independent node, not as a silent mutation in this completed node.

**Stop condition:** Node09 has delivered fully archived actual release CPU tests and fixed-policy calibration evidence. No more auditing loops or automatic next-stage execution; next node10 remains unstarted.
