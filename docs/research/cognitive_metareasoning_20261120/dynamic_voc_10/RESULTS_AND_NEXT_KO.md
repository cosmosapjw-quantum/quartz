# QUARTZ-COG-02-DYNAMIC-VOC-10 | 두 단계 계산 선택과 release 비용 보정

Date 2026-11-21 (Asia/Seoul). Baseline **`93e8e46cb8bfb6a891f8a38ef5ca0539d108611f`**. Branch **`research/quartz-cog02-dynamic-voc-20261121`**. GPU는 사용하지 않았다. **판정: 구현 및 제한된 유한실험 검증 완료, 비용을 일치시킨 근시안 정책 대비 독립적인 비근시안 우위는 미입증.**

## 1. 수학적 목표, 정리, 실제 추정량

**Exact abstract XOR example.** 공정한 독립 비트 `B1,B2`와 표적 `Z=B1 XOR B2`가 있고, 행위는 Z의 이진 예측이며 맞힐 때 보상1, 틀릴 때0이다. 가능한 4개 세계의 확률이 각각 1/4이고, 각 관측의 비용 `c=1/10`이다. 현재 Bayes 최적 기대효용 1/2, 첫 비트만 알려진 뒤에도 1/2, 두 비트 모두 알려진 뒤에는 1이다. 따라서 정확히
```text
EVSI(B1) = EVSI(B2) = 0
Net one computation = -1/10
EVSI(B1,B2) = +1/2
Net two computations = +1/2 - 2(1/10) = +3/10.
```
이 구성에서는 한 단계의 정보가치가 각각 0이어도 두 단계의 결합가치는 양수다. 이는 `prove_xor.py`의 `fractions.Fraction` 유한 열거와 별도의 Wolfram Language 정확 계산 `{1/2,1/2,1/2,1,0,1/2,-1/10,3/10}`이 일치한 **직접 유도/정확 계산**이다. 본 예제는 일반 Connect4 MCTS의 계산 모형이나 제어 알고리즘의 최적성을 증명하지 않는다.

**실제 게임의 empirical estimand.** 표준 7×6 중력 Connect Four에서 Pascal Pons 독립 완전게임 solver가 반환하는 합법 행동별 score `S(s,a)`를 사용하고 `u(s,a)=sgn S(s,a)∈{-1,0,+1}`로 정의한다. `-1000`은 불법 열 sentinel이다. root의 단계별 MCTS 최종행동 `a_b`는 방문예산 `b∈{8,16,32,64}`를 사용한다.
```text
R_b(s) = max_{a∈A_legal}u(s,a) - u(s,a_b)
G_b^(1)(s) = u(s,a_{2b}) - u(s,a_b)
G_b^(2)(s) = u(s,a_{4b}) - u(s,a_b)
C_b^(1), C_b^(2) = measured incremental CPU MCTS search wall time (ms).
```
`G_b`는 음수일 수 있다. 모델 `ĝ_b^(k)(X_b)=β_b^(k)·X_b`는 *현재* root에서 얻는 9차원 MCTS 특징으로 예측하는 **부호 있는 realized gain의 조건부 surrogate**이지 Bayesian posterior/preposterior knowledge gradient가 아니다. 목적함수와 유효범위를 혼동하지 않는다.

## 2. 자료 원천, 이력 및 실행조건

- 연구 정본은 `CONTRACT_KO.md`, `INPUT_LOCK.json` (16개 파일의 실제 byte SHA256)이다. 2026-11-21에 모델 구현 및 자료분할 전에 승인된 새로운 `COHORT_384_FROZEN.json` (SHA256 **`c12c86ab08adb614a80d196d842f31ddf3fc78f7929ddee17523b9569db46ce5`**)에는 총24개 신규 mirror-canonical 5-ply opening-family, 16개 root/family가 있다. `fresh_A`와 `fresh_B`는 각 12 family/192 root, side-to-move가 각 96:96. 기존 G07 train/val/test, C09 두 heldout, C06/old12 모든 prefix 및 canonical board와 겹치지 않는다. 서로 다른 family draw이지 서로 다른 stochastic MCTS trajectory가 아니다. Dirichlet off, seed17이 결정론적 root 행동에 영향을 주지 않는 조건에서 seed만 바꾼 재실험을 독립으로 주장하지 않는다.
- 적합(fit) 자료는 **G07 원래 train 40 family×16=640 root 및 validation 12×16=192 root**, 각 `uniform`, `new_mlp`, `conv` evaluator마다. 실행소스 커밋 **`0e36f3bf0ab5f43142ce2a33028d9845a76b67ca`**에서 2496 actual resident full64 release Rust searches를 수행했다. `evidence/release_train_val_2496_20261121/{RAW_2496_RELEASE_FULL64.json,JOINED_TRAIN_VAL_RELEASE.json,EXECUTION_MANIFEST.json,INDEPENDENT_READBACK.json}` 기록을 별도 프로그램으로 1-step, 2-step, CPU 시간 및 특징에 대해 독립 재계산했다.
- 외부 정확 점수의 접근 시점은 **모든 MCTS 검색이 종료된 이후**다. 연구용 Rust runner에는 `oracle`, `exact_action_scores` 등이 인자로 전달되지 않는다. 원래 G07 test 192, 과거 C09 test384, 이번 C10 new test384를 λ/가중치/비용모델 적합에 사용하지 않았다.
- 7×6 exact Pascal Pons solver와 공개 opening-book은 사용자 로컬의 외부 실행파일 및 자료이며 저장소/백업 아카이브에 재배포하지 않았다. 새 집합은 **384/384 모두 정확한 7-column label OK**, 33.76s bounded CPU solve, frozen file SHA256 **`4af28f7a4e61b1f5c3d2a0d9790f093c29200d91e72026fd4b8ab2b2034e76fa`**. `fresh_A` root WDL win/draw/loss 117/8/67, `fresh_B` 132/8/52. 독립 외부 solver 원본·좌우반전 24+24 확인, 실제 실행 후 추가 24+24 확인.
- Binary: Rust `cargo build --offline --locked --release --bin mcts_demo`, **`4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03`**. 실제 컴파일한 Rust source commit **`4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2`**. 본 연구 작업의 Python 실험 스크립트 추가가 바이너리 소스 identity를 바꾸지 않음. 실행은 CPU-only release debug GPU off, 단일 root/search thread. 시간은 CPU 프로세스 time이 아니라 **같은 호스트의 측정 wall-clock**. 평가기 네트워크는 서버 프로세스 내 캐시되고, model load/setup과 순수 MCTS stage wall을 분리 보존했다.

## 3. 구현 및 train-only 계량·val-only λ 선택

- `src/mcts/dynamic10.rs` 구현: `DynamicTwoStepPolicy`는 strict JSON schema `quartz_cog02_dynamic_two_step_v1`와 9개 current-root 특징, 3개 한 단계 gain, 2개 joint gain, release CPU log-cost Ridge, 양의 cost scale, λ를 해석한다. 8회 또는 16회 시점의 선택은 `stop | single | commit_two`. 두 단계 묶음을 선택했다면 16/32에서 정보를 볼 수는 있으나 **중간 중단 재판단을 생략**하며 이를 skip receipt에 기록한다. 32회에서는 one-step to64 또는 중단만 허용. 불법/비유한 weight, negative cost, 배타적 중단정책 충돌은 명시적으로 거부한다.
- `dynamic_model10.py`, `fit_release10.py`: 한 단계/두 단계 `G_b^(1),G_b^(2)`는 α=1 Ridge로, log(CPU ms)도 α=1 Ridge 후 positive exp 및 **train-only** 평균 scale 보정으로 학습. λ 후보 `{0,.25,.5,1,2,4} WDL/ms`. 원래 validation 192에서 `mean exact WDL - 1.0 × mean actual release search ms` 최대화. 타임캡은 validation-only grid 0.01..10ms(0.01ms step)에서 dynamic의 val 평균 search ms에 가장 가까운 값 선택. 비용 맞춤용 myopic λ 역시 **validation 시간만** 사용. 시험자료 최적화 금지.
- Frozen fit source commit **`f10dc235dd12b0710a9dc1f4d7e5cbfde790f173`**에서 실행한 freeze 모델 SHA256 **`112acbaa6e31be7cd49ba8ad6347daa191059b31056439554d956d8828e123ad`**, `evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json`. `audit_fit_release10.py`가 train-only 각 Ridge normal equation, log-positive cost mean, validation-only λ/cap 선택을 **다른 코드**로 재계산.

| evaluator | λ(dynamic) | λ(myopic) | λ(val-time-match myopic) | val fixed-clock cap | val joint commit count |
|---|---:|---:|---:|---:|---:|
| Uniform | 1 | 1 | 0.5 | 0.14ms | 109/192 |
| New MLP | 4 | 4 | 2 | 0.01ms | 0/192 |
| ConvNet | 4 | 4 | 1 | 0.01ms | 0/192 |

- `parity_live10.py`: 고정된 과거 val 24 roots(12 families)에서 288 actual Rust release requests. `uniform` 20 cases commit8→32 and 4 single; MLP/Conv 24/24 stop8. 동일 root의 Rust/Python coefficients, current-only 특징, skip, action, evaluator-call, visit distribution이 일치했다. 근시안 모델과 두 단계 모델이 달라질 수 있음을 실제 endpoint에서 확인했다.

## 4. 신규 테스트 두 집합에서 독립된 실제 Rust 8064 CPU root searches

**실행량:** 384 roots ×3 evaluator modes ×7 controller arms (`full64,dynamic,myopic,myopic_time_match,legacy_debug_myopic,live_fixed16,validation_clock`) = **8064 실제 release Rust searches**, 소요 약 **6.39s**(해당 Rust 서버 호출구간). Frozen models+λ+validation caps 게시 후 착수; 신규 exact oracle label은 모든 검색 완료 이후에만 읽었다. 실행 source commit **`83e02589c4b39236354636f3ec9de9bd6bfc6c1e`**.

### Uniform evaluation, exact WDL simple regret and measured search wall

| group | dynamic 2-step regret | one-step myopic regret | validation-time-match myopic regret | val-clock regret | genuine live16 regret | dynamic avg ms | joint commits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh A (192) | **0.078125** | 0.088542 | **0.078125** | 0.203125 | 0.286458 | 0.191406 | 118 |
| Fresh B (192) | **0.161458** | 0.166667 | **0.161458** | 0.223958 | 0.270833 | 0.204479 | 98 |
| Pooled (384) | **0.119792** | 0.127604 | **0.119792** | 0.213542 | 0.278646 | 0.197943 | **216** |

**Admitted matched-clock result:** dynamic minus validation-clock pooled exact regret difference **−0.09375**, 24-family fixed-model bootstrap conditional 95% **[−0.1380208, −0.0546875]**. Mean release CPU search time dynamic **0.197943ms**, clock **0.184870ms**, relative mismatch **6.60%**, under frozen **≤10%** mean-time admissibility. Separately Fresh A: Δregret−0.125, 3.76% cost mismatch; Fresh B Δregret−0.0625, 9.27% mismatch. The fixed checkpoint-clock can overshoot per-root time. No claim of equal CPU time on every root.

**Nonmyopic novelty gate fails:** dynamic versus myopic with same validation-selected λ1 had pooled regret difference **−0.0078125**, bootstrap 24-family 95% **[−0.02604,+0.01042]** and **19.08%** actual mean CPU-time mismatch, **NOT cost-matched**. Against myopic with validation-time-matched λ0.5, pooled regret difference **0**, including all paired-root WDL regret differences exactly zero (bootstrap [0,0]); but pooled actual CPU cost mismatch **11.68%** (Fresh A 4.44% matched, Fresh B 18.46% not). Neither comparison establishes independently superior *joint two-step valuation*. This must remain **unresolved**, although the online two-step mechanism and expected benefit of information in the separate XOR model are valid.

**Evaluator-specific failures:** New MLP dynamic and myopic both stop at 8 in all 384 roots, 0 joint commits; pooled regret both **0.169271**, release-clock comparison fails cost gate (20.39% gap). ConvNet also both stop8, 0 joint commits; pooled regret both **0.174479**, clock gap **10.97%** fails ≤10%. Different arm wall-times occur under identical visit/action decisions due to execution/OS timing variation and code-path overhead; do not infer a policy gain from this noise.

**Decision saturation:** In the new test 384 roots, only **209** (Fresh A 98, Fresh B 111) have at least two distinct legal-action exact WDL classes. The other 175 cannot reveal an improvement in sign(WDL)-regret from move selection. Reported aggregate probabilities are conditional on this synthetic group sample and should be accompanied by the informative-root subset (stored in `SUMMARY_TWO_STEP.json`).

### Release expected gain/cost calibration, pooled Uniform

| transition | mean forecast single signed gain | realized mean signed gain | forecast incremental CPU time | actual CPU time |
|---|---:|---:|---:|---:|
| 8→16 | +0.0893 | +0.0391 | 0.0709ms | 0.07397ms |
| 16→32 | +0.0977 | +0.1589 | 0.0728ms | 0.07719ms |
| 32→64 | −0.0516 | −0.1146 | 0.0819ms | 0.08697ms |

8→32 pooled joint forecast signed gain **+0.185756** vs realized **+0.197917**; joint incremental release cost forecast **0.143653ms** vs observed **0.151161ms**. There are **33/384** heldout roots where first additional-step realized gain ≤0 but combined 8→32 gain >0, a finite *observed horizon complementarity* sign. This does not show that the deployed joint option improves utility over a runtime conditional myopic baseline at matching real time.

## 5. Audits, reproducibility and failure classification

- Python `prove_xor.py`: exact Fraction enumeration; independent Wolfram exact crosscheck. Evidence `EXACT_XOR_SYNERGY.json`.
- Python `audit_train_release10.py`: 2496 actual training/validation full64 Rust traces, future signed WDL gain and actual stage costs independently recalc, no C09/C10 test labels to fit.
- Python `audit_fit_release10.py`: Ridge normal equations, release positive-cost calibration, validation-only selected λ and cap independently recomputed.
- `parity_live10.py`: actual 288 release Rust/Python online dynamic two-step parity checks on old validation roots.
- Exact source teacher file `evidence/teacher_new384_20261121/EXACT_384_LABELS.json`, `audit_teacher_c10.py` source+24 external exact originals+24 mirrored re-solves.
- `audit_live10.py`: **8064** actual release Rust roots independently recalc correct root hash, side, legal, action WDL, exact signed utilities, evaluation calls, physical timings, learned one/two-step current-only predictions, skipped intermediate decision, deterministic full64 prefixes, warm model cache. Additional 24 original+24 mirrored external solver re-solves. Frozen new test evidence `evidence/heldout_8064_release_20261121/{RAW_8064_RELEASE_ROOTS.json,PAIRED_384_NEW_ROOTS.json,SUMMARY_TWO_STEP.json,EXECUTION_MANIFEST.json,INDEPENDENT_READBACK.json,VERIFICATION_RECEIPT.json}`.
- Tests actually run in the current turn: `cargo test --offline --locked --release -- --test-threads=4` **596 passed / 0 failed / 89 ignored**; corresponding debug suite **596 passed / 0 failed / 89 ignored**. 11 related Python suites **105 passed / 0 failed**. All output SHA and binary/model source identity in execution manifest and receipt. All 8064 Rust research calls and 2496 train runs were executed, not emulated.
- Failure types: XOR exact rational synergy **derived/proven in finite model**; cost model and conditional signed utility **implementation-verified**; empirical mean regret **numerically checked on 24 fixed family clusters**; strong independent nonmyopic benefit under matched CPU **unresolved**; neural evaluator-specific decision controller **negative/degenerate**; strong general population/Elo/human cognitive claims **not tested**.
- `native_subagent: NOT_RUN` because this host currently exposes no native delegate tools. All work ran via actual serial remote Rust/Python CPU execution, Wolfram, and external solver independent reread. No GPU.

## 6. Posted stage identities and preservation

Every completed unit: Git commit → `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups` local `tar.gz` → `sha256sum` → `git push` → `git ls-remote` exact SHA readback. Distinct commits:

- A `0c3221e655cf945cc86ccade22df4c7740aefd31`: exact XOR, sealed C10 two new cohorts and SSOT.
- B1 `4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2`: actual Rust online two-step commitment.
- B2 `0e36f3bf0ab5f43142ce2a33028d9845a76b67ca`: release CPU train and signed gain/cost.
- C1 `cacf22b404f5b78cc6dd2436cffde9b6a3401b5a`: 2496 real release training/validation and independent audit.
- C2 `f10dc235dd12b0710a9dc1f4d7e5cbfde790f173`: train-only model and val-only λ/cost-match logic.
- C3 `ae741976a0629d0dc8f8d6384797adbbbb37a36e`: frozen models, train/val diagnostic and independent audit.
- C4 `1d5cf0e7ff3802b5bdac657548bca414cc6631c5`: Rust/Python online parity code.
- C5 `11ccbd78b58faad1f7c2258409f37966fe984867`: 288 real online parity evidence.
- D0 `3ccb276c4f1c782551094a3e95c1b2d411e4dcd4`: bounded standalone exact teacher source.
- D1 `c73e27c9d2a5935057f9e63121040178a14049b7`: complete exact 384 oracle, 24+24 re-solves.
- D2 `83e02589c4b39236354636f3ec9de9bd6bfc6c1e`: 7-arm 8064 release trial runner.
- D3 `910187115e93f6d5854c4969ce09f477a35450c1`: actual 8064 trial evidence and independent audit.
- E: final scientific verdict/handoff written here, append exact final SHA+backup after publication.

No author branch other than isolated C10 research branch was modified. Original developer worktree `/home/cosmosapjw/Dropbox/personal_projects/quartz` remains `agent/local-experiment-foundry` at `65e7d3e9596ca259ebfe523b0c74058692b6a755` with original untracked `QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md` untouched. External AGPL Pascal Pons source/binary/book is not vendored into repo/archive. Dropbox **cloud synchronization/restore ACK NOT_RUN**; only local archive existence+SHA and GitHub remote commit equality are verified.

## 7. Next bounded research ID: QUARTZ-COG-02-DYNAMIC-OPTION-COSTMATCH-11

**Do not automatically execute.** To isolate *incremental benefit of planning a two-step option* from extra cost or root budget allocation: pre-register a **validation-only randomized mixture** of myopic λ options matching the dynamic average release MCTS CPU time more closely, including assignment probabilities and pseudorandom source. Fix all policies and mixtures BEFORE constructing/accessing any **new third-party-free heldout 5-ply families**; do not fit to C10 new test or C09 heldouts. Run real myopic mixture independently in Rust with group/seed identity and compare it to frozen two-step dynamic and a fixed-clock baseline. At 10%-within-cost gate, preserve exact WDL regret difference, signed improvement/worsening, family-cluster conditional uncertainty, and a benchmark restricted to WDL-informative positions. Repeat on at least 2 fresh opening-family draws, with honest non-IID/seed limitations. A natural second route tests whether a true posterior/correlated joint observation model can be identified, but this requires source/measure/null/channel assumptions; do not relabel the present Ridge predictor as Bayesian. A separate `Einstellung/candidate attention reopening` study is appropriate **after** the cost-matched nonmyopic claim gate, not silently as part of this node.

**Stop:** C10 scientific claims and posted evidence are complete; actual separately conditioned two-step vs matched-myopic causally distinct superiority remains an open next research claim.
