# QUARTZ C19 | 범용 부호 있는 계산가치와 비용 기반 shadow 선택 연구 최종 보고

시점 2026-11-22 KST. 연구 ID QUARTZ-GENERAL-19-SIGNED-VOC-AND-SHADOW-SELECTIVITY, 상위 source C18 최종 Git SHA 097255bd00ffa436c3d43704ed201815e7cbb8ea. 독립 브랜치 research/quartz-general-c19-signed-voc-20261122. QUARTZ의 original frozen docs/THESIS.md P1–P4는 범용 bounded-rational computation allocation/stopping 연구 목표이며, 이 작업은 특정 게임의 최적화가 아니다. 이전 developer, C16/C17/C18/원본 THESIS·CLAIM_LEDGER·GameState 코드를 보존했고 live SearchPolicy에는 개입하지 않았다.

## 1. 이론 연구 R1–R10, 실제 수행·개별 Git backup/push

R01 원본 authority/계산 단위, R02 Russell–Wefald/Hay et al/Sezener–Dayan/Callaway 원전 대비, R03 관측변수와 실제 효용 정의, R04 2부분/상수/flip-only 모형, R05 조건부 정리 직접 유도, R06 비용·통계·사후선택 반례, R07 Wolfram exact rational 및 Python Fraction, R08 근거상태 판정, R09 구체적 Python/Rust 실험 계약, R10 이론→코딩 전이. 전체 10개는 각 stage commit, local SHA256 Git tar, GitHub remote SHA readback 후 후속으로 진행했다.

원리: Y=1[a_after≠a_before], Z=U(a_after,θ)-U(a_before,θ), Z=0 for unchanged action, q=P(Y=1|X,c), m=E[Z|Y=1,X,c] (q>0), gross G=q*m, priced N=G−λ E[C|X,c]. 두부분 계산가치 분해는 확률법칙이며 새 정리가 아님. q와 m 각각 신뢰할 수 있는 구간이 있으면 q*m의 구간은 양 끝 점들의 곱 네개 극값으로 구한다. λ>0이면 N 하한>0은 한 단계 계산의 선택을, N 상한<0은 중단을 충분히 정당화하나, **그 구간 자체의 calibration/coverage는 경험적으로 검증해야 한다**. Wolfram과 독립 Python Fraction: q=1/4, m=+1/2 vs−1/2, 비용=1/10이면 N=+1/40 vs−9/40. q∈[1/4,3/4],m∈[-1/2,+1/2]이면 N∈[-19/40,+11/40]로 강제 결정 불가. 정확한 Bayes 모델에서는 평균 gross EVSI≥0이나 실제 개별 utility Z는 음수가 될 수 있다. 논문의 신규성으로 주장하지 않는다.

## 2. 코딩 C1–C10 실제 자료·재현성

C1 C18 source/doc lock, C2 두 과제 interface, C3 숫자·비용·검정 동결, C4 240개 latent parent source bank로 사전 세계 고정, C5 Python signed_model test-driven RED5→GREEN5, C6 독립 Rust standalone SHADOW code test RED1→GREEN3+실제 Rust Debug/Release full regression, C7 사전 source/fit/heldout scorer code Git 게시, C07A TEST 이전 TRAIN/CAL raw·모델 고정 별도 게시, C08A TEST raw 원자료 별도 게시 후 C8 본 TEST 비교 결과, C9 완전히 독립적인 RNG/후험/score 감사와 실제 C08 tar 복원, C10 본 최종 보고/다음 연구 계약/게시 검증. **모든 R1–R10/C1–C10 단계는 별도 git commit+SHA256 hashed local tar archive+GitHub push/remote SHA equality**, C07A/C08A 포함 총 22개 아카이브를 최종 검증해야만 완료를 인정한다.

생성한 두 실제 계산 과정은 임의 장기 시뮬레이션 게임이 아니라 서로 구조적으로 다른 비게임 의사결정 문제다. **Beta-Bernoulli arm**: world마다 K2/3/4개 arm, latent mean Beta(2,2)이며 실제 initial3/arm + extra10/arm Bernoulli draws, posterior Beta(2+s,2+f)로 새 결정. **Finite categorical sensor**: 4-state latent world, state-conditioned 3-outcome noisy measurements, public state-action utility matrix4×3, initial2와 extra8 실제 sensor query 및 Bayes posterior update. True θ와 latent reward means는 source/scorer의 OFFLINE labels일 뿐 실시간 shared forecaster input으로 사용하지 않았다. 각 source family parent120개×3 observations=360 extra computations, 도메인당 TRAIN72/CAL24/TEST24 independent parent groups; 테스트는 72 episodes/domain =144. SOURCE_WORLDS_FROZEN.json SHA256 8fce353c5efd3a5ac2a643e5a799d46f8d69c6c429cdff0fb5418a43ea43292e. C19 물리 비용은 실제 Python monotone wall의 추가 관측·후험·최종 행동 계산만 측정했으며 C18 Rust 20ms TOTAL source cost와 혼동하지 않는다.

C07A TRAIN 실제 stochastic observation computations **576**: source family별 288. Beta의 decision flips95, mean signed true payoff+0.0522881, mean extra cost7.0903μs; finite sensor flips61, signed+0.0211506, cost12.816μs. TRAIN432, CAL144 observations만 사용해 pooled logistic q coefficient [−0.68610737,−4.37351397,+0.19072281], conditional m ridge [−0.02355221,−0.16553876,+0.12092599], CAL scalar q offset−0.22600146, m offset+0.01884282를 고정했다. Pooled q/m forecast feature는 generic initial utility margin, log1p(action count)뿐이며 domain ID 없이 적합; however K 분포가 일부 source family를 암시할 수 있으므로 strict transfer 증명은 아니다. TRAIN-only per-type incremental cost medians: beta6μs, sensor11μs. Frozen model byte SHA256 **5fdd172a13d51c4d17dc8e7829fb7c5518421028046ebac6e418a1158adf6735**. TRAIN/CAL raw SHA256 **5ecef7ca8a3074965f8b335ff98891c66abdf4d5e489246379980aa5651ed8f7**. TEST data never entered coefficients, thresholds, source-world split or cost tariffs.

C08 untouched TEST raw SHA256 **c9a8e3671a7612eac3e98c3b861ea72871bd9d28ca931f87f18d25bc8a624340** generated only after C07A publication, then independently committed at C08A before scoring. From 24 independent parent worlds per family, 3 different computation trials each: **72 genuine Beta extra computations** and **72 genuine finite sensor extra computations**, exactly 144. Beta action flips28 (20 positive utility deltas and8 negative), mean signed gain +0.02580558, mean extra wall4.875μs. Finite sensor flips18 (15 positive and3 negative), mean signed +0.04862100, actual additional wall9.0556μs. Sign labels use true expected terminal utility under the generated latent world, not a one-off observed terminal reward.

## 3. Independent heldout scientific comparison

| Domain, n72 each | Constant q Brier | Two-part q Brier | Constant signed Z MSE | Two-part signed Z MSE | q-only conditional mean signed Z MSE |
|---|---:|---:|---:|---:|---:|
| Beta arms | 0.249491 | 0.235168 | 0.00865174 | 0.00849368 | 0.00840036 |
| Finite sensor | 0.188406 | 0.177516 | 0.02231252 | 0.02234827 | 0.02240917 |

Revision q Brier point estimates improved in both, but q ECE10 worsened vs constant (beta0.16538 vs0.10880, sensor0.06432 vs0.03009). Signed Z two-part MSE improves marginally for beta yet worsens marginally for sensor; beta's q-only conditional-mean comparator did BETTER than fitted two-part m. Therefore the conditional signed-effect regression's cross-domain efficacy is **NOT_ESTABLISHED**; calibration/coverage of predicted uncertainty NOT_ESTABLISHED.

At pre-registered λ=0.004 terminal utility / actual measured extra microsecond, net gain versus immediate STOP from each already-executed computation pathway:

| Shadow policy (computed offline, no actual stop) | Beta arms | Finite sensor |
|---|---:|---:|
| NEVER | 0.000000 | 0.000000 |
| ALWAYS | +0.006306 | +0.012399 |
| FLIP_ONLY q>=0.20 | +0.008750 | +0.021809 |
| SIGNED two-part positive predicted net | +0.011453 | +0.000296 |

Beta signed shadow computed49/72, skipped2 realised positive-net, computed31 realised negative-net. Parent bootstrap500 descriptive percentile95 interval signed-shadow minus ALWAYS: **[−0.000119,+0.012235]**, includes0. Finite sensor shadow computed20/72, skipped9 positive-net, computed14 negative-net. Interval **[−0.059044,+0.019485]**, includes0. The important empirical negative is that the exact same two-part model may improve the beta-arm point mean while destroying the finite sensor mean gain. Cost TRAIN medians also exceeded heldout actual measured source times, contributing to risk of over-conservative action selection (descriptive POSTTEST mechanism, NOT refitted). Price sensitivities [0.001,0.010] are archived in C08 full data, not posthoc chosen to justify efficacy.

NO physical hardware computation was skipped because all extra observations were actually executed to label the potential counterfactual outcomes; avoided microseconds and discounted policy score are retrospectively calculated only. A future prospective active intervention study is needed to measure actual runtime saving or terminal noninferiority. The true utility of C18 original native MCTS game move remains **NOT_ESTIMABLE** and has not been imputed or changed. Human/GM statistical signature P3, physical P2 speedup or game-strength claim not studied.

## 4. Verification, preservation, limitations

C5 actual TDD 5 source tests RED from missing behavior, GREEN5 after implementation. C6 native compiled Rust standalone program: source-only q/m/price and posterior initial margin, rejects source domain/true labels/future observations; TDD RED1, GREEN3, one diagnosed Rust inference compiler E0283 repaired. Fresh full Rust DEBUG and RELEASE each primary MCTS 617 PASS, C17 CLI7, C18 CLI3, C19 CLI3 =630 PASS/0 FAIL; main89 ignored. Python focused C19 tests5 PASS. Actual compiled Rust model vs Python on96 source-only inputs: 480 float equalities, max abs error1.11e−16; on full144 heldout root inputs max abs error8.33e−17, no active SearchPolicy mutation. C19 compiled shadow release SHA256 **6761476e1901926f85ba3c0410f1221d34834d97c219f90a88f5f5f92bd58f68**.

C9 truly independent Python auditor imported neither source collector nor fit/scorer and replayed **all 144** actual source RNG observation sequences, posterior, initial/final decisions, utility labels, model score and all3 price scenarios; PASS. Actual C08 Git tar SHA256 **125f0c028c7ff458b3911dffd1784aca6836c85ae730447bafd238c365ccc25d** independently restored to temp, exact SHA byte match on13 key original/source/model/raw/report files, PASS. Git publish/remote equality and actual file restore distinct from Dropbox cloud sync ACK, which is NOT_VERIFIED. Native subagent NOT_RUN (serial fallback).

Science classification: literature-backed expected utility/EVSI, independently derived 2-part conditional bounds and exact finite examples, implementation-verified source-only Rust/Python models and 720 real extra stochastic calculations across both known-utility families (576 TRAIN/CAL +144 TEST), independent source-group audits PASS; general objective prediction/calibration and actual policy efficiency NOT_ESTABLISHED. No PR or merge to main, original developer/C16/C17/C18 frozen branches left untouched.

## 5. Next research plan and explicit ID

**QUARTZ-GENERAL-20-POSTERIOR-PREDICTIVE-VOC-AND-COST**. The C19 failure is scientifically meaningful: the coarse initial action-utility gap and action count do not suffice for a transferable signed-value forecast; cost predictor based on train source medians can be biased by hardware timing distributions. Instead of another heuristic threshold or sample gate, construct an observation-channel-aware **task-neutral Bayesian predictive value-of-computation reference** that integrates possible observations and subsequent Bayes terminal decisions, with independently measured compute resource prices. This retains project generality and does not embed game tactics in the shared controller.

Next R1–R10 must explicitly derive sufficient-statistic predictive value for both the independent Beta-Binomial arm model and finite hidden-state noisy sensor model, test exact EVSI vs heuristic q*m, include model uncertainty and group-dependence bounds. Then C1–C10 implement exact/independently checked generic posterior-updating interface, finite predictive integrations, code/Rust parity, pre-frozen source/heldout real computational experiments, scientific/efficiency and negative failure ledger, and stagewise commit+local SHA Git tar+remote provider readback. Do NOT activate real game MCTS SearchPolicy until independent exact game utility label and model transport are verified. C20 is NOT EXECUTED within this bounded C19 cycle.

Exact files here: theory/R01–R10, coding/, SOURCE_WORLDS_FROZEN.json, SHADOW_SIGNED_MODEL_FROZEN.json, evidence/C07_TRAIN_CAL/, evidence/C08_TEST/, evidence/C09_INDEPENDENT_AUDIT.json, evidence/C09_LOCAL_RESTORE.json, C10_FINAL_RELEASE_MANIFEST.json, NEXT_STAGE.json.