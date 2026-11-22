# QUARTZ C20 최종 연구 보고 및 C21 인계

**노드:** QUARTZ-GENERAL-20-POSTERIOR-PREDICTIVE-VOC-AND-COST. 2026-11-22 KST. 부모 C19 SHA 2a359ad049b6a0be4047e52ad281b665cf68e967. 독립 브랜치 research/quartz-general-c20-predictive-voc-20261122. 원본 docs/THESIS.md P1–P4에 따른 연구 목적은 **범용 제한합리성 계산 선택, 배분, 중단**이지 특정 게임의 승률/기력 향상이 아니다. 본 연구는 베이지안 관측채널에 따른 계산가치와 이를 계산하는 추가 CPU 자원비용을 직접 연결했다. C19 실제 neural MCTS signed utility 미정 상태와 원래 THESIS/CLAIM_LEDGER 동결을 보존했다.

## 1. 이론 R01–R10 수행: 증명 가능한 식과 모델의 전제

이론 10단계 각각 개별 Git commit, SHA-256 Git tar archive, origin push/remote SHA 확인 후 다음 단계로 진행했다. R01 원본/질문, R02 1차 문헌 대비, R03 정확한 posterior predictive observation operator, R04 경쟁 구성, R05 EVSI/EVPI 증명·단위·복잡도, R06 관측 상관/likelihood/원가 반례, R07 Wolfram/독립 Python Fraction 정확 검산, R08 claim gate, R09 구현 계약, R10 코딩 인계.

분포 b의 현재 최적 결정 가치 M(b)=max_a E_b U(a,theta), 값이 지불되는 계산 c의 posterior predictive observation y 이후 M(b^y)로부터 **G_c(b)=E_{y|b,c} M(b^y)-M(b)**. 올바른 observation channel, 동일한 완전한 행동 집합, 최적 posterior 행동과 동일 payoff를 가정할 때 tower property와 max convexity에 의해 **0<=G_c<=EVPI=E_b max_a U(a,theta)-M(b)**. 물리적 resource price lambda utility/μs에서 이상적 한 단계 net=G−lambda E[C_task]; evaluator 자체 비용을 지불하는 현실적 net=G−lambda(E[C_task]+E[C_evaluator]). 개별 actual Z=U(a1,theta)-U(a0,theta)는 음수 가능.

Beta arm model: 초기3 Bernoulli/arm posterior Beta(a_j,b_j), 뒤따르는10/arm 결과 성공횟수 k_j의 PMF binom(10,k)*B(a_j+k,b_j+10−k)/B(a_j,b_j). 독립 arm 간 확률곱을 취하되 동일 arm 반복 관측은 같은 theta를 공유하는 Beta-binomial, unconditionally independent marginal Bernoulli로 오용하지 않음. K=2/3/4 각 11^K full count outcomes. Sensor finite4-state, 3 symbols, n=8 추가 관측의 45 multinomial count vectors는 **fixed theta mixture of multinomials**로 통합, posterior optimal terminal action expected payoff에 적용. Beta model 전체 Rust 구현은 이번 노드에서 미실행, finite sensor generic state-channel Rust 구현만 존재한다.

Live Wolfram+independent Python Fraction: binary fair-state 3/4 accurate sensor single observation EVSI=1/4; two independent Beta(1,1) arms 1 draw each EVSI=1/12, Beta(1,1) n2 counts each 1/3; shared latent observation correlation witness PASS. 이 식은 기존 베이지안 의사결정 이론의 재구성으로, 일반적인 새 EVSI 정리 또는 신경망 MCTS의 보정 증거가 아니다.

주요 출처: Hay/Russell/Tolpin/Shimony UAI2012 https://www.auai.org/uai2012/papers/123.pdf ; Callaway et al UAI2018 https://proceedings.mlr.press/r16/callaway18a.html ; Sezener/Dayan UAI2020 https://proceedings.mlr.press/v124/sezener20a.html ; Hironaka et al MLMC EVSI https://arxiv.org/abs/1909.00549 ; Fleischhacker et al finite multinomial EVSI https://arxiv.org/abs/2301.00729 .

## 2. 실제 구현 C01–C10 및 독립 prospective source

C01 source lock, C02 generic expected-utility operator, C03 numeric/cost contract, C04 NEW 240 independent source-world parent bank, C05 Python full exact Beta-binomial and fixed-theta multinomial mixture predictor with real TDD RED6→GREEN6, C06 compiled independent Rust finite-sensor observation-channel numerical predictor test RED1→GREEN3 and 96 sensor model parity, C07 source/fit/scoring runner Git publication BEFORE labels, extra C07A TRAIN/CAL raw and TRAIN-only cost model publication, extra C08A untouched TEST raw publication, C08 model/value/cost compared at test, C09 independent raw sequence/replay and exact-math audit, archive restore and timing, C10 final handoff/prototype.

**Source identity:** new SOURCE_WORLDS_C20_FROZEN.json SHA256 d6cf96c26d9ac913963f702d9b0418f0002038979f6a470da3865999368e4ae7, 120 distinct parent latent worlds per family, initial source seeds distinct from C19, predetermined TRAIN72/CAL24/TEST24 each. Three actual additional sample observation trajectories per parent = **720 physically executed stochastic extra computations** in total. TRAIN+CAL576 paths (192 independent parent worlds), TEST144 (48 independent worlds). Beta: true arm means sampled from Beta(2,2), online posterior matched to prior, initial3 and extra10 Bernoulli observations per arm. Sensor: four latent states, public prior+channel+4x3 utility, initial2 and extra8 actual categorical sensor queries, posterior Bayes updates. True mean/hidden state only offline for verifying terminal **expected true payoff**, not a single realized terminal reward or runtime predictor input. Typed source adapter strips hidden theta, extra draws, future chosen actions and measured future cost before calling exact EVSI or C19 coarse predictor.

Provenance wrinkle: C04 reused C19 generator with new independent seeds but inherited the JSON metadata key source_parent_C18 as a **legacy template label**. That key is not current C20 Git lineage. Actual parent = C19 exact SHA given above, C01 Git lock; immutable latent source bytes and splits retained rather than silently replacing them. This is a representation defect, not evidence of reused C19 test. No original developer branch/main/previous C16–C19 worktrees or original frozen THESIS modified.

C07 code frozen/pushed Git620df8133d416dff9bfaaf94761e358172ddf082, then C07A TRAIN resource model frozen/pushed Git39f820d7105e104f93e4a235d889bdc3df7f3512 with SHA256 **530ca204c96b5fda4f3b060e8ebd87b52e4c4ec7e97be42dac56da93ca81cb04**, TEST only generated after this. TRAIN/CAL source raw SHA256 **009774dd99f1e2034e74f449c74a13f3470731737725c60fd7e7a142cfdf4a9e**. C08A raw TEST sealed/pushed Git a331f8bfb3a94fa9cb331d6fb604eeec4d05e706 SHA256 **5ab524aff074d8f441f09679309807079c694035eded182d86c5517d82cf5c1f**, then scored with unchanged frozen C08 script.

## 3. 독립 TEST에서 계산가치 예측

| 신규 TEST (n72 each) | Historical frozen C19 coarse signed gain MSE | C20 exact posterior predictive EVSI MSE | C20 − C19 MSE 500 parent bootstrap 95% |
|---|---:|---:|---|
| Beta(2,2) arms | 0.01411918 | **0.01381147** | [-0.00195766, +0.00105184] |
| 4-state finite sensor | 0.01119080 | **0.01068388** | [-0.00285884, +0.00146938] |

Both exact model MSE point estimates improve on the frozen historical coarse C19 predictor, but 24-independent-parent bootstraps include0, so **general or statistically decisive superiority NOT_ESTABLISHED**. TEST Beta mean true signed Z +0.03883327 vs exact expected G+0.04716533; TEST sensor mean true signed Z +0.00304986 vs exact expected G+0.02775210. These are finite realized-source deviations; exact E[G|current posterior] does not force each finite heldout sample to match. No posterior model shift was introduced; all true tasks generated from matching declared laws. C19 model not refitted on TEST.

## 4. Meta-evaluation price: mathematical correctness vs actual CPU value

Actual C07A TRAIN medians of **underlying extra computation C_task / exact Python EVSI evaluator C_eval / old C19 model evaluator**, microseconds:
- Beta K2: 3 / 113.20 / 36.32
- Beta K3: 4 / 1083.13 / 40.24
- Beta K4: 5 / 12770.69 / 95.66
- finite sensor: 8 / 216.71 / 37.53

In NEW TEST, mean measured task extra wall Beta8.722 μs, exact Python evaluator4621.978 μs; sensor task16.958 μs, exact Python evaluator214.982 μs. Actual clock jitter/source/hardware environment and cost-model drift must be preserved, not assumed constant. Predeclared lambda=0.004 terminal utility per microsecond; full TEST paired counterfactual mean net vs no extra computation:

| Policy | Beta arms | Finite sensor |
|---|---:|---:|
| NEVER | 0 | 0 |
| ALWAYS compute without metalevel evaluator | +0.003944 | -0.064783 |
| Frozen C19 coarse shadow, ignore evaluator overhead | +0.009159 | -0.019075 |
| C20 exact shadow, ignore evaluator overhead | +0.009159 | -0.006020 |
| Frozen C19 coarse shadow **charged evaluator overhead** | -0.190434 | -0.149241 |
| C20 exact shadow **charged actual Python evaluator overhead** | **-18.487910** | **-0.859928** |

At all prespecified prices0.001,0.004,0.010, charged C20 shadow chose **STOP on all72** examples in both families under frozen TRAIN median evaluation costs, but it still paid the expensive exact evaluator for each decision; no computation was physically stopped because all task followups were executed to collect outcomes. This is rigorous **counterfactual shadow accounting**, not measured workload savings, and negative net is a meaningful implementation cost finding. Primary priced charged C20−ALWAYS bootstrap500 parent intervals Beta[-28.423,-9.736], sensor[-0.818,-0.762], both negative under these particular measured CPU and price conventions. Does not prove information acquisition undesirable, only this implementation of metareasoning has unacceptable overhead.

## 5. Independent verification and constructive C10 fast algorithm

Rust DEBUG+RELEASE full regression independently run post-C08: existing MCTS617 pass (89 ignored), C17 CLI7, C18 CLI3, C19 CLI3, new C20 CLI3 = **633 pass /0 fail in EACH** configuration. Python exact EVSI tests6 PASS plus typed adapter2 PASS. Compiled Rust finite state reference on TEST n72 initial-only cases vs Python max abs numeric difference **6.66e-16**, no Beta Rust port or SearchPolicy hook claimed. C09 separate Python auditor did NOT import production generator/predictor/scorer; replays all144 actual TEST stochastic paths/terminal actions and true utility, independently computes Beta Polya-urn predictive EVSI for24 paths and full ordered 3^8 sensor sequences for24 parent inputs. Max independent EVSI error **3.10e-14**; paired priced policy MSE/net values across3 fixed lambdas independently reproduced. C08 local Git archive SHA256 **e2c8041c978fc5a54cc284d835b9a5f49d2a549b64b866a381889162a69f9114** restored in new temp location with **15 source/metric/code file hashes byte identical**; Dropbox cloud synchronization not independently acknowledged.

Important separately measured native path: compiled standalone Rust **finite-sensor** EVSI 72 initial-only public input cases×3 repeats =216 actual Rust function invocations, internal Rust Instant median **2.7255 μs**, mean **2.96583 μs**, max8.947 μs, compared to Python exact sensor mean ~215 μs. This is **posttest exploratory native-only CPU evidence**, excludes Rust process launch, JSON I/O, Python adapter and actual PUCT/SearchPolicy integration. Thus the Python infeasibility is **implementation dependent**, not a no-go for exact metareasoning.

**New constructive C10 result on TRAIN only, not retrofitted to C20 TEST:** independent posterior Beta-binomial action score variables X_j=(alpha_j+k_j)/(alpha_j+beta_j+n) have independent discrete distributions F_j over possible post-computation posterior scores. Consequently CDF(max_j X_j at most v)=product_j F_j(v), and E(max_j X_j)=sum over ordered *unique score support* v of v*(F_max(v)-F_max(v_previous)). This converts an 11^K joint count enumeration into a sweep of at most K×11 score support points, without changing G mathematically. Implemented an **independent Polya-urn PMF + exact rational support** CDF-of-maximum proof-of-concept coding/c10_fast_beta_cdf_prototype.py, compared on **216 TRAIN-only initial models**, max abs EVSI disagreement **2.998e-15**. Measured Python median speed ratio of old full-grid vs CDF: K2 **0.94x (CDF slightly slower)**, K3 **6.17x**, K4 **53.95x**. K4 full-grid median12663.66 μs vs new CDF median235.16 μs. This is a bounded correct model construction with tested integer Beta shapes, not an independently tested C20 TEST policy, and still slower than raw 5 μs task at K4. Rust port and full mixed-language overhead remain outstanding.

The user-requested research stage publish contract applies to R01–R10 C01–C10 with extra C07A/C08A, 22 separate Git commits/local hash tar backups; C10 and final postrelease attestation must be verified by actual remote SHA and restore before declaring completed. The published source files contain full exact version identities and traces. No main merge, PR, artificial native subagents (NOT_RUN), real game-utility, P2 CPU savings/P3 human search or P4 production hotpath PASS claim.

## 6. Next independent full cycle

**QUARTZ-GENERAL-21-CDF-EXACT-EVSI-NATIVE-RESOURCE-GATE**. Rather than adding another heuristic or 26 new validation gates, prove and implement the exact independent-max CDF sweep for arbitrary proper Beta shapes, repeated support degeneracy and larger K with stable probabilities, then port it to Rust with tight TDD and independently cross-check against Fraction/Wolfram, full-grid Python and source-specific extra sampling. Benchmark **full predictor end-to-end overhead** including computation, state adapter/serialization and actual task resource cost. Integrate a read-only generic meta-action proposal interface into real Rust MCTS only after observation likelihood/true utility prerequisites exist; still no LIVE STOP or neural-MCTS game performance claim based on toy tasks. Require two genuinely independent known-utility task sources and prospective TRAIN/CAL/TEST before any policy efficiency or no-harm statement. Run R01–R10 theory followed by C01–C10 coding as a separate future authorized cycle, stagewise Git SHA backup and remote receipts. C21 has NOT been executed in C20.

See NEXT_STAGE.json for typed exact stage handoff, C10_FINAL_RELEASE_MANIFEST.json for source/commit/test/archive identities, theory/, coding/, evidence/C07_TRAIN_CAL, evidence/C08_TEST, evidence/C09_*, evidence/C10_FAST_BETA_MAX_CDF_WITNESS.json.