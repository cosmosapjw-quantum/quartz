# QUARTZ COG-03 EINSTELLUNG-REOPENING-12 | 실제 CPU 루트 후보 재개방 연구 결과

Date: 2026-11-21 (Asia/Seoul). Parent commit `6dbdcaeab28b2325fbd7f0a0326c51b214b339f3`. Isolated research branch `research/quartz-cog03-einstellung-reopening-20261121`.

**종료 판정:** 실제 Rust release Connect Four MCTS에서 초기 root 후보 상위 3개만 허용한 *조작적 후보 고착*을 구성하고, 외부 정답 없이 한 수 앞 전술 증거로 누락 후보를 재개방했다. 새로 고정한 24개 opening-family/384 root에서 trained MLP와 ConvNet evaluator는 동일 방식으로 재시작한 후보-제한 대조군과 세 무작위 재개방군 대비 낮은 정확 WDL regret를 보였다. **그러나 unrestricted/root-all-legal 대조군의 regret가 오히려 수치상 더 낮으며, 본 실험은 인간의 Einstellung 메커니즘이나 일반적인 엔진 우위의 입증이 아니다.** 단계별 코드/증거는 실제 게시·백업되었다.

## 1. 정의, 원전과 측정가능한 가설

표준 7×6 gravity Connect Four의 합법 root 착수를 `A(s)`, 실제 CPU evaluator가 출력한 root prior를 `p_E(a|s)`라 한다. Prior 내림차순, 값 동률이면 중앙 column 3과의 거리 오름차순, 다음 column 오름차순으로 순위를 고정한다. 초기에 `A0(s)`는 상위 3개 legal root 후보, `A_out=A\\A0`. 이 root만 legal 후보를 가린 새 Connect4 GameState를 만들고 실제 MCTS 8 visits를 실행한다. **자식 state는 기존 모든 합법 수로 복원되며 원래 TT hash를 사용한다.** masked root만 별도 TT hash를 갖는다.

`π_8(a)`는 실제 첫 탐색 root 방문분포. Fixation 조건 `F=max_a π_8(a) >= .5`, 초기 선택 `a8`. 원본 합법 게임의 자체 규칙만으로 `w(a)=is_winning_move(s,a)` 및 `v(a)=exists opponent legal reply b such that is_winning_move(s after a,b)` 계산. 온라인 재개방 조건은 (i) F이고 초기 A0 밖 즉시 승리착수가 있을 때 이를 최대 2개 추가, 또는 (ii) F이고 초기 a8이 상대 즉시 승리를 허용하며 A0 밖에 그런 응수를 피하는 착수가 있을 때 원래 prior가 높은 순서로 최대2개 추가. **Pascal Pons external complete game exact action-score는 온라인에 없음**.

이후 모든 비교군은 첫 tree를 폐기하고, 새로운 root tree에서 정확히 16 visits를 검색한다. 두 root expansion/setup, 사전확률 생성과 전술 probe의 wall-time을 실제 합산한다. **현재 구현은 same-tree root edge insertion이 아니고 8+16 두 개의 독립 검색**이다. Controls: `closed_sham` 동일 tactical probe하지만 열지 않음; `evidence_reopen`; `random_1/2/3` 동일 개수만큼 deterministic oracle-blind hashing으로 열기; `always_full_reopen` 모든 누락 후보 열기; `plain_full` 처음부터 모든 후보 허용; `closed_no_probe` probe overhead 확인. Fixed seeds `202611210201/202611210202/202611210203`은 candidate 배정값이지 MCTS의 독립 PRNG replicate가 아니다.

실제 게임이론적 label은 offline Pascal Pons solver의 legal action score `S(s,a)`; illegal sentinel -1000을 제외하고 `u(s,a)=sgn S(s,a)∈{-1,0,+1}`; `R(s,a)=max_legal u(s,a)-u(s,a)` (regret는 낮을수록 좋음). 'strictly decisive omitted'은 `a∈A_out` 중 `u(s,a)>max_{b∈A0}u(s,b)`인 행동이며, oracle-only POSTHOC 진단이다. 효과는 실제 **재개방된 결정적 후보 수, 최종 선택이 개선된 국면 수, 정확 WDL regret**로 평가한다.

원전 및 novelty boundary: Bilalić, McLeod, Gobet (2008), *Why good thoughts block better ones*, https://doi.org/10.1016/j.cognition.2008.05.005 ; Bilalić et al. (2010), https://doi.org/10.1177/0963721410363571 ; Sheridan & Reingold (2013), https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075796. 이는 인간 체스의 시각적 주의/고착 연구이며, 본 MCTS search control과 같은 인지 기제를 검증해 주지 않는다. Progressive unpruning/widening은 기존 기법이며 본 root gate 자체를 새 일반 알고리즘으로 주장하지 않는다.

## 2. Immutable inputs, 선행조건과 byte identity

- `CONTRACT_KO.md`, `INPUT_LOCK.json` 18개 원본 Source/engine/model/book SHA256. Baseline locking commit **`19786f5bd8b88ced0804feeda822a6d78ca356dc`**. 신규 cohort 생성 전 thresholds `TOP_K=3`, `F>=0.5`, `MAX_REOPEN=2`, first8+fresh16, 8 arms, random seeds 및 cost admission <=10% 모두 고정.
- 새 `COHORT_384_FROZEN.json`: SHA256 **`31e86dcda897a540f7914ee123e929e30dadf8a8e93117f98c8cd137abed380a`**. 24 mirror-distinct 5-ply families, 각16 descendant=384 roots, `reopen_A/reopen_B` 각12 families/192 roots. Black/white-to-move 각 그룹 96/96. G07 (train/val/test), C06, old12, C09,C10,C11의 root opening-family prefix 및 board state를 좌우 반전까지 제외. 원본 모델은 C07의 실제 **trained** MLP, ConvNet plus Uniform evaluator.
- 외부 Pascal Pons action-score `evidence/new_384_exact_20261121/EXACT_NEW384_LABELS.json`: SHA256 **`0e018e813e77fc2e0a25688824c63b9adb10d4a9fc047dcdad92a72b48d78012`**. 384/384 OK, 32.81s bounded CPU, 0 timeout/illegal labels. 24 newly solved representatives plus horizontal mirrored 24 independently re-solved before experiment, another 24+24 after. `reopen_A` exact root WDL win138/draw7/loss47, `reopen_B` win123/draw8/loss61. Decision-informative root positions **240/384** (=126 A,114 B); rest have legal moves all same exact WDL.
- Optimized release Rust executable source commit **`afdc426d8062f227c8affc3d571fd2bc42d1bb36`**, binary SHA256 **`af4c64eaad0ff325d390d30a661bb292d090a80709cd1ec048964c0d54a07a8e`**. Actual Python test runner committed source SHA **`f9b7caa45a420f09f1ed7665c827b8e6a6c8ea6e`**, before any new 384 root exact target outcomes joined.
- External solver code/binary/book remain outside Git repository, not redistributed. Compiler Rust `cargo --offline --locked --release`, one CPU host and thread per root, shared process for model weight cache. All timing is **elapsed wall-clock**, not CPU process-time or hard per-simulation time limit.

## 3. Real Rust experiment, 9216 actual endpoint requests

Test count: `384 roots ×3 evaluators ×8 experimental arms =9216` real Rust research server requests, with **two independently constructed and searched roots** each: 8 first visits +16 second visits, total **18,432 actual MCTS root search executions**, ~8.765s warm-server batch. No reuse of first tree in second; compare `root_evaluator_calls`, prior evaluation, both new tree setups, prior-probe, total end-to-end CPU decision wall-time. All 384 exact solver labels read **after completing all 9216 Rust requests**.

Main pooled results from `evidence/live_9216_20261121/SUMMARY_REOPEN12.json`:

| Evaluator | Closed top3 regret | Evidence-reopen regret | All legal root regret | True probe triggers | Omitted decisive reopened |
|---|---:|---:|---:|---:|---:|
| Uniform | .35677083 | .35677083 | **.23177083** | 30/384 | **0** |
| Trained new MLP | .2578125 | **.15364583** | **.13020833** | 79/384 | **23** |
| Trained Conv | .2109375 | **.140625** | **.1171875** | 54/384 | **16** |

Every evaluator and both new 12-family groups met **end-to-end mean wall-time relative mismatch ≤10%** for evidence_reopen vs closed_sham. Pooled end-to-end wall-ms: Uniform evidence **.41011**, closed **.42945** (4.71%); MLP evidence **.46239**, closed **.46824** (1.26%); Conv evidence **.86753**, closed **.91366** (5.32%). Unlike COG02 signed-gain tests, this resource metric includes initial full prior inference, both root setup costs, both actual search stages, one-ply probe (except explicit no-probe control), but excludes model-load time. Probe costs and measured residual OS jitter are in raw evidence; not asserting deterministic per-root time equality.

### MLP: reproducible genuine candidate recovery within engineered gate

Pooled evidence vs closed exact-WDL regret difference **−.10416667**, 24-opening-family fixed-policy paired bootstrap 95% **[−.1484375,−.06510417]**. Original prior top3 omitted strictly better exact-WDL actions in **38/384** roots. Evidence probe actually reintroduced such actions in **23/38**; final chosen action WDL improved on **21/384** roots, worsened **0**, unchanged 363. By new group: A evidence .177083 vs closed .255208 (37 triggers, 11 decisively reopened, 8 improved selected); B evidence .130208 vs closed .260417 (42 triggers, 12 decisively reopened,13 improved selected). Both groups separately had conditional bootstrap intervals excluding zero vs sham.

MLP vs each oracle-blind equal-size random-reopening arm pooled regret: `random_1=.21875, random_2=.223958, random_3=.226563`. Evidence−random regret differences respectively **−.065104, −.070313, −.072917**, 24-family bootstrap 95% **[−.1015625,−.03125]**, **[−.10677083,−.03645833]**, **[−.11979167,−.03385417]**. All means actual CPU gate passed ≤10%, showing the evidence-specific targeting does better than selecting as many alternatives at random **in this bounded experiment**.

### ConvNet: positive but lower retrieval coverage

Pooled evidence−closed regret **−.0703125**, 24-family bootstrap 95% **[−.10677083,−.03645833]**. **30/384** roots have omitted superior exact-WDL move; evidence opens **16/30**, final selected WDL improves in **15/384**, worsens0. Group A evidence .166667 vs closed .255208 (27 triggers,9 reopened decisive,10 WDL-improved); B evidence .114583 vs closed .166667 (27 triggers,7 reopened decisive,5 improved). Group effects same sign vs closed. Random controls pooled regrets `random_1=.192708, random_2=.182292, random_3=.184896`; evidence−random effects **−.052083,−.041667,−.044271**, corresponding pooled 95% intervals **[−.08072917,−.0234375]**, **[−.06510417,−.01822917]**, **[−.07291667,−.01822917]**. All real CPU means matched within 10%.

### Uniform negative control and stronger general baseline

Uniform has 49 roots where the original top3 omitted a strictly better exact-WDL move, but observed first-root fixation/probe trigger reopens **0/49 decisively better actions**. Regret evidence=.35677 equals closed=.35677, no game move WDL improvement. Full/unrestricted root provides **.23177** at comparable mean CPU timing, a materially better *descriptive* result.

Crucial limitation for both NN modes: **always_full_reopen** and **plain_full** allow all legal root actions and attain **MLP regret .130208**, **Conv regret .117188**, better than the selective tactical reopening (.153646 and .140625). These differences are small and the paired family bootstrap interval for evidence−full **includes zero**, so not a confirmed full-superiority theorem; however **the data do not support selective reopening over simply searching all legal candidates**. Root legal branching is ≤7, so pruning top3 is an aggressive artificial manipulation rather than a prerequisite of efficient 7-column search. The real scientific contribution here is recovery of preemptively suppressed, low-prior moves under a fixed attention gate, not superior unrestricted-game play or human cognitive architecture.

## 4. Verification: actual code, independent model/solver, and error taxonomy

- Root-only `Connect4::with_root_candidates`, root gate isolated TT key, child full legal moves preserved; TDD targeted RED→GREEN. Research Rust `handle_research_connect4_reopen_cpu` actual serial MCTS first8 and fresh16, exact added candidate actions, prior/cache/first+second costs. Bad/illegal arms and oracle injection rejected.
- Pure rules `mcts/reopening12.rs` one-ply current-win/opponent-win candidate probe, deterministic fixed original prior ties and oracle-blind random controls. Runtime Rust endpoint schema always states `MCTS_restart_instead_of_same_tree_reopening=true`.
- `audit_oracle12.py`: 384 exact board/column WDL sources +24+24 Pons re-solve. `audit_live12.py`: **independently reimplemented Connect Four move history replay, immediate-win geometry, opponent immediate-response vulnerability, fixture-free root prior ranking, all live random-opening seed hashes, full action WDL/utilities, root search timing, model cache, and 24-family bootstrap**. All 9216 Rust responses matched; another 24 originals and 24 mirrored solver queries also matched. Evidence `evidence/live_9216_20261121/{RAW_9216_RELEASE_ROOT_RESTARTS.json,PAIRED_384_WDL_ATTENTION.json,SUMMARY_REOPEN12.json,EXECUTION_MANIFEST.json,INDEPENDENT_READBACK.json,VERIFICATION_RECEIPT.json}`.
- Actual fresh Rust debug `cargo test --offline --locked -- --test-threads=4`: **601 PASS, 0 FAIL, 89 ignored**. Fresh Rust release `cargo test --offline --locked --release -- --test-threads=4`: **601 PASS, 0 FAIL, 89 ignored**. Related Python suites **123 PASS, 0 FAIL**, including 7 new COG03 tests. All source file and generated-result byte SHA identities, live source commit and compiled binary source commit in manifests.
- Failure classification: artificial candidate narrowing is an explicit *experimental intervention*; MLP/Conv candidate re-opening gives actual verified finite-sample outcome gain conditional on this intervention, while Uniform shows a genuine negative result; no whole-game minimax inference from one-ply threat score; release wall-time noise and restart overhead are runtime/implementation constraints; insufficient evidence to infer stronger general engine, human cognitive mechanisms, or a novel unpruning algorithm. Native subagents **`NOT_RUN`** (not available in current tool inventory).

## 5. Provenance: every bounded stage archived and remotely verified

All commits to dedicated branch `research/quartz-cog03-einstellung-reopening-20261121`: per unit `git commit`, local Dropbox `git archive HEAD | gzip`, archive `sha256sum`, GitHub `git push`, `git ls-remote` equals local exact HEAD. Local archive directory `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`.

- A **`19786f5bd8b88ced0804feeda822a6d78ca356dc`**: SSOT contract and original eighteen file identities frozen before new data.
- B1 **`9403123507eac5b864be76e3331210a54e054ad1`**: root-only legal gate and TT identity.
- B2 **`afdc426d8062f227c8affc3d571fd2bc42d1bb36`**: actual Rust endpoint, tactical probe, 8+16 restarted MCTS and unit/full tests, optimized binary SHA.
- C0 **`a4a523b201914ca3339388952034d2ad3bbddc99`**: new two-cohort 384 roots and old-board mirror disjointness.
- C1 **`33b7be4d758a1ced7b6c5733436bd50e5a7d5064`**: bounded exact teacher wrapper and tests.
- C2 **`7da57fb5d943c946d75c1b6569b6585fe341430c`**: 384 exact oracle outcomes + independent mirror verification.
- D0 **`f9b7caa45a420f09f1ed7665c827b8e6a6c8ea6e`**: 8-arm real Rust runner and scoring fixed before execution.
- D1 **`b5e5e2be401a160c5683775dda28752ff0644f2a`**: raw 9216 actual CPU experiments, independent tactics/WDL/source audit, test receipt.
- E: this final report/next-stage handoff, separately published and hashed.

Developer main checkout `/home/cosmosapjw/Dropbox/personal_projects/quartz` on `agent/local-experiment-foundry`, HEAD `65e7d3e9596ca259ebfe523b0c74058692b6a755` and its pre-existing untracked `QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md`, remain untouched. External Pons game solver/book not redistributed. **Local Dropbox file+hash is verified, Dropbox cloud sync confirmation or restore test NOT_RUN.**

## 6. Next bounded research ID: QUARTZ-COG-03-RESIDENT-REOPENING-13

**Move to actual resident-tree root unpruning rather than another elaborate validation meta-loop.** Current experiment demonstrates that selective evidence can partly recover artificially omitted candidates, but simply leaving all seven columns eligible is descriptively better. A meaningful next physical research question is whether **real same-tree root candidate reactivation** can preserve the first 8 useful MCTS visits and recover superior decisions over an *all-legal* baseline under a truly matched CPU budget, or whether there is no advantage to root pruning at Connect4 branching factor7.

1. Implement minimal root edge activation in actual MCTS Arena/edge materialization that reuses existing root visits and counters without reinitializing tree, avoids transposition-cache contamination and tests child legality/NN model prior identity. Do not claim same-tree before verifying actual visited-root prefixes and evaluator-call preservation.
2. Use a **fresh**, independently sealed Connect4 family test and exact external WDL only posthoc, compare dynamic reopening with root-all-legal unpruned MCTS, sham-restricted root and preexisting single-step policies, on actual end-to-end CPU including probe and any policy overhead. Pre-fix the cost/evaluator budget, and verify ≤10% within each group; regression test root actions and WDL benefit and crucial root candidate omission coverage.
3. On negative result, stop trying to make 7-column root pruning useful and prioritize a **separate transfer study to original 15×15 Gomoku or Renju** using credible publicly available pretrained engines (e.g. Rapfi with verified file identity, rules, license, protocol and actual CPU budget). Connect4 oracle score definitions and standard gravity semantics do NOT carry over to Renju forbidden-move rules; freeze each exact ruleset and an independent competence benchmark.
4. Return to original constructive cognitive question: does low-prior candidate attention reopen *new crucial information* under computational constraint in a real branching regime, versus already-normal exploration? A same-tree real branch intervention and matched CPU plus stronger genuine pretrained engine would be more scientifically meaningful than more meta-audits.

**Stop:** C12 study fully scoped and completed upon publishing this report. Human Einstellung, novel cognitive mechanism, a new Bayesian VOC, generally stronger MCTS play **remain unresolved**. `native_subagent: NOT_RUN`.
