# QUARTZ-COG-02-DYNAMIC-OPTION-COSTMATCH-11 | prospective myopic mixture vs nonmyopic option

2026-11-21 Asia/Seoul. Parent HEAD **`85a08755e2463538ed0abf84d443f19dacc29e02`**. New isolated branch **`research/quartz-cog02-dynamic-option-costmatch-20261121`**.

**Scientific verdict:** exact 7×6 Connect Four external-oracle WDL and true in-process Rust release MCTS show a genuine finite empirical **nonmyopic two-step selection mechanism**, but **independent playing-decision superiority over prospectively randomized one-step policies at matched CPU cost is NOT ESTABLISHED**. In three predeclared balanced assignment streams, the regret advantage is only 0 to 1/384, and the 24-opening-family conditional confidence intervals contain zero. This bounded negative result closes the extra cost-matching loop; neither a universal no-go theorem nor a new Bayesian KG algorithm is claimed.

## 1. Claim, assumptions and transport

Standard gravity Connect Four 7×6, 4-in-row, seed17 MCTS, Dirichlet off, serial CPU release build. External exact Pascal Pons action score `S(s,a)`; illegal column sentinel `-1000`. For legal a, `u(s,a)=sgn S(s,a)∈{-1,0,+1}`, `Regret(s,a)=max_legal u(s,a)-u(s,a)`. The observed pairwise per-root regret difference is `R_D-R_M`; **negative favors the dynamic two-step policy**. The sole physical resource metric used for matched-cost gate is actual **MCTS stage search wall time in milliseconds**, not visit count, root GPU time, evaluator-call count, or elapsed process CPU time. Full request wall time, setup and model caching were also recorded separately.

The previously fitted nonmyopic model `D` is the C10 9-feature Ridge signed realized-gain surrogate with a `commit_two` option; it is **not Bayesian preposterior expected value of computation**. For the same nine current-root features, the myopic baseline has joint-gain coefficients identically zero and uses the **same** one-step Ridge gain/cost predictions. Only `λ` differs between two predeclared, fixed candidate models. No exact solver action value, future MCTS checkpoint or WDL label enters runtime decision.

The C10 abstract XOR information-synergy result is *derived exactly* but does **not** prove that C11's game policy can exploit such synergy better than a matched online myopic policy. Prior relevant literature: Hay et al. (2012), https://proceedings.mlr.press/r10/hay12a.html ; Callaway et al. (2018), https://proceedings.mlr.press/r16/callaway18a.html ; Sezener & Dayan (2020), https://proceedings.mlr.press/v124/sezener20a.html. This is a bounded reproducible experiment, not a precedence or novelty claim.

## 2. Locked SSOT and genuinely prospective comparison

**A first, before any new C11 family exists.** The validation source is only previous G07 40 train families ×16=640 roots and 12 validation families ×16=192 roots, previously used to freeze C10 weights. We reused the unchanged frozen C10 policies, SHA256 **`112acbaa6e31be7cd49ba8ad6347daa191059b31056439554d956d8828e123ad`**. Uniform validation adaptive mean search time is **0.17375ms**. Existing one-step λ=1 costs **0.165255208333ms**, λ=.5 costs **0.17953125ms**. Exact validation-only continuous randomization fraction:
```text
P(high-cost myopic lambda=.5) =
 (0.17375 - 0.16525520833333332) /
 (0.17953125 - 0.16525520833333332) = 0.5950383071871573
P(low-cost myopic lambda=1) = 0.4049616928128427
E_val CPU search time = 0.17375ms.
```
This calculation uses **only validation CPU cost**. Old C09/C10 test performance, exact WDL, and all new C11 data play no role. The MLP and Conv negative controls are λ4-only (stop8): p(high)=0.

Three fixed non-game-search allocation seeds **202611210101, 202611210102, 202611210103** selected before new test generation, and hashed with SHA256 to rank new families and root IDs. **Complete-block randomized allocation:** each new 12-family test group has 192 roots and exactly 114 high-cost and 78 low-cost assignments (6 families with 10/16 higher, 6 with 9/16 higher). Actual block fraction 114/192=.59375 differs slightly from continuous p=.5950383. All three assignment streams are reported, not treated as additional independent opening-position samples and not selected ex-post.

Original training + validation source, neural models, Rust source and binary, and external solver/book identities were sealed in `INPUT_LOCK.json` (20 source file SHA256 entries). `MIXTURE_PLAN_FROZEN.json` SHA256 **`faad5102c2a45f37cd2b0981aad00cc33e050ede3a2546ea1ac4c21f31f0fa63`** was committed and remotely published **before** `COHORT_384_FROZEN.json` generation. New C11 root cohort SHA256 **`ec95fdffdfa9a35bc009e607bd3882229e27fb2ea2c72f7fe8ee1eb7f5b14edb`**, 3-seed assignment map SHA256 **`146e1d0afa835e98de457c0221e02bb103e957067d6b6971de22a7573dabda30`**, were published before any external solver label was queried. The two `new_A/new_B` cohorts contain 12 new mirror-canonical five-ply opening families ×16 descendants each; previous G07 all64, C06, historical12, C09 all24 and C10 all24 families and their canonical board states were excluded. Side-to-move balanced 96/96 within each new group.

Exact independent Pascal Pons 7-column labels: **384/384 OK**, no missing/timeouts, actual bounded external solve ~29.17 seconds. File `evidence/exact_new384_20261121/EXACT_NEW384_LABELS.json` SHA256 **`25f16693b73766f5bdd6983f5a79d1fced3464cbbbef754d9c7279af12a577c6`**. All WDL score arrays, legal sentinel, target optimal-action sets and root positions were independently checked; 24 source roots and their 24 mirrored counterparts were re-solved, and another 24+24 source/mirror queries were executed after the live test. External third-party source/binary/book is not vendored.

## 3. Actual prospective CPU experiment

Previous compiled optimized Rust `mcts_demo` binary SHA256 **`4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03`**, source commit **`4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2`**. The current experiment runner source was frozen in Git commit **`348ce4e0ad311ea0c14cd512b1ad15d2ca19b95c`**, and the real results generated from that source.

**384 roots ×3 evaluator types ×9 fully real CPU MCTS endpoint arms =10,368 actual Rust root searches**, elapsed server batch ~7.27s. Arms: `full64, dynamic, myopic_low, myopic_high, mixture_1, mixture_2, mixture_3, validation_clock, live_fixed16`. Each mixture's one-step λ was selected by the frozen SHA256 root assignment **before** running that root and Rust itself made the online stop/continue decision. Models were loaded once per server process and remained resident. In-process MCTS prefixes (best action, root policy, evaluator calls and state hash) matched the full64 deterministic baseline in every arm; posthoc exact oracle was joined **only after all 10,368 searches completed**.

**Exact root WDL class distribution:** pooled root class win241 / draw19 / forced-loss124. Exactly **215/384** roots have multiple legal-action WDL classes (new_A98/new_B117). Other 169 roots are WDL-decision-saturated under the sign-score metric; no strong playing-skill population claim from all384.

### Uniform evaluator (lower regret is better; physical CPU search in milliseconds)

| C11 test cohort | Dynamic R | Mix1 R | Mix2 R | Mix3 R | Dynamic search ms |
|---|---:|---:|---:|---:|---:|
| new_A, 12 families/192 roots | .156250 | .156250 | .156250 | .156250 | .172391 |
| new_B, 12 families/192 roots | .161458 | .166667 | .161458 | .166667 | .157839 |
| **pooled, 24 families/384 roots** | **.158854** | .161458 | **.158854** | .161458 | **.165115** |

Pooled actual mean search wall-time: Mix1 **.161690ms**, Mix2 **.161982ms**, Mix3 **.163990ms**. Pooled relative mismatch versus dynamic: **2.074%, 1.897%, 0.681%**; in **each** fresh 12-family group all three mixed arms also independently pass the pre-fixed **≤10%** mean-cost gate. Dynamic chose **218** two-step commitments, while each one-step mixture was unable to use a two-step option by construction.

**Cost-matched primary paired differences, pooled (dynamic−myopic regret, negative better):**
- Seed1: **−0.00260417**, conditional 24-family bootstrap 95% **[−0.0078125,0]**.
- Seed2: **0**, conditional 95% **[0,0]**.
- Seed3: **−0.00260417**, conditional 95% **[−0.0078125,0]**.

The bootstrap resamples **24 opening families**, not 384 descendant roots and not three assignment streams as independent game observations. WDL regret at the root level differed in **exactly one of 384** source positions for Mix1 and Mix3, and in zero positions for Mix2. Cost match is now established; **independent nonmyopic utility superiority is not**. The intervals touch zero, and the sign/size could be sensitive to a few positions or source generator choice. There is no negative general theorem that synergistic computation is impossible.

**Conditional expected randomized-policy comparison (not a fourth runtime arm):** combining the two actually run frozen myopic endpoint outcomes per root with predeclared p gives exact conditional-on-frozen-test expected mean regret **.1599087544** and mean search cost **.16207811ms**. Dynamic regret **.15885417** differs by only **−.00105459** and mean CPU cost is **.16511458ms** (1.84% mismatch). This expected mixture is a valid *ex-post conditional randomization expectation*, but it does not replace actual seeded online trials or provide extra IID observations.

**Other benchmarks / limits:** Uniform dynamic vs validation-only fixed-clock pooled regret .158854 vs .171875, but the clock arm actually used **.189021ms**, a **14.48% CPU mismatch** (new_B mismatch 20.3%) so the pooled comparison is **NOT admitted as cost matched**. Genuine live16 regret .1953125, search .148677ms, also not matched. Existing stronger `myopic_high λ=.5` achieves the *same pooled regret* .158854 at .167138ms; small time difference is descriptive, not robust algorithmic superiority.

### Negative controls

New MLP: dynamic+all myopic mixtures stop at 8 on every root, 0 two-step commitments, pooled regret **.250000** (genuine fixed16 .1953125 at higher CPU cost). ConvNet: same stop8 collapse, 0 joint commits, pooled regret **.205729** (fixed16 .148438 at higher CPU cost). The two-stage controller did not yield general cognitive-policy improvements across evaluator classes.

## 4. Statistical and computational verification

- `mix_plan11.py`: tests RED→GREEN and freezes val-only bracket/probability before creating any new test root. `freeze_cohort11.py`: all old mirror-canonical prefix/state exclusions and all 3 family-block assignments. `COHORT_ASSIGNMENT_READBACK.json`: independent recomputation of source exclusions/quotas/hash.
- `oracle11.py` + `audit_oracle11.py`: new exact external teacher status, complete 7-column labels, opening-family 24 original/24 mirror re-solves.
- `run_live11.py`: real pre-commitment allocated myopic controls inside one warmed Rust server; **all 10,368** responses, CPU ms, root actions, and oracle labels after execution recorded in `evidence/live_heldout_20261121/{RAW_10368_RELEASE_ROOTS.json,PAIRED_384_NEW_POSITIONS.json,SUMMARY_C11_COST_MATCH.json,EXECUTION_MANIFEST.json}`.
- `audit_live11.py`: full **independent** reproduction of frozen SHA256 blocked randomization, all online policy decisions from present-state 9-feature + fixed gains/cost, all WDL regrets/CPU wall/evaluator calls, 10,368 deterministic prefixes, actual model residency, and further 24+24 exact solver re-solves. `verify_receipt11.py` writes cryptographic provenance and test evidence `VERIFICATION_RECEIPT.json`.
- Fresh Rust release `cargo test --offline --locked --release -- --test-threads=4`: **596 PASS / 0 FAIL / 89 ignored**. Rust debug analogous **596 PASS / 0 FAIL / 89 ignored**. Related Python research suites **116 PASS / 0 FAIL**. The Rust library/actual engine executable was not mutated in C11, so the release binary SHA is identical to C10.
- `native_subagent: NOT_RUN` because no native delegate tool was exposed; all local calculation actually ran through authorized remote CPU terminal and the external exact game solver, plus read-only literature source comparisons. No GPU.
- All measured CPU figures are **wall-clock per root stage** on one host, not process CPU time, hard deadline or deterministic per-root latency. No independent search PRNG trajectory replications were used (the SHA seeds only randomize allocation). Results are conditional on two synthetic 12-family cohorts, 215 decision-informative positions, the frozen original training distribution and selected terminal exact WDL utility.

**Final claim state:** finite source/XOR `derived`; Rust online computation selection and external-score readback `implementation-verified`; costs, utilities and conditional family inference `numerically checked`; a distinct cost-matched nonmyopic advantage over real myopic computation `UNRESOLVED/NOT_ESTABLISHED`. No claim of strong no-go, human cognition, general Elo gain or novel Bayesian theorem.

## 5. Per-stage backups and GitHub publication

Each bounded completed unit is committed to dedicated branch, archived as `git archive HEAD ... | gzip` under `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`, SHA256 verified, pushed to GitHub, and exact remote `git ls-remote` SHA checked:

- A `8d1699414e6ec98d2103682d90c21c016cfe105e` validation-only mixture before test roots.
- B `45fbcdffae6ae54022c490e94669a0db8db6855b` fresh 24 families and 3 preassigned blocked randomization maps.
- C0 `d2f6995c5fa9b87b844d66c46f5e79fb19f30904` strict external solver runner.
- C1 `2757f922b19f9c17854d8257a9dd887c44fb1348` actual external exact384 WDL and mirror audit.
- D0 `348ce4e0ad311ea0c14cd512b1ad15d2ca19b95c` actual 9-arm real CPU runner, tested and frozen before search.
- D1 `94fb6f8a5cbe592cc4410b5141b49eee7332e256` actual 10,368 CPU game-search data and independent audit.
- E final: this conclusion/handoff; exact HEAD/tree/archive checks to be recorded during final publication.

Original developer checkout `/home/cosmosapjw/Dropbox/personal_projects/quartz` at `agent/local-experiment-foundry`, HEAD `65e7d3e9596ca259ebfe523b0c74058692b6a755`, with pre-existing untracked `QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md`, and unrelated worktrees remain untouched. Third-party AGPL sources/book/binaries are not redistributed. Local Dropbox folder existence and archive SHA do **not** establish cloud synchronization ACK or archive restore verification.

## 6. Next bounded research ID: QUARTZ-COG-03-EINSTELLUNG-REOPENING-12

**Do not automatically start a new cost-alignment meta-loop** to obtain a nominally significant epsilon-sized improvement. The COG02 next-step claim has been bounded and reached an informative negative result at actual matched CPU cost. Resume the *original* cognitive goal on **recognition-driven candidate reopening / Einstellung under compute pressure** rather than inflating VOC calibration gates.

Proposed science-first node:
1. Define a root-level attention fixation/candidate omission experiment on standard Connect4, including an exact solver-defined set of decisive moves used **only for posthoc diagnosis** and not fed into online attention. Freeze new families, position distribution, low-prior target criterion and CPU/time contract before testing.
2. Implement minimal **candidate-reopening operator** in Rust MCTS (root candidate policy/prior gate + independent anomaly or uncertainty evidence) with explicit observation features and a bounded two-step nonmyopic option as one optional controller. Compare with uniform, ordinary one-step policies and the currently frozen C11 dynamic model at same actual release time. Count opening-family clusters and decision-informative positions separately, preserving the best move's rank and prior under each policy.
3. Quantify whether reopening creates genuinely new **decisive-move recoveries** and a positive exact WDL utility change rather than just KL/policy improvement that does not change final action. Use new independent opening families; negative result is admissible.
4. If external real trained Gomoku/Rapfi engine is feasible on CPU and rules/protocol are checked, treat migration to 15×15 Gomoku/Renju as **a separate distribution and rule-transfer study**, not automatic extrapolation from Connect4 7×6. Preserve neural MLP/Conv failure as an explicit lesson.

**Stop:** C11 completed the requested *cost-matched* test and preserved its negative/near-null scientific outcome. The next node is a fresh scientific hypothesis, not a rescue pass for this result.
