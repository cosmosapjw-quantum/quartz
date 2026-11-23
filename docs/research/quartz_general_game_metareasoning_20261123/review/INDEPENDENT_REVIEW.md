# C26 independent code/science review

Reviewer: `/root/c26_review`. Date: 2026-11-23. This report and its JSON companion are on the **REMOTE cosmosapjw device**, not the shared local container.

## Verdict

**Looks good — scoped Review Gate passes.** No unresolved blocking finding remains. The parent-discovered P1 result-publication omission is repaired at C08 commit `1f29a3152828ce7d4d347be362eb2d4a79ec4735`, and the P3 theoretical endpoint wording is corrected at the exact worktree identities verified below. P0/P2: none. The completed TEST campaign does **not** pass the frozen positive node-priced gain criterion; preserving that result is the correct scientific disposition, not a code defect.

Reviewed repository: `/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123`, branch `research/quartz-general-c26-game-metareasoning-20261123`. Initial review commit: `b443458d4d8bae0f95d68f38f0dc8e23323f75fa`; C25 parent: `d6019d674ae76524b1895016f49b82d4f7b001b2`. CAL ran at `16141896adab22df4e423bd4946b0544a19d8d4e`; TEST and the raw review were observed at `8101e7610dc59602f3abe938e49b43bf2571f191`. Production source, harness, tests and binary retain their pre-cohort frozen identities.

Read-only confirmation: no project files edited, formatted, generated, staged, committed or published; no policy cohort run. The reviewer wrote only the two authorized temporary reports, `/tmp/quartz_c26_review.md` and `/tmp/quartz_c26_review.json`. Parent owns integration and final archive/publication evidence.

## Scope and identities

Read the complete native binary, complete independent campaign harness, focused tests, frozen CODE_CONTRACT/SOURCE_SPECS, R03/R05/R06/R08, full constructor proof, statistical-physics derivation, origin watchdog/map, narrow THESIS diff, and recorded test evidence. Directly read and hashed original PDF pages 5–6, archived PRD and diagonal-Lindblad text, and the COG-01 contract. Reviewed all stored CAL/TEST raw response records and independently reconstructed the primary inference.

| Artifact | SHA256 |
|---|---|
| Native source | `145b586fea7ff7526eb30d15d683580678f1c9dd3a433967f536697957327a01` |
| Campaign harness | `ee4501d5bfee66adcc15dbfb0b0f93ee50343e3313c7d1ce654143571ecee750` |
| Python tests | `9f265bc253cf62913591a72e9b5460a4c0f1f2288f97a338dcaec57c16084b19` |
| Native release binary | `8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c` |
| CODE_CONTRACT | `1821e46989ab21f9259ee9c61e4935824a362da5139d08db13d10fb86c1d0985` |
| SOURCE_SPECS | `07826834be13bdb7f779459d0ca7e964ace402af61af286bda1a108503b3a466` |
| TRAIN model | `0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126` |
| Source manifest | `1dfb2b92e4fca4bb6827fda0b955eb49e3c2a8b7550a27d7ed3dece4db5b8c16` |

`src/game.rs`, `src/games/tictactoe.rs`, `Cargo.toml` and `Cargo.lock` have no change from the C25 parent. Additional original-source and raw-file identities are in the JSON companion.

## Findings

### P0

None.

### Resolved P1 — C26-REVIEW-P1-02: C07 result publication omitted by inherited ignore rule

- **Discovery/provenance:** parent /root reported after the code/science review that inherited `.gitignore` ignores `coding/results/`. The eight CAL result files were not in C07 Git/its Git-based cumulative overlay, and the eight new TEST files were not staged. Independently reviewed raw files remain intact on the device.
- **Impact:** the per-stage evidence publication/backup contract was not met for CAL at C07. This is a provenance/packaging defect; it does not change actual execution or raw-result validity.
- **Completed repair:** parent force-added exactly the CAL and TEST result directories at C08 commit `1f29a3152828ce7d4d347be362eb2d4a79ec4735`. `evidence/C08_PACKAGING_CORRECTION.json` preserves the C07 omission and C08 correction and records a supplemental CAL archive, SHA256 `6c7159e0c32c5e2bf3096f6923e397c5c14ee396fc5dd0b47112e62e7076b177`, 897,421 bytes, with successful parent archive readback.
- **Verified closure:** remote process 302990 independently read the exact C08 tree and compared all sixteen result Git blobs to reviewed worktree bytes: eight CAL and eight TEST files, all byte-identical. Their raw and analysis hashes match the prior independent review. The supplemental archive receipt and its eight CAL hashes were inspected; archive extraction itself was performed by the parent and not duplicated here. P1 is closed. The history remains: omission at C07, correction at C08.

### P2

None.

### Resolved P3 — C26-REVIEW-P3-01: zero-step endpoint in the Markov positivity statement

- **Evidence:** `theory/STATPHYS_DERIVATION.md:21` and its generating text at `theory/statphys_check.py:416` first allow `0<=delta<=1/kappa`, then say that a delta strictly below `1/kappa` makes all entries of `P=I+delta W` positive.
- **Observed counterexample:** at `delta=0`, `P=I`, so off-diagonal entries are zero and relative entropy is unchanged. The subsequent contraction proof at derivation line 31 correctly assumes a strictly positive P.
- **Impact/severity:** bounded mathematical documentation condition. Actual finite checks use `delta=1/2`; the production policy directly samples Gibbs and does not use this transition map. No cohort result is affected. This is nonblocking P3.
- **Remedy:** state `0<delta<1/kappa` for strict positivity; retain `0<=delta<=1/kappa` for nonnegative column stochasticity. Correct both the displayed report and its generation string while preserving original bytes and execution evidence.
- **Validation:** direct substitution at zero and positivity of off-diagonal rates for positive delta suffice. No production or cohort rerun is warranted.
- **Verified closure:** remote process 302990 checked both current files against their preserved C08 bytes. Each differs by exactly one replacement from `With delta strictly below 1/kappa` to `With 0<delta<1/kappa`, with no other changed byte. The receipt is `evidence/C09_ENDPOINT_CORRECTION.json`. Corrected derivation SHA256: `d717a973a529bd339a259b824ddaa80e88cd054fc4baee44102a9f14360fc512`; corrected generation script SHA256: `ef81457919054c18d5a29c7b9c9af1d9003179798a235fbd4af523813c3a9826`. P3 is closed for these observed worktree bytes; parent owns their C09 commit/publication. Reviewer edited no project file.

### Nitpicks

None.

## Code and runtime conclusions

The review traced actual data flow, rather than relying on passing tests.

1. **Signs and exact cost.** Native lines 227–253 implement `p_root*p_terminal*outcome`, root-relative max/min, and one count for every entered state with no pruning/cache. Lines 283–325 charge terminal cues once while representing their known outcome with four feature counts. Lines 374–413 choose the common raw-rollout incumbent and retain the best known terminal/exact value. Lines 728–780 charge only newly revealed candidates, retain incumbents on ties, and satisfy `charged=cue+initial+additional`.

2. **Known-only deployment.** The selected move is always a terminal-observed or exactly queried candidate. Initialization handles known terminal wins, all-terminal children, tie ordering and a paid nonterminal initial verification. A reveal updates the incumbent by a maximum. This pathwise utility property remains true when learned probabilities are inaccurate.

3. **Public input boundary.** Request/model types deny unknown fields. Legal history replay precedes unchecked GameState move application. Invalid JSON/history/model requests recover on the next input line. Survey is an explicit TRAIN measurement interface; run requests contain only public history/seeds/controls and the frozen model. Unrevealed utilities are absent from run predictions.

4. **Policies.** GREEDY uses positive `g-d`; PANDORA uses `z>b`; deterministic ties retain ascending move order. Model probabilities and cost estimates remain fixed after reveals. Gibbs samples the public seeded categorical law, retains nonpositive-Q FOCUS support below `b=1`, and applies the explicit exact-win mask. ALL_ACTIONS deliberately ignores that certificate and continues exhaustive opening; RANDOM_ORDER uses it. This matters for interpreting savings.

5. **Timing.** The per-policy timer covers regenerated cues, feature lookup, initialization, allocation and exact work, controller arithmetic and temporary work-state cleanup. Root/model request transport and output serialization are outside the declared decision boundary. Trace passes are excluded from timing statistics. Raw timing components sum to elapsed, and logical signatures agree across nine repetitions and separate trace passes.

6. **Training and inference.** Model construction uses only nonterminal TRAIN candidates with add-one categorical smoothing and TRAIN arithmetic-mean node costs. The complete raw TRAIN refit independently reproduced all parameters: 90 parents, 296 nonterminal candidates, class counts `[183,65,48]`, 41 feature bins. Model provenance precedes CAL, which precedes TEST; the same model hash is bound to both campaigns.

The source-parent grouping, fixed split, and paired inference follow the frozen contract. The uncertainty interval is conditional on this fixed TRAIN fit. Shared game descendants can couple parent boards, and that limitation is explicitly recorded in the analysis rather than hidden behind an iid performance claim.

## Mathematical conclusions

**Exact STOP.** For independent unrevealed variables, the chosen query is measurable before its value is observed. Thus its conditional expected net incumbent increment is `g_a(b_t)-d_a`. Since `b_t>=b` and g decreases in b, all initial `g_a(b)<=d_a` make every expected future increment nonpositive; a finite sum closes the proof. A positive feasible one-query gain gives the converse. Equality allows neutral queries, so optimal STOP is not asserted to be unique.

**Pandora.** With `Y_a=min(U_a,z_a)` and `E_a=(U_a-z_a)_+`, independence gives `E[I_a E_a]=E[I_a]d_a`. Subtracting all opened nonnegative excesses yields the capped-reward upper bound. Descending reservation order attains it because only the final opened candidate can have a positive excess. A known fallback, atoms, negative thresholds and ties are covered. Binding additional-query caps and correlated updating change this problem; the full-horizon cap nine is nonbinding for these finite roots. Actual predicted costs are a model closure, not the fixed true charge assumed by the theorem.

**Categorical formula and Gibbs.** The guarded three-branch index solves the positive-excess equation at all support boundaries and degenerate cases. The identity `F-[q.Q-tau KL(q||pi)]=tau KL(q||q*)` gives the finite-temperature optimizer, gradient and covariance Hessian under fixed positive reference support. Actual sampling is present. Local Gibbs does not claim the unrestricted raw-net or multistep Bellman optimum.

**Fluctuations and dynamics.** The relevant quadratic information-value covariance is `Cov(E[U|Y])`, with `Cov(U)=E Cov(U|Y)+Cov(E[U|Y])`. The Hessian annihilates common shifts. The displayed range-based third-order bound supports a fixed-positive-temperature local expansion. Detailed-balance column rates, off-diagonal Lindblad population transfer and finite Kraus normalization are correct under their stated conditions, with the P3 zero-step wording now corrected. Diagonal jumps alone dephase. Utility temperature, algorithmic relaxation time and physical temperature/time remain distinct.

The correlation and budget counterexamples refute precise transported statements and motivate replacements. They are not promoted to a global no-go; NG1–NG7 admission is not claimed.

## Direct execution and reused evidence

The reviewer ran **20 focused Python tests**, exit 0, no failure/error/skip, in 0.332 s (remote process 282982). Source hashes before/after agreed. The only excluded test was the exhaustive all-reachable audit because it writes project evidence; its existing result was read and reused. These fixtures cover literal board signs/costs, model smoothing, index boundaries, Gibbs frequency/scale, no-oracle input, malformed-request recovery, all-policy traces and aggregation.

A standalone read-only raw audit (remote process 291664, exit 0) inspected **2,520 timed request records and 2,520 trace request records**, encompassing **136,080 timed runs and 15,120 trace runs**, across CAL and TEST. It verified file/source/model/request hashes, complete cap/seed/source cells, policy/repetition coverage, nine-repeat logic, trace/timing signatures, 27,401 decision transitions, and 246,127 trace additional-node charges. It independently reconstructed parent means and the 2,000-draw PCG64 bootstrap. A separate raw TRAIN reconstruction (remote process 293420, exit 0) reproduced every fitted parameter exactly.

Reused evidence, checked at matching identities:

- Native binary tests: 18 passed; focused Python suite: 21 passed.
- Every reachable nonterminal TicTacToe root: 4,520 roots, 16,167 candidate children; both players and all three root utilities; exact values, uncached node multiplicities and cue accounting agree.
- `cargo test --release --locked --offline`: exit 0; 754 successful test invocations, 89 ignored, across 12 targets. These are invocations, not 754 distinct independent scientific validations.
- Full Python attempt: exit 2 at collection because `tests/test_a18_evaluator_ablation.py` imports missing torch. No C26 torch dependency was introduced. The full Python suite is not reported as passing.
- Existing CI Rust and real-loop smoke jobs succeed; the Python job stops at inherited Ruff formatting before pytest. Read-only C25 Ruff 0.17.6 checking identifies 272 existing files; the seven new C26 Python files pass their formatting check. This is a repository-wide verification limitation, not a newly demonstrated C26 runtime defect.
- Exact theory oracle: 9,495 cases; statistical-physics checks: 72 assertions. These finite checks supplement the displayed analytic proofs; they do not establish game independence or efficacy.

One tool-runtime reset interrupted the response to a read-only TRAIN inspection. The background-call registry reported no retained calls; one read-only repetition returned the completed TRAIN reconstruction. No mutation or cohort was retried.

## Held-out result and claim admission

The independent raw reconstruction reproduces the frozen TEST primary:

`PANDORA minus INITIAL_STOP, cap=9, delta final_u - 2^-10*delta charged_nodes`.

Mean **0.005690104166666667**; 95% stratified paired parent percentile interval **[-0.013693033854166667, 0.029039388020833317]**; **positive_gain_gate=false**. There are 150 source parents, 50 at each of plies 4/5/6. NumPy 2.4.2 PCG64, seed 2600923, 2,000 draws, reproduces the stored endpoints. Positive expected priced gain is not confirmed by this criterion.

CAL gives 0.06865234375 with interval [0.027977294921875, 0.11551188151041666]. It remains diagnostic and does not replace the TEST result.

Full-cap TEST means, independently reconstructed from raw nine-repeat medians and four-seed means:

| Policy | Exact final utility | Charged nodes | Decision time, microseconds |
|---|---:|---:|---:|
| INITIAL_STOP | 0.6966667 | 58.3133 | 2.72411 |
| ALL_ACTIONS | 0.7266667 | 124.7200 | 4.88970 |
| RANDOM_ORDER | 0.7266667 | 85.2600 | 3.64306 |
| GREEDY_VOC | 0.7266667 | 83.4733 | 3.60116 |
| PANDORA | 0.7266667 | 83.2067 | 3.58708 |
| GIBBS_VOC | 0.7266667 | 81.1567 | 3.58152 |

Every full-cap TEST source/seed cell of the five nonbaseline policies matches ALL_ACTIONS utility, not merely their aggregate means. This equality is specific to the observed TEST cohort; CAL Gibbs has a small shortfall. Savings against ALL_ACTIONS combine ordering/stopping with its deliberately exhaustive known-win behavior. RANDOM_ORDER already saves substantial work, so the full difference cannot be attributed solely to Pandora or Gibbs.

TEST Gibbs has 1,390 actual sampled decision steps; minimum positive reported choice probability 2.558547591235855e-10 and zero underflow zeros on sampled steps. Thus its finite-temperature support requirement is realized in this cohort. TEST categorical Brier is 0.3272727088243262; predicted node-cost bias is -1.120632605550633 and MAE 4.691270319516157. The within-root residual correlation 0.17673145444527896 is a descriptive diagnostic, not a proof/test of conditional independence.

## Origin preservation and limitations

Original QUARTZ Eq. 1 genuinely contains the KL/Gibbs objective; Eq. 2 motivates priced computation value. The actual COG-01 contract supplies TRAIN-only recognition, verification and separate quality/value calibration. C26 introduces an explicit game-root implementation and an explicitly new Pandora connection. Narrow THESIS corrections allow runtime estimated VOC while keeping offline oracle labels/signatures out of control, and allow proved statistical mechanisms while retaining standards for physical and empirical claims.

The original same-budget Elo/CPU objective and cognitive P3 target remain open. C26 does not establish high-branch MCTS efficacy, cross-game transfer, human cognitive validity, a physical temperature, or a quantum hardware advantage. These are scope boundaries already preserved by the contract. No failed experiment or unproved extension is recast as impossibility.

## Handoff and checkpoint

- **Blockers:** none. The concrete P1 publication and P3 endpoint findings are resolved at independently inspected identities.
- **Completed corrections:** C08 now tracks all sixteen result files with matching blobs; the two C09 endpoint strings are corrected exactly, and the original C08 bytes/evidence remain available.
- **Restore evidence:** parent subsequently reported the actual four-archive restore completed with exit 0 at `/tmp/quartz_c26_restore_b7q8p1o8`, exactly C08 `1f29a3152828ce7d4d347be362eb2d4a79ec4735`: 2,768 tracked blobs, 1,106,732,654 bytes, modes checked, no unexpected files, fourteen older worktrees and protected sources unchanged. Evidence pointer: `evidence/C09_RESTORE.json`. This is explicitly reused parent execution evidence; the reviewer did not repeat the restore. Dropbox metadata acknowledgement is distinct from downloaded cloud-byte verification.
- **Scientific decision:** retain conditional theorems, verified implementation, observed safety and exact cohort results; do not admit positive TEST primary priced gain.
- **Next / upcoming task:** parent publishes C09 correction/review/restore evidence, then completes C10 closeout. The independent reviewer task is complete. No new cohort or price/temperature retuning.

Skill context actually read: PhysMath Research Loop 1.3.0, POLICY_INTEGRATION, SUBAGENT_MODE, CODING_LOOP C9 and RESEARCH_POLICY NG1–NG7; Keystone change-review 2.0.4 plus review/checkpoint gates. One bounded independent review pass; no child agents or review-of-review.
