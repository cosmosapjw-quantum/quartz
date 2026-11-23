# C25 C09 independent review

## Verdict

**Looks good for the frozen C25 research claims and handoff. Review Gate: PASS.**

No P0/P1/P2/P3 source or scientific-claim defect was found within the declared C25 scope. The retained-decision implementation and deterministic charged-work stopping theorem are supported under their explicit assumptions. The primary empirical claim of positive full time-priced benefit against PLAIN_STOP is **not supported on this fresh TEST cohort**; the negative result is correctly preserved in CLAIMS.json and CODING.md. This is a scientific result, not a failed implementation acceptance gate.

This verdict applies to scientific snapshot **d45173685f955820bd848bf4ddb2044382ca7e67**, with C24 parent **4998c771c08788e4f0da8d2723347121ae16fbf2**. It does not assert a universal hardware-time theorem, optimal computation allocation, or the original full MCTS/Gomoku P2 target.

Reviewer: actual native child **/root/c25_review**, one bounded fresh-context C09 pass. No recursive delegation. Review date: **2026-11-23 UTC**.

## Location, identity, and authorized changes

All paths in this report refer to the **REMOTE Desktop Commander device cosmosapjw**, device ID **edd1b263-56a0-4391-8a2a-677dd00eabf5**.

Remote repository:
 /home/cosmosapjw/quantum/research/quartz_general_meta25_20261123

Remote document root:
 /home/cosmosapjw/quantum/research/quartz_general_meta25_20261123/docs/research/quartz_general_incumbent_price_20261123

Remote reviewer outputs:

- review/C09_REVIEW.md
- review/C09_REVIEW_CHECKS.py
- review/C09_REVIEW_CHECKS.json
- review/C09_PYTHON_TESTS.log

The review did not edit production code, source specifications, scientific contracts, predictor, raw observations, existing test files, or previously published evidence. No git mutation, native benchmark, native build, or new source campaign was performed. The only writes were the authorized review artifacts above. Parent-owned restore/report metadata may appear in the worktree status and was not cleaned up or treated as a scientific defect.

C09 independently checked all **16 frozen inputs**, including the native source, analyzer/verifier, source specifications, theoretical contract, predictor, and exact release binary. The protected C24 numerical prefix is identical after accounting for the unused entry-point rename and trailing whitespace. The protected C24 file itself is unchanged. All **205 inputs** read and bound by the reviewer script had matching before/after SHA-256 values. Observed HEAD remained d45173685f955820bd848bf4ddb2044382ca7e67 throughout that run.

Key identities:

- C25 native source SHA-256: 84c3937e42836431145fbde74999a57f98540e6b65760f2d71c6a4f78271a8d8
- C24 source SHA-256: 5a7cc595c267d5337f268045d1df8bf5909093ed7920ee50293da76da94cd381
- Native release binary SHA-256: 2ddccac45f1fa9c171509bcc329a37f2f9b575fe3c17797e806a74ee4d8febf5
- Reviewer script SHA-256: a3b00b095d4adc3714455bcc063255c8866b41ec1ff155b59ff14cc6f0ed84d6

## Scope and behavior paths inspected

Read the actual derivation, R06 parent check, origin watchdog, code/source/run contracts, frozen inputs, claim ledger, coding record, all six new Python modules, changed Python tests, and C25 native controller suffix. Traced the controller through the inherited solver, ordinary/FAST evaluator, initial score tie rule, historical intersection, actual readout counters, and elapsed-time boundaries.

Principal source locations:

| Object | Evidence |
|---|---|
| Retention, box regret, paid-continuation proof | theory/DERIVATION.md, sections R05.2–R05.5 and R05.7; theory/R06_PARENT_CHECK.md |
| Initial ordinary-score identity | Native ordinary_or_fast_response at line 550; score_action at line 756; initial readout in run_c25_retention at line 2005 |
| Enclosure history and actual all-action reads | Native measure_bounds at line 776 |
| Switch, gap, zero case, price direction, stop priority | Native c25_decide at line 1909 |
| Mandatory fresh readout after every admitted block | Native run_c25_retention at line 2005 |
| Successful solver matvec accounting and errors | Native Solver::step at line 299 |
| Six-policy execution, cyclic order, invalid requests | Native execute_c25 at line 2178 and changed behavior tests |
| Public-only transport and frozen runs | coding/c25_run.py, to_wire at line 26 and main at line 51; coding/c25_io.py |
| TRAIN-only cost definition | coding/c25_fit_predictor.py, fit_predictor at line 15; SOURCE_SPECS predictor contract |
| Source-level statistics and paired bootstrap | coding/c25_analyze.py, parent_metrics at line 23 and summarize at line 118 |
| Exact verification and finite residual coverage | coding/c25_verify.py; C25_VERIFY_TEST_TRACE.json |
| Original objective versus SPD proxy | ORIGIN_WATCHDOG.md sections 5–6; CODE_CONTRACT origin bridge; CLAIMS T5 |

The executable seam is a bounded local JSONL scientific CLI. The runner sends public task data plus a frozen timing prediction; hidden manufactured solutions and oracle utilities remain offline. The parser rejects unknown/oracle fields, the runner verifies the exact public field set, the manifest reader confines chunk paths to the evidence directory, and failed requests are explicit error objects rather than successful policy outcomes. There is no new network/authentication/production-game service surface in this change.

## Scientific and numerical assessment

### Retention and regret

Under simultaneous enclosures of fixed utilities, the sole accepted switch is L_j > U_i, hence u_j >= L_j > U_i >= u_i. The initial incumbent uses the same ordinary score arithmetic, four initial CG updates, and smallest-index score tie break as PLAIN_STOP. The implementation preserves that incumbent through budget/no-progress returns and never replaces it unconditionally with the final ordinary winner.

For fixed incumbent i, G_i = max(0, max_{j != i} U_j - L_i) bounds true regret. Box sharpness follows by choosing the incumbent coordinate at its lower endpoint and a maximizing competitor at its upper endpoint. The derivation correctly distinguishes this rectangle from the more structured SPD feasible set. Historical intersection reduces G for a fixed incumbent; a strict switch to a maximum-lower-bound candidate also cannot increase it. Selecting the maximum lower endpoint is correctly not claimed to minimize robust regret.

The code explicitly returns zero when max_other U <= L_i and otherwise rounds the difference upward. It keeps weak zero-regret optimality separate from strict unique optimality and price stopping. These distinctions survived the independent raw checks.

### Charged-work stop and actual cost floor

The proved comparison is between the **current post-switch incumbent** and the declared positive continuation class. For any such continuation, its terminal improvement is bounded by G_i, so its incremental priced advantage is at most G_i - lambda C_future. The bound C_future >= K n gives the sufficient stop condition.

The implementation's positive continuation always performs a fresh FAST readout of all K actions after the CG block, including zero CG progress. In successful solver paths, each reported CG iteration corresponds to one actual operator application; failures after an application return an error rather than a successful charged outcome. Thus the explicit cell-stencil accounting supports the stated K n floor. The floor is not a count of every CPU operation.

The selected work price is exactly 2^-30 utility per cell-stencil charge. Reviewer checks covered all **889 side/K combinations** in side 2–128 and K 2–8 and confirmed exact binary64 representation of the dyadic threshold products. The implementation uses a conservative downward neighbor for the threshold and an upward gap, so boundary conservatism cannot produce a false certified stop.

The proof and code correctly handle free certified changes before paid comparisons, exclude sunk cost from future-cost bounds, and distinguish local remaining-charge dominance from whole-policy ex-ante net benefit. Skipped/cached action reads, partial batches, new candidate expansion, or a different output-action class would require a revised floor; C25 does not silently implement those variants.

### Empirical wall-time rule

The wall-time price is the fixed exact rational 1/1,000,000 utility per microsecond. Directed multiplication is conservative relative to that price but does not certify the prediction as a pathwise or universal conditional cost lower bound.

The fitted quantity is the TRAIN RETAIN_BLOCK continuation time per executed block at additional 32: mean over repetitions within each active source parent, then median over active parents of the same side. Inactive parents are excluded as prespecified. Native continuation timing excludes the initial validation/solve/readout and includes subsequent control/readout work and destruction. Full total_us includes initial expense and is used for policy comparison. Averaging per-block continuation cost is an empirical estimator; it is correctly not presented as the cost of every future block.

The predictor was created at 06:36:26.886411 UTC; pre-TEST freeze is 06:40:43.537310 UTC; TEST source generation starts at 06:41:20.024450 UTC. The independently reconstructed predictions are:

| Side | Active TRAIN parents | Predicted block cost, microseconds |
|---:|---:|---:|
| 8 | 7 | 8.914666666666667 |
| 32 | 10 | 116.66068055555556 |
| 96 | 10 | 1004.2051666666666 |

The rules, source specification, model, price, and analysis code remained frozen. No TEST outcome was used to refit this model.

## Executed independent validation

The reviewer script imports Python standard-library modules and NumPy only. It does **not** import the production generator, runner, verifier, analyzer, or predictor for its scientific calculations. The separate existing Python unit-test invocation imports those modules as its normal test subject.

Actual command, executed on the remote device from the repository root:

    venv/bin/python -B docs/research/quartz_general_incumbent_price_20261123/review/C09_REVIEW_CHECKS.py

Observed remote process PID **4108444** completed with **exit code 0**. Scientific check elapsed time was **7.6122 seconds**; host process receipt reported approximately **7.78 seconds**. The run occurred from 06:53:15.183138 to 06:53:22.796976 UTC. The exact command, environment versions, input identities, output status, and counts are in C09_REVIEW_CHECKS.json.

| Independently checked object | Finite coverage |
|---|---:|
| TEST manufactured solution equations A x = b | 741,888 exact components |
| TEST exact quadratic utilities from those solutions | 216 actions across 72 sources |
| TEST timing requests | 216 |
| TEST timing policy rows | 11,664 |
| TEST timing interval containments | 104,976 |
| TEST trace interval containments | 28,584 |
| TEST trace/timing deterministic policy matches | 1,296 policy-source-budget groups, each timing group has 9 matching repetitions |
| Distinct retained source-budget-policy noninferiority outcomes | 648 |
| Retention gap checkpoints | 1,529 |
| Actual fresh-readout floors after admitted blocks | 881 |
| Strict accepted switches | 76 |
| Certified charged-work price stops | 41 |
| Empirical wall-time price stops | 61 |
| Independently reconstructed primary-price policy-budget summaries and bootstrap intervals | 18 |
| TRAIN timing requests read for predictor reconstruction | 108 |

The reviewer recomputed all TEST per-source medians, exact regrets and utility differences, exact primary-price net means, and primary-price paired percentile bootstrap intervals at all three budgets. It generated 2,000 shared source-parent resampling indices with seed 2500923 in the prespecified sequential budget order 8, 32, 128. All 18 exact primary-price means and all 18 corresponding intervals matched the published summary. Exact mean work-priced differences were also reconstructed for all 18 policy-budget combinations.

The existing Python tests were executed as:

    venv/bin/python -B -m unittest discover -s tests -p test_c25_sources.py -v

Result: **18 tests passed, exit code 0**, with PYTHONDONTWRITEBYTECODE=1. Log: review/C09_PYTHON_TESTS.log.

Existing native debug/release and RED/GREEN logs were inspected rather than rebuilt: debug and release each record 40 passing native tests; 12 changed native behaviors had observed RED-to-GREEN evidence, with the initial/control identity test separately identified as reused. The Python original RED log records 10 failures and the analysis RED log 2 failures; current 18-test GREEN includes the six later regression tests. These are actual behavior failures, not fabricated execution failures.

The historical c24_test_34_1 fixture remains UNIT regression evidence and is excluded from fresh TEST parents. Its preserved source-budget results reproduce ordinary loss while all retained variants preserve the initial action. C09 read this existing fixture result rather than starting another native run.

## Independently reproduced primary outcome

Primary additional budget is 32, with 72 independent manufactured source parents and nine measured repetitions per source-policy reduced to a median.

| Policy | Mean of source median time, microseconds | Wrong source decisions |
|---|---:|---:|
| PLAIN_STOP | 238.08123611111108 | 12 |
| PLAIN_ALWAYS | 1625.49175 | 0 |
| FAST_BLOCK | 1092.4213472222223 | 0 |
| RETAIN_BLOCK | 1090.6769861111109 | 1 |
| RETAIN_WORK | 929.8645972222221 | 2 |
| RETAIN_TIME | 729.63325 | 3 |

The independent exact TIME-minus-STOP primary mean is

    -2548119488053517006043105599 / 5440166188265831286177792000000

or **-0.0004683900086636479**, with paired-parent percentile interval

**[-0.0006629366545796896, -0.00028780714297975434]**.

Retained utility nondecrease therefore coexists with negative full priced benefit against immediate stopping. The runtime reduction against the longer PLAIN_ALWAYS baseline and the missed improvements caused by conservative retention/early price stopping remain separate comparisons. The finite bootstrap is descriptive evidence under the declared synthetic source sampling and resampling scheme, not a simultaneous universal coverage theorem.

## Findings by severity

### P0

None.

### P1

None.

### P2

None.

### P3

None.

### Nitpicks

None raised. Packaging metadata was not converted into a new scientific blocker.

### Tests to add or update

None required for this frozen scope. Changed behaviors have targeted native/Python tests, complete cohort verification, and the independent raw reconstruction above. No additional native campaign, optional reviewer tree, or full historical rerun is requested.

## Limitations and original-objective assessment

This is an independent implementation/reconstruction review using the same preserved raw observations and machine, not an independent timing replication on another host. The parent-provided expected values did not serve as substitutes for raw calculation. The inherited C24 arithmetic-domain proof and unchanged numerical core were reused after identity checks; C09 did not repeat that entire historical proof/campaign.

The unchanged verifier's exact residual check is limited to the frozen nine TEST parents at additional 32, covering 108 action bounds from 42 distinct vector states. C09 inspected that selection and code path and reused its evidence; it does not label those residuals as independently recomputed here. General theorem validity follows the stated proof assumptions, not finite counts.

ORIGIN_WATCHDOG.md correctly restores both the original game-quality/resource objective and the general FOCUS/STOP idea. C25 covers a fixed complete action set with deterministic utility enclosures. It omits original EXPAND/MERGE, new-candidate reservoirs, learned MCTS observation semantics, and actual game-quality evaluation. CLAIMS T5 therefore remains UNRESOLVED. These are explicit transfer obligations, not defects blocking completion of this authorized SPD mechanism study.

## Handoff and checkpoint

- Active sequence: parent-owned theory R01–R10 then coding C01–C10; this child owns only C09.
- Current skill: bounded PhysMath C09 independent review with Change Review and computation-audit.
- Review Gate: PASS; no unresolved source/claim blockers.
- Evidence: C09_REVIEW_CHECKS.json PASS/exit 0, independent reconstruction and inspected source/proof paths.
- Next required work: **parent C10 final report and authorized publication/preservation**.
- First next check: preserve the scoped claims and negative primary result; bind the final package to frozen scientific inputs and the parent's restore evidence.
- Action: **continue now in the parent**, with this child returning the completed C09 artifacts. No new user permission is required by this review.
- Next / upcoming task: parent C10 finalization and preservation audit.
- Child stop boundary: C09 completed; no git/publishing work delegated to this child.

## skill_context_used

Actually read and applied:

- PhysMath Research Loop **1.3.0**: actual loaded package **skill://plugins_6acdfbba921881919e0c17843706aef4/physmath-research-loop**, including SKILL.md, references/CODING_LOOP.md C09, references/POLICY_INTEGRATION.md, and references/SUBAGENT_MODE.md. Applied finite review, declared write ownership, typed scientific identity, protected assumptions, and no recursive delegation.
- Keystone change-review: **skill://plugins_6a512fae91f881918208460aae465ff9/change-review/SKILL.md**. Actual review/checkpoint gate files were read from /home/agent/.codex/plugins/cache/openai-curated-remote/keystone/2.0.4/references/gates/. The cloud sibling-resource lookup was unavailable, resolved through the installed filesystem copy; no permission or execution blocker was introduced.
- Mathematical computation-audit: **skill://plugins_6a9174204a0481918ca3798d69d2e227/computation-audit/SKILL.md**, plus its installed references/checklist.md. Applied exact finite assertion, independent computation, documented command/exit, source coverage, outcome interpretation, and preserved provenance using the existing project evidence format.
