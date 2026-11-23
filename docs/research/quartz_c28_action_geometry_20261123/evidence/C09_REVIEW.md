# C09 — independent scientific code and computation review

## Verdict and exact scope

**PASS_SCOPED for the C28 mathematical implementation and retained finite evidence, with one low-severity wording correction.** I found no blocking defect in the pinned direct/lifted H2 kernels, the actual-game bridge, the independent finite-world checker, or the primary cost aggregation. The algebra is valid under the stated assumptions. The retained native results support finite implementation equivalence and the reported synthetic cost reduction. The resident-game measurements support near-equal measured cost on this fixed cohort.

The original C08 wording “timed-game summary mismatches” overstates which summaries were individually compared. Finding C09-F1 below preserves the precise issue. The parent accepted and applied the correction during this review; publication remains the parent's responsibility. This verdict does not depend on another benchmark or a scientific-source change.

Review identity:

- User SSOT: `87c15dc828d6eea519c219fd8a412fa21dfc464d`.
- Pinned review commit: `f791e96e651d1849df601dcd7b98c18bcc3fa082`, tree `13c2c0b8a25ce269b3963b12cc19a93ba68e55ee`.
- Branch observed at entry: `research/quartz-c28-action-geometry-20261123-486e`; the SSOT was verified as an ancestor.
- Native source SHA256: `c396aad885132e7457fe99b4b16c8bdccff2067427cb1db7f2ed05d13d1603de`.
- Current release binary SHA256: `f8ab538dd1511244963b60c79f8da12f1c4e7e11ef63443087d8f35ca29f607a`.
- Remote worktree: `/home/cosmosapjw/quantum/research/quartz_c28_action_geometry_20261123_486e`.
- Device: `edd1b263-56a0-4391-8a2a-677dd00eabf5`.

I reviewed the frozen contract/interface, canonical theory, complete native source and inline tests, Python tests and campaign, complete-world oracle, raw C06/C07/C08 evidence, native execution receipts, and C06–C08 interpretations. I read the C08 interpretation after independently inspecting and reaggregating its measurements. The origin watchdog was used to assess the current bounded claim; the historical project was not reopened.

Execution status is deliberately separated:

- **Native production execution in this review:** not run. Existing native runs were audited against their retained logs, inputs, source and binary identities.
- **Fresh review calculations:** completed. Independent raw-data reaggregation and exact contingent-plan reconstruction are recorded below.
- **Scientific evidence state:** derived algebra, numerically checked native outputs in the recorded finite range, and implementation-verified recorded protocol/game behavior.
- **Reviewer changes:** this review and C09 review receipts only. No scientific source, tests, contract, baseline, raw timing data, Git ref, or other worktree was changed by the reviewer.

## Contract and independent mathematical reconstruction

The state is a known incumbent \(b\), an unobserved candidate set \(S\), and a binary latent mean \(m\). Utilities lie in \(\{0,1/2,1\}\); query costs are fixed and nonnegative. A query reveals its candidate exactly, removes it, and permits deployment only among known values. The horizon is two additional queries. The signed family uses normalized \(p_i\), mean-centered bounded features \(f_i\), and \(|r_i|<1\).

For a supported outcome write \(t=r_i f_i(u)\), \(D=1+mt\), \(N=m+t\), and \(b_u=\max(b,u)\). Since \(|m|\leq1\) and \(|t|<1\), \(D>0\). The conditional posterior is \(N/D\), and the branch probability is \(p_i(u)D\). Consequently,

\[
p_i(u)D\max\!\left(0,\max_{j\ne i}\{\alpha_j(b_u)+(N/D)\beta_j(b_u)\}\right)
=
p_i(u)\max\!\left(0,\max_{j\ne i}\{D\alpha_j(b_u)+N\beta_j(b_u)\}\right).
\]

The equality follows from multiplication of every entry, including STOP, by the same positive \(D\). Adding the first-query gain \(\alpha_i(b)+m\beta_i(b)\) gives the two native kernels. The argument requires no cancellation through a zero likelihood and no assumption that the second action is fixed before the reveal. Unsupported \(p_i(u)=0\) branches are omitted. Fixed costs and the update \(b_u\) are essential.

The native Prepared rows [b, max(b,0.5), 1] are exactly \(\max(b,u)\) for the three supported outcomes, including off-grid initial incumbents. Both methods construct the same AtomLaw, \(\alpha\), and \(\beta\) arrays; the direct method divides once per first-outcome branch and reuses that posterior over all second candidates. Both exclude the queried index. This is the intended matched cached comparator.

As a second reconstruction, I wrote a separate exact-Fraction program that enumerates complete deterministic contingent plans and then takes their expectation over complete worlds. It imports neither the native implementation nor the project's world-recursive Bellman oracle. On the reference family, an observation of A gives a latent mean of \(\pm4/5\). In the signed case, one target then has net H1 gain \(11/100\) on either branch. Thus \(K_A=-1/100+11/100=1/10\); starting with a target gives \(-1/20+(1/2)(11/100)=1/200\). With common loadings, only one A branch offers the \(11/100\) continuation, giving \(K_A=9/200\), while a failed target does not create a profitable second target.

The independent program checked nine retained cases: five signed/common/product and cost-crossing fixtures, the exact \(\pm19/24\) crossings, and two certain-latent near-unit-loading cases. It recovered the exact reference gains, including tied \(13/120\) winning gains at the two crossings. Maximum comparison residual was \(5.551115123125783\times10^{-17}\) utility units. For the stress request, rational reconstruction uses the serialized decimal loading; the main fixtures and the named boundaries use their exact rational values. This distinction is explicit rather than conflating decimal, binary64 and original rational identities.

## Numerical choice, support and error scope

The choose function implements the frozen rule: STOP when the maximum raw gain is at most \(10^{-12}\); otherwise choose the smallest action id within \(10^{-12}\) of the maximum. PlanResult.value uses the raw optimum \(b+\max(0,\max_i K_i)\), independently of that epsilon-selected action. The Python checker separately verifies raw values and exact-rational epsilon choices; it does not infer action correctness merely from the \(5\times10^{-12}\) score tolerance.

This distinction also applies to H2 continuation: scores describe the raw optimal Bellman value, while the executed H1 decision uses the epsilon rule. With exact scores, the implemented two-step selection can lose at most \(2\times10^{-12}\) in modeled expected value relative to the raw optimum: at most one epsilon at each decision. Floating arithmetic adds its numerical error. This bound concerns the declared model and fixed costs, not realized game outcomes or model calibration.

The support/endpoint implementation is consistent with the proof:

- Synthetic probabilities are finite/nonnegative and normalized only inside the admitted \(10^{-12}\) sum tolerance; point masses and zero atoms are supported.
- AtomLaw::new uses only positive-support atoms for its scale and sets zero-mass/point-mass tilt to zero.
- Prepared::scores skips zero-mass branches. Empty and unary sets and \(b=1\) have the correct raw values.
- Belief::observe rejects duplicate, out-of-range, unsupported and off-support observations before changing its state. Certain-latent endpoints remain fixed; the interior projection only tolerates the stated small arithmetic overshoot.
- Actual-game laws are strictly positive before nonterminal reveals; terminal-known laws are point masses. The frozen game loading \(0.75\) keeps every supported likelihood denominator at least \(0.25\).
- Production uses binary64, not certified intervals. The API admits loadings closer to one and costs larger than the frozen synthetic validation range. The evidence does not establish a uniform error theorem over that entire API domain. Near a discontinuous epsilon-selection boundary, a score tolerance alone cannot certify universal choice identity. The reviewed finite choices were checked explicitly.

No tolerance change is needed for the recorded results.

## Actual-game path and oracle independence

The root-frame convention is consistent across `TicTacToe::outcome`, `root_value`, minimax and utility conversion: terminal outcome is expressed in the terminal side-to-move frame, so multiplication by root player and current player recovers the root frame. The exact solver charges every entered node, including terminal entries, and does not prune. The Python minimax uses board winners directly; its memoization accelerates the checker while its returned logical node count still sums every traversal occurrence.

The initial latent laws are constructed **before** the mandatory initial solve. Terminal-known children are observed with point-mass zero tilt. The actual initial solve is then conditioned using its original nondegenerate law, even when it does not become the incumbent. The game loop evaluates H2, observes the selected exact utility, removes the candidate, then evaluates H1. It stops on a winning incumbent, an empty remaining set, or explicit STOP, and permits at most two additional queries. Strict improvement is required to replace the incumbent, preserving it on equal reveals.

All cue nodes, mandatory initial-solve nodes, and additional exact-solve nodes enter `charged_nodes`. The node price is (1/1024) utility per node. Elapsed nanoseconds remain a separate measured quantity. The predicted fixed query-node cost and the subsequently observed exact solve count are not interchangeable quantities.

The runtime request structs reject unknown fields, including model-level oracle fields. Runtime utility revelation occurs through terminal child inspection or the actual exact solver. The Python checker creates offline game labels and never sends them to the native controller.

The Python scientific oracle is substantively independent of the native algebra: it constructs the full two-component joint world law with Fraction arithmetic, conditions by filtering worlds, and recursively evaluates terminal utility and costs. The Rust world-recursive test supplies additional structural checking but shares `AtomLaw` with production; it is less independent for detecting feature-construction mistakes. The separate Python exact law and this review's explicit plan enumeration address that shared-code risk.

Trace completeness was inspected directly. Besides per-decision comparisons, the checker rejects a trace ending while the incumbent is below one, candidates remain, the two-query cap is unspent, and no explicit STOP was recorded. The executed Python test log includes the negative controls for omitted active H2 and H1 decisions.

I independently reconstructed one nontrivial recorded root, `C26-p4-0059adf99c0886cf`, from its board, seeded rollouts, model lookup, full exact minimax and contingent-plan expectations. Initial solve move 0 loses and costs 38 nodes; cues cost 73 nodes. H2 chooses move 7, H1 then chooses move 8; their exact solve counts are 47 and 24. The final known move is 7 with utility (1/2), and the charged total is (73+38+47+24=182). The reconstructed initial posterior is approximately (-0.6346153846153847), so this case detects replacement of the initial reveal's prior by a point mass. Its H2 and H1 scores agree with the retained native output within (2.7755575615628914 × 10^{-17}). This fresh reconstruction supplements, rather than relabels, the existing 300-root check.

## Actual coverage and provenance

The read-only raw audit verified all nine retained gzip artifacts against both compressed and uncompressed hashes. It matched all 27 metadata code/input/binary identities to the pinned Git blobs or the current binary. Current native source matched the pinned source. `src/game.rs`, `src/games/tictactoe.rs`, `Cargo.toml` and `Cargo.lock` matched the user SSOT byte-for-byte. The stated frozen C26 source hash was independently recomputed.

| Evidence | Actual visited range | Interpretation |
|---|---|---|
| C06 raw synthetic verification | 108 cases: 5 fixtures, 10 boundary cases, 45 edge cases, 48 seeded rational cases; 39 H1 and 69 H2 cases; 0–5 candidates | Every retained action score/raw value and exact epsilon choice was checked. Maximum recorded oracle residual (2.220446049250313 × 10^{-16}). |
| C07 raw game verification | All 300 source ids, histories and cue seeds in exact source order; 200 positive-root and 100 negative-root cases; at most 5 legal candidates | 107 active roots, 193 saturated roots, 107 H2 and 103 H1 decisions, 210 queries. Maximum recorded world-oracle residual (1.6653345369377348 × 10^{-16}). |
| C08 synthetic benchmark | Full (8 × 5 × 2 × 2=160) labeled configuration product | 80 configurations with n ≤ 5 use independent full-world verification; 80 with larger n use direct/lifted numerical agreement. |
| C08 game benchmark | All 300 recorded roots and frozen model | Separate unmeasured summaries checked; 9,300 timing pairs retained. |

The 160 labeled synthetic configurations contain 150 distinct semantic requests: at (n=1), the alternating-sign pattern equals the common pattern. Moreover, probabilities/magnitudes/costs are generated once per candidate count and reused across belief/incumbent/pattern settings. These are not 160 independently sampled models. The prescribed configuration count is nevertheless correct; no iterator stride, filter or early exit silently removed a contract cell.

All 300 C07 verification rows say `complete_world_status=CHECKED`, but only 210 actual decision states required Bellman comparisons. The other 193 roots start with utility one. There were no explicit STOP decisions among the 210 active game decisions. STOP behavior is instead exercised by the synthetic and protocol controls. C07 already reports the active/saturated distinction correctly.

The native receipt records 15 Rust tests on the earlier source hash, followed by the strengthened initial-reveal test on the final source hash and a final release build. The 17-test Python suite and C06–C08 native campaigns bind the final source/binary hashes. I did not recast the first 15-test run as a new all-tests execution on the final hash or as a whole-repository pass.

## Timing audit and scientific interpretation

The source boundary matches the frozen contract. Synthetic timing includes law/coefficient preparation and planner evaluation on each iteration, with `black_box` preserving work. Game timing starts before cue generation and includes model predictions, belief/cache preparation, mandatory verification, posterior maintenance, planning, all exact queries and final selection. Parse/validation/history reconstruction, transport and serialization remain outside the named resident-controller boundary. Detailed decision arrays are collected separately.

The raw audit checked all 11,060 timing pairs, repeat indices, alternating order, positive integer ns fields and iteration counts. Synthetic totals are divided once by 256 or 32; game samples use one iteration. All 460 root/configuration medians and all 18 prespecified summary aggregate groups reproduced **exactly** from raw timings.

| Primary scope | Direct mean of medians, ns | Lifted mean of medians, ns | Direct minus lifted, ns | Lifted/direct |
|---|---:|---:|---:|---:|
| 160 synthetic configurations | 2101.5375244140623 | 1870.1374755859374 | 231.400048828125 | 0.8898901180017509 |
| 300 resident game roots | 2954.9266666666667 | 2954.6666666666665 | 0.26 | 0.9999120113527915 |

The synthetic path has about an 11.01% reduction in this fixed weighted collection. This is a measured property of the compiled matched implementations and tested configurations; both algorithms remain quadratic for fixed support.

A fresh read-only order stratification gives game differences of (+1.11) ns for direct-first samples and (-0.8) ns for lifted-first samples, using within-root/order medians. Thus the already tiny game effect changes sign with the order stratum. This supports C08's refusal to adopt a meaningful whole-controller speed advantage. It does not change the frozen primary aggregation. Synthetic order strata retain the same reduction direction.

The campaign is one resident process with fixed request order, alternating methods, specified warmups, CPU affinity 11 and one native thread. It makes no CPU-exclusivity claim and supplies no fresh process-level or population inference. Historical TRAIN/CAL/TEST labels do not create a fresh confirmatory test from these already observed roots. No additional timing campaign is required to accept the bounded engineering result.

## Findings and minimum correction

### C09-F1 — individually compared summaries are outside the timed repeats

**Severity:** low, documentation/provenance precision.  
**Target:** `stages/C08.md` (“zero timed-game summary mismatches”) and `evidence/c08_benchmark/analysis.json` key `timed_game_summary_mismatches`.  
**Evidence:** native `game()` collects direct/lifted summaries in separate unmeasured `run_game` calls. Each `time_game()` call passes its result through `black_box` and discards it; only elapsed ns is retained. The campaign checks the request's returned unmeasured summaries.  
**Impact:** the recorded zero count applies to 300 benchmark-request summary comparisons, not 9,300 individually serialized/compared timed-result pairs. The timed body is the same deterministic execution path with the same immutable inputs, which is supported by source inspection and the separate semantic checks.  
**Minimum correction:** rename or define the derived field as `benchmark_request_summary_mismatches`, and state that repeated timed summaries were not serialized or individually compared. Preserve raw requests/responses/timings, metrics, binary and source. No new benchmark is needed.  
**Disposition at review completion:** parent explicitly accepted and applied the correction in `evidence/C09_CORRECTIONS.json`, the derived `analysis.json`, and an explicit C09 note in `stages/C08.md`; publication remains the parent's responsibility. This review preserves the finding against the original pinned C08 snapshot.

### Nonblocking coverage qualifications

The 150 distinct synthetic requests, 107 active game roots and absence of active-cohort STOP decisions specify what the existing evidence covers. They do not require changing the frozen cohort or adding a new gate. The universal algebraic result rests on the positive-likelihood derivation; universal binary64 choice stability, empirical signed-game dependence and new-domain efficacy remain outside the finite assertion.

## Original-motivation check

The implemented construction connects the original contrast-variance intuition to paid action crossings, and connects the algebraic lifting to a matched native computation. It preserves known-only deployment and accounts for actual cue/solve node work while measuring controller wall cost separately. These are concrete contributions to resource-priced computation allocation.

C28 has not established improved playing strength at matched real CPU budgets, learned signed dependence in games, a deployed action-retention certificate, P2/P3 completion, or physical/quantum equivalence. Its own contract and C08 disposition retain those boundaries. The next substantive research obligation is an empirically justified contrast model and genuinely new resource-matched decision evidence; this review does not prescribe another audit loop.

## Review receipts and actual skill context

- `evidence/C09_REVIEW_RAW_AUDIT.json`: fresh pinned-blob/hash/iterator/raw timing and aggregation audit; status `PASS_SCOPED_RAW_AUDIT`.
- `evidence/C09_REVIEW_RECONSTRUCTION.json`: nine exact plan fixtures plus one independently reconstructed game root; status `PASS_SCOPED_INDEPENDENT_RECONSTRUCTION`.
- `evidence/C09_REVIEW_PROGRAMS.json`: exact review program text, commands and SHA256 values for reproducibility.
- `evidence/C09_REVIEW_RAW_AUDIT.py` and `evidence/C09_REVIEW_RECONSTRUCTION.py`: byte-identical durable copies of the executed scratch scripts. The copies were not rerun; the recorded scratch executions remain the execution evidence. No native production process was launched.

Actually read and applied (`skill_context_used)):

1. PhysMath Research Loop, package `skill://codex-home-skills/home/agent/.codex/skills/remote-skills/skill-6aa0657100b08191b7d67482d1f97c1d`: `SKILL.md`, `references/RESEARCH_POLICY.md`, `references/CODING_LOOP.md` (C09 and relevant neighboring evidence obligations), and `references/EVIDENCE_AND_HANDOFF.md`. The retrieved main instruction states built-in protocol 1.0.0; historical files' “1.3.0” wording was not adopted as a newly observed skill version.
2. Mathbox computation-audit, package `skill://plugins_6a9174204a0481918ca3798d69d2e227/computation-audit`: `SKILL.md` and `references/checklist.md`.

One fresh reviewer performed this bounded pass; no reviewer was spawned. The project evidence format was reused. No external manifest-schema validation was run or claimed. Git publication, backup restoration and C10 integration remain the parent's responsibility.
