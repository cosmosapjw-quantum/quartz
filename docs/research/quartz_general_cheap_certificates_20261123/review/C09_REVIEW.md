# C24 C09 independent review

## Verdict

**Looks good for the stated finite SPD research release.**
No P0/P1 blocker or concrete unresolved P2/P3 defect was found in the reviewed mathematical, numerical, implementation or experiment path. This verdict applies to the conditional enclosure theorem, the admitted input/state domain and the preserved C24 experiment. It does not establish a universal speed claim or the original game-project P2/P3/P4 claims.

Reviewer: native agent /root/c24_review, one fresh bounded pass on 2026-11-23.
Reviewed C08 publication: df34b7ae1b867f4917c2d302ea97aca40cc55077.
The review creates only /tmp/quartz_c24_review* artifacts. It changes no production source, contract, test, result, Git index or branch and performs no publication.

## Contract and inspected scope

The load-bearing claim is that the ordinary stored-iterate response plus conservative scalar error bounds encloses q=b^T A^-1 b/S for A=I+rho L_D; strict historical-interval separation certifies the unique exact optimum. An uncertified fixed-budget policy must return the same ordinary final score/action and solver iterate as PLAIN_ALWAYS. Cost reduction is a finite measurement at matched exact decision utility, not a mathematical consequence of the enclosure theorem.

Inspected:
- theory/DERIVATION.md in full: primitive rounding model, operation-count lemma, stencil/dot/residual error propagation, positive-square-sum inversion, raw clipping, supported domain and no-overflow bounds.
- src/bin/quartz_c24_cheap_certificate.rs: complete production path, including serde entry, ordinary operation order, CG, scalar interval primitives, validation, both evaluators, history, every controller branch, diagnostics, timer and actual policy loop.
- CODE_CONTRACT.json and C04_RUN_CONTRACT.json, including the fixed nine-parent residual audit.
- coding/c24_sources.py, c24_run.py, c24_verify.py and c24_analyze.py in full.
- C07_PRE_TEST_FREEZE.json and TEST source/run/verification/summary/parent records.
- C06 debug/release27-pass logs and final Python10-pass log; these were read, not rerun.
- CODING.md C07/C08 claims and their limits.

## Mathematical and implementation assessment

1. The additive underflow floor is explicit. For this straight-line sum-of-products graph, each rounding epsilon propagates only through subsequent (1+delta) factors; at most k such errors and factors yield F_k=k sigma/(1-ku). The use of sigma=smallest NORMAL is deliberately loose but conservative. A sequential n-term dot is safely overcounted by k=2n. The stencil count9 covers the diagonal product and four product/subtraction pairs.

2. The implementation uses n B X and n X Y bounds for the signed dots and adds n X E_y to propagate operator error into x^T A x. The true residual perturbation n(2 R E_r+E_r^2) is valid for arbitrary admitted stored iterates and does not assume CG orthogonality. All nonnegative budget operations are outward, and denominators use the correct endpoint directions.

3. positive_square_sum handles the lower-numerator sign before division. Upper substitutes for gamma and the floor enlarge the positive-dot enclosure. Raw q clipping against[0,S] is justified by A>=I and the rigorous input-norm upper check. The residual upper takes square roots before division and uses the rigorous positive input-norm lower denominator.

4. Input constraints are enforced before Solver construction. The previous bmax=2^-600 failure is excluded explicitly; exact zero is treated separately. This closes the stated false-success case by domain restriction and is not arbitrary-range CG repair. State range, nonfinite arithmetic and empty intervals remain errors.

5. The ordinary fallback score uses the same ph/qh order in all evaluators. Evaluators receive immutable solvers. Block interleaving changes when a solver is evaluated but does not change its per-action CG update sequence. Strict separation uses >, preserving true ties. FAST_ONCE pays for an initial enclosure and returns to an ordinary final evaluation if not certified; the final upper/goal fields are null. PLAIN policies also keep those fields null.

6. The native timer starts before validation and covers allocation, initial work, readouts, continuation/controller and explicit destruction of solver/validation workspaces. JSON decode/serialization and offline oracle verification remain outside the declared timer. Every policy is actually executed with cyclic order and no cross-policy state cache. Typed ordinary and interval stencil counts correctly distinguish the two arithmetic paths inside OUTWARD evaluation; wall time is the primary cost measure.

7. Public tasks contain only id/side/rho/scale/rhs. Oracles are not read by the runner or sent to Rust. String transport and returned parsed bits provide a concrete byte-identity bridge. The independent verifier does not import generator, runner or solver; its exact graph-energy construction checks boundary terms and full component/action coverage. Exact q interval and fallback checks cover all traced TEST parents/budgets; true-residual vectors cover the stated nine-parent primary subset. That subset is not described as whole-population residual recomputation.

## Independent executed check

Command actually run:

    /home/cosmosapjw/quantum/research/quartz_general_meta24_20261123/venv/bin/python /tmp/quartz_c24_review_checks.py > /tmp/quartz_c24_review_checks.log 2>&1

Observed process3919665: exit0, tool-reported runtime0.93s.

The script imports neither production analyzer nor verifier. It independently:
- verifies every C07-frozen source/binary/dependency hash and each TEST run-input binding;
- reads raw TEST timing output and the separate exact oracle;
- reconstructs parent medians from nine actual repetitions for all216 parent/budget requests and11,664 policy executions;
- recomputes exact rational regret, wrong-decision count, strict-certificate count, actual saved-CG count and exact utility differences in all18 policy/budget summary cells;
- independently forms paired resamples and computes percentile endpoints by sorted linear interpolation;
- recomputes the primary price comparison and descriptive break-even versus PLAIN_STOP.

Maximum difference from recorded floating means/percentiles was9.094947017729282e-13us. The1e-9us comparison threshold is only for alternative floating aggregation order; exact rational utility/count assertions use equality and no physical/interval tolerance is widened. All observed input hashes are identical before/after.

Primary additional32 independent results:
- PLAIN_ALWAYS1896.1466805555553us; FAST_BLOCK1324.4306666666666us.
- Saved571.7160138888889us; paired-parent95% interval[299.6238027777778,865.1939336805556].
- Exact mean utility difference0; positive-regret decisions1/72 for both.
- FAST_BLOCK52 certificates,49 actual earlier exits,4,038 CG calls versus7,321.
- FAST_BLOCK minus PLAIN_STOP net at lambda1e-6: -0.0010419339760477828.
- Cohort descriptive break-even8.183883610962888e-9 utility/us.

Check result:
    /tmp/quartz_c24_review_checks.json
SHA256:
    7e11e014b160513e8a0615d03c89bc4dccc40fdda6e898d4c357bb4d57a40a53

The JSON binds exact input identities, command, runtime/software, observed heads and the script digest. The raw log and check source are preserved beside it.

## Findings

### P0
None.

### P1
None.

### P2
None found within the reviewed release contract.

### P3
None requiring current-task correction.

### Nitpicks
None reported; style-only remarks would not improve the present scientific result.

## Interpretation and remaining limits

- The finite checks and statistical reduction do not replace the conditional proof. Its IEEE round-to-nearest, gradual-underflow, fixed-order and no-fast-math assumptions remain part of the admissible theorem.
- The relative-versus-absolute underflow treatment is conservative, and arbitrary small-norm or huge-state support remains excluded. A future domain expansion requires a new contract and consistent solver arithmetic.
- Exact noninferiority is to PLAIN_ALWAYS at the same finite budget. It is not monotonic true utility relative to the initial choice. C08 explicitly preserves the single tight-gap primary case where continuation changes an initially correct choice to an incorrect one.
- The positive runtime contrast is measured on the frozen balanced manufactured cohort and recorded Ryzen5900X runtime. The bootstrap resamples independent parent problems, not the nine timing repetitions. It is a descriptive finite-cohort interval with heterogeneous sizes and no general-hardware or multiplicity-adjusted claim.
- A faster matched-budget policy can still lose economically to PLAIN_STOP at the frozen price; the negative result is retained.
- Grid-size cases are discrete problem sizes, not a continuum-convergence study. The original THESIS P2/P3/P4 game-engine claims are not established.
- No new native benchmark or repeat of the full exact-vector campaign was performed in this review. Existing finite verification was inspected and its evidence/source identity checked. Parent-owned archive restoration and final preservation/publication checks are separate and were not represented as completed by this reviewer.

## Tests to add or update

No additional current-release test is required by an evidenced defect. Existing support-boundary, tiny-rejection, ties, zero states, nullable diagnostics, actual-work, fallback-state and exact-oracle checks address the load-bearing changed behavior. Any future arithmetic/domain or policy expansion must invalidate and recheck the affected contracts.

## Skill context used

Actually read:
- PhysMath Research Loop1.3.0 SKILL.md; CODING_LOOP C9; POLICY_INTEGRATION; SUBAGENT_MODE.
- Mathbox computation-audit4.0.0 SKILL.md and references/checklist.md.
- Keystone change-review2.0.4 SKILL.md.
- Keystone references/gates/review.md, references/gates/checkpoint.md and references/handoff-packet.md from the installed local package after cloud resource lookup for the first two failed.

Applied a single independent, read-only pass, finite assertion/proof separation, typed identities and bounded raw-data recomputation. No recursive reviewer was spawned.

## Review gate and handoff

Review Gate: PASS for reviewed scientific/implementation scope; no unresolved blocker.
Source skill: change-review.
Target: parent-owned shipping/C09–C10 closeout.
Goal: preserve review evidence, finish actual restoration, and publish the authorized C24 result.
Authorization snapshot: user said “매 단계마다 백업과 게시를 병행해줘.” The existing authorized action set is C24 stage backup and publication; it does not add a new research cycle or mutate the protected original engine.
Files: reviewed production/evidence are read-only to this child; parent may copy the /tmp review artifacts into C24 evidence.
Risks: finite-domain/hardware/price limitations above; archive restoration is independently owned by parent.
Next check: read this report/check record, verify parent restore result, then perform C09 and C10 checkpoint/publication.
Action: continue now through parent; no new user approval is required.
Todo tail: Next / upcoming task: parent C09 evidence integration and successful restore, then C10 release/handoff.
