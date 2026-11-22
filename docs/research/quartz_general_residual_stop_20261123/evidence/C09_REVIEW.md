# C23 C09 — bounded independent scientific/code review

Reviewer: native agent /root/c23_review. One review pass; no recursive reviewers.
Production source and benchmark outputs were not edited. The parent owns release and claim admission.

## Outcome

**No BLOCK_NOW finding for the frozen manufactured C23 cohort.** The residual-energy derivation, outward interval construction, strict decision certificate and shared-state fallback are consistent with the inspected code and independent finite checks. One concrete norm-underflow diagnostic defect remains outside the measured source range; it must be retained as a limitation rather than represented as a fully general accepted-input guarantee.

The measured efficiency result is negative against PLAIN_ALWAYS at the primary additional budget 32. Genuine solver work is withheld, but interval/controller costs exceed that saving in this comparison. This does not imply a general impossibility result.

## Skill context actually used

- PhysMath Research Loop 1.3.0, cloud c43/physmath-research-loop: SKILL.md, POLICY_INTEGRATION.md, CODING_LOOP.md C9 and SUBAGENT_MODE.md. Applied one bounded pass, exact versus finite evidence, typed identity, parent-owned admission, closed prior findings and no recursive delegation.
- Mathbox computation-audit 4.0.0, cloud c42/computation-audit: SKILL.md and references/checklist.md. Applied independent finite arithmetic, actual iterator coverage, input/output identity, resource scope and conservative interpretation.

## Reviewed objects and bindings

Read THEORY.md, CODING.md, coding/C05_PROTOCOL_AMENDMENT.md, the complete new Rust binary, c23_sources.py, c23_run.py, c23_verify.py, c23_analyze.py, C06_BINDING.json, frozen SOURCE_SPECS.json and C08 result/manifest/verifier records.

- Rust source SHA256: 8a5aad2d9c8b0a9fd8df0efd5310604f7b498d16447df0f222aa9619a15ad8ed
- Release binary SHA256: 812c57e75ad193fa4cca3729fc63ee59df73cac7118c50886adb4c0b47118ac0
- TEST execution source HEAD: c918fbad6255a7e3a1f0a3be3f1e70cbd409326c
- TEST public source SHA256: f44a318e8c8ff2e657eae0976788e212dd6b478dbd4f03273ce90bfb6e55a30c
- TEST timing SHA256: 91f4dc48591cf56787b9ca1d4eba8cbc8a8aaf9b31596b8834c5037b5395f027
- TEST trace SHA256: 9fdcc49d8b0d64832a3a0b7dd88f99b84d4caf7b6a7d84a3cf65a2cd1496d396

The binary/source hashes were unchanged across the independent native checks. Raw TEST sources and both result artifact hashes were independently recomputed and matched the manifests. Archive/chunk restore publication remains parent-owned; this review did not transform trace data.

## Independent checks actually executed

The reviewer script at /tmp/quartz_c23_review.py imports no production generator, solver, verifier or analysis module. Its complete source and the separate commands are preserved inside C09_REVIEW.json, so the temporary path is not the only reproduction record.

1. Independently assembled zero-exterior operators using undirected grid edges plus boundary terms. Used exact Fraction Gaussian elimination for small arbitrary RHS, exact edge-energy manufactured cases, a reflected tie, rho=0 and zero RHS.
2. Executed seven finite tasks through the frozen native binary: 42 policy executions, 457 input components checked by actual native bits, 682 exact q interval containments and 108 final-state outward-bound checks. The latter re-evaluate ell and the true residual norm exactly from the returned f64 iterate, independently of native arithmetic.
3. All 10 strict exits were exact-oracle optimal. Four noncertified comparisons matched ALWAYS's complete final state, iteration counts, action and final ell. All seven PLAIN_ALWAYS comparisons matched ALWAYS's complete final state and ell. The wide-gap case actually used fewer CG matvecs. The reflected tie never certified.
4. Independently read all 216 TEST timing envelopes. They cover precisely 72 frozen parents times budgets 8/32/128, two parents in each of 36 strata. Each of six policies has 1,944 executions; each parent/budget has exactly reps 0 through 8 with trace copies disabled. Canonical task hashes and artifact/binary bindings matched.
5. Recomputed exact gate-versus-ALWAYS utility comparisons for all 432 parent/budget/gate pairs. Independently reconstructed parent medians, regrets, certification/early-exit counts and 36 paired mean/bootstrap time summaries. All agreed with C23_TEST_SUMMARY.json to floating rounding precision.

Actual process exits, obtained from Remote Desktop Commander read_process_output:

| Check | PID | Exit | Runtime |
|---|---:|---:|---:|
| Seven-task independent native/Fraction script | 3298327 | 0 | 0.12 s |
| TRAIN/CAL norm scan and tiny-input enclosure check | 3300438 | 0 | 0.42 s |
| TEST coverage, bindings and independent aggregation | 3303835 | 0 | 1.51 s |

These are finite implementation checks, not a proof of universal f64 behavior. The exact-real theorem follows from q-ell=e^T A e=r^T A^-1 r and the spectral endpoints; the floating certificate remains conditional on the stated IEEE operations/no-fast-math contract.

The existing C08 independent source/trace verification is reused rather than rerun: 741,888 exact source components, 216 trace transport requests, 26,898 trace interval containments and 104,976 timing-output interval containments. Its reports are C23_VERIFY_TEST_TRACE.json and C23_VERIFY_TEST_TIMING.json.

## C09-F01 — norm underflow can falsely report a residual goal

**Priority: FIX_SOON before general domain extension; not BLOCK_NOW for frozen C23.**
Object: Solver::new/step and enclose_response/ordinary_response residual diagnostics in src/bin/quartz_c23_residual_gate.rs.

Reproducer: side=2, rho=1, common scale S=2^-1000, RHS arms
b0=(2^-600,0,0,0) and b1=(0,0,0,0), additional=8, one repetition, trace enabled. The exact JSON is preserved in C09_REVIEW.json.

Observed native output for RESIDUAL:

- iterations=[0,0], final x=0;
- residual_relative=[0,0];
- residual_goal_met=true and reason=RESIDUAL_GOAL;
- certified=false.

The nonzero arm has exact relative residual ||b-Ax||/||b||=1. Although b is a normal nonzero binary64 value, its square 2^-1200 underflows. The ordinary/recurrent sums become zero and the zero-norm diagnostic branch declares success.

This is a numerical underflow defect in field-accuracy/convergence diagnostics. It does not invalidate the outward decision certificate in this example: all 76 checked enclosures contain the exact normalized utility
u0=(23/105)2^-200 and u1=0; no gated policy certified.

The frozen source ranges are far from this reproducer. Minimum initial squared RHS norms were 416.5525 in TRAIN, 345.6977 in CAL and 303.6543 in TEST. In TRAIN's 5,832 timing outputs, no fresh relative residual was reported zero; the smallest positive reconstructed ||r||² from the reported relative residual and input norm was 5.5653e-27. Independent final-state residual recomputation for the first TRAIN/CAL trace gave positive minima 1.6699e-14 and 1.0154e-14 across 18 arm states per split.

No CAL_TIMING artifact was available: the first diagnostic query raised FileNotFoundError, then the targeted existence-aware command completed with CAL input coverage and sampled trace coverage only. CAL timing-wide diagnostics are NOT_RUN, not counted as a pass.

Minimal correction: preserve the frozen C23 measurements and explicitly limit the residual-goal claim to the studied range. Before broadening the supported domain, use scaled sum-of-squares with consistent solver rescaling, or validate/reject unsupported nonzero norm ranges. The regression must make this input either report true relative residual 1 at x=0 or be explicitly rejected; it must not return a met residual goal.

## Result interpretation that must survive closeout

Primary budget means common initial four iterations plus up to 32 additional iterations per arm. Times below are arithmetic means of the 72 parent medians, each median from nine physical runs; they are wall microseconds.

| Policy | Mean parent median us | Wrong decisions / 72 | Certified | Earlier than ALWAYS |
|---|---:|---:|---:|---:|
| NEVER | 835.32 | 10 | 0 | 72 |
| ALWAYS | 2752.76 | 2 | 0 | 0 |
| PLAIN_ALWAYS | 1579.51 | 2 | 0 | 0 |
| CERT_ONCE | 2434.33 | 2 | 17 | 17 |
| CERT_BLOCK | 3061.35 | 2 | 52 | 49 |
| RESIDUAL | 4680.61 | 0 | 0 | 40 |

CERT_ONCE saves 1,450 total solver iterations and CERT_BLOCK saves 3,236 versus ALWAYS. CERT_ONCE saves 318.43 us versus ALWAYS, descriptive paired bootstrap interval [92.13,607.57] us. Against PLAIN_ALWAYS its saving is -854.82 us, interval [-1204.99,-513.25] us. CERT_BLOCK's corresponding saving against PLAIN_ALWAYS is -1481.84 us, interval [-2163.24,-844.21] us. The efficient comparator changes the conclusion; reporting only ALWAYS would overstate efficiency.

The strict certificate guarantees exact optimality when it fires. It does not turn every finite-budget outcome into a certificate: the two wrong decisions at primary budget are noncertified fallbacks. CERT_BLOCK's 52 certified parents and 49 actual earlier exits are distinct quantities.

At lambda=1e-6 utility/us, net differences from measured NEVER are -0.0015477151 for CERT_ONCE and -0.0021747371 for CERT_BLOCK. Neither gate demonstrates positive priced continuation value under this selected cost scale. NEVER itself computes an initial outward enclosure, so it is a measured control, not an optimized bound-free STOP policy. This limits positive economic claims; no post-TEST policy addition or retiming is needed to support the current negative conclusion.

Budget128 has zero measured regret for the fixed and gated policies, but does not establish a general speed advantage. At that secondary budget CERT_BLOCK's mean saving relative to PLAIN_ALWAYS is negative and its interval crosses zero. Do not replace the frozen primary budget with a favorable subgroup or secondary result.

Bootstrap units are parents, not repeated timings. The intervals are descriptive summaries of a balanced manufactured mixture; they do not establish a production workload distribution. Timings include the declared native validation and initial-work costs, and exclude decoding/source/oracle work. RESIDUAL has a different field-accuracy stopping contract. No cycles, continuum-physics, universal-fastest-solver or original THESIS P2/P3/P4 claim follows.

## Closed prior findings and transition

The numeric-JSON input mismatch is closed for the frozen study's string/FromStr transport and checked trace bits. Numeric compatibility parsing is not asserted to have exact source identity. The inflated fixed comparator risk is addressed by PLAIN_ALWAYS, which is included in the reported negative comparison. Shared final-state fallback is supported by code structure, independent small checks and preserved full TEST trace verification.

C8 results and identities have been received and checked; no C8 obligation remains for this reviewer. Parent action is to integrate the one concrete accepted-domain limitation and the bounded negative runtime conclusion, then complete the already-authorized C9/C10 backup/publication. No additional review pass or benchmark rerun is requested.
