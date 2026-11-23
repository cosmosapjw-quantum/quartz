# C24 Coding and scientific execution

## C01 — Goal, baseline and acceptance

Theory R01–R10 is complete. Implement the selected scalar-envelope evaluator and explicit tiny-norm rejection in one isolated new Rust binary, preserving the frozen C23 binary/source/results and original engine/Cargo files. Parent/head/worktree/protected identities and actual venv/runtime are in BASELINE.json.

Correctness acceptance: no false strict certificate in exact-oracle checks; every noncertified fixed-budget gate has the same solver state, iterations and final ordinary ell/action as PLAIN_ALWAYS; true upper residual diagnostics are distinguished from ordinary estimates; the known tiny nonzero input is explicitly rejected. Exactness is Fraction-to-outward-endpoint containment without tolerance expansion.

Efficiency is an empirical claim, not a prerequisite for a completed experiment. Compare consumed CG calls, readout cost and total wall time at the frozen source/decision objective. A valid but slower method will be retained as a negative result. The historical unchanged main regression617pass/0fail/89ignored is evidence from C23, not a newly executed C24 pass; no shared engine/dependency change requires repeating that entire historical suite.

## C02 — Formula, data and executable mapping

New target src/bin/quartz_c24_cheap_certificate.rs. Keep C23 operator/solver/transport mechanisms and deterministic ordinary score order. Cheap validation -> cached max B, positive-dot input norm[BL,BU],normal power-of-two S check,and supported-domain errors. The primitive scalar budget helper -> upward gamma_k,F_k. Common state validation reduces max X in a separate array pass. Fast evaluator -> one ordinary yhat/rhat/P/Q/D pass with max Y/R, then Ey,Er,EP,EQ,positive-dot D0 interval,T,ell interval,raw clipping and normalized[lo,hi].
The exact formula and field semantics are CODE_CONTRACT.json and theory/DERIVATION.md. Plain readouts have trivial[0,1] bounds plus the ordinary score, no interval evaluation count and null residual_upper/goal. Fast or full-outward readouts have valid intervals and a certified fresh residual upper. If FAST_ONCE continues to a final plain readout, that final diagnostic is likewise null.
C24 source manufacture reuses C23's validated dyadic construction with new source IDs/seeds and independent exact edge-energy verification. The native controller receives only public task fields; oracle labels remain in separate compressed files. Native parsed input bits,final state,checkpoints and policy work counters are retained in trace output; trace copying is disabled for timed measurements.

## C03 — Numerical design and concrete failure classes

Truncation error from incomplete CG is captured by ell plus true-residual energy bounds. Rounding error from operator,products,summation and residual evaluation is bounded by the proved scalar envelopes; sigma is a derived conservative underflow floor, not a tunable epsilon. Empty intervals/nonfinite states/domain violations are errors, not certificates.
Reject nonzero RHS maxima below2^-400 before constructing Solver, allowing exact zero tasks separately. Validate the supported state ratio before response use. The raw q interval is clipped against[0,S] before division; conservative residual norm uses sqrt(DU)/sqrt(BL) outward. This repairs the demonstrated false-success fixture by an explicit supported-domain contract and does not claim arbitrary-range CG.
All fixed-budget paths use the same initial4 and common recursive exhaustion threshold1e-14. Full residual policy uses recursive1e-8/cap512 and records whether the certified fresh residual upper actually meets the goal. Evaluation cannot mutate a solver state. Strict ties finish uncertified unless another action is uniquely better; equality is not relaxed.
Numerical/domain errors,implementation defects,runtime failures and performance outcomes are recorded separately. Native IEEE default compilation and actual flags are bound in runtime evidence; no fast-math or price/checkpoint tuning is introduced.

## C04 — Frozen execution DAG, interfaces and resources

SOURCE_SPECS.json binds144 disjoint fresh parents and all budgets,policies,repeats,prices,bootstrap seed. CODE_CONTRACT.json fixes the native JSON-lines/diagnostic interface before delegated code edits. C04_RUN_CONTRACT.json records actual hardware/compiler flags and the predetermined primary-budget TEST subset for exact stored-iterate residual-upper checks. All TEST oracle interval checks still cover the entire TEST cohort.

DAG: theory+specs publication -> behavioral RED/GREEN implementation -> focused debug/release and native UNIT exact pipeline -> TRAIN/CAL source generation/publication -> trace/verification and TRAIN timing -> pre-TEST code freeze -> new TEST source publication -> raw trace+timing -> exact source/interval/fallback checks and parent statistics -> one independent C09 review -> base-plus-overlay restore -> C10 release.

Resource ceiling: single-thread Rust,side<=96 for cohorts,K3,additional<=128,field cap512,one warmup+nine repeats,stream one parent/request; no GPU/HPC/service purchase. Finish compilation and source generation before timed policy runs. Reuse unchanged exact source machinery and parent theorem instead of rerunning old campaigns. Exact large-source q containment covers all traces; exact full-state residual-upper checks cover the frozen primary subset plus focused native cases, rather than repeatedly rebuilding every exact vector for every timed repetition.
Write ownership: Rust child alone owns new binary; data child alone owns new coding/c24_*.py,test_c24_sources.py and their focused evidence. Parent owns THEORY/CODING/contracts,canonical publication and benchmark dispatch. One fresh C09 reviewer later inspects important code/math/measurement risks; no recursive review.


## C05 — Complete implementation with observed behavioral failures

The new Rust binary implements all six frozen policies, the scalar-envelope response, supported-domain errors, nullable plain residual diagnostics, full trace/native-bit transport and inclusive per-policy timing. Production source SHA256 is 5a7cc595c267d5337f268045d1df8bf5909093ed7920ee50293da76da94cd381. The unchanged C23 solver and deterministic ordinary readout order are preserved; no engine or dependency surface changed.

Rust's same focused behavioral command first produced six intended failures (exit101), then six passes (exit0). C05_RUST_RED_SOURCE.rs and actual stdout/stderr preserve the false tiny-norm success and the other diagnostic/controller failures. The Python path now provides complete source manufacture, runner, independent exact verification, paired analysis and scientific plots. Its observed failures covered truncated oracle/action data, repeated policy IDs, false plain residual certification, mismatched fallback score bits, understated true residual bounds, exact wrong-decision counts lost to floating underflow, and missing break-even reporting. Reused source manufacture checks remain characterizations, not newly observed RED.

Parent inspection corrected ambiguous operator-cost reporting before cohort execution: an OUTWARD evaluation computes an ordinary stencil and an interval stencil in the same loop. The analyzer records ordinary, interval and total stencil evaluations separately, with logical response passes also explicitly named. These are typed work counts; they do not assert equal costs per stencil. Wall elapsed time remains the prespecified primary efficiency endpoint.

Runnable code and focused tests are complete. Native exact integration and the frozen cohort studies are C06–C08; no cohort performance conclusion is inferred from implementation tests.
