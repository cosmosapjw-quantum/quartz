# C24 Coding and scientific execution

## C01 — Goal, baseline and acceptance

Theory R01–R10 is complete. Implement the selected scalar-envelope evaluator and explicit tiny-norm rejection in one isolated new Rust binary, preserving the frozen C23 binary/source/results and original engine/Cargo files. Parent/head/worktree/protected identities and actual venv/runtime are in BASELINE.json.

Correctness acceptance: no false strict certificate in exact-oracle checks; every noncertified fixed-budget gate has the same solver state, iterations and final ordinary ell/action as PLAIN_ALWAYS; true upper residual diagnostics are distinguished from ordinary estimates; the known tiny nonzero input is explicitly rejected. Exactness is Fraction-to-outward-endpoint containment without tolerance expansion.

Efficiency is an empirical claim, not a prerequisite for a completed experiment. Compare consumed CG calls, readout cost and total wall time at the frozen source/decision objective. A valid but slower method will be retained as a negative result. The historical unchanged main regression617pass/0fail/89ignored is evidence from C23, not a newly executed C24 pass; no shared engine/dependency change requires repeating that entire historical suite.

## C02 — Formula, data and executable mapping

New target src/bin/quartz_c24_cheap_certificate.rs. Keep C23 operator/solver/transport mechanisms and deterministic ordinary score order. Cheap validation -> cached max B, positive-dot input norm[BL,BU],normal power-of-two S check,and supported-domain errors. The primitive scalar budget helper -> upward gamma_k,F_k. Fast evaluator -> one ordinary yhat/rhat/P/Q/D pass, max X/Y/R, then Ey,Er,EP,EQ,positive-dot D0 interval,T,ell interval,raw clipping and normalized[lo,hi].
The exact formula and field semantics are CODE_CONTRACT.json and theory/DERIVATION.md. Plain readouts have trivial[0,1] bounds plus the ordinary score, no interval evaluation count and null residual_upper/goal. Fast or full-outward readouts have valid intervals and a certified fresh residual upper. If FAST_ONCE continues to a final plain readout, that final diagnostic is likewise null.
C24 source manufacture reuses C23's validated dyadic construction with new source IDs/seeds and independent exact edge-energy verification. The native controller receives only public task fields; oracle labels remain in separate compressed files. Native parsed input bits,final state,checkpoints and policy work counters are retained in trace output; trace copying is disabled for timed measurements.

## C03 — Numerical design and concrete failure classes

Truncation error from incomplete CG is captured by ell plus true-residual energy bounds. Rounding error from operator,products,summation and residual evaluation is bounded by the proved scalar envelopes; sigma is a derived conservative underflow floor, not a tunable epsilon. Empty intervals/nonfinite states/domain violations are errors, not certificates.
Reject nonzero RHS maxima below2^-400 before constructing Solver, allowing exact zero tasks separately. Validate the supported state ratio before response use. The raw q interval is clipped against[0,S] before division; conservative residual norm uses sqrt(DU)/sqrt(BL) outward. This repairs the demonstrated false-success fixture by an explicit supported-domain contract and does not claim arbitrary-range CG.
All fixed-budget paths use the same initial4 and common recursive exhaustion threshold1e-14. Full residual policy uses recursive1e-8/cap512 and records whether the certified fresh residual upper actually meets the goal. Evaluation cannot mutate a solver state. Strict ties finish uncertified unless another action is uniquely better; equality is not relaxed.
Numerical/domain errors,implementation defects,runtime failures and performance outcomes are recorded separately. Native IEEE default compilation and actual flags are bound in runtime evidence; no fast-math or price/checkpoint tuning is introduced.
