# C24 Coding and scientific execution

## C01 — Goal, baseline and acceptance

Theory R01–R10 is complete. Implement the selected scalar-envelope evaluator and explicit tiny-norm rejection in one isolated new Rust binary, preserving the frozen C23 binary/source/results and original engine/Cargo files. Parent/head/worktree/protected identities and actual venv/runtime are in BASELINE.json.

Correctness acceptance: no false strict certificate in exact-oracle checks; every noncertified fixed-budget gate has the same solver state, iterations and final ordinary ell/action as PLAIN_ALWAYS; true upper residual diagnostics are distinguished from ordinary estimates; the known tiny nonzero input is explicitly rejected. Exactness is Fraction-to-outward-endpoint containment without tolerance expansion.

Efficiency is an empirical claim, not a prerequisite for a completed experiment. Compare consumed CG calls, readout cost and total wall time at the frozen source/decision objective. A valid but slower method will be retained as a negative result. The historical unchanged main regression617pass/0fail/89ignored is evidence from C23, not a newly executed C24 pass; no shared engine/dependency change requires repeating that entire historical suite.
