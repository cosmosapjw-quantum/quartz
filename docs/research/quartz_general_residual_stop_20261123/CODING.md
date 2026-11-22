
# C23 Coding and numerical execution
## C01 — Baseline and acceptance
Parent release and immutable source hashes are in BASELINE.json. C23 modifies only new isolated files. Theory completed R01–R10 before this coding track.
Feature: choose and genuinely stop useful SPD solver computation using valid response intervals. Primary acceptance: all reported strict certificates select an exact-oracle optimal action; when not certified, both gated policies share ALWAYS's finite-budget final state/action; physical CG iteration/matvec counts shrink on early STOP. Performance is measured, never required to be positive to mark an experiment executed.
Regression scope: focused new binary debug/release tests, original Rust main test suite once after implementation, readback protected original hashes. Historical unrelated numerical campaigns are not rerun.
Correctness tolerance: exact Fraction comparison against outward interval endpoints for oracle containment; exact integer selected-action identity on shared finite fallback; full floating solution agreement uses explicit residual diagnostics rather than byte equality. No tolerance widening to rescue a failed certificate.
