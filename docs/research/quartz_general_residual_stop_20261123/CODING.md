
# C23 Coding and numerical execution
## C01 — Baseline and acceptance
Parent release and immutable source hashes are in BASELINE.json. C23 modifies only new isolated files. Theory completed R01–R10 before this coding track.
Feature: choose and genuinely stop useful SPD solver computation using valid response intervals. Primary acceptance: all reported strict certificates select an exact-oracle optimal action; when not certified, both gated policies share ALWAYS's finite-budget final state/action; physical CG iteration/matvec counts shrink on early STOP. Performance is measured, never required to be positive to mark an experiment executed.
Regression scope: focused new binary debug/release tests, original Rust main test suite once after implementation, readback protected original hashes. Historical unrelated numerical campaigns are not rerun.
Correctness tolerance: exact Fraction comparison against outward interval endpoints for oracle containment; exact integer selected-action identity on shared finite fallback; full floating solution agreement uses explicit residual diagnostics rather than byte equality. No tolerance widening to rescue a failed certificate.

## C02 — Formula, data and code mapping
A=I+rho L_D -> apply_operator(side,rho,x), row-major zero-Dirichlet stencil.
x_{k+1}=x_k+alpha p_k, alpha=r.r/(p.Ap), standard unpreconditioned CG -> Solver state x,r,p,rr,iterations,matvecs.
ell=2b.x-x.Ax and D=||b-Ax||² -> enclose_response with fresh operator evaluation and outward arithmetic; recurrent residual is never the certificate residual.
[L,U] -> ResponseBounds with ell_hat for deterministic finite fallback; history intersection is distinct from final fresh bounds.
strict L_i>max U_j -> separated_action; STOP returns before any next cg_step and records certified flag.
u oracle -> exact dyadic source numerator divided by common scale, read only by Python scoring/checking. Binary public Task has id,side,rho,scale,rhs; transport deny_unknown_fields rejects oracle injection.
Timer boundary: native run_policy from validation/allocations through final action; source generation, JSON decode/output and Python analysis outside. Each benchmark policy actually runs, so campaign resource use is not claimed saved.
