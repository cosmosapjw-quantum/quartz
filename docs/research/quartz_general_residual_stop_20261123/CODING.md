
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

## C03 — Stable numerical design and error classes
The chosen object is a finite discrete SPD solve; no continuum truncation-error claim. Algebraic truncation is controlled by genuine CG work, strict decision certification or finite cap. Solver stability is diagnosed by fresh true residual; recurrent residual only controls common numerical exhaustion.
f64 rounding is contained by explicit next_down/next_up operations. Input coefficients rho in{1,16,64} and diagonal1+4rho are exact integers. Source b is exact dyadic; reference integer arithmetic is exact. Scale is an exactly representable common power of two and is checked against rhs norm upper bound.
At each check: recompute outward Ax_i as diagonal product and neighbouring negative products; r_i=b_i-Ax_i; square interval with lower0 for a zero crossing; sum outward; form outward ell; add residual lower/beta or upper/alpha; divide by positive S; intersect [0,1] and previous bounds. Reject nonfinite or empty intervals. Sequential arithmetic and no fast-math are part of the compiler contract.
Ties never pass strict separation; they finish at the declared budget. Zero rhs produces immediate converged state. At a fixed cap, the fallback uses final ell_hat lowest-index argmax, independently of which intermediate bounds a gated path inspected. No solver state is mutated by interval checks, ensuring identical continuation.
Compare total elapsed and actual mathematical work rather than theoretical speed. More frequent checks can cost more than they save; no tuning after TEST is allowed.

## C04 — Frozen source DAG and resource budget
SOURCE_SPECS.json enumerates all144 independent parents with 36 strata balanced across TRAIN36,CAL36,TEST72. Stratum=(side,rho,target_gap); each split contains all dimensions. Thus the contiguous-index/budget confounding risk in older sources is removed by construction.
DAG: publish specs/contracts -> implement/test -> generate TRAIN/CAL exact source and check -> publish executable benchmark and source generator -> generate preserved TEST -> benchmark all preregistered budgets/policies -> exact-oracle verification and parent-level timing summary -> one independent review and restore -> release.
No outcome-fitted policy or resource-price model exists in this cycle. Resource caps: side<=96 in study, K3, max CG cap512, max additional128, single-thread Rust, one warmup+9 repeats, one parent per streamed request; no GPU/HPC or external paid compute. Store raw inputs in gzip, exact manufactured integers in a separate oracle record, and all measured outputs with source hash. Initial budget4 and check block8 are fixed.
Timing repeated executions are not independent science parents. All policies and controls physically execute for reproducibility. Source cohorts and split identities are frozen before solver outcomes. C04 manifest records source-spec SHA256 and actual runtime versions.
