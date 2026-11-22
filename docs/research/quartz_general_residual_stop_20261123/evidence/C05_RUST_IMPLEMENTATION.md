# C23 Rust implementation handoff
Worker /root/c23_rust; bounded C05 implementation and focused C06 tests. No worker commit or publication. Parent owns stage integration.
skill_context_used: parent-supplied PhysMath1.3.0 C5/C6 execution and finite-review contracts, Skillquiver behavioral red-green instructions, actual remote THEORY.md R05–R09 and CODING.md C02–C04 read before mutation at C04 HEAD abfc55dc1d50701b554b151798981695313e66c8.

## Implemented behavior and final interface
New standalone src/bin/quartz_c23_residual_gate.rs implements 2D zero-Dirichlet screened-diffusion CG, fresh variational response, per-operation outward intervals, strict response separation and physical early return before further CG.
One JSON request per stdin line: task{id,side,rho,scale,rhs}, additional0..128, repeats(default1,max32), warmup(default0,max4), trace(defaultfalse).
Study transport sends scale and every rhs scalar as decimal strings, parsed with Rust str::parse::<f64>(). Numeric JSON is retained for compatibility but has no assumed exact-source byte identity. Unknown fields are rejected.
Task bounds:2..8 actions, side2..128, integer rho0..64, finite bounded rhs, positive normal power-of-two common scale covering an outward rhs-norm upper bound.
All six policies run physically and independently with cyclic order: NEVER,ALWAYS,CERT_ONCE,CERT_BLOCK,RESIDUAL,PLAIN_ALWAYS.
Response{id,additional,results}; each result includes policy,rep,selected,initial_selected,certified,reason,iterations,solver_matvecs,bound_evaluations,ordinary_evaluations,total_us,initial_us,solve_us,bounds_us,initial_bounds,final_bounds,decision_bounds,residual_relative,residual_goal_met.
Bounds{lo,hi,ell_hat}; ell_hat is normalized by the common scale.
trace=true additionally returns checkpoint arrays{iterations,fresh_bounds,bounds},final_x, and top-level input_rhs_bits/input_scale_bits copied from the actually parsed public task. Trace metadata is outside the task timer.

## Meaning of comparisons and counts
CERT_ONCE/CERT_BLOCK certified=true denotes an actual certificate-return path. A block certificate may occur at its cap; certificate count is distinct from work saved.
At an uncertified fixed-budget cap, the gated policies have the same CG state and final ell_hat argmax as ALWAYS. Interval checks never modify CG state.
PLAIN_ALWAYS performs the same initial and additional CG updates with ordinary scalar response evaluation only; it returns valid trivial[0,1] intervals, zero bound_evaluations and the same ell_hat/state/action as ALWAYS. It closes the unnecessary final-enclosure baseline risk identified before any scientific cohort runs.
solver_matvecs counts actual CG operator applications, one per successful iteration. bound_evaluations counts whole-grid interval response evaluations across actions. ordinary_evaluations counts PLAIN_ALWAYS scalar readout sweeps. Validation-norm arithmetic is included in total time but is not counted as a response enclosure.
total_us includes validation, solver workspace allocation and destruction, initial CG, actual continuation, response evaluation, gate comparisons and result assembly up to the timer. JSON decoding/serialization, external source/oracle work and top-level input-bit metadata are outside.
initial_us covers workspace construction and initial CG. solve_us covers continuation CG. bounds_us measures response evaluation (interval work for the first five policies, ordinary readout for PLAIN_ALWAYS); remaining wrapper costs stay in total_us.
RESIDUAL reports whether its fresh residual meets1e-8 after stopping on the recurrent residual or reaching cap512; a false fresh check is never labelled success.

## Actual verification and diagnosed repair
Initial eight meaningful behavioral tests compiled and then failed on explicit missing-implementation paths, exit101. After implementation, all eight passed. Requested input-bit metadata had its own failing None-vs-Some assertion before its implementation.
Parent UNIT probe then found real numeric-JSON1ULP source discrepancies:164/3072 RHS values on side32 and2205/27648 on side96. The preserved probe is a transport/provenance failure, not a solver counterexample. Decimal-string transport was tested first as an unsupported-type RED, then GREEN with exact positive and negative f64 bit patterns. No oracle/enclosure tolerance was widened and Cargo was unchanged.
Parent inspection found solver-state deallocation after the timer. Explicit drop(solvers) now precedes the final timer capture.
PLAIN_ALWAYS and six-policy transport were tested RED first:4 enclosure evaluations instead of0 and10 results instead of12 for two repetitions. Minimal implementation then passed.
Final actual focused suite: debug11PASS0FAIL, release11PASS0FAIL, release binary build exit0. Final sequence PID3287410 completed. All initial/fix RED and GREEN logs are retained, including historical smaller suite counts.
Commands: cargo test --offline --locked --bin quartz_c23_residual_gate; corresponding named tests for input metadata/string transport; cargo test --offline --locked --release --bin quartz_c23_residual_gate; cargo build --offline --locked --release --bin quartz_c23_residual_gate.

## Remaining parent work
Independently verify exact manufactured input bits and rational oracle containment on the frozen scientific sources; evaluate actual policy regret, physical work and elapsed economics against BOTH ALWAYS and PLAIN_ALWAYS; perform scoped inherited regression and one requested independent review. These implementation tests do not establish runtime superiority, continuum accuracy or original neural-MCTS/game claims.
