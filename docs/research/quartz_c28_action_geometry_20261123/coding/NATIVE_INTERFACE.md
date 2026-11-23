# C28 native interface frozen at R09

Binary `quartz_c28_action_geometry` reads JSON lines, emits one JSON object per line, and flushes each response. Reject unknown request fields and malformed/unsupported inputs with `{schema:"c28-geometry-v1",error:"..."}`; one bad line must not prevent the next valid request. Never accept oracle labels in game mode.

## Synthetic request

`{mode:"synthetic",id:string,probabilities:number[n][3],loadings:number[n],costs:number[n],m:number,b:number,horizon:1|2,repeats:uint,warmup:uint,iterations:uint}`.

`id` defaults to empty, horizon2, repeats0, warmup0, iterations1. Candidate id is its zero-based array index, n can be0–64. Probabilities must be finite/nonnegative and sum to1 within1e-12 (normalize within that admitted arithmetic tolerance, reject a larger discrepancy); point masses and zeros allowed. Loadings finite with abs<1; costs finite/nonnegative; m in[-1,1], b in[0,1]. No empirical outcomes are accepted. Mean-centered feature is zero on zero-mass support and on point masses. Both planners use identical coefficient preparation. Synthetic timing includes preparation each iteration, with std::hint::black_box preventing dead-code elimination. `iterations`1–10000, repeats0–1000, warmup0–1000. These are API resource bounds, not mathematical restrictions.

Response fields: `schema`, `mode`, `id`, `direct:{scores:number[n],choice:index|null,value:number}`, `lifted:{scores,choice,value}`, `max_score_abs_difference`, `timings:[{repeat,direct_ns,lifted_ns,iterations,order}]`. Timing values are total ns for the stated number of iterations; Python divides once. `order` is `direct_first` or `lifted_first`, alternating by repeat. Empty set gives value b, null choice, empty scores. `value` is b plus max(0, maximum raw score); choice uses the declared1e-12 rule and may be epsilon-suboptimal. Report that distinction rather than silently replacing value with the chosen epsilon score.

## Game request

`{mode:"game",id:string,history:uint[],cue_seed:uint64,model:C26Model,rho:number,repeats:uint,warmup:uint,trace:bool}`.

rho defaults .75 and must be in[0,1); repeats0–1000 and warmup0–1000; trace defaultfalse. C26 model and legal history validation match the frozen C26 substrate. Both methods independently generate identical cues, assign frozen probabilities/costs, perform the same mandatory initial verification, condition the common latent belief on all known values, then query with H2 followed by H1, never more than2 additional solves. Known terminal laws are point masses. Only reveal through actual GameState/exact calls; retain exact-known incumbent, and retain its move on equal-value later reveal. If b=1 or S is empty, stop. Node price is1/1024utility/node. All loadings are positive rho for this compatibility bridge; no fitted signed-game claim.

Response fields: `schema`, `mode`, `id`, `direct:RunSummary`, `lifted:RunSummary`, `timings:[{repeat,direct_ns,lifted_ns,iterations:1,order}]`. A RunSummary must include `initial_move`, `final_move`, `initial_utility`, `utility`, `initial_nodes`, `additional_nodes`, `charged_nodes`, `additional_queries`, `query_moves`, `final_m`, and `decisions` (detailed scores only if trace=true). Each detailed decision contains current b,m,remaining move ids,horizon,scores,choice. Summaries can be obtained in separate unmeasured runs so trace allocation stays outside performance measurements. Time measurement runs must call both full independent execution paths with tracing disabled, starting before cue generation and ending after final selection. Both include belief/cache preparation, initial verification, updates, planning and additional exact queries. JSON parse/validation/history reconstruction and serialization remain outside the resident controller boundary and are named as exclusions.

The native implementation may include a third internal legacy evaluator for unit checking, but it is not the primary performance comparator. Do not modify C26/C27 binaries, core game files, model files, Cargo manifest/lock or other worktrees. New source is `src/bin/quartz_c28_action_geometry.rs`; inline Rust tests are owned by its implementation worker. Python tests and campaign are a distinct owner.

## Shared numeric decision rule

Scores have utility units. STOP if the largest increment is<=1e-12. Otherwise choose smallest id with score>=maximum-1e-12. Native score verification uses absolute5e-12, relative0, on the frozen contract's tested domain; exact math uses exact STOP-first ties. Do not widen tolerances to repair a discrepancy. Fixed b,S/model/cost H1 certificates are analysis results and oracle tests; no deployed certificate speedup is claimed.

## Python contract

Tests: `tests/test_c28_action_geometry.py` using stdlib unittest (pytest may run it too). Binary path from `C28_BINARY`; missing binary is a clear NOT_RUN/error in native execution, not an overall pass. Campaign: `docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py`, subcommands `verify`, `benchmark`, `figures` or equivalently clearly documented CLI. Every output directory is scoped to C28; retain raw JSONL.gz, exact request/input hashes, binary SHA, environment, commands and exit codes. Root parent publishes stages; workers do not commit/push/delete.

Code workers may negotiate a necessary field-name correction before C05 implementation is accepted, record it in this interface, and keep both ends consistent. Changing science, source cohorts, tie tolerance or the comparison metric requires parent reconciliation before results, not an ad-hoc test edit.
