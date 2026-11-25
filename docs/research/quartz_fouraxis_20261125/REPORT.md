# FOURAXIS-01 | Native four-stage metareasoning interventions
## Disposition
All requested R01–R10 and C01–C10 executed plus blind C08A. User SSOT 87c15dc828d6eea519c219fd8a412fa21dfc464d, original history verified. New direct-base research sibling, no merge into production. **All four operational axes implemented and actual native activation independently checked; joint performance policy NOT ADMITTED.** This is first constructive slice: 16 static masks, not yet a learned adaptive joint controller. FOCUS/EXPAND/MERGE/STOP are distinct metacontrol actions; fixed-budget STOP in this experiment, not live VOC-based STOP competition. Prior NN-DECISION-01–04 negative results immutable. Actual frozen Connect4 COG07 Conv3x3 neural evaluator, native select_core/materialization, two-thread Rayon work, TT reused. No new Gomoku transfer or full production strength/Elo claim. Existing prior 15x15 NN/parallel IPC evidence retained. Native subagent NOT_RUN, independent finite mathematics and separate grid/permutation analysis instead.
## Theory
Search information state H, feasible joint typed phase action z=(S,X,V,B), transition kernel K_z, resource-constrained metalevel Bellman with STOP payoff M(H), prior-weighted Gibbs joint action distribution. Requires fixed positive integer costs for recursive finite budget (zero-cost STOP is terminal); random cost needs an explicit transition/overrun contract. Signed actual-decision gain differs from coherent Bayesian information value. Factorial Mobius inversion and Shapley via interaction terms; do not add isolated axis gains assuming independence. Backup bounded shrinkage sign-equivariance, one path traversal one edge visit; algebraic perturbation bound 2[1-product_l(1-eta_l)] relative to original leaf return, not WDL bias bound. 32660 exact Fraction assertions, Wolfram symbolic checks. See DERIVATION.md. Primary sources Sezener/Dayan UAI2020, Leurent/Maillard ACML2020, Czech/Korus/Kersting ICAPS2021 and Saffidine UCD; no claim novelty of these general ideas.
## Implementation and source semantics
- S (bit1): node-local KL between smoothed visible prior and visit distributions, c_local=c0[1+0.5D/(1+D)]. Existing select_core; ordinary callers pass None, research calls use opt-in hooks.
- X (bit2): native PW alpha2 beta.5 and existing depth scaling; additionally cover .75 candidate prior mass and keep already visible edges. Native root starts one materialized action; no legality change, no confidence claim. Finite budget can hide legal actions.
- V (bit4): actual same frozen NN base and two top-prior child evaluations when |v|<=.65, exact terminal child outcome when known, blend .5 base +.5 max child negamax. No claim this is a stronger model or a true minimax bound; terminal leaf itself exact. A known winning child establishes a game-theoretic winning parent, but the chosen heuristic blends rather than certifies that fact; candidate correction for next cycle, not post-hoc alteration. Count actual extra evaluations (not only engine visits).
- B (bit8): per-engine raw child-perspective ledger keyed child TT identity and actual incoming parent hash/edge index; prefer prior other-context mean, otherwise own temporal mean, eta<=.25. Coherent serialized writers for n/W/M2; still weak concurrent reader snapshots. One sample per traversed edge; terminal leaf paths unmodified by shrinkage. No independent-evidence assumption. Existing baseline backup's repeated-edge RTT proxy is not a true distinct-path counter; opt-in B omits that proxy and counts actual distinct incoming contexts. Hash-based identity relies on existing TT keys (not proof against all collisions). B bundles temporal smoothing, cross-context blending and coherent updates, so its effect does not identify one mechanism alone.
No unsafe TT/node layout change. Research-only synchronous owned descent (not async or arbitrary games); standard caller uses original path. Direct optimized rustc built new binary e2cb6685f87fd81a5620d85692d9276f2b635183ac4338a2c88e5132cc7c74cc from C05 source with locked read-only prior Cargo release deps, 2 focused red-green tests, four exact 64-visit single-thread native mask0 versus standard run_par result matches, all 16 actual two-thread smoke masks, eight deep 512-visit searches on four prior TRAIN roots, 389 other-context blends. Unknown oracle input rejected. Full Cargo suite NOT_RUN; no proof of global parallel safety. Known legal histories 1234 and3214 independently have identical gravity board and native root hash 41ebbe23001eca8a; deterministic one-thread research B root results equal. Independent audit of all observed root visit sums, W, bounds, VL cleanup, evaluator counts, B root M2 bounds, source and score. No claim every internal M2 trace separately reconstructed.
## New population and execution
165 attempted unique canonical opening families (61 tactical,104 exploratory selfplay), 64 accepted roots (32 per stratum,16 each side). Excluded 4489 prior recorded opening families and4207 canonical boards incl NN02–04. Unknown unrecorded interrupted attempts cannot be excluded. 966 actual exploratory NN recommendation events; no WDL screening. TACTICAL has no current immediate win, SELFPLAY unfiltered by this property but exploratory/ply selected. Source freeze C07 before live; 2048 fresh independent-engine searches =16 masks x64 roots x2 seeds17/43 at64 cumulative visits using two native threads. Raw compressed C08A 7499b4ee17c51fce1d184f795f19510511b1acf4 posted before exact labels, SHA d4f24187d42311cd3619d455a5c05d2e194a12431db6c9a3c4325088c508d9ff. All64 external Pons exact WDL action vectors,6 fresh mirror spot-checks agreed. Host criterion passes; measured live interval3.479s, swap activity2.300 pages/s, endpoint max load1 2.170 on24 logical CPUs; all request sampled load1<=12. Timing remains descriptive short campaign, no stable equal-cost claim from equal visits.
## Actual interventions
For full mask15, counts summed across64 runs per stratum:
| Mechanism | TACTICAL | SELFPLAY |
|---|---:|---:|
| Local selection coefficient changed |10904|8201|
| Nonroot selection hooks |6820|4110|
| Width limited |7360|4990|
| Prior-mass widening beyond base/old |1096|934|
| Leaf refinements |828|811|
| Extra actual NN evaluations |1423|1483|
| B backup value changed |4322|2955|
| B same-edge temporal blends |4422|2974|
| B other-context merge blends |70|33|
| Observed shared-child identities (sum per engine) |69|45|
Across all eight B-enabled masks per stratum: other-context blends544/288, in168/133 runs; these counts are not independent state or statistical samples. 389 further deep smoke other-context blends separate. Thus actual S and X operate below root, V consumes actual inference, B changes return and observes graph reuse. Some terminal/saturated roots have no V or B blend.
## WDL and interactions
Mean exact regret, same 64 visits (not same NN calls or wall):
| Mask | Active axes | TACTICAL | SELFPLAY |
|---:|---|---:|---:|
|0|none|.0390625|.09375|
|1|S|.0546875|.09375|
|2|X|.0703125|.09375|
|3|SX|.0703125|.09375|
|4|V|.0625|.109375|
|5|SV|.078125|.109375|
|6|XV|.078125|.109375|
|7|SXV|.0625|.109375|
|8|B|.046875|.09375|
|9|SB|.03125|.09375|
|10|XB|.0703125|.09375|
|11|SXB|.03125|.09375|
|12|VB|.0625|.109375|
|13|SVB|.0625|.109375|
|14|XVB|.0625|.109375|
|15|SXVB|.0625|.109375|

Full minus baseline utility TACTICAL -3/128=-.0234375 (32 families, paired descriptive bootstrap [-.109375,.046875]); SELFPLAY -1/64=-.015625 ([-.046875,0]). Full mean native whole time per root 1.8630 vs1.4060 ms (32.51% mismatch), 1.7686 vs1.2529 ms (41.17%). Full mean NN calls 62.828125 vs39.453125 (1.5925x),57.515625 vs32.28125 (1.7817x). Therefore no equal-wall superiority, no full policy admission. Exact rational Shapley contribution to joint utility change, independently via 24 permutations:
| Axis | TACTICAL | SELFPLAY |
|---|---:|---:|
| S |1/1536|0|
| X |-17/1536|0|
| V |-29/1536|-1/64|
| B |3/512|0|
TACTICAL SB interaction +1/32, S isolated -1/64 and B isolated -1/128; joint SB gains +1/128 against control. Four-way interaction -5/128. SELFPLAY only V has nonzero aggregate Mobius at this scope. Descriptive, sample-specific, no global causal interpretation or isolated merge effect from bundled B. V consumes additional NN calls and performs worse in these two cohorts; do not assert general value lookahead impossibility. No retuning on TEST; mask9 and11 TACTICAL are exploratory observations only.
## Drift and next FOURAXIS-02
Original research goal same mean CPU time higher Elo, not merely expanding search. Current work restores actual node-level phases and distinct-path MERGE semantics rather than treating all computation as root visit allocation. Still need a live policy selecting FOCUS/EXPAND/MERGE/STOP, genuine resource matching, broader games. Next:
1. On separate TRAIN roots and exact bounded leaf-oracle data, decompose V error into omitted-action bias, base NN error and max-of-noisy-estimates bias. Preserve terminal proofs (winning-child certificate, all-child loss/draw only with full coverage), compare base, top2, full legal one-step where affordable, and disagreement-conditioned selective evaluation. No oracle at runtime. Test at matched NN call and wall budgets separately; use trained COG07 weights, no new network claim.
2. Split B into coherent-writer-only, same-context shrink, other-context-only merge and combined; log distinct incoming contexts with state witnesses and raw return covariance/continuation-policy shifts. Use known DAG fixtures and actual deeper native traces; keep path-visit count unchanged and exact terminal priority. B positive Shapley does not establish merge benefit because temporal and merge effects bundled. Compare TT reuse vs true edge evidence without unsafe layout changes.
3. Freeze S/X local controls, measure actual post-prefix state (prior KL, materialized mass, value gap, graph provenance); introduce paired reconstruction of identical search prefix under deterministic single-thread before targeted stage interventions, plus native two-worker fixed-work/time comparisons on new independent roots. Evaluate signed actual utility and joint costs; a full tuple adaptive Gibbs or max-net-VOC controller only after identifiable stage-action evidence. STOP must compete explicitly with FOCUS/EXPAND/MERGE, and need expansion/evaluation/backup transitions not merely metrics. Avoid adding larger controller before mechanisms are measurable. Evaluate robust fixed/equal-work and actual equal mean wall controls, prior negative evidence preserved.
4. Extend through actual existing 15x15 neural IPC/parallel MCTS when stage-level semantics and time/decision effects are established; paired gameplay vs same NN and fixed-budget baseline, no exact WDL claim for standard 15x15. No automatic production merge.
## Preservation
20 numbered phases plus blind C08A, per-phase git push and verified cumulative incremental bundle. Older same-run bundles replaced only after successor identity and remote acknowledgement. Prior unique research evidence and other worktrees untouched. Final actual independent shallow-base restore and binary hash recorded in FINAL_VERIFY.json outside Git. Local Dropbox ACK only, no cloud sync claim.
