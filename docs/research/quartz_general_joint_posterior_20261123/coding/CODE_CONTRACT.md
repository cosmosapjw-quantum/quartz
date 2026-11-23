# C27 frozen theory-to-code contract (R09)

Authority: user SSOT87c15dc828d6eea519c219fd8a412fa21dfc464d; only this isolated continuation branch. This document and EXPERIMENT_CONTRACT.json precede all new TRAIN fitting/timing and CAL/TEST outcomes. Changes after outcomes require an explicit recorded deviation, preserving old evidence; no post-hoc threshold/model replacement.

## Scientific task

Implement the R05 marginal-preserving positive binary-latent law and exact finite two-query controller on the existing native GameState/TicTacToe path. Measure conditional predictive quality and resource-priced game utility. This is an isolated research executable, not a production-engine/P2 integration.

U=(v+1)/2 for root-player exact minimax v in{-1,0,1}. STOP deploys the best exact-known candidate. Public terminal children are known. Common initialization: if there is a terminal win, skip the paid initial query; otherwise exactly solve the nonterminal candidate with highest raw4-rollout mean, ascending move on ties. Retain all known candidates. This initial observation is included once in joint history. Every additional query reveals one previously unknown candidate. Additional budget is exactly at most2; use H2 thenH1 thenstop.

## Belief and decision functions

For p_a, mu=sum u p, s=max positive-support |u-mu|, f=(u-mu)/s or0 if s=0. rho in[0,1). Initial log_odds=0. Every exact observation contributes log1p(rho f)-log1p(-rho f). Predictive p_a^h=p_a(1+rho tanh(log_odds/2)f). Terminal delta masses have f=0. Positive C26 smoothing covers all nonterminal outcomes; impossible support is an explicit error, never a reset. Repeat observations are rejected or structurally prevented.

For b and d_a=lambda_nodes*predicted_nodes_a,
g_a=sum_u p_a^h(u) max(0,u-b).
K1_a=g_a-d_a.
K2_a=K1_a+sum_u p_a^h(u)max(0,max_{j!=a}(g_j^{h+(a,u)}(max(b,u))-d_j)).
Select maximum with STOP=0. STOP if best<=1e-12; ascending move wins numerical ties within1e-12. Strict rho0 recovers the product controller's decisions. Hypothetical updates never alter the real history. On b=1 or no unknown candidate, stop without unnecessary planning.

Five policies:
- INITIAL_STOP: common initialization thenstop.
- RANDOM_TWO: uniform among unknown candidates using public policy RNG; stop atb=1, exhaustion orcap2. Belief diagnostics use the product law.
- PRODUCT_H2: base cue marginals remain fixed across initial/later reveals.
- INITIAL_PROJECTED_H2: condition the latent law on the common initial observation, snapshot each remaining marginal, then treat those as independent/frozen for all later predictions and planning.
- JOINT_H2: condition on initialization and every subsequent exact reveal; totalcap2.

Primary JOINT_H2 minus PRODUCT_H2 includes initial and later information propagation. Secondary joint minus INITIAL_PROJECTED_H2 isolates later propagation. All use identical initialization, modeled query costs and paid work accounting.

## Data selection before labels

Read-only C26 public_sources.jsonl SHA2569c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1, training_raw.jsonl.gz SHA256486a50f34867e382cbff040e339dcc5e051349d2418dbc93a074ec25acdb6659, model.json SHA2560843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126.

TRAIN is exactly the old90 TRAIN roots (30 each ply4/5/6). No oldCAL/TEST fitting. New eligible evaluation roots are every canonical nonterminal ply4/5/6 board excluded from all300 old roots: counts8/53/83. Within each ply sort SHA256("C27-eval-split-v1|"+board_text). First2/8/14 become CAL; the remaining6/45/69 become TEST. Totals24/120. Preserve each canonical board's legal deterministic replay history and source identity. Use C26's public board/hash convention and cue_seed=uint64(first16hex SHA256("C26-cue-v1|"+board_text)). Root IDs carry C27 prefix. Publish public assignment and hash before new oracle labels.

No new labels are required for rho: old TRAIN survey contains90 complete root vectors,360 candidate entries,296 nonterminal and64 publicly terminal. Keep vector grouping, move order and original cues. Terminal entries have likelihood1 and are excluded from coupling score. Empty vector contributes0 log-likelihood difference.

## Honest three-fold rho selection

Within each ply sort the30 TRAIN sources by source_hash. Contiguous chunks of10 form folds0/1/2. Each heldoutfold has30 roots; fit each complement60 with C26 fit_model (pure memory). Use fold-specific model_prediction on the heldout candidate cues, never the full90 fit as a supposedly heldout predictor.

Grid in ascending order[0,.25,.5,.75,.9]. For each heldout root compute logR_rho=log[(product(1+rho f(u))+product(1-rho f(u)))/2] stably with log1p and log-sum-exp. Score=mean over90 heldout root logR. Select the greatest score, with smallest rho within1e-12. No extra regularizer or test-driven rho change. rho0 always scores0. Record every root/fold/grid score and fold-training hashes.

The final deployed base p_a is the exact original full90 C26 model, not a newfit. Store chosenrho, scores and provenance in a separate C27 bundle. CV assesses this selection procedure on reused TRAIN only; it is not TEST evidence.

## TRAIN-only time price

New native survey receives public TRAIN root/cues only, and executes each nonterminal candidate exact solve5 times. Record move, repeat, solve_nodes and elapsed_ns around each exact call, including timer overhead without subtraction. Exclude64 already-known terminal candidates from pricing. Expected measured queries=296*5=1480. No warmup in this price protocol; all5 measured repeats enter.

exact_ns_per_node=sum(elapsed_ns)/sum(solve_nodes).
lambda_nodes=2^-10 utility/node.
lambda_time_ns=lambda_nodes/exact_ns_per_node utility/ns.

Require positive finite elapsed aggregate and nodes; freeze price and all timing raw records before CAL/TEST. This is instrumented exact-query amortized wall time per node on this machine, not pure CPU or universal hardware price. It is distinct from C26 model.tau, a Gibbs utility parameter.

Runtime planning continues to use fixed predicted query node costs from C26. This omits predicted planner cost and is an explicit decision approximation. Empirical evaluation charges actual full instrumented native controller elapsed time, including cues, initialization, posterior/planning, queried exact solves and trace construction. JSON decoding/serialization, IPC and process launch are reported separately, outside that native interval. Do not label it end-to-end application latency.

## Campaign and primary admission

Both CAL and TEST: policy_seeds[0,1,2,3], common deterministic cues perroot;5 measured repeats and1 discarded warmup of each policy perrequest. Native order rotates by(seed+repeat) mod5; serialization canonical. Repeated runs must have identical semantic traces for fixedroot/seed/policy. Record every timing, not only minimum. Reduce timing to median of5 perroot/seed/policy, then average4 seeds withinroot. Root is the unit; seeds/repeats are not additional roots. Quality/nodes deterministic repeat checks occur before aggregation.

Primary perroot difference:
D_r=(U_joint-U_product)-lambda_time_ns*(t_joint_ns-t_product_ns).
Primary summary is the equally weighted mean over120 TEST roots (actual6/45/69 ply mix, no post-hoc equal-ply reweighting). Diagnostic paired root bootstrap:10000 resamples, seed2709, percentile2.5/97.5. These are descriptive source-resampling intervals, without independent-tree or superpopulation coverage claims. Report ply strata and root signs. Shared descendants and previous whole-state semantic audit limit “new” to new campaign roots.

Operational empirical admission requires primary descriptive lower endpoint>0, quality-difference lower endpoint>=-0.02 utility, and no invalid/support/leakage/semantic failures. If this fails, preserve a negative/inconclusive result. This gate is not proof of population-wide gain. Secondary net/node controls and conditional calibration cannot rescue a failed primary.

CAL is diagnostic only: no re-fit, retuning, sample replacement, rho/price changes or gate threshold adjustment. A semantic/transport defect may be repaired with an explicit deviation and selective invalidation. A scientific negative does not cancel TEST.

Conditional diagnostics from independently available offline outcome labels: candidate log loss/Brier and selected direct-gain residuals at recorded histories, initial versus later history separately; joint NLL against product on complete root vectors; counts and exposure denominators explicit. Optional fixed bins for gain reliability[0,.05,.10,.20,.50,1.01]. Predictive, decision and resource outcomes are reported separately. Offline values/signature labels never enter run requests.

## Input/output and scope

New executable src/bin/quartz_c27_joint_posterior.rs may reuse frozen game/solver/cue/initialization/transport code. Do not alter C26, core GameState, THESIS or CLAIM_LEDGER. New Python harness lives coding/c27_campaign.py, tests/test_c27_joint_posterior.py. Model request remains original C26 model schema plus explicit request rho; arbitrary unrecognized input keys are rejected.

Retain C26 public request fields id,history,cue_seed,policy_seed,mode,additional,repeats,warmup,trace,model; add rho. Run validates additional==2 and rho finite0<=rho<1. Mode survey supports repeated nonterminal exact timing; mode run never receives offline labels. New schema c27-joint-v1.

Run response contains all5policies, repeat, known initial/selected move and utility, query history, cue/initial/additional/charged nodes, elapsed_ns and phase times, posterior log_odds, predictive probabilities, modeled gain/K1/K2 and chosen action in traces. Terminal and initial observations, remaininghorizon and projected marginals must be inspectable. Unknown-candidate actual utilities must not be exposed in runtime predictions. Survey response is offline TRAIN-only and may expose full labels/timings.

## Verification and reproducibility

Meaningful red: original product-only policy/model cannot deliver109/200 or posterior .82 on the frozen pure-information fixture; keep actual failed assertion log before adding the mechanism. Native tests must cover marginal preservation, rho0, degenerate/unsupported observations, orderinvariance, H1/H2 cap/STOP ties, projectedcontrol, signs, queryunique, malformed JSON recovery and publicinput rejection. Python tests use full joint-world conditional enumeration as an independent scoring oracle; match actual game traces and pairedcost arithmetic, test fold/source isolation and immutablebase identity.

Tolerance probability/score comparisons<=1e-12 (testmayallow2e-12 accumulation); exact utilities,nodes,legalactions and cardinalities match exactly. Full-state C26 game semantics are reused by unchanged sourceidentity. New controller tests and representative oldTRAIN integration must actually run. Whole-project results that are inherited/blocked are separately labeled.

Freeze new native/Python/test source hashes, binary SHA and model/price bundle before CAL. Save commands, environment, return codes, raw records and summaries. C08 performs TEST, deterministic repeat verification, timing dispersion, focused numerical boundary checks and restoration. C09 is one fresh independent review; fix concrete findings and selectively recheck. C10 finalizes20 stage receipts, remote HEAD, archive integrity, old15worktree status/protected-byte preservation and nextC28 recommendation without starting it.

## Resource contract

Use the C26 target cache with the new unique C27 bin, Cargo--locked--offline and-j2, since free disk is about2.3GB. Original C26 release binary SHA must remain8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c. Generated cache mutates; old tracked trees do not. No Cargo clean, old worktree deletion, Torch installation or duplicate historical campaigns. Recheck free space before restoration; preserve backup-chain identities and distinguish streamed archive verification from physical restore execution.

## Origin watchdog and claim limits

The posterior's two-state Gibbs/partition representation is implemented as stable log odds, with dimensionless field. The information-only H2 effect realizes observation-conditioned computation value. P1 ratio remains distinct from priced-net optimization. P2 Gomoku NN-eval target, P3 human signatures, full original NS/TT mechanism and physical phase-transition claims remain unresolved. A zero selectedrho or net-negative timing result is a local finding about this closure/implementation, not a no-go for the original research motivation.
