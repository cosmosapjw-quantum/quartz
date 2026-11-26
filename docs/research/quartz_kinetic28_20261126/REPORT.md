# K28: atomic resource floor and decision-calibrated stopping
## Origin, mathematics and literature
R01–R10/C01–C10 bounded CPU research; K27 exact parent 0f1ebd33da6048f8ccea7ccdb3f421c9aad3317a. Original README/THEORY at15f3a19, legacy paper §§5.1–5.3, frozen THESIS7241ad0 reread. Preserve root decision-loss reduction per computation cost, no throughput/entropy/refinement-count substitute. Wager supplied edition §14.2 Eqs14.19–14.21 and Polyanskiy–Wu §2.6.2 exact editions in K22 BOOKS; Sezener/Dayan, Russell/Wefald, Hay et al, Tuisov arXiv2607.27871 Eqs17/21–24/32, baseline-fallback motivation from Satija et al (SOURCES.md). No SPIBB/DR or formal safe-improvement claim. Derived atomic free shared/protected transition (s,p)->(s-k+min(p,k),p-min(p,k)), all-or-none if s+p>=k, base restricted to shared until join. Refund, consume, quiescent release preserve total. Same-state optional feasibility contains fixed-split feasible set, not trajectory or utility dominance. Signed actual second gain D2-.002 K2 and native STOP zero increment; disjoint-root crossfit q2 then q1 from other fold, final fit full TRAIN, CAL actual policy fallback. K21 exact Gibbs drift+reverse KL decomposition on raw score, no posterior interpretation. Exhaustive B2..64 integer states and demands0..2,524288 random transitions,8192 geometry cases, independent Wolfram conservation residuals0.
## Native implementation
New research_connect4_k28_cpu endpoint and packed AtomicU64 floor28 free pool. CAS claims base from shared, V draws protected first then shared in same transaction; zero demand valid; base terminal/ticket refund, release residual protected only after all join. K27 common/split unchanged. Runtime assert pre-phase base+borrow+free=B-r, actual evaluator count; full stage used<=B. Explicit second-boundary STOP sentinel20 returns actual mid recommendation or unchanged exact root proof with no second computation. Common prefix8, budgets8/16/32/64 per stage, cap4B visits; 20 actions (mask,worker1/2/4/8,common/split/floor,reserve0/2/4), actual S/X/V/B hooks, K25 ledger persists. Native old red/new green,84 smoke/4 invalid,48 serial paired K27 parity exact prefix/final/snapshot/counts and ledgers. Two new Rust pool unit tests incl concurrent claims, full650 pass,0 fail,89 ignored. No production defaults changed or main merge. No sanitizer or whole-tree race theorem.
## Data and hardware
Host observed24 logical CPUs,14GiB available RAM,299GB disk before work; nvidia-smi absent. Six source processes, three native servers each8-worker pool, eight solvers, four independent audit workers. Locked offline release build LTO=false/codegen-units4, binary d6a22eaac6f6955b166f5175e25e2da2ec0881408ed0d505a71d39cfbe91c299. Frozen source K17 and CNN/Pons/independent negamax identities verified. 1800 attempted families, exclude7408 recorded prior families and8629 boards,57 locks,629 quiet roots TRAIN301/CAL165/TEST163,1171 misses retained. Source23869 plies,13198 N32 requests,392046 NN. 629 exact WDL,3753 action labels agree,14975558 independent solver nodes. 20 diagonal+10 randomized cross paths per root/budget, actual concurrent intermediate states; sparse paths not factorial nor iid.
| phase | native requests | NN | visits |
|---|---:|---:|---:|
| TRAIN/CAL atlas | 55920 | 3677314 | 5352293 |
| CAL live | 2640 | 151120 | 271227 |
| TEST atlas | 19560 | 1296547 | 1813083 |
| TEST live/repeats | 5216 | 301154 | 525564 |
| main | 83336 | 5426135 | 7962167 |
Deep5 requests/506NN/2195 visits, timing576/36816/49323, half diagnostic652/45242/81705. Source+main+extras5900745 NN, excluding smoke, parity and regression. Raw compressed per-budget/worker shards plus source/labels retained.
## Scaling: 8-worker full profile, TEST163 per budget
| B | common V admitted/denied | split4 admitted/denied | floor4 admitted/denied | floor borrowed | common/split/floor regret |
|---|---|---|---|---:|---|
|8|17/795|337/231|350/233|0|.230061/.233129/.230061|
|16|650/570|567/721|726/436|311|.223926/.184049/.193252|
|32|1658/537|669/2242|1681/506|1842|.184049/.171779/.184049|
|64|3520/518|729/4950|3522/497|5102|.150307/.144172/.150307|
B64 floor restores common-like admission, no fixed quota throttling, though allocation paths and opportunities differ. B16 floor vs common8 utility+.030675, descriptive paired CI[.006135,.061350],163 both target; vs split -.009202 CI crosses0. B8,32,64 floor vs common mean utility difference0, floor vs split B8+.003067, B32-.012270, B64-.006135 all CIs cross0. No multiple-comparison-corrected or general superiority. Across all main stages exact152969,cap11811,unused226257,protected312040,consumed230885,returned81155,borrowed294059,extra770243,refinements442586,denied253118,parallel overlap129433. No overshoot, no repeated receipt; transfers218099,blocked candidate occurrences460465. 5 deep B64 raw traces2195 iterations/7462 edges,68 transfers and704 blocked independently reconstructed.
## Actual frozen policy and prediction
TRAIN301, per budget root-disjoint two folds, root-mean q2 direct signed utility change less actual NN, shrink8, cross-cell support6, STOP conservative empirical mean-minus2SE positive filter; out-of-fold q1, final q2 all train. No claim conditional sufficient statistics or formal confidence. Freeze model C07B498e3de, actual CAL165 compare serial/tree4/continue/stop, baseline serial all B; selected serial B8/16/32 and STOP B64; C07 2e79e4cc2f7dd31d64ad04b5972f8d1069d57b31 before TEST. CAL B64 stop net .408776 vs serial .328473, paired lower+.041301; no TEST tuning. TEST163 exact labels. First live run:
| B | serial regret/NN | tree4 regret/NN | continue regret/NN | stop regret/NN | stopped | chosen |
|---|---|---|---|---|---:|---|
|8|.214724/24.031|.211656/24.264|.226994/24.558|.230061/17.613|139|serial|
|16|.202454/39.497|.193252/40.086|.223926/40.442|.208589/24.442|163|serial|
|32|.174847/68.491|.184049/70.184|.187117/70.031|.223926/40.479|163|serial|
|64|.144172/119.791|.150307/124.798|.131902/124.693|.190184/70.264|163|stop|
B64 selected gain -0.046012 utility, 7 better15 worse, descriptive CI[-.092025,-.003067], NN decrease49.528 (41.35%), net gain+.053043 at price .002. Not noninferiority. Continue B64+.012270 CI includes0, matched96 -.005208. B8/16/32 fixed CAL fallback means no newly adopted dynamic continuation. Repeated utility SD B64 serial0/tree4.16147/continue.14653/stop.09185, recommendation disagreement0/41/39/19, JSD0/.029228/.022271/.006926. B8/16/32 repeat data saved.
Per budget4890 actual fixed-path second increments; forecast actual vs predicted mean, RMSE, correlation and raw geometry correlation:
| B | actual net gain | prediction | RMSE | prediction corr | geom corr |
|---|---:|---:|---:|---:|---:|
|8|.001532|.014459|.183980|.049811|.010426|
|16|-.007180|-.003035|.238626|.017010|.093908|
|32|-.045299|-.031965|.253013|.018101|.034043|
|64|-.095676|-.091048|.240617|-.028747|.050688|
Weak conditional prediction and overly extreme bins. B64 highest predicted bin mean+.156603 vs actual-.123667. Aggregate average net approximated much better than ordering per-root. Raw score geometry exactly reconstructed, not a posterior EVSI; no entropy reward. Negative observations include computation price, not all raw decision deterioration.
## Post-lock actual half-budget diagnostic
B64 all STOP suggests mere shorter budget, not adaptive advantage. New 652 actual requests on same TEST roots, serial and tree4 one-stage [0/2,20], two repeats each, no refit. Compare two-repeat average per root (not independent observations):
| policy | utility | NN | net |
|---|---:|---:|---:|
| full serial | .496933 |119.791|.257350|
| one-stage serial | .466258 |68.926|.328405|
| one-stage tree4 | .453988 |69.853|.314282|
| learned STOP | .455521 |70.209|.315104|
| continue | .496933 |124.764|.247405|
STOP vs full serial gain -.041411 CI[-.084356,0], net+.057755 CI[.015098,.098669]; vs one-stage serial gain-.010736 CI[-.035276,.010736], NN+1.282 CI[.674847,1.990798], net-.013301 CI[-.037632,.008442]. Static one-stage serial vs full gain-.030675 CI[-.070552,.009202], net+.071055 CI[.030061,.110699]. Learned stop descriptively worse than simple half serial at this price; not controller superiority. No retrospective policy replacement. 652 half requests independently audited.
## Time and reproducibility
Separate predeclared single-server timing12 TEST roots x2 repeats each budget; not host-isolated. B64 serial wall6.122ms tail5.085ms CPU5.986ms; 8-worker full common wall2.195 tail1.130 CPU9.397, floor4 wall2.144 tail1.124 CPU9.294; learned selected wall2.543 tail1.557 CPU3.949 with fewer NN. Same NN target not matched wall/actual NN; descriptive only. Main independent source, exact crossfit, CAL selection, 83336 requests/164780 stage accounting, geometry, repeat and deep chronology, 3753 stored WDL agreement. Actual solvers run acquisition, audit does not re-run CNN or solvers. W1 27 pass. Native subagent unavailable, direct tools and independent code. C09 CAUTION. Stage backups verified remote and ancestor, only same-cycle redundant bundles removed; isolated restore depends on K27 parent, not standalone full backup, cloud sync unverified.
## Next K29
1. Stop spending additional cycles on low-support vote-bin profile tables. Use K28 raw data as TRAIN-only exploratory evidence and new root families for confirmation. Actual comparison to fixed one-stage serial/tree4 at identical realized NN, and explicit price/quality frontier. Build calibrated action contrast joint law (K18 WDL support, K21 geometry only as feature, K22 readout correction). Predict signed decision response, not net reward alone; use independent calibration, proper scoring and support/abstention. Compare risk of native recommendation, learned readout and exact proof. Fixed-budget and price objectives reported separately. Use fresh independent validation of incremental gains vs stronger short baselines; don't retune TEST.
2. Implement real pending evaluator queue / CPU microbatch seam or available GPU after verifying hardware; count admission, actual NN and duplicate identity, batch occupancy, time to feedback, pending exclusion and cancellation. K28 packed floor is accounting foundation, not leaf parallelism. Compare same actual NN/latency with existing synchronous paths and independent correctness traces.
3. Prioritize exact graph proof propagation and actual frontier expansion, then Gomoku 7x7 paired play (P2 original predeclared noninferiority and >=30% actual NN reduction) with mature evaluator. Do not extrapolate C4 fixed-price STOP to Gomoku. P1 full calibrated global objective and P3 unresolved.
