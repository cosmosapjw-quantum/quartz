# K31: direct native gain and exact proof frontier
## Decision
Research checkpoint complete; preserve actual exact frontier expansion as a positive candidate, do not adopt fitted continuation/priced policies. No production merge. K32 priority is proof-guided native materialization and incremental exact graph certificates, not another broad gain regressor.

## Theory and provenance
K30 parent d754a8ed3d98116eaf08c0e2baa86badc51a1ab1, tree f444b981d259831e5e5f9fd734b11c44d074f495. Re-read original 15f3a19 README/THEORY, original paper sections 5.1-5.3, frozen THESIS7241ad0. Original root-decision regret per computation cost preserved; fixed-price assay explicitly not original ratio or P1/P2/P3 proof.
DERIVATION.md: direct paired gain squared loss targets conditional mean; exact finite logged contrast identity with known e=1/9 under stable independent assignment; interval negamax and worst-case regret witness. 8192 rational/integer test cases and independent Wolfram residual/monotonicity check in R07.log. Wager supplied PDF pp35-39,60-63; Polyanskiy-Wu pp75-77; Dudik et al DR, Allis proof numbers, Kishimoto et al DAG caution and proof-set/PN-MCTS prior art recorded in SOURCES.md. Existing exact graph closure reused; no general novelty assertion.

## Protocol and resource evidence
24 logical CPU threads; initially 13 GiB available RAM and 275 GiB disk, no visible GPU. Six source workers, eight exact-label solver workers, two native servers with eight-worker pools and persistent broker. New 3600 attempted histories, 1294 exact retained roots, 2306 not retained. Historical exclusion sets 16408 families, 11844 boards. Source 48207 plies, 26525 queries, 786410 NN. TRAIN641/CAL-A163/CAL-B162/TEST328. Historical TRAIN-only K29 657 and K30 635 for scalar-compatible arms (0,1,2,4); no old TEST/CAL training.
K31 native: common prefix8, serial32 logical/cap128 then nine interventions twice randomized, STOP9. Arms0 serial,1 tree4,2 tree8,3 tree8 queue/dedup,4 full S/X/V/B floor4 direct,5 full queued,6 graph-only,7 FIFO actual rule expansion,8 directed actual rule expansion. Proof graph built from observed midpoint paths and root children with exact board-side keys and terminal facts; expand at most eight unregistered nondominated states, score cap16; no oracle, NN or extra MCTS visits in proof tail. Root witness wins readout; otherwise retain native if nondominated, else visit-ranked nondominated. No actual hidden MCTS edge materialization.
Main actual counts:
| phase | requests | logical NN | physical NN | visits | rule_work |
|---|---:|---:|---:|---:|---:|
| TRAIN/CAL atlas |17388|1025675|1005362|1812260|2613776|
| CAL live |1620|92609|91216|170018|56694|
| TEST atlas |5904|343578|336800|633086|893728|
| TEST live |3280|183754|181622|354856|119860|
| total |28192|1645616|1615000|2970220|3684058|
Source plus main logical NN 2432026; extra diagnostic15 requests/807 logical/770 physical, timing216/12423 logical/12193 physical, separately recorded. WDL 7655 actions, oracle nodes33322255. Native locked binary SHA256 52bf2bdf6e2b09d0f41757c5c828c4da7c9d05f7c60168be832705f99c8aaa06.

## Main heldout results
328 independent root units by protocol, two actual executions averaged per root. Utility WDL [0,1].
| arm | utility | regret | net (.002,.00002) | logical NN | physical NN | rule_work | CPU ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|.533537|.140244|.400543|66.496951|66.496951|0|4.635725|4.833419|
|1|.523628|.150152|.389604|67.012195|67.012195|0|6.245486|4.238137|
|2|.529726|.144055|.395555|67.085366|67.085366|0|6.661720|4.145664|
|3|.533537|.140244|.399457|67.039634|59.698171|0|5.209491|4.285469|
|4|.521341|.152439|.382107|69.617378|69.617378|0|6.725361|4.026744|
|5|.521341|.152439|.382277|69.532012|66.541159|0|5.063127|4.183076|
|6|.519817|.153963|.433485|38.987805|38.987805|417.832317|3.436620|3.673459|
|7|.538110|.135671|.450646|38.987805|38.987805|474.399390|3.621305|3.917741|
|8|.541159|.132622|.453780|38.987805|38.987805|470.158537|3.888003|4.208209|
Graph-only initial witness 5/328; FIFO 41, directed60, intersection39, directed-only21, FIFO-only2. Two repeats per root deterministic. FIFO 4926 actual expansions, directed4532 with 4376 guided and 72446 optimistic scores. Directed and FIFO improve 7 and 6 roots vs short respectively, zero worse; utility gains .021341/.018293, descriptive root bootstrap 95% [.006098,.036585]/[.006098,.033537]. Directed vs FIFO .003049 [0,.009146], one better zero worse; not established universal directed superiority. Directed vs long post-lock diagnostic +.007622 [-.021341,.038110], 17 better/17 worse, logical/physical NN -27.509146 (41.37%); net at assay prices +.053237 [.024717,.083042]. Mean break-even rule price .000133232 at NN price .002; sensitivity not policy selection. Rule count omits planner score CPU, and timing includes trace overhead and host load. No same-wall/physical-cost controlled NI, no P2.
Queue arm3 vs arm2 utility +.003811 [-.004573,.013720], physical -7.387195/request. Full arm5 vs4 gain0 [-.007622,.007622], physical -3.076219. All four active410/656 direct,416/656 queued. Main queue 172987 logical,142371 physical,30616 coalesced,45820 batches. Evidence 18485 new receipts,16714 blocked repeat deliveries; no duplicate delivery found in audit.

## Regression and policy
Eight candidates constant/ridge(lambda .03,.3,3), with/without compatible historical TRAIN. Root-disjoint twofold OOF; CAL-A selects augmented .03 (MSE .03424099 vs constant .03472287; OOF .02828603 vs .02815029). TEST every nontrivial gain RMSE worse than marginal constant; gain correlations between -.021043 and .033625. Mean signs less distorted than K30, not a calibrated root-level ranking. No direct cross-cohort causal comparison.
CAL-B actual net short .408679, serial .380148, tree4 .372685, continue .343660, priced .424297. Against baseline, paired mean minus2SE continue -.067698 and priced -.008658, freeze serial/short. TEST live:
| policy | utility | regret | logical NN | physical NN | rules | net |
|---|---:|---:|---:|---:|---:|---:|
| short |.519817|.153963|38.987805|38.987805|0|.441841|
| serial |.533537|.140244|66.496951|66.496951|0|.400543|
| tree4 |.532774|.141006|66.900915|66.900915|0|.398973|
| continue |.522104|.151677|66.804878|63.824695|0|.388494|
| priced |.525915|.147866|40.922256|40.652439|182.713415|.440416|
Continue vs serial -.011433 [-.030488,.007622], priced vs short +.006098 [-.006098,.018293], net -.001426. No policy adopted from TEST. First-arm logged sparse DR fixed full -.006293, direct .015435, IPS .053341, DR .054113; conditional SD .815059 -> .797006. Priced full .00009878, direct .005019, IPS .021131, DR .021004; SD .334155 -> .333520. Stable potential rewards under order/scheduling are assumptions, not established; complete atlas algebra verifies score identity, not causal unbiasedness in the actual logging regime.

## Verification
C09_AUDIT.json/log independent root/grid/label and ledger reconstruction, 28192 main requests, 8180 graph requests and all 40796 actual expansions; full planner choice reconstructed on 242 graph requests (additional diagnostic9). Exact WDL checks all stored 7655 actions; no invalid witness/interval found. Refit all regression candidates and compare 1960 native forecasts; 155 diagnostic batch-scalar checks and 1609 receipt-trace edges. Cargo release built; full test650 passed,0 failed,89 ignored; W1 27; resident scalar parity24 pairs against K30. No independent whole CNN/solver rerun in C09. Review CAUTION (C09_REVIEW.md). K31 timing audit includes some diagnostic samples for graph arms, so timing table above uses full TEST atlas descriptively. No GPU, sanitizer, async nonblocking, Gomoku test. Caller cache identities not native content verified, only locked external collector.

## Next K32
1. Incremental exact graph registration during actual traversal; terminal/proof fact payload separate from raw deterministic NN observations and visit statistics. Root witness STOP and nondominated legality filter; retain correct budget/refund accounting. Compare graph-only, FIFO, directed, and genuine MCTS hidden-edge materialization with matched NN and measured rule/planner time.
2. Fresh prospective proof-opportunity strata defined before treatment; fixed rule cap/wall and resource prices; zero false certificates, paired root quality and cost, report failures and uncertain cases. Do not tune using K31 TEST.
3. Original Gomoku7x7 P2 paired actual games versus frozen A1_legacy_base with preregistered NI margin and >=30% NN reduction; mature evaluator and actual proof applicability. C4 is not substitute. Do not claim P1/P2/P3 until their gates established.
