# C27 R01–R03: bounded source context, population and origin watchdog

Status: `implementation-verified` public-state enumeration; `source-inspected` primary-history map; `unresolved` C27 predictive/utility improvement. Parent owns admission, source split, all repo mutations and publication. No training, minimax labels, policy campaign, build or repo write was performed by this child.

## Authority and narrow question

- SSOT: `87c15dc828d6eea519c219fd8a412fa21dfc464d`; remote `/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123`; device `cosmosapjw` (`edd1b263-56a0-4391-8a2a-677dd00eabf5`).
- R01: with initial cue-conditioned marginals, initialized exact incumbent, horizon and resource price held fixed, does observation-updated joint belief improve conditional computation-value prediction and paired net utility against the product of those same initial marginals? This is the actual C26 continuation proposal, not a fresh claim of engine efficiency.
- R02: reuse the C26 legal-state semantics, exact-solver/sign evidence, frozen 300 source identities, original primary formulas and COG observation-channel correction where assumptions match. New source counts below require actual enumeration and were executed. C26 campaign outcomes are motivation only, not evidence that omitted correlation caused its failure.
- R03: specify joint utility law, public cue/terminal observations, exact-reveal history, root-player utility, finite H2 action set, support policy, utility/resource price and controller overhead before fitting. Posterior dependence is a proposed statistical model, not a demonstrated physical hidden mode or human cognitive mechanism.
- Read skills: physmath-research-loop 1.3.0 SKILL; THEORY_LOOP.md R01–R03; POLICY_INTEGRATION.md; SUBAGENT_MODE.md. This report does not add workflow gates or repeat C26 efficacy audits.

## 1. Actual source availability

Executed `python3 /tmp/quartz_c27_population.py`, remote PID429289, exit0, runtime0.16s. It imports public semantic helpers from C26, enumerates only legal play until a public terminal state, uses D4 canonicalization, and subtracts all300 existing public root boards. It does not call `oracle`, `truth_record`, `build_sources`, `fit_model`, a native policy or a campaign. The imported oracle cache remains empty. Deterministic histories and every available board are retained in `/tmp/quartz_c27_context.json`, `available_roots` (327 records).

| Ply | All canonical nonterminal roots | C26 roots excluded | New available |
|---:|---:|---:|---:|
|0|1|0|1|
|1|3|0|3|
|2|12|0|12|
|3|38|0|38|
|4|108|100|8|
|5|153|100|53|
|6|183|100|83|
|7|95|0|95|
|8|34|0|34|
|Total|627|300|327|

All legal reachable boards, including terminals and without quotienting, total5478. The original ply4/5/6 source frame has only144 unused canonical roots. C26's balanced 100-per-ply,300-root experiment cannot be repeated on wholly new roots in that frame. No symmetry transform or RNG seed creates a new root.

Public structural diagnostic: available roots with at least3 nonterminal children before initialization occur at ply0:1,1:3,2:12,3:38,4:8,5:53,6:23. Thus84 of the144 same-frame roots offer at least the structural possibility of2 unresolved exact queries after1 initial query. This does not establish positive H2 information value: terminal wins and observed values may cause stopping. It is not outcome evidence and was not used to select a cohort.

Feasible source contracts for parent selection, before label generation:

| Option | TRAIN | New CAL | New TEST | Consequence |
|---|---|---|---|---|
| Reuse frozen C26 training | Original90 TRAIN only |24=(2,8,14) by ply4/5/6 |120=(6,45,69) | More fresh evaluation roots; training is explicitly reused; no old CAL/TEST fitting |
| Fully fresh same-frame |54=(4,20,30) |30=(1,10,19) |60=(3,23,34) | All splits new; weaker small-sample training/test and tiny ply4 strata |
| Expand frame |Some of all327 |Freeze before outcomes |Freeze before outcomes | Changes domain: early roots have different solver costs; late roots leave less room for H2 |

These are feasibility examples, not frozen allocations. Availability order uses SHA256(`C27-available-v1|`+board_text), sorted within ply; parent must choose split salt and allocation before labels. Do not choose by which option looks better after fitting or test outcomes. The finite source estimand and weighting must identify its actual unbalanced strata.

Dependence is substantial. Among the144 new same-frame roots,4249 of10296 unordered root pairs share at least1 canonical descendant. Also53 roots have at least1 old C26 root as a descendant. Disjoint root IDs therefore do not yield independent game trees or a never-seen-state population. The C26 semantic audit previously computed/checks all4520 legal nonterminal raw states; that is reused semantic evidence, not a fresh run here. New-root holdout means excluded from the300-root C26 campaign and from subsequent fitting as contracted; it cannot mean their minimax values were never computed anywhere. Root bootstrap intervals are conditional/descriptive for the frozen source collection unless an explicit sampling/generalization law is defended. Multiple timing runs and seeds within one root are repeated observations, not additional roots.

### Exact identity

- Enumeration script SHA256: `d829c2593c5a162259b8781b2c5bd19f3ffcccf90ff52aebf3f62ebdbc17a914`.
- Population JSON SHA256: `854e56d3fd8bbb53c4745be77f70e42416e196a0e98273bc0ee9ca49a894ca1f`.
- C26 public_sources.jsonl SHA256: `9c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1`.
- C26 campaign SHA256: `ee4501d5bfee66adcc15dbfb0b0f93ee50343e3313c7d1ce654143571ecee750`.

## 2. Bounded origin/history drift watchdog

Primary sources were actually reopened: original PDF pp5–7 via pdftotext, root-commit Rust KL and VOC formulas via git show, current THESIS P1–P4, COG01 contract/results, C26 CONTINUATION and NEXT_C27_PROPOSAL, SOURCE_SPECS and CODE_CONTRACT. The C26 ORIGIN_THEORY_MAP is a reused source map; its C25 snapshot hashes must not be mistaken for current C26 SSOT identities.

Verified Git chronology: earliest root `15f3a1986a00d548d3da56dab8233b336e931aee`,2026-04-09; THESIS introduction `7241ad0cfe1e92f28bd8cded5ad5f9fe7d4b4eab`,2026-07-10; archived intuition corpus inclusion `3bcaa435b6bdca2f821f8990230ea05ddf251a2f`,2026-07-12. PDF internal January29 date is not an independently verified January Git date. Exact `git show -s --format=%H %cI %s SHA` returned exit0 for all these and SSOT.

| Original object | Direct source content and continuing role | C27 admission boundary |
|---|---|---|
| PDF p5 Eq1 | `sum q_a mu_a - beta^-1 KL(q||pi)`; Gibbs optimum and log-partition derived under positive prior support | A real statistical-mechanical variational construction. Distribution over compute actions is distinct from joint belief over utilities. Last-step softmax does not prove recursively regularized Bellman optimality. |
| PDF pp5–6 Eq2 | Post-information root decision improvement less measured compute cost; FOCUS/EXPAND/MERGE/STOP | H2 uses the observation transition explicitly. Utility/time conversion and controller overhead must be declared. Historical fixed-current-mean reading would make positive information gain impossible; preserve the corrected posterior-predictive reading. |
| Original source continuity | At root quartz.rs607–617, smoothed KL;945 `p_flip*e_delta-cost_focus` | Establishes original implementation intent; not an original-engine efficacy rerun. |
| PDF p7 hidden modes | Outside prior mass; hidden maximum probability; rare-event Poisson proxy; multimodality/surprise/heavy-tail gate | A two-component mixture can implement learned cross-candidate update. It does not establish equivalence to NS evidence quadrature, original EXPAND reservoir, TT geometry or a physical phase transition. |
| COG01 recognition | Exact binary recognizer posterior, known fallback, perfect verification; signed utility and gain calibration retained, including inverted-cue harm | C27 must keep marginal option quality, the specified observation channel and conditional query value separate. More accurate marginal probabilities alone do not guarantee correct VOC. |
| COG01 correction | In five perfect-observation screens `gain_ECE=0.4*quality_ECE` identically; no independent capacity dissociation. Separate H/L channel example gives grossVOI1/5 vs0 at calibratedq1/2, channel-aware value13/20 vs3/5 | The H/L transition mechanism motivates C27 joint conditioning; C27 must actually test its conditional predictions and costs. Human cognition remains untested. |
| Existing reopening history | C26 map points to resident_reopening_13 candidate preservation and monotone activation/NWM2 continuity | Reuse as scoped design/evidence context; no fresh verification or causal human Einstellung claim here. |

Historical corrections already recorded in C26 are preserved, not rediscovered as new results: Gaussian rho=0 gives a conservative difference-variance treatment only with appropriate nonnegative-correlation assumptions; negative rho can increase variance. Ranking-error probability alone is not probability of a computation-induced ranking change. `E log Z(U)`, `log E Z(U)` and `log Z(EU)` differ. Joint finite masses provide an explicit alternative without denying valid Gaussian results under their assumptions. Earlier path-normalization/Lindblad issues remain archived bounded corrections; no new quantum-mechanism claims are imported into C27.

Current THESIS already incorporates C26's narrow policy correction allowing precise statistical-mechanical formulas and runtime estimated VOC from permitted observations while excluding offline oracle evaluation labels/signature targets. No further THESIS edit is required merely to run C27. The historical C26 origin report's THESIS SHA02ffa… belongs to its explicitly stated C25 snapshot; current SSOT SHA is4f5133… below.

### P1–P4 preservation

- P1: governing objective is expected simple-regret reduction per cost, not an independent heuristic knob surface. C27 priced-net Bellman is a declared finite resource formulation; net and gain/cost choices need not coincide at fixed lambda. Existing metalevel_decision.py already separates them. Do not claim C27 proves the full original engine policy is composed or wired.
- P2: at least30% reduction in nn_evals_per_move at non-inferior play quality on Gomoku7x7 short-budget self-play, budget normalized. TicTacToe exact-solver nodes or H2 net utility do not meet that claim. Remains unresolved.
- P3: preregistered macro-invariance plus skill-increasing VOC-tightness as emergent double pattern, not optimization targets; independent offline labels excluded from control. C27 does not test human-GM signatures. Remains unresolved.
- P4: CPU-friendly hot-path cost is an engineering admissibility gate. Measure actual controller, feature, initialization and exact work for this new path. Node savings do not establish CPU savings. Remains separate from P2's NN-call research estimand.

Concrete possible C27 drift and smallest corrections: (i) false fresh balanced300 sample -> use actual144-frame count and explicit split; (ii) marginal model relabeled as dynamic VOI -> condition on every permitted exact observation including initial/terminal reveals; (iii) latent mixture called literal historical NS/physics -> state testable statistical transport assumptions; (iv) H2 called full-horizon optimum -> limit claim to its finite horizon/model; (v) planning overhead omitted -> retain charged full runtime; (vi) marginal/conditional quality conflated -> separate conditional predictive and selected-gain diagnostics. These are bounded repairs, not universal no-go conclusions.

### Current primary identities (SHA256; Git blob)

| Source | SHA256 | Git blob at SSOT |
|---|---|---|
|docs/THESIS.md|4f5133edbe04dbef2699c3efab14660b8a81517e800e30822cc2a4a99023876c|350d6a3a8a9b69000aee4f05a027a2a7fa206fa2|
|docs/legacy/QUARTZ_paper.pdf|74d8848012f1ad95fbf2320a8f0392654c800c641d3d37ea90b81dead154547b|f8a981172d0b541cf42d6cf364e94c0a1c5dcaf0|
|COG01_CONTRACT.md|24b3645d5c9c2926d56f1e10651d5e9110748ce84cbf3768fd7a9acf9c9c9f71|d8989aeeb5f57b8de4803b694ee64ee6d52eaf8c|
|COG01_RESULTS_AND_NEXT_PLAN_KO.md|6f7932ac67dd2a1495c5b80a0cf37237bd0beb95d8dac4b2d764f3e6e40fe9a4|e3fb892fe1c7f1604f5b2af7ea0705986edb1fec|
|quartz/experiments/metalevel_decision.py|47f33420223c3c05e495a705ef0ef838825781c3fe9b989b0a351ab479a0c47f|7d874d466502fda70a20b345a71246393f7bf9a6|
|NEXT_C27_PROPOSAL.md|50012cf2fc3e033bbaf323f381dfbb1597a555fa17cd2b77eedb3191f110830f|77ccc3ab39c2e6f22319557f933dfc27249d31e9|
|coding/c26_campaign.py|ee4501d5bfee66adcc15dbfb0b0f93ee50343e3313c7d1ce654143571ecee750|fc2408d3397657e34af25f323822250b31ae850b|
|src/bin/quartz_c26_game_metareasoning.rs|145b586fea7ff7526eb30d15d683580678f1c9dd3a433967f536697957327a01|13fa67b82bfe39ff95b39438314fb4f4761ac16b|

COG files are under docs/research/cognitive_metareasoning_20261120; C26 proposal/coding paths under docs/research/quartz_general_game_metareasoning_20261123.

## 3. Source/API reuse map

| Existing substrate | Actual reusable surface | C27 boundary |
|---|---|---|
|c26_campaign.py154–246|winner, terminal, current_player, legal_moves, apply, replay, enumerate_reachable, D4, canonicalize, board_text|Used in enumeration; preserve semantics/signs and terminal stopping.|
|c26_campaign.py191,250,277|oracle, truth_record, build_sources|Offline truth only. build_sources is C26-specific and computes labels; do not call for a public population pass or blindly retain300 allocation.|
|c26_campaign.py325,349,385|SplitMix64, cue_candidates, feature_key|Preserve identical public cue/feature semantics if scientific comparison requires them.|
|c26_campaign.py389,438|fit_model, model_prediction|C26 TRAIN categorical independent model reference. New joint parameters must use declared TRAIN only; freeze initial marginals then productize those same marginals.|
|c26_campaign.py481,511,664–699|public_request, expected_run, comparison validators|Transport and reference patterns reusable; fixed-marginal policy semantics/schema are not C27 correctness validators.|
|c26_campaign.py762,844,902,932–1038|NativeClient, all-state audit, paired_bootstrap, root aggregation, comparisons, calibration|Reuse existing semantics evidence when identities match; keep root as replicate unit. New conditional/H2 arithmetic and timing require focused checks. No full old campaign rerun needed.|
|Rust binary129–283|request validation/replay, seeded PRNG, root_value/utility/exact, rollout/candidates|Existing GameState/TicTacToe path import and unpruned exact node accounting. Isolate new binary rather than editing core/lib.rs or overwriting C26.|
|Rust binary368–418,593–883|initialize, policy pattern, decision/run records, timing, JSON-line serve|Retain exact-known incumbent and malformed-request recovery. Extend explicitly for joint posterior, support failure and H2 overhead.|
|metalevel_decision.py44–178|Experiment, FiniteProblem, terminal values/risk, possible_outcomes, posterior_after, EVSI, net/ratio one_step, finite bellman|Independent finite arithmetic/observation-transition reference, not a trained MCTS belief or ready-made new campaign.|

Completion evidence: enumeration exit0; source metadata/git reads exit0; initial and post-investigation `git status --porcelain=v1` empty. Writes confined to3 authorized /tmp files. This child did not freeze a split or claim a statistical improvement, model fit, C27 PASS, publication or backup. Parent should incorporate approved evidence into its R01–R03 stage artifacts and receipts, then decide one source frame before labels and continue actual derivation/implementation.
