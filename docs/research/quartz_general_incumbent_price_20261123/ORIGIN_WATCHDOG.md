# C25 R01 — QUARTZ origin, historical drift, and a bounded constructive correction

Status: **completed read-only historical investigation** for parent integration. This artifact completes the delegated historical component, not the parent's entire R01 or C25. No repository files, worktrees, Git refs, original policies, source or archives were modified. No extra child or review pass was started.

## 1. Decisive finding

QUARTZ has a **dual documented origin**: a general, uncertainty-sensitive computation controller and a concrete experimental ambition to improve external game decisions under a fair resource budget on consumer CPUs. The broad metareasoning mechanism is present in the internally January-dated original paper; it was not invented in July or C17. The concrete game-quality and timing-distribution targets are equally explicit in the early PRD. Later claims that game quality was never part of the original scientific objective overstate the historical evidence. Generalization beyond games is a legitimate later research direction, but should be identified as generalization rather than substituted for the original success targets. [O1–O4, T1]

C25 incumbent retention and price-aware stopping is a faithful study of the original **FOCUS/STOP** mechanism. The SPD substrate changes the uncertainty model and removes many original difficulties. Its value is a tractable mechanism study: certified bounds on terminal decision quality and real accounting of computation. It does not establish original MCTS efficiency, widening/reconciliation/allocation behavior, emergent expert signatures, or production hot-path cost. C24's 30.1515% time saving versus PLAIN_ALWAYS is real within its reported experiment; the numeric resemblance to THESIS's 30% NN-evaluation target does not connect the two claims. [O5, T7–T8]

The smallest correction **inside C25** is to make the origin-to-proxy map explicit and tie the new result to terminal decision utility, incumbent loss, and total priced computation, including comparison with immediate stopping. Continue the useful mathematical and coding work already requested. Do not require a historical campaign rerun, new game tournament, or wholesale return to every original module before completing this cycle.

## 2. Authority, dating and inspected state

Primary remote device: `cosmosapjw`, `edd1b263-56a0-4391-8a2a-677dd00eabf5`.

| Object | Observed identity / dating | Meaning |
|---|---|---|
| Original developer worktree | `/home/cosmosapjw/Dropbox/personal_projects/quartz`, HEAD `65e7d3e9596ca259ebfe523b0c74058692b6a755` | Read before and after; unchanged HEAD and existing untracked `QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md` |
| C24 worktree | `/home/cosmosapjw/quantum/research/quartz_general_meta24_20261123`, HEAD `4998c771c08788e4f0da8d2723347121ae16fbf2` | Clean before and after |
| Earliest reachable Git root | `15f3a1986a00d548d3da56dab8233b336e931aee`, commit date `2026-04-09T21:19:02+09:00` | Actual root of both inspected histories; first developer history contains 183 commits |
| Archival PRD | `docs/legacy/QUARTZ_docs.zip` member `QUARTZ/PRD.md`, internally dated `2026-01-28` | Historical document date, **not** a verified January Git commit date |
| Archival paper | `docs/legacy/QUARTZ_paper.pdf`, title page January 29, 2026 | Same date qualification; original paper read directly via in-memory `pdftotext` |
| Legacy corpus import | `3bcaa435b6bdca2f821f8990230ea05ddf251a2f`, `2026-07-12T21:41:40+09:00`, “Track docs/legacy: pre-QUARTZ intuition corpus…” | Verifiable repository inclusion of the older materials |
| Frozen thesis | `7241ad0cfe1e92f28bd8cded5ad5f9fe7d4b4eab`, `2026-07-10T04:14:22+09:00` | Actual initial and only change to `docs/THESIS.md` on both inspected histories |

Read the original worktree's `AGENTS.md`. It identifies QUARTZ as Rust MCTS plus Python AlphaZero learning, specifies project runtimes, and protects the Idea Foundry provenance zip/patch. Ancestor locations on the two supplied paths were checked for AGENTS/CLAUDE files; only the original repository AGENTS file was returned. All research reads used `git --no-optional-locks` or Python standard-library in-memory file/archive analysis. No project code was executed and no tests/builds were rerun.

The present user request and parent's C25 scope remain the execution authority. Historical documents are primary evidence about scientific intent, not a license to reinstate every old workflow restriction.

## 3. Original problem, intuition and success targets

### 3.1 The early product/research question

The internally January-28 PRD identifies the problem as fixed-time/fixed-simulation MCTS wasting effort on easy decisions and underspending on difficult decisions; CPU/GPU inference overhead and the desire to avoid handwritten game strategies add constraints. The proposed treatment is state-dependent reallocation under a fair mean resource budget. Its concise original sentence is:

> “같은 평균 시간 예산에서 더 높은 Elo”

The archive README states:

> “추가 계산”이 아니라 “계산 재배치”로 이겨야 한다.

The name in that archive is **QUAntized Resource Tree Search**, unlike the April repository's **Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable**. Both are historical facts; calling the April expansion the only original expansion erases an earlier preserved definition. [O1–O3]

The PRD's declared success targets include an initial +50 Elo aim versus a common engine/NN/TT/parallel stack, a mean 5-second move budget, easy-move p50 ≤0.5 seconds, hard-move mean ≤15 seconds and p99 ≤30 seconds, plus stability and CPU usability. These are **specified targets**, not validated measurements in this investigation. The same archive discusses a CPU student distilled from a GPU teacher and a multi-game curriculum. Those are product/implementation paths, not prerequisites to the narrow C25 mechanism study. [O1]

The archival fairness text sometimes says to hold the same full time-distribution policy while changing dynamic halting/widening. That wording needs a causal interpretation when reused: hold the admissible resource envelope and evaluation/engine infrastructure fixed while explicitly naming the allocation treatment. Requiring identical realized scheduling would risk controlling away the proposed treatment. This is a historical ambiguity, not a current C25 blocker.

### 3.2 The general mechanism already appears in the original paper

The January-29 paper is titled **QUARTZ: Thermodynamic Value-of-Computation Control for Neural Monte Carlo Tree Search**. Its abstract already says that the agent selects computation to improve the root decision per unit cost. Pages 5–6 specify:

| Original computation | Original function | C25 correspondence |
|---|---|---|
| FOCUS | Reduce uncertainty among already-expanded actions | More numerical work on already-defined candidate responses |
| EXPAND | Evaluate or add an unrepresented candidate | Absent from the fixed SPD action set |
| MERGE | Reconcile inconsistent estimates through transpositions/curvature | Absent from independent fixed RHS solves |
| STOP | Spend no further computation at a node or whole search | Incumbent-preserving, priced stopping |

Section 5.1 says computation costs are measured wall time tracked by EMA. Section 5.2 Eq. (2) subtracts cost from expected root-decision improvement; Section 5.3 uses visit-based root-impact weighting and chooses the largest global VOC. The abstract's improvement-per-cost language and the body's net-value language coexist. C17's ratio-versus-net distinction is therefore a useful mathematical clarification of an existing ambiguity, not the invention of a new scientific subject. C25's explicit price λ with units utility/μs makes the utility/time conversion explicit where the original expression left it implicit. [O4, T1]

The physics intuition was also real and should not be erased. The paper links finite-temperature selection to a KL-regularized variational objective, discusses low-order cumulants as a one-loop analogue, nested-sampling-style rare-mode exploration, surprise modulation, and discrete curvature. Earlier `mcts_study` notes advance stronger literal QFT/Lindblad/RG claims. This investigation establishes that those were proposed mechanisms/analogies in the corpus; it does **not** verify the physical equivalences or large speedup claims. The July thesis explicitly narrows physics motifs to sources of statistical observables. A useful watchdog preserves the original intuition while keeping that evidence distinction. [O4–O6]

### 3.3 The earliest actual Git snapshot contains a substantial implementation

The April-9 root README defines an AlphaZero-style game-playing engine with an adaptive controller, P_flip stopping, prior refresh, Rust search, Python learning, common training/evaluation semantics, and multi-game support. Its theory and controller notes say score shaping is root-local, with a separate virtual-loss runtime controller. Actual `src/mcts/select.rs` gates the QUARTZ scoring branch on `depth == 0`. [O3, O7]

The root's `src/mcts/gvoc.rs` describes fixed-budget replacement by VOC-based allocation, including inside/outside proposal concepts, expansion/contraction, and spending more computation where it can change decisions. The source contains FOCUS/EXPAND/MERGE/STOP-related state and uses unified VOC signals to adjust visible candidate counts. These are **source-inspected mechanisms**; no claim that this exact historical root compiles or achieves its design claims is made. [O8]

The earliest `docs/PUBLICATION_ABLATION_PLAN.md` states the concrete hypothesis: a stronger evaluator allows adaptive stopping to save search budget while maintaining quality. It proposes weak/mid/strong checkpoints, P_flip trajectories, >30% savings for a strong evaluator, and ≥95% move agreement with a fixed-budget reference. Its asserted existing results are historical report claims, not new measurements here. This gives an important origin axis missing from SPD experiments: how information quality/learned evaluator strength changes useful allocation and stop behavior. [O9]

### 3.4 Preserve history without laundering inaccurate explanatory text

Two exact source/document differences in the root commit demonstrate why an origin watchdog should read both documentation and code:

1. `docs/QUARTZ_THEORY.md` lines 82–97 describes most-visited root arms and future visit-count reversal. `src/mcts/quartz.rs` lines 650–695 instead sorts current edge Q values, forms a top-two mean difference with edge uncertainties/correlation, and returns a Gaussian tail. Those are different observables.
2. The theory document describes a visit-proportional penalty. The source's `one_loop_visit_penalty` at lines 1223–1232 returns `-min(hbar_eff, cap)/N_a`; `effective_penalty_v2` uses inverse effective occupancy. Do not transport the document's expression as the actual original executable law.

These are **historical representation/implementation discrepancies**. They are relevant to attribution and transfer, but do not require modifying the protected old source or replaying its campaign during C25. [O6–O8]

## 4. July formalization and C17–C24 transitions

The July thesis itself declares that none of its propositions is VALIDATED. It crystallizes the single-principle objective, the explicit NN-evaluation efficiency target, the predicted signature conjunction, and CPU engineering admissibility. P2 is ≥30% fewer `nn_evals_per_move` on Gomoku 7×7 at paired non-inferior play quality versus A1/legacy_quartz; wall-clock is explicitly not that efficiency metric. P3 requires both macro search-statistic skill invariance and increasing **offline reference** VOC-tightness with evaluator skill. P4 concerns the actual production `observe/should_halt/score_adjustment` path at short budgets. [O5]

| Cycle / exact terminal anchor | Actual transition inspected | Relation to origin |
|---|---|---|
| C17 `e4dc4d6a0d5db03189d2f8465f6d5f019018f20c` | Finite Bayesian metalevel Bellman model; exact ratio/net counterexample; new general research master plan | Clarifies the original VOC principle. Calling games merely an environment and never a scientific objective is too categorical given the January PRD. |
| C18 `097255bd00ffa436c3d43704ed201815e7cbb8ea` | Actual Rust MCTS source traces plus non-game Bernoulli continuation; revision probability calibration | A real bridge attempt. The report explicitly says the first-eight-edge proxy is not the engine's actual initial best_move and real MCTS signed utility is NOT_ESTIMABLE. Shared calibration superiority was not established. |
| C19 `2a359ad049b6a0be4047e52ad281b665cf68e967` | Signed continuation utility, two-part q×m predictor, Beta arms and categorical sensors | Corrects flip probability/value confusion; shifts empirical work to known-utility non-game models. |
| C20 `ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4` | Exact model-based posterior-predictive EVSI and evaluator cost | Makes computation-value semantics explicit under correct known likelihoods; does not establish neural-MCTS likelihood calibration. |
| C21 `351c63ae810b1fa633fe2a75440bf7d0929f92e4` | Faster exact CDF integration and native cost accounting | Addresses cost of metareasoning in the finite model, preserving total-resource accounting. |
| C22 `6e91e3abfbc6727d58c0785ffdf80a7e3e7cabaf` | Actual native STOP/COMPUTE and skipped stochastic draws; total runtime/net worse than ALWAYS on tiny tasks | Constructive negative result: skipping primitive work alone does not show useful resource allocation. |
| C23 `a4417b081037c4d5b49976ddcaeaa3f0771773b2` (closeout amendment); theory `b10ee3238737667c69fe60ec1f841e31c77d623a` | Switch to SPD decision responses and rigorous residual enclosures, motivated by amortizing controller cost over useful work | Explicitly records that deterministic intervals are not a posterior/EVSI; also corrects C22's overgeneral description of P1–P4. Saves matvecs but increases total time. |
| C24 `4998c771c08788e4f0da8d2723347121ae16fbf2` | Fast conditional floating-point enclosures, successful fixed-budget time reduction, retained price-negative result versus STOP | A useful FOCUS/STOP mechanism and numerical engineering result. The utility-price question remains open and directly motivates C25. |

The C17 master plan already named a calibrated real MCTS observation model and actual runtime adapter as unfinished obligations. C18 made a partial attempt; later known-utility models and SPD studies do not silently discharge that obligation. This is an explicit unfinished bridge, not grounds to discard their independently valid results. [T1–T8]

## 5. What C24/C25 preserves, transforms, and leaves unconnected

| Original object | Preserved in SPD lane | Changed / absent | Consequence for C25 claims |
|---|---|---|---|
| Improve the terminal decision under finite compute | Exact normalized action utility and regret; actual work can be skipped | Utilities are manufactured quadratic responses, not game outcomes or learned values | State the decision problem and interpret results within it |
| Adapt effort to decision difficulty | Gaps, spectral conditioning and finite compute affect certificate timing | No evaluator-strength curriculum or learned prior quality | Test source strata as numerical task difficulty; do not call this skill/emergence evidence |
| Stop when further computation is not worth its price | C25 explicitly compares possible gain with priced future work | C24 optimized certificate cost at a fixed continuation cap; deterministic bounds replace expected VOI | Distinguish robust regret upper bound from calibrated expected gain |
| Allocation among compute actions/nodes | FOCUS-like continuation versus STOP | No EXPAND, MERGE, node impact weighting, or candidate omission | C25 is a two-action slice of the controller, not the composed original policy |
| Fair full-cost comparison | Same source, solver, baseline budgets; evaluator and controller time counted | MCTS/NN/TT/parallel stack absent | Preserve full-cost baselines, including PLAIN_STOP and C24 FAST_BLOCK |
| External practical success | Mechanism yields measurable true decision utility and runtime | No Elo, game WDL, NN-evaluation efficiency or original hot path | Original empirical target rows stay untested/unresolved |
| Physics-inspired structure | A mathematically concrete energy/residual construction | No derived map from original free-energy, curvature, rare modes to this SPD response task | Do not infer physical-origin continuity merely from an SPD/energy vocabulary |

### Grounded drift finding

**Object:** repeated C17–C22 “original goal was not game performance” framing plus the later tendency to emphasize increasingly cheap certification.

**Observed defect:** the original PRD explicitly targets matched-budget Elo and CPU timing; C24 still has negative priced improvement versus initial stopping at the inherited primary price despite a positive fixed-budget speed result.

**Impact:** without the dual-origin statement and terminal-utility/cost comparison, a useful proxy-method sequence can be mistaken for completion of the original research purpose.

**Smallest constructive correction:** label C25 as a certified incumbent/price-aware FOCUS–STOP mechanism study, attach the map above, and judge its new results against exact incumbent loss and full priced utility with unchanged controls. Keep the original target and transfer obligations visible. **Current relevance: FIX_SOON within R01/documented interpretation, not BLOCK_NOW for the authorized SPD derivation and implementation.** C23's P2/P4 correction is already closed and should be reused rather than rediscovered as a new blocker.

## 6. Concrete C25 correction to integrate now

1. **Fix the narrow scientific question before results.** “Under valid utility intervals and a specified computation-cost convention, can an incumbent-preserving STOP/FOCUS rule prevent pointwise utility deterioration and improve the full priced outcome compared with immediate stopping and C24 FAST_BLOCK?” This directly instantiates original root-decision improvement and cost sensitivity.
2. **Use precise semantics.** `G_i = max(0,max_{j!=i} U_j-L_i)` is a deterministic upper bound on remaining simple regret. It is not EVSI. A switch with `L_j>U_i` has a true-utility improvement guarantee under enclosure validity. The guarantee is conditional on the current arithmetic/model contract; retention may reject beneficial but uncertified switches.
3. **Separate the two cost statements.** An exact charged-work lower bound can justify a conditional economic stop theorem. Wall-time forecasts/medians support an empirical decision rule and empirical evaluation, not a mathematical lower-bound theorem. Keep λ's unit utility/μs explicit and count all controller/evaluator work in ex-ante outcomes.
4. **Measure the actual tradeoff, not only correctness.** The fresh campaign should count realized incumbent improvements, attempted/rejected switches, retained-incumbent regret, truly avoided work, full elapsed cost, and priced utility against the carried PLAIN_STOP/PLAIN_ALWAYS/C24 FAST_BLOCK controls. Preserve C24's adverse initial-to-final fixture as a regression, not a fresh independent outcome.
5. **Close the historical overstatement in one companion paragraph.** January primary sources support both game-neutral controller design and concrete game-quality/CPU goals. July P1–P4 and C17 generalization are later formulations, not replacements for all earlier intent. No protected historical document needs alteration.
6. **Leave one exact transfer obligation.** A future real MCTS adapter needs action-quality/uncertainty semantics that can link its native observations to terminal decision loss; C18's first-eight-edge flip proxy is insufficient for signed utility. Completing that bridge is distinct from finishing this C25. Do not invent a requirement to retrain, rerun all games, or impose fresh publication gates now.

If C25's net result remains negative, retain that result and identify the cost/utility regime. Pointwise incumbent retention can be a valid safety/correctness gain while the economic claim remains unsupported. Conversely, a positive proxy net result is useful evidence for this mechanism, not automatic P2/P3/P4 validation.

## 7. Compact per-turn origin watchdog proposal

The user requested this for every future research turn. Use one short record in the existing run note; do not create another recursive review system.

```text
ORIGIN_WATCHDOG
anchors: January PRD + January paper; April root source; July THESIS; current adopted scope
current_object: precise decision/computation question and cost unit
preserved: which original mechanism and external success quantity this work addresses
transformed: changed model/assumption/substrate and exact transport condition
evidence: new derivation/measurement, including negative outcomes and strongest relevant control
drift_and_action: none, or object -> observed defect -> effect -> one minimal correction now
open_bridge: one specific unresolved connection; closure evidence for any previously fixed finding
```

Reuse this origin dossier and immutable identities. Only inspect new history or a changed claim when it changes the mapping. A few sentences or one row can satisfy the watchdog on a small turn. The watchdog's output should move the authorized scientific work forward; it should not demand renewed approval or repeated historical audits.

## 8. Source index and exact provenance

All citations below are repository primary artifacts actually read in this investigation. References to previous numerical outcomes mean **historical report evidence inspected**, not independently rerun measurements.

- **O1** — `3bcaa435b6bdca2f821f8990230ea05ddf251a2f:docs/legacy/QUARTZ_docs.zip`, member `QUARTZ/PRD.md`. Archive SHA-256 `927d9fd0201aeb3e8a052b9a5e55943d8cf21608320ac93753ec0595ce133fa2`; member SHA-256 `0d84ea7f2a30532d14a33a4685a47192e2bf8a1ab76eb2d5b02f1b9b50e64c50`. Entire member read.
- **O2** — Same archive, `QUARTZ/README.md`, SHA-256 `7354839a5055fdd18b13307dfed9f181e8ded775e97d92c355418262cbb174c6`; `QUARTZ/docs/DESIGN.md`, SHA-256 `c656318be83b392338c200c3c769e7661f8c9603cbafc7f846be4380009841d2`; `QUARTZ/docs/BASELINE_FAIRNESS.md`, SHA-256 `5dc457809803159be0e537c3fd9d4f8fc3e63223080f94b00b3d9adb3cc4f27e`. Entire members read.
- **O3** — `15f3a1986a00d548d3da56dab8233b336e931aee:README.md`, Git blob `0b188ccc771f89e39d7a86ef4c1dc00b31c8ab33`, SHA-256 `7af0bcccb36b9fd4b97caad582a6260d0a558a28ba46c86d6b462cbe68a81dc3`. Entire file read.
- **O4** — `3bcaa435b6bdca2f821f8990230ea05ddf251a2f:docs/legacy/QUARTZ_paper.pdf`, Git blob `f8a981172d0b541cf42d6cf364e94c0a1c5dcaf0`, SHA-256 `74d8848012f1ad95fbf2320a8f0392654c800c641d3d37ea90b81dead154547b`. Title, abstract, contents and pages 5–6 read directly; not a full proof audit of the paper.
- **O5** — `7241ad0cfe1e92f28bd8cded5ad5f9fe7d4b4eab:docs/THESIS.md`, Git blob `dbb91db955d6cbc21683d27cf415f0245cf9f02d`, SHA-256 `02ffa31af60ecbb1fa46f8bfebe4590caa365aca22f3c3d17817017a8ef5cb8f`. Entire file read; history confirms the same sole file-change commit in both inspected lineages.
- **O6** — Root `docs/QUARTZ_THEORY.md`, Git blob `78de060129578a72afb931ff6feb0e96a6dac8e3`, SHA-256 `f0e69d54e2e947b6be70f05babc5a58e00f09ce8e377364a6c1c6b272c3cfed3`; entire 281-line text read. Legacy `docs/legacy/mcts_study/v6.0/quantum_mcts_foundation.md` at import commit: Git blob `445c1dfc36fd956a36fd11c1fddb67c3fbc75038`, SHA-256 `fc8bb41be8b8ec43a924d17c41a9a7528bc185a7df96a0d7da1a059d065389c6`; opening/motivation sections inspected, not complete proof validation. Also inspected opening sections of legacy `qft-mcts-research-proposal.md` and `effective-action-mcts.md` as historical hypotheses.
- **O7** — Root `src/mcts/select.rs`, Git blob `87c31dd6603c32ca974637931c8d3064ab7e8d4d`, SHA-256 `03c3e30048cd963b86859f3f591a5c8f7f436dfc56dac912eb09cd8ada8dfe15`; relevant scoring/root dispatch inspected. Root `docs/CONTROLLER_NOTES.md`, Git blob `ba08e663510819341ffb2556d586db2f9fc592e0`, SHA-256 `1491beadd124609730b9778b6ce435b086077b4992186964eb9766bfafd4291b`; entire file read.
- **O8** — Root `src/mcts/quartz.rs`, Git blob `75bfe84e59796e66f93a36050ee76eb776cce14a`, SHA-256 `34e4161f04561db48fa741aae3505d9ffbda4dde419c33def5d365209e9abb11`; selected statistics, compute actions, flip and penalty functions inspected. Root `src/mcts/gvoc.rs`, Git blob `acd0130270d1f70550e3c8b998ee8817efdd1af4`, SHA-256 `37c2064557ca66d2fe6c4280eec3710f2d462153d8ad57fd5059f218fbe12527`; design, state and update functions inspected.
- **O9** — Root `docs/PUBLICATION_ABLATION_PLAN.md`, Git blob `ce00b92f9ab92786ec7c66dc5fabc8ff9610f06b`, SHA-256 `c5f04037849d1b4bf4cb4ee8db0b642b50e72a74097b75a3b766c00c82fd562d`. Entire file read.
- **T1** — C17 `docs/research/quartz_general_metareasoning_20261122/QUARTZ_GENERAL_MASTER_PLAN_v1.md` at `e4dc4d6a0d5db03189d2f8465f6d5f019018f20c`, Git blob `95cf87778db5ed34928f9abc37b5a49d0601fe71`, SHA-256 `f1947f0b546ae7b3d70a8353994b0017344f67d3c287f8dd1b80c203c3838ad0`; entire file read, plus R01/R02 texts at file-change commits `2ec9a80b1b3adb88aea910c85f168735c6af0f94` and `72427c69afe1b02a17928c9c244d668dc2ec5159`.
- **T2** — C18 `docs/research/quartz_general_calibration_20261122/RESULTS_AND_NEXT_KO.md`, last change `097255bd00ffa436c3d43704ed201815e7cbb8ea`; inspected scientific/model/source/heldout sections.
- **T3** — C19 `docs/research/quartz_general_signed_voc_20261122/RESULTS_AND_NEXT_KO.md`, last change `2a359ad049b6a0be4047e52ad281b665cf68e967`; inspected model, source, signed utility and calibration sections.
- **T4** — C20 `docs/research/quartz_general_predictive_voc_20261122/RESULTS_AND_NEXT_KO.md`, last change `ba100d13b36a49151fc4a7b5dd379fa2eb79e4c4`; inspected theoretical and implementation/source definitions.
- **T5** — C21 `docs/research/quartz_general_cdf_native_20261122/RESULTS_AND_NEXT_KO.md`, last change `351c63ae810b1fa633fe2a75440bf7d0929f92e4`; inspected CDF law, actual source/implementation and numeric scope.
- **T6** — C22 `docs/research/quartz_general_active_gate_20261122/RESULTS_AND_NEXT_KO.md`, last change `6e91e3abfbc6727d58c0785ffdf80a7e3e7cabaf`; inspected theory/setup and entire closing scientific outcomes/handoff.
- **T7** — C23 `docs/research/quartz_general_residual_stop_20261123/THEORY.md` at file-change `b10ee3238737667c69fe60ec1f841e31c77d623a`; entire 80-line document read. Opening results/source definition in `RESULTS_AND_NEXT_KO.md`, last change `a4417b081037c4d5b49976ddcaeaa3f0771773b2`, also inspected.
- **T8** — C24 `docs/research/quartz_general_cheap_certificates_20261123/REPORT.md` at `4998c771c08788e4f0da8d2723347121ae16fbf2`, Git blob `5b6b80692ffe1206051560508ee0ff7ed67f823c`, SHA-256 `b1674efbdf306252391e11cb19cb8d4f3313d97f7b0cecd66ff916641ca3a8e9`; scientific report read. Same commit's `NEXT_STAGE.json`, Git blob `d5554e1f08b5285ec114b667137decacb5732343`, SHA-256 `3d4f4bb21e3e8ff3c9e4138ced1260476af61c9dae7e7a2cc4531cc8d05686af`; entire file read.

## 9. Actual commands, evidence status and handoff

Remote commands completed through `remote_desktop_commander_start_process` with Python standard-library orchestration. Principal commands were:

```text
git --no-optional-locks -C <repo> rev-parse HEAD
git --no-optional-locks -C <repo> status --short
git --no-optional-locks -C <repo> rev-list --max-parents=0 HEAD
git --no-optional-locks -C <repo> log --reverse --format='%H %ad %s' --date=iso-strict
git --no-optional-locks -C <repo> ls-tree -r --name-only <commit>
git --no-optional-locks -C <repo> show <commit>:<path>
git --no-optional-locks -C <repo> log -1 --format='%H %ad %s' --date=iso-strict -- <path>
git --no-optional-locks -C <repo> rev-parse <commit>:<path>
zipfile.ZipFile(io.BytesIO(git_show_bytes)).read(<member>)
pdftotext - -
pdftotext -f 5 -l 6 -layout - -
```

Read-only process receipts: PIDs `3971789`, `3972056`, `3972363`, `3972576`, `3972986`, `3972931`, `3973399`, `3973700`, `3973890`, `3974093`, `3974823`, `3975140`, `3975652`, `3976881`. Relevant Git reads and PDF extraction reported exit 0. A first log query used `--reverse --max-count=80`, which returns the latest 80 in reverse; it was corrected by reading the full original log and selecting the first entries. The root identity was independently established with `rev-list --max-parents=0`. Large combined outputs were followed by narrower reads of the decisive sources; no claim of a full corpus/proof audit is made.

**Evidence states:** provenance and text/source findings are directly inspected; historical scientific results are report-supported; the C25 crosswalk and minimal correction are reasoned recommendations; actual future MCTS transfer, historical paper proof validity and C25 implementation/science remain outside this delegated result. Confidence is high in the dual-origin and specific scope distinctions because the PRD, paper, earliest Git documents, source and later thesis are mutually traceable. Confidence in independent creation dates before the July corpus import is limited to document-internal dates.

**skill_context_used:**

- PhysMath Research Loop **1.3.0**, actually read `skill://plugins_6acdfbba921881919e0c17843706aef4/physmath-research-loop/SKILL.md`.
- Actually read its `references/THEORY_LOOP.md` and applied the R1 contract: narrow calculable question, original intent/assumptions and selected scope.
- Actually read `references/POLICY_INTEGRATION.md`: finite review, typed identity, reuse rather than historical rerun, minimal constructive correction and direct scientific progress.
- Actually read `references/SUBAGENT_MODE.md`: parent-owned integration, bounded read-only task, no recursive delegation/review and concrete evidence return.
- Keystone `skill://plugins_6a512fae91f881918208460aae465ff9/context-survey/SKILL.md`, actually read and applied primary-source/read-only/evidence-separation requirements. Its ancillary checkpoint-resource lookup returned unavailable; no missing content was invented or treated as an execution blocker.

**Checkpoint:** objective completed = origin and drift map; assumptions = current C25 user scope remains authoritative; artifacts = this report and optional machine-readable companion; verification = exact source identities, source/code comparison and unchanged final repository state; blocker = none for parent C25 integration; next action = parent incorporate the origin correction and compact watchdog in R01, then execute the already-authorized theory/coding sequence. Recommended Keystone handoff: **implementation by parent within the existing C25 task**, without additional permission or audit stages.
