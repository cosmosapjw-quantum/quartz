# Quartz next-cycle origin watchdog — R1–R3 source subtask

## Scope and exact authority

Read-only source investigation for the user-selected SSOT `87c15dc828d6eea519c219fd8a412fa21dfc464d` in `cosmosapjw-quantum/quartz`. Remote reads used `/home/cosmosapjw/Dropbox/personal_projects/quartz` on device `cosmosapjw` (`edd1b263-56a0-4391-8a2a-677dd00eabf5`). No repository/worktree/ref mutation, fetch, push, cleanup, build or empirical campaign was performed by this worker.

The exact next-cycle proposal at this SSOT is `docs/research/quartz_general_game_metareasoning_20261123/NEXT_C27_PROPOSAL.md`, SHA256 `50012cf2fc3e033bbaf323f381dfbb1597a555fa17cd2b77eedb3191f110830f`. It is a proposal, not evidence that C27 executed.

## Sources inspected directly

| Source | Exact authority and observed identity | Material read |
|---|---|---|
| Initial reachable commit | `15f3a1986a00d548d3da56dab8233b336e931aee`, 2026-04-09 21:19:02 +0900, subject `init` | Commit metadata; theory and Rust sources |
| Legacy import commit | `3bcaa435b6bdca2f821f8990230ea05ddf251a2f`, 2026-07-12 21:41:40 +0900 | Commit metadata and archive members |
| `docs/legacy/QUARTZ_paper.pdf` at SSOT | SHA256 `74d8848012f1ad95fbf2320a8f0392654c800c641d3d37ea90b81dead154547b` | Actual PDF pages 5–6, extracted via `pdftotext -layout` from Git bytes |
| Legacy ZIP member `QUARTZ/PRD.md` | SHA256 `0d84ea7f2a30532d14a33a4685a47192e2bf8a1ab76eb2d5b02f1b9b50e64c50` | Whole PRD, including goals, fairness and resource metrics |
| Legacy ZIP member `QUARTZ/references/user_uploads/quantum_mcts_foundation.md` | SHA256 `fc8bb41be8b8ec43a924d17c41a9a7528bc185a7df96a0d7da1a059d065389c6` | Original physical hypotheses, path normalizer, Lindblad, RG, criticality sections |
| Root `docs/QUARTZ_THEORY.md` | SHA256 `f0e69d54e2e947b6be70f05babc5a58e00f09ce8e377364a6c1c6b272c3cfed3` | Initial controller interpretation and constraints |
| Root `src/mcts/quartz.rs` | SHA256 `34e4161f04561db48fa741aae3505d9ffbda4dde419c33def5d365209e9abb11` | FOCUS/EXPAND/MERGE implementation and controller architecture |
| Root `src/mcts/gvoc.rs` | SHA256 `37c2064557ca66d2fe6c4280eec3710f2d462153d8ad57fd5059f218fbe12527` | VOC-driven widening/contraction and global score |
| C26 source syntheses | `docs/research/quartz_general_game_metareasoning_20261123/{ORIGIN_THEORY_MAP.md,ORIGIN_PROVENANCE.json,ORIGIN_DRIFT_WATCHDOG.md}` at SSOT | Reuse map and limits, compared to the original material above |

The January dates inside the original documents are document dates. They do not establish January Git publication. The reachable root and legacy-import dates above are directly checked commit metadata.

## R1: original objective retained by the proposed joint-posterior study

The original project has two linked objectives: general uncertainty-sensitive computation allocation and better actual game decisions under fair real resources. PRD §§0–2,7–8 explicitly require CPU-only deployment, the same mean time budget, and a baseline with the same neural network, transposition table and parallelism. Elo/playing strength under those conditions is the intended external outcome. The finite small-game joint-belief study is a bounded mechanism test along that path; its success cannot by itself settle Gomoku/NN-evaluation P2 or human-GM P3.

The original paper pp5–6 already contains both KL/Gibbs choice and rational metareasoning. These are mathematical mechanisms, not retrospective names. C27's observed-reveal update and H2 information query directly extend the original Eq(2) purpose: pay for computation only insofar as it changes a downstream decision enough to justify its resource cost. The extension is constructive even if its eventual empirical net gain is nonpositive.

## R2: exact source-to-computation connections

| Original source content | Concrete next-cycle connection | Assumption/evidence boundary |
|---|---|---|
| PDF p5 Eq(1): maximize `sum_a q_a mu_a - beta^-1 KL(q||pi)`; Gibbs optimizer and log-partition optimum | Preserve the action-selection variational identity; use a recursive soft Bellman only if claiming a regularized sequential optimum | Gibbs distribution over metacomputation actions and the joint prior over utilities are different objects. A final softmax over hard-Q values alone does not establish a soft dynamic-programming optimum |
| PDF pp5–6: FOCUS, EXPAND, MERGE, STOP; Eq(2) expected root-decision gain less measured computation cost | Update one coherent joint utility law after incumbent/other reveals; compare stop with an explicit depth-two lookahead | Only information actually available at the controller's history can enter runtime inputs. Keep exact oracle labels offline or behind paid authorized reveal |
| PDF p6 contrast variance `sigma_1^2 + sigma_2^2 - 2 rho sigma_1 sigma_2` | A latent joint family makes observation-dependent changes to unrevealed candidates explicit | Matching initial marginals is necessary for the product-vs-joint causal contrast; marginal prediction quality and dependence-update quality are separate |
| Legacy source hidden-mode/statistical-field intuition | A small latent mixture is a concrete candidate representation | A latent variable is not itself TT geometry, a physical hidden mode, or a derivation of RG/Wick equivalence |
| Root Rust FOCUS/EXPAND/MERGE and GVOC widening | Retain all actual candidate actions and the exact-known incumbent; record how the finite query policy relates to existing controller channels | H2 exact root queries do not automatically implement resident MCTS continuation, MERGE, or partial-budget high-branch search |
| PRD fair real CPU resources | Include feature extraction, posterior updates, branch enumeration, allocation and solver work in time/cost evidence | Node-only savings do not establish CPU efficiency. New seeds or symmetries do not create independent source roots |

## R3: precise correction and current claim states

**Derived local correction.** PDF p6 calls `rho=0` a conservative upper bound when correlation is unknown. For positive mean gap and Gaussian contrast this requires nonnegative correlation. If `sigma_1=sigma_2=1` and `rho=-1/2`, the true contrast variance is `3`; setting `rho=0` gives `2`, thereby underestimating the flip probability. This is a scope defect in the historical wording, not a refutation of the covariance mechanism or a reason to replay old campaigns. The minimal correction is to state the sign condition, or carry the joint law/correlation uncertainty explicitly. The proposed joint model supplies the latter within its own learned-model assumptions.

**Source-supported original intent.** KL/Gibbs, VOC, measured cost, general allocation, actual game decisions and hidden-mode/physical hypotheses genuinely appear in primary historical sources.

**Established conditional mathematics already reusable.** Finite KL variational duality holds with finite utilities and positive prior support. Bayes conditioning and a finite-horizon Bellman equation are valid for the declared joint probability law and cost model; this alone says nothing about fit to actual game distributions.

**Hypothesis to test.** The C26 selected-query calibration defect may be improved by a small latent dependence model and two-step information queries. Correlation omission is not established as the cause of C26's result. The product controller should have the same initial marginals, costs, horizon and information boundary.

**Unresolved original ambitions retained.** General matched-CPU playing-strength gains, NN-evaluation P2, human cognitive P3, TT geometric control, RG/coarse-graining, critical exponents, and physical equivalence remain open. C26's negative primary admission and its implementation/publication corrections must remain separately recorded.

## Bounded drift disposition

No original-goal drift is implied by the C27 proposal itself. It is a direct, calculable development of the original resource-priced computation-value question. The concrete drift risks are (i) replacing the real-time external objective by node counts, (ii) calling a fitted latent mixture a physical theory without a map, (iii) treating a small-game mechanism result as P2/P3 completion, or (iv) replacing substantive theory/code with repeated audit bookkeeping. The smallest correction is the matched-marginal comparison plus measured full controller cost and explicit claim scope, already present in the SSOT proposal. No additional global gate is needed.

## Skill and execution receipt

### Parent steering: distinct follow-up after concurrent C27

The parent reports another concurrent C27 workflow (C08 HEAD `7dfa3c96dc99c57ec5b52054211bf0280de412f5`) with a common-factor model whose posterior changes did not change the finite tested policy trajectories. Those empirical statements were not revalidated by this read-only historical worker. C27 is to remain untouched.

A distinct action-sensitive signed latent direction is historically well motivated by the original **contrast** variance, not merely by generic positive covariance. The PDF's load-bearing variable is `U_1-U_2`; a direction that raises one action while lowering another can affect the decision boundary even when a pure common shift does not. An explicit next-cycle map can use latent perturbations `p_a(u|z)` with action-dependent signed loadings and then determine when conditional expected positive excess or H2 continuation values cross stop/action boundaries. This is a concrete computation-value mechanism; it is not evidence for literal quantum/physical equivalence.

No-go pitfalls: unchanged actions on one source cohort and one fitted common factor do not refute joint beliefs, latent dependence, informational queries, or the original QUARTZ program. Posterior movement alone does not establish useful information. At a fixed history, if every policy action gap exceeds the model-induced change in its action score, the chosen action is stable; a useful affordable certificate should bound those score changes and identify histories where a potentially decisive refinement remains. Its validity must bind the particular cost/horizon/selection rule. A certificate can save planning cost only if its own cost is included. A bounded next result should therefore be a conditional action-gap/stability calculation plus actual crossing examples and measured implementation, retaining both positive examples and null results.

`skill_context_used`: PhysMath Research Loop 1.3.0 SKILL.md; THEORY_LOOP.md R1–R3; POLICY_INTEGRATION.md; Keystone context-survey 2.0.4. The skills were actually read through fully qualified `skill://` package identifiers after shorthand aliases were unavailable to this child. No recursive delegation. Source-read commands completed successfully on the remote device; byte identities above were recomputed from actual Git blobs/archive-member bytes. No empirical result was rerun or revalidated. This temporary report is intended for parent integration into the authorized repository artifact.
