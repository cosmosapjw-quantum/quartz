# QUARTZ Idea Foundry — Experiment Candidate Atlas

This directory is the design atlas for the experimental program layered on top
of the current Rust MCTS engine and Phase-15 Python trace infrastructure.  It is
not a claim that all candidates work.  It is a contract for keeping ideas
alive, falsifiable, and runnable without merging them into one un-auditable
score formula.

The companion skeletons live in:

- `src/mcts/foundry/` — Rust runtime and meta-action contracts; deliberately not
  wired into `mcts::mod.rs` by this scaffold.
- `quartz/idea_foundry/` — Python replay, counterfactual, model, and analysis
  skeletons.
- `configs/idea_foundry.axes.v1.json` — machine-readable axis registry.

## 1. Existing repository contracts to preserve

The current repository already provides the correct low-level seams:

1. `mcts::policy::SearchPolicy` exposes periodic `observe`, O(1)
   `score_adjustment`, `should_halt`, and telemetry.
2. `SearchSnapshot` and `EdgeView` expose root and edge statistics without
   handing a policy mutable tree access.
3. `PolicyCachePublisher` uses immutable `PolicyCache` values and `ArcSwap`,
   keeps arrays edge-position indexed, and rejects stale certificates.
4. Phase-15 distinguishes search-relevant signatures from pure post-hoc
   readouts so candidate comparisons can reuse one trace.
5. `experiment_manifest.py` records source hashes, resolved assumptions,
   prohibited inferences, runtime provenance, and output hashes.

The foundry extends these seams.  It must not bypass them with a second tree,
a second hidden prior, or untracked in-place mutations.

## 2. Shared design rule

Every online module emits a `MetaProposal`; it does not directly alter PUCT.

```text
MetaProposal
  axis_id
  action: STOP | SAMPLE | CHALLENGE | WIDEN | DEEPEN | PROVE |
          RESAMPLE_MODE | MERGE_OR_SHARE | SET_BATCH | SET_INFLIGHT |
          SET_THREADS | REANALYSE | ARCHIVE_STATE | NOOP
  expected decision-regret reduction: mean + conservative lower bound
  measured or predicted cost: NN evals, CPU ms, GPU ms, energy proxy
  confidence
  activation guard
  explanation + telemetry
```

The conservative arbitration score is

```text
net_lcb = LCB(expected decision-regret reduction)
          - price_nn * nn_evals
          - price_cpu * cpu_ms
          - price_gpu * gpu_ms
          - price_energy * energy_proxy
```

Only one explicit computation action is selected at a checkpoint.  Several
shadow observers may run, but several active modules may not independently add
bonuses to the same score.

## 3. Candidate atlas

| ID | Axis | Plane | Initial disposition | Rust skeleton | Python skeleton |
|---|---|---|---|---|---|
| A01 | Calibrated stop council | control | shadow | `A01StopCouncil` | `A01StopCouncil` |
| A02 | Static-anchor regularized policy improvement | policy | shadow | `A02StaticAnchorRpo` | `A02StaticAnchorRPO` |
| A03 | MC/epistemic/drift/bias uncertainty decomposition | belief | shadow | `A03UncertaintyDecomposition` | `A03UncertaintyDecomposition` |
| A04 | KG/VOC computation allocator | control | shadow; low-budget stop remains closed | `A04KgVocAllocator` | `A04KgVocAllocator` |
| A05 | Forked counterfactual meta-action teacher | learning | seed | `A05CounterfactualMetaTeacher` | `A05CounterfactualMetaTeacher` |
| A06 | Gumbel candidate selection + Sequential Halving | candidates | mechanism-valid / challenger | `A06GumbelSequentialHalving` | `A06GumbelSequentialHalving` |
| A07 | Residual partition-mass widening | candidates | seed | `A07ResidualEvidenceWidening` | `A07ResidualEvidenceWidening` |
| A08 | Tactical sentinel and proof backend | proof | conditional / game-specific | `A08TacticalProofBackend` | `A08TacticalProofBackend` |
| A09 | H3 entropy–margin change-point router | control | shadow; old zero-floor trigger not reused | `A09H3ChangePointRouter` | `A09H3ChangePointRouter` |
| A10 | Prior-refresh specialist | policy | dormant conditional expert | `A10PriorRefreshSpecialist` | `A10PriorRefreshSpecialist` |
| A11 | Dynamic live-set particle search | search backend | seed | `A11DynamicLiveSetParticles` | `A11DynamicLiveSetParticles` |
| A12 | JSD-preconditioned locally balanced root sampler | geometry/allocation | seed | `A12JsdLocallyBalancedSampler` | `A12JsdLocallyBalancedSampler` |
| A13 | Pending-flow / WU-UCT correction | parallel | conditional | `A13PendingFlowWuUct` | `A13PendingFlowWuUct` |
| A14 | Whole-path semantic LSH diversity | parallel | high-thread shadow | `A14SemanticPathLsh` | `A14SemanticPathLsh` |
| A15 | Service-curve batch/inflight scheduler | systems | mechanism-valid | `A15ServiceCurveScheduler` | `A15ServiceCurveScheduler` |
| A16 | Monte-Carlo graph/state sharing | graph | seed | `A16MonteCarloGraphSharing` | `A16MonteCarloGraphSharing` |
| A17 | B13 finite-N curvature readout | readout | mechanism-valid; decision-neutral evidence | `A17B13CurvatureReadout` | `A17B13CurvatureReadout` |
| A18 | Diffusion-regularized deterministic evaluator | representation | seed | `A18DiffusionRegularizedEvaluator` | `A18DiffusionRegularizedEvaluator` |
| A19 | RW-ResT Lite evaluator | representation | seed | `A19RwRestLiteEvaluator` | `A19RwRestLiteEvaluator` |
| A20 | Regret/instability state archive | training control | seed | `A20RegretStateArchive` | `A20RegretStateArchive` |
| A21 | Coherence-gated signed path feature | analysis | analysis-only shadow | `A21CoherenceSignedPathShadow` | `A21CoherenceSignedPathShadow` |
| A22 | Physics-analogy falsification dashboard | analysis | analysis-only | `A22PhysicsFalsificationDashboard` | `A22PhysicsFalsificationDashboard` |
| A23 | CPU incremental pattern student | deployment | seed | `A23CpuIncrementalPatternStudent` | `A23CpuIncrementalPatternStudent` |
| A24 | Learned state-dependent budget gate | control | seed | `A24LearnedBudgetGate` | `A24LearnedBudgetGate` |
| A25 | MENTS / soft-backup ablation | search policy | dormant | `A25MentsSoftBackup` | `A25MentsSoftBackup` |
| A26 | Exact nested-contour validation lab | analysis backend | analysis-only | `A26NestedContourExactLab` | `A26NestedContourExactLab` |

## 4. Documents

- [01 — Control, risk, policy, and counterfactual learning](01_control_policy.md)
- [02 — Candidate coverage, widening, proof, and allocation](02_candidates_allocation.md)
- [03 — Parallel execution, particle backends, path diversity, and graph sharing](03_parallel_backends.md)
- [04 — Readout, evaluator architectures, deployment, and training control](04_representation_training.md)
- [05 — Physics-inspired shadows, falsification, and evidence discipline](05_analysis_physics.md)
- [06 — Per-axis execution, sequential campaigns, and meta-analysis](06_execution_and_meta_analysis.md)

Each axis section includes:

- the question it answers;
- current evidence and non-claims;
- exact repo insertion points;
- required inputs and outputs;
- Rust and Python skeleton symbols;
- the first dependency-light test;
- the first live-engine test;
- promotion and demotion criteria.

## 5. Execution order

The recommended order is evidence-driven, not conceptual:

1. **Trace-only:** A01, A02, A03, A09, A17, A21, A22.
2. **Synthetic candidate bank:** A06, A07, A12, A25, A26.
3. **Counterfactual resident-root teacher:** A05 with STOP/SAMPLE/WIDEN first.
4. **Live root allocation:** A04, A06, A07, A08.
5. **Parallel/system lanes:** A13, A14, A15, A16, then A11.
6. **Training/deployment:** A18, A19, A20, A23, A24.
7. Add multi-action arbitration only after individual proposals have calibrated
   gain and cost labels.

## 6. Promotion ladder

```text
SEED
  code skeleton and explicit assumptions
MECHANISM_VALID
  unit/synthetic signature works
SHADOW
  runs on real traces without changing search
CONDITIONAL
  useful in a preregistered regime
ACTIVE_EXPERIMENTAL
  causal online effect under paired protocol
DEPLOYMENT_CANDIDATE
  repeated matched-time strength/efficiency result
```

`DORMANT` means a role is currently unsupported, not that the parent idea has
been deleted.  `ANALYSIS_ONLY` means the module may generate observables but is
not allowed to control search.

## 7. Baseline stack

Attribution requires a layered baseline rather than one weak PUCT baseline:

```text
B0 serial PUCT
B1 tree reuse + eval cache + micro-batching
B2 B1 + existing tree-parallel virtual-loss path
B3 B1 + explicit unobserved/pending-count correction
B4 B3 + TT as state/evaluation cache
B5 Monte-Carlo graph/DAG statistic sharing
B6 regularized policy or Gumbel/Sequential Halving
H1 learned dynamic stopping
H2 virtual-expansion/policy-stability stopping
Q  selected foundry module or arbiter
```

Every runtime result must report both fixed NN-evaluation and fixed wall-clock
comparisons.  Simulation counts are not comparable across proof, particle,
MCTS, and neural-evaluator backends.

## 8. Primary literature map

The atlas uses the following papers as controls and counterexamples, not as a
claim that QUARTZ already inherits their guarantees:

- Gumbel policy improvement: https://openreview.net/forum?id=bERaNdoegnO
- Regularized policy optimization: https://proceedings.mlr.press/v119/grill20a.html
- Convex-regularized MCTS: https://proceedings.mlr.press/v139/dam21a.html
- Value of computation in MCTS: https://proceedings.mlr.press/v124/sezener20a.html
- BAI-MCTS: https://arxiv.org/abs/1706.02986
- Dynamic stopping: https://ojs.aaai.org/index.php/AAAI/article/view/16100
- Virtual-expansion stopping: https://arxiv.org/abs/2210.12628
- Epistemic MCTS: https://openreview.net/forum?id=Tb8RiXOc3N
- Bayesian online planning: https://arxiv.org/abs/2406.02103
- WU-UCT: https://openreview.net/forum?id=I6fJI8kj1Z8H
- PMCTS: https://arxiv.org/abs/2605.08982
- PTSA/JSD state abstraction: https://arxiv.org/abs/2310.06513
- Locally balanced discrete MCMC: https://arxiv.org/abs/1711.07424
- Dynamic nested sampling: https://arxiv.org/abs/1704.03459
- Discrete diffusion: https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html
- RandWire: https://openaccess.thecvf.com/content_ICCV_2019/html/Xie_Exploring_Randomly_Wired_Neural_Networks_for_Image_Recognition_ICCV_2019_paper.html
- DDW: https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Differentiable_Dynamic_Wirings_for_Neural_Networks_ICCV_2021_paper.html
- Go-Exploit: https://arxiv.org/abs/2302.12359
- Regret-guided search control: https://arxiv.org/abs/2602.20809
- Rapfi CPU evaluator: https://arxiv.org/abs/2503.13178

## 9. Non-negotiable safety and attribution rules

- Stop never uses stale cache or pending evaluations as evidence.
- `edge_pos`, `action_id`, and policy index remain distinct.
- Search-time, readout-time, training-target, and deployment roles are separate
  experiments even when they share one formula.
- Candidate-omission risk and visible-candidate ranking risk are reported
  separately.
- Restart-per-chunk and true root-continuation traces are never pooled.
- A learned controller is split by position/game group, never by adjacent
  checkpoint rows from the same root.
- Hardware scheduler artifacts are profile-specific: CPU, CUDA, and ROCm are
  not interchangeable.
- Physics-inspired observables do not control search until they add predictive
  information beyond ordinary statistical features.
