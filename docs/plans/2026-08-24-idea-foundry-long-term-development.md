# Idea Foundry Long-Term Development Implementation Plan

> **For agentic workers:** Use the host's available task-by-task implementation workflow. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a lean, reproducible execution and analysis foundation, then evaluate the 26 registered MCTS hypotheses and compatible combinations under common baseline, A-only, B-only, and A+B comparisons.

**Architecture:** Keep execution identity, scientific study design, and claim promotion as separate layers.  Qualify individual axes before allowing them into a preregistered combination registry; use one shared treatment runner for baseline/A/B/A+B and reserve formal interaction or superadditivity language for a separately powered, held-out confirmatory design.

**Tech Stack:** Python 3.12, Rust/Cargo with the `idea-foundry` feature, PyTorch/CUDA where an axis requires it, strict JSON/JSONL artifacts, pytest, Ruff, and repository-local MCTS runners.

## Global Constraints

- QUARTZ Idea Foundry is trusted-local, personal, single-developer MCTS
  research.  Security hardening, hostile-owner resistance, mount/path attack
  proofs, signatures, and anti-tamper custody are out of scope.
- Scientific integrity remains mandatory: clean committed source, exact config
  and input identity, runtime identity, deterministic seeds, raw failures,
  immutable prior attempts, and explicit execution-to-analysis lineage.
- P0b-C1 is execution infrastructure only.  Establishing it does not validate
  an axis, a combination, efficiency, play strength, interaction, or promotion.
- The scientific target includes both individual hypotheses and compatible
  combinations.  For a pair `(A, B)`, every study uses the same baseline and
  reports baseline, A-only, B-only, and A+B under the same fixed-compute or
  fixed-runtime contract.
- Exploratory evidence that `gain(A+B)` exceeds each solo gain is not by itself
  a formal interaction or superadditivity result.  Those claims require the
  preregistered contrast `Y_AB - Y_A - Y_B + Y_0`, a declared effect scale,
  independent units, a held-out target family, and adequate design power.
- Do not run a new 26-axis omnibus scientific campaign.  The 26-axis runner may
  remain a contract/diagnostic substrate; scientific work proceeds by one
  qualified axis or one preregistered combination family at a time.
- Changed-file and changed-cell focused checks are the default.  Run a full
  suite only for a recorded cross-cutting integration, release/toolchain change,
  final preflight before an authorized claim-bearing run, or a focused failure
  that demonstrates broader risk.
- Scientific run/resume, new result publication, receipt creation, claim
  promotion, push, PR creation, and merge each require the authority applicable
  to that work unit.  A plan or implementation commit grants none of them.
- Preserve `quartz_idea_foundry_skeleton.zip` and
  `quartz_idea_foundry_skeleton.patch` byte-for-byte.

## Status and dependency map

| Stage | Deliverable | Scientific maturity | Next-stage gate |
|---|---|---|---|
| S0 | Trusted-local policy and this roadmap | policy only | external adversarial review |
| S1 | Lean P0b-C1 execution identity | contract only | exact-SHA focused gate and review |
| S2 | P0b-C2 campaign closure and raw-to-derived lineage | contract only | negative lineage matrix |
| S3 | Priority-axis mechanisms and estimands | diagnostic or ablation | axis-specific admissibility |
| S4 | Baseline/A/B/A+B combination substrate | contract only | synthetic known-contrast oracle |
| S5 | Preregistered exploratory pair studies | ablation | held-out eligibility decision |
| S6 | Separately powered confirmatory interaction studies | confirmatory only if all gates pass | claim-ledger review |

The initial axis sequence is A15, A19, A13, A16, then A01.  A02–A12, A14,
A17, A18, and A20–A26 remain `DIAGNOSTIC_ONLY` or
`INVALIDATED_ESTIMAND` until an axis-specific successor spec defines an
independent estimand and falsifier.

---

### Task 1: Freeze the lean trusted-local P0b-C1 successor specification

**Files:**
- Create: `docs/idea_foundry/P0B_C1_TRUSTED_LOCAL_SPEC.md`
- Modify: `docs/CLAIM_LEDGER.md`
- Test: no persistent test file; validate the specification text and links directly

**Interfaces:**
- Consumes: `validate_status_v2(value)` and canonical status keys from `quartz/idea_foundry/status_schema.py`; registered axis order from `load_workflow_specs()`
- Produces: exact writer-visible schemas for `ExecutionIdentity`, `InitialStatePlan`, and `ResumePlan`, plus a finite negative acceptance matrix

- [ ] **Step 1: Write the failing acceptance examples before implementation**

  Specify exact failures for dirty source/config, identity drift, partial state,
  duplicate local start, rejected resume mutation, stale attempt reuse, and
  state that fails `validate_status_v2()`.  The spec must explicitly omit
  hostile Git config, adversarial symlink, mount, ownership, and anti-tamper
  cases.

- [ ] **Step 2: Verify the implementation is still absent**

  Run: `rg -n "class ExecutionIdentity|class InitialStatePlan|class ResumePlan" quartz/idea_foundry`

  Expected: no complete implementation satisfying all three interfaces, so
  `P0B-C1=NOT_ESTABLISHED` remains truthful.

- [ ] **Step 3: Define the minimum observable contract**

  Record one clean committed source/config/input/runtime identity before state;
  derive state from captured config bytes; validate before every durable write;
  reject accidental concurrent writers; preserve previous attempts on every
  rejected resume; and allow only a failed, inactive campaign to resume.

- [ ] **Step 4: Verify the focused specification surface**

  Run: `git diff --check && rg -n "P0B-C1=NOT_ESTABLISHED|trusted-local|ExecutionIdentity|ResumePlan|out of scope" docs/idea_foundry/P0B_C1_TRUSTED_LOCAL_SPEC.md docs/CLAIM_LEDGER.md`

  Expected: zero diff errors and all contract boundaries present without claim
  promotion.

- [ ] **Step 5: Run the affected documentation check**

  Resolve every relative Markdown target introduced by the two changed files.
  Expected: every target exists inside the repository and no result or receipt
  file is added.

- [ ] **Step 6: Commit the passing deliverable**

  ```bash
  git add docs/idea_foundry/P0B_C1_TRUSTED_LOCAL_SPEC.md docs/CLAIM_LEDGER.md
  git commit -m "docs(foundry): specify lean execution identity"
  ```

### Task 2: Establish P0b-C1 on one exact implementation commit

**Files:**
- Create: `quartz/idea_foundry/execution_seal.py`
- Modify: `quartz/idea_foundry/status_schema.py`
- Modify: `quartz/idea_foundry/axis_workflow.py`
- Modify: `quartz/idea_foundry/sequential.py`
- Test: `tests/test_idea_foundry_execution_seal.py`
- Test: `tests/test_idea_foundry_axis_workflows.py`

**Interfaces:**
- Consumes: captured bytes for `configs/idea_foundry.axes.v1.json` and `configs/idea_lab.local.v2.json`; `validate_status_v2(value)`; `load_workflow_specs()` compatibility behavior
- Produces: `capture_execution_identity(...) -> ExecutionIdentity`, `publish_initial_state(...) -> InitialStatePlan`, and `plan_resume(...) -> ResumePlan`

- [ ] **Step 1: Add focused failing tests**

  Cover clean new-run success; dirty tracked or non-ignored input rejection;
  state derived from captured rather than reread config; failed-to-successful
  retry; zero-byte mutation on rejected resume; duplicate invocation rejection;
  partial-write recovery; and canonical status validation before every write.

- [ ] **Step 2: Verify the relevant failures**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_execution_seal.py tests/test_idea_foundry_axis_workflows.py -k "execution_identity or resume or status_v2"`

  Expected: behavior-specific assertion failures from missing identity/resume
  behavior, not import, fixture, or collection failures.

- [ ] **Step 3: Implement the minimum trusted-local behavior**

  Use ordinary Git status/blob identity and SHA-256 as research-object labels,
  canonical JSON, atomic temporary-file replacement, and one simple local
  single-writer lock.  Do not add hostile-config defenses, directory-FD path
  proofs, mount probes, signature keys, security ownership manifests, or a
  custom installer.

- [ ] **Step 4: Verify the focused pass**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_execution_seal.py tests/test_idea_foundry_axis_workflows.py -k "execution_identity or resume or status_v2"`

  Expected: all selected nodes pass with no skip or xfail increase.

- [ ] **Step 5: Run affected integration checks**

  Run: `./venv/bin/ruff check quartz/idea_foundry/execution_seal.py quartz/idea_foundry/status_schema.py quartz/idea_foundry/axis_workflow.py quartz/idea_foundry/sequential.py tests/test_idea_foundry_execution_seal.py tests/test_idea_foundry_axis_workflows.py && ./venv/bin/ruff format --check quartz/idea_foundry/execution_seal.py quartz/idea_foundry/status_schema.py quartz/idea_foundry/axis_workflow.py quartz/idea_foundry/sequential.py tests/test_idea_foundry_execution_seal.py tests/test_idea_foundry_axis_workflows.py && git diff --check`

  Expected: changed Python files are lint/format clean and the diff has no
  whitespace errors.  Do not run a full suite unless a recorded trigger from
  the global constraints occurs.

- [ ] **Step 6: Commit and review the passing deliverable**

  ```bash
  git add quartz/idea_foundry/execution_seal.py quartz/idea_foundry/status_schema.py quartz/idea_foundry/axis_workflow.py quartz/idea_foundry/sequential.py tests/test_idea_foundry_execution_seal.py tests/test_idea_foundry_axis_workflows.py
  git commit -m "feat(foundry): establish trusted-local execution identity"
  ```

  One adversarial code/provenance review must find no P0/P1 or
  acceptance-related P2.  Only then may the exact local SHA be labeled
  `P0B-C1_ESTABLISHED`; this label remains non-scientific.

### Task 3: Close campaign artifacts and raw-to-derived lineage

**Files:**
- Create: `quartz/idea_foundry/lineage.py`
- Modify: `quartz/idea_foundry/sequential.py`
- Modify: `quartz/idea_foundry/axis_workflow.py`
- Modify: `quartz/idea_foundry/meta_analysis.py`
- Test: `tests/test_idea_foundry_lineage.py`
- Test: `tests/test_idea_foundry_meta_analysis.py`

**Interfaces:**
- Consumes: `ExecutionIdentity`, terminal attempt manifests, canonical status-v2 objects, raw axis artifacts
- Produces: `AnalysisIdentity`, `TransformationManifest`, and an admissibility decision that distinguishes contract, diagnostic, ablation, and confirmatory evidence

- [ ] **Step 1: Add focused failing lineage tests**

  Reject empty raw inventories, missing expected artifacts, hybrid inputs from
  different executions, stale-running state, derived data without a declared
  transformation, opaque independent-group labels without member lists, and
  contract/diagnostic records submitted to a scientific pool.

- [ ] **Step 2: Verify the relevant failures**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_lineage.py tests/test_idea_foundry_meta_analysis.py -k "lineage or admissibility or independent"`

  Expected: each negative reaches its named lineage boundary and no existing
  historical artifact is rewritten.

- [ ] **Step 3: Implement closure and admissibility**

  Bind source/input/runtime identity before execution; close the expected raw
  inventory only at terminal state; give later analysis a separate identity;
  connect the two through a transformation manifest; and store explicit member
  lists plus a membership hash for every independent unit.

- [ ] **Step 4: Verify the focused pass**

  Run the Step 2 command again.

  Expected: all selected nodes pass, exact deterministic properties are emitted
  as proof/counterexample states rather than zero-width confidence intervals,
  and no scientific pooling accepts contract-only evidence.

- [ ] **Step 5: Run affected integration checks**

  Run Ruff check/format for the five changed Python files, then
  `./venv/bin/python -m pytest -q tests/test_idea_foundry_axis_workflows.py -k "analysis or resume"` and `git diff --check`.

  Expected: the execution-to-analysis boundary and existing status round-trip
  remain green without a full-suite run.

- [ ] **Step 6: Commit the passing deliverable**

  ```bash
  git add quartz/idea_foundry/lineage.py quartz/idea_foundry/sequential.py quartz/idea_foundry/axis_workflow.py quartz/idea_foundry/meta_analysis.py tests/test_idea_foundry_lineage.py tests/test_idea_foundry_meta_analysis.py
  git commit -m "feat(foundry): close campaign analysis lineage"
  ```

### Task 4: Qualify priority axes with axis-specific estimands

**Files:**
- Modify: `configs/idea_foundry.studies.v1.json`
- Modify: `quartz/idea_foundry/studies.py`
- Modify: `quartz/idea_foundry/control.py`
- Modify: `quartz/idea_foundry/search.py`
- Modify: `quartz/idea_foundry/learning.py`
- Test: `tests/test_idea_foundry_studies.py`
- Test: `tests/test_idea_foundry_contracts_v2.py`

**Interfaces:**
- Consumes: registered axis contracts, fixed-compute/fixed-runtime budget descriptors, explicit independent-unit membership
- Produces: one axis-specific `StudySpec` and falsifiable `StudyOutcome` per qualified axis; no automatic promotion

- [ ] **Step 1: Add one focused failing test set per axis work unit**

  In order: A15 requires same-GPU paired scheduling and interval telemetry; A19
  preserves graph-treatment and replay-seed independence for a held-out loss
  estimand; A13 compares against the adaptive virtual-loss incumbent; A16 uses a
  real pure-state cache with parent-edge statistics separated; A01 executes the
  tactical/VOC stop mechanism and earns `calibrated` only through an independent
  calibration gate.

- [ ] **Step 2: Verify each axis-specific failure**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_studies.py tests/test_idea_foundry_contracts_v2.py -k "a15 or a19 or a13 or a16 or a01"`

  Expected: the selected axis fails its missing mechanism or estimand assertion;
  unrelated axes are not executed.

- [ ] **Step 3: Implement one axis per independently reviewable commit**

  Keep the existing `idea-foundry` feature boundary.  Each implementation names
  its baseline, treatment, unit, endpoint, budget, falsifier, and prohibited
  inference.  A failed or non-estimable result remains a valid terminal outcome.

- [ ] **Step 4: Verify the focused pass for the current axis only**

  Run the Step 2 command with `-k` narrowed to the current axis ID.

  Expected: the current axis nodes and its directly affected shared-contract
  nodes pass; no result artifact or claim update is created.

- [ ] **Step 5: Run the affected integration check**

  For Python-only axes, run changed-file Ruff and the current axis test nodes.
  Name new Rust tests with a `foundry_aNN_` prefix.  For a Rust-wired axis, run
  its exact prefix, for example
  `cargo test --features idea-foundry foundry_a15_` for A15.

  Expected: selected tests execute at least one node and pass.  A full suite
  needs a separately recorded trigger.

- [ ] **Step 6: Commit each passing axis independently**

  Use `feat(foundry): qualify A15 estimand`, then the analogous A19, A13, A16,
  and A01 messages.  Do not combine multiple axis mechanisms in one commit or
  use one axis result to promote another.

### Task 5: Implement the baseline/A/B/A+B combination substrate

**Files:**
- Create: `configs/idea_foundry.combinations.v1.json`
- Create: `quartz/idea_foundry/combination_studies.py`
- Create: `scripts/idea_foundry_combination_study.py`
- Create: `tests/test_idea_foundry_combination_studies.py`
- Modify: `quartz/idea_foundry/meta_analysis.py`

**Interfaces:**
- Consumes: two qualified axis IDs, one common baseline identity, one budget contract, held-out family membership, and independent seed/opening/position groups
- Produces: four-arm rows for `baseline`, `A_only`, `B_only`, and `A_plus_B`; solo gains, joint gain, exploratory joint-dominance probability, and optional preregistered factorial contrast

- [ ] **Step 1: Add focused failing four-arm tests**

  Require all four arms for every independent block, common starting checkpoint,
  identical opponent/opening/color schedule, exact fixed NN-evaluation or fixed
  wall-clock budget, treatment trace coverage, and rejection of duplicated or
  missing units.  Add a synthetic oracle with known zero, positive, and negative
  interaction contrasts.

- [ ] **Step 2: Verify the relevant failures**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_combination_studies.py`

  Expected: the missing registry/runner and four-arm validation fail with
  behavior-specific assertions.

- [ ] **Step 3: Implement the minimum combination runner and estimators**

  Treat `Y` as the preregistered direction-normalized endpoint for which larger
  is better.  Define `gain_A = Y_A - Y_0`, `gain_B = Y_B - Y_0`,
  `gain_AB = Y_AB - Y_0`, and
  `interaction_AB = Y_AB - Y_A - Y_B + Y_0` on the preregistered effect scale.
  Report `Pr(gain_AB > max(gain_A, gain_B))` as exploratory unless Task 6's
  confirmatory contract applies.  Never pool incompatible axes or estimands.

- [ ] **Step 4: Verify the focused pass**

  Run the Step 2 command again.

  Expected: the synthetic oracle recovers its known contrasts, permutation of
  row order changes no estimate, and incomplete/mismatched blocks fail closed.

- [ ] **Step 5: Run affected integration checks**

  Run changed-file Ruff, `./venv/bin/python -m pytest -q tests/test_idea_foundry_meta_analysis.py -k "incompatible or independent or promotion"`, and `git diff --check`.

  Expected: combination records cannot enter incompatible pools or trigger
  promotion; no real campaign is launched.

- [ ] **Step 6: Commit the passing deliverable**

  ```bash
  git add configs/idea_foundry.combinations.v1.json quartz/idea_foundry/combination_studies.py scripts/idea_foundry_combination_study.py tests/test_idea_foundry_combination_studies.py quartz/idea_foundry/meta_analysis.py
  git commit -m "feat(foundry): add four-arm combination studies"
  ```

### Task 6: Run bounded exploration and separate confirmatory interaction work

**Files:**
- Create: `docs/idea_foundry/COMBINATION_STUDY_SPEC_V1.md`
- Modify: `configs/idea_foundry.combinations.v1.json`
- Modify: `docs/CLAIM_LEDGER.md`
- Test: `tests/test_idea_foundry_combination_studies.py`

**Interfaces:**
- Consumes: qualified solo-axis evidence, mechanism-compatible pair registry, four-arm substrate, simulation-based sample design, explicit run authorization
- Produces: immutable exploratory raw results or a separately identified confirmatory result; concise claim-ledger status with negative and inconclusive outcomes preserved

- [ ] **Step 1: Freeze pair eligibility and design before outcome access**

  Select pairs by mechanism compatibility, resource feasibility, and qualified
  solo mechanisms—not by observed combination performance.  Freeze the exact
  pair list, baseline/checkpoint, endpoint, effect scale, unit members, held-out
  family, compute contract, stopping rule, and prohibited claims.  Simulate the
  design and stop as `BLOCKED_DESIGN` if the declared false-pass and power
  targets cannot be met within the resource budget.

- [ ] **Step 2: Verify the preregistration gate**

  Run: `./venv/bin/python -m pytest -q tests/test_idea_foundry_combination_studies.py -k "preregistered or design or held_out"`

  Expected: the frozen registry is accepted, while post-outcome pair edits,
  missing unit membership, and budget drift are rejected.

- [ ] **Step 3: Execute only after separate explicit authorization**

  First run bounded exploratory families and preserve `NEGATIVE`,
  `INCONCLUSIVE`, and `NON_ESTIMABLE` results without outcome-seeking reruns.
  A confirmatory run uses a new run ID, untouched held-out family, exact candidate
  SHA, and its own preregistered interaction contrast.  The RTX 5070 Ti and 64 GB
  RAM affect feasible batch/parallel settings only, not evidence maturity.

- [ ] **Step 4: Verify focused analysis and raw-result closure**

  Run the exact combination test nodes plus the Task 3 lineage nodes against the
  new schemas.  Expected: raw rows, member lists, execution identity, analysis
  identity, and transformation manifest all validate; no historical artifact is
  overwritten.

- [ ] **Step 5: Run the authorized pre-scientific integration gate**

  Record that a claim-bearing execution is imminent.  This is one allowed reason
  for a full repository gate; run it only once on the exact clean candidate SHA,
  after all focused checks and adversarial review are green.  If any required
  Python, Rust, lineage, or design gate fails, preserve the failure and do not
  execute the scientific campaign.

- [ ] **Step 6: Commit evidence separately from implementation**

  The implementation branch contains no result or receipt.  After an authorized
  run, create a separate evidence commit containing immutable raw-result
  references and one concise ledger update.  State exactly whether the result is
  exploratory joint dominance, formal interaction, negative, inconclusive, or
  blocked; never infer efficacy from P0b-C1 or a contract pass.

## Adversarial review and anti-inflation rule

Each task or axis work unit has one falsifiable behavior, one mutating writer,
one adversarial review, and at most one repair-closeout.  Read-only reviewers
classify findings as blockers or follow-ups.  Security-only requests are
out-of-scope unless the owner first changes the trusted-local threat model.
Process/docs wrappers cannot close a mechanism or scientific finding, and new
results cannot appear in an implementation commit.

## Externally observable decisions deferred to their named gate

- The exact compatible pair list is intentionally not selected here.  Task 6
  freezes it before any combination outcome is available, using Task 4's
  mechanism qualification and the registry's compatibility rules.
- Each axis or pair chooses exactly one primary endpoint and one budget scale in
  its own preregistration; this roadmap does not force unlike estimands onto a
  common scale.
- No scientific execution date, sample count, or claim level is authorized by
  this plan.  Task 6's simulation and explicit execution approval decide them.
- Remote CI is optional external evidence for this owner-only project.  Local
  exact-SHA focused evidence remains distinct from remote status, and neither
  alone establishes a scientific claim.
