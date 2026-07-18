# Local Experiment Lab

This document is the execution runbook for the QUARTZ idea foundry. It is
intentionally broader than the current production controller: existing
mechanism labs are executable now, while proposed experiments are registered
with explicit blockers rather than being silently omitted.

The local-lab contract is:

1. **Every run gets an isolated output directory.**
2. **Every executed command is logged.**
3. **The environment, Git/source/input hashes, and registry hash are recorded
   before execution.**
4. **Planned lanes are visible but never pretended to be executable.**
5. **A failed role/regime does not delete the parent idea.**
6. **Execution status and scientific evidence status are independent.**
7. **A technical contract failure stops the serial campaign; a preregistered
   negative result is preserved and does not stop independent first gates.**

The orchestrator is `scripts/idea_lab.py`. The compatibility registry remains
`configs/idea_lab.local.v1.json`; the controlling 26-axis campaign registry is
`configs/idea_lab.local.v2.json`.

For one-command-per-axis execution and post-run aggregation, the dedicated
entrypoints are `scripts/idea_foundry/a01_*.py` through `a26_*.py`,
`scripts/idea_foundry_run_all.py`, `scripts/idea_foundry_analyze_campaign.py`,
and `scripts/idea_foundry_meta_analyze.py`. Their complete artifact and
effect-record contract is documented in
`docs/idea_foundry/06_execution_and_meta_analysis.md`. These commands reuse the
same v2 `first-gate-all` order; they do not replace or promote blocked live
lanes.

Before launching a real ablation, run the source-stable release bundle:

```bash
venv/bin/python scripts/idea_foundry_preflight.py \
  --run-id pre-ablation-<run-id> \
  --mode release \
  --python venv/bin/python
```

The bundle is an execution-readiness gate, not a scientific promotion gate.
Its finite branch inventory and retained artifacts are specified in
`docs/idea_foundry/06_execution_and_meta_analysis.md`.

---

## 0. Controlling 26-axis campaign

The v2 registry covers A01--A26 exactly once in the `first-gate-all` suite and
registers separate blocked live/CUDA promotion roles where needed. The default
suite runs this dependency order:

```text
A03.trace -> A09.trace -> A01.trace -> A02.shadow -> A17.trace
 -> A21.analysis -> A22.analysis
 -> A06.synthetic -> A07.synthetic -> A12.synthetic -> A25.synthetic
 -> A26.analysis -> A05.counterfactual
 -> A04.shadow -> A08.conditional_audit
 -> A13.synthetic -> A15.system -> A14.shadow -> A16.cache_only
 -> A11.synthetic
 -> A18.representation -> A19.representation -> A23.deployment
 -> A20.training_control -> A24.training_control
 -> A10.conditional_audit
```

Every step calls `scripts/idea_foundry_axis_gate.py`, which exercises the
registered Python skeleton with deterministic fixtures and emits versioned
artifacts. Despite the role-bearing lane names, all 26 commands in this suite
are **synthetic contract preflights**. The role and registry evidence fields
describe the intended later experiment and pre-existing registry state; they
are not evidence newly earned by this run. No Phase-15 trace bundle, resident
MCTS fork, live controller, training corpus, or hardware timing study is
consumed here.

Inspect the complete plan without executing it:

```bash
venv/bin/python scripts/idea_lab.py plan \
  --config configs/idea_lab.local.v2.json \
  --suite first-gate-all \
  --profile cpu \
  --json
```

Start and inspect a stable campaign:

```bash
venv/bin/python scripts/idea_lab.py run \
  --config configs/idea_lab.local.v2.json \
  --suite first-gate-all \
  --profile cpu \
  --run-id idea-foundry-contract-preflight

venv/bin/python scripts/idea_lab.py status \
  --run-id idea-foundry-contract-preflight
```

Resume only after the status report identifies the interrupted or failed step:

```bash
venv/bin/python scripts/idea_lab.py resume \
  --config configs/idea_lab.local.v2.json \
  --run-id idea-foundry-contract-preflight
```

Resume skips a step only when the registry, Git commit, tracked source, inputs,
expanded command/artifact plan, embedded manifest source hashes, and artifact
schema/SHA-256 still match. Once any upstream step is re-executed, every later
step in that lane is re-executed rather than mixing new upstream state with
stale downstream artifacts. `--overwrite` is not a v2 resume mechanism and is
rejected for a non-empty v2 campaign.

The runtime status vocabulary is `planned`, `blocked`, `running`, `succeeded`,
`completed_no_promotion`, `failed`, `timeout`, `interrupted`, and `skipped`.
`completed_no_promotion` is a scientific-negative terminal state, not a
technical failure. A10 records an ineligible audit as
`DORMANT_NO_ELIGIBLE_SLICE` while keeping the default policy `no_refresh`.

The promotion-only order is retained separately and remains fail-closed:

```text
A04.live -> A06.live -> A07.live -> A08.live
A15.cuda -> A18.cuda -> A19.cuda
```

The live lanes require resident snapshots/checkpoints or a tactical corpus, and
the promotion commands are intentionally unregistered until they consume those
inputs. CUDA hardware proof must be regenerated in the designated interpreter
at execution time, but a passing doctor alone cannot unblock a lane without its
matched workload, checkpoint, and real promotion command.

### A15--A19 ablation-readiness campaign

The executable readiness chain is separate from the blocked promotion chain:

```text
A15.ablation_readiness
  -> A18.ablation_readiness
  -> A19.ablation_readiness
```

Plan and run it with the CUDA interpreter:

```bash
venv/bin/python scripts/idea_lab.py plan \
  --config configs/idea_lab.local.v2.json \
  --suite a15-a19-ablation-readiness \
  --profile cuda \
  --json

venv/bin/python scripts/idea_lab.py run \
  --config configs/idea_lab.local.v2.json \
  --suite a15-a19-ablation-readiness \
  --profile cuda \
  --run-id a15-a19-ablation-readiness-<run-id>
```

Every lane terminates as `completed_no_promotion`. The chain resolves the old
execution-contract mismatches but does not turn `A15.cuda`, `A18.cuda`, or
`A19.cuda` into an executable promotion lane.

The source-current acceptance record is
`results/idea_lab_local/a15-a19-ablation-readiness-20260718-v3/`. Its
`campaign_state.json` records one attempt per lane and `verified_skip` for all
three steps after resume. Relative and absolute spellings of the same virtualenv
launcher are canonicalized before the resume fingerprint is compared; the
virtualenv launcher itself is retained rather than resolved to the base Python.

| Lane | What is actually exercised | What remains blocked |
|---|---|---|
| A15 | Four-cell paired CPU/CUDA grid, identical model state and paired input bytes, semantic output parity, pinned deterministic runtime | Full 24-cell profile on the actual shipped checkpoint plus paired search quality per wall-clock |
| A18 | Three real `latest.pt` bootstraps, matched direct architecture/parameters/FLOPs, two updates per arm, frozen controller, training-replay diagnostics | Execution of the preregistered 200-update offline study, game/trajectory-group holdout evidence, fixed-evaluation/fixed-time play result, and production-compatible export |
| A19 | Three real replay/status inputs with hash-bound `latest.pt` sources, frozen controller contract, deterministic split schedules, and eight exact-density graph topologies | In-repo proxy trainer/evaluator that parses model state and recomputes raw metrics, measured shortlist, and shortlisted self-play |

The A15 full service-curve command remains diagnostic:

```bash
venv/bin/python scripts/a15_matched_service_curve.py \
  --profile full \
  --output-dir results/idea_foundry/a15_matched_service_curve_full_<run-id>
```

A18 has a separate preregistered study spec. Its evaluation inputs are derived
reproducibly from a different source seed while excluding every exact float32
state found in the paired training replay:

```bash
venv/bin/python scripts/a18_prepare_holdouts.py

venv/bin/python scripts/a18_evaluator_ablation.py \
  --spec configs/a18_evaluator_ablation.study.v1.json \
  --output results/idea_foundry/a18_evaluator_study_<run-id> \
  --device cuda \
  inspect
```

The derivation receipts record the observed overlap, exclusions, source and
output hashes, and retained positions. This proves exact-state and source-seed
separation only; the available replay metadata does not support a game- or
trajectory-group generalization claim. The tracked
`docs/idea_foundry/A18_STUDY_INPUT_RECEIPT.json` records the current local
derived-input hashes and explicitly marks them as repo-local regenerable inputs,
not embedded portable evidence. The actual 200-update study is not part of the
readiness suite and must be launched deliberately by replacing `inspect` with
`run` and choosing a new output directory.

A18 readiness checkpoints use a deterministic QUARTZ ZIP container with an
`.a18ckpt` suffix and the local A18 loader. They are evidence artifacts, not yet a
drop-in `torch.load` or production deployment format.

A19 deliberately emits an empty `not_measured` shortlist. Supplying external
proxy rows is rejected with `PROXY_EXECUTOR_NOT_IMPLEMENTED`; a receipt cannot
authenticate arbitrary metric values. Measured finalization stays disabled
until an in-repository trainer/evaluator recomputes raw predictions and metrics
from the frozen split, controller, topology, and checkpoint identities.

---

## 1. Environment profiles

QUARTZ uses three local profiles:

| Profile | PyTorch device | Typical use |
|---|---|---|
| `cpu` | `cpu` | synthetic labs, analyzers, Rust tests, CPU service curve |
| `cuda` | `cuda` | NVIDIA training, live checkpoint runs, GPU service curve |
| `rocm` | `cuda` | AMD PyTorch/ROCm (PyTorch exposes ROCm through the CUDA device namespace) |

The orchestrator does not install or guess accelerator wheels. A wrong CUDA or
ROCm wheel is worse than a clean failure because it invalidates timing and
hardware claims.

### Bootstrap

CPU:

```bash
bash scripts/bootstrap_local_lab.sh --profile cpu
```

CUDA or ROCm, after choosing the official wheel index matching the local
runtime:

```bash
bash scripts/bootstrap_local_lab.sh \
  --profile cuda \
  --torch-index-url <official-pytorch-index-url>
```

```bash
bash scripts/bootstrap_local_lab.sh \
  --profile rocm \
  --torch-index-url <official-pytorch-index-url>
```

The bootstrap script creates `venv`, installs QUARTZ and development
dependencies, builds the Rust engine, runs the Rust tests unless explicitly
skipped, and finishes with a strict environment check.

Existing platform-specific details remain in `docs/INSTALL.md`.

---

## 2. Environment diagnosis

```bash
venv/bin/python scripts/idea_lab.py doctor --profile cpu
venv/bin/python scripts/idea_lab.py doctor --profile cuda --strict
venv/bin/python scripts/idea_lab.py doctor --profile rocm --strict
```

`--strict` additionally requires:

- `target/release/mcts_demo`;
- `pytest`;
- a working PyTorch installation for the selected profile.

The JSON form is suitable for bug reports:

```bash
venv/bin/python scripts/idea_lab.py doctor --profile cuda --json \
  > local_doctor.json
```

A profile-specific doctor failure should be fixed before a timing or live-engine
experiment. Dependency-light synthetic labs may still be run individually if
their own plan reports `READY`.

The doctor imports PyTorch in the interpreter passed through `--python`; it does
not use the orchestrator process as a proxy for another environment. CUDA
requires `torch.version.cuda`, a visible NVIDIA device, and matching
driver/runtime evidence. ROCm requires `torch.version.hip`, a visible AMD
device, and ROCm runtime evidence. PyTorch exposing both backends as the `cuda`
device namespace is not enough to treat their hardware proofs as interchangeable.

---

## 3. Discovering experiments

List every lane:

```bash
venv/bin/python scripts/idea_lab.py list
```

Useful filters:

```bash
venv/bin/python scripts/idea_lab.py list --status available
venv/bin/python scripts/idea_lab.py list --status planned
venv/bin/python scripts/idea_lab.py list --group parallel
venv/bin/python scripts/idea_lab.py list --group representation
```

Resolve commands and blockers without executing anything:

```bash
venv/bin/python scripts/idea_lab.py plan --suite smoke --profile cpu
venv/bin/python scripts/idea_lab.py plan --suite all-available --profile cuda
venv/bin/python scripts/idea_lab.py plan --suite foundry-roadmap
```

Pass `--config configs/idea_lab.local.v2.json` to list or plan the 26-axis
campaign. `first-gate-all` is the v2 default; `live-promotion-blocked` and
`accelerator-promotion-blocked` expose the prerequisites that intentionally
prevent premature execution.

`plan` exits non-zero when one or more selected lanes are blocked. This is
intentional and makes missing checkpoints, trace artifacts, dependencies, and
unimplemented modules visible before a long run starts.

---

## 4. Recommended execution order

### Stage A — dependency-light mechanism smoke

```bash
venv/bin/python scripts/idea_lab.py run \
  --suite smoke \
  --profile cpu
```

This runs:

- the orchestrator unit tests;
- Bernoulli root ranking-risk quick screen;
- candidate omission/ranking morphology quick screen;
- symmetry-orbit quick audit;
- pending-flow quick screen.

These are mechanism checks, not neural-MCTS efficacy results.

### Stage B — complete synthetic suite

```bash
venv/bin/python scripts/idea_lab.py run \
  --suite synthetic \
  --profile cpu \
  --run-id synthetic-baseline
```

Keep this run as the local baseline before modifying any shared primitive.

### Stage C — Rust and systems bridge

Quick Rust policy and evaluator service-curve checks:

```bash
venv/bin/python scripts/idea_lab.py run \
  --suite systems-smoke \
  --profile cpu
```

Real virtual-loss bridge plus canonical end-to-end smoke:

```bash
venv/bin/python scripts/idea_lab.py run \
  --suite engine-bridge \
  --profile cpu \
  --keep-going
```

The service-curve quick grid is deliberately tiny. It verifies the measurement
path; it is not sufficient for a throughput claim. Run the full
`scripts/service_curve_lab.py` configuration separately before tuning a
scheduler.

### Stage D — Stage-7 artifact replay

Artifact analyzers are CPU-safe, but they need existing results.

H1 versus `P_flip`:

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane stage7.flip_calibration \
  --profile cpu \
  --set trace_dir=results/phase15_stage7/trace_cache
```

H3/O6:

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane stage7.o6_precision \
  --profile cpu \
  --set trace_dir=results/phase15_stage7/trace_cache \
  --set online_rows=results/phase15_stage7/online/phase15_online_rows.jsonl
```

B13 replay:

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane stage7.b13_analysis \
  --profile cpu \
  --set posthoc_rows=results/phase15_stage7/posthoc/phase15_rows.jsonl
```

The orchestrator uses reduced bootstrap counts for local replay. Claim-bearing
reruns should call the original analyzer with its preregistered settings and the
full research-grade gate.

### Stage E — live trained-checkpoint smoke

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane stage7.kg_stop_live \
  --profile cuda \
  --set checkpoint=models/seed_101/latest.pt
```

The registered command is a small connectivity check, not a reopening of the
closed low-budget KG-stop claim. The prior Stage-7 result remains the relevant
baseline until a genuinely different regime or mechanism is tested.

---

## 5. Output contract

Each v1 invocation creates:

```text
results/idea_lab_local/<run-id>/
├── lab_manifest.json
├── commands.jsonl
├── summary.json
└── <lane-id>/
    ├── <step>.log
    └── experiment-owned artifacts
```

`lab_manifest.json` records:

- registry path and SHA-256;
- git branch, commit, and dirty state when available;
- selected profile and lanes;
- a local doctor snapshot;
- non-secret template variables.

`commands.jsonl` records start/finish events, exact argument vectors, elapsed
time, return codes, and log paths.

`summary.json` is updated after every lane, so an interrupted campaign retains a
usable partial result.

The repository already ignores `results/` and model artifacts. Do not place
checkpoints or large trace caches under tracked paths.

The v2 campaign additionally maintains `campaign_state.json` atomically after
every step and writes `campaign_summary.json` across all lanes. Each lane owns
`run_manifest.json`, `rows.jsonl`, `summary.json`, streaming stdout/stderr logs,
and recorded artifact hashes. Exit code zero is necessary but insufficient:
every required artifact must exist and pass its declared JSON/JSONL schema
before the lane can become `succeeded` or `completed_no_promotion`.

---

## 6. Running one lane or building a custom campaign

One lane:

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane synthetic.candidate_morphology.quick
```

Multiple lanes:

```bash
venv/bin/python scripts/idea_lab.py run \
  --lane synthetic.bernoulli_root.quick \
  --lane engine.rust.policy_smoke \
  --keep-going
```

Dry run:

```bash
venv/bin/python scripts/idea_lab.py run \
  --suite all-available \
  --profile cuda \
  --dry-run
```

Use `--run-id` for a stable directory name. The v1 compatibility path may use
`--overwrite`; the v2 campaign refuses to reuse a non-empty run directory and
requires the verified `resume` command instead.

---

## 7. Registered foundry roadmap

The v1 registry contains planned lanes for the original experimental program.
The v2 registry assigns every axis a runnable deterministic contract preflight
and keeps actual trace/live/training/hardware promotion roles explicitly
blocked until an input-consuming command exists:

- calibrated stop council;
- static-anchor regularized policy improvement;
- residual-partition-mass widening;
- forked counterfactual meta-action teacher;
- JSD-preconditioned locally balanced root sampling;
- dynamic live-set particle search;
- high-thread semantic path-overlap control;
- coherence-gated signed path shadow features;
- diffusion-regularized deterministic evaluator;
- RW-ResT Lite evaluator;
- physics-analogy falsification dashboard;
- regret/instability state archive.

A blocked lane is not a placeholder success. Its `blocked_by` list is the
minimum implementation and validation contract required before changing its
status to `available`.

When implementing a lane:

1. add the module and focused unit tests;
2. add a dependency-light or trace-replay smoke command;
3. define artifact names and failure behavior;
4. register the command and requirements;
5. add it to a suite only after the lane can run from a clean checkout;
6. keep live-engine efficacy separate from mechanism and wiring status.

---

## 8. Reproducibility and safety rules

- Pin explicit checkpoint paths. Do not select weak/mid/strong checkpoints by
  lexical truncation.
- Use paired positions, seeds, budgets, and runtime contracts for comparisons.
- Keep fixed-budget and adaptive-budget conclusions separate.
- Never pool restart-per-chunk and true root-continuation traces.
- Record actual NN evaluations, realized root visits, wall-clock, batch shape,
  queue wait, and hardware profile.
- Treat CPU, CUDA, and ROCm service curves as distinct artifacts.
- Preserve non-improvement and failure rows.
- Synthetic screens authorize or reject a mechanism experiment; they do not
  establish play strength.
- A negative result is scoped to `(module, role, game, budget, evaluator,
  hardware)`, not silently generalized to the whole idea.

For promotion above local screening, apply the existing research-grade gates in
`docs/CLAIM_LEDGER.md`, `docs/RESEARCH_READINESS.md`, and the Phase-15 analyzer.

---

## 9. Known limitations of this first scaffold

- It does not download checkpoints or external datasets.
- It does not guess CUDA/ROCm PyTorch wheels.
- It does not yet fork resident MCTS snapshots for counterfactual meta-actions.
- It does not synthesize Phase-15 trace bundles when none exist.
- Planned representation and particle-search models remain roadmap entries until
  their own code and tests land.
- The connector-created scaffold cannot substitute for running the commands on
  the target local hardware; timing and accelerator validation must happen there.

Those limitations are deliberate. The first goal is a truthful, repeatable
local execution surface on top of the current repository, not a single command
that claims to validate experiments whose prerequisites do not exist.
