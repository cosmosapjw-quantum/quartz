# Local Experiment Lab

This document is the execution runbook for the QUARTZ idea foundry. It is
intentionally broader than the current production controller: existing
mechanism labs are executable now, while proposed experiments are registered
with explicit blockers rather than being silently omitted.

The local-lab contract is:

1. **Every run gets an isolated output directory.**
2. **Every executed command is logged.**
3. **The environment and registry hash are recorded before execution.**
4. **Planned lanes are visible but never pretended to be executable.**
5. **A failed role/regime does not delete the parent idea.**

The orchestrator is `scripts/idea_lab.py`; the registry is
`configs/idea_lab.local.v1.json`.

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

Each invocation creates:

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

Use `--run-id` for a stable directory name. The orchestrator refuses to reuse a
non-empty run directory unless `--overwrite` is given.

---

## 7. Registered foundry roadmap

The registry contains planned lanes for the full experimental program:

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

A planned lane is not a placeholder success. Its `blocked_by` list is the
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
