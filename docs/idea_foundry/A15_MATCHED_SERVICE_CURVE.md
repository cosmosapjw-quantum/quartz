# A15 matched CPU/CUDA service-curve readiness

Status: **PREPARATION / DIAGNOSTIC ONLY**. This path does not promote an A15
scheduler, an efficiency claim, or a production default.

## Resolved mismatch

The earlier `service_curve_lab` artifact is a real CUDA-only curve. It does not
form a matched CPU/CUDA ablation because the CPU backend was not measured under
the same model state, input bytes, grid cells, and repeated timing contract.

`scripts/a15_matched_service_curve.py` now reuses the existing representative
evaluator builder and enforces:

- one fixed model seed and one `model_state_sha256` copied to both devices;
- byte-identical CPU-created tensors within every CPU/CUDA repetition pair;
- a `workload_identity_sha256` binding builder source, model state, profile,
  seeds, dtype, and pinned CPU thread contract;
- pre-timing CPU/CUDA policy/value parity, including maximum absolute error and
  policy argmax agreement, with fail-closed tolerances;
- alternating backend order by cell and repetition;
- pinned PyTorch CPU intra-op/inter-op threads (`1/1` by default);
- deterministic-algorithm and cuDNN deterministic flags with TF32/benchmark
  disabled and recorded;
- immutable raw rows plus exact paired rows, a summary, hashes, and a generated
  plot labeled `DIAGNOSTIC`.

The script builds in a sibling attempt directory and publishes the completed
result by atomic rename. An exact retry after publication verifies the profile,
source, input, and artifact hashes and returns idempotently; partial or drifted
final directories fail closed.

CPU `inflight` is serial work within a wave. CUDA `inflight` uses one stream per
outstanding batch and synchronizes at the wave boundary. This difference is
part of the measured service contract, not hidden as if both backends used the
same concurrency mechanism. Input generation and host-to-device copies are not
timed.

## Readiness smoke

Use a new output directory for every run:

```bash
venv/bin/python scripts/a15_matched_service_curve.py \
  --profile diagnostic \
  --output-dir results/idea_foundry/a15_matched_service_curve_<run-id>
```

The diagnostic profile contains four cells (`batch={8,64}` ×
`inflight={1,2}`) and two matched repetitions. It tests the measurement and
artifact contract; it is not a substitute for the full matrix.

## Full matrix

The separate full profile preserves the previous 24-cell grid and measures five
repetitions per backend:

```bash
venv/bin/python scripts/a15_matched_service_curve.py \
  --profile full \
  --output-dir results/idea_foundry/a15_matched_service_curve_full_<run-id>
```

The CPU half can be long-running. Do not shorten it post hoc and compare it to a
different CUDA contract; change the versioned config and rerun both backends if
the preregistered workload needs revision.

## Artifacts

- `run_manifest.json`: source/input hashes, fixed seed contract, device proof,
  runtime contract, semantic parity, and workload identity;
- `rows.jsonl`: every raw timing repetition;
- `service_curve_rows.v1.csv`: the same versioned raw data in tabular form;
- `paired_backend_rows.v1.csv`: exact CPU/CUDA joins with a descriptive ratio;
- `summary.json`: completeness/readiness only, with promotion disabled;
- `diagnostic.png` and `plot_metadata.v1.json`: generated DIAGNOSTIC figure and
  its data provenance.

## What the plot does not show

The plot does not measure search decision quality, play strength, controlled
energy, the shipped network, queue wait in the live evaluator, or a production
scheduler gain. A future claim-bearing scheduler study must use the actual
checkpoint/workload and add matched search-quality-per-wall-clock evidence.
