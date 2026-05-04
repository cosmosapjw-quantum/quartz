# Research Readiness Criteria

This document is the internal checklist for upgrading QUARTZ from an
exploratory AlphaZero-style research platform toward claim-bearing experiment
artifacts. The ablation runner records a passive `research_readiness` section in
`ablation_report.json`; runs continue even when criteria are missing. For
claim-bearing artifacts, pass `--research-grade` in report mode to make the same
criteria a command-line gate.

## Current Policy

Treat a result as **engineering evidence** until all required criteria below are
met by the artifact itself.

| Criterion | Required evidence | Why it matters |
| --- | --- | --- |
| Multi-seed per condition | At least three explicit seeds per condition | Single-seed curves are not publication-grade evidence. |
| Paired seed protocol | `seed_protocol_summary` shows compared conditions share at least three common seeds and that paired-seed eval rows were requested and realized | Equal seed counts are weaker than paired/common seed sets for controller attribution. |
| Evaluation matrix | `evaluation_matrix.json` with scored post-train matches | Loss-only selection does not validate playing strength. |
| Confidence intervals | Every eval row has a score-rate CI | Point estimates hide variance. |
| No stale eval cache | `discarded_matches` is empty | Reused rows with old search manifests break reproducibility. |
| Benchmark-safe eval path | Every eval condition records `benchmark_safe=true` in expected runtime contract | Fallback/serial paths must not be mixed into claims. |
| Eval seed contract | `expected_eval_seeds` is present and each expected manifest contains `eval_seed` | Eval RNG must participate in cache invalidation. |
| Evaluation protocol summary | `evaluation_protocol_summary.protocol_ready=true` with runtime contract hash, runner mode, eval seed coverage, per-condition search manifest hash, consistent game counts, and complete model-pair coverage across eval conditions | Same-evaluator, same-NN, and same-game-distribution claims need explicit protocol evidence. |
| Evaluator quality strata | `evaluator_quality_summary.stratification_ready=true` with quality proxy coverage for every evaluated model pair | Claims about robustness to weak/strong evaluators need the evaluated models' quality metadata, not just aggregate win rates. |
| Held-out evaluator calibration | `heldout_calibration_summary.calibration_ready=true` with `evaluator_calibration.json` covering every evaluated model | Train-log proxies are not enough to interpret evaluator-quality robustness. |
| Controller selection trace | Eval rows include nonzero `realized_budget_trace.selection_trace.root_selects` and `selection_trace_coverage_frac` for controller-claim runs | Root-snapshot metrics alone do not prove what changed selection. |
| Budget trace | Eval rows roll up into `budget_fairness_summary` with realized root visits and halt reasons | Controller effects cannot be interpreted if modes silently receive different realized search budgets. |
| Pipeline telemetry | Train rows roll up into `pipeline_telemetry_summary` with replay freshness, throughput, and concurrent-worker latency proxy telemetry when applicable | Pipeline claims need evidence that replay and self-play were actually flowing under the claimed runtime. |
| Hardware claim scope | `hardware_runtime_summary` records backend/device requests, observed runtime telemetry, and `profiler_artifact_present` | Backend names and throughput rows are not hardware profiling evidence. |
| Deployment source explicit | `champion.json` records `deployment_cfg_source` | Exported bundles must state whether train config or an eval condition supplied search settings. |

## Passive Artifact Field

`scripts/ablation_study.py --report ...` writes:

```json
{
  "research_readiness": {
    "schema_version": 1,
    "research_grade_ready": false,
    "unmet_criteria": ["multi_seed_per_condition"],
    "blocking": false
  }
}
```

`blocking=false` is the default reporting behavior. In `--research-grade` mode,
the CLI exits nonzero when `research_grade_ready=false`; use that mode for
external claims and release/checkpoint qualification.

## Upgrade Order

1. Keep manifest/cache contracts complete: search options, halt mode, eval seed,
   Rust binary identity, backend/device, and benchmark-safe path.
2. Keep controller telemetry tied to actual actuator use: selection trace,
   actuator coverage, halt reasons, realized budget, and prior-refresh source.
3. Make repeated-seed protocols the default documented path for claim-bearing
   studies, while keeping short smoke runs explicitly exploratory.
4. Promote replay freshness, queue latency, and throughput summaries into
   report-level artifacts.
5. Only after the report can satisfy the checklist without manual inspection,
   use the artifact for external research claims.

## Current Known Gaps

- Existing legacy reports without `eval_seed` in their search manifest should be
  regenerated.
- Reports with one seed per condition remain engineering signals.
- Reports whose compared conditions use different seed sets should not make
  paired-seed or low-variance controller attribution claims.
- Reports missing `evaluation_protocol_summary.protocol_ready=true` should not
  make same-evaluator, same-NN, or same-game-distribution claims. In particular,
  every eval row should expose pair IDs, runner mode, search manifest hash, and
  consistent games/eval seed/runtime contract metadata.
- Reports missing `evaluator_quality_summary.stratification_ready=true` should
  not make weak-evaluator, strong-evaluator, noisy-evaluator, or evaluator
  robustness claims. Current in-repo strata use train-log proxies (`loss`,
  `p_loss`, `v_loss`, `loss_ema`, published Elo, and score rate); stronger
  claims require a held-out calibration artifact.
- Reports missing `heldout_calibration_summary.calibration_ready=true` should
  not make evaluator-quality robustness claims. The expected artifact is
  `evaluator_calibration.json` with `n_positions`, `policy_nll`, `value_mse`,
  `top1_acc`, and `brier` for every evaluated model id.
- Reports whose `realized_budget_trace.selection_trace` is absent, uncovered, or
  zero should not make controller mechanism claims.
- Reports missing `budget_fairness_summary` should not make same-budget
  controller comparison claims; a `budget_fairness_flag=drift` result can still
  be useful, but must be interpreted as compute-changing.
- Reports missing `pipeline_telemetry_summary` should not make throughput,
  queue-latency-proxy, or replay-freshness claims.
- Reports whose `hardware_runtime_summary.claim_scope` is
  `runtime_telemetry_only` may report observed throughput, but must not claim
  GPU/ROCm/CUDA optimization or hardware efficiency.
- GPU/ROCm/CUDA hardware-performance claims require
  `profiler_artifact_present=true`; backend/device names alone are not enough.
