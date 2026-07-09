# QUARTZ Claim Ledger

This ledger is the claim firewall for QUARTZ. It records what is implemented,
what is smoke-tested, and what remains ablation-pending. Do not promote a claim
above its evidence tier without adding the cited artifact and test/benchmark
command.

## Status Vocabulary

- `SPECIFIED`: design exists, implementation is not complete.
- `PROPOSED`: research direction or planned extension.
- `IMPLEMENTED`: code exists and is integrated.
- `SMOKE-VALIDATED`: implementation passed a small functional smoke or unit
  check. This is not an algorithmic efficacy claim.
- `ABLATION-PENDING`: implemented or specified but not yet compared under the
  fixed ablation protocol.
- `VALIDATED`: benchmarked or ablated with fixed seeds, paired artifacts,
  runtime hashes, and reproducible commands.
- `DEPRECATED`: no longer controlling.
- `FORBIDDEN`: explicitly disallowed for the current research direction.

## Current Claims

| Claim | Status | Evidence | Risk | Required Fix |
|---|---|---|---|---|
| RTX 3080 Ti CUDA venv and runtime setup exists. | SMOKE-VALIDATED | `tests/test_cuda_runtime_setup.py`; CUDA model smoke | Smoke only, no throughput claim | Run explicit throughput/profile campaign before GPU efficiency claims |
| `torch_inference_runtime` centralizes Torch batch inference for runtime/facade paths. | IMPLEMENTED | `quartz/torch_inference_runtime.py`; targeted pytest | Runtime behavior could drift if callers bypass helper | Keep new Torch inference callers routed through this module |
| `search_manifest` is the canonical search/runtime manifest key authority. | IMPLEMENTED | `quartz/search_manifest.py`; runtime/training catalog tests | Rust-side semantics still own execution | Keep Python manifest as provenance schema, not execution proof |
| Phase15 `B1` dual-channel commit is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; unit tests | Same-trace readout only, not online efficacy | Run paired Phase15 ablation before quality claims |
| Phase15 `B2` root challenger is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; unit tests | Candidate coverage may not predict play quality | Run bucketed wrong-prior and do-no-harm assays |
| Phase15 `B3` budget routing is available as a budget-scheduler candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; `results/phase15_b11_deep_20260709_135106/phase15_b12_deep_analysis.json` | Latest rehearsal shows a larger apparent quality signal (`delta_accuracy_to_oracle=0.033333333333` vs `A4`) but also `budget_burst_rate=0.5515625` and `extra_budget_used_mean=21.208333333333`; same-budget readout claims would be invalid | Keep `budget_burst_triggered` and `extra_budget_used` in reports; compare in a scheduler lane with effective-budget-normalized baselines |
| Phase15 `B4` root dual posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; aligned small rehearsal | Low KL can coexist with poor top-1/tie-aware agreement | Add top-1/tie-aware safety gating before expanding |
| Phase15 `B5` root posterior snapshot is an A4-equivalent sanity anchor. | IMPLEMENTED | `quartz/phase15_ablation.py`; aligned small rehearsal equivalence | Misreading it as an independent candidate would inflate the candidate set | Collapse under A4 in semantic summaries; do not promote as champion |
| Phase15 `B6` confidence-bound posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; unit tests | Temporal volatility proxy may not match statistical uncertainty | Audit trace volatility vs reference/oracle buckets |
| Phase15 `B7` robust-valley posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; unit tests | Robustness proxy may favor diffuse policies | Track entropy, top-k recall, and candidate undercoverage |
| Phase15 `B8` entropy-annealed posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; unit tests | Temperature schedule is heuristic | Treat as exploration readout until ablated |
| Phase15 `B9` argmax/tie-guarded dual posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; `results/phase15_b9_aligned_analysis.json`; unit tests | Aligned rehearsal showed no accuracy gain vs `A4`/`B5`, worse KL/top-k, and high guard veto rate; treating it as a champion would be claim drift | Keep as diagnostic/safety-gate evidence; do not promote without paired improvement over `A4`/`B5` |
| Phase15 `B10` snapshot-safe trace-stabilized posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; `results/phase15_b10_aligned_analysis.json`; unit tests | Aligned rehearsal showed no top-1/top-k loss vs `A4`/`B5` and tiny KL reduction, but effect size is too small for a quality claim | Keep as a no-harm stabilizer signal; run a stronger adaptive variant before promotion claims |
| Phase15 `B11` adaptive snapshot-safe trace-stabilized posterior is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; `results/phase15_b11_deep_20260709_135106/phase15_b12_deep_analysis.json`; unit tests | Deep rehearsal showed no top-1/top-k gain and slightly higher KL vs `A4`/`B5`; broad adaptive smoothing is not a champion signal | Keep as a stress-test anchor; use its failure to motivate narrower oracle-free gates |
| Phase15 `B12` entropy-expansion-gated snapshot stabilizer is available as a posthoc candidate. | ABLATION-PENDING | `quartz/phase15_ablation.py`; `results/phase15_b11_deep_20260709_135106/phase15_b12_deep_analysis.json`; unit tests | Deep rehearsal showed unchanged top-1/top-k vs `A4`/`B5` and a tiny KL reduction (`delta_kl_to_oracle=-0.000790025449` vs `A4`), but the effect is analysis-only and too small for efficacy claims | Keep as the current narrow stabilizer candidate; run a larger paired ablation before promotion claims |
| Phase15 aligned result analysis is reproducible through a repo-local script. | IMPLEMENTED | `scripts/phase15_analyze_results.py`; `tests/test_phase15_ablation.py` | Analysis summaries can still be misread as validation, and explicit target lists can omit newly run telemetry candidates | Preserve `ANALYSIS-ONLY` claim status, paired-delta framing, and `analysis_coverage`/`--targets auto` checks |
| BQ++ `>=30% nn_evals_per_move` reduction at non-inferior quality is a target. | SPECIFIED | `docs/BQ_PLUS_PLUS_DESIGN.md`; README objective | Easy to misread as achieved performance | Keep phrasing as target until paired ablation/profile exists |
| Current Gomoku7 evidence points toward no-refresh legacy-family variants in exploratory rows. | SMOKE-VALIDATED | Prior repo-local sweep notes and README caveat | One-off/single-family evidence can overgeneralize | Re-run under fixed Phase15 protocol before default recommendation |
| `edge_sigma()`/`σ_Q` under parallel backup is an exact per-edge sample std-dev. | FORBIDDEN (revise to: approximation under contention) | `src/mcts/backup.rs` module doc (A3-b) | `(n_old, w_before)` is not read/updated as an atomic pair; concurrent backups on the same edge can pair a stale `n_old` with a `w_before` that already includes another thread's update, biasing the incremental Welford `delta_m2`. `σ_Q` feeds both `QuartzController` root penalty and `ParallelismController`'s adaptive split-VL magnitude, so the bias propagates to both | Full fix (CAS loop or per-edge lock over the {n,w,m2} triple) conflicts with the lock-free hot-path contract; not planned. Treat `σ_Q` as approximate under heavy contention when interpreting controller/VL behavior; do not cite it as an exact statistic in a research write-up |
| Game-specific strategy/rule/pattern injection is allowed in mainline QUARTZ. | FORBIDDEN | User research constraint; current game-agnostic design | Would undermine novelty and generality | Keep candidates based only on root/search statistics |
| Literal quantum/thermodynamic superiority claims are supported. | FORBIDDEN | Legacy docs are deprecated idea sources only | Physics metaphor drift | Translate legacy motifs into statistical search observables only |

## Ablation Start Conditions

Before promoting any candidate above `ABLATION-PENDING`, the run must record:

- fixed seed set and paired candidate comparison;
- identical checkpoint/bootstrap inputs;
- identical root visit budget unless the operator explicitly reports extra
  budget usage;
- identical hardware/runtime flags and runtime contract hash;
- identical promotion/evaluation criterion;
- artifact hashes for configs, checkpoints, position suite, and search manifest;
- separate `reference_policy` and `oracle_policy` fields;
- failure/non-improvement rows preserved in the output.
