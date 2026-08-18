# QUARTZ Idea Foundry: Formal Scientific Audit Report (26 Axes & 2x2 Factorial Sweep)

**Campaign Run ID**: `first-scientific-gates-full-20260817` (26-Axis Full Diagnostic Campaign)  
**Factorial Run ID**: `factorial-confirmatory-frontier-r2-20260818` (Metacontroller 2x2 Confirmatory Factorial, N=3 paired seeds, 960 matches)  
**Audit Date**: 2026-08-18  
**Repository Branch**: `agent/local-experiment-foundry`  
**Hardware Platform**:
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, Driver 595.84, CUDA 13.2 / PyTorch cu128)
- CPU: AMD Ryzen 9 5900X 12-Core Processor (24 threads, 64GB DDR4 RAM)
**Software Stack**: Python 3.12.3 (venv), PyTorch 2.11.0+cu128, Rust 1.84+ (`idea-foundry` feature gated)  
**Claim Scope**: `first_scientific_gate_diagnostic_only` / `analysis_only` / `paired_factorial_resource_frontier_analysis_only`

---

## 1. Executive Summary & Audit Judgment

This formal audit report records the end-to-end execution, diagnostic gate enforcement, within-axis meta-analysis, and $2\times 2$ factorial training/runtime sweep for the QUARTZ Idea Foundry.

#### Critical Status Disambiguation
Under the five-status contract framework (`schema_version: 1`), all results distinguish five independent status dimensions:
$$\text{execution\_status} \neq \text{contract\_status} \neq \text{effect\_status} \neq \text{evidence\_maturity} \neq \text{promotion\_status}$$

1. **26-Axis Diagnostic Study (`first-scientific-gates-full-20260817`)**:
   - **Execution & Contract**: 25 axes completed cleanly (`execution_status: success`, `legacy_status: completed_no_promotion`, `contract_status: passed`), 1 axis skipped (`A10: dormant`, `contract_status: not_applicable`), 0 technical failures.
   - **Scientific Scope**: Diagnostic and mechanism-contract checks only. Code and preregistered configurations explicitly prohibit inferring play-strength, Elo, production readiness, or cross-axis superiority from these first-gate outputs (`promotion.eligible = false`).
2. **Metacontroller $2\times 2$ Confirmatory Factorial (`factorial-confirmatory-frontier-r2-20260818`)**:
   - **Execution & Scope**: 960 arena matches executed cleanly across 8 held-out opening families (2 replicates per family, 16 total opening groups) with $N_{\rm paired}=3$ seeds (41, 42, 43) under `resource_frontier_confirmatory` with 100% selection trace coverage.
   - **Realized Compute Reduction**: Runtime A01 executed 439 early halts per active arm, delivering a paired compute contrast $\Delta C = -16.18$ NN evals/move (**$-27.78\%$ search compute reduction**, $42.06$ vs $58.24$, `compute_reduction_passed: true` across both arms).
   - **Quality Non-Inferiority**: Main runtime effect point estimate $\Delta Q = -0.03125$ ($0.46875$ vs $0.50000$). Seed-level aggregate CI is $[-0.03125, -0.03125] > -0.05$ (`seed_conditioned_quality_ni_passed: true`), while hierarchical 48-cluster sensitivity CI across opening groups is $[-0.1279, +0.0654]$ (`heldout_generalization_quality_ni_passed: false`, `confirmatory_quality_ni_passed: false`).
   - **Interaction Equivalence**: Factored interaction $\Delta_{TR} = 0.0000$ (seed-level CI $[0.0000, 0.0000] \subset [-0.05, 0.05]$ and hierarchical cluster estimate $0.0000$), satisfying the 95% CI containment equivalence criterion with zero observed antagonism (`interaction_equivalence_passed: true`).

---

## 2. 26-Axis Ablation Meta-Analysis Table

| Axis | Axis Name | Estimand | k | Fixed Effect [95% CI] | Random Effect [95% CI] | Uncertainty Kind | Evidence Domain | Maturity | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A01** | Calibrated Stop Council | `early_stop_compute_fraction` | 3 | $+0.0129$ $[+0.0016, +0.0241]$ | $+0.0129$ $[+0.0016, +0.0241]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A02** | Static-Anchor RPO | `oracle_logloss_reduction` | 3 | $+0.1957$ $[+0.0228, +0.3687]$ | $+0.1957$ $[+0.0228, +0.3687]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A03** | Uncertainty Decomposition | `conservative_radius_coverage_gain` | 3 | $+3.20\times 10^{-11}$ $[\pm 1.39\times 10^{-6}]$ | $+3.20\times 10^{-11}$ $[\pm 1.39\times 10^{-6}]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A04** | KG/VOC Allocator | `uncertainty_allocation_correction_gain` | 3 | Validated | Validated | Model Based | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A05** | Counterfactual Meta Teacher | `meta_action_regret_red_per_cost` | 3 | $+1.96\times 10^{-5}$ $[-4.28\times 10^{-5}, +8.19\times 10^{-5}]$ | $+1.96\times 10^{-5}$ $[-4.28\times 10^{-5}, +8.19\times 10^{-5}]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A06** | Gumbel + Sequential Halving | `simple_regret_reduction` | 3 | $-0.0005$ $[-0.0065, +0.0054]$ | $-0.0005$ $[-0.0065, +0.0054]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A07** | Residual-Evidence Widening | `omission_regret_reduction` | 3 | $0.0000$ $[\pm 1.13\times 10^{-6}]$ | $0.0000$ $[\pm 1.13\times 10^{-6}]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A08** | Tactical Proof Backend | `forced_action_recall_gain` | 3 | $+1.0000$ (Exact) | $+1.0000$ (Exact) | Exact | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A09** | H3 Change-Point Router | `change_router_brier_gain` | 3 | $+0.2621$ $[+0.2175, +0.3067]$ | $+0.2621$ $[+0.2093, +0.3149]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A10** | Prior-Refresh Specialist | `ood_recovery_gain` | — | Skipped (dormant) | Skipped (dormant) | Unavailable | `shadow_trace` | Dormant | Skipped |
| **A11** | Dynamic Live-Set Particles | `multimodal_est_error_reduction` | 3 | $+0.0497$ $[+0.0466, +0.0527]$ | $+0.0497$ $[+0.0466, +0.0527]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A12** | JSD Balanced Sampler | `target_total_variation_reduction` | 3 | $+0.0711$ $[+0.0660, +0.0762]$ | $+0.0711$ $[+0.0655, +0.0767]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A13** | Pending-Flow / WU-UCT | `duplicate_dispatch_rate_reduction`| 3 | $+0.3807$ $[+0.3768, +0.3847]$ | $+0.3807$ $[+0.3768, +0.3847]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A14** | Semantic Path LSH | `near_dup_detection_bal_acc_gain` | 3 | $+0.2495$ $[+0.1817, +0.3172]$ | $+0.2495$ $[+0.1817, +0.3172]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A15** | Service-Curve Scheduler | `cuda_throughput_ratio` | 6 | $+4.9171$ $[+4.9088, +4.9254]$ | $+4.9171$ $[+4.1166, +5.7175]$ | Sampling | `systems_benchmark` | Diagnostic | Pass (Contract) |
| **A16** | Graph / State Sharing | `evaluator_call_reduction` | 3 | $+14.27$ $[+13.92, +14.62]$ calls | $+14.27$ $[+13.92, +14.62]$ calls | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A17** | B13 Curvature Readout | `oracle_kl_reduction` | 6 | $+0.0297$ $[+0.0270, +0.0324]$ | $+0.0297$ $[+0.0270, +0.0324]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A18** | Diffusion Evaluator | `heldout_joint_loss_reduction` | 3 | Validated | Validated | Sampling | `paired_training` | Diagnostic | Pass (Contract) |
| **A19** | RW-ResT Lite Evaluator | `weighted_proxy_loss_delta` | 3 | $+0.0078$ $[-0.0124, +0.0279]$ | $+0.0078$ $[-0.0124, +0.0279]$ | Sampling | `paired_training` | Diagnostic | Pass (Contract) |
| **A20** | Regret State Archive | `future_error_capture_lift` | 3 | $+0.3094$ $[+0.2824, +0.3364]$ | $+0.3091$ $[+0.2706, +0.3475]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A21** | Signed-Path Coherence | `coherence_brier_gain` | 3 | $+0.0385$ $[+0.0153, +0.0618]$ | $+0.0385$ $[+0.0153, +0.0618]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A22** | Physics Falsification | `surrogate_null_r2_gain` | 3 | Validated | Validated | Model Based | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A23** | Incremental CPU Pattern | `incremental_feature_exact_match` | 3 | $+1.0000$ (Exact) | $+1.0000$ (Exact) | Exact | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A24** | Learned Budget Gate | `budget_utility_gain` | 3 | $+0.0234$ $[+0.0048, +0.0420]$ | $+0.0234$ $[+0.0048, +0.0420]$ | Sampling | `shadow_trace` | Diagnostic | Pass (Contract) |
| **A25** | MENTS Soft Backup | `finite_temperature_value_shift` | 3 | $-0.0731$ $[-0.0774, -0.0687]$ | $-0.0731$ $[-0.0774, -0.0687]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A26** | Exact Nested-Contour Lab | `contour_enum_absolute_error` | 3 | $5.17\times 10^{-17}$ (Exact) | $5.17\times 10^{-17}$ (Exact) | Exact | `synthetic_gate` | Diagnostic | Pass (Contract) |

---

## 3. Metacontroller $2\times 2$ Confirmatory Factorial Results ($N_{\rm paired}=3$, 960 Games)

### Cell Allocation & Observed Compute Delivery

| Cell | Training Arm | Runtime Arm | Score Rate vs Anchor | Realized NN Evals / Move | Early Halts / Active Cell |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M00** | OFF (`shadow_noop`) | OFF (`shadow_noop`) | $0.50000$ | $58.24$ | 0 |
| **M01** | OFF (`shadow_noop`) | ON (`active`) | $0.46875$ | **$42.06$ ($-27.78\%$)** | 439 |
| **M10** | ON (`active`) | OFF (`shadow_noop`) | $0.50000$ | $58.24$ | 0 |
| **M11** | ON (`active`) | ON (`active`) | $0.46875$ | **$42.06$ ($-27.78\%$)** | 439 |

### Statistical Evaluation of Factorial Contrasts

| Contrast | Estimand $\hat{\theta}$ | Seed-Level 95% CI ($N=3$) | Hierarchical 48-Cluster 95% CI | Margin $\delta$ | Criterion | Empirical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Compute Reduction ($\Delta C$)** | $-27.78\%$ ($-16.18$ evals) | — | — | $\Delta C < 0$ | $C_{01}<C_{00} \land C_{11}<C_{10}$ | **PASS** ($-16.18$ evals/move) |
| **Quality Non-Inferiority ($\Delta Q$)** | $-0.03125$ ($0.46875$ vs $0.50000$) | $[-0.03125, -0.03125]$ | $[-0.12786, +0.06536]$ | $\delta_Q = 0.05$ | One-sided 95% LB $> -\delta_Q$ | **Seed-Conditioned PASS / Generalization NOT ESTABLISHED** |
| **Interaction Equivalence ($\Delta_{TR}$)** | $0.00000$ | $[0.00000, 0.00000]$ | $[0.00000, 0.00000]$ | $\delta_I = 0.05$ | 95% CI Containment in $[-\delta_I, \delta_I]$ | **PASS** (Zero Antagonism) |
| **Training Main Effect ($\Delta_T$)** | $0.00000$ | $[0.00000, 0.00000]$ | $[0.00000, 0.00000]$ | — | — | Neutral |

- **Selection Trace Coverage**: $100\%$ ($1.000000$) across all 960 games under canonical `A01.live_pflip_v1`.
- **Durable Artifacts**: Normalized evidence serialized in [20260818_confirmatory_factorial_evidence.json](evidence/20260818_confirmatory_factorial_evidence.json).
- **Promotion Status**: `COMPLETED_NO_PROMOTION` (first scientific confirmation on Gomoku7; multi-game generalization and larger scale training scheduled under Phase 16).

---

## 4. Architectural Analysis & Governance Actions

### 1. A01 Council Mechanism & Empirical Continuation Calibration (Stage 7 / C8)
- **Empirical Calibration Findings** (288 real MCTS trace bundles, 864 decision records stratified by trace source):
  - **Dirichlet Posterior Stability ($s_{\rm H1}$)**: $\text{Brier} = \mathbf{0.1424}$, $\text{ECE} = \mathbf{0.0755}$ ($7.55\%$). Realized agreement closely matches predicted confidence across all 10 bins.
  - **Gaussian Top-Two Incumbent ($s_{P_{\rm flip}}$)**: $\text{Brier} = \mathbf{0.4464}$ ($3.13\times$ higher error), $\text{ECE} = \mathbf{0.5040}$ ($50.4\%$). Suffers from severe small-budget overconfidence ($99.8\%$ predicted confidence yielded only $22.3\%$ realized agreement).
- **Governance Resolution**: Confirmed that Dirichlet stability $H_1$ substantially alleviates the observed small-budget overconfidence pathology on the Stage-7 trace bank. Live search adapter is sealed under canonical variant ID `A01.live_pflip_v1` while $H_1$ continuation calibration is preregistered for Phase 16 multi-game search engine integration.

### 2. A13 (WU-UCT) & A16 (Graph State Cache) — Immediate Merges Blocked & Protocols Established
- **A13 (Pending-Flow / WU-UCT)**: Tested against the true production incumbent (**Adaptive Virtual Loss**, `VlMode::Adaptive`). Live engine merge is **STRICTLY BLOCKED** until differential testing proves net search throughput gain ($\Delta Q_{\rm throughput} > +0.05$) under real GPU batching on Gomoku15, Chess, and Go without thread contention. See [A13 Real-Engine Differential Protocol](protocols/A13_REAL_ENGINE_DIFFERENTIAL_PROTOCOL.md).
- **A16 (Graph State Sharing Cache)**: Proposes a decoupled Tier-2 pure state evaluation cache using composite key $K = (\text{state\_zobrist}, \text{evaluator\_checkpoint\_id}, \text{ruleset}, \text{input\_version}, \text{cache\_schema})$. Live merge is **STRICTLY BLOCKED** until the two-tier cache architecture is implemented in Rust and verified to reduce NN forward passes ($\Delta E_{\rm eval} < -0.10$) without memory bloat or lock contention. See [A16 Graph State Cache Protocol](protocols/A16_GRAPH_STATE_CACHE_PROTOCOL.md).

### 3. A15 (Service Curve) & A19 (Semantic Alignment) Protocols
- **A15 (Dynamic GPU Queue Scheduler)**: Corrected estimand from raw CUDA vs CPU ratio to a Pareto vector objective $V_{\rm obj} = (\text{QPS}_{p99 \le \tau}, p99, \mathbb{E}[W_q], \text{padding\_fraction}, \text{energy/query})$ under fixed GPU hardware. See [A15 Service Curve Estimand Protocol](protocols/A15_SERVICE_CURVE_ESTIMAND_PROTOCOL.md).
- **A19 (RW-ResT Lite Evaluator)**: Formally separated `A19.proxy_screen_v1` ($0.65\cdot L_{\rm policy\_KL} + 0.35\cdot L_{\rm value\_MSE}$, implemented adapter, existing $+0.0078$ screen result) from `A19.confirmatory_weighted_loss_v2` ($1.0\cdot L_{\rm policy\_CE} + 1.5\cdot L_{\rm value\_MSE}$, future confirmatory protocol). See [A19 Weighted Proxy Loss Protocol](protocols/A19_WEIGHTED_PROXY_LOSS_PROTOCOL.md).

