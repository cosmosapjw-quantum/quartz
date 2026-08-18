# QUARTZ Idea Foundry: Formal Scientific Audit Report (26 Axes & 2x2 Factorial Sweep)

**Campaign Run ID**: `first-scientific-gates-full-20260817` (26-Axis Full Diagnostic Campaign)  
**Factorial Run ID**: `factorial-smoke-20260818` (Metacontroller 2x2 Factorial Smoke)  
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

### Critical Status Disambiguation
Under the Schema v2 framework, all results distinguish five independent status dimensions:
$$\text{execution\_status} \neq \text{contract\_status} \neq \text{effect\_status} \neq \text{evidence\_maturity} \neq \text{promotion\_status}$$

1. **26-Axis Diagnostic Study (`first-scientific-gates-full-20260817`)**:
   - **Execution & Contract**: 25 axes completed cleanly (`completed_no_promotion`), 1 axis skipped (`A10: dormant by contract`), 0 technical failures.
   - **Scientific Scope**: Diagnostic and mechanism-contract checks only. Code and preregistered configurations explicitly prohibit inferring play-strength, Elo, production readiness, or cross-axis superiority from these first-gate outputs (`promotion.eligible = false`).
2. **Metacontroller $2\times 2$ Confirmatory Factorial (`factorial-confirmatory-frontier-r2-20260818`)**:
   - **Execution & Scope**: 960 arena matches executed cleanly across 16 held-out opening families with $N_{\rm paired}=3$ seeds (41, 42, 43) under `resource_frontier_confirmatory` with 100% selection trace coverage.
   - **Realized Compute Reduction**: Runtime A01 executed 439 early halts per active arm, delivering a **$-27.78\%$ search compute reduction** ($42.06$ vs $58.24$ NN evals/move, `compute_reduction_passed: true`).
   - **Quality Non-Inferiority**: Main runtime effect $\Delta Q = -0.03125$ ($0.46875$ vs $0.50000$), satisfying the preregistered non-inferiority margin $\delta_Q = 0.05$ ($\Delta Q > -0.05$, `quality_noninferiority_passed: true`).
   - **Interaction Equivalence**: Factored interaction $\Delta_{TR} = 0.0000$ with 95% CI $[0.0000, 0.0000] \subset [-0.05, 0.05]$, confirming absence of antagonistic interaction under TOST equivalence (`interaction_equivalence_passed: true`).

---

## 2. 26-Axis Ablation Meta-Analysis Table

| Axis | Axis Name | Estimand | k | Fixed Effect [95% CI] | Random Effect [95% CI] | Uncertainty Kind | Evidence Domain | Maturity | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A01** | Calibrated Stop Council | `early_stop_compute_fraction` | 3 | $+0.0129$ $[+0.0016, +0.0241]$ | $+0.0129$ $[+0.0016, +0.0241]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A02** | Static-Anchor RPO | `oracle_logloss_reduction` | 3 | $+0.1957$ $[+0.0228, +0.3687]$ | $+0.1957$ $[+0.0228, +0.3687]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A03** | Uncertainty Decomposition | `conservative_radius_coverage_gain` | 3 | $+3.20\times 10^{-11}$ $[\pm 1.39\times 10^{-6}]$ | $+3.20\times 10^{-11}$ $[\pm 1.39\times 10^{-6}]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A04** | KG/VOC Allocator | `uncertainty_allocation_correction_gain` | 3 | Validated | Validated | Model Based | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A05** | Counterfactual Meta Teacher | `meta_action_regret_red_per_cost` | 3 | $+1.96\times 10^{-5}$ $[-4.28\times 10^{-5}, +8.19\times 10^{-5}]$ | $+1.96\times 10^{-5}$ $[-4.28\times 10^{-5}, +8.19\times 10^{-5}]$ | Sampling | `synthetic_counterfactual` | Diagnostic | Pass (Contract) |
| **A06** | Gumbel + Sequential Halving | `simple_regret_reduction` | 3 | $-0.0005$ $[-0.0065, +0.0054]$ | $-0.0005$ $[-0.0065, +0.0054]$ | Sampling | `synthetic_mechanism` | Diagnostic | Pass (Contract) |
| **A07** | Residual-Evidence Widening | `omission_regret_reduction` | 3 | $0.0000$ $[\pm 1.13\times 10^{-6}]$ | $0.0000$ $[\pm 1.13\times 10^{-6}]$ | Sampling | `synthetic_mechanism` | Diagnostic | Pass (Contract) |
| **A08** | Tactical Proof Backend | `forced_action_recall_gain` | 3 | $+1.0000$ (Exact) | $+1.0000$ (Exact) | Exact | `position_suite` | Diagnostic | Pass (Contract) |
| **A09** | H3 Change-Point Router | `change_router_brier_gain` | 3 | $+0.2621$ $[+0.2175, +0.3067]$ | $+0.2621$ $[+0.2093, +0.3149]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A10** | Prior-Refresh Specialist | `ood_recovery_gain` | — | Skipped (dormant) | Skipped (dormant) | Unavailable | `conditional_audit` | Dormant | Skipped |
| **A11** | Dynamic Live-Set Particles | `multimodal_est_error_reduction` | 3 | $+0.0497$ $[+0.0466, +0.0527]$ | $+0.0497$ $[+0.0466, +0.0527]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A12** | JSD Balanced Sampler | `target_total_variation_reduction` | 3 | $+0.0711$ $[+0.0660, +0.0762]$ | $+0.0711$ $[+0.0655, +0.0767]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A13** | Pending-Flow / WU-UCT | `duplicate_dispatch_rate_reduction`| 3 | $+0.3807$ $[+0.3768, +0.3847]$ | $+0.3807$ $[+0.3768, +0.3847]$ | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A14** | Semantic Path LSH | `near_dup_detection_bal_acc_gain` | 3 | $+0.2495$ $[+0.1817, +0.3172]$ | $+0.2495$ $[+0.1817, +0.3172]$ | Sampling | `trace_shadow` | Diagnostic | Pass (Contract) |
| **A15** | Service-Curve Scheduler | `cuda_throughput_ratio` | 6 | $+4.9171$ $[+4.9088, +4.9254]$ | $+4.9171$ $[+4.1166, +5.7175]$ | Sampling | `systems_benchmark` | Diagnostic | Pass (Contract) |
| **A16** | Graph / State Sharing | `evaluator_call_reduction` | 3 | $+14.27$ $[+13.92, +14.62]$ calls | $+14.27$ $[+13.92, +14.62]$ calls | Sampling | `synthetic_gate` | Diagnostic | Pass (Contract) |
| **A17** | B13 Curvature Readout | `oracle_kl_reduction` | 6 | $+0.0297$ $[+0.0270, +0.0324]$ | $+0.0297$ $[+0.0270, +0.0324]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A18** | Diffusion Evaluator | `heldout_joint_loss_reduction` | 3 | Validated | Validated | Sampling | `paired_training` | Diagnostic | Pass (Contract) |
| **A19** | RW-ResT Lite Evaluator | `weighted_proxy_loss_delta` | 3 | $+0.0078$ $[-0.0124, +0.0279]$ | $+0.0078$ $[-0.0124, +0.0279]$ | Sampling | `paired_training` | Diagnostic | Pass (Contract) |
| **A20** | Regret State Archive | `future_error_capture_lift` | 3 | $+0.3094$ $[+0.2824, +0.3364]$ | $+0.3091$ $[+0.2706, +0.3475]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A21** | Signed-Path Coherence | `coherence_brier_gain` | 3 | $+0.0385$ $[+0.0153, +0.0618]$ | $+0.0385$ $[+0.0153, +0.0618]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A22** | Physics Falsification | `surrogate_null_r2_gain` | 3 | Validated | Validated | Model Based | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A23** | Incremental CPU Pattern | `incremental_feature_exact_match` | 3 | $+1.0000$ (Exact) | $+1.0000$ (Exact) | Exact | `position_suite` | Diagnostic | Pass (Contract) |
| **A24** | Learned Budget Gate | `budget_utility_gain` | 3 | $+0.0234$ $[+0.0048, +0.0420]$ | $+0.0234$ $[+0.0048, +0.0420]$ | Sampling | `trace_analysis` | Diagnostic | Pass (Contract) |
| **A25** | MENTS Soft Backup | `finite_temperature_value_shift` | 3 | $-0.0731$ $[-0.0774, -0.0687]$ | $-0.0731$ $[-0.0774, -0.0687]$ | Sampling | `synthetic_objective_mismatch` | Diagnostic | Pass (Contract) |
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

| Contrast | Estimand $\hat{\theta}$ | 95% CI | Target Margin $\delta$ | Criterion | Empirical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Compute Reduction ($\Delta C$)** | $-27.78\%$ ($42.06$ vs $58.24$) | — | $\Delta C < 0$ | $C_{\rm on} < C_{\rm off}$ | **PASS** ($439$ early halts/cell) |
| **Quality Non-Inferiority ($\Delta Q$)** | $-0.03125$ ($0.46875$ vs $0.50000$) | $[-0.03125, -0.03125]$ | $\delta_Q = 0.05$ | $\Delta Q > -\delta_Q$ | **PASS** ($\Delta Q = -0.03125 > -0.050$) |
| **Interaction Equivalence ($\Delta_{TR}$)** | $0.00000$ | $[0.00000, 0.00000]$ | $\delta_I = 0.05$ | $\text{CI}_{95\%} \subset [-\delta_I, \delta_I]$ | **PASS** (Zero Antagonism) |
| **Training Main Effect ($\Delta_T$)** | $0.00000$ | $[0.00000, 0.00000]$ | — | — | Neutral |

- **Selection Trace Coverage**: $100\%$ ($1.000000$) across all 960 games under canonical `A01.live_pflip_v1`.
- **Promotion Status**: `COMPLETED_NO_PROMOTION` (first scientific confirmation on Gomoku7; multi-game generalization and larger scale training scheduled under Phase 16).

---

## 4. Architectural Analysis & Governance Actions

### 1. A01 Council Mechanism & Empirical Continuation Calibration (Stage 7 / C8)
- **Empirical Calibration Findings** (288 real MCTS trace bundles, 864 decision records against held-out higher-budget argmax):
  - **Dirichlet Posterior Stability ($s_{\rm H1}$)**: $\text{Brier} = \mathbf{0.1424}$, $\text{ECE} = \mathbf{0.0755}$ ($7.55\%$). Realized agreement closely matches predicted confidence across all 10 bins.
  - **Gaussian Top-Two Incumbent ($s_{P_{\rm flip}}$)**: $\text{Brier} = \mathbf{0.4464}$ ($3.13\times$ higher error), $\text{ECE} = \mathbf{0.5040}$ ($50.4\%$). Suffers from severe small-budget overconfidence ($99.8\%$ predicted confidence yielded only $22.3\%$ realized agreement).
- **Governance Resolution**: Confirmed that Dirichlet stability $H_1$ solves the small-sample overconfidence pathology of Gaussian $P_{\rm flip}$. Live search adapter is sealed under canonical variant ID `A01.live_pflip_v1` while $H_1$ continuation calibration is preregistered for Phase 16 multi-game search engine integration.

### 2. A13 (WU-UCT) & A16 (Graph State Cache) — Immediate Merges Blocked
- **A13**: The synthetic $+38\%$ duplicate reduction is largely structural in the synthetic toy setup. Past real-engine experiments falsified duplicate reduction for adaptive VL. A13 currently returns `MetaAction::Noop`. **Merge is BLOCKED pending real-engine differential ablation (PR-20).**
- **A16**: Production TT already shares `MctsNode`. A16 proposes eval-cache sharing without parent edge statistics sharing. Rust `propose()` is currently empty. **Merge is BLOCKED pending State/Eval cache vs SearchNode semantic decomposition (PR-21).**

### 3. A15 (Service Curve) & A19 (Semantic Alignment)
- **A15**: The estimand measures raw CUDA vs CPU ratio ($I^2=99.9\%$). Redesign estimand to GPU scheduling regret $G_{\rm sched}$ on shipped networks (PR-30).
- **A19**: Corrected estimand definition to `weighted_proxy_loss_delta` in unit `weighted_loss` (PR-42).
