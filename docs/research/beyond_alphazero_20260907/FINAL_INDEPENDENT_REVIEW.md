# V01 — Independent final research review

Date: 2026-09-07. Verdict: **confirmed**, for the bounded research conclusions and proposed next-step plan after the three corrections recorded below. This verdict does **not** confirm beyond-AlphaZero strength, production correctness, or a new algorithm's high-budget advantage. The documents explicitly leave those claims unproved.

The reviewer prepared the separate L01 literature memo but did not author the main Korean report, mathematical report, implementation plan, pilot, analysis script or pilot review. This review independently assesses their synthesis and evidence. It is a quality review, not a publication permission gate.

## Evidence inspected

Read the complete `RESEARCH_UPGRADE_KO.md`, `MATHEMATICAL_UPGRADE_KO.md`, `IMPLEMENTATION_PLAN_KO.md`, `RESEARCH_RECORD.md` and `PILOT_INDEPENDENT_REVIEW.md`. Read `evidence/execution.json`, `evidence/source_probes.stdout`, `evidence/math_checks.stderr`, `evidence/pilot/summary.json` and `evidence/analysis/validation.json`.

Inspected the pilot summary's metadata, all 240 allocation rows, exact-risk availability, selected high-budget comparisons and coverage values through read-only JSON arithmetic. Independently recomputed the displayed high-budget relative reductions and the plan's evaluation-slot totals. Verified that the summary SHA256 matches the analysis validation record and that the compressed raw file SHA256 matches the pilot summary. Did not rerun the pilot or decompress/reaggregate the raw data: the earlier independent pilot review and recorded analysis provide that separate verification.

Spot-checked primary source meaning in `prototype/bqpp_prototype/kg.py`, `src/mcts/policy/kg_stop.rs`, `src/mcts/foundry/policy.rs`, `src/mcts/foundry/control.rs`, `src/mcts/mod.rs`, and `quartz/experiments/forked_voc.py`. Read the actual Q1, Q7, Q16, Q18, Q23 and Q24 statements in `docs/legacy/mcts_study/v6.0/mcts_conversation.txt`. Compared literature claims with the primary texts previously inspected for L01, especially Sezener–Dayan, correlated KG, Gumbel, MPV-MCTS and EMCTS.

Read the tracked changes and working-tree status at review time: changes were confined to this research directory. This is a bounded scope check, not an exhaustive repository audit. Publication provenance was still being completed by the coordinating author; this review does not certify a remote branch or push result.

## Findings supporting the verdict

1. **Original motivation is preserved without treating analogy as evidence.** The reports retain small-network structure recognition, latent utility beliefs, selective computation, multimodal/refutation search, statistical-field corrections and information geometry. They correctly distinguish inspectable historical questions from recovered personal-context summaries. The original questions explicitly seek a non-quantum-computing analogy and demand care about discreteness, gauge assumptions and parameter interpretation; the final report reflects these constraints rather than repeating the old assistant's stronger physical claims.

2. **Prior-art overlap is stated honestly.** Correlated Gaussian/GP values with VOC in MCTS are credited to Sezener–Dayan; correlated KG to Frazier and colleagues; regularized planning, Gumbel, mixed evaluator sizes and correlated model-error handling have explicit competing precedents. The proposal does not claim that combining these ideas establishes novelty. Gumbel and MPV-MCTS are correctly acknowledged to include substantial-budget experiments. The recent preprints are distinguished from established published references.

3. **The checked mathematical core is correct under the stated assumptions.** The linear Gaussian posterior update, two-action innovation-based KG, common-mode cancellation, Gibbs variational identity, soft-value Hessian and second-order contrast correction are consistent. The acyclic two-node Hessian is a valid counterexample to diagonality. The two-bit example correctly shows that zero one-step KG does not imply zero multistep value. The Laplace, non-Gaussian and information-geometric sections remain conditional methods to investigate, not validated performance claims.

4. **The source counterexamples are concrete and scoped.** The inspected Python and Rust KG wrappers set leader KG to zero, and use the recorded uncertainty proxy rather than the derived one-observation innovation. The live H1 input is `1-p_flip`, the snapshot sets `n_visible=n_children`, and the gain LCB proxy is `0.5*gain`, matching the report's findings. The report does not claim that the research fixes these production paths or validates the whole engine. Its reported leader KG, proxy scale, zero-support and policy-churn examples agree with the recorded outputs.

5. **Exact and Monte Carlo quantities remain distinct.** The exact Bayes-regret comparison is restricted to the correctly specified two-finalist Gaussian diagnostic. The misspecified rows have no exact Bayes-regret value. Coverage and realized losses come from 4096 prior worlds per scenario; the report does not call them conditional frequentist guarantees. It does not count dependent budget/width/method rows as independent experiments.

6. **The reported numbers check.** There are 240 rows and 82 distinct raw keys; every allocation count sums to its declared completed-observation budget. At budget 2048 and width 32, pending-greedy's exact regret reductions relative to equal allocation are 15.480871%, 15.407359% and 15.442908% for the independent, positive and negative scenarios. Relative to Neyman they are 0%, 0% and 0.000072869%. Misspecified coverage is 45.800781% at budget 8 and 93.359375% at 2048. These agree with the report. The conclusion that no useful additional high-budget advantage over the strong known-noise control was demonstrated is supported.

7. **Performed work and future work are separated.** Execution records show zero exits and no timeout for the recorded checks, source probes and one pilot; the report accurately calls the approximately 5.7 seconds a process duration, not simulator throughput. The 12-test log is identified as research checks rather than repository-wide tests. Missing Wolfram/Rust/SymPy execution is disclosed. No claim is made that Gaussian coordinate observations are actual NN calls, asynchronous GPU scheduling, self-play or training.

8. **The high-budget plan is actionable and proportionate.** W1–W7 specify observation identity, epoch invalidation, latent contrast calibration, pending incremental value, reversible reopening, engine connection points, fixed-evaluator comparisons, matched calls/time and workload reinvestment. Strong baselines, covariance misspecification, cold/warm roots and separate training endpoints are included. Slot totals independently recompute to 2,356,992 for W5, 14,128,128 for the W6 default curve and 2,359,296 for its stress layer. Sample sizes and effect thresholds are proposed design inputs, with pilot-based precision and insufficient-evidence exits, rather than asserted power guarantees.

## Initial corrections and their resolution

The first read found three mathematical presentation issues. The author made these changes; this reviewer read the revised sections and confirmed them:

| Initial issue | Corrected statement | Status |
|---|---|---|
| Section 7 introduced pending work P but could be read as applying standalone batch KG to its incremental value. | For fixed Gaussian designs, `KG(C|B,P)=g(delta,s_(P union C))-g(delta,s_P)`, with `s_A^2=d^T(Sigma-Sigma_A)d`; standalone `g(delta,s_C)` applies when P is empty. Adaptive choice after observing P requires its outer expectation. | Resolved |
| Section 8 appeared to require only a correct Gaussian likelihood for the single-covariance exact-risk formula. | The formula explicitly requires outcome-independent allocation and deterministic final posterior covariance, as in this pilot. General adaptive allocation/stopping requires evaluating the adaptive process. | Resolved |
| Section 10's stopping inequality left a computation-dependent cost outside the maximization. | `max_c[KG(c)-lambda*C(c)] <= 0`. | Resolved |

These corrections affect the general statement of the mathematics. They do not alter the executed pilot, its valid fixed-design assumptions or its reported numerical results.

## Remaining limits

No blocking correction remains for the deliverable's stated scope. Historical campaign raw data and recovered personal-context records were not independently reconstructed by V01; the report correctly treats their scope as limited. Figure rendering and a full source audit were not repeated. Production compilation, actual high-budget engine performance, learned covariance quality, and training gains remain untested here.

For future W2/W6 execution, a held-out coverage tolerance is a diagnostic gate, not by itself a proof of a valid sequential stopping certificate; the mathematics already requires separate time-uniform assumptions and checks. Future budgets, practical-effect choices and confirmatory stopping rules must be frozen in the proposed new manifests before their associated outcomes are observed. These are implementation obligations already consistent with the plan, not reasons to reopen this completed Gaussian diagnostic.

The confirmed outcome is a source-supported research reassessment, checked conditional mathematics, a reproducible illustrative pilot with its negative strong-baseline result intact, and a concrete plan to test the remaining claim. It is not evidence that Quartz currently surpasses AlphaZero.
