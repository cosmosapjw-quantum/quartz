# K31 change review
Verdict: CAUTION, research-only checkpoint; not production adoption or main merge.

Scope: K30 exact parent d754a8ed3d98116eaf08c0e2baa86badc51a1ab1 through C08D 0ec141ba2700eae12a8c1bcff0196c95976142fb. Reviewed new dispatcher, k31_cpu_server.rs, k31_gain.rs, k31_proof.rs, trial contract and existing proof graph/queue dependencies. Read-only review followed by separate evidence publication. Native subagent unavailable; direct review and independent numerical reconstruction. Shared Keystone gate reference not available from the skill service; no claim of a separate gate tool pass.

P0: none identified in bounded research scope.
P1: none identified for this checkpoint. Production qualification remains blocked by insufficient evidence.
P2:
- The identity fields in k31_cpu_server.rs::model and k31_gain.rs::load are caller-supplied cache identities; the native loader checks shape/syntax, not file-content digest. The locked Python collector hashes assets, but standalone calls can mislabel files. Keep trusted research use only; require native digest validation and invalidation tests before deployment.
- k31_proof.rs::execute builds from captured paths and registers real rule states, not actual MCTS hidden edges. Its planner is optimistic and adds unpriced traversal/score work beyond rule_work. Measure graph build, expansion, planning and wall separately; validate same physical budget in a fresh experiment.
- K31 direct regression was selected on CAL-A and tested without refit. All eight nontrivial TEST gain RMSEs exceed fresh marginal constant; CAL-B selected native serial/short, not learned policies. Do not adopt or claim general causal benefit.
- The exact interval proof is sound under legal full child enumeration and exact terminal rules. Non-witness nondominated readout does not prove improvement. TEST directed vs short improved seven roots and harmed zero; vs long 17 improved/17 harmed. No prespecified Gomoku noninferiority or general dominance. Add prospective heldout proof-guided native materialization and paired actual games.
- DR first-arm logging was randomized; the algebraic all-18-entry check conditions on recorded outcomes, not a proof of stable potential outcomes under scheduling/order. No IID or finite causal confidence claim.
P3:
- All arms collect internal traces; timing includes logging, setup and uncontrolled host load. The timing audit aggregates diagnostic samples for graph arms as well; use the complete TEST atlas or disaggregated timing only descriptively.
- GPU, sanitizer, nonblocking async and original Gomoku benchmark not tested. 89 ignored tests retained as ignored. Root labels exact on this cohort; not a new independent full solver/CNN rerun during C09.
Evidence: locked release exit0; 650 passed, 0 failed, 89 ignored; W1 27 passed; scalar resident parity 24 paired; independent audit exit0 covering 28,192 main requests, all 8,180 graph requests and 40,796 actual graph steps, 242 graph-request full planner checks, 1,960 native model predictions, 155 diagnostic batch-scalar parity evaluations; 7,655 exact stored WDL action comparisons. No invalid interval or witness detected. Frontier accounting cross-check passes.
Handoff: preserve research branch and raw observations; K32 prioritizes exact graph certificates integrated with real native search, cost-aware frontier and materialization; original P2 test separately. No default engine policy changed.
