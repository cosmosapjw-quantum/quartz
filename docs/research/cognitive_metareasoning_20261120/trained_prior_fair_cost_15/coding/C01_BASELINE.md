# C1. Source baseline and pre-registered scientific acceptance

The repository C14 main result C10 commit eaff0ae is preserved; this branch begins after the R10 theory commit. The developer worktree remains untouched. C01_INPUT_LOCK.json is generated from actual byte reads, verifying external Rapfi executable, classical model, freestyle NNUE and config against the existing C14 source lock. No model binary is committed.

New source cohorts: training 128, validation 32, natural heldout 48, constructed terminal-in-one diagnostic 16. Source groups are frozen before teacher labels. Rapfi teacher rule0 budget 300ms; optional stability 600ms. Physical CPU target 20ms per Rust request; the <=10% mean runtime mismatch acceptance is applied separately to natural and diagnostic strata for learned_gated versus learned_all_legal. Full original root edges remain present.

Comparators: learned_gated, learned_all_legal, uniform_all_legal. Policy-only log-linear model has no trained value head (v=0). Neither GPLv3 Rapfi binary nor pretrained weights may be redistributed in the Git source or stage archives.

Outputs: independent heldout policy top-k and teacher agreement, pairwise root control differences at real CPU costs, visits/evaluator calls, immediate wins in constructed diagnostics, runtime by stratum, and separately audited legality. This cannot establish Elo or full-game minimax strength.
