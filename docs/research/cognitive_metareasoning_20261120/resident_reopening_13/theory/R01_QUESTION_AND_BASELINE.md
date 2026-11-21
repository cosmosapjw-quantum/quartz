# R1. 원본 목적과 좁은 판별 주장

**Research ID:** QUARTZ-COG-03-RESIDENT-REOPENING-13, parent `6abccb31f00f5f07b886df6aaae9794a2524b384`, branch `research/quartz-cog03-resident-reopening-20261121`. Authorized full R1–R10, followed by C1–C10. Original developer checkout is protected.

**Physical/gaming domain:** standard gravity Connect Four, 7 columns × 6 rows, win length 4, legal 0-based columns and alternating ±1 player-to-move, deterministic CPU-only single-thread PUCT MCTS with original trained tiny MLP, Conv3×3, and Uniform evaluators separately. External Pascal Pons exact game-theoretic action scores are posthoc labels only, not online search input.

**Prior evidence (COG03-12):** 384 new root positions/24 distinct five-ply families, 9,216 real root requests, 8+16 visits on *two different trees*. Tactical root reopening lowered MLP WDL regret from .2578125 to .1536458333 and Conv from .2109375 to .140625 versus artificially restricted top-three, but plain all-legal remained better (.13020833 and .1171875 respectively). This is an engineered attention intervention, not human Einstellung.

**New narrow falsifiable question:** can a *single* root tree, with fixed **full legal prior and actual existing seven candidate-edge objects**, initially enable only top3 while retaining every first-stage visit/value, and then activate up to two omitted legal actions with one-ply tactical evidence? Does this outperform normal all-legal MCTS at **comparable actual end-to-end CPU milliseconds**, including one-time root evaluation, one-ply probe, and selection mask overhead?

**Theory deliverables:** root-active-set evolution, conservation/semantics lemma and counterexample, PUCT score contract and exact-WDL signed regret, comparison designs and declared claim gates. A new strong no-go theorem is neither required nor claimed. **R1 status: context/source identified, problem and acceptance fixed**. Next: verify original progressive-unpruning literature and project memory/layout.
