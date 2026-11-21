# C4. Frozen execution DAG and actual CPU resource contract

**Study source and operator are precommitted without external oracle:** two new 12-family×16 root draws seeds `202611210321/0322`, all prior COG and legacy mirror-canonical source excluded, `COHORT_384_FROZEN.json` SHA and `coding/INPUT_LOCK.json` before exact solver. Root action choice uses full legal prior; top3 ranked by original evaluator, ties center-first. One-ply tactical witness F≥0.5, immediate own win or opponent immediate reply safety; at most two extra root moves. All random control seeds `202611210301,0302,0303`. No per-root threshold tuning.

**Fixed real engine calls:** one full source prior CPU evaluator `eval.evaluate(&state)` for ranking, one serial resident `MctsEngine::new(state, evaluator, cfg.pw=None)`, full legal immutable root prior/edge slab, initial top3 selection mask; `FixedIterations(8)`; one fixed rule-based tactical probe; opt-in candidate activation via monotone `set_root_visible_moves`; `FixedIterations(24)` with 16 **additional** visits. Each arm shares the same full source root game rules, evaluator seed17, c_PUCT=2, no Dirichlet, root_forced_win false, single Rust thread.

**8 candidate-controller arms:** `resident_closed_sham`, `resident_evidence`, `resident_random_1/2/3`, `resident_all_after8`, `resident_full24` (no restriction at any stage), and `resident_closed_no_probe` (accounting-only overhead). All must preserve same root/edge histories and priors at checkpoint mask change; no engine restarts. Actual runtime active mask (not a missing-edge slab append) must admit newly activated action selection. Simulator cannot see exact Pascal Pons WDL.

**Temporal DAG:**
- C4 lock source + experiment and Git stage backup.
- C5 TDD RED→GREEN Rust engine masked selection and actual resident research endpoint with 8/24.
- C6 full Rust tests + release binary source/hash, Python regression and endpoint pilot.
- C7 independent new 384 exact oracle scores; query and mirror requery; publish.
- C8 run each of `384×3×8 = 9,216` ACTUAL serial Rust server requests, each two calls on *same* MCTS node with total24. Score with exact solver labels **only after all Rust requests**, compare actual end-to-end CPU time; repeated source/position held fixed.
- C9 independent Python tactical rule geometry + candidate mask/priors and per-edge N/W check with 48 independent mirror solver queries, family-cluster paired bootstrap, source SHA.
- C10 publish bounded scientific verdict and next node.

**Cost gate:** `mean[(prior evaluation)+(full root expansion)+(8 visits)+(oneply)+(mask update)+(16 visits)+(output)]` milliseconds, model-load excluded but reported, compare to normal all-legal full24 at relative ≤10% **for each mode/group**. Report search-only and evaluator-call physical costs separately. C4 source is new, no prior test root or external WDL label inspected for tuning. Conditional claim, not population mean or general brain cognition.
