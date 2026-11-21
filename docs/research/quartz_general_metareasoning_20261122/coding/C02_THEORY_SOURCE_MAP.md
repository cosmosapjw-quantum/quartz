# C2. Definition to source interfaces and test obligation

Mathematical belief b over S states maps to immutable posterior tuple/list (positive total, no NaN, sum1); finite terminal utility U is a matrix S×A, unconstrained signed finite f64 entries; experiment c has per-state observation probabilities L_c (S×Y_c) and strictly positive deterministic resource cost κ_c. Calibration/sampling input is an *experiment channel*, not a game or policy prior.

Expected Python implementation:
- FiniteProblem validates prior, utility and all experiments.
- terminal_value(b), oracle_value(b), simple_regret(b) implement K−M.
- possible_posteriors(b,c) iterates positive mass y, Bayes updates and skips zero-probability outcomes.
- evsi(b,c) uses post-update *terminal decision improvement*, not posterior entropy.
- one_step_indices(b,λ) returns gross EVSI, rate, net and both ratio/net choices with an independent STOP condition.
- optimal_bellman(b,H,λ) computes priced H-step optimal cost-to-go and first computation/STOP; a high-level test must show H0..H3 and exact input immutability.

Expected Rust module is a production-code-visible but **reference-only** module in src/mcts/policy/finite_metalevel.rs: named structs finite observations, explicit input validation, correct posterior update and one-step EVSI/net/Bellman. It must not change MCTS/P UCT semantics, the GameState trait, per-game heuristics or live pipeline. Unit tests use the same original rational fixture but independently calculate floats.

Independent source authority:
- R7 Python Fraction and Wolfram symbolic computation is the exact oracle for fixed finite model.
- C4 registered JSON scenario bank is the sole experiment input, frozen before implementation and run.
- C8 uses actual Python lab running independent randomized sample trajectories with fixed seeds, no external teacher.
- C9 independently checks empirical scores and source bytes, and restores a local Git backup archive.

Scientific boundary: passing finite Bayes tests does not mean calibrated MCTS posterior, actual trained search quality, CPU P4 hot path or full THESIS P3 emergence. Read BQ++ primitives, recognize overlap, do not insert redundant PUCT controls.