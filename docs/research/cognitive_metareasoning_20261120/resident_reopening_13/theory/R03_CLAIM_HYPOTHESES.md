# R3. Define precisely which claims and assumptions matter

Let `s` be a nonterminal gravity-Connect4 state, `A(s)` its legal actions (≤7), `P(a|s)≥0` original full-legal evaluator prior with `Σ_A P(a|s)=1`. Tied initial prior ranks favor center column 3, then index. `A_0` contains top min(3,|A|), `A_1 = A_0 ∪ R(s,π_8)` where R is COG03's precommitted one-ply legal tactical witness, up to two extra actions, and `A_0⊆A_1⊆A`.

Define `N_a,W_a,M_{2,a}` on a resident root's materialized edges; `N_root = Σ_a N_a` under one completed PUCT root path per iteration with no parallel pending virtual loss. All inactive edges exist as game-tree objects but are **excluded from root PUCT argmax**; they must have `N_a=0` before activation. Their existence adds real upfront setup/evaluator cost.

**H1 implementation:** one root node/TT identity and edge statistics survive mask growth from top3 at visit8 to subset A1 at visit24. No root re-creation or root-evaluator re-call during activation. Selection of newly activated legal actions is possible in the same root. Descendant legality unaffected.

**H2 finite game inference:** the same-tree evidence arm recovers exact-WDL decisive omitted actions, compared with sham and random candidates at matched CPU. External Pascal Pons labels may classify `R`, but never select A1.

**H3 important comparator:** full-legal root MCTS from visit0 at equal CPU search budget has *higher* or *lower* mean exact action utility than the resident reopening arm? A result in either direction closes the question. Under no circumstances can `H1⇒H3` be asserted.

**Conventions:** `U(s,a)=sgn S(s,a)∈{-1,0,+1}`, sentinel S=-1000 excluded, regret `Reg(s,a)=max_A U(s,a)-U(s,a)` ∈ {0,1,2}, physically measured CPU wall cost `C = full-root prior + initial node/materialization + 8+16 iterations + one-ply rule check + mask activation overhead` in milliseconds. Difference `Δ=Reg_resident-Reg_full`, negative better. Causality limited to fixed synthetic opening draws, deterministic evaluator and search seed, not an IID sample of players.

**Claim status:** H1 currently implementation design (UNVERIFIED until runtime). H2/H3 are hypotheses; COG03-12 is restart-only evidence; conditional family inference is not population inference.
