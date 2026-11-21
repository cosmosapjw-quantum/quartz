# R3. Scope, exact terms, definitions and observables

State s is a nonterminal Freestyle 15x15 alternating legal history. Let t be to-move player, o the opponent, E(s) the empty cells. For p in {t,o} define W_p(s)={c in E(s): placing p at c makes a line of >=5}. W_p is a *counterfactual legal board pattern* even if p is not currently to move; it is not a legal actual turn if p !=t.

A nonterminal fork certificate at candidate a in E(s) is F(s,a) iff:
(i) W_t(s)=empty and W_o(s)=empty;
(ii) placing t at a is NOT terminal;
(iii) in resulting s+a, |W_t(s+a)|>=2, with >=2 DISTINCT empty endpoint cells;
(iv) W_o(s+a)=empty (can be deduced from (i) in monotone Freestyle).
Full certificate witness stores a and two endpoints c1!=c2 with exact game-coordinate line verification, and rejects any illegal or source-terminal input.

F(s,a) guarantees a win by t in at most the next own turn (3 plies a,b,c), against any legal opponent reply b, but only for classic Freestyle with no capture/forbidden moves. W_t(s) empty enforces nonterminal novelty relative to trivial win-in-one. Any preexisting opponent immediate win would invalidate an unconditional guarantee.

Trained prior rank r_theta(s,a) sorts all original legal-action probabilities descending with action index tie-breaking. An *omitted fork* exists when F(s,a) and r_theta(s,a)>16. Actual activation means at least one such certified omitted candidate is admitted to the root action mask AFTER the first 8 MCTS visits, preserving full root prior/edges and N/W/M2. A selected certificate is distinct: best_move remains ordinary MCTS unless the arm explicitly declares certified proof-choice override (separate intervention).

Let A=learned_fork_mask, B=learned_all_legal, C=learned_fork_proof, D=learned_sham_mask. Comparisons A-B isolate mask+certificate-computation and C-A isolate proof-choice rule under matching total CPU, conditional on actual activations and nonnegative compute gate; C-B combines effects and is not pure reopening.

Primary diagnostic outcome source-local: indicator of selecting a theorem-certified action, not teacher exact recommendation. Full-game paired outcome (win=1, draw=1/2, loss=0) uses actual complete rule-replayed games, alternating colors from same frozen opening. Pair contrast Delta=(1/n) sum_i(W_A,i-W_B,i), with pairing as the *source opening cluster*, not individual moves; report counts and uncertainty without claiming Elo from small n. Per-stratum elapsed wall CPU match E=|mean tau_A-mean tau_B|/mean tau_A <=0.10; use actual wall time, equal visits alone invalid.