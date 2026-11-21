# R3. Formal domain and evidence identity

Let s be a legal nonterminal 15x15 Freestyle state, A(s) the legal empty-cell actions, B(s)=|A(s)|≤225. Let p(a|s)≥0, sum over legal p=1 be the actual QUARTZ precommitted evaluator/heuristic prior, NOT Rapfi action labels. A0(s) is the initial top k=16, and A1=A0 union R with at most r=4 omitted legal moves reopened.

Rules-only tactical witness: w(s,a)=1 exactly when a is a legal immediate five-in-a-row terminal win under the selected game variant; v(s,a)=1 when a permits the opponent an immediately winning legal reply. The source of legality must always be Gomoku15::legal_moves. Freestyle and Renju use distinct legality and victory conventions.

Rapfi teacher a_R(s) is a finite-budget alpha-beta/NNUE action recommendation at fixed rule and bounded time, not exact minimax. Define agreement I_R(s,a)=1[a=a_R(s)] and local exact immediate-win selection I_W. A teacher disagreement is NOT necessarily an incorrect move. Exact Connect4 Pascal Pons WDL is not transferable to Gomoku15.

Hypotheses: H1 existing root visit/value stats survive 225-bit mask activation at one serial checkpoint; H2 tactical evidence finds legal decisive omitted moves more often than equally-sized random opening; H3 selective search improves or fails to improve agreement and one-ply tactical success relative to full-legal 32-visit QUARTZ search at actual similar CPU. H1 is a code test, H2/H3 empirical; mathematical combinatorics alone cannot establish H3.
