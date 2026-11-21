# R8. Claim admission / stop decision before any C16 test

DERIVED: sound sufficient nonterminal fork certificate, monotone "two distinct winning cells" proof, exact 32-local-cell set-completeness under no preexisting own immediate win. This is existing classical threat-space content reformulated as a bounded online proof object, not invented threat search.

NUMERICALLY CHECKED: R7 one explicit 15x15 source, 218 legal opponent replies, 90 full-vs-local checks, Wolfram exact 32-local count and singleton counterexample. The math and examples are upstream of the Rust implementation.

LITERATURE-SUPPORTED: Allis et al 1993/1996 straight-four and threat-space search, progressive unpruning Chaslot 2008; PN-MCTS cost/performance can be unfavorable in Gomoku per Doe et al 2022/2023.

CONJECTURAL: finite cost-gated C16 source positions may contain missed fork certificates and evidence-guided activation may affect MCTS selection or real game WDL.

UNRESOLVED and hard admission gates: 20 independent heldout source families with omitted witness and actual trigger; matched observed per-stratum physical CPU; real Rust implementation preserving root semantics; actual paired full game WDL; statistical confidence. If no 20 independent activated sources, label conditional effect NOT_ESTIMABLE regardless of any constructed root percentage. If CPU exceeds or fails comparability, report NOT_COST_MATCHED. If real games show 0 diff, record finite null result without claim of universal failure.

Decision: implement source-only proof vs mask vs proof-choice controls under full-root legality and immediately report outcomes. No iterative meta-gates or threshold relaxation when heldout result negative.