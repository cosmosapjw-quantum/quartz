# C7. New C13 frozen Connect4 positions solved by independent external Pons oracle

Source absolute: `docs/research/cognitive_metareasoning_20261120/resident_reopening_13/COHORT_384_FROZEN.json`, SHA256 `3b79dd2341e0e56405a2219f41fb0e28047bf642a7372f417f23513459450b31`, frozen at C4 commit `88ffc56dfe80d36e6f9632ca8d35b14c2d45e17f` **BEFORE** any exact game label requests. Two new 12-family×16-root cohorts, no old G07–C12 mirrored overlap.

Exact Pascal Pons external solver source and opening-book identity locked via C1 SHA before data generation. Source-only TDD RED `/tmp/c13_C07_tdd_red.log` (unimplemented teacher mapping and status), GREEN `/tmp/c13_C07_tdd_GREEN.log` 4 Python tests. Source wrapper committed C07a `66ab4ef5f527d5ac266f77e8d7078b2199d20680` and archived/pushed BEFORE actual solver execution.

Original external game-theoretic solver run: 384/384 queries **OK**, 0 timeout/bad label, elapsed ~29.923s; full 7-column action score file `evidence/new384_exact_20261121/EXACT_NEW384_LABELS.json`, SHA256 `73cf3b37ddbcc7f14298ee0634b760f5008bfb2194bd5cd5fc31dc24ceb1a817`. Result-creation source commit exactly `66ab4ef5...`. Independent `coding/audit_oracle_resident13.py` recomputed all 384 board states, legal full-column sentinel -1000, WDL optimal legal set, target policy and root sign; queried 24 additional distinct root histories and 24 horizontal mirrors and compared reversed scores. All passed.

Root WDL decision-informative count: **resident_A 125/192, resident_B 127/192, total 252/384**. Others have legal root moves with same WDL sign and cannot demonstrate a WDL improvement from action selection. Root game-theoretic WDL counts A win136/draw8/loss48; B win139/draw12/loss41.

Important temporal contract: external exact labels were generated after the entire C5 Rust policy and C4 new source frozen, but **before** actual C8 search. They are never passed to Rust runtime, not used for any threshold/tactic/seed/model fitting. C8 runner opens exact target source **only after all Rust endpoint root searches finish**. The result is an external posthoc exact action oracle, not learned evaluator or causal source for online behavior.

Published C7 source/label SHA and local archive receipt are preserved separately; C8 new science remains unexecuted at this stage.
