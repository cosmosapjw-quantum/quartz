# C4. Frozen high-branch game distribution, before external teacher

Actual source generator freeze_gomoku14.py TDD RED→GREEN, original fixed seeds 202611210401 (12 neutral random nearby openings), 202611210402 (12 artificial off-center terminal-in-one stress states), exact Freestyle15 game. Each history consists of alternating Black/White flat indices y*15+x; 15x15 occupancy empty-cell legality, no prior five-in-a-row, nonterminal board. The source has 24 distinct histories, but the 12 stress roots deliberately reuse transformed motifs and are NOT 12 independent population families. No general pooled population bootstrap.

Initial center-only attention rule: rank legal moves by Chebyshev radius to (7,7), then Manhattan radius, then flat index; original online evaluator is UNIFORM (not a trained QUARTZ network). Top16 are selectable first8. Reopen <=4 rules-only immediate legal winning omitted moves (mandatory overriding inappropriate large-branch visitation concentration gate), or conditioned safe-reply candidates as explicitly held fixed. Root same-tree continues until total32 visits. Comparators: closed16, evidence, three frozen same-count random seeds, all-open-at8, all-legal32. All root edges originally allocated, physical cost included.

Freestyle NNUE Rapfi teacher is external, exact ELF/config/classical/freestyle weight hashes fixed C01, protocol rule=0. Teacher is not read or used before source freeze, not an oracle, cannot alter selection heuristics. Fixed requested timeout_turn=300ms and confirm move legal for each root. Secondary 600ms teacher used as stability context, never to select favorable positions.

Source hash: a744fa5166cb1cd54ec274128238649e7305a1ae3c4f15b894f35559602f5a9c. All 12 stress positions have at least one immediate legal win outside initial top16 by independent Python Freestyle rules; this is a constructed positive control, not unbiased outcome evidence.

C5 begins implementation only after this source is archived and published. No game-theoretic exact minimax available for 15x15, no Connect4 Pascal Pons target transfer.
