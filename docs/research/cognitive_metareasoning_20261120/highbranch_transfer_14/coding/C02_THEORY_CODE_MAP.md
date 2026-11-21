# C2. Direct math to actual source map

Domain: Freestyle15 actions 0..224, 0-based flat index y*15+x. Actual game source is src/games/gomoku15.rs::Gomoku15 with legal_moves, is_winning_move, apply_move, TT hash and nonterminal from original move history. Renju rules MUST be handled separately (variant Renju, legal_moves black forbidden checks). Do not fake gravity or use Connect4 WDL labels.

Selection mask is currently one AtomicU64 at src/mcts/mod.rs::MctsEngine::root_action_mask, and search entry points in src/mcts/select.rs::select_core filter only if idx<64. For full Gomoku15, action index can be224. Generalize to four atomic words. Keep existing 7-column method returning first word for backward compatibility; add full four-word snapshot method. Update owned/in-place/async selection call sites. Full original root edge slab remains immutable.

At checkpoint first8, preserve original full root hash, all edge snapshot (move,prior,visits,W,M2) and evaluator-call count, store monotone bit words, then resume FixedIterations(32) on same MctsEngine, not a fresh tree. Root mask is an evaluator/control policy, not a change to legal Gomoku game rules. Test first edge excluded and high word actions 64,127,128,191,192,224.

Original Rapfi Piskvork protocol is external: START 15, INFO rule 0, INFO timeout_turn 300, BOARD with x,y,player (1=black,2=white), DONE. Its stdout best move is not a QUARTZ prior and must not be passed into online selection. Path/config/weight SHA in C01 lock. Selected reference is finite alpha-beta/NNUE, not solved minimax.

Data and metrics: 12 neutral +12 tactical preconstructed Freestyle sources, rules-only immediate winning candidates vs fixed center/geometry prior top16, matched CPU end-to-end elapsed and independent Rapfi teacher agreement. Cost includes full root original materialization, every branch; restrict no user primary repository.
