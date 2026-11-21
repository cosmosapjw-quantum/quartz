# C2. Formula/data/code mapping

- Mathematical state s: src/games/gomoku15.rs Gomoku15::new(Freestyle), apply_move, current_player, legal_moves and board_as_12 (0 empty,1 black,2 white).
- Mathematical phi(s,a) from theory/R09_COMPUTATIONAL_CONTRACT.md: new Python feature_matrix in coding/policy15.py and exact Rust feature function in src/mcts_server/prior15_cpu_server.rs. Python reads source JSON histories and game Freestyle legality; Rust reconstructs same history independently.
- Teacher finite y(s): C14 coding/rapfi_teacher14.py query_one external subprocess, Piskvork rule0 and byte-frozen 250615 freestyle NNUE config; labels never passed as request fields.
- theta fit: full-batch finite exact softmax NLL plus fixed L2 (lambda=.01), code in policy15.py; resulting only 13 scalar coefficients and metadata JSON. Rust trained Prior15 implements Evaluator<Gomoku15>, normalizes policy across every legal action, value=0.
- Source build/transport: Cargo.toml existing serde/serde_json and MCTS engine; new research endpoint registered in src/mcts_server.rs, with separate command for raw numerical features/parity.
- Action control: existing src/mcts/select.rs resident_root allowed mask (four AtomicU64), root original N/W/M2 and priors retained. Gated run first eight root visits followed by optional legal oneply witness and continued same-root searches until real CPU budget target; learned_full and uniform_full baseline skip gating.
- Heldout testing: new C4 board histories, C7 only train/validation labels, freeze model first, C8 then heldout teacher labels. Per-stratum paired outcome and actual CPU wall cost read from Rust JSON, independently re-evaluated in C9.

No external NNUE binary/model is converted, overwritten or placed under Git.