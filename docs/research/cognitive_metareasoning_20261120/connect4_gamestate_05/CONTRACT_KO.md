# QUARTZ COG01B C5: Connect4 GameState and exact decision utility

Parent: bc44e5c85c1f37201f80880f18d59fdb55a509a8
Research branch: research/quartz-cog01b-connect4-gamestate-20261120
Requested stage: next unfinished implementation C5. Preserve all previous scientific branches and originals.

## Scientific contract
- Exact board: standard Connect4, 7 columns x 6 rows, gravity, 4 connected horizontal/vertical/diagonal. Action a is zero-based COLUMN (not cell); external solver uses 1-based column strings. Board representation: row-major bottom-up; current_player ±1.
- Implement GameState in src/games/connect4.rs and a game-specific bounded Rust JSON server command using MctsEngine<Connect4>, UniformEval initially (zero value, uniform legal prior). This is an actual QUARTZ MCTS baseline, NOT an NN or cognitive improvement.
- The research endpoint will preserve a single engine root and report best action at root visits 8,16,32,64, root TT hash, cumulative evaluator calls and actual per-step wall time, using serial baseline and fixed seed; protocol is opt-in to avoid changing existing live server paths.
- Strictly verify the input column history is legal and non-terminal, no invalid move, full column, or turn after victory. Reject invalid command with a structured error, not a fallback empty position.
- The existing external solver 12-source position corpus is pinned by INPUT_LOCK.json. Join exact game-theoretic per-column integer scores to MCTS-selected column, compute U(a)=sign(score(a)); for each adjacent budget pair report signed ΔU; keep all negative findings. Raw score magnitude is an ordinal win-timing preference, NOT a cardinal game payoff.
- Test first: gravity behavior, immediate wins in all four geometric directions, terminal winner and next-to-move sign, full column and full-board draw, stable hash and Undo, all 12 actual solver root histories and legal column sets. One test must fail before implementation.
- Independent Python 7x6 reference from previous standard_games.py and solver 12 histories will crosscheck every state/available action, plus mirrored boards and score symmetry.
- No GPU, no pre-trained Connect4 NN installed. Do not invent a Connect4 NN or claim metacognitive VOC strength. Strong baseline is a later optional stage once a trained recognition model is acquired.
- Cap experiment to 12 roots and budgets 8→16→32→64, plus a solver-based external check. No independent statistical inference from mined small sample.

## Stages and publishing
A. Freeze contract/input identity -> commit -> local Dropbox archive SHA256 -> push -> remote readback.
B. TDD Rust board + server endpoint, Rust tests and build -> commit -> backup -> push.
C. Run clean source, report exact sampled utility and cost, checksum output -> commit -> backup -> push.
D. Result verdict + next ID and reproduction commands -> commit -> backup -> push.
Root developer checkout and worktrees read-only. No external book or GPL/AGPL source/binaries are vendored or published.
