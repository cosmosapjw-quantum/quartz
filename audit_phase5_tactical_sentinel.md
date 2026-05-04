# Audit — BQ++ Phase 5: Tactical sentinel (Gomoku focus)

**Date:** 2026-05-04
**Scope:** cheap CPU-only forced-move solver. Gomoku gets a full
implementation (immediate-win + forced-block detection); Chess and
Go are deferred to follow-up patches because their move generation
is more involved and the user's primary game is Gomoku 7×7 per the
README's "Current Controller Status" section.

## Why a tactical sentinel

Hidden-win recall is the primary failure mode of pure PAC stops. A
low-prior winning move never enters the candidate set ⇒ the
certificate certifies the wrong best arm. Game-specific cheap
solvers detect "this move forces an immediate win" or "this opponent
move is forced" and inject the result into the candidate reservoir
as `forced_move_pos` (a field on `PolicyCache` from Phase 2).

The sentinel is **conservative**: returns `Some(action)` only when
the move is provably forcing. False negatives are acceptable (the
main search will eventually find them); false positives would
corrupt the search.

## What changed

### New file `src/mcts/policy/tactical.rs`

- `TacticalResult { Forced(action_id), None }`: pure-data result
  type with `is_forced()` and `action_id() -> Option<u32>` accessors.
- `gomoku_sentinel(state: &Gomoku) -> TacticalResult`: full
  implementation.
  - Iterates over every empty cell.
  - For each, checks whether placing the current player's stone
    completes `win_len`-in-a-row in any of 4 directions
    (horizontal, vertical, two diagonals). Time complexity:
    O(size² × win_len), e.g. 225 × 5 = 1125 array reads on
    Gomoku 15×15 — well under 10 μs.
  - **Priority: immediate win > forced block > None.**
  - Returns the first matching position (deterministic given the
    board state).
- `would_complete_win_at(state, pos, player, win_len) -> bool`:
  internal helper that walks the 4 direction vectors from `pos`,
  counting consecutive stones of `player` and returning true if the
  count (including the hypothetical placement at `pos`) ≥ `win_len`.

### `src/games/gomoku.rs`: 5 new public accessors

- `Gomoku::size_dim() -> usize`: board side length.
- `Gomoku::win_len_dim() -> usize`: win-condition (4 for 7×7, 5 for 15×15).
- `Gomoku::current_player_sign() -> i8`: +1 black / −1 white.
- `Gomoku::cell_is_empty(pos) -> bool`: in-bounds + empty.
- `Gomoku::cell_is_player(pos, player) -> bool`: in-bounds + matches.

These are minimal accessors (each ≤ 4 LOC). No internal Gomoku state
is exposed beyond what the sentinel needs. The accessors are also
useful to other future code that wants per-cell queries without
copying the full board.

### `src/mcts/policy/mod.rs`

Re-exports `gomoku_sentinel, TacticalResult`.

## Tests added (7)

1. `test_phase5_gomoku_empty_board_no_forced`: empty 7×7 ⇒ None.
2. `test_phase5_gomoku_immediate_four_detected`: black has 3-in-row
   at row 3, cols 0-2; black to move; sentinel returns `Forced(3*7+3)`.
3. `test_phase5_gomoku_forced_block_detected`: white has 3-in-row;
   black must block at the win-completion position. Sentinel returns
   the block position.
4. `test_phase5_gomoku_two_in_row_not_forced`: only 2-in-row each ⇒
   None.
5. `test_phase5_gomoku_terminal_state_no_forced`: terminal position
   ⇒ None (no further play possible).
6. `test_phase5_tactical_result_api`: `TacticalResult` accessors
   work correctly.
7. `test_phase5_gomoku_immediate_win_priority_over_block`: black has
   immediate win at (0, 3); white has immediate win at (6, 3). Black
   to move takes the win, not the block — verifies priority.

## Test results

- `cargo test --release`: **517 passed** (was 510 + 7 from Phase 5).
- All P01-P08 + Phase 0-4 tests still pass.

## Bug caught during port

The initial test code used `g.apply_move(mv)` (the immutable
GameState method `fn apply_move(&self, mv) -> Self`) and discarded
the result, so the test states never advanced. Fix: switched to
`apply_move_in_place_no_undo` which mutates in place.

This bug-class would have been caught at Rust compile-time IF
`apply_move` returned `()` instead of `Self`. The unused-result
warning is silent because `Self` is a type. **Lesson logged**:
when calling Gomoku trait methods in tests, always use the
`*_in_place*` variants for clarity.

## What is NOT in this patch

- **Chess sentinel.** Mate-in-1 detection requires the full
  legal-move generator with check verification. Chess move
  generation is correct in `src/games/chess.rs` but the sentinel
  composition is not trivial; deferred to a follow-up patch.
- **Go sentinel.** Atari / capture detection requires the existing
  Go ko-rule and group-tracking logic; deferred to a follow-up.
  Per the audit's "no false positives" priority, ladder detection
  is **out of scope** even in the eventual Go sentinel — it's too
  expensive on the hot path and risks misclassifying complex
  positions.
- **Engine integration.** The sentinel is a pure function; no
  policy currently calls it inside the search loop. Phase 8 will
  compose it with the BQPP policy.

## Adversarial review

### Why immediate-win priority over block?

If both are available, the player should win, not block. Blocking
when winning is a clear regression; winning when blocking is
correct (you don't need to defend against threats you can pre-empt).
This priority ordering is enforced by the sequential search:
immediate-win loop runs first; forced-block loop runs only if no
immediate win is found.

### What if multiple winning placements exist?

The sentinel returns the **first** (lowest-index) winning placement.
This is deterministic given the board state. In Gomoku, multiple
simultaneous wins are usually equivalent (they all complete the
game); the choice between them doesn't affect the outcome.

### Performance

Gomoku 15×15 with win_len=5: 225 cells × 4 directions × ≤ 5 stones
walked per direction = ≤ 4500 array reads per sentinel call. Modern
CPUs do this in 1-5 μs.

The sentinel is called at most once per `observe` boundary (the
PolicyCache publish point), so it's called every ~100 iterations,
not every selection. Overhead is negligible.

### What if the board has > win_len in a row?

The sentinel returns true for any line with ≥ win_len consecutive
stones of `player`. Pre-existing > win_len lines are valid (the
game would already be terminal); the `is_terminal()` check at the
top of `gomoku_sentinel` handles this case by returning None.

### Conservative bias

If the sentinel returns None on a forcing position (e.g. complex
threat-pattern that requires multi-ply lookahead to verify), the
search continues normally and may eventually find the win. If the
sentinel returns `Forced(action)` on a non-forcing position
(false positive), the search halts prematurely with a wrong move.

The implementation guarantees no false positives by construction:
it ONLY returns `Forced` when a single-ply placement completes
`win_len`-in-a-row. This is a one-ply lookahead; provably correct.

## Files touched

- `src/mcts/policy/tactical.rs` (NEW, 250 LOC incl. tests)
- `src/games/gomoku.rs` (+45 LOC; 5 public accessors)
- `src/mcts/policy/mod.rs` (+2 LOC; module + re-exports)

Net delta: **+297 / 0 LOC**.

## What unblocks next

- **Phase 6 (nested-reservoir)**: uses `gomoku_sentinel` (when
  available for the active game) to seed the candidate reservoir
  with forced-move positions before Gumbel-top-m sampling fills
  the rest.
- **Phase 8 (battle)**: the `BQPP` policy's `should_halt` consults
  `cache.forced_move_pos` (populated by the sentinel) and returns
  `Stop(TacticalForced)` with the forced action immediately, taking
  priority over EB-cert and KG-stop.
- **Follow-up patches**: chess and Go sentinels can be added
  incrementally without touching the trait surface or the cache
  schema.
