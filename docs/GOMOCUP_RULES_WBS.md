# Gomocup Rules Upgrade WBS

## Goal

Upgrade the current Gomoku/Renju code to support Gomocup league rules while
preserving the existing Korean omok variant as a separate rule mode.

This upgrade now has two tranches:

- tranche A: rule taxonomy and runtime semantics
- tranche B: Gomocup manager/protocol compatibility

The current implementation scope covers:

- rule taxonomy cleanup
- game-state semantics
- server/runtime variant routing
- regression protection
- Gomocup `pbrain` command handling
- Gomocup `INFO rule` / time-control handling
- BOARD/opening ingestion for prepared openings
- compatibility handling for `pbrain-*` executable naming and `YXBOARD`

## Constraints

- Do not remove or rewrite the existing Korean omok variant.
- Keep current search/MCTS code reusable across all rule variants.
- Prefer additive, test-backed changes over broad rewrites.
- Preserve current de facto freestyle behavior for the existing `gomoku15`
  route unless an explicit Gomocup-only route is selected.

## Source Rule Targets

- `Freestyle-15/20`: 5 or more in a row wins.
- `Standard-15`: exactly 5 wins, overline does not win.
- `Renju-15`: international renju tournament basis with Gomocup tournament
  constraints; current implementation focus is forbidden-move and exact-five
  behavior plus 200-move auto-draw.
- `Caro-15`: standard caro where only open exact-five wins; blocked-both-ends
  five, six, and longer lines do not win.
- `Omok-KR`: preserve current Korean omok semantics already encoded in repo.

## WBS

### WP1. Variant Taxonomy

- Introduce an explicit rule enum that separates:
  - `Freestyle`
  - `Standard`
  - `Omok`
  - `Renju`
  - `Caro`
- Rebind constructors and naming so the code no longer uses `Standard` for
  freestyle semantics.
- Preserve current external compatibility by keeping `gomoku15` mapped to
  freestyle unless a more explicit route is requested.

### WP2. Rule Semantics

- Implement win detection per variant:
  - `Freestyle`: `>=5`
  - `Standard`: `==5`
  - `Omok`: existing KR semantics
  - `Renju`: existing renju semantics + 200-move auto-draw
  - `Caro`: only open exact-five wins
- Keep forbidden-move logic restricted to `Omok` and `Renju`.
- Do not let Gomocup `Standard` inherit KR omok or freestyle behavior.

### WP3. Server Routing

- Add explicit Gomocup-facing aliases:
  - `gomoku15_free`
  - `gomoku15_std`
  - `gomoku15_renju`
  - `gomoku15_caro`
  - `gomoku15_omok`
- Keep legacy `gomoku15` alive as freestyle-preserving alias.
- Ensure search and self-play paths route through the same variant parser.

### WP4. Regression Tests

- Add rule-level tests before implementation for:
  - freestyle overline wins
  - standard overline does not win
  - caro blocked five does not win
  - caro open five wins
  - caro exact six does not win
  - omok preserved
  - renju preserved
- Add server parsing tests for variant aliases.

### WP5. Documentation Sync

- Update docs or code comments where `Standard` currently means freestyle.
- Mark `Omok` explicitly as Korean omok / non-Gomocup variant.

### WP6. Gomocup Protocol Adapter

- Add a dedicated Gomocup brain entrypoint on top of the current search stack.
- Support at minimum:
  - `START`
  - `RECTSTART`
  - `RESTART`
  - `BEGIN`
  - `TURN`
  - `BOARD ... DONE`
  - `YXBOARD ... DONE`
  - `TAKEBACK`
  - `INFO`
  - `ABOUT`
  - `END`
- Keep protocol parsing isolated and unit-testable.

### WP7. Rule Handshake And Time Control

- Interpret Gomocup/Piskvork-style `INFO rule` codes into internal variants.
- Enforce valid size/rule combinations:
  - freestyle on `15` or `20`
  - standard/caro/renju/omok only on `15`
- Add a deterministic time-budget policy using:
  - `INFO timeout_turn`
  - `INFO timeout_match`
  - `INFO time_left`
- Default to single-thread, no-ponder, manager-driven thinking.

### WP8. Opening And Board Ingestion

- Support prepared openings and arbitrary manager states via `BOARD`.
- Infer side-to-move from board counts for manager-provided positions.
- Keep Renju opening handling manager-facing rather than hardcoding swap logic.

### WP9. Verification

- Add protocol regression tests for:
  - rule-code mapping
  - size/rule rejection
  - command parsing
  - `BOARD` ingestion
  - move output format
  - time-budget computation
- Run focused Rust regressions plus existing Python regressions.

## Execution Order

1. Add failing tests for taxonomy and rule semantics.
2. Refactor enum and constructors.
3. Implement variant-specific win logic.
4. Wire server alias parsing.
5. Extend WBS for Gomocup protocol layer.
6. Add failing protocol/time-control tests.
7. Implement Gomocup brain adapter.
8. Run focused regressions.
9. Update docs/comments.

## Still Deferred After This Tranche

- full Windows packaging/submission automation
- GPU-backed standalone inference backend for Gomocup binary
- self-play opening-book generation beyond manager-provided openings
- full 20x20 training presets on the Python learner side
