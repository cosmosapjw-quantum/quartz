# Gomocup Brain Mode

## Scope

This mode exposes QUARTZ as a Gomocup/Piskvork-style `pbrain` process on top of
the existing CPU search stack.

Current support covers:

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

It supports these rule variants:

- freestyle `15x15`
- freestyle `20x20`
- standard `15x15`
- renju `15x15`
- caro `15x15`
- preserved Korean omok `15x15`

## Invocation

Two invocation paths are supported:

```bash
cargo run --release -- --gomocup
```

or by running a release binary whose executable name starts with `pbrain`, for
example:

```bash
cp target/release/mcts_demo target/release/pbrain-quartz
./target/release/pbrain-quartz
```

This keeps local development simple while allowing tournament-style manager
launch behavior.

There is also a helper script:

```bash
./scripts/build_gomocup_brain.sh
./target/release/pbrain-quartz
```

## `INFO` Handling

The brain currently parses and stores:

- `INFO rule`
- `INFO timeout_turn`
- `INFO timeout_match`
- `INFO time_left`
- `INFO time_increment`
- `INFO max_memory`
- `INFO thread_num`
- `INFO folder`

Time budgeting is deterministic and based on manager-provided time controls.
`thread_num` and `folder` are accepted for compatibility, but the current brain
still runs a single-process CPU search path.

## Rule Codes

Implemented rule-code mapping:

- `0` -> freestyle
- `1` -> standard
- `4` -> renju
- `8` / `9` -> caro
- `104` -> preserved Korean omok (internal extension, not Gomocup standard)

## Notes

- `BOARD` ingests manager-provided positions in `x,y,player` form.
- `YXBOARD` is accepted as a compatibility extension and swaps incoming
  coordinates during ingestion.
- Renju keeps the current repo's forbidden-move semantics plus the Gomocup
  `200`-move auto-draw limit.
- The current Gomocup brain is CPU-search only. A dedicated NN backend for
  tournament deployment remains deferred.
