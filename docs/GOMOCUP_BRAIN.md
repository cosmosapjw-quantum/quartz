# Gomocup Brain Mode

## Scope

QUARTZ can run as a Gomocup/Piskvork-style `pbrain` process.

Current implementation supports:

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

Supported rule/board combinations:

- freestyle `15x15`
- freestyle `20x20`
- standard `15x15`
- renju `15x15`
- caro `15x15`
- preserved Korean omok `15x15`

## Runtime modes

The brain now has two practical deployment modes.

### 1. Bundle-backed Gomocup mode

Recommended for tournament use.

The binary loads a bundle directory containing:

- `gomocup_manifest.json`
- `gomocup_model.onnx`
- optional `champion.pt`
- optional `pbrain-quartz` copied in by the build helper

The bundle manifest controls:

- deploy game / rule target
- search profile (`quartz`, `baseline`, `baseline_strict`)
- virtual-loss mode
- TT on/off
- `c_puct`
- `sigma_0`
- `min_visits`
- `check_interval`
- optional `budget_ms`
- optional `max_visits`
- ABOUT metadata
- source model path, condition/seed, and selection/training metrics

When built with `--features onnx`, the brain uses the exported ONNX model for
Gomoku 15×15 searches. If the bundle is missing or cannot be loaded, it falls
back to the built-in rollout evaluator.

### 2. CPU fallback mode

If no valid bundle is found, the brain still runs with the internal search
stack and `ShortRollout` evaluator. This is useful for protocol testing but is
not the intended tournament deployment path.

## Bundle discovery

The binary searches for a Gomocup bundle in this order:

1. `QUARTZ_GOMOCUP_BUNDLE_DIR`
2. `INFO folder <path>` from the Gomocup manager
3. the binary directory
4. the current working directory

## Invocation

Two entry paths are supported:

```bash
cargo run --release -- --gomocup
```

or by running a binary whose executable name starts with `pbrain`, for example:

```bash
cp target/release/mcts_demo target/release/pbrain-quartz
./target/release/pbrain-quartz
```

For Gomocup deployment, use the helper script instead:

```bash
scripts/build_gomocup_brain.sh \
  --bundle-dir results/ablation/gomoku15/gomocup_bundle \
  --target-name pbrain-quartz
```

That script builds the Rust binary with `--features onnx` by default and copies
the resulting executable into the bundle directory.

## Tournament folder layout

A practical deployable bundle directory looks like:

```text
gomocup_bundle/
├── gomocup_manifest.json
├── gomocup_model.onnx
├── champion.pt
└── pbrain-quartz
```

You can run the binary from inside that directory, point the manager at the
directory with `INFO folder`, or set `QUARTZ_GOMOCUP_BUNDLE_DIR` explicitly.

## Export flow from training/ablation

1. Run the ablation study.
2. Let the runner choose a champion.
3. Export the champion bundle.
4. Build the Gomocup binary against that bundle.

```bash
venv/bin/python scripts/ablation_study.py \
  --study search_vl \
  --game gomoku15 \
  --iterations 30 \
  --eval-games 80

venv/bin/python scripts/ablation_study.py \
  --report results/ablation/gomoku15 \
  --prepare-gomocup

scripts/build_gomocup_brain.sh \
  --bundle-dir results/ablation/gomoku15/gomocup_bundle \
  --target-name pbrain-quartz
```

## `INFO` handling

The brain parses:

- `INFO rule`
- `INFO timeout_turn`
- `INFO timeout_match`
- `INFO time_left`
- `INFO time_increment`
- `INFO max_memory`
- `INFO thread_num`
- `INFO folder`

Notes:

- `INFO folder` is now meaningful: it can point to the Gomocup bundle directory.
- `thread_num` is accepted for manager compatibility, but the current Gomocup
  brain still uses one local search engine instance rather than a manager-driven
  multi-process topology.
- If the bundle manifest sets `budget_ms`, the brain clamps manager-derived
  time allocation by that budget.
- The selected search profile and search hyperparameters come from
  `champion.json` via the bundle manifest, so tournament deployment stays aligned
  with the ablation winner rather than ad-hoc CLI overrides.

## Rule codes

Implemented rule-code mapping:

- `0` -> freestyle
- `1` -> standard
- `4` -> renju
- `8` / `9` -> caro
- `104` -> preserved Korean omok

## Notes

- `BOARD` consumes `x,y,player`.
- `YXBOARD` is accepted as a compatibility extension and swaps incoming coordinates.
- Renju preserves forbidden-move behavior and the `200`-move auto-draw limit.
- The ONNX bundle should match the intended Gomocup rule variant. If the bundle
  game and manager rule do not match, the binary falls back to the internal evaluator path.
