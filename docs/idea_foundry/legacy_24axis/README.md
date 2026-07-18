# Deprecated 24-axis snapshot

This directory preserves the remote 24-axis Idea Foundry catalog merged from
commit `435e8ac`. It is historical design evidence, not the current source of
truth.

- Current registry: `configs/idea_foundry.axes.v1.json` (A01--A26)
- Current package: `quartz/idea_foundry/`
- Legacy Python package: `quartz/idea_foundry/legacy_24axis/`
- Legacy registry: `configs/legacy/idea_foundry.axes.24axis.v1.json`
- Legacy Rust sketches: `src/mcts/foundry/legacy_24axis/` (not compiled)

The snapshot remains `DEPRECATED`. Its tests prove only that the historical
mechanism skeleton still imports and runs; they do not establish current-axis
coverage, efficacy, play strength, or promotion readiness.
