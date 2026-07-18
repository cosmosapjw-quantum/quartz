# QUARTZ Idea Foundry skeleton import status

> **Already applied — do not reapply the ZIP or patch.**

The 21 repository-relative payload files from
`quartz_idea_foundry_skeleton.zip` were applied in commit
`7f332d60f7152548717a671cbda690697e06a040`. Applying either the archive or
`quartz_idea_foundry_skeleton.patch` again can overwrite the subsequent local
campaign-runner work.

The immutable import hashes, baseline commit, and payload-count check are in
`docs/idea_foundry/IMPORT_RECEIPT.json`. `BUNDLE_FILE_LIST.txt` lists payload
files only; archive directory entries are intentionally omitted.

The Rust `src/mcts/foundry/` implementation is gated behind the Cargo
`idea-foundry` feature. It must remain absent from the default production build
until its feature-gated contract tests and the applicable experiment gate pass.

Before working on an experiment, read:

1. `docs/idea_foundry/00_INDEX_KO.md`;
2. `docs/idea_foundry/README.md`;
3. `docs/LOCAL_EXPERIMENT_LAB.md`;
4. `configs/idea_lab.local.v2.json`.

The original ZIP and patch are provenance inputs. Do not delete, regenerate, or
overwrite them.
