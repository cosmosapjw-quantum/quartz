# Applying the QUARTZ Idea Foundry skeleton bundle

The archive mirrors repository-relative paths.  From the QUARTZ repository
root, unpack or copy the archive contents without deleting existing files.
All paths in this bundle are new.

```bash
unzip quartz_idea_foundry_skeleton.zip -d /path/to/quartz
cd /path/to/quartz
venv/bin/python -m pytest -q tests/test_idea_foundry_skeletons.py
```

The Rust `src/mcts/foundry/` directory is deliberately **not imported** from
`src/mcts/mod.rs`; therefore unpacking the bundle does not change or compile
production search behavior.  Read `docs/idea_foundry/00_INDEX_KO.md` and
`docs/idea_foundry/README.md` before wiring it.

Suggested Git workflow:

```bash
git checkout agent/local-experiment-foundry
git add docs/idea_foundry quartz/idea_foundry src/mcts/foundry \
        configs/idea_foundry.axes.v1.json tests/test_idea_foundry_skeletons.py
git commit -m "Add idea foundry experiment atlas and code skeletons"
```
