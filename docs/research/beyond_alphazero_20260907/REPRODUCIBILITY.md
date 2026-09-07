# Execution, analysis and publication identity

The scientific source and the later documentation/publication are separate objects. Original production code was not changed.

| Object | Identity |
|---|---|
| Inspected main | `3e303aef0a0a534d1ccc8b72289a72b7fd0dc901` |
| Latest development base | `a74adb4e0176b0e7f0ad6cb7c1870a483b787749` |
| Base tree | `4ea3240170cd4e778a5a9219f9ed49b7fc21758c` |
| Frozen executed source | `253df550031d816249dac0a414c74596f27142b0` |
| Frozen analysis-source commit | `8c87120960917f1caa66d38cfebb5e4b6720d826` |
| Runtime | CPython 3.12.13, NumPy 2.3.5, SciPy 1.17.0 |
| Scientific pilot executions | 1; process exit 0; no timeout |
| Publication method | GitHub connector Git-data API, additive commit on development base; publication commit differs from local execution commits |

The [Git bundle](evidence/diagnostic_source.bundle) preserves both original execution and analysis commits and their parent relationship. It requires the development base above, already present in the original repository. `git bundle verify` succeeded. Bundle publication ensures local execution hashes remain reconstructable even though the connector creates a later publication commit with its own metadata. The remote publication tree is checked against the local final tree; the original execution tree is independently recorded in `evidence/pilot/summary.json`.

From a checkout of this published branch, the following read-only checks inspect original evidence:

```bash
git bundle verify docs/research/beyond_alphazero_20260907/evidence/diagnostic_source.bundle
python docs/research/beyond_alphazero_20260907/test_research_math.py
python docs/research/beyond_alphazero_20260907/test_research_math.py --probes
```

To replay the exact source, fetch the bundle into a new local ref and use an isolated worktree. Choose paths/ref names that do not already exist. These commands write only the new local ref/worktree/output; they do not resume an Idea Foundry campaign.

```bash
git fetch docs/research/beyond_alphazero_20260907/evidence/diagnostic_source.bundle HEAD:refs/heads/quartz-diagnostic-source-20260907
git worktree add /tmp/quartz-gaussian-replay-20260907 253df550031d816249dac0a414c74596f27142b0
cd /tmp/quartz-gaussian-replay-20260907
python docs/research/beyond_alphazero_20260907/gaussian_batch_pilot.py --output /tmp/quartz-gaussian-output-20260907 --worlds 4096
```

The runner refuses to overwrite output and checks tracked source cleanliness. It does not promise cross-version bitwise NumPy RNG/BLAS identity; pin the versions above for exact replay. The persisted raw file records 9 significant decimal digits. Summary aggregates use the unrounded float64 values; analysis allows rounding-level differences when reconstructing means.

`analyze_pilot.py` in the frozen analysis commit reads `evidence/pilot/` and writes a new `evidence/analysis/` directory, refusing an existing directory. For an independent analysis replay, use a fresh worktree at the analysis commit, copy the original `evidence/pilot/` from the publication checkout into that fresh worktree, and run the script there. It verifies the raw SHA, all 240 design aggregates and the logical budgets before drawing figures. Running it directly in an already-populated evidence directory intentionally fails rather than overwriting earlier analysis.

The full 240-design arrays share raw keys when counts and sufficient statistics are identical. There are 82 such designs, each with 4096 rows, totaling 335872 raw rows. This deduplication is valid because the two-finalist covariance allocation is outcome-independent. It would not automatically be valid for a later adaptive stopping/search experiment. The method × budget × width rows are not independent experimental replications.

All recorded outputs are diagnostic. This package contains no real GPU throughput measurement, AlphaZero gameplay, independent training run, or general MCTS posterior-coverage proof. Wolfram context/evaluator were attempted and failed before kernel evaluation with MCP 404. SciSpace and primary web literature were consulted. Missing local SymPy/pytest/Rust did not block the stdlib-unittest and SciPy calculations; no full repository test/build result is claimed.
