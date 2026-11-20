# COG-01B Resident Fork | observed cache-identity distinction

2026-11-20. Origin: bounded Rust+Python CPU smoke in the isolated research worktree. Source Rust telemetry was published in `82dedd303e7aee3e4fb3250efe4c43f9435eb391`.

## Direct observation

- Same actual root: the persistent Rust engine yielded `root_visits 8→16`, root hash `be89d07d0f7afbbe`, evaluator boundary calls `9→17`, and physical CPU PyTorch forward items `9→17`.
- In the initial fresh16 control with the default caching regime, Rust reported **17 evaluator-boundary calls**, but the *Python NN wrapper* counted **zero new physical forward items**: model counts `(9,8,0)` versus Rust boundary counts `(9,8,17)`. This persists after stopping/restarting just the Rust child server.
- Source traced to `quartz/eval_runtime.py` process-global `_NN_EVAL_CACHE`: `get_nn_eval_cache(cfg)` returns the same object across Python `NNSearchClient` session and Rust-server restarts unless explicitly cleared or `QUARTZ_DISABLE_NN_CACHE` is nonempty. Its `run_batched_eval_groups` reuses prior evaluation records without new PyTorch forward. Thus **Rust evaluator boundary invocation is not equal to new physical NN inference under cache-enabled conditions**.
- Smallest correction for the scientific CPU micro-experiment: require `QUARTZ_DISABLE_NN_CACHE=1`, and `QUARTZ_DISABLE_ASYNC_PIPELINE=1`, single Rust search thread, single source NN, no opt-in external controller. Compare Rust eval boundary and Python wrapper physical items independently. A separate Rust server is also restarted for each fresh16 control to avoid engine/session state reuse.
- After the correction, one actual CPU trained model on `P0003/seed17/C01_gen_5`: physical `NN_counts=[9,17,17]`, delta calls 8, same root hash, fresh16 matches resident16, smoke exit 0.

## Scientific interpretation

The original default cache behavior is not automatically a defect: caching exact repeated input is a valid performance optimization. However a *new computation* is not a new neural inference when it hits the cache, so accounting must distinguish physical inference calls, evaluator invocations, TT hits, and root visits. This exercise tests **uncached physical NN item costs**, not production cached throughput. Whether the global key includes a complete model/checkpoint epoch and can accidentally reuse outputs across different weight versions is a separate high-impact question; no actual cross-epoch contamination was established here, so do not assert one.

The earlier failed cache-enabled smoke is an **implementation-environment / accounting-contract failure**, not a mathematical failure and not a failed game-strength result. The corrected smoke is scope-limited to one root and does not prove the 72-trial campaign passes.
