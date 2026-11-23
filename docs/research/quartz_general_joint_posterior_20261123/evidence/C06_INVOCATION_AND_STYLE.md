# C06 invocation correction and scoped style changes

The first C06 exact-oracle command omitted its required output-path argument. It terminated with IndexError at the final result-write statement, exit1. This is an invocation failure, not a mathematical assertion failure. The original log and return code are retained. C06_EXACT_ORACLE_FINAL supplies a new output path and completes all53851 assertions with exit0.

Ruff formatting first preserved Python AST identity (C06_FORMATTING_EQUIVALENCE.json). Subsequent scoped lint repairs sorted standard-library imports, removed nonexecuted shebangs, used functools.cache in place of the equivalent unbounded lru_cache, made the publisher's existing unchecked subprocess result explicit and combined resource contexts/UTC alias. The final source is independently rerun; the AST-only receipt covers the formatting operation, not these later lint adapters. The native scientific source, native executable, model-selection harness and test file retain their C05 identities. No campaign output was used to choose these changes.

The user's cleanup instruction amended the delivery assumptions in IMPLEMENTATION_PLAN before the source freeze. The physical/mathematical experiment contract is unchanged.
