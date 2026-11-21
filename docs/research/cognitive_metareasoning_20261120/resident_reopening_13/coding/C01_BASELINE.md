# C1. Executable baseline and physical claim gate

Authoritative theoretical R10 commit `14cec023721f45cac6825a0c6f364a228f36ec2f`. Current source branch is isolated and clean before scientific coding.

**Original baseline:** root candidates are `OnceLock` and fixed arena slab. Prior C12 used 8+16 separate fresh MCTS roots, exact 384 external-WDL, Rust debug/release 601 PASS, related Python 123 PASS. Not a resident same-root continuation.

**New executable acceptance:** opt-in engine scalar root action mask, full legal root candidates and priors immutable, first 8 root visits limited to original top3 with zero visits to omitted edges, monotone mask activation, `run(FixedIterations(24))` on SAME engine with 16 new root visits and unchanged prior/TT/hash/eval count at activation. CPU must account first full root edge materialization and any preceding ranking/model root evaluation.

**Scientific evaluation:** use NEW 24 five-ply family clusters/384 legal roots, exact Pascal Pons only posthoc, Uniform/MLP/Conv and restricted sham/random/full baselines. Main success is WDL regret improvement vs all-legal full24 **at <10% actual end-to-end mean CPU mismatch in each new group and 24-family CI excluding zero**. Negative result terminates the attempted benefit claim; do not mask null with all-legal competitor exclusion.

**Current actual input lock:** `INPUT_LOCK.json` above contains SHA256 for all 23 source files. No new test or exact labels yet. Distinguish compiled Rust code identity from executable binary identity.

**Verification:** SHA256 of source set and baseline branch; source-only no tests asserted beyond prior C12 verified log. C1 does not pretend to have implemented C5.
