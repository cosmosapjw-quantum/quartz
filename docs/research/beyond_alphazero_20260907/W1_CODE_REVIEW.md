# Independent W1 code review — 2026-09-15

Scope: the standalone, single-owner Python observation reducer, its focused
tests, the exact Gaussian diagnostic, the W1 contract, and the original W1
plan. This review did not run the engine, a training loop, the September 7 pilot,
or a repository-wide test suite. It did not modify the implementation.

## Finding and disposition

**Resolved: stochastic block aliases could acquire contradictory RNG meanings.**
Initially, `BeliefStore.complete` checked canonical duplicates before checking
the declared independent-block binding, and recorded bindings only for newly
accepted evidence. A direct probe reproduced this sequence:

| RNG / declared block | Initial behavior | Required behavior |
| --- | --- | --- |
| R1 / B1 | accepted | accepted |
| R1 / B2 | duplicate | duplicate, retaining B2's binding |
| R2 / B2 | accepted | rejected |

The reverse conflict order also mattered: after accepting R1/B1 and R2/B2,
R1/B2 was incorrectly classified as a harmless duplicate. The patch checks an
existing declared-block binding before canonical duplicate handling and records
aliases for both accepted and duplicate receipts. The regression covers both
arrival orders. This enforces metadata consistency; it does not prove that
different RNG blocks are statistically independent.

The implementation owner also fixed invalidation followed by an identity
roundtrip. `invalidate` now remembers pending job IDs as invalidated, so returning
to an old identity does not make its old pending results current again. I read
the patch and independently ran the two focused regressions for the alias and
roundtrip cases: **2 tests passed, exit 0**. The publication's committed-source
verification is recorded separately by the research owner.

## Scientific checks and limits

- Canonical deduplication includes leaf, evaluator, model, representation, target
  epoch, and observation kind; stochastic identity also includes generating
  policy and RNG block. Delivery IDs and paths do not generate new evidence.
  The caller must encode the complete evaluator input in `leaf_hash`, including
  player/value perspective and relevant history, and give epochs their declared
  meaning. This reducer cannot discover an incorrectly labelled target.
- Completed block counts, attempted enqueues, accepted visible blocks, and
  cancellation/failure receipts have different documented meanings. None is a
  physical-compute measurement or effective sample size. A late result can
  contribute to both a terminal-status count and the completed-receipt count.
- Chan/Welford merging is appropriate for the represented numerical moments.
  Aggregate moments can span different latent targets and are bookkeeping only.
  W2 must group by target and observation operator before forming a likelihood.
  Opaque RNG IDs do not detect overlapping streams or within-block dependence.
- Invalidation clears visibility while retaining receipts; previously recorded
  deterministic evidence is not automatically reintroduced at a new root.
  Evidence transport and reinterpretation require the later explicit interface.
- For the Gaussian diagnostic, with prior variance 1, observation-mean noise
  variance R, and iid-assumed weight w = n/(n+1), the correct variance R/(1+R),
  actual mean-square error (1-w)^2 + w^2 R, and normal interval coverage used in
  the code are consistent. The coverage is prior-predictive at fixed n. It is
  not conditional frequentist coverage or an adaptive-stopping guarantee.

No further blocking defect was found within this bounded review. This is a
source audit and narrow regression check, not a concurrency, production
integration, learned-calibration, or high-budget performance certificate.

Reviewed source SHA-256:

```text
observation_contract.py
252274721bebfd3ce4c3c59756bbf4be1bbda5671a1a89c0ae29a10c13ccbbf5
duplicate_evidence_diagnostic.py
a64076d856e268fe704ec7cdd032c05e51b81a68d4cb162895904fd2936ca894
```
