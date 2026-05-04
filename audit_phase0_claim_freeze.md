# Audit — BQ++ Phase 0: Claim freeze + math fixes

**Date:** 2026-05-04
**Scope:** docs-only patch. No code changes. Annotates every claim
location identified by the external review (`report.md`) with a
pointer to the correction. Audit-trail integrity preserved (no silent
rewrites of doc body).

## What changed (4 docs amended)

### `docs/LEGACY_VS_BAYESIAN_QUARTZ.md`

Appended **Appendix A — Superseded sections**. Lists §2.1 (config
count), §3.3 (EB gap), §3.4 (χ² formal-test), §3.5 (VOI sign), §3.6 +
§5.5 (KL-LUCB δ-PAC scope), §7.3 (concurrency monotonicity), §3.5 +
§8.2 (edge_pos vs action_id index bug) as superseded with explicit
corrections. Sections NOT superseded are explicitly enumerated so a
reader can see what survives. Body of the document is preserved
verbatim.

### `~/.claude/plans/iridescent-giggling-bachman.md`

Header note added: P09 onwards superseded by `bq_plus_plus_plan.md`.
P01-P08 listed as still in force, with the note that P06's internal
mutability migrates `Mutex<Cache>` → `ArcSwap<Arc<PolicyCache>>` in
BQ++ Phase 2.

### `README.md`

New "Controller v2 (BQ++)" section inserted before the existing
"Current Controller Status" header. One-paragraph statement of:
- BQ++ replacing the cancelled BayesianQuartz design.
- The single-principle objective (decision-loss reduction per cost).
- The five-module structure (calibrated belief, certificate, KG,
  reservoir, ArcSwap cache).
- The user's primary objective: ≥30% reduction in `nn_evals_per_move`
  at non-inferior play quality.
- Pointers to the audit response, comparison doc, and BQ++ plan.

### `docs/QUARTZ_THEORY.md`

Appended **§10 — Successor (BQ++)**. The existing §9 already disclosed
the heuristic-family scope honestly, so §10 is purely a forward-pointer
to BQ++. Explicit statement that BQ++'s "free-energy" framing is
log-sum-exp / KL-regularized planning (MENTS-family), **not** literal
quantum mechanics. "Path measure" replaces "path integral";
saddle-point / one-loop language removed from the hot path.

## Verification

### Repo-wide grep for the wrong configuration count

```
$ grep -rn "229,376\|229376\|229k" *.md docs/*.md
audit_external_review_response.md:27  → discusses the wrong number
audit_external_review_response.md:33  → discusses the wrong number
report.md:31, 38, 822                 → external audit input (the source identifying the error)
docs/LEGACY_VS_BAYESIAN_QUARTZ.md:86, 415, 1089  → original doc body (preserved per audit-trail)
docs/LEGACY_VS_BAYESIAN_QUARTZ.md:1125            → Appendix A (the correction)
```

Every occurrence in the body is in a document that now has an
explicit pointer to the correction (Appendix A in
LEGACY_VS_BAYESIAN_QUARTZ; audit_external_review_response.md is itself
the correction; report.md is the source).

### Documents have a single canonical truth source

The truth chain is:
1. `report.md` (external audit input) → identifies the errors.
2. `audit_external_review_response.md` (commit `2332aaf`) →
   acknowledges them with file/line pointers.
3. `docs/LEGACY_VS_BAYESIAN_QUARTZ.md` Appendix A (this commit) →
   explicit superseded-sections table.
4. `~/.claude/plans/bq_plus_plus_plan.md` → forward-looking plan.
5. `~/.claude/plans/iridescent-giggling-bachman.md` (header note) →
   preserved as audit trail with explicit "P09 onwards superseded"
   pointer.

A reader entering at any of points 1, 3, or 5 reaches points 2 and 4
within one navigation hop.

### IDE diagnostics

Two cosmetic markdown lint warnings (MD060 table-column-style in the
new Appendix A, MD032 blank-line-around-list in the new §10). Not
blocking. The original docs in this repo do not enforce these styles
(many existing tables omit the surrounding spaces); leaving as-is for
consistency.

## Files touched

- `docs/LEGACY_VS_BAYESIAN_QUARTZ.md` (+34 LOC appendix, body unchanged)
- `~/.claude/plans/iridescent-giggling-bachman.md` (+19 LOC header note)
- `README.md` (+22 LOC new section)
- `docs/QUARTZ_THEORY.md` (+27 LOC §10)

Net delta: **+102 / −0 LOC**, all docs.

## Adversarial review

The Phase 0 risk is silent doc rewriting — i.e. fixing the body of
`docs/LEGACY_VS_BAYESIAN_QUARTZ.md` so the wrong numbers disappear
from history. This was deliberately avoided: the doc body is preserved
verbatim and the corrections sit in Appendix A. A reader reviewing
the audit trail can see what was originally claimed, what the audit
identified, and what the correction is — all without consulting git
history.

The other Phase 0 risk is breaking existing internal links. Search
for every `LEGACY_VS_BAYESIAN_QUARTZ.md` link target:

```
$ grep -rn "LEGACY_VS_BAYESIAN_QUARTZ" *.md docs/*.md ~/.claude/plans/*.md
audit_external_review_response.md → relative path to docs/
README.md → existing reference
docs/QUARTZ_THEORY.md → new §10 (this patch)
```

All link targets resolve correctly with the docs in their current
locations.

## What unblocks next

Phase 1 (Python prototype + 41+ numerical primitive tests) can begin.
The primitives that need numerical verification before Rust port:
Welford + empirical-Bayes shrinkage (already-correct in P06's
`EdgeView::sigma_a`, but re-test here as regression), empirical
Bernstein width (R=1 vs R=2 scale), full E[max(X,0)] expected
improvement (NOT the wrong `s·φ(z)` overestimate from the audit's
§1.2), Knowledge Gradient approximation, KL-LUCB at the corrected
β = 15.618 (already verified in P06 commit `3370f95`), Gumbel
without-replacement sampling, Sequential Halving bracket arithmetic,
nested-reservoir live-set quantile pruning, prior-surprise χ² as
*statistic-only* (no p-value).

Estimated Phase 1 LOC: ~900 in `prototype/` (new top-level dir).
