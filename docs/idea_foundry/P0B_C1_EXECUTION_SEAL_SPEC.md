# P0b-C1 execution-seal boundary

## Authority and observable behavior
- Authoritative base: `292d0336b246f990cb0fc7f99ef11f464d506219`.
- This specification supersedes the rejected `20548e1` candidate, the frozen PX
  draft, and ignored preflight notes for design authority only. Preserve all of
  them as forensic material; do not delete, edit, or treat them as evidence.
- One falsifiable behavior: `run` must seal a clean live workspace, with exact
  40-source/2-input/1-binary membership, before writing campaign state or an
  attempt; `resume` must verify that seal and current bytes before any mutation.

## Components and contracts

- Add `quartz/idea_foundry/execution_seal.py`. It owns frozen value models,
  canonical payload bytes, a live-workspace adapter, validation, hashing, and
  atomic publication. It must not own campaign orchestration.
- The seal builder depends on `axis_workflow.load_json_strict()` and
  `load_workflow_specs()`. Registry axis membership/order and the seal's exact
  campaign and terminal-attempt artifact lists are sealed and revalidated.
- The live adapter records a clean committed source commit/tree and observes Git
  with a sanitized environment and replacement objects disabled. Git failure,
  dirty tracked or untracked state, ambiguous identity, or drift fails closed.
- Source and input inventories contain canonical repo-relative paths, byte size,
  and SHA-256; reject absolute/escaping paths, duplicates, symlink components,
  non-regular files, membership drift, and byte drift. The binary inventory binds
  the absolute interpreter invocation, its resolved regular target, size, and hash.
- The canonical JSON encoding is deterministic UTF-8 with sorted keys, compact
  separators, one trailing newline, strict exact JSON types, duplicate-key and
  non-finite-number rejection, and no acceptance through coercion.
- Publish `campaign_execution_seal.json` without overwrite: write and fsync a
  same-directory temporary file, hard-link it to the absent final name, then fsync the directory.
  Any collision or durability failure is an error; preserve the existing file.
- `sequential.py` only calls capture/publish/load/verify operations. A new schema-v2
  state binds the SHA-256 of the exact canonical seal bytes. It may be created only
  after durable seal publication, and no attempt path may exist before that point.
- Resume loads the existing seal, verifies its canonical bytes, state binding,
  registry, Git identity, and live inventory before changing state, timestamps,
  logs, summaries, or attempts. Legacy schema-v1 remains inspection-only and may
  not be resumed, migrated in place, or resealed.
- Resume of a terminal campaign is explicitly rejected until P0b-C2 implements
  terminal closure and final state/summary agreement.

## Threat model and non-goals

- In scope: serialized JSON, filesystem alias/symlink/byte drift, Git
  environment/replace-ref/identity drift, inventory drift, and publication races.
- Out of scope: hostile Python subclasses, lying in-process equality, and module
  monkeypatching; these are not serialized trust-boundary inputs.
- No receipt-verifier refactor, declared-Git-tree adapter, analyzer/meta-analysis,
  terminal release closure, historical artifact rewrite, result or receipt output,
  scientific run/resume, promotion, push, PR, or merge belongs in this change.

## Scope and workflow gates

- Maximum six changed files. Target total churn is at most 560 lines; the absolute
  limit is 600. Stop without committing if the range would exceed that limit.
- Total stack is four commits relative to `d0f0fb2`: the two existing strict-loader
  commits, this specification commit, and one implementation commit.
- Use recorded RED then GREEN TDD. Review the immutable implementation SHA with
  independent code/statistics and provenance/claim reviewers; allow at most three
  fix/review loops. Fixes amend that commit; every changed SHA invalidates prior approvals.

## Acceptance matrix

| Boundary | Required proof |
| --- | --- |
| New run | Seal is exact 40/2/1, clean, canonical, durable, and precedes state/attempt bytes. |
| Resume | Malformed, mismatched, stale, aliased, symlinked, replaced-Git, or byte-drifted input fails before mutation. |
| Compatibility | Schema-v1 is readable by status inspection only; terminal resume is blocked. |
| Scope | No results/receipts; six files or fewer; churn `<=600`; four commits total. |
| Claim | `execution_status=success`, `contract_status=passed` only after gates, `effect_status=non_estimable`, `evidence_maturity=contract_only`, `promotion_status=ineligible`. |
