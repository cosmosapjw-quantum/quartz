# P0b-C1 prospective execution-seal specification

## 1. Authority, outcome, and non-goals

- Authoritative implementation base: `292d0336b246f990cb0fc7f99ef11f464d506219` plus this corrected specification commit.
- The only allowed history is:

  ```text
  292d033
    -> corrected P0b-C1 specification commit
      -> one P0b-C1 implementation commit
  ```

- This specification is the sole implementation authority. `914169e` and `0bcbdbf` remain archival checkpoints; `10c34c1` and `20548e1` remain rejected candidates. None may be salvaged, amended, or used as an implementation base.
- One falsifiable behavior: a new sequential campaign acquires an unused run root, seals one coherent clean live workspace, durably publishes the seal, proves the workspace is unchanged, and binds schema-v2 campaign state before any attempt mutation; resume verifies the same boundary before any mutation.
- This is prospective contract infrastructure only. It does not establish scientific efficacy, provenance completeness, execution readiness, receipt validity, or promotion eligibility.
- P0b-C2 terminal closure, analyzers, meta-analysis/statistics, scientific execution, results, receipts, claim promotion, README/archive maintenance, push, PR, and merge are out of scope.

## 2. Frozen implementation surface and process gate

The implementation commit may change only:

```text
quartz/idea_foundry/execution_seal.py
quartz/idea_foundry/axis_workflow.py
quartz/idea_foundry/sequential.py
tests/test_idea_foundry_execution_seal.py
tests/test_idea_foundry_axis_workflows.py
```

Production churn is measured only from the corrected specification SHA:

```bash
git diff --numstat <CORRECTED_SPEC_SHA>...HEAD -- \
  quartz/idea_foundry/axis_workflow.py \
  quartz/idea_foundry/execution_seal.py \
  quartz/idea_foundry/sequential.py
```

Let `M` be the sum of additions and deletions printed by that command. The hard gate is `M <= 360`. Necessary acceptance-matrix tests have no line-count cap. Aggregate churn is telemetry only and cannot fail or split the task.

There is one implementation commit; an authorized repair amends it. One fresh independent adversarial review is allowed, followed by at most one repair-closeout and recheck. A remaining P0/P1 or acceptance-related P2 after that recheck yields `BLOCKED_REVIEW_LIMIT`. Re-slicing, budget recalibration, a second repair, and parallel writers are forbidden.

## 3. Immutable membership contract

List order and membership are binding. Runtime discovery, sorting, substitution, and coordinated constant drift are forbidden. The implementation must compare exact built-in strings and containers against independent immutable literals.

```python
SEALED_SOURCE_PATHS_V1 = (
    "scripts/idea_foundry/a03_uncertainty_decomposition.py",
    "scripts/idea_foundry/a09_h3_change_point_router.py",
    "scripts/idea_foundry/a01_calibrated_stop_council.py",
    "scripts/idea_foundry/a02_static_anchor_rpo.py",
    "scripts/idea_foundry/a17_b13_curvature_readout.py",
    "scripts/idea_foundry/a21_coherence_signed_path_shadow.py",
    "scripts/idea_foundry/a22_physics_falsification_dashboard.py",
    "scripts/idea_foundry/a06_gumbel_sequential_halving.py",
    "scripts/idea_foundry/a07_residual_evidence_widening.py",
    "scripts/idea_foundry/a12_jsd_locally_balanced_sampler.py",
    "scripts/idea_foundry/a25_ments_soft_backup.py",
    "scripts/idea_foundry/a26_nested_contour_exact_lab.py",
    "scripts/idea_foundry/a05_counterfactual_meta_teacher.py",
    "scripts/idea_foundry/a04_kg_voc_allocator.py",
    "scripts/idea_foundry/a08_tactical_proof_backend.py",
    "scripts/idea_foundry/a13_pending_flow_wu_uct.py",
    "scripts/idea_foundry/a15_service_curve_scheduler.py",
    "scripts/idea_foundry/a14_semantic_path_lsh.py",
    "scripts/idea_foundry/a16_monte_carlo_graph_sharing.py",
    "scripts/idea_foundry/a11_dynamic_live_set_particles.py",
    "scripts/idea_foundry/a18_diffusion_regularized_evaluator.py",
    "scripts/idea_foundry/a19_rw_rest_lite_evaluator.py",
    "scripts/idea_foundry/a23_cpu_incremental_pattern_student.py",
    "scripts/idea_foundry/a20_regret_state_archive.py",
    "scripts/idea_foundry/a24_learned_budget_gate.py",
    "scripts/idea_foundry/a10_prior_refresh_specialist.py",
    "quartz/experiment_manifest.py",
    "quartz/idea_foundry/__init__.py",
    "quartz/idea_foundry/axis_workflow.py",
    "quartz/idea_foundry/contracts.py",
    "quartz/idea_foundry/control.py",
    "quartz/idea_foundry/execution_seal.py",
    "quartz/idea_foundry/gates.py",
    "quartz/idea_foundry/learning.py",
    "quartz/idea_foundry/search.py",
    "quartz/idea_foundry/sequential.py",
    "quartz/idea_foundry/serialization.py",
    "quartz/idea_foundry/status_schema.py",
    "scripts/idea_foundry_axis_gate.py",
    "scripts/idea_foundry_run_all.py",
)
assert len(SEALED_SOURCE_PATHS_V1) == len(set(SEALED_SOURCE_PATHS_V1)) == 40

SEALED_INPUT_PATHS_V1 = (
    "configs/idea_foundry.axes.v1.json",
    "configs/idea_lab.local.v2.json",
)

CAMPAIGN_ARTIFACTS_V1 = (
    "campaign_execution_seal.json",
    "campaign_state.json",
    "campaign_summary.json",
)

TERMINAL_ATTEMPT_ARTIFACTS_V1 = (
    "run_manifest.json",
    "rows.jsonl",
    "summary.json",
    "analysis/analysis_manifest.json",
    "analysis/analysis.json",
    "analysis/analysis_rows.jsonl",
    "analysis/diagnostic.png",
)

BINARY_DESCRIPTOR_KEYS_V1 = (
    "invocation_path",
    "resolved_target_path",
    "size_bytes",
    "sha256",
)
```

The source list deliberately preserves the live registry order for its first 26 members, followed by the 14 fixed support files in the displayed order. `execution_seal.py` is prospective and may be absent at the base SHA; it must exist in the implementation commit. Each source and input record has the exact keys `path`, `size_bytes`, and `sha256`, and appears in the corresponding literal order.

There is exactly one binary record with the exact keys above. `invocation_path` is the absolute configured invocation actually used for `sys.executable`; `resolved_target_path` is its absolute, fully resolved, stable regular-file target. The two paths, target size, and target bytes are bound. Direct-file substitution, relative invocation, a changed symlink target, a non-regular target, and any alias ambiguity fail closed.

## 4. Canonical seal and exact types

The seal is schema v2 and has exact top-level keys:

```text
schema_version, kind, suite, claim_scope, run_id, seed, git,
axis_order, sources, inputs, binaries, campaign_artifacts,
terminal_attempt_artifacts
```

Its fixed literals are:

```text
schema_version = 2                    # exact built-in int, not bool or float
kind = idea_foundry_campaign_execution_seal
suite = first-gate-all-sequential
claim_scope = synthetic_contract_execution_only
```

`run_id` is a non-empty safe built-in string and `seed` an exact built-in integer, not `bool`. `git` has only built-in-string `commit` and `tree` object IDs. `axis_order` is the exact 26 axis IDs implied by the first 26 source members in that same order. Inventories and artifact lists exactly match Section 3.

Every trust-boundary container and scalar must be an exact JSON built-in type. Missing or extra keys, subclasses, coercion, duplicate members, duplicate JSON object names, cycles, non-finite values including exponent overflow such as `1e400`, malformed UTF-8, non-canonical repository paths, duplicate paths, reordered list members, invalid hashes, and non-canonical bytes fail closed.

Canonical bytes are UTF-8 JSON with sorted object keys, compact separators, no ASCII coercion, and exactly one trailing newline. Loading must use the shared strict loader. Re-encoding must equal the persisted bytes before a digest or state binding is accepted.

## 5. Git and live-file capture

A capture `S` is one canonical seal value computed from one live repository observation:

1. Resolve and `lstat` the repository root and every path component without following symlinks. Reject symlinks, hard-link aliases where uniqueness is required, missing paths, FIFOs/devices/sockets/directories where a regular file is required, and path or stat changes during reading.
2. Invoke Git with a fixed executable/argument vector, a sanitized environment, global/system configuration disabled, and replacement objects disabled. Ambient `GIT_DIR`, `GIT_WORK_TREE`, object-directory, index, namespace, replace-ref, and config redirects must not influence the observation.
3. Resolve one literal `HEAD^{commit}` and its literal tree. Reject malformed identity, a moving HEAD/tree, replace refs, Git errors, and every tracked or non-ignored untracked change reported by Git.
4. Explicitly inspect index flags so `assume-unchanged` and `skip-worktree` cannot hide drift.
5. For every literal source and input, require a regular blob at the exact path in the captured commit tree and require the stable live bytes to equal the literal blob bytes. A commit-absent path, mode/type substitution, live byte drift, object substitution, or path alias fails closed.
6. Read and bind the stable resolved interpreter target independently; it is not asserted to be a Git-tree blob.

The implementation may use repeated `lstat`/size/inode/mtime checks plus streamed hashing, but it must prove stable no-follow reads rather than trust a single path lookup. Capture must either return one internally coherent canonical `S` or fail without publishing it.

## 6. Run-root ownership, durability, and temporal coherence

The only allowed new-run state transition is:

```text
ABSENT -> CLAIMED_UNSEALED -> SEALED -> STATE_CREATED
```

- `ABSENT -> CLAIMED_UNSEALED` uses exclusive creation of the final run-root directory. An existing empty directory, non-empty directory, file, symlink, or concurrent creator loses and fails closed. Parent creation, if allowed by the CLI contract, completes before this exclusive claim and cannot make an existing run root reusable.
- `CLAIMED_UNSEALED -> SEALED` writes canonical bytes to a same-directory temporary regular file, flushes and file-`fsync`s them, publishes the absent final seal name through a no-overwrite same-filesystem hard link or an equivalently exclusive primitive, then `fsync`s the run-root directory.
- Final or ancestor symlinks, a run root reached through a different filesystem identity, seal aliases or a final seal link count other than one after temporary-file removal, collisions, non-regular targets, and failure at any durability stage fail closed.
- `SEALED -> STATE_CREATED` occurs only after the post-publication equality gate below and writes state binding the SHA-256 of the exact canonical persisted seal bytes. No `axes/`, log, archive, summary, attempt metadata, attempt output, or mutable timestamp may exist before state creation.

New-run temporal order is exact:

```text
exclusive root claim
  -> S_pre = capture(live workspace)
  -> durable publish(canonical S_pre)
  -> S_post = capture(the same live workspace contract)
  -> require canonical S_post == canonical S_pre
  -> create schema-v2 state bound to sha256(persisted seal bytes)
  -> permit attempt mutation
```

If publication succeeds but post-capture or state creation fails, the root remains `SEALED` and the command fails. `CLAIMED_UNSEALED`, `SEALED`, temporary files, and any other crash residue are preserved as inspection-only evidence. P0b-C1 never automatically cleans, reuses, reseals, repairs, or resumes them. A repository lock is not required if capture and the exact pre/post equality gate satisfy this contract.

## 7. Status and campaign-state contract

The canonical successful campaign status is exactly:

```json
{
  "schema_version": 2,
  "execution": "success",
  "contract": "passed",
  "effect": "non_estimable",
  "evidence_maturity": "contract_only",
  "promotion": "ineligible"
}
```

The accepted keys are exactly `schema_version`, `execution`, `contract`, `effect`, `evidence_maturity`, and `promotion`, all with exact built-in types. Aliases such as `execution_status`, `contract_status`, `effect_status`, and `promotion_status`; missing/extra fields; `schema_version=true`; `schema_version=2.0`; and unknown enum values fail closed. Every row below has exact built-in `schema_version=2`; no other lattice-valid combination is writer-representable.

| Name | `execution` | `contract` | `effect` | `evidence_maturity` | `promotion` | Allowed writer |
| --- | --- | --- | --- | --- | --- | --- |
| `PLANNED` | `planned` | `not_evaluated` | `not_evaluated` | `contract_only` | `ineligible` | Axis only |
| `RUNNING` | `running` | `not_evaluated` | `not_evaluated` | `contract_only` | `ineligible` | Campaign or axis |
| `FAILED` | `failed` | `failed` | `invalidated` | `contract_only` | `ineligible` | Campaign or axis |
| `SUCCEEDED` | `success` | `passed` | `non_estimable` | `contract_only` | `ineligible` | Campaign or every axis except A10 |
| `SKIPPED` | `skipped` | `not_applicable` | `non_estimable` | `contract_only` | `ineligible` | A10 only |

These exact values are produced and validated through the shared status-schema-v2 code, never ad hoc strings or aliases. This is an implementation acceptance requirement; this specification does not claim the base validator already enforces every exact-type or writer-subset rule.

Schema-v2 campaign state has exact base keys:

```text
schema_version, run_id, suite, status, seed, created_at, updated_at,
execution_seal_sha256, claim_scope, axes
```

No other top-level key is accepted. `schema_version` is an exact built-in `int` equal to 2; identity, fixed literals, timestamps, status, and seal digest use exact built-in types and canonical shapes. The only top-level lifecycle combinations are:

| Campaign state | Status | `resumed_at` | `completed_at` |
| --- | --- | --- | --- |
| New or initial-invocation progress | `RUNNING` | Absent | Absent |
| Resumed progress | `RUNNING` | Required | Absent |
| Failed before any resume | `FAILED` | Absent | Absent |
| Failed after a resume | `FAILED` | Required | Absent |
| Successful initial invocation | `SUCCEEDED` | Absent | Required |
| Successful resumed invocation | `SUCCEEDED` | Required | Required |

Both lifecycle timestamps, when present, are exact built-in non-empty strings emitted by the shared UTC writer. A `PLANNED` or `SKIPPED` campaign status, `completed_at` on a non-success campaign, or any other timestamp/status combination fails closed.

`axes` is exactly 26 built-in dictionaries. Their `order_index` values are the exact integers `0..25`, bound in this suite order:

```text
A03, A09, A01, A02, A17, A21, A22, A06, A07, A12, A25, A26, A05,
A04, A08, A13, A15, A14, A16, A11, A18, A19, A23, A20, A24, A10
```

This is the same live registry order bound by the first 26 source paths; lexical `A01..A26` order is invalid. Each row's immutable base keys are exactly:

```text
order_index, axis_id, slug, lane_id, role, status, attempts
```

The values of `order_index`, `axis_id`, `slug`, `lane_id`, and `role` must exactly equal the corresponding current registry spec. Axis rows have only these writer-representable variants:

| Axis status | `attempts` | `current_attempt` | `resume_action` | axis `failure_reason` |
| --- | --- | --- | --- | --- |
| `PLANNED` | Empty | Absent | Absent | Absent |
| `RUNNING` | Non-empty; last attempt is `running` | Required; equals last `output_dir` | Absent | Absent |
| `SUCCEEDED` or A10 `SKIPPED` | Non-empty; last attempt is successful | Required; equals last `output_dir` | Absent, or exact `verified_skip` only on a terminal prefix when state has `resumed_at` | Absent |
| `FAILED` | Non-empty; last attempt is finished and unsuccessful | Required; equals last `output_dir` | Absent | Required non-empty built-in string |

Campaign/axis sequence is exact: `RUNNING` has a terminal prefix, optionally one immediately following `RUNNING` axis, then only `PLANNED` axes; `FAILED` has a terminal prefix, exactly one immediately following `FAILED` axis, then only `PLANNED` axes; `SUCCEEDED` has all axes terminal, with only A10 `SKIPPED`. A terminal prefix in a resumed state may carry `verified_skip`; no other row may. A hard-crash state containing a persisted running process is writer-readable but not resume-eligible.

Each attempt dictionary starts with the exact keys `attempt_number`, `started_at`, `output_dir`, `stdout`, `stderr`, and `process_outcome`. `attempt_number` is the exact one-based position in the list. Paths are exactly `axes/{axis_id}/attempt-{N:03d}`, `logs/{axis_id}.attempt-{N:03d}.stdout.log`, and `logs/{axis_id}.attempt-{N:03d}.stderr.log`. All are built-in strings and canonical contained relative paths.

| Attempt variant | Additional exact keys and relations |
| --- | --- |
| Active | `process_outcome="running"`; no `completed_at`, `returncode`, or attempt `failure_reason`; allowed only as the last attempt of a `RUNNING` axis |
| Process completed successfully | `process_outcome="completed"`, `completed_at` present, exact-int `returncode=0`, no attempt `failure_reason`; allowed only as the last attempt of a terminal axis |
| Process exited unsuccessfully | `process_outcome="completed"`, `completed_at` present, exact-int nonzero `returncode`, no attempt `failure_reason` |
| Launch failed | `process_outcome="failed"`, `completed_at` present, exact-int `returncode=126`, no attempt `failure_reason` |
| Analysis validation failed | `process_outcome="failed"`, `completed_at` present, exact-int `returncode=2`, required non-empty attempt `failure_reason` |
| Timed out | `process_outcome="timeout"`, `completed_at` present, exact-int `returncode=124`, no attempt `failure_reason` |
| Interrupted | `process_outcome="interrupted"`, `completed_at` present, exact-int `returncode=130`, no attempt `failure_reason` |

Every non-last attempt is a finished unsuccessful variant. The last attempt of a `FAILED` axis is a finished unsuccessful variant; a later retry appends a new active attempt rather than rewriting history. Unknown outcomes, `bool` return codes, outcome/returncode mismatches, missing or extra fields, nonconsecutive numbering, and path mismatch fail closed.

## 8. Resume gate

Resume performs all checks before any mutation:

1. Reject legacy schema v1, malformed/duplicate/non-canonical JSON, wrong top-level or nested types/keys, aliases, invalid lifecycle-field combinations, and invalid status-lattice combinations.
2. Reject a run root or seal with symlinked components, non-regular nodes, unexpected hard-link counts, canonical-byte mismatch, or digest mismatch.
3. Require exact run identity, seed, suite, claim scope, seal SHA, 40/2/1 membership and order, artifact tuples, 26 registry-bound axis rows, axis order, status payload, attempt structure, and contained paths.
4. Recapture the current live workspace using Section 5 and require its canonical value to equal the persisted seal exactly.
5. Reject a terminal campaign before mutation. Terminal resume remains blocked until P0b-C2 establishes final state/summary/artifact closure.

Only an exact `FAILED` campaign with no active attempt is resume-eligible. Validation and live recapture complete before one atomic state mutation changes the campaign to resumed `RUNNING`, appends the retry attempt, and marks only the already-terminal prefix `verified_skip`. `RUNNING` crash residue and `SUCCEEDED` campaigns are inspection-only. Every rejection must occur before changing or creating state, summary, attempt, log, archive, timestamp, or directory bytes. Tests use byte snapshots and mutation sentinels, not only exception messages, to prove this ordering.

## 9. Acceptance matrix

| Boundary | Must pass | Must fail before prohibited mutation |
| --- | --- | --- |
| Literal contract | Exact 40/2/1, artifact tuples, binary descriptor, registry bindings, canonical order | Missing, extra, duplicate, reordered, substituted, coordinated-constant, wrong-type, or coerced member |
| Canonical data | Exact built-in JSON values and canonical bytes | Duplicate key, malformed UTF-8, non-finite/exponent overflow, subclass/coercion, extra/missing key, non-canonical bytes |
| Git identity | Stable literal commit/tree and clean blob-equal live members | Moving HEAD/tree, replace/object/env redirect, Git failure, any tracked or non-ignored untracked change, index flags, commit-absent or non-blob member |
| Filesystem | Stable no-follow regular reads and one resolved interpreter target | Absolute/escaping/non-canonical repo path, missing file, byte/stat drift, symlink ancestor/final, hardlink alias, FIFO/device/socket/directory |
| Root claim | One process exclusively claims an absent root | Concurrent creator, pre-existing empty/non-empty root, file, symlink, collision, crash residue reuse |
| Publication | File fsync, exclusive publish, directory fsync, preserved canonical seal | Failure at any durability stage, overwrite, unexpected link count, cleanup/reseal/reuse after failure |
| Coherence | `S_pre == S_post` before state | Drift between capture, publication, recapture, or state creation |
| State/status | Exact schema-v2 keys/types, seal digest, 26 suite-ordered registry rows, writer-subset statuses, lifecycle table | Status aliases, `true`/`2.0`, unknown or lattice-valid-but-writer-invalid status, altered axis metadata/order, illegal lifecycle fields, stale seal |
| Resume | Exact failed state with no active attempt and current capture equal sealed capture | Legacy/malformed/aliased/stale/running/successful campaign or any rejection after a state/log/summary/attempt/archive/timestamp mutation |
| Scope | Five allowed implementation paths and `M <= 360` | Results, receipt, analyzer/meta, P0b-C2, science, promotion, archive rewrite, extra production surface |

Focused tests must cover every row with a positive case and table-driven negative cases, including concurrent new-run ownership, all crash states, pre/post source and Git drift, `assume-unchanged`, `skip-worktree`, hostile Git environment and replacements, source absent from the commit, hardlink/symlink/FIFO/non-regular aliases, exact membership/order, status aliases/type coercion, lattice-valid-but-writer-invalid tuples, illegal lifecycle/timestamp/axis-field combinations, unknown outcomes, returncode mismatches, resumed-terminal rejection, and fail-before-mutation byte sentinels.

## 10. Evidence and claim boundary

At this specification commit the status tuple is:

```text
execution=skipped
contract=not_applicable
effect=non_estimable
evidence_maturity=contract_only
promotion=ineligible
```

P0b-C1 execution seal and P0a receipt verification have distinct roles:

```text
P0b-C1 execution seal = prospective live execution identity
P0a receipt verifier   = declared execution/analysis Git-tree integrity
```

Neither alone, nor their conjunction before later closure, may be described as `provenance complete`. No scientific run, receipt, maturity increase, efficacy claim, promotion, or result rewrite is authorized by this specification.
