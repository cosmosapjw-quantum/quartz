> **SUPERSEDED — 2026-08-23**
>
> This security-heavy specification is preserved as a forensic design record
> only.  It is not an implementation authority under the single-developer,
> trusted-local research policy in
> [`RESEARCH_TRUST_MODEL_20260823.md`](RESEARCH_TRUST_MODEL_20260823.md).
> P0b-C1 remains `NOT_ESTABLISHED`; implementation requires a new lean
> successor specification and must not silently weaken or partially implement
> the contract below.

# P0b-C1 successor execution-seal specification

## 1. Authority, status, and one bounded outcome

This document is the sole prospective implementation authority for P0b-C1. Its authoritative history
is exactly:

```text
292d0336b246f990cb0fc7f99ef11f464d506219
  -> this one successor specification commit
    -> one P0b-C1 implementation commit
       (the one authorized repair, if needed, amends that commit)
```

The corrected specification `83095c9742d779c3f2fc95a8021fa7bb4ab95338` is superseded. Failed
candidate/checkpoint `f825c38aa4d1c16146b425f1b768dfbd6ec7ad1a` and blocked specification candidates
`0de5115f2837894a8ce52b7d28d8489894c8b8a4` and `cf6191d45fba6c5575d7f630f563ed1a6e1083ec` are frozen
read-only forensic falsifiers. None is an implementation base, code source to salvage, or authority
to amend. Earlier `914169e`, `0bcbdbf`, `10c34c1`, and `20548e1` histories remain archival or
rejected only.

The single falsifiable outcome is:

> A new sequential campaign exclusively owns a canonical run root, binds one
> coherent clean Git/live workspace and interpreter into an immutable durable
> seal, proves the workspace and owned filesystem identities did not change,
> and publishes a schema-v2 state whose registry/provenance-dependent fields
> derive only from captured bytes before any attempt mutation; resume obtains
> the same single-writer ownership, validates
> the entire eligibility boundary into an immutable plan, and only then makes
> one valid state transition.

At this specification commit, the controlling process status is exactly:

```text
SPECIFICATION_ONLY
SPECIFIED
IMPLEMENTATION_NOT_PRESENT
P0B-C1_NOT_ESTABLISHED
RUN_RESUME_FORBIDDEN
SCIENCE_BLOCKED
```

P0b-C2 terminal closure, analyzers, meta-analysis/statistics, scientific execution, results,
receipts, reports, claim promotion, hardware-based scale increases, push, PR, and merge are out of
scope. Changed GPU or RAM capacity is capacity information only and does not relax this boundary.

Idea Foundry is an MCTS research program for 26 falsifiable axes and for preregistered combinations
whose component contracts are compatible. A combination study asks whether selected `A+B` improves
the common baseline and both `A`-only and `B`-only treatments. Positive or superadditive interaction
is interpretable only under a meaningful, prespecified endpoint, effect scale, treatment contrast,
independent unit, held-out target, and fixed-budget/runtime contract. The `M00/M01/M10/M11` cells
are a comparison structure, not evidence; `M11-M00` alone is not synergy. P0b-C1 may enable
trustworthy execution of individual-axis and compatible-combination tests, but validates no
hypothesis, estimand, interaction, efficacy, generalization, or promotion.

## 2. Supported host, threat model, and path input

P0b-C1 supports Linux/POSIX local filesystems only. Before any campaign or `results/` mutation,
static admission proves `O_NOFOLLOW`, `O_DIRECTORY`, and `O_CLOEXEC`; required entries in
`os.supports_dir_fd` and `os.supports_follow_symlinks`; dir-FD-relative open/mkdir/stat/link/unlink/
rename; file and directory fsync; and nonblocking exclusive `fcntl.flock`. There is no path-based
fallback.

For each retained no-follow directory FD in the repository-to-run-root target chain,
`TargetMountIdentity` is the exact tuple:

```text
(fdinfo mnt_id, exactly one matching raw /proc/self/mountinfo record,
 parsed nonnegative major/minor, fstat(fd).st_dev, parsed filesystem type)
```

`/proc/self/fdinfo/<fd>` must contain one parseable `mnt_id`; that numeric ID must select exactly
one mountinfo record. Missing, duplicate, changing, or inconsistent records fail closed. The
mountinfo major and minor fields are exact built-in nonnegative integers and must satisfy:

```text
os.makedev(mountinfo.major, mountinfo.minor) == fstat(fd).st_dev
```

Allowed target filesystems are exactly `ext2`, `ext3`, `ext4`, `xfs`, and `btrfs`. `tmpfs`, overlay,
NFS/NFS4, CIFS/SMB, FUSE, autofs, and unknown types are rejected.

The repository root establishes the target identity. Every existing or newly created `results`,
campaign-parent, and run-root component must have the same `mnt_id`, `st_dev`, filesystem type, and
mountinfo record; all mount crossing is forbidden. If `results` is absent, the deepest existing
retained ancestor (at minimum repository root) is the target anchor. If it exists, its retained FD
must already match, and the deepest existing campaign ancestor becomes the anchor. Missing
components are created only after the probe below and each is opened no-follow and immediately
proven to match.

After `WorkspaceLease` and before any non-probe campaign/results artifact mutation, an exclusive
randomly named disposable probe runs on exactly that retained target anchor mount. Through dir-FD
calls it performs exclusive mkdir/open/flock, complete short-write-safe write, file fsync,
no-overwrite link, rename, unlink, then probe-directory fsync. It proves the retained parent/child
identity, rmdirs that exact empty owned probe directory, fsyncs the target-anchor directory, proves
zero residue, and only then permits campaign-parent creation. Probe-dir fsync failure leaves that
one empty directory inspection-only without rmdir or parent fsync. Rmdir failure leaves only the
owned residue. Target-anchor fsync failure occurs after successful rmdir and therefore has zero live
residue but still fails admission. No failure creates campaign, run-root, seal, or state paths. This
is syscall admission, not a benchmark or power-loss proof; it emits no manifest, storage registry,
benchmark, result, or receipt.

Target mount identity is rechecked after every created campaign component and at run-root claim,
seal publication, state publication, and their directory fsyncs. Resume repeats it under both locks
immediately before mutation. Unsupported primitives, sparse checkouts, and nonlocal or unproven
lock/durability semantics fail before campaign mutation.

The protected concurrency model includes all conforming QUARTZ writers and ordinary rename/symlink
races. A malicious same-UID process that ignores the advisory lock, including one performing a
transient configuration/path swap, mount-namespace or bind-mount attackers, kernel compromise, and
physical-storage rollback are explicitly outside the contract.

The CLI retains the raw campaign-root argument as an exact built-in `str` before validation;
`argparse` must not convert it to `Path`. Only these two lexically canonical forms are accepted:

```text
<exact canonical absolute REPO_ROOT>/results/<one-or-more-components>
results/<one-or-more-components>
```

The relative form is interpreted from the retained repository root, never the process CWD. String
subclasses, bytes, empty values, backslashes, NUL, repeated slashes, `.` or `..` components, a
trailing slash, `~`, an absolute path outside `REPO_ROOT/results`, and the `results` directory
itself fail before mutation. `run_id` remains an exact built-in non-empty safe string matching
`[A-Za-z0-9][A-Za-z0-9._-]{0,95}` with no `..` substring.

Validation preserves the raw spelling. It never calls `resolve()` or `realpath()` before checking
lexical containment and component identity. Starting from retained repository/results directory FDs,
every existing component is traversed with no-follow directory opens; permitted missing
campaign-parent components are created one at a time, opened no-follow, and their parent directories
are fsynced. The final run root is never reusable.

Trusted `run` and `resume` admission accepts only that raw built-in string and never calls a path
compatibility adapter. The existing `resolve_run_root(Path)` name may remain only as a read-only
adapter for `status` and legacy preflight callers; it cannot return an ownership object or reach a
mutation API. The CLI default is the built-in canonical string `"results/idea_foundry_sequential"`,
not a `Path`.

## 3. Frozen implementation surface, budget, and review limit

The one implementation commit changes exactly these six paths and no others:

```text
quartz/idea_foundry/execution_seal.py
quartz/idea_foundry/sequential.py
quartz/idea_foundry/axis_workflow.py
quartz/idea_foundry/status_schema.py
tests/test_idea_foundry_execution_seal.py
tests/test_idea_foundry_axis_workflows.py
```

For a production file `p`, `M_p` is additions plus deletions in the frozen `RANGE` defined and
identity-checked in Section 12. `M` is their sum. The only measurement command is Section 12's
NUL-delimited, no-renames/no-ext-diff/no-textconv `git_bound diff --numstat` pipeline and exact
parser.

There is no production-churn minimum. Planning context, not acceptance credit, is: the compressed
incomplete `f825c38` production diff was `M=414`; readable formatting of the same incomplete
production logic is approximately `M=778`; formatting and consolidating `status_schema.py` alone
accounts for `M=106` before its missing exact semantics; and the missing P0 ownership/configuration/
identity seams require additional code. These values may explain a candidate but never require it to
spend lines. Every hard cap below and the total cap must pass:

| Production component | Hard `M_p` cap | Responsibility |
| --- | ---: | --- |
| `execution_seal.py` | 580 | Host preflight, owned dir-FD IO, Git/live capture, seal publication |
| `sequential.py` | 420 | Exact state, root lifecycle, immutable resume planning and admission |
| `axis_workflow.py` | 40 | Shared strict captured-byte registry/JSON bridge only |
| `status_schema.py` | 200 | Shared exact-type and writer-subset status implementation |
| **Total** | **1160** | All four production files |

The four file caps sum to 1240; the independently tighter total `M <= 1160` provides shared slack
rather than allowing every component maximum at once.

Tests have no line cap, but may implement only the frozen B01--B12 matrix in Section 11 and common
fixture helpers directly needed by it. The implementation has `P=0`: documentation, wrappers,
workflows, results, receipts, reports, and process artifacts changed by the implementation commit
are exactly zero.

The current `status_schema.py` formatting suppression must be removed. New `# fmt: off`, `# noqa`,
minification, semicolon or compound one-line statements, lambda/generator wrapper indirection used
to game line counts, duplicate status logic, and disabling Ruff rules are forbidden. Readability and
exact shared validators outrank churn telemetry. Exceeding a file or total cap yields `STOP_BUDGET`;
P0b-C1 remains `NOT_ESTABLISHED` and the work must split into newly authorized bounded successor
tasks. In-place waiver, cap raise, or scope rebalance is forbidden.

There is one fresh independent adversarial review wave on the immutable implementation SHA and at
most one repair-closeout. An authorized repair amends the implementation commit and invalidates the
prior review. A remaining P0/P1 or acceptance-related P2 after recheck yields
`BLOCKED_REVIEW_LIMIT`; there is no second repair.

## 4. Immutable 40/2/1 and artifact membership

Order and membership below are binding. Runtime discovery, sorting, substitution, and coordinated
constant drift are forbidden. Validation compares implementation values against independent
immutable literals.

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
if not (
    len(SEALED_SOURCE_PATHS_V1)
    == len(set(SEALED_SOURCE_PATHS_V1))
    == 40
):
    raise RuntimeError("SEALED_SOURCE_PATHS_V1 must contain 40 unique members")

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

The first 26 sources preserve live registry order; the next 14 are fixed support files in displayed
order. `execution_seal.py` is absent at base `292d033` and must exist in the implementation commit.
Each source/input record has the exact key set `{path, size_bytes, sha256}`; record-list order is
the literal order above, while canonical JSON object-key serialization is sorted.

There is exactly one binary descriptor. `invocation_path` is the absolute configured invocation
exactly equal to `sys.executable`; `resolved_target_path` is its absolute final stable regular-file
target. Resolution walks each symlink edge explicitly from retained directory FDs, rejects loops or
edge changes, and opens the final target no-follow. Both paths, target size, and target bytes are
bound. Relative invocation, direct-file substitution, target drift, non-regular target, or alias
ambiguity fails closed.

## 5. Retained ownership and capture objects

The implementation uses these semantic ownership objects (names are binding; field layout may remain
private):

- `TargetMountIdentity`: the exact fdinfo mount ID, unique raw mountinfo
  record, `fstat().st_dev`, and allowed filesystem type for a retained target
  directory FD.
- `WorkspaceLease`: a no-artifact `LOCK_EX | LOCK_NB` flock on the retained
  repository-root directory FD opened with `O_CLOEXEC`. Conforming QUARTZ
  writers honor it; run/resume holds it from before target-mount probing and
  any campaign/results mutation through the full command.
- `RunRootHandle`: a validated raw-root/run-id interpretation plus retained
  repository/results/campaign-parent directory FDs and their `(st_dev, st_ino)`
  chain; it contains no trusted re-resolved absolute path.
- `OwnedRunRoot`: the exclusively created or opened run-root directory FD,
  nonblocking exclusive flock, parent FD, and stable `(st_dev, st_ino)` chain.
  The lock is retained for the full command lifetime.
- `HeadIdentity`: no-follow HEAD raw identity/bytes; symbolic or detached
  mode; canonical symbolic ref when present; effective loose-ref and
  packed-refs presence/absence plus identity/bytes; and resolved full commit and
  tree OIDs. Unknown or unprovable reference backends fail closed.
- `WorkspaceCapture`: canonical seal value and bytes, SHA-256 digest, captured
  source/input/interpreter and registry/config bytes, workflow specs derived
  from those bytes, and a complete `identity_graph` covering retained
  repository, Git/common directories, `HeadIdentity`, bound configuration,
  every effective index backing object, every source/input edge, and the
  interpreter chain.
- `PublishedSeal`: exact canonical raw bytes and digest returned by publication,
  with an open no-follow final-seal FD and final `(st_dev, st_ino, st_nlink)`.
- `PublishedState`: exact validated state bytes and digest with an open
  no-follow state FD and stable `(st_dev, st_ino, st_nlink)` retained through
  the post-state coherence gate. It is a same-process guard only, not the
  persisted cross-process baseline.
- `InitialStatePlan`: an immutable initial state and canonical bytes. While both
  workspace and run-root locks are held, it samples one exact UTC timestamp
  exactly once after `S_post`; `created_at == updated_at` uses that value. All
  registry/provenance fields derive from `WorkspaceCapture` and
  `PublishedSeal`, with no live registry/path reread.
- `ResumePlan`: an immutable in-memory value containing validated terminal-prefix
  decisions, next attempt/log names, the complete next state, and its canonical
  bytes. It owns no live mutable dictionary.

All trusted IO is relative to retained directory FDs. Component traversal and creation are
no-follow. Each parent-child edge is checked from both retained FDs before and after critical
operations. Rename of a pathname prefix cannot redirect capture or publication: the operation
continues on the owned FD and a changed edge fails before the next durable transition. No later
`Path.resolve`, absolute pathname reopen, or CWD lookup may replace an owned identity. Every
retained FD is opened with `O_CLOEXEC`.

## 6. Canonical seal and exact JSON types

The seal is schema v2 with exact top-level keys:

```text
schema_version, kind, suite, claim_scope, run_id, seed, git,
identity_graph, axis_order, sources, inputs, binaries, campaign_artifacts,
terminal_attempt_artifacts
```

Fixed literals are:

```text
schema_version = 2                    # exact built-in int, not bool or float
kind = idea_foundry_campaign_execution_seal
suite = first-gate-all-sequential
claim_scope = synthetic_contract_execution_only
```

`run_id` is an exact non-empty safe built-in string; `seed` is an exact built-in integer, not
`bool`. `git` has only built-in-string `commit` and `tree` full object IDs. `axis_order` is the
exact 26 IDs implied by the first 26 source members. Inventories and artifact lists exactly match
Section 4.

The seal's `identity_graph` is the canonical persisted `IdentityGraphV1` from `S_pre`, not an
in-memory-only comparison value. Its exact keys are:

```text
repository, git_dir, common_dir, head, configuration, index,
sources, inputs, interpreter, target_mount, run_root_chain
```

Reusable records have these exact key sets:

```text
DirectoryIdentity = {path, st_dev, st_ino, st_mode, mnt_id}
ContentIdentity = {
  path, st_dev, st_ino, st_mode, st_nlink, size_bytes, sha256
}
SymlinkIdentity = {
  path, st_dev, st_ino, st_mode, st_nlink, target
}
PathEdgeIdentity = {
  parent_identity, component, child_kind, child_identity
}
MemberPathIdentity = {path, edges}
InterpreterPathIdentity = {
  invocation_path, resolved_target_path, edges
}
PresenceIdentity = {
  path, present, parent_identity, content_identity
}
TargetMountIdentity = {
  mnt_id, major, minor, st_dev, filesystem_type, mountinfo_sha256
}
```

All numeric fields are exact built-in nonnegative integers, never `bool`; `present` is an exact
built-in boolean; hashes are 64 lowercase hexadecimal built-in strings. Every identity-record `path`
is a canonical absolute lexical path. Only `MemberPathIdentity.path` is a repository-relative POSIX
literal. `DirectoryIdentity` is a retained no-follow directory. `ContentIdentity` is a stable
no-follow regular file whose size/hash represent all raw bytes. `SymlinkIdentity` has symlink mode
and an exact built-in strict-UTF-8 `target` equal to the complete `readlink` bytes; non-UTF-8
targets fail closed.

`PathEdgeIdentity.parent_identity` is a `DirectoryIdentity`; `component` is an exact nonempty
built-in UTF-8 string without slash or NUL; `child_kind` is exactly `directory`, `regular`, or
`symlink`, selecting respectively an exact `DirectoryIdentity`, `ContentIdentity`, or
`SymlinkIdentity` as `child_identity`. The child path is the canonical lexical join of the parent
path and component. Capture obtains each child through the retained parent FD with no-follow
lookup/open/read and before/after identity checks. A later process reconstructs the same ordered
edges from retained roots and requires exact record equality.

For each source/input, `MemberPathIdentity.edges` starts at `identity_graph.repository`, consumes
every literal path component in order, has only directory intermediate children, and ends in one
regular child. Adjacent edges bind the prior directory child as the next parent. Every final child
requires exact `st_nlink == 1`; its mode/size/hash equals the literal Git blob and top-level
inventory record projected from that child without another live read.

`InterpreterPathIdentity.edges` is the complete no-follow resolver transcript from a retained `/` FD
through the absolute `invocation_path`, including every directory, symlink, and final regular
lookup. An absolute symlink target resets the resolver cursor to `/`; a relative target replaces the
pending queue from the symlink parent. Lexical `.`/`..` processing cannot escape `/`. Repeated
resolver state, more than 40 symlink edges, malformed target bytes, or any edge drift fails. The
last edge is the only terminal regular child; it requires exact `st_nlink == 1`, its path equals
`resolved_target_path`, and its size/hash equals the binary descriptor.

`PresenceIdentity.parent_identity` is always a `DirectoryIdentity`; `content_identity` is exactly
`null` iff `present=false`, and is an absolute `ContentIdentity` with the same `path` iff
`present=true`. No other null relation is valid.

`TargetMountIdentity.major`/`minor` are the unique raw mountinfo record's exact parsed fields,
`mountinfo_sha256` hashes that full raw record, and the Section 2 `makedev` equation must hold.
`repository`, `git_dir`, and `common_dir` are `DirectoryIdentity`. `configuration` is the exact
ordered pair of `PresenceIdentity` records for `common_dir/config` and `git_dir/config.worktree`.
`index` is the exact one-element ordered list holding the effective unsplit index `ContentIdentity`.
`sources` and `inputs` are the ordered `MemberPathIdentity` lists matching literal 40/2 membership;
`interpreter` is one `InterpreterPathIdentity`. `target_mount` is one `TargetMountIdentity`.
`run_root_chain` is the prefix-ordered nonempty list of `DirectoryIdentity` records from repository
root through existing/created results, campaign parents, and run root; each has the target mnt_id
and st_dev.

No successful capture stores only a terminal source/input/interpreter identity. The richer graph
lives only in the seal. State binds it through the canonical seal digest and
`execution_seal_identity`; graph duplication in state is forbidden.

`head` has exact keys:

```text
mode, head_file, symbolic_ref, loose_ref, packed_refs, commit, tree
```

`head_file` is a `ContentIdentity`; `packed_refs` is always a `PresenceIdentity`; `commit` and
`tree` are full lowercase OID strings and must equal `git.commit` and `git.tree`. For
`mode="detached"`, `symbolic_ref` and `loose_ref` are exactly `null`, and HEAD raw bytes resolve the
commit. For `mode="symbolic"`, `symbolic_ref` is one canonical `refs/...` string and `loose_ref` is
its `PresenceIdentity`; exactly one unambiguous loose-or-packed storage resolution supplies the
commit. The loose and packed presence/absence, identity, and bytes are bound even when absent or
non-effective. All HEAD/ref/packed raw bytes are represented by their `ContentIdentity` size and
hash. Symbolic chains, ambiguous dual definitions, and unknown backends fail closed.

The seal canonical bytes therefore persist the complete cross-process baseline. The state's
`execution_seal_sha256` hashes those bytes and thus binds the graph.

Every trust-boundary container/scalar is an exact JSON built-in type. Missing or extra keys,
subclasses, coercion, duplicate members or object names, cycles, non-finite values including
`1e400`, malformed UTF-8, non-canonical repository paths, duplicate/reordered paths, invalid hashes,
and non-canonical bytes fail closed.

Canonical bytes are UTF-8 JSON with sorted object keys, compact separators, no ASCII coercion, and
exactly one trailing newline. Loading uses the shared strict loader. Re-encoding must equal
persisted bytes before digest or state binding.

## 7. Git and coherent live-workspace capture

Every relevant Git command uses a fixed `/usr/bin/git` argv, explicit bound
`--git-dir=<retained-git-dir>` and `--work-tree=<retained-repository-root>`, a fresh allowlisted
environment, `LC_ALL=C`, global/system config disabled, replacement objects disabled, literal
pathspecs, and `--no-optional-locks`. Every argv also fixes:

```text
-c core.fsmonitor=false
-c core.untrackedCache=false
-c core.ignoreStat=false
-c core.fileMode=true
-c core.hooksPath=/dev/null
-c fsck.skipList=/dev/null
```

Ambient `GIT_*`, `HOME`, object/index/namespace/replace/config redirects cannot enter the child. No
command may refresh or rewrite the index.

Before trusting Git output, capture proves the no-follow filesystem identities of expected
repository root, Git directory, common directory, and index. It then proves all of:

1. `rev-parse --show-toplevel` is the exact expected repository root;
2. `rev-parse --absolute-git-dir` and `rev-parse --git-common-dir` match the
   expected retained identities;
3. repository-local `core.worktree` is absent; local/worktree configuration
   cannot activate or redirect fsmonitor, untracked cache, ignore-stat,
   file-mode suppression, hooks, fsck policy, object/index paths, sparse
   checkout, or any equivalent trust-boundary setting;
4. exact stable raw bytes/identities of `common_dir/config` and
   `git_dir/config.worktree` are the complete configuration closure described
   below and remain identical before/after capture;
5. replace refs are absent; and
6. the exact P0a integrity command suffix
   `fsck --full --strict --no-dangling --no-reflogs <captured_HEAD>` succeeds
   under the same fixed sanitized/no-replace invocation.

For each present configuration file, capture feeds its already stable raw bytes to this fixed parser
through stdin under the sanitized Git environment:

```text
/usr/bin/git config --file - --no-includes --null --list
```

Parser failure; malformed NUL records; non-UTF-8 names/values; duplicate keys; or any
ASCII-casefolded key in an `include` or `includeIf` section fails closed. `--no-includes` prevents
reading an include target during inspection. Because system/global/env configuration is disabled and
include/includeIf is forbidden, the represented pair plus the fixed command-line `-c` literals is
the complete effective closure; no unrepresented included file is claimed or read. The raw identity,
bytes, parse, and include-free property are repeated in `S_pre`, `S_post`, `S_final`, and resume
recapture.

`HeadIdentity` opens the Git-dir `HEAD` no-follow and binds its stable raw identity and bytes.
Detached HEAD is exactly one full OID plus newline. Symbolic HEAD is exactly `ref: refs/...` plus
newline with one normalized, contained canonical ref name. The effective loose-ref path and
`packed-refs` are each bound as present with no-follow identity/bytes or as proven absent from a
retained parent FD. Resolution must be explained entirely by those bytes; symbolic-ref chains,
reftable or another unknown backend, ambiguous loose/packed precedence, peeled/tag substitution, and
unprovable absence fail closed. The resulting commit and tree are literal stable full OIDs.

All effective index backing identities/stat/bytes are read before and after observation.
Split/shared index, sparse index/checkout, shared-index linkage, and an index fsmonitor extension
are rejected rather than followed. A full `git ls-files -v -z` parse rejects every assume-unchanged
or skip-worktree bit anywhere in the index, not only among the 42 sealed members. `status
--porcelain=v1 -z --untracked-files=all --ignore-submodules=none` must report no tracked or
non-ignored untracked change. HEAD/tree/status/config/effective-index identity and bytes are
rechecked at capture end.

For every literal source and input, literal tree lookup must return exactly one regular blob in mode
`100644` or `100755`. Modes `120000`, `160000`, or any other mode/type fail. Blob bytes are read by
the returned object ID, not a `commit:path` expression. The corresponding live member is opened
through the retained repository FD as the complete ordered `MemberPathIdentity` transcript. Its
final bytes must equal the literal blob. Independently, Git/live executable mode agrees: `100755`
requires executable bits and `100644` requires none. Symlink components/finals, hard-link aliases
where uniqueness is required, FIFO/device/socket/directory substitutions, short reads, or
byte/stat/edge drift fail closed.

The interpreter is the complete ordered root-to-invocation and symlink-resolution transcript in
Section 6, not only its terminal file. Every capture binds `HeadIdentity`, resolved commit/tree
OIDs, configuration closure, and the complete graph. A capture either returns one internally
coherent `WorkspaceCapture` or fails without publishing it.

## 8. Root ownership, durability, and temporal coherence

New-run state transitions are exactly:

```text
ABSENT -> CLAIMED_UNSEALED -> SEALED -> STATE_CREATED
```

Static host/raw-path checks complete first. `WorkspaceLease` is acquired, the exact target-anchor
probe in Section 2 passes with zero residue, and only then may missing campaign parents be created.
Each creation is fsynced, reopened no-follow, and proven to retain the one `TargetMountIdentity`.

`ABSENT -> CLAIMED_UNSEALED` uses directory-FD-relative exclusive `mkdir` of the final run-root
name, fsyncs its parent, opens no-follow, proves the parent-child edge and target mount, and obtains
`LOCK_EX | LOCK_NB`. Existing empty/nonempty roots, files, symlinks, residue, crossing mounts, or a
concurrent creator fail closed; no run root is reused.

`CLAIMED_UNSEALED -> SEALED` creates a same-directory exclusive temporary regular file. A
complete-write loop handles positive short writes and rejects zero/negative progress. It
file-fsyncs, publishes the absent final name via dir-FD no-overwrite same-filesystem hard link,
unlinks the temporary, proves the open final FD is the published inode with link count one, rechecks
target mount, and fsyncs the run root. It returns `PublishedSeal`; downstream digest is
`sha256(PublishedSeal.raw)`, never a pathname reread.

`SEALED -> STATE_CREATED` occurs only after `S_pre == S_post` for canonical raw/registry/workflow
bytes, `HeadIdentity`, resolved commit/tree OIDs, and the complete identity graph. State bytes are
validated before temporary creation, completely written and file-fsynced, atomically renamed within
the owned root, opened as `PublishedState`, target-mount checked, and made durable by run-root
fsync. Temporary files and crash residue are inspection-only; there is no automatic reuse, reseal,
repair, or campaign cleanup.

New-run initial order is exact:

```text
static host/raw-path admission
  -> exclusive no-artifact WorkspaceLease
  -> target-anchor probe -> zero probe residue
  -> same-mount parent creation
  -> exclusive owned root + durable edge + run-root flock
  -> S_pre full capture
  -> durable publish(PublishedSeal from S_pre)
  -> S_post full capture
  -> require S_pre == S_post for raw/registry/workflow,
     HeadIdentity/OIDs, and complete identity_graph
  -> recheck root/seal/target mount
  -> sample one UTC t0; build InitialStatePlan from captured fields
     with created_at == updated_at == t0
  -> validate, publish PublishedState, file fsync, run-root fsync
  -> S_final full capture
  -> require S_pre == S_post == S_final for HeadIdentity/OIDs
     and complete graph, with canonical capture bytes equal
  -> recheck owned-root, PublishedSeal, PublishedState, target mount
  -> only then allow another state transition or attempt IO
```

No provenance source is live-read outside the named `S_pre`, `S_post`, and `S_final` captures. After
`S_post`, state derivation uses only the persisted `S_pre.identity_graph`, captured
registry/workflow values, `PublishedSeal`, and the lifecycle UTC sample; it does not reread a live
provenance path. The `S_final` gate is repeated after every durable state publish that authorizes
attempt IO: the first active attempt, every next-axis active attempt, and a resume retry. Thus the
required trace is:

```text
state file fsync -> state rename -> run-root directory fsync
  -> S_final -> exact equality -> root/seal/state/mount rechecks
  -> attempt directory/log/Popen/other attempt IO
```

For a new run, the initial planned state receives this gate before the first active-state
transition, and that active state receives it again before its attempt IO. Resume admission state
already contains the appended active retry and receives the same gate. No attempt directory, log,
`Popen`, or other attempt IO precedes it.

If any post-state capture/equality/identity check fails, the already durable state-bearing root is
inspection-only. There is no second state write, summary, log, attempt path, subprocess, archive, or
cleanup. The process may report `POST_STATE_COHERENCE_FAILURE` to its caller/stderr, but that label
is never persisted. This is not a campaign `FAILED` transition.

## 9. Shared status, state, and attempt contract

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

Keys are exactly `schema_version`, `execution`, `contract`, `effect`, `evidence_maturity`, and
`promotion`. Aliases `execution_status`, `contract_status`, `effect_status`, and `promotion_status`,
missing/extra fields, `schema_version=true`, `schema_version=2.0`, and unknown enums fail.
`status_schema.py` is the sole exact-type and writer-subset implementation; no sequential duplicate
exists.

| Name | `execution` | `contract` | `effect` | `evidence_maturity` | `promotion` | Allowed writer |
| --- | --- | --- | --- | --- | --- | --- |
| `PLANNED` | `planned` | `not_evaluated` | `not_evaluated` | `contract_only` | `ineligible` | Axis only |
| `RUNNING` | `running` | `not_evaluated` | `not_evaluated` | `contract_only` | `ineligible` | Campaign or axis |
| `FAILED` | `failed` | `failed` | `invalidated` | `contract_only` | `ineligible` | Campaign or axis |
| `SUCCEEDED` | `success` | `passed` | `non_estimable` | `contract_only` | `ineligible` | Campaign or every axis except A10 |
| `SKIPPED` | `skipped` | `not_applicable` | `non_estimable` | `contract_only` | `ineligible` | A10 only |

Schema-v2 campaign state exact base keys are:

```text
schema_version, run_id, suite, status, seed, created_at, updated_at,
execution_seal_sha256, execution_seal_identity, claim_scope, axes
```

`execution_seal_identity` is the exact `ContentIdentity` of the still-open `PublishedSeal` after
exclusive publication and temporary-name removal. Its `sha256` equals `execution_seal_sha256`. It
binds the seal inode/device/mode/ link-count/size as well as bytes without putting a self-reference
in the seal. Same-byte seal replacement on a later process therefore fails resume.

No other top-level key is accepted except lifecycle fields in this table:

| Campaign state | Status | `resumed_at` | `completed_at` |
| --- | --- | --- | --- |
| Initial progress | `RUNNING` | Absent | Absent |
| Resumed progress | `RUNNING` | Required | Absent |
| Failed before resume | `FAILED` | Absent | Absent |
| Failed after resume | `FAILED` | Required | Absent |
| Successful initial invocation | `SUCCEEDED` | Absent | Required |
| Successful resumed invocation | `SUCCEEDED` | Required | Required |

`schema_version` and `seed` are exact built-in integers; identity, fixed literals, timestamps,
status, and seal digest use exact built-in canonical shapes. Lifecycle timestamps are exact
non-empty strings from the shared UTC writer. `PLANNED`/`SKIPPED` campaign states or other lifecycle
combinations fail.

`axes` is exactly 26 built-in dictionaries with exact integer `order_index` values `0..25` in this
order:

```text
A03, A09, A01, A02, A17, A21, A22, A06, A07, A12, A25, A26, A05,
A04, A08, A13, A15, A14, A16, A11, A18, A19, A23, A20, A24, A10
```

Each row's immutable base keys are exactly:

```text
order_index, axis_id, slug, lane_id, role, status, attempts
```

The five registry fields are exactly `order_index`, `axis_id`, `slug`, `lane_id`, and `role`; they
equal workflow specs derived from captured registry bytes. `status` and `attempts` are lifecycle
fields, not registry fields. Writer variants are:

| Axis status | `attempts` | `current_attempt` | `resume_action` | axis `failure_reason` |
| --- | --- | --- | --- | --- |
| `PLANNED` | Empty | Absent | Absent | Absent |
| `RUNNING` | Non-empty; last attempt running | Required, equals last `output_dir` | Absent | Absent |
| `SUCCEEDED` or A10 `SKIPPED` | Non-empty; last attempt successful | Required, equals last `output_dir` | Absent, or exact `verified_skip` on resumed terminal prefix | Absent |
| `FAILED` | Non-empty; last attempt finished unsuccessful | Required, equals last `output_dir` | Absent | Required non-empty built-in string |

Campaign sequence is exact: `RUNNING` has a terminal prefix, optionally one following `RUNNING`
axis, then `PLANNED`; `FAILED` has a terminal prefix, exactly one following `FAILED` axis, then
`PLANNED`; `SUCCEEDED` has all axes terminal, with only A10 `SKIPPED`. `verified_skip` is allowed
only on a terminal prefix in a state with `resumed_at`. A persisted active process is readable but
not resume-eligible.

The initial planned campaign state is `RUNNING` while every axis row remains `PLANNED`, so it
validly has zero active axis rows. B08's `running_without_active` negative means an axis row whose
status is `RUNNING` but whose required last active attempt is absent; it must not be applied to the
initial campaign state.

Each attempt begins with exact keys `attempt_number`, `started_at`, `output_dir`, `stdout`,
`stderr`, and `process_outcome`. Number is the exact one-based list position. Paths are exactly:

```text
axes/{axis_id}/attempt-{N:03d}
logs/{axis_id}.attempt-{N:03d}.stdout.log
logs/{axis_id}.attempt-{N:03d}.stderr.log
```

| Attempt variant | Additional exact keys and relations |
| --- | --- |
| Active | `process_outcome="running"`; no `completed_at`, `returncode`, or attempt `failure_reason`; last attempt of `RUNNING` axis only |
| Successful | `process_outcome="completed"`, `completed_at`, exact-int `returncode=0`, no attempt `failure_reason`; last attempt of terminal axis only |
| Process failure | `process_outcome="completed"`, `completed_at`, exact-int nonzero `returncode` other than reserved codes, no attempt `failure_reason` |
| Launch failure | `process_outcome="failed"`, `completed_at`, exact-int `returncode=126`, no attempt `failure_reason` |
| Analysis failure | `process_outcome="failed"`, `completed_at`, exact-int `returncode=2`, required non-empty attempt `failure_reason` |
| Timeout | `process_outcome="timeout"`, `completed_at`, exact-int `returncode=124`, no attempt `failure_reason` |
| Interrupted | `process_outcome="interrupted"`, `completed_at`, exact-int `returncode=130`, no attempt `failure_reason` |

Every non-last attempt is finished and unsuccessful. The last attempt of a `FAILED` axis is finished
unsuccessful; retry appends rather than rewrites. Unknown outcomes, bool return codes, relation
mismatches, missing/extra fields, nonconsecutive numbering, and path mismatch fail closed.

Every state writer, including initial, pre-attempt, post-attempt, failure, resume, and completion,
must validate its exact in-memory value and canonical round trip before creating a temporary file. A
writer-produced invalid state is an internal hard failure and cannot reach disk.

Existing analyzer/meta consumers remain schema-v1-only until P0b-C2. Given a new schema-v2 campaign
state, they must fail closed before creating or changing any output. No analyzer/meta source change
is authorized here. The sole allowed compatibility change is replacing the existing end-to-end
analyzer-success assertion in `tests/test_idea_foundry_axis_workflows.py` with an exact schema-v2
incompatibility/no-output assertion; no other existing assertion may be removed or weakened.

## 10. Resume ownership, immutable planning, and one admission mutation

Resume first obtains `WorkspaceLease`, opens the run root through `RunRootHandle`, obtains a
nonblocking exclusive flock before reading any campaign child, and holds both locks for the full
command. Under those locks it recomputes `TargetMountIdentity` for every retained target-chain FD
and rejects crossing or drift before mutation. It then performs, without mutation:

1. load the canonical state without mutation, validate its
   `execution_seal_identity`, and require the open seal's exact
   `ContentIdentity` to equal it, including
   `sha256 == execution_seal_sha256`;
2. validate canonical seal bytes and the persisted exact `IdentityGraphV1`,
   then require its `git.commit/tree` and `head.commit/tree` agreement;
3. validate the remaining exact state lifecycle, 26 captured-registry rows,
   sequence, attempts, status writer subset, contained names, and require an
   exact `FAILED` campaign with no active attempt;
4. validate every terminal-prefix attempt and all seven terminal artifacts
   against manifests/hashes/status using no-follow dir-FD IO;
5. recapture the current `IdentityGraphV1` and canonical workspace, and compare
   them exactly to the graph and identity persisted in the seal rather than an
   in-memory baseline;
6. validate absent/canonical next attempt/log names and all retained identities;
   and
7. construct immutable `ResumePlan`, including complete next state, then
   validate and canonical-round-trip it.

Only then does resume publish one admission state: campaign `FAILED -> RUNNING`, `resumed_at`,
validated-prefix `verified_skip`, old row `failure_reason` removed, and one consecutive active retry
appended without rewriting history. After state file fsync/rename and run-root fsync, resume
performs a full `S_final` capture and requires the persisted seal `IdentityGraphV1`, the
pre-admission recapture, and `S_final` to have identical `HeadIdentity`, OIDs, canonical capture,
and complete graph; it then rechecks root/seal/state/target mount. Only then may retry attempt IO
begin.

The same post-state `S_final` admission applies to the first and every next axis attempt. A
successful retry leaves no row `failure_reason` and validates through the same writer. Any pre-state
rejection preserves the complete byte/path snapshot. Any post-state coherence failure follows
Section 8: the durable state-bearing root is inspection-only, with no second write or attempt IO and
no persisted failure label.

Terminal-prefix validation proves resume eligibility only. P0b-C2 still owns final campaign
state/summary/artifact closure; P0b-C1 cannot claim it.

## 11. Frozen RED/GREEN behavior matrix

B01--B12 are the only current acceptance behaviors. Each exact pytest node contains one positive
control and only the finite negative vector IDs listed below. Common helpers may create temporary
Git repositories, synthetic 40/2/1 members, canonical synthetic attempt artifacts, deterministic
process barriers, bounded injected short-write/fsync/rename/drift seams, and byte/path snapshots.
They must not launch a real axis subprocess or a real campaign.

For every negative, a test asserts the named boundary through deterministic call-order/fault-seam
evidence and compares the complete pre/post run-root path set plus every regular-file byte string,
symlink target, mode, and link count. It also installs mutation sentinels on
state/summary/attempt/log/archive writers. An exception alone is not an oracle.

| ID and exact pytest node | Positive control | Finite negative vector IDs | Exact `f825c38` RED signature | Successor GREEN oracle/sentinel |
| --- | --- | --- | --- | --- |
| **B01** `tests/test_idea_foundry_execution_seal.py::test_b01_literal_canonical_status_exactness` | Exact literal 40/2/1, artifacts, canonical seal and each writer status round-trip | `source_missing`, `source_extra`, `source_duplicate`, `source_reordered`, `input_substituted`, `artifact_reordered`, `coordinated_constant_drift`, `duplicate_key`, `bad_utf8`, `nan`, `infinity`, `1e400`, `noncanonical_bytes`, `status_alias`, `schema_bool`, `schema_float`, `unknown_enum` | Monkeypatching the shared `SOURCES` support-member order changes both producer and validator, so coordinated drift is accepted instead of independently rejected | Exact canonical bytes round-trip; every negative stops before Git/member/publication seams and leaves snapshot/sentinels unchanged |
| **B02** `tests/test_idea_foundry_execution_seal.py::test_b02_git_root_config_object_and_mode_identity` | `LocalConfigBinding` admits the exact allowlist; fixed Git observes expected top/git/common/config identities, captured-HEAD strict fsck, stable HEAD/tree and live-mode-equal `100644`/`100755` blobs | `hostile_git_env`, `local_config_gate_failure`, `local_core_worktree`, `local_core_fsmonitor`, `local_untracked_cache`, `local_ignorestat`, `local_filemode`, `local_fsck_skiplist`, `top_mismatch`, `gitdir_mismatch`, `common_dir_mismatch`, `replace_ref`, `fsck_failure`, `reachable_object_corrupt`, `blob_missing`, `mode_120000`, `mode_160000`, `live_mode_mismatch` | Local config can redirect or weaken observation; mode `120000` is accepted as a blob under `core.symlinks=false`; no strict fsck rejects a corrupt reachable unsealed object | Config-gate negatives stop before candidate fsck and admit no downstream Git result; the remaining trace proves exact captured-HEAD fsck precedes member reads, with no publication/mutation |
| **B03** `tests/test_idea_foundry_execution_seal.py::test_b03_global_cleanliness_and_index_stability` | Whole unsplit index has no flags/extensions, all backing/config bytes are stable, porcelain is empty, HEAD/tree unchanged | `tracked_dirty`, `untracked_nonignored`, `assume_outside_inventory`, `skip_outside_inventory`, `split_index`, `shared_index`, `sparse_index`, `fsmonitor_index`, `index_byte_drift`, `index_stat_drift`, `head_moves`, `tree_moves`, `optional_lock_write` | `assume-unchanged`/`skip-worktree` outside the sealed 42 members is accepted; status is not invoked under the required no-optional-lock contract | Trace is `config/index_before -> whole_flags/extensions -> status_ignore_submodules_none -> identity_recheck -> config/index_after`; no backing bytes/stat or run-root snapshot changes |
| **B04** `tests/test_idea_foundry_execution_seal.py::test_b04_stable_nofollow_members_and_binary` | Every source/input terminal and interpreter terminal target is regular with exact `st_nlink == 1`; complete member and root-to-invocation/symlink transcripts match captured bytes | `ancestor_symlink`, `final_symlink`, `hardlink_alias`, `fifo`, `device`, `socket`, `directory`, `member_midread_drift`, `member_edge_swap`, `binary_loop`, `binary_target_drift`, `binary_nonregular`, `short_read` | Path walk is followed by path-based open/read, so a deterministic parent/member swap can redirect use after inspection; hostile alias tests can fail earlier only as generic Git dirt | `hardlink_alias` enforces `st_nlink == 1` on those exact 42 member terminals and interpreter target; retained FD/inode trace proves every edge/final identity, decoy unread and snapshot unchanged |
| **B05** `tests/test_idea_foundry_execution_seal.py::test_b05_raw_campaign_path_dirfd_and_parent_swap` | Allowed target mount and exclusive target-anchor probe complete `probe-dir fsync -> identity-checked rmdir -> target-anchor fsync -> zero residue -> campaign parent creation`; canonical paths produce one same-mount retained chain | `missing_dirfd`, `missing_nofollow`, `missing_odirectory`, `missing_cloexec`, `missing_follow_symlink_support`, `directory_fsync_unsupported`, `filesystem_tmpfs`, `filesystem_overlay`, `filesystem_nfs`, `filesystem_cifs`, `filesystem_fuse`, `filesystem_unknown`, `fdinfo_mnt_id_missing`, `mount_record_missing`, `mount_record_ambiguous`, `mount_device_mismatch`, `results_submount_nfs`, `results_submount_local`, `campaign_parent_mount_crossing`, `probe_mount_mismatch`, `probe_capability_failure`, `probe_directory_fsync_failure`, `probe_parent_directory_fsync_failure`, `probe_cleanup_failure`, `runroot_mount_changes`, `path_object`, `string_subclass`, `absolute_outside`, `bare_results`, `dot_component`, `dotdot`, `double_slash`, `trailing_slash`, `backslash`, `nul`, `bad_run_id`, `symlink_campaign_root`, `preexisting_root`, `parent_swap` | `Path.resolve()` erases raw symlink spelling and the prior probe may run on another mount; target mount identity/crossing is not closed | Every vector asserts its exact phase/residue; no campaign/run-root/seal/state path is created, successful probe residue is zero, and retained FDs keep decoys untouched |
| **B06** `tests/test_idea_foundry_execution_seal.py::test_b06_exclusive_root_full_write_and_durability` | Exclusive root, forced positive short writes, file fsync, no-overwrite publish, temp unlink, link-count proof and ordered directory fsyncs produce exact bytes | `concurrent_creator`, `existing_empty`, `existing_nonempty`, `existing_file`, `existing_symlink`, `write_zero`, `file_fsync_fail`, `publish_collision`, `temp_unlink_fail`, `bad_link_count`, `parent_fsync_fail`, `runroot_fsync_fail`, `state_fsync_fail`, `state_dir_fsync_fail` | One `os.write` truncates under a short-write seam; run-root parent and post-state directory are not fsynced, and temporary state residue is cleaned | Exact bytes/digest and fsync call order pass; each fault fails at named durability edge and preserves inspectable residue without reuse/cleanup |
| **B07** `tests/test_idea_foundry_execution_seal.py::test_b07_pre_post_capture_derived_state_and_seal_binding` | Exact ref-mode variants for initial/next-axis/resume prove `S_pre=S_post=S_final`, include-free configuration closure, complete persisted member/interpreter edges, seal identity, one UTC sample, and durable state before attempt IO | `workspace_lock_contended`, `persisted_identity_graph_missing`, `persisted_identity_graph_mismatch`, `seal_identity_state_mismatch`, `malformed_head`, `symbolic_ref_chain`, `unsupported_ref_backend`, `loose_packed_ambiguity`, `head_storage_presence_flip`, `same_tree_different_commit`, `source_drift_between`, `git_drift_between`, `registry_drift_after_post`, `same_bytes_identity_graph_replaced_between`, `head_moves_after_post`, `head_ref_storage_replaced`, `member_drift_during_state_publish`, `index_drift_during_state_publish`, `packed_refs_bytes_after_state`, `symbolic_target_same_oid_after_state`, `member_same_bytes_new_inode_after_post`, `interpreter_same_bytes_new_inode_after_post`, `index_same_bytes_new_inode_after_post`, `seal_path_replaced`, `seal_inode_drift`, `runroot_edge_drift_before_state`, `runroot_edge_drift_after_state`, `s_final_capture_error`, `next_axis_s_final_drift`, `resume_s_final_drift` | `_new_state()` rereads live registry, seal digest reopens a path, no complete persisted graph/config closure exists, and no post-state capture prevents attempt IO after drift | `persisted_identity_graph_mismatch` structurally covers every existing graph record family and `git_drift_between` covers config bytes/properties; phase-specific residue remains exact and trace remains `state_dir_fsync -> S_final -> equality -> root/seal/state/mount recheck -> attempt IO` |
| **B08** `tests/test_idea_foundry_axis_workflows.py::test_b08_writer_validator_roundtrip_before_every_state_publish` | Every initial/resumed running, failed, and successful writer state canonical-round-trips through the single shared validator; schema-v2 analyzer admission fails with no output | `top_extra`, `schema_bool`, `bad_lifecycle`, `bad_axis_order`, `bad_registry_field`, `planned_with_attempt`, `running_without_active`, `failed_without_reason`, `terminal_with_reason`, `bad_resume_action`, `bad_attempt_number`, `bool_returncode`, `outcome_code_mismatch`, `noncanonical_attempt_path`, `bad_sequence`, `schema_v2_analyzer_incompatible` | `_save_state` writes without self-validation; an invalid failed/terminal row can be durably persisted, duplicate compressed status logic diverges, and the old E2E expects analyzer success | Validator event precedes temporary creation on every writer path; all negatives leave state/output bytes/path unchanged, analyzer incompatibility is explicit, and shared status logic has one implementation |
| **B09** `tests/test_idea_foundry_axis_workflows.py::test_b09_terminal_prefix_validation_precedes_resume_mutation` | A failed campaign with a fully valid terminal prefix yields an immutable in-memory plan without disk mutation | `artifact_missing`, `artifact_extra`, `artifact_symlink`, `artifact_noncanonical`, `manifest_hash_wrong`, `analysis_status_wrong`, `analysis_lineage_wrong`, `prefix_status_wrong`, `attempt_path_alias` | A terminal-prefix validation failure rewrites campaign state to failed and creates/rewrites `campaign_summary.json` before raising | Trace completes all prefix checks before plan/admission; every negative has an identical full snapshot and every mutation sentinel remains untouched |
| **B10** `tests/test_idea_foundry_axis_workflows.py::test_b10_failed_to_successful_retry_is_valid_and_append_only` | Synthetic first attempt return 9 then retry return 0: old attempt unchanged, new attempt appended, row terminal, `failure_reason` absent, persisted state validates | `failed_row_missing_reason`, `attempt_name_exists`, `log_name_exists`, `nonconsecutive_next_number`, `success_analysis_mismatch`, `retry_failure_reason_wrong_source` | Retry changes status to running but retains old row `failure_reason`; success persists a terminal row that its own validator rejects | Pre-admission negatives are snapshot-immutable; positive canonical state validates before each publish and old attempt bytes/dictionary remain unchanged |
| **B11** `tests/test_idea_foundry_axis_workflows.py::test_b11_concurrent_resume_has_one_owned_writer` | Two spawned processes meet deterministic barrier; exactly one acquires flock and reserves the next attempt/state transition | `lock_already_held`, `lock_unsupported`, `runroot_inode_changes`, `attempt_reservation_collision` | No resume flock/CAS exists, so both processes compute the same attempt number and log/output names and may overwrite state | One winner performs one admission publish; loser reports contention before first seal read and its before/after full snapshot is identical |
| **B12** `tests/test_idea_foundry_axis_workflows.py::test_b12_legacy_running_success_rejection_is_locked_and_immutable` | Exact failed/no-active control reaches validated `ResumePlan`; rejection controls acquire lock before first read | `schema_v1`, `running_campaign`, `successful_campaign`, `active_attempt`, `completed_field_mismatch`, `malformed_seal`, `stale_capture`, `seal_digest_mismatch` | Simple terminal cases may reject, but there is no flock event before the first seal/state read, so rejection is not single-writer or race-immutable | Trace begins `flock_acquired -> seal_read`; every rejection has exact pre/post full byte/path equality and no writer/timestamp event |

B02 `local_config_gate_failure` has exactly these finite subcases: `disallowed_key`,
`duplicate_allowed_key`, `include_path`, `matching_includeif_gitdir`, `fsck_missingemail_ignore`,
`extensions_worktreeconfig`, `worktree_config_present`, `same_bytes_new_inode`, and
`allowed_bytes_changed`. Each rejects before candidate fsck and admits no downstream Git result.

B05 `probe_capability_failure` has exactly: `mkdir`, `open`, `flock`, `write`, `file_fsync`, `link`,
`rename`, and `unlink`. `probe_directory_fsync_failure` fails the probe-dir fsync, leaves one empty
inspection-only probe directory, and performs neither rmdir nor target-anchor fsync.
`probe_parent_directory_fsync_failure` performs an identity-checked successful rmdir, then fails
target-anchor fsync: live residue is zero but admission fails. `probe_cleanup_failure` is exactly
the separate identity-checked rmdir failure.

The RED proof uses a disposable detached worktree at read-only production commit `f825c38`,
overlaying only the two frozen successor test files. It records the row-specific RED signatures
above; it never treats `f825c38` as a base or copies its production code. Base `292d033` lacks
`execution_seal.py`, so it can prove absence but cannot provide a meaningful behavior-by-behavior
executable RED. The successor implementation must turn every exact node GREEN without changing the
matrix.

## 12. Required commands and pass/fail gate

Before implementation, record the two-test overlay RED transcript against detached `f825c38`. The
verification shell block starts in strict mode. `SUCCESSOR_SPEC_SHA`, `IMPLEMENTATION_SHA`,
`REPOSITORY_ROOT`, `GIT_DIR_PATH`, and `GIT_COMMON_DIR_PATH` are externally supplied frozen values.
No command derives an identity from mutable `HEAD`. An amend changes the implementation OID and
invalidates every prior review, range hash, and gate.

```bash
set -euo pipefail

: "${SUCCESSOR_SPEC_SHA:?supply frozen full successor specification OID}"
: "${IMPLEMENTATION_SHA:?supply frozen full implementation OID}"
: "${REPOSITORY_ROOT:?supply canonical absolute candidate worktree root}"
: "${GIT_DIR_PATH:?supply canonical absolute candidate git directory}"
: "${GIT_COMMON_DIR_PATH:?supply canonical absolute common git directory}"
readonly SUCCESSOR_SPEC_SHA IMPLEMENTATION_SHA
readonly REPOSITORY_ROOT GIT_DIR_PATH GIT_COMMON_DIR_PATH
readonly GIT=/usr/bin/git
readonly BASE_SHA=292d0336b246f990cb0fc7f99ef11f464d506219
readonly FULL_OID_RE='^([0-9a-f]{40}|[0-9a-f]{64})$'
[[ "$SUCCESSOR_SPEC_SHA" =~ $FULL_OID_RE ]]
[[ "$IMPLEMENTATION_SHA" =~ $FULL_OID_RE ]]

for bound_path in \
  "$REPOSITORY_ROOT" "$GIT_DIR_PATH" "$GIT_COMMON_DIR_PATH"
do
  [[ "$bound_path" = /* ]]
  test -d "$bound_path"
  test "$(cd -- "$bound_path" && /bin/pwd -P)" = "$bound_path"
done
PHYSICAL_PWD="$(/bin/pwd -P)"
readonly PHYSICAL_PWD
test "$PHYSICAL_PWD" = "$REPOSITORY_ROOT"

QUARTZ_TEST_VENV="${QUARTZ_TEST_VENV:-./venv}"
readonly QUARTZ_TEST_VENV
test -x "$QUARTZ_TEST_VENV/bin/python"
test -x "$QUARTZ_TEST_VENV/bin/ruff"

local_config_binding() {
  "$QUARTZ_TEST_VENV/bin/python" - \
    "$GIT_COMMON_DIR_PATH" "$GIT_DIR_PATH" "$GIT" <<'PY'
import hashlib, json, os, re, stat, subprocess, sys
common_dir, git_dir, git = sys.argv[1:]
NOFOLLOW = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
DIRECTORY = NOFOLLOW | os.O_DIRECTORY

def snapshot(value):
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_uid, value.st_gid, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )

def read_all(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)

def require_worktree_absent(git_fd):
    try:
        os.stat("config.worktree", dir_fd=git_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise SystemExit("config.worktree present or dangling")

common_fd = os.open(common_dir, DIRECTORY)
git_fd = os.open(git_dir, DIRECTORY)
try:
    require_worktree_absent(git_fd)
    config_fd = os.open("config", NOFOLLOW, dir_fd=common_fd)
    try:
        before = snapshot(os.fstat(config_fd))
        path_before = snapshot(os.stat("config", dir_fd=common_fd, follow_symlinks=False))
        if not stat.S_ISREG(before[2]) or before[3] != 1 or before[6] > 1048576:
            raise SystemExit("unsafe local config identity")
        raw = read_all(config_fd)
        if len(raw) != before[6] or snapshot(os.fstat(config_fd)) != before:
            raise SystemExit("unstable local config read")
        env = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_COUNT": "0",
        }
        parsed = subprocess.run(
            [git, "config", "--file", f"/proc/self/fd/{config_fd}",
             "--no-includes", "--null", "--list"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, pass_fds=(config_fd,),
        )
        after_raw = read_all(config_fd)
        after = snapshot(os.fstat(config_fd))
        path_after = snapshot(os.stat("config", dir_fd=common_fd, follow_symlinks=False))
    finally:
        os.close(config_fd)
    require_worktree_absent(git_fd)
    if (path_before != before or path_after != before or parsed.returncode != 0
            or parsed.stderr or raw != after_raw or before != after):
        raise SystemExit("local config parse or stability failure")
    records = parsed.stdout.split(b"\0")
    if not records or records[-1] != b"":
        raise SystemExit("malformed local config records")
    values = {}
    for record in records[:-1]:
        name_raw, separator, value_raw = record.partition(b"\n")
        if not separator:
            raise SystemExit("malformed local config entry")
        name = name_raw.decode("utf-8", "strict").casefold()
        value = value_raw.decode("utf-8", "strict")
        if name in values:
            raise SystemExit("duplicate local config key")
        values[name] = value
    required = {
        "core.repositoryformatversion": "0",
        "core.filemode": "true",
        "core.bare": "false",
        "core.logallrefupdates": "true",
    }
    optional = {"pull.rebase": "false", "coderabbit.defaultbranch": "main"}
    if any(values.get(key) != value for key, value in required.items()):
        raise SystemExit("required local config mismatch")
    for name, value in values.items():
        if (
            name.startswith("include.")
            or name.startswith("includeif.")
            or name.startswith("fsck.")
            or name == "extensions.worktreeconfig"
        ):
            raise SystemExit("forbidden local config key")
        if name in required:
            continue
        if name in optional:
            if value != optional[name]:
                raise SystemExit("optional local config mismatch")
            continue
        remote = re.fullmatch(r"remote\.([a-z0-9][a-z0-9._/-]*)\.(url|fetch)", name)
        branch = re.fullmatch(
            r"branch\.([a-z0-9][a-z0-9._/-]*)\.(remote|merge|vscode-merge-base)",
            name,
        )
        if (remote is None and branch is None) or not value:
            raise SystemExit("disallowed local config entry")
    payload = {
        "common_dir": snapshot(os.fstat(common_fd)),
        "git_dir": snapshot(os.fstat(git_fd)),
        "config_stat": before,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config_worktree_present": False,
        "values": sorted(values.items()),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    print(hashlib.sha256(encoded).hexdigest())
finally:
    os.close(git_fd)
    os.close(common_fd)
PY
}

INITIAL_LOCAL_CONFIG_BINDING="$(local_config_binding)"
readonly INITIAL_LOCAL_CONFIG_BINDING

git_bound() {
  local before after returncode
  before="$(local_config_binding)" || return 1
  test "$before" = "$INITIAL_LOCAL_CONFIG_BINDING" || return 1
  if /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
    GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_COMMON_DIR="$GIT_COMMON_DIR_PATH" "$GIT" \
      --no-replace-objects --no-optional-locks --literal-pathspecs \
      --git-dir="$GIT_DIR_PATH" --work-tree="$REPOSITORY_ROOT" \
      -c core.fsmonitor=false -c core.untrackedCache=false \
      -c core.ignoreStat=false -c core.fileMode=true \
      -c core.hooksPath=/dev/null -c fsck.skipList=/dev/null "$@"
  then
    returncode=0
  else
    returncode=$?
  fi
  after="$(local_config_binding)" || return 1
  test "$after" = "$before" || return 1
  return "$returncode"
}

assert_bound_context() {
  test "$(git_bound rev-parse --show-toplevel)" = "$REPOSITORY_ROOT"
  test "$(git_bound rev-parse --absolute-git-dir)" = "$GIT_DIR_PATH"
  test "$(git_bound rev-parse --git-common-dir)" = "$GIT_COMMON_DIR_PATH"
  test "$(/bin/pwd -P)" = "$REPOSITORY_ROOT"
}

assert_clean_candidate() {
  local status_output
  status_output="$(git_bound status --porcelain=v1 \
    --untracked-files=all --ignore-submodules=none)"
  test -z "$status_output"
  git_bound ls-files -v -z | "$QUARTZ_TEST_VENV/bin/python" -c '
import sys
rows = [row for row in sys.stdin.buffer.read().split(b"\0") if row]
raise SystemExit(any(row[:1] == b"S" or chr(row[0]).islower() for row in rows))
'
  test "$(git_bound rev-parse --verify 'HEAD^{commit}')" = "$IMPLEMENTATION_SHA"
}

assert_single_parent() {
  local line
  local -a parents
  line="$(git_bound show -s --format=%P "$1")"
  read -r -a parents <<< "$line"
  test "${#parents[@]}" -eq 1
  test "${parents[0]}" = "$2"
}

readonly -a DIFF_FLAGS=(--no-renames --no-ext-diff --no-textconv)
readonly -a PRODUCTION_PATHS=(
  quartz/idea_foundry/execution_seal.py
  quartz/idea_foundry/sequential.py
  quartz/idea_foundry/axis_workflow.py
  quartz/idea_foundry/status_schema.py
)
readonly -a EXPECTED_PATHS=(
  "${PRODUCTION_PATHS[@]}"
  tests/test_idea_foundry_execution_seal.py
  tests/test_idea_foundry_axis_workflows.py
)

range_hash() {
  git_bound diff "${DIFF_FLAGS[@]}" --binary --full-index "$RANGE" -- |
    /usr/bin/sha256sum | /usr/bin/awk '{print $1}'
}

assert_bound_context
test "$(git_bound cat-file -t "$SUCCESSOR_SPEC_SHA")" = commit
test "$(git_bound cat-file -t "$IMPLEMENTATION_SHA")" = commit
test "$(git_bound rev-parse --verify "${SUCCESSOR_SPEC_SHA}^{commit}")" = \
  "$SUCCESSOR_SPEC_SHA"
test "$(git_bound rev-parse --verify "${IMPLEMENTATION_SHA}^{commit}")" = \
  "$IMPLEMENTATION_SHA"
assert_single_parent "$SUCCESSOR_SPEC_SHA" "$BASE_SHA"
assert_single_parent "$IMPLEMENTATION_SHA" "$SUCCESSOR_SPEC_SHA"
test "$(git_bound rev-parse --verify 'HEAD^{commit}')" = "$IMPLEMENTATION_SHA"
git_bound fsck --full --strict --no-dangling --no-reflogs "$IMPLEMENTATION_SHA"
assert_clean_candidate
readonly RANGE="${SUCCESSOR_SPEC_SHA}...${IMPLEMENTATION_SHA}"

git_bound diff "${DIFF_FLAGS[@]}" --name-only -z "$RANGE" -- |
  "$QUARTZ_TEST_VENV/bin/python" -c '
import sys
expected = tuple(sys.argv[1:])
actual = tuple(x.decode() for x in sys.stdin.buffer.read().split(b"\0") if x)
if len(actual) != len(set(actual)) or sorted(actual) != sorted(expected):
    raise SystemExit(f"scope mismatch: {actual!r}")
' "${EXPECTED_PATHS[@]}"
git_bound diff "${DIFF_FLAGS[@]}" --check "$RANGE" --
RANGE_DIFF_SHA256="$(range_hash)"
readonly RANGE_DIFF_SHA256
printf 'RANGE_DIFF_SHA256=%s\n' "$RANGE_DIFF_SHA256"

git_bound diff "${DIFF_FLAGS[@]}" --numstat -z "$RANGE" -- \
  "${PRODUCTION_PATHS[@]}" | "$QUARTZ_TEST_VENV/bin/python" -c '
import sys
caps = dict(zip(sys.argv[1::2], map(int, sys.argv[2::2]), strict=True))
seen = {}
for record in filter(None, sys.stdin.buffer.read().split(b"\0")):
    fields = record.split(b"\t", 2)
    if len(fields) != 3:
        raise SystemExit("invalid production numstat record")
    added, deleted, raw_path = fields
    path = raw_path.decode()
    if not (added.isdigit() and deleted.isdigit()) or path not in caps or path in seen:
        raise SystemExit("invalid or duplicate production numstat")
    seen[path] = int(added) + int(deleted)
if set(seen) != set(caps):
    raise SystemExit("production path set mismatch")
if any(seen[path] > cap for path, cap in caps.items()) or sum(seen.values()) > 1160:
    raise SystemExit("production cap exceeded")
for path in sorted(seen):
    print(f"{seen[path]}\t{path}")
' \
  quartz/idea_foundry/execution_seal.py 580 \
  quartz/idea_foundry/sequential.py 420 \
  quartz/idea_foundry/axis_workflow.py 40 \
  quartz/idea_foundry/status_schema.py 200

"$QUARTZ_TEST_VENV/bin/ruff" check "${EXPECTED_PATHS[@]}"
"$QUARTZ_TEST_VENV/bin/ruff" format --check "${EXPECTED_PATHS[@]}"
"$QUARTZ_TEST_VENV/bin/python" -m pytest -q \
  tests/test_idea_foundry_execution_seal.py \
  tests/test_idea_foundry_axis_workflows.py
"$QUARTZ_TEST_VENV/bin/python" -m pytest -q tests/
cargo test --release --locked
cargo test --release --locked --features idea-foundry
"$QUARTZ_TEST_VENV/bin/python" scripts/verify_evidence_receipt.py \
  --allow-empty-diagnostic

assert_bound_context
test "$(git_bound rev-parse --verify 'HEAD^{commit}')" = "$IMPLEMENTATION_SHA"
assert_clean_candidate
FINAL_RANGE_DIFF_SHA256="$(range_hash)"
test "$FINAL_RANGE_DIFF_SHA256" = "$RANGE_DIFF_SHA256"
test "$(local_config_binding)" = "$INITIAL_LOCAL_CONFIG_BINDING"
printf 'FINAL_RANGE_DIFF_SHA256=%s\n' "$FINAL_RANGE_DIFF_SHA256"
```

All Git commands in the gate cross the single sanitized `git_bound` boundary. CI/normal checkouts
default to repository-local `./venv`; linked-worktree operators may explicitly point
`QUARTZ_TEST_VENV` to the canonical checkout venv. Exact base `292d033` has no receipt, so the
implementation adds zero receipt paths and `--allow-empty-diagnostic` checks only the empty set. It
grants no receipt/evidence claim; any invalid nonempty receipt still fails.

The NUL-delimited scope and numstat parsers must accept exactly the six paths and four production
paths respectively. Every per-file cap and total `M <= 1160`, Ruff/focused/full/Rust/receipt gate,
strict candidate fsck, exact base/spec/candidate parent and HEAD check, initial/final bound-root and
clean check, and exact range-hash stability must pass. No plain `git diff --check`, mutable-HEAD
range, rename detection, external diff, or text conversion is permitted. A red required gate blocks
review.

## 13. Evidence and claim firewall

The following is a non-runtime claim-ledger projection for this specification-only point, not a
persisted campaign status:

```text
execution=skipped
contract=not_applicable
effect=non_estimable
evidence_maturity=contract_only
promotion=ineligible
```

P0b-C1 and P0a remain distinct:

```text
P0b-C1 execution seal = prospective live execution identity
P0a receipt verifier   = declared execution/analysis Git-tree integrity
```

Neither alone nor together before later closure may be called `provenance complete`. This
specification generates no external result, receipt, report, or claim update and authorizes no
run/resume, push, PR, merge, scientific execution, evidence-maturity increase, efficacy claim, or
promotion.
