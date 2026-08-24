# P0B-C1 Trusted-Local Execution Identity Specification

## Authority and status

This is the lean successor specification for the personal, single-developer
Idea Foundry research environment.  It is controlled by the audited
[stacked-PR plan](../plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md)
and the [research trust model](RESEARCH_TRUST_MODEL_20260823.md).

`P0B-C1=NOT_ESTABLISHED`.

This document specifies the minimum execution-identity and resume boundary for
future implementation.  It does not implement that boundary, authorize a
campaign, create a result or receipt, or establish any scientific claim.  It
is infrastructure for the registered MCTS studies, including future common
baseline, A-only, B-only, and A+B comparisons.  It neither evaluates an axis
nor supports an interaction or superadditivity conclusion.

## Trusted-local scope

QUARTZ is personal MCTS research operated by one trusted developer on one
trusted-local machine.  Ordinary `git rev-parse HEAD` and a clean `git status`
are research identity observations, not security proof.  They are used to
detect accidental source drift before a run and to bind the observed source
revision to its local configuration and runtime.

This specification deliberately excludes hostile Git configuration, mount
attacks, symlink-race handling, signature systems, and anti-tamper mechanisms.
Those cases are neither requirements nor claims for ordinary trusted-local
operation.  This exclusion does not relax research integrity requirements for
clean source/config/input identity, deterministic treatment definitions, raw
failure preservation, or execution-to-analysis lineage.

## Interface declarations

The following declarations are interface contracts only.  They are not
implementation code and do not prescribe a security perimeter.

```python
@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True)
class RuntimeIdentity:
    python_executable: str
    python_version: str
    platform: str
    argv: tuple[str, ...]

@dataclass(frozen=True)
class ExecutionIdentity:
    schema_version: int
    git_head: str
    git_dirty: bool
    axis_registry: FileIdentity
    lab_registry: FileIdentity
    source_files: tuple[FileIdentity, ...]
    runtime: RuntimeIdentity

@dataclass(frozen=True)
class ExecutionCapture:
    identity: ExecutionIdentity
    identity_sha256: str
    axis_registry_bytes: bytes
    lab_registry_bytes: bytes
    specs: tuple[AxisWorkflowSpec, ...]

@dataclass(frozen=True)
class InitialStatePlan:
    identity_payload: dict[str, object]
    identity_sha256: str
    state_payload: dict[str, object]

@dataclass(frozen=True)
class ResumePlan:
    prior_state_sha256: str
    next_state: dict[str, object]
    retry_axis_id: str
    retry_attempt_number: int
    verified_terminal_axis_ids: tuple[str, ...]
```

`ExecutionIdentity` describes one observed local execution context.  The two
registry fields identify the exact axis and lab configuration files; every
listed source file has a repository-relative path, byte length, and SHA-256.
`RuntimeIdentity` records the executable and invocation that interpreted the
captured inputs.  `ExecutionCapture` retains the registry byte snapshots and
the workflow specifications parsed from those snapshots.

## New-run capture contract

Before any durable campaign state is created, a new run must:

1. Confirm a clean committed research workspace and record its ordinary Git
   revision, selected source-file identities, both registry identities, and
   runtime identity.
2. Read the axis and lab configuration bytes once.  Workflow specifications
   and the initial state must derive from those captured bytes, not from a
   later live reread of either configuration path.
3. Create the campaign root with `mkdir(exist_ok=False)` and retain a
   persistent local flock file for the lifetime of the writer.  This prevents
   accidental duplicate local starts; it is not a cross-host ownership or
   tamper-resistance protocol.
4. Build an `InitialStatePlan` in memory, validate all status payloads through
   the canonical status-schema authority, and only then write the first state.

The initial state records the canonical identity payload and its digest.  A
future writer must validate the state it produces before every durable write.
If capture, validation, or local ownership fails, the new run must fail without
being presented as an executed campaign.

## Resume contract

Resume is an idempotent local recovery operation, not a way to overwrite prior
research evidence.  Only a campaign state that is both `failed` and inactive
may resume.  Before a retry becomes eligible, the implementation must validate
the persisted state, its identity binding, and the complete terminal prefix of
prior attempts.

A rejected resume must leave the prior state, attempts, results, logs, and
summary bytes unchanged.  Validation and planning therefore occur in memory
before any retry state mutation.  A successful plan identifies exactly one
failed axis, its next attempt number, and the verified terminal-axis IDs through
`ResumePlan`; a later implementation must retain the local writer lock while it
applies that plan.

## Status and lineage boundaries

The canonical status schema is the sole writer-visible status authority.  A
successful contract-only execution state uses the canonical dimensions
`execution`, `contract`, `effect`, `evidence_maturity`, and `promotion`; it
does not imply that a scientific effect is estimable or promoted.  Alias keys,
missing keys, extra keys, coercions, and unknown enum values belong to the
future schema validator's negative matrix.

The execution identity is a prospective local research record.  It supports
later raw-to-derived lineage work but does not replace analysis identity,
effect-record admission, a preregistered estimand, independent-unit membership,
or held-out scientific validation.

## Frozen history and explicit non-goals

The historical v1 study registry and module are inspection-only and remain
frozen:

- `configs/idea_foundry.studies.v1.json`
- `quartz/idea_foundry/studies.py`

This PR must not mutate those files or authorize their mutation.  It must not
run or resume a campaign, create a real result, receipt, or evidence artifact,
or promote any claim.  In particular, it does not establish efficacy for a
single MCTS hypothesis or a compatible A/B pair, and it cannot establish that
`gain(A+B)` exceeds either solo gain.

## Acceptance boundary for the successor implementation

The successor implementation is eligible for local review only after focused
tests demonstrate all of the following observable behaviors:

- a clean captured identity and state derive from one immutable pair of
  configuration byte snapshots;
- dirty or incomplete local inputs fail before state publication;
- a duplicate local start is rejected while the first writer's flock is held;
- failed-to-successful retry removes incompatible failure-only state and passes
  the canonical writer validator before persistence; and
- every rejected resume preserves the complete prior forensic byte snapshot.

Passing those tests would establish only a local execution substrate at its
exact implementation SHA.  It would still not authorize a scientific run or a
claim about any MCTS axis, A/B combination, interaction, or superadditivity.
