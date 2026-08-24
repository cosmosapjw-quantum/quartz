# Idea Foundry PR04/PR05 failure log

**Status:** forensic log for external review
**Scope:** failed implementation attempts after accepted PR03 and before any
accepted PR05 successor
**Non-evidence statement:** these entries are process/debugging evidence only.
They are not campaign results, receipts, or scientific evidence.

This log preserves observed commands, failure signatures, and artifact
disposition without turning a failed implementation attempt into a successful
checkpoint.  Its summary and current status are in
[`PR00_PR05_PROGRESS_AND_FAILURE_AUDIT_20260825.md`](PR00_PR05_PROGRESS_AND_FAILURE_AUDIT_20260825.md).

## Evidence classes and limitation

Commit SHA, parent, changed-path, and worktree-reference statements below are
Git-verifiable.  PID, process-exit, temporary-directory teardown, and focused
pytest-count statements are 2026-08-24/25 operator-session observations.  The
raw terminal transcripts were not retained or tracked, so those observations
are recorded as audit context rather than independently replayable artifacts.
They do not support implementation acceptance, scientific evidence, or a claim
upgrade.

## 2026-08-24 — PR04 original candidate stopped

**Candidate:** `1cc3e6e508c3ad305f94a61d15e771e64b53ed4a`
**Parent:** `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3`
**Branch/worktree:** `agent/foundry-pr04-initial-state` /
`/tmp/quartz-idea-foundry-pr04-initial-state-20260824`
**Decision:** `STOP_INVALID`; not pushed.

### Review findings

| ID | Severity | Observed failure | Disposition |
|---|---|---|---|
| PR04-01 | P0 scope finding | The candidate touched `.writer.lock` but did not hold a real local `flock`. | Do not accept the candidate.  Later separation assigned the concrete `fcntl` resume-lock API to PR05 rather than silently expanding PR04. |
| PR04-02 | P0 | Sequential post-initial state writes bypassed the common state validator; the validator also excluded required runtime lifecycle checks. | Do not accept. |
| PR04-03 | P1 | A full sequential/resume integration test was skipped rather than replaced with an explicit deterministic synthetic boundary. | Do not accept. |

### Forbidden execution escape during repair

The repair restored the old test selected by:

```text
/home/cosmosapjw/Dropbox/personal_projects/quartz/venv/bin/python -m pytest -q \
  tests/test_idea_foundry_axis_workflows.py \
  -k full_sequential_campaign_and_resume_skip_validated_axes
```

Operator-session observation (raw terminal transcript not retained): parent PID
`478032`.

Before the bounded work was stopped, the operator observed this child process:

```text
PID 480098
/home/cosmosapjw/Dropbox/personal_projects/quartz/venv/bin/python \
  /tmp/quartz-idea-foundry-pr04-initial-state-20260824/scripts/idea_foundry/ \
  a26_nested_contour_exact_lab.py run-and-analyze \
  --output-dir /tmp/quartz-idea-foundry-pr04-initial-state-20260824/results/ \
  idea-foundry-test-9xdljixj/sequential-smoke/axes/A26/attempt-001 --seed 29
```

The operator observed this transient campaign root:

```text
results/idea-foundry-test-9xdljixj/sequential-smoke/
```

The operator observed transient `execution_identity.json` and
`campaign_state.json` files.  The root process was interrupted, the child had
already exited, and pytest's temporary-directory lifecycle removed the root;
PIDs were checked absent afterward.  These are unretained session observations,
not independently replayable artifacts.  No manual deletion, raw result
retention, receipt generation, analysis, promotion, or scientific
interpretation occurred.

**Classification:** `FORBIDDEN_ESCAPE`.  This incident is why future
foundation tests must not invoke a real campaign runner, axis script, GPU
program, or `run-and-analyze` command.

## 2026-08-25 — PR04 retry1 stopped at test collection

**Candidate:** `ee5d8bb47e3edc77cedc26157c6a4924a1661c82`
**Parent:** `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3`
**Branch/worktree:** `agent/foundry-pr04-initial-state-retry-20260825` /
`/tmp/quartz-idea-foundry-pr04-initial-state-retry-20260825`
**Decision:** `STOP_INVALID`; not pushed.

The registered PR04 focused command expected:

```text
tests/test_idea_foundry_campaign_state.py
```

but the candidate supplied a differently named path:

```text
tests/test_campaign_state.py
```

Consequently the exact registered command failed before collection.  The
writer's broader report did not replace that named acceptance boundary.  No
axis subprocess, campaign, result, receipt, or scientific execution occurred
in retry1.  The candidate was retained for inspection and not repaired in
place.

## 2026-08-25 — PR04 retry2 accepted only for its stated v2 state scope

**Accepted commit:** `9ddd3471904f2200f4656cb102886c994bd7bed0`
**Parent:** `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3`
**Branch:** `agent/foundry-pr04-initial-state-retry2-20260825`

This is not a failure entry, but it is recorded here to prevent confusion with
the two rejected PR04 candidates.  A fresh reviewer verified that the v2 new
path consumes captured descriptors and the v2 resume path reconstructs them
from persisted state.  The direct test patched
`sequential.load_workflow_specs` to fail and still passed.  Final focused
evidence was `22 passed, 47 deselected` with changed-file Ruff/format and diff
checks passing.  No axis runner or subprocess was invoked.

The acceptance does not cover legacy v1 inspection continuation, nor does it
establish PR05 resume behavior or `P0B-C1_ESTABLISHED`.

## 2026-08-25 — PR05 candidate stopped after adversarial review

**Candidate:** `55c21a7b42c93ffb886084d9868c3c1a41da5077`
**Parent:** `9ddd3471904f2200f4656cb102886c994bd7bed0`
**Branch/worktree:** `agent/foundry-pr05-resume-plan` /
`/tmp/quartz-idea-foundry-pr05-resume-plan-20260825`
**Decision:** `STOP_INVALID`; not pushed.

### P1 blockers

| ID | Finding | Exact consequence |
|---|---|---|
| PR05-01 | `build_resume_plan` appends retry `N+1` as running before public `run_campaign` evaluates resumability.  The public path then sees the running attempt as non-resumable and appends/runs `N+2`. | The `N+1` attempt is orphaned and the promised exactly-one retry contract is violated. |
| PR05-02 | Public rejected resume acquires `.writer.lock` by opening it with `a+b` before validation. | A rejection can create and leave a new zero-byte lock file, violating full-tree byte immutability. |

### P2 acceptance gaps

- Positive plan/apply tests did not snapshot all campaign files.
- The lock test snapshot began only after the lock file already existed.
- No test exercised the public `run_campaign(..., resume=True)` admission path,
  so neither P1 condition was observed by the candidate suite.
- The required foundation integration file
  `tests/test_idea_foundry_contracts_v2.py` was absent.

The operator-session focused test summary was `9 passed`; the raw terminal
transcript is not tracked.  This was not a campaign run and no axis script was
invoked, but the P1 findings are inside the registered acceptance behavior.
The scoped contract was therefore recorded as `STOP_INVALID` and deleted after
ordinary status reporting; no repair was made on the candidate.

## Current forensic disposition

```text
PR04_ORIGINAL=STOP_INVALID_UNPUSHED
PR04_RETRY1=STOP_INVALID_UNPUSHED
PR04_RETRY2=ACCEPTED_V2_INITIAL_STATE_SCOPE_ONLY
PR05_CANDIDATE=STOP_INVALID_UNPUSHED
P0B-C1=NOT_ESTABLISHED
RUN_RESUME=FORBIDDEN
SCIENTIFIC_EXECUTION=BLOCKED
```

The retained worktrees are evidence locations, not approved branch points.  A
new PR05 successor must branch from accepted `9ddd347...` and carry a newly
frozen bounded contract; it must not amend, push, or run either retained failed
candidate.
