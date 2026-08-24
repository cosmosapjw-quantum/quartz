# Idea Foundry PR00–PR05 progress and failure audit

**Snapshot date:** 2026-08-25
**Review base:** `9ddd3471904f2200f4656cb102886c994bd7bed0`
**Snapshot kind:** external-audit handoff; documentation only
**Implementation authority:** the trusted-local policy and the accepted PR chain
below.  This document is not an implementation authority, execution receipt, or
scientific result.

## Executive status

| Subject | Status | What this means |
|---|---|---|
| Trusted-local operating policy | IMPLEMENTED policy | QUARTZ is owner-operated MCTS research; security-only hardening is out of scope, while accidental-drift and scientific-integrity controls remain required. |
| PR00–PR04 components | IMPLEMENTED with focused checks | The accepted chain supplies status-writer, captured-registry, execution-identity, and initial-state substrates. |
| PR05 resume plan | STOP_INVALID, unpushed | The candidate has independent P1 defects and is retained only for forensic inspection. |
| `P0B-C1_ESTABLISHED` | **NOT ESTABLISHED** | The required validation-first, immutable public resume behavior is not yet accepted. |
| `run` / `resume` | **FORBIDDEN** | No branch in this audit authorizes campaign execution or resumption. |
| Scientific execution, result, receipt, promotion | **BLOCKED** | No axis, pair, interaction, or efficacy claim follows from this work. |

The controlling policy is
[`RESEARCH_TRUST_MODEL_20260823.md`](RESEARCH_TRUST_MODEL_20260823.md); the
machine-oriented plan is
[`../plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md`](../plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md).
The P0b-C1 interface boundary remains
[`P0B_C1_TRUSTED_LOCAL_SPEC.md`](P0B_C1_TRUSTED_LOCAL_SPEC.md).  The claim
firewall records the same current conclusion in
[`../CLAIM_LEDGER.md`](../CLAIM_LEDGER.md).

## Research purpose and non-inference boundary

Idea Foundry is a research substrate for the 26 registered MCTS hypotheses and
preregistered compatible combinations.  A later study may compare a common
baseline with A-only, B-only, and A+B, including the exploratory question of
whether `gain(A+B)` exceeds each solo gain.  A formal interaction or
superadditivity claim still requires its own preregistered estimand, factorial
contrast, independent units, held-out target, and fixed compute/runtime
contract.

PR00–PR05 concerns only local execution identity and recovery-state plumbing.
It does **not** measure an MCTS axis, a combination, throughput, quality,
calibration, interaction, or superadditivity.  No GPU campaign, result,
receipt, or promotion was created by the accepted chain.

## Accepted progress chain

Each row is a single accepted commit on the parent shown.  Listed test results
are contemporaneous focused evidence, not an assertion that the full repository
or remote CI is green.  Their raw terminal logs are not tracked in this audit
snapshot; the counts are retained as session summaries only.

| PR | Branch and commit | Parent | Observable contribution | Focused evidence and review disposition |
|---|---|---|---|---|
| PR00 | `agent/foundry-pr00-lean-spec` at `9da35b4d6189c38ddfdc30949605765ab0816e11` | `37535065776d4babb0186d21370e3ddd7b4217de` | Added the lean trusted-local P0b-C1 specification; retained the security-heavy predecessor as forensic-only history. | Documentation and policy review; does not implement P0b-C1. |
| PR01 | `agent/foundry-pr01-status-writers` at `d4998d35ddd155856eaa9367d8198d09850348cd` | `9da35b4d6189c38ddfdc30949605765ab0816e11` | Made writer-context status validation explicit in `status_schema.py`. | Repair restored terminal-only reader semantics after review found that analysis could admit running/failed campaign status. Final focused checks: `12 passed, 32 deselected`; meta terminal/manifest nodes: `3 passed`; changed-file Ruff and format passed. |
| PR02 | `agent/foundry-pr02-registry-snapshot` at `02ff780fe493b74a0e5f353f75cab4c525838665` | `d4998d35ddd155856eaa9367d8198d09850348cd` | Parses workflow specifications from captured axis/lab JSON bytes, preserving the no-argument compatibility path. | Repair moved the no-reread test to the real `Path.read_bytes` boundary. Final focused checks: `37 passed, 20 deselected`; changed-file Ruff and format passed. |
| PR03 | `agent/foundry-pr03-execution-identity` at `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3` | `02ff780fe493b74a0e5f353f75cab4c525838665` | Added trusted-local `ExecutionIdentity` capture and strict serialized identity validation. | Repair rejected non-exact schema, dirty identity, invalid/short source inventory, non-canonical paths, negative sizes, and non-SHA digests. Final focused checks: `6 passed`; changed-file Ruff and format passed. |
| PR04 | `agent/foundry-pr04-initial-state-retry2-20260825` at `9ddd3471904f2200f4656cb102886c994bd7bed0` | `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3` | Added identity-bound campaign state creation.  The v2 new path derives descriptors from capture; the v2 resume path reconstructs descriptors from persisted state. | Final fresh review found no P0/P1 blocker for the **v2 initial-state scope**. Focused checks: `22 passed, 47 deselected`; changed-file Ruff/format and `git diff --check` passed. No axis runner or subprocess was invoked in retry2. |

The PR04 reviewer expressly limited its conclusion to the v2 initial-state
path.  Legacy v1 inspection continuation may still read a live registry and is
not evidence for a v2 resume contract.

## Retained failed candidates

These commits are deliberately not ancestors of the review base and were not
pushed as accepted implementation branches.  They are retained in local
forensic worktrees so an auditor can inspect the exact diff; they must not be
salvaged in place or used as a run/resume base.

| Candidate | Parent | Verdict | Reason |
|---|---|---|---|
| PR04 initial candidate `1cc3e6e508c3ad305f94a61d15e771e64b53ed4a` | `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3` | `STOP_INVALID` | Review found a missing real lock, state writes that bypassed validation, and an integration test replaced by a skip.  Its repair also caused the forbidden A26 test-process escape described below. |
| PR04 retry1 `ee5d8bb47e3edc77cedc26157c6a4924a1661c82` | `1976c7f9e108c673fc7ed5131ccf1533a0c23ba3` | `STOP_INVALID` | The registered PR04 focused command failed before collection because the declared `tests/test_idea_foundry_campaign_state.py` was absent; the candidate named a different test path.  No campaign or axis subprocess ran in this retry. |
| PR05 resume candidate `55c21a7b42c93ffb886084d9868c3c1a41da5077` | `9ddd3471904f2200f4656cb102886c994bd7bed0` | `STOP_INVALID` | Public resume can create an orphan retry attempt and rejected public resume can create `.writer.lock`; both violate the intended immutable rejection/retry contract.  Required public admission and foundation integration coverage is absent. |

The chronological command-level evidence and artifact disposition are preserved
in [`FAILURE_LOG_PR04_PR05_20260825.md`](FAILURE_LOG_PR04_PR05_20260825.md).

## Exact failure boundary: PR04 A26 process escape

One rejected PR04 repair restored an old full sequential test.  Its pytest
process started this child command before the work unit was stopped:

```text
.../venv/bin/python scripts/idea_foundry/a26_nested_contour_exact_lab.py \
  run-and-analyze --output-dir .../results/idea-foundry-test-9xdljixj/ \
  sequential-smoke/axes/A26/attempt-001 --seed 29
```

This was a **FORBIDDEN_ESCAPE**, not a scientific experiment.  The run root was
inside pytest's temporary directory and was removed by `TemporaryDirectory`
teardown after graceful test-process termination; it was not manually deleted.
It yielded no retained result, receipt, evidence artifact, or scientific claim.
The operator-session PID, root, parent command, and disposition are recorded in
the failure log; the raw terminal transcript was not retained or tracked.

## PR05 blocker and next authority

PR05 is not repairable by quietly changing the retained candidate.  A successor
work unit must start from accepted PR04 `9ddd347...`, write a revised bounded
specification for the resume scope, and then implement only that new contract.
At minimum it must prove through the public `run_campaign(..., resume=True)`
path that:

1. a rejected resume leaves every pre-existing campaign byte unchanged;
2. exactly one retry attempt is created and applied for one failed axis;
3. validation completes before retry-state mutation;
4. the affected foundation/integration test is present and runs a deterministic
   synthetic boundary, not an axis script; and
5. `P0B-C1_ESTABLISHED` is not declared until the successor's focused gates and
   one fresh adversarial review close without blocker.

This is a next-action boundary, not permission to resume campaigns.

## Audit reproduction commands

The following commands inspect only Git objects and the locally retained
worktrees.  They do not run a campaign or mutate an implementation candidate.

```bash
git show -s --format='%H%n%P%n%s' 9da35b4d6189c38ddfdc30949605765ab0816e11
git show -s --format='%H%n%P%n%s' d4998d35ddd155856eaa9367d8198d09850348cd
git show -s --format='%H%n%P%n%s' 02ff780fe493b74a0e5f353f75cab4c525838665
git show -s --format='%H%n%P%n%s' 1976c7f9e108c673fc7ed5131ccf1533a0c23ba3
git show -s --format='%H%n%P%n%s' 9ddd3471904f2200f4656cb102886c994bd7bed0
git show -s --format='%H%n%P%n%s' 1cc3e6e508c3ad305f94a61d15e771e64b53ed4a
git show -s --format='%H%n%P%n%s' ee5d8bb47e3edc77cedc26157c6a4924a1661c82
git show -s --format='%H%n%P%n%s' 55c21a7b42c93ffb886084d9868c3c1a41da5077
git worktree list --porcelain
```

## Claim audit

| Claim | Status | Evidence | Risk | Required fix |
|---|---|---|---|---|
| PR00–PR04 provide portions of a trusted-local execution substrate. | IMPLEMENTED with focused component checks | Accepted commit chain and summarized contemporaneous focused-check outcomes in this document; raw test logs are not tracked in this audit snapshot | Calling components a completed P0b-C1 contract | Complete a successor PR05 from `9ddd347...`. |
| PR05 establishes immutable public resume. | FORBIDDEN | P1 findings in the retained candidate; failure log | An unpushed candidate might be mistaken for accepted code | Preserve it as forensic-only and replace it through a new bounded work unit. |
| The chain validates an MCTS hypothesis, solo gain, A+B gain, interaction, or superadditivity. | FORBIDDEN | No authorized scientific execution or analysis was performed | Execution plumbing could be misread as efficacy evidence | Require separately preregistered research designs and authorized runs. |

No claim is upgraded by this audit snapshot.
