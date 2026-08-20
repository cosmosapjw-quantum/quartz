# P0b-C1 recovery and harness checkpoint — 2026-08-20

## Authority and publication boundary

- Recovery baseline: `d0f0fb237a0b8b05b907513b6be2b32335403144`.
- Clean P0b-C1 specification anchor:
  `914169e02c04a225f34a4fb3d17ae3486f424193`.
- Implementation PR base:
  `292d0336b246f990cb0fc7f99ef11f464d506219`.
- This checkpoint publishes reviewed recovery history and a specification. It
  does not publish an accepted P0b-C1 implementation.
- This archival checkpoint commit is not an implementation base. Future P0b-C1
  work must branch from `914169e`, not from this publication head.
- The source-current scientific authority remains unset. The 2026-08-07,
  2026-08-17, and 2026-08-18 outputs remain historical internal diagnostics.

## Included recovery history

| Head | Recovery slice | Status |
| --- | --- | --- |
| `dd0e03a` | G0a CI portability | reviewed |
| `040cf79` | G0b hermetic fixtures | reviewed |
| `f1ab500` | S0a status round-trip | reviewed |
| `fc0b2e8` | C0 historical claim quarantine | reviewed |
| `09d934b` | P0a literal Git-tree receipt binding | reviewed |
| `d0f0fb2` | P0b-L legacy scientific-route freeze | reviewed |
| `292d033` | strict duplicate/non-finite JSON admission | reviewed |
| `914169e` | P0b-C1 execution-seal specification | specified only |

The reviewed labels above are scoped to their respective implementation and
contract checks. They do not establish scientific efficacy, release readiness,
or promotion eligibility.

## Rejected and non-durable material

- `10c34c112717a9a57b0506422f1cbd272a83bbd0` is a rejected salvage
  implementation. Independent R0 review reproduced run-root publication races,
  incoherent Git/live capture, alias acceptance, and permissive schema-v2 state
  validation. Its local Fix-1 test draft is not acceptance evidence.
- `20548e1fc0a959184da5f633b505cd345cad4324` is a rejected private-payload
  candidate and is not part of this branch.
- C1A produced an untracked tests-only draft and no production candidate.
- C1A1 produced no tracked implementation change.
- The forensic root at `65e7d3e9596ca259ebfe523b0c74058692b6a755`
  retains four tracked and three untracked user paths; none are included here.

## Harness meta-audit

The audited `CODEX_BOUNDED_WORK_HARNESS_20260813.zip` has SHA-256
`0e6b5f9270e0debacea1864ce57112766a7a762972855f08c01754aa6480d4e0`.
Its archive paths are safe, its bundled manifest verifies 9/9 payloads, and its
20 package tests pass. The current global install matches those runtime bytes.

The harness is advisory process control, not a trust or proof anchor:

- the bundled manifest does not authenticate a publisher;
- the installer does not itself enforce `SHA256SUMS`;
- installation is not transaction-wide and does not preserve arbitrary
  `AGENTS.md` bytes or mode exactly;
- ancestor symlinks are not rejected;
- the contract validator accepts duplicate members, `schema_version: true`,
  unregistered non-finite metadata, and an unproven
  `COMPLETE_ACCEPTANCE` declaration.

Therefore harness `check` establishes only bounded declarative consistency.
It cannot establish that evidence commands ran or that scientific claims hold.

## Corrected process and claim state

The aggregate 600-line cap made specifications and necessary tests compete
with production complexity. Because frozen scope was structurally estimated at
671–791 readable lines, the correct classification is:

`POLICY_MISFIT -> STALL_PROCESS_ACCRETION -> CHECKPOINT -> STOP`.

The rejected implementation report's `PASS/success/passed` labels are not
controlling. Current P0b-C1 status is:

`PARTIAL / REVIEW_FAILED / CONTRACT_NOT_ESTABLISHED /`
`SCIENTIFIC_EVIDENCE_NOT_EARNED / PROMOTION_INELIGIBLE`.

## One-time recalibrated control

- Resume only the original cohesive P0b-C1 task from clean `914169e`.
- Keep the original six-path implementation surface:
  `P0B_C1_EXECUTION_SEAL_SPEC.md`, `axis_workflow.py`,
  `execution_seal.py`, `sequential.py`, and the two focused test modules.
- Hard-limit production/runtime churn to `M <= 360`.
- Keep the specification at its committed 71 lines.
- Do not hard-cap necessary tests by LOC. Tests may cover only the frozen
  acceptance matrix; 380–430 lines is telemetry, not a target.
- Treat aggregate churn as telemetry. Keep at most six files and four commits
  from the recovery baseline.
- Use focused RED/GREEN and the smallest affected suites.
- Permit one independent adversarial review and at most one repair-closeout.
- Do not create another split, waiver, gate, budget recalibration, all-axis
  run, factorial run, GPU run, receipt, or scientific claim.

## Resume pointer

The next work unit must reconstruct the cohesive P0b-C1 implementation from
`914169e` under the control above. P0b-C2 and scientific execution remain
blocked until that implementation passes its contract review.
