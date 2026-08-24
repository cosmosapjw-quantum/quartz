# Idea Foundry trusted-local research policy

## Authority and purpose

This policy records the project owner's 2026-08-23 decision for QUARTZ Idea
Foundry.  QUARTZ is a personal MCTS research project operated by one trusted
developer on a trusted local machine.  It is not a multi-tenant service, a
hostile-input package, or a security product.

This policy supersedes the adversarial security and deliberate-tampering threat
model in P0b-C1 specification commit
`497965bb204922387755f2d6050042a3ddda6f0c`.  That specification remains a
read-only forensic record and is not an implementation authority after this
policy commit.  A smaller successor specification is required before P0b-C1
implementation resumes.

The high-level staged research roadmap is
[`../plans/2026-08-24-idea-foundry-long-term-development.md`](../plans/2026-08-24-idea-foundry-long-term-development.md).
The controlling machine-oriented audit and stacked-PR execution plan is
[`../plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md`](../plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md).
When the overview and the machine plan differ in implementation detail, the
machine plan controls; neither document authorizes a scientific run by itself.

## Trusted operating model

The default operating assumptions are:

- the repository, local Git configuration, Python interpreter, filesystem, and
  experiment inputs are controlled by the single project owner;
- no malicious same-UID process, hostile Git configuration, attacker-controlled
  symlink/path replacement, mount-namespace attack, or deliberately modified
  installer/runtime is in scope;
- ordinary mistakes, process interruption, accidental duplicate invocation,
  stale inputs, and incomplete result writes remain realistic failure modes;
- external collaborators, shared services, untrusted packages, or remote users
  require a separately approved threat-model change before they are admitted.

Consequently, new work must not add anti-tamper ownership manifests, hostile
configuration allowlists, adversarial mount/path proofs, security-specific
installer state machines, or security review gates unless the owner explicitly
requests them for a named task.

## Research integrity that remains required

The trusted-local model does not relax scientific discipline.  The following
controls are research-method requirements rather than security features:

- bind each scientific run to the clean committed source, configuration,
  interpreter/runtime identity, and declared inputs that generated it;
- preserve deterministic seeds, treatment definitions, independent-unit
  membership, raw results, negative outcomes, and failed runs;
- use atomic publication where an interrupted write could be mistaken for a
  complete state or result;
- make resume idempotent enough to avoid duplicate attempts or rewriting prior
  evidence;
- separate execution identity from later analysis identity and record the
  transformation when they differ;
- keep contract, diagnostic, ablation, and confirmatory evidence maturity
  distinct, and never promote a claim from a passing smoke or provenance check.

Hashes and clean-tree checks are retained only to identify the research object
and detect accidental drift.  They do not assert resistance to a malicious
owner or compromised machine.

## P0b-C1 successor boundary

The next P0b-C1 specification must be limited to the smallest observable
research behavior:

1. a new run records one clean committed source/config/input/runtime identity;
2. state is validated before publication and derives from the captured inputs;
3. interrupted or rejected resume does not rewrite prior attempts or evidence;
4. duplicate local invocation is rejected with a simple no-overwrite or
   single-writer mechanism sufficient for accidental concurrency;
5. focused tests for the changed behavior and its declared dependents pass on
   the exact implementation commit.

The successor must not carry forward security-only filesystem mount probes,
complete symlink-edge transcripts, hostile local-Git configuration defenses,
anti-tamper ownership manifests, or adversarial installer rollback machinery.
Any retained mechanism must name the accidental scientific error it prevents.

## Verification scope policy

Changed-file and changed-cell checks are the default.  Each work unit runs the
narrowest test nodes that exercise its observable behavior, the affected
integration boundary, formatting/lint for changed code, and `git diff --check`.
An unchanged passing cell may carry its existing evidence forward.

A full Python or Rust suite is not a routine completion requirement.  Run one
only when the reason is recorded before execution and at least one of these
conditions holds:

- a shared public contract changed and its dependents cannot be bounded by the
  focused dependency map;
- a release, merge integration, or toolchain/dependency upgrade needs a broad
  regression check;
- the repository is at the final preflight for an explicitly authorized,
  claim-bearing scientific execution; or
- a focused failure supplies concrete evidence of a broader regression.

Documentation-only changes use diff, link-target, and targeted content checks;
they do not require Python or Rust test suites.  Skipping a full suite under
this policy is a declared scope choice, not evidence that the full repository
is green.

## Current status

```text
TRUST_MODEL=TRUSTED_LOCAL_SINGLE_DEVELOPER_RESEARCH
SECURITY_HARDENING=OUT_OF_SCOPE
DELIBERATE_TAMPER_RESISTANCE=OUT_OF_SCOPE
RESEARCH_REPRODUCIBILITY=REQUIRED
CLAIM_FIREWALL=REQUIRED

497965b_SECURITY_HEAVY_SPEC=SUPERSEDED_FORENSIC_ONLY
H0_HARNESS_V2_CANDIDATE=REJECTED_NOT_INSTALLED
P0B-C1=NOT_ESTABLISHED
RUN_RESUME=FORBIDDEN_PENDING_LEAN_SPEC
SCIENTIFIC_EXECUTION=BLOCKED
```

The rejected H0 source candidate and its failed review remain historical
diagnostic material.  They must not be installed or reclassified as accepted
under this policy; a future harness change, if still needed, must be a new
bounded work unit whose acceptance contains only process-budget behavior.

## Research objective and claim boundary

Idea Foundry studies the 26 registered MCTS hypotheses and preregistered
compatible combinations.  Combination work asks whether `A+B` improves the
common baseline and exceeds both `A`-only and `B`-only gains.  This comparison
does not by itself establish formal interaction or superadditivity; those words
require a prespecified endpoint, effect scale, factorial contrast, independent
unit, held-out target, and fixed compute/runtime contract.

No policy, provenance check, execution seal, or hardware upgrade is scientific
evidence for an axis or combination.
