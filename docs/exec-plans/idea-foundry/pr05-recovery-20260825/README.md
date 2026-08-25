# Idea Foundry PR05 Recovery — Audit-Compiled Package

## Canonical status

```text
AUDIT_CONTEXT_SHA=dadcc7187bca6a927b5ec5132aba845608611679
ACCEPTED_CODE_BASE=9ddd3471904f2200f4656cb102886c994bd7bed0
P0B-C1=NOT_ESTABLISHED
PR05_PREVIOUS_ATTEMPT=STOP_INVALID
RUN_RESUME=BLOCKED
SCIENTIFIC_EXECUTION=BLOCKED
RESULT_PUBLICATION=BLOCKED
CLAIM_PROMOTION=BLOCKED
```

This directory is the external-audit output for the PR00–PR05 recovery state.
It compiles the audit into executable contracts for Codex. It is not an
implementation, receipt, scientific result, or authorization to run or resume a
campaign.

The project is trusted-local, personal, single-developer research code.
Hostile-Git, malicious same-UID, mount/path attack, signatures, and anti-tamper
custody are out of scope. Accidental evidence mutation, stale identity,
interruption, duplicate local writers, silent wrong science, and unsupported
claim promotion remain in scope.

## Files

- `AUDIT_COMPILED_EXEC_PLAN.yaml` is the canonical machine-readable package. It
  contains the initiative authority, P0/P1 catalogue, invariant/test matrix,
  exact PR05-R1/R2/R3 contracts, post-PR05 lineage/effect/science DAG,
  adversarial scientific claim audit, fresh-context review contract, and
  evidence-bundle contract.
- `CODEX_HANDOFF_PROMPT.md` is a self-contained prompt that executes **PR05-R1
  only** from accepted PR04 SHA
  `9ddd3471904f2200f4656cb102886c994bd7bed0`.

## Primary blockers

1. **Public/private resume divergence.** Failed PR05 work targeted internal
   plan/apply helpers while the public `run_campaign(..., resume=True)` path
   retained mutable inline orchestration.
2. **Rejected-resume mutation.** The accepted public path can rewrite state and
   summary after prior-artifact validation fails.
3. **Retry validity.** A failed-to-success retry must append history, clear
   stale failure state, and publish only validator-admissible states.
4. **A26 process escape.** Nested `SystemExit(0)` must become a controlled
   technical failure, never owner-process success.
5. **Completion evidence.** P0b-C1 cannot close from implementer self-report;
   evidence validation and a fresh-context verdict-only review are required.

These are research-integrity blockers, not security-hardening requirements.

## Exact next sequence

```text
PR05-R1 public resume plan-before-mutate
  -> fresh-context verdict-only review
PR05-R2 nested process boundary
  -> fresh-context verdict-only review
PR05-R3 evidence verifier and P0b-C1 closeout
  -> contract-only P0b-C1 decision
PR06-R1 terminal raw closure
PR06-R2 analysis identity and transformation lineage
PR06-R3 EffectRecordV2 and meta-analysis admission
  -> versioned prospective solo-axis studies
```

A work unit whose dependency accepted SHA is null in the YAML package must stop
as `BLOCKED_BY_UNACCEPTED_DEPENDENCY`.

## Scientific adversarial verdict

No PR00–PR05 work earns a scientific effect.

- The historical 2026-08-18 A01 factorial records a descriptive
  resource-frontier evaluator-call reduction (42.06 versus 58.24, −27.78%). It
  does not establish fixed-compute efficacy, quality non-inferiority, held-out
  generalization, interaction, or promotion.
- Stage-7 H1 was better calibrated than historical P_flip on its trace bank,
  but matched-budget stop efficacy was insufficient. P_flip is not a calibrated
  low-budget probability.
- Stage-7 KG-stop was wired but effectively inactive at tested budgets.
- H3/O6 delivered no treatment (0/288 triggers), so no outcome effect is
  estimable.
- B13 has a promising decision-neutral KL signal, not a play-strength result,
  and requires prospective lineage-closed replication before combination work.
- The historical service curve is workload/hardware-specific and does not
  establish shipped-network scheduler efficacy.
- Adaptive-VL duplicate-reduction rationale is killed; A13 may not reuse it.
- A13 and A16 are candidate mechanisms until real-engine differential studies.
- A19 v1 is a frozen proxy; a new held-out v2 estimand is required.
- A18 state-disjointness is not trajectory/family generalization.

The finding-by-finding classification, allowed claim, forbidden claims, risk,
and next gate are under `scientific_claim_audit` in the YAML package.

## Acceptance principle

Every foreseeable P0/P1 must have at least one of:

1. an executable regression, negative, differential, property, or metamorphic
   test;
2. a mechanically checked invariant or assertion;
3. a named STOP/BLOCKED gate when mechanical verification is impossible.

A prose warning alone is not coverage. The implementer never self-accepts. The
fresh reviewer receives only base SHA, final SHA, diff, selected contract,
evidence bundle, and command logs; the first pass is verdict-only and does not
modify code.
