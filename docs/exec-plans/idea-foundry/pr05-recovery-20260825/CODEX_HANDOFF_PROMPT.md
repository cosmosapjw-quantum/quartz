# Codex Handoff Prompt — Execute PR05-R1 Only

Use the text block below in a fresh Codex session. Do not append the previous
implementer's private reasoning, explanation, or self-assessment.

```text
You are the implementation agent for QUARTZ Idea Foundry PR05-R1.

PRIMARY RULE
Conform exactly to the audit-compiled contract. Do not redesign the resume
semantics. Make every declared P0/P1 fail closed.

REPOSITORY
https://github.com/cosmosapjw-quantum/quartz

AUDIT PACKAGE BRANCH
audit/idea-foundry-progress-and-failure-log-20260825

AUDIT PACKAGE PATH
docs/exec-plans/idea-foundry/pr05-recovery-20260825/AUDIT_COMPILED_EXEC_PLAN.yaml

ACCEPTED CODE BASE
9ddd3471904f2200f4656cb102886c994bd7bed0

TARGET BRANCH
agent/idea-foundry-pr05-r1-public-resume

DO NOT ASK USER QUESTIONS.
DO NOT GUESS ACROSS A SPECIFICATION BOUNDARY.
If repository sources do not uniquely determine a required semantic, stop with
a declared BLOCKED code and exact evidence. Do not choose a plausible answer.

THREAT MODEL
Trusted-local, single-developer research code. Do not add hostile Git,
malicious-symlink or mount, cryptographic anti-tamper, multi-tenant, or
malicious same-UID defenses. Retain accidental-concurrency, interruption,
stale-identity, evidence-preservation, and scientific-correctness controls.

STEP 0 — FETCH AND VALIDATE THE CONTRACT

git fetch origin audit/idea-foundry-progress-and-failure-log-20260825
AUDIT_PACKAGE_SHA="$(git rev-parse origin/audit/idea-foundry-progress-and-failure-log-20260825)"
mkdir -p /tmp/quartz-pr05-r1-contract

git show \
  "$AUDIT_PACKAGE_SHA:docs/exec-plans/idea-foundry/pr05-recovery-20260825/AUDIT_COMPILED_EXEC_PLAN.yaml" \
  > /tmp/quartz-pr05-r1-contract/package.yaml

python - <<'PY'
from pathlib import Path
import yaml

path = Path('/tmp/quartz-pr05-r1-contract/package.yaml')
package = yaml.safe_load(path.read_text())
assert package['schema'] == 'audit-compiled-exec-package/v1'
assert package['accepted_code_base_sha'] == '9ddd3471904f2200f4656cb102886c994bd7bed0'
assert package['contracts']['PR05-R1']['base_sha'] == '9ddd3471904f2200f4656cb102886c994bd7bed0'
assert package['current_status']['P0B_C1'] == 'NOT_ESTABLISHED'
assert package['current_status']['scientific_execution'] == 'BLOCKED'
print('CONTRACT_PARSE_PASS')
PY

Read the complete package sections before editing:

- agent_policy
- p0_p1_catalogue
- invariant_test_matrix
- contracts.PR05-R1
- fresh_review_contract
- evidence_bundle_contract

STEP 1 — CREATE AN ISOLATED BRANCH

First inspect the current worktree:

git status --short

Never reset, clean, stash, overwrite, or delete user work.

Use an isolated worktree:

git fetch origin
git worktree add ../quartz-pr05-r1 9ddd3471904f2200f4656cb102886c994bd7bed0
cd ../quartz-pr05-r1
git switch -c agent/idea-foundry-pr05-r1-public-resume

Verify:

test "$(git rev-parse HEAD)" = "9ddd3471904f2200f4656cb102886c994bd7bed0"
test -z "$(git status --porcelain=v1)"

Failure is BLOCKED_BY_WRONG_BASE_SHA or BLOCKED_BY_DIRTY_WORKTREE.

STEP 2 — EXECUTE EVERY PRECONDITION

Read contracts.PR05-R1 from the package. Run every precondition exactly.
Preserve command, stdout, stderr, and exit code.

Record SHA-256 for these frozen files before editing:

sha256sum \
  configs/idea_foundry.studies.v1.json \
  quartz/idea_foundry/studies.py

Do not continue after a baseline failure.

STEP 3 — TDD RED THROUGH THE PUBLIC API

Create only:

tests/test_idea_foundry_resume.py

Add every public test named by contracts.PR05-R1 and the invariant matrix. Tests
must invoke run_campaign with resume=True and, where specified, the CLI.
Internal helper-only tests cannot close PR05-R1.

Required cases:

1. invalid prior artifact -> every pre-existing run-root byte unchanged;
2. RUNNING, SUCCESS, legacy, or identity-drift resume -> byte unchanged;
3. captured registry bytes are used, not a live reread;
4. failed attempt 1 -> successful retry 2 preserves attempt 1;
5. stale axis failure_reason is absent in retry RUNNING/SUCCESS state;
6. stale or duplicate ResumePlan cannot apply;
7. public API calls plan_resume and apply_resume_plan exactly once.

Run:

./venv/bin/python -m pytest -q tests/test_idea_foundry_resume.py

Expected: behavior-specific assertion failures. Import, collection, fixture, or
syntax failures are invalid RED evidence.

STEP 4 — IMPLEMENT THE MINIMUM CONTRACT

Allowed changes are exactly:

CREATE quartz/idea_foundry/resume.py
MODIFY quartz/idea_foundry/sequential.py
CREATE tests/test_idea_foundry_resume.py

Implement the classes, signatures, ordered steps, invariants, and forbidden
changes exactly as specified by contracts.PR05-R1.

Mandatory semantics:

1. plan_resume validates the entire eligibility boundary without filesystem
   mutation.
2. Only inactive FAILED campaigns are eligible.
3. Prior terminal artifacts are validated before next_state is built.
4. State is derived from captured config/registry bytes, never a live reread.
5. next_state changes campaign to RUNNING, adds resumed_at, marks only validated
   terminal-prefix rows verified_skip, clears stale axis failure_reason, and
   appends exactly one active retry.
6. Existing attempt dictionaries remain equal to their source values.
7. apply_resume_plan uses a simple trusted-local single-writer mechanism,
   rechecks the source-state digest, validates next_state, and publishes once.
8. Public run_campaign with resume=True has no alternative inline mutation.
9. The loop launches the active retry already in the plan and does not append a
   duplicate attempt.
10. Every durable state write is preceded by canonical state validation.

Do not add security hardening outside this list.

STEP 5 — GREEN AND MUTATION-BACKED REGRESSION

Run every targeted, negative, regression, and static command in the contract.

Then prove the regression detects the original defect:

A. fixed implementation -> named regression PASS;
B. temporarily restore or bypass the old early resume mutation -> regression
   MUST FAIL;
C. restore fixed implementation -> regression PASS.

Do not commit the temporary mutation. Record all three command outputs and exit
codes.

STEP 6 — SCOPE AND FROZEN FILES

Run:

git diff --name-only 9ddd3471904f2200f4656cb102886c994bd7bed0...HEAD
git diff --check
sha256sum \
  configs/idea_foundry.studies.v1.json \
  quartz/idea_foundry/studies.py

Changed paths must be exactly the three allowed paths. Frozen hashes must equal
the precondition values.

STEP 7 — EVIDENCE BUNDLE

Create `/tmp/quartz-pr05-r1-evidence.json` with every field required by
`evidence_bundle_contract` in the package. Include exact command logs,
invariant results, mutation results, frozen-file hashes, skipped-gate reasons,
and unresolved blockers.

These values are mandatory:

"scientific_runs_executed": false
"results_or_receipts_created": false
"claim_status_change": "none"

An incomplete bundle is BLOCKED_BY_EVIDENCE_INCOMPLETE.

STEP 8 — COMMIT AND PUSH

Only after all required gates pass:

git add \
  quartz/idea_foundry/resume.py \
  quartz/idea_foundry/sequential.py \
  tests/test_idea_foundry_resume.py

git commit -m "fix(foundry): make public resume plan-before-mutate"
git push -u origin agent/idea-foundry-pr05-r1-public-resume

Do not create or merge a GitHub PR.
Do not mark P0B-C1 established.
Do not execute a scientific campaign.

FINAL RESPONSE FORMAT

PR_ID: PR05-R1
BRANCH:
BASE_SHA:
FINAL_SHA:
AUDIT_PACKAGE_SHA:
FILES_CHANGED:
OBSERVABLE_BEHAVIOR:
TEST_COMMANDS:
TEST_RESULTS:
MUTATION_REGRESSION_RESULTS:
RUFF_RESULTS:
DIFF_CHECK:
FROZEN_FILE_HASHES:
EVIDENCE_BUNDLE_PATH:
SCIENTIFIC_RUNS_EXECUTED: false
RESULTS_OR_RECEIPTS_CREATED: false
CLAIM_STATUS_CHANGE: none
UNRESOLVED_BLOCKERS:
NEXT_ACTION: FRESH_CONTEXT_VERDICT_ONLY_REVIEW

Do not claim completion or repository-wide health unless the recorded evidence
supports that exact statement.
```
