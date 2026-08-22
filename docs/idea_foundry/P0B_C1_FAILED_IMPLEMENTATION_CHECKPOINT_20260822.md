# P0b-C1 failed implementation checkpoint — 2026-08-22

## Purpose and verdict

This document is the concise external-auditor entry point for the first
implementation attempt against the corrected P0b-C1 execution-seal
specification. The draft is preserved because its failure state is useful
forensic evidence. It is not an accepted implementation or an execution base.

```text
ARCHIVAL_CHECKPOINT
IMPLEMENTATION_NOT_ESTABLISHED
REVIEW_NOT_REACHED
SCIENTIFIC_EVIDENCE_NOT_EARNED
PROMOTION_INELIGIBLE
STOP_BUDGET
```

`REVIEW_NOT_REACHED` describes implementation review: the candidate did not
earn a reviewable green state. A separate read-only review of this archival
publication may verify only that the checkpoint is accurate and does not
overclaim.

## Identity and authority

- Corrected specification anchor:
  `83095c9742d779c3f2fc95a8021fa7bb4ab95338`
- Anchor parent: `292d0336b246f990cb0fc7f99ef11f464d506219`
- Working branch:
  `audit/idea-foundry-p0bc1-implementation-20260822`
- Intended remote:
  `origin/audit/idea-foundry-p0bc1-implementation-20260822`
- Controlling specification:
  `docs/idea_foundry/P0B_C1_EXECUTION_SEAL_SPEC.md`
- The historical `914169e` and `0bcbdbf` checkpoints remain archival; rejected
  `10c34c1` and `20548e1` candidates are not implementation bases.
- The dirty `agent/local-experiment-foundry@65e7d3e` worktree was not stashed,
  reset, cleaned, staged, or included in this branch.

## Hardware snapshot and research-scale consequence

Live measurements on 2026-08-22:

| Resource | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, UUID `GPU-c6936482-f543-45b4-ce0d-905ef7c0da18` |
| VRAM | 24,576 MiB reported by `nvidia-smi`; 25,290,604,544 bytes reported by PyTorch |
| Driver/runtime | driver 595.84; PyTorch 2.11.0+cu128; CUDA runtime 12.8 |
| GPU topology | compute capability 8.6; 82 SMs; PCI `00000000:09:00.0`; 330 W limit |
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 threads |
| RAM/swap | 98,770,044 kB RAM (96 GB physical class); 75,497,464 kB swap |

This is a capacity update only. Relative to the previous 16 GB VRAM / 64 GB
RAM planning baseline, memory capacity is about 1.5 times larger. A provisional
working envelope of 18–20 GiB GPU allocation and 60–64 GiB resident RAM is
reasonable for future preflight design, but it is not throughput evidence and
does not authorize larger game, seed, family, or time budgets. The workspace
instruction snapshot active during this session still described the former
machine, but `AGENTS.md` is absent from the corrected-spec Git tree and is not
part of this branch. No operational hardware configuration was changed. Any
future scientific execution requires a clean committed tree, a fresh hardware
preflight, and separate authorization.

## Preserved draft

The draft attempts to add a prospective execution-seal boundary, strict JSON
decoding, schema-v2 state binding, exclusive run-root acquisition, durable seal
publication, pre/post live capture equality, and pre-mutation resume checks.
None of those mechanisms is accepted by this checkpoint.

Production churn from the corrected specification anchor is:

| Path | Additions | Deletions | Churn |
| --- | ---: | ---: | ---: |
| `quartz/idea_foundry/axis_workflow.py` | 10 | 5 | 15 |
| `quartz/idea_foundry/sequential.py` | 129 | 20 | 149 |
| `quartz/idea_foundry/execution_seal.py` | 250 | 0 | 250 |
| **Production total M** | **389** | **25** | **414** |

The frozen hard gate was `M <= 360`; therefore the draft stopped at
`STOP_BUDGET`. The focused test module adds 683 lines. Test size is telemetry,
not the reason for the stop.

Exact file-byte hashes before archival commit:

```text
5f2564826f9b18222f72ce54593d491d838f91997ffa69caf60f55fa4080e7e5  quartz/idea_foundry/axis_workflow.py
4910574f057718b3659b7b0ddd86dab375cc4cdf95bfed43f06f80a6fb1bce46  quartz/idea_foundry/sequential.py
eec896623306208e82d29bea2b469968050dba0f812caea93b28a31b52586cff  quartz/idea_foundry/execution_seal.py
81703f8a52f95942402cf58deaef37968f54f78bb076324a323572bb9a6666dd  tests/test_idea_foundry_execution_seal.py
```

## Verification evidence

Pre-edit local baseline at the corrected specification anchor:

```text
focused axis-workflow suite: 37 passed in 66.55s
full Python suite: 1037 passed, 12 skipped, 3 subtests passed in 92.20s
```

These are local baseline observations, not a remote CI attestation. At an
earlier formatted `M=360` draft, the frozen focused suite reported 45 passed and
30 failed. After the final bounded integration attempt produced `M=414`, the
exact preserved draft was rerun on 2026-08-22:

```text
62 passed, 13 failed in 13.99s
```

The 13 failures comprise:

- one hostile-Git-environment test whose comparison helper inherits the hostile
  environment and exits 128;
- eleven existing state-validation cases that pass a `Path` where the new
  strict-JSON boundary rejects non-built-in JSON values; and
- one sequential smoke test that requests a clean execution seal from the
  intentionally dirty draft worktree.

Additional checks on the exact draft:

```text
git diff --check (two tracked draft files): PASS
git diff --no-index --check (three new files): no whitespace diagnostics;
  exit 1 is expected because each new file differs from /dev/null
ruff format --check (four changed Python files): PASS
ruff check (four changed Python files): FAIL, 17 E701/E702/E731 findings
post-change full pytest: NOT RUN
post-change Rust tests: NOT RUN
receipt verification: NOT RUN
implementation adversarial review: NOT REACHED
```

No passing focused gate, full gate, receipt, review, or CI result exists for the
draft. Publication of this branch must not be interpreted as acceptance.

## Reproduction

From a checkout of the published branch, using the project virtual environment:

```bash
git diff --check 83095c9742d779c3f2fc95a8021fa7bb4ab95338...HEAD
./venv/bin/ruff check \
  quartz/idea_foundry/axis_workflow.py \
  quartz/idea_foundry/sequential.py \
  quartz/idea_foundry/execution_seal.py \
  tests/test_idea_foundry_execution_seal.py
./venv/bin/ruff format --check \
  quartz/idea_foundry/axis_workflow.py \
  quartz/idea_foundry/sequential.py \
  quartz/idea_foundry/execution_seal.py \
  tests/test_idea_foundry_execution_seal.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" ./venv/bin/pytest -q \
  -p no:cacheprovider \
  tests/test_idea_foundry_execution_seal.py \
  tests/test_idea_foundry_axis_workflows.py
git diff --numstat 83095c9742d779c3f2fc95a8021fa7bb4ab95338...HEAD -- \
  quartz/idea_foundry/axis_workflow.py \
  quartz/idea_foundry/execution_seal.py \
  quartz/idea_foundry/sequential.py
```

The focused test command is expected to fail at this checkpoint. A clean clone
may change the final smoke-test symptom, but that does not cure the remaining
contract, budget, or lint failures.

## Claim firewall and recovery

- This branch contains no new campaign, scientific result, receipt, evidence
  seal, or claim promotion.
- It does not revive the 2026-08-17/18 claims. Their historical-diagnostic and
  forbidden-claim classifications in `docs/CLAIM_LEDGER.md` remain controlling.
- The new GPU and RAM do not change estimands, independence units, confidence
  intervals, evidence maturity, or promotion status.
- Do not run or resume Idea Foundry campaigns from this archival branch.
- Recovery is non-destructive: retain this remote branch for audit, and start
  any future bounded repair from the corrected specification anchor or a newly
  authorized successor specification. Do not salvage this draft by silently
  raising the `M` limit.
- Reverting the single archival checkpoint commit removes only this preserved
  draft and summary; it does not modify results or the protected forensic
  worktree.
