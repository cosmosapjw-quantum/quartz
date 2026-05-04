# Audit — P02: Pre-flight hash gate + sha256_checkpoint helper

**Date:** 2026-05-04
**Patch:** P02 (15-patch QUARTZ v1.0 sequence)
**Scope:** add a single-pass gate that runs before the eval matrix executes,
catching missing checkpoints, missing search-manifest hashes, and silently
overloaded condition labels under `--paired-seed-eval`. Closes audit
weaknesses W5 (same-NN guarantee) and W6 (same-evaluator guard) at the
"hard fail before launch" level.

## What changed

### Python — scripts/ablation_study.py

- Added `sha256_checkpoint(path) -> str | None` helper. Returns the **full
  64-char SHA256** digest, distinct from the existing `sha256_file_prefix`
  (which truncates to 16 chars for log readability). Pre-flight uses the
  full digest — a 16-char prefix collides with non-trivial probability
  across the ~10⁵ checkpoint scale a long campaign produces (birthday-
  bound ~3×10⁻⁶ at 10⁵ items).

- Added `pre_flight_check(args, eligible, eval_conditions, expected_manifest_hashes) -> dict`
  that performs three independent checks and returns a JSON-shaped result
  with `schema_version: 1`, `ok: bool`, `errors: list`, `skipped_pairs: list[run_id]`:

  1. **Checkpoint reachability + fingerprinting**: SHA256 every
     `eligible[i]["model_path"]`. Missing/unreadable files become
     `checkpoint_missing_or_unreadable` errors. Side-effect: stamps
     `eligible[i]["candidate_hash"]` so downstream eval rows can record
     exactly which model bytes were evaluated.
  2. **Manifest-hash presence**: every entry in `expected_manifest_hashes`
     must be non-null. A null/empty value silently breaks cross-condition
     comparison (search_manifest_hash drift goes undetected).
  3. **Paired-seed cross-condition consistency** (only when `--paired-seed-eval`):
     within each `(condition, seed)` label, all rows must share the same
     `train_contract_hash`. Two `A_seed42` rows with different contract
     hashes mean the label was overloaded — the run was re-trained with
     a different config under the same identifier, so pairing it with
     `B_seed42` would silently mix incompatible models. Hard fail.

  Under `--research-grade`, any error raises `SystemExit` with a one-line
  reason summary. Without `--research-grade`, the function returns the
  failure summary and the caller continues with skipped pairs.

- Wired into `run_evaluation_matrix` immediately after the
  `expected_manifest_hashes` loop, BEFORE the per-pair eval loop. Result
  stashed on `eval_payload["pre_flight"]` (both the incremental save path
  and the final payload).

- Modified `should_compare(model_a, model_b)` to skip pairs whose ids are
  in `pre_flight_skip` (the set of run-ids returned in `skipped_pairs`).
  Without this guard, the eval loop would attempt to load a missing
  checkpoint and crash mid-pair, polluting `evaluation_matrix.json` with
  half-finished rows.

### Tests added (5)

`tests/test_ablation_study.py`:

1. `test_p02_sha256_checkpoint_full_digest` — verifies the helper returns
   a 64-char digest, equals the known SHA256 of `b"hello"`, returns None
   on missing/None paths, and is independent from `sha256_file_prefix`
   (the prefix is the first 16 chars of the full digest).
2. `test_p02_pre_flight_check_passes_on_clean_inputs` — clean two-row
   eligible list with two valid checkpoint files and a non-null manifest
   hash ⇒ `ok=True`, errors empty, candidate_hash stamped on rows.
3. `test_p02_pre_flight_check_blocks_missing_checkpoint` — one of two
   model_paths doesn't exist on disk ⇒ run_id ends up in skipped_pairs;
   under `--research-grade` ⇒ SystemExit with a clear reason.
4. `test_p02_pre_flight_check_blocks_null_manifest_hash` — a None entry
   in `expected_manifest_hashes` ⇒ `search_manifest_hash_missing` error.
5. `test_p02_pre_flight_check_catches_drifted_label_under_paired_seed` —
   two rows with the same `(condition=A, seed=42)` but different
   `train_contract_hash` values under `--paired-seed-eval` ⇒
   `duplicate_label_with_drifted_contract` error.

Existing test `test_run_evaluation_matrix_can_limit_to_paired_seed_comparisons`
updated to write actual bytes for the four `*.pt` files it references —
previously the test passed `model_path` strings without creating the
files (FakeCampaign didn't read them), but P02's pre-flight gate now
correctly identifies them as missing. One-line fix.

## Test results

- `pytest tests/test_ablation_study.py -q`: 62 passed (was 57; +5 new tests).
- All P02-targeted tests green via `-k "p02"` (5 passed).

## Adversarial review (red-team self-audit)

### What the gate catches

- Stale evaluation_matrix.json with a deleted/moved checkpoint — eval
  used to crash; now the affected pair is skipped before it runs.
- Bug in eval-runner config that produces a null search_manifest_hash —
  was silently aggregated into `expected_manifest_hashes`; now flagged.
- A user re-running condition `A` with a different `train_cfg` (e.g.
  changed `iterations`) under the same `seed`, intending to keep both
  for ablation — under `--paired-seed-eval` this is now an error,
  forcing the user to relabel.

### What the gate does NOT catch

- **Cross-condition hash equality is not required.** Conditions A and B
  with the same seed produce *different* models by design (different
  controllers ⇒ different SGD trajectories), so pre-flight does not
  compare candidate_hash across conditions. This is intentional; doing
  so would block the legitimate ablation case.
- **Truncated/corrupted .pt files**: `sha256_checkpoint` returns a hash
  even for a damaged tensor file — only the bytes are checked, not the
  loadability via torch.load. Loadability errors would surface later
  during eval engine construction.
- **Stale candidate_hash on incremental writes**: if the candidate file
  is rewritten between pre-flight and the eval-pair launch, the stamped
  hash reflects pre-flight state, not the bytes actually evaluated. The
  contract is "the eval row records the hash at pre-flight time" —
  document this in the row schema if downstream readers need stricter
  semantics.

### Schema discipline

- `pre_flight_summary.schema_version = 1`. Adding new error reasons is
  forward-compatible (consumers iterate over `errors` by `reason` string).
  Removing or renaming reasons requires a major bump.

### Concurrency

- Pre-flight runs single-threaded before any eval pair launches. No race
  conditions. Reading every checkpoint sequentially is O(N×file_size); on
  disk-cache it's fast enough that we don't bother parallelizing.

## Files touched

- `scripts/ablation_study.py` (+114 / -1)
- `tests/test_ablation_study.py` (+115 / -0; existing test +6)

Net delta: **+229 / -1 LOC**.

## What unblocks next

P02 stamps `candidate_hash` on every eligible row. Downstream patches:
- P14 `ArenaResult` dataclass can carry `candidate_hash` and
  `champion_hash` from these stamped values.
- P15 deprecation cleanup: rename `mean_prior_refresh_rate` (legacy)
  field in replay metrics now that the principled measurement has
  shipped via P01's `extended_measured_prior_refresh_rate`.
