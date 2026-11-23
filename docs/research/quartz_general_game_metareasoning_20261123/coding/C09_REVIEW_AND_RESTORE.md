# C09 — independent review, narrow correction and actual restoration

## Independent review

Final verdict: Looks good; scoped Review Gate PASS; no unresolved blocking finding. The reviewer read the complete controller, independent oracle/harness, contracts, original source records, proofs and calibration/inference artifacts. Its exact report is review/INDEPENDENT_REVIEW.md and machine-readable companion .json.

The reviewer directly ran 20 focused tests (exit 0), independently reconstructed all TRAIN parameters from 296 candidate records, and independently audited 136,080 CAL/TEST measured policy runs and 15,120 trace policy runs. It verified 27,401 decision transitions and 246,127 additional trace node charges. It reproduced the predeclared paired bootstrap; the primary empirical gate remains false.

Original reviewer temporary artifacts were copied byte-for-byte:
- Markdown SHA256 df6fcb81797af93c7bf2ae2853e99bc1aabf68d06abd313f93eeb1bbde2b40d3.
- JSON SHA256 c4aa68cc6330bb7763285a83116b04efa9fb50c6e26c8763933dd6931aac476f.

## Findings closed

P1 C07 publication omission: reviewer verified all 16 CAL/TEST result paths at exact C08 commit 1f29a3152828ce7d4d347be362eb2d4a79ec4735 and equality of every commit blob with reviewed bytes. Supplemental CAL backup receipt was checked. The defect and repair timing remain explicit in evidence/C08_PACKAGING_CORRECTION.json. No raw scientific value changed.

P3 positivity endpoint: corrected only the words requiring 0<delta<1/kappa for strict positivity in theory/STATPHYS_DERIVATION.md and its generator string in theory/statphys_check.py. The original C08 versions remain in Git and backup; evidence/C09_ENDPOINT_CORRECTION.json records before/after identities. The reviewer verified that the replacement is the only byte difference in each file. Existing 72 assertions use delta=1/2; no computational expression, production code, model or cohort changed, so another campaign is unwarranted.

The current scoped Python formatter check exited 0 (34 files already formatted). Whole-repository inherited formatting debt and missing-Torch collection blockage remain as described in C06; a scoped review pass is not an all-repository CI pass.

## Actual restore

Ran coding/verify_restore.py, process exit 0, elapsed 3.8457 seconds. It verified SHA256 and sizes for the full C23 base, cumulative C24 and C25 overlays, and cumulative C26 C08 overlay, then extracted them into /tmp/quartz_c26_restore_b7q8p1o8.

Exact C08 tree reconstruction verified:
- 2,768 Git-tracked files; 1,106,732,654 bytes.
- Git blob identity and executable mode for every file.
- Exact file set, zero unexpected files, zero symlinks.
- All 14 old worktrees retained their baseline HEAD, status and own protected hashes.
- All current protected source files retained baseline hashes.

The release executable was also compressed separately and decompressed through a SHA256 readback. Uncompressed binary SHA256 is 8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c. Detailed receipt: evidence/C09_RESTORE.json; original stdout: evidence/C09_RESTORE.log. This is actual restoration and identity checking, not another scientific run.

## Cloud acknowledgement scope

The Dropbox service listing in evidence/CLOUD_METADATA_C08.json has has_more=false and includes all 18 stage archives through C08 with exact expected names and byte sizes, all three base archives, the CAL supplement, TRAIN model and binary backup. This is independent cloud metadata acknowledgement. The archive content hashes and restoration above were checked from the local synchronized files; remote cloud bytes were not downloaded independently. Earlier stage receipts retain their original NOT_VERIFIED cloud state rather than being retroactively rewritten.

C10 now needs the final scientific report, continued origin-preservation record, one bounded future proposal and final commit/archive/remote verification. No new theoretical-performance claim or empirical retuning is authorized by the review outcome.
