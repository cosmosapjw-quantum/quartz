# Reproduce and preserve Quartz C25

Date: 2026-11-23, Asia/Seoul.

## Identity and evidence

Repository: https://github.com/cosmosapjw-quantum/quartz
Branch: `research/quartz-general-c25-incumbent-price-20261123`

- C24 parent: `4998c771c08788e4f0da8d2723347121ae16fbf2`
- TRAIN predictor freeze: `3cdc8e4a043be4d2b1c707dceba6954d388930bc`
- Pre-TEST code/model freeze: `21c71da38ba1243d92f4f7682f54ef485eb5c3c6`
- C08 scientific outcome snapshot: `d45173685f955820bd848bf4ddb2044382ca7e67`
- C09 review/restore publication: `61707611fdf3319bb5c0d7f26da7ff8cf6ac5254`
- Final C10 identity: final external `STAGE_RECEIPTS.jsonl` / `FINAL_PUBLICATION_AUDIT.json`; the last receipt must match the remote branch HEAD.

The C10 commit cannot contain its own hash without a circular self-reference. [STAGE_PUBLICATIONS.json](STAGE_PUBLICATIONS.json) therefore contains the complete pre-C10 receipt snapshot. The external final audit verifies that snapshot against all receipts, including C10, and verifies the final cumulative archive by reading every contained file.

Remote execution root:

```text
/home/cosmosapjw/quantum/research/quartz_general_meta25_20261123
```

Document root relative to that repository:

```text
docs/research/quartz_general_incumbent_price_20261123
```

These paths are on the authorized remote device cosmosapjw, not the ChatGPT scratch filesystem.

## Observed runtime

Actual C25 environment: Python 3.12.3, Rust/Cargo 1.99.0, NumPy 2.4.2, SciPy 1.17.0, Matplotlib 3.10.8, Linux x86_64, 24 logical CPUs. See [C01_ENVIRONMENT.json](evidence/C01_ENVIRONMENT.json) and the raw cohort manifests for exact platform strings and start/end load.

The isolated `venv` used installed system packages. No dependency installation was required. Scientific timing used one native worker, with:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

No fast-math flags were introduced. Build/review/native campaigns did not overlap on the timing device. OS background load was observed rather than assumed absent. Timing replication on another host will not have byte-identical wall times.

C25 native source SHA-256:
`84c3937e42836431145fbde74999a57f98540e6b65760f2d71c6a4f78271a8d8`

Observed release binary SHA-256:
`2ddccac45f1fa9c171509bcc329a37f2f9b575fe3c17797e806a74ee4d8febf5`

The exact binary is backed up separately. Rebuilding on another compiler/platform may change bytes; scientific output identity and binary byte identity are different checks.

## Commands actually executed

Native build and tests, with their actual logs/manifests preserved:

```bash
cargo test --bin quartz_c25_incumbent_price
cargo test --release --bin quartz_c25_incumbent_price
cargo build --release --bin quartz_c25_incumbent_price
venv/bin/python -m unittest discover -s tests -p test_c25_sources.py -v
```

Debug/release each passed 40 native tests. The Python suite passed 18. [C05C06_RUST_REPORT.md](evidence/C05C06_RUST_REPORT.md) and [C05C06_RUST_VERIFICATION.json](evidence/C05C06_RUST_VERIFICATION.json) distinguish reused tests, observed changed-behavior RED/GREEN, binary validation streams, and actual exit codes.

The executed cohort driver was an identical byte copy of [coding/run_cohort.py](coding/run_cohort.py) at remote `/tmp/quartz_c25_run_cohort.py`. The actual command order was:

1. TRAIN source generation → trace → timing → exact trace verification → exact timing verification → predictor fit → analysis.
2. Commit and publish the TRAIN-only predictor at C07A.
3. CAL generation → trace → timing → both verifications → analysis.
4. Freeze all scientific inputs at C07 and publish before TEST generation.
5. TEST generation → trace → timing → both verifications → analysis.

[RUN_TRAIN_COMMANDS.json](evidence/RUN_TRAIN_COMMANDS.json), [RUN_CAL_COMMANDS.json](evidence/RUN_CAL_COMMANDS.json), and [RUN_TEST_COMMANDS.json](evidence/RUN_TEST_COMMANDS.json) contain each actual argv, working identity, elapsed time, stdout log hash and exit code. TRAIN has seven commands; CAL/TEST each have six. All completed with exit 0.

The driver has an intentional original-machine path and refuses to overwrite preserved run evidence. It is an execution record, not a portable command to rerun on top of the final source tree.

Independent C09 reconstruction, actually run at the C08 snapshot:

```bash
venv/bin/python -B docs/research/quartz_general_incumbent_price_20261123/review/C09_REVIEW_CHECKS.py
```

This script has a **C08 HEAD guard** and produced [C09_REVIEW_CHECKS.json](review/C09_REVIEW_CHECKS.json) with actual exit 0. It does not import the production verifier/analyzer/predictor. Running that guarded script from C10 is expected to reject the changed snapshot, even though the frozen scientific files are identical. The reviewer script also binds ROOT to the recorded absolute remote path. It is preserved as the exact execution record; simply copying it into an arbitrary checkout does not relocate it. A separate independent replication must declare its new checkout path and preserve the original script alongside any path-adapted copy, while retaining the C08 scientific snapshot and hash checks. For portable reanalysis of the existing observations, use the supported --evidence-dir procedure below.

## Reanalyze existing TEST data without a new native campaign

The following procedure copies only input evidence to a fresh temporary directory and uses the preserved verifier/analyzer. It does not regenerate sources, rebuild the binary, alter the predictor, or replace published results. It is a reproduction recipe; the recorded independent C09 run is the actual independent execution evidence.

Run from a checked-out final C25 repository with the recorded Python dependencies available:

```python
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

root = Path.cwd().resolve()
doc = root / 'docs/research/quartz_general_incumbent_price_20261123'
source = doc / 'evidence'
replay = Path(tempfile.mkdtemp(prefix='quartz_c25_reanalysis_')).resolve()

names = [
    'C25_PUBLIC_TEST.jsonl.gz',
    'C25_ORACLE_TEST.jsonl.gz',
    'C25_SOURCES_TEST.json',
    'C25_TRAIN_COST_PREDICTOR.json',
    'C25_TEST_TRACE.manifest.json',
    'C25_TEST_TIMING.manifest.json',
]
for name in names:
    shutil.copy2(source / name, replay / name)
for name in ('C25_TEST_TRACE', 'C25_TEST_TIMING'):
    shutil.copytree(source / name, replay / name)

# Chunk paths must remain inside replay: use copies, not external symlinks.
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    env[name] = '1'

for mode in ('trace', 'timing'):
    command = [sys.executable, '-B', str(doc / 'coding/c25_verify.py'),
               '--split', 'TEST', '--mode', mode,
               '--evidence-dir', str(replay)]
    with (replay / f'verify_{mode}.log').open('x') as output:
        subprocess.run(command, cwd=root, env=env, check=True,
                       stdout=output, stderr=subprocess.STDOUT)

command = [sys.executable, '-B', str(doc / 'coding/c25_analyze.py'),
           '--split', 'TEST', '--evidence-dir', str(replay)]
with (replay / 'analysis.log').open('x') as output:
    subprocess.run(command, cwd=root, env=env, check=True,
                   stdout=output, stderr=subprocess.STDOUT)

original = json.loads((source / 'C25_TEST_SUMMARY.json').read_text())
recomputed = json.loads((replay / 'C25_TEST_SUMMARY.json').read_text())
assert recomputed['summaries'] == original['summaries']
assert recomputed['contrasts'] == original['contrasts']
print(replay)
```

Execute the recipe with the recorded Python environment. Result metadata such as argv paths can differ; scientific summaries must agree. No deleting original verifier/summary files is necessary.

## New native timing replication

A new timing study must use a new evidence directory and keep the current run immutable. The portable scientific CLIs accept `--evidence-dir`, but `c25_run.py` intentionally requires committed scientific inputs. For CAL/TEST this includes the predictor **at its actual evidence path inside the new repository**. An arbitrary untracked external predictor fails the provenance check.

A faithful replication therefore uses a new isolated checkout/branch, retains the frozen source rules and analysis, and follows the original sequence. If refitting the timing predictor on new hardware, label it a new replication model, run TRAIN first, and commit that model before CAL/TEST. If assessing the exact existing policy, preserve the existing predictor bytes instead. Do not mix these two interpretations or relabel the known C25 TEST set as previously unseen.

The primary statistic remains mean of nine-run source medians, not the mean of all runs treated as independent observations. Bootstrap indices use 72 source parents, 2,000 draws, seed 2500923 and budget order 8→32→128. Price sensitivities revalue the same primary policy outcomes; they are not rerun optimized policies.

## Backup chain and actual restore

Backup root:

```text
/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta25_backups
```

Required chain:

| Layer | File | SHA-256 |
|---|---|---|
| C23 full tree | BASE_C23_a4417b081037.tar.gz | c3bd869039619094905c2bb18f1ea9b427ce8fb2fe1c71f6c62a5a9630b26b05 |
| C24 cumulative changes | QUARTZ_GENERAL_C24_C10_4998c771c087.tar.gz | 0d9ff61369ae7b26b1421648f4cc3af13db4b394438b173a4bc678f02faa3efd |
| C25 cumulative changes | Final C10 archive named by the last stage receipt | Verify against that receipt and FINAL_PUBLICATION_AUDIT.json |

Extract into a **new directory** in the above order. Each C25 stage overlay is cumulative relative to C24, so only the selected C25 overlay is needed after C23+C24. Archive paths are relative to the repository tree. There are no C25 deletions of inherited files.

Actual C09 restore used C25 C08 archive:

- File: `QUARTZ_GENERAL_C25_C08_d45173685f95.tar.gz`
- SHA-256: `0e45b85cfce2778df2f2d2aa690874f645319f8ee751e6c7ea082d699e2d5612`
- Restored 2,603 tracked files / 1,097,124,692 bytes, each matched to its Git blob.
- Restored directory was `/tmp/quartz_c25_restore_p_1zhuqd` on the remote device.
- The final C10 audit reads back the final overlay and checks it against every artifact entry, including final metadata added after C08.

Exact release binary backup:
`C25_NATIVE_RELEASE_2ddccac45f1fa9c1.gz`.
Compressed SHA-256 `bab90fd64be08f5a868249cbaf6092da3fc93db35893c6fffafd95be1a720287`; decompressed SHA-256 is the release binary hash above.

The local Dropbox-path files, hashes, actual extraction and Git remote publication are independently meaningful. Dropbox cloud synchronization acknowledgement was not verified.

## Rendered figures and packaging

[render_c25_report.py](coding/render_c25_report.py) rendered the scientific PNG/SVG from the frozen TEST summary. The PNG was visually inspected and preserved unchanged.

Matplotlib's raw SVG contained trailing spaces in path data, so the publication whitespace precheck rejected the first C09 attempt **before commit**. [normalize_c25_svg.py](coding/normalize_c25_svg.py) preserved the raw SVG as compressed evidence, removed only line-end spaces and verified the order, text and attribute tokens of all 1,389 XML elements. The corrected publication then passed the same precheck. No publication gate was disabled and no scientific input changed.

[C09_SVG_PACKAGING.json](evidence/C09_SVG_PACKAGING.json), the failed preflight log, the original render receipt and compressed original SVG preserve both byte identities. This packaging repair does not constitute a native scientific rerun.

## Completion checks and scope

The final audit checks:

- All R01–R10 and C01–C10 in order, with unique helper stage C07A.
- Parent linkage of every stage commit, archive hashes/sizes and restore prerequisites.
- Final artifact-manifest completeness and archive-file readback.
- All 16 frozen scientific inputs and inherited protected files unchanged.
- All 13 earlier worktrees preserved using their own baseline identities.
- Remote branch equals final HEAD, clean current worktree.
- Independent review and actual restore evidence present.
- C26 not executed.

Detailed finite scientific evidence is in [REPORT.md](REPORT.md); expected original-game transfer work remains [NEXT_STAGE.json](NEXT_STAGE.json).
