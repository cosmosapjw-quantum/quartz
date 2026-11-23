# C28 Python harness handoff — C05

Status: complete runnable harness and tests; selected integration checks passed. The complete C06 unit suite, C07 300-root scientific campaign, C08 frozen benchmark, and figure rendering remain parent-owned and were not executed by this worker.

## Owned files and identity

- `tests/test_c28_action_geometry.py` — SHA256 `ab905e385c67627da482330619f5eed043ebf1b0cd5de8255f51197c1ec7e09d`; 17 stdlib unittest methods.
- `docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py` — SHA256 `f60e3bcfd54c7e0edeca26b056e00fc1d15f6dd74409e90b36583dc83563c446`.
- `evidence/HARNESS_C05_SMOKE.json`, `evidence/HARNESS_C05_RECONCILIATION.json`, `evidence/HARNESS_C05_CLI_SMOKE/`, and this receipt.

Native interface coordination: decision fields are `b`, `m`, `remaining_moves`, `horizon`, `scores`, `choice`; scores align with remaining board move ids. A game terminal root is rejected under C26 history validation; a nonterminal root with a known terminal winning child returns a coherent stopped controller. Native `cue_nodes` is an additive response field. No contract, formula, old binary, C27 worktree, or shared policy was edited. No commit, push, deletion, or full benchmark was performed.

Native binary used: `/home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry`, SHA256 `f8ab538dd1511244963b60c79f8da12f1c4e7e11ef63443087d8f35ca29f607a`.

Native source SHA256 at checks: `c396aad885132e7457fe99b4b16c8bdccff2067427cb1db7f2ed05d13d1603de`.

## Scientific and execution checks

The ground truth imports only the immutable Fraction complete-world model and Bellman recursion for native score verification. It does not compare the native lifted formula against a Python copy of the same formula. Synthetic controls include signed/common/product values, the c_A=.08 action crossing, exact and near ±19/24 boundaries, empty/unary and zero/point mass laws, independent and endpoint limits, near-unit signed loading stress, and fixed-seed rational models. Native acceptance stays abs5e-12, rel0; native epsilon choice remains distinct from exact optimal value.

The offline game checker reconstructs the C26 cue generation, solves TicTacToe through a separate Python minimax implementation, conditions complete worlds on actual observed utilities, and checks all supplied native decisions when the root has at most five candidate moves. The fixed 300-source cohort satisfies that small-world regime. These offline labels are never included in controller requests. It also verifies root-player sign, initial verification, known-only deployment, incumbent retention on equal values, additional query sequences, and logical cue/initial/additional node charges. Missing active H2/H1 trace decisions are rejected; a regression constructs otherwise coherent premature terminations and confirms this exact failure.

Selected C05 unittest command ran nine methods with exit0, including three pure controls and six native integration methods. The smoke CLI ran three synthetic and three game cases with exit0; maximum complete-world residual was 1.1102230246251565e-16, and two actual additional queries plus two complete-world game decisions were checked. Raw requests/responses and verification records are retained as gzip JSONL in `HARNESS_C05_CLI_SMOKE/`.

The false-PASS guard produced actual unittest exit5 with `NOT_RUN` and zero tests. The first receipt wrapper incorrectly expected exit1 and preserved that mismatch as ERROR. `HARNESS_C05_RECONCILIATION.json` records the correct acceptance: nonzero exit plus explicit NOT_RUN. This is a runtime receipt correction, not a native/scientific change or tolerance relaxation. The overwrite guard returned exit1, and every existing smoke output byte remained unchanged.

Full benchmark cardinality is 160 synthetic cells = 8 candidate counts × 5 latent means × 2 incumbents × 2 loading patterns; two methods give 320 method-cell results. Game roots remain exactly 300. The earlier description of 320 synthetic cells was a counting error corrected with the parent; no axis changed.

## Commands for the parent

Run these from `/home/cosmosapjw/quantum/research/quartz_c28_action_geometry_20261123_486e`. Each output path must be new. CLI output directories must lie inside the C28 document tree. No overwrite or tolerance override flag is provided.

### C06 complete Python contract suite

```bash
C28_BINARY=/home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry python3 -m unittest discover -s tests -p 'test_c28_action_geometry.py' -v
```

### C06 complete synthetic native-versus-world verification

```bash
python3 docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py verify --binary /home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry --suite synthetic --output docs/research/quartz_c28_action_geometry_20261123/evidence/c06_synthetic
```

### C07 frozen public-root scientific trace verification

```bash
python3 docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py verify --binary /home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry --suite game --output docs/research/quartz_c28_action_geometry_20261123/evidence/c07_game
```

### C08 frozen paired native cost campaign

```bash
python3 docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py benchmark --binary /home/cosmosapjw/quantum/research/quartz_c28_target_20261123_486e/release/quartz_c28_action_geometry --output docs/research/quartz_c28_action_geometry_20261123/evidence/c08_benchmark
```

The benchmark launches one resident native process with `taskset -c 11`, one native thread, fixed original model/source hashes, recorded cue seeds, and native binary/source hashes. It uses 31 pairs plus3 warmup pairs per game; 11 pairs plus3 warmups per synthetic cell; 256 iterations for n<16 and32 otherwise. Native sample nanoseconds are divided by iterations exactly once. Summaries take medians within root/method, then paired means, mean differences, ratio of paired means, and paired ratios. Raw timing records remain in response JSONL.gz. No bootstrap or new-generalization claim is made. A slower or equal lifted implementation is a valid measured result.

### Figures after the completed benchmark

```bash
python3 docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py figures --benchmark docs/research/quartz_c28_action_geometry_20261123/evidence/c08_benchmark --output docs/research/quartz_c28_action_geometry_20261123/evidence/c08_figures
```

Matplotlib is present on the remote Python runtime. The figure command generates PNG/SVG plots and exact rational source data: signed/common value and action geometry, followed by measured native costs. Figure manifest is explicitly `GENERATED_PENDING_VISUAL_REVIEW`; the parent must inspect actual rendered images. Analytic curves use exhaustive contingent-plan lines, not fitted game outcomes.

## Context and remaining obligations

Actual PhysMath Research Loop1.3.0 SKILL.md, CODING_LOOP C05/C06 definitions, and POLICY_INTEGRATION were read. Canonical remote CONTRACT.json and NATIVE_INTERFACE.md were read. Frozen theory oracle hash: `58c09680fca14ac5b05a5fb8e20d40d397202ba36762f6a54406cdffc29fc35e`; original data/model identities are enforced at runtime. This work implements the selected C05 feature path and bounded smoke checks. It does not claim completion of future stages, a learned signed game model, fresh TEST evidence, Elo gain, or the original Gomoku/cognitive program.

No current implementation blocker remains. Formal full-suite and campaign results are not yet admitted. Rebuilding or editing the native source/binary after measurements requires new affected measurements; this harness checks byte identities before and after each campaign. Publication/backup/restore and C09 independent review are parent-owned.
