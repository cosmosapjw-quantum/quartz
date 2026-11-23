# C26 independent oracle, data and campaign harness

## Implemented scope

`coding/c26_campaign.py` is the finite campaign entry point. Its independent board
representation uses signed integer tuples, absolute winner signs, explicit
root-player max/min minimax and legal-history replay. It does not import native
game or policy implementation code. The offline recurrence memoizes board results
for speed while summing every occurrence in the complete uncached search tree;
its reported node count therefore retains transposition multiplicity. Actual
charged traversals and their timing are performed by the native binary.

The enumerator visits exactly 5,478 reachable boards and stops when a game ends.
Of these, 4,520 roots are nonterminal. D4 canonical nonterminal counts at plies
4/5/6 are 108/153/183. The R09 SHA-256 ordering selects exactly 100 parents per
ply and assigns ranks 0:30 to TRAIN, 30:50 to CAL, and 50:100 to TEST. Each selected
canonical representative carries an actually replayable transformed legal
history. Source IDs, full source hashes and frozen cue seeds are preserved.

The public source file contains identity, split, ply, history and cue seed.
Offline boards, exact child values and exact node counts live in a separate
compressed oracle file. The native request builder selects allowed fields
explicitly; additional keys in an input metadata object cannot reach the native
request. TRAIN requests use the model-free survey interface. Every survey is
checked against the independent exact solver and SplitMix64 replay before fitting.

The TRAIN estimator records every feature-bin, remaining-count and global class
count, every node-count sum, and every source identity. It applies the frozen
three-class add-one rule and arithmetic mean costs. The frozen model is written
before CAL. `audit-model` reconstructs every parameter from the retained raw
TRAIN measurements and verifies model and input hashes. Neither CAL nor TEST
changes parameters or selects a new price, temperature, cap, split or contrast.

## Independent native verification

The meaningful red fixture used an importable initial-state-only scaffold:
the real checks observed 1 instead of 5,478 reachable states and 1 instead of
549,946 nodes for the empty board. `DATA_RED.log`,
`DATA_RED.json` and `DATA_RED_SCAFFOLD.py.txt` preserve this feature-absence
signal. It is not a claim that the preexisting game implementation had a defect.

The focused suite checks literal terminal root-frame signs for both players,
empty-board exact value and full tree count, D4 history compatibility, frozen
population/split cardinalities, SplitMix64 literal vectors, terminal cue charging
once, TRAIN smoothing/fallbacks, reservation-index branch boundaries, stable Gibbs
probabilities, shift/scale invariance and an actual 100,000-draw frequency check
with predeclared absolute tolerance 0.007.

Native boundary checks reject malformed requests, illegal moves, terminal
continuation/root, unapproved oracle fields, invalid caps/repetitions, missing
model and invalid model probabilities/costs. A subsequent valid line verifies
that an input error does not terminate the persistent native process.

Every reachable nonterminal root is independently surveyed: all child values,
full-tree node counts, cue features, cue cost and terminal semantics are compared.
The audit records the precise root/candidate cardinalities, tested players/value
signs, a digest of logical native output, source/binary hashes and actual duration
in `evidence/DATA_ALL_REACHABLE.json`. A digest avoids storing thousands of
redundant verification rows; the documented command regenerates all of them.

Trace fixtures cover both players, known immediate wins, all six policies and
caps 1/2/9. The Python replay reconstructs every scored candidate, expected gain,
Q, reservation index, probability, forced STOP, Gibbs draw, random action, exact
reveal and incumbent update. Nonterminal predictions cannot contain oracle
truth. An additional fixture checks nine timing repetitions and one warmup
against a separate one-repeat trace.

## Runtime records and estimands

Each source/cap/seed receives a native timing request with six policies, one
discarded warmup and nine retained repetitions, followed by a separate
one-repetition trace request. Policy order is rotated natively according to the
contract. Public requests, request digests, source hashes, model hashes, response
records and transport measurements are retained in compressed JSON-lines files.

Logical signatures are checked against the independent oracle on every
repetition; trace signatures must also match. Recorded elapsed times and their
four component times must be finite, nonnegative and additive before aggregation.

The timing estimator is the median of nine repetitions within each
source/cap/seed/policy cell, then the mean over four seeds, then an equal-weight
mean over source parents. Separate component medians need not add up to the
median total time. `seed_medians.jsonl.gz` and `parent_metrics.jsonl.gz` preserve
these intermediate estimands. Repetitions and seeds are never bootstrap units.

Reports include exact final utility, utility gain, regret against the independent
root optimum, actual cue/initial/additional/total charged nodes, actual query
counts, predicted additional nodes, wall time and both frozen priced utilities.
All 15 policy pairs are compared at each cap. Full-cap node/time savings against
ALL_ACTIONS are also reported directly.

The frozen primary contrast is TEST, cap 9, PANDORA minus INITIAL_STOP in
`final_u - 2^-10 * charged_nodes`. The 95% percentile interval uses 2,000 paired
source-parent bootstrap resamples stratified by ply, NumPy PCG64 seed 2600923,
and linear quantiles. The same deterministic resampling pattern is used for
the other contrasts. A positive priced-gain claim is gated on the primary lower
endpoint being strictly positive. CAL output labels this contrast diagnostic.

Inference is conditional on the fixed TRAIN model. Different game roots can
share descendants; the finite source-parent bootstrap does not establish
universal iid game performance.

Calibration files retain every evaluated nonterminal candidate and report
multiclass Brier loss, observed versus predicted class frequencies, cost bias,
MAE, RMSE and MAPE, by remaining count and observed feature bin. Within-root
residual dependence is descriptive: symmetrized candidate-pair moments with
equal source-parent weights. Finite Gibbs Q/index ranges, positive probability
ranges and underflow zeros on genuinely sampled steps are reported separately
from forced STOP probabilities.

Native measured decision, warmup, trace and execution times are recorded.
Transport/parser/model-validation/serialization residual and Python
oracle/I/O/orchestration residual are reported as experiment overhead. They are
not included in the runtime decision-price estimand.

## Commands and outputs

Run from
`/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123`.
The parent stage controller owns preparation and the TRAIN/CAL/TEST commands;
the implementation worker has not run those cohorts.

```bash
venv/bin/python -m unittest discover -s tests -p test_c26_game_metareasoning.py -v
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py prepare
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py train
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py audit-model
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py cal
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py test
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py validate --split CAL
venv/bin/python docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py validate --split TEST
```

The standalone exhaustive verification command is `audit-all`. It is already
part of the focused unit suite and need not be repeated without a changed
dependency or specific remaining risk. `analyze --split TEST` reconstructs
reports from retained raw data and revalidates them; `--split CAL` is also
supported. Optional `--data-dir`, `--results-dir` and `--binary` are explicit.

Default prepared artifacts are in `coding/data/`:

- `public_sources.jsonl`, `offline_oracle.jsonl.gz`, `source_manifest.json`;
- `training_raw.jsonl.gz`, `model.json`, `model_provenance.json`;
- `training_calibration.json`, `training_calibration_rows.jsonl.gz`.

Default CAL/TEST results are in `coding/results/CAL/` and
`coding/results/TEST/`:

- `run_manifest.json`, `timing_raw.jsonl.gz`, `trace_raw.jsonl.gz`;
- `seed_medians.jsonl.gz`, `parent_metrics.jsonl.gz`;
- `calibration.json`, `calibration_rows.jsonl.gz`, `analysis.json`;
- `validation.json` after an explicit stored-record validation.

CAL comprises 38,880 retained timing runs and 4,320 trace runs over 60 parents.
TEST comprises 97,200 retained timing runs and 10,800 trace runs over 150 parents.
Each has 12 timing requests and 12 trace requests per parent. TRAIN measures 90
parents once, with every legal candidate independently verified.

Every CLI command writes a timestamped `evidence/DATA_EXEC_*.json` record with
argv, code hash, actual status, exit code and elapsed time. A failed campaign
retains its partial compressed raw files and a failed run manifest. Existing
frozen data, TRAIN output or campaign directories are never overwritten by
another execution; use a new explicit results location only when preserving and
diagnosing a failed execution requires a rerun.

## Checkpoint and remaining execution

The implementation and bounded fixtures are ready for parent review. Actual
test/format commands, exit status and byte hashes are in
`evidence/DATA_FINAL_CHECK.json` and its logs. Cohort performance, model
calibration, the primary contrast and campaign raw-record checks remain unrun
until the parent's C06/C07 gates and subsequent commands. C09 review and
archive/restore/publication belong to the parent controller. No C27 work is
started.
