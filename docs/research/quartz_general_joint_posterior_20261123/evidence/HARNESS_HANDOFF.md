# C05 Python harness handoff — ready for parent execution

Final source is complete and frozen for C05. No prepare, rho fit, TRAIN price, new CAL/TEST label generation, frozen bundle creation or campaign was executed by this lane. All new empirical data remain for the parent's C06–C08 commands.

## Identities

- C04 authority HEAD: cb6266eed040e2a9c4637cf0722b52763ee84937.
- coding/c27_campaign.py SHA256: cfe068449f30a40350e5069513f9b0c6fa280517841f8ee95c3af7c900a5f75f.
- tests/test_c27_joint_posterior.py SHA256: d2f8a31f6097d2c9eaea685535585cdcd9dcc01095f1fdc6552495f9fc4714c5.
- Native final source SHA256:433f620f1c420042d795fc054e87fe6309969d04530e9333c14c664e6dd69dec.
- Native final binary SHA256:38126679ef622d7339fecdc7fd258ce2bb1428d82bbcb24545f26a7fafb66eef.
- Root/data paths derive from __file__; no hardcoded old worktree path is used as a production default. Every binary-using CLI requires --binary.

## Actual executed checks

1. Meaningful semantic RED: python3 -m unittest tests.test_c27_joint_posterior.SemanticRegression -v, remote PID538172, exit1. Product baseline yielded0.5 instead of109/200 and unchanged marginal0.5 instead of0.82. Log HARNESS_RED.log. This was not an import-error RED.
2. Initial explicit-joint/pure helper suite:17 tests passed; HARNESS_GREEN_CORE.log.
3. First independent native integration: fixed first old TRAIN root in each ply4/5/6 x rho0/.8/.9 x5policies x2repeats=90runs, native exit0. HARNESS_INTEGRATION_ATTEMPT.log. This used pre-tie-fix binary755c93c55711ebc9ecaa6c05907ee9d8dff381ae1942df95ee33e8b1dce2f989; it is retained as that historical execution, not attributed to the final binary.
4. A newly added mutation test initially failed because its fixture used1-u at u=.5, leaving the value unchanged. HARNESS_TEST_FIXTURE_FAILURE.log preserves it. Changed the mutation to a genuinely different utility; this was a test-fixture defect, not a controller/theory failure.
5. Final formatted suite:21 tests, exit0, PID567116, HARNESS_GREEN_FORMATTED.log. Covers exact Fraction fixture, H2 information value, posterior conditioning, rho0, support, ties including chained tolerance, projected update, CV isolation, ns pricing, matched diagnostics, repeat integrity, root aggregation and request blinding.
6. Final scoped Ruff check: exit0, PID566974, HARNESS_LINT_FINAL.log. Formatting touched only owned new source/tests.
7. Final CLI integration against binary381266…: one fixed oldTRAIN root, rho0/.8/.9,5policies,5measured repeats and1warmup =>75 validated measured runs; malformed/unknown offline label/rho1/cap3/repeatedmove requests rejected (5 cases), then normal requests succeed. Native exit0; harness process PID566352 exit0. HARNESS_INTEGRATION_FINAL.json and.log preserve hashes, request controls and transport. Independent reference checks posterior probabilities/fields, projections, g/K1/K2, legal choices, exact values/signs/nodes, known observations, time partition, execution rotation and deterministic repeats.
8. CLI --help runs with no model fit/data mutation. HARNESS_CLI_HELP.log.

Whole-project/Torch tests were not run in this bounded implementation lane. Do not treat these targeted checks as a new whole-project PASS. The parent owns any broader required gate, old-tree guard, publication and restoration.

## Runnable commands

From ROOT=/home/cosmosapjw/quantum/research/quartz_general_meta27_20261123:
DOC=docs/research/quartz_general_joint_posterior_20261123
H=$DOC/coding/c27_campaign.py
DATA=$DOC/coding/data
BIN=/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123/target/release/quartz_c27_joint_posterior

Run with PYTHONDONTWRITEBYTECODE=1:
python3 "$H" prepare --data-dir "$DATA"
python3 "$H" fit --data-dir "$DATA"
python3 "$H" price --data-dir "$DATA" --binary "$BIN"
python3 "$H" freeze --data-dir "$DATA" --binary "$BIN"
python3 "$H" campaign --data-dir "$DATA" --binary "$BIN" --split CAL --output-dir "$DOC/coding/results/CAL"
python3 "$H" verify --data-dir "$DATA" --output-dir "$DOC/coding/results/CAL" --report "$DOC/evidence/CAL_VERIFICATION.json"
python3 "$H" analyze --data-dir "$DATA" --output-dir "$DOC/coding/results/CAL"
python3 "$H" campaign --data-dir "$DATA" --binary "$BIN" --split TEST --output-dir "$DOC/coding/results/TEST"
python3 "$H" verify --data-dir "$DATA" --output-dir "$DOC/coding/results/TEST" --report "$DOC/evidence/TEST_VERIFICATION.json"
python3 "$H" analyze --data-dir "$DATA" --output-dir "$DOC/coding/results/TEST"

All write commands use exclusive creation and preserve existing evidence. Prepare requires a new empty data directory. Campaign requires a new empty output directory. Repeated analyze into the same directory intentionally refuses overwrite; verification is read-only unless --report is supplied. An interrupted campaign retains raw partial data plus failure.json and has no completed manifest/admission. No automatic retry, cohort replacement or tuning occurs.

Optional bounded integration:
python3 "$H" integration --binary "$BIN" --limit 3 --report "$DOC/evidence/C06_TARGETED_INTEGRATION.json"
--limit accepts1..6, fixed oldTRAIN only; it does not run rho fitting/price or select new roots.

New data/results can match inherited ignore rules. Parent must force-add the explicit intended files and confirm git membership; this worker did not commit or publish.

## Source/data/model behavior

prepare enumerates public legal states, excludes all300 C26 roots, creates exactly144 public roots split24CAL/120TEST with frozen salt/counts. It never calls truth_record, cue_candidates or oracle. Assignment is written before later labels.

fit uses only old90 TRAIN raw vectors. Contiguous10-per-ply folds independently refit60 base marginals; each heldout root contributes exactlyone logR perrho. It chooses the smallest rho within1e-12 of the global maximum CV root-mean score. Final base remains original full90 model SHA0843447d…; C26 artifacts are read-only and hash checked.

price uses only public oldTRAIN requests, requires296*5=1480 nonterminal exact measurements, checks exact values/nodes against independent C26 semantics, excludes terminal64, and freezes the instrumented ns/node conversion. Actual price calls are not run by this child.

freeze hashes new native/Python/test/contracts/reference sources, unchanged game semantics, binary and all public/model-selection/price artifacts. Later load_frozen rejects changed source/data/binary, so any post-freeze repair needs explicit parent deviation and renewed affected evidence.

campaign creates offline truth after freeze, sends only public requests, verifies every returned run using independent explicit joint-world conditioning/H2 enumeration, and writes full raw response/timing/transport. Every additional query is unique, at most2, retains exact incumbent; all5 repeats are checked for semantic identity. No optimized tanh/closed-H2 formula is used by the independent score oracle.

## Native request/response agreement

Request retains C26 keys id/history/cue_seed/policy_seed/mode/additional/repeats/warmup/trace/model, adding rho. Run uses additional2/repeats5/warmup1/trace=true; survey uses model=null,rho0,policy_seed0,additional2,repeats5,warmup0,trace=false.

Run response schema c27-joint-v1, mode,id,runs plus separate decode/serialize/execution/warmup times. Records keep policy/repeat/execution_slot, initial_move/initial_solve_move/initial_u,selected_move/final_u, node accounts/query_moves,elapsed_ns/cue_ns/initial_ns/query_ns/controller_ns,initial_log_odds/final_log_odds,known_observations,projection,predictions,decisions. H1 score k2 is null and q=k1; H2 q=k2. ForcedSTOP score list is empty. Full actual native time includes trace construction; Python oracle checking/transport is outside this interval.

## Analysis files and plotting keys

Each output-dir contains:
- raw.jsonl.gz: complete perroot/seed native request/response and transport; all repetitions/times retained.
- offline_oracle.jsonl.gz: independently generated exact root/candidate values; analyst-only.
- campaign_manifest.json: input,code,binary,bundle and raw-file identity; native exit; validated counts.
- seed_summaries.jsonl.gz: median5 timing and fixed semantic metrics perroot/seed/policy.
- root_summaries.jsonl.gz: mean4seed metrics perroot/policy.
- paired_root_differences.jsonl.gz: PRIMARY paired differences, one row perroot:
  {"id":..., "ply":4|5|6, "metrics":{"net_wall":...,"utility":...,"elapsed_ns":...,"net_nodes":...,"charged_nodes":...,"additional_nodes":...,"additional_queries":...,"regret":...}}.
  Direction JOINT_H2 minus PRODUCT_H2. This is the simplest plotting input.
- timing_dispersion.jsonl.gz: all5samples,min,median,max,population_stdev for each phase/root/seed/policy.
- conditional_diagnostics.jsonl.gz: generating_policy,history_index,stage,exact permitted history,remainingcandidate,predictor(joint/product/initial_projected),probabilities,offline truth,log_loss,Brier,selected direct gain/residual.
- joint_vector_diagnostics.jsonl.gz: root-level product/jointNLL and logR on all nonterminal candidates.
- analysis.json: primary.metrics.net_wall.{mean,ci95}, primary.metrics.utility.{mean,ci95}, primary.per_root,primary.root_signs_net_wall,primary.ply_strata, secondary_contrasts,policy_summaries, conditional_diagnostics,gate and output hashes.

No numeric example above is a new scientific result; the actual CAL/TEST outputs do not yet exist.

Primary bootstrap resamples full roots equally (actual6/45/69 TEST mix),10000 draws,PCG64 seed2709,linear percentile endpoints. It is not stratified to equal ply sizes. Operational TEST admission requires netwall lower>0 and quality lower>=-.02 with no semantic failures. CAL always remains diagnostic. A failed scientific gate is retained and does not prevent TEST.

Matched diagnostics evaluate joint/product/initial-projected on the SAME remainingcandidate at each recorded history, tagging its generating policy. Later differences between unrelated exposure means are not called predictive improvement. Perpredictor N/exposure/root counts and within-history paired differences are explicit. Direct gain is immediate incumbent improvement, not the complete K2 information-value target; this limitation is included in the summary.

## Boundaries

The complete operational commands were implemented but intentionally not executed on new data in C05. Their real source counts, CV outcome, hardware price and scientific performance remain for the parent's staged execution. Current validation is implementation evidence, not C27 efficacy. No P2/P3/globalphysical claim is promoted. Oldbase labels remain offline; rho0 selection or a negative time result is preserved without automatic model expansion.
