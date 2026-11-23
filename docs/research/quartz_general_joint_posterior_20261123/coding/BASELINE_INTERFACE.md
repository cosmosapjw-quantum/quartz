# C27 R09 interface feasibility handoff

Status: read-only source/schema/storage inspection completed. No implementation, model fitting, CV scoring, timing experiment, native process invocation, build or campaign was performed. The only artifact written by this follow-up is this report. Parent owns R09 freeze and later C01–C10 work.

SSOT: 87c15dc828d6eea519c219fd8a412fa21dfc464d.
Remote root R=/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123.
D=R/docs/research/quartz_general_game_metareasoning_20261123/coding.
Device cosmosapjw, edd1b263-56a0-4391-8a2a-677dd00eabf5.

## 1. Exact training-vector provenance and seeds

Actual D/data/training_raw.jsonl.gz contains 90 unique root rows, exactly matching public_sources.jsonl split==TRAIN. It has 360 candidate records: 64 terminal-known,296 nonterminal. Every root candidate array is in ascending move order. These are90 root samples, not360 independent vector samples, four rollout replicates or multiple policy-seed datasets.

Actual keys:
- public source: cue_seed,history,id,ply,source_hash,split.
- training row: id,request,request_sha256,response,source_hash,transport.
- request: additional,cue_seed,history,id,mode,model,policy_seed,repeats,trace,warmup.
- response: candidates,cue_nodes,id,measurement_elapsed_us,mode,schema.
- candidate: counts,cue_nodes,move,remaining,rollout_mean,solve_nodes,terminal_known,u.
- transport: request_bytes,response_bytes,roundtrip_us.
- model_provenance: binary_sha256,elapsed_seconds,model_sha256,native_exit_code,native_roundtrip_us,policy,process_launch_us,provenance,raw_counts,schema,source_manifest_sha256,survey_measurement_us,training_raw_sha256,training_source_hashes,training_source_ids.
- raw_counts: features,global,nonterminal_candidates,remaining,smoothing,sources.

Every recorded request uses mode="survey",policy_seed=0,repeats=1,warmup=0,trace=false,model=null,additional=9. Policy seed does not create different training labels or cue vectors. Each root's fixed cue_seed is the unsigned integer represented by the first16 hex characters of SHA256("C26-cue-v1|"+board_text). A nonterminal candidate's four uniform rollouts use SplitMix64(cue_seed XOR (((move+1)*0x9e3779b97f4a7c15) mod2^64)). Their counts are ordered loss/draw/win in the root player's utility convention. A terminal candidate has four counts in its already-known class, cue_nodes=1.

Native survey obtains every nonterminal candidate's exact minimax value by exact(&child,root.current_player()), maps -1,0,+1 to u=0,.5,1 and records unpruned solve_nodes. Terminal-known candidates use their public value with solve_nodes=1. The resulting joint vector is indexed by ordered move, not feature bucket or root's selected candidate. The rho likelihood must preserve this root grouping and candidate correspondence. Terminal-known entries can be omitted from the likelihood (or represented by point masses of likelihood1), consistent with the parent's frozen direction. Candidate vectors are offline TRAIN labels; do not send their u/solve_nodes fields as run-mode controller observations.

Example actual first root:
id=C26-p4-0059adf99c0886cf,history=[5,1,6,4],cue_seed=3069874382843753759;
ordered moves=[0,2,3,7,8],u=[0,0,0,.5,0],solve_nodes=[38,34,30,47,24].
This is a schema/provenance example, not a new fitted/outcome-selected example.

Exact artifact SHA256:
| Artifact | SHA256 |
|---|---|
|public_sources.jsonl|9c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1|
|training_raw.jsonl.gz|486a50f34867e382cbff040e339dcc5e051349d2418dbc93a074ec25acdb6659|
|model_provenance.json|3dbb9903c021735741b5fe471d9058c526a985e48f086f447ca34d22584aaf02|
|model.json|0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126|
|source_manifest.json|1dfb2b92e4fca4bb6827fda0b955eb49e3c2a8b7550a27d7ed3dece4db5b8c16|

The provenance records training code HEAD16141896adab22df4e423bd4946b0544a19d8d4e and native binary SHA8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c, native_exit_code0. Source/evidence became committed later. This historical execution HEAD differs legitimately from final SSOT87c15d; preserve both identities, not an invented clean-final-HEAD training execution.

## 2. Parent-selected three-fold root CV is directly feasible

Parent direction: sort original90 TRAIN by source_hash within each ply, assign ten roots per ply to each of three folds. Each validation fold therefore has30 roots (10 per ply); each fitting complement has60. The rho grid is [0,.25,.5,.75,.9]. Score each grid point by mean validation-root log-likelihood improvement over rho0, pooling out-of-fold roots, exclude terminal-degenerate entries, choose maximum with smallest-rho tie break. Final base marginals remain the exact original C26 full90 model.json. This report did not compute fold assignments or scores.

Minimal existing functions:
- fit_model(rows) -> (model,raw_counts), c26_campaign.py389–435.
- model_prediction(model,remaining,counts) -> (probabilities,cost),438–447.
- feature_key(remaining,counts) -> str,385.
- json_lines(path), read_json(path): artifact reading helpers.
- audit_frozen_model(data_dir) -> (model,record),1222: hashes and exact parameter-reproduction logic already exist, but no audit was executed in this follow-up.

A fold-fitting row needs only:
{"id": raw["id"], "source_hash": raw["source_hash"], "candidates": raw["response"]["candidates"]}.
Select the60 root rows outside that fold; call fit_model once. It is a pure in-memory statistics computation and neither invokes the native binary nor oracle, writes a model nor opens CAL/TEST. For every held-out nonterminal candidate, call model_prediction(fold_model,candidate["remaining"],candidate["counts"]); preserve root's ordered vector of candidate["u"] as the scoring label. Evaluate the declared joint rho family on that vector. Joint likelihood itself is C27 logic; it does not exist in C26.

fit_model skips terminal_known candidates. It aggregates exact class counts by remaining:loss:draw:win feature, by remaining only and globally, applying add-one smoothing in three utility classes. Prediction falls back feature -> remaining -> global and explicitly renormalizes. Costs are mean solve_nodes by remaining/global; fold refit also recomputes these, although joint likelihood need only the probabilities.

Do not use train(data_dir,binary) for CV. That method starts native surveys and writes frozen output names, refusing existing model/raw paths. Do not change model.json in place. Store rho/CV/fold evidence and the new elapsed-time price in a separate C27 bundle referencing the full90 model's exact SHA. This keeps the parent's final base-model identity literal.

A root contributes one joint log-likelihood difference even when candidate counts differ. The requested root-mean score is not a candidate-mean score. Keep each root wholly within one fold. Deterministic fold assignment, exact grid order, numerical tie tolerance and how an empty nonterminal vector contributes should be explicit in the frozen implementation contract; natural empty-product log-likelihood difference is0. These are implementation details, not requests for new external data.

## 3. Native transport and cost-measurement route

Reusable signatures:
public_request(source,*,mode="run",cap=9,policy_seed=0,trace=False,repeats=None,warmup=None,model=None) -> request dict.
NativeClient(binary): one persistent JSON-lines native process.
NativeClient.request(request) -> (response,transport_metadata).
NativeClient.send_raw(line) -> same form; .close() records exit_code, requiring0.
Transport records binary_sha256, launch_us, roundtrip_us; request timeout60s. Request helper creates only allowed public source/control fields and supplied TRAIN model. Existing native Request/Model deny_unknown_fields; new C27 fields/schema must be added in the new binary explicitly.

C26 survey(root,&Request) starts Instant before candidates(), which creates public cues, and then exactly solves all nonterminal candidates. It reports ONE root measurement_elapsed_us after all of that. Actual aggregate training survey_measurement_us=581.353; native_roundtrip_us=2872.121, separate transport. No candidate-specific solve elapsed_ns/us exists in the raw C26 artifact. It cannot provide pure exact-solver ns/node by dividing that root time by node counts.

Parent's new C27 TRAIN-only path is implementable as a new survey mode in the isolated binary:
- reuse public root/history/cue parsing and exact() semantics;
- for each TRAIN nonterminal candidate, run the exact query five times;
- retain repeat,move,solve_nodes,elapsed_ns per measurement;
- elapsed_ns measures around the exact call using the monotonic timer and includes instrumentation overhead, with no subtraction of estimated timer cost;
- define exact_ns_per_node = sum(elapsed_ns)/sum(solve_nodes) over the declared measurement population;
- lambda_nodes=2^-10 utility/node;
- lambda_time_ns=lambda_nodes/exact_ns_per_node, utility/ns;
- freeze this TRAIN-derived conversion before CAL/TEST.

Known terminal candidates are already revealed through public rules, so they are not paid exact-query calls. Recommended precise boundary is296 nonterminal candidates x5 repetitions=1480 instrumented queries; terminal64 entries stay in the source observations and contribute no calibration query. Parent must retain whichever population it freezes. This new measurement is an instrumented exact-query amortized ns/node price, not pure CPU time and not the old mixed survey timing.

Use a name such as exact_ns_per_node rather than tau: old C26 model["tau"]=1/32 has utility units as the Gibbs temperature. They are unrelated quantities. If run-mode totals remain elapsed_us, multiply by1000 before lambda_time_ns charging; alternatively expose total elapsed_ns directly. Total deployed controller elapsed must include actual cue/initialization/planning/query work, not only measured solver work. Transport/process start remain separately reported under the contract.

Five measurements of one candidate are technical repeats for price stability, not extra independent roots or training outcomes. No such timings have been executed here.

## 4. Build/cache feasibility and preservation boundary

Read actual C26 NATIVE_BUILD.json:
command=["cargo","build","--release","--offline","--bin","quartz_c26_game_metareasoning","-j","2"],
exit0,elapsed11.8475575s,
binary=R/target/release/quartz_c26_game_metareasoning,
binary_bytes618160, SHA8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c.

Current bounded storage reads:
- du -sh R/target ->404M.
- df -h /home/cosmosapjw/quantum/research /tmp -> same filesystem /dev/nvme0n1p2,915G total,867G used,2.3G available,rounded100%.
The reading is point-in-time; future backup/build work consumes this space.

C26 root .gitignore ignores /target/. Cargo.toml names package mcts_demo0.3.0, default features empty, opt-level3,fatLTO,codegen-units1. C26 isolated binary imports ../game.rs and ../games/tictactoe.rs by path. It need not alter core source to introduce another uniquely named bin.

Feasible later-build option A: execute Cargo against the C27 worktree's Cargo.toml, choose a new unique --bin name and set CARGO_TARGET_DIR to existing R/target; use --locked --offline and bounded -j2. Cargo can reuse dependencies while adding/replacing cache records. This does not require changing old tracked source or Cargo.lock; the old worktree should remain git-clean. It does mutate the old target cache, so source/status preservation is not byte-for-byte preservation of every old generated artifact. Keep the original C26 executable SHA before/after, record new source/binary SHA and do not overwrite or reuse the old C26 executable filename. Shared Cargo cache may serialize concurrent builds; do not run cargo clean there. This is a feasibility analysis, not an executed build guarantee.

Option B: new external target such as /home/cosmosapjw/quantum/research/quartz_c27_target_20261123 isolates generated artifacts and old cache completely, at the cost of rebuilding dependencies and extra disk. It is on the same nearly full filesystem; /tmp is not an extra disk here. The present404M old target is a capacity reference, not a guaranteed size for a fresh C27 build. Creating an additional full source/target copy consumes more space without improving source isolation beyond an existing worktree. Select based on the parent's backup and old-artifact preservation contract.

## Read-only evidence and completion

Read actual c26_campaign.py245–326,348–450,480–511,761–843,1162–1317; Rust0–29 and798–887; source/model provenance/raw first row and bounded JSON key/count/hash projection; Cargo.toml,.gitignore,NATIVE_BUILD.json. Shell reads used rg,gzip/head,du,df and a Python artifact-only JSON projection. No helper model function or native program was called. Available full90 vector provenance and pure fold refit meet the proposed R09 data requirements. New candidate timing fields are required later in coding; old artifacts do not contain them.
