# QUARTZ COG-01B Connect4 Recognition Distillation 06

Date: 2026-11-20. Parent study commit `c4e219f930065c4e4b56eeb4f779f473a6c5b007`.
Branch: `research/quartz-cog01b-connect4-recognizer-20261120`.
This is the **next incomplete bounded stage** and not a new ten-cycle.

## Question and experimental control
Does a CPU-trained, oracle-labelled policy/value *recognizer* improve the fixed-budget QUARTZ Rust MCTS decision utility relative to its existing uniform evaluator, using real Connect Four (7×6 gravity, 4 in row) and the exact Pascal Pons WDL action oracle? Compare actual game-theoretic utility `U(s,a)=sign(S(s,a))`; do not compare integer win-distance scores as cardinal payoffs. All results must include negative signed changes and exact extra evaluator calls and wall time.

## Frozen input, splitting and teacher labels
- Prior 12 exact-score evaluation roots in `../standard_external_04/POSITIONS_FIXED.json` are **locked final heldout**, never teacher-trained or model-selected. Previous 12-root baseline is a fixed comparator; do not rewrite prior results.
- 192 **new** nonterminal positions, 12 mirror-canonical five-ply opening families ×16 descendants, generation seed 20261120, 8/2/2 whole families train/val/test (128/32/32). Plies 9,10,13,14,17,18,21,22 with both current players ±1. Parent histories and reflected histories share a group. The 12 old benchmark root families and mirrors are excluded by their initial five moves.
- Independent exact teacher: standalone Pascal Pons `connect4` C++ `c4solver -a -b 7x6_small.book`, source SHA d6ba50d8aaf2308c769d9bf2abd42d90f34baf41, executable and book hashes recorded. Each oracle call has a 5s timeout and strict legal-column + score-range guard. Non-returned or invalid results are `TIMEOUT/INVALID` and **excluded from training**, but retained in the manifest with their original frozen split; never invent labels. No Git repo contains third-party AGPL engine, code or book bytes.
- Targets: `U_star=max_a sign(score_a)` over legal actions; policy target **uniform over the entire optimal-WDL action set** (not single arbitrary fastest-win action), value target `U_star`. Illegal actions masked in policy logits. Do not alter labels based on evaluation performance.
- Features: 3 channels × 6×7 flat, relative current-player occupied (42), relative opponent (42), current-player-first color plane (42), matching `Connect4::encode_planes_into` exactly. Public source excludes move history from the neural feature; history family governs split only.

## Frozen architecture and resource budget
- One small MLP: input 126 -> hidden 64 ReLU -> 8 outputs (7 policy logits + tanh scalar value), two layers; float32. Train with CPU-only Torch (2 threads) using batch 64, Adam lr=0.003, weight_decay=1e-4, 80 epochs, fixed seed 20261120 and mirrored augmentation **training only**; val/test never augmented. Loss: optimal-set cross-entropy + 0.5×mean squared value error. Select checkpoint with lowest validation loss, ties earlier epoch; no test-driven tuning. Preserve all train/val/test metrics and ground-truth labels. If no learning or overfit, report negative result, do not silently change model size or epoch count.
- Export only this self-trained compact JSON weights + model identity, not public full weights or third-party binaries. Rust independent inference from exported `w1/b1/w2/b2` with f32, ReLU and policy softmax over legal columns; `value=tanh(output[7])`. No oracle queries are permitted on this path.
- State/evaluator/Tensor shapes, parity and masked invalid must be asserted, including a Python ↔ Rust agreement within tolerances (policy 1e-5, value 1e-5).
- MCTS: same 12 previous heldout roots, single seed 17, `baseline_strict`, visits 8→16→32→64, root hash fixed within each run, same config excluding prior/value source (uniform vs learned). Actual evaluator calls, wall time and exact outcome regret. Measure physical small-MLP calls (in-process Rust inference), not simply NN items in Python. Account for different single eval latency as well as calls.
- Interpretation: a different evaluator can affect search tree trajectories and evaluator-call count. Use matched **visits** for primary comparison; costs are observed and should never be falsely called equal if they differ. Optional cost-matched curves are a later study, not silently inferred from matched visits.

## Publication/checkpoints
A. Freeze theory, training corpus, external input hashes and schema -> commit, local Dropbox archive SHA, push, remote SHA readback.
B. Solve 192 labelled cases with bounded CPU exact teacher, save negatives/timeouts; implement Python trainer + Rust learned evaluator with red/green and CPU parity smoke -> commit + backup + push.
C. Train/lock checkpoint and compare uniform vs trained evaluator on 12 unchanged heldout roots; record model SHA, raw outputs, exact oracle regret + cost; stage backup+push.
D. Scientific verdict with explicit mechanism status and next executable ID, final tests, backup and push. Never edit original user checkout, other branches/worktrees or external engine. Cloud Dropbox sync ACK and restore are separate unverified properties.
