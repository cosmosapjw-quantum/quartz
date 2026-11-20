# QUARTZ standard games / external solver CPU pilot: results and scientific handoff

Date 2026-11-20. ID QUARTZ-COG-01B-STANDARD-EXTERNAL-04.
Branch: research/quartz-cog01b-standard-external-engines-20261120.
Parent research commit: 6f6cd0a36ac94642c4cf1958ca96ca5ef9529ed3.
Full evidence generated at source commit: 8e0f8117e3b23d529a2571ca45b809c7020ceb93.
External/input identity: EXTERNAL_LOCK.json; deterministic root corpus: POSITIONS_FIXED.json.

## Research scope correction
The 7x7 non-gravity four-in-row experiment was only a CPU feasibility surrogate. Preserve original data/contract in decision_utility_03, but **do not** treat that as the ceiling on scientific goals. This pilot uses TWO different standard games: Connect4 7x6 gravity connect-four and Gomoku freestyle15 no gravity connect-five. Never compare utility or strength values across the two as if the rules coincided.

## A. Standard 7x6 Connect Four with a public exact oracle
Engine: PascalPons/connect4, public C++ source commit d6ba50d8aaf2308c769d9bf2abd42d90f34baf41, standalone compiled c4solver and official 7x6_small.book. Source AGPL-3.0. Compile and book SHA-256 recorded in EXTERNAL_LOCK.json. No third-party source or weights in this Git branch.

Generated 12 legal, nonterminal bottom-up gravity histories at 10,14,18,22 plies (3 at each), using a fixed pseudo-random process. All 12 yielded full seven-column solver-analyze vectors, including -1000 for full columns. Source-reported exact optimal root scores categorized:

| Root game-theoretic WDL from signed solver score | Roots |
|---|---:|
| Winning (+) | 10 |
| Drawing (0) | 1 |
| Losing (-) | 1 |

Mean standalone 7-action solve wall time 0.07064 s; maximum 0.16767 s on this host with small official book (no population performance claim). The exact integer scores also encode a win/loss speed preference and are not cardinal utilities. For subsequent action-value experiments choose explicitly U(a)=sign(score(a)) in {-1,0,+1}; comparison of raw score differences is a different ordinal/time-priority estimand.

Independent readback reran the external solver without -a and verified the **root score == max(action scores)** for each of 12, and reran each left-right mirrored history with -a, verifying full seven-action score vectors mirror for all 12. Independent 7x6 gravity/terminal checks found no illegal or already-won source history. This is a strong consistency check of **external-solver-reported exact values**, not a formal independent mathematical proof of the solver itself. QUARTZ does NOT yet implement the gravity-based Connect4 GameState, so these are an oracle corpus and **not** a QUARTZ Connect4 MCTS improvement result.

## B. 15x15 freestyle Gomoku with pretrained Rapfi NNUE
External strong proxy: official Rapfi tag 250615 standalone Linux AVX2 binary, configuration config.toml, and pretrained mix9svqfreestyle_bsmix.bin.lz4 NNUE weights included in author release. GPL-3.0. Input + binary hashes are in EXTERNAL_LOCK.json. Checked Piskvork START 15, INFO rule 0, 1 CPU thread, configured timeout_turn 500 ms, BOARD, DONE, and searched to full returned move before sending END. Every trial printed a real NNUE weight-load message.

QUARTZ model: local results/smoke_torch/gomoku15/models/F1_legacy_base/latest.pt, Torch CPU 2.9.1, 17 channels, 12 residual blocks, 128 filters, board 15, win=5. The checkpoint was produced by a SMOKE experiment and is NOT a validated strong pretrained Gomoku actor. 12 synthetic nonterminal local-center openings, at 4/6/8/10 plies, three each, with no already won five-in-a-row. All are BLACK to move due even-ply construction, an explicit severe coverage limitation.

QUARTZ Rust baseline_strict same-root resident MCTS performed 8 followed by +8 simulations with physical CPU NN cache disabled. Both 8 and 16 results were legal. All 12 positions gave the SAME action 173, before and after more search. None matched Rapfi's best action:

| Observable | Measured |
|---|---:|
| Rapfi NNUE-confirmed reference moves | 12 / 12 |
| QUARTZ initial 8-move best matching Rapfi | 0 / 12 |
| QUARTZ after 16 visits best matching Rapfi | 0 / 12 |
| 8->16 action changes | 0 / 12 |
| Physical NN item evaluations for extra steps | 96 (8 each) |
| Mean additional QUARTZ CPU search wall time | 100.99 ms |
| Mean Rapfi wall time (standalone Piskvork) | 603.9 ms |

Do not compare the two wall times as matched accuracy or equal-quality inference budgets: Rapfi is a strong NNUE alpha-beta engine with a ~500ms time allocation and QUARTZ is only 16 simulations using a weak smoke checkpoint. Neither the original metareasoning hypothesis nor an actual trained NN is tested for playing strength here.

## C. Source, execution, failed/negative evidence
- Raw evidence: evidence/20261120_standard_cpu/{c4_oracle_results.json,gomoku15_external_vs_quartz.json,summary.json,execution_manifest.json,INDEPENDENT_READBACK.json}. Source/input checksums are persistent.
- Independent audit: 12/12 root score consistency, 12/12 left/right score mirror symmetry, 12/12 pretrained Rapfi NNUE message identity, 12/12 Rust resident visits+physical eval checks. All 12 external input files matched SHA-256. Raw original values and disagreement cases retained.
- New unit tests 7 PASS plus previous QUARTZ scientific reference tests 50 PASS; new source contains a red-before-green implementation trace (initial NotImplementedError), and fixture correction (illegal-action expected failure requires valid pretrained-load precondition).
- A diagnostic first full smoke had an intermittent Rapfi no-move timeout and returned no external result. Following error instrumentation the next bounded smoke succeeded. No mechanism for the first transient error was demonstrated; do not equate retry success with an identified cause.
- All response files were produced in CPU-only environment. No native subagent spawn tool exists in this host; native_subagent=NOT_RUN. Parallel/Exa/SciSpace bibliographic evidence was used only for source discovery and not as proof of solver outputs.
- Standalone engine licenses require retaining attribution/appropriate source obligations if redistributed. External source, books, binaries, weights and license payloads were deliberately NOT vendored, pushed or included in QUARTZ archives. Local installations are separately hashed.

## D. Next executable research
PRIMARY: QUARTZ-COG-01B-CONNECT4-GAMESTATE-05. Add gravity legal moves, 7x6 four-in-a-row terminal rules and independent Rust/Python GameState parity tests. Wire the engine session to the existing CPU resident protocol; compare QUARTZ 8→16 actions against the 7-entry exact external signed oracle **at exactly matched logical root**. Stage outcome is Delta U = sign(score(action16)) - sign(score(action8)); sign is true game-theoretic W/D/L conditional on the external solver. Record actual NN evaluations, wall time, root hashes, and signed negative changes. Start with uniform and cheap heuristic evaluator if no real Connect4 pretrained network is sourced; separately train/distill a small recognizer from solver-labelled positions on CPU before claiming recognition-guided VOC. Do NOT inject oracle labels into live policy observation.

SECONDARY: QUARTZ-COG-01B-GOMOKU15-RAPFI-TEACHER-05. Use Rapfi 250615 NNUE as external teacher to curate larger legal, color-balanced, opening-family heldout freestyle-15 positions, with fixed time and independent checkpoint/position splits. Distill or obtain a truly pretrained QUARTZ-compatible policy/value checkpoint and test real same-root budget allocation. Rapfi's single move is not U(a) for every move. Claim no strength improvement until matched engine competitions against that reference and reproducible training evidence exist.

NEITHER follow-up requires a GPU for its initial small bounded CPU screening; larger NN training and broad competition performance are separately scoped resource decisions.

## Preservation and publishing
Stages A/B/C were committed, backed up to /home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups as archive tar.gz with hash, pushed to the dedicated GitHub research branch, and read back through ls-remote. Stage D publishes this handoff identically. The cloud Dropbox server's sync ACK and restored archive verification were NOT performed. Original QUARTZ developer checkout and untracked files remain untouched.

Primary sources: https://github.com/PascalPons/connect4 ; https://github.com/PascalPons/connect4/releases/tag/book ; https://github.com/dhbloo/rapfi ; https://github.com/dhbloo/rapfi/releases/tag/250615 ; https://proceedings.mlr.press/v124/sezener20a.html .
