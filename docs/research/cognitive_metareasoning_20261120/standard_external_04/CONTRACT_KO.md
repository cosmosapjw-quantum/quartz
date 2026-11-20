# QUARTZ-COG-01B-STANDARD-EXTERNAL-04 | CPU standard games with public engines
Date: 2026-11-20. Parent commit 6f6cd0a36ac94642c4cf1958ca96ca5ef9529ed3.
Branch: research/quartz-cog01b-standard-external-engines-20261120.

## User-authorized scientific scope change
The old non-gravity 7x7 connect-four-in-a-row is a hardware-limited reference only. Keep its preceding contract and evidence unchanged. New targets are DISTINCT games, not pooled:
1. Standard Connect Four: 7 columns by 6 rows, gravity, connect four. Evaluate exact action values using PascalPons/connect4.
2. Standard freestyle Gomoku: 15x15, no gravity, connect five. Evaluate external pretrained Rapfi NNUE proxy, plus actual QUARTZ CPU resident search 8->16.

## External reference identity and license
Pascal Pons solver official repo https://github.com/PascalPons/connect4 (AGPL-3.0), source SHA d6ba50d8aaf2308c769d9bf2abd42d90f34baf41. Locally compiled C++ solver uses -a -b 7x6_small.book, accepts 1-indexed column histories and emits seven action scores including -1000 invalid sentinel. The 6,291,513 byte opening book is the official release artifact.
Rapfi official repo https://github.com/dhbloo/rapfi (GPL-3.0), tag 250615 distribution Rapfi-engine.7z with Linux AVX2 executable and pretrained mix9svqfreestyle_bsmix weights. Tested legal Piskvork START 15, INFO rule 0, BOARD, DONE, END on CPU. It is a strong search proxy, not a full game solve.
QUARTZ 15x15 checkpoint from results/smoke_torch/gomoku15/models/F1_legacy_base/latest.pt is a SMOKE experiment model with unknown strength. CPU-only Torch 2.9.1; same-root 8->16 server diagnostics from resident fork source.
Exact external source, binary, book, model file SHA256 and version information are recorded in EXTERNAL_LOCK.json. Source/binaries/weights are not copied into git, ZIP handoff or source archives; standalone subprocess boundary is maintained.

## Fixed pilot design (CPU only)
Connect4: 12 legal nonterminal column histories, 3 at each ply count 10,14,18,22; deterministic generator random.Random(20261120), reject terminal histories. Ask solver for ALL seven move scores at each root. Interpret score sign as game-theoretic win/draw/loss, magnitude as relative win-time convention. Timeout is a separate unresolved status. Independent Python gravity and four-in-a-row checks required. Record exact raw output and program hashes.
Gomoku15: 12 legal nonterminal synthetic alternating local openings at plies 4,6,8,10 (3 each), generated with independent seed 20261120 and no already-won line. Rapfi on one CPU thread, rule 0, 500 ms configured time per move, capture full stdout and legal x,y. QUARTZ weak-smoke checkpoint same board, baseline_strict resident search 8->16, cache-disabled physical inference counts and signed reference-action agreement. Do NOT call Rapfi recommendation true U(a).
Never fabricate Elo, Bayesian VOC, correct Gomoku value, statistical independence, or strong NN performance from this small pilot. Keep both games separate and preserve all negative results.

## Publication and gates
A: lock rule, external source identity, input bytes, contract -> commit, Dropbox-folder archive SHA, GitHub push and ls-remote.
B: TDD generate positions and implement external invocation, rule verifier, QUARTZ CPU search -> tests, commit, backup, push.
C: execute 12+12 fixed cases from clean source; log raw results, executable/book/model/source SHA, independent readback -> commit, backup, push.
D: exact scientific claim verdict, failures and next executable step -> commit, backup, push. Never change main, old QUARTZ branches, other worktrees or their untracked files.

Next after this external pilot: integrate an explicit Connect4 GameState into QUARTZ for same-root 8->16 against exact oracle, OR acquire a demonstrably stronger Gomoku15 checkpoint and larger heldout positions versus Rapfi. No GPU prerequisite.
