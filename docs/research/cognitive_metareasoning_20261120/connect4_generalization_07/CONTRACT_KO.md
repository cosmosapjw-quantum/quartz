# QUARTZ Connect4 Recognizer Generalization 07 · CPU scientific contract

Date: 2026-11-21 (Asia/Seoul). Parent: `c92e77a1e818c0bb1359836dd4d8be09a529d287`. Branch: `research/quartz-cog01b-recognizer-generalization-20261121`.

## Scientific question

Can more diverse, exactly labelled standard 7×6 gravity Connect Four positions and simple spatial inductive bias improve action-value recognition **on held-out opening families**, compared with the prior 126→64→8 MLP, without a GPU? Does any improvement transfer to actual Rust MCTS decision WDL under bounded CPU visits, calls and wall-time? A better supervised classifier alone is not proof of Bayesian value-of-computation (VOC) control.

## Fixed source / data authority

- Frozen `COHORT_1024_FROZEN.json` generated with SEED=20261121, 64 different 5-ply prefix families modulo mirror. 16 positions/family, depths 9,10,13,14,17,18,21,22 twice, balanced current-to-move colors. 40 train families (640 positions), 12 validation families (192), 12 test families (192).
- Canonical left/right board identity is unique in this cohort and disjoint from ALL earlier COG06 192 positions and previous historical 12 benchmark roots. Their 5-ply prefixes, including mirrors, are also excluded. Data split is **frozen before labels**. All descendants and mirrored augmentations remain within their family/split.
- Exact label authority: standalone PascalPons Connect4 solver `c4solver -a -b 7x6_small.book`, source commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`; 5-second per position timeout and a 180-second whole process cap. Every one of the 7 integer action scores must match gravity legality and allowable score bounds. Timeouts remain explicit NOT_RESOLVED, not zero/draw. No third-party source/binary/book vendored or published.
- WDL utility per legal action `u(s,a)=sign score(s,a)` and optimal set `A*(s)={a:u(s,a)=max_b u(s,b)}`. Train target policy uniform on `A*(s)` and scalar value `v*(s)=max_b u(s,b)`. Costly win-distance scores are ordinal and NOT cardinal game utility.
- Training+validation/test label metric: selected action belongs to A*, mean exact realized regret `v*-u(s,a)`, value MSE, masked policy CE. Also compute random-legal exact regret **on the same rows**.
- Training input `3x6x7` relative piece/side planes exactly Rust `Connect4::encode_planes_into`. Value from own player's perspective. Mirror augmentation train only; never flip the WDL label sign on reflection.

## Fixed architecture comparison (before labels)

1. Baseline MLP retrained on new dataset, architecture `126→64 ReLU→8`, 7 policy logits + tanh(value). This distinguishes **more labels** from **inductive-bias** benefits; archive the older COG06 model as an immutable comparison.
2. Spatial ConvNet `Conv2d(3,16,3,padding=1)→ReLU→flatten(672)→Linear(672,8)`, seven policy logits and one tanh(value). One 3x3 layer preserves local 4-in-row geometry more naturally than flat MLP but is NOT guaranteed superior. No batchnorm, dropout or GPU.
3. Identical CPU optimizer and loss: Adam(lr=0.003,weight_decay=1e-4); batch64, 80 epochs, fixed seed 20261121, two Torch CPU threads, deterministic algorithm flag. `L=policy-optimal-set CE+0.5 MSE(value,v*)`. Select epoch by minimum validation loss for each architecture; test split must not be consulted to select a winner. Train source mirrored (1280 samples). No hyperparameter search.
4. Export portable float32 MLP and ConvNet weights with source/teacher hashes. Rust standalone in-process evaluator must reproduce 12+ fixed Python examples with tolerance ≤3e-5 on policy/value; invalid illegal action must get zero probability.
5. Original 12 roots remain out-of-domain fixed final benchmark. On the original 12, compare previous R06 MLP, retrained MLP, ConvNet and Uniform, at 8/16/32/64 cumulative MCTS root visits, seed17, single thread, same search profile. Report separately physical evaluator calls (per leaf) and actual *inference* time, model load/setup and total process wall-clock. Matched visits are not matched calls or wall time. All external labels joined ONLY AFTER MCTS search, never used as evaluator input.
6. For new family test roots, measure greedy policy WDL and optionally same-root search on a **separately frozen small representative subset** if resource feasible, without choosing cases based on observed model scores. Distinguish new test families from former 12 all-black roots.
7. No iid claim from 1024 descendant states: group all evaluation uncertainties by family (12 validation families, 12 test families). A single fixed model run does not support universal probability of outperforming other methods. Never manufacture p-values from 1024 correlated states.

## Acceptance and stop

- Cohort/split/board mirror and legal-root identity: byte SHA and deterministic reproducibility. Teacher result must preserve all 1024 rows and split/status counts; if incomplete, stop rather than quietly impute.
- TDD strict 3×6×7 encoder/target/architecture/export; one failing behavior test before each new implementation. No assertion weakening.
- Evaluate both models from same frozen data and optimization; independently inspect epoch selection, held-out error/regret and model SHA.
- Check Python↔Rust inference and MCTS parity with actual frozen weights; a failed test blocks the promotion of actual Rust results, not the earlier offline model output.
- Perform one science verification and at most one independent readback. Defects classified by theory/numerical/implementation/environment rather than repeatedly adding review gates.

## Bounded completion and publishing

A: cohort+contract+input bytes → Git commit + local Dropbox archive SHA256 + GitHub push + remote SHA readback.
B: exact teacher labels (full run) and tests → commit/archive/push/readback.
C: CPU MLP+ConvNet train/test, weights and diagnostic comparison → commit/archive/push/readback.
D: Rust learned-network parity and bounded CPU MCTS compare → commit/archive/push/readback.
E: final scoped scientific verdict, negative findings and next ID → commit/archive/push/readback.

Backup location is `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`; local folder SHA presence does not establish remote Dropbox cloud sync or restore success. Preserve developer checkout, all other branches/worktrees, external solver/book, and untracked files.

Related prior scientific work: Kong, Ram & Yu (2026), https://arxiv.org/abs/2607.08984 and Willemsen et al (2021), https://doi.org/10.1007/s00521-021-05928-5. Oracle-optimal action set training is already in literature; no new algorithmic novelty claim from its reuse here.
