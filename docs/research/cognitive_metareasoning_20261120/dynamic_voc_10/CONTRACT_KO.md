# QUARTZ-COG-02-DYNAMIC-VOC-10: original finite synergy and actual CPU two-step MCTS option

Date 2026-11-21 Asia/Seoul. Parent `93e8e46cb8bfb6a891f8a38ef5ca0539d108611f`. Branch `research/quartz-cog02-dynamic-voc-20261121`. GPU unavailable; CPU-only. User authorizes scoped experiment, backup and publication each completed step.

## Precise theoretical and empirical claims

Four-world XOR: independent fair bits `B1,B2`, correct action is the XOR `Z=B1 xor B2`, payoff 1 if predicted correctly, otherwise 0. Prior best expected payoff 1/2. Observing either bit alone changes no posterior on Z; gross one-bit VOC=0. Observing both reveals Z, gross VOC=1/2. At cost c=1/10 per bit, one-step net=-1/10 but two-step net=3/10. This is an exact **abstract Bayesian information synergy** example, not yet a result about any Connect4 MCTS search trajectory. Rational exhaustive calculation in `prove_xor.py` uses `Fraction` and was independently confirmed by Wolfram Language enumeration on 2026-11-21.

Actual 7x6 gravity Connect4 game root has external Pascal Pons oracle score `S(s,a)`; illegal sentinel -1000, legal action utility `u(s,a)=sgn S(s,a)∈{-1,0,+1}`. `G_{8,16}=u(a_16)-u(a_8)`, `G_{8,32}=u(a_32)-u(a_8)`; analogous 16/32/64. These **signed realized gains** can be negative. Ridge conditional prediction `E[G_{b,t}|X_b]` is an empirical surrogate, NOT Bayesian preposterior knowledge gradient. Exact solver action scores are never sent to Rust runtime controller; only used for offline train labels and post-search evaluation.

## Fixed model and decision rule before testing

- Three CPU evaluator modes: original `uniform`, COG07 `new_mlp`, `conv`. Same MCTS evaluation configuration and deterministic seed17; model weights in-process warm cache.
- Exactly nine present-root features from `src/mcts/cog02.rs::current_root_features`: bias, normalized budget, seven-column visit entropy, max probability, top1-top2 gap, fraction legal, evaluation calls/(visits+1), plies/42, black side.
- Fixed 8→16→32→64 stage visits. For each stage i, supervised **one-step gain** `g_i(X_i)=β_i·X_i`. For i=0,1 additionally fit a **two-step joint gain** `h_i(X_i)=γ_i·X_i` from `U_{i+2}-U_i`. Fit regularized Ridge `alpha=1`, intercept unpenalized, independently for each mode and time horizon.
- Fit positive **release CPU search cost** from previous G07 640 train family roots: log of measured next-batch ms (with small positive epsilon), using same nine features and fixed Ridge alpha1. For long horizons label the sum of two consecutive observed batch costs. Transform cost predictor back with exp and *train-only residual mean calibration scale*, constrain positive finite. Never use current C09 or new C10 test cohorts to fit coefficients or select thresholds.
- Cost-aware dynamic policy at i=0,1 considers `stop`, `single`, `commit-two`. Let one-step net `g_i-λ c_i`, joint net `h_i-λ d_i`. Stop if both ≤0. If joint net>max(0,single), run through i+2 without inspecting i+1 for stop (committed two-step option). Otherwise run one batch and re-evaluate current-root observation. At i=2, allow only stop or single to64. Deterministic ties prefer stop then single. `commit-two` is an actual commitment option, not a Bayesian belief update.
- Myopic comparator shares EXACTLY the same trained one-step gain and release cost models but does not have joint `h_i,d_i`. Frozen COG09 legacy debug-cost/myopic policy is another untuned baseline.
- Precommitted λ candidates `[0,.25,.5,1,2,4]` in units WDL/ms, fixed external validation objective `mean exact U - 1.0×mean completed release root search ms` on G07 192 validation roots only. Choose λ for dynamic and myopic independently on validation, ties lower measured CPU and then larger λ. Also pick the time-matched-to-dynamic myopic λ using ONLY validation search-time mismatch as secondary comparator.
- Release fixed-clock comparison: one cap per mode matching dynamic's average *validation measured search CPU time*, selected from 0.01..10.0ms in 0.01 increments (post-checkpoint stop; overshoot possible). Test fixed-clock and myopic policy compare only under actual mean CPU cost difference ≤10% **for each new test cohort separately**. Report every mismatch and label it, not force equivalence.
- Keep real root action choices, evaluator counts, search+model load/setup costs, full resident prefix sanity. Report family-bootstrap conditional (12 per new cohort, 24 pooled), side-to-move/true WDL and decision-informative roots. Negative actual gains (including regret worsening) retained.

## Input separation

G07 exact teacher original 640 train and 192 validation families are only inputs to fitting and validation. Old G07 192 test, old COG06 192, historical12, C09 384 roots and two new C10 sets **never** used for gain/cost fitting or λ selection. `COHORT_384_FROZEN.json` generated from RNG 20261127/20261128 before exact solver labels, 24 new 5-ply mirror-canonical opening families x16 roots, 384 total, each cohort 192 roots with balanced black/white. Disjoint modulo mirror 5-ply prefix and canonical board-state from ALL listed prior cohorts. Two draws are not independent search PRNG trajectories; frozen search seed17 and Dirichlet off may be deterministic; do not count inert seed changes as independent samples.

## Experiment and acceptance

A. Freeze exact XOR, its independent Wolfram cross-check, new disjoint 384-root data, contract and input byte hashes. Git commit→local Dropbox tar archive→SHA256→GitHub push→ls-remote exact equality.
B. Implement Rust online two-step option with strict model/schema validation, current-feature-only decision and a committed two-stage skip of the intervening stop check. TDD RED→GREEN and Python parity against same frozen coefficients; publish each complete code phase.
C. Generate 2496 actual release Rust full64 trajectories (G07 832 train+val roots ×3 modes); join exact teacher ONLY after actual search. Train and select Ridge one/two-step signed gains and release costs using train/val only; record observed calibration/failure and freeze model bytes. Publish separately source and evidence.
D. Query exact external solver for 384 new root labels with explicit status/timeouts, separately audit original+mirror. On these new roots, run true live full64, new dynamic, same-model myopic, original debug-cost myopic, true live fixed16, validation-clock, optionally time-matched-myopic for 384×3×(6 or 7) real Rust requests. Run 3 modes in one warm-cache server; validate exact prefix and never feed any future steps or solver into online decisions. Publish scored raw evidence, non-IID family bootstrap and cost-mismatch classification.
E. Full Rust/Python tests, exact-source+binary+model SHA, independent solver mirror queries, terminate with scoped results, next node ID and verified final archive+GitHub SHA. Do NOT silently execute another 10-stage loop.

No genuine multi-agent native delegation tool exposed in this chat: `native_subagent: NOT_RUN`, use actual remote CPU and independent mathematical/solver crosschecks. Local Dropbox path for archive is `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`; cloud sync/restoration not assumed.

## Precedent / novelty guard

Hay et al. (2012), https://proceedings.mlr.press/r10/hay12a.html; Callaway et al. (2018), https://proceedings.mlr.press/r16/callaway18a.html; Sezener & Dayan (2020), https://proceedings.mlr.press/v124/sezener20a.html. They establish wider metareasoning precedent. This stage's XOR construction is elementary and not claimed as a novel theorem. Dynamic signed-gain heuristic is not automatically equivalent to the optimal metalevel policy or positive Bayesian KG.
