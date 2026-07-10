# Phase15 B13 (H2 one-loop) — trained-checkpoint ablation smoke (2026-07-10)

> **Claim tier: SMOKE-VALIDATED.** Single lightly-trained checkpoint
> (gomoku7, 5 self-play generations), 4 fixed positions, 1 seed. This is a
> plumbing + direction check, **not** a research-grade quality claim
> (`docs/RESEARCH_READINESS.md` requires multi-seed, paired protocol, more
> positions). It closes the "run H2 on a trained checkpoint" deferred item
> from `docs/RESEARCH_PLAN_PARTB.md` §B2/H2.

## Setup

- GPU: RTX 3080 Ti, `torch 2.11.0+cu128`, `device: cuda` (all runs).
- Checkpoint: `quartz.train --game gomoku7 --iterations 30` (SIGTERM at the
  580s cap after generation 5; `best.pt` = gen_5). Architecture is the
  86-key `AlphaZeroNet(build_base_cfg("gomoku7"))` — **identical keyset** to
  the smoke fixture, so it loads through the phase15 pipeline unchanged.
- Ablation: `scripts/phase15_ablation_study.py --systems A4,B5,B13
  --budgets 8,16,32,64 --oracle-budget 128` over 4 positions.

## Result 1 — pipeline integrity (all green)

- Systems ran: A4, B5, B13. Trace-cache **8 hits / 4 misses** = exactly
  4 unique traces for 4 positions, all 3 systems sharing them →
  **same-trace paired-delta pairing holds on the trained net** (A0-b + B13
  registration confirmed again on real data).

## Result 2 — policy concentration (trained vs random-init)

Root visit-distribution `K_eff` per budget (49-cell board):

| bundle | budget 8 | 16 | 32 | 64 | note |
|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 | 1.0 | decided position (forced) |
| 1 | 3.8 | 4.0 | 8.2 | 20.6 | concentrates early, spreads late |
| 2 | 3.9 | 3.9 | 7.5 | 19.8 | " |
| 3 | 3.8 | 4.0 | 7.5 | 18.4 | " |

Trained net concentrates at low budget (`K_eff≈4` @8 vs random-init's ≈6.6)
but **still spreads with budget** (`K_eff→~20` @64): gomoku7 root search
opens new candidates rather than deepening the top arm as budget grows. So
the finite-N reframe's premise ("per-arm `N_a` grows with `N`") does **not**
cleanly hold even here — the tail keeps acquiring new `N_a≈1` arms.

## Result 3 — H2 kill test on the trained net

`one_loop_top1_delta` (mass pulled off the best arm), `curvature=1.0`:

| bundle | b8 | b16 | b32 | b64 | argmax preserved |
|---|---|---|---|---|---|
| 1 | 0.056 | 0.003 | 0.018 | 0.016 | ✔ all |
| 2 | 0.012 | 0.016 | 0.019 | 0.015 | ✔ all |
| 3 | 0.056 | 0.003 | 0.012 | 0.016 | ✔ all |

**Confirms the adversarial-verify correction generalizes to a trained net:**
`top1_delta` is **non-monotone** (largest at low budget on average, but
bumps up mid-ladder) and the **argmax is preserved at every budget**. The
earlier "monotone shrink" claim would have been wrong here too.

## Result 4 — B13 quality vs A4/B5 (the verdict)

Per-budget, B13 vs A4 and B5 are **identical on accuracy and top-k**:

| budget | acc_to_oracle (all 3) | topk (all 3) | kl_to_oracle A4/B5 | kl B13 |
|---|---|---|---|---|
| 8 | 1.000 | 1.000 | 0.8995 | 0.9117 |
| 16 | 1.000 | 1.000 | 0.9231 | 0.9254 |
| 32 | 0.750 | 1.000 | 0.5722 | **0.5654** |
| 64 | 0.250 | 1.000 | 0.1059 | **0.0948** |

A2-b analyzer (Bonferroni corrected_alpha = 0.025, paired bootstrap CI):

- `delta_accuracy_to_oracle = 0.0`, CI [0, 0] — indistinguishable.
- `delta_topk_recall = 0.0` — indistinguishable.
- `delta_kl_to_oracle = −0.0008`, CI **includes 0** — not distinguishable.
- Flag: `B13_rehearsal_lower_kl_without_accuracy_or_topk_loss_vs_A4`.

**Verdict: B13 is NON-HARMFUL but NOT a validated improvement.** This
**flips the random-init result** (where `curvature=1.0` gave a
distinguishably *worse* KL, +0.47): on the concentrated trained net the
correction is gentle and marginally KL-favorable at high budget (32/64),
but the effect is tiny (−0.0008) and statistically indistinguishable. The
A2-b guard **correctly refuses to crown B13** — no false champion.

## Conclusions

1. The full pipeline (train → GPU ablation → readout → analyzer) is intact
   on a trained, architecture-compatible checkpoint.
2. H2's finite-N curvature premise is only partially met at these budgets —
   gomoku7 search spreads rather than deepens — so `top1_delta` neither
   monotonically vanishes nor cleanly does on a trained net; **argmax is
   always preserved** (B13 is decision-safe).
3. B13 moved from *harmful* (random-init) to *non-harmful, marginally
   KL-favorable but indistinguishable* (trained). It is **not** a validated
   quality gain.

## Follow-ups (for a research-grade verdict)

- Longer training (the net was only gen_5) + multi-seed + more positions
  under `--research-grade`.
- Calibrate `curvature` on the concentrated regime (0.25 vs 1.0) and test
  whether the high-budget KL gain survives with a real CI.
- Stratify `one_loop_top1_delta` by the *decided vs contested* position
  bucket (bundle 0 was already decided; the effect lives in 1-3).
