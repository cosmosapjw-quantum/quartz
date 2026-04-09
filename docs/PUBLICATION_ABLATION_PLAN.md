# Publication Ablation Plan

## Core Hypothesis

QUARTZ adaptive stopping saves search budget when the evaluator is strong
enough (loss < ~1.0), while maintaining search quality.

## Key Finding (Current Results)

P_flip does NOT converge below threshold (0.159) with any non-NN evaluator:
- Uniform: P_flip ~ 0.45 (no information)
- BiasedStrong: P_flip ~ 0.44 (biased but still noisy)
- ShortRollout: P_flip ~ 0.47 (rollout noise too high)

This confirms the theoretical prediction: P_flip convergence requires
evaluator loss < ~1.0 (i.e., a trained neural network).

## Required Next Steps

### Phase 1: Train NN Models (varying quality)

Train gomoku7 models to different quality levels:
- Early checkpoint (5 iterations): high loss (~2.0), weak model
- Mid checkpoint (20 iterations): medium loss (~1.0), transitional
- Late checkpoint (50 iterations): low loss (<1.0), strong model

Command:
  python3 -m quartz.train --game gomoku7 --iterations 50 --seed 42

Save checkpoints at iterations 5, 10, 20, 30, 40, 50.

### Phase 2: P_flip Convergence with NN

For each checkpoint model:
1. Run search_nn with Rust MCTS + checkpoint NN
2. Track P_flip vs iteration count
3. Measure: at what loss level does P_flip drop below threshold?

Expected result:
- loss > 1.5: P_flip stays high (no convergence)
- loss ~ 1.0: P_flip starts dropping (transitional)
- loss < 0.8: P_flip converges below threshold → QUARTZ stops early

### Phase 3: Budget Savings Measurement

Compare Fixed vs VOC halt modes with each checkpoint:
- Fixed(800): always uses 800 iterations
- VOC: stops when P_flip < threshold AND VOC <= 0

Metric: budget_savings = 1 - (VOC_iters / Fixed_iters)

Expected:
- Weak model: savings ~ 0% (P_flip never converges)
- Strong model: savings > 30% (early convergence)
- Move agreement vs Fixed should be >= 95% (quality preserved)

### Phase 4: Publication Figures

1. P_flip convergence curves (Figure 1)
   - X: iteration count
   - Y: P_flip
   - Lines: different model qualities (loss levels)
   - Horizontal line: threshold = 0.159

2. Budget savings vs model quality (Figure 2)
   - X: training loss
   - Y: budget savings (%)
   - Error bars: std across positions

3. Search quality vs budget (Figure 3)
   - X: budget (iterations)
   - Y: move agreement with high-budget reference
   - Lines: Fixed vs VOC vs Threshold

4. QUARTZ mode comparison (Table 1)
   - Rows: None / GatedRefresh / SelfAdaptive / PFlipMixture
   - Columns: Agreement / P_flip / sigma_Q / NPS / Budget saved

## Current Ablation Results (Baseline)

### With ShortRollout (non-NN evaluator)

P_flip convergence: NO convergence (P_flip ~ 0.4-0.5 at all budgets)
Adaptive stopping: NO budget savings (all evaluators use full budget)
QUARTZ modes: None=75%, GatedRefresh=55%, SelfAdaptive=55%, PFlipMixture=65%

These results establish the baseline: P_flip-based adaptive stopping
requires a trained NN to be effective. The QUARTZ framework correctly
identifies this regime and uses full budget when appropriate.

## Adaptive VL Results (Parallel Search)

### With ShortRollout (4 threads, 500 iters)

Component isolation:
- VvisitOnly matches Adaptive in agreement
- Fixed VL over-pessimises by ~10x (AvgVV=1.0 vs 0.1)
- DupRate: Adaptive=0.38, Fixed=0.27 (Adaptive allows more exploration)

QUARTZ x VL interaction:
- SelfAdaptive + Fixed VL = worst combination (double pessimism)
- Adaptive VL auto-corrects via sigma_Q scaling

Default: VlMode::Adaptive (ablation-backed decision)
