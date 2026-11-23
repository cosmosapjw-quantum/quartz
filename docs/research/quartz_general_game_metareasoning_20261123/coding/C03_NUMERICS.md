# C03 — numerical and statistical error design

The solver carries integer -1/0/+1 root values; half-integer utilities are exact. There is no horizon truncation, alpha-beta pruning, cache or neural approximation in a complete query. Node counts are entered states. Candidate/incumbent ties are deterministic.

Empirical probabilities and mean costs use f64. Validate finite positive p and sum within1e-12, then normalize. Reservation formulas are continuous at support boundaries; near ties can change at roundoff scale. Test expected-excess residuals and independent exact fixtures. Max-subtracted Gibbs softmax handles large score contrasts; very small theoretical mass may underflow, so do not claim exact full support for arbitrary f64 inputs.

Every timing repetition reproduces the same path through frozen SplitMix64 and unbiased bounded-index rejection. Match final values, choices, query order and node components across9repeats and a separate trace. Four cues are features calibrated to minimax labels, not automatically a correct minimax posterior.

Error classes:
- Theory scope: correlation, changing costs, partial information and binding caps lie outside unrestricted Pandora optimality.
- Statistical closure: TRAIN probability/cost prediction error; measure Brier, cost residuals and within-root outcome residual dependence.
- Numeric: probability/index/softmax roundoff and elapsed-time noise; no minimax truncation.
- Implementation: signs, hidden oracle, duplicate query, forgotten cue/initial work and changed Gibbs support; target with independent oracle/replay.
- Runtime/environment: process, tool, build and storage failures, with preserved actual exits.

Primary TEST contrast is PANDORA minus INITIAL_STOP atcap9, priced2^-10utility/node. Median9timings perseed, average4seeds perparent, then equal-parent mean. Deterministic policy seed copies do not add sample size. Bootstrap the150parents (50perply) with2000paired draws stratified byply, seed2600923, percentile95%interval. Admit positive primary priced gain only if lower endpoint>0.

The finite-cohort mean is observed exactly for sampled paths. The interval assumes resampling comparable roots conditional on the frozen TRAIN model; it omits training uncertainty. D4-disjoint roots can share descendants. Do not promote this to universal or cross-game performance. Secondary comparisons are exploratory. Report wall-priced net at10^-6utility/microsecond and actual node/time totals even if they disagree with the primary surrogate. Original P2 NN-evaluation efficiency is not measured. Retain zero/negative gains.
