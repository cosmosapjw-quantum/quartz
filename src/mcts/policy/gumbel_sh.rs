//! BQ++ Phase 3: Gumbel without-replacement sampling + Sequential Halving.
//!
//! Direct match for the user's primary objective: reduce the number
//! of NN evals per move while preserving (or improving) play quality.
//! Pure PUCT at root visits all candidates roughly proportional to
//! prior; Gumbel SH replaces this with a candidate-set selection
//! (top-m by `log π̃₀(a) + g_a`, g_a ~ Gumbel(0, 1)) followed by
//! Sequential Halving over the candidate set.
//!
//! Reference:
//!     Danihelka, I., Guez, A., Schrittwieser, J., & Silver, D. (2022).
//!     "Policy Improvement by Planning with Gumbel." ICLR 2022.
//!
//!     Karnin, Z., Koren, T., & Somekh, O. (2013). "Almost optimal
//!     exploration in multi-armed bandits." ICML 2013. (Sequential
//!     Halving original.)
//!
//! The Python prototype at `prototype/bqpp_prototype/gumbel_sh.py`
//! validates the math; this module is the Rust port. Hand-derived
//! expected values cross-checked between the two implementations.

use rand::Rng;
use smallvec::SmallVec;

/// Sample one Gumbel(0, 1) variate via inverse-CDF: g = -ln(-ln(U)).
///
/// Standard reparameterization-trick technique; see
/// Maddison-Mnih-Teh 2017 "The Concrete Distribution" appendix.
/// The U value is clamped to (1e-12, 1 - 1e-12) to avoid log(0)
/// at the extreme tails.
#[inline]
pub fn sample_gumbel<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u: f32 = rng.gen();
    let u = u.clamp(1e-12, 1.0 - 1e-12);
    -((-u.ln()).ln())
}

/// Select top-m indices by `log π_i + g_i` (Gumbel-top-m sampling).
///
/// Returns indices of the m largest perturbed log-prior values, in
/// descending order. This is the without-replacement Plackett-Luce
/// sample (Yellott 1977 equivalence). Stable: with the same RNG state,
/// the same selection is produced.
///
/// For m = 1 this reduces to `argmax(log π_i + g_i)` which is a
/// single Gumbel-Max sample of the categorical distribution (the
/// textbook reparameterization trick).
pub fn gumbel_top_m<R: Rng + ?Sized>(
    log_priors: &[f32],
    m: usize,
    rng: &mut R,
) -> SmallVec<[u16; 32]> {
    let k = log_priors.len();
    if m == 0 || k == 0 {
        return SmallVec::new();
    }
    // Compute perturbed scores
    let mut perturbed: SmallVec<[(u16, f32); 64]> = SmallVec::with_capacity(k);
    for i in 0..k {
        let g = sample_gumbel(rng);
        perturbed.push((i as u16, log_priors[i] + g));
    }
    // Sort descending by perturbed score
    perturbed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let take = m.min(k);
    let mut out: SmallVec<[u16; 32]> = SmallVec::with_capacity(take);
    for i in 0..take {
        out.push(perturbed[i].0);
    }
    out
}

/// Anytime-resumable Sequential Halving bracket state.
///
/// Sequential Halving (Karnin-Koren-Somekh 2013) divides a fixed
/// budget B into ⌈log₂(m₀)⌉ rounds; each round halves the live-set
/// by dropping the bottom half by mean reward. Total visits per
/// round = ⌊B / (m_r * ⌈log₂(m₀)⌉)⌋ where m_r is the live-set size
/// in round r.
///
/// Anytime property: the bracket can be paused after any complete
/// round and resumed without changing the eventual selection (given
/// the same RNG state and arm_means).
#[derive(Clone, Debug)]
pub struct SequentialHalvingBracket {
    /// Currently-live candidate edge-local positions.
    pub candidates: SmallVec<[u16; 32]>,
    /// Total visit budget for the bracket.
    pub budget: u32,
    /// Initial candidate count (for round-budget arithmetic).
    pub n_initial_candidates: u16,
    /// Number of halving rounds completed.
    pub rounds_completed: u16,
    /// Cumulative visits consumed across all rounds.
    pub visits_consumed: u32,
}

impl SequentialHalvingBracket {
    pub fn new(candidates: SmallVec<[u16; 32]>, budget: u32) -> Self {
        let n_initial = candidates.len() as u16;
        Self {
            candidates,
            budget,
            n_initial_candidates: n_initial,
            rounds_completed: 0,
            visits_consumed: 0,
        }
    }

    /// Number of halving rounds in the bracket: ⌈log₂(m₀)⌉ for m₀ ≥ 2,
    /// else 1.
    pub fn n_total_rounds(&self) -> u16 {
        let m0 = self.n_initial_candidates.max(1);
        if m0 <= 1 {
            return 1;
        }
        ((m0 as f32).log2().ceil() as u16).max(1)
    }

    /// Per-arm visits in the current round.
    /// Formula: ⌊B / (m_r * total_rounds)⌋ where m_r is the current
    /// live-set size.
    pub fn round_budget(&self) -> u32 {
        let m_r = self.candidates.len() as u32;
        let rounds = self.n_total_rounds() as u32;
        if m_r == 0 || rounds == 0 {
            return 0;
        }
        (self.budget / (m_r * rounds)).max(1)
    }

    /// Bracket finished iff only 1 candidate left or all rounds used.
    pub fn is_done(&self) -> bool {
        self.candidates.len() <= 1 || self.rounds_completed >= self.n_total_rounds()
    }

    /// Advance one halving round. Drops the bottom half of the
    /// current candidates by `arm_means` (indexed by edge-local pos).
    /// Returns a new bracket; does not mutate self.
    pub fn advance_round(&self, arm_means: &[f32]) -> Self {
        if self.is_done() || self.candidates.len() <= 1 {
            return self.clone();
        }
        // Sort current candidates by mean (descending). Stable sort
        // for reproducibility under tie.
        let mut sorted: SmallVec<[u16; 32]> = self.candidates.clone();
        sorted.sort_by(|&a, &b| {
            let ma = arm_means.get(a as usize).copied().unwrap_or(f32::NEG_INFINITY);
            let mb = arm_means.get(b as usize).copied().unwrap_or(f32::NEG_INFINITY);
            mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep_n = (sorted.len() / 2).max(1);
        sorted.truncate(keep_n);
        let visits_this_round = self.round_budget() * (self.candidates.len() as u32);
        Self {
            candidates: sorted,
            budget: self.budget,
            n_initial_candidates: self.n_initial_candidates,
            rounds_completed: self.rounds_completed + 1,
            visits_consumed: self.visits_consumed + visits_this_round,
        }
    }

    /// Final selection: argmax mean over live candidates.
    pub fn select_winner(&self, arm_means: &[f32]) -> u16 {
        if self.candidates.is_empty() {
            return 0;
        }
        *self
            .candidates
            .iter()
            .max_by(|&&a, &&b| {
                let ma = arm_means.get(a as usize).copied().unwrap_or(f32::NEG_INFINITY);
                let mb = arm_means.get(b as usize).copied().unwrap_or(f32::NEG_INFINITY);
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }
}

/// Build the initial SH bracket from a Gumbel-top-m candidate set
/// over the prior.
pub fn initial_bracket<R: Rng + ?Sized>(
    log_priors: &[f32],
    m_initial: usize,
    budget: u32,
    rng: &mut R,
) -> SequentialHalvingBracket {
    let candidates = gumbel_top_m(log_priors, m_initial, rng);
    SequentialHalvingBracket::new(candidates, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Phase 3: Gumbel(0, 1) mean ≈ 0.5772 (Euler-Mascheroni constant).
    /// Cross-checked with Python prototype `test_sample_gumbel_distribution_mean`.
    #[test]
    fn test_phase3_gumbel_sample_mean() {
        let mut rng = StdRng::seed_from_u64(0);
        let n = 200_000;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            sum += sample_gumbel(&mut rng) as f64;
        }
        let mean = sum / n as f64;
        // Euler-Mascheroni constant ≈ 0.5772
        assert!((mean - 0.5772).abs() < 0.02, "sample mean = {mean}");
    }

    /// Phase 3: top-m returns the requested count (no duplicates).
    #[test]
    fn test_phase3_gumbel_top_m_count() {
        let log_priors: Vec<f32> = vec![0.5, 0.3, 0.1, 0.05, 0.05]
            .iter()
            .map(|p: &f32| p.ln())
            .collect();
        let mut rng = StdRng::seed_from_u64(0);
        let out = gumbel_top_m(&log_priors, 3, &mut rng);
        assert_eq!(out.len(), 3);
        let mut sorted = out.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "duplicates: {out:?}");
    }

    /// Phase 3: strong prior concentration ≥ 90% top-1.
    /// Cross-checked with Python prototype
    /// `test_gumbel_top_m_concentration_on_strong_prior`.
    #[test]
    fn test_phase3_gumbel_top_m_concentrates_on_strong_prior() {
        let log_priors: Vec<f32> = vec![0.97_f32.ln(), 0.01_f32.ln(), 0.01_f32.ln(), 0.01_f32.ln()];
        let mut rng = StdRng::seed_from_u64(0);
        let n_runs = 1000;
        let mut n_correct = 0;
        for _ in 0..n_runs {
            let top = gumbel_top_m(&log_priors, 1, &mut rng);
            if top[0] == 0 {
                n_correct += 1;
            }
        }
        let rate = n_correct as f32 / n_runs as f32;
        assert!(rate > 0.93, "rate = {rate}");
    }

    /// Phase 3: uniform prior distributes top-1 picks evenly.
    #[test]
    fn test_phase3_gumbel_top_m_uniform_distributes() {
        let k = 4;
        let log_priors: Vec<f32> = vec![(1.0_f32 / k as f32).ln(); k];
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = vec![0u32; k];
        let n_runs = 4000;
        for _ in 0..n_runs {
            let top = gumbel_top_m(&log_priors, 1, &mut rng);
            counts[top[0] as usize] += 1;
        }
        for c in counts {
            let rate = c as f32 / n_runs as f32;
            assert!((0.20..0.30).contains(&rate), "counts off");
        }
    }

    /// Phase 3: empty input returns empty output.
    #[test]
    fn test_phase3_gumbel_top_m_empty_input() {
        let mut rng = StdRng::seed_from_u64(0);
        assert!(gumbel_top_m(&[], 3, &mut rng).is_empty());
        let log_priors = vec![0.5_f32.ln(), 0.5_f32.ln()];
        assert!(gumbel_top_m(&log_priors, 0, &mut rng).is_empty());
    }

    /// Phase 3: SH bracket arithmetic — log_2(m0) rounds.
    #[test]
    fn test_phase3_sh_bracket_n_total_rounds() {
        for &(m, expected) in &[(2_u16, 1_u16), (4, 2), (8, 3), (5, 3), (1, 1)] {
            let bracket = SequentialHalvingBracket::new(
                (0..m).collect::<SmallVec<_>>(),
                100,
            );
            assert_eq!(
                bracket.n_total_rounds(),
                expected,
                "m={m} expected={expected} got={}",
                bracket.n_total_rounds()
            );
        }
    }

    /// Phase 3: advance_round halves the candidate set by mean.
    #[test]
    fn test_phase3_sh_advance_halves_candidates() {
        let bracket = SequentialHalvingBracket::new(
            (0_u16..4).collect::<SmallVec<_>>(),
            64,
        );
        let arm_means: Vec<f32> = vec![0.9, 0.7, 0.5, 0.3];
        let nb = bracket.advance_round(&arm_means);
        // After round 1: top half = arms 0, 1
        let mut got: Vec<u16> = nb.candidates.to_vec();
        got.sort();
        assert_eq!(got, vec![0, 1]);
        let nb2 = nb.advance_round(&arm_means);
        // After round 2: arm 0 only
        assert_eq!(nb2.candidates.as_slice(), &[0_u16]);
    }

    /// Phase 3: select_winner = argmax mean over live candidates.
    #[test]
    fn test_phase3_sh_select_winner() {
        let bracket = SequentialHalvingBracket::new(
            SmallVec::<[u16; 32]>::from_slice(&[0_u16, 2_u16]),
            64,
        );
        let arm_means: Vec<f32> = vec![0.9, 0.5, 0.7, 0.3];
        assert_eq!(bracket.select_winner(&arm_means), 0);
    }

    /// Phase 3: resumable property — pause-resume same as full bracket.
    /// Cross-checked with Python prototype
    /// `test_sh_resumable_property`.
    #[test]
    fn test_phase3_sh_resumable_property() {
        let log_priors: Vec<f32> = vec![0.1_f32.ln(); 8];
        let arm_means: Vec<f32> = vec![0.5, 0.6, 0.55, 0.7, 0.4, 0.3, 0.65, 0.45];

        // Run 1: full bracket
        let mut rng1 = StdRng::seed_from_u64(0);
        let b1 = initial_bracket(&log_priors, 8, 64, &mut rng1);
        let mut full = b1;
        while !full.is_done() {
            full = full.advance_round(&arm_means);
        }
        let full_winner = full.select_winner(&arm_means);

        // Run 2: same seed, manual pause-and-resume
        let mut rng2 = StdRng::seed_from_u64(0);
        let b2 = initial_bracket(&log_priors, 8, 64, &mut rng2);
        let paused = b2.advance_round(&arm_means);  // round 1 only
        let mut resumed = paused;
        while !resumed.is_done() {
            resumed = resumed.advance_round(&arm_means);
        }
        let resumed_winner = resumed.select_winner(&arm_means);

        assert_eq!(full_winner, resumed_winner);
    }

    /// Phase 3: bracket budget is non-zero for typical inputs.
    #[test]
    fn test_phase3_sh_round_budget_at_least_one() {
        let bracket = SequentialHalvingBracket::new(
            (0_u16..4).collect::<SmallVec<_>>(),
            8,  // small budget; per-arm-per-round = 8/(4*2) = 1
        );
        assert_eq!(bracket.round_budget(), 1);
    }
}
