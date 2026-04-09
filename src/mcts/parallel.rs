//! § ParallelismController — Adaptive Virtual Loss for tree-parallel MCTS
//!
//! Separated from the search-policy controller (QUARTZ) by design:
//! - QUARTZ controls **what** to search (penalty, refresh, halt)
//! - ParallelismController controls **how** to parallelise (VL magnitude)
//!
//! # Design: Split VL
//!
//! Traditional VL applies a single value: `n += 1, w -= 1`.
//! We split into two independently adaptive components:
//!
//! - **vvisit (λ_v)**: virtual visit inflation → reduces exploration bonus
//! - **vvalue (λ_q)**: virtual value penalty  → pessimises Q estimate
//!
//! # Hyperparameter-free derivation
//!
//! All parameters are derived from observable search state:
//!
//! - `λ_v = 1.0` (topological: 1 reservation per pending thread — constant)
//! - `λ_q = σ_Q × depth_decay` (scales with Q uncertainty — adaptive)
//! - `depth_decay = 1 / (1 + depth)` (root contention most expensive)
//!
//! # Evidence (ablation_vl.rs, gomoku7 × ShortRollout)
//!
//! - Adaptive VL default: +5-10%p agreement over Fixed VL at 4 threads
//! - Fixed VL=1 over-pessimises Q by ~10× (AvgVV=1.0 vs σ_Q-scaled=0.1)
//! - Fixed VL + SelfAdaptive QUARTZ penalty = destructive interaction (40%)
//! - Adaptive VL + SelfAdaptive = rescued to 50% (σ_Q auto-correction)
//! - VvisitOnly (reservation) is the primary mechanism; vvalue is refinement
//! - Adaptive advantage is largest at low budgets, converges at high budget

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ───────────────────────────────────────────
// § VL Mode
// ───────────────────────────────────────────

/// Virtual loss strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VlMode {
    /// Classic fixed VL=1 (baseline for ablation).
    Fixed,
    /// No virtual loss at all (serial-equivalent for ablation).
    Disabled,
    /// Adaptive: vvisit=1, vvalue=σ_Q/(1+depth).
    /// Hyperparameter-free — reads σ_Q from search state.
    Adaptive,
    /// Ablation: vvisit=1 only (reservation, no pessimism).
    VvisitOnly,
    /// Ablation: vvalue=σ_Q/(1+depth) only (pessimism, no reservation).
    VvalueOnly,
}

impl Default for VlMode {
    fn default() -> Self {
        VlMode::Adaptive
    }
}

// ───────────────────────────────────────────
// § Split VL values
// ───────────────────────────────────────────

/// Split virtual loss magnitudes for one select→backup cycle.
#[derive(Debug, Clone, Copy)]
pub struct VlSplit {
    /// Virtual visit: inflates N_eff, reduces PUCT exploration bonus.
    pub vvisit: f32,
    /// Virtual value: pessimises Q_eff, diverts other threads.
    pub vvalue: f32,
}

impl VlSplit {
    pub const FIXED: VlSplit = VlSplit {
        vvisit: 1.0,
        vvalue: 1.0,
    };
    pub const ZERO: VlSplit = VlSplit {
        vvisit: 0.0,
        vvalue: 0.0,
    };
}

// ───────────────────────────────────────────
// § Telemetry (parallel-specific)
// ───────────────────────────────────────────

/// Runtime parallel telemetry — accumulated across iterations.
pub struct ParallelTelemetry {
    pub total_selects: AtomicU32,
    pub dup_leaf_count: AtomicU32, // times a leaf was already pending
    pub max_pending: AtomicU32,    // peak pending leaves across all nodes
    pub vvalue_sum: AtomicU64,     // sum of applied vvalues (f64 bits)
}

impl ParallelTelemetry {
    pub fn new() -> Self {
        ParallelTelemetry {
            total_selects: AtomicU32::new(0),
            dup_leaf_count: AtomicU32::new(0),
            max_pending: AtomicU32::new(0),
            vvalue_sum: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn record_select(&self, pending_at_node: u32) {
        self.total_selects.fetch_add(1, Ordering::Relaxed);
        let prev = self.max_pending.load(Ordering::Relaxed);
        if pending_at_node > prev {
            self.max_pending.store(pending_at_node, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_dup_leaf(&self) {
        self.dup_leaf_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_vvalue(&self, v: f32) {
        // Simple add — not exact for concurrent but fine for telemetry
        let old = f64::from_bits(self.vvalue_sum.load(Ordering::Relaxed));
        self.vvalue_sum
            .store((old + v as f64).to_bits(), Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.total_selects.store(0, Ordering::Relaxed);
        self.dup_leaf_count.store(0, Ordering::Relaxed);
        self.max_pending.store(0, Ordering::Relaxed);
        self.vvalue_sum.store(0.0f64.to_bits(), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dup_rate(&self) -> f32 {
        let total = self.total_selects.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.dup_leaf_count.load(Ordering::Relaxed) as f32 / total as f32
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            total_selects: self.total_selects.load(Ordering::Relaxed),
            dup_leaf_count: self.dup_leaf_count.load(Ordering::Relaxed),
            max_pending: self.max_pending.load(Ordering::Relaxed),
            avg_vvalue: {
                let t = self.total_selects.load(Ordering::Relaxed);
                if t == 0 {
                    0.0
                } else {
                    let s = f64::from_bits(self.vvalue_sum.load(Ordering::Relaxed));
                    (s / t as f64) as f32
                }
            },
            dup_rate: self.dup_rate(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TelemetrySnapshot {
    pub total_selects: u32,
    pub dup_leaf_count: u32,
    pub max_pending: u32,
    pub avg_vvalue: f32,
    pub dup_rate: f32,
}

// ───────────────────────────────────────────
// § ParallelismController
// ───────────────────────────────────────────

/// Adaptive virtual loss controller. Separated from QUARTZ.
///
/// Reads QUARTZ diagnostic output (σ_Q) in one-way coupling.
/// Computes per-depth (vvisit, vvalue) split for each select step.
///
/// Thread-safe: all mutable state uses atomics.
pub struct ParallelismController {
    mode: VlMode,
    n_threads: AtomicU32,
    /// σ_Q from QUARTZ (updated periodically). Stored as f32 bits in AtomicU32.
    sigma_q: AtomicU32,
    /// Root policy entropy from QUARTZ. Stored as f32 bits in AtomicU32.
    root_entropy: AtomicU32,
    /// Cached `(clamp(root_entropy, 0.5, 2.0) / 2.0)` used in adaptive VL.
    entropy_factor: AtomicU32,
    pub telemetry: ParallelTelemetry,
}

impl ParallelismController {
    pub fn new(mode: VlMode, n_threads: u32) -> Self {
        ParallelismController {
            mode,
            n_threads: AtomicU32::new(n_threads.max(1)),
            sigma_q: AtomicU32::new(0.3_f32.to_bits()),
            root_entropy: AtomicU32::new(1.0_f32.to_bits()),
            entropy_factor: AtomicU32::new(0.5_f32.to_bits()),
            telemetry: ParallelTelemetry::new(),
        }
    }

    /// Update thread count (e.g., when switching between single and parallel search).
    pub fn set_n_threads(&self, n: u32) {
        self.n_threads.store(n.max(1), Ordering::Relaxed);
    }

    /// One-way read from QUARTZ: update σ_Q and root_entropy.
    /// Called by thread 0 periodically during run_par.
    pub fn update_from_search(&self, sigma_q: f32, root_entropy: f32) {
        let root_entropy = root_entropy.max(0.01);
        self.sigma_q
            .store(sigma_q.max(0.01).to_bits(), Ordering::Relaxed);
        self.root_entropy
            .store(root_entropy.to_bits(), Ordering::Relaxed);
        self.entropy_factor.store(
            (root_entropy.clamp(0.5, 2.0) / 2.0).to_bits(),
            Ordering::Relaxed,
        );
    }

    #[inline(always)]
    fn sigma_q_val(&self) -> f32 {
        f32::from_bits(self.sigma_q.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn root_entropy_val(&self) -> f32 {
        f32::from_bits(self.root_entropy.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn entropy_factor_val(&self) -> f32 {
        f32::from_bits(self.entropy_factor.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn adaptive_vvalue_at_depth(&self, depth: u32) -> f32 {
        let depth_decay = 1.0 / (1.0 + depth as f32);
        let sigma = self.sigma_q_val();
        let entropy_factor = self.entropy_factor_val();
        let dr = self.telemetry.dup_rate();
        let mp = self.telemetry.max_pending.load(Ordering::Relaxed) as f32;
        let contention = (mp / self.n_threads.load(Ordering::Relaxed).max(1) as f32).min(2.0);
        let amplifier = 1.0 + dr * (1.0 + contention);
        (sigma * depth_decay * entropy_factor * amplifier).max(0.01)
    }

    /// Compute split VL for a select step at given depth.
    ///
    /// State-derived control law with fixed constants:
    /// - vvisit = 1.0 (topological reservation)
    /// - vvalue = σ_Q × depth_decay × entropy_factor × contention_amplifier
    ///
    /// **Feedback loop** (2nd generation):
    /// - σ_Q: Q uncertainty from QUARTZ (one-way read)
    /// - depth_decay: 1/(1+d), root contention most expensive
    /// - entropy_factor: low entropy → less VL needed
    /// - contention_amplifier: f(dup_rate, max_pending, n_threads)
    ///   Combines duplicate frequency with contention severity.
    ///   High dup_rate + high pending → aggressive spread.
    ///   Low dup_rate + low pending → minimal overhead.
    #[inline(always)]
    pub fn vl_at_depth(&self, depth: u32) -> VlSplit {
        match self.mode {
            VlMode::Fixed => VlSplit::FIXED,
            VlMode::Disabled => VlSplit::ZERO,
            VlMode::Adaptive => VlSplit {
                vvisit: 1.0,
                vvalue: self.adaptive_vvalue_at_depth(depth),
            },
            VlMode::VvisitOnly => VlSplit {
                vvisit: 1.0,
                vvalue: 0.0,
            },
            VlMode::VvalueOnly => VlSplit {
                vvisit: 0.0,
                vvalue: self.adaptive_vvalue_at_depth(depth),
            },
        }
    }

    pub fn mode(&self) -> VlMode {
        self.mode
    }

    pub fn reset_for_search(&self) {
        self.telemetry.reset();
    }

    pub fn summary_string(&self) -> String {
        let snap = self.telemetry.snapshot();
        format!(
            "VL[{:?}] σ_Q={:.3} ent={:.3} dup_rate={:.3} max_pend={} avg_vv={:.3}",
            self.mode,
            self.sigma_q_val(),
            self.root_entropy_val(),
            snap.dup_rate,
            snap.max_pending,
            snap.avg_vvalue
        )
    }
}

// ───────────────────────────────────────────
// § Tests
// ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn reference_adaptive_vvalue(
        pc: &ParallelismController,
        depth: u32,
        include_vvisit: bool,
    ) -> VlSplit {
        let depth_decay = 1.0 / (1.0 + depth as f32);
        let sigma = pc.sigma_q_val();
        let ent = pc.root_entropy_val();
        let entropy_factor = (ent.min(2.0).max(0.5)) / 2.0;
        let dr = pc.telemetry.dup_rate();
        let mp = pc.telemetry.max_pending.load(Ordering::Relaxed) as f32;
        let contention = (mp / pc.n_threads.load(Ordering::Relaxed).max(1) as f32).min(2.0);
        let amplifier = 1.0 + dr * (1.0 + contention);
        let vvalue = (sigma * depth_decay * entropy_factor * amplifier).max(0.01);
        VlSplit {
            vvisit: if include_vvisit { 1.0 } else { 0.0 },
            vvalue,
        }
    }

    #[test]
    fn test_fixed_vl() {
        let pc = ParallelismController::new(VlMode::Fixed, 4);
        let vl = pc.vl_at_depth(0);
        assert_eq!(vl.vvisit, 1.0);
        assert_eq!(vl.vvalue, 1.0);
        let vl5 = pc.vl_at_depth(5);
        assert_eq!(vl5.vvisit, 1.0); // fixed doesn't decay
    }

    #[test]
    fn test_disabled_vl() {
        let pc = ParallelismController::new(VlMode::Disabled, 4);
        let vl = pc.vl_at_depth(0);
        assert_eq!(vl.vvisit, 0.0);
        assert_eq!(vl.vvalue, 0.0);
    }

    #[test]
    fn test_adaptive_depth_decay() {
        let pc = ParallelismController::new(VlMode::Adaptive, 4);
        pc.update_from_search(0.3, 1.5);

        let vl0 = pc.vl_at_depth(0);
        let vl3 = pc.vl_at_depth(3);
        let vl10 = pc.vl_at_depth(10);

        // vvisit always 1.0
        assert_eq!(vl0.vvisit, 1.0);
        assert_eq!(vl3.vvisit, 1.0);

        // vvalue decays with depth
        assert!(
            vl0.vvalue > vl3.vvalue,
            "depth 0 should have more VL than depth 3"
        );
        assert!(
            vl3.vvalue > vl10.vvalue,
            "depth 3 should have more VL than depth 10"
        );
    }

    #[test]
    fn test_adaptive_sigma_scaling() {
        let pc1 = ParallelismController::new(VlMode::Adaptive, 4);
        pc1.update_from_search(0.1, 1.5);
        let vl_low = pc1.vl_at_depth(0);

        let pc2 = ParallelismController::new(VlMode::Adaptive, 4);
        pc2.update_from_search(0.5, 1.5);
        let vl_high = pc2.vl_at_depth(0);

        assert!(
            vl_high.vvalue > vl_low.vvalue,
            "higher σ_Q should produce larger virtual value loss"
        );
    }

    #[test]
    fn test_entropy_modulation() {
        let pc_low = ParallelismController::new(VlMode::Adaptive, 4);
        pc_low.update_from_search(0.3, 0.2); // low entropy = dominant move
        let vl_low = pc_low.vl_at_depth(0);

        let pc_high = ParallelismController::new(VlMode::Adaptive, 4);
        pc_high.update_from_search(0.3, 2.0); // high entropy = uncertain
        let vl_high = pc_high.vl_at_depth(0);

        assert!(
            vl_high.vvalue > vl_low.vvalue,
            "higher entropy should produce larger VL (more contention risk)"
        );
    }

    #[test]
    fn test_telemetry() {
        let tel = ParallelTelemetry::new();
        tel.record_select(3);
        tel.record_select(5);
        tel.record_dup_leaf();
        let snap = tel.snapshot();
        assert_eq!(snap.total_selects, 2);
        assert_eq!(snap.dup_leaf_count, 1);
        assert_eq!(snap.max_pending, 5);
        assert!((snap.dup_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_reset() {
        let tel = ParallelTelemetry::new();
        tel.record_select(4);
        tel.record_dup_leaf();
        tel.record_vvalue(0.25);
        tel.reset();

        let snap = tel.snapshot();
        assert_eq!(snap.total_selects, 0);
        assert_eq!(snap.dup_leaf_count, 0);
        assert_eq!(snap.max_pending, 0);
        assert_eq!(snap.avg_vvalue, 0.0);
        assert_eq!(snap.dup_rate, 0.0);
    }

    #[test]
    fn test_cached_entropy_factor_matches_reference_formula() {
        let pc = ParallelismController::new(VlMode::Adaptive, 4);
        pc.update_from_search(0.37, 1.73);
        pc.telemetry.record_select(3);
        pc.telemetry.record_dup_leaf();

        let expected = reference_adaptive_vvalue(&pc, 2, true);
        let actual = pc.vl_at_depth(2);
        assert_eq!(actual.vvisit, expected.vvisit);
        assert_eq!(actual.vvalue.to_bits(), expected.vvalue.to_bits());
    }

    #[test]
    #[ignore]
    fn bench_parallel_vl_hot_path() {
        let pc = ParallelismController::new(VlMode::Adaptive, 8);
        pc.update_from_search(0.41, 1.61);
        for pending in [1, 2, 4, 3, 5, 2, 6, 1] {
            pc.telemetry.record_select(pending);
        }
        for _ in 0..3 {
            pc.telemetry.record_dup_leaf();
        }

        let rounds = 2_000_000u32;

        let start = Instant::now();
        let mut ref_acc = 0.0f32;
        for i in 0..rounds {
            let vl = reference_adaptive_vvalue(&pc, i % 8, true);
            ref_acc += black_box(vl.vvalue);
        }
        let ref_ns = start.elapsed().as_nanos();

        let start = Instant::now();
        let mut opt_acc = 0.0f32;
        for i in 0..rounds {
            let vl = pc.vl_at_depth(i % 8);
            opt_acc += black_box(vl.vvalue);
        }
        let opt_ns = start.elapsed().as_nanos();

        eprintln!(
            "\nParallel VL hot path: reference={}ns optimized={}ns speedup={:.2}x acc=({:.3},{:.3})",
            ref_ns,
            opt_ns,
            ref_ns as f64 / opt_ns as f64,
            ref_acc,
            opt_acc
        );
    }
}
