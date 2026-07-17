//! Shared meta-control contracts for the QUARTZ idea foundry.
//!
//! This module is intentionally a skeleton and is not wired into `mcts::mod.rs`
//! yet.  It extends, rather than replaces, `policy::{SearchSnapshot, EdgeView,
//! PolicyCache}`.  Heavy modules observe a root epoch and emit proposals; a
//! single arbiter chooses one explicit computation action.

use std::collections::BTreeMap;

use crate::mcts::policy::{EdgeView, SearchSnapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisStatus {
    Seed,
    MechanismValid,
    Shadow,
    Conditional,
    ActiveExperimental,
    DeploymentCandidate,
    Dormant,
    AnalysisOnly,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetaAction {
    Stop { edge_pos: Option<u16> },
    Sample { edge_pos: u16, visits: u32 },
    Challenge { best_pos: u16, challenger_pos: u16, visits: u32 },
    Widen { count: u16 },
    Deepen { edge_pos: u16, visits: u32 },
    Prove { edge_pos: u16, budget: u32 },
    ResampleMode { mode_id: u16, count: u16 },
    MergeOrShare { state_key: u64 },
    SetBatch { batch_size: u16 },
    SetInflight { credit: u16 },
    SetThreads { threads: u16 },
    Reanalyse { state_key: u64 },
    ArchiveState { priority: f32 },
    Noop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostVector {
    pub nn_evals: f32,
    pub cpu_ms: f32,
    pub gpu_ms: f32,
    pub energy_proxy: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostPrices {
    pub nn_eval: f32,
    pub cpu_ms: f32,
    pub gpu_ms: f32,
    pub energy_proxy: f32,
}

impl CostVector {
    pub fn weighted(self, prices: CostPrices) -> f32 {
        self.nn_evals * prices.nn_eval
            + self.cpu_ms * prices.cpu_ms
            + self.gpu_ms * prices.gpu_ms
            + self.energy_proxy * prices.energy_proxy
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProposalEstimate {
    pub regret_reduction_mean: f32,
    pub regret_reduction_lcb: f32,
    pub confidence: f32,
    pub cost: CostVector,
}

impl ProposalEstimate {
    pub fn net_lcb(self, prices: CostPrices) -> f32 {
        self.regret_reduction_lcb - self.cost.weighted(prices)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetaProposal {
    pub axis_id: &'static str,
    pub action: MetaAction,
    pub estimate: ProposalEstimate,
    pub activation_guard: &'static str,
    pub explanation: String,
    pub telemetry: BTreeMap<String, f64>,
}

impl MetaProposal {
    pub fn noop(axis_id: &'static str, explanation: impl Into<String>) -> Self {
        Self {
            axis_id,
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard: "skeleton-only",
            explanation: explanation.into(),
            telemetry: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UncertaintyChannels {
    pub mc: f32,
    pub epistemic: f32,
    pub drift: f32,
    pub bias: f32,
}

impl UncertaintyChannels {
    pub fn conservative_sum(self) -> f32 {
        self.mc.max(0.0) + self.epistemic.max(0.0) + self.drift.max(0.0) + self.bias.max(0.0)
    }

    pub fn rss(self) -> f32 {
        (self.mc.max(0.0).powi(2)
            + self.epistemic.max(0.0).powi(2)
            + self.drift.max(0.0).powi(2)
            + self.bias.max(0.0).powi(2))
        .sqrt()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RuntimeSnapshot {
    pub threads: u16,
    pub batch_size: u16,
    pub inflight: u16,
    pub queue_wait_ms: f32,
    pub eval_latency_ms: f32,
    pub edge_duplicate_rate: f32,
    pub semantic_path_overlap: f32,
    pub max_pending: u16,
    pub tt_wait_ns: u64,
}

/// Extra root observations not yet present in the production `SearchSnapshot`.
///
/// The live implementation should either extend `SearchSnapshot` in a schema
/// bump or publish this beside it.  Do not repurpose existing fields silently.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoundryRootExtras {
    pub root_hash: u64,
    pub checkpoint_hash: u64,
    pub entropy: f32,
    pub effective_branching: f32,
    pub top2_margin: f32,
    pub margin_slope: f32,
    pub entropy_slope: f32,
    pub h1_stability: Option<f32>,
    pub p_flip: Option<f32>,
    pub prior_visit_js: f32,
    pub omission_bound: f32,
    pub revision_count: u16,
    pub runtime: RuntimeSnapshot,
}

pub struct FoundryObservation<'a> {
    pub snap: &'a SearchSnapshot,
    pub edges: &'a [EdgeView<'a>],
    pub extras: &'a FoundryRootExtras,
}

pub trait FoundryAxis: Send + Sync {
    fn id(&self) -> &'static str;
    fn status(&self) -> AxisStatus;
    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>);
}

pub trait FoundryArbiter: Send + Sync {
    fn choose<'a>(&self, proposals: &'a [MetaProposal], prices: CostPrices) -> Option<&'a MetaProposal>;
}

#[derive(Default)]
pub struct ConservativeArbiter;

impl FoundryArbiter for ConservativeArbiter {
    fn choose<'a>(&self, proposals: &'a [MetaProposal], prices: CostPrices) -> Option<&'a MetaProposal> {
        proposals
            .iter()
            .filter(|proposal| {
                matches!(proposal.action, MetaAction::Stop { .. })
                    || proposal.estimate.net_lcb(prices) > 0.0
            })
            .max_by(|a, b| {
                a.estimate
                    .net_lcb(prices)
                    .total_cmp(&b.estimate.net_lcb(prices))
            })
    }
}
