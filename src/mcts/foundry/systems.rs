//! Readout, representation, training-control, analysis, and deployment skeletons: A17-A23.

use std::collections::BTreeMap;

use super::types::{
    AxisStatus, CostVector, FoundryAxis, FoundryObservation, MetaAction, MetaProposal,
    ProposalEstimate,
};

#[derive(Clone, Debug)]
pub struct A17B13CurvatureReadout {
    pub curvature: f32,
}

impl Default for A17B13CurvatureReadout {
    fn default() -> Self {
        Self { curvature: 1.0 }
    }
}

impl FoundryAxis for A17B13CurvatureReadout {
    fn id(&self) -> &'static str {
        "A17.b13_curvature_readout"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::MechanismValid
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let mut telemetry = BTreeMap::new();
        telemetry.insert("curvature".into(), self.curvature as f64);
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate {
                confidence: 0.5,
                ..ProposalEstimate::default()
            },
            activation_guard: "readout-only by default; selection and training-target roles separate",
            explanation: "invoke the pinned Phase-15 one-loop readout implementation".into(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A18DiffusionRegularizedEvaluator {
    pub denoise_weight: f32,
    pub discrete_masking: bool,
}

impl Default for A18DiffusionRegularizedEvaluator {
    fn default() -> Self {
        Self {
            denoise_weight: 0.1,
            discrete_masking: false,
        }
    }
}

impl FoundryAxis for A18DiffusionRegularizedEvaluator {
    fn id(&self) -> &'static str {
        "A18.diffusion_regularized_evaluator"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        out.push(MetaProposal::noop(
            self.id(),
            "training-only denoising auxiliary; Rust evaluator remains deterministic and direct",
        ));
    }
}

#[derive(Clone, Debug)]
pub struct A19RwRestLiteEvaluator {
    pub nodes: u16,
    pub channels: u16,
    pub graph_seed: u64,
}

impl Default for A19RwRestLiteEvaluator {
    fn default() -> Self {
        Self {
            nodes: 40,
            channels: 144,
            graph_seed: 0,
        }
    }
}

impl FoundryAxis for A19RwRestLiteEvaluator {
    fn id(&self) -> &'static str {
        "A19.rw_rest_lite_evaluator"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, _observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let mut telemetry = BTreeMap::new();
        telemetry.insert("nodes".into(), self.nodes as f64);
        telemetry.insert("channels".into(), self.channels as f64);
        telemetry.insert("graph_seed".into(), self.graph_seed as f64);
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard: "controller frozen; graph-seed screen; static-pruned deployment graph",
            explanation: "register evaluator architecture ablation".into(),
            telemetry,
        });
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A20RegretStateArchive;

impl FoundryAxis for A20RegretStateArchive {
    fn id(&self) -> &'static str {
        "A20.regret_state_archive"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let instability = 1.0 - observation.extras.h1_stability.unwrap_or(0.0);
        let priority = instability.max(0.0)
            + observation.extras.omission_bound.max(0.0)
            + 0.25 * observation.extras.revision_count as f32
            + observation.extras.prior_visit_js.max(0.0);
        if priority <= 0.0 {
            return;
        }
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::ArchiveState { priority },
            estimate: ProposalEstimate::default(),
            activation_guard: "training-only; deduplicate by position group; sampling bias recorded",
            explanation: format!("archive priority={priority:.4}"),
            telemetry: BTreeMap::new(),
        });
    }
}

#[derive(Clone, Debug)]
pub struct A21CoherenceSignedPathShadow {
    pub decay: f32,
}

impl Default for A21CoherenceSignedPathShadow {
    fn default() -> Self {
        Self { decay: 0.05 }
    }
}

impl FoundryAxis for A21CoherenceSignedPathShadow {
    fn id(&self) -> &'static str {
        "A21.coherence_signed_path_shadow"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::AnalysisOnly
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let stability = observation.extras.h1_stability.unwrap_or(0.0);
        let coherence = (-self.decay * observation.snap.root_visits as f32).exp()
            * (1.0 - stability).max(0.0);
        let mut telemetry = BTreeMap::new();
        telemetry.insert("coherence".into(), coherence as f64);
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard: "shadow-only until predictive lift beyond ordinary disagreement is proven",
            explanation: format!("coherence gate={coherence:.6}"),
            telemetry,
        });
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct A22PhysicsFalsificationDashboard;

impl FoundryAxis for A22PhysicsFalsificationDashboard {
    fn id(&self) -> &'static str {
        "A22.physics_falsification_dashboard"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::AnalysisOnly
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        let mut telemetry = BTreeMap::new();
        telemetry.insert("budget".into(), observation.snap.root_visits as f64);
        telemetry.insert("entropy".into(), observation.extras.entropy as f64);
        telemetry.insert(
            "effective_branching".into(),
            observation.extras.effective_branching as f64,
        );
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate::default(),
            activation_guard: "analysis-only; explicit nulls; no FDT/Jarzynski without protocols",
            explanation: "record beta residual, redundancy, susceptibility, and scale-flow observables".into(),
            telemetry,
        });
    }
}

#[derive(Clone, Debug)]
pub struct A23CpuIncrementalPatternStudent {
    pub quantized: bool,
    pub incremental: bool,
}

impl Default for A23CpuIncrementalPatternStudent {
    fn default() -> Self {
        Self {
            quantized: true,
            incremental: true,
        }
    }
}

impl FoundryAxis for A23CpuIncrementalPatternStudent {
    fn id(&self) -> &'static str {
        "A23.cpu_incremental_pattern_student"
    }

    fn status(&self) -> AxisStatus {
        AxisStatus::Seed
    }

    fn propose(&self, observation: &FoundryObservation<'_>, out: &mut Vec<MetaProposal>) {
        out.push(MetaProposal {
            axis_id: self.id(),
            action: MetaAction::Noop,
            estimate: ProposalEstimate {
                cost: CostVector {
                    cpu_ms: observation.extras.runtime.eval_latency_ms,
                    ..CostVector::default()
                },
                ..ProposalEstimate::default()
            },
            activation_guard: "teacher/controller frozen; incremental cache correctness and fixed-time Elo",
            explanation: "register pattern-codebook/NNUE-like CPU evaluator comparison".into(),
            telemetry: BTreeMap::new(),
        });
    }
}
