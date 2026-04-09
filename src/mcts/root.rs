//! 루트 착수 선택 + Dirichlet noise

use std::sync::atomic::Ordering;
use std::sync::Arc;

use rand::distributions::Distribution;
use rand_distr::Dirichlet;

use crate::mcts::node::MctsNode;

// ─────────────────────────────────────────────
// § Dirichlet Noise
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DirichletConfig {
    /// noise 집중도 (바둑=0.03, 체스=0.3, 오목≈0.15)
    pub alpha: f32,
    /// noise 비율 (AlphaZero: 0.25)
    pub epsilon: f32,
}

impl DirichletConfig {
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        DirichletConfig { alpha, epsilon }
    }
    pub fn alphazero_default() -> Self {
        DirichletConfig {
            alpha: 0.03,
            epsilon: 0.25,
        }
    }
    pub fn gomoku() -> Self {
        DirichletConfig {
            alpha: 0.15,
            epsilon: 0.25,
        }
    }
}

/// Dirichlet noise overlay 계산 → adjusted prior delta vec
/// 반환: 각 엣지에 더할 noise (ε · (dir[i] - p[i])) 형태
pub fn compute_dirichlet_noise(n_edges: usize, priors: &[f32], cfg: &DirichletConfig) -> Vec<f32> {
    if n_edges == 0 {
        return vec![];
    }
    let alphas = vec![cfg.alpha as f64; n_edges];
    let dir = Dirichlet::new(&alphas).unwrap();
    let sample: Vec<f64> = dir.sample(&mut rand::thread_rng());
    sample
        .iter()
        .zip(priors.iter().chain(std::iter::repeat(&0.0)))
        .map(|(&d, &p)| cfg.epsilon * (d as f32 - p))
        .collect()
}

// ─────────────────────────────────────────────
// § 착수 선택 (Temperature)
// ─────────────────────────────────────────────

/// temperature=0 → argmax N, temperature>0 → N^(1/T) 비례 샘플링
pub fn select_move_with_temperature<M: Copy + Send + Sync + 'static>(
    node: &Arc<MctsNode<M>>,
    temperature: f32,
) -> Option<M> {
    let edge_arcs = node.edge_snapshot(node.materialized_count());
    if edge_arcs.is_empty() {
        return None;
    }

    if temperature <= 0.0 {
        // Greedy argmax
        edge_arcs
            .iter()
            .max_by_key(|e| e.n.load(Ordering::Acquire))
            .map(|e| e.mv)
    } else {
        use rand::distributions::WeightedIndex;
        let weights: Vec<f64> = edge_arcs
            .iter()
            .map(|e| {
                let n = e.n.load(Ordering::Acquire) as f64;
                n.powf(1.0 / temperature as f64)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total < 1e-12 {
            return edge_arcs.first().map(|e| e.mv);
        }
        let wi = WeightedIndex::new(&weights).ok()?;
        let idx = wi.sample(&mut rand::thread_rng());
        Some(edge_arcs[idx].mv)
    }
}

/// 방문 분포 π (AlphaZero 학습 타겟)
pub fn visit_distribution<M: Copy + Send + Sync + 'static>(
    node: &Arc<MctsNode<M>>,
    temperature: f32,
) -> Vec<(M, f32)> {
    let edge_arcs = node.edge_snapshot(node.materialized_count());
    if edge_arcs.is_empty() {
        return vec![];
    }

    let counts: Vec<f64> = edge_arcs
        .iter()
        .map(|e| {
            let n = e.n.load(Ordering::Acquire) as f64;
            if temperature <= 0.0 {
                n
            } else {
                n.powf(1.0 / temperature as f64)
            }
        })
        .collect();
    let total: f64 = counts.iter().sum();
    if total < 1e-12 {
        return vec![];
    }

    edge_arcs
        .iter()
        .zip(counts.iter())
        .map(|(e, &c)| (e.mv, (c / total) as f32))
        .collect()
}
