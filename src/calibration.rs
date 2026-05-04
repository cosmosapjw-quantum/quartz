//! Cross-game σ₀ calibration — measures σ_Q and ħ_eff across all game types
//! to determine optimal σ₀ per game.
//!
//! Q9 (audit_codex_20260428.md W'9): in addition to printing aggregates to
//! stderr, the calibration suite now writes a JSON recommendation file
//! (`results/sigma_0_recommendation.json` by default) so Optuna sweeps and
//! per-game catalogs can consume the calibrated σ₀ value programmatically
//! instead of guessing. The recommendation is the σ₀ whose ħ_eff is
//! closest to 1.0 across the swept grid; ties resolve toward the smaller
//! σ₀ (more conservative).

use std::sync::Arc;
use std::time::Instant;

use crate::game::{Evaluator, GameState};
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::quartz::{QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine, PwConfig};

/// Q9: per-game observation row produced by `sigma_scan`. Aggregated across
/// all games and surfaced as JSON for downstream consumers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SigmaScanRow {
    pub label: String,
    pub sigma_0: f32,
    pub mean_sigma_q: f32,
    pub mean_hbar_eff: f32,
    pub mean_p_flip: f32,
    pub converged_count: u32,
    pub positions: u32,
}

impl SigmaScanRow {
    /// Q9: distance from the calibration target ħ_eff ≈ 1.0. The σ₀ that
    /// minimizes this is the recommended value for that game.
    pub fn hbar_distance(&self) -> f32 {
        (self.mean_hbar_eff - 1.0).abs()
    }
}

/// Q9: pure helper picking the σ₀ closest to ħ_eff=1 from a per-game scan.
/// Ties resolve toward the smaller σ₀ (more conservative). Returns None on
/// empty input.
pub fn recommend_sigma_0(rows: &[SigmaScanRow]) -> Option<f32> {
    rows.iter()
        .min_by(|a, b| {
            let da = a.hbar_distance();
            let db = b.hbar_distance();
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.sigma_0
                        .partial_cmp(&b.sigma_0)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .map(|row| row.sigma_0)
}

fn sigma_scan<G: GameState, E: Evaluator<G> + Send + Sync + 'static>(
    label: &str,
    positions: &[G],
    eval: Arc<E>,
    budget: u32,
    sigma_values: &[f32],
) -> Vec<SigmaScanRow> {
    eprintln!(
        "  {:>20}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
        label, "σ₀", "σ_Q", "ħ_eff", "P_flip", "conv"
    );
    eprintln!("  {:>20}  {}", "", "-".repeat(50));

    let mut rows = Vec::with_capacity(sigma_values.len());
    for &sigma_0 in sigma_values {
        let n = positions.len() as f32;
        let mut tot_sq = 0.0f32;
        let mut tot_hbar = 0.0f32;
        let mut tot_pf = 0.0f32;
        let mut tot_conv = 0u32;

        for state in positions {
            let qcfg = QuartzConfig {
                sigma_0,
                min_visits: 30,
                check_interval: 30,
                ..Default::default()
            };
            let config = MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5))
                .with_quartz(qcfg.clone());
            let engine = MctsEngine::new(state.clone(), eval.clone(), config);
            let mut ctrl = QuartzController::new(budget, qcfg);
            engine.run_quartz(&mut ctrl);
            let s = ctrl.last_stats();
            tot_sq += s.sigma_q;
            tot_hbar += s.hbar_eff;
            tot_pf += s.p_flip;
            if s.converged {
                tot_conv += 1;
            }
        }

        eprintln!(
            "  {:>20}  {:>6.2}  {:>8.4}  {:>8.4}  {:>8.4}  {:>4}/{}",
            "",
            sigma_0,
            tot_sq / n,
            tot_hbar / n,
            tot_pf / n,
            tot_conv,
            positions.len()
        );

        rows.push(SigmaScanRow {
            label: label.to_string(),
            sigma_0,
            mean_sigma_q: tot_sq / n,
            mean_hbar_eff: tot_hbar / n,
            mean_p_flip: tot_pf / n,
            converged_count: tot_conv,
            positions: positions.len() as u32,
        });
    }
    eprintln!();
    rows
}

/// Q9: write a JSON recommendation file. `per_game_rows` is a map from
/// game label to the full sigma scan for that game; `recommendations`
/// stores the picked σ₀ per game (and an "all" key for the cross-game
/// minimum-distance pick across all rows).
pub fn write_sigma_recommendation(
    path: &std::path::Path,
    per_game_rows: &std::collections::BTreeMap<String, Vec<SigmaScanRow>>,
) -> std::io::Result<()> {
    use std::collections::BTreeMap;
    let mut recommendations: BTreeMap<String, Option<f32>> = BTreeMap::new();
    let mut all_rows: Vec<SigmaScanRow> = Vec::new();
    for (label, rows) in per_game_rows {
        recommendations.insert(label.clone(), recommend_sigma_0(rows));
        all_rows.extend(rows.iter().cloned());
    }
    recommendations.insert("__cross_game__".to_string(), recommend_sigma_0(&all_rows));

    let payload = serde_json::json!({
        "schema_version": 1,
        "tool": "calibration::cross_game_sigma_calibration",
        "target": "ħ_eff ≈ 1.0",
        "tie_break": "prefer smaller σ₀",
        "per_game_rows": per_game_rows,
        "recommendations": recommendations,
    });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::chess::Chess;
    use crate::games::go::Go;
    use crate::games::gomoku15::{Gomoku15, GomokuVariant};

    fn random_mid_positions<G: GameState>(
        initial: G,
        n: usize,
        moves_range: (usize, usize),
    ) -> Vec<G> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let mut positions = Vec::new();
        for _ in 0..n * 5 {
            let n_moves = moves_range.0 + rng.gen::<usize>() % (moves_range.1 - moves_range.0);
            let mut s = initial.clone();
            for _ in 0..n_moves {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_moves();
                if legal.is_empty() {
                    break;
                }
                s = s.apply_move(legal[rng.gen::<usize>() % legal.len()]);
            }
            if !s.is_terminal() && s.legal_moves().len() >= 3 {
                positions.push(s);
                if positions.len() >= n {
                    break;
                }
            }
        }
        positions
    }

    #[test]
    #[ignore]
    fn cross_game_sigma_calibration() {
        let sigmas = [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
        let budget = 300u32;

        eprintln!("\n{}", "═".repeat(75));
        eprintln!(
            "  Cross-Game σ₀ Calibration (UniformEval, {} iters, 8 positions each)",
            budget
        );
        eprintln!("  Target: ħ_eff ≈ 1.0 (σ_Q ≈ σ₀)");
        eprintln!("{}\n", "═".repeat(75));

        let mut per_game: std::collections::BTreeMap<String, Vec<SigmaScanRow>> =
            std::collections::BTreeMap::new();

        // Gomoku15 Standard
        let positions = random_mid_positions(Gomoku15::standard(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        per_game.insert(
            "Gomoku15-Std".to_string(),
            sigma_scan("Gomoku15-Std", &positions, eval, budget, &sigmas),
        );

        // Gomoku15 Omok
        let positions = random_mid_positions(Gomoku15::omok(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        per_game.insert(
            "Gomoku15-Omok".to_string(),
            sigma_scan("Gomoku15-Omok", &positions, eval, budget, &sigmas),
        );

        // Chess
        let positions = random_mid_positions(Chess::standard(), 8, (6, 20));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        per_game.insert(
            "Chess".to_string(),
            sigma_scan("Chess", &positions, eval, budget, &sigmas),
        );

        // Go 9×9
        let positions = random_mid_positions(Go::new_9x9(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        per_game.insert(
            "Go-9x9".to_string(),
            sigma_scan("Go-9x9", &positions, eval, budget, &sigmas),
        );

        eprintln!("  Interpretation:");
        eprintln!("  - ħ_eff >> 1: σ₀ too low → QUARTZ over-corrects");
        eprintln!("  - ħ_eff << 1: σ₀ too high → QUARTZ under-corrects");
        eprintln!("  - ħ_eff ≈ 1: optimal → σ₀ matches actual evaluation noise\n");

        // Q9 (audit_codex_20260428.md W'9): emit JSON recommendation so
        // Optuna sweeps and per-game catalogs can consume the calibrated
        // σ₀ programmatically. The path is overridable via env var so
        // CI / scripted runs don't clobber the canonical artifact.
        let out_path = std::env::var("QUARTZ_SIGMA_RECOMMENDATION_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "results/rust_ablations/calibration/sigma_0_recommendation.json",
                )
            });
        match write_sigma_recommendation(&out_path, &per_game) {
            Ok(()) => eprintln!("  σ₀ recommendation -> {}", out_path.display()),
            Err(e) => eprintln!("  [warn] could not write {}: {}", out_path.display(), e),
        }
    }

    #[test]
    fn test_q9_recommend_sigma_0_picks_minimum_hbar_distance() {
        // Q9 (audit_codex_20260428.md W'9): pure-helper test, no engine.
        // Pin the contract: recommendation is the σ₀ whose mean ħ_eff is
        // closest to 1.0; ties resolve toward the smaller σ₀.
        let make = |sigma_0: f32, hbar: f32| SigmaScanRow {
            label: "g".to_string(),
            sigma_0,
            mean_sigma_q: 0.0,
            mean_hbar_eff: hbar,
            mean_p_flip: 0.0,
            converged_count: 0,
            positions: 1,
        };
        // Closest to 1.0 is σ=0.3 (hbar=0.95, distance 0.05).
        let rows = vec![
            make(0.1, 1.4),
            make(0.2, 1.2),
            make(0.3, 0.95),
            make(0.5, 0.7),
        ];
        assert_eq!(recommend_sigma_0(&rows), Some(0.3));

        // Tie-break: two rows with equal distance 0.10 — must pick the
        // smaller σ₀ (more conservative).
        let rows_tie = vec![make(0.2, 0.9), make(0.5, 1.1)];
        assert_eq!(recommend_sigma_0(&rows_tie), Some(0.2));

        // Empty input is None, not a panic.
        let none_rows: Vec<SigmaScanRow> = Vec::new();
        assert_eq!(recommend_sigma_0(&none_rows), None);
    }

    #[test]
    fn test_q9_write_sigma_recommendation_serializes_per_game_and_cross_game() {
        // Q9: full JSON contract — file exists, schema_version=1, has
        // both `per_game_rows` and `recommendations.__cross_game__`.
        use std::collections::BTreeMap;
        let mut per_game: BTreeMap<String, Vec<SigmaScanRow>> = BTreeMap::new();
        per_game.insert(
            "g1".to_string(),
            vec![
                SigmaScanRow {
                    label: "g1".to_string(),
                    sigma_0: 0.2,
                    mean_sigma_q: 0.18,
                    mean_hbar_eff: 0.9,
                    mean_p_flip: 0.05,
                    converged_count: 4,
                    positions: 8,
                },
                SigmaScanRow {
                    label: "g1".to_string(),
                    sigma_0: 0.5,
                    mean_sigma_q: 0.50,
                    mean_hbar_eff: 1.05,
                    mean_p_flip: 0.10,
                    converged_count: 5,
                    positions: 8,
                },
            ],
        );

        let tmp = std::env::temp_dir().join("quartz_q9_sigma_test.json");
        if tmp.exists() {
            let _ = std::fs::remove_file(&tmp);
        }
        write_sigma_recommendation(&tmp, &per_game).expect("write should succeed");
        let body = std::fs::read_to_string(&tmp).expect("file should exist");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["schema_version"], 1);
        // Per-game scan is preserved.
        assert!(json["per_game_rows"]["g1"].is_array());
        // Per-game recommendation: 0.5 (hbar=1.05 is closer to 1.0 than 0.9).
        assert_eq!(json["recommendations"]["g1"], 0.5);
        // Cross-game recommendation aggregates the same set.
        assert_eq!(json["recommendations"]["__cross_game__"], 0.5);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    #[ignore]
    fn cross_game_sigma_with_rollout() {
        let sigmas = [0.1f32, 0.2, 0.3, 0.5, 0.7];
        let budget = 300u32;

        eprintln!("\n{}", "═".repeat(75));
        eprintln!(
            "  Cross-Game σ₀ Calibration (ShortRollout(10), {} iters)",
            budget
        );
        eprintln!("{}\n", "═".repeat(75));

        let positions = random_mid_positions(Gomoku15::standard(), 8, (10, 30));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Gomoku15-Std(SR)", &positions, eval, budget, &sigmas);

        let positions = random_mid_positions(Chess::standard(), 8, (6, 20));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Chess(SR)", &positions, eval, budget, &sigmas);

        let positions = random_mid_positions(Go::new_9x9(), 8, (10, 30));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Go-9x9(SR)", &positions, eval, budget, &sigmas);

        eprintln!();
    }
}
