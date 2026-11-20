//! CPU-only bounded signed continuation-gain surrogate for gravity Connect4.
//!
//! This is NOT Bayesian KG: the signed realized gain can be negative.
//! It never sees external exact game-theoretic oracle labels at runtime.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use crate::game::{Evaluator, GameState};
use crate::games::connect4::Connect4;
use crate::games::connect4_conv::ConvConnect4Evaluator;
use crate::games::connect4_nn::TinyConnect4Evaluator;
use crate::mcts::eval::UniformEval;

pub const FEATURE_COUNT: usize = 9;
pub const STAGE_VISITS: [u32; 3] = [8, 16, 32];

pub struct CostedSignedGainPolicy {
    pub coefficients: [[f64; FEATURE_COUNT]; 3],
    pub forecast_cost_ms: [f64; 3],
    pub lambda_wdl_per_ms: f64,
}

impl CostedSignedGainPolicy {
    pub fn parse(json: &Value) -> Result<Self, String> {
        if json.get("schema").and_then(Value::as_str)
            != Some("quartz_cog02_signed_gain_surrogate_v1")
            || json.get("features").and_then(Value::as_str) != Some("root_visits_v1")
        {
            return Err("incompatible costed-signed-gain model schema".into());
        }
        let coefs = json
            .get("coefficients")
            .and_then(Value::as_array)
            .ok_or("missing COG02 coefficient matrix")?;
        let costs = json
            .get("forecast_cost_ms")
            .and_then(Value::as_array)
            .ok_or("missing COG02 per-transition cost forecast")?;
        if coefs.len() != 3 || costs.len() != 3 {
            return Err("COG02 requires exactly three stages".into());
        }
        let mut mat = [[0f64; FEATURE_COUNT]; 3];
        let mut c = [0f64; 3];
        for i in 0..3 {
            let row = coefs[i].as_array().ok_or("coefficient row missing")?;
            if row.len() != FEATURE_COUNT {
                return Err("COG02 requires nine current-root features per stage".into());
            }
            for j in 0..FEATURE_COUNT {
                let x = row[j].as_f64().ok_or("COG02 coefficient not numeric")?;
                if !x.is_finite() || x.abs() > 1e6 {
                    return Err("COG02 nonfinite/out-of-bounds coefficient".into());
                }
                mat[i][j] = x;
            }
            let cost = costs[i].as_f64().ok_or("COG02 cost not numeric")?;
            if !cost.is_finite() || !(0.0..=100000.0).contains(&cost) || cost == 0.0 {
                return Err("COG02 cost must be positive finite milliseconds".into());
            }
            c[i] = cost;
        }
        let lambda = json
            .get("lambda_wdl_per_ms")
            .and_then(Value::as_f64)
            .ok_or("missing COG02 cost price")?;
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err("COG02 lambda must be finite within [0,1]".into());
        }
        Ok(Self {
            coefficients: mat,
            forecast_cost_ms: c,
            lambda_wdl_per_ms: lambda,
        })
    }

    /// The input is the CURRENT observed MCTS state, before any next batch.
    pub fn predict(
        &self,
        stage: usize,
        features: &[f64; FEATURE_COUNT],
    ) -> Result<(f64, f64, bool), String> {
        if stage >= 3 || features.iter().any(|x| !x.is_finite()) {
            return Err("invalid COG02 stage/features".into());
        }
        let gain = self.coefficients[stage]
            .iter()
            .zip(features.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let cost = self.forecast_cost_ms[stage];
        // Continue only if forecast signed gain exceeds price of the next
        // batch. Equality stops deterministically.
        Ok((gain, cost, gain > self.lambda_wdl_per_ms * cost))
    }
}

/// All nine features are derived from available root statistics; in this exact
/// order the Python COG02 reference must reproduce them:
/// [1, budget/64, normalized_entropy, max_p, top1-top2,
///  number_legal/7, completed_evals/(budget+1), ply/42, black_to_move].
pub fn current_root_features(
    root: &Connect4,
    policy: &[(usize, f32)],
    eval_calls: u64,
    budget: u32,
) -> Result<[f64; FEATURE_COUNT], String> {
    if !STAGE_VISITS.contains(&budget) || policy.is_empty() {
        return Err("COG02 features only defined at 8/16/32 checkpoints".into());
    }
    let legal = root.legal_moves();
    let mut dense = [0f64; 7];
    let mut found = [false; 7];
    for &(col, p) in policy {
        if col >= 7 || !legal.contains(&col) || found[col] || !p.is_finite() || p < 0.0 {
            return Err("malformed current-root MCTS visit policy".into());
        }
        found[col] = true;
        dense[col] = p as f64;
    }
    let z = dense.iter().sum::<f64>();
    if (z - 1.0).abs() > 1e-4 {
        return Err("current-root visit policy not normalized".into());
    }
    for x in &mut dense {
        *x /= z;
    }
    let mut sorted = dense.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let largest = sorted[0];
    let second = sorted[1];
    let entropy = -dense
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| p * p.ln())
        .sum::<f64>()
        / 7f64.ln();
    let completed_ratio = (eval_calls as f64) / (budget as f64 + 1.0);
    if !completed_ratio.is_finite() || completed_ratio > 4.0 {
        return Err("unbounded evaluator-call fraction".into());
    }
    let features = [
        1.0,
        budget as f64 / 64.0,
        entropy,
        largest,
        largest - second,
        legal.len() as f64 / 7.0,
        completed_ratio,
        root.ply_count() as f64 / 42.0,
        if root.current_player() == 1 { 1.0 } else { 0.0 },
    ];
    if features.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite COG02 features".into());
    }
    Ok(features)
}

type DynConnect4Eval = Arc<dyn Evaluator<Connect4> + Send + Sync>;

/// Per-process research-only cache keyed by exact expected frozen content
/// digest supplied by the caller. The experiment runner independently hashes
/// every file before and after a run; this function does not claim to hash
/// arbitrary on-disk data at each lookup.
pub fn model_for_root(
    mode: &str,
    path: Option<&str>,
    frozen_id: Option<&str>,
    reuse: bool,
) -> Result<(DynConnect4Eval, bool), String> {
    if mode == "uniform" {
        return Ok((Arc::new(UniformEval), false));
    }
    if mode != "tiny_mlp" && mode != "conv3x3" {
        return Err("unknown research CPU evaluator mode".into());
    }
    let path = path
        .filter(|x| !x.is_empty())
        .ok_or("missing research model_path")?;
    let key = if reuse {
        let id = frozen_id.ok_or("cache_model requires exact frozen model SHA256")?;
        if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("cache_model requires 64-char SHA256 identity".into());
        }
        let canonical = std::fs::canonicalize(path)
            .map_err(|err| format!("missing cached model path: {err}"))?;
        Some(format!("{mode}|{}|{id}", canonical.display()))
    } else {
        None
    };
    static CACHE: OnceLock<Mutex<HashMap<String, DynConnect4Eval>>> = OnceLock::new();
    if let Some(ref key) = key {
        let lock = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let map = lock
            .lock()
            .map_err(|_| "COG02 evaluator cache lock poisoned")?;
        if let Some(value) = map.get(key) {
            return Ok((value.clone(), true));
        }
    }
    let evaluator: DynConnect4Eval = match mode {
        "tiny_mlp" => Arc::new(TinyConnect4Evaluator::from_file(path)?),
        "conv3x3" => Arc::new(ConvConnect4Evaluator::from_file(path)?),
        _ => return Err("unrecognized evaluator".into()),
    };
    if let Some(key) = key {
        let lock = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = lock
            .lock()
            .map_err(|_| "COG02 evaluator cache lock poisoned")?;
        map.insert(key, evaluator.clone());
    }
    Ok((evaluator, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_root_features_normalized_and_no_oracle() {
        let state = Connect4::from_history("1212").unwrap();
        let features = current_root_features(&state, &[(0, 0.75), (1, 0.25)], 5, 8).unwrap();
        assert_eq!(features.len(), 9);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[1], 0.125);
        assert!((features[3] - 0.75).abs() < 1e-9);
        assert!((features[4] - 0.5).abs() < 1e-9);
        assert_eq!(features[5], 1.0);
        assert_eq!(features[8], 1.0);
        assert!(features[2] > 0.0 && features[2] < 1.0);
    }
    #[test]
    fn reject_malformed_surrogate() {
        let bad = serde_json::json!({"schema":"quartz_cog02_signed_gain_surrogate_v1",
            "features":"root_visits_v1","coefficients":[[1]],"forecast_cost_ms":[1,1,1],
            "lambda_wdl_per_ms":0.0});
        assert!(CostedSignedGainPolicy::parse(&bad).is_err());
    }
}
