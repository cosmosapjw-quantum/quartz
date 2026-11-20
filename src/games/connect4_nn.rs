//! Small trained CPU-only policy/value MLP evaluator for standard gravity Connect4.
//! Actual weights are exported by the bounded Python teacher distillation.
//! No external exact oracle scores are read in this runtime path.
use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::connect4::Connect4;
use serde_json::Value;

pub struct TinyConnect4Evaluator {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl TinyConnect4Evaluator {
    fn vector(root: &Value, field: &str, expected: usize) -> Result<Vec<f32>, String> {
        let arr = root
            .get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing {field} vector"))?;
        if arr.len() != expected {
            return Err(format!(
                "{field} wrong vector length {} != {expected}",
                arr.len()
            ));
        }
        arr.iter()
            .map(|x| {
                let v = x.as_f64().ok_or_else(|| format!("{field} nonnumeric"))?;
                if !v.is_finite() || v.abs() > 1.0e12 {
                    return Err(format!("{field} nonfinite or huge weight"));
                }
                Ok(v as f32)
            })
            .collect()
    }

    fn matrix(
        root: &Value,
        field: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let matrix = root
            .get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing {field} matrix"))?;
        if matrix.len() != rows {
            return Err(format!(
                "{field} wrong matrix rows {} != {rows}",
                matrix.len()
            ));
        }
        matrix
            .iter()
            .map(|row| {
                let values = row
                    .as_array()
                    .ok_or_else(|| format!("{field} row not a vector"))?;
                if values.len() != cols {
                    return Err(format!("{field} row length {} != {cols}", values.len()));
                }
                values
                    .iter()
                    .map(|x| {
                        let v = x.as_f64().ok_or_else(|| format!("{field} nonnumeric"))?;
                        if !v.is_finite() || v.abs() > 1.0e12 {
                            return Err(format!("{field} nonfinite or huge matrix weight"));
                        }
                        Ok(v as f32)
                    })
                    .collect::<Result<Vec<f32>, String>>()
            })
            .collect()
    }

    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let json: Value =
            serde_json::from_str(text).map_err(|err| format!("invalid model JSON: {err}"))?;
        if json.get("format").and_then(Value::as_str) != Some("quartz_connect4_tiny_mlp_v1") {
            return Err("invalid tiny MLP format".into());
        }
        for (field, expected) in [("input_dim", 126), ("hidden_dim", 64), ("output_dim", 8)] {
            if json.get(field).and_then(Value::as_u64) != Some(expected) {
                return Err(format!("{field} incompatible with network"));
            }
        }
        let w1 = Self::matrix(&json, "w1", 64, 126)?;
        let b1 = Self::vector(&json, "b1", 64)?;
        let w2 = Self::matrix(&json, "w2", 8, 64)?;
        let b2 = Self::vector(&json, "b2", 8)?;
        Ok(Self { w1, b1, w2, b2 })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|err| format!("cannot read frozen recognizer weights: {err}"))?;
        Self::from_json_str(&contents)
    }

    pub fn predict_raw(&self, state: &Connect4) -> ([f32; 7], f32) {
        let input = state.encode_planes();
        assert_eq!(input.len(), 126, "Connect4 encoder must match frozen 3x6x7");
        let mut hidden = [0f32; 64];
        for (i, row) in self.w1.iter().enumerate() {
            let mut y = self.b1[i];
            for (weight, x) in row.iter().zip(input.iter()) {
                y += weight * x;
            }
            hidden[i] = y.max(0.0);
        }
        let mut out = [0f32; 8];
        for (i, row) in self.w2.iter().enumerate() {
            let mut y = self.b2[i];
            for (weight, x) in row.iter().zip(hidden.iter()) {
                y += weight * x;
            }
            out[i] = y;
        }
        let mut policy = [0f32; 7];
        policy.copy_from_slice(&out[..7]);
        (policy, out[7].tanh())
    }
}

impl Evaluator<Connect4> for TinyConnect4Evaluator {
    fn evaluate(&self, state: &Connect4) -> EvalResult<usize> {
        if state.is_terminal() {
            return EvalResult {
                policy: vec![],
                value: state.outcome(),
            };
        }
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult {
                policy: vec![],
                value: 0.0,
            };
        }
        let (logits, value) = self.predict_raw(state);
        let largest = legal
            .iter()
            .map(|&a| logits[a])
            .fold(f32::NEG_INFINITY, f32::max);
        let scores = legal
            .iter()
            .map(|&a| (logits[a] - largest).exp())
            .collect::<Vec<f32>>();
        let z = scores.iter().sum::<f32>();
        if !z.is_finite() || z <= 0.0 {
            return state.uniform_eval(value);
        }
        EvalResult {
            policy: legal
                .iter()
                .zip(scores.iter())
                .map(|(&a, &p)| (a, p / z))
                .collect(),
            value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_net() -> String {
        let w1 = vec![vec![0f32; 126]; 64];
        let w2 = vec![vec![0f32; 64]; 8];
        let b1 = vec![0f32; 64];
        let b2 = vec![0f32; 8];
        serde_json::json!({
            "format":"quartz_connect4_tiny_mlp_v1",
            "input_dim":126,"hidden_dim":64,"output_dim":8,
            "w1":w1,"b1":b1,"w2":w2,"b2":b2
        })
        .to_string()
    }

    #[test]
    fn tiny_nn_zero_weights_produce_legal_uniform_policy() {
        let model = TinyConnect4Evaluator::from_json_str(&zero_net()).unwrap();
        let full_col = Connect4::from_history("111111").unwrap();
        let eval = model.evaluate(&full_col);
        assert_eq!(eval.policy.len(), 6);
        assert_eq!(eval.value, 0.0);
        assert!(eval
            .policy
            .iter()
            .all(|&(a, p)| a != 0 && (p - 1.0 / 6.0).abs() < 1e-6));
        assert!(eval.policy.iter().any(|&(a, _)| a == 6));
    }

    #[test]
    fn tiny_nn_rejects_invalid_or_wrong_shape_weights() {
        assert!(TinyConnect4Evaluator::from_json_str("{}").is_err());
        let value = serde_json::json!({
            "format":"quartz_connect4_tiny_mlp_v1",
            "w1": vec![vec![0f32;126]; 63],
            "b1":vec![0f32;64],
            "w2":vec![vec![0f32;64];8],
            "b2":vec![0f32;8]
        });
        assert!(TinyConnect4Evaluator::from_json_str(&value.to_string()).is_err());
    }

    #[test]
    fn tiny_nn_real_weights_agree_with_independent_pytorch_parity_reference() {
        let file = concat!(
            "docs/research/cognitive_metareasoning_20261120/",
            "connect4_recognizer_06/evidence/nn_train_20261120/tiny_mlp_weights.json"
        );
        let expected = concat!(
            "docs/research/cognitive_metareasoning_20261120/",
            "connect4_recognizer_06/PARITY_REFERENCE.json"
        );
        let model = TinyConnect4Evaluator::from_file(file).expect("frozen CPU model");
        let data = std::fs::read_to_string(expected).expect("PyTorch parity reference");
        let reference: Value = serde_json::from_str(&data).unwrap();
        let rows = reference["records"].as_array().unwrap();
        assert_eq!(rows.len(), 12);
        for row in rows {
            let history = row["history"].as_str().unwrap();
            let state = Connect4::from_history(history).unwrap();
            let rust = model.evaluate(&state);
            let mut dense = [0f32; 7];
            for (col, p) in rust.policy {
                assert!(col < 7);
                dense[col] = p;
            }
            for (col, expected) in row["policy"].as_array().unwrap().iter().enumerate() {
                let target = expected.as_f64().unwrap() as f32;
                assert!(
                    (dense[col] - target).abs() <= 3e-5,
                    "policy disparity history={history} col={col}: rust={}, torch={target}",
                    dense[col]
                );
            }
            let v = row["value"].as_f64().unwrap() as f32;
            assert!(
                (rust.value - v).abs() <= 3e-5,
                "value mismatch history={history}"
            );
        }
    }

    #[test]
    fn updated_mlp_matches_generalization07_torch_parity() {
        let path = concat!(
            "docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/",
            "evidence/mlp_conv_train_20261121/mlp_model.json"
        );
        let expected = concat!(
            "docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/",
            "PARITY_REFERENCE.json"
        );
        let model = TinyConnect4Evaluator::from_file(path).expect("2026-11-21 fitted MLP");
        let raw = std::fs::read_to_string(expected).expect("36 frozen PyTorch traces");
        let data: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let samples = data["samples"].as_array().unwrap();
        assert_eq!(samples.len(), 36);
        for row in samples {
            let history = row["history"].as_str().unwrap();
            let state = Connect4::from_history(history).unwrap();
            let prediction = model.evaluate(&state);
            let mut policy = [0f32; 7];
            for (col, p) in prediction.policy {
                policy[col] = p;
            }
            for (col, expected) in row["mlp_policy"].as_array().unwrap().iter().enumerate() {
                let x = expected.as_f64().unwrap() as f32;
                assert!(
                    (x - policy[col]).abs() <= 3e-5,
                    "updated MLP policy mismatch {history} {col}"
                );
            }
            let value = row["mlp_value"].as_f64().unwrap() as f32;
            assert!(
                (prediction.value - value).abs() <= 3e-5,
                "updated MLP value mismatch {history}"
            );
        }
    }

    #[test]
    fn tiny_nn_real_export_loads_and_predicts_legal_probabilities() {
        let file = concat!(
            "docs/research/cognitive_metareasoning_20261120/",
            "connect4_recognizer_06/evidence/nn_train_20261120/tiny_mlp_weights.json"
        );
        let model = TinyConnect4Evaluator::from_file(file).expect("trained CPU model JSON");
        let state = Connect4::from_history("4433226223").unwrap();
        let output = model.evaluate(&state);
        assert_eq!(output.policy.len(), 7);
        assert!(output.value.is_finite() && output.value.abs() <= 1.0);
        assert!((output.policy.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
