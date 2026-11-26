//! Small spatial CPU-only policy/value evaluator for gravity Connect4.
//! Strictly frozen float32 Conv3x3(3→16) + ReLU + Linear(672→8).
//! No oracle actions or other teacher labels are used by runtime evaluation.
use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::connect4::Connect4;
use serde::Deserialize;

#[derive(Deserialize)]
struct SpatialWeights {
    format: String,
    input_planes: usize,
    board_height: usize,
    board_width: usize,
    conv_out_channels: usize,
    kernel_size: usize,
    padding: usize,
    head_dim: usize,
    conv_weight: Vec<Vec<Vec<Vec<f32>>>>,
    conv_bias: Vec<f32>,
    head_weight: Vec<Vec<f32>>,
    head_bias: Vec<f32>,
}

pub struct ConvConnect4Evaluator {
    weights: SpatialWeights,
}

impl ConvConnect4Evaluator {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let w: SpatialWeights = serde_json::from_str(text)
            .map_err(|err| format!("invalid Connect4 spatial model JSON: {err}"))?;
        if w.format != "quartz_connect4_conv3x3_v1"
            || w.input_planes != 3
            || w.board_height != 6
            || w.board_width != 7
            || w.conv_out_channels != 16
            || w.kernel_size != 3
            || w.padding != 1
            || w.head_dim != 8
        {
            return Err("incompatible Connect4 spatial model format/shape".into());
        }
        if w.conv_bias.len() != 16
            || w.conv_weight.len() != 16
            || w.head_weight.len() != 8
            || w.head_bias.len() != 8
        {
            return Err("invalid Connect4 spatial model vector/matrix size".into());
        }
        for output in &w.conv_weight {
            if output.len() != 3 {
                return Err("conv input channel count not three".into());
            }
            for input in output {
                if input.len() != 3 || input.iter().any(|row| row.len() != 3) {
                    return Err("conv kernel must be 3x3".into());
                }
                if input
                    .iter()
                    .flat_map(|row| row.iter())
                    .any(|v| !v.is_finite() || v.abs() > 1e12)
                {
                    return Err("invalid convolution weight".into());
                }
            }
        }
        if w.head_weight.iter().any(|row| row.len() != 672)
            || w.head_weight
                .iter()
                .flat_map(|row| row.iter())
                .any(|v| !v.is_finite() || v.abs() > 1e12)
            || w.conv_bias
                .iter()
                .chain(w.head_bias.iter())
                .any(|v| !v.is_finite() || v.abs() > 1e12)
        {
            return Err("invalid/nonfinite spatial head or bias weights".into());
        }
        Ok(Self { weights: w })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("missing trained spatial checkpoint: {e}"))?;
        Self::from_json_str(&contents)
    }

    /// Returns seven unmasked policy logits and current-player negamax value.
    pub fn predict_raw(&self, state: &Connect4) -> ([f32; 7], f32) {
        let input = state.encode_planes();
        assert_eq!(
            input.len(),
            126,
            "three 6x7 relative Connect4 planes required"
        );
        let mut hidden = [0f32; 672];
        for oc in 0..16 {
            for row in 0..6i32 {
                for col in 0..7i32 {
                    let mut val = self.weights.conv_bias[oc];
                    for ic in 0..3 {
                        for dy in 0..3 {
                            let yp = row + dy as i32 - 1;
                            if !(0..6).contains(&yp) {
                                continue;
                            }
                            for dx in 0..3 {
                                let xp = col + dx as i32 - 1;
                                if !(0..7).contains(&xp) {
                                    continue;
                                }
                                val += self.weights.conv_weight[oc][ic][dy][dx]
                                    * input[ic * 42 + yp as usize * 7 + xp as usize];
                            }
                        }
                    }
                    hidden[oc * 42 + row as usize * 7 + col as usize] = val.max(0.0);
                }
            }
        }
        let mut output = [0f32; 8];
        for (i, dense) in self.weights.head_weight.iter().enumerate() {
            let mut val = self.weights.head_bias[i];
            for (&w, &a) in dense.iter().zip(hidden.iter()) {
                val += w * a;
            }
            output[i] = val;
        }
        let mut policy = [0f32; 7];
        policy.copy_from_slice(&output[..7]);
        (policy, output[7].tanh())
    }
    /// True CPU multi-board kernel: same per-lane accumulation order as
    /// scalar inference, with batch lanes contiguous in the inner loop.
    pub fn evaluate_batch29(&self, states: &[Connect4]) -> Vec<EvalResult<usize>> {
        assert!(!states.is_empty() && states.len()<=8);
        assert!(states.iter().all(|s| !s.is_terminal()));
        let n=states.len();
        let mut input=[[0f32;8];126];
        for (b,s) in states.iter().enumerate() {
            let a=s.encode_planes();assert_eq!(a.len(),126);
            for i in 0..126 {input[i][b]=a[i];}
        }
        let mut hidden=[[0f32;8];672];
        for oc in 0..16 {
            for row in 0..6i32 {
                for col in 0..7i32 {
                    let mut v=[self.weights.conv_bias[oc];8];
                    for ic in 0..3 {
                        for dy in 0..3 {
                            let yp=row+dy as i32-1;if !(0..6).contains(&yp){continue;}
                            for dx in 0..3 {
                                let xp=col+dx as i32-1;if !(0..7).contains(&xp){continue;}
                                let w=self.weights.conv_weight[oc][ic][dy][dx];
                                let a=input[ic*42+yp as usize*7+xp as usize];
                                for b in 0..8 {v[b]+=w*a[b];}
                            }
                        }
                    }
                    for b in 0..8{v[b]=v[b].max(0.0);}
                    hidden[oc*42+row as usize*7+col as usize]=v;
                }
            }
        }
        let mut output=[[0f32;8];8];
        for (i,dense) in self.weights.head_weight.iter().enumerate() {
            let mut v=[self.weights.head_bias[i];8];
            for (&w,a) in dense.iter().zip(hidden.iter()){
                for b in 0..8{v[b]+=w*a[b];}
            }
            output[i]=v;
        }
        (0..n).map(|b|{
            let mut logits=[0f32;7];for i in 0..7{logits[i]=output[i][b];}
            self.finish29(&states[b],logits,output[7][b].tanh())
        }).collect()
    }
    fn batch30<const W: usize>(&self, states: &[Connect4]) -> Vec<EvalResult<usize>> {
        assert!(!states.is_empty() && states.len()<=W);
        assert!(states.iter().all(|s| !s.is_terminal()));
        let n=states.len();
        let mut input=[[0f32;W];126];
        for (b,s) in states.iter().enumerate() {
            let a=s.encode_planes();assert_eq!(a.len(),126);
            for i in 0..126 {input[i][b]=a[i];}
        }
        let mut hidden=[[0f32;W];672];
        for oc in 0..16 {
            for row in 0..6i32 {
                for col in 0..7i32 {
                    let mut v=[self.weights.conv_bias[oc];W];
                    for ic in 0..3 {
                        for dy in 0..3 {
                            let yp=row+dy as i32-1;if !(0..6).contains(&yp){continue;}
                            for dx in 0..3 {
                                let xp=col+dx as i32-1;if !(0..7).contains(&xp){continue;}
                                let w=self.weights.conv_weight[oc][ic][dy][dx];
                                let a=input[ic*42+yp as usize*7+xp as usize];
                                for b in 0..W {v[b]+=w*a[b];}
                            }
                        }
                    }
                    for b in 0..W{v[b]=v[b].max(0.0);}
                    hidden[oc*42+row as usize*7+col as usize]=v;
                }
            }
        }
        let mut output=[[0f32;W];8];
        for (i,dense) in self.weights.head_weight.iter().enumerate() {
            let mut v=[self.weights.head_bias[i];W];
            for (&w,a) in dense.iter().zip(hidden.iter()){
                for b in 0..W{v[b]+=w*a[b];}
            }
            output[i]=v;
        }
        (0..n).map(|b|{
            let mut logits=[0f32;7];for i in 0..7{logits[i]=output[i][b];}
            self.finish29(&states[b],logits,output[7][b].tanh())
        }).collect()
    }
    pub fn evaluate_batch30(&self, states: &[Connect4]) -> (Vec<EvalResult<usize>>,usize) {
        match states.len(){1=>(self.batch30::<1>(states),1),2=>(self.batch30::<2>(states),2),3|4=>(self.batch30::<4>(states),4),5..=8=>(self.batch30::<8>(states),8),_=>panic!("invalid batch30 size")}
    }
    fn finish29(&self,state:&Connect4,logits:[f32;7],value:f32)->EvalResult<usize>{
        let legal=state.legal_moves();
        let max_logit=legal.iter().map(|&c|logits[c]).fold(f32::NEG_INFINITY,f32::max);
        let weights=legal.iter().map(|&c|(logits[c]-max_logit).exp()).collect::<Vec<_>>();
        let z=weights.iter().sum::<f32>();
        if !z.is_finite()||z<=0.0{return state.uniform_eval(value);}
        EvalResult{policy:legal.iter().zip(weights.iter()).map(|(&col,&p)|(col,p/z)).collect(),value}
    }
}

impl Evaluator<Connect4> for ConvConnect4Evaluator {
    fn evaluate(&self, state: &Connect4) -> EvalResult<usize> {
        if state.is_terminal() {
            return EvalResult {
                policy: Vec::new(),
                value: state.outcome(),
            };
        }
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult {
                policy: Vec::new(),
                value: 0.0,
            };
        }
        let (logits, value) = self.predict_raw(state);
        let max_logit = legal
            .iter()
            .map(|&c| logits[c])
            .fold(f32::NEG_INFINITY, f32::max);
        let weights = legal
            .iter()
            .map(|&c| (logits[c] - max_logit).exp())
            .collect::<Vec<_>>();
        let z = weights.iter().sum::<f32>();
        if !z.is_finite() || z <= 0.0 {
            return state.uniform_eval(value);
        }
        EvalResult {
            policy: legal
                .iter()
                .zip(weights.iter())
                .map(|(&col, &p)| (col, p / z))
                .collect(),
            value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn zero_conv_weights_respect_gravity_legality() {
        let weights = serde_json::json!({
            "format":"quartz_connect4_conv3x3_v1",
            "input_planes":3,"board_height":6,"board_width":7,
            "conv_out_channels":16,"kernel_size":3,"padding":1,"head_dim":8,
            "conv_weight":vec![vec![vec![vec![0.0f32;3];3];3];16],
            "conv_bias":vec![0.0f32;16],
            "head_weight":vec![vec![0.0f32;672];8],
            "head_bias":vec![0.0f32;8]
        });
        let model = ConvConnect4Evaluator::from_json_str(&weights.to_string()).unwrap();
        let state = Connect4::from_history("111111").unwrap();
        let evaluated = model.evaluate(&state);
        assert_eq!(evaluated.policy.len(), 6);
        assert_eq!(evaluated.value, 0.0);
        assert!(evaluated
            .policy
            .iter()
            .all(|&(col, p)| col != 0 && (p - 1.0 / 6.0).abs() < 1e-6));
    }

    #[test]
    fn rejects_truncated_or_invalid_conv_models() {
        assert!(ConvConnect4Evaluator::from_json_str("{}").is_err());
        let bad = serde_json::json!({"format":"quartz_connect4_conv3x3_v1",
            "input_planes":3,"board_height":6,"board_width":7,
            "conv_out_channels":16,"kernel_size":3,"padding":1,"head_dim":8,
            "conv_weight":[],"conv_bias":[],"head_weight":[],"head_bias":[]});
        assert!(ConvConnect4Evaluator::from_json_str(&bad.to_string()).is_err());
    }

    #[test]
    fn trained_conv_matches_frozen_python_pytorch_outputs() {
        let net_path = concat!(
            "docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/",
            "evidence/mlp_conv_train_20261121/conv_model.json"
        );
        let parity_path = concat!(
            "docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/",
            "PARITY_REFERENCE.json"
        );
        let model = ConvConnect4Evaluator::from_file(net_path).unwrap();
        let test: Value =
            serde_json::from_str(&std::fs::read_to_string(parity_path).unwrap()).unwrap();
        let samples = test["samples"].as_array().unwrap();
        assert_eq!(samples.len(), 36);
        for sample in samples {
            let history = sample["history"].as_str().unwrap();
            let state = Connect4::from_history(history).unwrap();
            let guess = model.evaluate(&state);
            let mut pi = [0.0f32; 7];
            for (col, value) in guess.policy {
                pi[col] = value;
            }
            for (col, v) in sample["conv_policy"].as_array().unwrap().iter().enumerate() {
                let target = v.as_f64().unwrap() as f32;
                assert!(
                    (pi[col] - target).abs() <= 3e-5,
                    "wrong policy history={} col={} rust={} torch={}",
                    history,
                    col,
                    pi[col],
                    target
                );
            }
            let v = sample["conv_value"].as_f64().unwrap() as f32;
            assert!(
                (guess.value - v).abs() <= 3e-5,
                "wrong value history={} rust={} torch={}",
                history,
                guess.value,
                v
            );
        }
    }
}
