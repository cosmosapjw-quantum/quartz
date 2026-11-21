//! Bounded empirical 2-step signed-gain continuation option for Connect4.
//!
//! The candidate computations are ordinary MCTS 8->16->32->64 steps.
//! A chosen two-step option commits across an intermediate checkpoint, without
//! reading its next observed features or external exact solver scores.
//! The predictor is *not* a Bayesian posterior or knowledge gradient.
use super::cog02::FEATURE_COUNT;
use serde_json::Value;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Stop,
    Single,
    CommitTwo,
}
impl Action {
    pub fn label(&self) -> &'static str {
        match self {
            Action::Stop => "stop",
            Action::Single => "single",
            Action::CommitTwo => "commit_two",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Decision {
    pub action: Action,
    pub predicted_single_gain: f64,
    pub predicted_joint_gain: Option<f64>,
    pub predicted_single_cost_ms: f64,
    pub predicted_joint_cost_ms: Option<f64>,
    pub net_single: f64,
    pub net_joint: Option<f64>,
}

pub struct DynamicTwoStepPolicy {
    one: [[f64; FEATURE_COUNT]; 3],
    two: [[f64; FEATURE_COUNT]; 2],
    one_cost_log: [[f64; FEATURE_COUNT]; 3],
    two_cost_log: [[f64; FEATURE_COUNT]; 2],
    one_scale: [f64; 3],
    two_scale: [f64; 2],
    lambda: f64,
}
impl DynamicTwoStepPolicy {
    fn matrix<const R: usize>(v: &Value, field: &str) -> Result<[[f64; FEATURE_COUNT]; R], String> {
        let a = v
            .get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{field}: not a matrix"))?;
        if a.len() != R {
            return Err(format!("{field}: expected {R} rows"));
        }
        let mut dst = [[0f64; FEATURE_COUNT]; R];
        for (i, row) in a.iter().enumerate() {
            let items = row.as_array().ok_or(format!("{field}: row not an array"))?;
            if items.len() != FEATURE_COUNT {
                return Err(format!("{field}: incorrect number of current features"));
            }
            for (j, x) in items.iter().enumerate() {
                let z = x.as_f64().ok_or(format!("{field}: nonnumeric weight"))?;
                if !z.is_finite() || z.abs() > 1e5 {
                    return Err(format!("{field}: nonfinite or huge weight"));
                }
                dst[i][j] = z;
            }
        }
        Ok(dst)
    }
    fn scales<const R: usize>(v: &Value, field: &str) -> Result<[f64; R], String> {
        let rows = v
            .get(field)
            .and_then(Value::as_array)
            .ok_or(format!("{field}: missing scales"))?;
        if rows.len() != R {
            return Err(format!("{field}: wrong count"));
        }
        let mut dst = [0f64; R];
        for (i, x) in rows.iter().enumerate() {
            let z = x.as_f64().ok_or(format!("{field}: nonnumeric scale"))?;
            if !z.is_finite() || z <= 0.0 || z > 1e5 {
                return Err(format!("{field}: nonpositive/nonfinite or huge scale"));
            }
            dst[i] = z;
        }
        Ok(dst)
    }
    pub fn parse(v: &Value) -> Result<Self, String> {
        if v.get("schema").and_then(Value::as_str) != Some("quartz_cog02_dynamic_two_step_v1")
            || v.get("features").and_then(Value::as_str) != Some("root_visits_v1")
        {
            return Err("bad dynamic10 policy identity".into());
        }
        let lambda = v
            .get("lambda_wdl_per_ms")
            .and_then(Value::as_f64)
            .ok_or("missing dynamic10 lambda")?;
        if !lambda.is_finite() || !(0.0..=10.0).contains(&lambda) {
            return Err("dynamic lambda must be finite in [0,10]".into());
        }
        Ok(Self {
            one: Self::matrix::<3>(v, "one_step_gain_coefficients")?,
            two: Self::matrix::<2>(v, "two_step_gain_coefficients")?,
            one_cost_log: Self::matrix::<3>(v, "one_step_cost_log_coefficients")?,
            two_cost_log: Self::matrix::<2>(v, "two_step_cost_log_coefficients")?,
            one_scale: Self::scales::<3>(v, "one_step_cost_scale")?,
            two_scale: Self::scales::<2>(v, "two_step_cost_scale")?,
            lambda,
        })
    }
    fn dot(a: &[f64; FEATURE_COUNT], b: &[f64; FEATURE_COUNT]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    pub fn predict(
        &self,
        stage: usize,
        feature: &[f64; FEATURE_COUNT],
    ) -> Result<Decision, String> {
        if stage >= 3 || feature.iter().any(|x| !x.is_finite()) {
            return Err("bad dynamic10 stage/current root feature".into());
        }
        let gs = Self::dot(&self.one[stage], feature);
        let ms = (Self::dot(&self.one_cost_log[stage], feature)
            .clamp(-20.0, 10.0)
            .exp()
            * self.one_scale[stage])
            .max(1e-6);
        if !gs.is_finite() || !ms.is_finite() {
            return Err("dynamic10 nonfinite gain/cost".into());
        }
        let ns = gs - self.lambda * ms;
        if stage == 2 {
            return Ok(Decision {
                action: if ns > 0.0 {
                    Action::Single
                } else {
                    Action::Stop
                },
                predicted_single_gain: gs,
                predicted_joint_gain: None,
                predicted_single_cost_ms: ms,
                predicted_joint_cost_ms: None,
                net_single: ns,
                net_joint: None,
            });
        }
        let gj = Self::dot(&self.two[stage], feature);
        let mc = (Self::dot(&self.two_cost_log[stage], feature)
            .clamp(-20.0, 10.0)
            .exp()
            * self.two_scale[stage])
            .max(1e-6);
        if !gj.is_finite() || !mc.is_finite() {
            return Err("dynamic10 nonfinite joint gain/cost".into());
        }
        let nj = gj - self.lambda * mc;
        let action = if nj > 0.0 && nj > ns.max(0.0) {
            Action::CommitTwo
        } else if ns > 0.0 {
            Action::Single
        } else {
            Action::Stop
        };
        Ok(Decision {
            action,
            predicted_single_gain: gs,
            predicted_joint_gain: Some(gj),
            predicted_single_cost_ms: ms,
            predicted_joint_cost_ms: Some(mc),
            net_single: ns,
            net_joint: Some(nj),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_value_dominates_even_if_single_computation_negative() {
        let mut one = vec![vec![0f64; 9]; 3];
        one[2][0] = -1.0;
        let mut two = vec![vec![0f64; 9]; 2];
        two[0][0] = 1.0;
        let j = serde_json::json!({
            "schema":"quartz_cog02_dynamic_two_step_v1",
            "features":"root_visits_v1",
            "one_step_gain_coefficients":one,
            "two_step_gain_coefficients":two,
            "one_step_cost_log_coefficients":vec![vec![0f64;9];3],
            "two_step_cost_log_coefficients":vec![vec![0f64;9];2],
            "one_step_cost_scale":[0.1,0.1,0.1],
            "two_step_cost_scale":[0.2,0.2],
            "lambda_wdl_per_ms":0.5
        });
        let policy = DynamicTwoStepPolicy::parse(&j).unwrap();
        let features = [1.0, 0.125, 0.25, 0.5, 0.5, 1.0, 0.5, 0.2, 1.0];
        let at8 = policy.predict(0, &features).unwrap();
        assert_eq!(at8.action, Action::CommitTwo);
        assert!(at8.net_single < 0.0 && at8.net_joint.unwrap() > 0.0);
        assert_eq!(policy.predict(2, &features).unwrap().action, Action::Stop);
    }
    #[test]
    fn invalid_shape_and_nonfinite_or_negative_cost_rejected() {
        let bad = serde_json::json!({
            "schema":"quartz_cog02_dynamic_two_step_v1",
            "features":"root_visits_v1",
            "one_step_gain_coefficients":[[0]],
            "two_step_gain_coefficients":[],
            "one_step_cost_log_coefficients":[],
            "two_step_cost_log_coefficients":[],
            "one_step_cost_scale":[0,0,0],
            "two_step_cost_scale":[1,1],
            "lambda_wdl_per_ms":0
        });
        assert!(DynamicTwoStepPolicy::parse(&bad).is_err());
    }
}
