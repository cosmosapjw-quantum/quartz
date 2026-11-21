//! Game-independent finite Bayesian metalevel control reference.
//!
//! Grounded semantics only. This does not implement SearchPolicy or assume
//! that correlated neural-MCTS backups follow calibrated IID likelihoods.

const NORMALIZATION_TOLERANCE: f64 = 1e-10;
const TIE_TOLERANCE: f64 = 1e-12;
const MAX_REFERENCE_HORIZON: usize = 4;

#[derive(Debug, Clone)]
pub struct Computation {
    pub name: String,
    pub cost: f64,
    /// For each hidden world: a probability row over finite observations.
    pub likelihood: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct FiniteMetalevel {
    pub prior: Vec<f64>,
    /// For each hidden world: one utility for every final physical action.
    pub utilities: Vec<Vec<f64>>,
    pub computations: Vec<Computation>,
    /// Units: physical terminal utility per one computation-resource unit.
    pub resource_price: f64,
}

#[derive(Debug, Clone)]
pub struct OneStepDecision {
    pub gross_gain: Vec<f64>,
    pub gain_per_cost: Vec<f64>,
    pub net_gain: Vec<f64>,
    pub ratio_choice: Option<usize>,
    pub net_choice: Option<usize>,
}

fn validate_probability(row: &[f64]) -> Result<(), String> {
    if row.is_empty() || row.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return Err("nonfinite, out-of-bounds or empty probability row".into());
    }
    if (row.iter().sum::<f64>() - 1.0).abs() > NORMALIZATION_TOLERANCE {
        return Err("probability row must sum to one; no silent re-normalization".into());
    }
    Ok(())
}

fn validate_payoffs(belief: &[f64], utilities: &[Vec<f64>]) -> Result<usize, String> {
    validate_probability(belief)?;
    if utilities.len() != belief.len() || utilities.is_empty() || utilities[0].len() < 2 {
        return Err("utility state count mismatch or fewer than two physical actions".into());
    }
    let actions = utilities[0].len();
    if utilities.iter().any(|row| row.len() != actions || row.iter().any(|v| !v.is_finite())) {
        return Err("invalid finite rectangular state/action utility matrix".into());
    }
    Ok(actions)
}

pub fn terminal_value(belief: &[f64], utilities: &[Vec<f64>]) -> Result<f64, String> {
    let actions = validate_payoffs(belief, utilities)?;
    Ok((0..actions)
        .map(|a| belief.iter().zip(utilities).map(|(p, row)| p * row[a]).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max))
}

pub fn bayes_risk(belief: &[f64], utilities: &[Vec<f64>]) -> Result<f64, String> {
    validate_payoffs(belief, utilities)?;
    let oracle: f64 = belief
        .iter()
        .zip(utilities)
        .map(|(p, row)| p * row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .sum();
    let diff = oracle - terminal_value(belief, utilities)?;
    if diff < -NORMALIZATION_TOLERANCE {
        return Err("posterior Bayes simple regret below numerical tolerance".into());
    }
    Ok(diff.max(0.0))
}

impl FiniteMetalevel {
    pub fn new(
        prior: Vec<f64>,
        utilities: Vec<Vec<f64>>,
        computations: Vec<Computation>,
        resource_price: f64,
    ) -> Result<Self, String> {
        validate_payoffs(&prior, &utilities)?;
        if !resource_price.is_finite() || resource_price <= 0.0 {
            return Err("resource price must be strictly positive and finite".into());
        }
        if computations.is_empty() {
            return Err("at least one computational action required".into());
        }
        for (i, c) in computations.iter().enumerate() {
            if c.name.is_empty()
                || c.cost <= 0.0
                || !c.cost.is_finite()
                || computations[..i].iter().any(|prev| prev.name == c.name)
            {
                return Err("invalid or duplicate computational action".into());
            }
            if c.likelihood.len() != prior.len() || c.likelihood[0].is_empty() {
                return Err("observation channel wrong state count or empty observation".into());
            }
            let observations = c.likelihood[0].len();
            for row in &c.likelihood {
                if row.len() != observations {
                    return Err("nonrectangular observation channel".into());
                }
                validate_probability(row)?;
            }
        }
        Ok(Self {
            prior,
            utilities,
            computations,
            resource_price,
        })
    }

    fn outcomes(
        &self,
        belief: &[f64],
        computation: usize,
    ) -> Result<Vec<(f64, Vec<f64>)>, String> {
        validate_payoffs(belief, &self.utilities)?;
        let c = self
            .computations
            .get(computation)
            .ok_or_else(|| "invalid computation index".to_string())?;
        let mut result = Vec::new();
        for obs in 0..c.likelihood[0].len() {
            let mass: f64 = belief
                .iter()
                .zip(&c.likelihood)
                .map(|(p, row)| p * row[obs])
                .sum();
            if mass <= 0.0 {
                continue;
            }
            let posterior = belief
                .iter()
                .zip(&c.likelihood)
                .map(|(p, row)| p * row[obs] / mass)
                .collect();
            result.push((mass, posterior));
        }
        Ok(result)
    }

    pub fn gross_evsi(&self, belief: &[f64], c: usize) -> Result<f64, String> {
        let now = terminal_value(belief, &self.utilities)?;
        let mut future = 0.0;
        for (mass, posterior) in self.outcomes(belief, c)? {
            future += mass * terminal_value(&posterior, &self.utilities)?;
        }
        let gain = future - now;
        if gain < -NORMALIZATION_TOLERANCE {
            return Err("calibrated expected gain was negative beyond tolerance".into());
        }
        Ok(gain.max(0.0))
    }

    pub fn one_step(&self, belief: &[f64]) -> Result<OneStepDecision, String> {
        validate_payoffs(belief, &self.utilities)?;
        let mut gross_gain = Vec::new();
        let mut gain_per_cost = Vec::new();
        let mut net_gain = Vec::new();
        for (idx, c) in self.computations.iter().enumerate() {
            let gross = self.gross_evsi(belief, idx)?;
            gross_gain.push(gross);
            gain_per_cost.push(gross / c.cost);
            net_gain.push(gross - self.resource_price * c.cost);
        }
        let argmax = |values: &[f64]| -> usize {
            let mut best = 0;
            for i in 1..values.len() {
                if values[i] > values[best] {
                    best = i;
                }
            }
            best
        };
        let i_rate = argmax(&gain_per_cost);
        let i_net = argmax(&net_gain);
        Ok(OneStepDecision {
            ratio_choice: (gain_per_cost[i_rate] > self.resource_price + TIE_TOLERANCE)
                .then_some(i_rate),
            net_choice: (net_gain[i_net] > TIE_TOLERANCE).then_some(i_net),
            gross_gain,
            gain_per_cost,
            net_gain,
        })
    }

    pub fn finite_horizon(
        &self,
        belief: &[f64],
        horizon: usize,
    ) -> Result<(f64, Option<usize>), String> {
        if horizon > MAX_REFERENCE_HORIZON {
            return Err("reference horizon exceeds bounded cap".into());
        }
        validate_payoffs(belief, &self.utilities)?;
        self.value_recursive(belief, horizon)
    }

    fn value_recursive(
        &self,
        belief: &[f64],
        remaining: usize,
    ) -> Result<(f64, Option<usize>), String> {
        let mut best_value = bayes_risk(belief, &self.utilities)?;
        let mut best_action = None;
        if remaining == 0 {
            return Ok((best_value, best_action));
        }
        for (idx, exp) in self.computations.iter().enumerate() {
            let mut after = 0.0;
            for (mass, updated_belief) in self.outcomes(belief, idx)? {
                after += mass * self.value_recursive(&updated_belief, remaining - 1)?.0;
            }
            let estimate = self.resource_price * exp.cost + after;
            if estimate < best_value - TIE_TOLERANCE {
                best_value = estimate;
                best_action = Some(idx);
            }
        }
        Ok((best_value, best_action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> FiniteMetalevel {
        FiniteMetalevel::new(
            vec![0.5, 0.3, 0.2],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![
                Computation {
                    name: "partial".into(),
                    cost: 0.05,
                    likelihood: vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                },
                Computation {
                    name: "full".into(),
                    cost: 0.20,
                    likelihood: vec![
                        vec![1.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 1.0],
                    ],
                },
            ],
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn nonzero_bayes_regret_three_state() {
        let m = example();
        assert!((bayes_risk(&m.prior, &m.utilities).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn heterogeneous_cost_rate_and_net_rank_differ() {
        let m = example();
        let d = m.one_step(&m.prior).unwrap();
        assert_eq!(d.ratio_choice, Some(0));
        assert_eq!(d.net_choice, Some(1));
        assert!((d.gross_gain[0] - 0.2).abs() < 1e-12);
        assert!((d.gross_gain[1] - 0.5).abs() < 1e-12);
        assert!((d.gain_per_cost[0] - 4.0).abs() < 1e-12);
        assert!((d.gain_per_cost[1] - 2.5).abs() < 1e-12);
        assert!((d.net_gain[0] - 0.15).abs() < 1e-12);
        assert!((d.net_gain[1] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn finite_horizon_h0_through_h3_matches_exact_theory() {
        let m = example();
        assert!((m.finite_horizon(&m.prior, 0).unwrap().0 - 0.5).abs() < 1e-12);
        for horizon in 1..=3 {
            let r = m.finite_horizon(&m.prior, horizon).unwrap();
            assert!((r.0 - 0.2).abs() < 1e-12);
            assert_eq!(r.1, Some(1));
        }
        assert!(m.finite_horizon(&m.prior, 5).is_err());
    }

    #[test]
    fn information_without_action_impact_is_worth_zero() {
        let m = FiniteMetalevel::new(
            vec![0.5, 0.5],
            vec![vec![2.0, 1.0], vec![2.0, 1.0]],
            vec![Computation {
                name: "reveal_state".into(),
                cost: 0.1,
                likelihood: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }],
            1.0,
        ).unwrap();
        assert_eq!(m.gross_evsi(&m.prior, 0).unwrap(), 0.0);
        assert_eq!(m.one_step(&m.prior).unwrap().net_choice, None);
        assert_eq!(m.finite_horizon(&m.prior, 2).unwrap().1, None);
    }

    #[test]
    fn input_validation_and_cost_unit_equivalence() {
        let mut m = example();
        m.computations[0].likelihood[0] = vec![0.5, 0.25];
        assert!(FiniteMetalevel::new(m.prior.clone(),m.utilities.clone(),m.computations.clone(),1.0).is_err());
        let m = example();
        let mut scaled = m.clone();
        for exp in &mut scaled.computations {
            exp.cost *= 1000.0;
        }
        scaled.resource_price /= 1000.0;
        let x = m.finite_horizon(&m.prior, 2).unwrap();
        let y = scaled.finite_horizon(&scaled.prior, 2).unwrap();
        assert!((x.0-y.0).abs()<1e-12);
        assert_eq!(x.1,y.1);
    }

    #[test]
    fn action_permutations_preserve_terminal_risk_and_evsi() {
        let original = example();
        let r = bayes_risk(&original.prior,&original.utilities).unwrap();
        let mut permuted = original.clone();
        for utilities in &mut permuted.utilities {
            *utilities = vec![utilities[2], utilities[0], utilities[1]];
        }
        assert!((r-bayes_risk(&permuted.prior,&permuted.utilities).unwrap()).abs()<1e-12);
        assert!((original.gross_evsi(&original.prior,0).unwrap()
              -permuted.gross_evsi(&permuted.prior,0).unwrap()).abs()<1e-12);
    }
}
