//! C27 joint-posterior finite two-query research executable.
//! Frozen C26 substrate copied without changing game/solver/cue/initialization semantics.
#![allow(dead_code)]
#[allow(dead_code)]
#[path = "../game.rs"]
mod game;
#[allow(dead_code)]
#[path = "../games/tictactoe.rs"]
mod tictactoe;

use game::GameState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tictactoe::TicTacToe;

const LAMBDA: f64 = 1.0 / 1024.0;
const TAU: f64 = 1.0 / 32.0;
const GAMMA: u64 = 0x9e3779b97f4a7c15;
const POLICY_SALT: u64 = 0xd1b54a32d192ed03;
const SCHEMA: &str = "c27-joint-v1";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    schema: String,
    feature_probs: BTreeMap<String, [f64; 3]>,
    remaining_probs: BTreeMap<String, [f64; 3]>,
    global_probs: [f64; 3],
    remaining_costs: BTreeMap<String, f64>,
    global_cost: f64,
    lambda_nodes: f64,
    tau: f64,
}

fn remaining_key(key: &str) -> Result<usize, String> {
    let n: usize = key.parse().map_err(|_| "invalid remaining key")?;
    if n > 8 || n.to_string() != key {
        return Err("remaining keys must be canonical integers in 0..8".into());
    }
    Ok(n)
}

fn normalize_probs(p: &mut [f64; 3]) -> Result<(), String> {
    if p.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err("model probabilities must be finite and strictly positive".into());
    }
    let sum: f64 = p.iter().sum();
    if (sum - 1.0).abs() > 1e-12 {
        return Err("model probabilities must sum to one within 1e-12".into());
    }
    for x in p {
        *x /= sum;
    }
    Ok(())
}

fn valid_cost(x: f64) -> bool {
    x.is_finite() && (1.0..=1_000_000.0).contains(&x)
}

impl Model {
    fn validate(&mut self) -> Result<(), String> {
        if self.schema != "c26-categorical-v1" {
            return Err("unknown model schema".into());
        }
        if self.lambda_nodes != LAMBDA || self.tau != TAU {
            return Err("lambda_nodes and tau must equal the frozen constants".into());
        }
        normalize_probs(&mut self.global_probs)?;
        if !valid_cost(self.global_cost) {
            return Err("global_cost must be finite in [1,1000000]".into());
        }
        for (key, p) in &mut self.feature_probs {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 4 {
                return Err("feature key must be remaining:loss:draw:win".into());
            }
            remaining_key(parts[0])?;
            let mut total = 0usize;
            for part in &parts[1..] {
                let n: usize = part.parse().map_err(|_| "invalid cue count")?;
                if n > 4 || n.to_string() != *part {
                    return Err("cue counts must be canonical integers in 0..4".into());
                }
                total += n;
            }
            if total != 4 {
                return Err("feature counts must sum to four".into());
            }
            normalize_probs(p)?;
        }
        for (key, p) in &mut self.remaining_probs {
            remaining_key(key)?;
            normalize_probs(p)?;
        }
        for (key, cost) in &self.remaining_costs {
            remaining_key(key)?;
            if !valid_cost(*cost) {
                return Err("remaining costs must be finite in [1,1000000]".into());
            }
        }
        Ok(())
    }

    fn predict(&self, remaining: usize, counts: [u8; 3]) -> ([f64; 3], f64) {
        let rem = remaining.to_string();
        let key = format!("{}:{}:{}:{}", remaining, counts[0], counts[1], counts[2]);
        let p = self
            .feature_probs
            .get(&key)
            .or_else(|| self.remaining_probs.get(&rem))
            .copied()
            .unwrap_or(self.global_probs);
        let cost = self
            .remaining_costs
            .get(&rem)
            .copied()
            .unwrap_or(self.global_cost);
        (p, cost)
    }
}

fn replay_history(history: &[usize]) -> Result<TicTacToe, String> {
    if history.len() > 8 {
        return Err("history length must not exceed eight".into());
    }
    let mut state = TicTacToe::initial();
    for &mv in history {
        if state.is_terminal() {
            return Err("history continues after a terminal state".into());
        }
        if mv >= 9 || !state.legal_moves().contains(&mv) {
            return Err("history contains an illegal, duplicate or out-of-range move".into());
        }
        state = state.apply_move(mv);
    }
    if state.is_terminal() {
        return Err("root must be nonterminal".into());
    }
    Ok(state)
}

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next();
            if r >= threshold {
                return (r % n) as usize;
            }
        }
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

// Integer root-frame utility throughout exact traversal; f64 is only an output/model scale.
fn root_value<G: GameState>(state: &G, root: i8) -> i8 {
    debug_assert!(state.is_terminal());
    root * state.current_player() * state.outcome() as i8
}
fn utility(v: i8) -> f64 {
    (v as f64 + 1.0) * 0.5
}

fn exact<G: GameState>(state: &G, root: i8) -> (i8, u64) {
    if state.is_terminal() {
        return (root_value(state, root), 1);
    }
    let maximizing = state.current_player() == root;
    let mut best = if maximizing { -1 } else { 1 };
    let mut nodes = 1;
    for mv in state.legal_moves() {
        let (value, entered) = exact(&state.apply_move(mv), root);
        nodes += entered;
        best = if maximizing {
            best.max(value)
        } else {
            best.min(value)
        };
    }
    (best, nodes)
}

fn rollout(state: &TicTacToe, root: i8, rng: &mut SplitMix64) -> (i8, u64) {
    let mut current = state.clone();
    let mut nodes = 1;
    while !current.is_terminal() {
        let moves = current.legal_moves();
        current = current.apply_move(moves[rng.index(moves.len())]);
        nodes += 1;
    }
    (root_value(&current, root), nodes)
}

struct Candidate {
    mv: usize,
    child: TicTacToe,
    remaining: usize,
    counts: [u8; 3],
    cue_nodes: u64,
    terminal_known: bool,
    known: Option<i8>,
    probabilities: Option<[f64; 3]>,
    predicted_nodes: Option<f64>,
}
impl Candidate {
    fn rollout_mean(&self) -> f64 {
        (0.5 * self.counts[1] as f64 + self.counts[2] as f64) / 4.0
    }
}

fn candidates(root: &TicTacToe, cue_seed: u64, model: Option<&Model>) -> Vec<Candidate> {
    let mut moves = root.legal_moves();
    moves.sort_unstable();
    moves
        .into_iter()
        .map(|mv| {
            let child = root.apply_move(mv);
            let remaining = child.legal_moves().len();
            let terminal_known = child.is_terminal();
            let mut counts = [0u8; 3];
            let mut cue_nodes = 0;
            let known = if terminal_known {
                let v = root_value(&child, root.current_player());
                counts[(v + 1) as usize] = 4;
                cue_nodes = 1;
                Some(v)
            } else {
                let mut rng = SplitMix64::new(cue_seed ^ (mv as u64 + 1).wrapping_mul(GAMMA));
                for _ in 0..4 {
                    let (v, n) = rollout(&child, root.current_player(), &mut rng);
                    counts[(v + 1) as usize] += 1;
                    cue_nodes += n;
                }
                None
            };
            let prediction = if terminal_known {
                None
            } else {
                model.map(|m| m.predict(remaining, counts))
            };
            Candidate {
                mv,
                child,
                remaining,
                counts,
                cue_nodes,
                terminal_known,
                known,
                probabilities: prediction.map(|x| x.0),
                predicted_nodes: prediction.map(|x| x.1),
            }
        })
        .collect()
}

struct Initial {
    incumbent: usize,
    value: i8,
    solve_move: Option<usize>,
    nodes: u64,
}
fn initialize(candidates: &mut [Candidate], root_player: i8) -> Initial {
    let terminal_win = candidates.iter().any(|c| c.known == Some(1));
    let mut solve_move = None;
    let mut nodes = 0;
    if !terminal_win {
        let mut best_cue: Option<usize> = None;
        for (i, candidate) in candidates.iter().enumerate() {
            if candidate.known.is_none()
                && best_cue
                    .map(|j| candidate.rollout_mean() > candidates[j].rollout_mean())
                    .unwrap_or(true)
            {
                best_cue = Some(i);
            }
        }
        if let Some(i) = best_cue {
            let (v, n) = exact(&candidates[i].child, root_player);
            candidates[i].known = Some(v);
            solve_move = Some(candidates[i].mv);
            nodes = n;
        }
    }
    let mut incumbent = None;
    for (i, c) in candidates.iter().enumerate() {
        if let Some(v) = c.known {
            if incumbent
                .map(|j: usize| v > candidates[j].known.unwrap())
                .unwrap_or(true)
            {
                incumbent = Some(i);
            }
        }
    }
    let incumbent = incumbent.expect("nonterminal root has a known initialized candidate");
    Initial {
        incumbent,
        value: candidates[incumbent].known.unwrap(),
        solve_move,
        nodes,
    }
}

const SUPPORT: [f64; 3] = [0.0, 0.5, 1.0];
const EPS: f64 = 1e-12;

#[derive(Clone, Debug)]
struct AtomLaw {
    p: [f64; 3],
    f: [f64; 3],
}
impl AtomLaw {
    fn new(p: [f64; 3]) -> Result<Self, String> {
        if p.iter().any(|x| !x.is_finite() || *x < 0.0)
            || (p.iter().sum::<f64>() - 1.0).abs() > EPS
        {
            return Err("invalid categorical law".into());
        }
        let mu = p[1] * 0.5 + p[2];
        let scale = SUPPORT.iter().zip(p).filter(|(_, mass)| *mass > 0.0)
            .map(|(u, _)| (u - mu).abs()).fold(0.0_f64, f64::max);
        let f = if scale > 0.0 { SUPPORT.map(|u| (u - mu) / scale) } else { [0.0; 3] };
        Ok(Self { p, f })
    }
}

#[derive(Clone, Debug)]
struct Belief {
    laws: Vec<AtomLaw>,
    rho: f64,
    log_odds: f64,
    observed: Vec<bool>,
}
impl Belief {
    fn new(probabilities: &[[f64; 3]], rho: f64) -> Result<Self, String> {
        if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
            return Err("rho must be finite in [0,1)".into());
        }
        Ok(Self { laws: probabilities.iter().copied().map(AtomLaw::new).collect::<Result<_,_>>()?,
            rho, log_odds: 0.0, observed: vec![false; probabilities.len()] })
    }
    fn predict_at(&self, i: usize, _field: f64) -> [f64; 3] {
        // RED starting behavior: frozen C26 independent categorical predictions.
        self.laws[i].p
    }
    fn predict(&self, i: usize) -> [f64; 3] { self.predict_at(i, self.log_odds) }
    fn after(&self, i: usize, u: f64, _field: f64) -> Result<f64, String> {
        let atom = SUPPORT.iter().position(|x| *x == u).ok_or("utility outside support")?;
        if self.laws[i].p[atom] <= 0.0 { return Err("unsupported observation".into()); }
        // RED starting behavior: product beliefs do not propagate observations.
        Ok(0.0)
    }
    fn observe(&mut self, i: usize, u: f64) -> Result<(), String> {
        if self.observed[i] { return Err("duplicate observation".into()); }
        self.log_odds = self.after(i, u, self.log_odds)?;
        self.observed[i] = true;
        Ok(())
    }
}

fn positive_excess(p: [f64; 3], b: f64) -> f64 {
    p.iter().zip(SUPPORT).map(|(p,u)| p*(u-b).max(0.0)).sum()
}

#[derive(Serialize, Debug)]
struct Score {
    #[serde(rename = "move")]
    mv: usize,
    probabilities: [f64; 3],
    predicted_nodes: f64,
    g: f64,
    k1: f64,
    k2: Option<f64>,
    q: f64,
}

fn score_actions(belief: &Belief, remaining: &[usize], moves: &[usize], nodes: &[f64], b: f64, horizon: usize) -> Result<Vec<Score>, String> {
    if !(1..=2).contains(&horizon) { return Err("planning horizon must be one or two".into()); }
    let mut result = Vec::with_capacity(remaining.len());
    for &a in remaining {
        let p = belief.predict(a);
        let g = positive_excess(p, b);
        let k1 = g - LAMBDA * nodes[a];
        let k2 = if horizon == 2 {
            let mut continuation = 0.0;
            for (u, mass) in SUPPORT.into_iter().zip(p) {
                if mass == 0.0 { continue; }
                let field = belief.after(a, u, belief.log_odds)?;
                let mut last = 0.0_f64;
                for &j in remaining {
                    if j != a {
                        last = last.max(positive_excess(belief.predict_at(j, field), b.max(u)) - LAMBDA * nodes[j]);
                    }
                }
                continuation += mass * last;
            }
            Some(k1 + continuation)
        } else { None };
        result.push(Score { mv: moves[a], probabilities: p, predicted_nodes: nodes[a], g, k1, k2, q: k2.unwrap_or(k1) });
    }
    Ok(result)
}

fn best_action(scores: &[Score]) -> Option<usize> {
    let mut chosen: Option<usize> = None;
    for (i, score) in scores.iter().enumerate() {
        if score.q <= EPS { continue; }
        if chosen.map(|j| score.q > scores[j].q + EPS
            || ((score.q - scores[j].q).abs() <= EPS && score.mv < scores[j].mv)).unwrap_or(true) {
            chosen = Some(i);
        }
    }
    chosen
}

#[cfg(test)]
mod c27_tests {
    use super::*;
    fn close(actual: f64, expected: f64) {
        assert!((actual-expected).abs() <= 2e-12, "actual={actual:.17}, expected={expected:.17}");
    }
    fn information_fixture() -> Belief {
        Belief::new(&[[0.5,0.5,0.0],[0.5,0.0,0.5]],0.8).unwrap()
    }
    #[test]
    fn information_observation_updates_other_candidate_to_082() {
        let mut belief = information_fixture();
        belief.observe(0, 0.5).unwrap();
        close(belief.predict(1)[2], 0.82);
    }
    #[test]
    fn information_first_query_has_h2_value_109_over_200() {
        let belief = information_fixture();
        let scores = score_actions(&belief, &[0,1], &[0,1], &[0.01/LAMBDA,0.30/LAMBDA],0.5,2).unwrap();
        let value = 0.5 + best_action(&scores).map(|i| scores[i].q).unwrap_or(0.0);
        close(value, 0.545);
        assert_eq!(best_action(&scores), Some(0));
    }
}

fn main() {}
