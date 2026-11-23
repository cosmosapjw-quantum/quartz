//! C26 exact-query game metareasoning. Contract: C26 coding/CODE_CONTRACT.md.
//! Utility is dimensionless; lambda is utility/node and tau is utility.
//! Runtime observes only paid cues, terminal children and explicitly queried values.
//! The independent-law model is a frozen TRAIN closure, not actual game independence.
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
const SCHEMA: &str = "c26-game-v1";

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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    history: Vec<usize>,
    cue_seed: u64,
    policy_seed: u64,
    mode: String,
    additional: usize,
    repeats: usize,
    warmup: usize,
    trace: bool,
    model: Option<Model>,
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

impl Request {
    fn validate(&mut self) -> Result<TicTacToe, String> {
        if self.id.is_empty() || self.id.len() > 128 {
            return Err("id must contain 1..128 UTF-8 bytes".into());
        }
        if !(1..=9).contains(&self.additional)
            || !(1..=15).contains(&self.repeats)
            || self.warmup > 2
        {
            return Err("additional/repeats/warmup outside allowed bounds".into());
        }
        match self.mode.as_str() {
            "run" => self
                .model
                .as_mut()
                .ok_or("run requires model")?
                .validate()?,
            "survey" => {
                if self.model.is_some()
                    || self.additional != 9
                    || self.repeats != 1
                    || self.warmup != 0
                    || self.trace
                    || self.policy_seed != 0
                {
                    return Err("survey requires model=null, additional=9, repeats=1, warmup=0, trace=false, policy_seed=0".into());
                }
            }
            _ => return Err("mode must be run or survey".into()),
        }
        replay_history(&self.history)
    }
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

// Literal three-atom expectation. Includes the atom at zero for mathematical z<0 checks.
fn positive_excess(p: [f64; 3], b: f64) -> f64 {
    p[0] * (0.0 - b).max(0.0) + p[1] * (0.5 - b).max(0.0) + p[2] * (1.0 - b).max(0.0)
}
fn reservation_index(p: [f64; 3], d: f64) -> f64 {
    let mu = p[1] * 0.5 + p[2];
    if p[2] > 0.0 && d <= p[2] * 0.5 {
        1.0 - d / p[2]
    } else if p[1] + p[2] > 0.0 && d <= mu {
        (mu - d) / (p[1] + p[2])
    } else {
        mu - d
    }
}

// Uniform reference cancels in normalization; STOP is the first Q.
fn softmax(q: &[f64], tau: f64) -> Vec<f64> {
    let maximum = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = q.iter().map(|x| ((x - maximum) / tau).exp()).collect();
    let total: f64 = weights.iter().sum();
    for p in &mut weights {
        *p /= total;
    }
    weights
}
fn sample_probabilities(probabilities: &[f64], draw: f64) -> usize {
    let mut cumulative = 0.0;
    for (i, p) in probabilities.iter().enumerate() {
        cumulative += p;
        if draw < cumulative {
            return i;
        }
    }
    // Account for a final floating-point summation gap without selecting a zero-mass tail.
    probabilities
        .iter()
        .rposition(|p| *p > 0.0)
        .expect("positive total probability")
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Policy {
    InitialStop,
    AllActions,
    RandomOrder,
    GreedyVoc,
    Pandora,
    GibbsVoc,
}
impl Policy {
    const ALL: [Self; 6] = [
        Self::InitialStop,
        Self::AllActions,
        Self::RandomOrder,
        Self::GreedyVoc,
        Self::Pandora,
        Self::GibbsVoc,
    ];
    fn rational(self) -> bool {
        matches!(self, Self::GreedyVoc | Self::Pandora | Self::GibbsVoc)
    }
}

#[derive(Serialize)]
struct Prediction {
    #[serde(rename = "move")]
    mv: usize,
    remaining: usize,
    counts: [u8; 3],
    terminal_known: bool,
    probabilities: Option<[f64; 3]>,
    predicted_nodes: Option<f64>,
    observed_u: Option<f64>,
}

#[derive(Serialize)]
struct Score {
    #[serde(rename = "move")]
    mv: usize,
    g: f64,
    q: f64,
    z: f64,
    probabilities: [f64; 3],
    predicted_nodes: f64,
    probability: f64,
}
fn scores(candidates: &[Candidate], remaining: &[usize], b: f64, model: &Model) -> Vec<Score> {
    remaining
        .iter()
        .map(|&i| {
            let c = &candidates[i];
            let p = c.probabilities.unwrap();
            let predicted_nodes = c.predicted_nodes.unwrap();
            let d = model.lambda_nodes * predicted_nodes;
            let g = positive_excess(p, b);
            Score {
                mv: c.mv,
                g,
                q: g - d,
                z: reservation_index(p, d),
                probabilities: p,
                predicted_nodes,
                probability: 0.0,
            }
        })
        .collect()
}

struct Choice {
    position: Option<usize>,
    masses: Vec<f64>, // STOP first, then remaining in ascending move order
    draw: Option<f64>,
    reason: &'static str,
}
fn deterministic_choice(
    position: Option<usize>,
    n: usize,
    trace: bool,
    reason: &'static str,
) -> Choice {
    let masses = if trace {
        let mut p = vec![0.0; n + 1];
        p[position.map(|i| i + 1).unwrap_or(0)] = 1.0;
        p
    } else {
        Vec::new()
    };
    Choice {
        position,
        masses,
        draw: None,
        reason,
    }
}

fn choose(
    policy: Policy,
    n: usize,
    scores: &[Score],
    b: f64,
    rng: &mut SplitMix64,
    trace: bool,
    tau: f64,
) -> Choice {
    match policy {
        Policy::AllActions => deterministic_choice(Some(0), n, trace, "all_actions"),
        Policy::RandomOrder => {
            let position = rng.index(n);
            let masses = if trace {
                let mut p = vec![1.0 / n as f64; n + 1];
                p[0] = 0.0;
                p
            } else {
                Vec::new()
            };
            Choice {
                position: Some(position),
                masses,
                draw: None,
                reason: "random_order",
            }
        }
        Policy::GreedyVoc | Policy::Pandora => {
            let mut best = 0;
            let value = |s: &Score| {
                if policy == Policy::GreedyVoc {
                    s.q
                } else {
                    s.z
                }
            };
            for i in 1..n {
                if value(&scores[i]) > value(&scores[best]) {
                    best = i;
                }
            }
            let positive = if policy == Policy::GreedyVoc {
                scores[best].q > 0.0
            } else {
                scores[best].z > b
            };
            let reason = match (policy, positive) {
                (Policy::GreedyVoc, true) => "positive_voc",
                (Policy::GreedyVoc, false) => "nonpositive_voc",
                (_, true) => "reservation_above_incumbent",
                (_, false) => "reservation_below_incumbent",
            };
            deterministic_choice(if positive { Some(best) } else { None }, n, trace, reason)
        }
        Policy::GibbsVoc => {
            let mut q = Vec::with_capacity(n + 1);
            q.push(0.0);
            q.extend(scores.iter().map(|s| s.q));
            let masses = softmax(&q, tau);
            let draw = rng.uniform();
            let sampled = sample_probabilities(&masses, draw);
            Choice {
                position: if sampled == 0 {
                    None
                } else {
                    Some(sampled - 1)
                },
                masses,
                draw: Some(draw),
                reason: if sampled == 0 {
                    "sampled_stop"
                } else {
                    "gibbs_sample"
                },
            }
        }
        Policy::InitialStop => unreachable!("initial stop is handled before choosing"),
    }
}

#[derive(Serialize)]
struct Decision {
    b_before: f64,
    remaining: Vec<usize>,
    candidates: Vec<Score>,
    stop_probability: f64,
    chosen_move: Option<usize>,
    choice: &'static str,
    uniform_draw: Option<f64>,
    queried_u: Option<f64>,
    queried_nodes: u64,
    b_after: f64,
    reason: &'static str,
}

#[derive(Serialize)]
struct RunRecord {
    policy: Policy,
    repeat: usize,
    initial_move: usize,
    initial_solve_move: Option<usize>,
    initial_u: f64,
    selected_move: usize,
    final_u: f64,
    cue_nodes: u64,
    initial_nodes: u64,
    additional_nodes: u64,
    charged_nodes: u64,
    additional_queries: usize,
    query_moves: Vec<usize>,
    predicted_additional_nodes: f64,
    elapsed_us: f64,
    cue_us: f64,
    initial_us: f64,
    query_us: f64,
    controller_us: f64,
    stop_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    predictions: Option<Vec<Prediction>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decisions: Option<Vec<Decision>>,
}
fn micros(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e6
}

fn run_one(root: &TicTacToe, request: &Request, policy: Policy, repeat: usize) -> RunRecord {
    let model = request.model.as_ref().unwrap();
    let start = Instant::now();
    // All temporary candidates, remaining indices and control vectors drop before elapsed is read.
    // Only serializable output metadata escapes this scope.
    let mut output = {
        let cue_start = Instant::now();
        let mut candidates = candidates(root, request.cue_seed, Some(model));
        let cue_nodes = candidates.iter().map(|c| c.cue_nodes).sum::<u64>();
        let cue_us = micros(cue_start);
        let predictions = request.trace.then(|| {
            candidates
                .iter()
                .map(|c| Prediction {
                    mv: c.mv,
                    remaining: c.remaining,
                    counts: c.counts,
                    terminal_known: c.terminal_known,
                    probabilities: c.probabilities,
                    predicted_nodes: c.predicted_nodes,
                    observed_u: if c.terminal_known {
                        c.known.map(utility)
                    } else {
                        None
                    },
                })
                .collect()
        });
        let initial_start = Instant::now();
        let initial = initialize(&mut candidates, root.current_player());
        let initial_us = micros(initial_start);
        let initial_move = candidates[initial.incumbent].mv;
        let mut incumbent = initial.incumbent;
        let mut value = initial.value;
        let mut remaining: Vec<usize> = (0..candidates.len())
            .filter(|&i| candidates[i].known.is_none())
            .collect();
        let mut rng = SplitMix64::new(request.cue_seed ^ POLICY_SALT ^ request.policy_seed);
        let mut query_moves = Vec::new();
        let mut additional_nodes = 0u64;
        let mut predicted_additional_nodes = 0.0;
        let mut query_us = 0.0;
        let mut decisions = request.trace.then(Vec::new);
        let stop_reason;
        loop {
            let b = utility(value);
            let forced = if policy == Policy::InitialStop {
                Some("initial_stop")
            } else if query_moves.len() >= request.additional {
                Some("cap_reached")
            } else if remaining.is_empty() {
                Some("exhausted")
            } else if value == 1 && policy != Policy::AllActions {
                Some("known_win")
            } else {
                None
            };
            let mut scored = if request.trace || (forced.is_none() && policy.rational()) {
                scores(&candidates, &remaining, b, model)
            } else {
                Vec::new()
            };
            let choice = if let Some(reason) = forced {
                deterministic_choice(None, remaining.len(), request.trace, reason)
            } else {
                choose(
                    policy,
                    remaining.len(),
                    &scored,
                    b,
                    &mut rng,
                    request.trace,
                    model.tau,
                )
            };
            let before_moves = if request.trace {
                remaining.iter().map(|&i| candidates[i].mv).collect()
            } else {
                Vec::new()
            };
            if request.trace {
                for (j, score) in scored.iter_mut().enumerate() {
                    score.probability = choice.masses[j + 1];
                }
            }
            let mut queried_u = None;
            let mut queried_nodes = 0;
            let chosen_move = choice
                .position
                .map(|position| candidates[remaining[position]].mv);
            if let Some(position) = choice.position {
                let i = remaining.remove(position);
                let query_start = Instant::now();
                let (v, n) = exact(&candidates[i].child, root.current_player());
                query_us += micros(query_start);
                queried_u = Some(utility(v));
                queried_nodes = n;
                candidates[i].known = Some(v);
                additional_nodes += n;
                predicted_additional_nodes += candidates[i].predicted_nodes.unwrap();
                query_moves.push(candidates[i].mv);
                if v > value {
                    value = v;
                    incumbent = i;
                }
            }
            if let Some(trace) = &mut decisions {
                trace.push(Decision {
                    b_before: b,
                    remaining: before_moves,
                    candidates: scored,
                    stop_probability: choice.masses[0],
                    chosen_move,
                    choice: if chosen_move.is_some() {
                        "FOCUS"
                    } else {
                        "STOP"
                    },
                    uniform_draw: choice.draw,
                    queried_u,
                    queried_nodes,
                    b_after: utility(value),
                    reason: choice.reason,
                });
            }
            if chosen_move.is_none() {
                stop_reason = choice.reason;
                break;
            }
        }
        RunRecord {
            policy,
            repeat,
            initial_move,
            initial_solve_move: initial.solve_move,
            initial_u: utility(initial.value),
            selected_move: candidates[incumbent].mv,
            final_u: utility(value),
            cue_nodes,
            initial_nodes: initial.nodes,
            additional_nodes,
            charged_nodes: cue_nodes + initial.nodes + additional_nodes,
            additional_queries: query_moves.len(),
            query_moves,
            predicted_additional_nodes,
            elapsed_us: 0.0,
            cue_us,
            initial_us,
            query_us,
            controller_us: 0.0,
            stop_reason,
            predictions,
            decisions,
        }
    };
    output.elapsed_us = micros(start);
    output.controller_us =
        (output.elapsed_us - output.cue_us - output.initial_us - output.query_us).max(0.0);
    output
}

fn survey(root: &TicTacToe, request: &Request) -> Value {
    let start = Instant::now();
    let mut total_cue_nodes = 0u64;
    let measured = {
        let candidates = candidates(root, request.cue_seed, None);
        candidates
            .iter()
            .map(|c| {
                let (v, solve_nodes) = if let Some(v) = c.known {
                    (v, 1)
                } else {
                    exact(&c.child, root.current_player())
                };
                total_cue_nodes += c.cue_nodes;
                json!({
                    "move": c.mv, "terminal_known": c.terminal_known, "remaining": c.remaining,
                    "counts": c.counts, "rollout_mean": c.rollout_mean(), "u": utility(v),
                    "solve_nodes": solve_nodes, "cue_nodes": c.cue_nodes
                })
            })
            .collect::<Vec<_>>()
    };
    let measurement_elapsed_us = micros(start);
    json!({
        "schema": SCHEMA, "mode": "survey", "id": request.id, "candidates": measured,
        "cue_nodes": total_cue_nodes, "measurement_elapsed_us": measurement_elapsed_us
    })
}

fn run(root: &TicTacToe, request: &Request) -> Value {
    let dispatch = Instant::now();
    let offset = (request.policy_seed % 6) as usize;
    let mut warmup_elapsed_us = 0.0;
    for repeat in 0..request.warmup {
        for slot in 0..6 {
            let policy = Policy::ALL[(slot + repeat + offset) % 6];
            warmup_elapsed_us += run_one(root, request, policy, repeat).elapsed_us;
        }
    }
    let mut runs = Vec::with_capacity(request.repeats * 6);
    for repeat in 0..request.repeats {
        for slot in 0..6 {
            let policy = Policy::ALL[(slot + repeat + offset) % 6];
            runs.push(run_one(root, request, policy, repeat));
        }
    }
    // Execution is rotated; serialized records use canonical policy order within each repeat.
    runs.sort_by_key(|r| (r.repeat, r.policy as usize));
    let execution_elapsed_us = micros(dispatch);
    json!({
        "schema": SCHEMA, "mode": "run", "id": request.id, "runs": runs,
        "warmup_elapsed_us": warmup_elapsed_us, "execution_elapsed_us": execution_elapsed_us
    })
}

fn handle_line(line: &str) -> Value {
    let parsed = serde_json::from_str::<Request>(line)
        .map_err(|e| format!("invalid request: {e}"))
        .and_then(|mut request| {
            let root = request.validate()?;
            Ok(if request.mode == "survey" {
                survey(&root, &request)
            } else {
                run(&root, &request)
            })
        });
    match parsed {
        Ok(response) => response,
        Err(error) => json!({ "error": error }),
    }
}

fn serve(input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    for line in input.lines() {
        let response = match line {
            Ok(line) => handle_line(&line),
            Err(error) => json!({ "error": format!("stdin read error: {error}") }),
        };
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(error) = serve(io::stdin().lock(), io::stdout().lock()) {
        eprintln!("C26 I/O error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod c26_tests {
    use super::*;
    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-13 * (1.0 + expected.abs()),
            "actual={actual:.17}, expected={expected:.17}"
        );
    }
    fn model_json() -> Value {
        json!({"schema":"c26-categorical-v1","feature_probs":{},"remaining_probs":{},
            "global_probs":[0.2,0.4,0.4],"remaining_costs":{},"global_cost":17.0,
            "lambda_nodes":LAMBDA,"tau":TAU})
    }
    fn request_json() -> Value {
        json!({"id":"unit-fixture","history":[0,4,1,8],"cue_seed":123,"policy_seed":0,
            "mode":"run","additional":9,"repeats":1,"warmup":0,"trace":true,"model":model_json()})
    }
    fn read_request(value: Value) -> (Request, TicTacToe) {
        let mut request: Request = serde_json::from_value(value).unwrap();
        let root = request.validate().unwrap();
        (request, root)
    }
    #[test]
    fn red_root_frame_retains_actual_x_win() {
        let mut s = TicTacToe::initial();
        for m in [0, 3, 1, 4, 2] {
            s = s.apply_move(m);
        }
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), -1.0);
        assert_eq!(root_value(&s, 1), 1, "missing root-player sign conversion");
        assert_eq!(root_value(&s, -1), -1);
        let mut o = TicTacToe::initial();
        for m in [0, 3, 1, 4, 8, 5] {
            o = o.apply_move(m);
        }
        assert_eq!(root_value(&o, -1), 1);
    }
    #[test]
    fn red_positive_excess_three_point_literal() {
        close(positive_excess([0.2, 0.4, 0.4], 0.5), 0.2);
        close(positive_excess([0.2, 0.4, 0.4], 0.0), 0.6);
        close(positive_excess([0.2, 0.4, 0.4], 1.0), 0.0);
        close(positive_excess([0.2, 0.4, 0.4], -0.2), 0.8);
    }
    #[test]
    fn red_reservation_first_branch_literal() {
        close(reservation_index([0.2, 0.4, 0.4], 0.1), 0.75);
        close(reservation_index([0.2, 0.4, 0.4], 0.2), 0.5);
        close(reservation_index([0.2, 0.4, 0.4], 0.4), 0.25);
        close(reservation_index([0.2, 0.4, 0.4], 0.6), 0.0);
        close(reservation_index([0.2, 0.4, 0.4], 0.8), -0.2);
        close(reservation_index([1.0, 0.0, 0.0], 0.1), -0.1);
        close(reservation_index([0.6, 0.4, 0.0], 0.1), 0.25);
        close(reservation_index([0.0, 0.0, 1.0], 0.0), 1.0);
    }
    #[test]
    fn red_gibbs_nonpositive_focus_has_probability() {
        let p = softmax(&[0.0, -TAU * 3.0_f64.ln()], TAU);
        close(p[0], 0.75);
        close(p[1], 0.25);
    }
    #[test]
    fn sampled_gibbs_frequency_and_shift_scale_invariants() {
        let q = [0.0, 0.2, -0.1];
        let p = softmax(&q, 0.1);
        for (a, b) in p.iter().zip([
            0.11419519938459448,
            0.8437947344813395,
            0.042010066134066035,
        ]) {
            close(*a, b);
        }
        let shifted = softmax(&q.map(|x| x + 3.0), 0.1);
        let scaled = softmax(&q.map(|x| x * 8.0), 0.8);
        for i in 0..3 {
            close(p[i], shifted[i]);
            close(p[i], scaled[i]);
        }
        let stable = softmax(&[10000.0, 10000.0, -10000.0], TAU);
        assert_eq!(stable, vec![0.5, 0.5, 0.0]);
        // Prospective tolerance 0.006 at N=100000, fixed seed260023.
        let mut rng = SplitMix64::new(260023);
        let mut counts = [0usize; 3];
        for _ in 0..100000 {
            counts[sample_probabilities(&p, rng.uniform())] += 1;
        }
        for i in 0..3 {
            assert!((counts[i] as f64 / 100000.0 - p[i]).abs() < 0.006);
        }
        println!("fixed Gibbs frequency counts: {counts:?}");
    }
    #[test]
    fn rng_literals_and_uniform_index_bounds() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next(), 0xe220a8397b1dcdaf);
        assert_eq!(rng.next(), 0x6e789e6aa1b965f4);
        assert_eq!(rng.next(), 0x06c45d188009454f);
        for n in 1..10 {
            for _ in 0..100 {
                assert!(rng.index(n) < n);
            }
        }
        for _ in 0..1000 {
            assert!((0.0..1.0).contains(&rng.uniform()));
        }
    }
    #[test]
    fn exact_integer_minimax_counts_both_turns_without_pruning() {
        let root = replay_history(&[0, 1, 2, 4, 3, 5, 7]).unwrap();
        assert_eq!(root.current_player(), -1);
        assert_eq!(exact(&root, -1), (0, 5));
        assert_eq!(exact(&root.apply_move(6), -1), (0, 2));
        assert_eq!(exact(&root.apply_move(8), -1), (-1, 2));
        let earlier = replay_history(&[0, 1, 2, 4, 3, 5]).unwrap();
        assert_eq!(earlier.current_player(), 1);
        assert_eq!(exact(&earlier.apply_move(7), 1), (0, 5));
    }
    #[test]
    fn independent_literal_cues_and_minimax_fixture() {
        let root = replay_history(&[0, 4, 1, 8]).unwrap();
        let c = candidates(&root, 123, None);
        // Independently enumerated board oracle, not outputs of implementation helpers.
        let literal = [
            (2, [0, 0, 4], 1, 1.0, 1),
            (3, [0, 0, 4], 14, 1.0, 33),
            (5, [0, 2, 2], 16, 0.5, 47),
            (6, [0, 1, 3], 16, 1.0, 37),
            (7, [2, 0, 2], 14, 0.0, 43),
        ];
        for (actual, (mv, counts, cue, u, n)) in c.iter().zip(literal) {
            assert_eq!(actual.mv, mv);
            assert_eq!(actual.counts, counts);
            assert_eq!(actual.cue_nodes, cue);
            let (v, nodes) = exact(&actual.child, root.current_player());
            assert_eq!(utility(v), u);
            assert_eq!(nodes, n);
        }
        assert_eq!(c.iter().map(|x| x.cue_nodes).sum::<u64>(), 61);
    }
    #[test]
    fn terminal_win_initialization_and_all_actions_cost() {
        let (request, root) = read_request(request_json());
        let baseline = run_one(&root, &request, Policy::InitialStop, 0);
        assert_eq!(baseline.initial_move, 2);
        assert_eq!(baseline.initial_u, 1.0);
        assert_eq!(baseline.initial_solve_move, None);
        assert_eq!(baseline.initial_nodes, 0);
        assert_eq!(baseline.charged_nodes, 61);
        let full = run_one(&root, &request, Policy::AllActions, 0);
        assert_eq!(full.query_moves, vec![3, 5, 6, 7]);
        assert_eq!(full.additional_nodes, 160);
        assert_eq!(full.charged_nodes, 221);
        assert_eq!(full.selected_move, 2);
        assert_eq!(full.final_u, 1.0);
        for policy in [
            Policy::RandomOrder,
            Policy::GreedyVoc,
            Policy::Pandora,
            Policy::GibbsVoc,
        ] {
            let r = run_one(&root, &request, policy, 0);
            assert!(r.query_moves.is_empty());
            assert_eq!(r.stop_reason, "known_win");
        }
    }
    #[test]
    fn raw_cue_initialization_and_cap_safety() {
        let mut value = request_json();
        value["history"] = json!([0, 1, 2, 4]);
        value["additional"] = json!(1);
        let (request, root) = read_request(value);
        let mut c = candidates(&root, request.cue_seed, request.model.as_ref());
        let best = c
            .iter()
            .filter(|x| !x.terminal_known)
            .max_by(|a, b| {
                a.rollout_mean()
                    .total_cmp(&b.rollout_mean())
                    .then(b.mv.cmp(&a.mv))
            })
            .unwrap()
            .mv;
        let initial = initialize(&mut c, root.current_player());
        assert_eq!(initial.solve_move, Some(best));
        let base = run_one(&root, &request, Policy::InitialStop, 0);
        for p in Policy::ALL {
            let r = run_one(&root, &request, p, 0);
            assert_eq!(
                (r.initial_move, r.initial_u, r.initial_nodes, r.cue_nodes),
                (
                    base.initial_move,
                    base.initial_u,
                    base.initial_nodes,
                    base.cue_nodes
                )
            );
            assert!(r.final_u >= r.initial_u);
            assert!(r.additional_queries <= 1);
            assert_eq!(
                r.charged_nodes,
                r.cue_nodes + r.initial_nodes + r.additional_nodes
            );
            assert!(r.controller_us >= 0.0 && r.elapsed_us >= r.query_us);
            assert_eq!(r.decisions.as_ref().unwrap().last().unwrap().choice, "STOP");
        }
    }
    #[test]
    fn model_fallback_order_and_normalization() {
        let mut value = model_json();
        value["feature_probs"]["3:0:2:2"] = json!([0.1, 0.2, 0.7]);
        value["remaining_probs"]["3"] = json!([0.3, 0.4, 0.3]);
        value["remaining_costs"]["3"] = json!(23.0);
        let mut m: Model = serde_json::from_value(value).unwrap();
        m.validate().unwrap();
        assert_eq!(m.predict(3, [0, 2, 2]), ([0.1, 0.2, 0.7], 23.0));
        assert_eq!(m.predict(3, [4, 0, 0]), ([0.3, 0.4, 0.3], 23.0));
        assert_eq!(m.predict(4, [4, 0, 0]), ([0.2, 0.4, 0.4], 17.0));
        let mut p = [0.2, 0.4, 0.4000000000001];
        normalize_probs(&mut p).unwrap();
        close(p.iter().sum(), 1.0);
    }
    #[test]
    fn strict_request_and_model_boundaries_and_recovery() {
        let mut bads = Vec::new();
        let good = request_json();
        for (field, value) in [
            ("history", json!([0, 0])),
            ("history", json!([9])),
            ("history", json!([0, 3, 1, 4, 2])),
            ("history", json!([0, 3, 1, 4, 2, 5])),
            ("history", json!([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            ("additional", json!(0)),
            ("additional", json!(10)),
            ("repeats", json!(0)),
            ("repeats", json!(16)),
            ("warmup", json!(3)),
            ("id", json!("")),
            ("mode", json!("oracle")),
            ("model", Value::Null),
            ("offline_u", json!([1, 1, 1])),
        ] {
            let mut v = good.clone();
            v[field] = value;
            bads.push(v);
        }
        for (field, value) in [
            ("schema", json!("unknown")),
            ("global_probs", json!([0.0, 0.5, 0.5])),
            ("global_probs", json!([0.2, 0.3, 0.4])),
            ("global_cost", json!(0.0)),
            ("global_cost", json!(1000001.0)),
            ("lambda_nodes", json!(0.1)),
            ("tau", json!(0.0)),
            ("oracle", json!([1, 0, 0])),
            ("feature_probs", json!({"3:1:1:1":[0.2,0.4,0.4]})),
            ("remaining_probs", json!({"9":[0.2,0.4,0.4]})),
        ] {
            let mut v = good.clone();
            v["model"][field] = value;
            bads.push(v);
        }
        for bad in bads {
            assert!(
                handle_line(&bad.to_string()).get("error").is_some(),
                "{bad}"
            );
        }
        let stream = format!("not json\n{}\n", good);
        let mut out = Vec::new();
        serve(io::Cursor::new(stream), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(serde_json::from_str::<Value>(lines[0])
            .unwrap()
            .get("error")
            .is_some());
        assert_eq!(
            serde_json::from_str::<Value>(lines[1]).unwrap()["runs"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
    }
    #[test]
    fn repeated_logic_and_no_unqueried_oracle_leak() {
        let mut value = request_json();
        value["history"] = json!([0, 1, 2, 4]);
        value["repeats"] = json!(2);
        value["policy_seed"] = json!(3);
        let response = handle_line(&value.to_string());
        let runs = response["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 12);
        for i in 0..6 {
            for field in [
                "initial_move",
                "initial_solve_move",
                "initial_u",
                "selected_move",
                "final_u",
                "cue_nodes",
                "initial_nodes",
                "additional_nodes",
                "charged_nodes",
                "query_moves",
                "decisions",
            ] {
                assert_eq!(runs[i][field], runs[i + 6][field], "{field}");
            }
            for p in runs[i]["predictions"].as_array().unwrap() {
                assert!(p.get("u").is_none());
                if p["terminal_known"] == false {
                    assert!(p["observed_u"].is_null());
                }
            }
        }
    }
    #[test]
    fn survey_requires_model_free_convention_and_paid_terminal() {
        let mut value = request_json();
        value["mode"] = json!("survey");
        value["model"] = Value::Null;
        value["trace"] = json!(false);
        let response = handle_line(&value.to_string());
        assert_eq!(response["cue_nodes"], 61);
        let first = &response["candidates"][0];
        assert_eq!(first["remaining"], 0);
        assert_eq!(first["solve_nodes"], 1);
        assert_eq!(first["cue_nodes"], 1);
        assert_eq!(first["u"], 1.0);
        value["policy_seed"] = json!(1);
        assert!(handle_line(&value.to_string()).get("error").is_some());
    }
}
