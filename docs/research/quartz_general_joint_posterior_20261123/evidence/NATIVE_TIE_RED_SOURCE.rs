//! C27 joint-posterior finite two-query research executable.
//! Frozen C26 substrate copied without changing game/solver/cue/initialization semantics.
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
        if p.iter().any(|x| !x.is_finite() || *x < 0.0) || (p.iter().sum::<f64>() - 1.0).abs() > EPS
        {
            return Err("invalid categorical law".into());
        }
        let mu = p[1] * 0.5 + p[2];
        let scale = SUPPORT
            .iter()
            .zip(p)
            .filter(|(_, mass)| *mass > 0.0)
            .map(|(u, _)| (u - mu).abs())
            .fold(0.0_f64, f64::max);
        let f = if scale > 0.0 {
            SUPPORT.map(|u| (u - mu) / scale)
        } else {
            [0.0; 3]
        };
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
        Ok(Self {
            laws: probabilities
                .iter()
                .copied()
                .map(AtomLaw::new)
                .collect::<Result<_, _>>()?,
            rho,
            log_odds: 0.0,
            observed: vec![false; probabilities.len()],
        })
    }
    fn predict_at(&self, i: usize, field: f64) -> [f64; 3] {
        let m = (field * 0.5).tanh();
        let law = &self.laws[i];
        std::array::from_fn(|k| law.p[k] * (1.0 + self.rho * m * law.f[k]))
    }
    fn predict(&self, i: usize) -> [f64; 3] {
        self.predict_at(i, self.log_odds)
    }
    fn after(&self, i: usize, u: f64, field: f64) -> Result<f64, String> {
        let atom = SUPPORT
            .iter()
            .position(|x| *x == u)
            .ok_or("utility outside support")?;
        if self.laws[i].p[atom] <= 0.0 {
            return Err("unsupported observation".into());
        }
        let tilt = self.rho * self.laws[i].f[atom];
        let next = field + tilt.ln_1p() - (-tilt).ln_1p();
        if !next.is_finite() {
            return Err("nonfinite posterior log odds".into());
        }
        Ok(next)
    }
    fn project(&self) -> Result<Self, String> {
        let probabilities: Vec<_> = (0..self.laws.len()).map(|i| self.predict(i)).collect();
        let mut projected = Self::new(&probabilities, 0.0)?;
        projected.observed = self.observed.clone();
        Ok(projected)
    }
    fn observe(&mut self, i: usize, u: f64) -> Result<(), String> {
        if self.observed[i] {
            return Err("duplicate observation".into());
        }
        self.log_odds = self.after(i, u, self.log_odds)?;
        self.observed[i] = true;
        Ok(())
    }
}

fn positive_excess(p: [f64; 3], b: f64) -> f64 {
    p.iter()
        .zip(SUPPORT)
        .map(|(p, u)| p * (u - b).max(0.0))
        .sum()
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

fn score_actions(
    belief: &Belief,
    remaining: &[usize],
    moves: &[usize],
    nodes: &[f64],
    b: f64,
    horizon: usize,
) -> Result<Vec<Score>, String> {
    if !(1..=2).contains(&horizon) {
        return Err("planning horizon must be one or two".into());
    }
    let mut result = Vec::with_capacity(remaining.len());
    for &a in remaining {
        let p = belief.predict(a);
        let g = positive_excess(p, b);
        let k1 = g - LAMBDA * nodes[a];
        let k2 = if horizon == 2 {
            let mut continuation = 0.0;
            for (u, mass) in SUPPORT.into_iter().zip(p) {
                if mass == 0.0 {
                    continue;
                }
                let field = belief.after(a, u, belief.log_odds)?;
                let mut last = 0.0_f64;
                for &j in remaining {
                    if j != a {
                        last = last.max(
                            positive_excess(belief.predict_at(j, field), b.max(u))
                                - LAMBDA * nodes[j],
                        );
                    }
                }
                continuation += mass * last;
            }
            Some(k1 + continuation)
        } else {
            None
        };
        result.push(Score {
            mv: moves[a],
            probabilities: p,
            predicted_nodes: nodes[a],
            g,
            k1,
            k2,
            q: k2.unwrap_or(k1),
        });
    }
    Ok(result)
}

fn best_action(scores: &[Score]) -> Option<usize> {
    let mut chosen: Option<usize> = None;
    for (i, score) in scores.iter().enumerate() {
        if score.q <= EPS {
            continue;
        }
        if chosen
            .map(|j| {
                score.q > scores[j].q + EPS
                    || ((score.q - scores[j].q).abs() <= EPS && score.mv < scores[j].mv)
            })
            .unwrap_or(true)
        {
            chosen = Some(i);
        }
    }
    chosen
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
    rho: f64,
}
impl Request {
    fn validate(&mut self) -> Result<TicTacToe, String> {
        if self.id.is_empty() || self.id.len() > 128 {
            return Err("id must contain 1..128 UTF-8 bytes".into());
        }
        if self.additional != 2 || !(1..=15).contains(&self.repeats) || self.warmup > 2 {
            return Err("additional must be 2; repeats 1..15; warmup 0..2".into());
        }
        if !self.rho.is_finite() || !(0.0..1.0).contains(&self.rho) {
            return Err("rho must be finite in [0,1)".into());
        }
        match self.mode.as_str() {
            "run" => self
                .model
                .as_mut()
                .ok_or("run requires model")?
                .validate()?,
            "survey" => {
                if self.model.is_some()
                    || self.rho != 0.0
                    || self.repeats != 5
                    || self.warmup != 0
                    || self.trace
                    || self.policy_seed != 0
                {
                    return Err("survey requires model=null,rho=0,repeats=5,warmup=0,trace=false,policy_seed=0".into());
                }
            }
            _ => return Err("mode must be run or survey".into()),
        }
        replay_history(&self.history)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Policy {
    InitialStop,
    RandomTwo,
    ProductH2,
    InitialProjectedH2,
    JointH2,
}
impl Policy {
    const ALL: [Self; 5] = [
        Self::InitialStop,
        Self::RandomTwo,
        Self::ProductH2,
        Self::InitialProjectedH2,
        Self::JointH2,
    ];
    fn rational(self) -> bool {
        matches!(
            self,
            Self::ProductH2 | Self::InitialProjectedH2 | Self::JointH2
        )
    }
    fn uses_latent(self) -> bool {
        matches!(self, Self::InitialProjectedH2 | Self::JointH2)
    }
}

#[derive(Serialize)]
struct Prediction {
    #[serde(rename = "move")]
    mv: usize,
    remaining: usize,
    counts: [u8; 3],
    terminal_known: bool,
    probabilities: [f64; 3],
    predicted_nodes: Option<f64>,
    observed_u: Option<f64>,
}
#[derive(Serialize)]
struct Observation {
    #[serde(rename = "move")]
    mv: usize,
    u: f64,
    kind: &'static str,
}
#[derive(Serialize)]
struct Projection {
    #[serde(rename = "move")]
    mv: usize,
    probabilities: [f64; 3],
}
#[derive(Serialize)]
struct Decision {
    b_before: f64,
    b_after: f64,
    remaining: Vec<usize>,
    horizon: usize,
    log_odds_before: f64,
    log_odds_after: f64,
    candidates: Vec<Score>,
    chosen_move: Option<usize>,
    choice: &'static str,
    queried_u: Option<f64>,
    queried_nodes: u64,
    reason: &'static str,
}
#[derive(Serialize)]
struct RunRecord {
    policy: Policy,
    repeat: usize,
    execution_slot: usize,
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
    elapsed_ns: u64,
    cue_ns: u64,
    initial_ns: u64,
    query_ns: u64,
    controller_ns: u64,
    stop_reason: &'static str,
    initial_log_odds: f64,
    final_log_odds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    known_observations: Option<Vec<Observation>>,
    projection: Option<Vec<Projection>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    predictions: Option<Vec<Prediction>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decisions: Option<Vec<Decision>>,
}

fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn candidate_law(c: &Candidate) -> [f64; 3] {
    if let Some(p) = c.probabilities {
        p
    } else {
        let mut p = [0.0; 3];
        p[(c.known.expect("terminal candidate is known") + 1) as usize] = 1.0;
        p
    }
}

fn run_one(
    root: &TicTacToe,
    request: &Request,
    policy: Policy,
    repeat: usize,
    slot: usize,
) -> Result<RunRecord, String> {
    let start = Instant::now();
    let model = request.model.as_ref().ok_or("model is required")?;
    let mut output = {
        let cue_start = Instant::now();
        let mut candidates = candidates(root, request.cue_seed, Some(model));
        let cue_nodes = candidates.iter().map(|c| c.cue_nodes).sum::<u64>();
        let cue_ns = nanos(cue_start);
        let base: Vec<_> = candidates.iter().map(candidate_law).collect();
        let mut belief = Belief::new(
            &base,
            if policy.uses_latent() {
                request.rho
            } else {
                0.0
            },
        )?;
        let predictions = request.trace.then(|| {
            candidates
                .iter()
                .enumerate()
                .map(|(i, c)| Prediction {
                    mv: c.mv,
                    remaining: c.remaining,
                    counts: c.counts,
                    terminal_known: c.terminal_known,
                    probabilities: base[i],
                    predicted_nodes: c.predicted_nodes,
                    observed_u: if c.terminal_known {
                        c.known.map(utility)
                    } else {
                        None
                    },
                })
                .collect()
        });
        let mut known_observations = request.trace.then(Vec::new);
        for (i, c) in candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.terminal_known)
        {
            let u = utility(c.known.unwrap());
            belief.observe(i, u)?;
            if let Some(obs) = &mut known_observations {
                obs.push(Observation {
                    mv: c.mv,
                    u,
                    kind: "terminal",
                });
            }
        }
        let initial_start = Instant::now();
        let initial = initialize(&mut candidates, root.current_player());
        let initial_ns = nanos(initial_start);
        if let Some(mv) = initial.solve_move {
            let i = candidates.iter().position(|c| c.mv == mv).unwrap();
            let u = utility(candidates[i].known.unwrap());
            belief.observe(i, u)?;
            if let Some(obs) = &mut known_observations {
                obs.push(Observation {
                    mv,
                    u,
                    kind: "initial",
                });
            }
        }
        let initial_log_odds = belief.log_odds;
        let mut remaining: Vec<_> = (0..candidates.len())
            .filter(|&i| candidates[i].known.is_none())
            .collect();
        let projection = if policy == Policy::InitialProjectedH2 {
            let snapshot = request.trace.then(|| {
                remaining
                    .iter()
                    .map(|&i| Projection {
                        mv: candidates[i].mv,
                        probabilities: belief.predict(i),
                    })
                    .collect()
            });
            belief = belief.project()?;
            snapshot
        } else {
            None
        };
        let moves: Vec<_> = candidates.iter().map(|c| c.mv).collect();
        let nodes: Vec<_> = candidates
            .iter()
            .map(|c| c.predicted_nodes.unwrap_or(0.0))
            .collect();
        let initial_move = candidates[initial.incumbent].mv;
        let mut incumbent = initial.incumbent;
        let mut value = initial.value;
        let mut query_moves = Vec::new();
        let mut additional_nodes = 0u64;
        let mut predicted_additional_nodes = 0.0;
        let mut query_ns = 0u64;
        let mut decisions = request.trace.then(Vec::new);
        let mut rng = SplitMix64::new(request.cue_seed ^ POLICY_SALT ^ request.policy_seed);
        let stop_reason;
        loop {
            let b = utility(value);
            let horizon = request.additional - query_moves.len();
            let forced = if policy == Policy::InitialStop {
                Some("initial_stop")
            } else if horizon == 0 {
                Some("cap_reached")
            } else if remaining.is_empty() {
                Some("exhausted")
            } else if value == 1 {
                Some("known_win")
            } else {
                None
            };
            let log_odds_before = belief.log_odds;
            let before_moves = if request.trace {
                remaining.iter().map(|&i| moves[i]).collect()
            } else {
                Vec::new()
            };
            // Forced stops never pay for a needless Bellman evaluation, even in traced runs.
            let scored = if forced.is_none() && (request.trace || policy.rational()) {
                score_actions(&belief, &remaining, &moves, &nodes, b, horizon)?
            } else {
                Vec::new()
            };
            let (position, reason) = if let Some(reason) = forced {
                (None, reason)
            } else if policy == Policy::RandomTwo {
                (Some(rng.index(remaining.len())), "random_two")
            } else {
                let action = best_action(&scored);
                (
                    action,
                    if action.is_none() {
                        "nonpositive_voc"
                    } else if horizon == 2 {
                        "positive_h2"
                    } else {
                        "positive_h1"
                    },
                )
            };
            let chosen_move = position.map(|pos| moves[remaining[pos]]);
            let mut queried_u = None;
            let mut queried_nodes = 0;
            if let Some(pos) = position {
                let i = remaining.remove(pos);
                let query_start = Instant::now();
                let (v, n) = exact(&candidates[i].child, root.current_player());
                query_ns += nanos(query_start);
                queried_u = Some(utility(v));
                queried_nodes = n;
                belief.observe(i, utility(v))?;
                candidates[i].known = Some(v);
                additional_nodes += n;
                predicted_additional_nodes += nodes[i];
                query_moves.push(moves[i]);
                if v > value {
                    value = v;
                    incumbent = i;
                }
            }
            if let Some(trace) = &mut decisions {
                trace.push(Decision {
                    b_before: b,
                    b_after: utility(value),
                    remaining: before_moves,
                    horizon,
                    log_odds_before,
                    log_odds_after: belief.log_odds,
                    candidates: scored,
                    chosen_move,
                    choice: if chosen_move.is_some() {
                        "FOCUS"
                    } else {
                        "STOP"
                    },
                    queried_u,
                    queried_nodes,
                    reason,
                });
            }
            if chosen_move.is_none() {
                stop_reason = reason;
                break;
            }
        }
        RunRecord {
            policy,
            repeat,
            execution_slot: slot,
            initial_move,
            initial_solve_move: initial.solve_move,
            initial_u: utility(initial.value),
            selected_move: moves[incumbent],
            final_u: utility(value),
            cue_nodes,
            initial_nodes: initial.nodes,
            additional_nodes,
            charged_nodes: cue_nodes + initial.nodes + additional_nodes,
            additional_queries: query_moves.len(),
            query_moves,
            predicted_additional_nodes,
            elapsed_ns: 0,
            cue_ns,
            initial_ns,
            query_ns,
            controller_ns: 0,
            stop_reason,
            initial_log_odds,
            final_log_odds: belief.log_odds,
            known_observations,
            projection,
            predictions,
            decisions,
        }
    };
    output.elapsed_ns = nanos(start);
    output.controller_ns = output
        .elapsed_ns
        .saturating_sub(output.cue_ns + output.initial_ns + output.query_ns);
    Ok(output)
}

fn survey(root: &TicTacToe, request: &Request) -> Value {
    let start = Instant::now();
    let candidates = candidates(root, request.cue_seed, None);
    let cue_nodes = candidates.iter().map(|c| c.cue_nodes).sum::<u64>();
    let measured: Vec<_> = candidates
        .iter()
        .map(|c| {
            let mut timings = Vec::new();
            let mut known = c.known;
            let mut solve_nodes = 1;
            if !c.terminal_known {
                for repeat in 0..request.repeats {
                    let clock = Instant::now();
                    let (v, n) = exact(&c.child, root.current_player());
                    let elapsed_ns = nanos(clock);
                    known = Some(v);
                    solve_nodes = n;
                    timings.push(json!({"repeat":repeat,"solve_nodes":n,"elapsed_ns":elapsed_ns}));
                }
            }
            json!({"move":c.mv,"terminal_known":c.terminal_known,"remaining":c.remaining,
            "counts":c.counts,"rollout_mean":c.rollout_mean(),"u":utility(known.unwrap()),
            "solve_nodes":solve_nodes,"cue_nodes":c.cue_nodes,"timings":timings})
        })
        .collect();
    json!({"schema":SCHEMA,"mode":"survey","id":request.id,"candidates":measured,
        "cue_nodes":cue_nodes,"measurement_elapsed_ns":nanos(start)})
}

fn run(root: &TicTacToe, request: &Request) -> Result<Value, String> {
    let dispatch = Instant::now();
    let offset = (request.policy_seed % 5) as usize;
    let mut warmup_elapsed_ns = 0u64;
    for repeat in 0..request.warmup {
        for slot in 0..5 {
            warmup_elapsed_ns += run_one(
                root,
                request,
                Policy::ALL[(slot + repeat + offset) % 5],
                repeat,
                slot,
            )?
            .elapsed_ns;
        }
    }
    let mut runs = Vec::with_capacity(request.repeats * 5);
    for repeat in 0..request.repeats {
        for slot in 0..5 {
            runs.push(run_one(
                root,
                request,
                Policy::ALL[(slot + repeat + offset) % 5],
                repeat,
                slot,
            )?);
        }
    }
    runs.sort_by_key(|r| (r.repeat, r.policy as usize));
    let execution_elapsed_ns = nanos(dispatch);
    Ok(
        json!({"schema":SCHEMA,"mode":"run","id":request.id,"runs":runs,
        "warmup_elapsed_ns":warmup_elapsed_ns,"execution_elapsed_ns":execution_elapsed_ns}),
    )
}

fn handle_line(line: &str) -> Value {
    let decode = Instant::now();
    let parsed = serde_json::from_str::<Request>(line)
        .map_err(|e| format!("invalid request: {e}"))
        .and_then(|mut request| {
            let root = request.validate()?;
            Ok((request, root))
        });
    let decode_validate_ns = nanos(decode);
    let response = parsed.and_then(|(request, root)| {
        if request.mode == "survey" {
            Ok(survey(&root, &request))
        } else {
            run(&root, &request)
        }
    });
    let mut response = match response {
        Ok(v) => v,
        Err(error) => json!({"error":error}),
    };
    response["decode_validate_ns"] = json!(decode_validate_ns);
    response
}
fn serve(input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    for line in input.lines() {
        let response = match line {
            Ok(line) => handle_line(&line),
            Err(e) => json!({"error":format!("stdin read error: {e}")}),
        };
        let clock = Instant::now();
        let mut encoded = serde_json::to_vec(&response)?;
        let serialize_payload_ns = nanos(clock);
        // Serialize the result once. Only this final timing-field formatting and I/O
        // occur after the measured payload serialization interval.
        let closing = encoded.pop();
        debug_assert_eq!(closing, Some(b'}'));
        write!(
            &mut encoded,
            ",\"serialize_payload_ns\":{serialize_payload_ns}}}\n"
        )?;
        output.write_all(&encoded)?;
        output.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = serve(io::stdin().lock(), io::stdout().lock()) {
        eprintln!("C27 I/O error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod c27_tests {
    use super::*;
    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-12,
            "actual={actual:.17}, expected={expected:.17}"
        );
    }
    fn information_fixture() -> Belief {
        Belief::new(&[[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], 0.8).unwrap()
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
        let scores = score_actions(
            &belief,
            &[0, 1],
            &[0, 1],
            &[0.01 / LAMBDA, 0.30 / LAMBDA],
            0.5,
            2,
        )
        .unwrap();
        let value = 0.5 + best_action(&scores).map(|i| scores[i].q).unwrap_or(0.0);
        close(value, 0.545);
        assert_eq!(best_action(&scores), Some(0));
    }
    #[test]
    fn global_ties_use_the_maximum_band_not_nontransitive_pairwise_scanning() {
        let scores: Vec<Score> = [0.2, 0.2 + 0.9e-12, 0.2 + 1.8e-12]
            .into_iter().zip([2, 5, 8]).map(|(q,mv)| Score {
                mv, probabilities: [0.5, 0.0, 0.5], predicted_nodes: 1.0,
                g: 0.5, k1: q, k2: None, q,
            }).collect();
        assert_eq!(best_action(&scores), Some(1));
    }
    fn model_json() -> Value {
        json!({"schema":"c26-categorical-v1","feature_probs":{},"remaining_probs":{},
            "global_probs":[0.2,0.4,0.4],"remaining_costs":{},"global_cost":17.0,
            "lambda_nodes":LAMBDA,"tau":TAU})
    }
    fn request_json() -> Value {
        json!({"id":"c27-native-fixture","history":[0,1,2,4],"cue_seed":123,"policy_seed":0,
            "mode":"run","additional":2,"repeats":2,"warmup":1,"trace":true,
            "model":model_json(),"rho":0.8})
    }
    fn semantic(mut v: Value) -> Value {
        let obj = v.as_object_mut().unwrap();
        for k in [
            "repeat",
            "execution_slot",
            "elapsed_ns",
            "cue_ns",
            "initial_ns",
            "query_ns",
            "controller_ns",
        ] {
            obj.remove(k);
        }
        v
    }
    #[test]
    fn posterior_normalizes_preserves_prior_and_rejects_unsupported_or_repeat() {
        let mut b = Belief::new(&[[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]], 0.9).unwrap();
        for (a, p) in [[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]].iter().enumerate() {
            for k in 0..3 {
                close(b.predict(a)[k], p[k]);
            }
        }
        assert!(b.observe(1, 1.0).is_err());
        assert_eq!(b.observed, vec![false, false]);
        b.observe(1, 0.5).unwrap();
        close(b.log_odds, 0.0);
        assert!(b.observe(1, 0.5).is_err());
        b.observe(0, 1.0).unwrap();
        close(b.predict(0).iter().sum(), 1.0);
        assert!(Belief::new(&[[0.2, 0.3, 0.5]], 1.0).is_err());
        assert!(Belief::new(&[[0.2, 0.3, 0.5]], f64::NAN).is_err());
    }
    #[test]
    fn posterior_order_and_rho_zero_are_exact_product_limits() {
        let laws = [[0.2, 0.3, 0.5], [0.4, 0.2, 0.4], [0.1, 0.6, 0.3]];
        let mut a = Belief::new(&laws, 0.9).unwrap();
        let mut b = a.clone();
        a.observe(0, 1.0).unwrap();
        a.observe(1, 0.0).unwrap();
        b.observe(1, 0.0).unwrap();
        b.observe(0, 1.0).unwrap();
        close(a.log_odds, b.log_odds);
        for k in 0..3 {
            close(a.predict(2)[k], b.predict(2)[k]);
        }
        let mut product = Belief::new(&laws, 0.0).unwrap();
        product.observe(0, 1.0).unwrap();
        product.observe(1, 0.0).unwrap();
        assert_eq!(product.predict(2), laws[2]);
        close(product.log_odds, 0.0);
    }
    #[test]
    fn projected_control_freezes_initial_marginals_but_joint_updates_again() {
        let mut joint =
            Belief::new(&[[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]], 0.8).unwrap();
        joint.observe(0, 0.5).unwrap();
        let mut projected = joint.project().unwrap();
        close(projected.predict(2)[2], 0.82);
        close(projected.log_odds, 0.0);
        joint.observe(1, 0.0).unwrap();
        projected.observe(1, 0.0).unwrap();
        close(joint.predict(2)[2], 0.5);
        close(projected.predict(2)[2], 0.82);
        assert!(projected.observe(0, 0.5).is_err());
    }
    #[test]
    fn information_threshold_h1_and_stop_ties_keep_the_declared_horizon() {
        let b = information_fixture();
        let moves = [0, 1];
        let s1 =
            score_actions(&b, &[0, 1], &moves, &[0.01 / LAMBDA, 0.3 / LAMBDA], 0.5, 1).unwrap();
        assert!(best_action(&s1).is_none());
        assert!(s1.iter().all(|s| s.k2.is_none()));
        // rho^2=0.64 = 4*0.05+8*0.055 gives a literal zero-information-value boundary.
        let boundary =
            score_actions(&b, &[0, 1], &moves, &[0.055 / LAMBDA, 0.3 / LAMBDA], 0.5, 2).unwrap();
        assert!(best_action(&boundary).is_none());
        let win =
            score_actions(&b, &[0, 1], &moves, &[0.01 / LAMBDA, 0.3 / LAMBDA], 1.0, 2).unwrap();
        assert!(best_action(&win).is_none());
        let tie = Belief::new(&[[0.5, 0.0, 0.5]; 2], 0.0).unwrap();
        let same = score_actions(&tie, &[0, 1], &[8, 3], &[1.0, 1.0], 0.0, 1).unwrap();
        assert_eq!(same[best_action(&same).unwrap()].mv, 3);
    }
    #[test]
    fn h2_hypotheses_do_not_mutate_real_posterior_and_moving_threshold_has_negative_correction() {
        let b = Belief::new(&[[0.5, 0.0, 0.5]; 2], 0.9).unwrap();
        let scores = score_actions(&b, &[0, 1], &[0, 1], &[0.1 / LAMBDA; 2], 0.0, 2).unwrap();
        // Actual future option is zero; frozen-marginal option was .2.
        close(scores[0].k2.unwrap(), 0.4);
        close(scores[0].k1, 0.4);
        close(scores[0].k2.unwrap() - scores[0].k1 - 0.2, -0.2);
        close(b.log_odds, 0.0);
        assert_eq!(b.observed, vec![false, false]);
    }
    #[test]
    fn frozen_game_cues_root_frame_and_exact_nodes_remain_literal() {
        let root = replay_history(&[0, 4, 1, 8]).unwrap();
        let c = candidates(&root, 123, None);
        let literal = [
            (2, [0, 0, 4], 1, 1.0, 1),
            (3, [0, 0, 4], 14, 1.0, 33),
            (5, [0, 2, 2], 16, 0.5, 47),
            (6, [0, 1, 3], 16, 1.0, 37),
            (7, [2, 0, 2], 14, 0.0, 43),
        ];
        for (a, (mv, counts, cue, u, n)) in c.iter().zip(literal) {
            assert_eq!((a.mv, a.counts, a.cue_nodes), (mv, counts, cue));
            let (v, actual) = exact(&a.child, root.current_player());
            assert_eq!((utility(v), actual), (u, n));
        }
        let other = replay_history(&[0, 1, 2, 4, 3, 5, 7]).unwrap();
        assert_eq!(exact(&other, other.current_player()), (0, 5));
    }
    #[test]
    fn five_policy_execution_rotates_but_trace_semantics_repeat_and_queries_are_unique() {
        let response = handle_line(&request_json().to_string());
        assert!(response.get("error").is_none(), "{response}");
        let runs = response["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 10);
        for i in 0..5 {
            assert_eq!(semantic(runs[i].clone()), semantic(runs[i + 5].clone()));
            assert_eq!(runs[i]["execution_slot"], json!(i));
            assert_eq!(runs[i + 5]["execution_slot"], json!((i + 4) % 5));
            let r = &runs[i];
            let qs = r["query_moves"].as_array().unwrap();
            assert!(qs.len() <= 2);
            let mut unique = qs.iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), qs.len());
            assert_eq!(
                r["charged_nodes"].as_u64().unwrap(),
                r["cue_nodes"].as_u64().unwrap()
                    + r["initial_nodes"].as_u64().unwrap()
                    + r["additional_nodes"].as_u64().unwrap()
            );
            let phases = ["cue_ns", "initial_ns", "query_ns", "controller_ns"]
                .iter()
                .map(|k| r[k].as_u64().unwrap())
                .sum::<u64>();
            assert_eq!(r["elapsed_ns"].as_u64().unwrap(), phases);
            for prediction in r["predictions"].as_array().unwrap() {
                if prediction["terminal_known"] == false {
                    assert!(prediction["observed_u"].is_null());
                }
            }
            for d in r["decisions"].as_array().unwrap() {
                let h = d["horizon"].as_u64().unwrap();
                assert!(h <= 2);
                for s in d["candidates"].as_array().unwrap() {
                    assert_eq!(s["k2"].is_null(), h == 1);
                }
            }
        }
        assert!(runs[0]["query_moves"].as_array().unwrap().is_empty());
        assert!(runs[3]["projection"].is_array());
    }
    #[test]
    fn native_rho_zero_joint_and_projected_match_product_decisions() {
        let mut r = request_json();
        r["rho"] = json!(0.0);
        r["repeats"] = json!(1);
        r["warmup"] = json!(0);
        let output = handle_line(&r.to_string());
        let runs = output["runs"].as_array().unwrap();
        for i in [3, 4] {
            for key in [
                "query_moves",
                "selected_move",
                "final_u",
                "charged_nodes",
                "decisions",
            ] {
                assert_eq!(runs[2][key], runs[i][key]);
            }
        }
    }
    #[test]
    fn terminal_win_skips_initial_query_and_all_planning() {
        let mut r = request_json();
        r["history"] = json!([0, 4, 1, 8]);
        r["repeats"] = json!(1);
        r["warmup"] = json!(0);
        let out = handle_line(&r.to_string());
        for run in out["runs"].as_array().unwrap() {
            assert_eq!(run["initial_move"], json!(2));
            assert_eq!(run["initial_u"], json!(1.0));
            assert!(run["initial_solve_move"].is_null());
            assert_eq!(run["initial_nodes"], json!(0));
            assert_eq!(run["charged_nodes"], json!(61));
            assert_eq!(run["additional_queries"], json!(0));
            assert!(run["decisions"][0]["candidates"]
                .as_array()
                .unwrap()
                .is_empty());
        }
    }
    #[test]
    fn survey_times_five_real_solves_and_excludes_terminal_from_price_samples() {
        let mut r = request_json();
        r["history"] = json!([0, 4, 1, 8]);
        r["mode"] = json!("survey");
        r["model"] = Value::Null;
        r["rho"] = json!(0.0);
        r["repeats"] = json!(5);
        r["warmup"] = json!(0);
        r["trace"] = json!(false);
        let out = handle_line(&r.to_string());
        assert!(out.get("error").is_none(), "{out}");
        let cs = out["candidates"].as_array().unwrap();
        assert_eq!(cs.len(), 5);
        assert!(cs[0]["timings"].as_array().unwrap().is_empty());
        for (c, n) in cs.iter().skip(1).zip([33, 47, 37, 43]) {
            let times = c["timings"].as_array().unwrap();
            assert_eq!(times.len(), 5);
            for (repeat, t) in times.iter().enumerate() {
                assert_eq!(t["repeat"], json!(repeat));
                assert_eq!(t["solve_nodes"], json!(n));
                assert!(t["elapsed_ns"].as_u64().unwrap() > 0);
            }
        }
    }
    #[test]
    fn strict_public_request_and_model_reject_labels_invalid_support_and_continue_jsonl() {
        for key in ["offline_values", "oracle", "source_hash", "unknown"] {
            let mut r = request_json();
            r[key] = json!([1, 0, 1]);
            assert!(handle_line(&r.to_string()).get("error").is_some());
        }
        for (key, value) in [
            ("additional", json!(3)),
            ("rho", json!(1.0)),
            ("rho", json!(-0.1)),
            ("history", json!([0, 0])),
        ] {
            let mut r = request_json();
            r[key] = value;
            assert!(handle_line(&r.to_string()).get("error").is_some());
        }
        for value in [json!([0.0, 0.5, 0.5]), json!([0.2, 0.3, 0.6])] {
            let mut r = request_json();
            r["model"]["global_probs"] = value;
            assert!(handle_line(&r.to_string()).get("error").is_some());
        }
        let mut r = request_json();
        r["model"]["label"] = json!(1);
        assert!(handle_line(&r.to_string()).get("error").is_some());
        let input = format!("not-json\n{}\n", request_json());
        let mut output = Vec::new();
        serve(io::Cursor::new(input), &mut output).unwrap();
        let rows = String::from_utf8(output).unwrap();
        let parsed: Vec<Value> = rows
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].get("error").is_some());
        assert_eq!(parsed[1]["schema"], json!(SCHEMA));
        assert!(parsed[1]["serialize_payload_ns"].is_u64());
        assert!(parsed[1]["decode_validate_ns"].is_u64());
    }
}
