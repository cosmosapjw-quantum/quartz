//! C28 signed latent action geometry and cached direct/lifted finite-horizon Bellman kernels.
//! Authority: docs/research/quartz_c28_action_geometry_20261123/coding/CONTRACT.json.
//! Utilities and costs are dimensionless; node price is 1/1024 utility per entered node.
//! C26 game substrate copied without policy changes from SSOT 87c15dc828d6eea519c219fd8a412fa21dfc464d,
//! src/bin/quartz_c26_game_metareasoning.rs SHA256
//! 145b586fea7ff7526eb30d15d683580678f1c9dd3a433967f536697957327a01.
//! The game bridge uses a fixed common positive loading; synthetic laws permit signed loadings.
//! No game oracle outcomes are accepted as inputs. Direct and lifted have identical preparation.
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
use std::hint::black_box;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tictactoe::TicTacToe;

const LAMBDA: f64 = 1.0 / 1024.0;
const TAU: f64 = 1.0 / 32.0;
const GAMMA: u64 = 0x9e3779b97f4a7c15;
const SCHEMA: &str = "c28-geometry-v1";
const SUPPORT: [f64; 3] = [0.0, 0.5, 1.0];
const EPS: f64 = 1e-12;

// BEGIN frozen C26 model, game, cue and initial-verification substrate.

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

// END frozen C26 substrate.

fn default_horizon() -> u8 {
    2
}
fn default_iterations() -> usize {
    1
}
fn default_rho() -> f64 {
    0.75
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Request {
    Synthetic(SyntheticRequest),
    Game(GameRequest),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticRequest {
    #[serde(default)]
    id: String,
    probabilities: Vec<[f64; 3]>,
    loadings: Vec<f64>,
    costs: Vec<f64>,
    m: f64,
    b: f64,
    #[serde(default = "default_horizon")]
    horizon: u8,
    #[serde(default)]
    repeats: usize,
    #[serde(default)]
    warmup: usize,
    #[serde(default = "default_iterations")]
    iterations: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GameRequest {
    #[serde(default)]
    id: String,
    history: Vec<usize>,
    cue_seed: u64,
    model: Model,
    #[serde(default = "default_rho")]
    rho: f64,
    #[serde(default)]
    repeats: usize,
    #[serde(default)]
    warmup: usize,
    #[serde(default)]
    trace: bool,
}

fn validate_repeats(repeats: usize, warmup: usize) -> Result<(), String> {
    if repeats > 1000 || warmup > 1000 {
        return Err("repeats and warmup must be in 0..1000".into());
    }
    Ok(())
}

fn normalize_synthetic(p: &mut [f64; 3]) -> Result<(), String> {
    if p.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("probabilities must be finite and nonnegative".into());
    }
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || (sum - 1.0).abs() > EPS {
        return Err("probabilities must sum to one within 1e-12".into());
    }
    for x in p {
        *x /= sum;
    }
    Ok(())
}

impl SyntheticRequest {
    fn validate(&mut self) -> Result<(), String> {
        let n = self.probabilities.len();
        if n > 64 || self.loadings.len() != n || self.costs.len() != n {
            return Err("probabilities/loadings/costs must have equal length in 0..64".into());
        }
        for p in &mut self.probabilities {
            normalize_synthetic(p)?;
        }
        if self
            .loadings
            .iter()
            .any(|r| !r.is_finite() || r.abs() >= 1.0)
        {
            return Err("loadings must be finite with absolute value strictly below one".into());
        }
        if self.costs.iter().any(|d| !d.is_finite() || *d < 0.0) {
            return Err("costs must be finite and nonnegative".into());
        }
        if !self.m.is_finite() || !(-1.0..=1.0).contains(&self.m) {
            return Err("m must be finite in [-1,1]".into());
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err("b must be finite in [0,1]".into());
        }
        if self.horizon != 1 && self.horizon != 2 {
            return Err("horizon must be one or two".into());
        }
        validate_repeats(self.repeats, self.warmup)?;
        if !(1..=10000).contains(&self.iterations) {
            return Err("iterations must be in 1..10000".into());
        }
        Ok(())
    }
}

impl GameRequest {
    fn validate(&mut self) -> Result<TicTacToe, String> {
        validate_repeats(self.repeats, self.warmup)?;
        if !self.rho.is_finite() || !(0.0..1.0).contains(&self.rho) {
            return Err("rho must be finite in [0,1)".into());
        }
        self.model.validate()?;
        replay_history(&self.history)
    }
}

#[derive(Clone, Copy, Debug)]
struct AtomLaw {
    p: [f64; 3],
    tilt: [f64; 3],
}
impl AtomLaw {
    // Inputs have already passed protocol validation; zero-mass atoms carry zero tilt.
    fn new(p: [f64; 3], loading: f64) -> Self {
        let mean = 0.5 * p[1] + p[2];
        let scale = (0..3)
            .filter(|&k| p[k] > 0.0)
            .map(|k| (SUPPORT[k] - mean).abs())
            .fold(0.0_f64, f64::max);
        let tilt = std::array::from_fn(|k| {
            if p[k] == 0.0 || scale == 0.0 {
                0.0
            } else {
                loading * ((SUPPORT[k] - mean) / scale)
            }
        });
        Self { p, tilt }
    }
}

struct Belief {
    laws: Vec<AtomLaw>,
    observed: Vec<bool>,
    m: f64,
}
impl Belief {
    fn new(laws: Vec<AtomLaw>, m: f64) -> Self {
        Self {
            observed: vec![false; laws.len()],
            laws,
            m,
        }
    }
    fn observe(&mut self, i: usize, u: f64) -> Result<(), String> {
        if i >= self.laws.len() {
            return Err("observation index out of range".into());
        }
        if self.observed[i] {
            return Err("duplicate observation".into());
        }
        let k = SUPPORT
            .iter()
            .position(|&atom| atom == u)
            .ok_or("utility outside support")?;
        let law = self.laws[i];
        if law.p[k] == 0.0 {
            return Err("unsupported observation".into());
        }
        let denominator = 1.0 + self.m * law.tilt[k];
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("invalid posterior denominator".into());
        }
        // A certain latent state stays fixed. Interior Bayes updates can exceed an
        // endpoint by one rounding unit; projection only removes that arithmetic gap.
        let next = if self.m.abs() == 1.0 {
            self.m
        } else {
            (self.m + law.tilt[k]) / denominator
        };
        if !next.is_finite() || next.abs() > 1.0 + 8.0 * f64::EPSILON {
            return Err("nonfinite or invalid posterior mean".into());
        }
        self.m = next.clamp(-1.0, 1.0);
        self.observed[i] = true;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Method {
    Direct,
    Lifted,
}

// Rows correspond to the initial b and each possible updated incumbent max(b,u).
// Both kernels are O(3 n^2); the normalized comparator divides only once per outcome.
struct Prepared {
    laws: Vec<AtomLaw>,
    alpha: Vec<[f64; 3]>,
    beta: Vec<[f64; 3]>,
}
impl Prepared {
    fn new(laws: Vec<AtomLaw>, costs: &[f64], b: f64) -> Self {
        let thresholds = [b, b.max(0.5), 1.0];
        let mut alpha = Vec::with_capacity(laws.len());
        let mut beta = Vec::with_capacity(laws.len());
        for (law, &cost) in laws.iter().zip(costs) {
            let mut a = [0.0; 3];
            let mut c = [0.0; 3];
            for (row, &threshold) in thresholds.iter().enumerate() {
                for (k, &u) in SUPPORT.iter().enumerate() {
                    let excess = (u - threshold).max(0.0);
                    a[row] += law.p[k] * excess;
                    c[row] += law.p[k] * law.tilt[k] * excess;
                }
                a[row] -= cost;
            }
            alpha.push(a);
            beta.push(c);
        }
        Self { laws, alpha, beta }
    }

    fn scores(&self, m: f64, horizon: u8, method: Method) -> Vec<f64> {
        let n = self.laws.len();
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let mut score = self.alpha[i][0] + m * self.beta[i][0];
            if horizon == 2 {
                for k in 0..3 {
                    let p = self.laws[i].p[k];
                    if p == 0.0 {
                        continue;
                    }
                    let t = self.laws[i].tilt[k];
                    let denominator = 1.0 + m * t;
                    let numerator = m + t;
                    let mut best = 0.0_f64; // STOP is always an available continuation.
                    match method {
                        Method::Direct => {
                            let posterior = numerator / denominator;
                            for j in 0..n {
                                if j != i {
                                    best = best.max(self.alpha[j][k] + posterior * self.beta[j][k]);
                                }
                            }
                            score += p * denominator * best;
                        }
                        Method::Lifted => {
                            for j in 0..n {
                                if j != i {
                                    best = best.max(
                                        denominator * self.alpha[j][k]
                                            + numerator * self.beta[j][k],
                                    );
                                }
                            }
                            score += p * best;
                        }
                    }
                }
            }
            scores.push(score);
        }
        scores
    }
}

// Returns an array position. IDs supply the stated minimum-action tie rule even
// when this helper is called with a non-sorted candidate list.
fn choose(scores: &[f64], ids: &[usize]) -> Option<usize> {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum <= EPS {
        return None;
    }
    (0..scores.len())
        .filter(|&i| scores[i] >= maximum - EPS)
        .min_by_key(|&i| ids[i])
}

#[derive(Serialize)]
struct PlanResult {
    scores: Vec<f64>,
    choice: Option<usize>,
    value: f64,
}
fn synthetic_plan(request: &SyntheticRequest, method: Method) -> PlanResult {
    let laws = request
        .probabilities
        .iter()
        .zip(&request.loadings)
        .map(|(&p, &r)| AtomLaw::new(p, r))
        .collect();
    let prepared = Prepared::new(laws, &request.costs, request.b);
    let scores = prepared.scores(request.m, request.horizon, method);
    let ids: Vec<_> = (0..scores.len()).collect();
    let choice = choose(&scores, &ids);
    // The value uses the raw optimum, not the epsilon-selected action score.
    let value = request.b + scores.iter().copied().fold(0.0_f64, f64::max);
    PlanResult {
        scores,
        choice,
        value,
    }
}

#[derive(Serialize)]
struct Timing {
    repeat: usize,
    direct_ns: u64,
    lifted_ns: u64,
    iterations: usize,
    order: &'static str,
}
fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
fn time_synthetic(request: &SyntheticRequest, method: Method) -> u64 {
    let start = Instant::now();
    for _ in 0..request.iterations {
        black_box(synthetic_plan(black_box(request), method));
    }
    elapsed_ns(start)
}
fn synthetic(request: &SyntheticRequest) -> Value {
    let direct = synthetic_plan(request, Method::Direct);
    let lifted = synthetic_plan(request, Method::Lifted);
    let difference = direct
        .scores
        .iter()
        .zip(&lifted.scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    for pair in 0..request.warmup {
        if pair % 2 == 0 {
            black_box(time_synthetic(request, Method::Direct));
            black_box(time_synthetic(request, Method::Lifted));
        } else {
            black_box(time_synthetic(request, Method::Lifted));
            black_box(time_synthetic(request, Method::Direct));
        }
    }
    let mut timings = Vec::with_capacity(request.repeats);
    for repeat in 0..request.repeats {
        let (direct_ns, lifted_ns, order) = if repeat % 2 == 0 {
            let d = time_synthetic(request, Method::Direct);
            let l = time_synthetic(request, Method::Lifted);
            (d, l, "direct_first")
        } else {
            let l = time_synthetic(request, Method::Lifted);
            let d = time_synthetic(request, Method::Direct);
            (d, l, "lifted_first")
        };
        timings.push(Timing {
            repeat,
            direct_ns,
            lifted_ns,
            iterations: request.iterations,
            order,
        });
    }
    json!({"schema":SCHEMA,"mode":"synthetic","id":request.id,
        "direct":direct,"lifted":lifted,"max_score_abs_difference":difference,"timings":timings})
}

#[derive(Serialize)]
struct Decision {
    b: f64,
    m: f64,
    remaining_moves: Vec<usize>,
    horizon: u8,
    scores: Vec<f64>,
    choice: Option<usize>,
}
#[derive(Serialize)]
struct RunSummary {
    initial_move: usize,
    final_move: usize,
    initial_utility: f64,
    utility: f64,
    initial_nodes: u64,
    additional_nodes: u64,
    cue_nodes: u64,
    charged_nodes: u64,
    additional_queries: usize,
    query_moves: Vec<usize>,
    final_m: f64,
    decisions: Vec<Decision>,
}
fn candidate_law(candidate: &Candidate) -> [f64; 3] {
    if candidate.terminal_known {
        let mut p = [0.0; 3];
        p[(candidate
            .known
            .expect("terminal candidate has an exact value")
            + 1) as usize] = 1.0;
        p
    } else {
        candidate
            .probabilities
            .expect("nonterminal candidate has a frozen model law")
    }
}
fn run_game(
    root: &TicTacToe,
    request: &GameRequest,
    method: Method,
    trace: bool,
) -> Result<RunSummary, String> {
    let mut candidates = candidates(root, request.cue_seed, Some(&request.model));
    let cue_nodes = candidates.iter().map(|c| c.cue_nodes).sum::<u64>();
    // Build prior laws BEFORE the mandatory solve. Its revealed value must condition
    // its original nondegenerate law, not be replaced by a point mass.
    let laws = candidates
        .iter()
        .map(|c| AtomLaw::new(candidate_law(c), request.rho))
        .collect();
    let mut belief = Belief::new(laws, 0.0);
    for (i, candidate) in candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.terminal_known)
    {
        belief.observe(i, utility(candidate.known.unwrap()))?;
    }
    let initial = initialize(&mut candidates, root.current_player());
    if let Some(mv) = initial.solve_move {
        let i = candidates.iter().position(|c| c.mv == mv).unwrap();
        belief.observe(i, utility(candidates[i].known.unwrap()))?;
    }
    let mut incumbent = initial.incumbent;
    let mut best_value = initial.value;
    let mut additional_nodes = 0;
    let mut query_moves = Vec::new();
    let mut decisions = Vec::new();
    for horizon in [2, 1] {
        if best_value == 1 {
            break;
        }
        let remaining: Vec<_> = (0..candidates.len())
            .filter(|&i| candidates[i].known.is_none())
            .collect();
        if remaining.is_empty() {
            break;
        }
        let b = utility(best_value);
        let ids: Vec<_> = remaining.iter().map(|&i| candidates[i].mv).collect();
        let laws = remaining.iter().map(|&i| belief.laws[i]).collect();
        let costs: Vec<_> = remaining
            .iter()
            .map(|&i| LAMBDA * candidates[i].predicted_nodes.unwrap())
            .collect();
        let prepared = Prepared::new(laws, &costs, b);
        let scores = prepared.scores(belief.m, horizon, method);
        let selected = choose(&scores, &ids);
        if trace {
            decisions.push(Decision {
                b,
                m: belief.m,
                remaining_moves: ids.clone(),
                horizon,
                scores,
                choice: selected.map(|i| ids[i]),
            });
        }
        let Some(selected) = selected else {
            break;
        };
        let i = remaining[selected];
        let (v, nodes) = exact(&candidates[i].child, root.current_player());
        additional_nodes += nodes;
        query_moves.push(candidates[i].mv);
        candidates[i].known = Some(v);
        belief.observe(i, utility(v))?;
        if v > best_value {
            // Preserve the exact-known incumbent on an equal reveal.
            best_value = v;
            incumbent = i;
        }
    }
    Ok(RunSummary {
        initial_move: candidates[initial.incumbent].mv,
        final_move: candidates[incumbent].mv,
        initial_utility: utility(initial.value),
        utility: utility(best_value),
        initial_nodes: initial.nodes,
        additional_nodes,
        cue_nodes,
        charged_nodes: cue_nodes + initial.nodes + additional_nodes,
        additional_queries: query_moves.len(),
        query_moves,
        final_m: belief.m,
        decisions,
    })
}
fn time_game(root: &TicTacToe, request: &GameRequest, method: Method) -> Result<u64, String> {
    // Resident boundary: cues/model, initial verification, belief/cache, planning,
    // exact queries and final selection. Parsing/history replay/serialization excluded.
    let start = Instant::now();
    black_box(run_game(
        black_box(root),
        black_box(request),
        method,
        false,
    )?);
    Ok(elapsed_ns(start))
}
fn game(root: &TicTacToe, request: &GameRequest) -> Result<Value, String> {
    // Detailed traces are collected only in these separate, unmeasured executions.
    let direct = run_game(root, request, Method::Direct, request.trace)?;
    let lifted = run_game(root, request, Method::Lifted, request.trace)?;
    for pair in 0..request.warmup {
        if pair % 2 == 0 {
            black_box(time_game(root, request, Method::Direct)?);
            black_box(time_game(root, request, Method::Lifted)?);
        } else {
            black_box(time_game(root, request, Method::Lifted)?);
            black_box(time_game(root, request, Method::Direct)?);
        }
    }
    let mut timings = Vec::with_capacity(request.repeats);
    for repeat in 0..request.repeats {
        let (direct_ns, lifted_ns, order) = if repeat % 2 == 0 {
            let d = time_game(root, request, Method::Direct)?;
            let l = time_game(root, request, Method::Lifted)?;
            (d, l, "direct_first")
        } else {
            let l = time_game(root, request, Method::Lifted)?;
            let d = time_game(root, request, Method::Direct)?;
            (d, l, "lifted_first")
        };
        timings.push(Timing {
            repeat,
            direct_ns,
            lifted_ns,
            iterations: 1,
            order,
        });
    }
    Ok(json!({"schema":SCHEMA,"mode":"game","id":request.id,
        "direct":direct,"lifted":lifted,"timings":timings}))
}

fn handle_line(line: &str) -> Value {
    let result: Result<Value, String> = (|| {
        let request: Request =
            serde_json::from_str(line).map_err(|e| format!("invalid request: {e}"))?;
        match request {
            Request::Synthetic(mut request) => {
                request.validate()?;
                Ok(synthetic(&request))
            }
            Request::Game(mut request) => {
                let root = request.validate()?;
                game(&root, &request)
            }
        }
    })();
    match result {
        Ok(value) => value,
        Err(error) => json!({"schema":SCHEMA,"error":error}),
    }
}
fn serve<R: BufRead, W: Write>(input: R, output: &mut W) -> io::Result<()> {
    for line in input.lines() {
        let response = match line {
            Ok(line) => handle_line(&line),
            Err(error) => json!({"schema":SCHEMA,"error":format!("input read error: {error}")}),
        };
        serde_json::to_writer(&mut *output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    if let Err(error) = serve(stdin.lock(), &mut stdout.lock()) {
        eprintln!("c28 stream error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod c28_tests {
    use super::*;
    const SCORE_TOLERANCE: f64 = 5e-12;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= SCORE_TOLERANCE, "{a:.17e} != {b:.17e}");
    }
    fn fixture(cost_a: f64, loadings: [f64; 3], m: f64) -> SyntheticRequest {
        SyntheticRequest {
            id: "fixture".into(),
            probabilities: vec![[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]],
            loadings: loadings.to_vec(),
            costs: vec![cost_a, 0.3, 0.3],
            m,
            b: 0.5,
            horizon: 2,
            repeats: 0,
            warmup: 0,
            iterations: 1,
        }
    }
    fn synthetic_json() -> Value {
        json!({"mode":"synthetic","probabilities":[[0.5,0.5,0.0],[0.5,0.0,0.5],[0.5,0.0,0.5]],
            "loadings":[0.8,0.8,-0.8],"costs":[0.01,0.3,0.3],"m":0.0,"b":0.5})
    }
    fn model_json() -> Value {
        json!({"schema":"c26-categorical-v1","feature_probs":{},"remaining_probs":{},
            "global_probs":[0.25,0.5,0.25],"remaining_costs":{},"global_cost":1.0,
            "lambda_nodes":LAMBDA,"tau":TAU})
    }
    fn game_json(history: &[usize], trace: bool) -> Value {
        json!({"mode":"game","history":history,"cue_seed":123,"model":model_json(),"trace":trace})
    }

    #[test]
    fn signed_common_product_controls_have_distinct_scores_and_policy() {
        let cases = [
            ([0.8, 0.8, -0.8], [0.1, 0.005, 0.005], 0.6, Some(0)),
            ([0.8; 3], [0.045, -0.05, -0.05], 0.545, Some(0)),
            ([0.0; 3], [-0.01, -0.05, -0.05], 0.5, None),
        ];
        for (loading, expected, value, choice) in cases {
            for method in [Method::Direct, Method::Lifted] {
                let result = synthetic_plan(&fixture(0.01, loading, 0.0), method);
                for (a, b) in result.scores.iter().zip(expected) {
                    close(*a, b);
                }
                close(result.value, value);
                assert_eq!(result.choice, choice);
            }
        }
        for method in [Method::Direct, Method::Lifted] {
            assert_eq!(
                synthetic_plan(&fixture(0.08, [0.8; 3], 0.0), method).choice,
                None
            );
            let signed = synthetic_plan(&fixture(0.08, [0.8, 0.8, -0.8], 0.0), method);
            assert_eq!(signed.choice, Some(0));
            close(signed.value, 0.53);
        }
    }

    #[test]
    fn phase_boundaries_respect_exact_ties_and_nearby_switches() {
        let edge = 19.0 / 24.0;
        for method in [Method::Direct, Method::Lifted] {
            for (m, choice) in [
                (-edge - 1e-8, 2),
                (-edge, 0),
                (-edge + 1e-8, 0),
                (edge - 1e-8, 0),
                (edge, 0),
                (edge + 1e-8, 1),
            ] {
                let result = synthetic_plan(&fixture(0.01, [0.8, 0.8, -0.8], m), method);
                assert_eq!(result.choice, Some(choice), "m={m}");
                if m.abs() == edge {
                    close(result.value, 0.5 + 13.0 / 120.0);
                }
            }
        }
    }

    #[test]
    fn raw_optimum_and_epsilon_choice_are_separate() {
        let request = SyntheticRequest {
            probabilities: vec![[0.0, 0.0, 1.0]; 2],
            loadings: vec![0.0; 2],
            costs: vec![0.5 + 4e-13, 0.5],
            b: 0.0,
            horizon: 1,
            ..fixture(0.0, [0.0; 3], 0.0)
        };
        let result = synthetic_plan(&request, Method::Lifted);
        assert_eq!(result.choice, Some(0));
        assert_eq!(result.value, 0.5);
        assert!(result.value > result.scores[0]);
        assert_eq!(choose(&[5e-13], &[0]), None);
        assert_eq!(choose(&[2e-12, 2e-12], &[8, 3]), Some(1));
    }

    #[test]
    fn centered_features_zero_support_and_certain_latent_states() {
        let deterministic = AtomLaw::new([0.0, 1.0, 0.0], -0.99);
        assert_eq!(deterministic.tilt, [0.0; 3]);
        let law = AtomLaw::new([0.5, 0.5, 0.0], 0.8);
        assert_eq!(law.tilt, [-0.8, 0.8, 0.0]);
        close(law.p.iter().zip(law.tilt).map(|(p, t)| p * t).sum(), 0.0);
        for m in [-1.0, 1.0] {
            let mut belief = Belief::new(vec![law], m);
            belief.observe(0, 0.5).unwrap();
            assert_eq!(belief.m, m);
        }
    }

    #[test]
    fn posterior_rejects_duplicate_unsupported_and_invalid_observations() {
        let mut belief = Belief::new(vec![AtomLaw::new([0.5, 0.5, 0.0], 0.8)], 0.0);
        assert_eq!(
            belief.observe(0, 1.0).unwrap_err(),
            "unsupported observation"
        );
        assert_eq!(
            belief.observe(0, 0.3).unwrap_err(),
            "utility outside support"
        );
        assert_eq!(
            belief.observe(1, 0.5).unwrap_err(),
            "observation index out of range"
        );
        assert_eq!(belief.m, 0.0);
        assert!(!belief.observed[0]);
        belief.observe(0, 0.5).unwrap();
        close(belief.m, 0.8);
        assert_eq!(belief.observe(0, 0.0).unwrap_err(), "duplicate observation");
        close(belief.m, 0.8);
    }

    // Independent finite-world recursion: form the full joint distribution under
    // Z=+/-1, condition by filtering worlds, then evaluate expected terminal utility.
    // This oracle does not use affine coefficients or either planner's formula.
    #[derive(Clone)]
    struct World {
        outcomes: Vec<usize>,
        probability: f64,
    }
    fn worlds(laws: &[AtomLaw], m: f64) -> Vec<World> {
        let mut output = Vec::new();
        for mut index in 0..3usize.pow(laws.len() as u32) {
            let mut outcomes = Vec::new();
            let mut positive = (1.0 + m) * 0.5;
            let mut negative = (1.0 - m) * 0.5;
            for law in laws {
                let k = index % 3;
                index /= 3;
                outcomes.push(k);
                positive *= law.p[k] * (1.0 + law.tilt[k]);
                negative *= law.p[k] * (1.0 - law.tilt[k]);
            }
            let probability = positive + negative;
            if probability > 0.0 {
                output.push(World {
                    outcomes,
                    probability,
                });
            }
        }
        output
    }
    fn world_action(
        worlds: &[World],
        remaining: &[usize],
        b: f64,
        costs: &[f64],
        horizon: u8,
        action: usize,
    ) -> f64 {
        let total: f64 = worlds.iter().map(|w| w.probability).sum();
        let mut value = -costs[action];
        let next: Vec<_> = remaining.iter().copied().filter(|&i| i != action).collect();
        for (k, &u) in SUPPORT.iter().enumerate() {
            let branch: Vec<_> = worlds
                .iter()
                .filter(|w| w.outcomes[action] == k)
                .cloned()
                .collect();
            let mass: f64 = branch.iter().map(|w| w.probability).sum();
            if mass == 0.0 {
                continue;
            }
            let updated = b.max(u);
            let continuation = if horizon <= 1 || next.is_empty() {
                updated
            } else {
                world_value(&branch, &next, updated, costs, horizon - 1)
            };
            value += mass / total * continuation;
        }
        value
    }
    fn world_value(
        worlds: &[World],
        remaining: &[usize],
        b: f64,
        costs: &[f64],
        horizon: u8,
    ) -> f64 {
        remaining
            .iter()
            .map(|&i| world_action(worlds, remaining, b, costs, horizon, i))
            .fold(b, f64::max)
    }

    #[test]
    fn both_kernels_match_independent_world_recursion_on_edges_and_signed_laws() {
        let mut rng = SplitMix64::new(2808);
        let stress = 1.0 - 2.0_f64.powi(-20);
        for case in 0..80 {
            let n = case % 5;
            let mut probabilities = Vec::new();
            let mut loadings = Vec::new();
            let mut costs = Vec::new();
            for i in 0..n {
                let mut p = std::array::from_fn(|_| (rng.next() % 6) as f64);
                if p.iter().sum::<f64>() == 0.0 {
                    p[1] = 1.0;
                }
                let sum: f64 = p.iter().sum();
                for x in &mut p {
                    *x /= sum;
                }
                probabilities.push(p);
                loadings.push([0.0, 0.99, -0.99, stress, -stress][(case + i) % 5]);
                costs.push((rng.next() % 7) as f64 / 10.0);
            }
            let m = [-1.0, -0.8, 0.0, 0.8, 1.0][(case / 5) % 5];
            let b = [0.0, 0.37, 0.5, 0.81, 1.0][(case / 7) % 5];
            let laws: Vec<_> = probabilities
                .iter()
                .zip(&loadings)
                .map(|(&p, &r)| AtomLaw::new(p, r))
                .collect();
            let joint = worlds(&laws, m);
            let ids: Vec<_> = (0..n).collect();
            for horizon in [1, 2] {
                let request = SyntheticRequest {
                    id: String::new(),
                    probabilities: probabilities.clone(),
                    loadings: loadings.clone(),
                    costs: costs.clone(),
                    m,
                    b,
                    horizon,
                    repeats: 0,
                    warmup: 0,
                    iterations: 1,
                };
                for method in [Method::Direct, Method::Lifted] {
                    let result = synthetic_plan(&request, method);
                    for (i, &score) in result.scores.iter().enumerate() {
                        close(score, world_action(&joint, &ids, b, &costs, horizon, i) - b);
                    }
                    close(result.value, world_value(&joint, &ids, b, &costs, horizon));
                }
            }
        }
    }

    #[test]
    fn empty_single_zero_cost_and_saturation_have_literal_values() {
        for method in [Method::Direct, Method::Lifted] {
            let mut request = SyntheticRequest {
                probabilities: vec![],
                loadings: vec![],
                costs: vec![],
                b: 0.37,
                ..fixture(0.0, [0.0; 3], 0.0)
            };
            let empty = synthetic_plan(&request, method);
            assert!(empty.scores.is_empty());
            assert_eq!(empty.choice, None);
            close(empty.value, 0.37);
            request.probabilities = vec![[0.5, 0.0, 0.5]];
            request.loadings = vec![0.0];
            request.costs = vec![0.0];
            request.b = 0.0;
            let single = synthetic_plan(&request, method);
            close(single.scores[0], 0.5);
            close(single.value, 0.5);
            assert_eq!(single.choice, Some(0));
            request.b = 1.0;
            let saturated = synthetic_plan(&request, method);
            assert_eq!(saturated.choice, None);
            close(saturated.value, 1.0);
            close(saturated.scores[0], 0.0);
        }
    }

    #[test]
    fn protocol_rejects_invalid_inputs_and_unknown_fields() {
        let good = synthetic_json();
        for (field, value) in [
            ("m", json!(1.1)),
            ("b", json!(-0.1)),
            ("horizon", json!(3)),
            ("costs", json!([-0.1, 0.3, 0.3])),
            ("loadings", json!([1.0, 0.8, -0.8])),
            (
                "probabilities",
                json!([[0.5, 0.6, 0.0], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]),
            ),
            (
                "probabilities",
                json!([[-0.1, 1.1, 0.0], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]),
            ),
            ("loadings", json!([0.8])),
            ("oracle_values", json!([1.0])),
            ("repeats", json!(1001)),
            ("warmup", json!(1001)),
            ("iterations", json!(0)),
            ("iterations", json!(10001)),
        ] {
            let mut bad = good.clone();
            bad[field] = value;
            assert!(
                handle_line(&bad.to_string()).get("error").is_some(),
                "field={field}"
            );
        }
        for input in [
            "",
            "{",
            "{}",
            "null",
            "[]",
            "{\"mode\":\"bad\"}",
            "{\"mode\":\"synthetic\",\"m\":NaN}",
            "{\"mode\":\"synthetic\",\"m\":1e999}",
        ] {
            assert!(handle_line(input).get("error").is_some(), "input={input}");
        }
        let mut admitted = good.clone();
        admitted["probabilities"][0] = json!([0.5, 0.5 + 2e-13, 0.0]);
        assert!(handle_line(&admitted.to_string()).get("error").is_none());
        let mut too_many = good;
        too_many["probabilities"] = json!(vec![[0.0, 1.0, 0.0]; 65]);
        too_many["loadings"] = json!(vec![0.0; 65]);
        too_many["costs"] = json!(vec![0.0; 65]);
        assert!(handle_line(&too_many.to_string()).get("error").is_some());
    }

    #[test]
    fn stream_recovers_after_bad_line_and_flushes_each_response() {
        struct FlushSink {
            bytes: Vec<u8>,
            flushes: usize,
        }
        impl Write for FlushSink {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushes += 1;
                Ok(())
            }
        }
        let input = format!("{{bad\n{}\n", synthetic_json());
        let mut sink = FlushSink {
            bytes: Vec::new(),
            flushes: 0,
        };
        serve(io::Cursor::new(input), &mut sink).unwrap();
        assert_eq!(sink.flushes, 2);
        let lines: Vec<Value> = String::from_utf8(sink.bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].get("error").is_some());
        assert_eq!(lines[1]["direct"]["choice"], json!(0));
    }

    #[test]
    fn timing_orders_alternate_and_totals_name_iterations() {
        let mut request = synthetic_json();
        request["repeats"] = json!(3);
        request["warmup"] = json!(1);
        request["iterations"] = json!(2);
        let output = handle_line(&request.to_string());
        assert!(output.get("error").is_none(), "{output}");
        let timings = output["timings"].as_array().unwrap();
        assert_eq!(timings.len(), 3);
        for (i, row) in timings.iter().enumerate() {
            assert_eq!(row["repeat"], json!(i));
            assert_eq!(row["iterations"], json!(2));
            assert_eq!(
                row["order"],
                json!(if i % 2 == 0 {
                    "direct_first"
                } else {
                    "lifted_first"
                })
            );
            assert!(row["direct_ns"].as_u64().is_some());
            assert!(row["lifted_ns"].as_u64().is_some());
        }
    }

    #[test]
    fn frozen_game_cues_exact_nodes_and_root_player_sign_remain_literal() {
        let root = replay_history(&[0, 4, 1, 8]).unwrap();
        let candidates = candidates(&root, 123, None);
        let literal = [
            (2, [0, 0, 4], 1, 1.0, 1),
            (3, [0, 0, 4], 14, 1.0, 33),
            (5, [0, 2, 2], 16, 0.5, 47),
            (6, [0, 1, 3], 16, 1.0, 37),
            (7, [2, 0, 2], 14, 0.0, 43),
        ];
        for (candidate, (mv, counts, cues, u, nodes)) in candidates.iter().zip(literal) {
            assert_eq!(
                (candidate.mv, candidate.counts, candidate.cue_nodes),
                (mv, counts, cues)
            );
            let (value, actual) = exact(&candidate.child, root.current_player());
            assert_eq!((utility(value), actual), (u, nodes));
        }
        let odd = replay_history(&[0, 1, 2, 4, 3, 5, 7]).unwrap();
        assert_eq!(odd.current_player(), -1);
        assert_eq!(exact(&odd, -1), (0, 5));
        let terminal = replay_history(&[0, 3, 1, 4]).unwrap().apply_move(2);
        assert_eq!(root_value(&terminal, 1), 1);
        assert_eq!(root_value(&terminal, -1), -1);
    }

    #[test]
    fn terminal_win_skips_initial_solve_and_two_query_plan() {
        let output = handle_line(&game_json(&[0, 4, 1, 8], true).to_string());
        assert!(output.get("error").is_none(), "{output}");
        for method in ["direct", "lifted"] {
            let summary = &output[method];
            assert_eq!(summary["initial_move"], json!(2));
            assert_eq!(summary["final_move"], json!(2));
            assert_eq!(summary["initial_nodes"], json!(0));
            assert_eq!(summary["charged_nodes"], json!(61));
            assert_eq!(summary["additional_queries"], json!(0));
            assert_eq!(summary["utility"], json!(1.0));
            assert_eq!(summary["decisions"], json!([]));
        }
    }

    #[test]
    fn initial_reveal_conditions_prior_and_equal_queries_retain_incumbent() {
        let mut request: GameRequest =
            match serde_json::from_value::<Request>(game_json(&[0, 1, 2, 4, 3, 5, 7], true))
                .unwrap()
            {
                Request::Game(request) => request,
                _ => unreachable!(),
            };
        request.model.global_probs = [0.6, 0.3, 0.1];
        let root = request.validate().unwrap();
        let c = candidates(&root, request.cue_seed, Some(&request.model));
        let laws: Vec<_> = c
            .iter()
            .map(|candidate| AtomLaw::new(candidate_law(candidate), request.rho))
            .collect();
        let mut initialized = candidates(&root, request.cue_seed, Some(&request.model));
        let initial = initialize(&mut initialized, root.current_player());
        let mut expected = Belief::new(laws, 0.0);
        for (i, candidate) in initialized.iter().enumerate() {
            if let Some(value) = candidate.known {
                expected.observe(i, utility(value)).unwrap();
            }
        }
        assert!(
            expected.m.abs() > 0.1,
            "initial reveal must change this prior"
        );
        for method in [Method::Direct, Method::Lifted] {
            let summary = run_game(&root, &request, method, true).unwrap();
            let decision = summary
                .decisions
                .first()
                .expect("remaining candidate requires planning");
            close(decision.m, expected.m);
            assert_eq!(summary.initial_move, initialized[initial.incumbent].mv);
            assert!(summary.additional_queries <= 2);
            if summary.utility == summary.initial_utility {
                assert_eq!(summary.final_move, summary.initial_move);
            }
            assert_eq!(
                summary.charged_nodes,
                summary.cue_nodes + summary.initial_nodes + summary.additional_nodes
            );
        }
    }

    #[test]
    fn game_paths_agree_and_queries_are_unique_with_h2_then_h1() {
        for history in [
            vec![0, 4, 8],
            vec![0, 1, 4],
            vec![0, 1, 2, 4, 3],
            vec![0, 1, 2, 4, 3, 5, 7],
        ] {
            let mut request = game_json(&history, true);
            request["repeats"] = json!(2);
            request["warmup"] = json!(1);
            let output = handle_line(&request.to_string());
            assert!(output.get("error").is_none(), "{output}");
            for key in [
                "initial_move",
                "final_move",
                "initial_utility",
                "utility",
                "initial_nodes",
                "additional_nodes",
                "charged_nodes",
                "additional_queries",
                "query_moves",
                "final_m",
            ] {
                assert_eq!(output["direct"][key], output["lifted"][key], "{key}");
            }
            let direct = &output["direct"];
            let queries = direct["query_moves"].as_array().unwrap();
            assert!(queries.len() <= 2);
            if queries.len() == 2 {
                assert_ne!(queries[0], queries[1]);
            }
            let ds = direct["decisions"].as_array().unwrap();
            let ls = output["lifted"]["decisions"].as_array().unwrap();
            assert_eq!(ds.len(), ls.len());
            for (i, (d, l)) in ds.iter().zip(ls).enumerate() {
                assert_eq!(d["horizon"], json!(2 - i));
                assert_eq!(d["choice"], l["choice"]);
                assert_eq!(d["remaining_moves"], l["remaining_moves"]);
                for (a, b) in d["scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(l["scores"].as_array().unwrap())
                {
                    close(a.as_f64().unwrap(), b.as_f64().unwrap());
                }
            }
            assert_eq!(output["timings"][0]["order"], json!("direct_first"));
            assert_eq!(output["timings"][1]["order"], json!("lifted_first"));
            request["trace"] = json!(false);
            request["repeats"] = json!(0);
            request["warmup"] = json!(0);
            let quiet = handle_line(&request.to_string());
            assert_eq!(quiet["direct"]["decisions"], json!([]));
        }
    }

    #[test]
    fn game_model_history_and_oracle_field_errors_are_explicit() {
        for history in [
            vec![9],
            vec![0, 0],
            vec![0, 3, 1, 4, 2],
            vec![0, 3, 1, 4, 2, 5],
        ] {
            assert!(handle_line(&game_json(&history, false).to_string())
                .get("error")
                .is_some());
        }
        for (field, value) in [
            ("oracle_values", json!([1])),
            ("optimal_move", json!(4)),
            ("rho", json!(-0.1)),
            ("rho", json!(1.0)),
        ] {
            let mut bad = game_json(&[0, 4, 8], false);
            bad[field] = value;
            assert!(handle_line(&bad.to_string()).get("error").is_some());
        }
        for (field, value) in [
            ("global_probs", json!([0.0, 0.5, 0.5])),
            ("tau", json!(1.0)),
            ("lambda_nodes", json!(1.0)),
            ("global_cost", json!(0.5)),
            ("oracle", json!(true)),
        ] {
            let mut bad = game_json(&[0, 4, 8], false);
            bad["model"][field] = value;
            assert!(handle_line(&bad.to_string()).get("error").is_some());
        }
    }
}
