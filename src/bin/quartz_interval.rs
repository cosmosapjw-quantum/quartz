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

// C29 noisy-query continuation. Prior C26/C28 source is copied as a pinned
// read-only prefix; this file does not modify the protected game engine.
fn normalize_law(p: &mut [f64;3]) -> Result<(),String> {
    if p.iter().any(|v| !v.is_finite() || *v < 0.0) {return Err("invalid probability".into());}
    let s:f64=p.iter().sum();
    if !s.is_finite() || (s-1.0).abs()>1e-12 {return Err("probability normalization".into());}
    for v in p { *v/=s; }
    Ok(())
}

fn validate_channels(ch: &[Vec<Vec<f64>>], n:usize) -> Result<(),String>{
    if ch.len()!=n {return Err("channel candidate count".into());}
    for c in ch {
        if c.len()!=3 {return Err("channel utility rows".into());}
        let k=c[0].len();
        if k==0 || k>3 {return Err("channel observation count".into());}
        for row in c {
            if row.len()!=k || row.iter().any(|v|!v.is_finite()||*v<0.0) {return Err("invalid channel row".into());}
            if (row.iter().sum::<f64>()-1.0).abs()>1e-12 {return Err("channel normalization".into());}
        }
    }
    Ok(())
}

#[derive(Clone)]
struct Filter {
    q: [f64;2],                    // Z = (-1,+1)
    local: Vec<[[f64;3];2]>,       // P(U_i | Z=z, history)
    known: Vec<Option<usize>>,    // only exact evidence; not noisy Y
}
#[derive(Clone)]
struct Plan {
    value: f64,
    stop: f64,
    scores: Vec<Option<f64>>,
    choice: Option<usize>,
    guard_stops: usize,
    root_evpi_stop: bool,
}
impl Filter {
    fn new(mut p:Vec<[f64;3]>, r:Vec<f64>, m:f64)->Result<Self,String>{
        if p.len()>64 || p.len()!=r.len() || !m.is_finite() || m.abs()>1.0 {
            return Err("invalid prior dimensions or latent mean".into());
        }
        let mut local=Vec::new();
        for (prob,&loading) in p.iter_mut().zip(r.iter()){
            normalize_law(prob)?;
            if !loading.is_finite() || loading.abs()>=1.0 {return Err("invalid loading".into());}
            let mu=(0..3).map(|k|prob[k]*SUPPORT[k]).sum::<f64>();
            let spread=(0..3).filter(|&k|prob[k]>0.0).map(|k|(SUPPORT[k]-mu).abs()).fold(0f64,f64::max);
            let mut factors=[[0f64;3];2];
            for zi in 0..2 {
                let z=if zi==0 {-1.0} else {1.0};
                for k in 0..3 {
                    let f=if spread>0.0 && prob[k]>0.0 {(SUPPORT[k]-mu)/spread} else {0.0};
                    factors[zi][k]=prob[k]*(1.0+z*loading*f);
                }
                let s:f64=factors[zi].iter().sum();
                if s<=0.0 || !s.is_finite() {return Err("invalid component law".into());}
                for x in &mut factors[zi] { *x/=s; }
            }
            local.push(factors);
        }
        let n=local.len();
        Ok(Self{q:[(1.0-m)/2.0,(1.0+m)/2.0],local,known:vec![None;n]})
    }
    fn means(&self)->Vec<f64>{
        self.local.iter().map(|f| (0..2).map(|z|
            self.q[z]*(0..3).map(|k|SUPPORT[k]*f[z][k]).sum::<f64>()
        ).sum::<f64>()).collect()
    }
    fn terminal(&self,b:f64,known_only:bool)->f64{
        if known_only {
            self.known.iter().flatten().map(|&k|SUPPORT[k]).fold(b,f64::max)
        } else {self.means().into_iter().fold(b,f64::max)}
    }
    fn observe(&self,i:usize,y:usize,ch:&[Vec<Vec<f64>>],known_only:bool)->Option<(Self,f64)>{
        let mut a=[0.0;2];
        for z in 0..2 {for k in 0..3 {a[z]+=self.local[i][z][k]*ch[i][k][y];}}
        let d=self.q[0]*a[0]+self.q[1]*a[1];
        if d==0.0 {return None;}
        if !d.is_finite() || d<0.0 {return None;}
        let mut next=self.clone();
        for z in 0..2 {
            next.q[z]=self.q[z]*a[z]/d;
            if a[z]>0.0 {
                for k in 0..3 {next.local[i][z][k]=self.local[i][z][k]*ch[i][k][y]/a[z];}
            }
        }
        let s=next.q[0]+next.q[1];
        if s<=0.0 || !s.is_finite() {return None;}
        for v in &mut next.q { *v/=s; }
        if known_only {next.known[i]=Some(y);}
        Some((next,d))
    }
    fn perfect_information(&self, b: f64) -> f64 {
        let mut at_zero = [1.0; 2];
        let mut at_half = [1.0; 2];
        for candidate in &self.local {
            for z in 0..2 {
                at_zero[z] *= candidate[z][0];
                at_half[z] *= candidate[z][0] + candidate[z][1];
            }
        }
        let mix = |p: [f64; 2]| self.q[0] * p[0] + self.q[1] * p[1];
        1.0 - (0.5 - b).max(0.0) * mix(at_zero) - (1.0 - b.max(0.5)) * mix(at_half)
    }

    // The represented nonnegative masses are normalized as real numbers.
    // Each operation below encloses one real primitive in binary64.
    fn strict_evpi_stop(&self, b: f64, min_cost: f64, known_only: bool) -> bool {
        if !min_cost.is_finite() || min_cost <= 0.0 {
            return false;
        }
        let down = |x: f64| x.next_down().max(0.0);
        if (b + min_cost).next_down() > 1.0 {
            return true;
        }
        let q_sum = (self.q[0] + self.q[1]).next_up();
        let q = self.q.map(|x| down(x / q_sum));
        let mut f0 = [1.0; 2];
        let mut fh = [1.0; 2];
        let mut lower = b;
        for (i, candidate) in self.local.iter().enumerate() {
            let mut means = [0.0; 2];
            for z in 0..2 {
                let [p0, ph, p1] = candidate[z];
                let den = ((p0 + ph).next_up() + p1).next_up();
                f0[z] = down(f0[z] * down(p0 / den));
                fh[z] = down(fh[z] * down(down(p0 + ph) / den));
                means[z] = down(q[z] * down(down(down(0.5 * ph) + p1) / den));
            }
            if known_only {
                if let Some(k) = self.known[i] {
                    lower = lower.max(SUPPORT[k]);
                }
            } else {
                lower = lower.max(down(means[0] + means[1]));
            }
        }
        let mix_lo = |f: [f64; 2]| down(down(q[0] * f[0]) + down(q[1] * f[1]));
        let loss0 = down(down(0.5 - b) * mix_lo(f0));
        let lossh = down(down(1.0 - b.max(0.5)) * mix_lo(fh));
        let upper = (1.0 - down(loss0 + lossh)).next_up().min(1.0);
        upper < (lower + min_cost).next_down()
    }

    fn plan(&self,ch:&[Vec<Vec<f64>>],costs:&[f64],h:usize,b:f64,
        known_only:bool,disabled:&[bool])->Plan{
        self.plan_with_guard(ch, costs, h, b, known_only, disabled, false)
    }

    fn plan_with_guard(&self,ch:&[Vec<Vec<f64>>],costs:&[f64],h:usize,b:f64,
        known_only:bool,disabled:&[bool],guard:bool)->Plan{
        let stop=self.terminal(b,known_only);
        let mut scores=vec![None;self.local.len()];
        let mut guard_stops = 0;
        if h>0 {
            if guard {
                let min_cost = costs.iter().enumerate()
                    .filter(|&(i, _)| !disabled[i] && !(known_only && self.known[i].is_some()))
                    .map(|(_, &cost)| cost).fold(f64::INFINITY, f64::min);
                if self.strict_evpi_stop(b, min_cost, known_only) {
                    return Plan { value: stop, stop, scores, choice: None, guard_stops: 1, root_evpi_stop: true };
                }
            }
            for i in 0..self.local.len(){
                if disabled[i] || (known_only && self.known[i].is_some()) {continue;}
                let mut score=-costs[i];
                for y in 0..ch[i][0].len(){
                    if let Some((after,py))=self.observe(i,y,ch,known_only) {
                        let child = after.plan_with_guard(ch,costs,h-1,b,known_only,disabled,guard);
                        score += py * child.value;
                        guard_stops += child.guard_stops;
                    }
                }
                scores[i]=Some(score);
            }
        }
        let value=scores.iter().flatten().copied().fold(stop,f64::max);
        let choice=if value<=stop+EPS {None} else {
            (0..scores.len()).find(|&i|scores[i].is_some_and(|q|q>=value-EPS))
        };
        Plan{value,stop,scores,choice,guard_stops,root_evpi_stop:false}
    }
}
fn is_identity(ch:&[Vec<Vec<f64>>])->bool{
    ch.iter().all(|c|c.iter().enumerate().all(|(k,row)|
        row.len()==3 && row.iter().enumerate().all(|(y,v)|*v==if y==k{1.0}else{0.0})))
}
fn finite_costs(costs:&[f64])->bool{
    costs.iter().all(|c|c.is_finite()&&*c>=0.0)
}
#[derive(Deserialize)]
#[serde(tag="mode",rename_all="snake_case")]
enum Request {Synthetic(SyntheticRequest),Game(GameRequest)}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticRequest{
    #[serde(default)] id:String,
    probabilities:Vec<[f64;3]>,
    loadings:Vec<f64>,
    channels:Vec<Vec<Vec<f64>>>,
    costs:Vec<f64>,
    m:f64,
    b:f64,
    #[serde(default="default_horizon_c29")] horizon:usize,
    #[serde(default)] known_only:bool,
    #[serde(default)] guard:bool,
}
fn default_horizon_c29()->usize{2}
impl SyntheticRequest{
    fn validate(&self)->Result<(),String>{
        let n=self.probabilities.len();
        if n>64||self.loadings.len()!=n||self.costs.len()!=n||self.horizon>2||
            !self.b.is_finite()||!(0.0..=1.0).contains(&self.b)||!finite_costs(&self.costs) {
            return Err("synthetic dimensions, costs, horizon or incumbent".into());
        }
        validate_channels(&self.channels,n)?;
        if self.known_only && !is_identity(&self.channels) {return Err("known_only requires exact identity channel".into());}
        Ok(())
    }
}
fn synthetic(req:SyntheticRequest)->Result<Value,String>{
    req.validate()?;
    let f=Filter::new(req.probabilities,req.loadings,req.m)?;
    let disabled = vec![false; f.local.len()];
    let plan = if req.guard {
        f.plan_with_guard(&req.channels,&req.costs,req.horizon,req.b,req.known_only,&disabled,true)
    } else {
        f.plan(&req.channels,&req.costs,req.horizon,req.b,req.known_only,&disabled)
    };
    Ok(json!({"schema":"c29-noisy-v1","mode":"synthetic","id":req.id,
        "value":plan.value,"stop":plan.stop,"scores":plan.scores,"choice":plan.choice,
        "posterior_means":f.means(),"q":f.q,"guard_stops":plan.guard_stops,
        "root_evpi_stop":plan.root_evpi_stop,
        "perfect_information":f.perfect_information(req.b)}))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GameRequest{
    #[serde(default)] id:String,
    history:Vec<usize>,
    cue_seed:u64,
    model:Model,
    channel:[[f64;3];3],
    #[serde(default="default_rho_c29")] rho:f64,
    #[serde(default="default_horizon_c29")] horizon:usize,
    #[serde(default)] repeats:usize,
    #[serde(default)] guard:bool,
}
fn default_rho_c29()->f64{0.75}
impl GameRequest{
    fn validate(&mut self)->Result<TicTacToe,String>{
        self.model.validate()?;
        if self.horizon>2||self.repeats>31||!self.rho.is_finite()||self.rho.abs()>=1.0 {
            return Err("game horizon/repeats/loading invalid".into());
        }
        for row in &self.channel {
            if row.iter().any(|x|!x.is_finite()||*x<0.0) ||
                (row.iter().sum::<f64>()-1.0).abs()>1e-12 {return Err("game channel invalid".into());}
        }
        replay_history(&self.history)
    }
}
#[derive(Clone,Serialize,PartialEq)]
struct ObservationTrace{
    move_id:usize, y:usize, nodes:u64, q_plus:f64,
    expected_before:f64, expected_after:f64, planned_query_cost:f64,
}
#[derive(Clone)]
struct GameRun {
    initial_move:usize,
    baseline_move:usize,
    final_move:usize,
    initial_nodes:u64,
    cue_nodes:u64,
    query_nodes:u64,
    predicted_query_cost:f64,
    observations:Vec<ObservationTrace>,
    baseline_expected:f64,
    final_expected:f64,
    planned_value:f64,
    guard_stops:usize,
}
fn best_mean_index(f:&Filter,cs:&[Candidate])->usize{
    let means=f.means();
    (0..cs.len()).max_by(|&i,&j|
        means[i].partial_cmp(&means[j]).unwrap().then_with(||cs[j].mv.cmp(&cs[i].mv))
    ).expect("nonterminal root has candidate")
}
fn run_game(root:&TicTacToe, req:&GameRequest, noisy:bool, guard:bool)->Result<GameRun,String>{
    let mut cs=candidates(root,req.cue_seed,Some(&req.model));
    let cue_nodes=cs.iter().map(|c|c.cue_nodes).sum();
    let n=cs.len();
    // Fix the prior *before* mandatory initial exact verification, as in C28.
    // Replacing a newly verified candidate by a point mass before applying Bayes
    // would discard its information about the shared signed latent mode.
    let mut probs=Vec::with_capacity(n);let mut loads=Vec::with_capacity(n);
    for candidate in &cs {
        if candidate.terminal_known {
            let mut p=[0.0;3];p[(candidate.known.ok_or("terminal exact value missing")?+1) as usize]=1.0;
            probs.push(p);loads.push(0.0);
        } else {
            probs.push(candidate.probabilities.ok_or("missing model distribution")?);
            loads.push(req.rho);
        }
    }
    let mut filter=Filter::new(probs,loads,0.0)?;
    let exact_channel=vec![(0..3).map(|u|
        (0..3).map(|y|if u==y{1.0}else{0.0}).collect::<Vec<_>>()
    ).collect::<Vec<_>>();n];
    for (i,c) in cs.iter().enumerate().filter(|(_,c)|c.terminal_known) {
        let k=(c.known.ok_or("terminal value missing")?+1) as usize;
        filter=filter.observe(i,k,&exact_channel,true)
            .ok_or("initial terminal observation unsupported")?.0;
    }
    let initial=initialize(&mut cs,root.current_player());
    if let Some(mv)=initial.solve_move {
        let i=cs.iter().position(|c|c.mv==mv).ok_or("initial solve move missing")?;
        let k=(cs[i].known.ok_or("initial solved value missing")?+1) as usize;
        filter=filter.observe(i,k,&exact_channel,true)
            .ok_or("initial verification observation unsupported")?.0;
    }
    let disabled:Vec<bool>=cs.iter().map(|c|c.known.is_some()).collect();
    let costs:Vec<f64>=cs.iter().map(|c|if c.known.is_some(){0.0}
        else{LAMBDA*(c.cue_nodes as f64)/4.0}).collect();
    let channels=vec![req.channel.iter().map(|row|row.to_vec()).collect::<Vec<_>>();n];
    let baseline_idx=best_mean_index(&filter,&cs);
    let baseline_expected=filter.means()[baseline_idx];
    let mut initial_plan = noisy.then(|| {
        filter.plan_with_guard(&channels,&costs,req.horizon,0.0,false,&disabled,guard)
    });
    let planned_value = initial_plan.as_ref().map_or(baseline_expected, |p| p.value);
    let mut guard_stops = 0;
    let mut query_nodes=0;let mut predicted_query_cost=0.0;let mut observations=Vec::new();
    if noisy {
        for round in 0..req.horizon {
            let plan = initial_plan.take().unwrap_or_else(|| {
                filter.plan_with_guard(&channels,&costs,req.horizon-round,0.0,false,&disabled,guard)
            });
            guard_stops += plan.guard_stops;
            let Some(i)=plan.choice else {break;};
            let expected_before=filter.means()[i];
            let seed=req.cue_seed
                ^ (cs[i].mv as u64+1).wrapping_mul(GAMMA)
                ^ (round as u64+101).wrapping_mul(0xbf58476d1ce4e5b9);
            let mut rng=SplitMix64::new(seed);
            let (observed,entered)=rollout(&cs[i].child,root.current_player(),&mut rng);
            let y=(observed+1) as usize;
            let (next,_)=filter.observe(i,y,&channels,false).ok_or("observed signal has zero model likelihood")?;
            filter=next;
            query_nodes+=entered;
            predicted_query_cost+=costs[i];
            observations.push(ObservationTrace{move_id:cs[i].mv,y,nodes:entered,
                q_plus:filter.q[1], expected_before, expected_after:filter.means()[i],
                planned_query_cost:costs[i]});
        }
    }
    let final_idx=best_mean_index(&filter,&cs);
    Ok(GameRun {
        initial_move:cs[initial.incumbent].mv,baseline_move:cs[baseline_idx].mv,
        final_move:cs[final_idx].mv,initial_nodes:initial.nodes,cue_nodes,
        query_nodes,predicted_query_cost,observations,baseline_expected,
        final_expected:filter.means()[final_idx],planned_value,guard_stops
    })
}
fn offline_utility(root:&TicTacToe,mv:usize)->f64{
    utility(exact(&root.apply_move(mv),root.current_player()).0)
}
fn game(mut req:GameRequest)->Result<Value,String>{
    let root=req.validate()?;
    let baseline=run_game(&root,&req,false,false)?;
    let actual=run_game(&root,&req,true,req.guard)?;
    let equivalent = if req.guard {
        let full = run_game(&root, &req, true, false)?;
        Some(actual.final_move == full.final_move
            && actual.observations == full.observations
            && actual.planned_value == full.planned_value
            && actual.query_nodes == full.query_nodes)
    } else {
        None
    };
    let mut baseline_ns=Vec::new();let mut query_ns=Vec::new();
    let mut full_h2_ns=Vec::new();
    for k in 0..req.repeats {
        let measure=|noisy, guard| {
            let start=Instant::now();
            let r=run_game(&root,&req,noisy,guard).expect("validated run must not fail");
            let elapsed=start.elapsed().as_nanos() as u64;
            black_box(r);
            elapsed
        };
        if k % 2 == 0 {
            baseline_ns.push(measure(false, false));
            if req.guard { full_h2_ns.push(measure(true, false)); }
            query_ns.push(measure(true, req.guard));
        } else {
            query_ns.push(measure(true, req.guard));
            if req.guard { full_h2_ns.push(measure(true, false)); }
            baseline_ns.push(measure(false, false));
        }
    }
    // The following minimax evaluation occurs AFTER both chosen policies returned;
    // it is never supplied to run_game or its posterior and is outside timing.
    let utility_baseline=offline_utility(&root,baseline.baseline_move);
    let utility_query=offline_utility(&root,actual.final_move);
    Ok(json!({"schema":"c29-noisy-v1","mode":"game","id":req.id,
        "baseline_move":baseline.baseline_move,"noisy_move":actual.final_move,
        "initial_known_move":actual.initial_move,
        "observations":actual.observations,"query_count":actual.observations.len(),
        "baseline_expected":baseline.baseline_expected,"noisy_expected":actual.final_expected,
        "planned_value":actual.planned_value,
        "cue_nodes":actual.cue_nodes,"initial_nodes":actual.initial_nodes,
        "additional_rollout_nodes":actual.query_nodes,
        "baseline_charged_nodes":actual.cue_nodes+actual.initial_nodes,
        "noisy_charged_nodes":actual.cue_nodes+actual.initial_nodes+actual.query_nodes,
        "model_expected_query_cost":actual.predicted_query_cost,
        "actual_query_cost":actual.query_nodes as f64*LAMBDA,
        "offline_realized_baseline":utility_baseline,
        "offline_realized_noisy":utility_query,
        "timing_baseline_ns":baseline_ns,"timing_noisy_ns":query_ns,
        "timing_full_h2_ns":full_h2_ns,"guard_stops":actual.guard_stops,
        "equivalent_full_h2":equivalent,
        "offline_oracle_not_runtime_input":true}))
}
fn handle_line(line:&str)->Value{
    let result=serde_json::from_str::<Request>(line);
    match result {
        Ok(Request::Synthetic(s))=>synthetic(s).unwrap_or_else(|e|json!({"error":e})),
        Ok(Request::Game(g))=>game(g).unwrap_or_else(|e|json!({"error":e})),
        Err(e)=>json!({"error":e.to_string()})
    }
}
fn serve<R:BufRead,W:Write>(input:R,output:&mut W)->io::Result<()>{
    for line in input.lines(){
        let line=line?;
        let value=handle_line(&line);
        serde_json::to_writer(&mut *output,&value)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
fn main(){
    let stdin=io::stdin();let stdout=io::stdout();
    let mut out=io::BufWriter::new(stdout.lock());
    if let Err(e)=serve(stdin.lock(),&mut out){eprintln!("C29 IO: {e}");std::process::exit(1);}
}
#[cfg(test)]
mod tests{
    use super::*;
    fn channel()->Vec<Vec<Vec<f64>>>{
        vec![vec![vec![0.75,0.25],vec![0.5,0.5],vec![0.25,0.75]]]
    }
    #[test]
    fn noisy_repeat_nonmyopic(){
        let f=Filter::new(vec![[0.5,0.0,0.5]],vec![0.0],0.0).unwrap();
        let a=f.plan(&channel(),&[0.01],1,0.6,false,&[false]);
        let b=f.plan(&channel(),&[0.01],2,0.6,false,&[false]);
        assert!((a.value-0.665).abs()<1e-13);
        assert!((b.value-0.67875).abs()<1e-13);
        assert_eq!(b.choice,Some(0));
    }
    #[test]
    fn independent_scalar_not_sufficient(){
        let f=Filter::new(vec![[0.5,0.0,0.5]],vec![0.0],0.0).unwrap();
        let (g,_)=f.observe(0,1,&channel(),false).unwrap();
        assert_eq!(g.q,f.q);
        assert!((g.means()[0]-0.75).abs()<1e-13);
    }
    #[test]
    fn uninformative_positive_cost_stop(){
        let f=Filter::new(vec![[0.5,0.0,0.5]],vec![0.0],0.0).unwrap();
        let ch=vec![vec![vec![0.5,0.5];3]];
        let p=f.plan(&ch,&[0.01],2,0.6,false,&[false]);
        assert_eq!(p.choice,None);
        assert!((p.value-0.6).abs()<1e-13);
    }
    #[test]
    fn evpi_does_not_count_empty_menu() {
        let f = Filter::new(vec![[0.5, 0.0, 0.5]], vec![0.0], 0.0).unwrap();
        let p = f.plan_with_guard(&channel(), &[0.5], 2, 0.6, false, &[true], true);
        assert_eq!(p.guard_stops, 0);
        assert!(!p.root_evpi_stop);
    }

    #[test]
    fn costly_supported_query_is_skipped() {
        let f = Filter::new(vec![[0.5, 0.0, 0.5]], vec![0.0], 0.0).unwrap();
        let p = f.plan_with_guard(&channel(), &[0.3], 2, 0.6, false, &[false], true);
        assert_eq!(f.perfect_information(0.6), 0.8);
        assert_eq!(p.guard_stops, 1);
        assert!(p.root_evpi_stop);
        assert_eq!(p.value, 0.6);
    }

    #[test]
    fn bad_channel_rejected(){
        assert!(validate_channels(&[vec![vec![0.2,0.2];3]],1).is_err());
    }
}
