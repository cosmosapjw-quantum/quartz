//! Isolated finite-horizon channel heterogeneity experiment.
#![allow(dead_code)]
#[path = "../game.rs"]
mod game;
#[path = "../games/tictactoe.rs"]
mod tictactoe;
use game::GameState;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::time::Instant;
use tictactoe::TicTacToe;
const GAMMA: u64 = 0x9e3779b97f4a7c15;
const LAMBDA: f64 = 1.0 / 1024.0;
const TOL: f64 = 1e-12;
// Following helper block copied unchanged from SSOT C26 exact, root_value, rollout, SplitMix64.
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    schema: String,
    costs: Vec<f64>,
    table: Vec<Vec<Vec<Vec<[f64; 4]>>>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    history: Vec<usize>,
    cue_seed: u64,
    repeats: usize,
}
#[derive(Clone, Copy)]
struct State {
    known: Option<f64>,
    rem: usize,
    cue: usize,
    h: [usize; 3],
}
fn hidx(h: [usize; 3]) -> usize {
    let n = h.iter().sum::<usize>();
    (0..n).map(|k| (k + 1) * (k + 2) / 2).sum::<usize>() + cidx(h)
}
fn cidx(h: [usize; 3]) -> usize {
    let n = h.iter().sum::<usize>();
    (0..h[0]).map(|i| n - i + 1).sum::<usize>() + h[1]
}
impl Model {
    fn row(&self, s: State, arm: usize) -> [f64; 4] {
        if let Some(v) = s.known {
            return [v, 0., 0., 0.];
        }
        self.table[s.rem][arm][s.cue][hidx(s.h)]
    }
    fn validate(&self) -> Result<(), String> {
        if self.schema != "quartz-channel-table-v1"
            || self.costs.len() != 9
            || self.table.len() != 9
        {
            return Err("shape".into());
        }
        if self
            .costs
            .iter()
            .any(|c| !c.is_finite() || *c < 1. || *c > 9.)
        {
            return Err("cost".into());
        }
        for depth in &self.table {
            if depth.len() != 3 {
                return Err("arms".into());
            }
            for arm in depth {
                if arm.len() != 15 {
                    return Err("cues".into());
                }
                for cue in arm {
                    if cue.len() != 10 {
                        return Err("history".into());
                    }
                    for (i, r) in cue.iter().enumerate() {
                        if r.iter().any(|x| !x.is_finite() || *x < 0. || *x > 1.) {
                            return Err("invalid row".into());
                        }
                        if i < 4 && (r[1..].iter().sum::<f64>() - 1.).abs() > 1e-12 {
                            return Err("normalization".into());
                        }
                        if i >= 4 && r[1..] != [0., 0., 0.] {
                            return Err("last row predictive sentinel".into());
                        }
                    }
                }
            }
            // Explicit same first prior, predictive, and post-first utility for F/A.
            for c in 0..15 {
                for i in 0..4 {
                    if (depth[1][c][i][0] - depth[2][c][i][0]).abs() > TOL {
                        return Err("F/A first posterior mismatch".into());
                    }
                }
                for a in 0..3 {
                    if (depth[1][c][0][a + 1] - depth[2][c][0][a + 1]).abs() > TOL {
                        return Err("F/A first predictive mismatch".into());
                    }
                }
                if (depth[0][c][0][0] - depth[2][c][0][0]).abs() > TOL {
                    return Err("common cue posterior mismatch".into());
                }
            }
        }
        Ok(())
    }
}
fn best(s: &[State], m: &Model, arm: usize, known_only: bool) -> (f64, usize) {
    let mut out = (-1., 0);
    for (i, x) in s.iter().enumerate() {
        if known_only && x.known.is_none() {
            continue;
        }
        let v = m.row(*x, arm)[0];
        if v > out.0 + TOL
            || ((v - out.0).abs() <= TOL && x.known.is_some() && s[out.1].known.is_none())
        {
            out = (v, i)
        }
    }
    out
}
fn plan(s: &mut [State], m: &Model, arm: usize, h: usize) -> (f64, Option<usize>) {
    let mut out = (best(s, m, arm, false).0, None);
    if h == 0 || out.0 >= 1. - TOL {
        return out;
    }
    for i in 0..s.len() {
        if s[i].known.is_some() {
            continue;
        }
        let r = m.row(s[i], arm);
        let mut q = -LAMBDA * m.costs[s[i].rem];
        for y in 0..3 {
            if r[y + 1] == 0. {
                continue;
            }
            s[i].h[y] += 1;
            q += r[y + 1] * plan(s, m, arm, h - 1).0;
            s[i].h[y] -= 1;
        }
        if q > out.0 + TOL {
            out = (q, Some(i))
        }
    }
    out
}
struct Candidate {
    mv: usize,
    child: TicTacToe,
    rng: SplitMix64,
    cue: [usize; 3],
}
fn one(root: &TicTacToe, m: &Model, seed: u64, policy: usize, horizon: usize) -> Value {
    let t = Instant::now();
    let player = root.current_player();
    let mut ms = root.legal_moves();
    ms.sort_unstable();
    let mut ss = Vec::new();
    let mut cs = Vec::new();
    let mut cue_nodes = 0u64;
    for mv in ms {
        let child = root.apply_move(mv);
        let mut rng = SplitMix64::new(seed ^ (mv as u64 + 1).wrapping_mul(GAMMA));
        let mut cue = [0usize; 3];
        let known = if child.is_terminal() {
            let v = root_value(&child, player);
            cue[(v + 1) as usize] = 4;
            cue_nodes += 1;
            Some(utility(v))
        } else {
            for _ in 0..4 {
                let (v, n) = rollout(&child, player, &mut rng);
                cue[(v + 1) as usize] += 1;
                cue_nodes += n
            }
            None
        };
        let rem = child.legal_moves().len();
        ss.push(State {
            known,
            rem,
            cue: cidx(cue),
            h: [0; 3],
        });
        cs.push(Candidate {
            mv,
            child,
            rng,
            cue,
        });
    }
    let mut exact_nodes = 0u64;
    let mut solved = None;
    if !ss.iter().any(|s| s.known == Some(1.)) {
        let mut idx = None;
        let mut maxscore = 0;
        for (i, s) in ss.iter().enumerate() {
            let c = cs[i].cue;
            let q = c[1] + 2 * c[2];
            if s.known.is_none() && (idx.is_none() || q > maxscore) {
                idx = Some(i);
                maxscore = q
            }
        }
        if let Some(i) = idx {
            let (v, n) = exact(&cs[i].child, player);
            ss[i].known = Some(utility(v));
            exact_nodes = n;
            solved = Some(cs[i].mv)
        }
    }
    let (initial, ii) = best(&ss, m, 0, true);
    let arm = if policy < 2 { 0 } else { policy - 2 };
    let mut qm = [None; 2];
    let mut yy = [None; 2];
    let mut qn = [0u64; 2];
    let mut pv = best(&ss, m, arm, policy == 0).0;
    if policy >= 2 {
        for step in 0..horizon {
            let (v, idx) = plan(&mut ss, m, arm, horizon - step);
            if step == 0 {
                pv = v
            }
            if let Some(i) = idx {
                let c = &mut cs[i];
                let (o, n) = rollout(&c.child, player, &mut c.rng);
                let y = (o + 1) as usize;
                ss[i].h[y] += 1;
                qm[step] = Some(c.mv);
                yy[step] = Some(y);
                qn[step] = n;
            } else {
                break;
            }
        }
    }
    let (terminal, fi) = best(&ss, m, arm, policy == 0);
    let first_move = cs[ii].mv;
    let final_move = cs[fi].mv;
    drop(ss);
    drop(cs);
    let elapsed = t.elapsed().as_nanos() as u64;
    json!({"policy":policy,"horizon":horizon,"initial_move":first_move,"initial_u":initial,"solved":solved,
          "final_move":final_move,"terminal_mean":terminal,"plan_value":pv,
          "query_moves":qm,"observations":yy,"query_nodes":qn,"cue_nodes":cue_nodes,
          "exact_nodes":exact_nodes,"elapsed_ns":elapsed})
}
fn run(req: Request, m: &Model) -> Result<Value, String> {
    if req.id.is_empty()
        || req.id.len() > 128
        || req.history.len() > 8
        || req.repeats == 0
        || req.repeats > 5
    {
        return Err("request range".into());
    }
    let mut root = TicTacToe::initial();
    for mv in req.history {
        if root.is_terminal() || !root.legal_moves().contains(&mv) {
            return Err("illegal history".into());
        }
        root = root.apply_move(mv);
    }
    if root.is_terminal() {
        return Err("terminal root".into());
    }
    let mut runs = Vec::new();
    // H1 controls are executed separately from timing.
    for p in [3, 4] {
        runs.push(one(&root, m, req.cue_seed, p, 1))
    }
    // Warmup each, rotate policy order in each measured repeat.
    for p in 0..5 {
        let _ = one(&root, m, req.cue_seed, p, 2);
    }
    for r in 0..req.repeats {
        for j in 0..5 {
            let p = (r + j + (req.cue_seed % 5) as usize) % 5;
            let mut v = one(&root, m, req.cue_seed, p, 2);
            v["repeat"] = json!(r);
            runs.push(v);
        }
    }
    Ok(json!({"id":req.id,"cue_seed":req.cue_seed,"runs":runs}))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("model path required")?;
    let m: Model = serde_json::from_reader(std::fs::File::open(path)?)?;
    m.validate()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let out = match serde_json::from_str::<Request>(&line) {
            Ok(req) => run(req, &m).unwrap_or_else(|e| json!({"error":e})),
            Err(e) => json!({"error":e.to_string()}),
        };
        println!("{}", out);
    }
    Ok(())
}
