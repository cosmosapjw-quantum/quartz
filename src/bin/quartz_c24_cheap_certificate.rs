//! C24 scalar-envelope computation-choice experiment; isolated from game engines.
//! Formula and domain: docs/research/quartz_general_cheap_certificates_20261123/theory/DERIVATION.md.
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    side: usize,
    rho: u32,
    #[serde(deserialize_with = "deserialize_scalar")]
    scale: f64,
    #[serde(deserialize_with = "deserialize_rhs")]
    rhs: Vec<Vec<f64>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    task: Task,
    additional: usize,
    #[serde(default = "one")]
    repeats: usize,
    #[serde(default)]
    warmup: usize,
    #[serde(default)]
    trace: bool,
}
fn one() -> usize {
    1
}

// The study uses decimal strings and Rust's correctly rounded str::parse.
// Numeric JSON remains a compatibility path; its input bits must not be
// assumed identical to an independently manufactured binary64 source.
#[derive(Deserialize)]
#[serde(untagged)]
enum JsonScalar {
    Number(f64),
    Text(String),
}
impl JsonScalar {
    fn value(self) -> Result<f64, String> {
        match self {
            Self::Number(x) => Ok(x),
            Self::Text(s) => {
                if s.len() > 64 {
                    return Err("scalar string exceeds64bytes".into());
                }
                s.parse::<f64>()
                    .map_err(|e| format!("invalid scalar string: {e}"))
            }
        }
    }
}
fn deserialize_scalar<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    JsonScalar::deserialize(deserializer)?
        .value()
        .map_err(serde::de::Error::custom)
}
fn deserialize_rhs<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Vec<f64>>, D::Error> {
    let raw = Vec::<Vec<JsonScalar>>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|v| v.value().map_err(serde::de::Error::custom))
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn invalid() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }
    fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
    fn add(self, b: Self) -> Self {
        if !self.finite() || !b.finite() {
            return Self::invalid();
        }
        Self {
            lo: (self.lo + b.lo).next_down(),
            hi: (self.hi + b.hi).next_up(),
        }
    }
    fn sub(self, b: Self) -> Self {
        if !self.finite() || !b.finite() {
            return Self::invalid();
        }
        Self {
            lo: (self.lo - b.hi).next_down(),
            hi: (self.hi - b.lo).next_up(),
        }
    }
    fn mul(self, b: Self) -> Self {
        if !self.finite() || !b.finite() {
            return Self::invalid();
        }
        let p = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        if p.iter().any(|v| !v.is_finite()) {
            return Self::invalid();
        }
        Self {
            lo: p.iter().copied().fold(f64::INFINITY, f64::min).next_down(),
            hi: p
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    fn square(self) -> Self {
        if !self.finite() {
            return Self::invalid();
        }
        let a = self.lo * self.lo;
        let b = self.hi * self.hi;
        if !a.is_finite() || !b.is_finite() {
            return Self::invalid();
        }
        Self {
            lo: if self.lo <= 0.0 && self.hi >= 0.0 {
                0.0
            } else {
                a.min(b).next_down().max(0.0)
            },
            hi: a.max(b).next_up(),
        }
    }
    fn div_positive(self, d: f64) -> Self {
        if !self.finite() || !d.is_finite() || d <= 0.0 {
            return Self::invalid();
        }
        Self {
            lo: (self.lo / d).next_down(),
            hi: (self.hi / d).next_up(),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
struct Bounds {
    lo: f64,
    hi: f64,
    ell_hat: f64,
}
#[derive(Clone, Debug, Serialize)]
struct Checkpoint {
    iterations: Vec<usize>,
    fresh_bounds: Vec<Bounds>,
    bounds: Vec<Bounds>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Policy {
    PlainStop,
    PlainAlways,
    FastOnce,
    FastBlock,
    OutwardBlock,
    Residual,
}
impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::PlainStop => "PLAIN_STOP",
            Self::PlainAlways => "PLAIN_ALWAYS",
            Self::FastOnce => "FAST_ONCE",
            Self::FastBlock => "FAST_BLOCK",
            Self::OutwardBlock => "OUTWARD_BLOCK",
            Self::Residual => "RESIDUAL",
        }
    }
    fn initial_kind(self) -> EvaluationKind {
        match self {
            Self::PlainStop | Self::PlainAlways => EvaluationKind::Plain,
            Self::OutwardBlock => EvaluationKind::Outward,
            _ => EvaluationKind::Fast,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
struct Outcome {
    policy: String,
    rep: usize,
    selected: usize,
    initial_selected: usize,
    certified: bool,
    reason: String,
    iterations: Vec<usize>,
    solver_matvecs: usize,
    fast_evaluations: usize,
    outward_evaluations: usize,
    bound_evaluations: usize,
    ordinary_evaluations: usize,
    total_us: f64,
    validation_us: f64,
    initial_us: f64,
    solve_us: f64,
    bounds_us: f64,
    initial_bounds: Vec<Bounds>,
    final_bounds: Vec<Bounds>,
    decision_bounds: Vec<Bounds>,
    // Ordinary estimates only. A certificate is carried exclusively by residual_upper.
    residual_relative: Vec<f64>,
    residual_upper: Option<Vec<f64>>,
    residual_goal_met: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<Vec<Checkpoint>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_x: Option<Vec<Vec<f64>>>,
}
#[derive(Debug, Serialize)]
struct Response {
    id: String,
    additional: usize,
    results: Vec<Outcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_rhs_bits: Option<Vec<Vec<u64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_scale_bits: Option<u64>,
}

// All loops below preserve sequential binary64 operations. No mul_add, FMA or
// reassociation is used. These are the ordinary orders in DERIVATION R5.2.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * y[i];
    }
    sum
}
fn stencil_value(side: usize, rho: u32, x: &[f64], i: usize) -> f64 {
    let r = rho as f64;
    let d = 1.0 + 4.0 * r; // Exact for the supported integer rho.
    let row = i / side;
    let col = i % side;
    let mut y = d * x[i];
    if row > 0 {
        y -= r * x[i - side];
    }
    if row + 1 < side {
        y -= r * x[i + side];
    }
    if col > 0 {
        y -= r * x[i - 1];
    }
    if col + 1 < side {
        y -= r * x[i + 1];
    }
    y
}
fn apply_operator(side: usize, rho: u32, x: &[f64], out: &mut [f64]) {
    for i in 0..x.len() {
        out[i] = stencil_value(side, rho, x, i);
    }
}
struct Solver {
    x: Vec<f64>,
    r: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    rr: f64,
    rr0: f64,
    iterations: usize,
}
impl Solver {
    fn new(b: &[f64]) -> Self {
        let rr = dot(b, b);
        Self {
            x: vec![0.0; b.len()],
            r: b.to_vec(),
            p: b.to_vec(),
            ap: vec![0.0; b.len()],
            rr,
            rr0: rr,
            iterations: 0,
        }
    }
    fn step(&mut self, side: usize, rho: u32, rel_tol: f64) -> Result<bool, String> {
        if self.rr == 0.0 || self.rr <= self.rr0 * (rel_tol * rel_tol) {
            return Ok(false);
        }
        apply_operator(side, rho, &self.p, &mut self.ap);
        let pap = dot(&self.p, &self.ap);
        if !pap.is_finite() || pap <= 0.0 {
            return Err("nonpositive/nonfinite CG curvature".into());
        }
        let alpha = self.rr / pap;
        if !alpha.is_finite() {
            return Err("nonfinite CG step".into());
        }
        for i in 0..self.x.len() {
            self.x[i] += alpha * self.p[i];
            self.r[i] -= alpha * self.ap[i];
            if !self.x[i].is_finite() || !self.r[i].is_finite() {
                return Err("nonfinite CG state".into());
            }
        }
        let next = dot(&self.r, &self.r);
        if !next.is_finite() || next < 0.0 {
            return Err("invalid recurrent residual".into());
        }
        let beta = next / self.rr;
        for i in 0..self.p.len() {
            self.p[i] = self.r[i] + beta * self.p[i];
        }
        self.rr = next;
        self.iterations += 1;
        Ok(true)
    }
}

const UNIT_ROUNDOFF: f64 = f64::from_bits(0x3ca0_0000_0000_0000); // 2^-53
const RHS_MIN: f64 = f64::from_bits(623u64 << 52); // 2^-400
const RHS_MAX: f64 = f64::from_bits(1123u64 << 52); // 2^100
const STATE_MAX: f64 = f64::from_bits(1223u64 << 52); // 2^200
const RESIDUAL_TOL: f64 = 1.0e-8;

fn add_up(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}
fn sub_down(a: f64, b: f64) -> f64 {
    (a - b).next_down()
}
fn mul_up(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}
fn div_up(a: f64, b: f64) -> f64 {
    (a / b).next_up()
}
fn div_down(a: f64, b: f64) -> f64 {
    (a / b).next_down()
}

#[derive(Clone, Copy, Debug)]
struct ErrorBudget {
    gamma: f64,
    floor: f64,
}
impl ErrorBudget {
    fn new(k: usize) -> Result<Self, String> {
        let ku = mul_up(k as f64, UNIT_ROUNDOFF);
        let denominator = sub_down(1.0, ku);
        if denominator <= 0.0 {
            return Err("invalid rounding-budget denominator".into());
        }
        let out = Self {
            gamma: div_up(ku, denominator),
            floor: div_up(mul_up(k as f64, f64::MIN_POSITIVE), denominator),
        };
        if !out.gamma.is_finite() || !out.floor.is_finite() || out.gamma >= 1.0 {
            return Err("nonfinite rounding budget".into());
        }
        Ok(out)
    }
}
// R5.4: invert |h - exact nonnegative square sum| <= gamma*sum + floor.
// Clipping a nonpositive lower numerator before dividing avoids any sign issue
// when replacing the exact gamma by its floating upper bound.
fn positive_square_sum(h: f64, budget: ErrorBudget) -> Result<Interval, String> {
    if !h.is_finite() || h < 0.0 {
        return Err("invalid ordinary square sum".into());
    }
    let lo = if h <= budget.floor {
        0.0
    } else {
        div_down(sub_down(h, budget.floor), add_up(1.0, budget.gamma)).max(0.0)
    };
    let hi = div_up(add_up(h, budget.floor), sub_down(1.0, budget.gamma));
    let result = Interval { lo, hi };
    if !result.finite() {
        return Err("nonfinite positive square-sum enclosure".into());
    }
    Ok(result)
}
#[derive(Clone, Copy, Debug)]
struct InputNorm {
    max_abs: f64,
    ordinary: f64,
    bounds: Interval,
}
#[derive(Debug)]
struct Validated {
    norms: Vec<InputNorm>,
    dot_budget: ErrorBudget,
}
fn validate(task: &Task) -> Result<Validated, String> {
    if task.id.is_empty() || task.id.len() > 128 {
        return Err("invalid task id".into());
    }
    if !(2..=128).contains(&task.side) || task.rho > 64 {
        return Err("operator outside bounded domain".into());
    }
    if !(2..=8).contains(&task.rhs.len()) {
        return Err("need 2..8 actions".into());
    }
    let bits = task.scale.to_bits();
    if !task.scale.is_normal() || task.scale <= 0.0 || (bits & ((1u64 << 52) - 1)) != 0 {
        return Err("scale must be a positive normal power of two".into());
    }
    let n = task.side * task.side;
    let dot_budget = ErrorBudget::new(2 * n)?;
    let mut norms = Vec::with_capacity(task.rhs.len());
    for b in &task.rhs {
        if b.len() != n {
            return Err("rhs shape or finite-range violation".into());
        }
        let mut max_abs: f64 = 0.0;
        let mut ordinary = 0.0;
        for &v in b {
            if !v.is_finite() {
                return Err("rhs shape or finite-range violation".into());
            }
            max_abs = max_abs.max(v.abs());
            ordinary += v * v;
        }
        if max_abs != 0.0 && !(RHS_MIN..=RHS_MAX).contains(&max_abs) {
            return Err(
                "UNSUPPORTED_RHS_NORM_RANGE: nonzero max|b| must lie in [2^-400,2^100]".into(),
            );
        }
        let bounds = if max_abs == 0.0 {
            Interval::point(0.0)
        } else {
            positive_square_sum(ordinary, dot_budget)?
        };
        if max_abs != 0.0 && bounds.lo <= 0.0 {
            return Err(
                "UNSUPPORTED_RHS_NORM_RANGE: squared norm lower bound is not positive".into(),
            );
        }
        if bounds.hi > task.scale {
            return Err("scale does not cover rigorous rhs norm upper bound".into());
        }
        norms.push(InputNorm {
            max_abs,
            ordinary,
            bounds,
        });
    }
    Ok(Validated { norms, dot_budget })
}
fn state_max(task: &Task, b: &[f64], x: &[f64], norm: InputNorm) -> Result<f64, String> {
    if x.len() != b.len() || x.len() != task.side * task.side {
        return Err("invalid response state shape".into());
    }
    let mut max_abs: f64 = 0.0;
    for &v in x {
        if !v.is_finite() {
            return Err("nonfinite response state".into());
        }
        max_abs = max_abs.max(v.abs());
    }
    if norm.max_abs == 0.0 {
        if max_abs != 0.0 {
            return Err("UNSUPPORTED_STATE: zero rhs requires zero state".into());
        }
    } else if max_abs > STATE_MAX || max_abs > RHS_MAX * norm.max_abs {
        return Err("UNSUPPORTED_STATE: require X <= min(2^200,2^100 B)".into());
    }
    Ok(max_abs)
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum EvaluationKind {
    Plain,
    Fast,
    Outward,
}
#[derive(Debug)]
struct Evaluation {
    bound: Bounds,
    approximate_relative: f64,
    residual_upper: Option<f64>,
}
fn approximate_relative(residual_square: f64, norm: InputNorm) -> Result<f64, String> {
    let value = if norm.max_abs == 0.0 {
        0.0
    } else {
        residual_square.sqrt() / norm.ordinary.sqrt()
    };
    if !value.is_finite() {
        return Err("nonfinite approximate residual".into());
    }
    Ok(value)
}
fn certified_relative_upper(residual_upper: f64, norm: InputNorm) -> Result<f64, String> {
    if norm.max_abs == 0.0 {
        return Ok(0.0);
    }
    // R5.6: sqrt FIRST, then divide; the squared ratio could overflow.
    let numerator = residual_upper.sqrt().next_up();
    let denominator = norm.bounds.lo.sqrt().next_down();
    if denominator <= 0.0 {
        return Err("nonpositive input norm denominator".into());
    }
    let result = div_up(numerator, denominator);
    if !result.is_finite() {
        return Err("nonfinite certified relative residual".into());
    }
    Ok(result)
}
fn normalize_response(
    task: &Task,
    ell: Interval,
    residual: Interval,
    ell_hat: f64,
) -> Result<Bounds, String> {
    let beta = 1.0 + 8.0 * (task.rho as f64);
    if !ell.finite() || !residual.finite() || residual.lo < 0.0 || !ell_hat.is_finite() {
        return Err("nonfinite response enclosure".into());
    }
    let lower = ell.add(residual.div_positive(beta));
    let upper = ell.add(residual);
    if !lower.finite() || !upper.finite() {
        return Err("nonfinite raw response enclosure".into());
    }
    // R5.5: clip RAW quantities first, so a loose bound / tiny S cannot overflow.
    let raw_lo = lower.lo.max(0.0);
    let raw_hi = upper.hi.min(task.scale);
    if raw_lo > raw_hi {
        return Err("empty raw response enclosure".into());
    }
    let lo = div_down(raw_lo, task.scale).max(0.0);
    let hi = div_up(raw_hi, task.scale).min(1.0);
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err("empty normalized response enclosure".into());
    }
    Ok(Bounds { lo, hi, ell_hat })
}
fn ordinary_or_fast_response(
    task: &Task,
    b: &[f64],
    x: &[f64],
    norm: InputNorm,
    dot_budget: ErrorBudget,
    fast: bool,
) -> Result<Evaluation, String> {
    let xmax = state_max(task, b, x, norm)?;
    let (mut ph, mut qh, mut dh) = (0.0, 0.0, 0.0);
    let (mut ymax, mut rmax): (f64, f64) = (0.0, 0.0);
    for i in 0..x.len() {
        let yhat = stencil_value(task.side, task.rho, x, i);
        let rhat = b[i] - yhat;
        ph += b[i] * x[i];
        qh += x[i] * yhat;
        dh += rhat * rhat;
        if fast {
            ymax = ymax.max(yhat.abs());
            rmax = rmax.max(rhat.abs());
        }
    }
    let ell_hat = (2.0 * ph - qh) / task.scale;
    if !ph.is_finite() || !qh.is_finite() || !dh.is_finite() || !ell_hat.is_finite() {
        return Err("nonfinite ordinary response".into());
    }
    let approximate_relative = approximate_relative(dh, norm)?;
    if !fast {
        return Ok(Evaluation {
            bound: Bounds {
                lo: 0.0,
                hi: 1.0,
                ell_hat,
            },
            approximate_relative,
            residual_upper: None,
        });
    }
    if norm.max_abs == 0.0 {
        return Ok(Evaluation {
            bound: Bounds {
                lo: 0.0,
                hi: 0.0,
                ell_hat,
            },
            approximate_relative: 0.0,
            residual_upper: Some(0.0),
        });
    }
    let stencil_budget = ErrorBudget::new(9)?;
    let n = x.len() as f64;
    let beta = 1.0 + 8.0 * (task.rho as f64);
    // R5.2/R5.3: every error-budget operation is rounded upward.
    let ey = add_up(
        mul_up(mul_up(stencil_budget.gamma, beta), xmax),
        stencil_budget.floor,
    );
    let er = add_up(
        add_up(ey, mul_up(UNIT_ROUNDOFF, add_up(norm.max_abs, ymax))),
        f64::MIN_POSITIVE,
    );
    let ep = add_up(
        mul_up(mul_up(mul_up(dot_budget.gamma, n), norm.max_abs), xmax),
        dot_budget.floor,
    );
    let eq = add_up(
        add_up(
            mul_up(mul_up(mul_up(dot_budget.gamma, n), xmax), ymax),
            dot_budget.floor,
        ),
        mul_up(mul_up(n, xmax), ey),
    );
    let p = Interval {
        lo: sub_down(ph, ep),
        hi: add_up(ph, ep),
    };
    let q = Interval {
        lo: sub_down(qh, eq),
        hi: add_up(qh, eq),
    };
    let ell = Interval::point(2.0).mul(p).sub(q);
    let d0 = positive_square_sum(dh, dot_budget)?;
    let tail = mul_up(n, add_up(mul_up(mul_up(2.0, rmax), er), mul_up(er, er)));
    let residual = Interval {
        lo: sub_down(d0.lo, tail).max(0.0),
        hi: add_up(d0.hi, tail),
    };
    if [ey, er, ep, eq, tail]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("nonfinite scalar response budget".into());
    }
    let bound = normalize_response(task, ell, residual, ell_hat)?;
    let residual_upper = Some(certified_relative_upper(residual.hi, norm)?);
    Ok(Evaluation {
        bound,
        approximate_relative,
        residual_upper,
    })
}
// Inherited full per-element outward evaluator, with the same raw clipping,
// common support guard and honest residual-upper denominator as the fast path.
fn outward_response(
    task: &Task,
    b: &[f64],
    x: &[f64],
    norm: InputNorm,
) -> Result<Evaluation, String> {
    state_max(task, b, x, norm)?;
    let side = task.side;
    let r = task.rho as f64;
    let d = 1.0 + 4.0 * r;
    let mut p = Interval::point(0.0);
    let mut q = Interval::point(0.0);
    let mut rr = Interval::point(0.0);
    let (mut ph, mut qh, mut dh) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let row = i / side;
        let col = i % side;
        let mut y = Interval::point(d).mul(Interval::point(x[i]));
        if row > 0 {
            y = y.sub(Interval::point(r).mul(Interval::point(x[i - side])));
        }
        if row + 1 < side {
            y = y.sub(Interval::point(r).mul(Interval::point(x[i + side])));
        }
        if col > 0 {
            y = y.sub(Interval::point(r).mul(Interval::point(x[i - 1])));
        }
        if col + 1 < side {
            y = y.sub(Interval::point(r).mul(Interval::point(x[i + 1])));
        }
        let yh = stencil_value(side, task.rho, x, i);
        p = p.add(Interval::point(b[i]).mul(Interval::point(x[i])));
        q = q.add(Interval::point(x[i]).mul(y));
        rr = rr.add(Interval::point(b[i]).sub(y).square());
        ph += b[i] * x[i];
        qh += x[i] * yh;
        let ri = b[i] - yh;
        dh += ri * ri;
    }
    let ell_hat = (2.0 * ph - qh) / task.scale;
    if !dh.is_finite() || !ell_hat.is_finite() {
        return Err("nonfinite outward response readout".into());
    }
    let approximate_relative = approximate_relative(dh, norm)?;
    if norm.max_abs == 0.0 {
        return Ok(Evaluation {
            bound: Bounds {
                lo: 0.0,
                hi: 0.0,
                ell_hat,
            },
            approximate_relative: 0.0,
            residual_upper: Some(0.0),
        });
    }
    let ell = Interval::point(2.0).mul(p).sub(q);
    // Interval addition can push an exact zero lower endpoint below zero.
    rr.lo = rr.lo.max(0.0);
    let bound = normalize_response(task, ell, rr, ell_hat)?;
    let residual_upper = Some(certified_relative_upper(rr.hi, norm)?);
    Ok(Evaluation {
        bound,
        approximate_relative,
        residual_upper,
    })
}
fn evaluate(
    task: &Task,
    b: &[f64],
    x: &[f64],
    norm: InputNorm,
    validated: &Validated,
    kind: EvaluationKind,
) -> Result<Evaluation, String> {
    match kind {
        EvaluationKind::Outward => outward_response(task, b, x, norm),
        EvaluationKind::Plain | EvaluationKind::Fast => ordinary_or_fast_response(
            task,
            b,
            x,
            norm,
            validated.dot_budget,
            kind == EvaluationKind::Fast,
        ),
    }
}
fn separated_action(bounds: &[Bounds]) -> Option<usize> {
    let mut best = 0;
    for i in 1..bounds.len() {
        if bounds[i].lo > bounds[best].lo {
            best = i;
        }
    }
    if bounds
        .iter()
        .enumerate()
        .all(|(i, b)| i == best || bounds[best].lo > b.hi)
    {
        Some(best)
    } else {
        None
    }
}
fn score_action(bounds: &[Bounds]) -> usize {
    let mut best = 0;
    for i in 1..bounds.len() {
        if bounds[i].ell_hat > bounds[best].ell_hat {
            best = i;
        }
    }
    best
}
#[derive(Default)]
struct EvaluationCounts {
    fast: usize,
    outward: usize,
    ordinary: usize,
}
struct Measurement {
    fresh: Vec<Bounds>,
    relative: Vec<f64>,
    upper: Option<Vec<f64>>,
}
fn measure_bounds(
    task: &Task,
    solvers: &[Solver],
    validated: &Validated,
    history: &mut Vec<Bounds>,
    trace: &mut Option<Vec<Checkpoint>>,
    kind: EvaluationKind,
    counts: &mut EvaluationCounts,
) -> Result<Measurement, String> {
    let mut fresh = Vec::with_capacity(solvers.len());
    let mut relative = Vec::with_capacity(solvers.len());
    let mut upper = if kind == EvaluationKind::Plain {
        None
    } else {
        Some(Vec::with_capacity(solvers.len()))
    };
    for (i, s) in solvers.iter().enumerate() {
        let evaluation = evaluate(
            task,
            &task.rhs[i],
            &s.x,
            validated.norms[i],
            validated,
            kind,
        )?;
        fresh.push(evaluation.bound);
        relative.push(evaluation.approximate_relative);
        if let Some(values) = &mut upper {
            values.push(
                evaluation
                    .residual_upper
                    .ok_or("missing certified residual upper")?,
            );
        }
    }
    match kind {
        EvaluationKind::Plain => counts.ordinary += solvers.len(),
        EvaluationKind::Fast => counts.fast += solvers.len(),
        EvaluationKind::Outward => counts.outward += solvers.len(),
    }
    if history.is_empty() {
        *history = fresh.clone();
    } else {
        for (old, new) in history.iter_mut().zip(&fresh) {
            old.lo = old.lo.max(new.lo);
            old.hi = old.hi.min(new.hi);
            old.ell_hat = new.ell_hat;
            if old.lo > old.hi {
                return Err("empty historical interval intersection".into());
            }
        }
    }
    if let Some(points) = trace {
        points.push(Checkpoint {
            iterations: solvers.iter().map(|s| s.iterations).collect(),
            fresh_bounds: fresh.clone(),
            bounds: history.clone(),
        });
    }
    Ok(Measurement {
        fresh,
        relative,
        upper,
    })
}
fn advance(solvers: &mut [Solver], task: &Task, steps: usize, tol: f64) -> Result<usize, String> {
    let mut count = 0;
    for s in solvers {
        for _ in 0..steps {
            if !s.step(task.side, task.rho, tol)? {
                break;
            }
            count += 1;
        }
    }
    Ok(count)
}
fn micros(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1.0e6
}
fn run_policy(
    task: &Task,
    additional: usize,
    policy: Policy,
    want_trace: bool,
) -> Result<Outcome, String> {
    let total_start = Instant::now();
    let validation_start = Instant::now();
    let validated = validate(task)?;
    if additional > 128 {
        return Err("additional iteration cap exceeds128".into());
    }
    let validation_us = micros(validation_start);
    let initial_start = Instant::now();
    let mut solvers: Vec<Solver> = task.rhs.iter().map(|b| Solver::new(b)).collect();
    advance(&mut solvers, task, 4, 1.0e-14)?;
    let initial_us = micros(initial_start);
    let mut trace = if want_trace { Some(Vec::new()) } else { None };
    let mut history = Vec::new();
    let mut counts = EvaluationCounts::default();
    let kind = policy.initial_kind();
    let bound_start = Instant::now();
    let mut measured = measure_bounds(
        task,
        &solvers,
        &validated,
        &mut history,
        &mut trace,
        kind,
        &mut counts,
    )?;
    let mut bounds_us = micros(bound_start);
    let initial_bounds = measured.fresh.clone();
    let initial_selected = score_action(&measured.fresh);
    let mut selected = initial_selected;
    let mut certified = false;
    let mut reason = "INITIAL_ONLY".to_string();
    let mut solve_us = 0.0;
    if matches!(
        policy,
        Policy::FastOnce | Policy::FastBlock | Policy::OutwardBlock
    ) {
        if let Some(i) = separated_action(&history) {
            selected = i;
            certified = true;
            reason = "INITIAL_CERTIFICATE".into();
        }
    }
    if !certified && policy != Policy::PlainStop {
        if policy == Policy::Residual {
            let t = Instant::now();
            for s in &mut solvers {
                while s.iterations < 512 {
                    if !s.step(task.side, task.rho, RESIDUAL_TOL)? {
                        break;
                    }
                }
            }
            solve_us += micros(t);
            let t = Instant::now();
            measured = measure_bounds(
                task,
                &solvers,
                &validated,
                &mut history,
                &mut trace,
                kind,
                &mut counts,
            )?;
            bounds_us += micros(t);
            selected = score_action(&measured.fresh);
            let met = measured
                .upper
                .as_ref()
                .is_some_and(|values| values.iter().all(|r| *r <= RESIDUAL_TOL));
            reason = if met {
                "RESIDUAL_GOAL"
            } else {
                "RESIDUAL_GOAL_NOT_MET"
            }
            .into();
        } else if matches!(policy, Policy::FastBlock | Policy::OutwardBlock) {
            let mut used = 0;
            while used < additional {
                let batch = 8.min(additional - used);
                let t = Instant::now();
                let advanced = advance(&mut solvers, task, batch, 1.0e-14)?;
                solve_us += micros(t);
                used += batch;
                if advanced == 0 {
                    break;
                }
                let t = Instant::now();
                measured = measure_bounds(
                    task,
                    &solvers,
                    &validated,
                    &mut history,
                    &mut trace,
                    kind,
                    &mut counts,
                )?;
                bounds_us += micros(t);
                if let Some(i) = separated_action(&history) {
                    selected = i;
                    certified = true;
                    reason = "BLOCK_CERTIFICATE".into();
                    break;
                }
            }
            if !certified {
                selected = score_action(&measured.fresh);
                reason = "UNSEPARATED_BUDGET".into();
            }
        } else {
            let t = Instant::now();
            advance(&mut solvers, task, additional, 1.0e-14)?;
            solve_us += micros(t);
            let t = Instant::now();
            // FAST_ONCE has no additional enclosure after a failed initial gate.
            measured = measure_bounds(
                task,
                &solvers,
                &validated,
                &mut history,
                &mut trace,
                EvaluationKind::Plain,
                &mut counts,
            )?;
            bounds_us += micros(t);
            selected = score_action(&measured.fresh);
            reason = if policy == Policy::PlainAlways {
                "FIXED_BUDGET_PLAIN"
            } else {
                "UNSEPARATED_BUDGET"
            }
            .into();
        }
    }
    let residual_goal_met = measured
        .upper
        .as_ref()
        .map(|values| values.iter().all(|r| *r <= RESIDUAL_TOL));
    let iterations: Vec<usize> = solvers.iter().map(|s| s.iterations).collect();
    let solver_matvecs = iterations.iter().sum();
    let final_x = if want_trace {
        Some(solvers.iter().map(|s| s.x.clone()).collect())
    } else {
        None
    };
    let mut out = Outcome {
        policy: policy.label().into(),
        rep: 0,
        selected,
        initial_selected,
        certified,
        reason,
        iterations,
        solver_matvecs,
        fast_evaluations: counts.fast,
        outward_evaluations: counts.outward,
        ordinary_evaluations: counts.ordinary,
        bound_evaluations: counts.fast + counts.outward,
        total_us: 0.0,
        validation_us,
        initial_us,
        solve_us,
        bounds_us,
        initial_bounds,
        final_bounds: measured.fresh,
        decision_bounds: history,
        residual_relative: measured.relative,
        residual_upper: measured.upper,
        residual_goal_met,
        trace,
        final_x,
    };
    // Timing includes destruction of solver and validation workspaces.
    drop(solvers);
    drop(validated);
    out.total_us = micros(total_start);
    Ok(out)
}
fn execute(req: Request) -> Result<Response, String> {
    if req.repeats == 0 || req.repeats > 32 || req.warmup > 4 || req.additional > 128 {
        return Err("request repeat/warmup/budget cap violation".into());
    }
    let policies = [
        Policy::PlainStop,
        Policy::PlainAlways,
        Policy::FastOnce,
        Policy::FastBlock,
        Policy::OutwardBlock,
        Policy::Residual,
    ];
    let offset = req
        .task
        .id
        .bytes()
        .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
        % policies.len();
    let mut results = Vec::with_capacity(req.repeats * policies.len());
    for pass in 0..req.warmup + req.repeats {
        for k in 0..policies.len() {
            let policy = policies[(offset + pass + k) % policies.len()];
            let mut out = run_policy(&req.task, req.additional, policy, req.trace)?;
            if pass >= req.warmup {
                out.rep = pass - req.warmup;
                results.push(out);
            }
        }
    }
    let input_rhs_bits = if req.trace {
        Some(
            req.task
                .rhs
                .iter()
                .map(|b| b.iter().map(|v| v.to_bits()).collect())
                .collect(),
        )
    } else {
        None
    };
    let input_scale_bits = if req.trace {
        Some(req.task.scale.to_bits())
    } else {
        None
    };
    Ok(Response {
        id: req.task.id,
        additional: req.additional,
        results,
        input_rhs_bits,
        input_scale_bits,
    })
}
fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let result = line.map_err(|e| e.to_string()).and_then(|s| {
            if s.len() > 32 * 1024 * 1024 {
                return Err("request line exceeds32MiB".into());
            }
            serde_json::from_str::<Request>(&s)
                .map_err(|e| e.to_string())
                .and_then(execute)
        });
        let value = match result {
            Ok(response) => serde_json::to_value(response)
                .unwrap_or_else(|e| serde_json::json!({"error":e.to_string()})),
            Err(error) => serde_json::json!({"error":error}),
        };
        if writeln!(stdout, "{}", value).is_err() || stdout.flush().is_err() {
            break;
        }
    }
}

// Test-only convenience seams: production policies use cached validation.
#[cfg(test)]
fn test_response(
    task: &Task,
    b: &[f64],
    x: &[f64],
    kind: EvaluationKind,
) -> Result<(Bounds, f64), String> {
    let validated = validate(task)?;
    let i = task
        .rhs
        .iter()
        .position(|rhs| rhs.as_slice() == b)
        .ok_or("test rhs missing")?;
    let result = evaluate(task, b, x, validated.norms[i], &validated, kind)?;
    Ok((result.bound, result.approximate_relative))
}
#[cfg(test)]
fn enclose_response(task: &Task, b: &[f64], x: &[f64]) -> Result<(Bounds, f64), String> {
    test_response(task, b, x, EvaluationKind::Fast)
}
#[cfg(test)]
fn ordinary_response(task: &Task, b: &[f64], x: &[f64]) -> Result<(Bounds, f64), String> {
    test_response(task, b, x, EvaluationKind::Plain)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Task {
        let n = 64;
        let b: Vec<f64> = (0..n)
            .map(|i| (((i * 17) % 23) as f64 - 11.0) / 16.0)
            .collect();
        let c: Vec<f64> = (0..n)
            .map(|i| (((i * 7) % 19) as f64 - 9.0) / 1024.0)
            .collect();
        Task {
            id: "strong".into(),
            side: 8,
            rho: 16,
            scale: 64.0,
            rhs: vec![b, c],
        }
    }
    #[test]
    fn signed_interval_operations_enclose_known_exact_results() {
        let a = Interval { lo: -2.0, hi: 1.0 };
        let b = Interval { lo: 3.0, hi: 4.0 };
        let p = a.mul(b);
        assert!(p.lo <= -8.0 && p.hi >= 4.0);
        let s = a.square();
        assert_eq!(s.lo, 0.0);
        assert!(s.hi >= 4.0);
        let d = Interval::point(1.0).div_positive(3.0);
        assert!(d.lo < 1.0 / 3.0 && d.hi > 1.0 / 3.0);
    }
    #[test]
    fn rational_response_example_and_one_step_certificate() {
        let task = Task {
            id: "exact".into(),
            side: 2,
            rho: 1,
            scale: 2.0,
            rhs: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.5, 0.5, 0.0]],
        };
        let mut bs = Vec::new();
        for (b, q) in task.rhs.iter().zip([23.0 / 210.0, 5.0 / 84.0]) {
            let x: Vec<f64> = b.iter().map(|v| v / 5.0).collect();
            let (bound, _) = enclose_response(&task, b, &x).unwrap();
            assert!(bound.lo <= q && q <= bound.hi, "{bound:?} excludes {q}");
            bs.push(bound);
        }
        assert_eq!(separated_action(&bs), Some(0));
    }
    #[test]
    fn actual_stop_withholds_solver_matvecs() {
        let task = fixture();
        let always = run_policy(&task, 32, Policy::PlainAlways, true).unwrap();
        let gate = run_policy(&task, 32, Policy::FastOnce, true).unwrap();
        assert!(gate.certified);
        assert_eq!(gate.selected, 0);
        assert!(gate.solver_matvecs < always.solver_matvecs);
        assert_eq!(gate.iterations, vec![4, 4]);
    }
    #[test]
    fn plain_baseline_matches_state_and_action_without_interval_enclosures() {
        let mut task = fixture();
        task.rhs[1] = task.rhs[0].clone();
        let a = run_policy(&task, 32, Policy::OutwardBlock, true).unwrap();
        let p = run_policy(&task, 32, Policy::PlainAlways, true).unwrap();
        assert_eq!(p.selected, a.selected);
        assert_eq!(p.initial_selected, a.initial_selected);
        assert_eq!(p.iterations, a.iterations);
        assert_eq!(p.final_x, a.final_x);
        assert_eq!(p.bound_evaluations, 0);
        assert_eq!(
            p.final_bounds.iter().map(|b| b.ell_hat).collect::<Vec<_>>(),
            a.final_bounds.iter().map(|b| b.ell_hat).collect::<Vec<_>>()
        );
    }
    #[test]
    fn tied_uncertified_fallback_has_identical_state_and_action() {
        let mut task = fixture();
        task.rhs[1] = task.rhs[0].clone();
        let a = run_policy(&task, 8, Policy::PlainAlways, true).unwrap();
        for p in [Policy::FastOnce, Policy::FastBlock, Policy::OutwardBlock] {
            let b = run_policy(&task, 8, p, true).unwrap();
            assert!(!b.certified);
            assert_eq!(a.selected, b.selected);
            assert_eq!(a.iterations, b.iterations);
            assert_eq!(a.final_x, b.final_x);
        }
    }
    #[test]
    fn zero_rhs_needs_no_solver_work_and_does_not_claim_unique_winner() {
        let task = Task {
            id: "zero".into(),
            side: 2,
            rho: 0,
            scale: 1.0,
            rhs: vec![vec![0.0; 4]; 2],
        };
        let o = run_policy(&task, 32, Policy::FastBlock, true).unwrap();
        assert_eq!(o.solver_matvecs, 0);
        assert!(!o.certified);
        assert_eq!(o.selected, 0);
    }
    #[test]
    fn invalid_task_and_oracle_field_are_rejected() {
        let mut t = fixture();
        t.scale = 0.0;
        assert!(validate(&t).is_err());
        let raw = r#"{"task":{"id":"x","side":2,"rho":1,"scale":2,"rhs":[[1,0,0,0],[0,0,0,0]],"oracle":[1,0]},"additional":8}"#;
        assert!(serde_json::from_str::<Request>(raw).is_err());
    }
    #[test]
    fn conventional_solver_meets_its_stated_residual_goal() {
        let o = run_policy(&fixture(), 32, Policy::Residual, false).unwrap();
        assert_eq!(o.residual_goal_met, Some(true), "{:?}", o.residual_upper);
        assert!(o.residual_relative.iter().all(|v| *v <= 1.0e-8));
    }
    #[test]
    fn decimal_string_transport_preserves_exact_binary64_bits() {
        let raw = r#"{"task":{"id":"exact_decimal","side":2,"rho":1,"scale":"1024.0","rhs":[["1.8920180462300777","-10.504335880279541","0.0","0.0"],["0.0","0.0","0.0","0.0"]]},"additional":8}"#;
        let req: Request =
            serde_json::from_str(raw).expect("decimal string transport must be accepted");
        assert_eq!(req.task.rhs[0][0].to_bits(), 4611199710940626944);
        assert_eq!(req.task.rhs[0][1].to_bits(), 13845475070304059392);
        assert_eq!(req.task.scale.to_bits(), 1024.0f64.to_bits());
        validate(&req.task).unwrap();
    }
    #[test]
    fn trace_metadata_identifies_parsed_public_inputs() {
        let task = fixture();
        let expected: Vec<Vec<u64>> = task
            .rhs
            .iter()
            .map(|b| b.iter().map(|v| v.to_bits()).collect())
            .collect();
        let scale = task.scale.to_bits();
        let response = execute(Request {
            task,
            additional: 8,
            repeats: 1,
            warmup: 0,
            trace: true,
        })
        .unwrap();
        assert_eq!(response.input_rhs_bits, Some(expected));
        assert_eq!(response.input_scale_bits, Some(scale));
    }
    #[test]
    fn request_executes_every_control_with_numbered_repetitions() {
        let response = execute(Request {
            task: fixture(),
            additional: 8,
            repeats: 2,
            warmup: 1,
            trace: false,
        })
        .unwrap();
        assert_eq!(response.results.len(), 12);
        for rep in 0..2 {
            assert_eq!(response.results.iter().filter(|r| r.rep == rep).count(), 6);
        }
        assert!(response
            .results
            .iter()
            .all(|r| r.total_us.is_finite() && r.total_us >= 0.0));
    }
}

#[cfg(test)]
mod c24_behavior_tests {
    use super::*;
    fn task(tied: bool) -> Task {
        let b: Vec<f64> = (0..64)
            .map(|i| (((i * 17) % 23) as f64 - 11.0) / 16.0)
            .collect();
        let c = if tied {
            b.clone()
        } else {
            (0..64)
                .map(|i| (((i * 7) % 19) as f64 - 9.0) / 1024.0)
                .collect()
        };
        Task {
            id: "c24_behavior".into(),
            side: 8,
            rho: 16,
            scale: 64.0,
            rhs: vec![b, c],
        }
    }
    #[test]
    fn c24_tiny_nonzero_rhs_rejected_before_solver() {
        let t = Task {
            id: "tiny_nonzero".into(),
            side: 2,
            rho: 1,
            scale: 2.0f64.powi(-1000),
            rhs: vec![vec![2.0f64.powi(-600), 0.0, 0.0, 0.0], vec![0.0; 4]],
        };
        let got = run_policy(&t, 8, Policy::Residual, false);
        assert!(
            got.is_err(),
            "tiny nonzero RHS must reject before a false-success solve: {got:?}"
        );
        assert!(got.unwrap_err().contains("UNSUPPORTED_RHS_NORM_RANGE"));
    }
    #[test]
    fn c24_initial_control_performs_no_interval_work() {
        let out = run_policy(&task(false), 8, Policy::PlainStop, false).unwrap();
        assert_eq!(
            out.bound_evaluations, 0,
            "plain initial control must not compute enclosures"
        );
        assert_eq!(out.ordinary_evaluations, 2);
    }
    #[test]
    fn c24_plain_residual_goal_is_null() {
        let out =
            serde_json::to_value(run_policy(&task(false), 8, Policy::PlainAlways, false).unwrap())
                .unwrap();
        assert_eq!(
            out.get("residual_goal_met"),
            Some(&serde_json::Value::Null),
            "ordinary estimates cannot certify the residual goal"
        );
        assert_eq!(out.get("residual_upper"), Some(&serde_json::Value::Null));
    }
    #[test]
    fn c24_public_policy_labels_exact() {
        let out = execute(Request {
            task: task(false),
            additional: 8,
            repeats: 1,
            warmup: 0,
            trace: false,
        })
        .unwrap();
        let mut labels: Vec<_> = out.results.iter().map(|o| o.policy.as_str()).collect();
        labels.sort();
        let mut expected = vec![
            "PLAIN_STOP",
            "PLAIN_ALWAYS",
            "FAST_ONCE",
            "FAST_BLOCK",
            "OUTWARD_BLOCK",
            "RESIDUAL",
        ];
        expected.sort();
        assert_eq!(labels, expected);
    }
    #[test]
    fn c24_once_fallback_final_readout_is_ordinary() {
        let out = run_policy(&task(true), 8, Policy::FastOnce, true).unwrap();
        assert!(!out.certified);
        assert_eq!(
            out.ordinary_evaluations, 2,
            "FAST_ONCE fallback needs one final ordinary readout per action"
        );
        let value = serde_json::to_value(out).unwrap();
        assert_eq!(
            value.get("residual_goal_met"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(value.get("residual_upper"), Some(&serde_json::Value::Null));
    }
    #[test]
    fn c24_response_rejects_state_outside_supported_ratio() {
        let b = 2.0f64.powi(-400);
        let t = Task {
            id: "ratio_guard".into(),
            side: 2,
            rho: 1,
            scale: 2.0f64.powi(-798),
            rhs: vec![vec![b, 0.0, 0.0, 0.0]; 2],
        };
        let x = vec![2.0f64.powi(-299), 0.0, 0.0, 0.0];
        assert!(
            enclose_response(&t, &t.rhs[0], &x).is_err(),
            "state violates X <= 2^100 B"
        );
        assert!(
            ordinary_response(&t, &t.rhs[0], &x).is_err(),
            "plain readout shares support guard"
        );
    }
}

#[cfg(test)]
mod c24_numerical_contract_tests {
    use super::*;
    fn tied_task() -> Task {
        let b: Vec<f64> = (0..64)
            .map(|i| (((i * 17) % 23) as f64 - 11.0) / 16.0)
            .collect();
        Task {
            id: "c24_tied_contract".into(),
            side: 8,
            rho: 16,
            scale: 64.0,
            rhs: vec![b.clone(), b],
        }
    }
    #[test]
    fn all_uncertified_budget_edges_match_plain_state_score_and_action() {
        let task = tied_task();
        for additional in [0, 1, 9, 128] {
            let plain = run_policy(&task, additional, Policy::PlainAlways, true).unwrap();
            for policy in [Policy::FastOnce, Policy::FastBlock, Policy::OutwardBlock] {
                let gated = run_policy(&task, additional, policy, true).unwrap();
                assert!(!gated.certified, "tie must remain uncertified");
                assert_eq!(gated.selected, plain.selected);
                assert_eq!(gated.initial_selected, plain.initial_selected);
                assert_eq!(gated.iterations, plain.iterations);
                assert_eq!(gated.solver_matvecs, plain.solver_matvecs);
                let bits = |state: Option<Vec<Vec<f64>>>| {
                    state
                        .unwrap()
                        .into_iter()
                        .map(|v| v.into_iter().map(f64::to_bits).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(gated.final_x.clone()), bits(plain.final_x.clone()));
                assert_eq!(
                    gated
                        .final_bounds
                        .iter()
                        .map(|b| b.ell_hat.to_bits())
                        .collect::<Vec<_>>(),
                    plain
                        .final_bounds
                        .iter()
                        .map(|b| b.ell_hat.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn gate_evaluation_preserves_every_solver_workspace_field() {
        let task = tied_task();
        let validated = validate(&task).unwrap();
        let mut solvers: Vec<Solver> = task.rhs.iter().map(|b| Solver::new(b)).collect();
        advance(&mut solvers, &task, 4, 1.0e-14).unwrap();
        let snapshot = |values: &[Solver]| {
            values
                .iter()
                .map(|s| {
                    (
                        s.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        s.r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        s.p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        s.ap.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        s.rr.to_bits(),
                        s.rr0.to_bits(),
                        s.iterations,
                    )
                })
                .collect::<Vec<_>>()
        };
        let before = snapshot(&solvers);
        for kind in [
            EvaluationKind::Plain,
            EvaluationKind::Fast,
            EvaluationKind::Outward,
        ] {
            measure_bounds(
                &task,
                &solvers,
                &validated,
                &mut Vec::new(),
                &mut None,
                kind,
                &mut EvaluationCounts::default(),
            )
            .unwrap();
            assert_eq!(snapshot(&solvers), before);
        }
    }
    #[test]
    fn supported_extreme_norms_and_mixed_subnormals_remain_finite() {
        for exponent in [-400, 100] {
            let value = 2.0f64.powi(exponent);
            let b = vec![value, f64::from_bits(1), value / 2.0, -value / 4.0];
            let task = Task {
                id: format!("edge_{exponent}"),
                side: 2,
                rho: 64,
                scale: 2.0f64.powi(2 * exponent + 2),
                rhs: vec![b.clone(), b],
            };
            let response = execute(Request {
                task,
                additional: 9,
                repeats: 1,
                warmup: 0,
                trace: true,
            })
            .unwrap();
            for out in response.results {
                assert!(!out.certified);
                assert!(out
                    .final_bounds
                    .iter()
                    .all(|b| b.lo <= b.hi && b.lo >= 0.0 && b.hi <= 1.0 && b.ell_hat.is_finite()));
                assert!(out.residual_relative.iter().all(|r| r.is_finite()));
                if let Some(upper) = out.residual_upper {
                    assert!(upper.iter().all(|r| r.is_finite() && *r >= 0.0));
                    assert_eq!(
                        out.residual_goal_met,
                        Some(upper.iter().all(|r| *r <= RESIDUAL_TOL))
                    );
                } else {
                    assert_eq!(out.residual_goal_met, None);
                }
            }
        }
    }
    #[test]
    fn exact_zero_problem_requires_zero_state_for_every_readout() {
        let task = Task {
            id: "zero_guard".into(),
            side: 2,
            rho: 64,
            scale: f64::MIN_POSITIVE,
            rhs: vec![vec![0.0; 4]; 2],
        };
        let validation = validate(&task).unwrap();
        for kind in [
            EvaluationKind::Plain,
            EvaluationKind::Fast,
            EvaluationKind::Outward,
        ] {
            let nonzero = vec![f64::from_bits(1), 0.0, 0.0, 0.0];
            assert!(evaluate(
                &task,
                &task.rhs[0],
                &nonzero,
                validation.norms[0],
                &validation,
                kind
            )
            .is_err());
            let good = evaluate(
                &task,
                &task.rhs[0],
                &[0.0; 4],
                validation.norms[0],
                &validation,
                kind,
            )
            .unwrap();
            assert_eq!(good.approximate_relative, 0.0);
            if kind != EvaluationKind::Plain {
                assert_eq!(good.residual_upper, Some(0.0));
            }
        }
    }
    #[test]
    fn rho_zero_enclosures_cover_exact_energy_at_arbitrary_iterates() {
        let b = vec![0.5, -0.25, 0.125, -0.0625];
        let task = Task {
            id: "identity".into(),
            side: 2,
            rho: 0,
            scale: 1.0,
            rhs: vec![b.clone(), b.clone()],
        };
        let validated = validate(&task).unwrap();
        // All entries and exact q are dyadic; this value is exact in binary64.
        let exact_q = 0.25 + 0.0625 + 0.015625 + 0.00390625;
        for x in [vec![0.0; 4], b, vec![-1.0, 0.5, 0.25, -0.125]] {
            for kind in [EvaluationKind::Fast, EvaluationKind::Outward] {
                let result = evaluate(
                    &task,
                    &task.rhs[0],
                    &x,
                    validated.norms[0],
                    &validated,
                    kind,
                )
                .unwrap();
                assert!(
                    result.bound.lo <= exact_q && exact_q <= result.bound.hi,
                    "{kind:?} {:?}",
                    result.bound
                );
                let exact_d: f64 = task.rhs[0]
                    .iter()
                    .zip(&x)
                    .map(|(a, c)| (a - c) * (a - c))
                    .sum();
                let upper = result.residual_upper.unwrap();
                assert!(upper * upper * exact_q >= exact_d);
            }
        }
    }
    #[test]
    fn positive_square_sum_encloses_exact_small_dyadic_norm() {
        let exact = 2.0f64.powi(-800);
        let bound = positive_square_sum(exact, ErrorBudget::new(8).unwrap()).unwrap();
        assert!(bound.lo > 0.0 && bound.lo <= exact && bound.hi >= exact);
        let zero = positive_square_sum(0.0, ErrorBudget::new(8).unwrap()).unwrap();
        assert_eq!(zero.lo, 0.0);
        assert!(zero.hi > 0.0);
    }
    #[test]
    fn residual_upper_uses_sqrt_before_division() {
        let norm = InputNorm {
            max_abs: RHS_MIN,
            ordinary: 2.0f64.powi(-800),
            bounds: Interval::point(2.0f64.powi(-800)),
        };
        let du = 2.0f64.powi(440);
        assert!(
            (du / norm.bounds.lo).is_infinite(),
            "fixture must expose squared-ratio overflow"
        );
        let upper = certified_relative_upper(du, norm).unwrap();
        assert!(upper.is_finite() && upper >= 2.0f64.powi(620));
    }
    #[test]
    fn conservative_scale_and_absolute_state_cap_are_enforced() {
        let mut task = Task {
            id: "domain".into(),
            side: 2,
            rho: 1,
            scale: 1.0,
            rhs: vec![vec![1.0, 0.0, 0.0, 0.0]; 2],
        };
        assert!(
            validate(&task).is_err(),
            "exact norm is not enough if the rigorous upper exceeds S"
        );
        task.scale = 2.0;
        let validated = validate(&task).unwrap();
        for kind in [
            EvaluationKind::Plain,
            EvaluationKind::Fast,
            EvaluationKind::Outward,
        ] {
            assert!(evaluate(
                &task,
                &task.rhs[0],
                &[2.0f64.powi(201), 0.0, 0.0, 0.0],
                validated.norms[0],
                &validated,
                kind
            )
            .is_err());
        }
    }
    #[test]
    fn counters_traces_and_final_diagnostics_match_actual_readout_kind() {
        let response = execute(Request {
            task: tied_task(),
            additional: 9,
            repeats: 1,
            warmup: 0,
            trace: true,
        })
        .unwrap();
        for out in response.results {
            let trace = out.trace.as_ref().unwrap();
            assert_eq!(
                out.bound_evaluations,
                out.fast_evaluations + out.outward_evaluations
            );
            assert_eq!(
                out.bound_evaluations + out.ordinary_evaluations,
                2 * trace.len()
            );
            assert!(
                out.validation_us.is_finite()
                    && out.validation_us >= 0.0
                    && out.total_us >= out.validation_us
            );
            assert_eq!(trace.last().unwrap().iterations, out.iterations);
            if out.policy.starts_with("PLAIN") {
                assert_eq!(out.bound_evaluations, 0);
                assert_eq!(out.residual_upper, None);
                assert_eq!(out.residual_goal_met, None);
                assert!(out.final_bounds.iter().all(|b| b.lo == 0.0 && b.hi == 1.0));
            } else if out.policy == "FAST_ONCE" {
                assert_eq!(out.fast_evaluations, 2);
                assert_eq!(out.ordinary_evaluations, 2);
                assert_eq!(out.residual_upper, None);
                assert_eq!(out.residual_goal_met, None);
            } else {
                assert_eq!(out.ordinary_evaluations, 0);
                assert_eq!(
                    out.residual_goal_met,
                    Some(
                        out.residual_upper
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|r| *r <= RESIDUAL_TOL)
                    )
                );
            }
            for adjacent in trace.windows(2) {
                for (old, new) in adjacent[0].bounds.iter().zip(&adjacent[1].bounds) {
                    assert!(new.lo >= old.lo && new.hi <= old.hi);
                }
            }
        }
    }
    #[test]
    fn trace_false_omits_vector_copies_but_keeps_plain_null_diagnostics() {
        let response = execute(Request {
            task: tied_task(),
            additional: 1,
            repeats: 1,
            warmup: 0,
            trace: false,
        })
        .unwrap();
        let value = serde_json::to_value(response).unwrap();
        assert!(value.get("input_rhs_bits").is_none());
        assert!(value.get("input_scale_bits").is_none());
        for out in value["results"].as_array().unwrap() {
            assert!(out.get("trace").is_none());
            assert!(out.get("final_x").is_none());
            if out["policy"].as_str().unwrap().starts_with("PLAIN") {
                assert!(out["residual_upper"].is_null());
                assert!(out["residual_goal_met"].is_null());
            }
        }
    }
}
