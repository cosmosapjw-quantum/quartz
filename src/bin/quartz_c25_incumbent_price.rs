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
#[allow(dead_code)]
fn c24_main() {
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


// C25 extends only the controller/output contract; the complete C24 numerical
// implementation and inherited tests above are reused without arithmetic edits.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct C25Request {
    task: Task,
    additional: usize,
    #[serde(default = "one")]
    repeats: usize,
    #[serde(default)]
    warmup: usize,
    #[serde(default)]
    trace: bool,
    #[serde(default)]
    predicted_block_us: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum C25Policy {
    PlainStop,
    PlainAlways,
    FastBlock,
    RetainBlock,
    RetainWork,
    RetainTime,
}
impl C25Policy {
    fn label(self) -> &'static str {
        match self {
            Self::PlainStop => "PLAIN_STOP",
            Self::PlainAlways => "PLAIN_ALWAYS",
            Self::FastBlock => "FAST_BLOCK",
            Self::RetainBlock => "RETAIN_BLOCK",
            Self::RetainWork => "RETAIN_WORK",
            Self::RetainTime => "RETAIN_TIME",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum C25Reason {
    Continue,
    InitialCertificate,
    BlockCertificate,
    ZeroRegretOptimum,
    Budget,
    NoProgress,
    PriceWork,
    PriceTime,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum C25CostKind {
    None,
    CellStencil,
    TrainPredictedUs,
}
#[derive(Clone, Debug, Serialize)]
struct C25SwitchProof {
    from: usize,
    to: usize,
    challenger_lower: f64,
    incumbent_upper: f64,
}
#[derive(Clone, Debug, Serialize)]
struct C25Decision {
    checkpoint: usize,
    incumbent_before: usize,
    incumbent_after: usize,
    gain_upper: f64,
    price_threshold: Option<f64>,
    cost_kind: C25CostKind,
    reason: C25Reason,
    switch_proof: Option<C25SwitchProof>,
    executed_blocks: usize,
    charged_cell_stencils: usize,
}
#[derive(Clone, Debug, Serialize)]
struct C25Metadata {
    executed_blocks: usize,
    continuation_us: Option<f64>,
    charged_cell_stencils: usize,
    final_gain_upper: Option<f64>,
    weakly_optimal: Option<bool>,
    final_ordinary_selected: usize,
    price_stop_certified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    decisions: Option<Vec<C25Decision>>,
}
#[derive(Clone, Debug, Serialize)]
struct C25Outcome {
    #[serde(flatten)]
    base: Outcome,
    c25: C25Metadata,
}
#[derive(Debug, Serialize)]
struct C25Response {
    id: String,
    additional: usize,
    results: Vec<C25Outcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_rhs_bits: Option<Vec<Vec<u64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_scale_bits: Option<u64>,
}
fn c25_control_metadata(out: &Outcome, side: usize, actions: usize) -> C25Metadata {
    C25Metadata {
        executed_blocks: (out.fast_evaluations
            + out.ordinary_evaluations
            + out.outward_evaluations)
            / actions
            - 1,
        continuation_us: None,
        charged_cell_stencils: side
            * side
            * (out.solver_matvecs + out.fast_evaluations + out.ordinary_evaluations),
        final_gain_upper: None,
        weakly_optimal: None,
        final_ordinary_selected: score_action(&out.final_bounds),
        price_stop_certified: false,
        decisions: None,
    }
}
// RETAIN operates on inherited historical enclosures, never on oracle inputs.
impl C25Reason {
    fn label(self) -> &'static str {
        match self {
            Self::Continue => "CONTINUE",
            Self::InitialCertificate => "INITIAL_CERTIFICATE",
            Self::BlockCertificate => "BLOCK_CERTIFICATE",
            Self::ZeroRegretOptimum => "ZERO_REGRET_OPTIMUM",
            Self::Budget => "BUDGET",
            Self::NoProgress => "NO_PROGRESS",
            Self::PriceWork => "PRICE_WORK",
            Self::PriceTime => "PRICE_TIME",
        }
    }
}
fn c25_decide(
    bounds: &[Bounds],
    incumbent: usize,
    policy: C25Policy,
    prediction: f64,
    floor: usize,
    remaining: usize,
    last_advanced: Option<usize>,
    checkpoint: usize,
    charged: usize,
) -> C25Decision {
    let mut candidate = 0;
    for j in 1..bounds.len() {
        if bounds[j].lo > bounds[candidate].lo {
            candidate = j;
        }
    }
    let switch_proof = if bounds[candidate].lo > bounds[incumbent].hi {
        Some(C25SwitchProof {
            from: incumbent,
            to: candidate,
            challenger_lower: bounds[candidate].lo,
            incumbent_upper: bounds[incumbent].hi,
        })
    } else {
        None
    };
    let selected = switch_proof.as_ref().map_or(incumbent, |proof| proof.to);
    let other_upper = bounds
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != selected)
        .map(|(_, b)| b.hi)
        .fold(0.0, f64::max);
    let gain_upper = if other_upper <= bounds[selected].lo {
        0.0
    } else {
        (other_upper - bounds[selected].lo).next_up()
    };
    let unique = bounds
        .iter()
        .enumerate()
        .all(|(j, b)| j == selected || bounds[selected].lo > b.hi);
    // Price is always rounded downward. The wall price lower enclosure is
    // conservative for the exact rational 1/1_000_000, not a cost theorem.
    let (price_threshold, cost_kind) = match policy {
        C25Policy::RetainWork => {
            let price = f64::from_bits(993u64 << 52); // exactly 2^-30
            (
                Some((price * floor as f64).next_down().max(0.0)),
                C25CostKind::CellStencil,
            )
        }
        C25Policy::RetainTime => {
            let price_lower = 1.0e-6_f64.next_down();
            (
                Some((price_lower * prediction).next_down().max(0.0)),
                C25CostKind::TrainPredictedUs,
            )
        }
        _ => (None, C25CostKind::None),
    };
    let reason = if unique {
        if checkpoint == 0 {
            C25Reason::InitialCertificate
        } else {
            C25Reason::BlockCertificate
        }
    } else if gain_upper == 0.0 {
        C25Reason::ZeroRegretOptimum
    } else if remaining == 0 {
        C25Reason::Budget
    } else if last_advanced == Some(0) {
        C25Reason::NoProgress
    } else if price_threshold.is_some_and(|threshold| gain_upper <= threshold) {
        if policy == C25Policy::RetainWork {
            C25Reason::PriceWork
        } else {
            C25Reason::PriceTime
        }
    } else {
        C25Reason::Continue
    };
    C25Decision {
        checkpoint,
        incumbent_before: incumbent,
        incumbent_after: selected,
        gain_upper,
        price_threshold,
        cost_kind,
        reason,
        switch_proof,
        executed_blocks: checkpoint,
        charged_cell_stencils: charged,
    }
}
fn run_c25_retention(
    task: &Task,
    additional: usize,
    policy: C25Policy,
    prediction: f64,
    want_trace: bool,
) -> Result<C25Outcome, String> {
    let total_start = Instant::now();
    if !prediction.is_finite() || prediction < 0.0 {
        return Err("predicted_block_us must be finite and nonnegative".into());
    }
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
    let bound_start = Instant::now();
    let mut measured = measure_bounds(
        task,
        &solvers,
        &validated,
        &mut history,
        &mut trace,
        EvaluationKind::Fast,
        &mut counts,
    )?;
    let mut bounds_us = micros(bound_start);
    let initial_bounds = measured.fresh.clone();
    let initial_selected = score_action(&measured.fresh);
    let mut selected = initial_selected;
    // Excludes validation, initial solver and initial FAST readout. Includes
    // all subsequent control, CG, fresh readouts and workspace destruction.
    let continuation_start = Instant::now();
    let mut decisions = if want_trace { Some(Vec::new()) } else { None };
    let n = task.side * task.side;
    let floor = task.rhs.len() * n;
    let mut used = 0;
    let mut executed_blocks = 0;
    let mut last_advanced = None;
    let mut solve_us = 0.0;
    let final_decision = loop {
        let solver_matvecs: usize = solvers.iter().map(|s| s.iterations).sum();
        let charged = n * (solver_matvecs + counts.fast + counts.ordinary);
        let decision = c25_decide(
            &history,
            selected,
            policy,
            prediction,
            floor,
            additional - used,
            last_advanced,
            executed_blocks,
            charged,
        );
        selected = decision.incumbent_after;
        if let Some(points) = &mut decisions {
            points.push(decision.clone());
        }
        if decision.reason != C25Reason::Continue {
            break decision;
        }
        let batch = 8.min(additional - used);
        let start = Instant::now();
        let advanced = advance(&mut solvers, task, batch, 1.0e-14)?;
        solve_us += micros(start);
        used += batch;
        executed_blocks += 1;
        last_advanced = Some(advanced);
        // This unconditional all-K readout is part of the admitted action.
        // In particular, advanced==0 does not bypass its K*n charged cost.
        let start = Instant::now();
        measured = measure_bounds(
            task,
            &solvers,
            &validated,
            &mut history,
            &mut trace,
            EvaluationKind::Fast,
            &mut counts,
        )?;
        bounds_us += micros(start);
    };
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
    let final_ordinary_selected = score_action(&measured.fresh);
    let base = Outcome {
        policy: policy.label().into(),
        rep: 0,
        selected,
        initial_selected,
        certified: matches!(
            final_decision.reason,
            C25Reason::InitialCertificate | C25Reason::BlockCertificate
        ),
        reason: final_decision.reason.label().into(),
        iterations,
        solver_matvecs,
        fast_evaluations: counts.fast,
        outward_evaluations: 0,
        bound_evaluations: counts.fast,
        ordinary_evaluations: counts.ordinary,
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
    let c25 = C25Metadata {
        executed_blocks,
        continuation_us: Some(0.0),
        charged_cell_stencils: n * (solver_matvecs + counts.fast + counts.ordinary),
        final_gain_upper: Some(final_decision.gain_upper),
        weakly_optimal: Some(final_decision.gain_upper == 0.0),
        final_ordinary_selected,
        price_stop_certified: final_decision.reason == C25Reason::PriceWork,
        decisions,
    };
    let mut out = C25Outcome { base, c25 };
    drop(solvers);
    drop(validated);
    out.c25.continuation_us = Some(micros(continuation_start));
    out.base.total_us = micros(total_start);
    Ok(out)
}
fn run_c25_policy(
    task: &Task,
    additional: usize,
    policy: C25Policy,
    prediction: f64,
    want_trace: bool,
) -> Result<C25Outcome, String> {
    let old = match policy {
        C25Policy::PlainStop => Some(Policy::PlainStop),
        C25Policy::PlainAlways => Some(Policy::PlainAlways),
        C25Policy::FastBlock => Some(Policy::FastBlock),
        _ => None,
    };
    if let Some(old) = old {
        let start = Instant::now();
        let base = run_policy(task, additional, old, want_trace)?;
        let c25 = c25_control_metadata(&base, task.side, task.rhs.len());
        let mut out = C25Outcome { base, c25 };
        out.base.total_us = micros(start);
        Ok(out)
    } else {
        run_c25_retention(task, additional, policy, prediction, want_trace)
    }
}
fn execute_c25(req: C25Request) -> Result<C25Response, String> {
    if req.repeats == 0 || req.repeats > 32 || req.warmup > 4 || req.additional > 128 {
        return Err("request repeat/warmup/budget cap violation".into());
    }
    if !req.predicted_block_us.is_finite() || req.predicted_block_us < 0.0 {
        return Err("predicted_block_us must be finite and nonnegative".into());
    }
    let policies = [
        C25Policy::PlainStop,
        C25Policy::PlainAlways,
        C25Policy::FastBlock,
        C25Policy::RetainBlock,
        C25Policy::RetainWork,
        C25Policy::RetainTime,
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
            let mut out = run_c25_policy(
                &req.task,
                req.additional,
                policy,
                req.predicted_block_us,
                req.trace,
            )?;
            if pass >= req.warmup {
                out.base.rep = pass - req.warmup;
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
    Ok(C25Response {
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
            serde_json::from_str::<C25Request>(&s)
                .map_err(|e| e.to_string())
                .and_then(execute_c25)
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
#[cfg(test)]
mod c25_behavior_tests {
    use super::*;
    fn bound(lo: f64, hi: f64, score: f64) -> Bounds {
        Bounds {
            lo,
            hi,
            ell_hat: score,
        }
    }
    fn tied_task(rho: u32) -> Task {
        let b: Vec<f64> = (0..64)
            .map(|i| (((i * 17) % 23) as f64 - 11.0) / 16.0)
            .collect();
        Task {
            id: "c25_tied".into(),
            side: 8,
            rho,
            scale: 64.0,
            rhs: vec![b.clone(), b],
        }
    }
    fn decide(
        bounds: &[Bounds],
        policy: C25Policy,
        prediction: f64,
        remaining: usize,
        last_advanced: Option<usize>,
    ) -> C25Decision {
        c25_decide(
            bounds,
            0,
            policy,
            prediction,
            8,
            remaining,
            last_advanced,
            0,
            8,
        )
    }
    // Defect: an uncertified final ordinary score replaces a better incumbent.
    #[test]
    fn retained_budget_rejects_a_worse_ordinary_reselection() {
        let b = [
            bound(271.0 / 1800.0, 131.0 / 200.0, 7.0 / 80.0),
            bound(47.0 / 400.0, 63.0 / 400.0, 9.0 / 80.0),
        ];
        let d = decide(&b, C25Policy::RetainBlock, 0.0, 0, None);
        assert_eq!(d.incumbent_after, 0);
        assert_eq!(d.reason, C25Reason::Budget);
        assert!(d.switch_proof.is_none());
        assert!(d.gain_upper >= 1.0 / 144.0);
    }
    // Defect: price stopping happens before an already certified free switch.
    #[test]
    fn strict_max_lower_switch_precedes_price_and_is_auditable() {
        let b = [
            bound(0.125, 0.25, 0.125),
            bound(0.375, 0.5, 0.1),
            bound(0.75, 0.875, 0.0),
        ];
        let d = decide(&b, C25Policy::RetainTime, 1.0e9, 8, None);
        assert_eq!(d.incumbent_after, 2);
        let proof = d.switch_proof.unwrap();
        assert_eq!((proof.from, proof.to), (0, 2));
        assert_eq!(proof.challenger_lower, 0.75);
        assert_eq!(proof.incumbent_upper, 0.25);
        assert_eq!(d.reason, C25Reason::InitialCertificate);
        assert_eq!(d.gain_upper, 0.0);
    }
    // Defect: equality is silently accepted as a strict utility improvement.
    #[test]
    fn endpoint_equality_does_not_switch_and_lower_ties_use_first_index() {
        let equality = [bound(0.0, 0.5, 0.0), bound(0.5, 0.75, 0.7)];
        let d = decide(&equality, C25Policy::RetainBlock, 0.0, 8, None);
        assert_eq!(d.incumbent_after, 0);
        assert!(d.switch_proof.is_none());
        let tied = [
            bound(0.0, 0.125, 0.0),
            bound(0.5, 0.75, 0.0),
            bound(0.5, 0.875, 1.0),
        ];
        let d = decide(&tied, C25Policy::RetainBlock, 0.0, 8, None);
        assert_eq!(d.incumbent_after, 1);
    }
    // Defect: ties are claimed uniquely optimal, or gain rounds zero upward.
    #[test]
    fn zero_gap_certifies_weak_optimality_without_a_unique_winner() {
        let b = [bound(0.25, 0.25, 0.25), bound(0.25, 0.25, 0.25)];
        let d = decide(&b, C25Policy::RetainWork, 0.0, 8, None);
        assert_eq!(d.reason, C25Reason::ZeroRegretOptimum);
        assert_eq!(d.gain_upper.to_bits(), 0.0f64.to_bits());
        assert!(d.switch_proof.is_none());
    }
    // Defect: nearest arithmetic admits a false stop exactly at the price edge.
    #[test]
    fn work_gate_uses_outward_gap_and_inward_exact_dyadic_threshold() {
        let exact = f64::from_bits(996u64 << 52); // 8 * 2^-30 = 2^-27.
        let b = [bound(0.0, 1.0, 1.0), bound(0.0, exact, 0.0)];
        let d = decide(&b, C25Policy::RetainWork, 0.0, 8, None);
        assert_eq!(d.gain_upper.to_bits(), exact.to_bits() + 1);
        assert_eq!(d.price_threshold.unwrap().to_bits(), exact.to_bits() - 1);
        assert_eq!(d.reason, C25Reason::Continue);
        let inside = [
            bound(0.0, 1.0, 1.0),
            bound(0.0, f64::from_bits(exact.to_bits() - 2), 0.0),
        ];
        let d = decide(&inside, C25Policy::RetainWork, 0.0, 8, None);
        assert_eq!(d.reason, C25Reason::PriceWork);
        assert_eq!(d.cost_kind, C25CostKind::CellStencil);
    }
    // Defect: predicted wall price is presented as deterministic cost, or 0 fails.
    #[test]
    fn time_gate_is_separate_and_zero_prediction_disables_positive_stop() {
        let b = [bound(0.0, 1.0, 1.0), bound(0.0, 2.0f64.powi(-30), 0.0)];
        let off = decide(&b, C25Policy::RetainTime, 0.0, 8, None);
        assert_eq!(off.reason, C25Reason::Continue);
        assert_eq!(off.price_threshold, Some(0.0));
        let on = decide(&b, C25Policy::RetainTime, 1.0, 8, None);
        assert_eq!(on.reason, C25Reason::PriceTime);
        assert_eq!(on.cost_kind, C25CostKind::TrainPredictedUs);
        // Independent bit value: lower neighbor twice below nearest decimal.
        assert_eq!(on.price_threshold.unwrap().to_bits(), 0x3eb0c6f7a0b5ed8b);
    }
    // Defect: exhaustion is reported as a price optimum over unavailable blocks.
    #[test]
    fn budget_and_zero_progress_precede_positive_price_gate() {
        let b = [bound(0.0, 1.0, 1.0), bound(0.0, 2.0f64.powi(-30), 0.0)];
        assert_eq!(
            decide(&b, C25Policy::RetainWork, 0.0, 0, None).reason,
            C25Reason::Budget
        );
        assert_eq!(
            decide(&b, C25Policy::RetainWork, 0.0, 8, Some(0)).reason,
            C25Reason::NoProgress
        );
    }
    // Defect: an admitted zero-CG block skips FAST and violates K*n cost floor.
    #[test]
    fn zero_progress_block_still_reads_every_action_once() {
        let t = tied_task(0);
        let out = run_c25_policy(&t, 16, C25Policy::RetainBlock, 0.0, true).unwrap();
        assert_eq!(out.c25.executed_blocks, 1);
        assert_eq!(out.base.fast_evaluations, 4);
        assert_eq!(out.base.reason, "NO_PROGRESS");
        let ds = out.c25.decisions.as_ref().unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(
            ds[1].charged_cell_stencils - ds[0].charged_cell_stencils,
            128
        );
        assert!(out.c25.continuation_us.unwrap() >= 0.0);
        assert_eq!(
            out.c25.charged_cell_stencils,
            64 * (out.base.solver_matvecs + out.base.fast_evaluations)
        );
    }
    // Defect: unavailable budget performs new readouts or drops retain diagnostics.
    #[test]
    fn zero_budget_keeps_initial_action_and_exposes_one_checkpoint() {
        let out = run_c25_policy(&tied_task(16), 0, C25Policy::RetainBlock, 0.0, true).unwrap();
        assert_eq!(out.base.selected, out.base.initial_selected);
        assert_eq!(out.base.reason, "BUDGET");
        assert_eq!(out.c25.executed_blocks, 0);
        assert_eq!(out.base.fast_evaluations, 2);
        assert_eq!(out.c25.decisions.as_ref().unwrap().len(), 1);
        assert_eq!(
            out.c25.final_gain_upper,
            Some(out.c25.decisions.as_ref().unwrap()[0].gain_upper)
        );
    }
    // Defect: public suite retains obsolete controls or fails cyclic interleaving.
    #[test]
    fn public_suite_runs_exact_six_policies_with_cyclic_repetitions() {
        let out = execute_c25(C25Request {
            task: tied_task(16),
            additional: 8,
            repeats: 2,
            warmup: 1,
            trace: true,
            predicted_block_us: 0.0,
        })
        .unwrap();
        assert_eq!(out.results.len(), 12);
        let wanted = [
            "PLAIN_STOP",
            "PLAIN_ALWAYS",
            "FAST_BLOCK",
            "RETAIN_BLOCK",
            "RETAIN_WORK",
            "RETAIN_TIME",
        ];
        let first: Vec<_> = out
            .results
            .iter()
            .filter(|o| o.base.rep == 0)
            .map(|o| o.base.policy.as_str())
            .collect();
        let second: Vec<_> = out
            .results
            .iter()
            .filter(|o| o.base.rep == 1)
            .map(|o| o.base.policy.as_str())
            .collect();
        for policy in wanted {
            assert!(first.contains(&policy), "missing {policy}");
        }
        assert_eq!(
            second,
            first
                .iter()
                .cycle()
                .skip(1)
                .take(6)
                .copied()
                .collect::<Vec<_>>()
        );
        let initial = out.results[0].base.initial_selected;
        assert!(out
            .results
            .iter()
            .all(|o| o.base.initial_selected == initial));
    }
    // Defect: invalid cost prediction or oracle data enters the online interface.
    #[test]
    fn public_input_rejects_negative_nonfinite_prediction_and_oracle_fields() {
        for prediction in [-1.0, f64::INFINITY, f64::NAN] {
            let r = execute_c25(C25Request {
                task: tied_task(16),
                additional: 8,
                repeats: 1,
                warmup: 0,
                trace: false,
                predicted_block_us: prediction,
            });
            assert!(r.is_err(), "prediction {prediction:?} must reject");
        }
        let raw = r#"{"task":{"id":"x","side":2,"rho":0,"scale":2,"rhs":[[1,0,0,0],[0,0,0,0]]},"additional":8,"predicted_block_us":0,"oracle":[1,0]}"#;
        assert!(serde_json::from_str::<C25Request>(raw).is_err());
    }
    // Defect: new controller changes the shared 4-step ordinary action or controls.
    // This is a reused invariant; no new-behavior RED claim is made for this test.
    #[test]
    fn inherited_controls_and_initial_scores_are_identical() {
        let task = tied_task(16);
        for (new, old) in [
            (C25Policy::PlainStop, Policy::PlainStop),
            (C25Policy::PlainAlways, Policy::PlainAlways),
            (C25Policy::FastBlock, Policy::FastBlock),
        ] {
            let got = run_c25_policy(&task, 9, new, 0.0, true).unwrap();
            let expected = run_policy(&task, 9, old, true).unwrap();
            assert_eq!(got.base.selected, expected.selected);
            assert_eq!(got.base.iterations, expected.iterations);
            assert_eq!(got.base.final_x, expected.final_x);
            assert_eq!(got.base.fast_evaluations, expected.fast_evaluations);
            assert_eq!(got.base.ordinary_evaluations, expected.ordinary_evaluations);
            assert!(got.c25.continuation_us.is_none());
        }
        let plain = run_policy(&task, 9, Policy::PlainStop, true).unwrap();
        for mode in [
            C25Policy::RetainBlock,
            C25Policy::RetainWork,
            C25Policy::RetainTime,
        ] {
            let got = run_c25_policy(&task, 9, mode, 0.0, true).unwrap();
            assert_eq!(got.base.initial_selected, plain.initial_selected);
            assert_eq!(
                got.base
                    .initial_bounds
                    .iter()
                    .map(|b| b.ell_hat.to_bits())
                    .collect::<Vec<_>>(),
                plain
                    .initial_bounds
                    .iter()
                    .map(|b| b.ell_hat.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    // Defect: timing mode materializes vectors or omits required final diagnostics.
    #[test]
    fn trace_off_omits_vectors_and_preserves_compact_decision_result() {
        let out = run_c25_policy(&tied_task(16), 9, C25Policy::RetainBlock, 0.0, false).unwrap();
        let value = serde_json::to_value(out).unwrap();
        assert!(value.get("trace").is_none());
        assert!(value.get("final_x").is_none());
        assert!(value["c25"].get("decisions").is_none());
        assert!(value["c25"]["final_gain_upper"].is_number());
        assert!(value["c25"]["continuation_us"].is_number());
        assert!(value["c25"]["weakly_optimal"].is_boolean());
    }
}
