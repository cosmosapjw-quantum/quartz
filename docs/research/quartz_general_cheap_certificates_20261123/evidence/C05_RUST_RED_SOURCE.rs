//! C23 isolated numerical computation-choice experiment. No game engine integration.
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    side: usize,
    rho: u32,
    #[serde(deserialize_with="deserialize_scalar")]
    scale: f64,
    #[serde(deserialize_with="deserialize_rhs")]
    rhs: Vec<Vec<f64>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    task: Task,
    additional: usize,
    #[serde(default = "one")] repeats: usize,
    #[serde(default)] warmup: usize,
    #[serde(default)] trace: bool,
}
fn one() -> usize { 1 }

// The study uses decimal strings and Rust's correctly rounded str::parse.
// Numeric JSON remains a compatibility path; its input bits must not be
// assumed identical to an independently manufactured binary64 source.
#[derive(Deserialize)]
#[serde(untagged)]
enum JsonScalar { Number(f64), Text(String) }
impl JsonScalar {
    fn value(self)->Result<f64,String> {
        match self {
            Self::Number(x)=>Ok(x),
            Self::Text(s)=>{
                if s.len()>64 {return Err("scalar string exceeds64bytes".into());}
                s.parse::<f64>().map_err(|e|format!("invalid scalar string: {e}"))
            }
        }
    }
}
fn deserialize_scalar<'de,D:serde::Deserializer<'de>>(deserializer:D)->Result<f64,D::Error> {
    JsonScalar::deserialize(deserializer)?.value().map_err(serde::de::Error::custom)
}
fn deserialize_rhs<'de,D:serde::Deserializer<'de>>(deserializer:D)->Result<Vec<Vec<f64>>,D::Error> {
    let raw=Vec::<Vec<JsonScalar>>::deserialize(deserializer)?;
    raw.into_iter().map(|row|row.into_iter().map(|v|v.value().map_err(serde::de::Error::custom)).collect()).collect()
}

#[derive(Clone, Copy, Debug)]
struct Interval { lo: f64, hi: f64 }
impl Interval {
    fn point(x: f64) -> Self { Self {lo:x,hi:x} }
    fn invalid() -> Self { Self {lo:f64::NAN,hi:f64::NAN} }
    fn finite(self) -> bool { self.lo.is_finite() && self.hi.is_finite() && self.lo<=self.hi }
    fn add(self, b:Self) -> Self {
        if !self.finite()||!b.finite() { return Self::invalid(); }
        Self{lo:(self.lo+b.lo).next_down(),hi:(self.hi+b.hi).next_up()}
    }
    fn sub(self, b:Self) -> Self {
        if !self.finite()||!b.finite() { return Self::invalid(); }
        Self{lo:(self.lo-b.hi).next_down(),hi:(self.hi-b.lo).next_up()}
    }
    fn mul(self, b:Self) -> Self {
        if !self.finite()||!b.finite() { return Self::invalid(); }
        let p=[self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi];
        if p.iter().any(|v|!v.is_finite()) { return Self::invalid(); }
        Self{lo:p.iter().copied().fold(f64::INFINITY,f64::min).next_down(),
             hi:p.iter().copied().fold(f64::NEG_INFINITY,f64::max).next_up()}
    }
    fn square(self) -> Self {
        if !self.finite() { return Self::invalid(); }
        let a=self.lo*self.lo;let b=self.hi*self.hi;
        if !a.is_finite()||!b.is_finite() { return Self::invalid(); }
        Self{lo:if self.lo<=0.0&&self.hi>=0.0 {0.0}else{a.min(b).next_down().max(0.0)},
             hi:a.max(b).next_up()}
    }
    fn div_positive(self,d:f64) -> Self {
        if !self.finite()||!d.is_finite()||d<=0.0 { return Self::invalid(); }
        Self{lo:(self.lo/d).next_down(),hi:(self.hi/d).next_up()}
    }
}
#[derive(Clone, Debug, Serialize)]
struct Bounds { lo: f64, hi: f64, ell_hat: f64 }
#[derive(Clone, Debug, Serialize)]
struct Checkpoint { iterations: Vec<usize>, fresh_bounds: Vec<Bounds>, bounds: Vec<Bounds> }
#[derive(Clone, Copy, Debug, PartialEq)]
enum Policy { Never, Always, CertOnce, CertBlock, Residual, PlainAlways }
impl Policy {
    fn label(self)->&'static str {
        match self {Self::Never=>"NEVER",Self::Always=>"ALWAYS",Self::CertOnce=>"CERT_ONCE",
            Self::CertBlock=>"CERT_BLOCK",Self::Residual=>"RESIDUAL",Self::PlainAlways=>"PLAIN_ALWAYS"}
    }
}
#[derive(Clone, Debug, Serialize)]
struct Outcome {
    policy: String, rep: usize, selected: usize, initial_selected: usize,
    certified: bool, reason: String, iterations: Vec<usize>, solver_matvecs: usize,
    bound_evaluations: usize, ordinary_evaluations: usize, total_us: f64, initial_us: f64, solve_us: f64, bounds_us: f64,
    initial_bounds: Vec<Bounds>, final_bounds: Vec<Bounds>, decision_bounds: Vec<Bounds>,
    residual_relative: Vec<f64>, residual_goal_met: bool,
    #[serde(skip_serializing_if="Option::is_none")] trace: Option<Vec<Checkpoint>>,
    #[serde(skip_serializing_if="Option::is_none")] final_x: Option<Vec<Vec<f64>>>,
}
#[derive(Debug, Serialize)]
struct Response {
    id:String, additional:usize, results:Vec<Outcome>,
    #[serde(skip_serializing_if="Option::is_none")]
    input_rhs_bits: Option<Vec<Vec<u64>>>,
    #[serde(skip_serializing_if="Option::is_none")]
    input_scale_bits: Option<u64>,
}

fn dot(x:&[f64],y:&[f64])->f64 { x.iter().zip(y).map(|(a,b)|a*b).sum() }
fn apply_operator(side:usize,rho:u32,x:&[f64],out:&mut[f64]) {
    let r=rho as f64;let d=1.0+4.0*r;
    for i in 0..x.len() {
        let row=i/side;let col=i%side;
        let mut y=d*x[i];
        if row>0 {y-=r*x[i-side];}
        if row+1<side {y-=r*x[i+side];}
        if col>0 {y-=r*x[i-1];}
        if col+1<side {y-=r*x[i+1];}
        out[i]=y;
    }
}
struct Solver { x:Vec<f64>,r:Vec<f64>,p:Vec<f64>,ap:Vec<f64>,rr:f64,rr0:f64,iterations:usize }
impl Solver {
    fn new(b:&[f64])->Self {
        let rr=dot(b,b);
        Self{x:vec![0.0;b.len()],r:b.to_vec(),p:b.to_vec(),ap:vec![0.0;b.len()],
            rr,rr0:rr,iterations:0}
    }
    fn step(&mut self,side:usize,rho:u32,rel_tol:f64)->Result<bool,String> {
        if self.rr==0.0||self.rr<=self.rr0*(rel_tol*rel_tol) {return Ok(false);}
        apply_operator(side,rho,&self.p,&mut self.ap);
        let pap=dot(&self.p,&self.ap);
        if !pap.is_finite()||pap<=0.0 {return Err("nonpositive/nonfinite CG curvature".into());}
        let alpha=self.rr/pap;
        if !alpha.is_finite() {return Err("nonfinite CG step".into());}
        for i in 0..self.x.len() {
            self.x[i]+=alpha*self.p[i];
            self.r[i]-=alpha*self.ap[i];
            if !self.x[i].is_finite()||!self.r[i].is_finite() {return Err("nonfinite CG state".into());}
        }
        let next=dot(&self.r,&self.r);
        if !next.is_finite()||next<0.0 {return Err("invalid recurrent residual".into());}
        let beta=next/self.rr;
        for i in 0..self.p.len() {self.p[i]=self.r[i]+beta*self.p[i];}
        self.rr=next;self.iterations+=1;
        Ok(true)
    }
}
fn validate(task:&Task)->Result<(),String> {
    if task.id.is_empty()||task.id.len()>128 {return Err("invalid task id".into());}
    if !(2..=128).contains(&task.side)||task.rho>64 {return Err("operator outside bounded domain".into());}
    if !(2..=8).contains(&task.rhs.len()) {return Err("need 2..8 actions".into());}
    let bits=task.scale.to_bits();
    if !task.scale.is_normal()||task.scale<=0.0||(bits&((1u64<<52)-1))!=0 {
        return Err("scale must be a positive normal power of two".into());
    }
    let n=task.side*task.side;
    for b in &task.rhs {
        if b.len()!=n||b.iter().any(|v|!v.is_finite()||v.abs()>2.0f64.powi(100)) {
            return Err("rhs shape or finite-range violation".into());
        }
        let mut norm=Interval::point(0.0);
        for &v in b {norm=norm.add(Interval::point(v).square());}
        if !norm.finite()||norm.hi>task.scale {
            return Err("scale does not cover outward rhs norm upper bound".into());
        }
    }
    Ok(())
}
fn enclose_response(task:&Task,b:&[f64],x:&[f64])->Result<(Bounds,f64),String> {
    if x.len()!=b.len()||x.len()!=task.side*task.side||x.iter().any(|v|!v.is_finite()) {
        return Err("invalid enclosure state".into());
    }
    let side=task.side;let r=task.rho as f64;let d=1.0+4.0*r;
    let mut p=Interval::point(0.0);let mut q=Interval::point(0.0);let mut rr=Interval::point(0.0);
    let (mut ph,mut qh,mut rh,mut bn)=(0.0,0.0,0.0,0.0);
    for i in 0..x.len() {
        let row=i/side;let col=i%side;
        let mut y=Interval::point(d).mul(Interval::point(x[i]));
        let mut yh=d*x[i];
        let mut neighbour=|j:usize| {
            y=y.sub(Interval::point(r).mul(Interval::point(x[j])));
            yh-=r*x[j];
        };
        if row>0 {neighbour(i-side);}
        if row+1<side {neighbour(i+side);}
        if col>0 {neighbour(i-1);}
        if col+1<side {neighbour(i+1);}
        if !y.finite()||!yh.is_finite() {return Err("nonfinite interval stencil".into());}
        p=p.add(Interval::point(b[i]).mul(Interval::point(x[i])));
        q=q.add(Interval::point(x[i]).mul(y));
        rr=rr.add(Interval::point(b[i]).sub(y).square());
        ph+=b[i]*x[i];qh+=x[i]*yh;
        let ri=b[i]-yh;rh+=ri*ri;bn+=b[i]*b[i];
    }
    let ell=Interval::point(2.0).mul(p).sub(q);
    let beta=1.0+8.0*r;
    let lower=ell.add(rr.div_positive(beta)).div_positive(task.scale);
    let upper=ell.add(rr).div_positive(task.scale);
    if !lower.finite()||!upper.finite()||!rh.is_finite()||!bn.is_finite() {
        return Err("nonfinite response enclosure".into());
    }
    let lo=lower.lo.max(0.0);let hi=upper.hi.min(1.0);
    let ell_hat=(2.0*ph-qh)/task.scale;
    if lo>hi||!ell_hat.is_finite() {return Err("empty response enclosure".into());}
    let relative=if bn==0.0 {if rh==0.0 {0.0}else{f64::INFINITY}}else{(rh/bn).sqrt()};
    if !relative.is_finite() {return Err("nonfinite fresh relative residual".into());}
    Ok((Bounds{lo,hi,ell_hat},relative))
}
// Ordinary response readout for the comparator: no residual interval work.
fn ordinary_response(task:&Task,b:&[f64],x:&[f64])->Result<(Bounds,f64),String> {
    let side=task.side;let r=task.rho as f64;let d=1.0+4.0*r;
    let (mut p,mut q,mut rr,mut bn)=(0.0,0.0,0.0,0.0);
    for i in 0..x.len() {
        let row=i/side;let col=i%side;
        let mut y=d*x[i];
        if row>0 {y-=r*x[i-side];}
        if row+1<side {y-=r*x[i+side];}
        if col>0 {y-=r*x[i-1];}
        if col+1<side {y-=r*x[i+1];}
        p+=b[i]*x[i];q+=x[i]*y;
        let v=b[i]-y;rr+=v*v;bn+=b[i]*b[i];
    }
    let ell_hat=(2.0*p-q)/task.scale;
    let relative=if bn==0.0 {if rr==0.0 {0.0}else{f64::INFINITY}}else{(rr/bn).sqrt()};
    if !ell_hat.is_finite()||!relative.is_finite() {return Err("nonfinite ordinary response".into());}
    Ok((Bounds{lo:0.0,hi:1.0,ell_hat},relative))
}
fn separated_action(bounds:&[Bounds])->Option<usize> {
    let mut best=0;
    for i in 1..bounds.len() {if bounds[i].lo>bounds[best].lo {best=i;}}
    if bounds.iter().enumerate().all(|(i,b)|i==best||bounds[best].lo>b.hi) {Some(best)}else{None}
}
fn score_action(bounds:&[Bounds])->usize {
    let mut best=0;
    for i in 1..bounds.len() {if bounds[i].ell_hat>bounds[best].ell_hat {best=i;}}
    best
}
fn measure_bounds(task:&Task,solvers:&[Solver],history:&mut Vec<Bounds>,trace:&mut Option<Vec<Checkpoint>>,plain:bool)
    ->Result<(Vec<Bounds>,Vec<f64>),String> {
    let mut fresh=Vec::with_capacity(solvers.len());let mut rel=Vec::with_capacity(solvers.len());
    for (b,s) in task.rhs.iter().zip(solvers) {
        let (bound,r)=if plain {ordinary_response(task,b,&s.x)?}else{enclose_response(task,b,&s.x)?};
        fresh.push(bound);rel.push(r);
    }
    if history.is_empty() {*history=fresh.clone();}else{
        for (old,new) in history.iter_mut().zip(&fresh) {
            old.lo=old.lo.max(new.lo);old.hi=old.hi.min(new.hi);old.ell_hat=new.ell_hat;
            if old.lo>old.hi {return Err("empty historical interval intersection".into());}
        }
    }
    if let Some(points)=trace {
        points.push(Checkpoint{iterations:solvers.iter().map(|s|s.iterations).collect(),
            fresh_bounds:fresh.clone(),bounds:history.clone()});
    }
    Ok((fresh,rel))
}
fn advance(solvers:&mut[Solver],task:&Task,steps:usize,tol:f64)->Result<usize,String> {
    let mut count=0;
    for s in solvers {
        for _ in 0..steps {
            if !s.step(task.side,task.rho,tol)? {break;}
            count+=1;
        }
    }
    Ok(count)
}
fn micros(start:Instant)->f64 {start.elapsed().as_secs_f64()*1.0e6}
fn run_policy(task:&Task,additional:usize,policy:Policy,want_trace:bool)->Result<Outcome,String> {
    let total_start=Instant::now();
    validate(task)?;
    if additional>128 {return Err("additional iteration cap exceeds128".into());}
    let initial_start=Instant::now();
    let mut solvers:Vec<Solver>=task.rhs.iter().map(|b|Solver::new(b)).collect();
    advance(&mut solvers,task,4,1.0e-14)?;
    let initial_us=micros(initial_start);
    let mut trace=if want_trace {Some(Vec::new())}else{None};
    let mut history=Vec::new();
    let ordinary_only=policy==Policy::PlainAlways;
    let bound_start=Instant::now();
    let (mut fresh,mut relative)=measure_bounds(task,&solvers,&mut history,&mut trace,ordinary_only)?;
    let mut bounds_us=micros(bound_start);
    let initial_bounds=fresh.clone();
    let initial_selected=score_action(&fresh);
    let mut selected=initial_selected;let mut certified=false;
    let mut reason="INITIAL_ONLY".to_string();let mut solve_us=0.0;
    let mut bound_evaluations=solvers.len();
    if matches!(policy,Policy::CertOnce|Policy::CertBlock) {
        if let Some(i)=separated_action(&history) {
            selected=i;certified=true;reason="INITIAL_CERTIFICATE".into();
        }
    }
    if !certified&&policy!=Policy::Never {
        if policy==Policy::Residual {
            let t=Instant::now();
            for s in &mut solvers {
                while s.iterations<512 {
                    if !s.step(task.side,task.rho,1.0e-8)? {break;}
                }
            }
            solve_us+=micros(t);
            let t=Instant::now();
            (fresh,relative)=measure_bounds(task,&solvers,&mut history,&mut trace,ordinary_only)?;
            bounds_us+=micros(t);bound_evaluations+=solvers.len();
            selected=score_action(&fresh);
            reason=if relative.iter().all(|r|*r<=1.0e-8) {"RESIDUAL_GOAL"}else{"RESIDUAL_GOAL_NOT_MET"}.into();
        }else if policy==Policy::CertBlock {
            let mut used=0;
            while used<additional {
                let batch=8.min(additional-used);
                let t=Instant::now();let advanced=advance(&mut solvers,task,batch,1.0e-14)?;
                solve_us+=micros(t);used+=batch;
                if advanced==0 {break;}
                let t=Instant::now();
                (fresh,relative)=measure_bounds(task,&solvers,&mut history,&mut trace,ordinary_only)?;
                bounds_us+=micros(t);bound_evaluations+=solvers.len();
                if let Some(i)=separated_action(&history) {
                    selected=i;certified=true;reason="BLOCK_CERTIFICATE".into();break;
                }
            }
            if !certified {selected=score_action(&fresh);reason="UNSEPARATED_BUDGET".into();}
        }else{
            let t=Instant::now();advance(&mut solvers,task,additional,1.0e-14)?;
            solve_us+=micros(t);
            let t=Instant::now();
            (fresh,relative)=measure_bounds(task,&solvers,&mut history,&mut trace,ordinary_only)?;
            bounds_us+=micros(t);bound_evaluations+=solvers.len();
            selected=score_action(&fresh);
            reason=if policy==Policy::Always {"FIXED_BUDGET"}else if policy==Policy::PlainAlways {"FIXED_BUDGET_PLAIN"}else{"UNSEPARATED_BUDGET"}.into();
        }
    }
    let residual_goal_met=relative.iter().all(|r|*r<=1.0e-8);
    let iterations:Vec<usize>=solvers.iter().map(|s|s.iterations).collect();
    let solver_matvecs=iterations.iter().sum();
    let final_x=if want_trace {Some(solvers.iter().map(|s|s.x.clone()).collect())}else{None};
    let mut out=Outcome{policy:policy.label().into(),rep:0,selected,initial_selected,
        certified,reason,iterations,solver_matvecs,
        ordinary_evaluations:if ordinary_only {bound_evaluations}else{0},
        bound_evaluations:if ordinary_only {0}else{bound_evaluations},total_us:0.0,
        initial_us,solve_us,bounds_us,initial_bounds,final_bounds:fresh,decision_bounds:history,
        residual_relative:relative,residual_goal_met,trace,final_x};
    // Include destruction of the numerical workspace in the measured task.
    drop(solvers);
    out.total_us=micros(total_start);
    Ok(out)
}
fn execute(req:Request)->Result<Response,String> {
    if req.repeats==0||req.repeats>32||req.warmup>4||req.additional>128 {
        return Err("request repeat/warmup/budget cap violation".into());
    }
    let policies=[Policy::Never,Policy::Always,Policy::CertOnce,Policy::CertBlock,Policy::Residual,Policy::PlainAlways];
    let offset=req.task.id.bytes().fold(0usize,|a,b|a.wrapping_mul(31).wrapping_add(b as usize))%policies.len();
    let mut results=Vec::with_capacity(req.repeats*policies.len());
    for pass in 0..req.warmup+req.repeats {
        for k in 0..policies.len() {
            let policy=policies[(offset+pass+k)%policies.len()];
            let mut out=run_policy(&req.task,req.additional,policy,req.trace)?;
            if pass>=req.warmup {out.rep=pass-req.warmup;results.push(out);}
        }
    }
    let input_rhs_bits=if req.trace {
        Some(req.task.rhs.iter().map(|b|b.iter().map(|v|v.to_bits()).collect()).collect())
    }else{None};
    let input_scale_bits=if req.trace {Some(req.task.scale.to_bits())}else{None};
    Ok(Response{id:req.task.id,additional:req.additional,results,input_rhs_bits,input_scale_bits})
}
fn main() {
    let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let result=line.map_err(|e|e.to_string()).and_then(|s|{
            if s.len()>32*1024*1024 {return Err("request line exceeds32MiB".into());}
            serde_json::from_str::<Request>(&s).map_err(|e|e.to_string()).and_then(execute)
        });
        let value=match result {
            Ok(response)=>serde_json::to_value(response).unwrap_or_else(|e|serde_json::json!({"error":e.to_string()})),
            Err(error)=>serde_json::json!({"error":error}),
        };
        if writeln!(stdout,"{}",value).is_err()||stdout.flush().is_err() {break;}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Task {
        let n=64;
        let b:Vec<f64>=(0..n).map(|i|(((i*17)%23) as f64-11.0)/16.0).collect();
        let c:Vec<f64>=(0..n).map(|i|(((i*7)%19) as f64-9.0)/1024.0).collect();
        Task{id:"strong".into(),side:8,rho:16,scale:64.0,rhs:vec![b,c]}
    }
    #[test]
    fn signed_interval_operations_enclose_known_exact_results() {
        let a=Interval{lo:-2.0,hi:1.0}; let b=Interval{lo:3.0,hi:4.0};
        let p=a.mul(b); assert!(p.lo<=-8.0&&p.hi>=4.0);
        let s=a.square(); assert_eq!(s.lo,0.0);assert!(s.hi>=4.0);
        let d=Interval::point(1.0).div_positive(3.0);
        assert!(d.lo<1.0/3.0&&d.hi>1.0/3.0);
    }
    #[test]
    fn rational_response_example_and_one_step_certificate() {
        let task=Task{id:"exact".into(),side:2,rho:1,scale:1.0,
            rhs:vec![vec![1.0,0.0,0.0,0.0],vec![0.0,0.5,0.5,0.0]]};
        let mut bs=Vec::new();
        for (b,q) in task.rhs.iter().zip([23.0/105.0,5.0/42.0]) {
            let x:Vec<f64>=b.iter().map(|v|v/5.0).collect();
            let (bound,_)=enclose_response(&task,b,&x).unwrap();
            assert!(bound.lo<=q&&q<=bound.hi,"{bound:?} excludes {q}");
            bs.push(bound);
        }
        assert_eq!(separated_action(&bs),Some(0));
    }
    #[test]
    fn actual_stop_withholds_solver_matvecs() {
        let task=fixture();
        let always=run_policy(&task,32,Policy::Always,true).unwrap();
        let gate=run_policy(&task,32,Policy::CertOnce,true).unwrap();
        assert!(gate.certified);
        assert_eq!(gate.selected,0);
        assert!(gate.solver_matvecs<always.solver_matvecs);
        assert_eq!(gate.iterations,vec![4,4]);
    }
    #[test]
    fn plain_baseline_matches_state_and_action_without_interval_enclosures() {
        let task=fixture();
        let a=run_policy(&task,32,Policy::Always,true).unwrap();
        let p=run_policy(&task,32,Policy::PlainAlways,true).unwrap();
        assert_eq!(p.selected,a.selected);assert_eq!(p.initial_selected,a.initial_selected);
        assert_eq!(p.iterations,a.iterations);assert_eq!(p.final_x,a.final_x);
        assert_eq!(p.bound_evaluations,0);
        assert_eq!(p.final_bounds.iter().map(|b|b.ell_hat).collect::<Vec<_>>(),
                   a.final_bounds.iter().map(|b|b.ell_hat).collect::<Vec<_>>());
    }
    #[test]
    fn tied_uncertified_fallback_has_identical_state_and_action() {
        let mut task=fixture();task.rhs[1]=task.rhs[0].clone();
        let a=run_policy(&task,8,Policy::Always,true).unwrap();
        for p in [Policy::CertOnce,Policy::CertBlock] {
            let b=run_policy(&task,8,p,true).unwrap();
            assert!(!b.certified);assert_eq!(a.selected,b.selected);
            assert_eq!(a.iterations,b.iterations);assert_eq!(a.final_x,b.final_x);
        }
    }
    #[test]
    fn zero_rhs_needs_no_solver_work_and_does_not_claim_unique_winner() {
        let task=Task{id:"zero".into(),side:2,rho:0,scale:1.0,rhs:vec![vec![0.0;4];2]};
        let o=run_policy(&task,32,Policy::CertBlock,true).unwrap();
        assert_eq!(o.solver_matvecs,0);assert!(!o.certified);assert_eq!(o.selected,0);
    }
    #[test]
    fn invalid_task_and_oracle_field_are_rejected() {
        let mut t=fixture();t.scale=0.0;assert!(validate(&t).is_err());
        let raw=r#"{"task":{"id":"x","side":2,"rho":1,"scale":2,"rhs":[[1,0,0,0],[0,0,0,0]],"oracle":[1,0]},"additional":8}"#;
        assert!(serde_json::from_str::<Request>(raw).is_err());
    }
    #[test]
    fn conventional_solver_meets_its_stated_residual_goal() {
        let o=run_policy(&fixture(),32,Policy::Residual,false).unwrap();
        assert!(o.residual_goal_met,"{:?}",o.residual_relative);
        assert!(o.residual_relative.iter().all(|v|*v<=1.0e-8));
    }
    #[test]
    fn decimal_string_transport_preserves_exact_binary64_bits() {
        let raw=r#"{"task":{"id":"exact_decimal","side":2,"rho":1,"scale":"1024.0","rhs":[["1.8920180462300777","-10.504335880279541","0.0","0.0"],["0.0","0.0","0.0","0.0"]]},"additional":8}"#;
        let req:Request=serde_json::from_str(raw).expect("decimal string transport must be accepted");
        assert_eq!(req.task.rhs[0][0].to_bits(),4611199710940626944);
        assert_eq!(req.task.rhs[0][1].to_bits(),13845475070304059392);
        assert_eq!(req.task.scale.to_bits(),1024.0f64.to_bits());
        validate(&req.task).unwrap();
    }
    #[test]
    fn trace_metadata_identifies_parsed_public_inputs() {
        let task=fixture();
        let expected:Vec<Vec<u64>>=task.rhs.iter().map(|b|b.iter().map(|v|v.to_bits()).collect()).collect();
        let scale=task.scale.to_bits();
        let response=execute(Request{task,additional:8,repeats:1,warmup:0,trace:true}).unwrap();
        assert_eq!(response.input_rhs_bits,Some(expected));
        assert_eq!(response.input_scale_bits,Some(scale));
    }
    #[test]
    fn request_executes_every_control_with_numbered_repetitions() {
        let response=execute(Request{task:fixture(),additional:8,repeats:2,warmup:1,trace:false}).unwrap();
        assert_eq!(response.results.len(),12);
        for rep in 0..2 {
            assert_eq!(response.results.iter().filter(|r|r.rep==rep).count(),6);
        }
        assert!(response.results.iter().all(|r|r.total_us.is_finite()&&r.total_us>=0.0));
    }
}

#[cfg(test)]
mod c24_behavior_tests {
    use super::*;
    fn task(tied: bool) -> Task {
        let b:Vec<f64>=(0..64).map(|i|(((i*17)%23) as f64-11.0)/16.0).collect();
        let c=if tied {b.clone()} else {(0..64).map(|i|(((i*7)%19) as f64-9.0)/1024.0).collect()};
        Task{id:"c24_behavior".into(),side:8,rho:16,scale:64.0,rhs:vec![b,c]}
    }
    #[test]
    fn c24_tiny_nonzero_rhs_rejected_before_solver() {
        let t=Task{id:"tiny_nonzero".into(),side:2,rho:1,scale:2.0f64.powi(-1000),
            rhs:vec![vec![2.0f64.powi(-600),0.0,0.0,0.0],vec![0.0;4]]};
        let got=run_policy(&t,8,Policy::Residual,false);
        assert!(got.is_err(),"tiny nonzero RHS must reject before a false-success solve: {got:?}");
        assert!(got.unwrap_err().contains("UNSUPPORTED_RHS_NORM_RANGE"));
    }
    #[test]
    fn c24_initial_control_performs_no_interval_work() {
        let out=run_policy(&task(false),8,Policy::Never,false).unwrap();
        assert_eq!(out.bound_evaluations,0,"plain initial control must not compute enclosures");
        assert_eq!(out.ordinary_evaluations,2);
    }
    #[test]
    fn c24_plain_residual_goal_is_null() {
        let out=serde_json::to_value(run_policy(&task(false),8,Policy::PlainAlways,false).unwrap()).unwrap();
        assert_eq!(out.get("residual_goal_met"),Some(&serde_json::Value::Null),"ordinary estimates cannot certify the residual goal");
        assert_eq!(out.get("residual_upper"),Some(&serde_json::Value::Null));
    }
    #[test]
    fn c24_public_policy_labels_exact() {
        let out=execute(Request{task:task(false),additional:8,repeats:1,warmup:0,trace:false}).unwrap();
        let mut labels:Vec<_>=out.results.iter().map(|o|o.policy.as_str()).collect();labels.sort();
        let mut expected=vec!["PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK","RESIDUAL"];expected.sort();
        assert_eq!(labels,expected);
    }
    #[test]
    fn c24_once_fallback_final_readout_is_ordinary() {
        let out=run_policy(&task(true),8,Policy::CertOnce,true).unwrap();
        assert!(!out.certified);
        assert_eq!(out.ordinary_evaluations,2,"FAST_ONCE fallback needs one final ordinary readout per action");
        let value=serde_json::to_value(out).unwrap();
        assert_eq!(value.get("residual_goal_met"),Some(&serde_json::Value::Null));
        assert_eq!(value.get("residual_upper"),Some(&serde_json::Value::Null));
    }
    #[test]
    fn c24_response_rejects_state_outside_supported_ratio() {
        let b=2.0f64.powi(-400);
        let t=Task{id:"ratio_guard".into(),side:2,rho:1,scale:2.0f64.powi(-798),rhs:vec![vec![b,0.0,0.0,0.0];2]};
        let x=vec![2.0f64.powi(-299),0.0,0.0,0.0];
        assert!(enclose_response(&t,&t.rhs[0],&x).is_err(),"state violates X <= 2^100 B");
        assert!(ordinary_response(&t,&t.rhs[0],&x).is_err(),"plain readout shares support guard");
    }
}
