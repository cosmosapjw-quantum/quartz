//! C21 project-level, game-neutral posterior predictive EVSI for independent Beta arms.
//! Exact discrete model algebra via product-of-CDF event sweep, with double precision.
//! Live SearchPolicy / PUCT / game heuristics are untouched. True hidden arm means
//! occur only in the separate bench_task generator and are rejected in EVSI requests.
use serde_json::{json,Value};
use rand::{Rng,SeedableRng};
use rand::rngs::StdRng;
use std::io::{self,BufRead,Write};
use std::time::Instant;

const MAX_ARMS:usize=32;
const MAX_BATCH:usize=20;
const PMF_MASS_TOL:f64=1e-10;
const CDF_TOL:f64=1e-9;
const SHAPE_MIN:f64=0.1;
const SHAPE_MAX:f64=100.0;

#[derive(Clone,Copy,Debug)]
struct ScoreEvent { score:f64, arm:usize, weight:f64 }

#[derive(Clone,Debug)]
struct BetaEVSI {
    before:f64, after:f64, gain:f64,
    predictive_mass:f64, support_points:usize, event_count:usize,
    function_us:f64,
}
fn valid_shape(a:f64,b:f64)->bool {
    a.is_finite() && b.is_finite()
    && (SHAPE_MIN..=SHAPE_MAX).contains(&a)
    && (SHAPE_MIN..=SHAPE_MAX).contains(&b)
}
fn pmf_beta_binomial(a:f64,b:f64,n:usize)->Result<Vec<f64>,String> {
    if !valid_shape(a,b)||n>MAX_BATCH {
        return Err("Beta posterior parameters outside validated proper finite range".into())
    }
    let mut probs=vec![0.;n+1];
    let mut p=1.0f64;
    for k in 0..n {
        p *= (b+k as f64)/(a+b+k as f64);
    }
    probs[0]=p;
    for k in 0..n {
        let factor=(n-k) as f64/(k+1) as f64 * (a+k as f64)/(b+(n-k-1) as f64);
        p*=factor;
        probs[k+1]=p;
    }
    if probs.iter().any(|x|!x.is_finite()||*x<0.) {
        return Err("invalid Beta-binomial predictive probability".into())
    }
    let mass=probs.iter().sum::<f64>();
    if !mass.is_finite()||mass<=0.|| (mass-1.).abs()>PMF_MASS_TOL {
        return Err(format!("Beta-binomial probability mass violates bound: {mass}"))
    }
    for x in &mut probs {*x/=mass;}
    Ok(probs)
}
fn kahan(sum:&mut f64,correction:&mut f64,x:f64) {
    let y=x-*correction;
    let t=*sum+y;
    *correction=(t-*sum)-y;
    *sum=t;
}
fn evaluate(arms:&[(f64,f64)],budgets:&[usize])->Result<BetaEVSI,String> {
    let start=Instant::now();
    let k=arms.len();
    if !(2..=MAX_ARMS).contains(&k)||budgets.len()!=k {
        return Err("expected 2..32 posterior action arms and matching budgets".into())
    }
    let mut events=Vec::<ScoreEvent>::new();
    let mut before=f64::NEG_INFINITY;
    for (j,(&(a,b),&n)) in arms.iter().zip(budgets.iter()).enumerate(){
        if !valid_shape(a,b)||n>MAX_BATCH {
            return Err("invalid Beta arm posterior or sample batch".into())
        }
        let old=a/(a+b);
        before=before.max(old);
        let weights=pmf_beta_binomial(a,b,n)?;
        let denom=a+b+n as f64;
        for (success,p) in weights.into_iter().enumerate(){
            let score=(a+success as f64)/denom;
            if !score.is_finite()||!(0.0..=1.0).contains(&score){
                return Err("post-computation utility score out of unit range".into())
            }
            events.push(ScoreEvent{score,arm:j,weight:p});
        }
    }
    events.sort_by(|x,y|x.score.total_cmp(&y.score)
        .then_with(||x.arm.cmp(&y.arm)));
    let event_count=events.len();
    let mut cdf=vec![0f64;k];
    let mut zero_factors=k;
    let mut product_nonzero=1.0f64;
    let mut previous_score=0f64;
    let mut previous_cdf=0f64;
    let mut mean_max=0f64;
    let mut compensation=0f64;
    let mut support_points=0usize;
    let mut i=0usize;
    while i<event_count {
        let score=events[i].score;
        // E[T]=integral_0^1 (1-F_T(t))dt. This survival sum avoids
        // subtracting adjacent CDF values close to one.
        let gap=score-previous_score;
        if gap<0.0{return Err("event order not monotone".into());}
        kahan(&mut mean_max,&mut compensation,gap*(1.0-previous_cdf));
        previous_score=score;
        // Update ALL exactly equal floating supports before emitting one jump.
        while i<event_count && events[i].score.total_cmp(&score).is_eq() {
            let ScoreEvent{score:_,arm,weight}=events[i];
            let old=cdf[arm];
            let new=(old+weight).min(1.0);
            if new+1e-15<old || !new.is_finite(){
                return Err("marginal CDF decreased or nonfinite".into());
            }
            if old==0.0 && new>0.0 {
                zero_factors-=1;
                product_nonzero*=new;
            }else if old>0.0{
                product_nonzero*=new/old;
            }
            cdf[arm]=new;
            i+=1;
        }
        let joint=if zero_factors==0 {product_nonzero}else{0.0};
        if !joint.is_finite()||joint>1.0+CDF_TOL
            ||joint+CDF_TOL<previous_cdf {
            return Err("joint maximum CDF not a probability distribution".into());
        }
        previous_cdf=joint.min(1.0);
        support_points+=1;
    }
    let mass=previous_cdf;
    if (mass-1.0).abs()>CDF_TOL {
        return Err(format!("maximum predictive CDF failed total mass check: {mass}"));
    }
    if !mean_max.is_finite()||mean_max<0.0||mean_max>1.0+CDF_TOL{
        return Err("expected terminal utility outside unit range".into())
    }
    let gross=mean_max-before;
    if gross< -CDF_TOL||gross>1.0-before+CDF_TOL{
        return Err(format!("Bayes EVSI sign/bound violated: {gross}"))
    }
    let gain=gross.max(0.0);
    Ok(BetaEVSI{
        before,after:mean_max,gain,predictive_mass:mass,
        support_points,event_count,function_us:start.elapsed().as_secs_f64()*1e6,
    })
}
pub fn beta_evsi(arms:&[(f64,f64)],budgets:&[usize])->Result<f64,String>{
    Ok(evaluate(arms,budgets)?.gain)
}
fn array_arms(v:&Value)->Result<Vec<(f64,f64)>,String> {
    let arr=v.as_array().ok_or("missing posterior list")?;
    arr.iter().map(|x|{
        let pair=x.as_array().ok_or("Beta arm must be pair [a,b]")?;
        if pair.len()!=2{return Err("Beta arm must have exactly two shapes".into());}
        let a=pair[0].as_f64().ok_or("non-numeric Beta shape a")?;
        let b=pair[1].as_f64().ok_or("non-numeric Beta shape b")?;
        if !valid_shape(a,b){return Err("posterior shape out of validated range".into());}
        Ok((a,b))
    }).collect()
}
fn array_budgets(v:&Value)->Result<Vec<usize>,String>{
    let arr=v.as_array().ok_or("missing additional sample budgets")?;
    arr.iter().map(|x|{
        let n=x.as_u64().ok_or("noninteger computation count")?;
        if n>MAX_BATCH as u64 {return Err("too many additional samples".into())}
        Ok(n as usize)
    }).collect()
}
fn strict_keys(v:&Value,allowed:&[&str])->Result<(),String> {
    let obj=v.as_object().ok_or("expected source-only JSON object")?;
    if obj.keys().any(|s|!allowed.contains(&s.as_str())) {
        return Err("forbidden extra/future/latent field in source-only request".into());
    }
    Ok(())
}
fn run_eval_case(v:&Value)->Result<Value,String>{
    strict_keys(v,&["posterior","extra_per_arm"])?;
    let arms=array_arms(&v["posterior"])?;
    let budgets=array_budgets(&v["extra_per_arm"])?;
    let r=evaluate(&arms,&budgets)?;
    Ok(json!({"before_best":r.before,"after_expected_max":r.after,
         "gross_evsi":r.gain,"joint_mass":r.predictive_mass,
         "support_points":r.support_points,"events":r.event_count,
         "function_internal_us":r.function_us,
         "hidden_truth_used":false}))
}
fn run_sim_task(v:&Value)->Result<Value,String>{
    // OFFLINE-only workload measurement, never routed to run_eval_case.
    strict_keys(v,&["posterior","extra_per_arm","theta_offline_means","seed"])?;
    let arms=array_arms(&v["posterior"])?;
    let budgets=array_budgets(&v["extra_per_arm"])?;
    let true_mu=v["theta_offline_means"].as_array().ok_or("offline sampler theta required")?;
    let seed=v["seed"].as_u64().ok_or("offline seed required")?;
    if true_mu.len()!=arms.len()||budgets.len()!=arms.len(){
        return Err("offline sampler dimension mismatch".into())
    }
    let mut rng=StdRng::seed_from_u64(seed);
    let start=Instant::now();
    let mut draws=0usize;
    let mut means=Vec::new();
    for (j,(&(a,b),&n)) in arms.iter().zip(budgets.iter()).enumerate() {
        let mu=true_mu[j].as_f64().ok_or("offline theta must be numeric")?;
        if !mu.is_finite()||!(0.0..=1.0).contains(&mu){
            return Err("invalid offline theta outside [0,1]".into())
        }
        let mut successes=0usize;
        for _ in 0..n {
            if rng.gen::<f64>()<mu {successes+=1;}
            draws+=1;
        }
        means.push((a+successes as f64)/(a+b+n as f64));
    }
    let action=means.iter().enumerate().max_by(|(j,a),(k,b)|
        a.total_cmp(b).then_with(||k.cmp(j))).unwrap().0;
    let elapsed=start.elapsed().as_secs_f64()*1e6;
    Ok(json!({"offline_workload_only":true,"task_function_us":elapsed,
        "task_num_actual_bernoulli_draws":draws,"sampled_terminal_action":action}))
}
fn run(req:&Value)->Result<Value,String> {
    let obj=req.as_object().ok_or("JSON request object required")?;
    let cmd=obj.get("cmd").and_then(Value::as_str).ok_or("missing command")?;
    match cmd {
        "beta_eval"=>{
            strict_keys(req,&["cmd","cases"])?;
            let cases=req["cases"].as_array().ok_or("missing cases")?;
            if cases.is_empty()||cases.len()>1024 {
                return Err("empty/excessive batch".into())
            }
            let result:Result<Vec<_>,_>=cases.iter().map(run_eval_case).collect();
            Ok(json!({"schema":"quartz_C21_beta_exact_native_v1",
                "results":result?,"no_SearchPolicy_mutation":true,
                "source_truth_used":false}))
        }
        "bench_task"=>{
            strict_keys(req,&["cmd","cases"])?;
            let cases=req["cases"].as_array().ok_or("missing task bench cases")?;
            if cases.is_empty()||cases.len()>1024{
                return Err("empty/excessive offline batch".into())
            }
            let result:Result<Vec<_>,_>=cases.iter().map(run_sim_task).collect();
            Ok(json!({"schema":"quartz_C21_task_benchmark_offline_v1",
                "results":result?,"source_truth_is_offline_sampler_only":true,
                "no_SearchPolicy_mutation":true}))
        }
        _=>Err("unsupported command".into())
    }
}
fn server(){
    let stdin=io::stdin();
    let mut stdout=io::stdout().lock();
    for line in stdin.lock().lines(){
        let Ok(text)=line else{break};
        if text.trim().is_empty(){continue;}
        if text.trim()==r#"{"cmd":"quit"}"# {break}
        let res=serde_json::from_str::<Value>(&text)
             .map_err(|e|e.to_string()).and_then(|v|run(&v));
        let out=match res {Ok(v)=>v,Err(e)=>json!({"error":e})};
        if writeln!(stdout,"{}",out).is_err(){break}
        if stdout.flush().is_err(){break}
    }
}
fn main(){
    let args:Vec<String>=std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("--server"){
        server();
        return;
    }
    let Some(file)=args.get(1)else{
        eprintln!("usage: quartz_c21_fast_evsi INPUT.json | --server");
        std::process::exit(2);
    };
    let res=(||->Result<Value,String>{
        let data=std::fs::read(file).map_err(|e|e.to_string())?;
        let val:Value=serde_json::from_slice(&data).map_err(|e|e.to_string())?;
        run(&val)
    })();
    match res {
       Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
       Err(e)=>{eprintln!("C21 exact posterior contract error: {e}");std::process::exit(3)}
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tied_three_arm_exact_one_hundred_nine_over_eight_hundred_forty(){
        let r=evaluate(&[(1.,1.),(2.,2.),(3.,3.)],&[2,2,2]).unwrap();
        assert!((r.gain-109.0/840.0).abs()<1e-12);
        assert!((r.after-529.0/840.0).abs()<1e-12);
        assert_eq!(r.support_points,7);
        assert_eq!(r.event_count,9);
    }
    #[test]
    fn two_uniform_arms_one_draw_and_half_integer_posterior(){
        let r=evaluate(&[(1.,1.),(1.,1.)],&[1,1]).unwrap();
        assert!((r.gain-1.0/12.0).abs()<1e-12);
        assert!((r.predictive_mass-1.0).abs()<1e-12);
        assert!((pmf_beta_binomial(0.5,1.5,2).unwrap()[0]-5.0/8.0).abs()<1e-12);
    }
    #[test]
    fn zero_budget_and_heterogeneous_budget_conservation(){
        let r=evaluate(&[(4.,2.),(2.,6.),(7.,3.)],&[0,0,0]).unwrap();
        assert!(r.gain.abs()<1e-12);
        let h=evaluate(&[(4.,2.),(2.,6.),(7.,3.)],&[0,2,10]).unwrap();
        assert!(h.gain>=0.0 && h.gain<=1.0 && (h.predictive_mass-1.).abs()<1e-10);
    }
    #[test]
    fn large_K32_bounded_and_invalid_source_rejected(){
        let a=vec![(0.1,100.);32];
        let b=vec![20usize;32];
        let result=evaluate(&a,&b).unwrap();
        assert!((result.predictive_mass-1.).abs()<1e-9);
        assert!(result.gain>=0.);
        assert!(evaluate(&[(1.,1.),(2.,2.)],&[21,1]).is_err());
        assert!(evaluate(&[(0.,1.),(1.,1.)],&[2,2]).is_err());
        assert!(evaluate(&[(f64::NAN,1.),(1.,1.)],&[2,2]).is_err());
    }
    #[test]
    fn strict_source_only_input_and_distinct_offline_sampling(){
        let v=json!({"cmd":"beta_eval","cases":[{"posterior":[[1.,1.],[1.,1.]],
             "extra_per_arm":[1,1]}]});
        assert_eq!(run(&v).unwrap()["no_SearchPolicy_mutation"],true);
        let mut bad=v.clone();
        bad["cases"][0]["theta_offline_means"]=json!([0.6,0.4]);
        assert!(run(&bad).is_err());
        let task=json!({"cmd":"bench_task","cases":[{"posterior":[[1.,1.],[1.,1.]],
             "extra_per_arm":[3,3],"theta_offline_means":[0.6,0.4],"seed":9}]});
        let result=run(&task).unwrap();
        assert_eq!(result["results"][0]["task_num_actual_bernoulli_draws"],6);
    }
}
