//! C22 in-process rational computation actuation for two KNOWN utility models.
//! Domain-neutral metalevel decision; no game tree / PUCT / SearchPolicy edits.
//! Native conditional STOP truly avoids future RNG draws. Distinct ALWAYS control
//! is independently executed with identical seeds for paired scientific auditing.
#[path="quartz_c21_fast_evsi.rs"]
mod exact_beta;
#[path="quartz_c20_shadow.rs"]
mod exact_sensor;

use std::time::Instant;
use std::io::{self,BufRead,Write};
use rand::{SeedableRng,Rng};
use rand::rngs::StdRng;
use serde_json::{Value,json};

#[derive(Clone,Debug)]
enum PublicTask {
    Beta { arms:Vec<(f64,f64)>, budgets:Vec<usize> },
    Sensor { prior:Vec<f64>, likelihood:Vec<Vec<f64>>,
             utilities:Vec<Vec<f64>>, n:usize },
}
#[derive(Clone,Debug)]
enum OfflineTruth { Beta(Vec<f64>), Sensor(usize) }
#[derive(Clone,Copy,Debug)]
struct ResourcePrice { lambda:f64, task_est_us:f64, exact_est_us:f64 }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Policy { Never, Always, CertifiedOnly, Gated, FullExact }
impl Policy {
    fn as_str(self)->&'static str {
        match self {Self::Never=>"NEVER",Self::Always=>"ALWAYS",
          Self::CertifiedOnly=>"CERTIFIED_ONLY",Self::Gated=>"GATED",
          Self::FullExact=>"FULL_EXACT"}
    }
}
#[derive(Clone,Debug)]
struct Decided {
    compute:bool, reason:&'static str,
    cheap_us:f64, exact_us:f64,
    upper:f64, gross:Option<f64>, certified_zero:bool,
}
#[derive(Clone,Debug)]
struct ActionResult {
    before:usize,after:usize,draws:usize,
    signed_utility_gain:f64,task_us:f64,
}
#[derive(Clone,Debug)]
struct RunResult {
    policy:Policy, decision:Decided, action:ActionResult,
    total_internal_us:f64, net_realized:f64,
}
#[derive(Clone,Debug)]
pub struct ActiveOutcome {
    pub compute:bool,pub draws:usize,pub before:usize,
    pub after:usize,pub delta:f64,
}
fn argmax(vals:&[f64])->usize {
    (0..vals.len()).max_by(|&i,&j|vals[i].total_cmp(&vals[j])
       .then_with(||j.cmp(&i))).unwrap()
}
fn current_utilities(task:&PublicTask)->Vec<f64> {
    match task {
       PublicTask::Beta{arms,..}=>arms.iter().map(|(a,b)|a/(a+b)).collect(),
       PublicTask::Sensor{prior,utilities,..}=> (0..utilities[0].len())
           .map(|a|(0..prior.len()).map(|t|prior[t]*utilities[t][a]).sum()).collect()
    }
}
fn validate(task:&PublicTask)->Result<(),String> {
    match task{
      PublicTask::Beta{arms,budgets}=>{
        if !(2..=4).contains(&arms.len())||arms.len()!=budgets.len() ||
           budgets.iter().any(|&n|n>20) ||
           arms.iter().any(|&(a,b)|!a.is_finite()||!b.is_finite()||
             !(0.1..=100.).contains(&a)||!(0.1..=100.).contains(&b)){
          return Err("C22 Beta bounded proper posterior or budget invalid".into());
        }
      },
      PublicTask::Sensor{prior,likelihood,utilities,n}=>{
         if prior.len()!=4||likelihood.len()!=4||utilities.len()!=4 ||
             *n>10 || likelihood.iter().any(|r|r.len()!=3) ||
             utilities.iter().any(|r|r.len()!=3) ||
             prior.iter().any(|p|!p.is_finite()||*p<0.) ||
             (prior.iter().sum::<f64>()-1.).abs()>1e-9 ||
             likelihood.iter().any(|row|row.iter().any(|p|!p.is_finite()||*p<0.) ||
                        (row.iter().sum::<f64>()-1.).abs()>1e-9) ||
             utilities.iter().any(|row|row.iter().any(|u|!u.is_finite()||!(0.0..=1.0).contains(u))){
            return Err("C22 finite sensor posterior, channel or payoff invalid".into());
         }
      },
    }
    Ok(())
}
fn validate_price(p:ResourcePrice)->Result<(),String>{
    if !p.lambda.is_finite()||p.lambda<0. ||
       !p.task_est_us.is_finite()||p.task_est_us<=0. ||
       !p.exact_est_us.is_finite()||p.exact_est_us<=0. {
        return Err("invalid strictly positive train-only resource estimates".into())
    }
    Ok(())
}
fn cheap_upper(task:&PublicTask)->Result<(bool,f64),String>{
    validate(task)?;
    match task {
       PublicTask::Beta{arms,budgets}=>{
          let mu:Vec<f64>=arms.iter().map(|(a,b)|a/(a+b)).collect();
          let i0=argmax(&mu);
          let lows:Vec<f64>=arms.iter().zip(budgets.iter())
              .map(|(&(a,b),&n)|a/(a+b+n as f64)).collect();
          let highs:Vec<f64>=arms.iter().zip(budgets.iter())
              .map(|(&(a,b),&n)|(a+n as f64)/(a+b+n as f64)).collect();
          let dominance=(0..arms.len()).filter(|&j|j!=i0).all(|j|lows[i0]>highs[j]);
          // When all budgets are zero, G=0 independently of current tie.
          if dominance||budgets.iter().all(|&n|n==0) {return Ok((true,0.))}
          let var:Vec<f64>=arms.iter().zip(budgets.iter()).map(|(&(a,b),&n)|{
              let s=a+b; (n as f64)*a*b/(s*s*(s+1.)*(s+n as f64))
          }).collect();
          let mut bound=0.;
          for j in 0..arms.len() {
              if j==i0 {continue}
              let delta=mu[j]-mu[i0];
              bound+=0.5*((delta*delta+var[i0]+var[j]).sqrt()+delta);
          }
          Ok((false,bound.max(0.).min(1.-mu[i0])))
       },
       PublicTask::Sensor{prior,utilities,..}=>{
          let scores=current_utilities(task);
          let best=argmax(&scores);
          let dominance=(0..utilities[0].len()).filter(|&a|a!=best).all(|a|
             (0..prior.len()).filter(|&t|prior[t]>0.).all(|t|
                utilities[t][best]>utilities[t][a]));
          if dominance {return Ok((true,0.))}
          let vpi=(0..prior.len()).map(|t|prior[t]*
              utilities[t].iter().cloned().fold(f64::NEG_INFINITY,f64::max))
              .sum::<f64>() - scores[best];
          Ok((false,vpi.max(0.).min(1.-scores[best])))
       }
    }
}
fn full_gross(task:&PublicTask)->Result<f64,String> {
    validate(task)?;
    match task {
      PublicTask::Beta{arms,budgets}=>exact_beta::beta_evsi(arms,budgets),
      PublicTask::Sensor{prior,likelihood,utilities,n}=>{
         exact_sensor::sensor_evsi(prior,likelihood,utilities,*n)
      }
    }
}
fn decide(task:&PublicTask,policy:Policy,price:ResourcePrice)->Result<Decided,String>{
    validate(task)?;
    validate_price(price)?;
    if policy==Policy::Never {return Ok(Decided{compute:false,reason:"NEVER",
        cheap_us:0.,exact_us:0.,upper:0.,gross:None,certified_zero:false})}
    if policy==Policy::Always {return Ok(Decided{compute:true,reason:"ALWAYS",
        cheap_us:0.,exact_us:0.,upper:0.,gross:None,certified_zero:false})}
    let mut cheap_us=0.;
    let mut upper=1.;
    let mut zero=false;
    if policy!=Policy::FullExact {
       let start=Instant::now();
       let (z,b)=cheap_upper(task)?;
       cheap_us=start.elapsed().as_secs_f64()*1e6;
       zero=z;upper=b;
       if z {
          return Ok(Decided{compute:false,reason:"CERTIFIED_ZERO",
            cheap_us,exact_us:0.,upper,gross:Some(0.),certified_zero:true});
       }
       if policy==Policy::CertifiedOnly {
          return Ok(Decided{compute:true,reason:"CERT_ONLY_COMPUTE",
            cheap_us,exact_us:0.,upper,gross:None,certified_zero:false});
       }
       if upper<=price.lambda*price.task_est_us {
          return Ok(Decided{compute:false,reason:"PRICE_UPPER_SCREEN",
             cheap_us,exact_us:0.,upper,gross:None,certified_zero:false});
       }
    }
    let started=Instant::now();
    let g=full_gross(task)?;
    let exact_us=started.elapsed().as_secs_f64()*1e6;
    if g< -1e-9 || g>1.+1e-9{return Err("exact Bayesian EVSI invalid".into())}
    let compute=g>price.lambda*(price.task_est_us+price.exact_est_us);
    Ok(Decided{compute,reason:if compute {"EXACT_COMPUTE"}else{"EXACT_STOP"},
       cheap_us,exact_us,upper,gross:Some(g),certified_zero:false})
}
fn validate_truth(task:&PublicTask,truth:&OfflineTruth)->Result<(),String>{
    match (task,truth){
       (PublicTask::Beta{arms,..},OfflineTruth::Beta(v))=>{
           if v.len()!=arms.len()||v.iter().any(|x|!x.is_finite()||!(0.0..=1.0).contains(x)){
               return Err("offline Beta actual mean dimension/value invalid".into())
           }
       },
       (PublicTask::Sensor{prior,..},OfflineTruth::Sensor(t))=>{
           if *t>=prior.len() {return Err("offline hidden sensor state invalid".into())}
       },
       _=>return Err("offline source kind mismatch".into())
    }
    Ok(())
}
fn execute_actual(task:&PublicTask,truth:&OfflineTruth,seed:u64,compute:bool)
        ->Result<ActionResult,String>{
    validate(task)?;validate_truth(task,truth)?;
    let before=argmax(&current_utilities(task));
    if !compute {
        // Critical: no RNG initialization, no posterior update and no future observation.
        return Ok(ActionResult{before,after:before,draws:0,
            signed_utility_gain:0.,task_us:0.});
    }
    let started=Instant::now();
    let mut rng=StdRng::seed_from_u64(seed);
    let (after,draws,gain)=match (task,truth){
       (PublicTask::Beta{arms,budgets},OfflineTruth::Beta(theta))=>{
          let mut values=Vec::with_capacity(arms.len());
          let mut samples=0usize;
          for (j,(&(a,b),&n)) in arms.iter().zip(budgets.iter()).enumerate(){
             let mut successes=0usize;
             for _ in 0..n{
                if rng.gen::<f64>()<theta[j]{successes+=1}
                samples+=1;
             }
             values.push((a+successes as f64)/(a+b+n as f64));
          }
          let chosen=argmax(&values);
          (chosen,samples,theta[chosen]-theta[before])
       },
       (PublicTask::Sensor{prior,likelihood,utilities,n},OfflineTruth::Sensor(theta))=>{
           let mut b=prior.clone();
           for _ in 0..*n {
              let u=rng.gen::<f64>();
              let mut y=likelihood[*theta].len()-1;
              let mut cdf=0.;
              for (j,prob) in likelihood[*theta].iter().enumerate(){
                  cdf+=prob;
                  if u<cdf {y=j;break}
              }
              let mut norm=0.;
              for t in 0..b.len(){
                 b[t]*=likelihood[t][y];
                 norm+=b[t];
              }
              if norm<=0. {return Err("zero posterior predictive likelihood".into())}
              for x in &mut b {*x/=norm;}
           }
           let aftermeans:(Vec<f64>)=(0..utilities[0].len())
              .map(|a|(0..b.len()).map(|t|b[t]*utilities[t][a]).sum::<f64>()).collect();
           let chosen=argmax(&aftermeans);
           (chosen,*n,utilities[*theta][chosen]-utilities[*theta][before])
       },
       _=>return Err("task/offline source mismatch".into())
    };
    let elapsed_us=started.elapsed().as_secs_f64()*1e6;
    Ok(ActionResult{before,after,draws,signed_utility_gain:gain,task_us:elapsed_us})
}
fn execute_policy(task:&PublicTask,truth:&OfflineTruth,seed:u64,
        policy:Policy,price:ResourcePrice)->Result<RunResult,String>{
    let start=Instant::now();
    let d=decide(task,policy,price)?;
    let a=execute_actual(task,truth,seed,d.compute)?;
    let total_internal_us=start.elapsed().as_secs_f64()*1e6;
    let net=a.signed_utility_gain-price.lambda*total_internal_us;
    if !d.compute && (a.draws!=0 || a.before!=a.after || a.signed_utility_gain!=0.){
        return Err("failed actual STOP invariant".into())
    }
    Ok(RunResult{policy,decision:d,action:a,total_internal_us,net_realized:net})
}
pub fn run_beta_active(posterior:&[(f64,f64)],budgets:&[usize],
    theta:&[f64],seed:u64,lambda:f64,task_est:f64,meta_est:f64)
    -> Result<ActiveOutcome,String> {
    let t=PublicTask::Beta{arms:posterior.to_vec(),budgets:budgets.to_vec()};
    let p=ResourcePrice{lambda,task_est_us:task_est,exact_est_us:meta_est};
    let r=execute_policy(&t,&OfflineTruth::Beta(theta.to_vec()),seed,Policy::Gated,p)?;
    Ok(ActiveOutcome{compute:r.decision.compute,draws:r.action.draws,
       before:r.action.before,after:r.action.after,delta:r.action.signed_utility_gain})
}
fn only_keys(v:&Value,keys:&[&str])->Result<(),String>{
    let map=v.as_object().ok_or("source-only object required")?;
    if map.keys().any(|key|!keys.contains(&key.as_str())){
       return Err("forbidden extra field / hidden theta in public decision".into())
    }
    Ok(())
}
fn float_vec(v:&Value)->Result<Vec<f64>,String>{
    let a=v.as_array().ok_or("numeric vector required")?;
    a.iter().map(|e|e.as_f64().ok_or("invalid scalar".into())).collect()
}
fn float_matrix(v:&Value)->Result<Vec<Vec<f64>>,String>{
    let a=v.as_array().ok_or("matrix required")?;
    a.iter().map(float_vec).collect()
}
fn public_from_value(v:&Value)->Result<PublicTask,String>{
    let family=v["family"].as_str().ok_or("missing public family")?;
    match family{
      "beta_arms"=>{
         only_keys(v,&["family","posterior","extra_per_arm"])?;
         let arr=float_matrix(&v["posterior"])?;
         let mut arms=Vec::new();
         for x in arr{
            if x.len()!=2{return Err("Beta shape requires exactly [a,b]".into())}
            arms.push((x[0],x[1]));
         }
         let budgets=v["extra_per_arm"].as_array().ok_or("missing future batch count")?
           .iter().map(|x|x.as_u64().map(|n|n as usize)
             .ok_or("nonnegative integer batch required".into())).collect::<Result<Vec<_>,String>>()?;
         let task=PublicTask::Beta{arms,budgets};
         validate(&task)?;Ok(task)
      },
      "finite_sensor"=>{
         only_keys(v,&["family","prior","likelihood","utilities","extra"])?;
         let n=v["extra"].as_u64().ok_or("nonnegative sensor budget required")?as usize;
         let task=PublicTask::Sensor{
           prior:float_vec(&v["prior"])?,
           likelihood:float_matrix(&v["likelihood"])?,
           utilities:float_matrix(&v["utilities"])?,
           n,
         };
         validate(&task)?;Ok(task)
      },
      _=>Err("unknown non-game source family".into())
    }
}
fn truth_from_value(task:&PublicTask,v:&Value)->Result<(OfflineTruth,u64),String>{
    let seed=v["seed"].as_u64().ok_or("offline future sampling seed missing")?;
    let t=match task{
      PublicTask::Beta{..}=>{
        only_keys(v,&["seed","theta_offline_arm_means"])?;
        OfflineTruth::Beta(float_vec(&v["theta_offline_arm_means"])?)
      },
      PublicTask::Sensor{..}=>{
        only_keys(v,&["seed","theta_offline_state"])?;
        let index=v["theta_offline_state"].as_u64().ok_or("offline hidden state index required")? as usize;
        OfflineTruth::Sensor(index)
      }
    };
    validate_truth(task,&t)?;
    Ok((t,seed))
}
fn price_from_value(v:&Value,lambda:f64)->Result<ResourcePrice,String>{
    only_keys(v,&["task_est_us","exact_est_us"])?;
    let p=ResourcePrice{
       lambda,
       task_est_us:v["task_est_us"].as_f64().ok_or("missing task cost forecast")?,
       exact_est_us:v["exact_est_us"].as_f64().ok_or("missing exact model cost forecast")?,
    };
    validate_price(p)?;Ok(p)
}
fn serialize_run(r:&RunResult)->Value {
   json!({"policy":r.policy.as_str(),"COMPUTE":r.decision.compute,
     "decision_reason":r.decision.reason,
     "certified_zero_information":r.decision.certified_zero,
     "cheap_upper":r.decision.upper,
     "gross_predictive_EVSI":r.decision.gross,
     "actual_future_draw_count":r.action.draws,
     "before_action":r.action.before,
     "after_action":r.action.after,
     "true_signed_terminal_utility_change":r.action.signed_utility_gain,
     "cheap_gate_us":r.decision.cheap_us,
     "exact_evaluator_us":r.decision.exact_us,
     "actual_future_task_us":r.action.task_us,
     "active_total_internal_us":r.total_internal_us,
     "net_utility_change_after_measured_cost":r.net_realized,
     "active_not_shadow_only":true})
}
fn active_experiment(req:&Value)->Result<Value,String>{
    only_keys(req,&["cmd","lambda","cases","policy_set"])?;
    let lambda=req["lambda"].as_f64().ok_or("missing resource tariff")?;
    if !lambda.is_finite()||lambda<=0. {return Err("positive finite price required".into())}
    let cases=req["cases"].as_array().ok_or("missing experiment cases")?;
    if cases.is_empty()||cases.len()>1024{return Err("bounded nonempty source batch required".into())}
    let policies=[Policy::Never,Policy::Always,Policy::CertifiedOnly,Policy::Gated,Policy::FullExact];
    let mut outputs=Vec::new();
    let mut stops=0usize;let mut avoided=0usize;
    for (i,v) in cases.iter().enumerate(){
       only_keys(v,&["id","group_id","split","rep","public","offline","price"])?;
       let public=public_from_value(&v["public"])?;
       let (truth,seed)=truth_from_value(&public,&v["offline"])?;
       let price=price_from_value(&v["price"],lambda)?;
       let mut runs=Vec::new();
       // Alternate evaluation order to reduce systematic thermal/order bias.
       let ordered:Vec<Policy>=if i%2==0{policies.to_vec()}
         else{policies.iter().rev().cloned().collect()};
       for policy in ordered{
          runs.push(execute_policy(&public,&truth,seed,policy,price)?);
       }
       let baseline=runs.iter().find(|r|r.policy==Policy::Always).unwrap();
       let base_a=baseline.action.after;
       let base_delta=baseline.action.signed_utility_gain;
       let mut by_policy=serde_json::Map::new();
       for run in &runs{
          if run.policy!=Policy::Never && run.decision.compute {
             if run.action.after!=base_a ||
                (run.action.signed_utility_gain-base_delta).abs()>1e-12 ||
                run.action.draws!=baseline.action.draws {
                return Err("paired same seed active COMPUTE not identical to ALWAYS control".into());
             }
          }
          if run.policy==Policy::Gated && !run.decision.compute{
              stops+=1;
              avoided+=baseline.action.draws;
          }
          by_policy.insert(run.policy.as_str().to_string(),serialize_run(run));
       }
       outputs.push(json!({
           "id":v["id"],"group_id":v["group_id"],"split":v["split"],"rep":v["rep"],
           "source_family":match &public{PublicTask::Beta{..}=>"beta_arms",
                  PublicTask::Sensor{..}=>"finite_sensor"},
           "input_true_hidden_theta_used_by_public_selector":false,
           "baseline_seed_replayed_in_separate_timed_control":true,
           "baseline_and_active_both_physically_executed_for_comparison":true,
           "results":Value::Object(by_policy)
       }));
    }
    Ok(json!({"schema":"quartz_C22_ACTIVE_INPROCESS_v1",
       "number_cases":outputs.len(),"lambda":lambda,
       "GATED_true_STOP_count":stops,"GATED_actually_unexecuted_future_draws":avoided,
       "source_policy": "two_source_true_utility_native_runtime",
       "original_SearchPolicy_unchanged":true,
       "any_global_runtime_savings_claim":false,
       "rows":outputs}))
}
fn main(){
   let Some(path)=std::env::args().nth(1)else{
       eprintln!("usage: quartz_c22_active_gate CASES.json");
       std::process::exit(2);
   };
   let result=(||->Result<Value,String>{
       let contents=std::fs::read_to_string(path).map_err(|e|e.to_string())?;
       let req:Value=serde_json::from_str(&contents).map_err(|e|e.to_string())?;
       if req["cmd"]!="active_experiment"{return Err("only bounded active_experiment supported".into())}
       active_experiment(&req)
   })();
   match result{
      Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
      Err(e)=>{eprintln!("C22 in-process active decision ERROR: {e}");std::process::exit(3)}
   }
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn dominant_posterior_actively_stops_without_rng(){
  let o=run_beta_active(&[(12.,2.),(2.,12.)],&[1,1],&[0.8,0.2],7,0.004,0.2,1.).unwrap();
  assert!(!o.compute && o.draws==0 && o.before==o.after && o.delta==0.);
 }
 #[test] fn source_only_nonzero_evsi_and_same_seed_actual_compute(){
   let t=PublicTask::Beta{arms:vec![(1.,1.),(1.,1.)],budgets:vec![1,1]};
   let price=ResourcePrice{lambda:0.004,task_est_us:0.2,exact_est_us:0.1};
   let truth=OfflineTruth::Beta(vec![0.2,0.8]);
   assert!((full_gross(&t).unwrap()-1./12.).abs()<1e-12);
   let active=execute_policy(&t,&truth,1313,Policy::Gated,price).unwrap();
   let always=execute_policy(&t,&truth,1313,Policy::Always,price).unwrap();
   assert!(active.decision.compute && active.action.draws==2);
   assert_eq!(active.action.after,always.action.after);
   assert!((active.action.signed_utility_gain-always.action.signed_utility_gain).abs()<1e-15);
 }
 #[test] fn upper_bound_can_actively_price_stop_without_any_random_draws(){
   let t=PublicTask::Beta{arms:vec![(1.,1.),(1.,1.)],budgets:vec![1,1]};
   let price=ResourcePrice{lambda:10.0,task_est_us:0.2,exact_est_us:0.1};
   let active=execute_policy(&t,&OfflineTruth::Beta(vec![0.1,0.9]),9,Policy::Gated,price).unwrap();
   assert!(!active.decision.compute && active.action.draws==0);
   assert_eq!(active.decision.reason,"PRICE_UPPER_SCREEN");
 }
 #[test] fn sensor_dominant_action_actively_avoids_future_categorical_observations(){
    let t=PublicTask::Sensor{prior:vec![0.25;4],
       likelihood:vec![vec![0.7,0.2,0.1];4],
       utilities:vec![vec![0.8,0.3,0.1];4],n:8};
    let price=ResourcePrice{lambda:0.004,task_est_us:0.3,exact_est_us:2.};
    let c=execute_policy(&t,&OfflineTruth::Sensor(0),29,Policy::Gated,price).unwrap();
    assert!(!c.decision.compute && c.action.draws==0 && c.action.before==c.action.after);
    assert_eq!(c.decision.reason,"CERTIFIED_ZERO");
 }
 #[test] fn no_oracle_in_public_beta_or_sensor_and_pairwise_controls(){
    let mut beta=json!({"cmd":"active_experiment","lambda":0.004,"cases":[{
      "id":"test0","group_id":"oneparent","split":"train","rep":0,
      "public":{"family":"beta_arms","posterior":[[1.,1.],[1.,1.]],"extra_per_arm":[1,1]},
      "offline":{"theta_offline_arm_means":[0.3,0.7],"seed":81},
      "price":{"task_est_us":0.1,"exact_est_us":0.1}}]});
    let result=active_experiment(&beta).unwrap();
    assert_eq!(result["rows"][0]["results"]["GATED"]["actual_future_draw_count"],2);
    assert_eq!(result["rows"][0]["results"]["NEVER"]["actual_future_draw_count"],0);
    beta["cases"][0]["public"]["theta_offline_arm_means"]=json!([0.3,0.7]);
    assert!(active_experiment(&beta).is_err());
 }
}
