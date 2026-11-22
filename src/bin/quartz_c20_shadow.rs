//! Bounded independent Rust source-law EVSI reference for *finite latent channels*.
//! This is read-only and is not SearchPolicy/PUCT or a Beta-Binomial Rust port.
use serde_json::{json,Value};
use std::time::Instant;

#[derive(Clone,Debug)]
struct ResultEVSI { evsi:f64, mass:f64, before:f64, after:f64, count:usize, elapsed_us:f64 }

fn factorial(n:usize)->u64 { (1..=n as u64).product() }
fn each_counts<F:FnMut(&[usize])>(depth:usize,remaining:usize,k:&mut [usize],f:&mut F){
    if depth+1==k.len() {k[depth]=remaining;f(k);return}
    for v in 0..=remaining { k[depth]=v;each_counts(depth+1,remaining-v,k,f); }
}
fn posterior_evsi(prior:&[f64],l:&[Vec<f64>],u:&[Vec<f64>],n:usize)
     ->Result<ResultEVSI,String>{
    let started=Instant::now();
    let s=prior.len();
    if !(2..=8).contains(&s)||n>10||l.len()!=s||u.len()!=s
       { return Err("invalid bounded latent state dimension or budget".into()) }
    let m=l[0].len();let a=u[0].len();
    if !(2..=4).contains(&m)||!(2..=8).contains(&a)
       {return Err("invalid sensor outcome/terminal action dimension".into())}
    let valid_probability=|row:&[f64]|{
        !row.is_empty()&&row.iter().all(|x|x.is_finite()&&*x>=0.&&*x<=1.)
        &&(row.iter().sum::<f64>()-1.).abs()<=1e-9
    };
    if !valid_probability(prior)||l.iter().any(|r|r.len()!=m||!valid_probability(r))
       ||u.iter().any(|r|r.len()!=a||r.iter().any(|v|!v.is_finite()||*v<0.||*v>1.))
       {return Err("invalid normalized source prior/channel or bounded utility".into())}
    let mut baseline=0f64;
    for action in 0..a{
        baseline=baseline.max((0..s).map(|t|prior[t]*u[t][action]).sum::<f64>());
    }
    let mut mass=0.;let mut after=0.;let mut count=0usize;
    let mut k=vec![0; m];
    each_counts(0,n,&mut k,&mut |counts|{
        let factor=factorial(n)as f64 /
           counts.iter().map(|x|factorial(*x)as f64).product::<f64>();
        let weights:Vec<f64>=(0..s).map(|t| {
            prior[t]*(0..m).map(|y| l[t][y].powi(counts[y]as i32)).product::<f64>()
        }).collect();
        mass+=factor*weights.iter().sum::<f64>();
        let best=(0..a).map(|act|
            (0..s).map(|t|weights[t]*u[t][act]).sum::<f64>()
        ).fold(f64::NEG_INFINITY,f64::max);
        after+=factor*best;
        count+=1;
    });
    let g=after-baseline;
    if (mass-1.).abs()>1e-9||g< -1e-9||g>1.-baseline+1e-8{
        return Err("Bayesian posterior predictive normalization/bounds violation".into());
    }
    Ok(ResultEVSI{evsi:g.max(0.),mass,before:baseline,after,count,
        elapsed_us:started.elapsed().as_secs_f64()*1e6})
}
pub fn sensor_evsi(prior:&[f64],l:&[Vec<f64>],u:&[Vec<f64>],n:usize)
     ->Result<f64,String> {
    Ok(posterior_evsi(prior,l,u,n)?.evsi)
}
fn read_vec(v:&Value)->Result<Vec<f64>,String>{
    let arr=v.as_array().ok_or("expected vector array")?;
    arr.iter().map(|x| x.as_f64().ok_or("invalid finite numeric coefficient".into())).collect()
}
fn matrix(v:&Value)->Result<Vec<Vec<f64>>,String>{
    let rows=v.as_array().ok_or("expected matrix")?;
    rows.iter().map(read_vec).collect()
}
fn evaluate(request:&Value)->Result<Value,String>{
    let cases=request["cases"].as_array().ok_or("missing cases array")?;
    if cases.is_empty()||cases.len()>1024 {return Err("empty/excessive request".into())}
    let mut output=Vec::with_capacity(cases.len());
    for v in cases {
        if ["theta_offline_true_state","label","signed_gain","final_action",
            "extra_observations","actual_extra_cost_us","domain"]
            .iter().any(|k|v.get(k).is_some()){
             return Err("oracle or future source data forbidden in read-only EVSI".into());
        }
        let prior=read_vec(&v["prior"])?;
        let l=matrix(&v["likelihood"])?;
        let u=matrix(&v["utilities"])?;
        let n=v["extra"].as_u64().ok_or("batch count required")?as usize;
        let r=posterior_evsi(&prior,&l,&u,n)?;
        output.push(json!({"evsi":r.evsi,"before_best":r.before,
          "after_expected_best":r.after,"total_predictive_mass":r.mass,
          "outcome_count":r.count,"evaluation_us":r.elapsed_us,
          "true_hidden_state_was_used":false}));
    }
    Ok(json!({"schema":"quartz_c20_rust_finite_channel_shadow_v1",
      "predictions":output,"no_SearchPolicy_mutation":true,
      "beta_binomial_Rust_port":false}))
}
fn main(){
    let Some(p)=std::env::args_os().nth(1)else{
        eprintln!("usage: quartz_c20_shadow FINITE_CHANNEL_INPUT.json");
        std::process::exit(2);
    };
    let result=(||->Result<Value,String>{
        let data=std::fs::read(&p).map_err(|e|e.to_string())?;
        let req:Value=serde_json::from_slice(&data).map_err(|e|e.to_string())?;
        evaluate(&req)
    })();
    match result {
        Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
        Err(e)=>{eprintln!("C20 shadow error: {e}");std::process::exit(3)}
    }
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]
 fn accurate_binary_sensor_has_quarter_value(){
   let a=sensor_evsi(&[0.5,0.5],&vec![vec![0.75,0.25],vec![0.25,0.75]],
       &vec![vec![1.,0.],vec![0.,1.]],1).unwrap();
   assert!((a-0.25).abs()<1e-12);
 }
 #[test]
 fn repeated_signal_shared_hidden_state_is_correlated(){
   let a=posterior_evsi(&[0.5,0.5],&vec![vec![1.,0.],vec![0.,1.]],
        &vec![vec![1.,0.],vec![0.,1.]],2).unwrap();
   assert!((a.evsi-0.5).abs()<1e-12&&a.count==3);
   assert!((a.mass-1.).abs()<1e-12);
 }
 #[test]
 fn no_information_and_future_leak_rejected(){
   let a=sensor_evsi(&[0.5,0.5],&vec![vec![0.4,0.6],vec![0.4,0.6]],
       &vec![vec![1.,0.],vec![0.,1.]],2).unwrap();
   assert!(a.abs()<1e-12);
   let j=json!({"cases":[{"prior":[0.5,0.5],
      "likelihood":[[0.75,0.25],[0.25,0.75]],
      "utilities":[[1.,0.],[0.,1.]],
      "extra":1,"theta_offline_true_state":1}]});
   assert!(evaluate(&j).is_err());
   assert!(sensor_evsi(&[0.5,0.5],&vec![vec![0.5,0.5],vec![0.4,0.6]],
       &vec![vec![1.,0.],vec![0.,1.]],50).is_err());
 }
}
