//! C21 exploratory TRAIN-only independent full-joint Rust reference and same-compiler
//! timing benchmark. Not used to refit the prospective C21 heldout policy.
#[path="quartz_c21_fast_evsi.rs"]
mod fast;
use serde_json::{json,Value};
use std::time::Instant;
use std::hint::black_box;

fn pmf(a:f64,b:f64,n:usize)->Result<Vec<f64>,String> {
    if !a.is_finite()||!b.is_finite()||a<=0.0||b<=0.0||n>10{
        return Err("invalid bounded Beta posterior grid reference".into());
    }
    let mut p=1.0f64;
    for k in 0..n {p*=(b+k as f64)/(a+b+k as f64);}
    let mut v=vec![p];
    for k in 0..n {
        p*=((n-k) as f64/(k+1) as f64)*((a+k as f64)/(b+(n-k-1) as f64));
        v.push(p);
    }
    let sum=v.iter().sum::<f64>();
    if (sum-1.).abs()>1e-10{return Err("reference predictive mass invalid".into());}
    for x in &mut v {*x/=sum;}
    Ok(v)
}
fn full_grid(arms:&[(f64,f64)],ns:&[usize])->Result<f64,String> {
    if !(2..=4).contains(&arms.len())||arms.len()!=ns.len()||ns.iter().any(|n|*n>10){
        return Err("full grid restricted K<=4 n<=10".into());
    }
    let prob:Vec<Vec<f64>>=arms.iter().zip(ns.iter())
        .map(|(&(a,b),&n)|pmf(a,b,n)).collect::<Result<_,_>>()?;
    fn visit(level:usize,arms:&[(f64,f64)],ns:&[usize],prob:&[Vec<f64>],
        mass:f64,current_max:f64,overall:&mut f64,weighted:&mut f64){
        if level==arms.len(){
            *overall+=mass;
            *weighted+=mass*current_max;
            return;
        }
        let (a,b)=arms[level];
        let n=ns[level];
        for k in 0..=n {
            let posterior_mean=(a+k as f64)/(a+b+n as f64);
            visit(level+1,arms,ns,prob,mass*prob[level][k],
                current_max.max(posterior_mean),overall,weighted);
        }
    }
    let mut weighted=0f64;let mut mass=0f64;
    visit(0,arms,ns,&prob,1.,0.,&mut mass,&mut weighted);
    if (mass-1.).abs()>1e-9 {return Err("full joint predictive mass failed".into());}
    let before=arms.iter().map(|(a,b)|a/(a+b)).fold(f64::NEG_INFINITY,f64::max);
    let gross=weighted-before;
    if gross< -1e-9 {return Err("full joint Bayes EVSI negative".into());}
    Ok(gross.max(0.))
}
fn median(mut v:Vec<f64>)->f64 {
    v.sort_by(f64::total_cmp);
    let n=v.len();
    if n%2==1 {v[n/2]}else{(v[n/2-1]+v[n/2])/2.}
}
fn run(request:&Value)->Result<Value,String>{
    let raw=request["cases"].as_array().ok_or("expected cases")?;
    if raw.is_empty()||raw.len()>1024{return Err("bad cases count".into());}
    let repeat=request["repetitions"].as_u64().ok_or("integer repetitions required")? as usize;
    if !(2..=100).contains(&repeat) {return Err("repetitions require 2..100".into());}
    let mut out=Vec::new();
    for c in raw{
        let arr=c["posterior"].as_array().ok_or("no posterior")?;
        let nsarr=c["extra_per_arm"].as_array().ok_or("no extra sample counts")?;
        let arms:Vec<(f64,f64)>=arr.iter().map(|v|{
            let z=v.as_array().ok_or("bad Beta pair")?;
            if z.len()!=2 {return Err("bad Beta dimensions".into());}
            Ok((z[0].as_f64().ok_or("bad a")?,z[1].as_f64().ok_or("bad b")?))
        }).collect::<Result<_,String>>()?;
        let ns:Vec<usize>=nsarr.iter().map(|x|x.as_u64().map(|n|n as usize)
            .ok_or("bad budget".into())).collect::<Result<_,String>>()?;
        let reference=full_grid(&arms,&ns)?;
        let fast_value=fast::beta_evsi(&arms,&ns)?;
        if (reference-fast_value).abs()>1e-10 {
            return Err(format!("reference/fast mismatch: {reference} vs {fast_value}"));
        }
        let mut gridtime=Vec::new();let mut fasttime=Vec::new();
        for i in 0..repeat {
            if i%2==0 {
                let t=Instant::now();
                black_box(full_grid(black_box(&arms),black_box(&ns))?);
                gridtime.push(t.elapsed().as_secs_f64()*1e6);
                let t=Instant::now();
                black_box(fast::beta_evsi(black_box(&arms),black_box(&ns))?);
                fasttime.push(t.elapsed().as_secs_f64()*1e6);
            }else{
                let t=Instant::now();
                black_box(fast::beta_evsi(black_box(&arms),black_box(&ns))?);
                fasttime.push(t.elapsed().as_secs_f64()*1e6);
                let t=Instant::now();
                black_box(full_grid(black_box(&arms),black_box(&ns))?);
                gridtime.push(t.elapsed().as_secs_f64()*1e6);
            }
        }
        let fastmed=median(fasttime);let gridmed=median(gridtime);
        out.push(json!({"K":arms.len(),"n":ns,"grid_evsi":reference,
          "fast_evsi":fast_value,
          "abs_error":(reference-fast_value).abs(),
          "grid_native_median_us":gridmed,
          "fast_native_median_us":fastmed,
          "native_grid_over_fast_speedratio":gridmed/fastmed}));
    }
    Ok(json!({"schema":"quartz_C21_same_compiler_native_fast_vs_joint_grid_TRAIN_only",
        "same_Rust_compiler_target":true,
        "repetitions":repeat,"n_cases":out.len(),
        "results":out,"prospective_C21_TEST_input_used":false,
        "not_retrofitted_to_C21_frozen_TEST_policy":true}))
}
fn main(){
    let Some(path)=std::env::args().nth(1)else{
        eprintln!("usage: quartz_c21_grid_reference TRAIN_INPUT.json");
        std::process::exit(2);
    };
    let res=(||->Result<Value,String>{
        let text=std::fs::read_to_string(path).map_err(|e|e.to_string())?;
        let parsed:Value=serde_json::from_str(&text).map_err(|e|e.to_string())?;
        run(&parsed)
    })();
    match res{
        Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
        Err(e)=>{eprintln!("C21 native grid reference: {e}");std::process::exit(3)}
    }
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]
 fn full_grid_and_fast_three_tied_beta_arms(){
    let a=[(1.,1.),(2.,2.),(3.,3.)];let n=[2,2,2];
    let x=full_grid(&a,&n).unwrap();
    assert!((x-109.0/840.0).abs()<1e-12);
    assert!((x-fast::beta_evsi(&a,&n).unwrap()).abs()<1e-12);
 }
 #[test]
 fn full_grid_and_fast_heterogeneous_fractional(){
    let a=[(0.5,1.5),(2.5,3.5),(100.,0.1),(1.,1.)];
    let n=[3,1,4,2];
    let x=full_grid(&a,&n).unwrap();
    assert!((x-fast::beta_evsi(&a,&n).unwrap()).abs()<1e-10);
 }
}
