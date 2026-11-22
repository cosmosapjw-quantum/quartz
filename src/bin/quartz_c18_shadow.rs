//! Game-independent, read-only C18 computation-revision forecast.
//! Standalone research CLI, NOT a live stop/computation SearchPolicy hook.
//! Inputs are initial-only margin and action count; never final/teacher data.
use serde_json::{json,Value};
use std::path::Path;
const EPS:f64=1e-6;

pub fn predict(theta:[f64;3],offset:f64,gap:f64,n_actions:usize)->Result<f64,String>{
    if !theta.iter().all(|x|x.is_finite())||!offset.is_finite()
        ||!gap.is_finite()||!(0.0..=1.0).contains(&gap)||n_actions<2 {
        return Err("invalid finite generic initial-only observation inputs".into());
    }
    let z=theta[0]+theta[1]*gap+theta[2]*(1.0+n_actions as f64).ln()+offset;
    let q=if z>=0.0 {1.0/(1.0+(-z).exp())} else {z.exp()/(1.0+z.exp())};
    Ok(q.clamp(EPS,1.0-EPS))
}
fn evaluate(request:&Value)->Result<Value,String>{
    let weights=request["theta"].as_array().ok_or("theta array required")?;
    if weights.len()!=3{return Err("three coefficients required".into());}
    let mut t=[0.0;3];
    for (i,x) in weights.iter().enumerate(){t[i]=x.as_f64().ok_or("nonfinite theta")?;}
    let offset=request["offset"].as_f64().ok_or("fixed offset required")?;
    let rows=request["rows"].as_array().ok_or("input row array required")?;
    if rows.is_empty(){return Err("nonempty generic rows required".into());}
    let result:Result<Vec<_>,_>=rows.iter().map(|row|{
        if row.get("label").is_some()||row.get("domain").is_some()
          ||row.get("final_mcts_move").is_some()
          ||row.get("true_utility_means").is_some(){
          return Err("shadow read-only forecast must not receive future/oracle labels".to_string());
        }
        let gap=row["gap"].as_f64().ok_or("gap required")?;
        let n=row["n_actions"].as_u64().ok_or("n_actions required")? as usize;
        predict(t,offset,gap,n)
    }).collect();
    Ok(json!({"schema":"quartz_general_c18_readonly_shadow_v1",
        "prediction":result?,"online_actions_mutated":false,
        "model_is_not_bayesian_utility_posterior":true}))
}
fn main(){
    let Some(p)=std::env::args_os().nth(1) else {
        eprintln!("usage: quartz_c18_shadow INPUT.json");
        std::process::exit(2);
    };
    let answer=(||->Result<Value,String>{
        let b=std::fs::read(Path::new(&p)).map_err(|e|e.to_string())?;
        let req:Value=serde_json::from_slice(&b).map_err(|e|e.to_string())?;
        evaluate(&req)
    })();
    match answer{
        Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
        Err(err)=>{eprintln!("C18 shadow contract error: {err}");std::process::exit(3);}
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn zero_coefficients_return_one_half(){
        assert!((predict([0.0;3],0.0,0.25,4).unwrap()-0.5).abs()<1e-12);
    }
    #[test]
    fn nontrivial_logistic_and_invalid_source_rejected(){
        let z=0.4-2.0*0.1+0.25*5f64.ln()-0.05;
        let expected=1.0/(1.0+(-z).exp());
        let actual=predict([0.4,-2.0,0.25],-0.05,0.1,4).unwrap();
        assert!((actual-expected).abs()<1e-12);
        assert!(predict([0.0;3],0.0,-0.1,4).is_err());
        assert!(predict([0.0;3],0.0,0.1,0).is_err());
        assert!(predict([f64::NAN,0.0,0.0],0.0,0.1,4).is_err());
    }
    #[test]
    fn no_future_label_or_domain_input_allowed(){
        let valid=json!({"theta":[0.0,0.0,0.0],"offset":0.0,
                "rows":[{"gap":0.1,"n_actions":3}]});
        assert_eq!(evaluate(&valid).unwrap()["online_actions_mutated"],false);
        let mut bad=valid.clone();
        bad["rows"][0]["label"]=json!(1);
        assert!(evaluate(&bad).is_err());
        bad=valid.clone();
        bad["rows"][0]["domain"]=json!("Gomoku");
        assert!(evaluate(&bad).is_err());
    }
}
