//! Domain-neutral signed value-of-computation SHADOW evaluator.
//! No GameState, rollout, predictor training, data labels or true state.
use serde_json::{json,Value};
use std::path::Path;
const EPS:f64=1e-6;
#[derive(Debug,Clone)]
pub struct Forecast {
   pub q:f64,pub conditional_signed:f64,pub gross:f64,
   pub estimated_cost_us:f64,pub net:f64,pub compute:bool
}
fn predict(qw:[f64;3],mw:[f64;3],qo:f64,mo:f64,
    gap:f64,n_actions:usize,price:f64,cost:f64)->Result<Forecast,String>{
    if qw.iter().chain(mw.iter()).any(|p|!p.is_finite()) ||
       !qo.is_finite()||!mo.is_finite()||!gap.is_finite()||
       !(0.0..=1.0).contains(&gap)||n_actions<2||
       !price.is_finite()||price<0.0||!cost.is_finite()||cost<=0.0{
       return Err("invalid finite initial-only prediction or priced cost".into());
    }
    let x=[1.0,gap,(1.0+n_actions as f64).ln()];
    let qlogit=qw[0]*x[0]+qw[1]*x[1]+qw[2]*x[2]+qo;
    let mlinear=mw[0]*x[0]+mw[1]*x[1]+mw[2]*x[2]+mo;
    let q=if qlogit>=0.0{
        1.0/(1.0+(-qlogit).exp())
    }else{
        qlogit.exp()/(1.0+qlogit.exp())
    }.clamp(EPS,1.0-EPS);
    let m=mlinear.clamp(-1.0,1.0);
    let gross=q*m;
    let net=gross-price*cost;
    if !net.is_finite(){return Err("nonfinite signed net utility".into());}
    Ok(Forecast{q,conditional_signed:m,gross,estimated_cost_us:cost,
       net,compute:net>0.0})
}
pub fn signed_net(qw:[f64;3],mw:[f64;3],qo:f64,mo:f64,
    gap:f64,n_actions:usize,price:f64,estimated_cost_us:f64)->Result<f64,String>{
    Ok(predict(qw,mw,qo,mo,gap,n_actions,price,estimated_cost_us)?.net)
}
fn triplet(v:&Value)->Result<[f64;3],String>{
    let a=v.as_array().ok_or("expected three numeric model parameters")?;
    if a.len()!=3{return Err("model coefficient vector not 3-dimensional".into());}
    Ok([a[0].as_f64().ok_or("bad coefficient")?,
        a[1].as_f64().ok_or("bad coefficient")?,
        a[2].as_f64().ok_or("bad coefficient")?])
}
fn evaluate(req:&Value)->Result<Value,String>{
    let q=triplet(&req["q_coefficients"])?;
    let m=triplet(&req["m_coefficients"])?;
    let qo=req["q_offset"].as_f64().ok_or("missing q_offset")?;
    let mo=req["m_offset"].as_f64().ok_or("missing m_offset")?;
    let price=req["resource_price"].as_f64().ok_or("missing price")?;
    let rows=req["rows"].as_array().ok_or("missing initial-only rows")?;
    if rows.is_empty(){return Err("no initial-only rows".into());}
    let result:Result<Vec<Value>,String>=rows.iter().map(|r|->Result<Value,String>{
        if r.get("label").is_some()||r.get("signed_gain").is_some()||
           r.get("domain").is_some()||r.get("true_utility").is_some()||
           r.get("final_action").is_some()||r.get("observed_cost_us").is_some(){
           return Err("forbidden post-compute/latent task data in shadow prediction".into());
        }
        let gap=r["gap"].as_f64().ok_or("missing normalized gap")?;
        let n=r["n_actions"].as_u64().ok_or("missing terminal action count")? as usize;
        let c=r["estimated_cost_us"].as_f64().ok_or("missing predeclared cost forecast")?;
        let pred=predict(q,m,qo,mo,gap,n,price,c)?;
        Ok(json!({"flip_probability":pred.q,
          "conditional_signed_value":pred.conditional_signed,
          "gross_signed_value":pred.gross,
          "estimated_incremental_cost_us":pred.estimated_cost_us,
          "predicted_net_value":pred.net,
          "shadow_choice":if pred.compute {"COMPUTE"}else{"STOP"},
          "runtime_search_unchanged":true}))
    }).collect();
    Ok(json!({"schema":"quartz_general_c19_shadow_signed_v1",
      "predictions":result?,"no_search_or_decision_mutation":true,
      "true_utility_input_used":false}))
}
fn main(){
    let Some(path)=std::env::args_os().nth(1)else{
        eprintln!("usage: quartz_c19_shadow FROZEN_REQUEST.json");
        std::process::exit(2);
    };
    let result=(||->Result<Value,String>{
        let data=std::fs::read(Path::new(&path)).map_err(|e|e.to_string())?;
        let req:Value=serde_json::from_slice(&data).map_err(|e|e.to_string())?;
        evaluate(&req)
    })();
    match result{
       Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
       Err(err)=>{eprintln!("C19 shadow contract ERROR: {err}");std::process::exit(3);}
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn zero_sign_change_but_10us_cost_is_negative_net(){
       let net=signed_net([0.;3],[0.;3],0.,0.,0.25,3,0.004,10.).unwrap();
       assert!((net+0.04).abs()<1e-12);
    }
    #[test]
    fn signed_gain_and_cost_unit_invariance(){
       let x=predict([0.1,0.7,-0.2],[0.2,-0.2,0.04],0.15,0.03,0.2,4,0.004,6.).unwrap();
       assert!((x.gross-x.q*x.conditional_signed).abs()<1e-12);
       assert!((x.net-(x.gross-0.024)).abs()<1e-12);
       let same=predict([0.1,0.7,-0.2],[0.2,-0.2,0.04],0.15,0.03,
                 0.2,4,0.000004,6000.).unwrap();
       assert!((x.net-same.net).abs()<1e-12);
    }
    #[test]
    fn deny_later_observations_and_bad_inputs(){
        let valid=json!({"q_coefficients":[0.,0.,0.],
            "m_coefficients":[0.,0.,0.],
            "q_offset":0.,"m_offset":0.,"resource_price":0.004,
            "rows":[{"gap":0.3,"n_actions":3,"estimated_cost_us":10.}]});
        assert_eq!(evaluate(&valid).unwrap()["no_search_or_decision_mutation"],true);
        let mut bad=valid.clone();
        bad["rows"][0]["signed_gain"]=json!(0.3);
        assert!(evaluate(&bad).is_err());
        bad=valid.clone();
        bad["rows"][0]["domain"]=json!("specific_game");
        assert!(evaluate(&bad).is_err());
        assert!(predict([0.;3],[f64::NAN,0.,0.],0.,0.,0.1,3,0.004,3.).is_err());
        assert!(predict([0.;3],[0.;3],0.,0.,0.1,3,0.004,0.).is_err());
    }
}
