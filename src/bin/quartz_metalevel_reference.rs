//! Pure finite metalevel reference CLI. Not a production SearchPolicy.
#[path = "../mcts/policy/finite_metalevel.rs"]
mod finite_metalevel;
use finite_metalevel::{Computation, FiniteMetalevel};
use serde_json::{json, Value};
use std::path::Path;

fn number(v:&Value)->Result<f64,String>{
    let s=if let Some(s)=v.as_str(){s.to_owned()}else{v.to_string()};
    let f=if let Some((n,d))=s.split_once('/'){
        let a=n.parse::<f64>().map_err(|e|e.to_string())?;
        let b=d.parse::<f64>().map_err(|e|e.to_string())?;
        if b==0.0{return Err("zero denominator".into());}
        a/b
    }else{s.parse::<f64>().map_err(|e|e.to_string())?};
    if f.is_finite(){Ok(f)}else{Err("nonfinite scalar".into())}
}
fn vnum(v:&Value)->Result<Vec<f64>,String>{
    v.as_array().ok_or("expected numeric array")?.iter().map(number).collect()
}
fn mnum(v:&Value)->Result<Vec<Vec<f64>>,String>{
    v.as_array().ok_or("expected matrix")?.iter().map(vnum).collect()
}
fn evaluate(row:&Value)->Result<Value,String>{
    let mut comps=vec![];
    for e in row["experiments"].as_array().ok_or("missing experiments")?{
        comps.push(Computation{name:e["name"].as_str().ok_or("missing name")?.into(),
             cost:number(&e["cost"])?,likelihood:mnum(&e["likelihood"])?});
    }
    let m=FiniteMetalevel::new(vnum(&row["posterior"])?,mnum(&row["utilities"])?,comps,number(&row["lambda"])?)?;
    let d=m.one_step(&m.prior)?;
    let name=|x:Option<usize>|x.map(|i|m.computations[i].name.clone()).unwrap_or("STOP".to_string());
    let mut steps=vec![];
    let h=row["horizon"].as_u64().ok_or("invalid horizon")? as usize;
    for j in 0..=h{
        let (value,action)=m.finite_horizon(&m.prior,j)?;
        steps.push(json!({"h":j,"value":value,"choice":name(action)}));
    }
    let vars:Vec<_>=m.computations.iter().enumerate().map(|(i,e)|json!({
        "name":e.name,"cost":e.cost,"gross":d.gross_gain[i],
        "ratio":d.gain_per_cost[i],"net":d.net_gain[i]
    })).collect();
    Ok(json!({"id":row["id"],"initial_risk":finite_metalevel::bayes_risk(&m.prior,&m.utilities)?,
        "terminal_value":finite_metalevel::terminal_value(&m.prior,&m.utilities)?,
        "rate_choice":name(d.ratio_choice),"net_choice":name(d.net_choice),
        "experiments":vars,"horizons":steps}))
}
fn execute(path:&Path)->Result<Value,String>{
    let doc:Value=serde_json::from_slice(&std::fs::read(path).map_err(|e|e.to_string())?)
       .map_err(|e|e.to_string())?;
    if doc["schema"]!="quartz_general_meta17_frozen_finite_Bayes_tasks_v1"{
        return Err("wrong immutable experiment schema".into());
    }
    let cases=doc["cases"].as_array().ok_or("missing case list")?;
    if cases.len()!=5{return Err("expected exactly five frozen tasks".into());}
    let result:Result<Vec<_>,_>=cases.iter().map(evaluate).collect();
    Ok(json!({"schema":"quartz_meta17_rust_game_neutral_reference_v1",
       "game_specific_input_used":false,"tasks":result?}))
}
fn main(){
    let Some(arg)=std::env::args_os().nth(1)else{
        eprintln!("usage: quartz_metalevel_reference CASES.json");
        std::process::exit(2);
    };
    match execute(Path::new(&arg)){
        Ok(v)=>println!("{}",serde_json::to_string_pretty(&v).unwrap()),
        Err(err)=>{eprintln!("model or input error: {err}");std::process::exit(3);}
    }
}
#[cfg(test)]
mod cli_tests{
    use super::*;
    #[test]
    fn actual_frozen_bank_is_cross_task_reference(){
        let p=Path::new(concat!(env!("CARGO_MANIFEST_DIR"),"/configs/metalevel_reference_cases.v1.json"));
        let r=execute(p).unwrap();
        assert_eq!(r["tasks"].as_array().unwrap().len(),5);
        assert_eq!(r["game_specific_input_used"],false);
        assert_eq!(r["tasks"][0]["rate_choice"],"partial_state_3");
        assert_eq!(r["tasks"][0]["net_choice"],"reveal_all");
    }
}
