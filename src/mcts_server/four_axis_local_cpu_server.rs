//! Single-root same-prefix local research endpoint; exact oracle is offline.
use super::*;
use crate::mcts::four_axis::{Hooks,X};
use serde::Deserialize;
use serde_json::{json,Value};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,arm:String,mode:String,seed:u64,model_path:String,model_identity:String,#[serde(default)]target_us:u64,#[serde(default)]reserve_us:u64,#[serde(default)]retention:f64}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn snap(engine:&MctsEngine<Connect4>,state:&Connect4,h:&Hooks,dump:bool)->Value{
    let r=engine.four_axis_snapshot();let normal=r["best"].as_u64().unwrap() as usize;
    let vs:Vec<_>=r["edges"].as_array().unwrap().iter().map(|e|(e["move"].as_u64().unwrap() as usize,e["n"].as_u64().unwrap() as u32)).collect();
    let p=h.root_proof(state,normal,&vs,dump);
    let best=if p.is_null(){normal}else{p["best"].as_u64().unwrap() as usize};
    json!({"result":r,"proof":p,"best":best,"hooks":h.snapshot()})
}
fn retention(pre:&Value,post:&Value,threshold:f64)->(usize,Option<f64>){
    let a=pre["best"].as_u64().unwrap() as usize;let b=post["best"].as_u64().unwrap() as usize;
    let p=&post["proof"];
    if a==b||p.is_null()||!p["witness"].is_null()||!p["allowed"].as_array().unwrap().iter().any(|m|m.as_u64()==Some(a as u64)){return(b,None);}
    let r=&post["result"];let edges=r["edges"].as_array().unwrap();
    let n=|m:usize|edges.iter().find(|e|e["move"].as_u64()==Some(m as u64)).and_then(|e|e["n"].as_u64()).unwrap_or(0);
    if n(a)==0||n(b)==0{return(b,None);}
    let margin=(n(b) as f64-n(a) as f64)/r["root_n"].as_u64().unwrap() as f64;
    (if threshold>0.&&margin<=threshold{a}else{b},Some(margin))
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_fouraxis_local_cpu"||!matches!(q.mode.as_str(),"DIAG"|"WORK"|"NN"|"TIME"){return Err("bad contract".into());}
    if ![0.,0.05,0.10,0.20].contains(&q.retention){return Err("bad retention".into());}
    if q.mode=="TIME"&&(!(3000..=20000).contains(&q.target_us)||q.reserve_us>=q.target_us)||q.mode!="TIME"&&(q.target_us!=0||q.reserve_us!=0){return Err("bad budget".into());}
    let (sx,gm,local,v,b,priority)=match q.arm.as_str(){
        "BASE"=>(0,0,0,0,0,false),"SAFE"=>(3,2,0,2,0,false),"ACTIVE"=>(3,2,0,2,0,true),
        "FOCUS"=>(3,2,1,2,0,false),"EXPAND"=>(3,2,2,2,0,false),"FRONTIER"=>(3,2,3,2,0,false),
        "EVAL"=>(3,2,0,4,0,false),"WRITER"=>(3,2,0,2,1,false),"MERGE"=>(3,2,0,2,3,false),_=>return Err("bad arm".into())
    };
    let state=Connect4::from_history(&q.history)?;if state.is_terminal(){return Err("terminal root".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
    if sx&X!=0{cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));}
    let t=Instant::now();let engine=MctsEngine::new(state.clone(),ev,cfg);let setup_us=us(t);
    let mut h=Hooks::new_modes(sx,if gm==0{0}else{2},0,false).with_graph(gm).with_safe_priority().with_repeat_trace();
    h.proof_register(&state);
    let t=Instant::now();engine.four_axis_block(32,1,&h);let prefix_us=us(t);
    let prefix=snap(&engine,&state,&h,q.mode=="DIAG");
    h.configure_local(local,v,b,priority);
    let max=match q.mode.as_str(){"DIAG"=>64,"WORK"=>128,_=>256};
    let threads=if q.mode=="DIAG"{1}else{2};
    let mut steps=Vec::new();let mut visits=32u32;
    let optional=Instant::now();
    let cause=loop{
        if gm!=0&&h.root_control(&state).0.is_some(){break "CERTIFIED";}
        if q.mode=="NN"&&engine.evaluator_call_count()>=64{break "NN";}
        if q.mode=="TIME"&&us(start)>=q.target_us-q.reserve_us{break "TIME";}
        if visits>=max{break if matches!(q.mode.as_str(),"NN"|"TIME"){"CAP"}else{"VISITS"};}
        let t=Instant::now();let plan=engine.prepare_local(&mut h);let plan_us=us(t);
        let t=Instant::now();engine.four_axis_block(8,threads,&h);let search_us=us(t);visits+=8;
        let post=snap(&engine,&state,&h,false);
        steps.push(json!({"plan":plan,"plan_us":plan_us,"search_us":search_us,"post":post}));
    };
    let optional_us=us(optional);
    let last=steps.last().map(|s|&s["post"]).unwrap_or(&prefix);
    let best=last["best"].as_u64().unwrap() as usize;
    let (guard_best,margin)=retention(&prefix,last,q.retention);
    let before_drop_us=us(start);let t=Instant::now();drop(engine);drop(h);let drop_us=us(t);
    let whole_us=us(start);
    Ok(json!({"schema":"quartz_fouraxis05_local_v1","history":q.history,"arm":q.arm,"mode":q.mode,"seed":q.seed,"threads":threads,"model_identity":q.model_identity,"cache_hit":hit,"target_us":q.target_us,"reserve_us":q.reserve_us,"retention":q.retention,"prefix":prefix,"steps":steps,"cause":cause,"best":best,"guard_best":guard_best,"retention_margin":margin,"setup_us":setup_us,"prefix_us":prefix_us,"optional_us":optional_us,"before_drop_us":before_drop_us,"drop_us":drop_us,"whole_us":whole_us}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
