//! Research-only resident four-root certificate scheduler. No online oracle.
use super::*;
use crate::mcts::four_axis::{Hooks,X};
use serde::Deserialize;
use serde_json::{json,Value};
use std::collections::HashSet;
#[path="four_axis_pool_plan.rs"]
mod plan;
use plan::{Slot,Policy};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job{id:String,history:String}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,arm:String,mode:String,seed:u64,model_path:String,model_identity:String,jobs:Vec<Job>,#[serde(default)]target_us:u64,#[serde(default)]reserve_us:u64}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
struct Root{
    id:String,history:String,state:Connect4,
    engine:Option<MctsEngine<Connect4>>,hooks:Option<Hooks>,
    prefix:Value,ph:Value,initial:Value,slot:Slot,witness:Option<usize>,
    setup_us:u64,prefix_us:u64,finished:Option<Value>,
}
fn status(r:&Root)->Value{json!({"blocks":r.slot.blocks,"spent_us":r.slot.spent_us,"risk2":r.slot.risk2,"contenders":r.slot.contenders,"witness":r.witness,"visits":8+8*r.slot.blocks,"evals":r.engine.as_ref().unwrap().evaluator_call_count()})}
fn refresh(r:&mut Root){
    let (w,d,h)=r.hooks.as_ref().unwrap().root_control(&r.state);
    assert!(d<=r.slot.risk2);r.slot.risk2=d;r.slot.contenders=h;r.witness=w;
}
fn finish(r:&mut Root,reason:&str)->u64{
    assert!(!r.slot.retired);
    let t=Instant::now();
    let engine=r.engine.take().unwrap();let h=r.hooks.take().unwrap();
    let result=engine.four_axis_snapshot();let normal=result["best"].as_u64().unwrap() as usize;
    let vs:Vec<_>=result["edges"].as_array().unwrap().iter().map(|e|(e["move"].as_u64().unwrap() as usize,e["n"].as_u64().unwrap() as u32)).collect();
    let proof=h.root_proof(&r.state,normal,&vs,false);
    let best=if proof.is_null(){normal}else{proof["best"].as_u64().unwrap() as usize};
    if reason=="CERTIFIED"{assert!(r.witness.is_some());assert_eq!(Some(best),r.witness);}
    let hooks=h.snapshot();
    let st=Instant::now();drop(engine);drop(h);let drop_us=us(st);
    r.slot.retired=true;
    r.finished=Some(json!({"id":r.id,"history":r.history,"prefix":std::mem::take(&mut r.prefix),"prefix_hooks":std::mem::take(&mut r.ph),"initial":std::mem::take(&mut r.initial),"result":result,"best":best,"proof":proof,"hooks":hooks,"blocks":r.slot.blocks,"spent_us":r.slot.spent_us,"reason":reason,"setup_us":r.setup_us,"prefix_us":r.prefix_us,"drop_us":drop_us,"finish_us":us(t)}));
    drop_us
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_fouraxis_pool_cpu"||q.jobs.len()!=4||!matches!(q.mode.as_str(),"WORK"|"TIME"){return Err("invalid pool contract".into());}
    if q.mode=="TIME"&&(!(6000..=40000).contains(&q.target_us)||q.reserve_us>=q.target_us){return Err("bad time budget".into());}
    if q.mode=="WORK"&&(q.target_us!=0||q.reserve_us!=0){return Err("unexpected time budget".into());}
    let (sx,gm,bm,stop,reuse,bound,safe)=match q.arm.as_str(){
        "BASE"=>(0,0,0,false,false,false,false),
        "ACTIVE_FIXED"=>(3,2,0,false,false,false,false),
        "STOP_FIXED"=>(3,2,0,true,false,false,false),
        "RR"=>(3,2,0,true,true,false,false),
        "BOUND"=>(3,2,0,true,true,true,false),
        "RR_SAFE"=>(3,2,0,true,true,false,true),
        "RR_WRITER"=>(3,2,1,true,true,false,false),
        "RR_TEMPORAL"=>(3,2,2,true,true,false,false),
        "RR_MERGE"=>(3,2,3,true,true,false,false),
        _=>return Err("unknown arm".into())
    };
    let mut ids=HashSet::new();let mut boards=HashSet::new();let mut states=Vec::new();
    for j in &q.jobs{
        if !ids.insert(j.id.clone()){return Err("duplicate root id".into());}
        let state=Connect4::from_history(&j.history)?;
        if state.is_terminal()||!boards.insert((state.board_flat(),state.current_player())){return Err("terminal or duplicate state".into());}
        states.push(state);
    }
    let (eval,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let mut roots=Vec::new();
    for (j,state) in q.jobs.iter().zip(states){
        let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed);
        cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
        if sx&X!=0{cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));}
        let t=Instant::now();let engine=MctsEngine::new(state.clone(),eval.clone(),cfg);let setup_us=us(t);
        let ph=Hooks::new_modes(sx,0,0,false);
        let t=Instant::now();engine.four_axis_block(8,1,&ph);let prefix_us=us(t);
        let prefix=engine.four_axis_snapshot();let ph=ph.snapshot();
        let mut h=Hooks::new_modes(sx,if gm==0{0}else{2},bm,false).with_graph(gm);
        if safe{h=h.with_safe_priority();}
        h.proof_register(&state);
        let mut r=Root{id:j.id.clone(),history:j.history.clone(),state,engine:Some(engine),hooks:Some(h),prefix,ph,initial:Value::Null,slot:Slot{risk2:2,..Slot::default()},witness:None,setup_us,prefix_us,finished:None};
        refresh(&mut r);r.initial=status(&r);roots.push(r);
    }
    let mandatory_us=us(start);
    let allowance=q.target_us.saturating_sub(q.reserve_us).saturating_sub(mandatory_us)/4;
    let policy=if bound{Policy::Bound}else if reuse{Policy::Reuse}else if q.mode=="WORK"{Policy::FixedWork}else{Policy::FixedTime(allowance)};
    let mut trace=Vec::new();let mut retired=Vec::new();let mut drop_total=0;
    if stop{
        for (i,r) in roots.iter_mut().enumerate(){if r.witness.is_some(){
            drop_total+=finish(r,"CERTIFIED");retired.push(json!({"i":i,"after":0,"reason":"CERTIFIED"}));
        }}
    }
    let mut cursor=0;let mut total=0u32;
    let cause=loop{
        if q.mode=="WORK"&&total>=60{break "WORK";}
        if q.mode=="TIME"&&us(start)>=q.target_us-q.reserve_us{break "TIME";}
        let slots:Vec<_>=roots.iter().map(|r|r.slot).collect();
        let Some(i)=plan::choose(&slots,cursor,policy)else{break "NO_ELIGIBLE";};
        cursor=(i+1)%roots.len();
        let r=&mut roots[i];let before=status(r);let t=Instant::now();
        r.engine.as_ref().unwrap().four_axis_block(8,2,r.hooks.as_ref().unwrap());
        r.slot.blocks+=1;refresh(r);
        let charge=us(t).max(1);r.slot.spent_us+=charge;total+=1;
        let after=status(r);
        trace.push(json!({"i":i,"before":before,"after":after,"charge_us":charge}));
        if stop&&r.witness.is_some(){drop_total+=finish(r,"CERTIFIED");retired.push(json!({"i":i,"after":total,"reason":"CERTIFIED"}));}
    };
    let before_final_us=us(start);
    for (i,r) in roots.iter_mut().enumerate(){
        if r.slot.retired{continue;}
        let reason=if r.slot.blocks>=63{"CAP"}else if !reuse&&q.mode=="WORK"&&r.slot.blocks>=15{"QUOTA"}else if !reuse&&q.mode=="TIME"&&r.slot.spent_us>=allowance{"QUOTA"}else{cause};
        drop_total+=finish(r,reason);retired.push(json!({"i":i,"after":total,"reason":reason}));
    }
    let blocks:Vec<_>=roots.iter().map(|r|r.slot.blocks).collect();
    let cert:Vec<_>=roots.iter().map(|r|r.finished.as_ref().unwrap()["reason"]=="CERTIFIED").collect();
    let account=if q.mode=="WORK"{let(c,e,s,u)=plan::account(&blocks,&cert,15);json!({"credit":c,"extra":e,"shortfall":s,"unused":u})}else{Value::Null};
    let rows:Vec<_>=roots.into_iter().map(|r|r.finished.unwrap()).collect();
    let whole_us=us(start);
    Ok(json!({"schema":"quartz_fouraxis04_pool_v1","arm":q.arm,"mode":q.mode,"seed":q.seed,"model_identity":q.model_identity,"cache_hit":hit,"threads":2,"target_us":q.target_us,"reserve_us":q.reserve_us,"mandatory_us":mandatory_us,"allowance_us":allowance,"before_final_us":before_final_us,"whole_us":whole_us,"drop_total_us":drop_total,"total_blocks":total,"cause":cause,"account":account,"trace":trace,"retired":retired,"rows":rows}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
