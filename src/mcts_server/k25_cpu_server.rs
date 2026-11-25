//! K25 joint profile and actual CPU scheduling comparison. Research only.
use super::*;
use crate::mcts::four_axis::{Hooks,LiteRoot};
use serde::Deserialize;
use serde_json::{json,Value};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Req{cmd:String,history:String,method:String,seed:u64,model_path:String,model_identity:String,
    #[serde(default)]profile:Option<u8>,#[serde(default)]table:Option<[u8;4]>,#[serde(default)]trace:bool}
fn us(t:Instant)->u64{t.elapsed().as_micros() as u64}
fn aggregate(lanes:&[LiteRoot])->Value{
    let a=&lanes[0];let mut n=vec![0u32;a.len];let mut w=vec![0.;a.len];let mut nn=0;let mut visits=0;
    for r in lanes{assert_eq!(a.len,r.len);assert_eq!(a.moves,r.moves);assert_eq!(a.prior,r.prior);
        nn+=r.nn;visits+=r.visits;for i in 0..a.len{n[i]+=r.n[i];w[i]+=r.w[i];}}
    let i=n.iter().enumerate().max_by_key(|&(_,v)|*v).unwrap().0;
    json!({"moves":a.moves[..a.len],"n":n,"w":w,"visits":visits,"nn":nn,"best":a.moves[i]})
}
fn bin(a:&Value)->usize{
    let mut n:Vec<_>=a["n"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()).collect();n.sort_unstable();
    let top=n[n.len()-1];let gap=top-n[n.len()-2];2*usize::from(4*top>=3*a["visits"].as_u64().unwrap())+usize::from(gap>=4)
}
fn interval(root:&Connect4)->Vec<(usize,f64,f64)>{
    root.legal_moves().iter().map(|&m|{let s=root.apply_move(m);
        let (l,u)=if s.is_terminal(){let v=(1.-s.outcome() as f64)/2.;(v,v)}
        else if s.legal_moves().iter().any(|&a|s.is_winning_move(a)){(0.,0.)}else{(0.,1.)};
        (m,l,u)}).collect()
}
fn stage(engine:&MctsEngine<Connect4>,h:&Hooks,method:&str,delta:u32,t:Instant)->(u64,u64){
    let start=us(t);
    if method=="WAVE2"{for _ in 0..delta/8{engine.four_axis_k25_block(8,2,h);}}
    else{engine.four_axis_k25_block(delta,if method=="TREE2"{2}else{1},h);}
    (start,us(t))
}
fn inner(line:&str)->Result<Value,String>{
    let start=Instant::now();let q:Req=serde_json::from_str(line).map_err(|e|e.to_string())?;
    if q.cmd!="research_connect4_k25_cpu"||!matches!(q.method.as_str(),"SERIAL"|"TREE2"|"WAVE2"|"ROOT2")||
       q.profile.is_some()==q.table.is_some()||q.profile.is_some_and(|v|v>15)||q.table.is_some_and(|t|t.iter().any(|&v|v>15)){return Err("bad K25 contract".into());}
    let state=Connect4::from_history(&q.history)?;let legal=state.legal_moves();
    if state.is_terminal()||legal.len()<3{return Err("bad root".into());}
    if q.method!="SERIAL"&&rayon::current_num_threads()<2{return Err("K25 requires two rayon workers".into());}
    let (ev,hit)=model_for_root("conv3x3",Some(&q.model_path),Some(&q.model_identity),true)?;
    let t=Instant::now();let count=if q.method=="ROOT2"{2}else{1};
    let mut engines=Vec::with_capacity(count);let mut hs=Vec::with_capacity(count);
    for i in 0..count{
        let mut cfg=MctsConfig::evaluation(2.).with_seed(q.seed.wrapping_add(i as u64));
        cfg.root_forced_win=false;cfg.dirichlet=None;cfg.temperature=0.;cfg.quartz=None;
        cfg.pw=Some(crate::mcts::PwConfig::new(2.,0.5));
        engines.push(MctsEngine::new(state.clone(),ev.clone(),cfg));hs.push(Hooks::new_modes(0,0,0,false).with_k25(q.trace));
    }
    let setup_us=us(t);
    let t=Instant::now();let prefix_lanes=if count==2{
        let(a,b)=rayon::join(||stage(&engines[0],&hs[0],"SERIAL",16,t),||stage(&engines[1],&hs[1],"SERIAL",16,t));vec![a,b]
    }else{vec![stage(&engines[0],&hs[0],"SERIAL",32,t)]};let prefix_us=us(t);
    let pre:Vec<_>=engines.iter().map(|e|e.four_axis_lite()).collect();
    let prefix=aggregate(&pre);let b=bin(&prefix);let profile=q.profile.unwrap_or_else(||q.table.unwrap()[b]);
    let ph:Vec<_>=hs.iter().map(|h|h.k25_report(false)).collect();
    for h in &mut hs{h.set_k25_mask(profile);}
    let t=Instant::now();let tail_lanes=if count==2{
        let(a,b)=rayon::join(||stage(&engines[0],&hs[0],"SERIAL",48,t),||stage(&engines[1],&hs[1],"SERIAL",48,t));vec![a,b]
    }else{vec![stage(&engines[0],&hs[0],&q.method,96,t)]};let tail_us=us(t);
    let end:Vec<_>=engines.iter().map(|e|e.four_axis_lite()).collect();let result=aggregate(&end);
    assert_eq!(result["visits"],128);let proof=interval(&state);let ub=proof.iter().map(|p|p.2).fold(0.,f64::max);
    let cert=proof.iter().find(|p|p.1>=ub).map(|p|p.0);
    let best=cert.unwrap_or(result["best"].as_u64().unwrap() as usize);
    let lanes:Vec<_>=engines.iter().enumerate().map(|(i,e)|json!({"prefix":pre[i],"prefix_hooks":ph[i],"end":end[i],
        "tail_hooks":hs[i].k25_report(q.trace),"snapshot":e.four_axis_snapshot(),
        "prefix_interval_us":prefix_lanes[i],"tail_interval_us":tail_lanes[i]})).collect();
    let calls=result["nn"].as_u64().unwrap();drop(engines);drop(hs);
    Ok(json!({"schema":"quartz_k25_c4_v1","history":q.history,"method":q.method,"profile":profile,"bin":b,"seed":q.seed,
        "model_identity":q.model_identity,"cache_hit":hit,"legal":legal,"prefix":prefix,"result":result,"best":best,
        "proof":proof,"visits":128,"NN_calls":calls,"lanes":lanes,"setup_us":setup_us,"prefix_us":prefix_us,"tail_us":tail_us,"whole_us":us(start)}))
}
pub(super) fn handle(line:&str)->Value{match inner(line){Ok(v)=>v,Err(e)=>json!({"error":e})}}
